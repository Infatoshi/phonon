import Foundation

/// Transcribes recorded meetings both while recording is in progress and after it stops, one
/// at a time through the engine's meeting queue (`{"cmd":"meeting_span"}`, answered by
/// `{"type":"meeting_span"}`). While a meeting records, closed chunks roll into spans once buffered
/// and each finished span updates `transcript.partial.json`, `transcript.txt`, and
/// `Meetings/latest.txt`; after the recording stops, any trailing tail is finished and
/// `transcript.json` is written. The engine decides when each span may use the speech model, so a
/// dictation always goes first. Logs carry meeting ids, counts and timings only, never what was
/// said.
@MainActor
final class MeetingTranscriber {
    /// Sends a command to the engine.
    var send: (([String: Any]) -> Void)?
    var onJobsChanged: (([String: MeetingTranscriptionJob]) -> Void)?
    /// A meeting's transcript.json is written.
    var onFinished: ((String) -> Void)?
    /// Where a meeting's transcript files go: its own folder (a check can point elsewhere).
    var outputDirectory: (URL) -> URL = { $0 }

    private(set) var jobs: [String: MeetingTranscriptionJob] = [:] {
        didSet { if jobs != oldValue { onJobsChanged?(jobs) } }
    }
    private var waiting: [URL] = []
    private var current: Job?
    private var liveMetadata: [String: MeetingMetadata] = [:]
    private(set) var engineReady = false

    private struct Job {
        let id: String
        let directory: URL
        let output: URL
        var metadata: MeetingMetadata
        var live: Bool
        var progress: MeetingTranscriptProgress
        var inFlight: (id: String, span: MeetingSpan)?
        let started = Date()
        var spans = 0
        var dictatedWordsRemoved = 0
    }

    var isIdle: Bool { current == nil && waiting.isEmpty }

    /// Update (or start tracking) a live recording so closed chunks stream into `transcript.txt`
    /// and `Meetings/latest.txt` while the call is still under way.
    func updateLive(directory: URL, metadata: MeetingMetadata) {
        let id = directory.lastPathComponent
        let output = outputDirectory(directory)
        guard
            !FileManager.default.fileExists(
                atPath: output.appendingPathComponent(MeetingTranscript.fileName).path)
        else { return }
        liveMetadata[id] = metadata
        if var job = current, job.id == id {
            job.metadata = metadata
            job.live = true
            current = job
            sendNextSpan()
            return
        }
        try? FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let textURL = output.appendingPathComponent(MeetingTranscript.textFileName)
        if !FileManager.default.fileExists(atPath: textURL.path) {
            let progress = MeetingTranscriptProgress.load(from: output) ?? MeetingTranscriptProgress()
            try? TurnMerger.transcript(progress, metadata: metadata).saveText(
                to: output, metadata: metadata, status: .recording)
            MeetingTranscriptFiles.updateLatestText(for: output, id: id, metadata: metadata)
        }
        if !waiting.contains(where: { $0.lastPathComponent == id }) {
            waiting.append(directory)
            jobs[id] = .running(0)
            startNext()
        }
    }

    /// Cancel and forget a discarded live recording (Undo).
    func cancel(directory: URL) {
        let id = directory.lastPathComponent
        let output = outputDirectory(directory)
        liveMetadata.removeValue(forKey: id)
        waiting.removeAll { $0.lastPathComponent == id }
        if current?.id == id { current = nil }
        jobs[id] = nil
        if output.lastPathComponent == id {
            MeetingTranscriptFiles.refreshLatestTextAfterRemoval(
                of: id, root: output.deletingLastPathComponent())
        }
        startNext()
    }

    /// Transcribe the meeting in `directory` (after its recording stopped, or from the Notes
    /// button), resuming from its progress file when there is one.
    func enqueue(directory: URL) {
        let id = directory.lastPathComponent
        let wasLive = liveMetadata.removeValue(forKey: id) != nil
        if wasLive {
            if var job = current, job.id == id {
                job.live = false
                if let fresh = Self.loadMetadata(from: directory) { job.metadata = fresh }
                if job.metadata.micChunks.isEmpty && job.metadata.systemChunks.isEmpty {
                    dropEmptyJob(job)
                    return
                }
                try? writeText(job, status: .transcribing)
                current = job
                if job.inFlight == nil { sendNextSpan() }
                return
            }
            if waiting.contains(where: { $0.lastPathComponent == id }) {
                startNext()
                return
            }
        }
        guard current?.id != id, !waiting.contains(where: { $0.lastPathComponent == id }) else {
            return
        }
        let output = outputDirectory(directory)
        if FileManager.default.fileExists(
            atPath: output.appendingPathComponent(MeetingTranscript.fileName).path)
        {
            return
        }
        waiting.append(directory)
        jobs[id] = .queued
        startNext()
    }

    /// The engine is up (its `ready` event) or gone. A span in flight to an engine that went
    /// away is sent again once the next one is ready.
    func setEngineReady(_ ready: Bool) {
        engineReady = ready
        if !ready {
            current?.inFlight = nil
            return
        }
        if current == nil { startNext() } else { sendNextSpan() }
    }

    /// Meetings whose transcription a quit or crash interrupted (a progress file without an
    /// error and no transcript): picked up again at launch. A progress file beside a written
    /// transcript means the quit came after the transcript and before its hand-off to cleanup
    /// (`finish` drops the file only after `onFinished`): the hand-off is made now.
    func resumePending(in root: URL, fileManager: FileManager = .default) {
        let folders =
            (try? fileManager.contentsOfDirectory(
                at: root, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles])) ?? []
        for folder in folders.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            guard MeetingNote.isMeetingDirectory(folder, fileManager: fileManager) else { continue }
            let output = outputDirectory(folder)
            guard let progress = MeetingTranscriptProgress.load(from: output) else { continue }
            let marker = output.appendingPathComponent(MeetingTranscriptProgress.fileName)
            if fileManager.fileExists(atPath: output.appendingPathComponent(MeetingTranscript.fileName).path) {
                onFinished?(folder.lastPathComponent)
                try? fileManager.removeItem(at: marker)
                continue
            }
            guard progress.error == nil else { continue }
            enqueue(directory: folder)
        }
    }

    /// Leave a progress marker for a meeting whose recording stopped as the app quit, so the
    /// next launch transcribes it.
    nonisolated static func markPending(directory: URL) {
        guard MeetingTranscriptProgress.load(from: directory) == nil else { return }
        try? MeetingTranscriptProgress().save(to: directory)
    }

    private static func loadMetadata(from directory: URL) -> MeetingMetadata? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("metadata.json")) else {
            return nil
        }
        return try? JSONDecoder().decode(MeetingMetadata.self, from: data)
    }

    private func startNext() {
        guard current == nil, !waiting.isEmpty else { return }
        let directory = waiting.removeFirst()
        let id = directory.lastPathComponent
        let output = outputDirectory(directory)
        let isLive = liveMetadata[id] != nil
        guard let metadata = liveMetadata[id] ?? Self.loadMetadata(from: directory) else {
            liveMetadata.removeValue(forKey: id)
            jobs[id] = .failed("This meeting's metadata.json could not be read.")
            startNext()
            return
        }
        if metadata.micChunks.isEmpty && metadata.systemChunks.isEmpty && !isLive {
            // A recording that never started: nothing to hear, and Notes does not list it.
            try? FileManager.default.removeItem(
                at: output.appendingPathComponent(MeetingTranscriptProgress.fileName))
            jobs[id] = nil
            startNext()
            return
        }
        var progress = MeetingTranscriptProgress.load(from: output) ?? MeetingTranscriptProgress()
        progress.error = nil
        let job = Job(
            id: id, directory: directory, output: output, metadata: metadata, live: isLive,
            progress: progress)
        do {
            try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
            try progress.save(to: output)
            try writeText(job, status: isLive ? .recording : .transcribing)
        } catch {
            liveMetadata.removeValue(forKey: id)
            jobs[id] = .failed("Could not save transcription progress: \(error.localizedDescription)")
            startNext()
            return
        }
        NSLog(
            "phonon meeting transcript: \(id) started (mic at \(progress.mic.cursor), "
                + "system at \(progress.system.cursor)\(isLive ? ", live" : ""))")
        current = job
        sendNextSpan()
    }

    private func dropEmptyJob(_ job: Job) {
        try? FileManager.default.removeItem(
            at: job.output.appendingPathComponent(MeetingTranscriptProgress.fileName))
        try? FileManager.default.removeItem(
            at: job.output.appendingPathComponent(MeetingTranscript.textFileName))
        current = nil
        jobs[job.id] = nil
        if job.output.lastPathComponent == job.id {
            MeetingTranscriptFiles.refreshLatestTextAfterRemoval(
                of: job.id, root: job.output.deletingLastPathComponent())
        }
        startNext()
    }

    private func writeText(_ job: Job, status: MeetingTranscript.LiveStatus) throws {
        let transcript = TurnMerger.transcript(job.progress, metadata: job.metadata)
        try transcript.saveText(to: job.output, metadata: job.metadata, status: status)
        MeetingTranscriptFiles.updateLatestText(
            for: job.output, id: job.id, metadata: job.metadata)
    }

    private func chunks(_ track: MeetingTrack, of metadata: MeetingMetadata) -> [MeetingChunk] {
        track == .mic ? metadata.micChunks : metadata.systemChunks
    }

    /// How far the meeting is, by track samples transcribed, 0...1.
    static func fraction(
        _ progress: MeetingTranscriptProgress, metadata: MeetingMetadata, live: Bool = false
    ) -> Double {
        var done: Int64 = 0
        var total: Int64 = 0
        for track in [MeetingTrack.mic, .system] {
            let chunks = track == .mic ? metadata.micChunks : metadata.systemChunks
            let length = chunks.map { $0.startSample + $0.sampleCount }.max() ?? 0
            total += length
            done += progress[track].done ? length : min(length, max(0, progress[track].cursor))
        }
        if total == 0 { return live ? 0 : 1 }
        return Double(done) / Double(total)
    }

    private func sendNextSpan() {
        guard engineReady, var job = current, job.inFlight == nil else { return }
        job.live = liveMetadata[job.id] != nil
        if job.live, let fresh = liveMetadata[job.id] { job.metadata = fresh }
        let tracks: [MeetingTrack] =
            job.live && job.progress.system.cursor < job.progress.mic.cursor
            ? [.system, .mic] : [.mic, .system]
        for track in tracks where job.live || !job.progress[track].done {
            if let span = MeetingSpanPlanner.next(
                track: track, chunks: chunks(track, of: job.metadata),
                cursor: job.progress[track].cursor, rate: job.metadata.sampleRate, live: job.live)
            {
                let spanID = "\(job.id):\(track.rawValue):\(span.startSample)"
                job.inFlight = (spanID, span)
                current = job
                jobs[job.id] = .running(
                    Self.fraction(job.progress, metadata: job.metadata, live: job.live))
                let folder = job.directory.appendingPathComponent(track.rawValue, isDirectory: true)
                send?([
                    "cmd": "meeting_span", "id": spanID,
                    "chunks": span.chunks.map {
                        [
                            "path": folder.appendingPathComponent($0.file).path,
                            "start_sample": $0.startSample,
                        ] as [String: Any]
                    },
                    "start_sample": span.startSample, "end_sample": span.endSample,
                ])
                return
            }
            if !job.live { job.progress[track].done = true }
        }
        if job.live {
            if !waiting.isEmpty {
                current = nil
                waiting.append(job.directory)
                jobs[job.id] = .queued
                startNext()
            } else {
                current = job
                jobs[job.id] = .running(
                    Self.fraction(job.progress, metadata: job.metadata, live: true))
            }
            return
        }
        current = job
        finish(job)
    }

    /// `{"type":"meeting_span"}` from the engine.
    func handle(_ ev: [String: Any]) {
        guard let id = ev["id"] as? String, var job = current, let inFlight = job.inFlight,
            inFlight.id == id
        else { return }
        job.inFlight = nil
        job.live = liveMetadata[job.id] != nil
        if let fresh = liveMetadata[job.id] ?? Self.loadMetadata(from: job.directory) {
            job.metadata = fresh
        }
        if let error = ev["error"] as? String {
            fail(job, message: error)
            return
        }
        let sentences = Self.sentences(ev["sentences"])
        let (planned, cursor) = MeetingSpanPlanner.advance(
            inFlight.span, sentences: sentences, rate: job.metadata.sampleRate)
        let track = inFlight.span.track
        var kept = planned
        if track == .mic {
            let filtered = DictationFilter.remove(kept, windows: job.metadata.dictationWindows())
            kept = filtered.kept
            job.dictatedWordsRemoved += filtered.removedWords
        }
        // word times are only for that; the progress file keeps sentences
        job.progress[track].sentences += kept.map { TimedSentence(start: $0.start, end: $0.end, text: $0.text) }
        job.progress[track].cursor = cursor
        if let model = ev["model"] as? String, !model.isEmpty { job.progress.model = model }
        job.spans += 1
        do {
            try job.progress.save(to: job.output)
            try writeText(job, status: job.live ? .recording : .transcribing)
        } catch {
            fail(job, message: "Could not save transcription progress: \(error.localizedDescription)")
            return
        }
        current = job
        sendNextSpan()
    }

    static func sentences(_ value: Any?) -> [TimedSentence] {
        guard let rows = value as? [[String: Any]] else { return [] }
        return rows.compactMap { row in
            guard let text = row["text"] as? String,
                let start = (row["start"] as? NSNumber)?.doubleValue
            else { return nil }
            let end = (row["end"] as? NSNumber)?.doubleValue ?? start
            let words = (row["words"] as? [[String: Any]])?.compactMap { word -> TimedWord? in
                guard let text = word["text"] as? String, let start = (word["start"] as? NSNumber)?.doubleValue
                else { return nil }
                let end = (word["end"] as? NSNumber)?.doubleValue ?? start
                return TimedWord(start: start, end: max(start, end), text: text)
            }
            return TimedSentence(start: start, end: max(start, end), text: text, words: words)
        }
    }

    private func fail(_ job: Job, message: String) {
        liveMetadata.removeValue(forKey: job.id)
        var progress = job.progress
        progress.error = message
        try? progress.save(to: job.output)
        NSLog("phonon meeting transcript: \(job.id) failed after \(job.spans) spans")
        current = nil
        jobs[job.id] = .failed(message)
        startNext()
    }

    private func finish(_ job: Job) {
        var finalJob = job
        if let fresh = Self.loadMetadata(from: finalJob.directory) { finalJob.metadata = fresh }
        let transcript = TurnMerger.transcript(finalJob.progress, metadata: finalJob.metadata)
        do {
            try transcript.save(to: finalJob.output)
            try transcript.saveText(
                to: finalJob.output, metadata: finalJob.metadata, status: .complete)
            MeetingTranscriptFiles.updateLatestText(
                for: finalJob.output, id: finalJob.id, metadata: finalJob.metadata)
        } catch {
            fail(finalJob, message: "Could not save the transcript: \(error.localizedDescription)")
            return
        }
        NSLog(
            "phonon meeting transcript: \(finalJob.id) done: \(transcript.turns.count) turns, "
                + "\(transcript.wordCount) words, \(finalJob.dictatedWordsRemoved) dictated words left out "
                + "(\(finalJob.metadata.dictations?.count ?? 0) dictations), \(finalJob.spans) spans in "
                + String(format: "%.1f s", Date().timeIntervalSince(finalJob.started)))
        current = nil
        jobs[finalJob.id] = nil
        // Hand off (cleanup writes its own progress file) before the marker goes, so a quit in
        // between still hands it off at the next launch (`resumePending`).
        onFinished?(finalJob.id)
        try? FileManager.default.removeItem(
            at: finalJob.output.appendingPathComponent(MeetingTranscriptProgress.fileName))
        startNext()
    }
}
