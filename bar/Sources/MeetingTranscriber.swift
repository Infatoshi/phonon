import Foundation

/// Transcribes recorded meetings one at a time through the engine's meeting queue
/// (`{"cmd":"meeting_span"}`, answered by `{"type":"meeting_span"}`). It sends one span, waits
/// for its answer, saves the progress, and sends the next; the engine decides when each span
/// may use the speech model, so a dictation always goes first. Logs carry meeting ids, counts
/// and timings only, never what was said.
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
    private(set) var engineReady = false

    private struct Job {
        let id: String
        let directory: URL
        let output: URL
        let metadata: MeetingMetadata
        var progress: MeetingTranscriptProgress
        var inFlight: (id: String, span: MeetingSpan)?
        let started = Date()
        var spans = 0
    }

    var isIdle: Bool { current == nil && waiting.isEmpty }

    /// Transcribe the meeting in `directory` (after its recording stopped, or from the Notes
    /// button), resuming from its progress file when there is one.
    func enqueue(directory: URL) {
        let id = directory.lastPathComponent
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
    /// error and no transcript): picked up again at launch.
    func resumePending(in root: URL, fileManager: FileManager = .default) {
        let folders =
            (try? fileManager.contentsOfDirectory(
                at: root, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])) ?? []
        for folder in folders.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            let output = outputDirectory(folder)
            guard let progress = MeetingTranscriptProgress.load(from: output),
                progress.error == nil,
                !fileManager.fileExists(
                    atPath: output.appendingPathComponent(MeetingTranscript.fileName).path)
            else { continue }
            enqueue(directory: folder)
        }
    }

    /// Leave a progress marker for a meeting whose recording stopped as the app quit, so the
    /// next launch transcribes it.
    nonisolated static func markPending(directory: URL) {
        guard MeetingTranscriptProgress.load(from: directory) == nil else { return }
        try? MeetingTranscriptProgress().save(to: directory)
    }

    private func startNext() {
        guard current == nil, !waiting.isEmpty else { return }
        let directory = waiting.removeFirst()
        let id = directory.lastPathComponent
        let output = outputDirectory(directory)
        guard
            let data = try? Data(contentsOf: directory.appendingPathComponent("metadata.json")),
            let metadata = try? JSONDecoder().decode(MeetingMetadata.self, from: data)
        else {
            jobs[id] = .failed("This meeting's metadata.json could not be read.")
            startNext()
            return
        }
        if metadata.micChunks.isEmpty && metadata.systemChunks.isEmpty {
            // A recording that never started: nothing to hear, and Notes does not list it.
            try? FileManager.default.removeItem(
                at: output.appendingPathComponent(MeetingTranscriptProgress.fileName))
            jobs[id] = nil
            startNext()
            return
        }
        var progress = MeetingTranscriptProgress.load(from: output) ?? MeetingTranscriptProgress()
        progress.error = nil
        do {
            try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
            try progress.save(to: output)
        } catch {
            jobs[id] = .failed("Could not save transcription progress: \(error.localizedDescription)")
            startNext()
            return
        }
        NSLog(
            "phonon meeting transcript: \(id) started (mic at \(progress.mic.cursor), "
                + "system at \(progress.system.cursor))")
        current = Job(id: id, directory: directory, output: output, metadata: metadata, progress: progress)
        sendNextSpan()
    }

    private func chunks(_ track: MeetingTrack, of metadata: MeetingMetadata) -> [MeetingChunk] {
        track == .mic ? metadata.micChunks : metadata.systemChunks
    }

    /// How far the meeting is, by track samples transcribed, 0...1.
    static func fraction(_ progress: MeetingTranscriptProgress, metadata: MeetingMetadata) -> Double {
        var done: Int64 = 0
        var total: Int64 = 0
        for track in [MeetingTrack.mic, .system] {
            let chunks = track == .mic ? metadata.micChunks : metadata.systemChunks
            let length = chunks.map { $0.startSample + $0.sampleCount }.max() ?? 0
            total += length
            done += progress[track].done ? length : min(length, max(0, progress[track].cursor))
        }
        return total > 0 ? Double(done) / Double(total) : 1
    }

    private func sendNextSpan() {
        guard engineReady, var job = current, job.inFlight == nil else { return }
        for track in [MeetingTrack.mic, .system] where !job.progress[track].done {
            if let span = MeetingSpanPlanner.next(
                track: track, chunks: chunks(track, of: job.metadata),
                cursor: job.progress[track].cursor, rate: job.metadata.sampleRate)
            {
                let spanID = "\(job.id):\(track.rawValue):\(span.startSample)"
                job.inFlight = (spanID, span)
                current = job
                jobs[job.id] = .running(Self.fraction(job.progress, metadata: job.metadata))
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
            job.progress[track].done = true
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
        if let error = ev["error"] as? String {
            fail(job, message: error)
            return
        }
        let sentences = Self.sentences(ev["sentences"])
        let (kept, cursor) = MeetingSpanPlanner.advance(
            inFlight.span, sentences: sentences, rate: job.metadata.sampleRate)
        let track = inFlight.span.track
        job.progress[track].sentences += kept
        job.progress[track].cursor = cursor
        if let model = ev["model"] as? String, !model.isEmpty { job.progress.model = model }
        job.spans += 1
        do {
            try job.progress.save(to: job.output)
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
            return TimedSentence(start: start, end: max(start, end), text: text)
        }
    }

    private func fail(_ job: Job, message: String) {
        var progress = job.progress
        progress.error = message
        try? progress.save(to: job.output)
        NSLog("phonon meeting transcript: \(job.id) failed after \(job.spans) spans")
        current = nil
        jobs[job.id] = .failed(message)
        startNext()
    }

    private func finish(_ job: Job) {
        let transcript = TurnMerger.transcript(job.progress, metadata: job.metadata)
        do {
            try transcript.save(to: job.output)
            try? FileManager.default.removeItem(
                at: job.output.appendingPathComponent(MeetingTranscriptProgress.fileName))
        } catch {
            fail(job, message: "Could not save the transcript: \(error.localizedDescription)")
            return
        }
        NSLog(
            "phonon meeting transcript: \(job.id) done: \(transcript.turns.count) turns, "
                + "\(transcript.wordCount) words, \(job.spans) spans in "
                + String(format: "%.1f s", Date().timeIntervalSince(job.started)))
        current = nil
        jobs[job.id] = nil
        onFinished?(job.id)
        startNext()
    }
}
