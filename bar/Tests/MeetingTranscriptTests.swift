import XCTest

@testable import PhononBar

final class MeetingTranscriptTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("phonon-transcript-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func sentence(_ start: Double, _ end: Double, _ text: String) -> TimedSentence {
        TimedSentence(start: start, end: end, text: text)
    }

    /// Chunks of `seconds` each at `rate`, contiguous from `from` (in samples).
    private func chunks(
        _ count: Int, seconds: Double = 5, rate: Int = 1_000, from: Int64 = 0, firstIndex: Int = 0,
        last: Double? = nil
    ) -> [MeetingChunk] {
        var out: [MeetingChunk] = []
        var at = from
        for index in 0..<count {
            let length = Int64(((index == count - 1 ? last : nil) ?? seconds) * Double(rate))
            out.append(
                MeetingChunk(
                    index: firstIndex + index, file: ChunkedWavWriter.fileName(firstIndex + index),
                    startSample: at, sampleCount: length))
            at += length
        }
        return out
    }

    // MARK: Merging

    func testTracksMergeByTimeAndASpeakersSentencesJoinIntoOneTurn() {
        let mic = [sentence(0.5, 2, "Hi all."), sentence(2.2, 4, "Can you hear me?"), sentence(9, 10, "Great.")]
        let call = [sentence(4.5, 6, "Yes."), sentence(6.5, 8.5, "Loud and clear."), sentence(12, 13, "Bye.")]

        let turns = TurnMerger.turns(mic: mic, call: call)

        XCTAssertEqual(turns.map(\.speaker), [.me, .call, .me, .call])
        XCTAssertEqual(turns[0].text, "Hi all. Can you hear me?")
        XCTAssertEqual(turns[0].startSeconds, 0.5)
        XCTAssertEqual(turns[0].endSeconds, 4)
        XCTAssertEqual(turns[1].text, "Yes. Loud and clear.")
        XCTAssertEqual(turns[1].endSeconds, 8.5)
        XCTAssertEqual(turns.last?.text, "Bye.")
        XCTAssertEqual(TurnMerger.turns(mic: [sentence(1, 2, "  ")], call: []), [])
    }

    func testTheTranscriptPutsEachTrackOnTheSharedTimelineByItsOffset() {
        var metadata = MeetingMetadata(id: "m", startedAt: "2026-11-09T12:00:00Z", startedHostNs: 0)
        metadata.sampleRate = 48_000
        // as the recorder writes them: the mic's sample 0 came 2 s before the call's
        metadata.micStartOffsetSamples = 48_000
        metadata.systemStartOffsetSamples = 144_000
        var progress = MeetingTranscriptProgress()
        progress.model = "parakeet"
        progress.mic.sentences = [sentence(1, 2, "First."), sentence(5, 6, "Third.")]
        progress.system.sentences = [sentence(1, 2, "Second.")]  // 3 s on the timeline

        let transcript = TurnMerger.transcript(
            progress, metadata: metadata, createdAt: Date(timeIntervalSince1970: 0))

        XCTAssertEqual(transcript.turns.map(\.text), ["First.", "Second.", "Third."])
        XCTAssertEqual(transcript.turns.map(\.startSeconds), [1, 3, 5])
        XCTAssertEqual(transcript.model, "parakeet")
        XCTAssertEqual(transcript.createdAt, "1970-01-01T00:00:00Z")
        XCTAssertEqual(transcript.plainText, "0:01 You: First.\n\n0:03 Call: Second.\n\n0:05 You: Third.")
        XCTAssertEqual(transcript.wordCount, 3)
    }

    // MARK: Planning

    func testSpansAreAboutAMinuteAndAShortRemainderJoinsTheLastOne() throws {
        let track = chunks(20)  // 100 s
        let first = try XCTUnwrap(MeetingSpanPlanner.next(track: .mic, chunks: track, cursor: 0, rate: 1_000))
        XCTAssertEqual(first.startSample, 0)
        XCTAssertEqual(first.endSample, 60_000)
        XCTAssertFalse(first.reachesRunEnd)
        XCTAssertEqual(first.chunks.map(\.index), Array(0..<12))

        let second = try XCTUnwrap(
            MeetingSpanPlanner.next(track: .mic, chunks: track, cursor: 57_300, rate: 1_000))
        XCTAssertEqual(second.startSample, 57_300)
        XCTAssertEqual(second.endSample, 100_000)  // 42.7 s left: one span
        XCTAssertTrue(second.reachesRunEnd)
        XCTAssertEqual(second.chunks.first?.index, 11)
        XCTAssertNil(MeetingSpanPlanner.next(track: .mic, chunks: track, cursor: 100_000, rate: 1_000))
    }

    func testAShortLastChunkEndsTheTrackAndNeverPushesASpanPastTheLimit() throws {
        // 14 chunks of 5 s and a last one of 2.3 s: 72.3 s, one span
        let short = chunks(15, last: 2.3)
        let whole = try XCTUnwrap(MeetingSpanPlanner.next(track: .system, chunks: short, cursor: 0, rate: 1_000))
        XCTAssertEqual(whole.endSample, 72_300)
        XCTAssertTrue(whole.reachesRunEnd)

        // 82.3 s: over the 75 s limit, so 60 s and then the rest
        let long = chunks(17, last: 2.3)
        let first = try XCTUnwrap(MeetingSpanPlanner.next(track: .system, chunks: long, cursor: 0, rate: 1_000))
        XCTAssertEqual(first.endSample, 60_000)
        let rest = try XCTUnwrap(
            MeetingSpanPlanner.next(track: .system, chunks: long, cursor: 60_000, rate: 1_000))
        XCTAssertEqual(rest.endSample, 82_300)
        XCTAssertLessThanOrEqual(
            Double(rest.endSample - rest.startSample) / 1_000, MeetingSpanPlanner.maxSeconds)
    }

    func testAGapInTheChunksEndsASpanAndTheNextStartsAfterIt() throws {
        // 0-20 s, then nothing until 30 s, then 30-40 s
        let track = chunks(4) + chunks(2, from: 30_000, firstIndex: 4)
        let before = try XCTUnwrap(MeetingSpanPlanner.next(track: .mic, chunks: track, cursor: 0, rate: 1_000))
        XCTAssertEqual(before.endSample, 20_000)
        XCTAssertTrue(before.reachesRunEnd)
        let after = try XCTUnwrap(
            MeetingSpanPlanner.next(track: .mic, chunks: track, cursor: 20_000, rate: 1_000))
        XCTAssertEqual(after.startSample, 30_000)
        XCTAssertEqual(after.endSample, 40_000)
        XCTAssertEqual(after.chunks.map(\.index), [4, 5])
        // chunks out of order and empty ones are fine
        let shuffled = track.reversed() + [MeetingChunk(index: 9, file: "x", startSample: 50_000, sampleCount: 0)]
        XCTAssertEqual(
            MeetingSpanPlanner.next(track: .mic, chunks: Array(shuffled), cursor: 0, rate: 1_000), before)
        XCTAssertNil(MeetingSpanPlanner.next(track: .mic, chunks: [], cursor: 0, rate: 1_000))
    }

    func testASentenceRunningIntoTheCutIsTranscribedAgainByTheNextSpan() throws {
        let span = try XCTUnwrap(
            MeetingSpanPlanner.next(track: .mic, chunks: chunks(20), cursor: 10_000, rate: 1_000))
        let heard = [sentence(1, 20, "Kept."), sentence(30, 45, "Also kept."), sentence(52, 59.6, "Cut off at")]

        let (kept, cursor) = MeetingSpanPlanner.advance(span, sentences: heard, rate: 1_000)

        XCTAssertEqual(kept.map(\.text), ["Kept.", "Also kept."])
        XCTAssertEqual(kept.first?.start, 11)  // track seconds
        XCTAssertEqual(cursor, 10_000 + 51_850)  // 0.15 s before the sentence it gives back

        // a last sentence well before the cut, or one that began near the start, stays
        let quiet = MeetingSpanPlanner.advance(span, sentences: [sentence(1, 50, "Done.")], rate: 1_000)
        XCTAssertEqual(quiet.kept.count, 1)
        XCTAssertEqual(quiet.cursor, span.endSample)
        let monologue = MeetingSpanPlanner.advance(span, sentences: [sentence(2, 60, "On and on")], rate: 1_000)
        XCTAssertEqual(monologue.kept.count, 1)
        XCTAssertEqual(monologue.cursor, span.endSample)
        // silence: nothing kept, on to the next minute
        XCTAssertEqual(MeetingSpanPlanner.advance(span, sentences: [], rate: 1_000).cursor, span.endSample)

        // the last span of a run keeps everything
        let end = try XCTUnwrap(
            MeetingSpanPlanner.next(track: .mic, chunks: chunks(20), cursor: 60_000, rate: 1_000))
        let last = MeetingSpanPlanner.advance(end, sentences: [sentence(30, 40, "Bye")], rate: 1_000)
        XCTAssertEqual(last.kept.map(\.start), [90])
        XCTAssertEqual(last.cursor, 100_000)
    }

    // MARK: Files

    func testTranscriptJSONDecodes() throws {
        let json = """
            {"schema_version":1,"model":"mlx-community/parakeet-tdt-0.6b-v2",
             "created_at":"2026-11-09T12:00:00Z",
             "turns":[{"start_seconds":0.5,"end_seconds":3.25,"speaker":"me","text":"Morning."},
                      {"start_seconds":4,"end_seconds":6,"speaker":"call","text":"Hi there."}]}
            """
        try Data(json.utf8).write(to: directory.appendingPathComponent(MeetingTranscript.fileName))

        let transcript = try XCTUnwrap(MeetingTranscript.load(from: directory))

        XCTAssertEqual(transcript.schemaVersion, 1)
        XCTAssertEqual(transcript.turns.count, 2)
        XCTAssertEqual(transcript.turns[0].speaker, .me)
        XCTAssertEqual(transcript.turns[1].speaker, .call)
        XCTAssertEqual(transcript.turns[0].endSeconds, 3.25)
        // and it round-trips under the same keys
        try transcript.save(to: directory)
        let saved = try String(contentsOf: directory.appendingPathComponent(MeetingTranscript.fileName))
        XCTAssertTrue(saved.contains("\"start_seconds\""))
        XCTAssertEqual(MeetingTranscript.load(from: directory), transcript)
        XCTAssertNil(MeetingTranscript.load(from: directory.appendingPathComponent("none")))
    }

    // MARK: Notes states

    func testNotesShowsTheTranscriptRunningOrTheButton() {
        let transcript = MeetingTranscript(model: "m", createdAt: "", turns: [])
        XCTAssertEqual(
            NoteTranscriptState.resolve(transcript: transcript, job: .queued, savedError: "x", recording: false),
            .ready(transcript))
        XCTAssertEqual(
            NoteTranscriptState.resolve(transcript: nil, job: nil, savedError: nil, recording: true), .recording)
        XCTAssertEqual(
            NoteTranscriptState.resolve(transcript: nil, job: .queued, savedError: nil, recording: false),
            .transcribing(progress: nil))
        XCTAssertEqual(
            NoteTranscriptState.resolve(transcript: nil, job: .running(0.4), savedError: nil, recording: false),
            .transcribing(progress: 0.4))
        XCTAssertEqual(
            NoteTranscriptState.resolve(transcript: nil, job: .failed("no model"), savedError: nil, recording: false),
            .failed("no model"))
        // after a relaunch the saved error is what is left
        XCTAssertEqual(
            NoteTranscriptState.resolve(transcript: nil, job: nil, savedError: "disk full", recording: false),
            .failed("disk full"))
        XCTAssertEqual(
            NoteTranscriptState.resolve(transcript: nil, job: nil, savedError: nil, recording: false),
            .notTranscribed)
    }

    // MARK: The transcriber

    /// A meeting folder: metadata only (the stand-in engine never reads audio). Mic 130 s,
    /// call 40 s, the call's sample 0 one second after the mic's.
    private func writeMeeting(_ id: String) throws -> URL {
        let folder = directory.appendingPathComponent("Meetings/\(id)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        var metadata = MeetingMetadata(id: id, startedAt: "2026-11-09T12:00:00Z", startedHostNs: 0)
        metadata.sampleRate = 1_000
        metadata.micChunks = chunks(26)
        metadata.systemChunks = chunks(8)
        metadata.micStartOffsetSamples = 0
        metadata.systemStartOffsetSamples = 1_000
        metadata.endedAt = "2026-11-09T12:02:10Z"
        try JSONEncoder().encode(metadata).write(to: folder.appendingPathComponent("metadata.json"))
        return folder
    }

    /// Answers each span with one sentence per 10 s of its audio, the last one ending 0.5 s
    /// before the span does.
    @MainActor
    private func answer(_ command: [String: Any]) -> [String: Any] {
        let start = command["start_sample"] as! Int64
        let end = command["end_sample"] as! Int64
        let seconds = Double(end - start) / 1_000
        var sentences: [[String: Any]] = []
        var at = 0.0
        while at + 10 <= seconds + 0.001 {
            let close = min(at + 9.5, seconds - 0.5)
            sentences.append(["start": at + 0.2, "end": close, "text": "Words at \(Int(Double(start) / 1_000 + at))."])
            at += 10
        }
        return ["type": "meeting_span", "id": command["id"] as! String, "sentences": sentences, "model": "test-model"]
    }

    @MainActor
    func testATranscriptionResumesAfterAQuitAndFinishesTheFile() throws {
        let folder = try writeMeeting("meeting_1")
        var sent: [[String: Any]] = []
        let first = MeetingTranscriber()
        first.send = { sent.append($0) }
        first.enqueue(directory: folder)
        XCTAssertTrue(sent.isEmpty, "nothing goes out before the engine is ready")
        XCTAssertEqual(first.jobs["meeting_1"], .queued)
        first.setEngineReady(true)
        XCTAssertEqual(sent.count, 1)
        XCTAssertEqual(sent[0]["cmd"] as? String, "meeting_span")
        XCTAssertEqual(sent[0]["end_sample"] as? Int64, 60_000)
        let paths = (sent[0]["chunks"] as? [[String: Any]])?.compactMap { $0["path"] as? String } ?? []
        XCTAssertEqual(paths.count, 12)
        XCTAssertTrue(paths[0].hasSuffix("meeting_1/mic/0000.wav"))

        // a stale or foreign answer changes nothing
        first.handle(["type": "meeting_span", "id": "meeting_1:mic:999", "sentences": []])
        XCTAssertEqual(sent.count, 1)

        first.handle(answer(sent[0]))
        XCTAssertEqual(sent.count, 2)
        // the sixth sentence ended 0.5 s before the cut, so the next span starts before it
        XCTAssertEqual(sent[1]["start_sample"] as? Int64, 50_050)
        let saved = try XCTUnwrap(MeetingTranscriptProgress.load(from: folder))
        XCTAssertEqual(saved.mic.cursor, 50_050)
        XCTAssertEqual(saved.mic.sentences.count, 5)
        if case .running(let fraction) = first.jobs["meeting_1"] {
            XCTAssertEqual(fraction, 50_050.0 / 170_000.0, accuracy: 1e-9)
        } else {
            XCTFail("running")
        }

        // the app quits with that span in flight; the next launch resumes from the saved cursor
        var resumed: [[String: Any]] = []
        let second = MeetingTranscriber()
        second.send = { resumed.append($0) }
        second.setEngineReady(true)
        second.resumePending(in: directory.appendingPathComponent("Meetings"))
        XCTAssertEqual(resumed.first?["start_sample"] as? Int64, 50_050)
        var finished: [String] = []
        second.onFinished = { finished.append($0) }
        var guardCount = 0
        while let command = resumed.last, finished.isEmpty, guardCount < 20 {
            second.handle(answer(command))
            guardCount += 1
            if resumed.last.map({ ($0["id"] as? String) == (command["id"] as? String) }) == true { break }
        }
        XCTAssertEqual(finished, ["meeting_1"])
        XCTAssertNil(second.jobs["meeting_1"])
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: folder.appendingPathComponent(MeetingTranscriptProgress.fileName).path))
        let transcript = try XCTUnwrap(MeetingTranscript.load(from: folder))
        XCTAssertEqual(transcript.model, "test-model")
        // every 10 s of the mic once (13), and the call's four on the timeline 1 s later
        let words = transcript.turns.map(\.text).joined(separator: " ")
        // (the mic's last span is 29.9 s: its sentences start at 100 and 110)
        for second in stride(from: 0, through: 110, by: 10) {
            XCTAssertEqual(words.components(separatedBy: "Words at \(second).").count - 1, second < 40 ? 2 : 1, "\(second)")
        }
        XCTAssertEqual(transcript.turns.first?.speaker, .me)
        XCTAssertTrue(second.isIdle)

        // done is done: asking again sends nothing
        second.enqueue(directory: folder)
        XCTAssertEqual(second.jobs["meeting_1"], nil)
    }

    @MainActor
    func testAFailureIsSavedAndTheButtonRetriesFromWhereItStopped() throws {
        let folder = try writeMeeting("meeting_2")
        var sent: [[String: Any]] = []
        let transcriber = MeetingTranscriber()
        transcriber.send = { sent.append($0) }
        transcriber.setEngineReady(true)
        transcriber.enqueue(directory: folder)
        transcriber.handle(answer(sent[0]))
        transcriber.handle(["type": "meeting_span", "id": sent[1]["id"] as! String, "error": "transcribe span failed"])

        XCTAssertEqual(transcriber.jobs["meeting_2"], .failed("transcribe span failed"))
        let saved = try XCTUnwrap(MeetingTranscriptProgress.load(from: folder))
        XCTAssertEqual(saved.error, "transcribe span failed")
        XCTAssertEqual(saved.mic.cursor, 50_050)
        let notes = MeetingNote.loadAll(from: directory.appendingPathComponent("Meetings"))
        XCTAssertEqual(notes.first?.transcriptError, "transcribe span failed")

        // a failed one is not resumed by itself at launch
        let relaunched = MeetingTranscriber()
        var resent: [[String: Any]] = []
        relaunched.send = { resent.append($0) }
        relaunched.setEngineReady(true)
        relaunched.resumePending(in: directory.appendingPathComponent("Meetings"))
        XCTAssertTrue(resent.isEmpty)
        // the button tries again from the saved cursor, error cleared
        relaunched.enqueue(directory: folder)
        XCTAssertEqual(resent.first?["start_sample"] as? Int64, 50_050)
        XCTAssertNil(MeetingTranscriptProgress.load(from: folder)?.error)
    }

    @MainActor
    func testAnEngineRestartResendsTheSpanInFlight() throws {
        let folder = try writeMeeting("meeting_3")
        var sent: [[String: Any]] = []
        let transcriber = MeetingTranscriber()
        transcriber.send = { sent.append($0) }
        transcriber.setEngineReady(true)
        transcriber.enqueue(directory: folder)
        transcriber.setEngineReady(false)
        transcriber.setEngineReady(true)
        XCTAssertEqual(sent.count, 2)
        XCTAssertEqual(sent[0]["id"] as? String, sent[1]["id"] as? String)
    }

    func testAQuitDuringRecordingLeavesAMarkerTheNextLaunchPicksUp() throws {
        let folder = try writeMeeting("meeting_4")
        MeetingTranscriber.markPending(directory: folder)
        let marker = try XCTUnwrap(MeetingTranscriptProgress.load(from: folder))
        XCTAssertEqual(marker.mic.cursor, 0)
        XCTAssertNil(marker.error)
        // a second mark keeps the progress already there
        var progressed = marker
        progressed.mic.cursor = 5_000
        try progressed.save(to: folder)
        MeetingTranscriber.markPending(directory: folder)
        XCTAssertEqual(MeetingTranscriptProgress.load(from: folder)?.mic.cursor, 5_000)
    }

    @MainActor
    func testAStoppedRecordingIsQueuedAndListedOnlyOnceItsFilesAreClosed() throws {
        let store = NativeAppStore(supportDirectory: directory)
        XCTAssertTrue(store.meetings.isEmpty)
        var queued: [String] = []
        store.onTranscribeMeeting = { queued.append($0.lastPathComponent) }
        let folder = try writeMeeting("meeting_6")
        // recording (or still closing its last chunks): no transcription from the button
        store.recordingMeetingID = "meeting_6"
        let note = MeetingNote(metadata: MeetingMetadata(id: "meeting_6", startedAt: "", startedHostNs: 0), directory: folder)
        XCTAssertEqual(store.transcriptState(note), .recording)
        store.transcribeMeeting(note)
        XCTAssertTrue(queued.isEmpty)

        // the recorder's stop finished: queued, listed, no longer "recording"
        store.meetingRecordingFinished(directory: folder)
        XCTAssertEqual(queued, ["meeting_6"])
        XCTAssertNil(store.recordingMeetingID)
        XCTAssertEqual(store.meetings.map(\.id), ["meeting_6"])
        store.meetingTranscription["meeting_6"] = .queued
        XCTAssertEqual(store.transcriptState(store.meetings[0]), .transcribing(progress: nil))

        // an older recording finishing its stop never clears a newer one
        store.recordingMeetingID = "meeting_7"
        store.meetingRecordingFinished(directory: folder)
        XCTAssertEqual(store.recordingMeetingID, "meeting_7")
        store.transcribeMeeting(store.meetings[0])
        XCTAssertEqual(queued, ["meeting_6", "meeting_6", "meeting_6"])
    }

    @MainActor
    func testTheStopMarkerIsResumedAndGoesOnlyWithTheTranscript() throws {
        let folder = try writeMeeting("meeting_8")
        let marker = folder.appendingPathComponent(MeetingTranscriptProgress.fileName)
        // written at stop, before the files close; the app quits before anything is queued
        MeetingTranscriber.markPending(directory: folder)
        var sent: [[String: Any]] = []
        let transcriber = MeetingTranscriber()
        transcriber.send = { sent.append($0) }
        transcriber.setEngineReady(true)
        transcriber.resumePending(in: directory.appendingPathComponent("Meetings"))
        XCTAssertEqual(sent.first?["start_sample"] as? Int64, 0)
        var finished = false
        transcriber.onFinished = { _ in finished = true }
        while let command = sent.last, !finished, sent.count < 20 {
            XCTAssertTrue(FileManager.default.fileExists(atPath: marker.path), "kept while it runs")
            let before = sent.count
            transcriber.handle(answer(command))
            if sent.count == before { break }
        }
        XCTAssertTrue(finished)
        XCTAssertFalse(FileManager.default.fileExists(atPath: marker.path))
        XCTAssertNotNil(MeetingTranscript.load(from: folder))
    }

    @MainActor
    func testOutputCanGoElsewhereThanTheMeetingFolder() throws {
        let folder = try writeMeeting("meeting_5")
        let elsewhere = directory.appendingPathComponent("out", isDirectory: true)
        var sent: [[String: Any]] = []
        let transcriber = MeetingTranscriber()
        transcriber.outputDirectory = { elsewhere.appendingPathComponent($0.lastPathComponent) }
        transcriber.send = { sent.append($0) }
        transcriber.setEngineReady(true)
        transcriber.enqueue(directory: folder)
        XCTAssertNotNil(MeetingTranscriptProgress.load(from: elsewhere.appendingPathComponent("meeting_5")))
        XCTAssertNil(MeetingTranscriptProgress.load(from: folder))
    }
}
