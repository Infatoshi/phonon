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

    // MARK: Dictations during the call (GOAL N9)

    private func word(_ start: Double, _ text: String) -> TimedWord {
        TimedWord(start: start, end: start + 0.4, text: text)
    }

    func testDictatedWordsLeaveTheSentenceAndTheRestStays() {
        let mixed = TimedSentence(
            start: 8, end: 12.4, text: "so we ship. Remind me Friday",
            words: [word(8, "so"), word(8.5, "we"), word(9, "ship."), word(11, "Remind"), word(11.5, "me"), word(12, "Friday")])
        let untouched = TimedSentence(
            start: 20, end: 21.4, text: "Sounds good.", words: [word(20, "Sounds"), word(21, "good.")])
        let noTimes = TimedSentence(start: 30.5, end: 31.5, text: "Dear team,")
        let (kept, removed) = DictationFilter.remove(
            [mixed, untouched, noTimes], windows: [10.8...12.5, 30...32])
        XCTAssertEqual(removed, 3 + 2)
        XCTAssertEqual(kept.map(\.text), ["so we ship.", "Sounds good."])
        XCTAssertEqual(kept[0].start, 8)
        XCTAssertEqual(kept[0].end, 9.4)
        XCTAssertEqual(kept[1], untouched, "a sentence no dictation touches keeps its text exactly")
        // the pad: a word whose middle is 0.2 s before the key went down goes too
        XCTAssertEqual(DictationFilter.remove([mixed], windows: [11.4...12.5]).removedWords, 3)
        XCTAssertEqual(DictationFilter.remove([mixed], windows: []).kept, [mixed])
    }

    func testDictationHostTimesLandOnTheMicTrackAcrossAReanchor() {
        var metadata = MeetingMetadata(id: "m", startedAt: "", startedHostNs: 1_000_000_000)
        metadata.sampleRate = 16_000
        metadata.micChunks = [
            MeetingChunk(index: 0, file: "0000.wav", startSample: 0, sampleCount: 80_000, startHostNs: 2_000_000_000),
            MeetingChunk(index: 1, file: "0001.wav", startSample: 80_000, sampleCount: 80_000, startHostNs: 7_000_000_000),
            // the clock re-anchored: this chunk's first sample was captured 60 s later than the count says
            MeetingChunk(
                index: 2, file: "0002.wav", startSample: 160_000, sampleCount: 80_000, startHostNs: 72_000_000_000),
        ]
        XCTAssertEqual(metadata.micSeconds(hostNs: 3_500_000_000)!, 1.5, accuracy: 1e-9)
        XCTAssertEqual(metadata.micSeconds(hostNs: 8_000_000_000)!, 6, accuracy: 1e-9)
        XCTAssertEqual(metadata.micSeconds(hostNs: 73_000_000_000)!, 11, accuracy: 1e-9)
        XCTAssertEqual(metadata.micSeconds(hostNs: 1_500_000_000)!, -0.5, accuracy: 1e-9, "before the track")
        metadata.dictations = [MeetingDictation(startHostNs: 3_000_000_000, endHostNs: 4_000_000_000)]
        XCTAssertEqual(metadata.dictationWindows(), [1...2])
        XCTAssertNil(MeetingMetadata(id: "e", startedAt: "", startedHostNs: 0).micSeconds(hostNs: 5))
    }

    func testTheIndexRecordsDictationTimesOnly() throws {
        let index = try MeetingIndex(
            directory: directory, metadata: MeetingMetadata(id: "m", startedAt: "", startedHostNs: 0))
        index.dictation(MeetingDictation(startHostNs: 10, endHostNs: 20))
        index.dictation(MeetingDictation(startHostNs: 30, endHostNs: 45))
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("metadata.json")))
                as? [String: Any])
        XCTAssertEqual(
            object["dictations"] as? [[String: Int]],
            [["start_host_ns": 10, "end_host_ns": 20], ["start_host_ns": 30, "end_host_ns": 45]])
    }

    /// The N9 count at the logic level: a 5-minute call with 20 dictations of 3 s each. The
    /// stand-in engine "hears" a word every 0.5 s on both tracks, named by its second and by
    /// whether the user was dictating then; sentences of 8 words run across dictation edges.
    @MainActor
    func testTwentyDictationsDuringACallAreAllLeftOutOfTheMeTrack() throws {
        let base: UInt64 = 50_000_000_000
        let folder = directory.appendingPathComponent("Meetings/meeting_n9", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        var metadata = MeetingMetadata(id: "meeting_n9", startedAt: "2026-11-10T12:00:00Z", startedHostNs: base)
        metadata.sampleRate = 1_000
        metadata.micChunks = chunks(60).map {
            var chunk = $0
            chunk.startHostNs = base + UInt64(chunk.startSample) * 1_000_000
            return chunk
        }
        metadata.systemChunks = metadata.micChunks
        metadata.micStartOffsetSamples = 0
        metadata.systemStartOffsetSamples = 0
        let windows: [ClosedRange<Double>] = (0..<20).map { (index: Int) -> ClosedRange<Double> in
            let start = Double(10 + 14 * index)
            return start...(start + 3)
        }
        metadata.dictations = windows.map { (window: ClosedRange<Double>) -> MeetingDictation in
            let startNs: UInt64 = base + UInt64(window.lowerBound * 1e9)
            let endNs: UInt64 = base + UInt64(window.upperBound * 1e9)
            return MeetingDictation(startHostNs: startNs, endHostNs: endNs)
        }
        try JSONEncoder().encode(metadata).write(to: folder.appendingPathComponent("metadata.json"))

        func dictated(_ middle: Double) -> Int? {
            windows.firstIndex { $0.contains(middle) }
        }
        func answer(_ command: [String: Any]) -> [String: Any] {
            let id = command["id"] as! String
            let isMic = id.contains(":mic:")
            let start = Double(command["start_sample"] as! Int64) / 1_000
            let end = Double(command["end_sample"] as! Int64) / 1_000
            var words: [[String: Any]] = []
            var at = (start * 2).rounded(.up) / 2
            while at + 0.4 <= end {
                let middle = at + 0.2
                let name =
                    !isMic ? "c\(Int(at * 10))" : dictated(middle).map { "d\($0)x\(Int(at * 10))" } ?? "m\(Int(at * 10))"
                words.append(["start": at - start, "end": at - start + 0.4, "text": name])
                at += 0.5
            }
            let sentences = stride(from: 0, to: words.count, by: 8).map { from -> [String: Any] in
                let group = Array(words[from..<min(words.count, from + 8)])
                return [
                    "start": group.first!["start"]!, "end": group.last!["end"]!,
                    "text": group.map { $0["text"] as! String }.joined(separator: " "), "words": group,
                ]
            }
            return ["type": "meeting_span", "id": id, "sentences": sentences, "model": "test-model"]
        }

        var sent: [[String: Any]] = []
        let transcriber = MeetingTranscriber()
        transcriber.send = { sent.append($0) }
        var finished = false
        transcriber.onFinished = { _ in finished = true }
        transcriber.setEngineReady(true)
        transcriber.enqueue(directory: folder)
        var answered = 0
        while answered < sent.count, answered < 50 {
            transcriber.handle(answer(sent[answered]))
            answered += 1
        }
        XCTAssertTrue(finished)
        let transcript = try XCTUnwrap(MeetingTranscript.load(from: folder))
        let me = transcript.turns.filter { $0.speaker == .me }.flatMap { $0.text.split(separator: " ").map(String.init) }
        let call = transcript.turns.filter { $0.speaker == .call }.flatMap { $0.text.split(separator: " ").map(String.init) }
        let removed = (0..<20).filter { index in !me.contains { $0.hasPrefix("d\(index)x") } }
        XCTAssertEqual(removed.count, 20, "every dictation is out of the \"me\" track")
        // what the user said to the call stays, except a word touching a dictation's edge
        let meetingWords = Set((0..<600).map { Double($0) / 2 }.filter { dictated($0 + 0.2) == nil }.map { "m\(Int($0 * 10))" })
        let lost = meetingWords.subtracting(me)
        XCTAssertLessThanOrEqual(lost.count, 20, "at most one call word per dictation, at its end")
        XCTAssertTrue(lost.allSatisfy { name in
            let middle = Double(name.dropFirst())! / 10 + 0.2
            return windows.contains { abs(middle - $0.lowerBound) <= 0.25 || abs(middle - $0.upperBound) <= 0.25 }
        })
        // the call's side is never filtered, even while the user dictates
        XCTAssertTrue(Set((0..<600).map { "c\($0 * 5)" }).isSubset(of: Set(call)))
        print("N9 logic check: \(removed.count) of 20 dictations removed, \(lost.count) of the user's call words lost at dictation edges")
    }
}
