import XCTest

@testable import PhononBar

final class MeetingCleanupTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("phonon-cleanup-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private static let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()

    /// `count` words, sentences of `sentence` words, each word distinct ("w<seed>x<i>").
    private func words(_ count: Int, seed: Int, sentence: Int = 10) -> String {
        (0..<count).map { index in
            let word = "w\(seed)x\(index)"
            return (index + 1) % sentence == 0 || index == count - 1 ? word + "." : word
        }.joined(separator: " ")
    }

    private func turn(_ speaker: TranscriptSpeaker, _ text: String, at start: Double = 0) -> MeetingTranscript.Turn {
        MeetingTranscript.Turn(startSeconds: start, endSeconds: start + 1, speaker: speaker, text: text)
    }

    // MARK: Chunking

    func testLongTurnsSplitAtASentenceEndAfter120WordsOrAt160() {
        let lines = MeetingCleanupChunker.lines([
            turn(.call, words(250, seed: 1)), turn(.me, words(170, seed: 2, sentence: 1_000)),
        ])
        // 250 words, sentence ends every 10: 120 + 120 + 10
        XCTAssertEqual(lines.map(\.wordCount), [120, 120, 10, 160, 10])
        XCTAssertEqual(lines.map(\.turn), [0, 0, 0, 1, 1])
        // labels by first appearance in the meeting: the call spoke first here
        XCTAssertEqual(Set(lines.filter { $0.turn == 0 }.map(\.label)), ["Speaker 1"])
        XCTAssertEqual(Set(lines.filter { $0.turn == 1 }.map(\.label)), ["Speaker 2"])
    }

    func testChunksAreWholeLinesOf120To220WordsClosedAtASentenceEnd() {
        var turns: [MeetingTranscript.Turn] = []
        for index in 0..<40 {
            turns.append(turn(index % 2 == 0 ? .me : .call, words(17 + index % 9, seed: index, sentence: 6), at: Double(index)))
        }
        let chunks = MeetingCleanupChunker.chunks(turns)
        XCTAssertGreaterThan(chunks.count, 3)
        for chunk in chunks.dropLast() {
            XCTAssertGreaterThanOrEqual(chunk.wordCount, 120)
            XCTAssertLessThanOrEqual(chunk.wordCount, 220)
            XCTAssertTrue(MeetingCleanupChunker.endsSentence(chunk.lines.last!.text))
        }
        // every turn's words once, in order
        let all = chunks.flatMap(\.lines)
        XCTAssertEqual(all.map(\.turn), Array(0..<40))
        XCTAssertEqual(all.map(\.text), turns.map(\.text))
        XCTAssertEqual(chunks.map(\.index), Array(0..<chunks.count))
        // the user turn the adapter was trained on
        XCTAssertEqual(chunks[0].prompt.components(separatedBy: "\n").first, "Speaker 1: " + turns[0].text)
        XCTAssertEqual(chunks[0].maxTokens, Int(2 * Double(chunks[0].wordCount) * 1.4) + 96)
    }

    func testARunThatCannotReach120WordsBecomesAShortChunkInsteadOfBeingDropped() {
        // 100 words with no sentence end, then a 150-word line: training would drop the 100
        let turns = [turn(.me, words(100, seed: 1, sentence: 1_000)), turn(.call, words(150, seed: 2, sentence: 1_000))]
        let chunks = MeetingCleanupChunker.chunks(turns)
        XCTAssertEqual(chunks.map(\.wordCount), [100, 150])
        XCTAssertEqual(MeetingCleanupChunker.chunks([]), [])
    }

    // MARK: Checking answers

    private struct FixtureCase: Decodable {
        var id: String
        var input: String
        var output: String
        var verdict: String
    }

    /// A chunk from "Speaker N: words" lines (fixture input).
    private func chunk(_ text: String) -> CleanupChunk {
        let lines = text.components(separatedBy: "\n").enumerated().compactMap { index, line -> CleanupLine? in
            guard let labelled = MeetingCleanupValidator.labelled(line) else { return nil }
            return CleanupLine(turn: index, label: labelled.label, text: labelled.text)
        }
        return CleanupChunk(index: 0, lines: lines)
    }

    func testThePublicEvalAnswersGetTheReferenceValidatorsVerdicts() throws {
        struct Fixture: Decodable { var cases: [FixtureCase] }
        let data = try Data(contentsOf: Self.repo.appendingPathComponent("fixtures/meeting_cleanup_cases.json"))
        let cases = try JSONDecoder().decode(Fixture.self, from: data).cases
        XCTAssertGreaterThanOrEqual(cases.count, 8)
        for item in cases {
            let input = chunk(item.input)
            switch MeetingCleanupValidator.check(input, output: item.output) {
            case .cleaned(let lines):
                XCTAssertEqual("ok", item.verdict, item.id)
                XCTAssertEqual(lines.count, input.lines.count, item.id)
                XCTAssertFalse(lines.compactMap { $0 }.contains { $0.contains("TL;DR") }, item.id)
                XCTAssertTrue(item.output.contains("TL;DR:"), "the answer had one to strip: \(item.id)")
            case .rejected(let reason):
                XCTAssertEqual(reason, item.verdict, item.id)
            }
        }
    }

    func testTheTLDRIsStrippedAndFillerOnlyLinesMayBeDropped() {
        let input = CleanupChunk(index: 0, lines: [
            CleanupLine(turn: 0, label: "Speaker 1", text: "so um we we ship the the build on Tuesday"),
            CleanupLine(turn: 1, label: "Speaker 2", text: "Mm-hmm. Yeah."),
            CleanupLine(turn: 2, label: "Speaker 1", text: "and then uh the docs on Friday"),
        ])
        let answer = """
            **Speaker 1:** We ship the build on Tuesday.
            Speaker 1: And then the docs on Friday.

            TL;DR: Speaker 1 says the build ships Tuesday and the docs Friday.
            """
        XCTAssertEqual(
            MeetingCleanupValidator.check(input, output: answer),
            .cleaned(["We ship the build on Tuesday.", nil, "And then the docs on Friday."]))
        // a filler phrase emptied under its kept label
        XCTAssertEqual(
            MeetingCleanupValidator.check(
                input, output: "Speaker 1: We ship the build on Tuesday.\nSpeaker 2:\nSpeaker 1: And then the docs on Friday."),
            .cleaned(["We ship the build on Tuesday.", nil, "And then the docs on Friday."]))
        // no TL;DR at all is fine too
        XCTAssertEqual(
            MeetingCleanupValidator.check(
                input, output: "Speaker 1: We ship the build on Tuesday.\nSpeaker 2: Yeah.\nSpeaker 1: And then the docs on Friday."),
            .cleaned(["We ship the build on Tuesday.", "Yeah.", "And then the docs on Friday."]))
    }

    func testAnswersThatLoseMoveOrInventWordsAreRejected() {
        let input = CleanupChunk(index: 0, lines: [
            CleanupLine(turn: 0, label: "Speaker 1", text: "we ship the build on Tuesday after the review with the design team"),
            CleanupLine(turn: 1, label: "Speaker 2", text: "Okay, great."),
            CleanupLine(turn: 2, label: "Speaker 1", text: "and then the docs go out on Friday morning"),
        ])
        func reason(_ answer: String) -> String? {
            if case .rejected(let reason) = MeetingCleanupValidator.check(input, output: answer) { return reason }
            return nil
        }
        let first = "Speaker 1: We ship the build on Tuesday after the review with the design team."
        let last = "Speaker 1: And then the docs go out on Friday morning."
        XCTAssertNil(reason([first, "Speaker 2: Okay, great.", last].joined(separator: "\n")))
        XCTAssertEqual(reason([first, last].joined(separator: "\n")), "speaker order", "a line with content dropped")
        XCTAssertEqual(reason([first, "Speaker 2: Okay, great."].joined(separator: "\n")), "speaker line lost")
        XCTAssertEqual(reason([first, "Speaker 2: Okay, great.", last, last].joined(separator: "\n")), "extra speaker line")
        XCTAssertEqual(reason([first, "Okay, great.", last].joined(separator: "\n")), "unlabelled line")
        // a label kept with its words gone: fine for "you know,", not for a real line
        XCTAssertEqual(reason(["Speaker 1:", "Speaker 2: Okay, great.", last].joined(separator: "\n")), "content lost")
        XCTAssertEqual(
            reason(["Speaker 1: We ship it.", "Speaker 2: Okay, great.", "Speaker 1: Docs Friday."].joined(separator: "\n")),
            "content lost")
        XCTAssertEqual(
            reason([first, "Speaker 2: Okay, great. Also the launch party moves to the new office downtown.", last].joined(separator: "\n")),
            "words added")
        let long = Array(repeating: "We ship the build on Tuesday after the review with the design team.", count: 5)
            .joined(separator: " ")
        XCTAssertEqual(reason(["Speaker 1: " + long, "Speaker 2: Okay, great.", last].joined(separator: "\n")), "too long")
        XCTAssertEqual(MeetingCleanupValidator.tokens("Don't re-run it, Bob!"), ["don't", "re", "run", "it", "bob"])
    }

    func testALongLineCutShortIsRejectedEvenWhenTheChunkKeepsEnoughWords() {
        let long = (0..<40).map { "alpha\($0)" }.joined(separator: " ")
        let others = (0..<200).map { "beta\($0)" }.joined(separator: " ")
        let input = CleanupChunk(index: 0, lines: [
            CleanupLine(turn: 0, label: "Speaker 1", text: long),
            CleanupLine(turn: 1, label: "Speaker 2", text: others),
        ])
        let cut = (0..<15).map { "alpha\($0)" }.joined(separator: " ")
        XCTAssertEqual(
            MeetingCleanupValidator.check(input, output: "Speaker 1: \(cut)\nSpeaker 2: \(others)"), .rejected("line cut short"))
    }

    // MARK: Putting turns back together

    func testCleanedTurnsAreAssembledAndAFailedChunkKeepsItsRawLines() {
        let transcript = MeetingTranscript(model: "m", createdAt: "", turns: [
            turn(.me, "first raw", at: 0), turn(.call, "mm-hmm", at: 1), turn(.me, "split raw a split raw b", at: 2),
            turn(.call, "last raw", at: 3),
        ])
        let chunks = [
            CleanupChunk(index: 0, lines: [
                CleanupLine(turn: 0, label: "Speaker 1", text: "first raw"),
                CleanupLine(turn: 1, label: "Speaker 2", text: "mm-hmm"),
                CleanupLine(turn: 2, label: "Speaker 1", text: "split raw a"),
            ]),
            CleanupChunk(index: 1, lines: [
                CleanupLine(turn: 2, label: "Speaker 1", text: "split raw b"),
                CleanupLine(turn: 3, label: "Speaker 2", text: "last raw"),
            ]),
        ]
        let results = [
            CleanupChunkResult(chunk: 0, outcome: .cleaned(["First.", nil, "Split A."])),
            CleanupChunkResult(chunk: 1, outcome: .rejected("speaker order")),
        ]
        let cleaned = CleanedTranscript.assemble(
            transcript: transcript, chunks: chunks, results: results, runtime: "rt", adapterSHA256: "ab")
        XCTAssertEqual(cleaned.turns.map(\.text), ["First.", nil, "Split A. split raw b", "last raw"])
        XCTAssertEqual(cleaned.turns.map(\.source), [.cleaned, .dropped, .mixed, .raw])
        XCTAssertEqual(cleaned.chunks, 2)
        XCTAssertEqual(cleaned.rawChunks, 1)
        XCTAssertTrue(cleaned.matches(transcript))
        var edited = transcript
        edited.turns[0].text = "changed"
        XCTAssertFalse(cleaned.matches(edited), "a cleanup of another transcript is never shown")

        try? cleaned.save(to: directory)
        XCTAssertEqual(CleanedTranscript.load(from: directory), cleaned)
        let json = try? String(contentsOf: directory.appendingPathComponent(CleanedTranscript.fileName), encoding: .utf8)
        XCTAssertTrue(json?.contains("\"raw_chunks\" : 1") == true)
        XCTAssertFalse(json?.contains("TL;DR") == true)
    }

    // MARK: Memory gate

    func testTheMemoryGate() {
        let gib: UInt64 = 1 << 30
        XCTAssertNotNil(MeetingCleanupMemory.refusal(physical: 8 * gib, available: 6 * gib), "an 8 GB Mac never")
        XCTAssertNil(MeetingCleanupMemory.refusal(physical: 16 * gib, available: 3 * gib))
        XCTAssertNil(MeetingCleanupMemory.refusal(physical: 16 * gib, available: nil), "unknown: let it run")
        let low = MeetingCleanupMemory.refusal(physical: 16 * gib, available: 1_200_000_000)
        XCTAssertEqual(low, "Not cleaned: only 1.2 GB of memory was free (2.5 GB needed).")
        XCTAssertNotNil(MeetingCleanupMemory.availableBytes())
    }

    // MARK: Notes

    func testNotesShowsCleanedTextWhenPresentAndRawOnRequest() {
        let transcript = MeetingTranscript(model: "m", createdAt: "", turns: [
            turn(.me, "so um hi", at: 0), turn(.call, "yeah", at: 4), turn(.call, "uh hello there", at: 65),
        ])
        let cleaned = CleanedTranscript(
            source: CleanupSource(transcript), runtime: "rt", adapterSHA256: "ab", createdAt: "", chunks: 1, rawChunks: 0,
            turns: [.init(text: "Hi.", source: .cleaned), .init(text: nil, source: .dropped), .init(text: "Hello there.", source: .cleaned)])
        let state = NoteCleanupState.resolve(transcript: transcript, cleaned: cleaned, job: nil, savedError: nil)
        XCTAssertEqual(state, .cleaned(cleaned))
        let rows = NoteTranscriptRows.rows(transcript, cleanup: state, showRaw: false)
        XCTAssertEqual(rows.map(\.text), ["Hi.", "Hello there."])
        XCTAssertEqual(rows.map(\.startSeconds), [0, 65])
        XCTAssertEqual(NoteTranscriptRows.plainText(rows), "0:00 You: Hi.\n\n1:05 Call: Hello there.")
        XCTAssertEqual(
            NoteTranscriptRows.rows(transcript, cleanup: state, showRaw: true).map(\.text),
            ["so um hi", "yeah", "uh hello there"])

        // a cleanup of an older transcript, running, failed, or none
        var other = transcript
        other.turns.removeLast()
        XCTAssertEqual(NoteCleanupState.resolve(transcript: other, cleaned: cleaned, job: nil, savedError: nil), .none)
        XCTAssertEqual(
            NoteTranscriptRows.rows(other, cleanup: .none, showRaw: false).map(\.text), ["so um hi", "yeah"])
        XCTAssertEqual(
            NoteCleanupState.resolve(transcript: transcript, cleaned: nil, job: .running(0.5, paused: true), savedError: nil),
            .cleaning(progress: 0.5, paused: true))
        XCTAssertEqual(
            NoteCleanupState.resolve(transcript: transcript, cleaned: nil, job: .queued, savedError: nil),
            .cleaning(progress: nil, paused: false))
        XCTAssertEqual(
            NoteCleanupState.resolve(transcript: transcript, cleaned: nil, job: nil, savedError: "low memory"),
            .failed("low memory"))
        XCTAssertEqual(
            NoteCleanupState.resolve(transcript: transcript, cleaned: nil, job: .failed("x"), savedError: "y"), .failed("x"))
        XCTAssertEqual(NoteCleanupState.resolve(transcript: nil, cleaned: nil, job: .queued, savedError: nil), .none)
        XCTAssertEqual(NoteDetail.cleaningLabel(progress: 0.25, paused: false), "Cleaning up… 25%")
        XCTAssertEqual(NoteDetail.cleaningLabel(progress: 0.25, paused: true), "Cleaning up… (paused while you dictate)")
        XCTAssertEqual(NoteDetail.cleanupSummary(cleaned), "Cleaned on this Mac.")
    }

    @MainActor
    func testNotesLoadsTheCleanupAndItsError() throws {
        let folder = directory.appendingPathComponent("meeting_1", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        var metadata = MeetingMetadata(id: "meeting_1", startedAt: "2026-11-10T12:00:00Z", startedHostNs: 0)
        metadata.micChunks = [MeetingChunk(index: 0, file: "0000.wav", startSample: 0, sampleCount: 1_000)]
        try MeetingTranscriptFiles.write(metadata, to: folder.appendingPathComponent("metadata.json"))
        try MeetingCleanupProgress(error: "low memory").save(to: folder)
        let note = try XCTUnwrap(MeetingNote.loadAll(from: directory).first)
        XCTAssertEqual(note.cleanupError, "low memory")
        XCTAssertNil(note.cleaned)
    }

    // MARK: Manifest and runtime command line

    func testTheManifestPinsTheRefinerTheAdapterWasTrainedOn() throws {
        let paths = try String(
            contentsOf: Self.repo.appendingPathComponent("crates/phonon-llm/src/paths.rs"), encoding: .utf8)
        XCTAssertTrue(paths.contains("POLISH_MODEL_REVISION: &str = \"\(MeetingCleanupManifest.refinerRevision)\""))
        XCTAssertTrue(paths.contains("POLISH_MODEL_ID: &str = \"\(MeetingCleanupManifest.refinerRepo)\""))
        let win = try String(
            contentsOf: Self.repo.appendingPathComponent("crates/phonon-win/src/manifest.rs"), encoding: .utf8)
        XCTAssertTrue(win.contains("LLAMA_BUILD: &str = \"\(MeetingCleanupManifest.llamaBuild)\""), "the Windows/Linux build")
        for component in [MeetingCleanupManifest.runtime, MeetingCleanupManifest.adapter] {
            XCTAssertEqual(component.asset.sha256.count, 64)
            XCTAssertGreaterThan(component.asset.bytes, 0)
        }
        XCTAssertNil(MeetingCleanupManifest.adapter.asset.url, "not published yet")
        let candidates = MeetingCleanupManifest.baseCandidates(support: directory, environment: ["HF_HUB_CACHE": "/hub"])
        XCTAssertEqual(
            candidates.last?.path,
            "/hub/models--Infatoshi--phonon-refiner-1.7b/snapshots/\(MeetingCleanupManifest.refinerRevision)/"
                + MeetingCleanupManifest.refinerGGUF)
    }

    @MainActor
    func testTheRuntimeRunsOnTheCPUAtLowPriorityOnLoopback() {
        let setup = LlamaServerRuntime.Setup(
            server: URL(fileURLWithPath: "/rt/llama-server"), model: URL(fileURLWithPath: "/m/q4.gguf"),
            adapter: URL(fileURLWithPath: "/a/lora.gguf"), adapterSHA256: "ab", threads: 4,
            pidFile: URL(fileURLWithPath: "/tmp/x.pid"))
        let arguments = LlamaServerRuntime.arguments(setup, port: 12_345)
        XCTAssertEqual(Array(arguments.prefix(3)), ["-n", "15", "/rt/llama-server"])
        let joined = arguments.joined(separator: " ")
        for part in ["--device none", "-ngl 0", "--no-repack", "--lora /a/lora.gguf", "-m /m/q4.gguf", "-t 4",
                     "--host 127.0.0.1", "--port 12345", "--repeat-penalty 1.0"] {
            XCTAssertTrue(joined.contains(part), part)
        }
        XCTAssertEqual(LlamaServerRuntime.threads(cores: 16), 4)
        XCTAssertEqual(LlamaServerRuntime.threads(cores: 6), 3)
        XCTAssertGreaterThan(LlamaServerRuntime.freePort(), 0)
    }
}

// MARK: - The cleaner with a stand-in runtime

@MainActor
private final class FakeRuntime: MeetingCleanupRuntime {
    var started = 0
    var stopped = 0
    var prompts: [String] = []
    var paused = false
    var pauseCalls: [Bool] = []
    var failNext = 0
    /// Answers each prompt with its lines capitalised, plus a TL;DR.
    var answer: @MainActor (String) -> String = { prompt in
        prompt.components(separatedBy: "\n").map { line in
            guard let colon = line.firstIndex(of: ":") else { return line }
            return String(line[...colon]) + " CLEAN" + String(line[line.index(after: colon)...])
        }.joined(separator: "\n") + "\nTL;DR: something was said."
    }
    var label: String { "fake" }

    func start() async throws { started += 1 }

    func complete(system: String, user: String, maxTokens: Int) async throws -> String {
        XCTAssertFalse(paused, "never asked while paused")
        XCTAssertEqual(system, MeetingCleanupPrompt.system)
        prompts.append(user)
        if failNext > 0 {
            failNext -= 1
            throw URLError(.networkConnectionLost)
        }
        let reply = answer(user)
        // like a stopped process: an answer in flight comes back only once it runs again
        while freezeWhilePaused && paused {
            try await Task.sleep(nanoseconds: 5_000_000)
        }
        return reply
    }

    var freezeWhilePaused = false

    func setPaused(_ paused: Bool) {
        self.paused = paused
        pauseCalls.append(paused)
    }

    func stop() { stopped += 1 }
}

final class MeetingCleanerTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("phonon-cleaner-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    /// A meeting folder with a transcript of `turns` turns of about 60 words each.
    private func meeting(_ id: String, turns: Int = 10) throws -> (URL, MeetingTranscript) {
        let folder = root.appendingPathComponent(id, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let transcript = MeetingTranscript(
            model: "m", createdAt: "",
            turns: (0..<turns).map { index in
                MeetingTranscript.Turn(
                    startSeconds: Double(index * 20), endSeconds: Double(index * 20 + 19),
                    speaker: index % 2 == 0 ? .me : .call,
                    text: (0..<60).map { "t\(index)w\($0)" }.joined(separator: " ") + ".")
            })
        try transcript.save(to: folder)
        return (folder, transcript)
    }

    @MainActor
    private func cleaner(
        _ runtime: FakeRuntime, available: Bool = true, physical: UInt64 = 32 << 30, free: UInt64? = 8 << 30
    ) -> MeetingCleaner {
        MeetingCleaner(
            environment: .init(
                available: { available }, physicalMemory: { physical }, availableMemory: { free },
                makeRuntime: { (runtime, "adaptersha") }))
    }

    @MainActor
    private func waitUntil(_ what: String, _ condition: () -> Bool) async {
        for _ in 0..<400 where !condition() {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        XCTAssertTrue(condition(), what)
    }

    @MainActor
    func testACleanupRunsEveryChunkWritesTheCleanedFileAndStopsTheRuntime() async throws {
        let (folder, transcript) = try meeting("meeting_1")
        let runtime = FakeRuntime()
        let cleaner = cleaner(runtime)
        var finished: [String] = []
        cleaner.onFinished = { finished.append($0) }
        cleaner.setEngineReady(true)
        cleaner.setDictationQuiet(true)
        cleaner.enqueue(directory: folder)
        await waitUntil("finished") { finished == ["meeting_1"] }

        let chunks = MeetingCleanupChunker.chunks(transcript.turns)
        XCTAssertEqual(runtime.prompts, chunks.map(\.prompt))
        XCTAssertEqual(runtime.started, 1)
        XCTAssertGreaterThanOrEqual(runtime.stopped, 1, "not resident after the meeting")
        let cleaned = try XCTUnwrap(CleanedTranscript.load(from: folder))
        XCTAssertTrue(cleaned.matches(transcript))
        XCTAssertEqual(cleaned.rawChunks, 0)
        XCTAssertEqual(cleaned.adapterSHA256, "adaptersha")
        XCTAssertTrue(cleaned.turns.allSatisfy { $0.text?.hasPrefix("CLEAN ") == true && $0.source == .cleaned })
        XCTAssertFalse(cleaned.turns.contains { $0.text?.contains("TL;DR") == true })
        XCTAssertNil(MeetingCleanupProgress.load(from: folder), "the partial file goes")
        XCTAssertEqual(MeetingTranscript.load(from: folder), transcript, "the raw transcript is untouched")
        XCTAssertTrue(cleaner.isIdle)

        // done is done
        cleaner.enqueue(directory: folder)
        XCTAssertNil(cleaner.jobs["meeting_1"])
    }

    @MainActor
    func testDictationPausesTheRuntimeAndHoldsTheNextChunk() async throws {
        let (folder, _) = try meeting("meeting_2", turns: 12)
        let runtime = FakeRuntime()
        let cleaner = cleaner(runtime)
        cleaner.setEngineReady(true)
        // a dictation is under way when the meeting is queued
        cleaner.setDictationQuiet(false)
        cleaner.enqueue(directory: folder)
        try await Task.sleep(nanoseconds: 50_000_000)
        XCTAssertEqual(runtime.started, 0, "not even started while dictating")
        XCTAssertTrue(runtime.prompts.isEmpty)
        if case .running(_, let paused) = cleaner.jobs["meeting_2"] { XCTAssertTrue(paused) } else { XCTFail("running") }

        // the next dictation: paused at once, and nothing new is asked until it is over
        runtime.answer = { prompt in
            // a key goes down while this chunk is in flight
            cleaner.setDictationQuiet(false)
            return prompt + "\nTL;DR: x"
        }
        cleaner.setDictationQuiet(true)
        await waitUntil("first chunk") { runtime.prompts.count >= 1 }
        await waitUntil("paused") { runtime.paused }
        XCTAssertEqual(runtime.started, 1)
        let asked = runtime.prompts.count
        try await Task.sleep(nanoseconds: 50_000_000)
        XCTAssertEqual(runtime.prompts.count, asked)
        XCTAssertEqual(runtime.pauseCalls.last, true)
        runtime.answer = { $0 }
        cleaner.setDictationQuiet(true)
        XCTAssertFalse(runtime.paused)
        await waitUntil("done") { CleanedTranscript.load(from: folder) != nil }
    }

    @MainActor
    func testAChunkInFlightWhenADictationStartsIsCountedOnceWhenItResumes() async throws {
        let (folder, transcript) = try meeting("meeting_12", turns: 12)
        let chunks = MeetingCleanupChunker.chunks(transcript.turns)
        let runtime = FakeRuntime()
        runtime.freezeWhilePaused = true
        let cleaner = cleaner(runtime)
        let normal = runtime.answer
        var pausedOnce = false
        runtime.answer = { prompt in
            if !pausedOnce && prompt == chunks[1].prompt {
                pausedOnce = true
                cleaner.setDictationQuiet(false)  // the key goes down mid-chunk
            }
            return normal(prompt)
        }
        cleaner.setEngineReady(true)
        cleaner.setDictationQuiet(true)
        cleaner.enqueue(directory: folder)
        await waitUntil("paused mid-chunk") { runtime.paused }
        try await Task.sleep(nanoseconds: 300_000_000)
        XCTAssertEqual(MeetingCleanupProgress.load(from: folder)?.results.count, 1, "chunk 1 is still in flight")
        if case .running(_, let paused) = cleaner.jobs["meeting_12"] { XCTAssertTrue(paused) } else { XCTFail("running") }
        cleaner.setDictationQuiet(true)
        await waitUntil("done") { CleanedTranscript.load(from: folder) != nil }
        XCTAssertEqual(runtime.prompts, chunks.map(\.prompt), "each chunk asked once: no retry after the pause")
        XCTAssertEqual(runtime.started, 1)
        let cleaned = try XCTUnwrap(CleanedTranscript.load(from: folder))
        XCTAssertEqual(cleaned.rawChunks, 0)
        XCTAssertEqual(cleaned.chunks, chunks.count)
    }

    @MainActor
    func testAnEngineThatGoesAwayHoldsCleanupUntilTheNewOneSaysItIsQuiet() async throws {
        let (folder, _) = try meeting("meeting_11", turns: 12)
        let runtime = FakeRuntime()
        let cleaner = cleaner(runtime)
        cleaner.setEngineReady(true)
        cleaner.setDictationQuiet(true)
        runtime.answer = { prompt in
            cleaner.setEngineReady(false)
            return prompt
        }
        cleaner.enqueue(directory: folder)
        await waitUntil("paused") { runtime.paused }
        let asked = runtime.prompts.count
        cleaner.setEngineReady(true)
        try await Task.sleep(nanoseconds: 50_000_000)
        XCTAssertTrue(runtime.paused, "ready alone is not enough: the new engine's quiet state is unknown")
        XCTAssertEqual(runtime.prompts.count, asked)
        runtime.answer = { $0 }
        cleaner.setDictationQuiet(true)
        await waitUntil("done") { CleanedTranscript.load(from: folder) != nil }
    }

    @MainActor
    func testACleanupResumesAfterAQuitAtTheNextChunk() async throws {
        let (folder, transcript) = try meeting("meeting_3", turns: 12)
        let chunks = MeetingCleanupChunker.chunks(transcript.turns)
        XCTAssertGreaterThanOrEqual(chunks.count, 3)
        // the last run saved chunk 0, then the app quit
        var saved = MeetingCleanupProgress(source: CleanupSource(transcript))
        saved.results = [CleanupChunkResult(chunk: 0, outcome: .cleaned(chunks[0].lines.map { _ in "saved" }))]
        try saved.save(to: folder)
        // another meeting failed for good: not resumed by itself
        let (failed, _) = try meeting("meeting_4", turns: 4)
        try MeetingCleanupProgress(error: "low memory").save(to: failed)

        let runtime = FakeRuntime()
        let cleaner = cleaner(runtime)
        cleaner.setEngineReady(true)
        cleaner.setDictationQuiet(true)
        cleaner.resumePending(in: root)
        await waitUntil("finished") { CleanedTranscript.load(from: folder) != nil }
        XCTAssertEqual(runtime.prompts, chunks.dropFirst().map(\.prompt), "only the chunks not saved")
        let cleaned = try XCTUnwrap(CleanedTranscript.load(from: folder))
        XCTAssertEqual(cleaned.turns.first?.text, "saved")
        XCTAssertNil(CleanedTranscript.load(from: failed))
        XCTAssertNil(cleaner.jobs["meeting_4"])

        // a progress file of another transcript (re-transcribed) starts over
        let (again, transcript5) = try meeting("meeting_5", turns: 4)
        var stale = MeetingCleanupProgress(source: CleanupSource(MeetingTranscript(model: "x", createdAt: "", turns: [])))
        stale.results = [CleanupChunkResult(chunk: 0, outcome: .cleaned(["stale"]))]
        try stale.save(to: again)
        runtime.prompts = []
        cleaner.resumePending(in: root)
        await waitUntil("restarted") { CleanedTranscript.load(from: again) != nil }
        XCTAssertEqual(runtime.prompts, MeetingCleanupChunker.chunks(transcript5.turns).map(\.prompt))
    }

    @MainActor
    func testARejectedOrFailingChunkKeepsItsRawTurns() async throws {
        let (folder, transcript) = try meeting("meeting_6", turns: 8)
        let chunks = MeetingCleanupChunker.chunks(transcript.turns)
        let runtime = FakeRuntime()
        let normal = runtime.answer
        runtime.answer = { prompt in
            // the first chunk's answer loses a speaker line
            prompt == chunks[0].prompt ? String(prompt.split(separator: "\n").first!) : normal(prompt)
        }
        runtime.failNext = 0
        let cleaner = cleaner(runtime)
        cleaner.setEngineReady(true)
        cleaner.setDictationQuiet(true)
        cleaner.enqueue(directory: folder)
        await waitUntil("finished") { CleanedTranscript.load(from: folder) != nil }
        var cleaned = try XCTUnwrap(CleanedTranscript.load(from: folder))
        XCTAssertEqual(cleaned.rawChunks, 1)
        let firstTurns = Set(chunks[0].lines.map(\.turn))
        for (index, turn) in cleaned.turns.enumerated() where firstTurns.contains(index) && chunks.count > 1 {
            if turn.source == .raw { XCTAssertEqual(turn.text, transcript.turns[index].text) }
        }

        // a runtime that fails twice on a chunk: restarted, then that chunk stays raw
        try FileManager.default.removeItem(at: folder.appendingPathComponent(CleanedTranscript.fileName))
        runtime.answer = normal
        runtime.failNext = 2
        runtime.started = 0
        cleaner.enqueue(directory: folder)
        await waitUntil("finished again") { CleanedTranscript.load(from: folder) != nil }
        cleaned = try XCTUnwrap(CleanedTranscript.load(from: folder))
        XCTAssertEqual(cleaned.rawChunks, 1)
        XCTAssertEqual(runtime.started, 3, "the first start and one restart per failure")
    }

    @MainActor
    func testTheMemoryGateSkipsCleanupAndSaysWhy() async throws {
        let (folder, _) = try meeting("meeting_7", turns: 4)
        let runtime = FakeRuntime()
        let cleaner = cleaner(runtime, free: 1_000_000_000)
        cleaner.setEngineReady(true)
        cleaner.setDictationQuiet(true)
        cleaner.enqueue(directory: folder)
        await waitUntil("skipped") { cleaner.isIdle }
        XCTAssertEqual(runtime.started, 0, "never loads a second model copy")
        if case .failed(let message) = cleaner.jobs["meeting_7"] {
            XCTAssertTrue(message.hasPrefix("Not cleaned: only 1.0 GB"))
        } else {
            XCTFail("failed")
        }
        XCTAssertEqual(MeetingCleanupProgress.load(from: folder)?.error?.hasPrefix("Not cleaned"), true)
        XCTAssertNil(CleanedTranscript.load(from: folder))

        // an 8 GB Mac, and a Mac without the adapter: nothing at all
        let small = self.cleaner(runtime, physical: 8 << 30)
        small.setEngineReady(true)
        small.setDictationQuiet(true)
        let (other, _) = try meeting("meeting_8", turns: 4)
        small.enqueue(directory: other)
        await waitUntil("skipped") { small.isIdle }
        XCTAssertEqual(runtime.started, 0)
        let none = self.cleaner(runtime, available: false)
        let (third, _) = try meeting("meeting_9", turns: 4)
        none.enqueue(directory: third)
        XCTAssertTrue(none.isIdle)
        XCTAssertNil(none.jobs["meeting_9"])
        XCTAssertNil(MeetingCleanupProgress.load(from: third))
    }

    @MainActor
    func testQuittingStopsTheRuntimeAndLeavesTheProgressFile() async throws {
        let (folder, _) = try meeting("meeting_10", turns: 12)
        let runtime = FakeRuntime()
        let cleaner = cleaner(runtime)
        cleaner.setEngineReady(true)
        cleaner.setDictationQuiet(true)
        runtime.answer = { prompt in
            cleaner.setDictationQuiet(false)
            return prompt
        }
        cleaner.enqueue(directory: folder)
        await waitUntil("one chunk") { runtime.prompts.count == 1 }
        await waitUntil("paused") { runtime.paused }
        cleaner.shutdown()
        XCTAssertGreaterThanOrEqual(runtime.stopped, 1)
        XCTAssertFalse(runtime.paused, "a stopped process is continued first, so SIGTERM lands")
        let progress = try XCTUnwrap(MeetingCleanupProgress.load(from: folder))
        XCTAssertNil(progress.error)
        XCTAssertNil(CleanedTranscript.load(from: folder))
    }
}

// MARK: - Runtime timeouts count running time only

/// A flag shared with `@Sendable` stand-ins.
private final class Flag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    private var count = 0
    var isSet: Bool { lock.withLock { value } }
    var calls: Int { lock.withLock { count } }
    func set() { lock.withLock { value = true } }
    func call() { lock.withLock { count += 1 } }
}

final class LlamaServerRuntimeTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("phonon-runtime-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    /// A stand-in server: a real process (so SIGSTOP / SIGCONT really happen) that only sleeps.
    @MainActor
    private func runtime(
        startBudget: TimeInterval = 0.5, requestBudget: TimeInterval = 0.5, ready: Flag,
        send: (@Sendable (URLRequest) async throws -> Data)? = nil
    ) throws -> LlamaServerRuntime {
        let script = root.appendingPathComponent("llama-server")
        try "#!/bin/sh\nexec sleep 30\n".write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
        let setup = LlamaServerRuntime.Setup(
            server: script, model: root.appendingPathComponent("m.gguf"), adapter: root.appendingPathComponent("a.gguf"),
            adapterSHA256: "ab", threads: 1, pidFile: root.appendingPathComponent("rt.pid"))
        return LlamaServerRuntime(
            setup: setup, startBudget: startBudget, requestBudget: requestBudget,
            probe: { _ in ready.isSet }, send: send)
    }

    private func state(_ pidFile: URL) -> String {
        guard let pid = try? String(contentsOf: pidFile, encoding: .utf8) else { return "" }
        let ps = Process()
        ps.executableURL = URL(fileURLWithPath: "/bin/ps")
        ps.arguments = ["-o", "stat=", "-p", pid.trimmingCharacters(in: .whitespacesAndNewlines)]
        let pipe = Pipe()
        ps.standardOutput = pipe
        try? ps.run()
        ps.waitUntilExit()
        return String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    @MainActor
    func testAStartPausedLongerThanItsBudgetWaitsInsteadOfFailing() async throws {
        let ready = Flag()
        let runtime = try runtime(ready: ready)
        defer { runtime.stop() }
        var outcome: Result<Void, Error>?
        Task { @MainActor in
            do { try await runtime.start(); outcome = .success(()) } catch { outcome = .failure(error) }
        }
        try await Task.sleep(nanoseconds: 200_000_000)  // 0.2 s of the 0.5 s budget runs
        runtime.setPaused(true)  // a dictation while the model loads
        XCTAssertTrue(state(runtime.setup.pidFile).hasPrefix("T"), "the process is stopped")
        try await Task.sleep(nanoseconds: 1_500_000_000)  // 3x the budget, paused
        XCTAssertNil(outcome, "a paused start is not a timeout")
        runtime.setPaused(false)
        ready.set()  // the model finishes loading once it runs again
        for _ in 0..<100 where outcome == nil { try await Task.sleep(nanoseconds: 10_000_000) }
        guard case .success = outcome else { return XCTFail("started: \(String(describing: outcome))") }
    }

    @MainActor
    func testAPauseAskedBeforeLaunchHoldsTheStartToo() async throws {
        let ready = Flag()
        ready.set()  // would answer at once if it ran
        let runtime = try runtime(ready: ready)
        defer { runtime.stop() }
        runtime.setPaused(true)
        var outcome: Result<Void, Error>?
        Task { @MainActor in
            do { try await runtime.start(); outcome = .success(()) } catch { outcome = .failure(error) }
        }
        try await Task.sleep(nanoseconds: 1_000_000_000)
        XCTAssertNil(outcome, "no health check while stopped, and no timeout")
        XCTAssertTrue(state(runtime.setup.pidFile).hasPrefix("T"), "launched stopped")
        runtime.setPaused(false)
        for _ in 0..<100 where outcome == nil { try await Task.sleep(nanoseconds: 10_000_000) }
        guard case .success = outcome else { return XCTFail("started: \(String(describing: outcome))") }
    }

    @MainActor
    func testAStartThatNeverAnswersWhileRunningTimesOut() async throws {
        let runtime = try runtime(ready: Flag())
        let began = Date()
        do {
            try await runtime.start()
            XCTFail("timeout")
        } catch LlamaServerRuntime.Failure.startTimeout {}
        XCTAssertLessThan(Date().timeIntervalSince(began), 2)
        XCTAssertFalse(FileManager.default.fileExists(atPath: runtime.setup.pidFile.path), "stopped")
    }

    private static let reply = Data(#"{"choices":[{"message":{"content":"Speaker 1: Hi."}}]}"#.utf8)

    @MainActor
    func testAChunkInFlightWhenStoppedIsNeitherTimedOutNorSentTwice() async throws {
        let ready = Flag()
        ready.set()
        let answered = Flag()
        let runtime = try runtime(ready: ready) { _ in
            answered.call()
            // the stopped server answers only after it runs again
            while !answered.isSet { try await Task.sleep(nanoseconds: 10_000_000) }
            return Self.reply
        }
        defer { runtime.stop() }
        try await runtime.start()
        var outcome: Result<String, Error>?
        Task { @MainActor in
            do {
                outcome = .success(try await runtime.complete(system: "s", user: "u", maxTokens: 8))
            } catch { outcome = .failure(error) }
        }
        try await Task.sleep(nanoseconds: 200_000_000)
        runtime.setPaused(true)
        try await Task.sleep(nanoseconds: 1_500_000_000)  // 3x the request budget, paused
        XCTAssertNil(outcome, "a paused chunk is not a timeout")
        runtime.setPaused(false)
        answered.set()
        for _ in 0..<100 where outcome == nil { try await Task.sleep(nanoseconds: 10_000_000) }
        guard case .success(let text) = outcome else { return XCTFail("answered: \(String(describing: outcome))") }
        XCTAssertEqual(text, "Speaker 1: Hi.")
        XCTAssertEqual(answered.calls, 1, "one request, not re-sent")
    }

    @MainActor
    func testAChunkThatNeverAnswersWhileRunningTimesOut() async throws {
        let ready = Flag()
        ready.set()
        let runtime = try runtime(ready: ready) { _ in
            try await Task.sleep(nanoseconds: 30_000_000_000)
            return Self.reply
        }
        defer { runtime.stop() }
        try await runtime.start()
        let began = Date()
        do {
            _ = try await runtime.complete(system: "s", user: "u", maxTokens: 8)
            XCTFail("timeout")
        } catch LlamaServerRuntime.Failure.requestTimeout {}
        XCTAssertLessThan(Date().timeIntervalSince(began), 2)
    }

    func testTheActiveClockLeavesOutPausedTime() {
        var now = Date(timeIntervalSince1970: 0)
        var clock = ActiveClock(now: { now })
        let mark = clock.mark()
        now += 2
        clock.pause()
        now += 100
        XCTAssertEqual(clock.active(since: mark), 2, accuracy: 1e-9, "paused now")
        clock.resume()
        now += 3
        XCTAssertEqual(clock.active(since: mark), 5, accuracy: 1e-9)
        clock.pause()
        let inside = clock.mark()  // an operation that starts while paused
        now += 50
        clock.resume()
        now += 1
        XCTAssertEqual(clock.active(since: inside), 1, accuracy: 1e-9)
        XCTAssertEqual(clock.active(since: mark), 6, accuracy: 1e-9)
    }
}

// MARK: - Downloads

/// Serves one file, honouring Range; `cutAfter` ends the first response early with an error.
private final class StubServer: URLProtocol {
    nonisolated(unsafe) static var body = Data()
    nonisolated(unsafe) static var cutAfter: Int?
    nonisolated(unsafe) static var ranges: [String?] = []

    override class func canInit(with request: URLRequest) -> Bool { request.url?.host == "stub.test" }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let range = request.value(forHTTPHeaderField: "Range")
        Self.ranges.append(range)
        var start = 0
        if let range, let value = Int(range.dropFirst("bytes=".count).dropLast()) { start = value }
        let status = start > 0 ? 206 : 200
        let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: "HTTP/1.1", headerFields: [:])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        let rest = Self.body.subdata(in: start..<Self.body.count)
        if let cut = Self.cutAfter {
            // the connection drops after `cut` bytes have arrived
            Self.cutAfter = nil
            client?.urlProtocol(self, didLoad: rest.prefix(cut))
            DispatchQueue.global().asyncAfter(deadline: .now() + 0.2) { [self] in
                client?.urlProtocol(self, didFailWithError: URLError(.networkConnectionLost))
            }
            return
        }
        client?.urlProtocol(self, didLoad: rest)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

final class PinnedDownloaderTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("phonon-download-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    private func configuration() -> URLSessionConfiguration {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubServer.self]
        return configuration
    }

    func testAnInterruptedDownloadResumesFromTheBytesOnDiskAndIsVerifiedBeforeInstall() async throws {
        let body = Data((0..<300_000).map { UInt8($0 % 251) })
        let file = root.appendingPathComponent("src.bin")
        try body.write(to: file)
        let component = PinnedComponent(
            dir: "models/test", label: "test file",
            asset: PinnedAsset(
                name: "test.bin", url: URL(string: "https://stub.test/test.bin"),
                sha256: try PinnedFiles.sha256(file), bytes: Int64(body.count)),
            unpack: .none, sentinel: "test.bin")
        StubServer.body = body
        StubServer.cutAfter = 100_000
        StubServer.ranges = []
        let downloader = PinnedDownloader(root: root, configuration: configuration())
        do {
            _ = try await downloader.ensure(component)
            XCTFail("the first try is cut off")
        } catch {}
        XCTAssertNil(downloader.installed(component), "nothing half-written is installed")
        let partial = root.appendingPathComponent("downloads/test.bin.part")
        XCTAssertEqual((try FileManager.default.attributesOfItem(atPath: partial.path)[.size] as? NSNumber)?.intValue, 100_000)

        let installed = try await downloader.ensure(component)
        XCTAssertEqual(StubServer.ranges, [nil, "bytes=100000-"])
        XCTAssertEqual(try Data(contentsOf: installed), body)
        XCTAssertEqual(installed.path, root.appendingPathComponent("models/test/test.bin").path)
        XCTAssertFalse(FileManager.default.fileExists(atPath: partial.path))
        // installed: no network
        _ = try await downloader.ensure(component)
        XCTAssertEqual(StubServer.ranges.count, 2)
    }

    func testABadHashIsNeverInstalledAndAnUnpublishedAssetIsNotFetched() async throws {
        StubServer.body = Data("not the file".utf8)
        StubServer.cutAfter = nil
        StubServer.ranges = []
        let downloader = PinnedDownloader(root: root, configuration: configuration())
        var component = PinnedComponent(
            dir: "models/bad", label: "bad file",
            asset: PinnedAsset(
                name: "bad.bin", url: URL(string: "https://stub.test/bad.bin"), sha256: String(repeating: "0", count: 64),
                bytes: Int64(StubServer.body.count)),
            unpack: .none, sentinel: "bad.bin")
        do {
            _ = try await downloader.ensure(component)
            XCTFail("verification")
        } catch let error as PinnedDownloader.Failure {
            XCTAssertEqual(error, .verification("bad.bin"))
        }
        XCTAssertNil(downloader.installed(component))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("downloads/bad.bin.part").path))

        component.asset.url = nil
        StubServer.ranges = []
        do {
            _ = try await downloader.ensure(component)
            XCTFail("unpublished")
        } catch let error as PinnedDownloader.Failure {
            XCTAssertEqual(error, .unpublished("bad file"))
        }
        XCTAssertTrue(StubServer.ranges.isEmpty)
        // a verified copy put in place by hand is used
        let dir = root.appendingPathComponent("models/bad", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try StubServer.body.write(to: dir.appendingPathComponent("bad.bin"))
        component.asset.sha256 = try PinnedFiles.sha256(dir.appendingPathComponent("bad.bin"))
        let found = try await downloader.ensure(component)
        XCTAssertEqual(found.lastPathComponent, "bad.bin")
    }

    func testAnArchiveIsUnpackedIntoItsFolderOnlyWhenItHoldsTheSentinel() async throws {
        // a tar.gz of llama-test/{llama-server,lib.dylib}
        let source = root.appendingPathComponent("src", isDirectory: true)
        let top = source.appendingPathComponent("llama-test", isDirectory: true)
        try FileManager.default.createDirectory(at: top, withIntermediateDirectories: true)
        try Data("#!/bin/sh\n".utf8).write(to: top.appendingPathComponent("llama-server"))
        try Data("lib".utf8).write(to: top.appendingPathComponent("lib.dylib"))
        let archive = root.appendingPathComponent("rt.tar.gz")
        let tar = Process()
        tar.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        tar.arguments = ["-czf", archive.path, "-C", source.path, "llama-test"]
        try tar.run()
        tar.waitUntilExit()
        StubServer.body = try Data(contentsOf: archive)
        StubServer.cutAfter = nil
        let component = PinnedComponent(
            dir: "runtime/test", label: "runtime",
            asset: PinnedAsset(
                name: "rt.tar.gz", url: URL(string: "https://stub.test/rt.tar.gz"), sha256: try PinnedFiles.sha256(archive),
                bytes: Int64(StubServer.body.count)),
            unpack: .tarGz(stripTo: "llama-test"), sentinel: "llama-server")
        let downloader = PinnedDownloader(root: root, configuration: configuration())
        let server = try await downloader.ensure(component)
        XCTAssertEqual(server.path, root.appendingPathComponent("runtime/test/llama-server").path)
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("runtime/test/lib.dylib").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("downloads/rt.tar.gz").path))

        var wrong = component
        wrong.dir = "runtime/wrong"
        wrong.sentinel = "missing"
        do {
            _ = try await downloader.ensure(wrong)
            XCTFail("no sentinel")
        } catch {}
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("runtime/wrong").path))
    }
}
