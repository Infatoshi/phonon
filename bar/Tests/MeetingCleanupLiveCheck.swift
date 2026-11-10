import XCTest

@testable import PhononBar

/// End-to-end check of meeting cleanup on the real runtime, opt in only (public eval chunks,
/// never a user's meeting):
///
///     PHONON_CLEANUP_CHECK_SERVER=<llama-b12132/llama-server> \
///     PHONON_CLEANUP_CHECK_ADAPTER=<phonon-meeting-lora-q8_0.gguf> \
///     PHONON_CLEANUP_CHECK_CHUNKS=<meetings_v1 chunks/eval2.jsonl> \
///     PHONON_CLEANUP_CHECK_OUT=<empty folder> \
///         swift test --filter MeetingCleanupLiveCheck
///
/// Builds a two-speaker transcript from the eval set's 1-2 speaker chunks (Speaker 1 as You,
/// Speaker 2 as Call), runs `MeetingCleaner` with `LlamaServerRuntime` on the installed
/// correction model (found and verified as the app finds it), pauses it once for 3 s as a
/// dictation would, and prints counts and timings. PHONON_CLEANUP_CHECK_ANSWERS (jsonl of
/// {id, text, verdict}) also checks the validator against reference verdicts.
final class MeetingCleanupLiveCheck: XCTestCase {
    private struct Row: Decodable {
        var id: String
        var text: String
        var bin: String?
        var verdict: String?
    }

    private func rows(_ path: String) throws -> [Row] {
        try String(contentsOfFile: path, encoding: .utf8).split(separator: "\n").map {
            try JSONDecoder().decode(Row.self, from: Data($0.utf8))
        }
    }

    func testValidatorAgreesWithTheReferenceVerdicts() throws {
        let env = ProcessInfo.processInfo.environment
        guard let chunksPath = env["PHONON_CLEANUP_CHECK_CHUNKS"], let answersPath = env["PHONON_CLEANUP_CHECK_ANSWERS"]
        else { throw XCTSkip("set PHONON_CLEANUP_CHECK_CHUNKS and PHONON_CLEANUP_CHECK_ANSWERS to run") }
        let inputs = Dictionary(uniqueKeysWithValues: try rows(chunksPath).map { ($0.id, $0.text) })
        var agree = 0
        var verdicts: [String: Int] = [:]
        let answers = try rows(answersPath)
        for answer in answers {
            let lines = inputs[answer.id]!.components(separatedBy: "\n").enumerated().compactMap { index, line in
                MeetingCleanupValidator.labelled(line).map { CleanupLine(turn: index, label: $0.label, text: $0.text) }
            }
            let verdict: String
            switch MeetingCleanupValidator.check(CleanupChunk(index: 0, lines: lines), output: answer.text) {
            case .cleaned: verdict = "ok"
            case .rejected(let reason): verdict = reason
            }
            verdicts[verdict, default: 0] += 1
            if verdict == answer.verdict { agree += 1 } else { print("disagree", answer.id, verdict, answer.verdict ?? "") }
        }
        print("validator: \(answers.count) answers, \(agree) agree with the reference, verdicts \(verdicts)")
        XCTAssertEqual(agree, answers.count)
    }

    @MainActor
    func testCleanAPublicMeetingOnTheRealRuntime() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let server = env["PHONON_CLEANUP_CHECK_SERVER"], let adapter = env["PHONON_CLEANUP_CHECK_ADAPTER"],
            let chunksPath = env["PHONON_CLEANUP_CHECK_CHUNKS"], let outPath = env["PHONON_CLEANUP_CHECK_OUT"]
        else { throw XCTSkip("set PHONON_CLEANUP_CHECK_SERVER, _ADAPTER, _CHUNKS and _OUT to run") }
        let support = PhononDataPaths.supportDirectory()
        let base = try XCTUnwrap(
            PinnedFiles.firstMatching(
                MeetingCleanupManifest.baseCandidates(support: support), sha256: MeetingCleanupManifest.baseSHA256,
                bytes: MeetingCleanupManifest.baseBytes),
            "the shipped correction model")
        let adapterSHA = try PinnedFiles.sha256(URL(fileURLWithPath: adapter))
        XCTAssertEqual(adapterSHA, MeetingCleanupManifest.adapter.asset.sha256)

        // a two-speaker meeting from public chunks: 1-2 speaker bin, labels to You / Call
        let count = Int(env["PHONON_CLEANUP_CHECK_N"] ?? "") ?? 12
        var turns: [MeetingTranscript.Turn] = []
        var at = 0.0
        for row in try rows(chunksPath) where row.bin == "1-2" && turns.count < 400 {
            if turns.count > 0 && MeetingCleanupChunker.chunks(turns).count >= count { break }
            for line in row.text.components(separatedBy: "\n") {
                guard let labelled = MeetingCleanupValidator.labelled(line),
                    labelled.label == "Speaker 1" || labelled.label == "Speaker 2"
                else { continue }
                let speaker: TranscriptSpeaker = labelled.label == "Speaker 1" ? .me : .call
                if var last = turns.last, last.speaker == speaker {
                    last.text += " " + labelled.text
                    turns[turns.count - 1] = last
                } else {
                    turns.append(.init(startSeconds: at, endSeconds: at + 5, speaker: speaker, text: labelled.text))
                }
                at += 5
            }
        }
        let out = URL(fileURLWithPath: outPath, isDirectory: true)
        let folder = out.appendingPathComponent("meeting_check", isDirectory: true)
        try? FileManager.default.removeItem(at: folder)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let transcript = MeetingTranscript(model: "eval2", createdAt: "", turns: turns)
        try transcript.save(to: folder)
        let chunks = MeetingCleanupChunker.chunks(turns)

        let setup = LlamaServerRuntime.Setup(
            server: URL(fileURLWithPath: server), model: base, adapter: URL(fileURLWithPath: adapter),
            adapterSHA256: adapterSHA, threads: LlamaServerRuntime.threads(),
            pidFile: out.appendingPathComponent("cleanup.pid"))
        let runtime = LlamaServerRuntime(setup: setup)
        let cleaner = MeetingCleaner(
            environment: .init(available: { true }, makeRuntime: { (runtime, adapterSHA) }))
        var finished = false
        cleaner.onFinished = { _ in finished = true }
        let started = Date()
        cleaner.setEngineReady(true)
        cleaner.setDictationQuiet(true)
        cleaner.enqueue(directory: folder)
        var paused = false
        var pausedAt: Date?
        while !finished && Date().timeIntervalSince(started) < 600 {
            try await Task.sleep(nanoseconds: 100_000_000)
            if case .failed(let message) = cleaner.jobs["meeting_check"] { XCTFail(message); break }
            let done = MeetingCleanupProgress.load(from: folder)?.results.count ?? 0
            if !paused && done == chunks.count / 2 {
                paused = true
                pausedAt = Date()
                cleaner.setDictationQuiet(false)  // a dictation, 3 s
                try await Task.sleep(nanoseconds: 3_000_000_000)
                XCTAssertEqual(MeetingCleanupProgress.load(from: folder)?.results.count, done, "nothing finished while paused")
                cleaner.setDictationQuiet(true)
            }
        }
        let cleaned = try XCTUnwrap(CleanedTranscript.load(from: folder))
        let seconds = Date().timeIntervalSince(started)
        let words = turns.reduce(0) { $0 + $1.text.split(separator: " ").count }
        print(
            "cleanup check: \(turns.count) turns, \(words) words, \(chunks.count) chunks, "
                + "\(cleaned.rawChunks) kept raw, \(String(format: "%.1f", seconds)) s wall incl. 3 s pause "
                + "(paused at \(pausedAt.map { String(format: "%.1f", $0.timeIntervalSince(started)) } ?? "-") s)")
        XCTAssertTrue(cleaned.matches(transcript))
        XCTAssertFalse(cleaned.turns.contains { $0.text?.contains("TL;DR") == true })
        XCTAssertLessThan(cleaned.rawChunks, max(2, chunks.count / 4))
        XCTAssertFalse(FileManager.default.fileExists(atPath: setup.pidFile.path), "the runtime is gone")
    }
}
