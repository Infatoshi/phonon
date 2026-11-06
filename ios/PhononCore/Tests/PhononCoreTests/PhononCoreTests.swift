import CryptoKit
import XCTest
@testable import PhononCore

private struct Golden: Decodable {
    struct Sys: Decodable { let vocab: [String]; let mode: String; let out: String }
    struct Budget: Decodable { let tokens: Int; let out: Int }
    struct GuardCase: Decodable { let spoken: String; let output: String; let out: [GuardValue] }
    enum GuardValue: Decodable {
        case s(String), b(Bool)
        init(from d: Decoder) throws {
            let c = try d.singleValueContainer()
            if let b = try? c.decode(Bool.self) { self = .b(b) } else { self = .s(try c.decode(String.self)) }
        }
    }
    struct Clean: Decodable { let `in`: String; let out: String }
    struct Restore: Decodable { let spoken: String; let output: String; let vocab: [String]; let out: String }
    let systemPrompt: String
    let systems: [Sys]
    let budgets: [Budget]
    let guards: [GuardCase]
    let cleans: [Clean]
    let restores: [Restore]
}

final class RefinerPromptTests: XCTestCase {
    private func golden() throws -> Golden {
        let url = Bundle.module.url(forResource: "golden", withExtension: "json")!
        return try JSONDecoder().decode(Golden.self, from: Data(contentsOf: url))
    }

    func testSystemPromptMatchesMacSource() throws {
        let g = try golden()
        XCTAssertEqual(RefinerPrompt.systemPrompt, g.systemPrompt)
        for s in g.systems {
            XCTAssertEqual(RefinerPrompt.system(vocabulary: s.vocab, modeLine: s.mode), s.out)
        }
    }

    func testBudgetGuardCleanRestoreMatchMacSource() throws {
        let g = try golden()
        for b in g.budgets { XCTAssertEqual(RefinerPrompt.tokenBudget(inputTokens: b.tokens), b.out, "budget \(b.tokens)") }
        for c in g.guards {
            let (text, fired) = RefinerPrompt.guardOutput(spoken: c.spoken, output: c.output)
            guard case .s(let wantText) = c.out[0], case .b(let wantFired) = c.out[1] else { return XCTFail() }
            XCTAssertEqual(text, wantText)
            XCTAssertEqual(fired, wantFired)
        }
        for c in g.cleans { XCTAssertEqual(RefinerPrompt.clean(c.in), c.out) }
        for r in g.restores {
            XCTAssertEqual(RefinerPrompt.restoreTerms(spoken: r.spoken, output: r.output, vocabulary: r.vocab), r.out)
        }
    }

    /// transformers' apply_chat_template(enable_thinking=False, add_generation_prompt=True)
    /// on the pinned chat_template.jinja, captured 2026-11-06.
    func testRenderMatchesChatTemplate() {
        let rendered = RefinerPrompt.render(system: "SYS LINE 1\nSYS LINE 2", user: "hello world")
        XCTAssertEqual(rendered, "<|startoftext|><|im_start|>system\nSYS LINE 1\nSYS LINE 2<|im_end|>\n<|im_start|>user\nhello world<|im_end|>\n<|im_start|>assistant\n<think></think>")
    }

    func testVocabularyCap() {
        let terms = (0..<200).map { "T\($0)" } + ["T1", " ", "T2"]
        XCTAssertEqual(RefinerPrompt.capVocabulary(terms, transcript: "one two").count, 30)
        XCTAssertEqual(RefinerPrompt.capVocabulary(terms, transcript: Array(repeating: "w", count: 20).joined(separator: " ")).count, 60)
        XCTAssertEqual(RefinerPrompt.capVocabulary(terms, transcript: Array(repeating: "w", count: 50).joined(separator: " ")).count, 80)
        XCTAssertEqual(RefinerPrompt.capVocabulary(["a", "a", " b "], transcript: "x"), ["a", "b"])
    }
}

/// Serves one file from memory, honouring Range, optionally ignoring it or failing midway.
final class StubProtocol: URLProtocol {
    static var body = Data()
    static var ignoreRange = false
    static var failAfter: Int? = nil
    static var requests: [String] = []

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let range = request.value(forHTTPHeaderField: "Range")
        Self.requests.append(range ?? "full")
        var start = 0
        if let range, !Self.ignoreRange, let n = Int(range.dropFirst(6).dropLast()) { start = n }
        let status = start > 0 ? 206 : 200
        let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: "HTTP/1.1", headerFields: nil)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        var payload = Self.body.subdata(in: start..<Self.body.count)
        if let fail = Self.failAfter {
            Self.failAfter = nil
            payload = payload.prefix(fail)
            client?.urlProtocol(self, didLoad: payload)
            client?.urlProtocol(self, didFailWithError: URLError(.networkConnectionLost))
            return
        }
        client?.urlProtocol(self, didLoad: payload)
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

final class DownloaderTests: XCTestCase {
    var root: URL!

    override func setUp() {
        root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        StubProtocol.body = Data((0..<300_000).map { UInt8($0 % 251) })
        StubProtocol.ignoreRange = false
        StubProtocol.failAfter = nil
        StubProtocol.requests = []
    }

    override func tearDown() { try? FileManager.default.removeItem(at: root) }

    private func file(sha: String? = nil) -> ModelFile {
        let digest = SHA256.hash(data: StubProtocol.body).map { String(format: "%02x", $0) }.joined()
        return ModelFile(repo: "r/r", revision: "abc", path: "dir/f.bin", folder: "pack",
                         size: Int64(StubProtocol.body.count), sha256: sha ?? digest)
    }

    private func downloader() -> ModelDownloader {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StubProtocol.self]
        return ModelDownloader(root: root, configuration: config)
    }

    func testResumesAfterDropAndVerifies() async throws {
        let f = file()
        StubProtocol.failAfter = 100_000
        let d = downloader()
        try await d.fetchForTest(f)
        XCTAssertTrue(d.isReady(f))
        XCTAssertEqual(StubProtocol.requests, ["full", "bytes=100000-"])
        XCTAssertEqual(try Data(contentsOf: f.localURL(root: root)), StubProtocol.body)
    }

    func testServerIgnoringRangeRestartsCleanly() async throws {
        let f = file()
        StubProtocol.failAfter = 50_000
        StubProtocol.ignoreRange = true
        let d = downloader()
        try await d.fetchForTest(f)
        XCTAssertTrue(d.isReady(f))
        XCTAssertEqual(try Data(contentsOf: f.localURL(root: root)), StubProtocol.body)
    }

    func testHashMismatchDeletesAndFails() async {
        let f = file(sha: String(repeating: "0", count: 64))
        let d = downloader()
        do {
            try await d.fetchForTest(f)
            XCTFail("expected a hash failure")
        } catch {
            XCTAssertEqual(error as? ModelDownloader.Failure, .hashMismatch("dir/f.bin"))
        }
        XCTAssertFalse(d.isReady(f))
        XCTAssertFalse(FileManager.default.fileExists(atPath: f.localURL(root: root).path))
    }

    func testPinnedManifestIsSane() {
        XCTAssertEqual(ModelManifest.refinerFiles.count, 1)
        XCTAssertEqual(ModelManifest.refinerFiles[0].sha256, "059e6e503c6c2ec50cb61a39bd0d556fb6d946bb58e965fb9ddb8b0176dfc6cd")
        for f in ModelManifest.allFiles {
            XCTAssertEqual(f.sha256.count, 64)
            XCTAssertTrue(f.url.absoluteString.contains("/resolve/\(f.revision)/"), f.url.absoluteString)
            XCTAssertFalse(f.url.absoluteString.contains("/resolve/main/"))
        }
        let names = Set(ModelManifest.asrFiles.map { String($0.path.split(separator: "/")[0]) })
        XCTAssertEqual(names, ["Preprocessor.mlmodelc", "Encoder.mlmodelc", "Decoder.mlmodelc", "JointDecision.mlmodelc", "parakeet_vocab.json"])
    }
}

extension ModelDownloader {
    func fetchForTest(_ f: ModelFile) async throws {
        retryDelay = 0
        try await download(files: [f]) { _ in }
    }
}

final class SharedStoreTests: XCTestCase {
    func testSingleWriterFilesRoundTrip() {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let store = SharedStore(root: root)
        var s = AppState()
        s.phase = .recording
        s.ackSeq = 7
        s.latest = Dictation(id: "d1", raw: "hello world", text: "Hello world.", refined: true)
        store.writeAppState(s)
        XCTAssertEqual(store.appState, s)

        store.writeCommand(KeyboardCommand(seq: 8, kind: .stop))
        XCTAssertEqual(store.command?.kind, .stop)

        var k = store.keyboardState
        for i in 0..<60 { k.markInserted("id\(i)") }
        k.markInserted("id10")
        store.writeKeyboardState(k)
        XCTAssertEqual(store.keyboardState.inserted.count, 50)
        XCTAssertEqual(store.keyboardState.inserted.last, "id10")

        for i in 0..<40 { store.appendHistory(Dictation(id: "h\(i)", raw: "r", text: "t", refined: false)) }
        XCTAssertEqual(store.history.count, 30)
        XCTAssertEqual(store.history.first?.id, "h39")
    }
}

final class JournalTests: XCTestCase {
    func testUnfinishedRecordingSurvivesAndReadsBack() {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let j = RecordingJournal(directory: dir)
        j.begin(id: "a", mode: .casual)
        j.append([0.25, -0.5, 1])
        j.append([0.75])
        j.endRecording()
        // A new process sees the same journal.
        let again = RecordingJournal(directory: dir)
        XCTAssertEqual(again.unfinished().map(\.id), ["a"])
        XCTAssertEqual(RecordingJournal.readSamples(again.audioURL("a")), [0.25, -0.5, 1, 0.75])
        again.transcribed("a", raw: "hi")
        XCTAssertEqual(again.unfinished().first?.raw, "hi")
        again.finish("a", stage: .delivered)
        XCTAssertTrue(again.unfinished().isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: again.audioURL("a").path))
    }
}
