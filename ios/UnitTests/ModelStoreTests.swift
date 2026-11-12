import CryptoKit
import XCTest

/// ModelStore against an in-process HTTP server (no network): full download, resume from a partial file (also
/// through a redirect), retry after a dropped connection, a server that ignores Range, a hash mismatch, files
/// already complete, and adopting an older build's FluidAudio cache. Mirrors
/// android/app/src/test/java/ai/phonon/keyboard/ModelStoreTest.kt.
final class ModelStoreTests: XCTestCase {
    private var server: TestServer!
    private var root: URL!
    private let fm = FileManager.default

    override func setUpWithError() throws {
        server = try TestServer()
        XCTAssertNotEqual(server.port, 0)
        root = fm.temporaryDirectory.appendingPathComponent("modelstore-\(UUID().uuidString)")
    }

    override func tearDownWithError() throws {
        try? fm.removeItem(at: root)
        server = nil
    }

    private func blob(_ n: Int, seed: UInt8 = 7) -> Data {
        Data((0..<n).map { UInt8(truncatingIfNeeded: $0 &* 31 &+ Int(seed)) })
    }

    private func sha(_ d: Data) -> String { SHA256.hash(data: d).map { String(format: "%02x", $0) }.joined() }

    private func pinned(_ path: String, _ data: Data, via: String = "") -> ModelFile {
        server.set(path, data)
        return ModelFile(path: path, url: server.url(via + path), sha256: sha(data), size: Int64(data.count))
    }

    private func store() -> ModelStore { ModelStore(root: root, sleep: { _ in }) }

    private func partURL(_ f: ModelFile) -> URL { root.appendingPathComponent(".partial").appendingPathComponent(f.path) }

    private func writePart(_ f: ModelFile, _ d: Data) throws {
        let p = partURL(f)
        try fm.createDirectory(at: p.deletingLastPathComponent(), withIntermediateDirectories: true)
        try d.write(to: p)
    }

    func testDownloadsVerifiesAndMarks() async throws {
        let data = blob(300_000)
        let f = pinned("Encoder.mlmodelc/weights/weight.bin", data)
        let s = store()
        XCTAssertFalse(s.isReady(f))
        XCTAssertEqual(s.remaining([f]), 300_000)
        try await s.ensure([f])
        XCTAssertTrue(s.isReady(f))
        XCTAssertEqual(try Data(contentsOf: s.file(f)), data)
        XCTAssertFalse(fm.fileExists(atPath: partURL(f).path))
        XCTAssertEqual(s.remaining([f]), 0)
    }

    func testResumesFromAPartialFileThroughARedirect() async throws {
        let data = blob(200_000)
        let f = pinned("Decoder.mlmodelc/weights/weight.bin", data, via: "redirect/")
        try writePart(f, data.prefix(50_000))
        let s = store()
        XCTAssertEqual(s.remaining([f]), 150_000)
        try await s.ensure([f])
        XCTAssertTrue(s.isReady(f))
        XCTAssertEqual(try Data(contentsOf: s.file(f)), data)
        let reqs = server.requests
        XCTAssertEqual(reqs.map(\.path), ["redirect/Decoder.mlmodelc/weights/weight.bin",
                                          "Decoder.mlmodelc/weights/weight.bin"])
        XCTAssertEqual(reqs.map(\.range), ["bytes=50000-", "bytes=50000-"], "the Range goes along the redirect")
    }

    func testRetriesADroppedConnectionFromWhereItStopped() async throws {
        let data = blob(400_000)
        let f = pinned("model.mil", data)
        server.cut(after: [100_000, 100_000])
        try await store().ensure([f])
        XCTAssertEqual(try Data(contentsOf: store().file(f)), data)
        XCTAssertEqual(server.requests.map(\.range), [nil, "bytes=100000-", "bytes=200000-"])
    }

    func testServerIgnoringRangeRestartsTheFile() async throws {
        let data = blob(120_000)
        let f = pinned("parakeet_vocab.json", data)
        try writePart(f, Data(repeating: 0xEE, count: 40_000))  // junk that a 206 would have kept
        server.ignoreRange = true
        try await store().ensure([f])
        XCTAssertEqual(try Data(contentsOf: store().file(f)), data)
        XCTAssertEqual(server.requests.map(\.range), ["bytes=40000-"])
    }

    func testHashMismatchDeletesThePartAndFailsAfterOneRefetch() async throws {
        let data = blob(64_000)
        server.set("bad.bin", data)
        let f = ModelFile(path: "bad.bin", url: server.url("bad.bin"), sha256: String(repeating: "0", count: 64),
                          size: Int64(data.count))
        let s = store()
        do {
            try await s.ensure([f])
            XCTFail("a file that does not hash to its pin must not load")
        } catch let e as ModelStore.Failure {
            XCTAssertTrue(e.message.contains("sha256 mismatch"), e.message)
        }
        XCTAssertEqual(server.requests.count, 2, "one fresh fetch after the bad file, then an error")
        XCTAssertFalse(fm.fileExists(atPath: partURL(f).path))
        XCTAssertFalse(fm.fileExists(atPath: s.file(f).path))
        XCTAssertFalse(s.isReady(f))
    }

    func testCompleteFilesAreSkippedAndAWrongSizeIsNotReady() async throws {
        let a = pinned("a.bin", blob(10_000, seed: 1))
        let b = pinned("dir/b.bin", blob(20_000, seed: 2))
        let s = store()
        try await s.ensure([a, b])
        XCTAssertEqual(server.requests.count, 2)
        try await s.ensure([a, b])
        XCTAssertEqual(server.requests.count, 2, "nothing fetched when every file is ready")
        // a file cut short after its marker was written is fetched again
        try Data(blob(10_000, seed: 1).prefix(5_000)).write(to: s.file(a))
        XCTAssertFalse(s.isReady(a))
        try await s.ensure([a, b])
        XCTAssertEqual(server.requests.map(\.path).suffix(1), ["a.bin"])
        XCTAssertTrue(s.allReady([a, b]))
    }

    func testAdoptsMatchingFilesFromAnOlderCache() async throws {
        let good = blob(30_000, seed: 3)
        let a = pinned("Preprocessor.mlmodelc/weights/weight.bin", good)
        let b = pinned("Preprocessor.mlmodelc/model.mil", blob(8_000, seed: 4))
        let legacy = root.appendingPathComponent("legacy-cache")
        for (f, d) in [(a, good), (b, Data(repeating: 1, count: 8_000))] {  // b: right size, wrong bytes
            let u = legacy.appendingPathComponent(f.path)
            try fm.createDirectory(at: u.deletingLastPathComponent(), withIntermediateDirectories: true)
            try d.write(to: u)
        }
        let s = ModelStore(root: root.appendingPathComponent("store"), sleep: { _ in })
        XCTAssertEqual(s.adopt(from: legacy, files: [a, b]), 1)
        XCTAssertTrue(s.isReady(a))
        XCTAssertFalse(s.isReady(b))
        try await s.ensure([a, b])
        XCTAssertEqual(server.requests.map(\.path), ["Preprocessor.mlmodelc/model.mil"])
        XCTAssertTrue(s.allReady([a, b]))
    }

    func testParakeetEnsureReusesAnOlderCacheAndRemovesIt() async throws {
        let a = pinned("Encoder.mlmodelc/coremldata.bin", blob(500, seed: 5))
        let legacy = root.appendingPathComponent("FluidAudio/Models/x")
        let u = legacy.appendingPathComponent(a.path)
        try fm.createDirectory(at: u.deletingLastPathComponent(), withIntermediateDirectories: true)
        try blob(500, seed: 5).write(to: u)
        let b = pinned("Encoder.mlmodelc/model.mil", blob(700, seed: 6))  // not in the old cache: downloaded
        let s = ModelStore(root: root.appendingPathComponent("store"), sleep: { _ in })
        let dir = try await ParakeetModel.ensure(store: s, files: [a, b], legacy: legacy)
        XCTAssertEqual(dir, s.root)
        XCTAssertTrue(s.allReady([a, b]))
        XCTAssertEqual(server.requests.map(\.path), ["Encoder.mlmodelc/model.mil"])
        XCTAssertFalse(fm.fileExists(atPath: legacy.path), "the old cache is removed once taken over")
        XCTAssertEqual(try s.root.resourceValues(forKeys: [.isExcludedFromBackupKey]).isExcludedFromBackup, true)
        // a later start with everything ready touches neither the network nor the old location
        _ = try await ParakeetModel.ensure(store: s, files: [a, b], legacy: legacy)
        XCTAssertEqual(server.requests.count, 1)
    }

    func testParakeetPinsCoverFluidAudiosV2Files() {
        let files = ParakeetModel.files
        XCTAssertEqual(files.count, 21)
        XCTAssertEqual(Set(files.map { $0.path.split(separator: "/")[0] }),
                       ["Preprocessor.mlmodelc", "Encoder.mlmodelc", "Decoder.mlmodelc", "JointDecision.mlmodelc",
                        "parakeet_vocab.json"])
        for dir in ["Preprocessor", "Encoder", "Decoder", "JointDecision"] {
            for leaf in ["analytics/coremldata.bin", "coremldata.bin", "metadata.json", "model.mil", "weights/weight.bin"] {
                XCTAssertTrue(files.contains { $0.path == "\(dir).mlmodelc/\(leaf)" }, "\(dir) \(leaf)")
            }
        }
        XCTAssertEqual(files.reduce(0) { $0 + $1.size }, 464_413_247)
        for f in files {
            XCTAssertEqual(f.sha256.count, 64, f.path)
            XCTAssertTrue(f.sha256.allSatisfy { "0123456789abcdef".contains($0) }, f.path)
            XCTAssertEqual(f.url.absoluteString,
                           "https://huggingface.co/\(ParakeetModel.repo)/resolve/\(ParakeetModel.revision)/\(f.path)")
        }
        XCTAssertEqual(ParakeetModel.revision.count, 40)
    }
}
