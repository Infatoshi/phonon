import XCTest

@testable import PhononBar

final class DataMirrorTests: XCTestCase {
    private var home: URL!
    private var support: URL!
    private var mirror: URL!
    private var fileManager: FileManager!

    override func setUpWithError() throws {
        // The mirror location is injected so the tests never touch the real
        // ~/.phonon.
        fileManager = FileManager()
        home = URL(
            fileURLWithPath: NSTemporaryDirectory(), isDirectory: true
        ).appendingPathComponent("phonon-mirror-\(UUID().uuidString)", isDirectory: true)
        support = home.appendingPathComponent("Support", isDirectory: true)
        mirror = home.appendingPathComponent(".phonon/backup", isDirectory: true)
        try fileManager.createDirectory(at: support, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? fileManager.removeItem(at: home)
    }

    private func writeDictionary(_ count: Int) throws {
        let entries = (0..<count).map { ["phrase": "term\($0)"] }
        let data = try JSONSerialization.data(withJSONObject: ["entries": entries])
        try data.write(to: support.appendingPathComponent("dictionary.json"))
    }

    private func writeCorpus(_ count: Int) throws {
        let corpus = support.appendingPathComponent("Corpus", isDirectory: true)
        try fileManager.createDirectory(at: corpus, withIntermediateDirectories: true)
        for index in 0..<count {
            try fileManager.createDirectory(
                at: corpus.appendingPathComponent("item\(index)"),
                withIntermediateDirectories: true)
        }
    }

    func testCaptureRecordsWhatTheStoreHolds() throws {
        try writeDictionary(7)
        try writeCorpus(3)

        let manifest = try XCTUnwrap(
            PhononDataMirror.capture(from: support, mirror: mirror, fileManager: fileManager))

        XCTAssertEqual(manifest.dictionaryEntries, 7)
        XCTAssertEqual(manifest.historyItems, 3)
        let reread = try XCTUnwrap(
            PhononDataMirror.manifest(mirror: mirror, fileManager: fileManager))
        XCTAssertEqual(reread.dictionaryEntries, manifest.dictionaryEntries)
        XCTAssertEqual(reread.historyItems, manifest.historyItems)
    }

    func testAWipedStoreCannotOverwriteAGoodBackup() throws {
        try writeDictionary(7)
        try writeCorpus(3)
        let good = try XCTUnwrap(
            PhononDataMirror.capture(from: support, mirror: mirror, fileManager: fileManager))

        // This is the exact event the backup exists for. It must not consume it.
        try fileManager.removeItem(at: support)
        try fileManager.createDirectory(at: support, withIntermediateDirectories: true)

        XCTAssertNil(
            PhononDataMirror.capture(from: support, mirror: mirror, fileManager: fileManager))
        let survivor = try XCTUnwrap(
            PhononDataMirror.manifest(mirror: mirror, fileManager: fileManager))
        XCTAssertEqual(survivor.dictionaryEntries, good.dictionaryEntries)
        XCTAssertEqual(survivor.historyItems, good.historyItems)
    }

    func testEmptyStoreIsNeverCaptured() throws {
        XCTAssertNil(PhononDataMirror.capture(from: support, mirror: mirror, fileManager: fileManager))
        XCTAssertNil(PhononDataMirror.manifest(mirror: mirror, fileManager: fileManager))
    }

    func testRestoreIsOfferedOnlyWhenTheStoreIsEmpty() throws {
        try writeDictionary(5)
        PhononDataMirror.capture(from: support, mirror: mirror, fileManager: fileManager)

        XCTAssertNil(PhononDataMirror.restorableManifest(for: support, mirror: mirror, fileManager: fileManager))

        try fileManager.removeItem(at: support.appendingPathComponent("dictionary.json"))
        let offered = try XCTUnwrap(
            PhononDataMirror.restorableManifest(for: support, mirror: mirror, fileManager: fileManager))
        XCTAssertEqual(offered.dictionaryEntries, 5)
    }

    func testRestorePutsTheDictionaryBackWithoutClobbering() throws {
        try writeDictionary(5)
        let settings = support.appendingPathComponent("settings.json")
        try Data("{\"streaming\":true}".utf8).write(to: settings)
        PhononDataMirror.capture(from: support, mirror: mirror, fileManager: fileManager)

        try fileManager.removeItem(at: support.appendingPathComponent("dictionary.json"))
        try Data("{\"streaming\":false}".utf8).write(to: settings)

        try PhononDataMirror.restore(into: support, mirror: mirror, fileManager: fileManager)

        let restored = try Data(contentsOf: support.appendingPathComponent("dictionary.json"))
        let object = try JSONSerialization.jsonObject(with: restored) as? [String: Any]
        XCTAssertEqual((object?["entries"] as? [Any])?.count, 5)
        // A file the store already had wins, so a newer setting is not undone.
        XCTAssertEqual(
            String(data: try Data(contentsOf: settings), encoding: .utf8), "{\"streaming\":false}")
    }

    func testForgetRemovesTheBackup() throws {
        try writeDictionary(2)
        PhononDataMirror.capture(from: support, mirror: mirror, fileManager: fileManager)
        XCTAssertNotNil(PhononDataMirror.manifest(mirror: mirror, fileManager: fileManager))

        try PhononDataMirror.forget(mirror: mirror, fileManager: fileManager)

        XCTAssertNil(PhononDataMirror.manifest(mirror: mirror, fileManager: fileManager))
        XCTAssertNoThrow(try PhononDataMirror.forget(mirror: mirror, fileManager: fileManager))
    }

    func testTheMirrorLivesOutsideLibrary() {
        let path = PhononDataMirror.directory().path
        XCTAssertFalse(path.contains("/Library/"), "an uninstaller sweeps ~/Library: \(path)")
        XCTAssertTrue(path.hasSuffix("/.phonon/backup"))
    }
}
