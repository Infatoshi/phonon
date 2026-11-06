import XCTest
@testable import Phonon

/// A missing or broken Parakeet file must fail with the re-download notice. FluidAudio's own
/// loader would instead delete the folder and download an unpinned copy.
final class LocalAsrModelsTests: XCTestCase {
    func testMissingModelFailsWithoutDownloading() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertThrowsError(try LocalAsrModels.load(from: dir)) { error in
            XCTAssertTrue(error.localizedDescription.contains("re-download the models in Settings"), error.localizedDescription)
        }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: dir.path), [])
    }

    func testCorruptModelFailsWithoutDownloading() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let fake = dir.appendingPathComponent("Encoder.mlmodelc")
        try FileManager.default.createDirectory(at: fake, withIntermediateDirectories: true)
        try Data("not a model".utf8).write(to: fake.appendingPathComponent("model.mil"))
        try Data(#"{"0": "a"}"#.utf8).write(to: dir.appendingPathComponent("parakeet_vocab.json"))
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertThrowsError(try LocalAsrModels.load(from: dir)) { error in
            XCTAssertTrue(error.localizedDescription.contains("Encoder.mlmodelc"), error.localizedDescription)
        }
        XCTAssertEqual(Set(try FileManager.default.contentsOfDirectory(atPath: dir.path)), ["Encoder.mlmodelc", "parakeet_vocab.json"])
    }
}
