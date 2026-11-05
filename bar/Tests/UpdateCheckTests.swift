import XCTest

@testable import PhononBar

final class UpdateCheckTests: XCTestCase {
    func testVersionParts() {
        XCTAssertEqual(UpdateCheck.versionParts("v0.2.4"), [0, 2, 4])
        XCTAssertEqual(UpdateCheck.versionParts("0.2.10"), [0, 2, 10])
        XCTAssertNil(UpdateCheck.versionParts("v0.2.0-win-beta.1"))
        XCTAssertNil(UpdateCheck.versionParts("v0.2"))
        XCTAssertNil(UpdateCheck.versionParts("v0.2.x"))
        XCTAssertNil(UpdateCheck.versionParts("v+1.2.3"))
        XCTAssertNil(UpdateCheck.versionParts("v0..3"))
        XCTAssertNil(UpdateCheck.versionParts(""))
    }

    private func release(
        _ tag: String, draft: Bool = false, prerelease: Bool = false,
        assets: [String] = ["Phonon.dmg"]
    ) -> String {
        let names = assets.map { #"{"name":"\#($0)"}"# }.joined(separator: ",")
        return #"{"tag_name":"\#(tag)","html_url":"https://github.com/Infatoshi/phonon/releases/tag/\#(tag)","draft":\#(draft),"prerelease":\#(prerelease),"assets":[\#(names)],"body":"notes"}"#
    }

    private func releases(_ items: [String]) -> Data {
        Data("[\(items.joined(separator: ","))]".utf8)
    }

    func testNewestSkipsBetasDraftsAndReleasesWithoutTheDMG() throws {
        let data = releases([
            release("v0.3.0", assets: []),
            release("v0.2.12", draft: true),
            release("v0.2.11", prerelease: true),
            release("v0.2.10"),
            release("v0.2.9"),
            release("v0.2.3-win-beta.2", prerelease: true, assets: ["phonon-win.exe"]),
            release("v0.2.3"),
        ])
        let update = try XCTUnwrap(UpdateCheck.newest(in: data, current: "0.2.4"))
        XCTAssertEqual(update.version, "0.2.10", "numeric, not string, order")
        XCTAssertEqual(
            update.pageURL.absoluteString,
            "https://github.com/Infatoshi/phonon/releases/tag/v0.2.10")
        XCTAssertNil(try UpdateCheck.newest(in: data, current: "0.2.10"))
        XCTAssertNil(try UpdateCheck.newest(in: data, current: "0.3.1"))
    }

    func testOlderPublishedReleaseIsNotAnUpdate() throws {
        // A local build ahead of GitHub, as on the owner's Mac before a release.
        let data = releases([release("v0.2.3"), release("v0.2.2")])
        XCTAssertNil(try UpdateCheck.newest(in: data, current: "0.2.4"))
        XCTAssertEqual(try UpdateCheck.newest(in: data, current: "0.2.2")?.version, "0.2.3")
    }

    func testUnknownRunningVersionNeverAnnounces() throws {
        let data = releases([release("v9.9.9")])
        XCTAssertNil(try UpdateCheck.newest(in: data, current: "0"))
        XCTAssertNil(try UpdateCheck.newest(in: data, current: "16.0"))
    }

    func testMalformedResponseThrows() {
        XCTAssertThrowsError(try UpdateCheck.newest(in: Data("{\"message\":\"rate limited\"}".utf8), current: "0.2.4"))
    }

    func testSettingsDefaultToCheckingAndRoundTrip() throws {
        let empty = try JSONDecoder().decode(NativeSettings.self, from: Data("{}".utf8))
        XCTAssertTrue(empty.checkForUpdates)
        XCTAssertEqual(empty.updateAnnounced, "")
        var settings = NativeSettings()
        settings.checkForUpdates = false
        settings.updateAnnounced = "0.2.5"
        let encoded = try JSONEncoder().encode(settings)
        let text = String(decoding: encoded, as: UTF8.self)
        XCTAssertTrue(text.contains("\"check_for_updates\":false"))
        XCTAssertTrue(text.contains("\"update_announced\":\"0.2.5\""))
        XCTAssertEqual(try JSONDecoder().decode(NativeSettings.self, from: encoded), settings)
    }
}
