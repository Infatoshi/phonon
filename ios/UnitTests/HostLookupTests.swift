import XCTest

/// The start link carries the host's pid and name when the bundle id is hidden (iOS 26.7), and a bar tap that had to
/// open Phonon is remembered long enough to dictate when the keyboard is back.
final class HostLookupTests: XCTestCase {
    func testStartURLRoundTrip() throws {
        let url = try XCTUnwrap(HostLookup.startURL(host: nil, via: "none|tok@0:pid=800:nosig0", pid: 800, processName: "MobileSMS"))
        XCTAssertEqual(url.scheme, "phonon")
        XCTAssertEqual(url.host, "start")
        let hint = HostLookup.parse(url)
        XCTAssertNil(hint.host)
        XCTAssertEqual(hint.pid, 800)
        XCTAssertEqual(hint.processName, "MobileSMS")
        XCTAssertEqual(hint.via, "none|tok@0:pid=800:nosig0")
    }

    func testStartURLWithKnownHostAndNoPid() throws {
        let url = try XCTUnwrap(HostLookup.startURL(host: "com.apple.mobilenotes", via: "_hostBundleID@0", pid: 0, processName: nil))
        let hint = HostLookup.parse(url)
        XCTAssertEqual(hint, HostLookup.Hint(host: "com.apple.mobilenotes", via: "_hostBundleID@0", pid: nil, processName: nil))
    }

    func testOldLinkWithoutNewFieldsStillParses() throws {
        let hint = HostLookup.parse(try XCTUnwrap(URL(string: "phonon://start?via=none&host=")))
        XCTAssertNil(hint.host)
        XCTAssertNil(hint.pid)
        XCTAssertNil(hint.processName)
    }

    func testKnownAppsAreFoundAndUnknownAreNotGuessed() {
        XCTAssertEqual(HostLookup.bundle(forProcessName: "MobileSMS"), "com.apple.MobileSMS")
        XCTAssertEqual(HostLookup.bundle(forProcessName: "WhatsApp"), "net.whatsapp.WhatsApp")
        XCTAssertNil(HostLookup.bundle(forProcessName: "SomeUnknownApp"))
        XCTAssertNil(HostLookup.bundle(forProcessName: nil))
        for (name, id) in HostLookup.knownApps {
            XCTAssertTrue(id.contains("."), "\(name) -> \(id)")
            XCTAssertLessThanOrEqual(name.utf8.count, 16, "p_comm holds at most 16 bytes: \(name)")
        }
    }

    func testProcessNameOfThisProcessAndOfNothing() {
        XCTAssertNotNil(HostLookup.processName(pid: getpid()))
        XCTAssertNil(HostLookup.processName(pid: 0))
        XCTAssertNil(HostLookup.processName(pid: -5))
    }

    func testStartWishResumesOnlyWhenFreshAndAlive() {
        let t = 1_000_000.0
        XCTAssertTrue(PendingStart.shouldResume(wantedAt: t, now: t + 5, alive: true))
        XCTAssertTrue(PendingStart.shouldResume(wantedAt: t, now: t + PendingStart.ttl, alive: true))
        XCTAssertFalse(PendingStart.shouldResume(wantedAt: t, now: t + 5, alive: false), "the session is not up yet: keep waiting")
        XCTAssertFalse(PendingStart.shouldResume(wantedAt: t, now: t + PendingStart.ttl + 1, alive: true), "stale")
        XCTAssertFalse(PendingStart.shouldResume(wantedAt: 0, now: t, alive: true), "no wish")
        XCTAssertFalse(PendingStart.shouldResume(wantedAt: t + 10, now: t, alive: true), "clock went backwards")
    }

    func testStaleWishIsDroppedButAFreshOneWaitingForTheSessionIsKept() {
        let t = 1_000_000.0
        XCTAssertTrue(PendingStart.isStale(wantedAt: t, now: t + PendingStart.ttl + 1))
        XCTAssertTrue(PendingStart.isStale(wantedAt: t + 10, now: t))
        XCTAssertFalse(PendingStart.isStale(wantedAt: t, now: t + 5))
        XCTAssertFalse(PendingStart.isStale(wantedAt: 0, now: t))
    }
}
