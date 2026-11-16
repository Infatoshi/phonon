import XCTest

@testable import PhononBar

/// Issue #21 (fn did not consistently start dictation) and the Input Monitoring
/// onboarding: edge detection after a lost release, the
/// Globe-key conflict and what the permission banner shows.
final class HotkeyReliabilityTests: XCTestCase {
    private func ms(_ value: UInt64) -> UInt64 { value * 1_000_000 }

    func testEdgesFollowTheFlags() {
        var edges = HoldKeyEdges()
        XCTAssertEqual(edges.update(pressed: true, ownKey: true), .down)
        XCTAssertEqual(edges.update(pressed: true, ownKey: false), .none, "another modifier")
        XCTAssertEqual(edges.update(pressed: false, ownKey: true), .up)
        XCTAssertEqual(edges.update(pressed: false, ownKey: false), .none)
    }

    /// The old handler ignored this press (fnDown was still true), and its release
    /// then stopped nothing: one dictation lost per lost release.
    func testAPressAfterALostReleaseIsARepress() {
        var edges = HoldKeyEdges()
        XCTAssertEqual(edges.update(pressed: true, ownKey: true), .down)
        // the release never arrived; the next event is the key's own press
        XCTAssertEqual(edges.update(pressed: true, ownKey: true), .repress)
        XCTAssertTrue(edges.isDown)
        XCTAssertEqual(edges.update(pressed: false, ownKey: true), .up)
    }

    /// Replays what AppController.holdKey does on a repress when nothing is recording:
    /// the lost release is applied, then the press starts.
    func testRepressWhileIdleStartsADictation() {
        var latch = HoldKeyTapLatch()
        XCTAssertEqual(latch.keyDown(at: ms(0)), .start)
        _ = latch.keyUp(at: ms(5_000))  // the lost release, applied late
        XCTAssertEqual(latch.keyDown(at: ms(5_000)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(8_000)), .stop)
    }


    func testGlobeKeyConflict() {
        XCTAssertFalse(GlobeKeySetting.conflicts(usageType: 0))
        for usage in [nil, 1, 2, 3] {
            XCTAssertTrue(GlobeKeySetting.conflicts(usageType: usage), "\(String(describing: usage))")
        }
        XCTAssertEqual(GlobeKeySetting.action(usageType: 2), "shows Emoji & Symbols")
        XCTAssertEqual(GlobeKeySetting.action(usageType: 3), "starts Apple's dictation")
    }

    func testHoldKeyModes() {
        XCTAssertTrue(ShortcutPolicy.usesHoldKey(mode: "fn"))
        XCTAssertTrue(ShortcutPolicy.usesHoldKey(mode: "both"))
        XCTAssertTrue(ShortcutPolicy.usesHoldKey(mode: "right_option"))
        XCTAssertFalse(ShortcutPolicy.usesHoldKey(mode: "control_space"))
        XCTAssertEqual(ShortcutPolicy.holdKeyName(mode: "fn_and_control_space"), "Globe (fn)")
        XCTAssertEqual(ShortcutPolicy.holdKeyName(mode: "right_option"), "Right Option")
    }

    func testMissingInputMonitoringComesFirstAndBlocks() {
        let items = PermissionAttention.items(
            inputMonitoring: false, microphone: true, accessibility: false, shortcutMode: "fn",
            globeUsageType: 2)
        XCTAssertEqual(
            items,
            [.inputMonitoring(key: "Globe (fn)"), .accessibility,
             .globeKey(action: "shows Emoji & Symbols")])
        XCTAssertTrue(items[0].blocksDictation)
        XCTAssertFalse(items[1].blocksDictation)
        XCTAssertTrue(items[0].title.contains("Globe (fn)"))
        XCTAssertTrue(items[0].detail.contains("Input Monitoring"))
    }

    func testControlSpaceNeedsNoInputMonitoringAndNoGlobeAdvice() {
        XCTAssertEqual(
            PermissionAttention.items(
                inputMonitoring: false, microphone: true, accessibility: true,
                shortcutMode: "control_space", globeUsageType: 2),
            [])
        XCTAssertEqual(
            PermissionAttention.items(
                inputMonitoring: false, microphone: false, accessibility: true,
                shortcutMode: "right_option", globeUsageType: nil),
            [.inputMonitoring(key: "Right Option"), .microphone])
    }

    func testAllGrantedShowsNothing() {
        XCTAssertEqual(
            PermissionAttention.items(
                inputMonitoring: true, microphone: true, accessibility: true, shortcutMode: "fn",
                globeUsageType: 0),
            [])
    }
}
