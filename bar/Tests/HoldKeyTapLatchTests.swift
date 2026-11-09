import XCTest

@testable import PhononBar

/// Same cases as `crates/phonon-hotkey`: hold to talk, or tap to start and tap to stop.
final class HoldKeyTapLatchTests: XCTestCase {
    private func ms(_ value: UInt64) -> UInt64 { value * 1_000_000 }

    func testHoldStartsOnDownAndStopsOnRelease() {
        var latch = HoldKeyTapLatch()
        XCTAssertEqual(latch.keyDown(at: ms(0)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(900)), .stop)
        XCTAssertFalse(latch.latched)
    }

    func testOneTapLatchesAndTheNextPressStops() {
        var latch = HoldKeyTapLatch()
        XCTAssertEqual(latch.keyDown(at: ms(0)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(120)), .none, "one tap keeps the recording going")
        XCTAssertTrue(latch.latched)
        // While latched the press itself stops; its release is ignored.
        XCTAssertEqual(latch.keyDown(at: ms(4_000)), .stop)
        XCTAssertFalse(latch.latched)
        XCTAssertEqual(latch.keyUp(at: ms(4_100)), .none)
        // And again: the stop press leaves nothing pending.
        XCTAssertEqual(latch.keyDown(at: ms(4_300)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(4_400)), .none)
        XCTAssertTrue(latch.latched)
    }

    /// Two quick taps used to latch; now the first latches and the second stops.
    func testADoubleTapIsAStartThenAStop() {
        var latch = HoldKeyTapLatch()
        XCTAssertEqual(latch.keyDown(at: ms(0)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(150)), .none)
        XCTAssertEqual(latch.keyDown(at: ms(400)), .stop)
        XCTAssertEqual(latch.keyUp(at: ms(550)), .none)
        XCTAssertFalse(latch.latched)
    }

    func testAHoldAfterAHoldIsPushToTalk() {
        var latch = HoldKeyTapLatch()
        XCTAssertEqual(latch.keyDown(at: ms(0)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(600)), .stop)
        XCTAssertEqual(latch.keyDown(at: ms(700)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(2_700)), .stop)
        XCTAssertFalse(latch.latched)
    }

    func testThresholdIsInclusive() {
        var latch = HoldKeyTapLatch(tapMaxNs: ms(250))
        XCTAssertEqual(latch.keyDown(at: ms(0)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(250)), .none)
        XCTAssertTrue(latch.latched)
        latch.reset()
        XCTAssertEqual(latch.keyDown(at: ms(1_000)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(1_251)), .stop)
        XCTAssertFalse(latch.latched)
    }

    func testResetClearsTheLatch() {
        var latch = HoldKeyTapLatch()
        XCTAssertEqual(latch.keyDown(at: ms(400)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(500)), .none)
        XCTAssertTrue(latch.latched)
        // Something else ended the recording: the next press starts, not stops.
        latch.reset()
        XCTAssertFalse(latch.latched)
        XCTAssertEqual(latch.keyDown(at: ms(9_000)), .start)
    }

    func testAClockThatGoesBackwardsIsNotATap() {
        var latch = HoldKeyTapLatch()
        XCTAssertEqual(latch.keyDown(at: ms(1_000)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(900)), .stop)
        XCTAssertFalse(latch.latched)
    }

    /// The menu-bar glyph is drawn larger than the menu font; marks after it are not.
    func testStatusTitleEnlargesOnlyTheGlyph() {
        let title = StatusTitle.attributed("ϕ ●")
        XCTAssertEqual(title.string, "ϕ ●")
        let glyph = title.attribute(.font, at: 0, effectiveRange: nil) as? NSFont
        let mark = title.attribute(.font, at: 2, effectiveRange: nil) as? NSFont
        XCTAssertEqual(glyph?.pointSize, StatusTitle.glyphSize)
        XCTAssertEqual(mark?.pointSize, StatusTitle.markSize)
    }
}
