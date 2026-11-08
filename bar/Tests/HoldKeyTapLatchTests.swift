import XCTest

@testable import PhononBar

final class HoldKeyTapLatchTests: XCTestCase {
    private func ms(_ value: UInt64) -> UInt64 { value * 1_000_000 }

    func testHoldStartsOnDownAndStopsOnRelease() {
        var latch = HoldKeyTapLatch()
        XCTAssertEqual(latch.keyDown(at: ms(0)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(900)), .stop)
        XCTAssertFalse(latch.latched)
    }

    func testLoneTapIsAShortCaptureWithoutLatch() {
        var latch = HoldKeyTapLatch()
        XCTAssertEqual(latch.keyDown(at: ms(0)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(200)), .stop)
        XCTAssertFalse(latch.latched)
        // A press well after the window is a plain hold again.
        XCTAssertEqual(latch.keyDown(at: ms(2_000)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(2_800)), .stop)
        XCTAssertFalse(latch.latched)
    }

    func testDoubleTapLatchesAndNextPressStops() {
        var latch = HoldKeyTapLatch()
        XCTAssertEqual(latch.keyDown(at: ms(0)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(150)), .stop)
        XCTAssertEqual(latch.keyDown(at: ms(400)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(550)), .none)
        XCTAssertTrue(latch.latched)
        // While latched the press itself stops; its release is ignored.
        XCTAssertEqual(latch.keyDown(at: ms(5_000)), .stop)
        XCTAssertFalse(latch.latched)
        XCTAssertEqual(latch.keyUp(at: ms(5_100)), .none)
        // The stop press does not seed a new double-tap.
        XCTAssertEqual(latch.keyDown(at: ms(5_200)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(5_300)), .stop)
        XCTAssertFalse(latch.latched)
    }

    func testSecondPressHeldLongIsAHoldNotALatch() {
        var latch = HoldKeyTapLatch()
        XCTAssertEqual(latch.keyDown(at: ms(0)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(150)), .stop)
        XCTAssertEqual(latch.keyDown(at: ms(400)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(1_200)), .stop)
        XCTAssertFalse(latch.latched)
    }

    func testSecondTapOutsideWindowDoesNotLatch() {
        var latch = HoldKeyTapLatch()
        XCTAssertEqual(latch.keyDown(at: ms(0)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(150)), .stop)
        // Window is measured from the first release to the second press.
        XCTAssertEqual(latch.keyDown(at: ms(150 + 351)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(650)), .stop)
        XCTAssertFalse(latch.latched)
    }

    func testFirstPressHeldLongDoesNotSeedADoubleTap() {
        var latch = HoldKeyTapLatch()
        XCTAssertEqual(latch.keyDown(at: ms(0)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(600)), .stop)
        XCTAssertEqual(latch.keyDown(at: ms(700)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(800)), .stop)
        XCTAssertFalse(latch.latched)
    }

    func testThresholdsAreInclusive() {
        var latch = HoldKeyTapLatch(tapMaxNs: ms(250), doubleTapWindowNs: ms(350))
        XCTAssertEqual(latch.keyDown(at: ms(0)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(250)), .stop)
        XCTAssertEqual(latch.keyDown(at: ms(600)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(850)), .none)
        XCTAssertTrue(latch.latched)
    }

    func testTapToToggleLatchesOnOneTapAndTheNextTapStops() {
        var latch = HoldKeyTapLatch()
        latch.tapToggles = true
        XCTAssertEqual(latch.keyDown(at: ms(0)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(120)), .none, "one tap keeps the recording going")
        XCTAssertTrue(latch.latched)
        XCTAssertFalse(latch.awaitingSecondTap)
        XCTAssertEqual(latch.keyDown(at: ms(4_000)), .stop)
        XCTAssertEqual(latch.keyUp(at: ms(4_100)), .none)
        XCTAssertFalse(latch.latched)
        // and again: the stop tap does not leave anything pending
        XCTAssertEqual(latch.keyDown(at: ms(4_300)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(4_400)), .none)
        XCTAssertTrue(latch.latched)
    }

    func testTapToToggleKeepsHoldAsPushToTalk() {
        var latch = HoldKeyTapLatch()
        latch.tapToggles = true
        XCTAssertEqual(latch.keyDown(at: ms(0)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(251)), .stop)
        XCTAssertFalse(latch.latched)
        XCTAssertEqual(latch.keyDown(at: ms(1_000)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(3_000)), .stop)
        XCTAssertFalse(latch.latched)
    }

    func testDefaultIsOffSoALoneTapStillStops() {
        var latch = HoldKeyTapLatch()
        XCTAssertFalse(latch.tapToggles)
        XCTAssertEqual(latch.keyDown(at: ms(0)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(100)), .stop)
        XCTAssertTrue(latch.awaitingSecondTap, "a lone tap may still become a double-tap")
        XCTAssertFalse(latch.latched)
    }

    func testResetClearsLatchAndPendingTap() {
        var latch = HoldKeyTapLatch()
        XCTAssertEqual(latch.keyDown(at: ms(0)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(100)), .stop)
        latch.reset()
        XCTAssertEqual(latch.keyDown(at: ms(200)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(300)), .stop)
        XCTAssertFalse(latch.latched)
        // Latched, then reset because something else ended the recording.
        XCTAssertEqual(latch.keyDown(at: ms(400)), .start)
        XCTAssertEqual(latch.keyUp(at: ms(500)), .none)
        XCTAssertTrue(latch.latched)
        latch.reset()
        XCTAssertFalse(latch.latched)
        XCTAssertEqual(latch.keyDown(at: ms(9_000)), .start)
    }
}
