import XCTest

@testable import PhononBar

/// Issue #23 (Left Option triggered the Right Option hold); cases from outside PR #29.
final class RightOptionKeyTests: XCTestCase {
    private let alt: UInt64 = 0x80000  // NX_ALTERNATEMASK
    private let leftBit: UInt64 = 0x20  // NX_DEVICELALTKEYMASK
    private let rightBit: UInt64 = 0x40  // NX_DEVICERALTKEYMASK
    private let shift: UInt64 = 0x20000

    func testRightOptionAloneIsDown() {
        XCTAssertTrue(RightOptionKey.isDown(keyCode: 61, rawFlags: alt | rightBit, wasDown: false))
    }

    func testLeftOptionAloneIsNotRightOption() {
        XCTAssertFalse(RightOptionKey.isDown(keyCode: 58, rawFlags: alt | leftBit, wasDown: false))
    }

    func testRightStaysDownWhileLeftIsPressedAndReleased() {
        let both = alt | leftBit | rightBit
        XCTAssertTrue(RightOptionKey.isDown(keyCode: 58, rawFlags: both, wasDown: true))
        XCTAssertTrue(RightOptionKey.isDown(keyCode: 58, rawFlags: alt | rightBit, wasDown: true))
    }

    func testReleasingRightWhileLeftHeldCountsAsRelease() {
        XCTAssertFalse(RightOptionKey.isDown(keyCode: 61, rawFlags: alt | leftBit, wasDown: true))
    }

    func testReleaseClearsEverything() {
        XCTAssertFalse(RightOptionKey.isDown(keyCode: 61, rawFlags: 0, wasDown: true))
    }

    func testOtherModifiersSuppressTheKey() {
        XCTAssertFalse(
            RightOptionKey.isDown(keyCode: 61, rawFlags: alt | rightBit | shift, wasDown: false))
    }

    func testEventsWithoutDeviceBitsFallBackToTheKeyCode() {
        XCTAssertTrue(RightOptionKey.isDown(keyCode: 61, rawFlags: alt, wasDown: false))
        XCTAssertFalse(RightOptionKey.isDown(keyCode: 58, rawFlags: alt, wasDown: false))
        XCTAssertTrue(RightOptionKey.isDown(keyCode: 56, rawFlags: alt, wasDown: true))
    }

    /// The left key through the whole edge path never starts a recording.
    func testLeftOptionPressAndReleaseProducesNoEdge() {
        var edges = HoldKeyEdges()
        let down = RightOptionKey.isDown(keyCode: 58, rawFlags: alt | leftBit, wasDown: edges.isDown)
        XCTAssertEqual(edges.update(pressed: down, ownKey: false), .none)
        let up = RightOptionKey.isDown(keyCode: 58, rawFlags: 0, wasDown: edges.isDown)
        XCTAssertEqual(edges.update(pressed: up, ownKey: false), .none)
    }
}
