import XCTest

@testable import PhononBar

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
        // Left released while right is still held: the generic Option flag stays set.
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
        // Some other modifier changed while Option state is unknowable: keep the last answer.
        XCTAssertTrue(RightOptionKey.isDown(keyCode: 56, rawFlags: alt, wasDown: true))
    }
}
