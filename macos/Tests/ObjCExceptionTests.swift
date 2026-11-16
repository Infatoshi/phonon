import Foundation
import PhononObjC
import XCTest

/// 2026-10-29 crash: joining a Discord call switched audio devices, the mic tap
/// was installed in the format the hardware had just left, and AVFAudio raised
/// an Objective-C exception that aborted the app. The catcher turns it into an
/// error the recorder can retry on.
final class ObjCExceptionTests: XCTestCase {
    func testARaisedExceptionBecomesAnError() {
        var error: NSError?
        let ok = PhononCatchException({
            NSException(
                name: NSExceptionName("com.apple.coreaudio.avfaudio"),
                reason: "Failed to create tap due to format mismatch", userInfo: nil
            ).raise()
        }, &error)
        XCTAssertFalse(ok)
        XCTAssertEqual(error?.domain, "com.apple.coreaudio.avfaudio")
        XCTAssertEqual(error?.localizedDescription, "Failed to create tap due to format mismatch")
    }

    func testABlockThatReturnsIsUntouched() {
        var error: NSError?
        var ran = false
        XCTAssertTrue(PhononCatchException({ ran = true }, &error))
        XCTAssertTrue(ran)
        XCTAssertNil(error)
    }
}
