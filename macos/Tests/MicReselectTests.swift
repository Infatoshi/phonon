import XCTest

@testable import PhononBar

/// AVAudioEngine posts a configuration change right after a start that had to rebuild its
/// graph (the first start after the system tore the default-device aggregate down, which is
/// the first dictation after some idle minutes). `MicRecorder` re-resolves the device once
/// that burst settles, and that stops the engine. With Instant microphone off the recorder is
/// not primed, and the stop used to be final: the dictation under way kept its 0.4 s of
/// audio and heard nothing more ("Phonon didn't hear anything"), while the next press
/// started the engine from scratch and worked.
final class MicReselectTests: XCTestCase {
    func testADictationUnderWayGetsItsMicrophoneBackEvenWhenNotPrimed() {
        XCTAssertTrue(MicRecorder.shouldRestartAfterReselect(primed: false, capturing: true))
    }

    func testAWarmMicrophoneComesBack() {
        XCTAssertTrue(MicRecorder.shouldRestartAfterReselect(primed: true, capturing: false))
        XCTAssertTrue(MicRecorder.shouldRestartAfterReselect(primed: true, capturing: true))
    }

    func testAnIdleReleasedMicrophoneStaysOff() {
        XCTAssertFalse(MicRecorder.shouldRestartAfterReselect(primed: false, capturing: false))
    }
}
