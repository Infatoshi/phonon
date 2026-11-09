import XCTest

@testable import PhononBar

/// A latched dictation stops on its own: dropped when nothing was said, committed after a
/// long silence that follows speech. Meter readings arrive about every 43 ms.
final class LatchAutoStopTests: XCTestCase {
    private let reading = 0.043

    /// Voice on every reading from `from` for `seconds`.
    private func speak(_ stop: inout LatchAutoStop, from: Double, seconds: Double) {
        var t = from
        while t <= from + seconds {
            stop.note(voice: true, at: t)
            t += reading
        }
        stop.note(voice: false, at: t)
    }

    func testAStrayTapWithNoSpeechIsDiscardedAfterTheLimit() {
        var stop = LatchAutoStop(startedAt: 100)
        var t = 100.0
        while t < 100 + LatchAutoStop.noSpeechSeconds - 0.1 {
            stop.note(voice: false, at: t)
            XCTAssertEqual(stop.decision(at: t), .keepRecording)
            t += reading
        }
        XCTAssertEqual(stop.decision(at: 100 + LatchAutoStop.noSpeechSeconds), .discard)
        XCTAssertFalse(stop.heardSpeech)
    }

    func testNoMeterReadingsAtAllStillDiscards() {
        // A mic that delivers no buffers never calls the meter; the timer still decides.
        let stop = LatchAutoStop(startedAt: 0)
        XCTAssertEqual(stop.decision(at: 7.9), .keepRecording)
        XCTAssertEqual(stop.decision(at: 8.0), .discard)
    }

    func testKeystrokesAreNotSpeech() {
        var stop = LatchAutoStop(startedAt: 0)
        // Typing: one voiced reading per key, quiet between, for the whole window.
        var t = 0.0
        while t < 10 {
            stop.note(voice: true, at: t)
            stop.note(voice: false, at: t + reading)
            t += 0.2
        }
        XCTAssertFalse(stop.heardSpeech)
        XCTAssertEqual(stop.decision(at: 10), .discard)
    }

    func testSpeechThenLongSilenceCommits() throws {
        var stop = LatchAutoStop(startedAt: 0)
        speak(&stop, from: 1, seconds: 3)
        XCTAssertTrue(stop.heardSpeech)
        let lastSpeech = try XCTUnwrap(stop.lastSpeechAt)
        // Past the no-speech limit, but speech was heard, so it keeps going.
        XCTAssertEqual(stop.decision(at: 12), .keepRecording)
        XCTAssertEqual(
            stop.decision(at: lastSpeech + LatchAutoStop.silenceAfterSpeechSeconds - 0.1),
            .keepRecording)
        XCTAssertEqual(
            stop.decision(at: lastSpeech + LatchAutoStop.silenceAfterSpeechSeconds), .commit)
    }

    func testSpeechAfterAPauseRestartsTheSilenceClock() {
        var stop = LatchAutoStop(startedAt: 0)
        speak(&stop, from: 1, seconds: 2)
        speak(&stop, from: 25, seconds: 2)
        // 33 s after the first speech, but only ~6 s after the second.
        XCTAssertEqual(stop.decision(at: 33), .keepRecording)
        XCTAssertEqual(stop.decision(at: 27 + LatchAutoStop.silenceAfterSpeechSeconds + 0.1), .commit)
    }

    func testSpeechJustBeforeTheNoSpeechLimitKeepsRecording() {
        var stop = LatchAutoStop(startedAt: 0)
        speak(&stop, from: 7.5, seconds: 0.4)
        XCTAssertTrue(stop.heardSpeech)
        XCTAssertEqual(stop.decision(at: 8.5), .keepRecording)
    }

    func testTheLimitsAreTheChosenDefaults() {
        XCTAssertEqual(LatchAutoStop.noSpeechSeconds, 8)
        XCTAssertEqual(LatchAutoStop.silenceAfterSpeechSeconds, 30)
        XCTAssertLessThan(LatchAutoStop.noSpeechSeconds, LatchAutoStop.silenceAfterSpeechSeconds)
    }
}
