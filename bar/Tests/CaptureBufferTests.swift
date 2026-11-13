import XCTest

@testable import PhononBar

/// A microphone or rate change mid-dictation: each stretch of audio is resampled at the rate it
/// was captured in, so the 16 kHz WAV keeps its true length and pitch.
final class CaptureBufferTests: XCTestCase {
    static func sine(seconds: Double, rate: Double, hz: Double = 440, from t0: Double = 0) -> [Float] {
        (0..<Int(seconds * rate)).map { Float(sin(2 * .pi * hz * (t0 + Double($0) / rate))) }
    }

    /// Fed in 1024-frame buffers, like the tap.
    static func feed(_ buffer: inout CaptureBuffer, _ samples: [Float], rate: Double) {
        var i = 0
        while i < samples.count {
            buffer.append(Array(samples[i..<min(i + 1_024, samples.count)]), rate: rate)
            i += 1_024
        }
    }

    func testARateChangeMidDictationKeepsEachStretchAtItsOwnRate() {
        var buffer = CaptureBuffer()
        Self.feed(&buffer, Self.sine(seconds: 1, rate: 48_000), rate: 48_000)
        Self.feed(&buffer, Self.sine(seconds: 1, rate: 16_000, from: 1), rate: 16_000)
        XCTAssertEqual(buffer.segments.map(\.rate), [48_000, 16_000])
        XCTAssertEqual(buffer.seconds, 2, accuracy: 1e-9)
        let out = buffer.mono16k()
        // Resampling all 64,000 samples at the newest rate (16 kHz) made 4 s of audio from 2.
        XCTAssertEqual(out.count, 32_000)
        // And it is the same 440 Hz tone all the way through.
        let expected = Self.sine(seconds: 2, rate: 16_000)
        let worst = zip(out, expected).map { abs($0 - $1) }.max() ?? 1
        XCTAssertLessThan(worst, 0.02)
    }

    func testOneRateIsUnchanged() {
        var buffer = CaptureBuffer()
        let audio = Self.sine(seconds: 1.3, rate: 44_100)
        Self.feed(&buffer, audio, rate: 44_100)
        XCTAssertEqual(buffer.segments.count, 1)
        XCTAssertEqual(buffer.mono16k(), MicRecorder.resample(audio, from: 44_100, to: 16_000))
    }

    func testPastTheCapTheOldestAudioGoes() {
        var buffer = CaptureBuffer(maxSeconds: 1)
        let audio = (0..<72_000).map { Float($0) }
        Self.feed(&buffer, audio, rate: 48_000)
        XCTAssertEqual(buffer.segments.first?.samples, Array(audio.suffix(48_000)), "the last second")
        // Across a rate change, a whole old segment can go.
        Self.feed(&buffer, Self.sine(seconds: 1, rate: 16_000), rate: 16_000)
        XCTAssertEqual(buffer.segments.map(\.rate), [16_000])
        XCTAssertEqual(buffer.seconds, 1, accuracy: 1e-9)
    }
}
