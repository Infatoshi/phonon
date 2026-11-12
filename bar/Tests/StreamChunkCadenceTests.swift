import XCTest

@testable import PhononBar

/// M15: the live text card streams the opening 2 s in 0.1 s chunks; everything else keeps 0.4 s.
final class StreamChunkCadenceTests: XCTestCase {
    /// Chunk end times (s) for `seconds` of audio delivered in `buffer`-frame buffers.
    private func chunkEnds(seconds: Double, rate: Double, buffer: Int, fastStart: Bool) -> [Double] {
        var sent = 0
        var pending = 0
        var ends: [Double] = []
        var captured = 0
        while Double(captured) < seconds * rate {
            captured += buffer
            pending += buffer
            if StreamChunkCadence.shouldSend(
                sentFrames: sent, pendingFrames: pending, rate: rate, fastStart: fastStart)
            {
                sent += pending
                pending = 0
                ends.append(Double(sent) / rate)
            }
        }
        return ends
    }

    func testWithoutFastStartTheCadenceIsTheOldFixedOne() {
        for rate in [16_000.0, 44_100, 48_000] {
            for pending in [0, 1, Int(rate * 0.4) - 1, Int(rate * 0.4), Int(rate * 0.4) + 1] {
                for sent in [0, Int(rate), Int(rate * 5)] {
                    XCTAssertEqual(
                        StreamChunkCadence.shouldSend(
                            sentFrames: sent, pendingFrames: pending, rate: rate, fastStart: false),
                        pending >= Int(rate * 0.4), "rate \(rate) pending \(pending)")
                }
            }
        }
    }

    func testFastStartSendsShortChunksForTwoSecondsThenTheUsualOnes() {
        let ends = chunkEnds(seconds: 4.0, rate: 48_000, buffer: 480, fastStart: true)
        let early = ends.filter { $0 <= 2.0 + 1e-9 }
        XCTAssertEqual(early.count, 20)
        XCTAssertEqual(early.first!, 0.1, accuracy: 1e-9)
        XCTAssertEqual(
            Array(ends.drop(while: { $0 <= 2.0 + 1e-9 })), [2.4, 2.8, 3.2, 3.6, 4.0].map { $0 },
            "after the opening, the boundaries of the fixed cadence")
        let fixed = chunkEnds(seconds: 4.0, rate: 48_000, buffer: 480, fastStart: false)
        XCTAssertEqual(fixed.filter { $0 > 2.0 }, ends.filter { $0 > 2.0 })
    }

    func testLargeBuffersStillSendEveryBuffer() {
        // A device delivering 4,800-frame buffers (0.1 s): one chunk per buffer while early.
        let ends = chunkEnds(seconds: 1.0, rate: 48_000, buffer: 4_800, fastStart: true)
        XCTAssertEqual(ends.count, 10)
    }
}
