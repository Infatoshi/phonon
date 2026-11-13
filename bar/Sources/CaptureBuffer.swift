import Foundation

/// One dictation's captured audio, mono, kept at the rate each buffer arrived in. A microphone
/// or format change mid-dictation (a headset connecting, the device renegotiating its rate)
/// starts a new segment, so every stretch is resampled at its own rate. Mixing them and
/// resampling everything at the newest rate garbled the audio before the change (played too
/// fast or too slow) and with it the transcript.
struct CaptureBuffer {
    /// 20 minutes (230 MB at 48 kHz, held only while recording). The old 120 s kept just the
    /// tail of longer dictations, so their corpus audio no longer matched the transcript (26 of
    /// 1,203 entries by 2026-11-04). Past it the oldest audio is dropped.
    static let maxSeconds: TimeInterval = 1_200

    struct Segment: Equatable {
        var rate: Double
        var samples: [Float]
    }

    let maxSeconds: TimeInterval
    private(set) var segments: [Segment] = []

    init(maxSeconds: TimeInterval = CaptureBuffer.maxSeconds) {
        self.maxSeconds = maxSeconds
    }

    var isEmpty: Bool { segments.allSatisfy { $0.samples.isEmpty } }

    var seconds: TimeInterval {
        segments.reduce(0) { $0 + Double($1.samples.count) / $1.rate }
    }

    mutating func append(_ chunk: [Float], rate: Double) {
        guard !chunk.isEmpty, rate > 0 else { return }
        if let last = segments.indices.last, segments[last].rate == rate {
            segments[last].samples.append(contentsOf: chunk)
        } else {
            segments.append(Segment(rate: rate, samples: chunk))
        }
        trim()
    }

    /// Drop the oldest audio past `maxSeconds`.
    private mutating func trim() {
        while let first = segments.first {
            let total = seconds
            guard total > maxSeconds else { return }
            let firstSeconds = Double(first.samples.count) / first.rate
            let keep = Int(((maxSeconds - (total - firstSeconds)) * first.rate))
            if keep <= 0 {
                segments.removeFirst()
            } else {
                segments[0].samples.removeFirst(first.samples.count - keep)
                return
            }
        }
    }

    /// 16 kHz mono for the WAV and the engine: each segment resampled at its own rate.
    func mono16k() -> [Float] {
        segments.flatMap { MicRecorder.resample($0.samples, from: $0.rate, to: 16_000) }
    }
}
