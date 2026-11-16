import Foundation

/// One dictation's captured audio, mono, kept at the rate each buffer arrived in. A microphone
/// or format change mid-dictation (a headset connecting, the device renegotiating its rate)
/// starts a new segment, so every stretch is resampled at its own rate. Mixing them and
/// resampling everything at the newest rate garbled the audio before the change (played too
/// fast or too slow) and with it the transcript.
///
/// Stored as pieces of at most `pieceFrames`; past `maxSeconds` the oldest audio is dropped by
/// moving a start index (whole pieces released, a partial one skipped by an offset), never by
/// shifting the array: `removeFirst` on the whole buffer moved up to 230 MB on every tap buffer
/// (about 47 a second) once a dictation passed the cap.
struct CaptureBuffer {
    /// 20 minutes (230 MB at 48 kHz, held only while recording). The old 120 s kept just the
    /// tail of longer dictations, so their corpus audio no longer matched the transcript (26 of
    /// 1,203 entries by 2026-11-04). Past it the oldest audio is dropped.
    static let maxSeconds: TimeInterval = 1_200
    /// About a third of a second at 48 kHz.
    static let pieceFrames = 16_384

    struct Segment: Equatable {
        var rate: Double
        var samples: [Float]
    }

    private struct Piece {
        var rate: Double
        var samples: [Float]
    }

    let maxSeconds: TimeInterval
    private var pieces: [Piece] = []
    /// The first live piece, and how many of its samples were dropped.
    private var head = 0
    private var headOffset = 0
    /// Live frames per rate, so the length in seconds is exact.
    private var frames: [Double: Int] = [:]

    init(maxSeconds: TimeInterval = CaptureBuffer.maxSeconds) {
        self.maxSeconds = maxSeconds
    }

    var isEmpty: Bool { frames.values.allSatisfy { $0 == 0 } }

    var seconds: TimeInterval {
        frames.reduce(0) { $0 + Double($1.value) / $1.key }
    }

    /// Pieces kept in storage, dropped ones excluded (for tests: dropping is O(1)).
    var storedPieces: Int { pieces.count - head }

    mutating func append(_ chunk: [Float], rate: Double) {
        guard !chunk.isEmpty, rate > 0 else { return }
        if pieces.count > head, let last = pieces.indices.last, pieces[last].rate == rate,
            pieces[last].samples.count < Self.pieceFrames
        {
            pieces[last].samples.append(contentsOf: chunk)
        } else {
            pieces.append(Piece(rate: rate, samples: chunk))
        }
        frames[rate, default: 0] += chunk.count
        trim()
    }

    /// Drop the oldest audio past `maxSeconds`.
    private mutating func trim() {
        while head < pieces.count {
            let total = seconds
            guard total > maxSeconds else { break }
            let rate = pieces[head].rate
            let live = pieces[head].samples.count - headOffset
            let rateSeconds = Double(frames[rate] ?? 0) / rate
            // Frames of this rate to keep so the total is `maxSeconds` (as the old cap counted).
            let keepOfRate = Int((maxSeconds - (total - rateSeconds)) * rate)
            let excess = (frames[rate] ?? 0) - max(0, keepOfRate)
            if excess >= live {
                frames[rate, default: 0] -= live
                pieces[head].samples = []
                head += 1
                headOffset = 0
            } else {
                frames[rate, default: 0] -= excess
                headOffset += excess
                break
            }
        }
        // Released pieces leave empty slots; compact once they are the majority (amortized O(1)).
        if head > 64 && head * 2 > pieces.count {
            pieces.removeFirst(head)
            head = 0
        }
    }

    /// The live audio as runs of one rate, oldest first.
    var segments: [Segment] {
        var out: [Segment] = []
        for index in head..<pieces.count {
            let piece = pieces[index]
            let live = index == head ? piece.samples[headOffset...] : piece.samples[...]
            guard !live.isEmpty else { continue }
            if let last = out.indices.last, out[last].rate == piece.rate {
                out[last].samples.append(contentsOf: live)
            } else {
                out.append(Segment(rate: piece.rate, samples: Array(live)))
            }
        }
        return out
    }

    /// 16 kHz mono for the WAV and the engine: each segment resampled at its own rate.
    func mono16k() -> [Float] {
        segments.flatMap { MicRecorder.resample($0.samples, from: $0.rate, to: 16_000) }
    }
}
