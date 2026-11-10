import Accelerate
import Foundation

// Only the owner's voice while a meeting records (SPEC "Meeting notes", "Only your voice"). A call
// played through the speakers reaches the microphone; the meeting's system track is a clean copy
// of everything that is not the owner, on the same host clock. `CallEchoGate` finds the delay at
// which the mic follows that reference and silences the stretches of mic audio the reference
// explains. It is used on the meeting's "me" track (`MeetingEchoPlanner`) and on a dictation made
// while a meeting records (`DictationEchoGate`). It fails open: no reference, too little of it,
// or no delay at which it explains the mic, and the audio is left exactly as it was.

/// The gate itself, on 16 kHz mono audio.
enum CallEchoGate {
    static let rate = 16_000
    static let window = 512
    static let hop = 160
    /// The mic may lag the reference by up to this (output latency plus the room) ...
    static let maxDelayFrames = 50
    /// ... or lead it by this much (ScreenCaptureKit stamps its buffers a little late).
    static let maxLeadFrames = 10
    /// The reference covers mic time from `-lookbackSamples` to `mic.count + leadSamples`.
    static var lookbackSamples: Int { maxDelayFrames * hop }
    static var leadSamples: Int { maxLeadFrames * hop }
    /// Bins 7..<128 of 512 at 16 kHz: about 220 Hz to 4 kHz, where speech is.
    static let lowBin = 7
    static let highBin = 128
    /// At least this many frames (1 s) of reference sound before anything is decided.
    static let minReferenceFrames = 100
    /// At the best delay the reference must explain at least this share of the mic's energy in
    /// the frames where it plays, and this many times the median over all delays (a delay that
    /// stands out, not a broad bias).
    static let minExplained: Float = 0.2
    static let delayContrast: Float = 4
    /// A frame is the call's when the reference predicts at least this share of its energy.
    static let dominance: Float = 0.6
    /// Frames this close to one where the owner is clearly talking (the reference predicts under
    /// this share of it) are never silenced.
    static let hangoverFrames = 2
    static let ownerShare: Float = 0.3
    /// Lag search uses at most this many frames (evenly spread), to bound the cost of long spans.
    static let maxSearchFrames = 1_500
    /// Reference frames quieter than this (band energy, about -70 dBFS) are not sound.
    static let silentBandEnergy: Float = 0.01

    struct Analysis: Equatable {
        /// The mic lags the reference by this many frames; nil when no delay was found (open).
        var delayFrames: Int?
        /// Share of the mic's energy the reference explains at that delay.
        var explained: Float
        /// Per mic frame: silenced.
        var muted: [Bool]
        /// Per mic frame: the share of its energy the reference predicts (empty when open).
        var echoShare: [Float] = []
        /// The reference was moved this many samples later on top of `delayFrames`.
        var shift = 0

        var mutedFrames: Int { muted.filter { $0 }.count }
        var alignment: Alignment? { delayFrames.map { Alignment(delayFrames: $0, shift: shift) } }
    }

    /// Where the mic follows the reference: whole frames plus a sub-frame shift in samples.
    struct Alignment: Equatable {
        var delayFrames: Int
        var shift: Int
    }

    static func shifted(_ samples: [Float], by shift: Int) -> [Float] {
        guard shift != 0 else { return samples }
        var out = [Float](repeating: 0, count: samples.count)
        let lower = max(0, shift)
        let upper = min(samples.count, samples.count + shift)
        if upper > lower { out[lower..<upper] = samples[(lower - shift)..<(upper - shift)] }
        return out
    }

    /// Frames of `samples` (16 kHz): how many 512-sample windows fit at a 160-sample hop.
    static func frameCount(_ samples: Int) -> Int {
        samples < window ? 0 : (samples - window) / hop + 1
    }

    /// `reference` covers mic time from `-lookbackSamples` to `mic.count + leadSamples`. With
    /// `known` (an alignment an earlier analysis of the same recording found), the delay is not
    /// searched again: a stretch where the owner talks over the call throughout still has its
    /// echo found.
    static func analyze(mic: [Float], reference: [Float], known: Alignment? = nil) -> Analysis {
        let micFrames = frameCount(mic.count)
        let open = Analysis(delayFrames: nil, explained: 0, muted: Array(repeating: false, count: micFrames))
        let refFrames = frameCount(reference.count)
        guard micFrames > 0, refFrames > 0 else { return open }
        let m = Spectra(mic)
        var r = Spectra(shifted(reference, by: known?.shift ?? 0))
        let bins = m.bins
        func soundingFrames(_ spectra: Spectra) -> [Bool] {
            let floor = max((spectra.power.max() ?? 0) * 1e-4, silentBandEnergy)
            return spectra.power.map { $0 > floor }
        }
        var sounding = soundingFrames(r)
        guard sounding.contains(true) else { return open }

        // The delay: every frame counts the same (its spectrum scaled to unit energy), so a loud
        // owner over a quiet echo does not hide the frames where the call plays alone.
        let mn = m.normalized()
        var rn = r.normalized()
        let stride = max(1, micFrames / maxSearchFrames)
        var best: (delay: Int, explained: Float) = (0, -1)
        var all: [Float] = []
        var sxyRe = [Float](repeating: 0, count: bins)
        var sxyIm = [Float](repeating: 0, count: bins)
        var sxx = [Float](repeating: 0, count: bins)
        var syy = [Float](repeating: 0, count: bins)
        func explainedShare() -> Float {
            var explainedEnergy: Float = 0
            var micEnergy: Float = 0
            for k in 0..<bins {
                micEnergy += syy[k]
                if sxx[k] > 0 { explainedEnergy += (sxyRe[k] * sxyRe[k] + sxyIm[k] * sxyIm[k]) / sxx[k] }
            }
            return micEnergy > 0 ? explainedEnergy / micEnergy : 0
        }
        func accumulate(delay: Int, stride: Int, re mRe: [Float], _ mIm: [Float], _ rRe: [Float], _ rIm: [Float])
            -> Int
        {
            var frames = 0
            sxyRe.withUnsafeMutableBufferPointer { aRe in
                sxyIm.withUnsafeMutableBufferPointer { aIm in
                    sxx.withUnsafeMutableBufferPointer { ax in
                        syy.withUnsafeMutableBufferPointer { ay in
                            mRe.withUnsafeBufferPointer { mr in
                                mIm.withUnsafeBufferPointer { mi in
                                    rRe.withUnsafeBufferPointer { rr in
                                        rIm.withUnsafeBufferPointer { ri in
                                            sounding.withUnsafeBufferPointer { on in
                                                for k in 0..<bins {
                                                    aRe[k] = 0
                                                    aIm[k] = 0
                                                    ax[k] = 0
                                                    ay[k] = 0
                                                }
                                                var t = 0
                                                while t < micFrames {
                                                    let u = t + maxDelayFrames - delay
                                                    if u >= 0, u < refFrames, on[u] {
                                                        frames += 1
                                                        let mo = t * bins
                                                        let ro = u * bins
                                                        for k in 0..<bins {
                                                            let a = mr[mo + k]
                                                            let b = mi[mo + k]
                                                            let c = rr[ro + k]
                                                            let d = ri[ro + k]
                                                            // mic times the conjugate of the reference
                                                            aRe[k] += a * c + b * d
                                                            aIm[k] += b * c - a * d
                                                            ax[k] += c * c + d * d
                                                            ay[k] += a * a + b * b
                                                        }
                                                    }
                                                    t += stride
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return frames
        }
        var bestShift = known?.shift ?? 0
        if let known {
            _ = accumulate(delay: known.delayFrames, stride: stride, re: mn.re, mn.im, rn.re, rn.im)
            best = (known.delayFrames, explainedShare())
        } else {
            for delay in -maxLeadFrames...maxDelayFrames {
                let frames = accumulate(delay: delay, stride: stride, re: mn.re, mn.im, rn.re, rn.im)
                guard frames * stride >= minReferenceFrames else { continue }
                let explained = explainedShare()
                all.append(explained)
                if explained > best.explained { best = (delay, explained) }
            }
            guard !all.isEmpty else { return open }
            let median = all.sorted()[all.count / 2]
            guard best.explained >= minExplained, best.explained >= delayContrast * median else {
                return Analysis(delayFrames: nil, explained: max(0, best.explained), muted: open.muted)
            }

            // Within the frame: shift the reference by up to half a hop, 2 ms at a time, so the
            // per-frame echo estimate below lines up within 1 ms, not within the 10 ms hop (on the
            // synthetic mixes in CallEchoTests that is the difference between ~12 and ~40 dB).
            for shift in Swift.stride(from: -hop / 2, through: hop / 2, by: rate / 500) where shift != 0 {
                let candidate = Spectra(shifted(reference, by: shift))
                let kept = (sounding, rn)
                sounding = soundingFrames(candidate)
                rn = candidate.normalized()
                let frames = accumulate(delay: best.delay, stride: stride, re: mn.re, mn.im, rn.re, rn.im)
                let explained = explainedShare()
                if frames * stride >= minReferenceFrames, explained > best.explained {
                    best.explained = explained
                    bestShift = shift
                    r = candidate
                } else {
                    (sounding, rn) = kept
                }
            }
        }

        // The echo path at that delay, at true levels (the owner's speech is uncorrelated with
        // the call, so it does not bias it), then per frame the echo it predicts against what
        // the mic holds.
        _ = accumulate(delay: best.delay, stride: 1, re: m.re, m.im, r.re, r.im)
        let gain = (0..<bins).map {
            sxx[$0] > 0 ? (sxyRe[$0] * sxyRe[$0] + sxyIm[$0] * sxyIm[$0]) / (sxx[$0] * sxx[$0]) : 0
        }
        var call = [Bool](repeating: false, count: micFrames)
        var share = [Float](repeating: 0, count: micFrames)
        for t in 0..<micFrames {
            let u = t + maxDelayFrames - best.delay
            guard u >= 0, u < refFrames, sounding[u], m.power[t] > 0 else { continue }
            var echo: Float = 0
            let ro = u * bins
            for k in 0..<bins {
                echo += gain[k] * (r.re[ro + k] * r.re[ro + k] + r.im[ro + k] * r.im[ro + k])
            }
            share[t] = echo / m.power[t]
            call[t] = echo >= dominance * m.power[t]
        }
        let micFloor = m.power.sorted()[micFrames / 10]
        let owner = (0..<micFrames).map {
            share[$0] < ownerShare && m.power[$0] > max(10 * micFloor, silentBandEnergy)
        }
        var muted = call
        for t in 0..<micFrames where call[t] {
            let lo = max(0, t - hangoverFrames)
            let hi = min(micFrames - 1, t + hangoverFrames)
            if owner[lo...hi].contains(true) { muted[t] = false }
        }
        return Analysis(
            delayFrames: best.delay, explained: best.explained, muted: muted, echoShare: share,
            shift: bestShift)
    }

    /// The mic sample range [lower, upper) frame `t` stands for: the middle hop of its window
    /// (the first frame also takes the samples before it, the last the samples after it).
    static func samples(ofFrame t: Int, frames: Int, count: Int) -> Range<Int> {
        let pad = (window - hop) / 2
        let lower = t == 0 ? 0 : t * hop + pad
        let upper = t == frames - 1 ? count : (t + 1) * hop + pad
        return lower..<max(lower, min(count, upper))
    }

    /// Silenced stretches as 16 kHz sample ranges of the mic, merged.
    static func mutedRanges(_ analysis: Analysis, count: Int) -> [Range<Int>] {
        var out: [Range<Int>] = []
        let frames = analysis.muted.count
        for t in 0..<frames where analysis.muted[t] {
            let range = samples(ofFrame: t, frames: frames, count: count)
            if let last = out.last, last.upperBound >= range.lowerBound {
                out[out.count - 1] = last.lowerBound..<max(last.upperBound, range.upperBound)
            } else {
                out.append(range)
            }
        }
        return out
    }

    /// 16 kHz ranges as ranges of audio at `rate`.
    static func scale(_ ranges: [Range<Int>], to rate: Double, limit: Int) -> [Range<Int>] {
        let factor = rate / Double(Self.rate)
        return ranges.compactMap {
            let lower = min(limit, Int((Double($0.lowerBound) * factor).rounded()))
            let upper = min(limit, Int((Double($0.upperBound) * factor).rounded()))
            return upper > lower ? lower..<upper : nil
        }
    }

    /// Per-sample gains: 0 inside the ranges, 1 outside, with 5 ms ramps at each inner edge.
    static func gains(count: Int, silenced ranges: [Range<Int>], rate: Double) -> [Float] {
        var gains = [Float](repeating: 1, count: count)
        let ramp = max(1, Int(rate * 0.005))
        for range in ranges {
            let lower = max(0, range.lowerBound)
            let upper = min(count, range.upperBound)
            guard upper > lower else { continue }
            for i in lower..<upper { gains[i] = 0 }
            // fade out before the range and back in after it
            for step in 1...ramp {
                let g = Float(step) / Float(ramp + 1)
                let before = lower - step
                if before >= 0 { gains[before] = min(gains[before], g) }
                let after = upper - 1 + step
                if after < count { gains[after] = min(gains[after], g) }
            }
        }
        return gains
    }

    /// Mono audio at `rate` as 16 kHz: a 3-sample average from 48 kHz, else linear.
    static func to16k(_ input: [Float], rate: Double) -> [Float] {
        guard !input.isEmpty, rate > 0 else { return [] }
        if abs(rate - 48_000) < 0.5 {
            let n = input.count / 3
            return (0..<n).map { (input[3 * $0] + input[3 * $0 + 1] + input[3 * $0 + 2]) / 3 }
        }
        if abs(rate - Double(Self.rate)) < 0.5 { return input }
        let count = max(1, Int((Double(input.count) * Double(Self.rate) / rate).rounded()))
        let step = rate / Double(Self.rate)
        let last = input.count - 1
        return (0..<count).map { i in
            let position = Double(i) * step
            let i0 = min(last, Int(position))
            let i1 = min(i0 + 1, last)
            let fraction = Float(position - Double(i0))
            return input[i0] * (1 - fraction) + input[i1] * fraction
        }
    }

    /// Hann-windowed spectra of the speech band, one row of `bins` per frame, with each frame's
    /// band energy.
    struct Spectra {
        let bins = CallEchoGate.highBin - CallEchoGate.lowBin
        var re: [Float]
        var im: [Float]
        var power: [Float]

        init(_ samples: [Float]) {
            let frames = CallEchoGate.frameCount(samples.count)
            let n = CallEchoGate.window
            re = [Float](repeating: 0, count: frames * bins)
            im = [Float](repeating: 0, count: frames * bins)
            power = [Float](repeating: 0, count: frames)
            guard frames > 0,
                let setup = vDSP_DFT_zrop_CreateSetup(nil, vDSP_Length(n), .FORWARD)
            else { return }
            defer { vDSP_DFT_DestroySetup(setup) }
            let half = n / 2
            let bins = self.bins
            let hann = vDSP.window(ofType: Float.self, usingSequence: .hanningDenormalized, count: n, isHalfWindow: false)
            // even samples (and window values) first, odd ones after: the split input zrop takes
            var hannEven = [Float](repeating: 0, count: half)
            var hannOdd = [Float](repeating: 0, count: half)
            for j in 0..<half {
                hannEven[j] = hann[2 * j]
                hannOdd[j] = hann[2 * j + 1]
            }
            var even = [Float](repeating: 0, count: half)
            var odd = [Float](repeating: 0, count: half)
            var outRe = [Float](repeating: 0, count: half)
            var outIm = [Float](repeating: 0, count: half)
            var reAll = [Float](repeating: 0, count: frames * bins)
            var imAll = [Float](repeating: 0, count: frames * bins)
            var power = [Float](repeating: 0, count: frames)
            defer {
                re = reAll
                im = imAll
                self.power = power
            }
            samples.withUnsafeBufferPointer { x in
                reAll.withUnsafeMutableBufferPointer { reOut in
                    imAll.withUnsafeMutableBufferPointer { imOut in
                        for t in 0..<frames {
                            let start = x.baseAddress! + t * CallEchoGate.hop
                            vDSP_vmul(start, 2, hannEven, 1, &even, 1, vDSP_Length(half))
                            vDSP_vmul(start + 1, 2, hannOdd, 1, &odd, 1, vDSP_Length(half))
                            vDSP_DFT_Execute(setup, even, odd, &outRe, &outIm)
                            let row = t * bins
                            var energy: Float = 0
                            for k in 0..<bins {
                                let a = outRe[CallEchoGate.lowBin + k]
                                let b = outIm[CallEchoGate.lowBin + k]
                                reOut[row + k] = a
                                imOut[row + k] = b
                                energy += a * a + b * b
                            }
                            power[t] = energy
                        }
                    }
                }
            }
        }

        /// Each frame scaled to unit band energy (silent frames stay zero).
        func normalized() -> (re: [Float], im: [Float]) {
            var nre = re
            var nim = im
            for t in 0..<power.count where power[t] > 0 {
                let scale = 1 / power[t].squareRoot()
                for k in (t * bins)..<((t + 1) * bins) {
                    nre[k] *= scale
                    nim[k] *= scale
                }
            }
            return (nre, nim)
        }
    }
}

/// The call audio of the meeting being recorded, as a short 16 kHz history on the host clock,
/// for a dictation made during the meeting. Written by the meeting recorder's system track, read
/// by the dictation recorder; inactive (and empty) when no meeting records.
final class CallReference: @unchecked Sendable {
    static let shared = CallReference()
    static let keepSeconds = 30.0
    /// A buffer this far from where the last one ended is a jump: padded with silence if later,
    /// started over if earlier.
    static let jumpNanoseconds: Int64 = 20_000_000

    private let lock = NSLock()
    private var active = false
    private var buffer: [Float] = []
    private var startHostNs: Int64 = 0
    private var carry: [Float] = []

    var isActive: Bool {
        lock.lock()
        defer { lock.unlock() }
        return active
    }

    func begin() {
        lock.lock()
        active = true
        buffer = []
        carry = []
        lock.unlock()
    }

    func end() {
        lock.lock()
        active = false
        buffer = []
        carry = []
        lock.unlock()
    }

    private static func ns(samples: Int) -> Int64 { Int64(samples) * 1_000_000_000 / Int64(CallEchoGate.rate) }
    private static func samples(ns: Int64) -> Int {
        Int((Double(ns) * Double(CallEchoGate.rate) / 1e9).rounded())
    }

    /// Call audio at `rate` whose first sample was captured at `hostNs`.
    func append(_ samples: [Float], rate: Double, hostNs: UInt64?) {
        lock.lock()
        defer { lock.unlock() }
        guard active, !samples.isEmpty else { return }
        if let hostNs {
            let host = Int64(clamping: hostNs)
            if buffer.isEmpty {
                startHostNs = host
                carry = []
            } else {
                let delta = host - (startHostNs + Self.ns(samples: buffer.count))
                if delta > Self.jumpNanoseconds, Double(delta) < Self.keepSeconds * 1e9 {
                    buffer.append(contentsOf: [Float](repeating: 0, count: Self.samples(ns: delta)))
                    carry = []
                } else if delta < -Self.jumpNanoseconds || Double(delta) >= Self.keepSeconds * 1e9 {
                    buffer = []
                    carry = []
                    startHostNs = host
                }
            }
        } else if buffer.isEmpty {
            return  // nothing to anchor it to
        }
        var input = samples
        if abs(rate - 48_000) < 0.5 {
            input = carry + samples
            let usable = input.count / 3 * 3
            carry = Array(input[usable...])
            input = Array(input[..<usable])
        }
        buffer.append(contentsOf: CallEchoGate.to16k(input, rate: rate))
        let keep = Int(Self.keepSeconds) * CallEchoGate.rate
        if buffer.count > keep {
            let drop = buffer.count - keep
            buffer.removeFirst(drop)
            startHostNs += Self.ns(samples: drop)
        }
    }

    /// `count` samples of call audio from host time `startHostNs` (which may be before the
    /// history), silence where there is none; nil when inactive or none of it is covered.
    func window(startHostNs requested: Int64, count: Int) -> [Float]? {
        lock.lock()
        defer { lock.unlock() }
        guard active, !buffer.isEmpty, count > 0 else { return nil }
        let offset = Self.samples(ns: requested - startHostNs)
        let lower = max(0, offset)
        let upper = min(buffer.count, offset + count)
        guard upper > lower else { return nil }
        var out = [Float](repeating: 0, count: count)
        for i in lower..<upper { out[i - offset] = buffer[i] }
        return out
    }
}

/// The gate on one dictation made while a meeting records: fed the dictation's audio block by
/// block (as it is streamed), it returns the gains for each block, or nil to leave it as is.
final class DictationEchoGate {
    /// The delay and echo path are estimated over at most this much of the dictation.
    static let historySeconds = 6.0

    let reference: CallReference
    let sourceRate: Double
    private var firstHostNs: Int64?
    private var history: [Float] = []
    /// Dictation samples (16 kHz) before `history[0]`.
    private var historyStart = 0
    private(set) var mutedSamples = 0
    private(set) var alignment: CallEchoGate.Alignment?

    /// A gate only while a meeting records: otherwise nil, and dictation is untouched.
    static func make(reference: CallReference = .shared, sourceRate: Double) -> DictationEchoGate? {
        reference.isActive && sourceRate > 0 ? DictationEchoGate(reference: reference, sourceRate: sourceRate) : nil
    }

    init(reference: CallReference, sourceRate: Double) {
        self.reference = reference
        self.sourceRate = sourceRate
    }

    /// The host time of the dictation's first sample (later calls are ignored).
    func noteStart(hostNs: UInt64?) {
        guard firstHostNs == nil, let hostNs else { return }
        firstHostNs = Int64(clamping: hostNs)
    }

    /// Gains for the next block of the dictation (at `sourceRate`), nil to leave it as it is.
    func gains(for block: [Float]) -> [Float]? {
        let block16 = CallEchoGate.to16k(block, rate: sourceRate)
        history.append(contentsOf: block16)
        let keep = Int(Self.historySeconds) * CallEchoGate.rate
        if history.count > keep {
            let drop = history.count - keep
            history.removeFirst(drop)
            historyStart += drop
        }
        guard let firstHostNs, !block16.isEmpty else { return nil }
        let historyHostNs =
            firstHostNs + Int64(historyStart) * 1_000_000_000 / Int64(CallEchoGate.rate)
        let lookback = CallEchoGate.lookbackSamples
        guard
            let ref = reference.window(
                startHostNs: historyHostNs - Int64(lookback) * 1_000_000_000 / Int64(CallEchoGate.rate),
                count: lookback + history.count + CallEchoGate.leadSamples)
        else { return nil }
        // The speaker-to-mic path does not move within a dictation: once found, the delay is
        // kept, so a stretch where the owner talks over the call is still gated.
        let analysis = CallEchoGate.analyze(mic: history, reference: ref, known: alignment)
        if alignment == nil { alignment = analysis.alignment }
        guard analysis.delayFrames != nil, analysis.mutedFrames > 0 else { return nil }
        let blockStart = history.count - block16.count
        let ranges = CallEchoGate.mutedRanges(analysis, count: history.count).compactMap {
            range -> Range<Int>? in
            let lower = max(range.lowerBound, blockStart) - blockStart
            let upper = range.upperBound - blockStart
            return upper > lower ? lower..<upper : nil
        }
        guard !ranges.isEmpty else { return nil }
        let scaled = CallEchoGate.scale(ranges, to: sourceRate, limit: block.count)
        mutedSamples += scaled.reduce(0) { $0 + $1.count }
        return CallEchoGate.gains(count: block.count, silenced: scaled, rate: sourceRate)
    }
}

/// The silenced stretches of a meeting's mic span: the call (system track at the same moments)
/// is the reference. Empty (open) for system spans, unknown offsets, a re-anchored clock, or
/// chunks that cannot be read.
enum MeetingEchoPlanner {
    static func silence(span: MeetingSpan, metadata: MeetingMetadata, directory: URL) -> [Range<Int64>] {
        guard span.track == .mic, span.endSample > span.startSample,
            let micOffset = metadata.micStartOffsetSamples,
            let systemOffset = metadata.systemStartOffsetSamples,
            (metadata.micDiscontinuities ?? []).isEmpty,
            (metadata.systemDiscontinuities ?? []).isEmpty,
            !metadata.systemChunks.isEmpty
        else { return [] }
        let rate = Double(max(1, metadata.sampleRate))
        let lookback = Int64((Double(CallEchoGate.lookbackSamples) * rate / Double(CallEchoGate.rate)).rounded())
        let lead = Int64((Double(CallEchoGate.leadSamples) * rate / Double(CallEchoGate.rate)).rounded())
        // The system track's sample at the span's first mic sample, on the shared timeline.
        let systemStart = span.startSample + micOffset - systemOffset
        guard
            let mic = MeetingChunkReader.read(
                chunks: span.chunks, folder: directory.appendingPathComponent("mic"),
                from: span.startSample, to: span.endSample),
            let system = MeetingChunkReader.read(
                chunks: metadata.systemChunks, folder: directory.appendingPathComponent("system"),
                from: systemStart - lookback, to: systemStart + (span.endSample - span.startSample) + lead)
        else { return [] }
        let mic16 = CallEchoGate.to16k(mic.samples, rate: rate)
        let ref16 = CallEchoGate.to16k(system.samples, rate: rate)
        guard system.covered else { return [] }
        let analysis = CallEchoGate.analyze(mic: mic16, reference: ref16)
        guard analysis.delayFrames != nil else { return [] }
        let ranges = CallEchoGate.scale(
            CallEchoGate.mutedRanges(analysis, count: mic16.count), to: rate, limit: mic.samples.count)
        return ranges.map {
            (span.startSample + Int64($0.lowerBound))..<(span.startSample + Int64($0.upperBound))
        }
    }
}

/// Reads one meeting track's samples [from, to) out of its 16-bit mono chunk WAVs; what no chunk
/// covers is silence. `covered` is false when no chunk overlaps at all.
enum MeetingChunkReader {
    static func read(chunks: [MeetingChunk], folder: URL, from: Int64, to: Int64)
        -> (samples: [Float], covered: Bool)?
    {
        guard to > from else { return nil }
        var out = [Float](repeating: 0, count: Int(to - from))
        var covered = false
        for chunk in chunks where chunk.sampleCount > 0 {
            let lower = max(from, chunk.startSample)
            let upper = min(to, chunk.startSample + chunk.sampleCount)
            guard upper > lower else { continue }
            guard let handle = try? FileHandle(forReadingFrom: folder.appendingPathComponent(chunk.file))
            else { return nil }
            defer { try? handle.close() }
            guard let header = try? handle.read(upToCount: 4_096), let dataStart = dataOffset(header)
            else { return nil }
            let first = lower - chunk.startSample
            do {
                try handle.seek(toOffset: UInt64(dataStart) + UInt64(first) * 2)
                guard let bytes = try handle.read(upToCount: Int(upper - lower) * 2) else { continue }
                bytes.withUnsafeBytes { raw in
                    let values = raw.bindMemory(to: Int16.self)
                    let base = Int(lower - from)
                    for i in 0..<values.count {
                        out[base + i] = Float(Int16(littleEndian: values[i])) / 32_768
                    }
                }
                covered = true
            } catch {
                return nil
            }
        }
        return (out, covered)
    }

    /// Byte offset of the samples in a 16-bit mono PCM WAV header; nil for anything else.
    static func dataOffset(_ header: Data) -> Int? {
        let bytes = [UInt8](header)
        guard bytes.count >= 12, bytes[0..<4] == [0x52, 0x49, 0x46, 0x46] else { return nil }
        func u32(_ at: Int) -> Int {
            Int(bytes[at]) | Int(bytes[at + 1]) << 8 | Int(bytes[at + 2]) << 16 | Int(bytes[at + 3]) << 24
        }
        func u16(_ at: Int) -> Int { Int(bytes[at]) | Int(bytes[at + 1]) << 8 }
        var at = 12
        var pcm16Mono = false
        while at + 8 <= bytes.count {
            let id = String(bytes: bytes[at..<(at + 4)], encoding: .ascii)
            let size = u32(at + 4)
            if id == "fmt ", at + 8 + 16 <= bytes.count {
                pcm16Mono = u16(at + 8) == 1 && u16(at + 10) == 1 && u16(at + 22) == 16
            }
            if id == "data" { return pcm16Mono ? at + 8 : nil }
            at += 8 + size + (size & 1)
        }
        return nil
    }
}
