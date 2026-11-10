import XCTest

@testable import PhononBar

/// Synthetic mixes for the call-echo gate (SPEC "Only your voice"): a known "owner" signal plus a
/// delayed, attenuated, filtered copy of a "call" signal on the mic, and the clean call on the
/// system track.
final class CallEchoTests: XCTestCase {
    static let rate = 48_000
    static let seconds = 12.0

    /// Deterministic noise.
    struct Noise {
        var state: UInt64
        mutating func next() -> Float {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            z ^= z >> 31
            return Float(Double(z >> 11) / Double(1 << 53)) * 2 - 1
        }
    }

    /// Speech-like: noise through two resonances, under a 3-5 Hz syllable envelope, present
    /// only in `segments` (seconds).
    static func talker(
        seed: UInt64, formants: [Double], syllableHz: Double, segments: [ClosedRange<Double>],
        count: Int = Int(seconds * Double(rate))
    ) -> [Float] {
        var noise = Noise(state: seed)
        var out = [Float](repeating: 0, count: count)
        let fs = Double(rate)
        var states = formants.map { _ in (0.0, 0.0) }
        for i in 0..<count {
            let t = Double(i) / fs
            let input = Double(noise.next())
            var y = 0.0
            for (j, f) in formants.enumerated() {
                let r = 0.995
                let a1 = -2 * r * cos(2 * Double.pi * f / fs)
                let a2 = r * r
                let v = input - a1 * states[j].0 - a2 * states[j].1
                states[j] = (v, states[j].0)
                y += v * 0.002
            }
            let syllable = 0.5 - 0.5 * cos(2 * Double.pi * syllableHz * t)
            let on = segments.contains { $0.contains(t) } ? 1.0 : 0.0
            out[i] = Float(y * syllable * on)
        }
        return out
    }

    /// What the mic hears of the call through speakers: delayed, low-passed, a reflection, quieter.
    static func leak(_ call: [Float], delaySeconds: Double, gainDB: Double) -> [Float] {
        let delay = Int(delaySeconds * Double(rate))
        let reflection = Int(0.008 * Double(rate))
        let g = Float(pow(10, gainDB / 20))
        var out = [Float](repeating: 0, count: call.count)
        var lowpass: Float = 0
        for i in 0..<call.count {
            let direct = i >= delay ? call[i - delay] : 0
            let bounce = i >= delay + reflection ? call[i - delay - reflection] : 0
            lowpass = 0.5 * lowpass + 0.5 * (direct + 0.3 * bounce)
            out[i] = g * lowpass
        }
        return out
    }

    static func add(_ a: [Float], _ b: [Float]) -> [Float] { zip(a, b).map { $0 + $1 } }

    static func energy(_ x: [Float], _ range: ClosedRange<Double>) -> Double {
        let lo = Int(range.lowerBound * Double(rate))
        let hi = min(x.count, Int(range.upperBound * Double(rate)))
        return x[lo..<hi].reduce(0) { $0 + Double($1) * Double($1) }
    }

    static func db(_ ratio: Double) -> Double { 10 * log10(max(ratio, 1e-30)) }

    // call only 0-3 s, owner only 3-6 s, both 6-9 s, call only 9-12 s
    static let callSegments: [ClosedRange<Double>] = [0...3, 6...12]
    static let ownerSegments: [ClosedRange<Double>] = [3...9]
    static let owner: [Float] = talker(
        seed: 1, formants: [520, 1_480], syllableHz: 4, segments: ownerSegments)
    static let call: [Float] = talker(
        seed: 2, formants: [730, 2_240], syllableHz: 3.3, segments: callSegments)

    /// Writes a two-track meeting and returns its folder and metadata.
    static func meeting(mic: [Float], system: [Float], micOffset: Int64 = 0, systemOffset: Int64 = 0)
        throws -> (URL, MeetingMetadata)
    {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("phonon-echo-\(UUID().uuidString)", isDirectory: true)
        var micChunks: [MeetingChunk] = []
        var systemChunks: [MeetingChunk] = []
        let micWriter = try ChunkedWavWriter(directory: folder.appendingPathComponent("mic")) {
            micChunks.append($0)
        }
        let systemWriter = try ChunkedWavWriter(directory: folder.appendingPathComponent("system")) {
            systemChunks.append($0)
        }
        try micWriter.append(mic)
        try micWriter.finish()
        try systemWriter.append(system)
        try systemWriter.finish()
        var metadata = MeetingMetadata(id: "m", startedAt: "2026-11-09T12:00:00Z", startedHostNs: 0)
        metadata.micChunks = micChunks
        metadata.systemChunks = systemChunks
        metadata.micStartOffsetSamples = micOffset
        metadata.systemStartOffsetSamples = systemOffset
        return (folder, metadata)
    }

    /// The mic track as the speech model gets it: every span's silence applied.
    static func gatedMeetingMic(_ mic: [Float], system: [Float], micOffset: Int64 = 0, systemOffset: Int64 = 0)
        throws -> (gated: [Float], silencedSeconds: Double)
    {
        let (folder, metadata) = try meeting(
            mic: mic, system: system, micOffset: micOffset, systemOffset: systemOffset)
        defer { try? FileManager.default.removeItem(at: folder) }
        var gated = mic
        var cursor: Int64 = 0
        var silenced: Int64 = 0
        while let span = MeetingSpanPlanner.next(
            track: .mic, chunks: metadata.micChunks, cursor: cursor, rate: metadata.sampleRate)
        {
            for range in MeetingEchoPlanner.silence(span: span, metadata: metadata, directory: folder) {
                XCTAssertGreaterThanOrEqual(range.lowerBound, span.startSample)
                XCTAssertLessThanOrEqual(range.upperBound, span.endSample)
                for i in range { gated[Int(i)] = 0 }
                silenced += range.upperBound - range.lowerBound
            }
            cursor = span.endSample
        }
        return (gated, Double(silenced) / Double(rate))
    }

    struct Scores {
        var suppressionDB: Double
        var ownerChangeDB: Double
        var bothChangeDB: Double
        var droppedOwnerFrames: Double
    }

    /// Suppression of the call in the call-only stretches, the owner's level change alone and
    /// while both talk, and the share of 10 ms frames with the owner audible (above -30 dB of
    /// the owner's peak frame) that were silenced while both talk.
    static func score(before: [Float], after: [Float], owner: [Float]) -> Scores {
        let callOnly = [0.2...2.8, 9.2...11.8]
        let suppression = db(
            callOnly.map { energy(before, $0) }.reduce(0, +)
                / callOnly.map { energy(after, $0) }.reduce(0, +))
        let ownerChange = db(energy(after, 3.2...5.8) / energy(before, 3.2...5.8))
        let bothChange = db(energy(after, 6.2...8.8) / energy(before, 6.2...8.8))
        let frame = rate / 100
        var frames: [(owner: Double, kept: Bool)] = []
        var start = Int(6.2 * Double(rate))
        while start + frame <= Int(8.8 * Double(rate)) {
            let o = owner[start..<(start + frame)].reduce(0) { $0 + Double($1) * Double($1) }
            let kept = after[start..<(start + frame)].contains { $0 != 0 }
            frames.append((o, kept))
            start += frame
        }
        let peak = frames.map(\.owner).max() ?? 0
        let audible = frames.filter { $0.owner > peak * 1e-3 }
        let dropped = Double(audible.filter { !$0.kept }.count) / Double(max(1, audible.count))
        return Scores(
            suppressionDB: suppression, ownerChangeDB: ownerChange, bothChangeDB: bothChange,
            droppedOwnerFrames: dropped)
    }

    func testMeetingMicTrackLosesTheCallAndKeepsTheOwner() throws {
        let owner = Self.owner
        let call = Self.call
        for (delay, gain) in [(0.037, -12.0), (0.120, -6.0), (0.012, -20.0)] {
            let mic = Self.add(owner, Self.leak(call, delaySeconds: delay, gainDB: gain))
            let (gated, _) = try Self.gatedMeetingMic(mic, system: call)
            let s = Self.score(before: mic, after: gated, owner: owner)
            print(
                String(
                    format: "echo meeting delay %.0f ms gain %.0f dB: call suppressed %.1f dB, owner %.2f dB, "
                        + "both %.2f dB, owner frames dropped while both talk %.1f%%",
                    delay * 1000, gain, s.suppressionDB, s.ownerChangeDB, s.bothChangeDB,
                    100 * s.droppedOwnerFrames))
            XCTAssertGreaterThanOrEqual(s.suppressionDB, 20, "delay \(delay) gain \(gain)")
            XCTAssertLessThanOrEqual(abs(s.ownerChangeDB), 1, "delay \(delay) gain \(gain)")
            XCTAssertLessThanOrEqual(s.droppedOwnerFrames, 0.10, "delay \(delay) gain \(gain)")
        }
    }

    func testTimelineOffsetsAreHonoured() throws {
        // The system track starts 0.25 s after the mic: its sample 0 sits at mic sample 12,000.
        let owner = Self.owner
        let call = Self.call
        let mic = Self.add(owner, Self.leak(call, delaySeconds: 0.03, gainDB: -10))
        let lateSystem = Array(call[12_000...])
        let (gated, _) = try Self.gatedMeetingMic(mic, system: lateSystem, micOffset: 0, systemOffset: 12_000)
        let s = Self.score(before: mic, after: gated, owner: owner)
        XCTAssertLessThanOrEqual(abs(s.ownerChangeDB), 1)
        // Before the system track starts there is no reference: that stretch stays as it was.
        XCTAssertEqual(Self.energy(gated, 0...0.25), Self.energy(mic, 0...0.25), accuracy: 1e-9)
        let callOnly = [0.5...2.8, 9.2...11.8]
        let suppression = Self.db(
            callOnly.map { Self.energy(mic, $0) }.reduce(0, +)
                / callOnly.map { Self.energy(gated, $0) }.reduce(0, +))
        XCTAssertGreaterThanOrEqual(suppression, 20)
    }

    func testFailsOpenWithoutAMatchingReference() throws {
        let n = 7 * Self.rate
        let owner = Array(Self.owner[0..<n])
        let call = Array(Self.call[0..<n])
        // Headphones: the call never reaches the mic.
        var result = try Self.gatedMeetingMic(owner, system: call)
        XCTAssertEqual(result.silencedSeconds, 0)
        XCTAssertEqual(result.gated, owner)
        // A reference that is some other sound than what leaked.
        let other = Self.talker(
            seed: 9, formants: [610, 1_900], syllableHz: 3.7, segments: Self.callSegments, count: n)
        let mic = Self.add(owner, Self.leak(call, delaySeconds: 0.04, gainDB: -10))
        result = try Self.gatedMeetingMic(mic, system: other)
        XCTAssertEqual(result.silencedSeconds, 0)
        // Misaligned by 2 s, beyond the delays the gate looks at.
        let shifted = [Float](repeating: 0, count: 96_000) + call.dropLast(96_000)
        result = try Self.gatedMeetingMic(mic, system: shifted)
        XCTAssertEqual(result.silencedSeconds, 0)
        // A silent system track.
        result = try Self.gatedMeetingMic(mic, system: [Float](repeating: 0, count: call.count))
        XCTAssertEqual(result.silencedSeconds, 0)
    }

    func testNoSilenceForSystemSpansOrAReanchoredClock() throws {
        let call = Self.call
        let mic = Self.add(Self.owner, Self.leak(call, delaySeconds: 0.04, gainDB: -10))
        var (folder, metadata) = try Self.meeting(mic: mic, system: call)
        defer { try? FileManager.default.removeItem(at: folder) }
        let micSpan = try XCTUnwrap(
            MeetingSpanPlanner.next(track: .mic, chunks: metadata.micChunks, cursor: 0, rate: 48_000))
        XCTAssertFalse(MeetingEchoPlanner.silence(span: micSpan, metadata: metadata, directory: folder).isEmpty)
        let systemSpan = try XCTUnwrap(
            MeetingSpanPlanner.next(track: .system, chunks: metadata.systemChunks, cursor: 0, rate: 48_000))
        XCTAssertTrue(MeetingEchoPlanner.silence(span: systemSpan, metadata: metadata, directory: folder).isEmpty)
        metadata.systemDiscontinuities = [MeetingDiscontinuity(sample: 100, hostNs: 1, jumpNs: 50_000_000)]
        XCTAssertTrue(MeetingEchoPlanner.silence(span: micSpan, metadata: metadata, directory: folder).isEmpty)
        metadata.systemDiscontinuities = nil
        metadata.micStartOffsetSamples = nil
        XCTAssertTrue(MeetingEchoPlanner.silence(span: micSpan, metadata: metadata, directory: folder).isEmpty)
    }

    /// A dictation during a meeting: the call reaches the reference in 10 ms buffers stamped on
    /// the host clock, the dictation is gated in 0.4 s blocks as it streams.
    static func gatedDictation(mic: [Float], call: [Float]?, reference: CallReference = CallReference())
        -> (gated: [Float], gate: DictationEchoGate?)
    {
        let start: UInt64 = 5_000_000_000
        if let call {
            reference.begin()
            let buffer = rate / 100
            var i = 0
            while i < call.count {
                let piece = Array(call[i..<min(call.count, i + buffer)])
                let host = start + UInt64(i) * 1_000_000_000 / UInt64(rate)
                reference.append(piece, rate: Double(rate), hostNs: host)
                i += buffer
            }
        }
        guard let gate = DictationEchoGate.make(reference: reference, sourceRate: Double(rate)) else {
            return (mic, nil)
        }
        gate.noteStart(hostNs: start)
        var out: [Float] = []
        let block = Int(0.4 * Double(rate))
        var i = 0
        while i < mic.count {
            let piece = Array(mic[i..<min(mic.count, i + block)])
            if let gains = gate.gains(for: piece) {
                XCTAssertEqual(gains.count, piece.count)
                out += zip(piece, gains).map { $0 * $1 }
            } else {
                out += piece
            }
            i += block
        }
        return (out, gate)
    }

    func testDictationDuringAMeetingLosesTheCall() {
        let owner = Self.owner
        let call = Self.call
        let mic = Self.add(owner, Self.leak(call, delaySeconds: 0.037, gainDB: -12))
        let (gated, gate) = Self.gatedDictation(mic: mic, call: call)
        XCTAssertNotNil(gate)
        // The first second of call audio trains the gate and is left as it was.
        let warm = Self.db(Self.energy(mic, 0...1.0) / Self.energy(gated, 0...1.0))
        let callOnly = [1.2...2.8, 9.2...11.8]
        let suppression = Self.db(
            callOnly.map { Self.energy(mic, $0) }.reduce(0, +)
                / callOnly.map { Self.energy(gated, $0) }.reduce(0, +))
        let s = Self.score(before: mic, after: gated, owner: owner)
        print(
            String(
                format: "echo dictation: first second %.1f dB, call suppressed after it %.1f dB, owner %.2f dB, "
                    + "both %.2f dB, owner frames dropped while both talk %.1f%%",
                warm, suppression, s.ownerChangeDB, s.bothChangeDB, 100 * s.droppedOwnerFrames))
        XCTAssertGreaterThanOrEqual(suppression, 20)
        XCTAssertLessThanOrEqual(abs(s.ownerChangeDB), 1)
        XCTAssertLessThanOrEqual(s.droppedOwnerFrames, 0.10)
    }

    func testDictationIsUntouchedWithoutAMeeting() {
        let mic = Self.add(Self.owner, Self.leak(Self.call, delaySeconds: 0.037, gainDB: -12))
        // No meeting records: no gate at all.
        let idle = CallReference()
        XCTAssertNil(DictationEchoGate.make(reference: idle, sourceRate: 48_000))
        XCTAssertFalse(CallReference.shared.isActive)
        XCTAssertNil(DictationEchoGate.make(sourceRate: 48_000))
        XCTAssertEqual(Self.gatedDictation(mic: mic, call: nil).gated, mic)
        // A meeting that has no call audio yet: a gate, but every block left alone.
        let empty = CallReference()
        empty.begin()
        let gate = DictationEchoGate.make(reference: empty, sourceRate: 48_000)
        XCTAssertNotNil(gate)
        gate?.noteStart(hostNs: 1_000_000_000)
        XCTAssertNil(gate?.gains(for: Array(mic[0..<19_200])))
        // Headphones during a meeting: the reference never matches the mic.
        let owner = Array(Self.owner[0..<(6 * Self.rate)])
        let (gated, used) = Self.gatedDictation(mic: owner, call: Array(Self.call[0..<(6 * Self.rate)]))
        XCTAssertEqual(gated, owner)
        XCTAssertEqual(used?.mutedSamples, 0)
        // A dictation with no host time for its audio cannot be aligned.
        let reference = CallReference()
        reference.begin()
        reference.append(Self.call, rate: 48_000, hostNs: 1_000_000_000)
        let unstamped = DictationEchoGate(reference: reference, sourceRate: 48_000)
        unstamped.noteStart(hostNs: nil)
        XCTAssertNil(unstamped.gains(for: Array(mic[0..<19_200])))
    }

    func testCallReferenceKeepsHostTimeAcrossGapsAndJumps() {
        let reference = CallReference()
        XCTAssertNil(reference.window(startHostNs: 0, count: 10))
        reference.begin()
        let second: UInt64 = 1_000_000_000
        reference.append([Float](repeating: 0.5, count: 48_000), rate: 48_000, hostNs: 10 * second)
        // A 0.5 s gap is silence.
        reference.append([Float](repeating: 0.25, count: 48_000), rate: 48_000, hostNs: 11 * second + second / 2)
        let window = reference.window(startHostNs: Int64(10 * second), count: 40_000)
        XCTAssertEqual(window?[100], 0.5)
        XCTAssertEqual(window?[20_000], 0)
        XCTAssertEqual(window?[30_000], 0.25)
        // Before the history: silence, then what there is.
        let early = reference.window(startHostNs: Int64(9 * second), count: 32_000)
        XCTAssertEqual(early?[0], 0)
        XCTAssertEqual(early?[16_100], 0.5)
        XCTAssertNil(reference.window(startHostNs: Int64(20 * second), count: 100))
        // A buffer stamped earlier than where the history ends starts it over.
        reference.append([Float](repeating: 0.75, count: 4_800), rate: 48_000, hostNs: 3 * second)
        XCTAssertEqual(reference.window(startHostNs: Int64(3 * second), count: 10)?[5], 0.75)
        reference.end()
        XCTAssertFalse(reference.isActive)
        XCTAssertNil(reference.window(startHostNs: Int64(3 * second), count: 10))
    }

    func testGainsRampAtEdgesAndRangesScale() {
        let gains = CallEchoGate.gains(count: 1_000, silenced: [400..<600], rate: 16_000)
        XCTAssertEqual(gains[0], 1)
        XCTAssertEqual(gains[450], 0)
        XCTAssertGreaterThan(gains[399], 0)
        XCTAssertLessThan(gains[399], 1)
        XCTAssertGreaterThan(gains[600], 0)
        XCTAssertLessThan(gains[600], 1)
        XCTAssertEqual(gains[999], 1)
        XCTAssertEqual(CallEchoGate.scale([10..<20], to: 48_000, limit: 1_000), [30..<60])
        XCTAssertEqual(CallEchoGate.scale([10..<20], to: 48_000, limit: 45), [30..<45])
        XCTAssertEqual(CallEchoGate.to16k([1, 2, 3, 4, 5, 6, 7], rate: 48_000), [2, 5])
    }
}
