import XCTest

@testable import PhononBar

/// The call-echo gate on a real recorded meeting, opt in only:
///
///     PHONON_ECHO_CHECK_DIR=<Meetings/<id>> [PHONON_ECHO_CHECK_INJECT_DB=-20] \
///         swift test --filter CallEchoLiveCheck
///
/// Reads the meeting's chunks (never writes anything) and prints counts, levels and timings
/// only, never audio or text. Without PHONON_ECHO_CHECK_INJECT_DB it runs `MeetingEchoPlanner` on
/// every mic span as the transcriber would and reports how much it silences (on a recording
/// made on headphones: nothing, it must fail open). With it, a speaker leak of the meeting's
/// own system track (that many dB, 40 ms late, low-passed) is added to the mic in memory, and
/// the gate's suppression of it and its effect on the owner's real speech are measured.
final class CallEchoLiveCheck: XCTestCase {
    func testGateOnARecordedMeeting() throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["PHONON_ECHO_CHECK_DIR"] else {
            throw XCTSkip("set PHONON_ECHO_CHECK_DIR to run")
        }
        let folder = URL(fileURLWithPath: path, isDirectory: true)
        let metadata = try JSONDecoder().decode(
            MeetingMetadata.self, from: Data(contentsOf: folder.appendingPathComponent("metadata.json")))
        var spans: [MeetingSpan] = []
        var cursor: Int64 = 0
        while let span = MeetingSpanPlanner.next(
            track: .mic, chunks: metadata.micChunks, cursor: cursor, rate: metadata.sampleRate)
        {
            spans.append(span)
            cursor = span.endSample
        }
        let rate = Double(metadata.sampleRate)
        if let inject = env["PHONON_ECHO_CHECK_INJECT_DB"].flatMap(Double.init) {
            try injected(gainDB: inject, spans: spans, metadata: metadata, folder: folder, rate: rate)
            return
        }
        var silenced: Int64 = 0
        var spansWithSilence = 0
        var slowest = 0.0
        let started = Date()
        for span in spans {
            let t0 = Date()
            let ranges = MeetingEchoPlanner.silence(span: span, metadata: metadata, directory: folder)
            slowest = max(slowest, Date().timeIntervalSince(t0))
            if !ranges.isEmpty { spansWithSilence += 1 }
            silenced += ranges.reduce(0) { $0 + ($1.upperBound - $1.lowerBound) }
        }
        print(
            String(
                format: "ECHO live: %d mic spans, %d with silence, %.2f s silenced of %.0f s; "
                    + "%.2f s in all, slowest span %.3f s",
                spans.count, spansWithSilence, Double(silenced) / rate,
                Double(cursor) / rate, Date().timeIntervalSince(started), slowest))
    }

    private func injected(
        gainDB: Double, spans: [MeetingSpan], metadata: MeetingMetadata, folder: URL, rate: Double
    ) throws {
        let micOffset = try XCTUnwrap(metadata.micStartOffsetSamples)
        let systemOffset = try XCTUnwrap(metadata.systemStartOffsetSamples)
        let lookback = Int64(Double(CallEchoGate.lookbackSamples) * rate / 16_000)
        let lead = Int64(Double(CallEchoGate.leadSamples) * rate / 16_000)
        let g = Float(pow(10, gainDB / 20))
        let delay = Int(0.040 * 16_000)
        // Per 1 s frame, over all spans: energies before and after, and what was talking.
        struct Row {
            var owner: Bool, call: Bool
            var before = 0.0, after = 0.0, leak = 0.0, leakAfter = 0.0, ownerEnergy = 0.0, ownerAfter = 0.0
            /// 10 ms frames where the owner is louder than the leak, and those silenced.
            var ownerFrames = 0, ownerMuted = 0
        }
        var rows: [Row] = []
        var found = 0
        for span in spans {
            let systemStart = span.startSample + micOffset - systemOffset
            guard
                let mic = MeetingChunkReader.read(
                    chunks: span.chunks, folder: folder.appendingPathComponent("mic"),
                    from: span.startSample, to: span.endSample),
                let system = MeetingChunkReader.read(
                    chunks: metadata.systemChunks, folder: folder.appendingPathComponent("system"),
                    from: systemStart - lookback, to: systemStart + (span.endSample - span.startSample) + lead)
            else { continue }
            let owner16 = CallEchoGate.to16k(mic.samples, rate: rate)
            let ref16 = CallEchoGate.to16k(system.samples, rate: rate)
            let base = CallEchoGate.lookbackSamples
            var leak = [Float](repeating: 0, count: owner16.count)
            var lowpass: Float = 0
            for i in 0..<owner16.count {
                let k = base + i - delay
                let x = k >= 0 && k < ref16.count ? ref16[k] : 0
                lowpass = 0.5 * lowpass + 0.5 * x
                leak[i] = g * lowpass
            }
            let mixed = zip(owner16, leak).map { $0 + $1 }
            let analysis = CallEchoGate.analyze(mic: mixed, reference: ref16)
            if analysis.delayFrames != nil { found += 1 }
            if ProcessInfo.processInfo.environment["PHONON_ECHO_CHECK_VERBOSE"] != nil {
                let mutedShares = zip(analysis.muted, analysis.echoShare).filter { $0.0 }.map { $0.1 }.sorted()
                print(
                    "ECHO span \(span.startSample / 48_000) s: delay \(String(describing: analysis.delayFrames)) "
                        + "shift \(analysis.shift) explained \(analysis.explained) muted \(analysis.mutedFrames) "
                        + "median muted share \(mutedShares.isEmpty ? 0 : mutedShares[mutedShares.count / 2])")
            }
            let gains = CallEchoGate.gains(
                count: mixed.count, silenced: CallEchoGate.mutedRanges(analysis, count: mixed.count),
                rate: 16_000)
            let second = 16_000
            var start = 0
            while start + second <= mixed.count {
                var row = Row(owner: false, call: false)
                for i in start..<(start + second) {
                    let o = Double(owner16[i])
                    let l = Double(leak[i])
                    let gi = Double(gains[i])
                    row.ownerEnergy += o * o
                    row.ownerAfter += o * o * gi * gi
                    row.leak += l * l
                    row.leakAfter += l * l * gi * gi
                    row.before += (o + l) * (o + l)
                    row.after += (o + l) * (o + l) * gi * gi
                }
                var f = start
                while f + 160 <= start + second {
                    let frame = f..<(f + 160)
                    let o = frame.reduce(0.0) { $0 + Double(owner16[$1]) * Double(owner16[$1]) }
                    let l = frame.reduce(0.0) { $0 + Double(leak[$1]) * Double(leak[$1]) }
                    if o / 160 > 1e-4, o > l {
                        row.ownerFrames += 1
                        if frame.allSatisfy({ gains[$0] == 0 }) { row.ownerMuted += 1 }
                    }
                    f += 160
                }
                // 1 s levels of the real tracks: owner above -40 dBFS talks, under -45 is quiet;
                // the call above -50 dBFS plays, under -60 is silent.
                let ownerPower = row.ownerEnergy / Double(second)
                let callPower = row.leak / Double(second) / Double(g * g)
                let ownerTalks = ownerPower > 1e-4
                let ownerQuiet = ownerPower < 3.2e-5
                let callPlays = callPower > 1e-5
                let callSilent = callPower < 1e-6
                if (ownerTalks || ownerQuiet) && (callPlays || callSilent) {
                    row.owner = ownerTalks
                    row.call = callPlays
                    rows.append(row)
                }
                start += second
            }
        }
        func db(_ x: Double) -> Double { 10 * log10(max(x, 1e-30)) }
        let callOnly = rows.filter { $0.call && !$0.owner }
        let ownerOnly = rows.filter { $0.owner && !$0.call }
        let both = rows.filter { $0.owner && $0.call }
        func sum(_ rows: [Row], _ key: KeyPath<Row, Double>) -> Double { rows.reduce(0) { $0 + $1[keyPath: key] } }
        let mixSuppression = db(sum(callOnly, \.before) / sum(callOnly, \.after))
        let leakSuppression = db(sum(callOnly, \.leak) / sum(callOnly, \.leakAfter))
        let ownerChange = db(sum(ownerOnly, \.after) / sum(ownerOnly, \.before))
        let bothOwnerKept = db(sum(both, \.ownerAfter) / sum(both, \.ownerEnergy))
        let bothLeakSuppression = db(sum(both, \.leak) / sum(both, \.leakAfter))
        let bothDrop =
            Double(both.reduce(0) { $0 + $1.ownerMuted }) / Double(max(1, both.reduce(0) { $0 + $1.ownerFrames }))
        print(
            String(
                format: "ECHO injected %.0f dB: delay found in %d of %d spans; seconds call-only %d, owner-only %d, "
                    + "both %d. Call-only: leak suppressed %.1f dB (mic incl. its own noise %.1f dB). Owner-only: "
                    + "level %.2f dB. Both: owner energy %.2f dB, leak suppressed %.1f dB, owner-louder 10 ms "
                    + "frames silenced %.1f%%",
                gainDB, found, spans.count, callOnly.count, ownerOnly.count, both.count, leakSuppression,
                mixSuppression, ownerChange, bothOwnerKept, bothLeakSuppression, 100 * bothDrop))
    }
}
