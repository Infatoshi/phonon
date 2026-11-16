import XCTest

@testable import PhononBar

/// GOAL N9 on recorded audio: a synthetic call is written through the recorder's own chunk
/// writer and index (16 kHz, 5 s chunks, host-stamped buffers with jitter, a mic stall padded
/// with silence and a sleep that re-anchors the clock), with 20 dictations stamped on the host
/// clock as the app stamps them. A stand-in engine reads the chunk WAVs the transcriber points
/// it at, as the sidecar does, and "hears" each tone burst as a word: the user's own speech is a
/// 300 Hz tone, the call's 500 Hz, and dictation `i`'s words are 1,000 + 50 i Hz, so a word in
/// the transcript says which dictation it came from. Nothing here is told where the dictations
/// fell on the track; that comes from the audio and the metadata alone.
final class MeetingDictationAudioTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("phonon-n9-audio-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private static let rate = MeetingFormat.sampleRate
    private static let base: UInt64 = 5_000_000_000

    /// One tone burst on the host timeline (seconds).
    private struct Burst {
        let start: Double
        let length: Double
        let hz: Double
    }

    /// Host seconds with audio: a 2 s mic stall at 150 s (padded with silence), then the lid
    /// closed from 200 s to 900 s (beyond TrackClock.maxGapSeconds, so the clock re-anchors).
    private static let segments: [ClosedRange<Double>] = [0...150, 152...200, 900...1060]

    /// 20 dictations of 4 s: 10, 3 and 7 in the three stretches.
    private static let dictations: [ClosedRange<Double>] = {
        var out: [ClosedRange<Double>] = []
        for (segment, count) in zip(segments, [10, 3, 7]) {
            for index in 0..<count {
                let start = segment.lowerBound + 8 + 14 * Double(index)
                out.append(start...(start + 4))
            }
        }
        return out
    }()

    private static func dictationHz(_ index: Int) -> Double { 1_000 + 50 * Double(index) }

    /// The user talks to the call on a 0.5 s grid and stops 50 ms before pressing the key and
    /// resumes 50 ms after releasing it; inside a dictation they dictate (words from 0.25 s
    /// after the key went down to 0.15 s before it came up).
    private static func micBursts() -> [Burst] {
        var bursts: [Burst] = []
        for segment in segments {
            var at = segment.lowerBound
            while at + 0.35 <= segment.upperBound {
                let word = at...(at + 0.35)
                let clear = dictations.allSatisfy { window in
                    word.upperBound < window.lowerBound - 0.05 || word.lowerBound > window.upperBound + 0.05
                }
                if clear { bursts.append(Burst(start: at, length: 0.35, hz: 300)) }
                at += 0.5
            }
        }
        for (index, window) in dictations.enumerated() {
            var at = window.lowerBound + 0.25
            while at + 0.35 <= window.upperBound - 0.15 {
                bursts.append(Burst(start: at, length: 0.35, hz: dictationHz(index)))
                at += 0.5
            }
        }
        return bursts.sorted { $0.start < $1.start }
    }

    private static func systemBursts() -> [Burst] {
        segments.flatMap { segment in
            stride(from: segment.lowerBound + 0.1, through: segment.upperBound - 0.4, by: 1.3).map {
                Burst(start: $0, length: 0.3, hz: 500)
            }
        }
    }

    /// Writes one track through ChunkedWavWriter in 100 ms buffers whose host stamps carry up
    /// to 6 ms of jitter (the samples themselves are continuous within a stretch).
    private static func record(_ bursts: [Burst], to writer: ChunkedWavWriter, jitterSeed: UInt64) throws {
        let frames = rate / 10
        var seed = jitterSeed
        var next = 0
        for segment in segments {
            var buffer = 0
            while true {
                let t0 = segment.lowerBound + Double(buffer * frames) / Double(rate)
                guard t0 + Double(frames) / Double(rate) <= segment.upperBound else { break }
                var samples = [Float](repeating: 0, count: frames)
                while next < bursts.count, bursts[next].start + bursts[next].length < t0 { next += 1 }
                var index = next
                while index < bursts.count, bursts[index].start < t0 + 0.1 {
                    let burst = bursts[index]
                    for frame in 0..<frames {
                        let t = t0 + Double(frame) / Double(rate) - burst.start
                        if t >= 0, t < burst.length {
                            samples[frame] += Float(0.3 * sin(2 * .pi * burst.hz * t))
                        }
                    }
                    index += 1
                }
                seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                let jitterNs = Int64(seed >> 33) % 12_000_000 - 6_000_000
                let stamp = UInt64(Int64(base) + Int64((t0 * 1e9).rounded()) + jitterNs)
                try writer.append(samples, hostNs: stamp)
                buffer += 1
            }
        }
        try writer.finish()
    }

    /// The sidecar's `read_span`: samples [start, end) of a track from the chunk WAVs that
    /// cover it, silence where none does.
    private static func readSpan(_ command: [String: Any]) throws -> [Float] {
        let start = command["start_sample"] as! Int64
        let end = command["end_sample"] as! Int64
        var out = [Float](repeating: 0, count: Int(end - start))
        for chunk in command["chunks"] as! [[String: Any]] {
            let data = try Data(contentsOf: URL(fileURLWithPath: chunk["path"] as! String))
            let first = chunk["start_sample"] as! Int64
            let count = (data.count - 44) / 2
            data.withUnsafeBytes { raw in
                for index in 0..<count {
                    let at = first + Int64(index)
                    guard at >= start, at < end else { continue }
                    let value = raw.loadUnaligned(fromByteOffset: 44 + index * 2, as: Int16.self)
                    out[Int(at - start)] = Float(Int16(littleEndian: value)) / 32768
                }
            }
        }
        return out
    }

    /// Words heard in a span: runs of samples above a threshold (gaps under 25 ms joined), each
    /// named by its tone and by its first sample on the track (stable across re-transcribed
    /// overlaps).
    private static func hear(_ samples: [Float], trackStart: Int64) -> [[String: Any]] {
        var words: [[String: Any]] = []
        var index = 0
        let threshold: Float = 0.01
        while index < samples.count {
            guard abs(samples[index]) > threshold else {
                index += 1
                continue
            }
            let first = index
            var last = index
            var probe = index + 1
            while probe < samples.count, probe - last < 400 {
                if abs(samples[probe]) > threshold { last = probe }
                probe += 1
            }
            var crossings = 0
            for at in (first + 1)...max(first + 1, last) where at < samples.count {
                if (samples[at - 1] < 0) != (samples[at] < 0) { crossings += 1 }
            }
            let seconds = Double(last - first + 1) / Double(rate)
            let hz = Double(crossings) / 2 / seconds
            let name: String
            if hz < 400 {
                name = "m\(trackStart + Int64(first))"
            } else if hz < 700 {
                name = "c\(trackStart + Int64(first))"
            } else {
                name = "d\(Int(((hz - 1_000) / 50).rounded()))"
            }
            words.append([
                "start": Double(first) / Double(rate), "end": Double(last + 1) / Double(rate), "text": name,
            ])
            index = last + 1
        }
        return words
    }

    private struct Run {
        var heard: [String: Set<String>]
        var me: Set<String>
        var call: Set<String>
        var spans: Int
    }

    /// Records the call, transcribes it through MeetingTranscriber with the tone engine, and
    /// returns what was heard on each track and what each side of the transcript kept.
    @MainActor
    private func run(stampDictations: Bool) throws -> Run {
        let id = stampDictations ? "meeting_n9_audio" : "meeting_n9_unstamped"
        let folder = directory.appendingPathComponent(id, isDirectory: true)
        let index = try MeetingIndex(
            directory: folder,
            metadata: MeetingMetadata(id: id, startedAt: "2026-11-12T07:00:00Z", startedHostNs: Self.base))
        let mic = try ChunkedWavWriter(
            directory: folder.appendingPathComponent("mic", isDirectory: true),
            onChunkClosed: { index.chunkClosed($0, track: .mic) })
        let system = try ChunkedWavWriter(
            directory: folder.appendingPathComponent("system", isDirectory: true),
            onChunkClosed: { index.chunkClosed($0, track: .system) })
        try Self.record(Self.micBursts(), to: mic, jitterSeed: 1)
        try Self.record(Self.systemBursts(), to: system, jitterSeed: 2)
        // as MeetingController stamps them: capture start and end on the host clock
        for window in Self.dictations where stampDictations {
            index.dictation(
                MeetingDictation(
                    startHostNs: Self.base + UInt64(window.lowerBound * 1e9),
                    endHostNs: Self.base + UInt64(window.upperBound * 1e9)))
        }
        index.end(reason: "call_ended")
        XCTAssertEqual(index.metadata.micDiscontinuities?.count, 1, "the sleep re-anchored the mic clock")

        var heard: [String: Set<String>] = ["mic": [], "system": []]
        var sent: [[String: Any]] = []
        let transcriber = MeetingTranscriber()
        transcriber.send = { sent.append($0) }
        var finished = false
        transcriber.onFinished = { _ in finished = true }
        transcriber.setEngineReady(true)
        transcriber.enqueue(directory: folder)
        var answered = 0
        while answered < sent.count, answered < 100 {
            let command = sent[answered]
            let id = command["id"] as! String
            let track = id.contains(":mic:") ? "mic" : "system"
            let words = Self.hear(try Self.readSpan(command), trackStart: command["start_sample"] as! Int64)
            heard[track]!.formUnion(words.map { $0["text"] as! String })
            let sentences = stride(from: 0, to: words.count, by: 8).map { from -> [String: Any] in
                let group = Array(words[from..<min(words.count, from + 8)])
                return [
                    "start": group.first!["start"]!, "end": group.last!["end"]!,
                    "text": group.map { $0["text"] as! String }.joined(separator: " "), "words": group,
                ]
            }
            transcriber.handle(["type": "meeting_span", "id": id, "sentences": sentences, "model": "tones"])
            answered += 1
        }
        XCTAssertTrue(finished)
        let transcript = try XCTUnwrap(MeetingTranscript.load(from: folder))
        func words(_ speaker: TranscriptSpeaker) -> [String] {
            transcript.turns.filter { $0.speaker == speaker }.flatMap {
                $0.text.split(separator: " ").map { String($0).trimmingCharacters(in: .punctuationCharacters) }
            }
        }
        return Run(heard: heard, me: Set(words(.me)), call: Set(words(.call)), spans: answered)
    }

    @MainActor
    func testTwentyDictationsInARecordedCallAreLeftOutOfTheMeTranscript() throws {
        let result = try run(stampDictations: true)
        let heard = result.heard
        let me = result.me
        let call = result.call
        // every dictation was on the recorded mic track, and none of it is in "me"
        let onTrack = (0..<20).filter { heard["mic"]!.contains("d\($0)") }
        XCTAssertEqual(onTrack.count, 20, "all 20 dictations were recorded and heard")
        let removed = onTrack.filter { !me.contains("d\($0)") }
        XCTAssertEqual(removed.count, 20, "20 of 20 dictations are out of the \"me\" transcript")
        XCTAssertFalse(me.contains { $0.hasPrefix("d") })
        // what the user said to the call stays, right up to 50 ms from each key press and release
        let own = heard["mic"]!.filter { $0.hasPrefix("m") }
        let lost = own.subtracting(me)
        XCTAssertEqual(lost.count, 0, "none of the user's own words are lost")
        // the call's side is never filtered
        let callHeard = heard["system"]!
        XCTAssertEqual(callHeard.subtracting(call).count, 0)
        print(
            "N9 recorded audio: \(removed.count) of 20 dictations removed from \"me\"; own words kept "
                + "\(own.count - lost.count) of \(own.count); call words kept "
                + "\(callHeard.count - callHeard.subtracting(call).count) of \(callHeard.count); "
                + "\(result.spans) spans")
    }

    /// The control: the same recording without the dictation stamps keeps all 20 in "me", so
    /// the count above comes from the stamps, not from the tone engine.
    @MainActor
    func testWithoutDictationStampsTheDictationsStayInMe() throws {
        let result = try run(stampDictations: false)
        XCTAssertEqual((0..<20).filter { result.me.contains("d\($0)") }.count, 20)
    }
}
