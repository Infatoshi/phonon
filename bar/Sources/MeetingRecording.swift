import AVFoundation
import CoreMedia
import Foundation
import PhononObjC
import ScreenCaptureKit
import os

// Meeting capture: the user's microphone and the system audio, each as its
// own 48 kHz mono 16-bit track, in closed 5 s WAV chunks under
// ~/Library/Application Support/Phonon/Meetings/<id>/{mic,system}/NNNN.wav
// with a metadata.json index rewritten atomically as each chunk closes. A
// crash loses at most the open chunk, and a rolling transcriber can read
// chunks as they close; the last, partial chunk is closed at the stop.
// Format agreed with the main session 2026-10-29; 5 s chunks because the
// meeting lane measured chunk length to be the end-of-meeting latency lever.
//
// The two tracks share one clock, the host clock (ns since boot): sample n
// of a track was captured at `<track>_first_sample_host_ns + n / 48 kHz`,
// and `<track>_start_offset_samples` is that first sample's offset from
// `started_host_ns`. To sum the tracks sample-aligned, put sample n of each
// track at timeline sample `<track>_start_offset_samples + n`. A track that
// receives no audio for a while (a stalled device, or no system sound) is
// padded with silence so it stays on the clock; samples are never dropped,
// and each chunk records the host time of its first sample so any drift
// between the clocks of the two devices stays visible.
//
// The microphone is recorded as captured: no voice processing, echo
// cancellation or gating (the pipeline sums the tracks and handles echo).
//
// Recording never starts by itself: detection asks first ("Meeting detected:
// take notes?") unless the user opted into auto-start, which is off by
// default. System audio comes from ScreenCaptureKit, under the Screen
// Recording permission Phonon already asks for, so there is no new prompt.

enum MeetingFormat {
    static let sampleRate = 48_000
    static let chunkSeconds = 5
}

/// The host clock in nanoseconds since boot: the clock of
/// `DispatchTime.now().uptimeNanoseconds`, of AVAudioTime host times (in
/// mach ticks) and of ScreenCaptureKit presentation timestamps.
enum HostClock {
    static func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

    private static let timebase: mach_timebase_info_data_t = {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        return info
    }()

    static func nanoseconds(machTicks: UInt64) -> UInt64 {
        ticksToNanoseconds(machTicks, numer: timebase.numer, denom: timebase.denom)
    }

    /// Split so that ticks times numer cannot overflow.
    static func ticksToNanoseconds(_ ticks: UInt64, numer: UInt32, denom: UInt32) -> UInt64 {
        guard denom != 0 else { return ticks }
        let numer = UInt64(numer)
        let denom = UInt64(denom)
        return ticks / denom * numer + ticks % denom * numer / denom
    }

    /// A host-clock CMTime in nanoseconds; nil when invalid.
    static func nanoseconds(_ time: CMTime) -> UInt64? {
        guard time.isValid, time.isNumeric, time.value >= 0 else { return nil }
        let converted = CMTimeConvertScale(
            time, timescale: 1_000_000_000, method: .roundHalfAwayFromZero)
        return converted.isValid && converted.value >= 0 ? UInt64(converted.value) : nil
    }

    /// A buffer timestamp only if it is plausibly on this clock (within
    /// `window` of now); a timestamp on some other clock places nothing.
    static func plausible(
        _ hostNs: UInt64?, now: UInt64, window: UInt64 = 10_000_000_000
    ) -> UInt64? {
        guard let hostNs else { return nil }
        let distance = hostNs > now ? hostNs - now : now - hostNs
        return distance <= window ? hostNs : nil
    }

    static func nanoseconds(samples: Int64, rate: Int) -> Int64 {
        Int64((Double(samples) * 1e9 / Double(rate)).rounded())
    }

    static func samples(from start: UInt64, to end: UInt64, rate: Int) -> Int64 {
        let delta = Int64(bitPattern: end &- start)
        return Int64((Double(delta) * Double(rate) / 1e9).rounded())
    }

    static func offset(_ hostNs: UInt64, byNanoseconds delta: Int64) -> UInt64 {
        delta >= 0 ? hostNs &+ UInt64(delta) : hostNs &- UInt64(-delta)
    }
}

/// Where a track's clock was re-anchored: a jump in buffer timestamps too
/// large to be silence (`TrackClock.maxGapSeconds`), so the track sample
/// `sample` was captured at `hostNs` rather than where the samples before
/// it would put it. `jumpNs` is how far off that was.
struct MeetingDiscontinuity: Codable, Equatable {
    let sample: Int64
    let hostNs: UInt64
    let jumpNs: Int64

    enum CodingKeys: String, CodingKey {
        case sample
        case hostNs = "host_ns"
        case jumpNs = "jump_ns"
    }
}

/// Keeps one track's sample index on the host clock. A buffer that arrives
/// later than the samples placed so far account for (beyond jitter) is
/// preceded by silence for the gap; a buffer that arrives early is placed
/// as is, so nothing captured is ever dropped. A jump beyond
/// `maxGapSeconds` either way is not filled: the clock is re-anchored on
/// that buffer and the jump recorded, so later stalls are padded again.
struct TrackClock {
    /// Gaps up to this long are timestamp jitter or converter delay.
    static let gapToleranceSeconds = 0.02
    /// The longest gap filled at once; a larger jump is a clock error.
    static let maxGapSeconds = 600.0

    let sampleRate: Int
    /// Host time the track's sample 0 would have under the current anchor,
    /// once any buffer had a timestamp.
    private(set) var anchorHostNs: UInt64?
    /// Samples placed so far, silence included.
    private(set) var placed: Int64 = 0
    private(set) var silenceInserted: Int64 = 0
    private(set) var discontinuities: [MeetingDiscontinuity] = []

    struct Placement: Equatable {
        /// Zeros to write before the buffer.
        let silence: Int
        /// Host time of the first sample written (silence included).
        let startHostNs: UInt64?
        /// The clock was re-anchored on this buffer.
        var discontinuity: MeetingDiscontinuity?
    }

    init(sampleRate: Int) { self.sampleRate = sampleRate }

    private func anchor(for hostNs: UInt64) -> UInt64 {
        HostClock.offset(
            hostNs, byNanoseconds: -HostClock.nanoseconds(samples: placed, rate: sampleRate))
    }

    mutating func place(frames: Int, hostNs: UInt64?) -> Placement {
        var silence = 0
        var start: UInt64?
        var discontinuity: MeetingDiscontinuity?
        if let hostNs {
            if let anchor = anchorHostNs {
                let gap = HostClock.samples(from: anchor, to: hostNs, rate: sampleRate) - placed
                let tolerance = Int64(Self.gapToleranceSeconds * Double(sampleRate))
                let limit = Int64(Self.maxGapSeconds * Double(sampleRate))
                if abs(gap) > limit {
                    let jump = HostClock.nanoseconds(samples: gap, rate: sampleRate)
                    discontinuity = MeetingDiscontinuity(
                        sample: placed, hostNs: hostNs, jumpNs: jump)
                    discontinuities.append(discontinuity!)
                    anchorHostNs = self.anchor(for: hostNs)
                } else if gap > tolerance {
                    silence = Int(gap)
                }
            } else {
                anchorHostNs = anchor(for: hostNs)
            }
            start = HostClock.offset(
                hostNs,
                byNanoseconds: -HostClock.nanoseconds(samples: Int64(silence), rate: sampleRate))
        } else if let anchor = anchorHostNs {
            start = HostClock.offset(
                anchor, byNanoseconds: HostClock.nanoseconds(samples: placed, rate: sampleRate))
        }
        placed += Int64(silence + frames)
        silenceInserted += Int64(silence)
        return Placement(silence: silence, startHostNs: start, discontinuity: discontinuity)
    }
}

/// One closed chunk in a track's index.
struct MeetingChunk: Codable, Equatable {
    let index: Int
    let file: String
    /// Offset of the first sample from the start of the track, in samples.
    let startSample: Int64
    let sampleCount: Int64
    /// Host time (ns since boot) of the chunk's first sample, as measured
    /// from the buffer it came from; nil when the buffers had no timestamp.
    var startHostNs: UInt64?
    /// Clock re-anchors inside this chunk (see `MeetingDiscontinuity`).
    var discontinuities: [MeetingDiscontinuity]?

    enum CodingKeys: String, CodingKey {
        case index, file
        case startSample = "start_sample"
        case sampleCount = "sample_count"
        case startHostNs = "start_host_ns"
        case discontinuities
    }
}

struct MeetingMetadata: Codable, Equatable {
    var schemaVersion = 2
    var id: String
    /// Wall clock (ISO 8601) and host monotonic time (ns since boot, the
    /// clock e2e traces use) when recording began. `startedHostNs` is the
    /// shared timeline's sample 0.
    var startedAt: String
    var startedHostNs: UInt64
    var appBundleID: String?
    var appName: String?
    var windowTitle: String?
    var conferenceURL: String?
    var micDevice: String?
    var systemSource = "screencapturekit"
    var sampleRate = MeetingFormat.sampleRate
    var chunkSeconds = MeetingFormat.chunkSeconds
    var micChunks: [MeetingChunk] = []
    var systemChunks: [MeetingChunk] = []
    /// Host time of each track's sample 0, and its offset from
    /// `startedHostNs` in samples (negative if captured before it).
    var micFirstSampleHostNs: UInt64?
    var micStartOffsetSamples: Int64?
    var systemFirstSampleHostNs: UInt64?
    var systemStartOffsetSamples: Int64?
    /// Clock re-anchors per track (also listed on their chunks): from each
    /// one on, the track's samples follow its `host_ns`, not the offset.
    var micDiscontinuities: [MeetingDiscontinuity]?
    var systemDiscontinuities: [MeetingDiscontinuity]?
    var endedAt: String?
    var endReason: String?

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case id
        case startedAt = "started_at"
        case startedHostNs = "started_host_ns"
        case appBundleID = "app_bundle_id"
        case appName = "app_name"
        case windowTitle = "window_title"
        case conferenceURL = "conference_url"
        case micDevice = "mic_device"
        case systemSource = "system_source"
        case sampleRate = "sample_rate"
        case chunkSeconds = "chunk_seconds"
        case micChunks = "mic_chunks"
        case systemChunks = "system_chunks"
        case micFirstSampleHostNs = "mic_first_sample_host_ns"
        case micStartOffsetSamples = "mic_start_offset_samples"
        case systemFirstSampleHostNs = "system_first_sample_host_ns"
        case systemStartOffsetSamples = "system_start_offset_samples"
        case micDiscontinuities = "mic_discontinuities"
        case systemDiscontinuities = "system_discontinuities"
        case endedAt = "ended_at"
        case endReason = "end_reason"
    }
}

/// Writes one mono track as consecutive closed WAV chunks. Not thread-safe;
/// `MeetingRecorder` feeds it from one serial queue.
final class ChunkedWavWriter {
    let directory: URL
    let sampleRate: Int
    let samplesPerChunk: Int
    private let onChunkClosed: (MeetingChunk) -> Void
    private var handle: FileHandle?
    private var chunkIndex = 0
    private var chunkSamples = 0
    private var trackSamples: Int64 = 0
    private var chunkStart: Int64 = 0
    private var chunkStartHostNs: UInt64?
    /// Re-anchors not yet reported with the chunk that holds them.
    private var pendingDiscontinuities: [MeetingDiscontinuity] = []
    private(set) var clock: TrackClock
    /// Set by `finish`: buffers that arrive after the stop are dropped
    /// rather than opening a chunk nobody will close.
    private(set) var finished = false

    init(
        directory: URL, sampleRate: Int = MeetingFormat.sampleRate,
        chunkSeconds: Double = Double(MeetingFormat.chunkSeconds),
        onChunkClosed: @escaping (MeetingChunk) -> Void
    ) throws {
        self.directory = directory
        self.sampleRate = sampleRate
        self.samplesPerChunk = max(1, Int(Double(sampleRate) * chunkSeconds))
        self.onChunkClosed = onChunkClosed
        clock = TrackClock(sampleRate: sampleRate)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    static func fileName(_ index: Int) -> String { String(format: "%04d.wav", index) }

    /// Append float samples in [-1, 1] at `sampleRate` whose first sample
    /// was captured at `hostNs` (host clock), when known. The samples are
    /// written as given (only clamped and quantized to 16 bits); a gap
    /// before them is filled with silence (see `TrackClock`).
    func append(_ samples: [Float], hostNs: UInt64? = nil) throws {
        guard !finished else { return }
        let placement = clock.place(frames: samples.count, hostNs: hostNs)
        if let discontinuity = placement.discontinuity {
            pendingDiscontinuities.append(discontinuity)
        }
        try write(placement.silence, firstHostNs: placement.startHostNs) { _ in 0 }
        let samplesHostNs = placement.startHostNs.map {
            HostClock.offset(
                $0,
                byNanoseconds: HostClock.nanoseconds(
                    samples: Int64(placement.silence), rate: sampleRate))
        }
        try write(samples.count, firstHostNs: samplesHostNs) { samples[$0] }
    }

    private func write(_ count: Int, firstHostNs: UInt64?, sample: (Int) -> Float) throws {
        var offset = 0
        while offset < count {
            if handle == nil {
                try openChunk(
                    hostNs: firstHostNs.map {
                        HostClock.offset(
                            $0,
                            byNanoseconds: HostClock.nanoseconds(
                                samples: Int64(offset), rate: sampleRate))
                    })
            }
            let room = samplesPerChunk - chunkSamples
            let take = min(room, count - offset)
            var pcm = Data(count: take * 2)
            pcm.withUnsafeMutableBytes { raw in
                let out = raw.bindMemory(to: Int16.self)
                for index in 0..<take {
                    out[index] = Self.pcm16(sample(offset + index)).littleEndian
                }
            }
            handle?.write(pcm)
            chunkSamples += take
            trackSamples += Int64(take)
            offset += take
            if chunkSamples >= samplesPerChunk { try closeChunk() }
        }
    }

    static func pcm16(_ value: Float) -> Int16 {
        guard !value.isNaN else { return 0 }
        return Int16(max(-1, min(1, value)) * Float(Int16.max))
    }

    /// Close the open chunk, if it holds any samples. Final: later appends
    /// are no-ops.
    func finish() throws {
        finished = true
        try closeChunk()
    }

    private func openChunk(hostNs: UInt64?) throws {
        let url = directory.appendingPathComponent(Self.fileName(chunkIndex))
        FileManager.default.createFile(atPath: url.path, contents: Self.header(samples: 0, rate: sampleRate))
        handle = try FileHandle(forWritingTo: url)
        handle?.seekToEndOfFile()
        chunkSamples = 0
        chunkStart = trackSamples
        chunkStartHostNs = hostNs
    }

    private func closeChunk() throws {
        guard let handle else { return }
        handle.seek(toFileOffset: 0)
        handle.write(Self.header(samples: chunkSamples, rate: sampleRate))
        try handle.close()
        self.handle = nil
        let end = chunkStart + Int64(chunkSamples)
        let inside = pendingDiscontinuities.filter { $0.sample < end }
        pendingDiscontinuities.removeAll { $0.sample < end }
        let chunk = MeetingChunk(
            index: chunkIndex, file: Self.fileName(chunkIndex), startSample: chunkStart,
            sampleCount: Int64(chunkSamples), startHostNs: chunkStartHostNs,
            discontinuities: inside.isEmpty ? nil : inside)
        chunkIndex += 1
        onChunkClosed(chunk)
    }

    static func header(samples: Int, rate: Int) -> Data {
        let dataBytes = UInt32(samples * 2)
        var data = Data()
        func append<T: FixedWidthInteger>(_ value: T) {
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        data.append(contentsOf: Array("RIFF".utf8))
        append(UInt32(36) + dataBytes)
        data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8))
        append(UInt32(16))
        append(UInt16(1))  // PCM
        append(UInt16(1))  // mono
        append(UInt32(rate))
        append(UInt32(rate * 2))
        append(UInt16(2))
        append(UInt16(16))
        data.append(contentsOf: Array("data".utf8))
        append(dataBytes)
        return data
    }
}

/// The metadata.json of one meeting, rewritten atomically on every change.
final class MeetingIndex {
    let directory: URL
    private(set) var metadata: MeetingMetadata
    private let lock = OSAllocatedUnfairLock()

    init(directory: URL, metadata: MeetingMetadata) throws {
        self.directory = directory
        self.metadata = metadata
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try save()
    }

    func chunkClosed(_ chunk: MeetingChunk, track: MeetingTrack) {
        lock.withLock {
            // The first timed chunk fixes where the track's sample 0 sits on
            // the shared clock.
            let first = chunk.startHostNs.map {
                HostClock.offset(
                    $0,
                    byNanoseconds: -HostClock.nanoseconds(
                        samples: chunk.startSample, rate: metadata.sampleRate))
            }
            let offset = first.map {
                HostClock.samples(
                    from: metadata.startedHostNs, to: $0, rate: metadata.sampleRate)
            }
            let jumps = chunk.discontinuities ?? []
            switch track {
            case .mic:
                metadata.micChunks.append(chunk)
                if metadata.micFirstSampleHostNs == nil, let first {
                    metadata.micFirstSampleHostNs = first
                    metadata.micStartOffsetSamples = offset
                }
                if !jumps.isEmpty { metadata.micDiscontinuities = (metadata.micDiscontinuities ?? []) + jumps }
            case .system:
                metadata.systemChunks.append(chunk)
                if metadata.systemFirstSampleHostNs == nil, let first {
                    metadata.systemFirstSampleHostNs = first
                    metadata.systemStartOffsetSamples = offset
                }
                if !jumps.isEmpty {
                    metadata.systemDiscontinuities = (metadata.systemDiscontinuities ?? []) + jumps
                }
            }
        }
        try? save()
    }

    func end(reason: String, at date: Date = Date()) {
        lock.withLock {
            metadata.endedAt = ISO8601DateFormatter().string(from: date)
            metadata.endReason = reason
        }
        try? save()
    }

    /// The call window's title and conference URL, read after recording
    /// started.
    func setWindowInfo(title: String?, conferenceURL: String?) {
        lock.withLock {
            metadata.windowTitle = title
            metadata.conferenceURL = conferenceURL
        }
        try? save()
    }

    /// Saves run one at a time and each takes its snapshot inside, so the
    /// last write always holds every change made before it (a chunk closing
    /// on the writer queue and the window info arriving from elsewhere).
    private let saving = NSLock()

    func save() throws {
        saving.lock()
        defer { saving.unlock() }
        let snapshot = lock.withLock { metadata }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(snapshot).write(
            to: directory.appendingPathComponent("metadata.json"), options: .atomic)
    }
}

enum MeetingTrack: String {
    case mic, system
}

/// When a detected meeting is over: the call app (or conference tab) has not
/// held the microphone for `grace` seconds.
struct MeetingEndPolicy {
    static let grace: TimeInterval = 10
    private(set) var lastSeen: TimeInterval

    init(now: TimeInterval) { lastSeen = now }

    mutating func observe(stillInCall: Bool, now: TimeInterval) -> Bool {
        if stillInCall { lastSeen = now }
        return now - lastSeen >= Self.grace
    }
}

/// Records the two tracks. Mic through its own AVAudioEngine (the dictation
/// recorder keeps working alongside), system audio through a ScreenCaptureKit
/// audio-only stream that excludes Phonon's own sound.
/// Start and stop of a recorder can interleave: Stop can be pressed (or the
/// call can end) while ScreenCaptureKit is still starting. A stop wins, and
/// anything that finishes starting after it must be stopped at once.
struct RecorderLifecycle: Equatable {
    enum Phase: Equatable { case idle, starting, running, stopped }

    private(set) var phase: Phase = .idle

    /// Begin starting; false when already started or stopped.
    mutating func beginStart() -> Bool {
        guard phase == .idle else { return false }
        phase = .starting
        return true
    }

    /// After each await in start: keep going only if no stop came in.
    var mayContinue: Bool { phase == .starting }

    /// A device change may bring the microphone back only while recording.
    var mayRestartInput: Bool { phase == .starting || phase == .running }

    /// Start finished; false when a stop came in meanwhile, and the caller
    /// must undo what it just started.
    mutating func finishStart() -> Bool {
        guard phase == .starting else { return false }
        phase = .running
        return true
    }

    /// True the first time only: that caller does the stopping.
    mutating func stop() -> Bool {
        guard phase != .stopped else { return false }
        phase = .stopped
        return true
    }
}

enum MeetingRecorderError: LocalizedError {
    case noMicrophone

    var errorDescription: String? {
        switch self {
        case .noMicrophone: "no microphone input"
        }
    }
}

final class MeetingRecorder: NSObject, SCStreamOutput, @unchecked Sendable {
    private let queue = DispatchQueue(label: "phonon.meeting.writer")
    /// Serializes AVAudioEngine start and stop, which arrive from different
    /// executors.
    private let control = DispatchQueue(label: "phonon.meeting.control")
    private let index: MeetingIndex
    private let mic: ChunkedWavWriter
    private let system: ChunkedWavWriter
    private let engine = AVAudioEngine()
    /// Set and cleared on `control` only.
    private var configurationObserver: NSObjectProtocol?
    /// The lifecycle and the running stream, changed together under one lock
    /// so a stop can never miss a stream that is being attached.
    private struct State {
        var lifecycle = RecorderLifecycle()
        var stream: SCStream?
        var systemRestarts = RestartBudget(limit: 3)
    }
    private let state = OSAllocatedUnfairLock(uncheckedState: State())
    private let target = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: Double(MeetingFormat.sampleRate),
        channels: 1, interleaved: false)!
    /// Fed the system track while recording (see `CallEcho.swift`).
    private let callReference = CallReference.shared

    init(directory: URL, metadata: MeetingMetadata) throws {
        let index = try MeetingIndex(directory: directory, metadata: metadata)
        self.index = index
        mic = try ChunkedWavWriter(
            directory: directory.appendingPathComponent("mic", isDirectory: true),
            onChunkClosed: { index.chunkClosed($0, track: .mic) })
        system = try ChunkedWavWriter(
            directory: directory.appendingPathComponent("system", isDirectory: true),
            onChunkClosed: { index.chunkClosed($0, track: .system) })
    }

    func start() async throws {
        guard state.withLock({ $0.lifecycle.beginStart() }) else { return }
        // The call audio is the reference that keeps it out of a dictation made meanwhile.
        callReference.begin()
        do {
            try control.sync { try startMicrophone() }
        } catch {
            callReference.end()
            throw error
        }
        guard state.withLock({ $0.lifecycle.mayContinue }) else {
            // Stopped while the mic was starting; stop() already closed it,
            // but the engine may have come up after that.
            control.sync { stopMicrophone() }
            return
        }
        let stream = try await startSystemAudio()
        let attached = state.withLock { current -> Bool in
            guard current.lifecycle.mayContinue else { return false }
            current.stream = stream
            return current.lifecycle.finishStart()
        }
        if !attached {
            // A stop came in while ScreenCaptureKit was starting.
            try? await stream?.stopCapture()
            control.sync { stopMicrophone() }
        }
    }

    func setWindowInfo(title: String?, conferenceURL: String?) {
        index.setWindowInfo(title: title, conferenceURL: conferenceURL)
    }

    /// System audio stopped by itself (a display change, a capture error):
    /// start it again while recording, a few times per meeting at most. The
    /// track clock pads the gap with silence.
    private func restartSystemAudio(after stopped: SCStream) async {
        let allowed = state.withLock { current -> Bool in
            guard current.lifecycle.phase == .running, current.stream === stopped,
                current.systemRestarts.take()
            else { return false }
            current.stream = nil
            return true
        }
        guard allowed else { return }
        try? await Task.sleep(for: .seconds(1))
        guard let stream = try? await startSystemAudio() else { return }
        let attached = state.withLock { current -> Bool in
            guard current.lifecycle.phase == .running, current.stream == nil else { return false }
            current.stream = stream
            return true
        }
        if !attached { try? await stream.stopCapture() }
    }

    func stop(reason: String) async {
        guard let stream = takeStreamForStop() else { return }
        if let running = stream { try? await running.stopCapture() }
        finishFiles(reason: reason)
    }

    /// Stop the microphone and close both open chunks now, without waiting
    /// for ScreenCaptureKit; used when the app quits.
    func closeFiles(reason: String) {
        guard let stream = takeStreamForStop() else { return }
        if let running = stream { Task { try? await running.stopCapture() } }
        finishFiles(reason: reason)
    }

    /// Marks the recorder stopped. Nil when it was already stopped; else the
    /// running stream, if any, which the caller stops.
    private func takeStreamForStop() -> SCStream?? {
        state.withLock { current -> SCStream?? in
            guard current.lifecycle.stop() else { return nil }
            let stream = current.stream
            current.stream = nil
            return .some(stream)
        }
    }

    private func finishFiles(reason: String) {
        callReference.end()
        control.sync { stopMicrophone() }
        queue.sync {
            try? mic.finish()
            try? system.finish()
        }
        if index.metadata.endReason == nil { index.end(reason: reason) }
    }

    private func startMicrophone() throws {
        try installMicrophone()
        // A device change mid-call (AirPods connecting, a USB mic unplugged)
        // stops the engine; take the new input and carry on. The track
        // clock pads the gap with silence from the buffer timestamps.
        configurationObserver = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: engine, queue: nil
        ) { [weak self] _ in
            guard let self else { return }
            self.control.async { self.restartMicrophone() }
        }
    }

    private func installMicrophone() throws {
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0,
            let converter = AVAudioConverter(from: format, to: target)
        else { throw MeetingRecorderError.noMicrophone }
        // No voice processing: the input node is the raw device signal. The
        // converter belongs to this tap alone, so a restart never swaps it
        // under a buffer in flight. A device switching formats mid-call makes
        // AVFAudio raise; that becomes a thrown error, not an abort.
        var raised: NSError?
        let installed = PhononCatchException({
            input.installTap(onBus: 0, bufferSize: 4_096, format: format) { [weak self] buffer, when in
                guard let self, let samples = self.convert(buffer, with: converter) else { return }
                let hostNs = HostClock.plausible(
                    when.isHostTimeValid ? HostClock.nanoseconds(machTicks: when.hostTime) : nil,
                    now: HostClock.now())
                self.queue.async { try? self.mic.append(samples, hostNs: hostNs) }
            }
        }, &raised)
        guard installed else { throw raised ?? MeetingRecorderError.noMicrophone }
        engine.prepare()
        try engine.start()
    }

    /// On `control`: re-install the tap on the new input while recording.
    private func restartMicrophone() {
        guard state.withLock({ $0.lifecycle.mayRestartInput }) else { return }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        do {
            try installMicrophone()
        } catch {
            NSLog("phonon meeting: microphone did not come back: \(error.localizedDescription)")
        }
    }

    private func stopMicrophone() {
        if let configurationObserver {
            NotificationCenter.default.removeObserver(configurationObserver)
            self.configurationObserver = nil
        }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
    }

    private func convert(_ buffer: AVAudioPCMBuffer, with converter: AVAudioConverter)
        -> [Float]?
    {
        let ratio = target.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio + 32)
        guard let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else {
            return nil
        }
        var consumed = false
        var error: NSError?
        converter.convert(to: output, error: &error) { _, status in
            if consumed {
                status.pointee = .noDataNow
                return nil
            }
            consumed = true
            status.pointee = .haveData
            return buffer
        }
        guard error == nil, let channel = output.floatChannelData?[0] else { return nil }
        return Array(UnsafeBufferPointer(start: channel, count: Int(output.frameLength)))
    }

    /// The started stream; the caller attaches it or stops it.
    private func startSystemAudio() async throws -> SCStream? {
        let content = try await SCShareableContent.excludingDesktopWindows(
            false, onScreenWindowsOnly: true)
        guard let display = content.displays.first else { return nil }
        let filter = SCContentFilter(display: display, excludingWindows: [])
        let configuration = SCStreamConfiguration()
        configuration.capturesAudio = true
        configuration.excludesCurrentProcessAudio = true
        configuration.sampleRate = MeetingFormat.sampleRate
        configuration.channelCount = 1
        // Audio only: the smallest, slowest video the API accepts.
        configuration.width = 2
        configuration.height = 2
        configuration.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        let stream = SCStream(filter: filter, configuration: configuration, delegate: self)
        try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: queue)
        try await stream.startCapture()
        return stream
    }

    func stream(
        _ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of type: SCStreamOutputType
    ) {
        guard type == .audio, let samples = Self.floatSamples(sampleBuffer) else { return }
        // ScreenCaptureKit stamps buffers on the host time clock.
        let hostNs = HostClock.plausible(
            HostClock.nanoseconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer)),
            now: HostClock.now())
        // Already on `queue`.
        try? system.append(samples, hostNs: hostNs)
        callReference.append(samples, rate: Double(MeetingFormat.sampleRate), hostNs: hostNs)
    }

    /// Mono float samples from a ScreenCaptureKit audio buffer (deinterleaved
    /// float32; the first channel when there are more).
    static func floatSamples(_ sampleBuffer: CMSampleBuffer) -> [Float]? {
        var blockBuffer: CMBlockBuffer?
        var list = AudioBufferList()
        let status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer, bufferListSizeNeededOut: nil, bufferListOut: &list,
            bufferListSize: MemoryLayout<AudioBufferList>.size, blockBufferAllocator: nil,
            blockBufferMemoryAllocator: nil, flags: 0, blockBufferOut: &blockBuffer)
        guard status == noErr, let data = list.mBuffers.mData else { return nil }
        let count = Int(list.mBuffers.mDataByteSize) / MemoryLayout<Float>.size
        return Array(UnsafeBufferPointer(start: data.assumingMemoryBound(to: Float.self), count: count))
    }
}

extension MeetingRecorder: SCStreamDelegate {
    func stream(_ stream: SCStream, didStopWithError error: Error) {
        NSLog("phonon meeting: system audio stopped: \(error.localizedDescription)")
        Task { await self.restartSystemAudio(after: stream) }
    }
}

/// How many times something may be retried.
struct RestartBudget: Equatable {
    let limit: Int
    private(set) var used = 0

    init(limit: Int) { self.limit = limit }

    /// True, and one fewer left, while any are left.
    mutating func take() -> Bool {
        guard used < limit else { return false }
        used += 1
        return true
    }
}
