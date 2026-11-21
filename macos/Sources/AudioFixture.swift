import Foundation

/// Headless benchmark harness: with `PHONON_AUDIO_FIXTURE` a dictation records a WAV file
/// instead of the microphone (no device, no permission), and with `PHONON_NO_PASTE` the
/// final text goes to `last_fixture_output.txt` in the data directory instead of the
/// focused app. Pair with `PHONON_DATA_DIR` for an isolated store.
struct AudioFixtureConfig: Equatable {
    static let fixtureKey = "PHONON_AUDIO_FIXTURE"
    static let noPasteKey = "PHONON_NO_PASTE"
    static let outputFileName = "last_fixture_output.txt"

    let fixtureURL: URL?
    let noPaste: Bool

    static func from(
        environment: [String: String], fileManager: FileManager = .default
    ) -> AudioFixtureConfig {
        var url: URL?
        if let raw = environment[fixtureKey]?.trimmingCharacters(in: .whitespacesAndNewlines),
            !raw.isEmpty
        {
            let path = (raw as NSString).expandingTildeInPath
            var isDirectory: ObjCBool = false
            if fileManager.fileExists(atPath: path, isDirectory: &isDirectory),
                !isDirectory.boolValue, fileManager.isReadableFile(atPath: path)
            {
                url = URL(fileURLWithPath: path)
            }
        }
        let flag = environment[noPasteKey]?.trimmingCharacters(in: .whitespaces).lowercased()
        return AudioFixtureConfig(fixtureURL: url, noPaste: flag == "1" || flag == "true")
    }

    static let current = AudioFixtureConfig.from(environment: ProcessInfo.processInfo.environment)

    /// Write the final text where the harness reads it, atomically.
    static func writeOutput(_ text: String, directory: URL) throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent(outputFileName)
        try Data(text.utf8).write(to: url, options: .atomic)
        return url
    }
}

enum AudioFixtureError: Error, Equatable {
    case notRIFFWave
    case missingChunk(String)
    case unsupportedFormat(String)
}

/// Reads a 16-bit PCM WAV (any rate, any channel count) as 16 kHz mono float32.
enum AudioFixtureLoader {
    static let targetRate = 16_000.0

    static func loadMono16kPCM(from url: URL) throws -> [Float] {
        try decodeMono16kPCM(Data(contentsOf: url))
    }

    static func decodeMono16kPCM(_ data: Data) throws -> [Float] {
        let bytes = [UInt8](data)
        func u32(_ at: Int) -> UInt32 {
            UInt32(bytes[at]) | UInt32(bytes[at + 1]) << 8 | UInt32(bytes[at + 2]) << 16
                | UInt32(bytes[at + 3]) << 24
        }
        func u16(_ at: Int) -> UInt16 { UInt16(bytes[at]) | UInt16(bytes[at + 1]) << 8 }
        guard bytes.count >= 12, String(decoding: bytes[0..<4], as: UTF8.self) == "RIFF",
            String(decoding: bytes[8..<12], as: UTF8.self) == "WAVE"
        else { throw AudioFixtureError.notRIFFWave }

        var offset = 12
        var channels = 0
        var sampleRate = 0
        var bitsPerSample = 0
        var format: UInt16 = 0
        var dataRange: Range<Int>?
        while offset + 8 <= bytes.count {
            let id = String(decoding: bytes[offset..<offset + 4], as: UTF8.self)
            let size = Int(u32(offset + 4))
            let body = offset + 8
            let end = min(bytes.count, body + size)
            if id == "fmt " {
                guard size >= 16, body + 16 <= bytes.count else {
                    throw AudioFixtureError.unsupportedFormat("short fmt chunk")
                }
                format = u16(body)
                channels = Int(u16(body + 2))
                sampleRate = Int(u32(body + 4))
                bitsPerSample = Int(u16(body + 14))
                if format == 0xFFFE, size >= 26, body + 26 <= bytes.count {
                    format = u16(body + 24)  // WAVE_FORMAT_EXTENSIBLE: the sub-format GUID
                }
            } else if id == "data" {
                dataRange = body..<end  // a streaming writer may leave the size short or 0
                if size == 0 || size == 0xFFFF_FFFF { dataRange = body..<bytes.count }
            }
            offset = body + size + (size & 1)
        }
        guard channels > 0, sampleRate > 0 else { throw AudioFixtureError.missingChunk("fmt ") }
        guard let dataRange else { throw AudioFixtureError.missingChunk("data") }
        guard format == 1, bitsPerSample == 16 else {
            throw AudioFixtureError.unsupportedFormat(
                "format \(format), \(bitsPerSample)-bit (16-bit PCM only)")
        }
        let frameBytes = 2 * channels
        let frames = dataRange.count / frameBytes
        var mono = [Float](repeating: 0, count: frames)
        for frame in 0..<frames {
            var sum: Float = 0
            let base = dataRange.lowerBound + frame * frameBytes
            for channel in 0..<channels {
                sum += Float(Int16(bitPattern: u16(base + channel * 2))) / 32_768
            }
            mono[frame] = sum / Float(channels)
        }
        return MicRecorder.resample(mono, from: Double(sampleRate), to: targetRate)
    }
}

/// Feeds loaded samples in 10 ms chunks (160 at 16 kHz) at real-time pace from a background
/// timer, like a microphone would, without touching any audio device.
final class AudioFixtureStreamer {
    static let chunkFrames = 160
    static let sampleRate = 16_000.0

    private let samples: [Float]
    private let queue = DispatchQueue(label: "phonon.audio-fixture", qos: .userInitiated)
    private let lock = NSLock()
    private var timer: DispatchSourceTimer?
    private var position = 0
    private var generation = 0
    private let chunkInterval: TimeInterval

    /// `pace`: 1 is real time; tests run faster.
    init(samples: [Float], pace: Double = 1) {
        self.samples = samples
        chunkInterval = Double(Self.chunkFrames) / Self.sampleRate / max(pace, 0.001)
    }

    var isRunning: Bool {
        lock.lock()
        defer { lock.unlock() }
        return timer != nil
    }

    /// `onChunk` gets each chunk on the streamer's queue; `onFinished` once, after the last
    /// chunk, unless stopped first.
    func start(onChunk: @escaping ([Float]) -> Void, onFinished: @escaping () -> Void = {}) {
        lock.lock()
        timer?.cancel()
        generation += 1
        let mine = generation
        position = 0
        let source = DispatchSource.makeTimerSource(queue: queue)
        timer = source
        lock.unlock()
        source.schedule(
            deadline: .now(), repeating: chunkInterval, leeway: .milliseconds(1))
        source.setEventHandler { [weak self] in
            guard let self else { return }
            self.lock.lock()
            guard self.generation == mine, self.timer != nil else {
                self.lock.unlock()
                return
            }
            let start = self.position
            let end = min(self.samples.count, start + Self.chunkFrames)
            self.position = end
            let done = end >= self.samples.count
            if done {
                self.timer?.cancel()
                self.timer = nil
            }
            self.lock.unlock()
            if end > start { onChunk(Array(self.samples[start..<end])) }
            if done { onFinished() }
        }
        source.resume()
    }

    /// Stop at once; no chunk or finish callback follows.
    func stop() {
        lock.lock()
        generation += 1
        timer?.cancel()
        timer = nil
        lock.unlock()
    }

    deinit { timer?.cancel() }
}
