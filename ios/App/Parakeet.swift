import AVFoundation
import FluidAudio
import Foundation

/// Parakeet TDT 0.6B v2 on this iPhone (CoreML, Neural Engine), the same recognizer as the Mac app. Downloaded once
/// from Hugging Face on first start at a pinned commit with every file's sha256 checked (ParakeetModel), then
/// loaded from that folder with FluidAudio's no-download loader.
actor Parakeet {
    private var manager: AsrManager?

    var ready: Bool { manager != nil }

    func load(progress: @escaping @Sendable (Int64, Int64) -> Void = { _, _ in }) async throws -> Int {
        if manager != nil { return 0 }
        let t0 = Date()
        let store = ModelStore(root: ParakeetModel.directory, log: { print("Parakeet", $0) })
        let dir = try await ParakeetModel.ensure(store: store, progress: progress)
        let models = try AsrModels.loadLocal(from: dir, version: .v2)
        let m = AsrManager(config: .default)
        try await m.loadModels(models)
        manager = m
        return Int(Date().timeIntervalSince(t0) * 1000)
    }

    func transcribe(_ samples: [Float]) async throws -> String {
        guard let manager else { throw NSError(domain: "Parakeet", code: 1) }
        // under one second of audio is padded by the model; under 0.15 s there is nothing to hear
        guard samples.count >= 2400 else { return "" }
        // each dictation is independent: a fresh decoder state, no context carried from the last one
        var state = TdtDecoderState.make()
        return try await manager.transcribe(samples, decoderState: &state).text
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// Collects the microphone tap as 16 kHz mono Float32 for the current utterance. Touched from the audio thread.
final class SampleSink: @unchecked Sendable {
    private let lock = NSLock()
    private var converter: AVAudioConverter?
    private var inFormat: AVAudioFormat?
    private var samples: [Float] = []
    private var active = false
    private var lastLevel = 0.0
    /// Called (on the audio thread) about 15 times a second while an utterance is open, with a 0...1 level.
    var onLevel: (@Sendable (Double) -> Void)?
    private let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1,
                                          interleaved: false)!

    func begin() {
        lock.lock()
        samples.removeAll(keepingCapacity: true)
        active = true
        lock.unlock()
    }

    func end() -> [Float] {
        lock.lock()
        active = false
        let s = samples
        samples.removeAll()
        lock.unlock()
        return s
    }

    func push(_ buffer: AVAudioPCMBuffer) {
        lock.lock()
        defer { lock.unlock() }
        guard active else { return }
        if inFormat != buffer.format {
            inFormat = buffer.format
            converter = AVAudioConverter(from: buffer.format, to: outFormat)
        }
        guard let converter else { return }
        let ratio = outFormat.sampleRate / buffer.format.sampleRate
        let cap = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 32
        guard let out = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: cap) else { return }
        var fed = false
        var err: NSError?
        converter.convert(to: out, error: &err) { _, status in
            if fed {
                status.pointee = .noDataNow
                return nil
            }
            fed = true
            status.pointee = .haveData
            return buffer
        }
        if err == nil, let p = out.floatChannelData?[0] {
            let chunk = UnsafeBufferPointer(start: p, count: Int(out.frameLength))
            samples.append(contentsOf: chunk)
            let now = CACurrentMediaTime()
            if now - lastLevel > 0.066, !chunk.isEmpty, let onLevel {
                lastLevel = now
                let rms = (chunk.reduce(0) { $0 + Double($1 * $1) } / Double(chunk.count)).squareRoot()
                // -55 dBFS (silence) .. -15 dBFS (close speech) mapped to 0...1
                let db = 20 * log10(max(rms, 1e-6))
                onLevel(max(0, min(1, (db + 55) / 40)))
            }
        }
    }
}
