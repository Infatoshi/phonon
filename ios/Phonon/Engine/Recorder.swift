import AVFoundation
import Foundation

/// The microphone. While the session is warm the engine keeps running with nothing kept,
/// which is what lets the keyboard start a dictation while Phonon is in the background: iOS
/// lets an app that already holds a running record session keep recording, but not start
/// one from the background. Samples are converted to 16 kHz mono Float32 for Parakeet.
final class Recorder: @unchecked Sendable {
    enum Event {
        case interrupted          // a call or Siri took the microphone; capture stopped
        case resumed              // the interruption ended and the engine is back
        case routeChanged(String) // input moved (AirPods in or out); capture continues
        case failed(String)
    }

    static let sampleRate: Double = 16_000

    var onSamples: (([Float]) -> Void)?
    var onLevel: ((Float) -> Void)?
    var onEvent: ((Event) -> Void)?

    private let engine = AVAudioEngine()
    private let queue = DispatchQueue(label: "phonon.recorder")
    private var converter: AVAudioConverter?
    private let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false)!
    private var capturing = false
    private var captured: [Float] = []
    private(set) var isRunning = false
    private var observers: [NSObjectProtocol] = []

    init() {
        let nc = NotificationCenter.default
        let session = AVAudioSession.sharedInstance()
        observers.append(nc.addObserver(forName: AVAudioSession.interruptionNotification, object: session, queue: .main) { [weak self] note in
            self?.handleInterruption(note)
        })
        observers.append(nc.addObserver(forName: AVAudioSession.routeChangeNotification, object: session, queue: .main) { [weak self] note in
            self?.handleRouteChange(note)
        })
        observers.append(nc.addObserver(forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main) { [weak self] _ in
            self?.restartAfterConfigurationChange()
        })
        observers.append(nc.addObserver(forName: AVAudioSession.mediaServicesWereResetNotification, object: session, queue: .main) { [weak self] _ in
            self?.isRunning = false
            try? self?.start()
        })
    }

    static func configureSession() throws {
        let session = AVAudioSession.sharedInstance()
        // mixWithOthers: dictating does not pause the music or podcast he is listening to.
        try session.setCategory(.playAndRecord, mode: .default,
                                options: [.mixWithOthers, .allowBluetoothHFP, .defaultToSpeaker])
        try session.setPreferredSampleRate(48_000)
        try session.setPreferredIOBufferDuration(0.02)
        try session.setActive(true)
    }

    /// Opens the session and starts the engine without keeping audio.
    func start() throws {
        if isRunning { return }
        try Self.configureSession()
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw NSError(domain: "Phonon", code: 1, userInfo: [NSLocalizedDescriptionKey: "No microphone input is available."])
        }
        converter = AVAudioConverter(from: format, to: target)
        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: 2048, format: format) { [weak self] buffer, _ in
            self?.handle(buffer)
        }
        engine.prepare()
        try engine.start()
        isRunning = true
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        isRunning = false
        queue.sync { capturing = false }
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    func beginCapture() {
        queue.sync {
            captured = []
            captured.reserveCapacity(16_000 * 60)
            capturing = true
        }
    }

    /// Ends the capture and returns everything recorded since `beginCapture`.
    func endCapture() -> [Float] {
        queue.sync {
            capturing = false
            let out = captured
            captured = []
            return out
        }
    }

    var isCapturing: Bool { queue.sync { capturing } }

    private func handle(_ buffer: AVAudioPCMBuffer) {
        guard let converter else { return }
        let ratio = target.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio + 64)
        guard let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { return }
        var fed = false
        var error: NSError?
        converter.convert(to: out, error: &error) { _, status in
            if fed {
                status.pointee = .noDataNow
                return nil
            }
            fed = true
            status.pointee = .haveData
            return buffer
        }
        guard error == nil, let data = out.floatChannelData else { return }
        let samples = Array(UnsafeBufferPointer(start: data[0], count: Int(out.frameLength)))
        queue.async { [weak self] in
            guard let self, self.capturing else { return }
            self.captured.append(contentsOf: samples)
            self.onSamples?(samples)
            var sum: Float = 0
            for s in samples { sum += s * s }
            let rms = samples.isEmpty ? 0 : (sum / Float(samples.count)).squareRoot()
            self.onLevel?(min(1, rms * 12))
        }
    }

    private func handleInterruption(_ note: Notification) {
        guard let raw = note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt,
              let type = AVAudioSession.InterruptionType(rawValue: raw) else { return }
        switch type {
        case .began:
            isRunning = false
            onEvent?(.interrupted)
        case .ended:
            do {
                try start()
                onEvent?(.resumed)
            } catch {
                onEvent?(.failed(error.localizedDescription))
            }
        @unknown default:
            break
        }
    }

    private func handleRouteChange(_ note: Notification) {
        let name = AVAudioSession.sharedInstance().currentRoute.inputs.first?.portName ?? "microphone"
        onEvent?(.routeChanged(name))
    }

    /// The input format changed (headset in or out). Rebuild the tap; a capture in progress
    /// keeps its samples and continues on the new input.
    private func restartAfterConfigurationChange() {
        guard isRunning else { return }
        isRunning = false
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        do {
            try start()
        } catch {
            onEvent?(.failed(error.localizedDescription))
        }
    }
}
