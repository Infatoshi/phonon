import AVFoundation
import FluidAudio
import Foundation
import os
import PhononCore
import UIKit

/// Owns the microphone session, both models and the pipeline, and talks to the keyboard
/// through the App Group (`SharedStore`) and Darwin notifications.
@MainActor
final class DictationEngine: ObservableObject {
    enum ModelPhase: Equatable {
        case missing
        case downloading(Double)
        case loading(String)
        case ready
        case failed(String)
    }

    @Published private(set) var phase: SessionPhase = .off
    @Published private(set) var modelPhase: ModelPhase = .missing
    @Published private(set) var level: Float = 0
    @Published private(set) var recordingStartedAt: Date?
    @Published private(set) var latest: Dictation?
    @Published private(set) var history: [Dictation] = []
    @Published private(set) var notice: String?
    @Published private(set) var downloadProgress: ModelDownloader.Progress?
    @Published private(set) var coldStart: ColdStart?
    /// Why the cleanup model is not running, when it is not.
    @Published private(set) var refinerError: String?
    @Published var settings: SharedSettings {
        didSet { store.writeSettings(settings); scheduleWarmExpiry() }
    }
    /// Set when a keyboard tap opened the app: the screen tells him to go back.
    @Published var openedFromKeyboard = false

    struct ColdStart: Codable, Equatable {
        var asrSeconds: Double
        var refinerSeconds: Double
    }

    let store: SharedStore
    let modelsRoot: URL
    let downloader: ModelDownloader
    private let journal: RecordingJournal
    private let recorder = Recorder()
    private var asr: AsrManager?
    private var refiner: LlamaRefiner?
    private var currentID: String?
    private var currentMode: WritingMode?
    private var warmTimer: Timer?
    private var lastLevelWrite = Date.distantPast
    private var processing: Task<Void, Never>?
    private let log = Logger(subsystem: "com.abhaymettu.phonon", category: "engine")

    init() {
        let fm = FileManager.default
        let support = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        modelsRoot = support.appendingPathComponent("Models", isDirectory: true)
        downloader = ModelDownloader(root: modelsRoot)
        journal = RecordingJournal(directory: support.appendingPathComponent("Recordings", isDirectory: true))
        // Without the App Group the app still works on its own (in-app dictation and copy).
        store = SharedStore.appGroup() ?? SharedStore(root: support.appendingPathComponent("Shared", isDirectory: true))
        settings = store.settings
        history = store.history
        latest = history.first

        ModelCache.resetIfNewInstall()
        DispatchQueue.global(qos: .utility).async { ModelCache.pruneAbandoned() }
        recorder.onSamples = { [journal] samples in journal.append(samples) }
        recorder.onLevel = { [weak self] value in
            Task { @MainActor in self?.updateLevel(value) }
        }
        recorder.onEvent = { [weak self] event in
            Task { @MainActor in self?.handle(event) }
        }
        DarwinNote.observe(PhononIDs.commandNote) { [weak self] in
            Task { @MainActor in self?.handleKeyboardCommand() }
        }
        publish()
    }

    var modelsDownloaded: Bool { downloader.isReady(.asr) && downloader.isReady(.refiner) }
    var asrReady: Bool { asr != nil }
    var refinerReady: Bool { refiner != nil }
    var refinerThreads: Int32 { refiner?.threads ?? 0 }

    // MARK: Models

    func downloadModels() {
        switch modelPhase {
        case .missing, .failed: break
        default: return
        }
        modelPhase = .downloading(downloadProgress?.fraction ?? 0)
        UIApplication.shared.isIdleTimerDisabled = true
        Task {
            do {
                try await downloader.download([.asr, .refiner]) { progress in
                    Task { @MainActor in
                        self.downloadProgress = progress
                        self.modelPhase = .downloading(progress.fraction)
                    }
                }
                UIApplication.shared.isIdleTimerDisabled = false
                await loadModels()
            } catch {
                UIApplication.shared.isIdleTimerDisabled = false
                if (error as? ModelDownloader.Failure) == .cancelled {
                    modelPhase = .missing
                } else {
                    modelPhase = .failed(error.localizedDescription)
                }
            }
        }
    }

    func pauseDownload() { downloader.cancel() }

    func deleteModels() {
        asr = nil
        refiner = nil
        downloader.delete(.asr)
        downloader.delete(.refiner)
        modelPhase = .missing
        downloadProgress = nil
        publish()
    }

    /// Loads both models. The first load of the Parakeet encoder compiles it for the Neural
    /// Engine and takes longer; later launches reuse the system's compiled copy.
    private var loadingModels = false

    func loadModels() async {
        guard !loadingModels else { return }
        loadingModels = true
        defer { loadingModels = false }
        guard modelsDownloaded else {
            modelPhase = .missing
            return
        }
        if asr != nil && refiner != nil {
            modelPhase = .ready
            return
        }
        var cold = ColdStart(asrSeconds: 0, refinerSeconds: 0)
        do {
            if asr == nil {
                modelPhase = .loading("Loading speech recognition")
                let started = Date()
                let dir = ModelManifest.asrDirectory(root: modelsRoot)
                let models = try await Task.detached(priority: .userInitiated) { try LocalAsrModels.load(from: dir) }.value
                let manager = AsrManager(config: .default)
                try await manager.initialize(models: models)
                asr = manager
                cold.asrSeconds = Date().timeIntervalSince(started)
            }
            if refiner == nil {
                modelPhase = .loading("Loading cleanup model")
                let url = ModelManifest.refinerURL(root: modelsRoot)
                do {
                    refiner = try await Task.detached(priority: .userInitiated) { try LlamaRefiner(modelURL: url) }.value
                    cold.refinerSeconds = refiner?.loadSeconds ?? 0
                    refinerError = nil
                } catch {
                    // Speech recognition still works; say plainly that cleanup does not.
                    refinerError = error.localizedDescription
                }
            }
            coldStart = cold
            modelPhase = .ready
            publish()
            await recoverUnfinished()
        } catch {
            log.error("model load failed: \(error.localizedDescription, privacy: .public)")
            modelPhase = .failed(error.localizedDescription)
            publish()
        }
    }

    func bootstrap() {
        if modelsDownloaded {
            Task { await loadModels() }
        } else if downloader.bytesOnDisk(.asr) + downloader.bytesOnDisk(.refiner) > 0 {
            modelPhase = .failed("Download paused. Tap Resume to continue where it stopped.")
        }
        // A keyboard tap that launched the app left its command unacknowledged.
        handleKeyboardCommand()
    }

    // MARK: Session

    var microphoneAllowed: Bool { AVAudioApplication.shared.recordPermission == .granted }
    var microphoneDenied: Bool { AVAudioApplication.shared.recordPermission == .denied }

    func requestMicrophone() async -> Bool {
        await AVAudioApplication.requestRecordPermission()
    }

    func startSession() async -> Bool {
        if recorder.isRunning { return true }
#if DEBUG
        let allowed = Recorder.fixtureSamples != nil ? true : await requestMicrophone()
#else
        let allowed = await requestMicrophone()
#endif
        guard allowed else {
            notice = "Microphone access is off. Turn it on in Settings > Phonon."
            phase = .failed
            publish()
            return false
        }
        do {
            try recorder.start()
            phase = .ready
            scheduleWarmExpiry()
            publish()
            return true
        } catch {
            notice = error.localizedDescription
            phase = .failed
            publish()
            return false
        }
    }

    func endSession() {
        if phase == .recording { cancelRecording() }
        recorder.stop()
        warmTimer?.invalidate()
        phase = .off
        publish()
    }

    private func scheduleWarmExpiry() {
        warmTimer?.invalidate()
        guard recorder.isRunning else { return }
        let seconds = TimeInterval(max(1, settings.warmMinutes) * 60)
        warmTimer = Timer.scheduledTimer(withTimeInterval: seconds, repeats: false) { [weak self] _ in
            Task { @MainActor in
                guard let self, self.phase == .ready else { return }
                self.endSession()
            }
        }
        publish()
    }

    // MARK: Dictation

    private var starting = false

    func startRecording(mode: WritingMode? = nil) async {
        if phase == .transcribing || phase == .refining {
            // Still processing (this dictation or one a kill left behind): start right after.
            log.info("start queued behind \(String(describing: self.phase), privacy: .public)")
            pendingStart = mode ?? settings.writingMode ?? .casual
            publish()
            return
        }
        guard !starting, phase != .recording, phase != .transcribing, phase != .refining else {
            log.info("start ignored in phase \(String(describing: self.phase), privacy: .public)")
            return
        }
        starting = true
        defer { starting = false }
        guard await startSession() else {
            log.error("start failed: \(self.notice ?? "no session", privacy: .public)")
            return
        }
        log.info("recording")
        let id = UUID().uuidString
        currentID = id
        currentMode = mode ?? settings.writingMode ?? .casual
        journal.begin(id: id, mode: currentMode)
        recorder.beginCapture()
        recordingStartedAt = Date()
        phase = .recording
        notice = nil
        warmTimer?.invalidate()
        publish()
    }

    func stopRecording() {
        guard phase == .recording, let id = currentID else { return }
        let samples = recorder.endCapture()
        journal.endRecording()
        recordingStartedAt = nil
        level = 0
        process(id: id, samples: samples, mode: currentMode, recovered: false)
    }

    func cancelRecording() {
        pendingStart = nil
        guard let id = currentID else { return }
        if phase == .recording {
            _ = recorder.endCapture()
            journal.endRecording()
        }
        cancelFlagStorage = true
        journal.finish(id, stage: .discarded)
        currentID = nil
        recordingStartedAt = nil
        level = 0
        phase = recorder.isRunning ? .ready : .off
        scheduleWarmExpiry()
        publish()
    }

    private func process(id: String, samples: [Float], mode: WritingMode?, recovered: Bool, raw knownRaw: String? = nil) {
        cancelFlagStorage = false
        phase = knownRaw == nil ? .transcribing : .refining
        publish()
        let background = UIApplication.shared.beginBackgroundTask(withName: "phonon.dictation")
        processing = Task {
            defer { UIApplication.shared.endBackgroundTask(background) }
            let audioSeconds = Double(samples.count) / Recorder.sampleRate
            var asrSeconds = 0.0
            var raw = knownRaw ?? ""
            do {
                if knownRaw == nil {
                    guard let asr else { throw EngineError.modelsNotReady }
                    // Parakeet wants at least a second; pad short taps with silence.
                    let padded = samples.count < 16_000 ? samples + [Float](repeating: 0, count: 16_000 - samples.count) : samples
                    let started = Date()
                    let result = try await asr.transcribe(padded, source: .microphone)
                    try? await asr.resetDecoderState(for: .microphone)
                    asrSeconds = Date().timeIntervalSince(started)
                    raw = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
                    journal.transcribed(id, raw: raw)
                }
                if cancelFlagStorage { return }
                if raw.isEmpty {
                    journal.finish(id, stage: .discarded)
                    notice = "No speech heard."
                    finishProcessing()
                    return
                }
                var dictation = Dictation(id: id, raw: raw, text: raw, refined: false, recovered: recovered,
                                          audioSeconds: audioSeconds, asrSeconds: asrSeconds, mode: mode)
                if settings.refinerEnabled, let refiner {
                    phase = .refining
                    publish()
                    let vocabulary = settings.vocabulary
                    let started = Date()
                    let output = try await refiner.refineAsync(transcript: raw, mode: mode, vocabulary: vocabulary) { [weak self] in
                        self?.cancelFlag ?? false
                    }
                    dictation.refineSeconds = Date().timeIntervalSince(started)
                    dictation.text = output.text
                    dictation.refined = true
                    dictation.guardFired = output.guardFired
                    log.info("refined: prompt \(output.stats.promptTokens) reused \(output.stats.reusedTokens) out \(output.stats.outputTokens) prefill \(output.stats.prefillTokPerSec, format: .fixed(precision: 1)) tok/s decode \(output.stats.decodeTokPerSec, format: .fixed(precision: 1)) tok/s")
                } else if settings.refinerEnabled {
                    notice = "Cleanup model not loaded; inserted the raw transcript."
                }
                if cancelFlagStorage { return }
                deliver(dictation)
                journal.finish(id, stage: .delivered)
            } catch {
                notice = error.localizedDescription
                if !raw.isEmpty {
                    // ASR worked: deliver the raw transcript rather than lose the dictation.
                    deliver(Dictation(id: id, raw: raw, text: raw, refined: false, recovered: recovered,
                                      audioSeconds: audioSeconds, asrSeconds: asrSeconds, mode: mode))
                    journal.finish(id, stage: .delivered)
                }
            }
            finishProcessing()
        }
    }

    /// Read from the refiner's thread between tokens.
    nonisolated(unsafe) private var cancelFlagStorage = false
    nonisolated var cancelFlag: Bool { cancelFlagStorage }

    private func finishProcessing() {
        currentID = nil
        phase = recorder.isRunning ? .ready : .off
        scheduleWarmExpiry()
        publish()
        if !recovering, let mode = pendingStart {
            pendingStart = nil
            Task { await startRecording(mode: mode) }
        }
        if recoveryWaiting { Task { await recoverUnfinished() } }
    }

    private func deliver(_ d: Dictation) {
        latest = d
        store.appendHistory(d)
        history = store.history
        publish()
    }

    /// Finishes dictations a crash or a kill interrupted, from the raw transcript or the audio.
    /// Set when recovery had to wait for a dictation in progress; run when that one is done.
    private var recoveryWaiting = false
    private var recovering = false
    /// A keyboard start that arrived while a dictation was processing; it runs right after.
    private var pendingStart: WritingMode?

    /// Finishes dictations a kill or crash left behind. Never while a dictation is recording or
    /// being processed: a cold launch from the keyboard records first, recovers after.
    private func recoverUnfinished() async {
        guard phase != .recording, phase != .transcribing, phase != .refining, !starting else {
            recoveryWaiting = true
            return
        }
        recoveryWaiting = false
        recovering = true
        defer { recovering = false }
        for entry in journal.unfinished() where entry.id != currentID {
            if phase == .recording || starting { recoveryWaiting = true; return }
            if let mode = pendingStart {
                pendingStart = nil
                recoveryWaiting = true
                recovering = false
                await startRecording(mode: mode)
                return
            }
            log.info("recovering \(entry.id, privacy: .public)")
            let samples = RecordingJournal.readSamples(journal.audioURL(entry.id))
            if entry.raw == nil && samples.count < 8_000 {
                journal.finish(entry.id, stage: .discarded)
                continue
            }
            process(id: entry.id, samples: samples, mode: entry.mode, recovered: true, raw: entry.raw)
            await processing?.value
        }
        recovering = false
        if let mode = pendingStart {
            pendingStart = nil
            await startRecording(mode: mode)
        }
    }

    func clearHistory() {
        store.clearHistory()
        history = []
    }

    // MARK: Keyboard bridge

    func handleKeyboardCommand() {
        guard let command = store.command else { return }
        guard command.seq > max(ackSeq, store.appState.ackSeq) else { return }
        log.info("keyboard command \(command.seq) \(String(describing: command.kind), privacy: .public)")
        // A command older than a minute is a stale tap from a previous run; acknowledge only.
        let fresh = Date().timeIntervalSince(command.sentAt) < 60
        ackSeq = command.seq
        guard fresh else { publish(); return }
        switch command.kind {
        case .start:
            let hinted = command.hostHint.flatMap(WritingMode.init(rawValue:))
            Task { await startRecording(mode: settings.writingMode ?? hinted) }
        case .stop:
            stopRecording()
        case .cancel:
            cancelRecording()
        case .ping:
            break
        }
        publish()
    }

    private var ackSeq: Int = 0

    private func updateLevel(_ value: Float) {
        level = value
        let now = Date()
        if now.timeIntervalSince(lastLevelWrite) > 0.12 {
            lastLevelWrite = now
            publish()
        }
    }

    private func handle(_ event: Recorder.Event) {
        switch event {
        case .interrupted:
            // A call took the microphone: keep what was said so far.
            if phase == .recording {
                notice = "Recording stopped by a call. Kept what you said."
                stopRecording()
            } else {
                phase = .off
                publish()
            }
        case .resumed:
            if phase == .off { phase = .ready; scheduleWarmExpiry() }
            publish()
        case .routeChanged(let name):
            if phase == .recording { notice = "Now using \(name)." }
            publish()
        case .failed(let message):
            notice = message
            if phase == .recording { stopRecording() } else { phase = .off }
            publish()
        }
    }

    /// Writes the state file the keyboard reads and pings it.
    private func publish() {
        var s = AppState()
        s.phase = phase
        s.ackSeq = max(ackSeq, store.appState.ackSeq)
        s.updatedAt = Date()
        s.recordingStartedAt = recordingStartedAt
        s.level = level
        s.message = notice
        s.latest = latest
        s.warmUntil = (phase == .ready && recorder.isRunning) ? Date().addingTimeInterval(TimeInterval(settings.warmMinutes * 60)) : nil
        s.modelsReady = asr != nil
        store.writeAppState(s)
        DarwinNote.post(PhononIDs.stateNote)
    }

    enum EngineError: LocalizedError {
        case modelsNotReady
        var errorDescription: String? { "Models are not loaded yet. Open Phonon to finish setup." }
    }

    // MARK: Self-test

    struct SelfTestRow: Codable, Identifiable {
        var id: String { name }
        var name: String
        var expected: String
        var raw: String
        var refined: String
        var audioSeconds: Double
        var asrSeconds: Double
        var refineSeconds: Double
        var promptTokens: Int
        var outputTokens: Int
        var prefillTokPerSec: Double
        var decodeTokPerSec: Double
    }

    /// Runs the bundled speech fixtures through both models, in the foreground.
    func runSelfTest() async -> [SelfTestRow] {
        guard let asr, let refiner else { return [] }
        var rows: [SelfTestRow] = []
        let urls = (Bundle.main.urls(forResourcesWithExtension: "wav", subdirectory: "Fixtures") ?? []).sorted { $0.path < $1.path }
        let expected = (try? JSONDecoder().decode([String: String].self, from: Data(contentsOf: Bundle.main.url(forResource: "expected", withExtension: "json", subdirectory: "Fixtures")!))) ?? [:]
        for url in urls {
            let name = url.deletingPathExtension().lastPathComponent
            guard let samples = try? AudioFile16k.read(url) else { continue }
            let started = Date()
            let raw = (try? await asr.transcribe(samples, source: .system).text) ?? ""
            try? await asr.resetDecoderState(for: .system)
            let asrSeconds = Date().timeIntervalSince(started)
            let refStart = Date()
            let out = try? await refiner.refineAsync(transcript: raw, mode: .casual, vocabulary: [])
            rows.append(SelfTestRow(name: name, expected: expected[name] ?? "", raw: raw, refined: out?.text ?? "",
                                    audioSeconds: Double(samples.count) / 16_000, asrSeconds: asrSeconds,
                                    refineSeconds: Date().timeIntervalSince(refStart),
                                    promptTokens: out?.stats.promptTokens ?? 0, outputTokens: out?.stats.outputTokens ?? 0,
                                    prefillTokPerSec: out?.stats.prefillTokPerSec ?? 0, decodeTokPerSec: out?.stats.decodeTokPerSec ?? 0))
        }
        return rows
    }
}

/// Reads any audio file into 16 kHz mono Float32.
enum AudioFile16k {
    static func read(_ url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false)!
        guard let input = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)) else { return [] }
        try file.read(into: input)
        guard let converter = AVAudioConverter(from: file.processingFormat, to: target) else { return [] }
        let capacity = AVAudioFrameCount(Double(input.frameLength) * 16_000 / file.processingFormat.sampleRate + 1024)
        guard let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { return [] }
        var fed = false
        var error: NSError?
        converter.convert(to: out, error: &error) { _, status in
            if fed { status.pointee = .endOfStream; return nil }
            fed = true
            status.pointee = .haveData
            return input
        }
        if let error { throw error }
        return Array(UnsafeBufferPointer(start: out.floatChannelData![0], count: Int(out.frameLength)))
    }
}
