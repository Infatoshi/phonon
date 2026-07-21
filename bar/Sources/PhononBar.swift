// Phonon — native macOS app plus a bottom-anchored dictation capsule.
//
// Idle: tiny warm bar. Hold Option → the same bar expands. Key-up → it breathes.
// Done → type text and return to idle.
//
// First-pass fix: wait for warm engine, pre-warm mic, queue work until ready.

import AVFoundation
import AppKit
import ApplicationServices
import AudioToolbox
import Carbon.HIToolbox
import CoreAudio
import Darwin
import Foundation
import QuartzCore
import ScreenCaptureKit
import SwiftUI
import Vision

final class SingleInstanceLock {
    private let descriptor: Int32

    private init(descriptor: Int32) {
        self.descriptor = descriptor
    }

    static func acquire() -> SingleInstanceLock? {
        guard
            let base = FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask
            ).first
        else { return nil }
        let directory = base.appendingPathComponent("Phonon", isDirectory: true)
        try? FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        return acquire(at: directory.appendingPathComponent("bar.lock"))
    }

    static func acquire(at url: URL) -> SingleInstanceLock? {
        let descriptor = Darwin.open(url.path, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR)
        guard descriptor >= 0 else { return nil }
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            Darwin.close(descriptor)
            return nil
        }
        return SingleInstanceLock(descriptor: descriptor)
    }

    deinit {
        flock(descriptor, LOCK_UN)
        Darwin.close(descriptor)
    }
}

// MARK: - Theme (black shell, white ink)

enum Theme {
    static let panel = NSColor.black
    static let ink = NSColor.white
    static let inkDim = NSColor(white: 1.0, alpha: 0.22)
}

// MARK: - State

@MainActor
final class PillState: ObservableObject {
    enum Mode {
        case hidden
        case idle
        case listening
        case working
    }

    @Published var mode: Mode = .hidden
    @Published var level = 0.0
    @Published var streamingPreviewEnabled = true
    @Published var previewText = ""
    @Published var previewStage = "Listening"
    @Published var panelHeight: CGFloat = 116
}

struct ModelLoadStream: Identifiable {
    let id: String
    let title: String
    var state = "Waiting"
    var progress = 0.0
    var detail = "Waiting for engine"
    var loadMilliseconds: Double?
}

@MainActor
final class ModelStartupState: ObservableObject {
    @Published private(set) var streams = [
        ModelLoadStream(id: "asr", title: "Speech"),
        ModelLoadStream(id: "fluid-1", title: "Language model"),
        ModelLoadStream(id: "mtp", title: "Speculative decoder"),
    ]
    @Published private(set) var ready = false

    var progress: Double {
        streams.map { stream in
            stream.state == "Ready" ? 1 : stream.progress
        }.reduce(0, +) / Double(streams.count)
    }

    func apply(name: String, state: String, progress: Double, detail: String, loadMs: Double?) {
        guard let index = streams.firstIndex(where: { $0.id == name }) else { return }
        var updated = streams
        updated[index].state = state.prefix(1).uppercased() + state.dropFirst()
        updated[index].progress = state == "loading"
            ? max(updated[index].progress, progress)
            : progress
        if !detail.isEmpty { updated[index].detail = detail }
        if let loadMs { updated[index].loadMilliseconds = loadMs }
        streams = updated
    }

    func markReady() {
        ready = true
    }

    func markRestarting() {
        ready = false
        streams = streams.map { stream in
            var stream = stream
            stream.state = "Waiting"
            stream.progress = 0
            stream.detail = "Engine restarting"
            stream.loadMilliseconds = nil
            return stream
        }
    }
}

struct ModelStartupView: View {
    @ObservedObject var state: ModelStartupState

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(state.ready ? "Phonon is ready" : "Preparing Phonon")
                        .font(.title2.weight(.semibold))
                    Text("Models are warmed with real demo requests before dictation.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Text("\(Int((state.progress * 100).rounded()))%")
                    .font(.system(.title3, design: .monospaced).weight(.medium))
            }

            ProgressView(value: state.progress)
                .progressViewStyle(.linear)
                .tint(EmberTheme.accent)

            VStack(spacing: 0) {
                ForEach(state.streams) { stream in
                    HStack(alignment: .top, spacing: 12) {
                        Circle()
                            .fill(color(for: stream.state))
                            .frame(width: 7, height: 7)
                            .padding(.top, 6)
                        VStack(alignment: .leading, spacing: 3) {
                            HStack {
                                Text(stream.title).fontWeight(.medium)
                                Text(stream.state)
                                    .font(.caption.weight(.medium))
                                    .foregroundStyle(color(for: stream.state))
                                Spacer()
                                if let milliseconds = stream.loadMilliseconds {
                                    Text("\(milliseconds, specifier: "%.0f") ms")
                                        .font(.caption.monospacedDigit())
                                        .foregroundStyle(.secondary)
                                }
                            }
                            Text(stream.detail)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(2)
                        }
                    }
                    .padding(.vertical, 9)
                    if stream.id != state.streams.last?.id { Divider() }
                }
            }
        }
        .padding(24)
        .frame(width: 520)
        .foregroundStyle(EmberTheme.text)
        .background(
            EmberTheme.surface,
            in: RoundedRectangle(cornerRadius: 18, style: .continuous)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .stroke(EmberTheme.border.opacity(0.8), lineWidth: 1)
        )
        .preferredColorScheme(.dark)
    }

    private func color(for state: String) -> Color {
        switch state.lowercased() {
        case "ready": return EmberTheme.healthy
        case "loading": return EmberTheme.accent
        case "error": return .red
        case "missing": return .secondary
        default: return .secondary
        }
    }
}

enum PreviewRevisionPolicy {
    static func stableAcoustic(previous: String, candidate: String) -> String {
        wordCount(candidate) < wordCount(previous) ? previous : candidate
    }

    static func canShowPolish(
        source: String, latestAcoustic: String, polished: String, newerCandidateWaiting: Bool
    ) -> Bool {
        !newerCandidateWaiting
            && source == latestAcoustic
            && wordCount(polished) >= wordCount(source)
    }

    private static func wordCount(_ text: String) -> Int {
        text.split(whereSeparator: { $0.isWhitespace }).count
    }
}

enum ShortcutPolicy {
    static func allows(mode: String, source: String) -> Bool {
        switch mode {
        case "right_option": return source != "control-space"
        case "control_space": return source != "right-option"
        default: return true
        }
    }
}

enum LiveTranscriptFormatter {
    static func format(_ text: String) -> String {
        var result = text
        let replacements = [
            (#"(?i)\b(?:full stop|period)\b"#, "."),
            (#"(?i)\bquestion mark\b"#, "?"),
            (#"(?i)\b(?:exclamation mark|exclamation point)\b"#, "!"),
            (#"(?i)\bcomma\b"#, ","),
            (#"(?i)\bsemicolon\b"#, ";"),
            (#"(?i)\bcolon\b"#, ":"),
            (#"(?i)\bnew line\b"#, "\n"),
            (#"(?i)\b(?:uh+|um+|ah+)\b[\s,]*"#, ""),
            (#"(?i)\b([[:alnum:]']+)(?:\s+\1\b)+"#, "$1"),
            (#"\s+([,.!?;:])"#, "$1"),
            (#"([,.!?;:])(?=\S)"#, "$1 "),
            (#"[ \t]{2,}"#, " "),
        ]
        for (pattern, replacement) in replacements {
            result = replacing(pattern: pattern, in: result, with: replacement)
        }
        return capitalizeSentences(
            result.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    private static func replacing(pattern: String, in text: String, with replacement: String)
        -> String
    {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return text }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return regex.stringByReplacingMatches(
            in: text, range: range, withTemplate: replacement)
    }

    private static func capitalizeSentences(_ text: String) -> String {
        var output = ""
        var shouldCapitalize = true
        for character in text {
            if shouldCapitalize, character.isLetter {
                output.append(contentsOf: character.uppercased())
                shouldCapitalize = false
            } else {
                output.append(character)
                if !character.isWhitespace && !"\"'([{«.».".contains(character) {
                    shouldCapitalize = false
                }
            }
            if ".!?\n".contains(character) {
                shouldCapitalize = true
            }
        }
        return output
    }
}

// MARK: - Transient screen context

struct BarSettings: Decodable {
    let streaming: Bool?
    let screenContext: Bool?
    let microphonePriority: [String]?

    enum CodingKeys: String, CodingKey {
        case streaming
        case screenContext = "screen_context"
        case microphonePriority = "microphone_priority"
    }

    private static func load() -> Self? {
        guard
            let base = FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask
            ).first,
            let data = try? Data(
                contentsOf: base.appendingPathComponent("Phonon/settings.json")),
            let settings = try? JSONDecoder().decode(Self.self, from: data)
        else { return nil }
        return settings
    }

    static func streamingEnabled() -> Bool {
        load()?.streaming ?? true
    }

    static func screenContextEnabled() -> Bool {
        load()?.screenContext ?? true
    }

    static func microphonePriorities() -> [String] {
        load()?.microphonePriority ?? ["Yeti", "MacBook Pro Microphone"]
    }

    static func setStreamingEnabled(_ enabled: Bool) {
        guard
            let base = FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask
            ).first
        else { return }
        let url = base.appendingPathComponent("Phonon/settings.json")
        guard
            let data = try? Data(contentsOf: url),
            var object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return }
        object["streaming"] = enabled
        guard let updated = try? JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted])
        else { return }
        try? updated.write(to: url, options: .atomic)
    }
}

struct AudioInputDevice: Equatable {
    let id: AudioDeviceID
    let name: String
    let isDefault: Bool
}

enum MicrophonePriorityResolver {
    static func resolve(devices: [AudioInputDevice], priorities: [String]) -> AudioInputDevice? {
        for priority in priorities {
            if let match = devices.first(where: {
                $0.name.localizedCaseInsensitiveContains(priority)
            }) {
                return match
            }
        }
        return devices.first(where: \.isDefault) ?? devices.first
    }
}

enum CoreAudioInputDevices {
    static func resolve(priorities: [String]) -> AudioInputDevice? {
        MicrophonePriorityResolver.resolve(devices: all(), priorities: priorities)
    }

    static func inputNames() -> [String] {
        all().map(\.name).sorted { $0.localizedCaseInsensitiveCompare($1) == .orderedAscending }
    }

    private static func all() -> [AudioInputDevice] {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size) == noErr
        else { return [] }

        let count = Int(size) / MemoryLayout<AudioDeviceID>.size
        var ids = [AudioDeviceID](repeating: 0, count: count)
        guard AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &ids) == noErr
        else { return [] }

        let defaultID = defaultInputID()
        return ids.compactMap { id in
            guard hasInputStreams(id), let name = deviceName(id) else { return nil }
            return AudioInputDevice(id: id, name: name, isDefault: id == defaultID)
        }
    }

    private static func defaultInputID() -> AudioDeviceID? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var id = AudioDeviceID(kAudioObjectUnknown)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        guard AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &id) == noErr
        else { return nil }
        return id
    }

    private static func hasInputStreams(_ id: AudioDeviceID) -> Bool {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreams,
            mScope: kAudioDevicePropertyScopeInput,
            mElement: kAudioObjectPropertyElementMain
        )
        var size: UInt32 = 0
        return AudioObjectGetPropertyDataSize(id, &address, 0, nil, &size) == noErr && size > 0
    }

    private static func deviceName(_ id: AudioDeviceID) -> String? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioObjectPropertyName,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var name: Unmanaged<CFString>?
        var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        guard AudioObjectGetPropertyData(id, &address, 0, nil, &size, &name) == noErr,
            let name
        else {
            return nil
        }
        return name.takeUnretainedValue() as String
    }
}

enum ScreenContextCapture {
    static func recognizeAllDisplays() async -> String {
        guard CGPreflightScreenCaptureAccess() || CGRequestScreenCaptureAccess() else {
            return ""
        }
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(
                false, onScreenWindowsOnly: true)
            let ownApplication = content.applications.first {
                $0.processID == ProcessInfo.processInfo.processIdentifier
            }
            var recognized = [String]()
            for display in content.displays {
                guard !Task.isCancelled else { return "" }
                let filter = SCContentFilter(
                    display: display,
                    excludingApplications: ownApplication.map { [$0] } ?? [],
                    exceptingWindows: [])
                let configuration = SCStreamConfiguration()
                configuration.width = display.width
                configuration.height = display.height
                configuration.showsCursor = false
                let image = try await SCScreenshotManager.captureImage(
                    contentFilter: filter, configuration: configuration)
                guard !Task.isCancelled else { return "" }
                let text = recognize(image)
                recognized.append(text)
            }
            return recognized.filter { !$0.isEmpty }.joined(separator: "\n")
        } catch {
            NSLog("phonon screen context: \(error.localizedDescription)")
            return ""
        }
    }

    private static func recognize(_ image: CGImage) -> String {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true
        let handler = VNImageRequestHandler(cgImage: image)
        do {
            try handler.perform([request])
            return (request.results ?? [])
                .compactMap { $0.topCandidates(1).first?.string }
                .joined(separator: "\n")
        } catch {
            return ""
        }
    }
}

struct PillView: View {
    @ObservedObject var state: PillState

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30.0)) { context in
            let energy = visualEnergy(at: context.date)
            let size = shellSize(energy: energy)
            Capsule(style: .continuous)
                .fill(
                    LinearGradient(
                        stops: [
                            .init(color: .black, location: 0),
                            .init(
                                color: Color(red: 0.42, green: 0.12, blue: 0.035)
                                    .opacity(warmth(energy)),
                                location: 0.5),
                            .init(color: .black, location: 1),
                        ],
                        startPoint: .leading,
                        endPoint: .trailing
                    )
                )
                .shadow(
                    color: Color(red: 0.55, green: 0.16, blue: 0.04)
                        .opacity(state.mode == .idle ? 0 : 0.22 + energy * 0.12),
                    radius: 4,
                    y: 1
                )
                .frame(width: size.width, height: size.height)
                .frame(width: 80, height: 16, alignment: .bottom)
                .opacity(state.mode == .hidden ? 0 : 1)
                .animation(
                    .spring(response: 0.22, dampingFraction: 0.86),
                    value: size
                )
        }
    }

    private func visualEnergy(at date: Date) -> Double {
        switch state.mode {
        case .hidden, .idle:
            return 0
        case .listening:
            return min(1, max(0.08, state.level))
        case .working:
            let breath = (sin(date.timeIntervalSinceReferenceDate * .pi * 2 / 1.6) + 1) / 2
            return 0.18 + breath * 0.18
        }
    }

    private func warmth(_ energy: Double) -> Double {
        switch state.mode {
        case .hidden, .idle:
            return 0
        case .listening, .working:
            return 0.50 + energy * 0.38
        }
    }

    private func shellSize(energy: Double) -> CGSize {
        switch state.mode {
        case .hidden, .idle:
            return CGSize(width: 40, height: 8)
        case .listening:
            return CGSize(
                width: 66 + 14 * energy,
                height: 13 + 3 * energy
            )
        case .working:
            return CGSize(
                width: 38 + 10 * energy,
                height: 10 + 3 * energy
            )
        }
    }
}

final class PillPanel: NSPanel {
    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }

    init(contentRect: NSRect) {
        super.init(
            contentRect: contentRect,
            styleMask: [.borderless, .nonactivatingPanel, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        isFloatingPanel = true
        level = .floating
        hidesOnDeactivate = false
        collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        isMovableByWindowBackground = false
        backgroundColor = .clear
        isOpaque = false
        hasShadow = false
        ignoresMouseEvents = true
    }

    static func screenFrame() -> NSRect {
        let screen =
            NSScreen.screens.first { NSMouseInRect(NSEvent.mouseLocation, $0.frame, false) }
            ?? NSScreen.main
        return screen?.frame ?? .zero
    }

    static func frame(size: NSSize) -> NSRect {
        PanelGeometry.restingFrame(screen: screenFrame(), size: size)
    }

    static func hiddenFrame(size: NSSize) -> NSRect {
        PanelGeometry.hiddenFrame(screen: screenFrame(), size: size)
    }
}

enum PanelGeometry {
    static let bottomInset: CGFloat = 14

    static func restingFrame(screen: NSRect, size: NSSize) -> NSRect {
        return NSRect(
            x: screen.midX - size.width / 2,
            y: screen.minY + bottomInset,
            width: size.width,
            height: size.height
        )
    }

    static func hiddenFrame(screen: NSRect, size: NSSize) -> NSRect {
        NSRect(
            x: screen.midX - size.width / 2,
            y: screen.minY - size.height,
            width: size.width,
            height: size.height
        )
    }

}

// MARK: - Engine client

final class EngineClient {
    private var process: Process?
    private var stdin: FileHandle?
    private var readSource: DispatchSourceRead?
    private var buffer = Data()
    private var shuttingDown = false
    var onEvent: (([String: Any]) -> Void)?
    var onExit: (() -> Void)?

    func start() throws {
        readSource?.cancel()
        readSource = nil
        try? stdin?.close()
        stdin = nil
        process = nil
        buffer.removeAll(keepingCapacity: true)
        shuttingDown = false
        let phonon = resolvePhonon()
        let p = Process()
        p.executableURL = URL(fileURLWithPath: phonon)
        p.arguments = ["engine"]
        let inPipe = Pipe()
        let outPipe = Pipe()
        p.standardInput = inPipe
        p.standardOutput = outPipe
        p.standardError = FileHandle.nullDevice
        try p.run()
        p.terminationHandler = { [weak self] _ in
            DispatchQueue.main.async {
                guard let self, !self.shuttingDown else { return }
                self.onExit?()
            }
        }
        process = p
        stdin = inPipe.fileHandleForWriting

        let fh = outPipe.fileHandleForReading
        let source = DispatchSource.makeReadSource(fileDescriptor: fh.fileDescriptor, queue: .main)
        source.setEventHandler { [weak self] in
            let chunk = fh.availableData
            guard !chunk.isEmpty else { return }
            self?.buffer.append(chunk)
            self?.drainLines()
        }
        source.resume()
        readSource = source
    }

    private func resolvePhonon() -> String {
        if let env = ProcessInfo.processInfo.environment["PHONON_BIN"], !env.isEmpty {
            return env
        }
        for c in [
            NSString(string: "~/.local/bin/phonon").expandingTildeInPath,
            "/Users/infatoshi/.local/bin/phonon",
        ] where FileManager.default.isExecutableFile(atPath: c) {
            return c
        }
        return "phonon"
    }

    private func drainLines() {
        while let range = buffer.range(of: Data([0x0A])) {
            let lineData = buffer.subdata(in: buffer.startIndex..<range.lowerBound)
            buffer.removeSubrange(buffer.startIndex...range.lowerBound)
            if let line = String(data: lineData, encoding: .utf8),
                let data = line.data(using: .utf8),
                let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            {
                onEvent?(obj)
            }
        }
    }

    func send(_ obj: [String: Any]) {
        guard let stdin,
            let data = try? JSONSerialization.data(withJSONObject: obj),
            var line = String(data: data, encoding: .utf8)
        else { return }
        line += "\n"
        if let d = line.data(using: .utf8) {
            try? stdin.write(contentsOf: d)
        }
    }

    func shutdown() {
        shuttingDown = true
        onExit = nil
        send(["cmd": "shutdown"])
        readSource?.cancel()
        process?.terminate()
        process = nil
    }
}

// MARK: - Mic (buffer + offline 16 kHz WAV)

final class MicRecorder {
    private let engine = AVAudioEngine()
    private let lock = NSLock()
    private var samples: [Float] = []
    private var hwSampleRate: Double = 48_000
    private var meter = 0.0
    private var meterFrame = 0
    private var streamSamples: [Float] = []
    private var captureEnabled = false
    private var configuredDeviceID: AudioDeviceID?
    private(set) var wavPath: String?
    private(set) var selectedDeviceName = "Resolving…"
    var onAmplitude: ((Double) -> Void)?
    var onStreamPCM16: ((Data) -> Void)?
    private var primed = false

    /// Keep the CoreAudio graph alive so key-down only gates sample retention.
    func prewarm() {
        guard !primed else { return }
        do {
            try ensureHardwareRunning()
            primed = true
        } catch {
            NSLog("mic prewarm: \(error.localizedDescription)")
        }
    }

    func refreshPreferredDevice() {
        do {
            try ensureHardwareRunning()
            primed = true
        } catch {
            NSLog("mic refresh: \(error.localizedDescription)")
        }
    }

    func start() throws {
        try ensureHardwareRunning()
        lock.lock()
        samples = []
        meter = 0
        meterFrame = 0
        streamSamples.removeAll(keepingCapacity: true)
        captureEnabled = true
        lock.unlock()
    }

    private func ensureHardwareRunning() throws {
        let status = AVCaptureDevice.authorizationStatus(for: .audio)
        if status == .denied || status == .restricted {
            throw NSError(
                domain: "PhononBar", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Microphone permission denied"])
        }

        let device = CoreAudioInputDevices.resolve(
            priorities: BarSettings.microphonePriorities())
        if let device, configuredDeviceID != device.id {
            if engine.isRunning {
                engine.stop()
            }
            let input = engine.inputNode
            if configuredDeviceID != nil {
                input.removeTap(onBus: 0)
            }
            guard let audioUnit = input.audioUnit else {
                throw NSError(
                    domain: "PhononBar", code: 4,
                    userInfo: [NSLocalizedDescriptionKey: "Microphone audio unit unavailable"])
            }
            var deviceID = device.id
            let result = AudioUnitSetProperty(
                audioUnit,
                kAudioOutputUnitProperty_CurrentDevice,
                kAudioUnitScope_Global,
                0,
                &deviceID,
                UInt32(MemoryLayout<AudioDeviceID>.size)
            )
            guard result == noErr else {
                throw NSError(
                    domain: "PhononBar", code: Int(result),
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "Could not select microphone \(device.name) (CoreAudio \(result))"
                    ])
            }
            selectedDeviceName = device.name
            configuredDeviceID = device.id
            NSLog("phonon mic: Auto → \(device.name)")

            let inFmt = input.inputFormat(forBus: 0)
            guard inFmt.sampleRate > 0, inFmt.channelCount > 0 else {
                throw NSError(
                    domain: "PhononBar", code: 3,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "No mic format (sr=\(inFmt.sampleRate) ch=\(inFmt.channelCount))"
                    ])
            }
            hwSampleRate = inFmt.sampleRate
            input.installTap(onBus: 0, bufferSize: 1024, format: inFmt) { [weak self] buffer, _ in
                self?.ingest(buffer)
            }
        }
        if !engine.isRunning {
            engine.prepare()
            try engine.start()
        }
    }

    private func ingest(_ buffer: AVAudioPCMBuffer) {
        lock.lock()
        let shouldCapture = captureEnabled
        lock.unlock()
        guard shouldCapture else { return }
        guard let ch = buffer.floatChannelData, buffer.frameLength > 0 else { return }
        let frames = Int(buffer.frameLength)
        let nCh = Int(buffer.format.channelCount)

        var chunk = [Float](repeating: 0, count: frames)
        var sumSq = 0.0
        for i in 0..<frames {
            var s: Float = 0
            for c in 0..<nCh { s += ch[c][i] }
            s /= Float(max(1, nCh))
            chunk[i] = s
            let d = Double(s)
            sumSq += d * d
        }
        let rms = sqrt(sumSq / Double(frames))
        let db = rms > 1e-9 ? 20.0 * log10(rms) : -60.0
        let norm = min(1, max(0, (db - (-55.0)) / 55.0))
        lock.lock()
        guard captureEnabled else {
            lock.unlock()
            return
        }
        meter = 0.18 * meter + 0.82 * norm
        meterFrame += 1
        let amp = meter
        let shouldPublishAmplitude = meterFrame.isMultiple(of: 2)
        streamSamples.append(contentsOf: chunk)
        var streamPCM: Data?
        if streamSamples.count >= Int(hwSampleRate * 0.4) {
            let mono16k = Self.resample(streamSamples, from: hwSampleRate, to: 16_000)
            streamPCM = Self.pcm16Data(mono16k)
            streamSamples.removeAll(keepingCapacity: true)
        }

        samples.append(contentsOf: chunk)
        let maxSamples = Int(hwSampleRate * 120)
        if samples.count > maxSamples {
            samples.removeFirst(samples.count - maxSamples)
        }
        lock.unlock()

        if shouldPublishAmplitude {
            DispatchQueue.main.async { [weak self] in
                self?.onAmplitude?(amp)
            }
        }
        if let streamPCM {
            DispatchQueue.main.async { [weak self] in
                self?.onStreamPCM16?(streamPCM)
            }
        }
    }

    func endCapture(keepHardwareWarm: Bool = true) {
        lock.lock()
        captureEnabled = false
        meter = 0
        streamSamples.removeAll(keepingCapacity: true)
        lock.unlock()
        if !keepHardwareWarm { releaseHardware() }
    }

    func releaseHardware() {
        if configuredDeviceID != nil {
            engine.inputNode.removeTap(onBus: 0)
        }
        engine.stop()
        configuredDeviceID = nil
        primed = false
    }

    func takeSamples() -> (samples: [Float], sampleRate: Double) {
        lock.lock()
        let captured = samples
        let rate = hwSampleRate
        samples.removeAll(keepingCapacity: true)
        lock.unlock()
        return (captured, rate)
    }

    nonisolated static func writeWav(from captured: [Float], sampleRate srcRate: Double) -> String? {
        guard !captured.isEmpty else { return nil }
        // Require at least ~80ms so we don't ship a silent stub from a cold start.
        let minFrames = Int(srcRate * 0.08)
        guard captured.count >= minFrames else { return nil }

        let base = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
            .appendingPathComponent("Phonon", isDirectory: true)
            .appendingPathComponent("Corpus", isDirectory: true)
        let id = "bar_\(Int(Date().timeIntervalSince1970 * 1000))_\(ProcessInfo.processInfo.processIdentifier)"
        let directory = base.appendingPathComponent(id, isDirectory: true)
        try? FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let path = directory.appendingPathComponent("audio.wav").path
        let mono16k = resample(captured, from: srcRate, to: 16_000)
        do {
            try writeWavPCM16(path: path, samples: mono16k, sampleRate: 16_000)
            return path
        } catch {
            NSLog("wav write failed: \(error.localizedDescription)")
            return nil
        }
    }

    nonisolated static func discardWav(at path: String) {
        let url = URL(fileURLWithPath: path)
        guard url.lastPathComponent == "audio.wav",
            url.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent
                == "Corpus"
        else { return }
        try? FileManager.default.removeItem(at: url.deletingLastPathComponent())
    }

    func stop(writeFile: Bool = true) {
        endCapture()
        if writeFile {
            let (captured, rate) = takeSamples()
            wavPath = Self.writeWav(from: captured, sampleRate: rate)
        } else {
            _ = takeSamples()
            wavPath = nil
        }
        releaseHardware()
    }

    nonisolated private static func resample(_ input: [Float], from src: Double, to dst: Double)
        -> [Float]
    {
        guard !input.isEmpty, src > 0, dst > 0 else { return input }
        if abs(src - dst) < 0.5 { return input }
        let outCount = max(1, Int((Double(input.count) * dst / src).rounded()))
        var out = [Float](repeating: 0, count: outCount)
        let scale = src / dst
        let last = input.count - 1
        for i in 0..<outCount {
            let srcPos = Double(i) * scale
            let i0 = Int(srcPos)
            let i1 = min(i0 + 1, last)
            let frac = Float(srcPos - Double(i0))
            out[i] = input[min(i0, last)] + (input[i1] - input[min(i0, last)]) * frac
        }
        return out
    }

    nonisolated private static func writeWavPCM16(path: String, samples: [Float], sampleRate: Int)
        throws
    {
        let pcm = pcm16Data(samples)
        let dataSize = UInt32(pcm.count)
        let sampleRateU = UInt32(sampleRate)
        let byteRate = sampleRateU * 2
        let blockAlign = UInt16(2)
        let bitsPerSample = UInt16(16)
        let channels = UInt16(1)

        var data = Data()
        func append(_ s: String) { data.append(contentsOf: s.utf8) }
        func appendU16(_ v: UInt16) {
            var le = v.littleEndian
            withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
        }
        func appendU32(_ v: UInt32) {
            var le = v.littleEndian
            withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
        }

        append("RIFF")
        appendU32(36 + dataSize)
        append("WAVE")
        append("fmt ")
        appendU32(16)
        appendU16(1)
        appendU16(channels)
        appendU32(sampleRateU)
        appendU32(byteRate)
        appendU16(blockAlign)
        appendU16(bitsPerSample)
        append("data")
        appendU32(dataSize)
        data.append(pcm)
        try data.write(to: URL(fileURLWithPath: path), options: .atomic)
    }

    nonisolated private static func pcm16Data(_ samples: [Float]) -> Data {
        var pcm = Data(capacity: samples.count * 2)
        for s in samples {
            let clipped = max(-1.0, min(1.0, s))
            var v = Int16((clipped * Float(Int16.max)).rounded()).littleEndian
            withUnsafeBytes(of: &v) { pcm.append(contentsOf: $0) }
        }
        return pcm
    }
}

// MARK: - Typer

enum Typer {
    static func emit(_ text: String) throws {
        guard !text.isEmpty else { return }
        guard AXIsProcessTrusted() else {
            throw NSError(
                domain: "PhononBar", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Grant Accessibility to PhononBar"])
        }
        let utf16 = Array(text.utf16)
        guard let down = CGEvent(keyboardEventSource: nil, virtualKey: 0, keyDown: true),
            let up = CGEvent(keyboardEventSource: nil, virtualKey: 0, keyDown: false)
        else { return }
        let chunk = 32
        var i = 0
        while i < utf16.count {
            let end = min(i + chunk, utf16.count)
            var slice = Array(utf16[i..<end])
            down.keyboardSetUnicodeString(stringLength: slice.count, unicodeString: &slice)
            up.keyboardSetUnicodeString(stringLength: slice.count, unicodeString: &slice)
            down.post(tap: .cghidEventTap)
            up.post(tap: .cghidEventTap)
            i = end
        }
    }
}

// MARK: - End-to-end profiling

struct E2EProfileRecord: Codable {
    let timestamp: String
    let passId: String
    let source: String
    let textCharacters: Int
    let insertionSucceeded: Bool
    let keyEventToCallbackMs: Double
    let callbackToMainMs: Double
    let keyDownToRecordingMs: Double
    let keyDownToPanelMs: Double
    let keyDownToFirstStreamChunkMs: Double?
    let keyDownToFirstPartialMs: Double?
    let keyDownToFirstPreviewMs: Double?
    let dictationMs: Double
    let keyUpToMainMs: Double
    let keyUpToCaptureEndMs: Double
    let wavEncodeMs: Double
    let asrMs: Double
    let polishMs: Double
    let insertionPostMs: Double
    let keyUpToInsertMs: Double
    let keyDownToInsertMs: Double
    let asrReportedMs: Double?
    let llmReportedWallMs: Double?
    let llmTtftMs: Double?
    let llmTokensPerSecond: Double?

    enum CodingKeys: String, CodingKey {
        case timestamp
        case passId = "pass_id"
        case source
        case textCharacters = "text_characters"
        case insertionSucceeded = "insertion_succeeded"
        case keyEventToCallbackMs = "key_event_to_callback_ms"
        case callbackToMainMs = "callback_to_main_ms"
        case keyDownToRecordingMs = "key_down_to_recording_ms"
        case keyDownToPanelMs = "key_down_to_panel_ms"
        case keyDownToFirstStreamChunkMs = "key_down_to_first_stream_chunk_ms"
        case keyDownToFirstPartialMs = "key_down_to_first_partial_ms"
        case keyDownToFirstPreviewMs = "key_down_to_first_preview_ms"
        case dictationMs = "dictation_ms"
        case keyUpToMainMs = "key_up_to_main_ms"
        case keyUpToCaptureEndMs = "key_up_to_capture_end_ms"
        case wavEncodeMs = "wav_encode_ms"
        case asrMs = "asr_ms"
        case polishMs = "polish_ms"
        case insertionPostMs = "insertion_post_ms"
        case keyUpToInsertMs = "key_up_to_insert_ms"
        case keyDownToInsertMs = "key_down_to_insert_ms"
        case asrReportedMs = "asr_reported_ms"
        case llmReportedWallMs = "llm_reported_wall_ms"
        case llmTtftMs = "llm_ttft_ms"
        case llmTokensPerSecond = "llm_tokens_per_second"
    }
}

final class E2ETrace {
    let passId: String
    let source: String
    let keyDownEventNs: UInt64
    let keyDownCallbackNs: UInt64
    let keyDownHandledNs: UInt64
    var recordingStartedNs: UInt64?
    var panelShownNs: UInt64?
    var firstStreamChunkNs: UInt64?
    var firstPartialNs: UInt64?
    var firstPreviewNs: UInt64?
    var keyUpEventNs: UInt64?
    var keyUpHandledNs: UInt64?
    var captureEndedNs: UInt64?
    var wavReadyNs: UInt64?
    var asrSubmittedNs: UInt64?
    var asrResultNs: UInt64?
    var polishSubmittedNs: UInt64?
    var polishResultNs: UInt64?
    var insertionStartedNs: UInt64?
    var insertionCompletedNs: UInt64?
    var asrReportedMs: Double?
    var llmReportedWallMs: Double?
    var llmTtftMs: Double?
    var llmTokensPerSecond: Double?

    init(
        passId: String, source: String, keyDownEventNs: UInt64,
        keyDownCallbackNs: UInt64, keyDownHandledNs: UInt64
    ) {
        self.passId = passId
        self.source = source
        self.keyDownEventNs = keyDownEventNs
        self.keyDownCallbackNs = keyDownCallbackNs
        self.keyDownHandledNs = keyDownHandledNs
    }

    private func milliseconds(from start: UInt64?, to end: UInt64?) -> Double? {
        guard let start, let end, end >= start else { return nil }
        return Double(end - start) / 1_000_000
    }

    func finish(textCharacters: Int, insertionSucceeded: Bool) -> E2EProfileRecord? {
        guard
            let recordingStartedNs,
            let keyUpEventNs,
            let keyUpHandledNs,
            let captureEndedNs,
            let wavReadyNs,
            let asrSubmittedNs,
            let asrResultNs,
            let polishSubmittedNs,
            let polishResultNs,
            let insertionStartedNs,
            let insertionCompletedNs
        else { return nil }
        return E2EProfileRecord(
            timestamp: ISO8601DateFormatter().string(from: Date()),
            passId: passId,
            source: source,
            textCharacters: textCharacters,
            insertionSucceeded: insertionSucceeded,
            keyEventToCallbackMs: milliseconds(
                from: keyDownEventNs, to: keyDownCallbackNs) ?? 0,
            callbackToMainMs: milliseconds(from: keyDownCallbackNs, to: keyDownHandledNs) ?? 0,
            keyDownToRecordingMs: milliseconds(from: keyDownEventNs, to: recordingStartedNs) ?? 0,
            keyDownToPanelMs: milliseconds(from: keyDownEventNs, to: panelShownNs) ?? 0,
            keyDownToFirstStreamChunkMs: milliseconds(
                from: keyDownEventNs, to: firstStreamChunkNs),
            keyDownToFirstPartialMs: milliseconds(from: keyDownEventNs, to: firstPartialNs),
            keyDownToFirstPreviewMs: milliseconds(from: keyDownEventNs, to: firstPreviewNs),
            dictationMs: milliseconds(from: keyDownEventNs, to: keyUpEventNs) ?? 0,
            keyUpToMainMs: milliseconds(from: keyUpEventNs, to: keyUpHandledNs) ?? 0,
            keyUpToCaptureEndMs: milliseconds(from: keyUpEventNs, to: captureEndedNs) ?? 0,
            wavEncodeMs: milliseconds(from: captureEndedNs, to: wavReadyNs) ?? 0,
            asrMs: milliseconds(from: asrSubmittedNs, to: asrResultNs) ?? 0,
            polishMs: milliseconds(from: polishSubmittedNs, to: polishResultNs) ?? 0,
            insertionPostMs: milliseconds(from: insertionStartedNs, to: insertionCompletedNs) ?? 0,
            keyUpToInsertMs: milliseconds(from: keyUpEventNs, to: insertionCompletedNs) ?? 0,
            keyDownToInsertMs: milliseconds(from: keyDownEventNs, to: insertionCompletedNs) ?? 0,
            asrReportedMs: asrReportedMs,
            llmReportedWallMs: llmReportedWallMs,
            llmTtftMs: llmTtftMs,
            llmTokensPerSecond: llmTokensPerSecond
        )
    }
}

enum E2EProfileStore {
    static func logURL() -> URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/Phonon/e2e.jsonl")
    }

    static func append(_ record: E2EProfileRecord) {
        do {
            let url = logURL()
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            var data = try JSONEncoder().encode(record)
            data.append(0x0A)
            if FileManager.default.fileExists(atPath: url.path) {
                let handle = try FileHandle(forWritingTo: url)
                try handle.seekToEnd()
                try handle.write(contentsOf: data)
                try handle.close()
            } else {
                try data.write(to: url, options: .atomic)
            }
            NSLog("phonon e2e: \(String(data: data, encoding: .utf8) ?? "")")
        } catch {
            NSLog("phonon e2e log failed: \(error.localizedDescription)")
        }
    }
}

// MARK: - App

@MainActor
final class AppController: NSObject, NSApplicationDelegate {
    private let state = PillState()
    private let modelStartupState = ModelStartupState()
    private let appStore = NativeAppStore()
    private var panel: PillPanel?
    private var modelStartupWindow: NSPanel?
    private var mainWindow: NSWindow?
    private let engine = EngineClient()
    private let recorder = MicRecorder()
    private var statusItem: NSStatusItem?
    private var streamingMenuItem: NSMenuItem?
    private var microphoneMenuItem: NSMenuItem?
    private var pass = 0
    private var activeId: String?
    private var rawText = ""
    private var isRecording = false
    private var optionDown = false
    private var eventTap: CFMachPort?
    private var toggleHotKey: EventHotKeyRef?
    private var toggleHandler: EventHandlerRef?
    private var enginesReady = false
    private var hideWork: DispatchWorkItem?
    /// First dict often lands before models are warm — hold until ready.
    private var pendingTranscribe: (path: String, id: String)?
    private var submittedTranscribe: (path: String, id: String)?
    private var processing = false
    private var streamActive = false
    private var previewPolishInput: String?
    private var pendingPreviewText: String?
    private var previewRevision = 0
    private var lastPreviewPolishInput = ""
    private var latestAcousticPreview = ""
    private var e2eTrace: E2ETrace?
    private var screenContextText = ""
    private var screenContextReady = true
    private var pendingFinalPolish: (text: String, id: String)?
    private var screenContextTask: Task<Void, Never>?
    private var engineRestartWork: DispatchWorkItem?
    private var terminating = false
    private var activeWavPath: String?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        appStore.onDictionaryChanged = { [weak self] in
            self?.engine.send(["cmd": "reload_dictionary"])
        }
        appStore.onMicrophonePermissionGranted = { [weak self] in
            self?.warmMicrophoneIfAuthorized()
        }
        state.streamingPreviewEnabled = appStore.settings.streaming
        setupPanel()
        setupModelStartupWindow()
        setupMainWindow()
        setupStatusItem()

        if let demo = ProcessInfo.processInfo.environment["PHONON_UI_DEMO"] {
            if demo == "main" {
                showMainWindow()
                snapshotMainWindowIfRequested()
                return
            } else if demo == "startup" {
                showModelStatus()
                snapshotModelWindowIfRequested()
                return
            } else if demo == "spam" {
                runPanelSpamDemo()
                return
            } else if demo == "idle" {
                showIdle()
            } else if demo == "processing" {
                showWorking()
            } else {
                state.streamingPreviewEnabled = demo != "compact"
                showListening()
                state.level = 0.72
                if state.streamingPreviewEnabled {
                    setPreview(
                        ProcessInfo.processInfo.environment["PHONON_UI_DEMO_TEXT"]
                            ?? "The live transcript expands upward from the bottom dock.",
                        stage: "Polished live"
                    )
                }
            }
            snapshotPanelIfRequested()
            return
        }

        promptPermissions()
        showMainWindow()
        showModelStatus()
        startEngine()
        installToggleHotKey()
        installHotkey()
        showIdle()

        warmMicrophoneIfAuthorized()
    }

    private func warmMicrophoneIfAuthorized() {
        guard AVCaptureDevice.authorizationStatus(for: .audio) == .authorized else { return }
        if appStore.settings.instantMic { recorder.prewarm() }
        updateMicrophoneStatus()
    }

    private func setupPanel() {
        let size = Self.containerSize
        let panel = PillPanel(contentRect: PillPanel.frame(size: size))
        let host = NSHostingController(rootView: PillView(state: state))
        host.view.frame = NSRect(origin: .zero, size: size)
        panel.contentView = host.view
        self.panel = panel
    }

    private func setupModelStartupWindow() {
        let host = NSHostingController(rootView: ModelStartupView(state: modelStartupState))
        let window = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 520, height: 300),
            styleMask: [.titled, .closable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.title = "Phonon model status"
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.isMovableByWindowBackground = true
        window.backgroundColor = .clear
        window.isOpaque = false
        window.hasShadow = true
        window.contentViewController = host
        window.center()
        modelStartupWindow = window
    }

    private func setupMainWindow() {
        let initialPage = ProcessInfo.processInfo.environment["PHONON_MAIN_PAGE"]
            .flatMap(NativeAppPage.init(rawValue:)) ?? .home
        let root = PhononMainView(
            store: appStore,
            initialPage: initialPage,
            onSettingsChanged: { [weak self] in self?.applySettings() },
            onShowModelStatus: { [weak self] in self?.showModelStatus() }
        )
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 980, height: 680),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.title = "Phonon"
        window.titlebarAppearsTransparent = true
        window.backgroundColor = EmberTheme.nsBackground
        window.appearance = NSAppearance(named: .darkAqua)
        window.contentViewController = NSHostingController(rootView: root)
        window.setContentSize(NSSize(width: 980, height: 680))
        window.minSize = NSSize(width: 900, height: 620)
        window.center()
        mainWindow = window
    }

    private func setupStatusItem() {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        item.button?.title = "ϕ"
        item.button?.toolTip = "Phonon — hold ⌥ to dictate"
        let menu = NSMenu()
        menu.addItem(
            NSMenuItem(title: "Open Phonon…", action: #selector(showMainWindow), keyEquivalent: ","))
        menu.addItem(.separator())
        menu.addItem(
            NSMenuItem(title: "Toggle record", action: #selector(toggleRecord), keyEquivalent: "r"))
        let previewItem = NSMenuItem(
            title: "Streaming", action: #selector(toggleStreamingPreview), keyEquivalent: ""
        )
        previewItem.state = state.streamingPreviewEnabled ? .on : .off
        menu.addItem(previewItem)
        streamingMenuItem = previewItem
        menu.addItem(
            NSMenuItem(
                title: "Model status…", action: #selector(showModelStatus), keyEquivalent: ""))
        let microphoneItem = NSMenuItem(
            title: "Microphone: Auto → Resolving…", action: nil, keyEquivalent: "")
        microphoneItem.isEnabled = false
        menu.addItem(microphoneItem)
        microphoneMenuItem = microphoneItem
        menu.addItem(.separator())
        menu.addItem(NSMenuItem(title: "Quit", action: #selector(quit), keyEquivalent: "q"))
        item.menu = menu
        statusItem = item
    }

    @objc private func showModelStatus() {
        NSApp.activate(ignoringOtherApps: true)
        modelStartupWindow?.center()
        modelStartupWindow?.orderFrontRegardless()
    }

    @objc private func showMainWindow() {
        appStore.reloadAll()
        appStore.refreshPermissions()
        NSApp.activate(ignoringOtherApps: true)
        mainWindow?.makeKeyAndOrderFront(nil)
    }

    private func updateMicrophoneStatus() {
        microphoneMenuItem?.title = "Microphone: Auto → \(recorder.selectedDeviceName)"
        appStore.selectedMicrophone = recorder.selectedDeviceName
    }

    private func applySettings() {
        if isRecording {
            stopDictation(source: "settings")
        }
        state.streamingPreviewEnabled = appStore.settings.streaming
        streamingMenuItem?.state = state.streamingPreviewEnabled ? .on : .off
        if appStore.settings.instantMic {
            recorder.refreshPreferredDevice()
            updateMicrophoneStatus()
        } else if !isRecording {
            recorder.releaseHardware()
        }
    }

    private func promptPermissions() {
        let opts = ["AXTrustedCheckOptionPrompt": true] as CFDictionary
        _ = AXIsProcessTrustedWithOptions(opts)
    }

    private func startEngine() {
        engine.onEvent = { [weak self] ev in
            Task { @MainActor in self?.handleEngine(ev) }
        }
        engine.onExit = { [weak self] in
            self?.recoverFromEngineExit()
        }
        do {
            try engine.start()
        } catch {
            NSLog("phonon engine failed: \(error.localizedDescription)")
            scheduleEngineRestart()
        }
    }

    private func recoverFromEngineExit() {
        guard !terminating else { return }
        enginesReady = false
        appStore.engineReady = false
        appStore.engineMessage = "Engine restarting"
        streamActive = false
        statusItem?.button?.title = "ϕ…"
        modelStartupState.markRestarting()
        showModelStatus()
        if processing, let submittedTranscribe, submittedTranscribe.id == activeId {
            pendingTranscribe = submittedTranscribe
        } else if processing, let activeId, !rawText.isEmpty {
            pendingFinalPolish = (rawText, activeId)
        }
        scheduleEngineRestart()
    }

    private func scheduleEngineRestart() {
        guard !terminating else { return }
        engineRestartWork?.cancel()
        let work = DispatchWorkItem { [weak self] in self?.startEngine() }
        engineRestartWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 1, execute: work)
    }

    private func handleEngine(_ ev: [String: Any]) {
        let type = ev["type"] as? String ?? ""
        switch type {
        case "stream":
            appStore.engineMessage = ev["msg"] as? String ?? "Loading local models"
            modelStartupState.apply(
                name: ev["name"] as? String ?? "",
                state: ev["state"] as? String ?? "loading",
                progress: ev["pct"] as? Double ?? 0,
                detail: ev["msg"] as? String ?? "",
                loadMs: ev["load_ms"] as? Double
            )
        case "ready":
            enginesReady = true
            appStore.engineReady = true
            appStore.engineMessage = "Ready"
            modelStartupState.markReady()
            engineRestartWork?.cancel()
            engineRestartWork = nil
            statusItem?.button?.title = "ϕ"
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) { [weak self] in
                guard self?.enginesReady == true else { return }
                self?.modelStartupWindow?.orderOut(nil)
            }
            if isRecording && state.streamingPreviewEnabled && !streamActive {
                startAsrStream()
            }
            // Flush first-pass work that arrived early.
            if let pending = pendingTranscribe {
                pendingTranscribe = nil
                sendTranscribe(path: pending.path, id: pending.id)
            } else if screenContextReady, let pending = pendingFinalPolish {
                pendingFinalPolish = nil
                submitFinalPolish(pending.text, id: pending.id)
            }
        case "result":
            let kind = ev["kind"] as? String ?? ""
            let id = ev["id"] as? String

            if kind == "asr" {
                let text = (ev["text"] as? String ?? "").trimmingCharacters(
                    in: .whitespacesAndNewlines)
                let partial = ev["partial"] as? Bool ?? false
                if partial {
                    guard isRecording, id == activeId, !text.isEmpty else { return }
                    if e2eTrace?.passId == id, e2eTrace?.firstPartialNs == nil {
                        e2eTrace?.firstPartialNs = DispatchTime.now().uptimeNanoseconds
                    }
                    let stable = PreviewRevisionPolicy.stableAcoustic(
                        previous: latestAcousticPreview, candidate: text)
                    guard stable != latestAcousticPreview else { return }
                    latestAcousticPreview = stable
                    setPreview(LiveTranscriptFormatter.format(stable), stage: "Live cleanup")
                    if e2eTrace?.passId == id, e2eTrace?.firstPreviewNs == nil {
                        e2eTrace?.firstPreviewNs = DispatchTime.now().uptimeNanoseconds
                    }
                    return
                }
                guard id == activeId else { return }
                if e2eTrace?.passId == id {
                    e2eTrace?.asrResultNs = DispatchTime.now().uptimeNanoseconds
                    e2eTrace?.asrReportedMs = (ev["seconds"] as? Double).map { $0 * 1_000 }
                }
                rawText = text
                submittedTranscribe = nil
                let pid = activeId ?? "p\(pass)"
                if text.isEmpty {
                    // Empty ASR — don't hang the loader forever.
                    NSLog("phonon: empty ASR for \(pid)")
                    processing = false
                    finishRecordingRetention()
                    hideAfter(0.1)
                    return
                }
                submitFinalPolish(text, id: pid)
            } else if kind == "polish" {
                if let id, let activeId, id.hasPrefix("\(activeId):preview:") {
                    let source = previewPolishInput ?? ""
                    previewPolishInput = nil
                    if isRecording {
                        let polished = (ev["text"] as? String ?? "").trimmingCharacters(
                            in: .whitespacesAndNewlines)
                        if PreviewRevisionPolicy.canShowPolish(
                            source: source,
                            latestAcoustic: latestAcousticPreview,
                            polished: polished,
                            newerCandidateWaiting: pendingPreviewText != nil
                        ) {
                            setPreview(polished, stage: "Polished live")
                        }
                        submitPendingPreviewPolish()
                    }
                    return
                }
                guard id == activeId else { return }
                if e2eTrace?.passId == id {
                    e2eTrace?.polishResultNs = DispatchTime.now().uptimeNanoseconds
                    e2eTrace?.llmReportedWallMs = ev["latency_ms"] as? Double
                    e2eTrace?.llmTtftMs = ev["ttft_ms"] as? Double
                    e2eTrace?.llmTokensPerSecond = ev["tok_s"] as? Double
                }
                let text = ev["text"] as? String ?? rawText
                let out = text.trimmingCharacters(in: .whitespacesAndNewlines)
                let final = out.isEmpty ? rawText : out
                processing = false
                if !final.isEmpty {
                    setPreview(final, stage: "Inserted")
                    e2eTrace?.insertionStartedNs = DispatchTime.now().uptimeNanoseconds
                    var insertionSucceeded = false
                    do {
                        try Typer.emit(final)
                        insertionSucceeded = true
                    } catch {
                        NSLog("phonon insertion failed: \(error.localizedDescription)")
                    }
                    e2eTrace?.insertionCompletedNs = DispatchTime.now().uptimeNanoseconds
                    if let record = e2eTrace?.finish(
                        textCharacters: final.count,
                        insertionSucceeded: insertionSucceeded
                    ) {
                        E2EProfileStore.append(record)
                    }
                    e2eTrace = nil
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(final, forType: .string)
                    finishRecordingRetention()
                    hideAfter(state.streamingPreviewEnabled ? 0.55 : 0)
                } else {
                    hidePill()
                }
            }
        case "error":
            let msg = ev["msg"] as? String ?? "error"
            appStore.lastError = msg
            NSLog("phonon engine error: \(msg)")
            if let id = ev["id"] as? String, id.contains(":preview:") {
                guard let activeId, id.hasPrefix("\(activeId):preview:") else { return }
                previewPolishInput = nil
                if isRecording { submitPendingPreviewPolish() }
                return
            }
            if let id = ev["id"] as? String, id != activeId { return }
            if isRecording { return }
            // Don't kill a queued first pass on boot-stream noise.
            if pendingTranscribe != nil { return }
            processing = false
            finishRecordingRetention()
            hideAfter(0.12)
        default:
            break
        }
    }

    // MARK: Visibility

    private static let containerSize = NSSize(width: 80, height: 16)

    private func showIdle() {
        hideWork?.cancel()
        state.level = 0
        state.mode = .idle
        present()
    }

    private func showListening() {
        hideWork?.cancel()
        state.level = 0
        setPreview("", stage: enginesReady ? "Listening" : "Loading speech engine")
        state.mode = .listening
        present()
    }

    private func showWorking() {
        hideWork?.cancel()
        state.level = 0
        state.mode = .working
        state.previewStage = "Final pass"
        present()
    }

    private func present() {
        guard let panel else { return }
        hideWork?.cancel()
        let target = PillPanel.frame(size: Self.containerSize)
        panel.contentView?.setFrameSize(Self.containerSize)
        if !panel.isVisible {
            panel.setFrame(target, display: false)
            panel.alphaValue = 1
            panel.orderFrontRegardless()
        }
    }

    private func hidePill() {
        hideWork?.cancel()
        guard let panel else { return }
        if !panel.isVisible {
            showIdle()
            return
        }
        state.level = 0
        state.mode = .idle
    }

    private func hideAfter(_ seconds: TimeInterval) {
        hideWork?.cancel()
        let id = activeId
        let work = DispatchWorkItem { [weak self] in
            guard let self, self.activeId == id, !self.isRecording else { return }
            self.hidePill()
        }
        hideWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + seconds, execute: work)
    }

    private func setPreview(_ text: String, stage: String) {
        state.previewText = text
        state.previewStage = stage
    }

    private func snapshotPanelIfRequested() {
        guard let path = ProcessInfo.processInfo.environment["PHONON_UI_SNAPSHOT"] else { return }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) { [weak self] in
            guard let view = self?.panel?.contentView,
                let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds)
            else { return }
            view.cacheDisplay(in: view.bounds, to: rep)
            guard let png = rep.representation(using: .png, properties: [:]) else { return }
            try? png.write(to: URL(fileURLWithPath: path), options: .atomic)
            if ProcessInfo.processInfo.environment["PHONON_UI_EXIT_AFTER_SNAPSHOT"] == "1" {
                NSApp.terminate(nil)
            }
        }
    }

    private func snapshotMainWindowIfRequested() {
        snapshotWindow(mainWindow)
    }

    private func snapshotModelWindowIfRequested() {
        snapshotWindow(modelStartupWindow)
    }

    private func snapshotWindow(_ window: NSWindow?) {
        guard let path = ProcessInfo.processInfo.environment["PHONON_UI_SNAPSHOT"] else { return }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            guard let view = window?.contentView,
                let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds)
            else { return }
            view.cacheDisplay(in: view.bounds, to: rep)
            guard let png = rep.representation(using: .png, properties: [:]) else { return }
            try? png.write(to: URL(fileURLWithPath: path), options: .atomic)
            if ProcessInfo.processInfo.environment["PHONON_UI_EXIT_AFTER_SNAPSHOT"] == "1" {
                NSApp.terminate(nil)
            }
        }
    }

    private func runPanelSpamDemo() {
        let transitions = 24
        for index in 0..<transitions {
            DispatchQueue.main.asyncAfter(deadline: .now() + Double(index) * 0.015) { [weak self] in
                guard let self else { return }
                if index.isMultiple(of: 2) {
                    self.showListening()
                    self.state.level = 0.72
                } else {
                    self.hidePill()
                }
                if index == transitions - 1 {
                    self.showListening()
                    self.state.level = 0.72
                    self.snapshotPanelIfRequested()
                }
            }
        }
    }

    // MARK: Record

    @objc func toggleRecord() {
        let now = DispatchTime.now().uptimeNanoseconds
        toggleRecordFromKeyboard(source: "menu", eventNs: now, callbackNs: now)
    }

    @objc private func toggleStreamingPreview() {
        state.streamingPreviewEnabled.toggle()
        appStore.updateSettings { $0.streaming = state.streamingPreviewEnabled }
        streamingMenuItem?.state = state.streamingPreviewEnabled ? .on : .off
        switch state.mode {
        case .hidden, .idle, .listening, .working:
            break
        }
    }

    private func toggleRecordFromKeyboard(source: String, eventNs: UInt64, callbackNs: UInt64) {
        guard shortcutAllows(source: source) else { return }
        if isRecording {
            stopDictation(source: source, eventNs: eventNs, callbackNs: callbackNs)
        } else {
            startDictation(source: source, eventNs: eventNs, callbackNs: callbackNs)
        }
    }

    func startDictation(
        source: String = "direct", eventNs: UInt64 = DispatchTime.now().uptimeNanoseconds,
        callbackNs: UInt64 = DispatchTime.now().uptimeNanoseconds
    ) {
        // If previous pass still polishing, abandon it for snappy re-trigger.
        if processing {
            if let pendingTranscribe {
                MicRecorder.discardWav(at: pendingTranscribe.path)
            }
            pendingTranscribe = nil
            submittedTranscribe = nil
            pendingFinalPolish = nil
            processing = false
        }
        guard !isRecording else { return }
        activeWavPath = nil
        pass += 1
        activeId = "p\(pass)"
        rawText = ""
        e2eTrace = E2ETrace(
            passId: activeId ?? "p\(pass)",
            source: source,
            keyDownEventNs: eventNs,
            keyDownCallbackNs: callbackNs,
            keyDownHandledNs: DispatchTime.now().uptimeNanoseconds
        )
        isRecording = true
        previewPolishInput = nil
        pendingPreviewText = nil
        previewRevision = 0
        lastPreviewPolishInput = ""
        latestAcousticPreview = ""
        screenContextText = ""
        screenContextReady = !appStore.settings.screenContext
        pendingFinalPolish = nil
        screenContextTask?.cancel()
        if appStore.settings.screenContext {
            let contextPass = activeId
            screenContextTask = Task { [weak self] in
                let text = await ScreenContextCapture.recognizeAllDisplays()
                guard !Task.isCancelled, let self, self.activeId == contextPass else { return }
                self.screenContextText = text
                self.screenContextReady = true
                if let pending = self.pendingFinalPolish {
                    self.pendingFinalPolish = nil
                    self.submitFinalPolish(pending.text, id: pending.id)
                }
            }
        }
        showListening()
        e2eTrace?.panelShownNs = DispatchTime.now().uptimeNanoseconds
        recorder.onAmplitude = { [weak self] a in
            Task { @MainActor in
                guard let self, self.isRecording else { return }
                self.state.level = a
            }
        }
        recorder.onStreamPCM16 = { [weak self] pcm in
            Task { @MainActor in
                guard let self, self.isRecording, self.streamActive else { return }
                if self.e2eTrace?.firstStreamChunkNs == nil {
                    self.e2eTrace?.firstStreamChunkNs = DispatchTime.now().uptimeNanoseconds
                }
                self.engine.send([
                    "cmd": "stream_chunk",
                    "id": self.activeId ?? "p\(self.pass)",
                    "pcm16": pcm.base64EncodedString(),
                ])
            }
        }
        if enginesReady && state.streamingPreviewEnabled {
            startAsrStream()
        }
        do {
            try recorder.start()
            updateMicrophoneStatus()
            e2eTrace?.recordingStartedNs = DispatchTime.now().uptimeNanoseconds
        } catch {
            isRecording = false
            hidePill()
            NSLog("mic: \(error.localizedDescription)")
        }
    }

    func stopDictation(
        source _: String = "direct", eventNs: UInt64 = DispatchTime.now().uptimeNanoseconds,
        callbackNs _: UInt64 = DispatchTime.now().uptimeNanoseconds
    ) {
        guard isRecording else { return }
        e2eTrace?.keyUpEventNs = eventNs
        e2eTrace?.keyUpHandledNs = DispatchTime.now().uptimeNanoseconds
        isRecording = false
        processing = true
        pendingPreviewText = nil
        if streamActive {
            engine.send(["cmd": "stream_stop", "id": activeId ?? "p\(pass)"])
            streamActive = false
        }

        // Instant fireball on key-up.
        showWorking()

        recorder.endCapture(keepHardwareWarm: appStore.settings.instantMic)
        e2eTrace?.captureEndedNs = DispatchTime.now().uptimeNanoseconds
        let (captured, rate) = recorder.takeSamples()
        let id = activeId ?? "p\(pass)"

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let path = MicRecorder.writeWav(from: captured, sampleRate: rate)
            DispatchQueue.main.async {
                guard let self else { return }
                guard self.activeId == id, self.processing else {
                    if let path { MicRecorder.discardWav(at: path) }
                    return
                }
                if self.e2eTrace?.passId == id {
                    self.e2eTrace?.wavReadyNs = DispatchTime.now().uptimeNanoseconds
                }
                guard let path else {
                    NSLog(
                        "phonon: empty/short capture (\(captured.count) frames @ \(rate) Hz) — pass \(id)"
                    )
                    // Stay visible briefly so it doesn't feel broken, then hide.
                    self.processing = false
                    self.hideAfter(0.25)
                    return
                }
                self.activeWavPath = path
                self.submitTranscribe(path: path, id: id)
            }
        }
    }

    private func submitTranscribe(path: String, id: String) {
        guard activeId == id, processing else {
            MicRecorder.discardWav(at: path)
            return
        }
        if enginesReady {
            sendTranscribe(path: path, id: id)
        } else {
            // First-pass classic failure: models still loading. Queue + keep fireball.
            NSLog("phonon: queueing transcribe until engine ready (\(id))")
            pendingTranscribe = (path, id)
            showWorking()
        }
    }

    private func sendTranscribe(path: String, id: String) {
        guard activeId == id, processing else { return }
        if e2eTrace?.passId == id, e2eTrace?.asrSubmittedNs == nil {
            e2eTrace?.asrSubmittedNs = DispatchTime.now().uptimeNanoseconds
        }
        submittedTranscribe = (path, id)
        engine.send(["cmd": "transcribe", "path": path, "id": id, "source": "bar"])
    }

    private func finishRecordingRetention() {
        defer {
            activeWavPath = nil
            appStore.reloadAll()
        }
        guard !appStore.settings.localHistory, let activeWavPath else { return }
        MicRecorder.discardWav(at: activeWavPath)
    }

    private func shortcutAllows(source: String) -> Bool {
        ShortcutPolicy.allows(mode: appStore.settings.shortcutMode, source: source)
    }

    private func submitFinalPolish(_ text: String, id: String) {
        guard activeId == id, processing else { return }
        guard screenContextReady else {
            pendingFinalPolish = (text, id)
            return
        }
        if e2eTrace?.passId == id {
            e2eTrace?.polishSubmittedNs = DispatchTime.now().uptimeNanoseconds
        }
        engine.send([
            "cmd": "polish", "text": text, "context": screenContextText, "id": id,
        ])
    }

    private func startAsrStream() {
        guard !streamActive else { return }
        let id = activeId ?? "p\(pass)"
        engine.send(["cmd": "stream_start", "id": id])
        streamActive = true
    }

    private func queuePreviewPolish(_ text: String) {
        guard enginesReady, text != lastPreviewPolishInput else { return }
        if previewPolishInput != nil {
            pendingPreviewText = text
            return
        }
        previewRevision += 1
        previewPolishInput = text
        lastPreviewPolishInput = text
        state.previewStage = "Polishing live"
        let id = "\(activeId ?? "p\(pass)"):preview:\(previewRevision)"
        engine.send([
            "cmd": "polish", "text": text,
            "context": screenContextReady ? screenContextText : "", "id": id,
        ])
    }

    private func submitPendingPreviewPolish() {
        guard let text = pendingPreviewText else { return }
        pendingPreviewText = nil
        queuePreviewPolish(text)
    }

    // MARK: Hotkey

    /// Carbon hotkeys do not require Input Monitoring, so toggle recording remains
    /// usable while macOS is waiting for Right-Option monitoring permission.
    private func installToggleHotKey() {
        var eventSpec = EventTypeSpec(
            eventClass: OSType(kEventClassKeyboard),
            eventKind: UInt32(kEventHotKeyPressed)
        )
        let callback: EventHandlerUPP = { _, _, refcon in
            guard let refcon else { return OSStatus(eventNotHandledErr) }
            let app = Unmanaged<AppController>.fromOpaque(refcon).takeUnretainedValue()
            let callbackNs = DispatchTime.now().uptimeNanoseconds
            Task { @MainActor in
                app.toggleRecordFromKeyboard(
                    source: "control-space", eventNs: callbackNs, callbackNs: callbackNs)
            }
            return noErr
        }
        let handlerStatus = InstallEventHandler(
            GetApplicationEventTarget(),
            callback,
            1,
            &eventSpec,
            Unmanaged.passUnretained(self).toOpaque(),
            &toggleHandler
        )

        let hotKeyId = EventHotKeyID(signature: 0x50484F4E, id: 1)  // PHON
        let hotKeyStatus = RegisterEventHotKey(
            UInt32(kVK_Space),
            UInt32(controlKey),
            hotKeyId,
            GetApplicationEventTarget(),
            0,
            &toggleHotKey
        )
        if handlerStatus != noErr || hotKeyStatus != noErr {
            NSLog(
                "phonon: Ctrl+Space registration failed (handler=\(handlerStatus), hotkey=\(hotKeyStatus))"
            )
        }
    }

    private func installHotkey() {
        let mask = 1 << CGEventType.flagsChanged.rawValue
        let callback: CGEventTapCallBack = { _, type, event, refcon in
            guard let refcon else { return Unmanaged.passUnretained(event) }
            let app = Unmanaged<AppController>.fromOpaque(refcon).takeUnretainedValue()
            if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
                if let tap = app.eventTap {
                    CGEvent.tapEnable(tap: tap, enable: true)
                }
                return Unmanaged.passUnretained(event)
            }
            app.handleCGEvent(type: type, event: event)
            return Unmanaged.passUnretained(event)
        }
        guard
            let tap = CGEvent.tapCreate(
                tap: .cgSessionEventTap,
                place: .headInsertEventTap,
                options: .listenOnly,
                eventsOfInterest: CGEventMask(mask),
                callback: callback,
                userInfo: Unmanaged.passUnretained(self).toOpaque()
            )
        else {
            appStore.inputMonitoringAvailable = false
            statusItem?.button?.toolTip =
                "Phonon — Ctrl+Space ready; grant Input Monitoring for hold ⌥"
            NSLog("phonon: Right-Option unavailable (Input Monitoring not granted)")
            return
        }
        eventTap = tap
        appStore.inputMonitoringAvailable = true
        let src = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
        CFRunLoopAddSource(CFRunLoopGetMain(), src, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)
    }

    nonisolated private func handleCGEvent(type: CGEventType, event: CGEvent) {
        if type == .flagsChanged {
            let flags = event.flags
            let anyAlt =
                flags.contains(.maskAlternate)
                && !flags.contains(.maskCommand)
                && !flags.contains(.maskControl)
                && !flags.contains(.maskShift)
            let eventNs = event.timestamp
            let callbackNs = DispatchTime.now().uptimeNanoseconds
            Task { @MainActor in
                guard self.shortcutAllows(source: "right-option") else {
                    self.optionDown = false
                    return
                }
                if anyAlt && !self.optionDown {
                    self.optionDown = true
                    self.startDictation(
                        source: "right-option", eventNs: eventNs, callbackNs: callbackNs)
                } else if !anyAlt && self.optionDown {
                    self.optionDown = false
                    self.stopDictation(
                        source: "right-option", eventNs: eventNs, callbackNs: callbackNs)
                }
            }
        }
    }

    @objc func quit() {
        terminating = true
        engineRestartWork?.cancel()
        screenContextTask?.cancel()
        engine.shutdown()
        NSApp.terminate(nil)
    }

    func applicationWillTerminate(_ notification: Notification) {
        terminating = true
        engineRestartWork?.cancel()
        screenContextTask?.cancel()
        engine.shutdown()
        recorder.stop(writeFile: false)
    }

    func applicationDidBecomeActive(_ notification: Notification) {
        appStore.refreshPermissions()
    }

    func applicationShouldHandleReopen(
        _ sender: NSApplication, hasVisibleWindows flag: Bool
    ) -> Bool {
        if !flag { showMainWindow() }
        return true
    }
}

@main
enum PhononBarMain {
    private static var instanceLock: SingleInstanceLock?

    static func main() {
        let lock = ProcessInfo.processInfo.environment["PHONON_INSTANCE_LOCK_PATH"]
            .flatMap { SingleInstanceLock.acquire(at: URL(fileURLWithPath: $0)) }
            ?? SingleInstanceLock.acquire()
        guard let lock else { return }
        instanceLock = lock
        let app = NSApplication.shared
        let delegate = AppController()
        app.delegate = delegate
        app.run()
    }
}
