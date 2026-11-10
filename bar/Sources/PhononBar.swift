// Phonon: the native macOS app plus a bottom-anchored dictation capsule.
//
// Idle, the capsule is a small bar. Holding the shortcut (Globe by default)
// expands it; on key-up it breathes while the text is corrected, then the text
// is typed and the capsule returns to idle. Work that arrives before the engine
// is warm is queued, not dropped.

import AVFoundation
import AppKit
import ApplicationServices
import AudioToolbox
import Carbon.HIToolbox
import CoreAudio
import Darwin
import Foundation
import PhononObjC
import QuartzCore
import ScreenCaptureKit
import SwiftUI
import Vision
import os

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
    /// A meeting is being recorded: the red light shows in the capsule.
    @Published var meetingRecording = false
    /// The pointer is over the recording capsule: the stop button replaces the light.
    @Published var stopButtonShown = false
    /// Settings `show_idle_capsule`: off hides the capsule between dictations.
    @Published var showWhenIdle = true

    /// Whether the capsule is drawn. Idle shows it only when the setting allows, or
    /// while a meeting records (its red light and stop button live in the capsule).
    var capsuleVisible: Bool {
        switch mode {
        case .hidden: return false
        case .idle: return showWhenIdle || meetingRecording
        case .listening, .working: return true
        }
    }

    func apply(_ flow: MeetingFlow) {
        if meetingRecording != flow.isRecording { meetingRecording = flow.isRecording }
        if stopButtonShown != flow.showsStopButton { stopButtonShown = flow.showsStopButton }
    }
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
        ModelLoadStream(id: "llm", title: "Correction model"),
    ]
    @Published private(set) var ready = false
    /// The main window shows the loader in place of the app until startup
    /// finishes (and again while the engine restarts).
    @Published private(set) var loaderVisible = true

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

    func hideLoader() {
        loaderVisible = false
    }

    func markRestarting() {
        ready = false
        loaderVisible = true
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
                    Text("Phonon checks both models with a short test before you dictate.")
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

/// The one app window: the startup loader until both models pass their checks,
/// then the full app. The app stays mounted underneath so first-run permission
/// and privacy sheets still appear at launch.
struct PhononWindowRoot: View {
    @ObservedObject var startup: ModelStartupState
    let main: PhononMainView

    var body: some View {
        ZStack {
            main.accessibilityHidden(startup.loaderVisible)
            if startup.loaderVisible {
                ZStack {
                    EmberTheme.background
                    VStack(spacing: 4) {
                        ModelStartupView(state: startup)
                        // A missing permission is fixed while the models load.
                        PermissionBanner(store: main.store).frame(width: 640)
                    }
                }
                .contentShape(Rectangle())
                .transition(.opacity)
            }
        }
        .animation(.easeOut(duration: 0.25), value: startup.loaderVisible)
    }
}

/// Streaming refine (SPEC "Streaming refine"): the engine decides (settings.json
/// `stream_refine`, PHONON_STREAM_REFINE) and says so in its ready event.
enum StreamRefine {
    static func enabled(readyEvent ev: [String: Any]) -> Bool {
        ev["stream_refine"] as? Bool ?? false
    }
}

enum PreviewRevisionPolicy {
    static func stableAcoustic(previous: String, candidate: String) -> String {
        wordCount(candidate) < wordCount(previous) ? previous : candidate
    }

    static func safeFinalText(source: String, candidate: String) -> String {
        let trimmed = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return source }
        let sourceWords = wordCount(source)
        let candidateWords = wordCount(trimmed)
        let looksLikeEnvelope = trimmed.contains("<phonon_dictionary")
            || trimmed.contains("</phonon_dictionary>")
            || trimmed.contains("canonical_terms:")
        let expandedTooFar = candidateWords > sourceWords * 2 + 8
            || trimmed.count > source.count * 4 + 64
        return looksLikeEnvelope || expandedTooFar ? source : trimmed
    }

    private static func wordCount(_ text: String) -> Int {
        text.split(whereSeparator: { $0.isWhitespace }).count
    }
}

enum ShortcutPolicy {
    /// A mode names exactly the event sources it accepts. A hold key and the
    /// Control-Space toggle are independent, so a mode may enable either or both.
    static func sources(for mode: String) -> Set<String> {
        switch mode {
        case "right_option": return ["right-option"]
        case "fn": return ["fn"]
        case "control_space": return ["control-space"]
        case "fn_and_control_space": return ["fn", "control-space"]
        default: return ["right-option", "control-space"]
        }
    }

    static func allows(mode: String, source: String) -> Bool {
        sources(for: mode).contains(source)
    }

    /// The mode has a hold key (fn or Right Option), which needs Input Monitoring.
    static func usesHoldKey(mode: String) -> Bool {
        let sources = sources(for: mode)
        return sources.contains("fn") || sources.contains("right-option")
    }

    /// What the hold key is called in the UI.
    static func holdKeyName(mode: String) -> String {
        sources(for: mode).contains("fn") ? "Globe (fn)" : "Right Option"
    }
}

/// Tells the right Option key from the left one (issue #23; ported from outside
/// PR #29). The generic Option flag is set while either is held, so the
/// device-dependent bits (IOKit `NX_DEVICERALTKEYMASK` and `NX_DEVICELALTKEYMASK`)
/// say which one it is.
enum RightOptionKey {
    static let alternate: UInt64 = 0x80000
    static let deviceLeft: UInt64 = 0x20
    static let deviceRight: UInt64 = 0x40
    static let otherModifiers: UInt64 = 0x20000 | 0x40000 | 0x100000  // shift, control, command
    static let rightKeyCode = 61
    static let leftKeyCode = 58

    static func isDown(keyCode: Int, rawFlags: UInt64, wasDown: Bool) -> Bool {
        guard rawFlags & alternate != 0, rawFlags & otherModifiers == 0 else { return false }
        if rawFlags & (deviceLeft | deviceRight) != 0 {
            return rawFlags & deviceRight != 0
        }
        // No device bits (a synthetic or remapped keyboard): the key code decides.
        switch keyCode {
        case rightKeyCode: return true
        case leftKeyCode: return false
        default: return wasDown
        }
    }
}

/// Turns the hold key's level (is it held, per the event flags) into edges.
///
/// A flags-changed event for the key's own key code while it already reads as
/// held means its release never reached Phonon (the event tap was disabled for
/// a moment, issue #21): before this, that press was ignored and its release
/// stopped nothing, so the press after a lost release never started dictation.
struct HoldKeyEdges {
    enum Edge: Equatable {
        case down
        case up
        /// Pressed again without a release in between.
        case repress
        case none
    }

    private(set) var isDown = false

    mutating func update(pressed: Bool, ownKey: Bool) -> Edge {
        switch (pressed, isDown) {
        case (true, false):
            isDown = true
            return .down
        case (false, true):
            isDown = false
            return .up
        case (true, true):
            return ownKey ? .repress : .none
        case (false, false):
            return .none
        }
    }

    mutating func reset() { isDown = false }
}

/// macOS's own use of the Globe key (System Settings > Keyboard > "Press 🌐 key to",
/// `com.apple.HIToolbox` `AppleFnUsageType`): 0 does nothing, 1 changes the input
/// source, 2 shows emoji, 3 starts Apple's dictation. Anything but 0 also fires on
/// the taps Phonon listens for. An absent key is the system default, which is not 0.
enum GlobeKeySetting {
    static func conflicts(usageType: Int?) -> Bool { usageType != 0 }

    static func action(usageType: Int?) -> String {
        switch usageType {
        case 1: return "changes the input source"
        case 2: return "shows Emoji & Symbols"
        case 3: return "starts Apple's dictation"
        default: return "does something of its own"
        }
    }

    static func currentUsageType() -> Int? {
        UserDefaults(suiteName: "com.apple.HIToolbox")?.object(forKey: "AppleFnUsageType")
            as? Int
    }
}

/// The menu-bar title. At the menu font's 13 pt the "ϕ" read as too small next to
/// the system icons, so the glyph is drawn larger and the marks after it (●, …, ⚠)
/// stay at menu size. At 19 pt the glyph is about as tall as the Wi-Fi icon.
enum StatusTitle {
    static let glyphSize: CGFloat = 19
    static let markSize: CGFloat = 13

    static func attributed(_ text: String) -> NSAttributedString {
        let out = NSMutableAttributedString()
        for (index, character) in text.enumerated() {
            let isGlyph = index == 0
            out.append(
                NSAttributedString(
                    string: String(character),
                    attributes: [
                        .font: NSFont.systemFont(
                            ofSize: isGlyph ? glyphSize : markSize,
                            weight: isGlyph ? .medium : .regular),
                        .foregroundColor: NSColor.labelColor,
                    ]))
        }
        return out
    }
}

/// Hold or tap for a dictation key, fed raw key-down and key-up events with
/// monotonic nanosecond timestamps. The rules match `crates/phonon-hotkey`.
///
/// Key-down always starts a recording, so a hold has no latency and release
/// stops it (push-to-talk). A release inside `tapMaxNs` is a tap, and a tap
/// latches the recording on (press to toggle): the recording that began at
/// key-down keeps going, and the next key-down stops it. There is no double-tap.
struct HoldKeyTapLatch {
    enum Action: Equatable {
        case start
        case stop
        case none
    }

    static let defaultTapMaxNs: UInt64 = 250_000_000

    let tapMaxNs: UInt64
    private(set) var latched = false
    private var downAt: UInt64?

    init(tapMaxNs: UInt64 = HoldKeyTapLatch.defaultTapMaxNs) {
        self.tapMaxNs = tapMaxNs
    }

    mutating func keyDown(at now: UInt64) -> Action {
        if latched {
            reset()
            return .stop
        }
        downAt = now
        return .start
    }

    mutating func keyUp(at now: UInt64) -> Action {
        // A key-up with no matching key-down follows a latched stop.
        guard let downAt else { return .none }
        self.downAt = nil
        if now >= downAt && now - downAt <= tapMaxNs {
            latched = true
            return .none
        }
        return .stop
    }

    mutating func reset() {
        latched = false
        downAt = nil
    }
}

/// When a latched dictation (one tap of fn or Right Option) stops on its own, so a stray tap
/// cannot leave the mic recording until the next press. A held key never stops this way (its
/// release does), and neither do Ctrl+Space or a meeting.
///
/// Speech is the capsule's `VoiceDetector` (the level meter against the room's floor), the
/// same signal the live-text dot answers to, but only once it has lasted `speechRunSeconds`
/// without a break: a keystroke or a click near the mic is one or two meter readings.
/// Timestamps are seconds on any monotonic clock; the caller supplies them.
struct LatchAutoStop {
    enum Decision: Equatable {
        case keepRecording
        /// Nothing was said since the start: drop it, with no paste and no card.
        case discard
        /// Speech, then a long silence: stop and commit as the next press would.
        case commit
    }

    // The two limits live here and nowhere else.
    /// No speech at all this long after the start. Someone who tapped on purpose starts
    /// talking within a second or two (the capsule and the start cue say it is listening);
    /// 8 s leaves room to gather a thought and still frees the mic soon after a stray tap.
    static let noSpeechSeconds: TimeInterval = 8
    /// Silence this long after speech. Someone dictating a long prompt stops to read or think
    /// for tens of seconds, and a commit pastes into whatever is focused, so this is a full
    /// minute; someone who has walked away still gets their words committed.
    static let silenceAfterSpeechSeconds: TimeInterval = 60
    /// Voice this long without a break counts as speech: about a syllable, and longer than
    /// a keystroke. Time, not a count of readings, because the meter's rate follows the
    /// device's buffer size.
    static let speechRunSeconds: TimeInterval = 0.15

    let startedAt: TimeInterval
    private(set) var lastSpeechAt: TimeInterval?
    private var voiceSince: TimeInterval?
    private var lastVoiceAt: TimeInterval?

    init(startedAt: TimeInterval) {
        self.startedAt = startedAt
    }

    var heardSpeech: Bool { lastSpeechAt != nil }

    /// One reading of the voice detector.
    mutating func note(voice: Bool, at now: TimeInterval) {
        guard voice else {
            voiceSince = nil
            return
        }
        let since = voiceSince ?? now
        voiceSince = since
        lastVoiceAt = now
        if now - since >= Self.speechRunSeconds { lastSpeechAt = now }
    }

    func decision(at now: TimeInterval) -> Decision {
        // Voice that has just begun and not yet lasted `speechRunSeconds` holds a limit
        // off, so talking right at the limit is not cut. Only while readings keep
        // arriving: a mic that stopped mid-voice must not hold it off for ever.
        if voiceSince != nil, let lastVoiceAt, now - lastVoiceAt < 1 {
            return .keepRecording
        }
        if let lastSpeechAt {
            return now - lastSpeechAt >= Self.silenceAfterSpeechSeconds ? .commit : .keepRecording
        }
        return now - startedAt >= Self.noSpeechSeconds ? .discard : .keepRecording
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
        load()?.microphonePriority ?? []
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
    /// Bluetooth headset microphones drop the headset to phone-call audio the
    /// moment any process opens them, so they are never chosen by inference.
    var isBluetooth: Bool = false
    /// CoreAudio's persistent device UID: stable across reconnects and
    /// reboots, unlike `id`. The new-microphone prompt keys on it.
    var uid: String = ""
    /// Aggregate and virtual inputs (call apps' voice-processing aggregates,
    /// loopback drivers) are never offered as a new microphone.
    var isVirtual: Bool = false

    var isBuiltIn: Bool {
        name.localizedCaseInsensitiveContains("MacBook")
            || name.localizedCaseInsensitiveContains("Built-in")
    }
}

/// Ranked names win. With nothing ranked, Phonon follows the system input the
/// way every other app does, except that a Bluetooth headset microphone is
/// skipped in favour of the built-in one: macOS makes a headset the default
/// input when it connects, and using its mic degrades the user's headphones.
enum MicrophonePriorityResolver {
    static func resolve(devices: [AudioInputDevice], priorities: [String]) -> AudioInputDevice? {
        for priority in priorities {
            if let match = devices.first(where: {
                $0.name.localizedCaseInsensitiveContains(priority)
            }) {
                return match
            }
        }
        if let system = devices.first(where: \.isDefault), !system.isBluetooth {
            return system
        }
        return devices.first(where: \.isBuiltIn)
            ?? devices.first(where: { !$0.isBluetooth })
            ?? devices.first
    }
}

enum CoreAudioInputDevices {
    static func resolve(priorities: [String]) -> AudioInputDevice? {
        MicrophonePriorityResolver.resolve(devices: all(), priorities: priorities)
    }

    static func inputNames() -> [String] {
        all().map(\.name).sorted { $0.localizedCaseInsensitiveCompare($1) == .orderedAscending }
    }

    static func all() -> [AudioInputDevice] {
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
            let transport = transportType(id)
            return AudioInputDevice(
                id: id, name: name, isDefault: id == defaultID,
                isBluetooth: transport == kAudioDeviceTransportTypeBluetooth
                    || transport == kAudioDeviceTransportTypeBluetoothLE,
                uid: deviceUID(id) ?? "",
                isVirtual: transport == kAudioDeviceTransportTypeAggregate
                    || transport == kAudioDeviceTransportTypeAutoAggregate
                    || transport == kAudioDeviceTransportTypeVirtual)
        }
    }

    private static func transportType(_ id: AudioDeviceID) -> UInt32? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyTransportType,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var transport: UInt32 = 0
        var size = UInt32(MemoryLayout<UInt32>.size)
        guard AudioObjectGetPropertyData(id, &address, 0, nil, &size, &transport) == noErr
        else { return nil }
        return transport
    }

    /// The caller owns the returned CFString (CoreAudio's rule for this property).
    private static func deviceUID(_ id: AudioDeviceID) -> String? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceUID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var uid: Unmanaged<CFString>?
        var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        guard AudioObjectGetPropertyData(id, &address, 0, nil, &size, &uid) == noErr, let uid
        else { return nil }
        return uid.takeRetainedValue() as String
    }

    /// Fires on the main queue whenever a device appears or disappears.
    static func observeDeviceListChanges(_ handler: @escaping () -> Void) {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        AudioObjectAddPropertyListenerBlock(
            AudioObjectID(kAudioObjectSystemObject), &address, DispatchQueue.main
        ) { _, _ in handler() }
    }

    /// Fires on the main queue whenever the system default input changes.
    static func observeDefaultInputChanges(_ handler: @escaping () -> Void) {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        AudioObjectAddPropertyListenerBlock(
            AudioObjectID(kAudioObjectSystemObject), &address, DispatchQueue.main
        ) { _, _ in handler() }
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
    /// One capture's OCR text and the windows it covered.
    struct Read {
        let keys: [String]
        let windowIDs: [UInt32]
        let text: String
    }

    /// On-screen windows, front to back, from the window server. Titles need
    /// the Screen Recording permission screen context already requires.
    static func onScreenWindows() -> [ScreenWindow] {
        guard
            let info = CGWindowListCopyWindowInfo(
                [.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID)
                as? [[String: Any]]
        else { return [] }
        var bundles = [pid_t: String?]()
        return info.compactMap { entry in
            guard let number = (entry[kCGWindowNumber as String] as? NSNumber)?.uint32Value,
                let pid = (entry[kCGWindowOwnerPID as String] as? NSNumber)?.int32Value,
                let bounds = entry[kCGWindowBounds as String] as? NSDictionary,
                let frame = CGRect(dictionaryRepresentation: bounds as CFDictionary)
            else { return nil }
            if bundles[pid] == nil {
                bundles[pid] = NSRunningApplication(processIdentifier: pid)?.bundleIdentifier
            }
            return ScreenWindow(
                id: number, pid: pid, bundleID: bundles[pid] ?? nil, frame: frame,
                layer: (entry[kCGWindowLayer as String] as? NSNumber)?.intValue ?? 0,
                alpha: (entry[kCGWindowAlpha as String] as? NSNumber)?.doubleValue ?? 1,
                title: entry[kCGWindowName as String] as? String)
        }
    }

    /// The window `pid` has in front, the one a switch just brought up.
    static func frontWindow(of pid: pid_t) -> ScreenWindow? {
        onScreenWindows().first { $0.pid == pid && $0.layer == 0 }
    }

    /// The focused browser window and its top-level page host, when the
    /// frontmost app is a browser and the focused element is in a page of
    /// one of its standard windows. The window id is the focused element's
    /// own window (`BrowserPage.focusedWindow`), never one picked separately
    /// from the window list, so a popup or a window switch between two reads
    /// cannot pair one window's host with another window. Blocking AX; call
    /// off the main thread.
    static func focusedBrowserPage(pid: pid_t?, isBrowser: Bool) -> (
        windowID: UInt32, host: String
    )? {
        guard isBrowser, let pid else { return nil }
        return BrowserPage.focusedWindow(pid: pid)
    }

    /// The reads to keep once the capture is done: all of them while the
    /// focused browser page is still the one the capture was allowed for;
    /// otherwise none that covered that browser window, whose tab may have
    /// changed under the capture.
    static func keptReads(
        _ reads: [Read], allowed: (windowID: UInt32, host: String)?,
        now: (windowID: UInt32, host: String)?
    ) -> [Read] {
        guard let allowed, !samePage(allowed, now) else { return reads }
        return reads.filter { !$0.windowIDs.contains(allowed.windowID) }
    }

    /// The page a switched-to window may be read for: the focused element's
    /// own window and host, from one walk, only if that window is the one
    /// in front.
    static func switchedPage(_ focus: BrowserPage.Focus, window: UInt32) -> (
        windowID: UInt32, host: String
    )? {
        guard let page = focus.page, page.windowID == window else { return nil }
        return page
    }

    /// Whether the focused browser page is still the one a capture was
    /// allowed for: same window, same host. Checked after the capture.
    static func samePage(
        _ before: (windowID: UInt32, host: String)?, _ after: (windowID: UInt32, host: String)?
    ) -> Bool {
        guard let before, let after else { return before == nil && after == nil }
        return before.windowID == after.windowID && before.host == after.host
    }

    /// Read what can be seen on each display: one capture per display that
    /// contains only that display's visible windows (`ScreenWindowSelector`),
    /// so a window buried under others, Phonon's own and private ones are
    /// never read.
    static func readVisible(excluded: @escaping (ScreenWindow) -> Bool) async -> [Read] {
        guard CGPreflightScreenCaptureAccess() || CGRequestScreenCaptureAccess() else {
            return []
        }
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(
                false, onScreenWindowsOnly: true)
            let displays = content.displays.map {
                ScreenDisplay(id: $0.displayID, frame: $0.frame)
            }
            let picked = ScreenWindowSelector.visibleWindows(
                windows: onScreenWindows(), displays: displays, excluded: excluded)
            var reads = [Read]()
            for display in content.displays {
                guard !Task.isCancelled else { return reads }
                let chosen = picked[display.displayID] ?? []
                let ids = Set(chosen.map(\.id))
                let windows = content.windows.filter { ids.contains($0.windowID) }
                guard !windows.isEmpty else { continue }
                let filter = SCContentFilter(display: display, including: windows)
                let configuration = SCStreamConfiguration()
                configuration.width = display.width
                configuration.height = display.height
                configuration.showsCursor = false
                let image = try await SCScreenshotManager.captureImage(
                    contentFilter: filter, configuration: configuration)
                guard !Task.isCancelled else { return reads }
                reads.append(
                    Read(
                        keys: chosen.map(\.captureKey), windowIDs: chosen.map(\.id),
                        text: recognize(image)))
            }
            return reads
        } catch {
            NSLog("phonon screen context: \(error.localizedDescription)")
            return []
        }
    }

    /// Read one window, for a window or tab the user switched to mid-dictation.
    static func readWindow(id: UInt32) async -> String? {
        guard CGPreflightScreenCaptureAccess() else { return nil }
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(
                false, onScreenWindowsOnly: true)
            guard let window = content.windows.first(where: { $0.windowID == id }) else {
                return nil
            }
            let configuration = SCStreamConfiguration()
            configuration.width = max(1, Int(window.frame.width))
            configuration.height = max(1, Int(window.frame.height))
            configuration.showsCursor = false
            let image = try await SCScreenshotManager.captureImage(
                contentFilter: SCContentFilter(desktopIndependentWindow: window),
                configuration: configuration)
            guard !Task.isCancelled else { return nil }
            return recognize(image)
        } catch {
            NSLog("phonon screen context window: \(error.localizedDescription)")
            return nil
        }
    }

    /// The final correction waits for the screen read, and the first accurate text
    /// recognition in a process loads Vision's models (cold after a reboot it took
    /// ~14 s and held the first dictation). Recognize one rendered word at launch
    /// so that load happens before the first dictation; no screen is captured.
    static func warmUp() {
        Task.detached(priority: .utility) {
            let width = 192
            let height = 48
            guard
                let context = CGContext(
                    data: nil, width: width, height: height, bitsPerComponent: 8,
                    bytesPerRow: 0, space: CGColorSpaceCreateDeviceGray(),
                    bitmapInfo: CGImageAlphaInfo.none.rawValue)
            else { return }
            context.setFillColor(gray: 1, alpha: 1)
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            let word = NSAttributedString(
                string: "Phonon",
                attributes: [.font: NSFont.systemFont(ofSize: 28), .foregroundColor: NSColor.black])
            context.textPosition = CGPoint(x: 12, y: 12)
            CTLineDraw(CTLineCreateWithAttributedString(word), context)
            guard let image = context.makeImage() else { return }
            _ = recognize(image)
        }
    }

    private static func recognize(_ image: CGImage) -> String {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        // Language correction "fixes" identifiers toward English words and
        // costs about 30% of the time; screen context exists to read exactly
        // those identifiers. Measured 2026-10-29 (synthetic Retina screens,
        // 316 dictionary terms, warm, M4 Max): on 96.9/96.1% of terms exact
        // in 202/218 ms (code/UI screen), off 96.9/98.7% in 156/135 ms.
        request.usesLanguageCorrection = false
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
    /// The stop button's action while a meeting records.
    var onStop: () -> Void = {}
    /// Renders one instant (offscreen tests); nil follows the clock.
    var fixedTime: Date?

    var body: some View {
        // Idle and hidden draw nothing that changes over time, so the clock stops there: an unpaused timeline redrew
        // the still capsule 30 times a second forever (about 5% of a core on an M4 Air, 2026-11-05).
        TimelineView(
            .animation(
                minimumInterval: 1.0 / 30.0,
                paused: (state.mode == .idle || state.mode == .hidden) && !state.meetingRecording)
        ) { context in
            let date = fixedTime ?? context.date
            let energy = visualEnergy(at: date)
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
                        .opacity(glow(energy)),
                    radius: 4,
                    y: 1
                )
                .frame(width: size.width, height: size.height)
                .overlay(alignment: dictating ? .leading : .center) {
                    if state.meetingRecording {
                        recordingMark(at: date)
                            .padding(.leading, dictating ? 4 : 0)
                    }
                }
                .frame(width: 80, height: 16, alignment: .bottom)
                .opacity(state.capsuleVisible ? 1 : 0)
                .animation(.easeOut(duration: 0.18), value: state.capsuleVisible)
                .animation(
                    .spring(response: 0.22, dampingFraction: 0.86),
                    value: size
                )
        }
    }

    /// Dictating while a meeting records: the capsule keeps its energy-driven
    /// size and warmth, and the red light moves to its dark leading end.
    private var dictating: Bool { state.mode == .listening || state.mode == .working }

    /// The red light, or on hover the stop button over it.
    private func recordingMark(at date: Date) -> some View {
        let light = RecordLight.opacity(at: date.timeIntervalSinceReferenceDate)
        let shown = state.stopButtonShown
        return ZStack {
            Circle()
                .fill(RecordLight.color)
                .frame(width: RecordLight.diameter, height: RecordLight.diameter)
                .shadow(color: RecordLight.color, radius: RecordLight.glowRadius(opacity: light))
                .opacity(shown ? 0 : light)
                .animation(.easeOut(duration: 0.12), value: shown)
            Button(action: onStop) { CapsuleStopButton() }
                .buttonStyle(.plain)
                .scaleEffect(shown ? 1 : 0.5)
                .opacity(shown ? 1 : 0)
                .allowsHitTesting(shown)
                .animation(.spring(response: 0.16, dampingFraction: 0.62), value: shown)
        }
        .frame(width: RecordLight.diameter, height: RecordLight.diameter)
    }

    private func glow(_ energy: Double) -> Double {
        if state.mode == .idle { return state.meetingRecording ? 0.18 : 0 }
        return 0.22 + energy * 0.12
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
            // Recording a meeting: a faint warm glow at the idle size (mock).
            return state.meetingRecording ? 0.35 : 0
        case .listening, .working:
            return 0.50 + energy * 0.38
        }
    }

    private func shellSize(energy: Double) -> CGSize {
        switch state.mode {
        case .hidden, .idle:
            // Recording a meeting: slightly larger, so the light fits inside.
            return state.meetingRecording
                ? CGSize(width: 46, height: 12) : CGSize(width: 40, height: 8)
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
        // Clicks pass through to the app below, except while a meeting
        // records (the stop button). Never key or main either way.
        ignoresMouseEvents = true
        becomesKeyOnlyIfNeeded = true
    }

    /// The screen under the pointer, its bottom raised above the Dock. The Dock's window
    /// level is above `.floating`, so a panel placed in the Dock's strip draws behind it.
    static func screenFrame() -> NSRect {
        let screen =
            NSScreen.screens.first { NSMouseInRect(NSEvent.mouseLocation, $0.frame, false) }
            ?? NSScreen.main
        guard let screen else { return .zero }
        return PanelGeometry.clearOfDock(frame: screen.frame, visibleFrame: screen.visibleFrame)
    }

    static func frame(size: NSSize) -> NSRect {
        PanelGeometry.restingFrame(screen: screenFrame(), size: size)
    }

    static func hiddenFrame(size: NSSize) -> NSRect {
        PanelGeometry.hiddenFrame(screen: screenFrame(), size: size)
    }
}

/// The two recording cues. A rising sweep opens a pass, a falling one closes it.
///
/// Both players are decoded once at launch. Building them on the keypress would
/// put file I/O on the path the capsule is trying to open in a few milliseconds.
final class CueSounds {
    private let start: AVAudioPlayer?
    private let stop: AVAudioPlayer?

    init() {
        start = CueSounds.load("record_start")
        stop = CueSounds.load("record_stop")
    }

    func playStart() { CueSounds.restart(start) }
    func playStop() { CueSounds.restart(stop) }

    private static func load(_ name: String) -> AVAudioPlayer? {
        guard let resources = Bundle.main.resourcePath else { return nil }
        let url = URL(fileURLWithPath: resources)
            .appendingPathComponent("assets")
            .appendingPathComponent("\(name).wav")
        guard let player = try? AVAudioPlayer(contentsOf: url) else { return nil }
        player.volume = 0.55
        player.prepareToPlay()
        return player
    }

    private static func restart(_ player: AVAudioPlayer?) {
        guard let player else { return }
        // A held key can retrigger before the previous cue finishes.
        player.currentTime = 0
        player.play()
    }
}

enum PanelGeometry {
    static let bottomInset: CGFloat = 14

    /// `frame` with its bottom edge moved up to `visibleFrame`'s, so a bottom Dock is
    /// left clear. The sides stay the full screen's: a side Dock does not reach the
    /// bottom center, and the panels stay centered on the screen. With the Dock hidden
    /// the two bottoms (nearly) match and nothing moves.
    static func clearOfDock(frame: NSRect, visibleFrame: NSRect) -> NSRect {
        let bottom = min(max(visibleFrame.minY, frame.minY), frame.maxY)
        return NSRect(
            x: frame.minX, y: bottom, width: frame.width, height: frame.maxY - bottom)
    }

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

/// The engine's stderr (and, labeled, its sidecars') appended to
/// ~/Library/Logs/Phonon/engine.log, rotated to engine.log.1 past 5 MB at each engine start,
/// so a helper that dies leaves its last words behind.
enum EngineLog {
    static let maxBytes = 5_000_000

    static var defaultURL: URL? {
        FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask).first?
            .appendingPathComponent("Logs/Phonon/engine.log")
    }

    static func handle(at url: URL?) -> FileHandle? {
        guard let url else { return nil }
        let manager = FileManager.default
        try? manager.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        if let size = (try? manager.attributesOfItem(atPath: url.path))?[.size] as? Int,
            size > maxBytes
        {
            let previous = URL(fileURLWithPath: url.path + ".1")
            try? manager.removeItem(at: previous)
            try? manager.moveItem(at: url, to: previous)
        }
        if !manager.fileExists(atPath: url.path) {
            manager.createFile(atPath: url.path, contents: nil)
        }
        guard let handle = try? FileHandle(forWritingTo: url) else { return nil }
        _ = try? handle.seekToEnd()
        let stamp = ISO8601DateFormatter().string(from: Date())
        try? handle.write(contentsOf: Data("=== engine started \(stamp)\n".utf8))
        return handle
    }
}

final class EngineClient {
    private var process: Process?
    private var stdin: FileHandle?
    private var readSource: DispatchSourceRead?
    private var buffer = Data()
    private var shuttingDown = false
    /// Bumped at every start, so a late exit notice from a replaced engine is ignored.
    private var generation = 0
    private var exitReported = false
    var onEvent: (([String: Any]) -> Void)?
    var onExit: (() -> Void)?
    /// The engine to run and its arguments (tests swap in a stand-in; nil is the app's own).
    var executable: String?
    var arguments = ["engine"]
    var logURL: URL? = EngineLog.defaultURL
    var processIdentifier: Int32? { process?.processIdentifier }

    func start() throws {
        readSource?.cancel()
        readSource = nil
        try? stdin?.close()
        stdin = nil
        process = nil
        buffer.removeAll(keepingCapacity: true)
        shuttingDown = false
        exitReported = false
        generation += 1
        let started = generation
        let phonon = executable ?? resolvePhonon()
        let p = Process()
        p.executableURL = URL(fileURLWithPath: phonon)
        p.arguments = arguments
        var environment = ProcessInfo.processInfo.environment
        if let resources = Bundle.main.resourcePath {
            environment["PHONON_ROOT"] = resources
        }
        p.environment = environment
        let inPipe = Pipe()
        let outPipe = Pipe()
        p.standardInput = inPipe
        p.standardOutput = outPipe
        p.standardError = EngineLog.handle(at: logURL) ?? FileHandle.nullDevice
        p.terminationHandler = { [weak self] _ in
            DispatchQueue.main.async {
                guard let self, self.generation == started else { return }
                self.reportExit()
            }
        }
        try p.run()
        process = p
        stdin = inPipe.fileHandleForWriting

        let fh = outPipe.fileHandleForReading
        let source = DispatchSource.makeReadSource(fileDescriptor: fh.fileDescriptor, queue: .main)
        source.setEventHandler { [weak self, weak source] in
            let chunk = fh.availableData
            guard !chunk.isEmpty else {
                // End of file: the engine closed its output. A readable source at EOF fires
                // without end, so stop it; the exit itself arrives via the termination handler.
                source?.cancel()
                return
            }
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
        if let bundled = Bundle.main.builtInPlugInsURL?
            .deletingLastPathComponent()
            .appendingPathComponent("Helpers/phonon").path,
            FileManager.default.isExecutableFile(atPath: bundled)
        {
            return bundled
        }
        for c in [
            NSString(string: "~/.local/bin/phonon").expandingTildeInPath,
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
        guard !shuttingDown, let stdin,
            let data = try? JSONSerialization.data(withJSONObject: obj),
            var line = String(data: data, encoding: .utf8)
        else { return }
        line += "\n"
        guard let d = line.data(using: .utf8) else { return }
        do {
            try stdin.write(contentsOf: d)
        } catch {
            // The engine closed its end (it exited, or is exiting). SIGPIPE is ignored for
            // the whole app (PhononBarMain), so this is an error, not the app's death:
            // stop writing and restart the engine.
            engineLost("write failed: \(error.localizedDescription)")
        }
    }

    private func engineLost(_ reason: String) {
        guard !shuttingDown, stdin != nil else { return }
        NSLog("phonon engine lost: \(reason)")
        try? stdin?.close()
        stdin = nil
        if let process, process.isRunning {
            process.terminate()  // its termination handler reports the exit
        } else {
            reportExit()
        }
    }

    private func reportExit() {
        guard !shuttingDown, !exitReported else { return }
        exitReported = true
        try? stdin?.close()
        stdin = nil
        onExit?()
    }

    /// Stop writing first, then let the engine go: one shutdown line, end of input, and
    /// terminate. Safe to call twice (quit and applicationWillTerminate both do).
    func shutdown() {
        guard !shuttingDown else { return }
        let input = stdin
        shuttingDown = true
        onExit = nil
        stdin = nil
        if let input {
            try? input.write(contentsOf: Data("{\"cmd\":\"shutdown\"}\n".utf8))
            try? input.close()
        }
        readSource?.cancel()
        readSource = nil
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
    private var tapFormat: AVAudioFormat?
    private var meter = 0.0
    private var meterFrame = 0
    /// This dictation's loudest buffer and how many frames it has captured.
    private var peakRms = 0.0
    private var capturedFrames = 0
    private var streamSamples: [Float] = []
    private var captureEnabled = false
    private var configuredDeviceID: AudioDeviceID?
    private(set) var wavPath: String?
    private(set) var selectedDeviceName = "Resolving…"
    var onAmplitude: ((Double) -> Void)?
    var onStreamPCM16: ((Data) -> Void)?
    private var primed = false
    private var pendingReselect: DispatchWorkItem?
    private var reselectRetries = 0
    /// Device changes arrive in bursts (a call app switching a Bluetooth
    /// headset fires a default-input change and several engine rebuilds), and
    /// the input format is not final until the burst ends.
    static let reselectSettle: TimeInterval = 0.4
    static let reselectRetryLimit = 3

    init() {
        // AVAudioEngine rebuilds its graph when the default input changes or a
        // device (dis)appears, and the input unit can come back on the new
        // system default instead of the device Phonon selected. Re-resolve.
        NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main
        ) { [weak self] _ in self?.scheduleReselect(reason: "engine configuration change") }
        CoreAudioInputDevices.observeDefaultInputChanges { [weak self] in
            self?.scheduleReselect(reason: "system default input change")
        }
    }

    /// Re-resolve once the burst of device notifications has settled.
    private func scheduleReselect(reason: String, delay: TimeInterval = MicRecorder.reselectSettle) {
        pendingReselect?.cancel()
        let work = DispatchWorkItem { [weak self] in self?.reselectDevice(reason: reason) }
        pendingReselect = work
        DispatchQueue.main.asyncAfter(deadline: .now() + delay, execute: work)
    }

    private func reselectDevice(reason: String) {
        NSLog("phonon mic: reselect (\(reason))")
        pendingReselect = nil
        forgetDevice()
        guard primed else { return }
        do {
            try ensureHardwareRunning()
            reselectRetries = 0
        } catch {
            NSLog("mic reselect: \(error.localizedDescription)")
            // The hardware may still be switching formats; try again shortly.
            guard reselectRetries < Self.reselectRetryLimit else {
                reselectRetries = 0
                return
            }
            reselectRetries += 1
            scheduleReselect(reason: "retry after \(reason)", delay: 1.0)
        }
    }

    /// Install the capture tap in the hardware's current format. A format that
    /// changed under us raises inside AVFAudio; that becomes a thrown error
    /// instead of an abort, and the caller retries once the device settles.
    private func installCaptureTap(on input: AVAudioInputNode, format: AVAudioFormat) throws {
        var raised: NSError?
        let installed = PhononCatchException({
            input.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
                self?.ingest(buffer)
            }
        }, &raised)
        guard installed else {
            tapFormat = nil
            throw raised
                ?? NSError(
                    domain: "PhononBar", code: 5,
                    userInfo: [NSLocalizedDescriptionKey: "Could not install the microphone tap"])
        }
        hwSampleRate = format.sampleRate
        tapFormat = format
    }

    /// Drop the selected device so the next `ensureHardwareRunning` picks again.
    private func forgetDevice() {
        if engine.isRunning {
            engine.stop()
        }
        if configuredDeviceID != nil {
            engine.inputNode.removeTap(onBus: 0)
        }
        configuredDeviceID = nil
        tapFormat = nil
    }

    /// The device the input unit is actually on, which can differ from the one
    /// Phonon set after a graph rebuild.
    private func liveDeviceID() -> AudioDeviceID? {
        guard let audioUnit = engine.inputNode.audioUnit else { return nil }
        var id = AudioDeviceID(kAudioObjectUnknown)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        guard
            AudioUnitGetProperty(
                audioUnit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0,
                &id, &size) == noErr
        else { return nil }
        return id
    }

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
        peakRms = 0
        capturedFrames = 0
        streamSamples.removeAll(keepingCapacity: true)
        captureEnabled = true
        lock.unlock()
    }

    /// Below this RMS (-90 dBFS) the input is digital silence: a muted, disconnected or
    /// blocked mic. A quiet room on a working mic sits near -60 dBFS or above.
    static let silentRms = 3e-5

    /// Whether this dictation has captured at least `seconds` of audio and all of it was
    /// digital silence.
    func capturedOnlySilence(atLeast seconds: Double) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return Double(capturedFrames) >= seconds * hwSampleRate && peakRms < Self.silentRms
    }

    /// Whether this dictation has captured any audio at all.
    var capturedAnything: Bool {
        lock.lock()
        defer { lock.unlock() }
        return capturedFrames > 0
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
        if let configured = configuredDeviceID, let live = liveDeviceID(), live != configured {
            NSLog("phonon mic: input unit drifted off the selected device, reselecting")
            forgetDevice()
        }
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
            do {
                try installCaptureTap(on: input, format: inFmt)
            } catch {
                // Leave nothing half-configured: the next call starts over.
                configuredDeviceID = nil
                throw error
            }
        } else if configuredDeviceID != nil {
            reinstallTapIfHardwareFormatChanged()
        }
        if !engine.isRunning {
            engine.prepare()
            do {
                try engine.start()
            } catch {
                guard reinstallTapIfHardwareFormatChanged() else { throw error }
                engine.prepare()
                try engine.start()
            }
        }
    }

    /// The configured device can renegotiate its format while idle, e.g. when a
    /// Bluetooth output on another clock connects. A tap installed at the old
    /// rate then fails every `engine.start()` with kAudioUnitErr_FormatNotSupported.
    @discardableResult
    private func reinstallTapIfHardwareFormatChanged() -> Bool {
        let input = engine.inputNode
        let inFmt = input.inputFormat(forBus: 0)
        guard inFmt.sampleRate > 0, inFmt.channelCount > 0, inFmt != tapFormat else {
            return false
        }
        if engine.isRunning {
            engine.stop()
        }
        input.removeTap(onBus: 0)
        do {
            try installCaptureTap(on: input, format: inFmt)
        } catch {
            NSLog("mic tap reinstall: \(error.localizedDescription)")
            return false
        }
        return true
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
        peakRms = max(peakRms, rms)
        capturedFrames += frames
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
        // 20 minutes (230 MB at 48 kHz, held only while recording). The old 120 s kept just
        // the tail of longer dictations, so their corpus audio no longer matched the
        // transcript (26 of 1,203 entries by 2026-11-04).
        let maxSamples = Int(hwSampleRate * 1200)
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

    /// The streamed audio not yet sent (under one 0.4 s chunk), as 16 kHz PCM16, so the stream
    /// ends where the recording ends.
    func drainStreamPCM16() -> Data? {
        lock.lock()
        let pending = streamSamples
        let rate = hwSampleRate
        streamSamples.removeAll(keepingCapacity: true)
        lock.unlock()
        guard !pending.isEmpty else { return nil }
        return Self.pcm16Data(Self.resample(pending, from: rate, to: 16_000))
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
        // one preallocated buffer filled in place: the per-sample append was 21 ms at 117 s of audio (1.3 ms this way)
        var pcm = Data(count: samples.count * 2)
        pcm.withUnsafeMutableBytes { raw in
            let out = raw.bindMemory(to: Int16.self)
            for (i, s) in samples.enumerated() {
                let clipped = max(-1.0, min(1.0, s))
                out[i] = Int16((clipped * Float(Int16.max)).rounded()).littleEndian
            }
        }
        return pcm
    }
}

// MARK: - Typer

enum Typer {
    /// The target app reads the pasteboard some time after ⌘V; the old contents come back
    /// after this, and only if nothing else was copied meanwhile.
    static let restoreDelay: TimeInterval = 0.6

    /// The text as typed: one trailing space, so the next dictation flows on without a check
    /// of timing or of what is in the field (owner, 2026-11-05). Not after a space or newline.
    static func spaced(_ text: String) -> String {
        guard let last = text.last, !last.isWhitespace else { return text }
        return text + " "
    }

    @MainActor
    static func emit(_ text: String, target: NSRunningApplication?, restoreClipboard: Bool = false)
        throws
    {
        guard !text.isEmpty else { return }
        guard AXIsProcessTrusted() else {
            throw NSError(
                domain: "PhononBar", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Grant Accessibility to PhononBar"])
        }

        let pasteboard = NSPasteboard.general
        let previous: [NSPasteboardItem] =
            restoreClipboard
            ? (pasteboard.pasteboardItems ?? []).map { item in
                let copy = NSPasteboardItem()
                for type in item.types {
                    if let data = item.data(forType: type) { copy.setData(data, forType: type) }
                }
                return copy
            } : []
        pasteboard.clearContents()
        guard pasteboard.setString(text, forType: .string) else {
            throw NSError(
                domain: "PhononBar", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Could not copy dictation to the pasteboard"])
        }

        let ownPID = ProcessInfo.processInfo.processIdentifier
        if NSWorkspace.shared.frontmostApplication?.processIdentifier == ownPID,
            let target, !target.isTerminated
        {
            target.activate()
            RunLoop.current.run(until: Date().addingTimeInterval(0.05))
        }

        guard let (down, up) = makePasteEvents() else {
            throw NSError(
                domain: "PhononBar", code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Could not create paste keyboard events"])
        }
        down.post(tap: .cghidEventTap)
        up.post(tap: .cghidEventTap)
        guard restoreClipboard, !previous.isEmpty else { return }
        let ours = pasteboard.changeCount
        DispatchQueue.main.asyncAfter(deadline: .now() + restoreDelay) {
            guard pasteboard.changeCount == ours else { return }
            pasteboard.clearContents()
            pasteboard.writeObjects(previous)
        }
    }

    static func makePasteEvents() -> (CGEvent, CGEvent)? {
        let source = CGEventSource(stateID: .hidSystemState)
        guard
            let down = CGEvent(
                keyboardEventSource: source, virtualKey: CGKeyCode(kVK_ANSI_V), keyDown: true),
            let up = CGEvent(
                keyboardEventSource: source, virtualKey: CGKeyCode(kVK_ANSI_V), keyDown: false)
        else { return nil }
        down.flags = .maskCommand
        up.flags = .maskCommand
        return (down, up)
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
    /// Final transcript to correction request: the time spent waiting for
    /// screen context (bounded by `ScreenContextWait`).
    var screenWaitMs: Double? = nil
    /// Since launch: times macOS disabled the event tap, and hold-key presses that
    /// arrived with no release before them (a lost release; issue #21).
    var eventTapDisables: Int? = nil
    var hotkeyRepresses: Int? = nil

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
        case screenWaitMs = "screen_wait_ms"
        case eventTapDisables = "event_tap_disables"
        case hotkeyRepresses = "hotkey_represses"
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
            llmTokensPerSecond: llmTokensPerSecond,
            screenWaitMs: milliseconds(from: asrResultNs, to: polishSubmittedNs)
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
    /// Live text: the corrected text so far above the capsule (engine `draft` results).
    private let liveText = LiveTextState()
    private var liveTextPanel: LiveTextPanel?
    /// A dictation with no text box to go to shows here with a copy button; so do the
    /// mic warnings (silent mic while held, nothing heard after).
    private let notPasted = NotPastedController()
    private var recordingStartedAt: Date?
    /// Tells speech from room noise on the level meter; live text changes only on speech.
    private var voiceDetector = VoiceDetector()
    /// How long the key was held for the dictation being finished.
    private var heldSeconds: TimeInterval = 0
    private var micSilenceCheck: DispatchWorkItem?
    /// Digital silence for this long while held shows the mic warning.
    static let micSilenceSeconds: TimeInterval = 1.5
    /// An empty result from a hold at least this long says "didn't hear anything"; a
    /// shorter one was most likely an accidental tap.
    static let notHeardMinSeconds: TimeInterval = 1.0

    private func checkMicSilence() {
        guard isRecording, let started = recordingStartedAt,
            Date().timeIntervalSince(started) >= Self.micSilenceSeconds
        else { return }
        let silent =
            !recorder.capturedAnything
            || recorder.capturedOnlySilence(atLeast: Self.micSilenceSeconds)
        if silent {
            if notPasted.current == nil {
                notPasted.show(.micSilent(mic: recorder.selectedDeviceName), stays: true)
            }
        } else if case .micSilent = notPasted.current {
            notPasted.dismiss()
        }
    }
    private let cues = CueSounds()
    /// Settings `background_audio`: lowers or mutes other audio while the key is held.
    private let ducker = AudioDucker()
    private var modelStartupWindow: NSPanel?
    private var mainWindow: NSWindow?
    private let engine = EngineClient()
    private let recorder = MicRecorder()
    private var statusItem: NSStatusItem?
    /// The menu-bar title before the permission warning: "ϕ", "ϕ ●" (meeting), "ϕ…".
    private var statusBase = "ϕ"
    /// Top of the menu-bar menu while a permission keeps dictation from working.
    private var attentionMenuItem: NSMenuItem?
    /// Re-checks permissions every few seconds while one is missing, so granting it
    /// in System Settings clears the warning without a click in Phonon.
    private var permissionWatch: Timer?
    private var lastPermissionState: [PermissionAttention] = []
    private var requestedInputMonitoring = false
    private var streamingMenuItem: NSMenuItem?
    private var showInDockMenuItem: NSMenuItem?
    private var microphoneMenuItem: NSMenuItem?
    private var pass = 0
    private var activeId: String?
    private var rawText = ""
    private var isRecording = false
    private var optionEdges = HoldKeyEdges()
    private var fnEdges = HoldKeyEdges()
    private var fnTapLatch = HoldKeyTapLatch()
    private var optionTapLatch = HoldKeyTapLatch()
    /// Hears speech from the start of every dictation; acted on only once a tap latched it.
    private var latchAutoStop = LatchAutoStop(startedAt: 0)
    private var latchAutoStopTimer: Timer?
    /// Times macOS disabled the event tap (a lost key release is likely each time).
    private var eventTapDisables = 0
    private var hotkeyRepresses = 0
    private var eventTap: CFMachPort?
    private var eventTapSource: CFRunLoopSource?
    private var toggleHotKey: EventHotKeyRef?
    private var toggleHandler: EventHandlerRef?
    private var enginesReady = false
    /// Screen text last sent to the engine for the open stream (streaming refine).
    private var streamScreenTextSent = ""
    /// The engine refines while the speaker talks (its ready event says so).
    private var engineStreamRefine = false
    private var hideWork: DispatchWorkItem?
    /// First dict often lands before models are warm — hold until ready.
    private var pendingTranscribe: (path: String, id: String)?
    private var submittedTranscribe: (path: String, id: String)?
    private var processing = false
    private var streamActive = false
    private var latestAcousticPreview = ""
    /// Live text is on for the open dictation, so partials show as its uncorrected tail.
    private var liveTextPass = false
    private var liveTrace = LiveTextTrace()
    private var e2eTrace: E2ETrace?
    private var screenContextReady = true
    private var pendingFinalPolish: (text: String, id: String)?
    private var screenContextTask: Task<Void, Never>?
    /// What screen context read this pass: the visible windows at key-down
    /// plus windows and tabs switched to while speaking.
    private var screenCollection = ScreenContextCollection()
    private var screenInitialDone = true
    /// A browser dictation's screen read waits for the tab lookup.
    private var screenStartAwaitsTarget = false
    private var screenReads = ScreenReadLedger()
    private var screenSwitchTasks: [Task<Void, Never>] = []
    private var screenSwitchTimer: Timer?
    private var screenWaitWork: DispatchWorkItem?
    private var engineRestartWork: DispatchWorkItem?
    private var terminating = false
    private var activeWavPath: String?
    private var insertionTarget: NSRunningApplication?
    /// Where this pass is going (app, browser site), and whether it is on the
    /// private list. Read off the main thread at key-down.
    private var dictationTarget: DictationTarget?
    private var dictationTargetTask: Task<Void, Never>?
    /// The text around the cursor at key-down (setting `textbox_context`).
    private var dictationTextbox: TextboxContext?
    private lazy var competitors = CompetitorCoordinator(store: appStore)
    /// True while edit learning watches a field; read on the event-tap thread.
    nonisolated private let editKeyWatch = OSAllocatedUnfairLock(initialState: false)
    private lazy var editLearning = EditLearningController(
        store: appStore, keyWatch: editKeyWatch,
        send: { [weak self] command in self?.engine.send(command) })
    private lazy var meetings = MeetingController(store: appStore)
    private let meetingTranscriber = MeetingTranscriber()
    private lazy var meetingCleaner = MeetingCleaner.live(
        support: PhononDataPaths.supportDirectory())
    private lazy var newMicrophones = NewMicrophoneController(store: appStore)
    private var meetingMenuItem: NSMenuItem?
    private lazy var updates = UpdateController(store: appStore)
    private var updateMenuItem: NSMenuItem?

    func applicationDidFinishLaunching(_ notification: Notification) {
        ducker.recoverAfterCrash()
        applyDockVisibility()
        setupMainMenu()
        appStore.onDictionaryChanged = { [weak self] in
            self?.engine.send(["cmd": "reload_dictionary"])
        }
        appStore.onMicrophonePermissionGranted = { [weak self] in
            self?.warmMicrophoneIfAuthorized()
        }
        appStore.onPermissionsRefresh = { [weak self] in
            self?.refreshPermissionDependentServices()
        }
        state.streamingPreviewEnabled = appStore.settings.streaming
        state.showWhenIdle = appStore.settings.showIdleCapsule
        offerBackupRestoreIfNeeded()
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
                showMainWindow()
                snapshotMainWindowIfRequested()
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
            }
            snapshotPanelIfRequested()
            return
        }

        promptPermissions()
        showMainWindow()
        startEngine()
        installToggleHandler()
        syncToggleHotKey()
        installHotkey()
        refreshStatusTitle()
        showIdle()

        warmMicrophoneIfAuthorized()
        if appStore.settings.screenContext { ScreenContextCapture.warmUp() }
        // The https handler list is looked up once per process (7-11 ms,
        // 3 runs); do it now, off the main thread, not at the first key-down.
        Task.detached(priority: .utility) { _ = BrowserCatalog.httpsHandlers() }
        meetings.onRecordingChanged = { [weak self] app in self?.meetingRecordingChanged(app) }
        meetings.onFlowChanged = { [weak self] flow in self?.meetingFlowChanged(flow) }
        meetings.refresh()
        // Meeting transcripts: after each recording stops, from the Notes button, and at
        // launch for any a quit interrupted. Spans go through the engine's meeting queue.
        meetingTranscriber.send = { [weak self] command in self?.engine.send(command) }
        meetingTranscriber.onJobsChanged = { [weak self] jobs in
            self?.appStore.meetingTranscription = jobs
        }
        meetingTranscriber.onFinished = { [weak self] id in
            guard let self else { return }
            self.appStore.loadMeetings()
            // then cleaned in the background, on the CPU, dictation first
            self.meetingCleaner.enqueue(
                directory: self.appStore.meetingsDirectory.appendingPathComponent(id, isDirectory: true))
        }
        // From the Notes button and from `meetingRecordingFinished` (a recording that stopped).
        appStore.onTranscribeMeeting = { [weak self] directory in
            self?.meetingTranscriber.enqueue(directory: directory)
        }
        meetingTranscriber.resumePending(in: appStore.meetingsDirectory)
        appStore.meetingCleanupAvailable = meetingCleaner.isAvailable
        meetingCleaner.onJobsChanged = { [weak self] jobs in self?.appStore.meetingCleanup = jobs }
        meetingCleaner.onFinished = { [weak self] _ in self?.appStore.loadMeetings() }
        appStore.onCleanMeeting = { [weak self] directory in
            self?.meetingCleaner.enqueue(directory: directory)
        }
        meetingCleaner.resumePending(in: appStore.meetingsDirectory)

        // A competitor quit with consent leaves Phonon's tap in place; re-install
        // it so it sits at the head of the tap chain (SPEC step 5).
        competitors.onQuitFinished = { [weak self] in
            guard let self else { return }
            self.removeEventTap()
            self.installHotkey()
            NSLog("phonon competitors: hotkey tap re-installed at head")
        }
        competitors.subscribeToLaunches()
        competitors.check(trigger: "launch")

        // Ask once per new microphone (SPEC "Microphone"); the first run of
        // this only records what is connected.
        newMicrophones.isBusy = { [weak self] in
            guard let self else { return false }
            return self.isRecording || self.processing
        }
        newMicrophones.onRankingChanged = { [weak self] in self?.applySettings() }
        newMicrophones.start()

        updates.isBusy = { [weak self] in
            guard let self else { return false }
            return self.isRecording || self.processing
        }
        updates.showMessage = { [weak self] title, detail, action, perform in
            self?.notPasted.showMessage(
                title: title, detail: detail, action: action, perform: perform)
        }
        updates.onAvailableChanged = { [weak self] update in
            self?.updateMenuItem?.isHidden = update == nil
            self?.updateMenuItem?.title = update.map { "Update to Phonon \($0.version)…" } ?? ""
        }
        appStore.onShowUpdate = { [weak self] in self?.updates.showAvailable() }
        appStore.onCheckForUpdates = { [weak self] in self?.updates.check(manual: true) }
        updates.start()
    }

    /// The dictation-start scan defers its prompt; show it once the pipeline is
    /// idle so the alert never lands while text is still being inserted.
    private func presentCompetitorPromptWhenIdle(attempt: Int = 0) {
        guard competitors.hasPendingPrompt else { return }
        if isRecording || processing {
            guard attempt < 40 else { return }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
                self?.presentCompetitorPromptWhenIdle(attempt: attempt + 1)
            }
            return
        }
        competitors.presentPendingPrompt()
    }

    @objc private func quitCompetingApps() {
        if !competitors.checkManually() {
            let alert = NSAlert()
            alert.messageText = "No other dictation app is running"
            alert.informativeText =
                "Phonon checks for "
                + CompetingAppTable.apps.map(\.name).joined(separator: ", ") + "."
            alert.addButton(withTitle: "OK")
            NSApp.activate(ignoringOtherApps: true)
            alert.runModal()
        }
    }

    private func warmMicrophoneIfAuthorized() {
        guard AVCaptureDevice.authorizationStatus(for: .audio) == .authorized else { return }
        if appStore.settings.instantMic { recorder.prewarm() }
        updateMicrophoneStatus()
    }

    private func setupPanel() {
        let panel = PillPanel(contentRect: PillPanelLayout.frame(screen: PillPanel.screenFrame()))
        let host = PillHostingView(
            rootView: PillPanelContent(
                state: state, onStop: { [weak self] in self?.meetings.stopFromUser() }))
        host.frame = NSRect(origin: .zero, size: PillPanelLayout.size)
        host.onHover = { [weak self] inside in self?.meetings.hover(inside) }
        panel.contentView = host
        self.panel = panel
        let livePanel = LiveTextPanel()
        let liveHost = NSHostingController(rootView: LiveTextView(state: liveText))
        liveHost.view.frame = NSRect(origin: .zero, size: LiveTextPanel.size)
        livePanel.contentView = liveHost.view
        liveTextPanel = livePanel
    }

    /// A draft of the open dictation: shown through the same final check as the paste. It
    /// covers the transcript the engine committed for it (`source`); later partial words
    /// show after it as the uncorrected tail.
    private func showLiveText(_ text: String, source: String, complete: Bool) {
        let shown = PreviewRevisionPolicy.safeFinalText(source: source, candidate: text)
        let sourceWords = LiveTextPolicy.words(source)
        let covered = sourceWords.count
        guard liveText.draftMatches(source: sourceWords) else {
            liveTrace.record(
                "draft_stale", input: text, applied: false, covered: covered,
                card: liveText.shownWords)
            return
        }
        guard !shown.isEmpty else {
            liveTrace.record(
                "draft_rejected", input: text, applied: false, covered: covered,
                card: liveText.shownWords)
            return
        }
        let applied = liveText.applyDraft(
            shown, coveredWords: covered, complete: complete, source: sourceWords)
        liveTrace.record(
            "draft", input: shown, applied: applied, covered: covered, card: liveText.shownWords)
        presentLiveText()
    }

    /// The latest speech-recognition partial: the words no draft covers yet show as the tail.
    private func showLiveTextTail(_ transcript: String) {
        guard liveTextPass else { return }
        let applied = liveText.setTranscript(transcript)
        liveTrace.record("partial", input: transcript, applied: applied, card: liveText.shownWords)
        if !liveText.isEmpty { presentLiveText() }
    }

    private func presentLiveText() {
        guard let liveTextPanel else { return }
        let target = LiveTextPanel.frame(screen: PillPanel.screenFrame())
        if liveTextPanel.frame != target {
            liveTextPanel.setFrame(target, display: false)
        }
        if !liveTextPanel.isVisible {
            liveTextPanel.orderFrontRegardless()  // never makeKey: it must not take focus
        }
    }

    private func hideLiveText() {
        liveTrace.finish()
        liveText.reset()
        liveTextPanel?.orderOut(nil)
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
        window.backgroundColor = EmberTheme.nsBackground
        window.appearance = NSAppearance(named: .darkAqua)
        window.hasShadow = true
        window.contentViewController = host
        window.center()
        modelStartupWindow = window
    }

    private func setupMainWindow() {
        let initialPage = ProcessInfo.processInfo.environment["PHONON_MAIN_PAGE"]
            .flatMap(NativeAppPage.init(rawValue:)) ?? .home
        let root = PhononWindowRoot(
            startup: modelStartupState,
            main: PhononMainView(
                store: appStore,
                initialPage: initialPage,
                onSettingsChanged: { [weak self] in self?.applySettings() },
                onShowModelStatus: { [weak self] in self?.showModelStatus() }
            ))
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 980, height: 680),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.title = "Phonon"
        // A programmatically created NSWindow releases itself when closed, which
        // ARC does not account for, so `mainWindow` would keep pointing at freed
        // memory and the next reopen would message it and crash.
        window.isReleasedWhenClosed = false
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
        item.button?.attributedTitle = StatusTitle.attributed("ϕ")
        item.button?.toolTip = "Phonon"
        let menu = NSMenu()
        let attention = NSMenuItem(title: "", action: #selector(showMainWindow), keyEquivalent: "")
        attention.isHidden = true
        menu.addItem(attention)
        attentionMenuItem = attention
        menu.addItem(
            NSMenuItem(title: "Open Phonon…", action: #selector(showMainWindow), keyEquivalent: ","))
        let updateItem = NSMenuItem(title: "", action: #selector(showUpdate), keyEquivalent: "")
        updateItem.isHidden = true
        menu.addItem(updateItem)
        updateMenuItem = updateItem
        menu.addItem(.separator())
        menu.addItem(
            NSMenuItem(
                title: "Start or Stop Dictation", action: #selector(toggleRecord),
                keyEquivalent: "r"))
        let previewItem = NSMenuItem(
            title: "Streaming", action: #selector(toggleStreamingPreview), keyEquivalent: ""
        )
        previewItem.state = state.streamingPreviewEnabled ? .on : .off
        menu.addItem(previewItem)
        streamingMenuItem = previewItem
        let dockItem = NSMenuItem(
            title: "Show in Dock", action: #selector(toggleShowInDock), keyEquivalent: "")
        dockItem.state = appStore.settings.showInDock ? .on : .off
        menu.addItem(dockItem)
        showInDockMenuItem = dockItem
        menu.addItem(
            NSMenuItem(
                title: "Model Status…", action: #selector(showModelStatus), keyEquivalent: ""))
        let microphoneItem = NSMenuItem(
            title: "Microphone: Auto → Resolving…", action: nil, keyEquivalent: "")
        microphoneItem.isEnabled = false
        menu.addItem(microphoneItem)
        microphoneMenuItem = microphoneItem
        menu.addItem(
            NSMenuItem(
                title: "Quit Other Dictation Apps…", action: #selector(quitCompetingApps),
                keyEquivalent: ""))
        let meetingItem = NSMenuItem(
            title: "Stop Meeting Notes", action: #selector(stopMeetingNotes), keyEquivalent: "")
        meetingItem.isHidden = true
        menu.addItem(meetingItem)
        meetingMenuItem = meetingItem
        menu.addItem(.separator())
        menu.addItem(NSMenuItem(title: "Quit Phonon", action: #selector(quit), keyEquivalent: "q"))
        item.menu = menu
        statusItem = item
        refreshStatusTitle()
    }

    /// Tells the engine about the user's own endpoint (issue #24); the key comes from
    /// the Keychain and only ever travels over the engine's stdin. Read only when the
    /// endpoint is on, so a user who never set one is never asked about the Keychain.
    private func sendOwnEndpoint() {
        let settings = appStore.settings
        let key = settings.byokEnabled ? OwnEndpoint.readKey() : nil
        engine.send(OwnEndpoint.engineCommand(settings: settings, apiKey: key))
    }

    /// The menu-bar title and its warning line follow `PermissionAttention`.
    private func refreshStatusTitle() {
        let items = appStore.permissionAttention
        lastPermissionState = items
        let blocking = items.first(where: \.blocksDictation)
        statusItem?.button?.attributedTitle = StatusTitle.attributed(
            blocking == nil ? statusBase : statusBase + " ⚠")
        statusItem?.button?.toolTip = blocking.map { "Phonon: \($0.title)" } ?? "Phonon"
        if let first = blocking ?? items.first {
            attentionMenuItem?.title = first.menuTitle
            attentionMenuItem?.isHidden = false
        } else {
            attentionMenuItem?.isHidden = true
        }
        if items.isEmpty {
            permissionWatch?.invalidate()
            permissionWatch = nil
        } else if permissionWatch == nil {
            permissionWatch = Timer.scheduledTimer(withTimeInterval: 3, repeats: true) {
                [weak self] _ in
                Task { @MainActor in self?.checkPermissionsChanged() }
            }
        }
    }

    /// Cheap checks; a full refresh (and the event tap retry) only on a change.
    private func checkPermissionsChanged() {
        let now = PermissionAttention.items(
            inputMonitoring: CGPreflightListenEventAccess(),
            microphone: appStore.microphonePermission,
            accessibility: appStore.accessibilityPermission,
            shortcutMode: appStore.settings.shortcutMode,
            globeUsageType: GlobeKeySetting.currentUsageType())
        guard now != lastPermissionState else { return }
        appStore.refreshPermissions()  // calls refreshPermissionDependentServices
        refreshStatusTitle()
    }

    @objc private func showModelStatus() {
        // While the main window shows the loader, that is the model status.
        if modelStartupState.loaderVisible {
            showMainWindow()
            return
        }
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

    @objc private func showSettings() {
        appStore.pageRequest = .settings
        showMainWindow()
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
        state.showWhenIdle = appStore.settings.showIdleCapsule
        streamingMenuItem?.state = state.streamingPreviewEnabled ? .on : .off
        if appStore.settings.instantMic {
            recorder.refreshPreferredDevice()
            updateMicrophoneStatus()
        } else if !isRecording {
            recorder.releaseHardware()
        }
        applyDockVisibility()
        refreshStatusTitle()
        if enginesReady { sendOwnEndpoint() }
        syncToggleHotKey()
        competitors.settingsChanged()
        if !appStore.settings.meetingCapture { meetings.stop(reason: "turned_off") }
        meetings.refresh()
    }

    /// A recording meeting is always visible: a dot on the menu bar item and
    /// a Stop item in its menu.
    private func meetingRecordingChanged(_ app: String?) {
        statusBase = app == nil ? "ϕ" : "ϕ ●"
        refreshStatusTitle()
        meetingMenuItem?.isHidden = app == nil
        meetingMenuItem?.title = app.map { "Stop Meeting Notes (\($0))" } ?? "Stop Meeting Notes"
    }

    /// The capsule follows the meeting UI: red light while recording, the stop
    /// button on hover, and the pointer only while recording.
    private func meetingFlowChanged(_ flow: MeetingFlow) {
        state.apply(flow)
        panel?.ignoresMouseEvents = !flow.capsuleAcceptsMouse
    }

    @objc private func stopMeetingNotes() {
        meetings.stopFromUser()
    }

    @objc private func showUpdate() {
        updates.showAvailable()
    }

    @objc private func checkForUpdates() {
        updates.check(manual: true)
    }

    private func promptPermissions() {
        let opts = ["AXTrustedCheckOptionPrompt": true] as CFDictionary
        _ = AXIsProcessTrustedWithOptions(opts)
    }

    private func startEngine() {
        // The WebSocket API's token used to live in the Keychain; with the API on and no token
        // in settings.json the engine would not listen, so make one (phones pair again).
        if appStore.settings.websocketAPI && appStore.settings.websocketAPIToken.count < 16 {
            appStore.updateSettings { WebSocketAPI.turnOn(&$0) }
        }
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
        meetingTranscriber.setEngineReady(false)
        meetingCleaner.setEngineReady(false)
        appStore.engineReady = false
        appStore.websocketAPIStatus = nil
        appStore.websocketLive = false
        appStore.engineMessage = "Engine restarting"
        streamActive = false
        hideLiveText()
        statusBase = "ϕ…"
        refreshStatusTitle()
        modelStartupState.markRestarting()
        showMainWindow()
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
            engineStreamRefine = StreamRefine.enabled(readyEvent: ev)
            appStore.engineReady = true
            appStore.engineMessage = "Ready"
            modelStartupState.markReady()
            engineRestartWork?.cancel()
            engineRestartWork = nil
            statusBase = meetings.isRecording ? "ϕ ●" : "ϕ"
            refreshStatusTitle()
            sendOwnEndpoint()
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) { [weak self] in
                guard self?.enginesReady == true else { return }
                self?.modelStartupState.hideLoader()
                self?.modelStartupWindow?.orderOut(nil)
            }
            meetingTranscriber.setEngineReady(true)
            meetingCleaner.setEngineReady(true)
            if isRecording && state.streamingPreviewEnabled && !streamActive {
                startAsrStream()
            }
            // Flush first-pass work that arrived early.
            if let pending = pendingTranscribe {
                pendingTranscribe = nil
                sendTranscribe(path: pending.path, id: pending.id)
            } else if let pending = pendingFinalPolish {
                // Waits for screen context again, bounded, if it is still reading.
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
                    if e2eTrace?.passId == id, e2eTrace?.firstPreviewNs == nil {
                        e2eTrace?.firstPreviewNs = DispatchTime.now().uptimeNanoseconds
                    }
                    showLiveTextTail(stable)
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
                    if heldSeconds >= Self.notHeardMinSeconds {
                        hideLiveText()
                        notPasted.show(.notHeard(mic: recorder.selectedDeviceName))
                    }
                    hideAfter(0.1)
                    return
                }
                submitFinalPolish(text, id: pid)
            } else if kind == "polish" {
                guard id == activeId else { return }
                if e2eTrace?.passId == id {
                    e2eTrace?.polishResultNs = DispatchTime.now().uptimeNanoseconds
                    e2eTrace?.llmReportedWallMs = ev["latency_ms"] as? Double
                    e2eTrace?.llmTtftMs = ev["ttft_ms"] as? Double
                    e2eTrace?.llmTokensPerSecond = ev["tok_s"] as? Double
                }
                let text = ev["text"] as? String ?? rawText
                // A dictation that is exactly a macro's trigger types its expansion.
                let final = MacroBook.expand(
                    PreviewRevisionPolicy.safeFinalText(source: rawText, candidate: text),
                    macros: appStore.macros)
                processing = false
                if !final.isEmpty {
                    e2eTrace?.insertionStartedNs = DispatchTime.now().uptimeNanoseconds
                    if let category = ev["app_category"] as? String, let id {
                        editLearning.noteCategory(category, passID: id)
                    }
                    // The field as read while the correction ran (never in a
                    // private app or site), so the dictation can be told apart
                    // from the text around it after the paste.
                    let editSnapshot = id.flatMap { editLearning.takeSnapshot(passID: $0) }
                    var insertionSucceeded = false
                    let target = insertionTarget ?? NSWorkspace.shared.frontmostApplication
                    if let target,
                        TextboxAccess.pasteHasNoTarget(
                            pid: target.processIdentifier, bundleID: target.bundleIdentifier)
                    {
                        // Nowhere to paste: keep it on the clipboard and show it to copy.
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(final, forType: .string)
                        hideLiveText()
                        notPasted.show(final)
                    } else {
                        do {
                            try Typer.emit(
                                Typer.spaced(final), target: insertionTarget,
                                restoreClipboard: appStore.settings.clipboardAfterPaste == .restore)
                            insertionSucceeded = true
                        } catch {
                            NSLog("phonon insertion failed: \(error.localizedDescription)")
                            if !AXIsProcessTrusted() {
                                NSPasteboard.general.clearContents()
                                NSPasteboard.general.setString(final, forType: .string)
                                hideLiveText()
                                notPasted.show(.needsAccessibility(final))
                            }
                        }
                    }
                    e2eTrace?.insertionCompletedNs = DispatchTime.now().uptimeNanoseconds
                    // Live text ends on the pasted text itself, so a last-second revision
                    // ("scratch that") shows as the paste lands and a pending tail resolves
                    // into it. It is updated after the paste, so the paste never waits on it.
                    if liveText.showsCard {
                        liveText.applyFinal(final)
                        liveTrace.record(
                            "final", input: final, applied: true, card: liveText.shownWords)
                        // the card fades out as the paste lands, not with the capsule
                        let pass = activeId
                        DispatchQueue.main.asyncAfter(
                            deadline: .now() + LiveTextMotion.hold + LiveTextMotion.leave + 0.02
                        ) { [weak self] in
                            guard let self, self.activeId == pass, !self.isRecording else { return }
                            self.hideLiveText()
                        }
                    }
                    if insertionSucceeded, let editSnapshot, let id {
                        editLearning.didPaste(
                            editSnapshot, passID: id, pasted: Typer.spaced(final),
                            record: appStore.settings.localHistory)
                    }
                    if var record = e2eTrace?.finish(
                        textCharacters: final.count,
                        insertionSucceeded: insertionSucceeded
                    ) {
                        // issue #21 evidence: how often keys had to be resynced
                        record.eventTapDisables = eventTapDisables
                        record.hotkeyRepresses = hotkeyRepresses
                        E2EProfileStore.append(record)
                    }
                    e2eTrace = nil
                    finishRecordingRetention()
                    hideAfter(state.streamingPreviewEnabled ? 0.55 : 0)
                } else {
                    hidePill()
                }
            } else if kind == "draft" {
                // Live text: the engine sends these only with settings.json `live_text` on,
                // for the open stream. Display only; the polish result is what gets pasted.
                guard isRecording, id == activeId else { return }
                showLiveText(
                    ev["text"] as? String ?? "", source: ev["source"] as? String ?? "",
                    complete: ev["complete"] as? Bool ?? true)
            } else if kind == "edit" {
                editLearning.handleEngineResult(ev)
            }
        case "status":
            switch ev["kind"] as? String {
            case "websocket_api":
                appStore.websocketAPIStatus = ev
            case "websocket_dictation":
                if ev["state"] as? String == "started" {
                    appStore.websocketLive = true
                } else {
                    appStore.recordWebSocketDictation(
                        words: ev["words"] as? Int ?? 0,
                        audioMilliseconds: (ev["audio_ms"] as? NSNumber)?.uint64Value ?? 0)
                }
            default:
                break
            }
        case "meeting_span":
            meetingTranscriber.handle(ev)
        case "dictation":
            // the engine's meeting holds: meeting cleanup pauses while a dictation runs
            meetingCleaner.setDictationQuiet(ev["quiet"] as? Bool ?? false)
        case "error":
            let msg = ev["msg"] as? String ?? "error"
            appStore.lastError = msg
            NSLog("phonon engine error: \(msg)")
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

    /// The capsule's own box; the panel adds room for the stop button (PillPanelLayout).
    private static let containerSize = PillPanelLayout.capsule

    private func showIdle() {
        hideWork?.cancel()
        state.level = 0
        state.mode = .idle
        present()
    }

    private func showListening() {
        hideWork?.cancel()
        state.level = 0
        state.mode = .listening
        present()
    }

    private func showWorking() {
        hideWork?.cancel()
        state.level = 0
        state.mode = .working
        present()
    }

    private func present() {
        guard let panel else { return }
        hideWork?.cancel()
        // The frame is recomputed every time, not only on the first show. The
        // panel is never ordered out, so gating this on visibility left the
        // capsule stranded on whichever display it first appeared on. It now
        // follows the screen holding the pointer, like the rest of the UI.
        let target = PillPanelLayout.frame(screen: PillPanel.screenFrame())
        panel.contentView?.setFrameSize(PillPanelLayout.size)
        if panel.frame != target {
            panel.setFrame(target, display: false)
        }
        if !panel.isVisible {
            panel.alphaValue = 1
            panel.orderFrontRegardless()
        }
    }

    private func hidePill() {
        hideWork?.cancel()
        hideLiveText()
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

    @objc private func toggleShowInDock() {
        appStore.updateSettings { $0.showInDock.toggle() }
        applyDockVisibility()
    }

    /// A regular app (Dock icon, app menu) or an accessory one (menu bar item only),
    /// per Settings > System > Show in Dock.
    private func applyDockVisibility() {
        let show = appStore.settings.showInDock
        showInDockMenuItem?.state = show ? .on : .off
        let policy: NSApplication.ActivationPolicy = show ? .regular : .accessory
        guard NSApp.activationPolicy() != policy else { return }
        let windowWasVisible = mainWindow?.isVisible == true
        NSApp.setActivationPolicy(policy)
        // Becoming an accessory app deactivates it, which drops the open window
        // behind the frontmost app. Keep it in front when the switch came from it.
        if windowWasVisible {
            DispatchQueue.main.async { [weak self] in
                NSApp.activate(ignoringOtherApps: true)
                self?.mainWindow?.makeKeyAndOrderFront(nil)
            }
        }
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

    private func applyHoldKey(
        _ action: HoldKeyTapLatch.Action, source: String, eventNs: UInt64, callbackNs: UInt64
    ) {
        switch action {
        case .start:
            startDictation(source: source, eventNs: eventNs, callbackNs: callbackNs)
        case .stop:
            stopDictation(source: source, eventNs: eventNs, callbackNs: callbackNs)
        case .none:
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
        notPasted.dismiss()
        editLearning.interrupt(reason: .nextDictation)
        let ownPID = ProcessInfo.processInfo.processIdentifier
        if let frontmost = NSWorkspace.shared.frontmostApplication,
            frontmost.processIdentifier != ownPID
        {
            insertionTarget = frontmost
        } else {
            insertionTarget = nil
        }
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
        // Off the key-down path: a competitor that appeared since launch is
        // noted now and asked about after this dictation ends.
        DispatchQueue.main.async { [weak self] in self?.competitors.noteDictationStart() }
        // The correction model is slow on its first request after an idle
        // minute; waking it now hides that behind the user's speech.
        if enginesReady { engine.send(["cmd": "prewarm"]) }
        latestAcousticPreview = ""
        hideLiveText()  // a previous pass abandoned mid-correction left its text
        // The engine reads `live_text` at the stream start too; drafts need streaming refine.
        liveTextPass =
            appStore.settings.liveText && engineStreamRefine && state.streamingPreviewEnabled
        if liveTextPass { liveTrace.begin() }
        let denylist = appStore.settings.screenContextDenylist
        let target = DictationTarget.resolve(
            bundleID: insertionTarget?.bundleIdentifier, name: insertionTarget?.localizedName,
            url: nil, denylist: denylist, urlLookedUp: false)
        dictationTarget = target
        resolveDictationTarget(
            app: insertionTarget, denylist: denylist,
            readTextbox: appStore.settings.textboxContext && !target.denied)
        pendingFinalPolish = nil
        streamScreenTextSent = ""
        let wantScreen = appStore.settings.screenContext && !target.denied
        if wantScreen && target.awaitsTabLookup {
            // A browser tab might be a private site: read nothing until the
            // tab lookup says it is not (it takes milliseconds).
            startScreenContext(enabled: false)
            screenStartAwaitsTarget = true
            screenInitialDone = false
            screenContextReady = false
        } else {
            startScreenContext(enabled: wantScreen)
        }
        if appStore.settings.soundFeedback { cues.playStart() }
        ducker.start(appStore.settings.backgroundAudio)
        showListening()
        e2eTrace?.panelShownNs = DispatchTime.now().uptimeNanoseconds
        recorder.onAmplitude = { [weak self] a in
            Task { @MainActor in
                guard let self, self.isRecording else { return }
                self.state.level = a
                self.checkMicSilence()
                let voice = self.voiceDetector.isVoice(a)
                self.latchAutoStop.note(voice: voice, at: ProcessInfo.processInfo.systemUptime)
                if self.liveTextPass {
                    // the dot answers the voice at once, before any word is recognized
                    self.liveText.noteLevel(a)
                    if voice, self.liveText.noteVoice() { self.presentLiveText() }
                } else if voice {
                    self.liveText.noteVoice()
                }
                self.liveTrace.voice(voice)
            }
        }
        recordingStartedAt = Date()
        voiceDetector = VoiceDetector()
        stopLatchAutoStop()
        latchAutoStop = LatchAutoStop(startedAt: ProcessInfo.processInfo.systemUptime)
        // No buffers at all never calls onAmplitude, so check on a timer too.
        micSilenceCheck?.cancel()
        let check = DispatchWorkItem { [weak self] in self?.checkMicSilence() }
        micSilenceCheck = check
        DispatchQueue.main.asyncAfter(
            deadline: .now() + Self.micSilenceSeconds + 0.3, execute: check)
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
            // The engine heard `prewarm` and maybe `stream_start`: end both there.
            if streamActive {
                engine.send(["cmd": "stream_stop", "id": activeId ?? "p\(pass)"])
                streamActive = false
            } else {
                engine.send(["cmd": "dictation_end", "id": activeId ?? "p\(pass)"])
            }
            ducker.stop()
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
        stopLatchAutoStop()
        ducker.stop()  // the audio comes back while the text is corrected
        micSilenceCheck?.cancel()
        if case .micSilent = notPasted.current { notPasted.dismiss() }
        heldSeconds = recordingStartedAt.map { Date().timeIntervalSince($0) } ?? 0
        liveText.listening = false  // the dot goes; a pending tail waits for the final text
        liveText.showEndMark()
        stopScreenSwitchWatch()
        if streamActive {
            let streamId = activeId ?? "p\(pass)"
            if engineStreamRefine {
                // Streaming refine: send the last partial chunk too, so the stream holds the
                // whole recording, and finish it now. The release transcription then runs
                // while the WAV is written; the transcribe that follows releases its text.
                if let tail = recorder.drainStreamPCM16() {
                    engine.send([
                        "cmd": "stream_chunk", "id": streamId,
                        "pcm16": tail.base64EncodedString(),
                    ])
                }
                engine.send(["cmd": "stream_finish", "id": streamId])
            } else {
                engine.send(["cmd": "stream_stop", "id": streamId])
            }
            streamActive = false
        }

        // Instant fireball on key-up.
        if appStore.settings.soundFeedback { cues.playStop() }
        showWorking()
        if competitors.hasPendingPrompt {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { [weak self] in
                self?.presentCompetitorPromptWhenIdle()
            }
        }

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
                    // No WAV, so no transcribe: tell the engine the dictation is over.
                    self.engine.send(["cmd": "dictation_end", "id": id])
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

    /// Watch a dictation a tap just latched, until it stops (`LatchAutoStop`). A timer, not
    /// the meter, because a mic that delivers no buffers never calls `onAmplitude`.
    private func armLatchAutoStop() {
        guard isRecording, latchAutoStopTimer == nil else { return }
        latchAutoStopTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) {
            [weak self] _ in
            MainActor.assumeIsolated { self?.checkLatchAutoStop() }
        }
    }

    private func stopLatchAutoStop() {
        latchAutoStopTimer?.invalidate()
        latchAutoStopTimer = nil
    }

    private func checkLatchAutoStop() {
        guard isRecording, fnTapLatch.latched || optionTapLatch.latched else {
            stopLatchAutoStop()
            return
        }
        let decision = latchAutoStop.decision(at: ProcessInfo.processInfo.systemUptime)
        guard decision != .keepRecording else { return }
        // The key is up (it latched), so the next press must start, not stop.
        fnTapLatch.reset()
        optionTapLatch.reset()
        switch decision {
        case .commit:
            NSLog(
                "phonon: latched dictation silent for \(Int(LatchAutoStop.silenceAfterSpeechSeconds)) s after speech; stopping"
            )
            stopDictation(source: "latch-auto-stop")
        case .discard:
            NSLog(
                "phonon: latched dictation heard no speech in \(Int(LatchAutoStop.noSpeechSeconds)) s; discarding"
            )
            discardDictation()
        case .keepRecording:
            break
        }
    }

    /// End the open dictation as if it never started: no stop cue, no WAV, no transcription,
    /// no paste and no card. For a latched dictation nobody spoke into.
    private func discardDictation() {
        guard isRecording else { return }
        isRecording = false
        stopLatchAutoStop()
        ducker.stop()
        micSilenceCheck?.cancel()
        if case .micSilent = notPasted.current { notPasted.dismiss() }
        dictationTargetTask?.cancel()
        dropScreenContext()
        if streamActive {
            // Without stream_finish or a transcribe the engine emits no result for it; the
            // next stream_start replaces it.
            engine.send(["cmd": "stream_stop", "id": activeId ?? "p\(pass)"])
            streamActive = false
        } else {
            // No stream to stop, but the engine heard the key go down (`prewarm`): tell it the
            // dictation is over, so meeting transcription does not wait for it.
            engine.send(["cmd": "dictation_end", "id": activeId ?? "p\(pass)"])
        }
        recorder.endCapture(keepHardwareWarm: appStore.settings.instantMic)
        _ = recorder.takeSamples()
        e2eTrace = nil
        hidePill()
        if competitors.hasPendingPrompt {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { [weak self] in
                self?.presentCompetitorPromptWhenIdle()
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

    /// Read the browser tab (or Electron page) URL off the main thread. A
    /// site on the private list drops whatever screen context this pass has.
    private func resolveDictationTarget(
        app: NSRunningApplication?, denylist: [String], readTextbox: Bool
    ) {
        dictationTargetTask?.cancel()
        dictationTextbox = nil
        let pass = activeId
        let bundleID = app?.bundleIdentifier
        let name = app?.localizedName
        let pid = app?.processIdentifier
        let watchEdits = appStore.settings.editLearning
        let readsFields = readTextbox || watchEdits
        let screenContext = appStore.settings.screenContext
        dictationTargetTask = Task { [weak self] in
            let (target, field) = await Task.detached(priority: .userInitiated) {
                let target = DictationTarget.capture(
                    bundleID: bundleID, name: name, pid: pid, readsFields: readsFields,
                    screenContext: screenContext, denylist: denylist)
                // One read of the focused field, only if it belongs to the
                // dictation's app and neither the app nor its page is private:
                // the text around the cursor (with the setting on) and the
                // anchors edit learning needs after the paste.
                let field = KeyDownFieldRead.read(
                    target: target, targetPID: pid, wantTextbox: readTextbox,
                    wantEditSnapshot: watchEdits)
                return (target, field)
            }.value
            guard let self, !Task.isCancelled, self.activeId == pass, let pass else { return }
            self.dictationTarget = target
            if self.streamActive, self.isRecording, !target.enginePayload.isEmpty {
                self.engine.send(["cmd": "stream_app", "id": pass, "app": target.enginePayload])
            }
            self.dictationTextbox = field.textbox
            self.editLearning.store(field.editSnapshot, passID: pass)
            if target.denied {
                self.dropScreenContext()
            } else if self.screenStartAwaitsTarget {
                self.screenStartAwaitsTarget = false
                if self.isRecording {
                    self.startScreenContext(enabled: true)
                } else {
                    // Key-up came first: correct without screen context
                    // rather than start reading now.
                    self.screenInitialDone = true
                    self.screenContextUpdated()
                }
            }
        }
    }

    // MARK: Screen context

    /// Read the visible windows of every display now, then any window or tab
    /// the user switches to until key-up.
    private func startScreenContext(enabled: Bool) {
        screenStartAwaitsTarget = false
        screenSwitchTimer?.invalidate()
        screenSwitchTimer = nil
        screenWaitWork?.cancel()
        screenWaitWork = nil
        cancelScreenReads()
        screenCollection = ScreenContextCollection()
        screenInitialDone = !enabled
        screenContextReady = !enabled
        guard enabled else { return }
        let token = screenReads.generation
        let denylist = appStore.settings.screenContextDenylist
        let ownPID = ProcessInfo.processInfo.processIdentifier
        let front = NSWorkspace.shared.frontmostApplication
        let frontPID = front?.processIdentifier
        let frontIsBrowser = ContextPolicy.isBrowser(front?.bundleIdentifier)
        screenContextTask = Task { [weak self] in
            // Only the focused browser window can be read, and only with its
            // top-level page known; background browser windows never are.
            let focusedPage = await Task.detached(priority: .userInitiated) {
                ScreenContextCapture.focusedBrowserPage(pid: frontPID, isBrowser: frontIsBrowser)
            }.value
            var reads = await ScreenContextCapture.readVisible(
                excluded: ContextPolicy.screenExclusion(
                    denylist: denylist, ownPID: ownPID, focusedPage: focusedPage))
            if focusedPage != nil {
                // The browser window was read for one page; if the focus
                // moved to another window or tab meanwhile, drop its read.
                let after = await Task.detached(priority: .userInitiated) {
                    ScreenContextCapture.focusedBrowserPage(
                        pid: frontPID, isBrowser: frontIsBrowser)
                }.value
                reads = ScreenContextCapture.keptReads(reads, allowed: focusedPage, now: after)
            }
            guard !Task.isCancelled, let self, self.screenReads.isCurrent(token) else { return }
            for read in reads { self.screenCollection.add(keys: read.keys, text: read.text) }
            self.screenInitialDone = true
            self.screenContextUpdated()
        }
        screenSwitchTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) {
            [weak self] _ in
            MainActor.assumeIsolated { self?.readSwitchedWindow() }
        }
    }

    /// A window or browser tab that came to the front mid-dictation and has
    /// not been read yet is read now, unless it is private.
    private func readSwitchedWindow() {
        // The key-down read covers whatever is in front until it finishes.
        guard isRecording, screenInitialDone, !screenCollection.isFull,
            let app = NSWorkspace.shared.frontmostApplication,
            app.processIdentifier != ProcessInfo.processInfo.processIdentifier
        else { return }
        let denylist = appStore.settings.screenContextDenylist
        let ownPID = ProcessInfo.processInfo.processIdentifier
        // App-level check first; a browser's tab is decided once its page
        // is known (the switch focused this window).
        guard let window = ScreenContextCapture.frontWindow(of: app.processIdentifier),
            screenCollection.wants(window.captureKey),
            !ContextPolicy.isPrivateApp(window, denylist: denylist, ownPID: ownPID)
        else { return }
        screenCollection.markSeen([window.captureKey])
        let token = screenReads.begin()
        screenContextReady = false
        let bundleID = app.bundleIdentifier
        let name = app.localizedName
        let pid = app.processIdentifier
        let isBrowser = ContextPolicy.isBrowser(bundleID)
        let task = Task { [weak self] in
            // One walk gives the page, its host and its window.
            let (target, focus) = await Task.detached(priority: .userInitiated) {
                DictationTarget.captureWithFocus(
                    bundleID: bundleID, name: name, pid: pid, screenContext: true,
                    denylist: denylist)
            }.value
            // A browser window is read only if it is the focused element's
            // own window (not a popup, not a window the focus left).
            let focusedPage = ScreenContextCapture.switchedPage(focus, window: window.id)
            let isPrivate =
                target.denied
                || ContextPolicy.isPrivateWindow(
                    window, denylist: denylist, ownPID: ownPID, focusedPage: focusedPage)
            var text =
                isPrivate || Task.isCancelled
                ? nil : await ScreenContextCapture.readWindow(id: window.id)
            if text != nil, isBrowser {
                // Still the same window and page after the capture?
                let after = await Task.detached(priority: .userInitiated) {
                    BrowserPage.focusedWindow(pid: pid)
                }.value
                if !ScreenContextCapture.samePage(focusedPage, after) { text = nil }
            }
            // A read from before a reset (new dictation, private target) is
            // dropped whole: its text must never reach the correction.
            guard let self, self.screenReads.finish(token), !Task.isCancelled else { return }
            if let text { self.screenCollection.add(keys: [window.captureKey], text: text) }
            self.screenContextUpdated()
        }
        screenSwitchTasks.append(task)
    }

    /// Cancel every screen read in flight and invalidate their results.
    private func cancelScreenReads() {
        screenContextTask?.cancel()
        screenContextTask = nil
        screenSwitchTasks.forEach { $0.cancel() }
        screenSwitchTasks.removeAll()
        screenReads.reset()
    }

    private func screenContextUpdated() {
        sendStreamScreenContext()
        screenContextReady = ScreenContextWait.shouldSubmit(
            pendingReads: screenReads.pending, initialDone: screenInitialDone, waited: 0)
        if screenContextReady, let pending = pendingFinalPolish {
            pendingFinalPolish = nil
            submitFinalPolish(pending.text, id: pending.id)
        }
    }

    /// Streaming refine (PHONON_STREAM_REFINE=1): the engine refines the transcript so far
    /// while the speaker talks. With the screen text read so far in hand, its background
    /// prompts carry the same dictionary envelope as the final correction, which then reuses
    /// their cached state. Sent again only when the collected text changed.
    private func sendStreamScreenContext() {
        guard engineStreamRefine, isRecording else { return }
        let id = activeId ?? "p\(pass)"  // the stream's id (startAsrStream)
        let text = screenCollection.text
        guard !text.isEmpty, text != streamScreenTextSent else { return }
        streamScreenTextSent = text
        engine.send(["cmd": "screen_context", "id": id, "text": text])
    }

    private func stopScreenSwitchWatch() {
        screenSwitchTimer?.invalidate()
        screenSwitchTimer = nil
    }

    private func dropScreenContext() {
        screenStartAwaitsTarget = false
        cancelScreenReads()
        stopScreenSwitchWatch()
        screenCollection = ScreenContextCollection()
        screenInitialDone = true
        guard !screenContextReady else { return }
        screenContextReady = true
        if let pending = pendingFinalPolish {
            pendingFinalPolish = nil
            submitFinalPolish(pending.text, id: pending.id)
        }
    }

    /// The final correction waits for screen context, but at most
    /// `ScreenContextWait.afterFinalTranscript`; then it goes with whatever
    /// was read by then.
    private func submitFinalPolish(_ text: String, id: String, force: Bool = false) {
        guard activeId == id, processing else { return }
        guard screenContextReady || force else {
            pendingFinalPolish = (text, id)
            if screenWaitWork == nil {
                let work = DispatchWorkItem { [weak self] in
                    guard let self, let pending = self.pendingFinalPolish, pending.id == id else {
                        return
                    }
                    self.pendingFinalPolish = nil
                    NSLog("phonon screen context: wait bounded, correcting without the rest")
                    self.submitFinalPolish(pending.text, id: pending.id, force: true)
                }
                screenWaitWork = work
                DispatchQueue.main.asyncAfter(
                    deadline: .now() + ScreenContextWait.afterFinalTranscript, execute: work)
            }
            return
        }
        screenWaitWork?.cancel()
        screenWaitWork = nil
        if e2eTrace?.passId == id {
            e2eTrace?.polishSubmittedNs = DispatchTime.now().uptimeNanoseconds
        }
        engine.send(polishCommand(text: text, id: id, context: screenCollection.text))
    }

    private func polishCommand(text: String, id: String, context: String) -> [String: Any] {
        PolishCommand.make(
            text: text, id: id, screenText: context, target: dictationTarget,
            textbox: appStore.settings.textboxContext ? dictationTextbox : nil,
            localHistory: appStore.settings.localHistory)
    }

    /// The target app rides along so the engine's background refines use the writing mode
    /// of the final correction; a browser tab's host follows in `stream_app`.
    private func startAsrStream() {
        guard !streamActive else { return }
        let id = activeId ?? "p\(pass)"
        var command: [String: Any] = ["cmd": "stream_start", "id": id]
        if let app = dictationTarget?.enginePayload, !app.isEmpty { command["app"] = app }
        engine.send(command)
        streamActive = true
    }

    // MARK: Hotkey

    /// Handles the toggle hotkey; `syncToggleHotKey()` registers it.
    private func installToggleHandler() {
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
        if handlerStatus != noErr {
            NSLog("phonon: Ctrl+Space handler installation failed (\(handlerStatus))")
        }
    }

    /// Registers the toggle hotkey only while the shortcut mode uses it. Carbon
    /// hotkeys work without Input Monitoring, unlike the hold keys, but a
    /// registered one never reaches other apps.
    private func syncToggleHotKey() {
        let wanted = shortcutAllows(source: "control-space")
        if !wanted, let toggleHotKey {
            UnregisterEventHotKey(toggleHotKey)
            self.toggleHotKey = nil
        } else if wanted, toggleHotKey == nil {
            let hotKeyId = EventHotKeyID(signature: 0x50484F4E, id: 1)  // PHON
            let status = RegisterEventHotKey(
                UInt32(kVK_Space),
                UInt32(controlKey),
                hotKeyId,
                GetApplicationEventTarget(),
                0,
                &toggleHotKey
            )
            if status != noErr {
                toggleHotKey = nil
                NSLog("phonon: Ctrl+Space registration failed (\(status))")
            }
        }
    }

    private func installHotkey() {
        if let eventTap, CFMachPortIsValid(eventTap) {
            CGEvent.tapEnable(tap: eventTap, enable: true)
            appStore.inputMonitoringAvailable = CGEvent.tapIsEnabled(tap: eventTap)
            return
        }
        // Key-downs are listened to only so edit learning can tell a sent
        // message from a cleared field; they are ignored unless a field is
        // being watched, and only their category is kept.
        let mask = (1 << CGEventType.flagsChanged.rawValue) | (1 << CGEventType.keyDown.rawValue)
        let callback: CGEventTapCallBack = { _, type, event, refcon in
            guard let refcon else { return Unmanaged.passUnretained(event) }
            let app = Unmanaged<AppController>.fromOpaque(refcon).takeUnretainedValue()
            if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
                if let tap = app.eventTap {
                    CGEvent.tapEnable(tap: tap, enable: true)
                }
                // Key events while the tap was off are gone, a release among them
                // perhaps: re-read which modifiers are held now.
                let held = CGEventSource.flagsState(.combinedSessionState)
                let reason = type == .tapDisabledByTimeout ? "timeout" : "user input"
                Task { @MainActor in app.resyncHoldKeys(held: held, reason: reason) }
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
            NSLog("phonon: Globe/Right-Option unavailable (Input Monitoring not granted)")
            // Once per launch: macOS shows its own prompt the first time and adds
            // Phonon to the Input Monitoring list, so only the switch is left.
            if !requestedInputMonitoring
                && ShortcutPolicy.usesHoldKey(mode: appStore.settings.shortcutMode)
            {
                requestedInputMonitoring = true
                _ = CGRequestListenEventAccess()
            }
            refreshStatusTitle()
            return
        }
        eventTap = tap
        appStore.inputMonitoringAvailable = true
        let src = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
        eventTapSource = src
        CFRunLoopAddSource(CFRunLoopGetMain(), src, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)
    }

    private func refreshPermissionDependentServices() {
        // After the checks below settle `inputMonitoringAvailable`.
        defer { DispatchQueue.main.async { [weak self] in self?.refreshStatusTitle() } }
        warmMicrophoneIfAuthorized()
        guard CGPreflightListenEventAccess() else {
            if let eventTap { CGEvent.tapEnable(tap: eventTap, enable: false) }
            appStore.inputMonitoringAvailable = false
            return
        }
        if let eventTap, CFMachPortIsValid(eventTap) {
            CGEvent.tapEnable(tap: eventTap, enable: true)
            appStore.inputMonitoringAvailable = CGEvent.tapIsEnabled(tap: eventTap)
            return
        }
        removeEventTap()
        installHotkey()
    }

    private func removeEventTap() {
        if let eventTapSource {
            CFRunLoopRemoveSource(CFRunLoopGetMain(), eventTapSource, .commonModes)
        }
        if let eventTap { CFMachPortInvalidate(eventTap) }
        eventTapSource = nil
        eventTap = nil
    }

    nonisolated private func handleCGEvent(type: CGEventType, event: CGEvent) {
        if type == .keyDown {
            guard editKeyWatch.withLock({ $0 }) else { return }
            let keyCode = Int(event.getIntegerValueField(.keyboardEventKeycode))
            let flags = event.flags
            Task { @MainActor in
                self.editLearning.noteKey(
                    keyCode: keyCode, command: flags.contains(.maskCommand),
                    option: flags.contains(.maskAlternate), shift: flags.contains(.maskShift))
            }
            return
        }
        if type == .flagsChanged {
            let flags = event.flags
            let keyCode = Int(event.getIntegerValueField(.keyboardEventKeycode))
            // Right Option only: the left key sets the same generic flag (#23).
            let rightOption = { (wasDown: Bool) in
                RightOptionKey.isDown(keyCode: keyCode, rawFlags: flags.rawValue, wasDown: wasDown)
            }
            // The Globe/fn key reports through this same tap as its own flag.
            // Require it to be the only modifier, so fn-qualified keys such as
            // the arrow cluster can never open a recording.
            let onlyFn =
                flags.contains(.maskSecondaryFn)
                && !flags.contains(.maskAlternate)
                && !flags.contains(.maskCommand)
                && !flags.contains(.maskControl)
                && !flags.contains(.maskShift)
            let eventNs = event.timestamp
            let callbackNs = DispatchTime.now().uptimeNanoseconds
            Task { @MainActor in
                if self.shortcutAllows(source: "right-option") {
                    // Same logic as the Globe key: hold to talk, or tap to start
                    // and tap again to stop.
                    let edge = self.optionEdges.update(
                        pressed: rightOption(self.optionEdges.isDown),
                        ownKey: keyCode == RightOptionKey.rightKeyCode)
                    self.holdKey(edge, source: "right-option", eventNs: eventNs, callbackNs: callbackNs)
                } else {
                    self.optionEdges.reset()
                    self.optionTapLatch.reset()
                }
                if self.shortcutAllows(source: "fn") {
                    let edge = self.fnEdges.update(
                        pressed: onlyFn, ownKey: keyCode == Self.functionKeyCode)
                    self.holdKey(edge, source: "fn", eventNs: eventNs, callbackNs: callbackNs)
                } else {
                    self.fnEdges.reset()
                    self.fnTapLatch.reset()
                }
            }
        }
    }

    /// kVK_Function: the flags-changed event of the Globe/fn key itself.
    nonisolated static let functionKeyCode = 63

    private func tapLatch(for source: String) -> HoldKeyTapLatch {
        source == "fn" ? fnTapLatch : optionTapLatch
    }

    private func setTapLatch(_ latch: HoldKeyTapLatch, for source: String) {
        if source == "fn" { fnTapLatch = latch } else { optionTapLatch = latch }
    }

    /// One edge of a hold key through its tap latch.
    private func holdKey(
        _ edge: HoldKeyEdges.Edge, source: String, eventNs: UInt64, callbackNs: UInt64
    ) {
        var latch = tapLatch(for: source)
        // Something else (Ctrl+Space, the menu) may have ended a latched
        // recording; the next press must start, not stop.
        if latch.latched && !isRecording { latch.reset() }
        var actions: [HoldKeyTapLatch.Action] = []
        switch edge {
        case .down:
            actions = [latch.keyDown(at: eventNs)]
        case .up:
            actions = [latch.keyUp(at: eventNs)]
        case .repress:
            // The release before this press was lost (#21). A latched recording
            // stops on this press. A hold that is still recording ends here, as that
            // release would have ended it; if the lost release made it a tap, it
            // latched and this press is the stop. Idle, this press starts a
            // dictation like any other.
            hotkeyRepresses += 1
            NSLog("phonon: \(source) pressed again without a release; resynced")
            if latch.latched {
                actions = [latch.keyDown(at: eventNs)]
            } else if isRecording {
                let up = latch.keyUp(at: eventNs)
                actions = up == .stop ? [.stop] : [latch.keyDown(at: eventNs)]
            } else {
                latch.reset()
                actions = [latch.keyDown(at: eventNs)]
            }
        case .none:
            return
        }
        setTapLatch(latch, for: source)
        for action in actions {
            applyHoldKey(action, source: source, eventNs: eventNs, callbackNs: callbackNs)
        }
        if latch.latched && isRecording { armLatchAutoStop() }
    }

    /// After macOS turned the event tap off and on again: a hold key that reads
    /// as held but no longer is had its release lost; release it now.
    func resyncHoldKeys(held: CGEventFlags, reason: String) {
        eventTapDisables += 1
        NSLog("phonon: event tap was disabled (\(reason)), \(eventTapDisables) so far; resyncing keys")
        let nowNs = DispatchTime.now().uptimeNanoseconds
        if fnEdges.isDown && !held.contains(.maskSecondaryFn) {
            holdKey(fnEdges.update(pressed: false, ownKey: false), source: "fn",
                eventNs: nowNs, callbackNs: nowNs)
        }
        if optionEdges.isDown && !held.contains(.maskAlternate) {
            holdKey(optionEdges.update(pressed: false, ownKey: false), source: "right-option",
                eventNs: nowNs, callbackNs: nowNs)
        }
    }


    /// A `.regular` app built without a nib gets no menu bar, and AppKit hangs
    /// every standard key equivalent off menu items. Without this, Command-Q did
    /// nothing and the text fields had no Command-C, V or A either.
    private func setupMainMenu() {
        let mainMenu = NSMenu()

        let appItem = NSMenuItem()
        let appMenu = NSMenu(title: "Phonon")
        appMenu.addItem(
            withTitle: "About Phonon",
            action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
        let checkUpdates = NSMenuItem(
            title: "Check for Updates…", action: #selector(checkForUpdates), keyEquivalent: "")
        checkUpdates.target = self
        appMenu.addItem(checkUpdates)
        appMenu.addItem(.separator())
        let settings = NSMenuItem(
            title: "Settings…", action: #selector(showSettings), keyEquivalent: ",")
        settings.target = self
        appMenu.addItem(settings)
        appMenu.addItem(.separator())
        appMenu.addItem(
            withTitle: "Hide Phonon", action: #selector(NSApplication.hide(_:)),
            keyEquivalent: "h")
        appMenu.addItem(.separator())
        let quitItem = NSMenuItem(title: "Quit Phonon", action: #selector(quit), keyEquivalent: "q")
        quitItem.target = self
        appMenu.addItem(quitItem)
        appItem.submenu = appMenu
        mainMenu.addItem(appItem)

        let editItem = NSMenuItem()
        let editMenu = NSMenu(title: "Edit")
        editMenu.addItem(
            withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
        editMenu.addItem(
            withTitle: "Redo", action: Selector(("redo:")), keyEquivalent: "Z")
        editMenu.addItem(.separator())
        editMenu.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        editMenu.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        editMenu.addItem(
            withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        editMenu.addItem(
            withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        editItem.submenu = editMenu
        mainMenu.addItem(editItem)

        let windowItem = NSMenuItem()
        let windowMenu = NSMenu(title: "Window")
        windowMenu.addItem(
            withTitle: "Close", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
        windowMenu.addItem(
            withTitle: "Minimize", action: #selector(NSWindow.miniaturize(_:)), keyEquivalent: "m")
        windowItem.submenu = windowMenu
        mainMenu.addItem(windowItem)

        NSApp.mainMenu = mainMenu
        NSApp.windowsMenu = windowMenu
    }

    /// An uninstall sweeps ~/Library, so a reinstall can arrive with an empty
    /// store while the off-Library mirror still holds the dictionary. Offer it
    /// back rather than restoring silently: someone who uninstalled may have
    /// meant it, so declining and deleting are both one click.
    private func offerBackupRestoreIfNeeded() {
        guard let manifest = appStore.restorableBackup() else { return }
        let stamp = DateFormatter.localizedString(
            from: manifest.savedAt, dateStyle: .medium, timeStyle: .short)
        let alert = NSAlert()
        alert.messageText = "Restore your Phonon data?"
        alert.informativeText =
            "This Mac has a backup from \(stamp) holding "
            + "\(manifest.dictionaryEntries) dictionary entries and "
            + "\(manifest.historyItems) recordings, but Phonon's data folder is "
            + "empty. Uninstalling removes everything under ~/Library, so Phonon "
            + "keeps a copy of the small irreplaceable files in ~/.phonon."
        alert.addButton(withTitle: "Restore")
        alert.addButton(withTitle: "Not Now")
        alert.addButton(withTitle: "Delete Backup")
        alert.alertStyle = .informational
        NSApp.activate(ignoringOtherApps: true)
        switch alert.runModal() {
        case .alertFirstButtonReturn:
            appStore.restoreFromBackup()
            engine.send(["cmd": "reload_dictionary"])
        case .alertThirdButtonReturn:
            appStore.deleteBackup()
        default:
            break
        }
    }

    @objc func quit() {
        terminating = true
        engineRestartWork?.cancel()
        screenContextTask?.cancel()
        stopScreenSwitchWatch()
        editLearning.interrupt(reason: .appTermination)
        meetings.stopNow(reason: "app_quit")
        meetingCleaner.shutdown()
        engine.shutdown()
        NSApp.terminate(nil)
    }

    func applicationWillTerminate(_ notification: Notification) {
        terminating = true
        ducker.restoreNow()
        engineRestartWork?.cancel()
        screenContextTask?.cancel()
        stopScreenSwitchWatch()
        meetings.stopNow(reason: "app_quit")
        meetingCleaner.shutdown()
        engine.shutdown()
        recorder.stop(writeFile: false)
        removeEventTap()
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

    /// A write to a helper that already exited (the engine, during a quit or after a crash)
    /// must be an error the caller handles, not a signal that kills the app.
    static func ignoreBrokenPipes() {
        signal(SIGPIPE, SIG_IGN)
    }

    static func main() {
        ignoreBrokenPipes()
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
