// Competing dictation apps.
//
// Two apps holding the same key both record and both insert. Phonon finds the
// competitor, asks once, quits it only with consent, then re-installs its own
// event tap so it sits at the head of the tap chain. See SPEC.md
// "Competing dictation apps" for the table and the five-step policy.

import AppKit
import CoreAudio
import Darwin
import Foundation

// MARK: - Table

/// A bare helper process that outlives its app. Matched by exact process name
/// plus a fragment of the executable path, so nothing with a similar name is
/// ever signalled.
struct CompetitorHelperProcess: Equatable {
    let name: String
    let pathFragment: String
}

/// One dictation app Phonon knows how to detect and quit.
struct CompetingApp: Equatable {
    /// Where the app keeps the shortcut that starts a recording.
    enum HotkeySource: Equatable {
        /// `~/Library/Application Support/Wispr Flow/config.json`
        case wisprConfig
        /// `~/Library/Application Support/Aqua Voice/settings.json`
        case aquaSettings
        /// `defaults read com.prakashjoshipax.VoiceInk Shortcut_primaryRecording`
        case voiceInkDefaults
        /// No known on-disk location. The app is assumed to conflict.
        case unknown
    }

    let name: String
    /// Main app bundle identifiers. `NSWorkspace` lists these.
    let bundleIDs: [String]
    /// Helper apps with a bundle of their own. `NSWorkspace` lists these too.
    let helperBundleIDs: [String]
    /// Helper processes without a bundle.
    let helperProcesses: [CompetitorHelperProcess]
    let hotkeySource: HotkeySource
    /// How the app comes back after a quit. Shown in the prompt.
    let returnNote: String?

    init(
        name: String,
        bundleIDs: [String],
        helperBundleIDs: [String] = [],
        helperProcesses: [CompetitorHelperProcess] = [],
        hotkeySource: HotkeySource = .unknown,
        returnNote: String? = nil
    ) {
        self.name = name
        self.bundleIDs = bundleIDs
        self.helperBundleIDs = helperBundleIDs
        self.helperProcesses = helperProcesses
        self.hotkeySource = hotkeySource
        self.returnNote = returnNote
    }

    /// The identifier that keys "don't ask again" and the once-per-session guard.
    var primaryBundleID: String { bundleIDs[0] }
}

enum CompetingAppTable {
    static let apps: [CompetingApp] = [
        CompetingApp(
            name: "Wispr Flow",
            bundleIDs: ["com.electron.wispr-flow"],
            helperBundleIDs: ["com.electron.wispr-flow.accessibility-mac-app"],
            hotkeySource: .wisprConfig,
            returnNote:
                "Wispr Flow opens at login by default, so it comes back after the next "
                + "sign-in unless you turn that off in its own settings."
        ),
        CompetingApp(
            name: "Aqua Voice",
            bundleIDs: ["com.electron.aqua-voice"],
            helperProcesses: [
                CompetitorHelperProcess(name: "AquaMacOSBridge", pathFragment: "/Aqua Voice.app/")
            ],
            hotkeySource: .aquaSettings
        ),
        CompetingApp(
            name: "superwhisper",
            bundleIDs: ["com.superduper.superwhisper", "com.superduper.superwhisper-setapp"],
            hotkeySource: .unknown
        ),
        CompetingApp(
            name: "VoiceInk",
            bundleIDs: ["com.prakashjoshipax.VoiceInk"],
            hotkeySource: .voiceInkDefaults
        ),
        CompetingApp(
            name: "MacWhisper",
            bundleIDs: ["com.goodsnooze.MacWhisper"],
            hotkeySource: .unknown
        ),
        CompetingApp(
            name: "Typeless",
            bundleIDs: ["now.typeless.desktop"],
            hotkeySource: .unknown
        ),
    ]

    static func app(forBundleID bundleID: String) -> CompetingApp? {
        apps.first { $0.bundleIDs.contains(bundleID) }
    }

    static func isHelper(bundleID: String) -> Bool {
        apps.contains { $0.helperBundleIDs.contains(bundleID) }
    }

    static var watchedBundleIDs: Set<String> {
        Set(apps.flatMap { $0.bundleIDs + $0.helperBundleIDs })
    }
}

// MARK: - Shortcut chords

/// A competitor shortcut reduced to the parts that matter for a clash with
/// Phonon: which modifiers it holds, whether Space is the key, and whether any
/// other key is involved.
struct CompetitorChord: Equatable {
    enum Kind: Equatable {
        /// Globe/fn alone or fn+Space. Phonon's fn hold fires on the fn press.
        case fn
        /// Option alone or Option+Space. Phonon's Right Option hold fires on any
        /// Option press.
        case option
        /// Control+Space. Phonon's toggle.
        case controlSpace
        case other
    }

    var fn = false
    var option = false
    var control = false
    var command = false
    var shift = false
    var space = false
    var otherKeys = 0

    var kind: Kind {
        guard otherKeys == 0 else { return .other }
        if fn, !option, !control, !command, !shift { return .fn }
        if option, !fn, !control, !command, !shift { return .option }
        if control, space, !fn, !option, !command, !shift { return .controlSpace }
        return .other
    }

    /// Carbon virtual key codes as Wispr Flow and VoiceInk store them.
    static func fromKeyCodes(_ codes: [Int]) -> CompetitorChord {
        var chord = CompetitorChord()
        for code in codes {
            switch code {
            case 63: chord.fn = true
            case 58, 61: chord.option = true
            case 59, 62: chord.control = true
            case 54, 55: chord.command = true
            case 56, 60: chord.shift = true
            case 49: chord.space = true
            default: chord.otherKeys += 1
            }
        }
        return chord
    }

    /// Key names as Aqua Voice stores them: `"Fn+Space"`, `"Meta+Control+KeyV"`.
    static func fromTokens(_ tokens: [String]) -> CompetitorChord {
        var chord = CompetitorChord()
        for raw in tokens {
            let token = raw.trimmingCharacters(in: .whitespaces).lowercased()
            switch token {
            case "": continue
            case "fn", "globe", "function": chord.fn = true
            case "alt", "option", "opt": chord.option = true
            case "control", "ctrl": chord.control = true
            case "meta", "command", "cmd", "super": chord.command = true
            case "shift": chord.shift = true
            case "space": chord.space = true
            default: chord.otherKeys += 1
            }
        }
        return chord
    }

    /// `NSEvent.ModifierFlags` raw bits plus a key code, as `KeyboardShortcuts`
    /// style stores write them.
    static func fromKeyCode(_ code: Int, modifierFlags: Int) -> CompetitorChord {
        var chord = fromKeyCodes([code])
        if modifierFlags & (1 << 17) != 0 { chord.shift = true }
        if modifierFlags & (1 << 18) != 0 { chord.control = true }
        if modifierFlags & (1 << 19) != 0 { chord.option = true }
        if modifierFlags & (1 << 20) != 0 { chord.command = true }
        if modifierFlags & (1 << 23) != 0 { chord.fn = true }
        return chord
    }

    /// Carbon modifier bits plus a key code.
    static func fromKeyCode(_ code: Int, carbonModifiers: Int) -> CompetitorChord {
        var chord = fromKeyCodes([code])
        if carbonModifiers & 256 != 0 { chord.command = true }
        if carbonModifiers & 512 != 0 { chord.shift = true }
        if carbonModifiers & 2048 != 0 { chord.option = true }
        if carbonModifiers & 4096 != 0 { chord.control = true }
        return chord
    }

    var label: String {
        switch kind {
        case .fn: return space ? "Globe (fn) + Space" : "Globe (fn)"
        case .option: return space ? "Option + Space" : "Option"
        case .controlSpace: return "Control + Space"
        case .other: return "another key"
        }
    }
}

/// What a read-only look at the competitor's settings found.
enum CompetitorHotkey: Equatable {
    case chords([CompetitorChord])
    /// File missing, unparsable, or no known location.
    case unreadable
}

// MARK: - Settings readers (pure)

enum CompetitorHotkeyReader {
    /// Wispr Flow: `prefs.user.shortcuts` is `{"63": "ptt", "49+63": "popo", ...}`
    /// with `+`-joined Carbon key codes as keys.
    static func wispr(config: Any) -> CompetitorHotkey {
        guard let root = config as? [String: Any],
            let prefs = root["prefs"] as? [String: Any],
            let user = prefs["user"] as? [String: Any],
            let shortcuts = user["shortcuts"] as? [String: Any]
        else { return .unreadable }
        let chords = shortcuts.keys.sorted().compactMap { key -> CompetitorChord? in
            let parts = key.split(separator: "+").map { Int($0.trimmingCharacters(in: .whitespaces)) }
            guard !parts.isEmpty, !parts.contains(nil) else { return nil }
            return CompetitorChord.fromKeyCodes(parts.compactMap { $0 })
        }
        return .chords(chords)
    }

    static func wisprOpensAtLogin(config: Any) -> Bool? {
        guard let root = config as? [String: Any],
            let prefs = root["prefs"] as? [String: Any],
            let user = prefs["user"] as? [String: Any]
        else { return nil }
        return user["openAtLogin"] as? Bool
    }

    /// Aqua Voice: `hotkeys` is `[{"keys": "Fn", "action": "activate"}, ...]`.
    static func aqua(settings: Any) -> CompetitorHotkey {
        guard let root = settings as? [String: Any],
            let hotkeys = root["hotkeys"] as? [[String: Any]]
        else { return .unreadable }
        let chords = hotkeys.compactMap { entry -> CompetitorChord? in
            guard let keys = entry["keys"] as? String, !keys.isEmpty else { return nil }
            return CompetitorChord.fromTokens(keys.split(separator: "+").map(String.init))
        }
        return .chords(chords)
    }

    /// VoiceInk: the `Shortcut_primaryRecording` default is JSON with a key code
    /// and either AppKit or Carbon modifier bits.
    static func voiceInk(shortcut: Any?) -> CompetitorHotkey {
        var object: [String: Any]?
        if let dictionary = shortcut as? [String: Any] {
            object = dictionary
        } else {
            let data: Data?
            if let string = shortcut as? String {
                data = string.data(using: .utf8)
            } else {
                data = shortcut as? Data
            }
            if let data {
                object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
            }
        }
        guard let object else { return .unreadable }
        let code = (object["keyCode"] as? Int) ?? (object["carbonKeyCode"] as? Int)
        guard let code else { return .unreadable }
        if let flags = object["modifierFlags"] as? Int {
            return .chords([CompetitorChord.fromKeyCode(code, modifierFlags: flags)])
        }
        let carbon = (object["carbonModifiers"] as? Int) ?? 0
        return .chords([CompetitorChord.fromKeyCode(code, carbonModifiers: carbon)])
    }

    /// Read from disk. Read-only: Phonon never writes a competitor's files.
    static func read(_ source: CompetingApp.HotkeySource, home: URL = FileManager.default.homeDirectoryForCurrentUser) -> CompetitorHotkey {
        let support = home.appendingPathComponent("Library/Application Support", isDirectory: true)
        switch source {
        case .wisprConfig:
            guard let json = json(at: support.appendingPathComponent("Wispr Flow/config.json"))
            else { return .unreadable }
            return wispr(config: json)
        case .aquaSettings:
            guard let json = json(at: support.appendingPathComponent("Aqua Voice/settings.json"))
            else { return .unreadable }
            return aqua(settings: json)
        case .voiceInkDefaults:
            let domain = "com.prakashjoshipax.VoiceInk"
            let key = "Shortcut_primaryRecording"
            if let value = UserDefaults(suiteName: domain)?.object(forKey: key) {
                return voiceInk(shortcut: value)
            }
            let container = home.appendingPathComponent(
                "Library/Containers/\(domain)/Data/Library/Preferences/\(domain).plist")
            if let data = try? Data(contentsOf: container),
                let plist = try? PropertyListSerialization.propertyList(from: data, format: nil)
                    as? [String: Any]
            {
                return voiceInk(shortcut: plist[key])
            }
            return .unreadable
        case .unknown:
            return .unreadable
        }
    }

    static func wisprOpensAtLogin(home: URL = FileManager.default.homeDirectoryForCurrentUser) -> Bool? {
        let url = home.appendingPathComponent("Library/Application Support/Wispr Flow/config.json")
        guard let json = json(at: url) else { return nil }
        return wisprOpensAtLogin(config: json)
    }

    private static func json(at url: URL) -> Any? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONSerialization.jsonObject(with: data)
    }
}

// MARK: - Decision (pure)

enum CompetitorPolicy {
    enum Reason: Equatable {
        /// The competitor's stored shortcut fires Phonon's shortcut too.
        case sameHotkey(CompetitorChord)
        /// Phonon could not read the shortcut, so a clash cannot be ruled out.
        case unreadable
    }

    /// Nil means the two shortcuts cannot both fire, so there is nothing to ask.
    static func conflict(hotkey: CompetitorHotkey, phononMode: String) -> Reason? {
        let sources = ShortcutPolicy.sources(for: phononMode)
        switch hotkey {
        case .unreadable:
            return .unreadable
        case .chords(let chords):
            for chord in chords {
                switch chord.kind {
                case .fn where sources.contains("fn"):
                    return .sameHotkey(chord)
                case .option where sources.contains("right-option"):
                    return .sameHotkey(chord)
                case .controlSpace where sources.contains("control-space"):
                    return .sameHotkey(chord)
                default:
                    continue
                }
            }
            return nil
        }
    }

    /// Ask once per session per app, plus at most one more time when the app
    /// relaunches (SPEC step 5). Never for an app the user muted with "Don't
    /// ask again".
    static func shouldPrompt(
        bundleID: String, muted: [String], promptCount: Int, relaunch: Bool
    ) -> Bool {
        if muted.contains(bundleID) { return false }
        if promptCount == 0 { return true }
        return relaunch && promptCount < 2
    }

    /// A helper process may be signalled only when both its name and its path
    /// match the table entry.
    static func helperMatches(
        _ helper: CompetitorHelperProcess, processName: String, executablePath: String
    ) -> Bool {
        processName == helper.name && executablePath.contains(helper.pathFragment)
    }
}

// MARK: - Microphone activity

enum MicrophoneActivity {
    enum State: Equatable {
        case free
        /// Another process is pulling from an input device: someone may be
        /// mid-dictation.
        case busyElsewhere
        /// Phonon's own instant-mic prewarm holds the device, which masks the
        /// "running somewhere" flag.
        case heldByPhonon
    }

    static func state() -> State {
        var ownsAny = false
        for device in CoreAudioInputDevices.all() {
            let somewhere = flag(device.id, kAudioDevicePropertyDeviceIsRunningSomewhere)
            let ours = flag(device.id, kAudioDevicePropertyDeviceIsRunning)
            if ours { ownsAny = true }
            if somewhere && !ours { return .busyElsewhere }
        }
        return ownsAny ? .heldByPhonon : .free
    }

    private static func flag(_ id: AudioDeviceID, _ selector: AudioObjectPropertySelector) -> Bool {
        var address = AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var value: UInt32 = 0
        var size = UInt32(MemoryLayout<UInt32>.size)
        guard AudioObjectGetPropertyData(id, &address, 0, nil, &size, &value) == noErr else {
            return false
        }
        return value != 0
    }
}

// MARK: - Bare processes

enum CompetitorProcesses {
    /// PIDs of running processes that match a table helper by name and path.
    static func pids(matching helper: CompetitorHelperProcess) -> [pid_t] {
        let count = proc_listallpids(nil, 0)
        guard count > 0 else { return [] }
        var pids = [pid_t](repeating: 0, count: Int(count) * 2)
        let filled = proc_listallpids(&pids, Int32(pids.count * MemoryLayout<pid_t>.size))
        guard filled > 0 else { return [] }
        var matches: [pid_t] = []
        for pid in pids.prefix(Int(filled)) where pid > 0 {
            var nameBuffer = [CChar](repeating: 0, count: Int(MAXPATHLEN))
            guard proc_name(pid, &nameBuffer, UInt32(nameBuffer.count)) > 0 else { continue }
            let name = String(cString: nameBuffer)
            guard name == helper.name else { continue }
            var pathBuffer = [CChar](repeating: 0, count: Int(MAXPATHLEN) * 4)
            guard proc_pidpath(pid, &pathBuffer, UInt32(pathBuffer.count)) > 0 else { continue }
            let path = String(cString: pathBuffer)
            if CompetitorPolicy.helperMatches(helper, processName: name, executablePath: path) {
                matches.append(pid)
            }
        }
        return matches
    }
}

// MARK: - Coordinator

/// One running competitor plus why Phonon wants to ask about it.
struct CompetitorFinding {
    let app: CompetingApp
    let reason: CompetitorPolicy.Reason
    let running: [NSRunningApplication]
}

@MainActor
final class CompetitorCoordinator {
    /// Called after a consented quit finished, so the owner can put Phonon's
    /// event tap back at the head of the chain.
    var onQuitFinished: (() -> Void)?

    private let store: NativeAppStore
    /// Prompts shown per primary bundle ID this session.
    private var promptCounts: [String: Int] = [:]
    private var lastShortcutMode: String
    private var pendingFindings: [CompetitorFinding] = []
    private var launchObserver: NSObjectProtocol?
    private var prompting = false

    /// SPEC step 3: poll `isTerminated` at 250 ms, force-quit at 5 s.
    static let terminatePollInterval: TimeInterval = 0.25
    static let forceTerminateAfter: TimeInterval = 5
    /// SPEC step 4: never quit mid-dictation. Wait this long for the mic.
    static let microphoneWaitLimit: TimeInterval = 10

    init(store: NativeAppStore) {
        self.store = store
        self.lastShortcutMode = store.settings.shortcutMode
    }

    deinit {
        if let launchObserver {
            NSWorkspace.shared.notificationCenter.removeObserver(launchObserver)
        }
    }

    // MARK: Triggers

    /// SPEC step 1: subscribe to relaunches.
    func subscribeToLaunches() {
        guard launchObserver == nil else { return }
        launchObserver = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didLaunchApplicationNotification, object: nil, queue: .main
        ) { [weak self] notification in
            guard
                let app = notification.userInfo?[NSWorkspace.applicationUserInfoKey]
                    as? NSRunningApplication,
                let bundleID = app.bundleIdentifier,
                CompetingAppTable.app(forBundleID: bundleID) != nil
            else { return }
            NSLog("phonon competitors: \(bundleID) launched")
            // Let the app finish starting before its settings are read.
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                Task { @MainActor in self?.check(trigger: "relaunch") }
            }
        }
    }

    /// SPEC step 1: rescan on every hotkey change. A new shortcut changes the
    /// verdict, so earlier "not now" answers no longer apply.
    func settingsChanged() {
        let mode = store.settings.shortcutMode
        guard mode != lastShortcutMode else { return }
        lastShortcutMode = mode
        promptCounts.removeAll()
        check(trigger: "hotkey-change")
    }

    /// Scan now and prompt if anything qualifies. Runs the alert modally.
    func check(trigger: String) {
        let findings = scan(ignoreMuted: false, relaunch: trigger == "relaunch")
        guard !findings.isEmpty else { return }
        NSLog(
            "phonon competitors: \(trigger) found \(findings.map(\.app.name).joined(separator: ", "))")
        present(findings, manual: false)
    }

    /// Called as a dictation starts. Only records the finding; the prompt is
    /// shown once the dictation is over, so a sheet never lands mid-sentence.
    func noteDictationStart() {
        let findings = scan(ignoreMuted: false)
        guard !findings.isEmpty else { return }
        pendingFindings = findings
        NSLog(
            "phonon competitors: found during dictation, prompt deferred: "
                + findings.map(\.app.name).joined(separator: ", "))
    }

    var hasPendingPrompt: Bool { !pendingFindings.isEmpty }

    /// Show the prompt recorded by `noteDictationStart` once Phonon is idle.
    func presentPendingPrompt() {
        let findings = pendingFindings
        pendingFindings = []
        // The running set may have changed since the dictation started.
        let stillRunning = findings.filter { finding in
            finding.running.contains { !$0.isTerminated }
        }
        guard !stillRunning.isEmpty else { return }
        present(stillRunning, manual: false)
    }

    /// Menu action: ask about every running competitor, muted or not, with no
    /// hotkey test. The user asked, so the answer is what is running.
    func checkManually() -> Bool {
        let findings = scan(ignoreMuted: true, requireConflict: false)
        guard !findings.isEmpty else { return false }
        present(findings, manual: true)
        return true
    }

    // MARK: Scan

    func scan(
        ignoreMuted: Bool, requireConflict: Bool = true, relaunch: Bool = false
    ) -> [CompetitorFinding] {
        let running = NSWorkspace.shared.runningApplications
        let mode = store.settings.shortcutMode
        let muted = store.settings.competitorQuitMuted
        var findings: [CompetitorFinding] = []
        for app in CompetingAppTable.apps {
            let instances = running.filter { instance in
                guard let id = instance.bundleIdentifier else { return false }
                return app.bundleIDs.contains(id) && !instance.isTerminated
            }
            guard !instances.isEmpty else { continue }
            if !ignoreMuted,
                !CompetitorPolicy.shouldPrompt(
                    bundleID: app.primaryBundleID, muted: muted,
                    promptCount: promptCounts[app.primaryBundleID, default: 0],
                    relaunch: relaunch)
            {
                continue
            }
            let hotkey = CompetitorHotkeyReader.read(app.hotkeySource)
            let reason: CompetitorPolicy.Reason
            if let conflict = CompetitorPolicy.conflict(hotkey: hotkey, phononMode: mode) {
                reason = conflict
            } else if requireConflict {
                NSLog("phonon competitors: \(app.name) runs on a different shortcut; no prompt")
                continue
            } else {
                reason = .unreadable
            }
            findings.append(CompetitorFinding(app: app, reason: reason, running: instances))
        }
        return findings
    }

    // MARK: Prompt

    private func present(_ findings: [CompetitorFinding], manual: Bool) {
        guard !prompting else { return }
        prompting = true
        defer { prompting = false }
        for finding in findings {
            promptCounts[finding.app.primaryBundleID, default: 0] += 1
        }

        let names = findings.map(\.app.name)
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText =
            names.count == 1
            ? "Quit \(names[0]) while Phonon runs?"
            : "Quit \(names.count) other dictation apps while Phonon runs?"
        alert.informativeText = Self.informativeText(for: findings, manual: manual)
        alert.addButton(withTitle: names.count == 1 ? "Quit \(names[0])" : "Quit Them")
        alert.addButton(withTitle: "Not Now")
        alert.addButton(withTitle: "Don't Ask Again")
        NSApp.activate(ignoringOtherApps: true)
        switch alert.runModal() {
        case .alertFirstButtonReturn:
            NSLog("phonon competitors: consent to quit \(names.joined(separator: ", "))")
            quit(findings)
        case .alertThirdButtonReturn:
            let ids = findings.map(\.app.primaryBundleID)
            NSLog("phonon competitors: muted \(ids.joined(separator: ", "))")
            store.updateSettings { settings in
                for id in ids where !settings.competitorQuitMuted.contains(id) {
                    settings.competitorQuitMuted.append(id)
                }
            }
        default:
            NSLog("phonon competitors: user kept \(names.joined(separator: ", "))")
        }
    }

    static func informativeText(for findings: [CompetitorFinding], manual: Bool) -> String {
        var lines: [String] = []
        for finding in findings {
            switch finding.reason {
            case .sameHotkey(let chord):
                lines.append(
                    "\(finding.app.name) is running and also records on \(chord.label), "
                        + "so both apps would insert text on every dictation.")
            case .unreadable where manual:
                lines.append("\(finding.app.name) is running.")
            case .unreadable:
                lines.append(
                    "\(finding.app.name) is running. Phonon could not read its shortcut, "
                        + "so it may also record on yours.")
            }
            if let note = finding.app.returnNote {
                lines.append(note)
            }
        }
        lines.append(
            "Phonon only quits the app; it never changes the app's login items or settings. "
                + "To keep both, change Phonon's shortcut in Settings › Shortcut.")
        return lines.joined(separator: "\n\n")
    }

    // MARK: Quit

    private func quit(_ findings: [CompetitorFinding]) {
        waitForMicrophone(deadline: Date().addingTimeInterval(Self.microphoneWaitLimit)) {
            [weak self] free in
            guard let self else { return }
            guard free else {
                let names = findings.map(\.app.name).joined(separator: ", ")
                NSLog("phonon competitors: microphone still busy; not quitting \(names)")
                self.store.lastError =
                    "Phonon did not quit \(names): a microphone is still in use. "
                    + "Finish the other dictation, then choose Phonon › Quit competing "
                    + "dictation apps from the menu bar."
                return
            }
            let mainApps = findings.flatMap(\.running)
            self.terminate(
                mainApps, forceAfter: Self.forceTerminateAfter, label: "main"
            ) { [weak self] in
                self?.terminateHelpers(of: findings.map(\.app))
                self?.onQuitFinished?()
            }
        }
    }

    /// SPEC step 4: another process pulling from an input device may be
    /// mid-dictation. Wait for it, up to the limit.
    private func waitForMicrophone(deadline: Date, completion: @escaping (Bool) -> Void) {
        switch MicrophoneActivity.state() {
        case .free:
            completion(true)
        case .heldByPhonon:
            NSLog("phonon competitors: mic held by Phonon's prewarm; activity signal masked")
            completion(true)
        case .busyElsewhere:
            guard Date() < deadline else {
                completion(false)
                return
            }
            NSLog("phonon competitors: microphone busy elsewhere; waiting")
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                Task { @MainActor in
                    self.waitForMicrophone(deadline: deadline, completion: completion)
                }
            }
        }
    }

    /// `terminate()`, poll `isTerminated`, `forceTerminate()` at the limit.
    private func terminate(
        _ apps: [NSRunningApplication], forceAfter: TimeInterval, label: String,
        completion: @escaping () -> Void
    ) {
        let live = apps.filter { !$0.isTerminated }
        guard !live.isEmpty else {
            completion()
            return
        }
        for app in live {
            let name = app.bundleIdentifier ?? "?"
            let sent = app.terminate()
            NSLog("phonon competitors: terminate \(label) \(name) pid \(app.processIdentifier) sent=\(sent)")
        }
        let start = Date()
        var forced = false
        func poll() {
            let remaining = live.filter { !$0.isTerminated }
            if remaining.isEmpty {
                NSLog("phonon competitors: \(label) exited after \(Int(Date().timeIntervalSince(start) * 1000)) ms")
                completion()
                return
            }
            if !forced, Date().timeIntervalSince(start) >= forceAfter {
                forced = true
                for app in remaining {
                    let sent = app.forceTerminate()
                    NSLog(
                        "phonon competitors: forceTerminate \(label) "
                            + "\(app.bundleIdentifier ?? "?") pid \(app.processIdentifier) sent=\(sent)")
                }
            } else if forced, Date().timeIntervalSince(start) >= forceAfter + 2 {
                NSLog("phonon competitors: \(label) did not exit after forceTerminate; giving up")
                completion()
                return
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + Self.terminatePollInterval) {
                Task { @MainActor in poll() }
            }
        }
        poll()
    }

    /// SPEC step 3: rescan for orphaned helpers once the main app is gone.
    private func terminateHelpers(of apps: [CompetingApp]) {
        let running = NSWorkspace.shared.runningApplications
        let helperIDs = Set(apps.flatMap(\.helperBundleIDs))
        let helperApps = running.filter { instance in
            guard let id = instance.bundleIdentifier else { return false }
            return helperIDs.contains(id) && !instance.isTerminated
        }
        if !helperApps.isEmpty {
            terminate(helperApps, forceAfter: 2, label: "helper") {}
        }
        for helper in apps.flatMap(\.helperProcesses) {
            for pid in CompetitorProcesses.pids(matching: helper) {
                let result = kill(pid, SIGTERM)
                NSLog("phonon competitors: SIGTERM \(helper.name) pid \(pid) result=\(result)")
            }
        }
    }
}
