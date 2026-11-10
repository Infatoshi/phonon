import AppKit
import ApplicationServices
import AVFoundation
import Foundation
import ScreenCaptureKit
import Security
import ServiceManagement

struct NativeSettings: Codable, Equatable {
    /// Schema 2 made screen OCR and recording retention opt-in. Schema 1 files
    /// keep the values their owner already saw in Settings. Schema 3 turned
    /// meeting detection on by default (see `meetingCapture`).
    static let currentSchemaVersion = 3
    /// 0 keeps recordings until they are deleted by hand.
    static let keepRecordingsForever = 0

    var schemaVersion: Int
    var streaming: Bool
    var localHistory: Bool
    var screenContext: Bool
    var microphonePriority: [String]
    /// CoreAudio UIDs of every input Phonon has already seen, so the
    /// new-microphone prompt asks once per device. nil (key absent: a fresh
    /// install or an upgrade) is seeded with what is connected, without asking.
    var microphonesSeen: [String]?
    var instantMic: Bool
    var soundFeedback: Bool
    var shortcutMode: String
    var privacyChoiceMade: Bool
    var historyRetentionDays: Int
    /// Bundle IDs of competing dictation apps the owner answered "Don't ask
    /// again" for. See `CompetingApps.swift`.
    var competitorQuitMuted: [String]
    /// Watch the field after insertion and offer what the user corrected.
    var editLearning: Bool
    /// Approved style preferences by app category, e.g. `["chat":
    /// ["lowercase_start"]]`. Written only when the user says yes.
    var stylePreferences: [String: [String]]
    /// Apps (bundle ids) and sites (hosts) whose screen, text fields and
    /// edits Phonon never reads. Signal and Discord by default.
    var screenContextDenylist: [String]
    /// Send the text around the cursor to the local correction model.
    var textboxContext: Bool
    /// Notice calls and offer to record meeting notes (mic + call audio). On by
    /// default since schema 3: it only asks, it never records by itself.
    var meetingCapture: Bool
    /// Start recording a detected call without asking. Off by default:
    /// recording other people needs their consent in some places.
    var meetingAutoStart: Bool
    /// Ask GitHub once a day whether a newer release is out (SPEC "Update check").
    var checkForUpdates: Bool
    /// The last version whose update alert was shown, so each release is
    /// announced once; the menu and sidebar keep offering it afterwards.
    var updateAnnounced: String
    /// Live text: show the corrected text so far above the capsule while the key is held.
    /// The engine reads it from settings.json at every dictation start. On by default
    /// since 1.0.
    var liveText: Bool
    /// How live text streams in: "fade" (default) or "blur". Unknown values read as fade.
    var liveTextStyle: LiveTextStyle
    /// Settings > Advanced > Writing style: follow the app (default) or one mode
    /// everywhere. The engine reads it from settings.json for every request.
    var writingStyle: WritingStyle
    /// Streaming refine; nil leaves it to the engine (on with 16 GB or more).
    var streamRefine: Bool?
    /// What the clipboard holds after a paste: the dictation (default, so ⌘V pastes it
    /// again) or what was copied before. Asked at first run with the privacy choices.
    var clipboardAfterPaste: ClipboardAfterPaste
    /// Show the Dock icon. Off keeps Phonon in the menu bar only (an accessory app).
    var showInDock: Bool
    /// Keep the small capsule on screen between dictations. Off shows it only while
    /// dictating (and while a meeting records, so the red light stays visible).
    var showIdleCapsule: Bool
    /// Other audio while the key is held: left alone, lowered (default) or muted.
    var backgroundAudio: BackgroundAudio
    /// Per-app writing mode, bundle id -> refiner mode (`minimal`, `casual`,
    /// `formal`, `prompt`). Wins over `writing_style` and the app's category.
    var writingModesByApp: [String: String]
    /// Your own OpenAI-compatible endpoint refines instead of the local refiner
    /// (issue #24). Off by default; the key lives in the Keychain, never here.
    var byokEnabled: Bool
    var byokBaseURL: String
    var byokModel: String
    /// The WebSocket API: another device (a phone keyboard) dictates through this Mac's
    /// engine. Off by default. The engine reads these keys from settings.json and follows a
    /// change without a restart. The token is kept here rather than in the Keychain: an app
    /// built and signed on this Mac would otherwise ask for the login password at every
    /// launch after each rebuild.
    var websocketAPI: Bool
    var websocketAPIPort: Int
    var websocketAPIToken: String

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case streaming
        case localHistory = "local_history"
        case screenContext = "screen_context"
        case microphonePriority = "microphone_priority"
        case microphonesSeen = "microphones_seen"
        case instantMic = "instant_mic"
        case soundFeedback = "sound_feedback"
        case shortcutMode = "shortcut_mode"
        case privacyChoiceMade = "privacy_choice_made"
        case historyRetentionDays = "history_retention_days"
        case competitorQuitMuted = "competitor_quit_muted"
        case editLearning = "edit_learning"
        case stylePreferences = "style_preferences"
        case screenContextDenylist = "screen_context_denylist"
        case textboxContext = "textbox_context"
        case meetingCapture = "meeting_capture"
        case meetingAutoStart = "meeting_auto_start"
        case checkForUpdates = "check_for_updates"
        case updateAnnounced = "update_announced"
        case liveText = "live_text"
        case liveTextStyle = "live_text_style"
        case writingStyle = "writing_style"
        case streamRefine = "stream_refine"
        case clipboardAfterPaste = "clipboard_after_paste"
        case showInDock = "show_in_dock"
        case showIdleCapsule = "show_idle_capsule"
        case backgroundAudio = "background_audio"
        case writingModesByApp = "writing_modes_by_app"
        case byokEnabled = "byok_enabled"
        case byokBaseURL = "byok_base_url"
        case byokModel = "byok_model"
        case websocketAPI = "websocket_api"
        case websocketAPIPort = "websocket_api_port"
        case websocketAPIToken = "websocket_api_token"
    }

    init(
        schemaVersion: Int = NativeSettings.currentSchemaVersion,
        streaming: Bool = true,
        localHistory: Bool = false,
        screenContext: Bool = false,
        microphonePriority: [String] = [],
        microphonesSeen: [String]? = nil,
        instantMic: Bool = true,
        soundFeedback: Bool = false,
        shortcutMode: String = "fn",
        privacyChoiceMade: Bool = false,
        historyRetentionDays: Int = NativeSettings.keepRecordingsForever,
        competitorQuitMuted: [String] = [],
        editLearning: Bool = true,
        stylePreferences: [String: [String]] = [:],
        screenContextDenylist: [String] = ContextPolicy.defaultDenylist,
        textboxContext: Bool = false,
        meetingCapture: Bool = true,
        meetingAutoStart: Bool = false,
        checkForUpdates: Bool = true,
        updateAnnounced: String = "",
        liveText: Bool = false,
        liveTextStyle: LiveTextStyle = .fade,
        writingStyle: WritingStyle = .app,
        streamRefine: Bool? = nil,
        clipboardAfterPaste: ClipboardAfterPaste = .dictation,
        showInDock: Bool = true,
        showIdleCapsule: Bool = true,
        backgroundAudio: BackgroundAudio = .lower,
        writingModesByApp: [String: String] = [:],
        byokEnabled: Bool = false,
        byokBaseURL: String = "",
        byokModel: String = "",
        websocketAPI: Bool = false,
        websocketAPIPort: Int = WebSocketAPI.defaultPort,
        websocketAPIToken: String = ""
    ) {
        self.schemaVersion = schemaVersion
        self.streaming = streaming
        self.localHistory = localHistory
        self.screenContext = screenContext
        self.microphonePriority = microphonePriority
        self.microphonesSeen = microphonesSeen
        self.instantMic = instantMic
        self.soundFeedback = soundFeedback
        self.shortcutMode = shortcutMode
        self.privacyChoiceMade = privacyChoiceMade
        self.historyRetentionDays = historyRetentionDays
        self.competitorQuitMuted = competitorQuitMuted
        self.editLearning = editLearning
        self.stylePreferences = stylePreferences
        self.screenContextDenylist = screenContextDenylist
        self.textboxContext = textboxContext
        self.meetingCapture = meetingCapture
        self.meetingAutoStart = meetingAutoStart
        self.checkForUpdates = checkForUpdates
        self.updateAnnounced = updateAnnounced
        self.liveText = liveText
        self.liveTextStyle = liveTextStyle
        self.writingStyle = writingStyle
        self.streamRefine = streamRefine
        self.clipboardAfterPaste = clipboardAfterPaste
        self.showInDock = showInDock
        self.showIdleCapsule = showIdleCapsule
        self.backgroundAudio = backgroundAudio
        self.writingModesByApp = writingModesByApp
        self.byokEnabled = byokEnabled
        self.byokBaseURL = byokBaseURL
        self.byokModel = byokModel
        self.websocketAPI = websocketAPI
        self.websocketAPIPort = websocketAPIPort
        self.websocketAPIToken = websocketAPIToken
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let storedSchema = try values.decodeIfPresent(Int.self, forKey: .schemaVersion) ?? 1
        schemaVersion = storedSchema
        streaming = try values.decodeIfPresent(Bool.self, forKey: .streaming) ?? true
        // Before schema 2 both defaulted to on, so an absent key on an existing
        // install means the owner was running with it enabled.
        let legacyDefault = storedSchema < 2
        localHistory = try values.decodeIfPresent(Bool.self, forKey: .localHistory) ?? legacyDefault
        screenContext =
            try values.decodeIfPresent(Bool.self, forKey: .screenContext) ?? legacyDefault
        microphonePriority = try values.decodeIfPresent([String].self, forKey: .microphonePriority)
            ?? []
        microphonesSeen = try? values.decodeIfPresent([String].self, forKey: .microphonesSeen)
        instantMic = try values.decodeIfPresent(Bool.self, forKey: .instantMic) ?? true
        // Silent unless asked for. Dictation is held down mid-sentence and a
        // cue on every press is intrusive, so nobody gets one by default.
        soundFeedback = try values.decodeIfPresent(Bool.self, forKey: .soundFeedback) ?? false
        shortcutMode = try values.decodeIfPresent(String.self, forKey: .shortcutMode) ?? "fn"
        // An existing install already chose these in Settings; do not re-prompt.
        privacyChoiceMade =
            try values.decodeIfPresent(Bool.self, forKey: .privacyChoiceMade) ?? legacyDefault
        historyRetentionDays =
            try values.decodeIfPresent(Int.self, forKey: .historyRetentionDays)
            ?? NativeSettings.keepRecordingsForever
        competitorQuitMuted =
            try values.decodeIfPresent([String].self, forKey: .competitorQuitMuted) ?? []
        editLearning = try values.decodeIfPresent(Bool.self, forKey: .editLearning) ?? true
        stylePreferences =
            try values.decodeIfPresent([String: [String]].self, forKey: .stylePreferences) ?? [:]
        // Absent means never edited: the defaults apply. An empty list is the
        // user's own choice and stays empty.
        screenContextDenylist =
            try values.decodeIfPresent([String].self, forKey: .screenContextDenylist)
            ?? ContextPolicy.defaultDenylist
        // Off until the user turns it on: it reads what they are typing.
        textboxContext = try values.decodeIfPresent(Bool.self, forKey: .textboxContext) ?? false
        // Every save writes every key, so a stored false before schema 3 is the old
        // off-by-default, indistinguishable from a choice. Those files turn it on once;
        // the next save writes schema 3, and from then on a false is the owner's.
        // When the migration is what turns detection on, auto-start goes off with it: an
        // owner who once enabled auto-start and then switched detection off must not start
        // recording calls unprompted after the upgrade (they can turn it back on).
        let storedCapture = try values.decodeIfPresent(Bool.self, forKey: .meetingCapture)
        let storedAutoStart =
            try values.decodeIfPresent(Bool.self, forKey: .meetingAutoStart) ?? false
        let forcedOn = storedSchema < 3 && storedCapture != true
        meetingCapture = storedSchema < 3 ? true : (storedCapture ?? true)
        meetingAutoStart = forcedOn ? false : storedAutoStart
        checkForUpdates = try values.decodeIfPresent(Bool.self, forKey: .checkForUpdates) ?? true
        updateAnnounced =
            try values.decodeIfPresent(String.self, forKey: .updateAnnounced) ?? ""
        liveText = try values.decodeIfPresent(Bool.self, forKey: .liveText) ?? false
        liveTextStyle =
            (try? values.decodeIfPresent(String.self, forKey: .liveTextStyle))
            .flatMap { $0 }.flatMap(LiveTextStyle.init(rawValue:)) ?? .fade
        writingStyle =
            (try? values.decodeIfPresent(String.self, forKey: .writingStyle))
            .flatMap { $0 }.flatMap(WritingStyle.init(rawValue:)) ?? .app
        streamRefine = try? values.decodeIfPresent(Bool.self, forKey: .streamRefine)
        clipboardAfterPaste =
            (try? values.decodeIfPresent(String.self, forKey: .clipboardAfterPaste))
            .flatMap { $0 }.flatMap(ClipboardAfterPaste.init(rawValue:)) ?? .dictation
        showInDock = try values.decodeIfPresent(Bool.self, forKey: .showInDock) ?? true
        showIdleCapsule =
            try values.decodeIfPresent(Bool.self, forKey: .showIdleCapsule) ?? true
        backgroundAudio =
            (try? values.decodeIfPresent(String.self, forKey: .backgroundAudio))
            .flatMap { $0 }.flatMap(BackgroundAudio.init(rawValue:)) ?? .lower        // Unknown modes are dropped on read, so a hand edit cannot pin a mode the
        // refiner does not have.
        writingModesByApp =
            ((try? values.decodeIfPresent([String: String].self, forKey: .writingModesByApp))
            ?? nil ?? [:]).filter { AppWritingMode(rawValue: $0.value) != nil }
        byokEnabled = (try? values.decodeIfPresent(Bool.self, forKey: .byokEnabled)) ?? false
        byokBaseURL = ((try? values.decodeIfPresent(String.self, forKey: .byokBaseURL)) ?? nil) ?? ""
        byokModel = ((try? values.decodeIfPresent(String.self, forKey: .byokModel)) ?? nil) ?? ""
        websocketAPI = (try? values.decodeIfPresent(Bool.self, forKey: .websocketAPI)) ?? false
        // A port the engine could not listen on reads as the default.
        websocketAPIPort =
            ((try? values.decodeIfPresent(Int.self, forKey: .websocketAPIPort)) ?? nil)
            .flatMap { WebSocketAPI.validPort($0) ? $0 : nil } ?? WebSocketAPI.defaultPort
        websocketAPIToken =
            ((try? values.decodeIfPresent(String.self, forKey: .websocketAPIToken)) ?? nil) ?? ""
    }
}

/// One dictation another device made over the WebSocket API.
struct WebSocketDictation: Equatable {
    let end: Date
    let words: Int
    /// Length of its audio; 0 when unknown.
    let audioSeconds: Double
}

/// Home's graph of WebSocket dictation: words per minute for each of the last 60 minutes.
enum WebSocketActivity {
    static let minutes = 60
    /// How long a dictation is kept for the graph.
    static let keep: TimeInterval = 61 * 60

    /// Words in each of the last [minutes] clock minutes, oldest first, the current (partial)
    /// minute last. A dictation's words are spread evenly over the minutes its audio spanned,
    /// so a three-minute dictation shows as three minutes of speech, not one spike.
    static func perMinute(_ dictations: [WebSocketDictation], now: Date) -> [Double] {
        var buckets = [Double](repeating: 0, count: minutes)
        let current = floor(now.timeIntervalSince1970 / 60)
        let first = current - Double(minutes - 1)  // the oldest minute shown
        func add(_ words: Double, minute: Double) {
            let index = Int(minute - first)
            if index >= 0 && index < minutes { buckets[index] += words }
        }
        for dictation in dictations where dictation.words > 0 {
            let end = dictation.end.timeIntervalSince1970
            let duration = max(dictation.audioSeconds, 0)
            guard duration > 0 else {
                add(Double(dictation.words), minute: floor(end / 60))
                continue
            }
            let start = end - duration
            var minute = floor(start / 60)
            while minute * 60 < end {
                let overlap = min(end, (minute + 1) * 60) - max(start, minute * 60)
                if overlap > 0 { add(Double(dictation.words) * overlap / duration, minute: minute) }
                minute += 1
            }
        }
        return buckets
    }

    /// Words over the last [minutes] minutes, and the speaking rate across those dictations
    /// (words per minute of audio; nil when no audio length is known).
    static func summary(_ dictations: [WebSocketDictation], now: Date) -> (words: Int, rate: Int?) {
        let recent = dictations.filter {
            now.timeIntervalSince($0.end) < Double(minutes) * 60 && $0.words > 0
        }
        let words = recent.reduce(0) { $0 + $1.words }
        let timed = recent.filter { $0.audioSeconds > 0 }
        let seconds = timed.reduce(0) { $0 + $1.audioSeconds }
        guard seconds > 0 else { return (words, nil) }
        let timedWords = timed.reduce(0) { $0 + $1.words }
        return (words, Int((Double(timedWords) * 60 / seconds).rounded()))
    }

    /// The graph's top: the busiest minute rounded up to a multiple of 20, at least 40.
    static func scale(_ perMinute: [Double]) -> Double {
        let peak = perMinute.max() ?? 0
        return max(40, (peak / 20).rounded(.up) * 20)
    }
}

/// The WebSocket API (engine `remote.rs`): its port, its token and this Mac's addresses.
enum WebSocketAPI {
    static let defaultPort = 8377
    /// Letters and digits that are not mistaken for one another when typed on a phone.
    static let tokenAlphabet = Array("abcdefghjkmnpqrstuvwxyz23456789")
    static let tokenLength = 24

    static func validPort(_ port: Int) -> Bool { (1...65_535).contains(port) }

    /// 24 characters from a 31-character alphabet: about 119 bits.
    static func newToken() -> String {
        var generator = SystemRandomNumberGenerator()
        return String(
            (0..<tokenLength).map { _ in tokenAlphabet.randomElement(using: &generator)! })
    }

    /// The switch turned on: the token stays as it is, or is made now if there is none.
    static func turnOn(_ settings: inout NativeSettings) {
        settings.websocketAPI = true
        if settings.websocketAPIToken.count < 16 { settings.websocketAPIToken = newToken() }
    }

    /// This Mac's IPv4 addresses a phone could reach: the local network's and a tailnet's
    /// (100.64.0.0/10), never loopback or link-local.
    static func reachableAddresses() -> [String] {
        var first: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&first) == 0, let first else { return [] }
        defer { freeifaddrs(first) }
        var addresses: [String] = []
        for pointer in sequence(first: first, next: { $0.pointee.ifa_next }) {
            let interface = pointer.pointee
            let flags = Int32(interface.ifa_flags)
            guard flags & IFF_UP != 0, flags & IFF_LOOPBACK == 0,
                let address = interface.ifa_addr, address.pointee.sa_family == UInt8(AF_INET)
            else { continue }
            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            guard
                getnameinfo(
                    address, socklen_t(address.pointee.sa_len), &host, socklen_t(host.count),
                    nil, 0, NI_NUMERICHOST) == 0
            else { continue }
            let text = String(cString: host)
            if !text.hasPrefix("169.254."), !addresses.contains(text) { addresses.append(text) }
        }
        return addresses
    }

    /// What the pairing QR code holds: `phonon://pair?url=ws://…&token=…`. The Android app's
    /// "Scan pairing code" (and any camera app, through its link) fills its fields from it.
    static func pairingURI(address: String, port: Int, token: String) -> String {
        var components = URLComponents()
        components.scheme = "phonon"
        components.host = "pair"
        components.queryItems = [
            URLQueryItem(name: "url", value: "ws://\(address):\(port)"),
            URLQueryItem(name: "token", value: token),
        ]
        // URLComponents leaves ":" and "/" as they are in a query; encode them so every
        // scanner hands the value over intact.
        var allowed = CharacterSet.urlQueryAllowed
        allowed.remove(charactersIn: ":/?&=+")
        let query = (components.queryItems ?? []).map { item in
            let value = item.value?.addingPercentEncoding(withAllowedCharacters: allowed) ?? ""
            return "\(item.name)=\(value)"
        }.joined(separator: "&")
        return "phonon://pair?\(query)"
    }

    /// The address to pair with when the user has not picked one: a tailnet address
    /// (100.64.0.0/10) first, then a home network's, then whatever comes first.
    static func preferredAddress(_ addresses: [String]) -> String? {
        func tailnet(_ address: String) -> Bool {
            let parts = address.split(separator: ".").compactMap { Int($0) }
            return parts.count == 4 && parts[0] == 100 && (64...127).contains(parts[1])
        }
        return addresses.first(where: tailnet)
            ?? addresses.first(where: { $0.hasPrefix("192.168.") })
            ?? addresses.first
    }

    /// What Settings shows under the switch: where it listens, or why it does not.
    static func statusLine(engineStatus: [String: Any]?, engineReady: Bool) -> String {
        guard engineReady else { return "Starts when the models are loaded." }
        guard let engineStatus else { return "Starting…" }
        if engineStatus["listening"] as? Bool == true {
            let address = engineStatus["address"] as? String ?? ""
            let port = address.split(separator: ":").last.map(String.init) ?? ""
            return port.isEmpty ? "Listening." : "Listening on port \(port)."
        }
        let reason = engineStatus["msg"] as? String ?? ""
        return reason.isEmpty ? "Not listening." : "Not listening: \(reason)."
    }
}

/// Your own model endpoint (issue #24): what the engine is told, and the API key
/// in the login Keychain (a generic password only Phonon reads without asking).
enum OwnEndpoint {
    static let keychainService = "com.infatoshi.phonon.byok"
    static let keychainAccount = "api_key"

    /// An http(s) URL with a host.
    static func validBaseURL(_ raw: String) -> Bool {
        let trimmed = raw.trimmingCharacters(in: .whitespaces)
        guard let url = URL(string: trimmed), let scheme = url.scheme?.lowercased(),
            scheme == "https" || scheme == "http", url.host?.isEmpty == false
        else { return false }
        return true
    }

    /// The engine's `byok` command: the endpoint when it is on and complete, else off.
    static func engineCommand(settings: NativeSettings, apiKey: String?) -> [String: Any] {
        let model = settings.byokModel.trimmingCharacters(in: .whitespaces)
        guard settings.byokEnabled, validBaseURL(settings.byokBaseURL), !model.isEmpty else {
            return ["cmd": "byok"]
        }
        return [
            "cmd": "byok",
            "base_url": settings.byokBaseURL.trimmingCharacters(in: .whitespaces),
            "model": model,
            "api_key": apiKey ?? "",
        ]
    }

    static func readKey(service: String = keychainService) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: keychainAccount,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
            let data = item as? Data
        else { return nil }
        return String(data: data, encoding: .utf8)
    }

    /// Saves (or with an empty key deletes) the API key. Returns the Keychain status.
    @discardableResult
    static func saveKey(_ key: String, service: String = keychainService) -> OSStatus {
        let match: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: keychainAccount,
        ]
        SecItemDelete(match as CFDictionary)
        let trimmed = key.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return errSecSuccess }
        var add = match
        add[kSecValueData as String] = Data(trimmed.utf8)
        add[kSecAttrLabel as String] = "Phonon model endpoint API key"
        add[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        return SecItemAdd(add as CFDictionary, nil)
    }
}

/// A per-app writing mode (settings.json `writing_modes_by_app`): the refiner's four
/// modes (SPEC "Modes"). The refiner has no verbatim mode.
enum AppWritingMode: String, CaseIterable, Identifiable {
    case minimal, casual, formal, prompt

    var id: String { rawValue }

    var title: String {
        switch self {
        case .minimal: return "Minimal"
        case .casual: return "Casual"
        case .formal: return "Formal"
        case .prompt: return "Prompt"
        }
    }
}

/// settings.json `clipboard_after_paste`.
enum ClipboardAfterPaste: String, Codable, CaseIterable, Identifiable {
    /// The dictation stays on the clipboard: ⌘V pastes it again.
    case dictation
    /// What was copied before comes back once the paste has landed.
    case restore

    var id: String { rawValue }

    var title: String {
        switch self {
        case .dictation: return "Your dictation"
        case .restore: return "What you copied before"
        }
    }
}

/// settings.json `writing_style`. `app` lets each app pick its mode (Slack and
/// email formal, Discord casual, terminals and AI tools as prompts); the others
/// use that one refiner mode everywhere.
enum WritingStyle: String, Codable, CaseIterable, Identifiable {
    case app, minimal, casual, formal

    var id: String { rawValue }

    var title: String {
        switch self {
        case .app: return "Follow the app"
        case .minimal: return "Minimal"
        case .casual: return "Casual"
        case .formal: return "Formal"
        }
    }
}

/// The app a dictation went into, as the engine keeps it in the corpus entry.
struct NativeAppContext: Codable, Equatable {
    var bundleID: String
    var appName: String?
    var urlHost: String?
    var category: String?

    enum CodingKeys: String, CodingKey {
        case bundleID = "bundle_id"
        case appName = "app_name"
        case urlHost = "url_host"
        case category
    }
}

struct NativeDictionaryEntry: Codable, Identifiable, Equatable {
    var phrase: String
    var replacement: String?
    var spokenForms: [String]
    var source: String
    var starred: Bool
    var usageCount: UInt64

    var id: String { "\(phrase.lowercased())\u{0}\(replacement ?? "")" }

    enum CodingKeys: String, CodingKey {
        case phrase, replacement, source, starred
        case spokenForms = "spoken_forms"
        case usageCount = "usage_count"
    }

    init(
        phrase: String,
        replacement: String? = nil,
        spokenForms: [String] = [],
        source: String = "phonon-app",
        starred: Bool = false,
        usageCount: UInt64 = 0
    ) {
        self.phrase = phrase
        self.replacement = replacement
        self.spokenForms = spokenForms
        self.source = source
        self.starred = starred
        self.usageCount = usageCount
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        phrase = try values.decode(String.self, forKey: .phrase)
        replacement = try values.decodeIfPresent(String.self, forKey: .replacement)
        spokenForms = try values.decodeIfPresent([String].self, forKey: .spokenForms) ?? []
        source = try values.decodeIfPresent(String.self, forKey: .source) ?? ""
        starred = try values.decodeIfPresent(Bool.self, forKey: .starred) ?? false
        usageCount = try values.decodeIfPresent(UInt64.self, forKey: .usageCount) ?? 0
    }
}

/// What one edit-learning approval changed in the dictionary.
struct LearnedTermUndo: Equatable {
    static let source = "edit-learning"

    enum Change: Equatable {
        case addedEntry(id: String)
        case addedSpokenForm(entryID: String, form: String)
    }

    let change: Change
}

struct NativeDictionaryFile: Codable {
    var schemaVersion: Int
    var updatedAtUnixMs: UInt64
    var entries: [NativeDictionaryEntry]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case updatedAtUnixMs = "updated_at_unix_ms"
        case entries
    }
}

struct NativeLlmMetadata: Codable, Equatable {
    var latencyMs: Double?
    var ttftMs: Double?
    var tokensPerSecond: Double?

    enum CodingKeys: String, CodingKey {
        case latencyMs = "latency_ms"
        case ttftMs = "ttft_ms"
        case tokensPerSecond = "tokens_per_second"
    }
}

struct NativeAppliedCorrection: Codable, Equatable {
    var count: UInt64

    enum CodingKeys: String, CodingKey {
        case count
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        count = try values.decodeIfPresent(UInt64.self, forKey: .count) ?? 1
    }
}

struct NativeRecordingMetadata: Codable, Equatable {
    var id: String
    var createdAtUnixMs: UInt64
    var source: String
    var audioFile: String
    var microphone: String?
    var audioDurationMs: UInt64?
    var speechDetected: Bool?
    var rawTranscript: String
    var finalTranscript: String
    var intendedTranscript: String?
    var screenContextTerms: [String]
    var dictionaryCorrections: [NativeAppliedCorrection]
    var llm: NativeLlmMetadata?
    /// What the user did to the dictation after it was typed, when edit
    /// learning watched it (the engine's `user_edit`).
    var userEdit: NativeUserEdit?
    /// The app it was pasted into (kept only with local history on).
    var appContext: NativeAppContext?

    enum CodingKeys: String, CodingKey {
        case id, source, microphone, llm
        case userEdit = "user_edit"
        case appContext = "app_context"
        case createdAtUnixMs = "created_at_unix_ms"
        case audioFile = "audio_file"
        case audioDurationMs = "audio_duration_ms"
        case speechDetected = "speech_detected"
        case rawTranscript = "raw_transcript"
        case finalTranscript = "final_transcript"
        case intendedTranscript = "intended_transcript"
        case screenContextTerms = "screen_context_terms"
        case dictionaryCorrections = "dictionary_corrections"
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        id = try values.decode(String.self, forKey: .id)
        createdAtUnixMs = try values.decode(UInt64.self, forKey: .createdAtUnixMs)
        source = try values.decodeIfPresent(String.self, forKey: .source) ?? ""
        audioFile = try values.decodeIfPresent(String.self, forKey: .audioFile) ?? "audio.wav"
        microphone = try values.decodeIfPresent(String.self, forKey: .microphone)
        audioDurationMs = try values.decodeIfPresent(UInt64.self, forKey: .audioDurationMs)
        speechDetected = try values.decodeIfPresent(Bool.self, forKey: .speechDetected)
        rawTranscript = try values.decodeIfPresent(String.self, forKey: .rawTranscript) ?? ""
        finalTranscript = try values.decodeIfPresent(String.self, forKey: .finalTranscript) ?? ""
        intendedTranscript = try values.decodeIfPresent(String.self, forKey: .intendedTranscript)
        screenContextTerms = try values.decodeIfPresent([String].self, forKey: .screenContextTerms)
            ?? []
        dictionaryCorrections =
            try values.decodeIfPresent([NativeAppliedCorrection].self, forKey: .dictionaryCorrections)
            ?? []
        llm = try values.decodeIfPresent(NativeLlmMetadata.self, forKey: .llm)
        // A malformed record must not hide the dictation from History.
        userEdit = try? values.decodeIfPresent(NativeUserEdit.self, forKey: .userEdit)
        appContext = try? values.decodeIfPresent(NativeAppContext.self, forKey: .appContext)
    }
}

/// The parts of an engine `user_edit` record History shows.
struct NativeUserEdit: Codable, Equatable {
    var pastedText: String
    var editedText: String
    var outcome: String
    var endReason: String

    enum CodingKeys: String, CodingKey {
        case outcome
        case pastedText = "pasted_text"
        case editedText = "edited_text"
        case endReason = "end_reason"
    }

    /// The user changed what Phonon typed (including deleting it).
    var changed: Bool {
        pastedText.trimmingCharacters(in: .whitespacesAndNewlines)
            != editedText.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

struct NativeHistoryItem: Identifiable, Equatable {
    let metadata: NativeRecordingMetadata
    let directoryURL: URL
    let metadataURL: URL

    var id: String { metadata.id }
    var displayText: String {
        if let intended = metadata.intendedTranscript, !intended.isEmpty { return intended }
        if !metadata.finalTranscript.isEmpty { return metadata.finalTranscript }
        return metadata.rawTranscript
    }
    var date: Date { Date(timeIntervalSince1970: Double(metadata.createdAtUnixMs) / 1_000) }
    /// The user's edit, when they changed what Phonon typed.
    var changedEdit: NativeUserEdit? { metadata.userEdit.flatMap { $0.changed ? $0 : nil } }
}

struct NativeUsageStats: Equatable {
    var recordings = 0
    var words = 0
    var speakingMilliseconds: UInt64 = 0
    var dictionaryFixes = 0
    var wordsToday = 0
    var recordingsToday = 0
    var activeDays = 0

    var wordsPerMinute: Int {
        guard speakingMilliseconds > 0 else { return 0 }
        return Int((Double(words) * 60_000 / Double(speakingMilliseconds)).rounded())
    }

    var averageWordsPerRecording: Int {
        guard recordings > 0 else { return 0 }
        return Int((Double(words) / Double(recordings)).rounded())
    }
}

enum PhononDataPaths {
    static func supportDirectory(fileManager: FileManager = .default) -> URL {
        let base = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? fileManager.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support")
        return base.appendingPathComponent("Phonon", isDirectory: true)
    }

    static func settings(fileManager: FileManager = .default) -> URL {
        supportDirectory(fileManager: fileManager).appendingPathComponent("settings.json")
    }

    static func dictionary(fileManager: FileManager = .default) -> URL {
        supportDirectory(fileManager: fileManager).appendingPathComponent("dictionary.json")
    }

    static func corpus(fileManager: FileManager = .default) -> URL {
        supportDirectory(fileManager: fileManager).appendingPathComponent("Corpus", isDirectory: true)
    }
}

enum PermissionGuide: String, Identifiable {
    case inputMonitoring
    case screenRecording

    var id: String { rawValue }

    var title: String {
        switch self {
        case .inputMonitoring: return "Allow Input Monitoring"
        case .screenRecording: return "Allow Screen Recording"
        }
    }

    var detail: String {
        switch self {
        case .inputMonitoring:
            return "Turn on Phonon so the hold-to-talk shortcut works everywhere."
        case .screenRecording:
            return "Turn on Phonon so local screen context can improve technical terms."
        }
    }

    var manualInstructions: String {
        "Click + in System Settings, type Phonon, press Return, then turn Phonon on."
    }

    var pane: PrivacyPane {
        switch self {
        case .inputMonitoring: return .inputMonitoring
        case .screenRecording: return .screenRecording
        }
    }
}

@MainActor
final class NativeAppStore: ObservableObject {
    @Published private(set) var settings = NativeSettings()
    @Published private(set) var dictionaryEntries: [NativeDictionaryEntry] = []
    @Published private(set) var history: [NativeHistoryItem] = []
    @Published private(set) var usage = NativeUsageStats()
    /// macros.json: whole-dictation triggers and what to type instead.
    @Published private(set) var macros: [PhononMacro] = []
    /// Recorded meetings under Meetings/, newest first.
    @Published private(set) var meetings: [MeetingNote] = []
    /// Meetings being transcribed, queued or failed this session (`MeetingTranscriber`).
    @Published var meetingTranscription: [String: MeetingTranscriptionJob] = [:]
    /// The meeting being recorded now, if any.
    @Published var recordingMeetingID: String?
    /// The Notes page's Transcribe button: transcribe the meeting in this folder.
    var onTranscribeMeeting: ((URL) -> Void)?
    /// Meetings being cleaned, queued or failed this session (`MeetingCleaner`).
    @Published var meetingCleanup: [String: MeetingCleanupJob] = [:]
    /// Whether transcripts can be cleaned on this Mac (the meeting adapter installed or published).
    @Published var meetingCleanupAvailable = false
    /// The Notes page's Clean up button.
    var onCleanMeeting: ((URL) -> Void)?
    @Published var engineReady = false
    /// The engine's last word on the WebSocket API (`status`, kind `websocket_api`).
    @Published var websocketAPIStatus: [String: Any]?
    /// Dictations other devices made over the WebSocket API, newest last: the engine's
    /// `websocket_dictation` events, seeded from local history at launch.
    @Published private(set) var websocketDictations: [WebSocketDictation] = []
    /// A WebSocket client is dictating right now.
    @Published var websocketLive = false
    private var websocketSeeded = false

    /// A client's dictation ended (the engine's `websocket_dictation` event, state `ended`).
    func recordWebSocketDictation(words: Int, audioMilliseconds: UInt64, at end: Date = Date()) {
        websocketLive = false
        guard words > 0 else { return }
        var list = websocketDictations.filter { end.timeIntervalSince($0.end) < WebSocketActivity.keep }
        list.append(
            WebSocketDictation(
                end: end, words: words, audioSeconds: Double(audioMilliseconds) / 1_000))
        websocketDictations = list
    }

    /// The last hour's WebSocket dictations from local history, once per launch, so the
    /// graph does not start empty after a restart.
    private func seedWebSocketDictations() {
        guard !websocketSeeded else { return }
        websocketSeeded = true
        let now = Date()
        let seeded = history.compactMap { item -> WebSocketDictation? in
            guard item.metadata.source == "websocket",
                now.timeIntervalSince(item.date) < WebSocketActivity.keep
            else { return nil }
            let words = item.displayText.split(whereSeparator: { $0.isWhitespace }).count
            guard words > 0 else { return nil }
            return WebSocketDictation(
                end: item.date, words: words,
                audioSeconds: Double(item.metadata.audioDurationMs ?? 0) / 1_000)
        }
        websocketDictations = (seeded + websocketDictations).sorted { $0.end < $1.end }
    }
    @Published var engineMessage = "Loading local models"
    @Published var selectedMicrophone = "Resolving…"
    @Published var availableMicrophones: [String] = []
    @Published var inputMonitoringAvailable = false
    @Published var permissionGuide: PermissionGuide?
    @Published var lastError: String?
    /// A page the main window should switch to, set by the app menu's
    /// Settings… item and cleared once the window shows it.
    @Published var pageRequest: NativeAppPage?
    /// Last known state of the off-Library copy. See `PhononDataMirror`.
    @Published private(set) var mirrorManifest: MirrorManifest?
    /// A newer release found by the update check. See `UpdateController`.
    @Published var availableUpdate: AvailableUpdate?
    var onDictionaryChanged: (() -> Void)?
    var onMicrophonePermissionGranted: (() -> Void)?
    var onPermissionsRefresh: (() -> Void)?
    /// Show the update alert (sidebar and Settings buttons).
    var onShowUpdate: (() -> Void)?
    /// Check GitHub now and report the result (Settings button).
    var onCheckForUpdates: (() -> Void)?

    private let fileManager: FileManager
    private let supportDirectory: URL
    /// Where `PhononDataMirror` keeps the off-Library copy of this store.
    private let mirrorDirectory: URL

    /// A store opened on an injected directory (tests, tools) mirrors inside
    /// that directory unless told otherwise. Before this, every test that
    /// saved a dictionary entry captured its temporary store into the owner's
    /// real `~/.phonon/backup`, which is the copy a wipe is restored from.
    init(
        fileManager: FileManager = .default, supportDirectory: URL? = nil,
        mirrorDirectory: URL? = nil
    ) {
        self.fileManager = fileManager
        let support = supportDirectory
            ?? PhononDataPaths.supportDirectory(fileManager: fileManager)
        self.supportDirectory = support
        self.mirrorDirectory = Self.mirrorDirectory(
            explicit: mirrorDirectory, injectedSupport: supportDirectory, fileManager: fileManager)
        reloadAll()
        mirrorManifest = PhononDataMirror.capture(
            from: support, mirror: self.mirrorDirectory, fileManager: fileManager)
            ?? PhononDataMirror.manifest(mirror: self.mirrorDirectory, fileManager: fileManager)
    }

    static func mirrorDirectory(
        explicit: URL?, injectedSupport: URL?, fileManager: FileManager = .default
    ) -> URL {
        if let explicit { return explicit }
        if let injectedSupport {
            return injectedSupport.appendingPathComponent(".mirror", isDirectory: true)
        }
        return PhononDataMirror.directory(fileManager: fileManager)
    }

    func reloadAll() {
        loadSettings()
        loadDictionary()
        loadHistory()
        seedWebSocketDictations()
        loadMacros()
        loadMeetings()
        refreshAvailableMicrophones()
    }

    // MARK: Macros

    func loadMacros() {
        macros = MacroBook.load(from: macrosURL)
    }

    /// Add a macro, or replace the one with the same trigger.
    func upsertMacro(trigger: String, expansion: String, replacing oldID: String? = nil) {
        let trigger = trigger.trimmingCharacters(in: .whitespacesAndNewlines)
        let expansion = expansion.trimmingCharacters(in: .newlines)
        guard !MacroBook.normalize(trigger).isEmpty,
            !expansion.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return }
        let macro = PhononMacro(trigger: trigger, expansion: expansion)
        var list = macros.filter { $0.id != macro.id && $0.id != oldID }
        if let oldID, let index = macros.firstIndex(where: { $0.id == oldID }) {
            list.insert(macro, at: min(index, list.count))
        } else {
            list.insert(macro, at: 0)
        }
        saveMacros(list)
    }

    func removeMacro(id: String) {
        saveMacros(macros.filter { $0.id != id })
    }

    private func saveMacros(_ list: [PhononMacro]) {
        do {
            try MacroBook.save(list, to: macrosURL)
            macros = list
            lastError = nil
            captureMirror()
        } catch {
            lastError = "Could not save macros: \(error.localizedDescription)"
        }
    }

    // MARK: Meetings

    func loadMeetings() {
        meetings = MeetingNote.loadAll(from: meetingsDirectory, fileManager: fileManager)
    }

    var meetingsDirectory: URL {
        supportDirectory.appendingPathComponent("Meetings", isDirectory: true)
    }

    func transcriptState(_ note: MeetingNote) -> NoteTranscriptState {
        NoteTranscriptState.resolve(
            transcript: note.transcript, job: meetingTranscription[note.id],
            savedError: note.transcriptError, recording: recordingMeetingID == note.id)
    }

    func cleanupState(_ note: MeetingNote) -> NoteCleanupState {
        NoteCleanupState.resolve(
            transcript: note.transcript, cleaned: note.cleaned, job: meetingCleanup[note.id],
            savedError: note.cleanupError)
    }

    /// The Clean up button.
    func cleanMeeting(_ note: MeetingNote) {
        onCleanMeeting?(note.directory)
    }

    /// The Transcribe button. Never for the meeting being recorded (or still closing its last
    /// chunks): its metadata is not final until the recording has fully stopped.
    func transcribeMeeting(_ note: MeetingNote) {
        guard recordingMeetingID != note.id else { return }
        onTranscribeMeeting?(note.directory)
    }

    /// A recording has fully stopped (chunks closed, metadata.json final): it is no longer the
    /// one recording (unless a newer one already is), it is queued for transcription, and the
    /// Notes list picks it up, so an open Notes page shows it transcribing.
    func meetingRecordingFinished(directory: URL) {
        if recordingMeetingID == directory.lastPathComponent { recordingMeetingID = nil }
        onTranscribeMeeting?(directory)
        loadMeetings()
    }

    // MARK: Dictionary chips

    /// The Dictionary page's add field: "misheard → right" (or "->") becomes a
    /// replacement; anything else is a term to spell right.
    func addDictionaryText(_ text: String) {
        if let (heard, right) = Self.parseFix(text) {
            upsertDictionary(originalID: nil, phrase: heard, replacement: right, spokenForms: [])
        } else {
            upsertDictionary(originalID: nil, phrase: text, replacement: nil, spokenForms: [])
        }
    }

    static func parseFix(_ text: String) -> (String, String)? {
        for arrow in ["→", "->", "=>"] {
            let parts = text.components(separatedBy: arrow)
            guard parts.count == 2 else { continue }
            let heard = parts[0].trimmingCharacters(in: .whitespacesAndNewlines)
            let right = parts[1].trimmingCharacters(in: .whitespacesAndNewlines)
            guard !heard.isEmpty, !right.isEmpty else { return nil }
            return (heard, right)
        }
        return nil
    }

    /// Merge another dictionary: a Phonon dictionary.json (or a bare list of
    /// entries), or a text file with one term or "misheard → right" per line.
    /// Entries already here are left as they are. Returns how many were added.
    @discardableResult
    func importDictionary(from url: URL) -> Int {
        guard let data = try? Data(contentsOf: url) else {
            lastError = "Could not read \(url.lastPathComponent)."
            return 0
        }
        var incoming: [NativeDictionaryEntry] = []
        let decoder = JSONDecoder()
        if let file = try? decoder.decode(NativeDictionaryFile.self, from: data) {
            incoming = file.entries
        } else if let list = try? decoder.decode([NativeDictionaryEntry].self, from: data) {
            incoming = list
        } else if let text = String(data: data, encoding: .utf8) {
            for line in text.split(whereSeparator: \.isNewline) {
                let line = line.trimmingCharacters(in: .whitespaces)
                guard !line.isEmpty, !line.hasPrefix("#") else { continue }
                if let (heard, right) = Self.parseFix(line) {
                    incoming.append(NativeDictionaryEntry(phrase: heard, replacement: right))
                } else {
                    incoming.append(NativeDictionaryEntry(phrase: line))
                }
            }
        }
        let (entries, added, _) = Self.merge(
            incoming.map {
                var copy = $0
                copy.usageCount = 0
                copy.source = "import"
                return copy
            }, into: dictionaryEntries, addSpokenForms: false)
        if added > 0 { saveDictionary(entries) }
        return added
    }

    /// Entries already present (same phrase ignoring case, same replacement)
    /// are kept as they are; with `addSpokenForms` they gain the incoming
    /// spoken forms they lack. Returns the list, how many were added, and
    /// whether anything changed.
    static func merge(
        _ incoming: [NativeDictionaryEntry], into existing: [NativeDictionaryEntry],
        addSpokenForms: Bool
    ) -> ([NativeDictionaryEntry], Int, Bool) {
        var entries = existing
        var added = 0
        var changed = false
        for entry in incoming {
            let phrase = entry.phrase.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !phrase.isEmpty else { continue }
            if let index = entries.firstIndex(where: {
                $0.phrase.caseInsensitiveCompare(phrase) == .orderedSame
                    && ($0.replacement ?? "") == (entry.replacement ?? "")
            }) {
                guard addSpokenForms else { continue }
                let missing = entry.spokenForms.filter { form in
                    !entries[index].spokenForms.contains {
                        $0.caseInsensitiveCompare(form) == .orderedSame
                    }
                }
                if !missing.isEmpty {
                    entries[index].spokenForms = (entries[index].spokenForms + missing).sorted {
                        $0.localizedCaseInsensitiveCompare($1) == .orderedAscending
                    }
                    changed = true
                }
                continue
            }
            var copy = entry
            copy.phrase = phrase
            entries.append(copy)
            added += 1
            changed = true
        }
        return (entries, added, changed)
    }

    // MARK: Import from other dictation apps

    /// Apply what `DictationImportScanner` found (SPEC "Import from other
    /// dictation apps"): words and replacements into the dictionary, snippets
    /// into macros, past dictations into `Imported/` and `profile/extract/`,
    /// and the shortcut and microphone when Phonon has not been given its own.
    /// Nothing is written to the other apps' files.
    @discardableResult
    func applyDictationImport(_ findings: [DictationAppFindings]) -> DictationImportSummary {
        var summary = DictationImportSummary()
        let (entries, added, changed) = Self.merge(
            findings.flatMap(\.terms), into: dictionaryEntries, addSpokenForms: true)
        if changed { saveDictionary(entries) }
        summary.addedTerms = added

        var list = macros
        for macro in findings.flatMap(\.macros) where !list.contains(where: { $0.id == macro.id }) {
            list.append(macro)
            summary.addedMacros += 1
        }
        if summary.addedMacros > 0 { saveMacros(list) }

        var failure: String?
        for finding in findings where !finding.dictations.isEmpty {
            do {
                try DictationImportFiles.write(
                    finding.dictations, app: finding.app, support: supportDirectory)
                summary.savedDictations += finding.dictations.count
            } catch {
                failure = "Could not save \(finding.app.name) history: \(error.localizedDescription)"
            }
        }

        // Phonon's own default is fn, so only a different key is a change. The
        // first app in table order that names a key Phonon offers wins.
        if let mode = findings.lazy.compactMap(\.shortcutMode).first,
            mode != settings.shortcutMode
        {
            updateSettings { $0.shortcutMode = mode }
            summary.shortcutMode = mode
        }
        // A ranking the user already made in Phonon is theirs; leave it. A
        // Bluetooth headset is not ranked from another app's choice: opening
        // its mic drops the headset to call audio (SPEC "Microphone").
        let devices = CoreAudioInputDevices.all()
        if settings.microphonePriority.isEmpty,
            let microphone = findings.lazy.compactMap(\.microphone).first(where: { name in
                devices.first { $0.name.localizedCaseInsensitiveContains(name) }?.isBluetooth
                    != true
            })
        {
            updateSettings { $0.microphonePriority = [microphone] }
            summary.microphone = microphone
        }
        if let failure { lastError = failure }
        return summary
    }

    /// Scan the given apps off the main thread with the user's private list.
    func scanDictationApps(_ apps: [DictationApp]) async -> [DictationAppFindings] {
        let env = DictationImportEnvironment.live(denylist: settings.screenContextDenylist)
        return await Task.detached(priority: .userInitiated) {
            DictationImportScanner.scan(apps, env: env)
        }.value
    }

    func detectDictationApps() -> [DictationApp] {
        DictationImportScanner.detect(
            DictationImportEnvironment.live(denylist: settings.screenContextDenylist))
    }

    /// The microphone list is whatever CoreAudio reports right now, so a device
    /// that has been unplugged since it was ranked still shows in the ranking but
    /// can be told apart from one that is present.
    func refreshAvailableMicrophones() {
        availableMicrophones = CoreAudioInputDevices.inputNames()
    }

    func updateSettings(_ mutate: (inout NativeSettings) -> Void) {
        var updated = settings
        mutate(&updated)
        updated.schemaVersion = NativeSettings.currentSchemaVersion
        settings = updated
        do {
            try writeSettingsKeepingUnknownKeys(updated)
            lastError = nil
            captureMirror()
        } catch {
            lastError = "Could not save settings: \(error.localizedDescription)"
        }
    }

    func upsertDictionary(
        originalID: String?, phrase: String, replacement: String?, spokenForms: [String]
    ) {
        let phrase = phrase.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !phrase.isEmpty else { return }
        let cleanReplacement = replacement?.trimmingCharacters(in: .whitespacesAndNewlines)
        let cleanForms = Array(
            Set(spokenForms.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter {
                !$0.isEmpty
            })
        ).sorted { $0.localizedCaseInsensitiveCompare($1) == .orderedAscending }
        var entries = dictionaryEntries
        if let originalID, let index = entries.firstIndex(where: { $0.id == originalID }) {
            entries[index].phrase = phrase
            entries[index].replacement = cleanReplacement?.isEmpty == false ? cleanReplacement : nil
            entries[index].spokenForms = cleanForms
        } else if !entries.contains(where: {
            $0.phrase.caseInsensitiveCompare(phrase) == .orderedSame
                && ($0.replacement ?? "") == (cleanReplacement ?? "")
        }) {
            entries.append(
                NativeDictionaryEntry(
                    phrase: phrase,
                    replacement: cleanReplacement?.isEmpty == false ? cleanReplacement : nil,
                    spokenForms: cleanForms
                ))
        }
        saveDictionary(entries)
    }

    func removeDictionary(id: String) {
        saveDictionary(dictionaryEntries.filter { $0.id != id })
    }

    /// Add a term the user approved on the edit-learning panel. An entry that
    /// already spells the term gains the heard form as a spoken form; any
    /// other term becomes a new entry carrying it. Returns what changed, for
    /// Undo, or nil when there was nothing to add.
    @discardableResult
    func approveLearnedTerm(term: String, heardAs: String) -> LearnedTermUndo? {
        let term = term.trimmingCharacters(in: .whitespacesAndNewlines)
        let heard = heardAs.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !term.isEmpty else { return nil }
        let same = { (left: String, right: String) in
            left.caseInsensitiveCompare(right) == .orderedSame
        }
        var entries = dictionaryEntries
        if let index = entries.firstIndex(where: {
            same($0.phrase, term) || ($0.replacement.map { same($0, term) } ?? false)
        }) {
            let entry = entries[index]
            guard !heard.isEmpty, !same(heard, entry.phrase), !same(heard, term),
                !entry.spokenForms.contains(where: { same($0, heard) })
            else { return nil }
            entries[index].spokenForms = (entry.spokenForms + [heard]).sorted {
                $0.localizedCaseInsensitiveCompare($1) == .orderedAscending
            }
            saveDictionary(entries)
            return LearnedTermUndo(change: .addedSpokenForm(entryID: entry.id, form: heard))
        }
        let forms = heard.isEmpty || same(heard, term) ? [] : [heard]
        let entry = NativeDictionaryEntry(
            phrase: term, spokenForms: forms, source: LearnedTermUndo.source)
        entries.append(entry)
        saveDictionary(entries)
        return LearnedTermUndo(change: .addedEntry(id: entry.id))
    }

    /// Take back exactly what one approval added, if it is still there.
    @discardableResult
    func undoLearnedTerm(_ undo: LearnedTermUndo) -> Bool {
        switch undo.change {
        case let .addedEntry(id):
            guard dictionaryEntries.contains(where: { $0.id == id }) else { return false }
            saveDictionary(dictionaryEntries.filter { $0.id != id })
            return true
        case let .addedSpokenForm(entryID, form):
            var entries = dictionaryEntries
            guard let index = entries.firstIndex(where: { $0.id == entryID }),
                entries[index].spokenForms.contains(form)
            else { return false }
            entries[index].spokenForms.removeAll { $0 == form }
            saveDictionary(entries)
            return true
        }
    }

    /// Save a style preference the user approved for one app category.
    func approveStylePreference(_ preference: String, category: String) {
        updateSettings {
            var list = $0.stylePreferences[category] ?? []
            if !list.contains(preference) { list.append(preference) }
            $0.stylePreferences[category] = list
        }
    }

    /// Pin one writing mode for dictations into `bundleID`; nil goes back to the default.
    func setAppWritingMode(_ bundleID: String, _ mode: AppWritingMode?) {
        let id = bundleID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !id.isEmpty else { return }
        updateSettings { $0.writingModesByApp[id] = mode?.rawValue }
    }

    /// Apps dictated into, most used first (from local history), for the per-app
    /// writing mode picker.
    var dictatedApps: [(name: String, bundleID: String)] {
        var counts: [String: (name: String, count: Int)] = [:]
        for item in history {
            guard let app = item.metadata.appContext, !app.bundleID.isEmpty else { continue }
            let name = app.appName ?? counts[app.bundleID]?.name ?? app.bundleID
            counts[app.bundleID] = (name, (counts[app.bundleID]?.count ?? 0) + 1)
        }
        return counts.sorted { $0.value.count > $1.value.count }
            .map { (name: $0.value.name, bundleID: $0.key) }
    }

    /// Add an app (bundle id) or site (host or URL) to the private list.
    func addToDenylist(_ raw: String) {
        guard let entry = ContextPolicy.normalizeEntry(raw) else { return }
        updateSettings {
            guard
                !$0.screenContextDenylist.contains(where: {
                    $0.caseInsensitiveCompare(entry) == .orderedSame
                })
            else { return }
            $0.screenContextDenylist.append(entry)
        }
    }

    func removeFromDenylist(_ entry: String) {
        updateSettings { $0.screenContextDenylist.removeAll { $0 == entry } }
    }

    func removeStylePreference(_ preference: String, category: String) {
        updateSettings {
            var list = $0.stylePreferences[category] ?? []
            list.removeAll { $0 == preference }
            $0.stylePreferences[category] = list.isEmpty ? nil : list
        }
    }

    func saveIntendedTranscript(itemID: String, text: String) {
        guard let item = history.first(where: { $0.id == itemID }) else { return }
        do {
            let data = try Data(contentsOf: item.metadataURL)
            guard var object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw CocoaError(.fileReadCorruptFile)
            }
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty {
                object.removeValue(forKey: "intended_transcript")
            } else {
                object["intended_transcript"] = trimmed
            }
            try writeJSONObject(object, to: item.metadataURL)
            loadHistory()
            lastError = nil
        } catch {
            lastError = "Could not save the transcript: \(error.localizedDescription)"
        }
    }

    var needsPrivacyChoice: Bool { !settings.privacyChoiceMade }

    /// Record the first-run answer for the two settings that retain data.
    func recordPrivacyChoice(
        localHistory: Bool, screenContext: Bool, clipboard: ClipboardAfterPaste = .dictation
    ) {
        updateSettings {
            $0.localHistory = localHistory
            $0.screenContext = screenContext
            $0.clipboardAfterPaste = clipboard
            $0.privacyChoiceMade = true
        }
    }

    func clearAllHistory() {
        var failure: String?
        for item in history {
            do {
                _ = try fileManager.trashItem(at: item.directoryURL, resultingItemURL: nil)
            } catch {
                failure = error.localizedDescription
            }
        }
        loadHistory()
        lastError = failure.map { "Could not clear every recording: \($0)" }
    }

    /// Trash recordings older than the retention window. Never runs when the
    /// window is "keep forever", which is the default for every install.
    func pruneExpiredRecordings(now: Date = Date()) {
        let days = settings.historyRetentionDays
        guard days > NativeSettings.keepRecordingsForever else { return }
        let cutoff = now.addingTimeInterval(-Double(days) * 86_400)
        var removed = false
        for item in history where item.date < cutoff {
            if (try? fileManager.trashItem(at: item.directoryURL, resultingItemURL: nil)) != nil {
                removed = true
            }
        }
        if removed { loadHistory() }
    }

    func trashRecording(itemID: String) {
        guard let item = history.first(where: { $0.id == itemID }) else { return }
        do {
            _ = try fileManager.trashItem(at: item.directoryURL, resultingItemURL: nil)
            loadHistory()
            lastError = nil
        } catch {
            lastError = "Could not move recording to Trash: \(error.localizedDescription)"
        }
    }

    func setLaunchAtLogin(_ enabled: Bool) {
        do {
            if enabled {
                try SMAppService.mainApp.register()
            } else {
                try SMAppService.mainApp.unregister()
            }
            lastError = nil
            objectWillChange.send()
        } catch {
            lastError = "Could not update launch at login: \(error.localizedDescription)"
        }
    }

    var launchAtLoginEnabled: Bool {
        SMAppService.mainApp.status == .enabled
    }

    var microphonePermission: Bool {
        AVCaptureDevice.authorizationStatus(for: .audio) == .authorized
    }

    var microphoneAuthorizationStatus: AVAuthorizationStatus {
        AVCaptureDevice.authorizationStatus(for: .audio)
    }

    var microphoneStatusText: String {
        switch microphoneAuthorizationStatus {
        case .authorized: return "Granted"
        case .notDetermined: return "Not requested"
        case .denied: return "Denied"
        case .restricted: return "Restricted"
        @unknown default: return "Needs access"
        }
    }

    var microphoneActionTitle: String {
        microphoneAuthorizationStatus == .notDetermined ? "Request Access" : "Settings"
    }

    func performMicrophonePermissionAction() {
        guard microphoneAuthorizationStatus == .notDetermined else {
            NSWorkspace.shared.open(PrivacyPane.microphone.settingsURL)
            return
        }
        AVCaptureDevice.requestAccess(for: .audio) { [weak self] granted in
            DispatchQueue.main.async {
                guard let self else { return }
                self.refreshPermissions()
                if granted { self.onMicrophonePermissionGranted?() }
            }
        }
    }

    var accessibilityPermission: Bool { AXIsProcessTrusted() }
    var screenRecordingPermission: Bool { CGPreflightScreenCaptureAccess() }

    var inputMonitoringActionTitle: String {
        inputMonitoringAvailable ? "Settings" : "Request Access"
    }

    var screenRecordingActionTitle: String {
        screenRecordingPermission ? "Settings" : "Request Access"
    }

    func performInputMonitoringPermissionAction() {
        guard !CGPreflightListenEventAccess() else {
            NSWorkspace.shared.open(PrivacyPane.inputMonitoring.settingsURL)
            return
        }
        let granted = CGRequestListenEventAccess()
        refreshPermissions()
        if !granted { permissionGuide = .inputMonitoring }
    }

    func performScreenRecordingPermissionAction() {
        guard !CGPreflightScreenCaptureAccess() else {
            NSWorkspace.shared.open(PrivacyPane.screenRecording.settingsURL)
            return
        }
        let granted = CGRequestScreenCaptureAccess()
        refreshPermissions()
        if !granted { permissionGuide = .screenRecording }
    }

    func refreshPermissions() {
        inputMonitoringAvailable = CGPreflightListenEventAccess()
        globeUsageType = GlobeKeySetting.currentUsageType()
        onPermissionsRefresh?()
        objectWillChange.send()
    }

    /// System Settings › Keyboard › "Press 🌐 key to", read at each permission refresh.
    var globeUsageType: Int? = GlobeKeySetting.currentUsageType()

    /// What keeps dictation from working, for the banner and the menu-bar item.
    var permissionAttention: [PermissionAttention] {
        PermissionAttention.items(
            inputMonitoring: inputMonitoringAvailable, microphone: microphonePermission,
            accessibility: accessibilityPermission, shortcutMode: settings.shortcutMode,
            globeUsageType: globeUsageType)
    }

    func performAttentionAction(_ item: PermissionAttention) {
        switch item {
        case .inputMonitoring: performInputMonitoringPermissionAction()
        case .microphone: performMicrophonePermissionAction()
        case .accessibility:
            // The prompt adds Phonon to the list, so the user only flips the switch.
            let options = ["AXTrustedCheckOptionPrompt": true] as CFDictionary
            if !AXIsProcessTrustedWithOptions(options) {
                NSWorkspace.shared.open(PrivacyPane.accessibility.settingsURL)
            }
            refreshPermissions()
        case .globeKey: NSWorkspace.shared.open(PermissionAttention.keyboardPane)
        }
    }

    private func loadSettings() {
        let url = settingsURL
        guard let data = try? Data(contentsOf: url),
            let decoded = try? JSONDecoder().decode(NativeSettings.self, from: data)
        else {
            var fresh = NativeSettings()
            fresh.microphonePriority = Self.defaultMicrophonePriority()
            settings = fresh
            try? writeJSON(settings, to: url)
            return
        }
        settings = decoded
    }

    /// Nothing ranked on a fresh install: Phonon follows the system input the
    /// user already chose (Bluetooth headset mics excepted, see
    /// `MicrophonePriorityResolver`). Ranking is the override, not the default.
    private static func defaultMicrophonePriority() -> [String] {
        []
    }

    private func loadDictionary() {
        let url = dictionaryURL
        guard let data = try? Data(contentsOf: url),
            let decoded = try? JSONDecoder().decode(NativeDictionaryFile.self, from: data)
        else {
            dictionaryEntries = []
            return
        }
        dictionaryEntries = decoded.entries.sorted {
            $0.phrase.localizedCaseInsensitiveCompare($1.phrase) == .orderedAscending
        }
    }

    private func captureMirror() {
        if let written = PhononDataMirror.capture(
            from: supportDirectory, mirror: mirrorDirectory, fileManager: fileManager)
        {
            mirrorManifest = written
        }
    }

    /// A backup worth offering, or nil. Only non-empty when the store itself
    /// has nothing, which is what an uninstall or a first run looks like.
    func restorableBackup() -> MirrorManifest? {
        PhononDataMirror.restorableManifest(
            for: supportDirectory, mirror: mirrorDirectory, fileManager: fileManager)
    }

    func restoreFromBackup() {
        do {
            try PhononDataMirror.restore(
                into: supportDirectory, mirror: mirrorDirectory, fileManager: fileManager)
            reloadAll()
            onDictionaryChanged?()
            lastError = nil
        } catch {
            lastError = "Could not restore backup: \(error.localizedDescription)"
        }
    }

    func deleteBackup() {
        do {
            try PhononDataMirror.forget(mirror: mirrorDirectory, fileManager: fileManager)
            mirrorManifest = nil
            lastError = nil
        } catch {
            lastError = "Could not delete backup: \(error.localizedDescription)"
        }
    }

    func exportEverything(to destination: URL) {
        do {
            try PhononDataMirror.exportAll(
                from: supportDirectory, to: destination, fileManager: fileManager)
            lastError = nil
        } catch {
            lastError = "Could not export: \(error.localizedDescription)"
        }
    }

    func revealBackupInFinder() {
        NSWorkspace.shared.activateFileViewerSelecting([mirrorDirectory])
    }

    private func saveDictionary(_ entries: [NativeDictionaryEntry]) {
        let sorted = entries.sorted {
            $0.phrase.localizedCaseInsensitiveCompare($1.phrase) == .orderedAscending
        }
        let file = NativeDictionaryFile(
            schemaVersion: 1,
            updatedAtUnixMs: UInt64(Date().timeIntervalSince1970 * 1_000),
            entries: sorted
        )
        do {
            try writeJSON(file, to: dictionaryURL)
            dictionaryEntries = sorted
            lastError = nil
            captureMirror()
            onDictionaryChanged?()
        } catch {
            lastError = "Could not save dictionary: \(error.localizedDescription)"
        }
    }

    private func loadHistory() {
        let corpus = corpusURL
        let directories = (try? fileManager.contentsOfDirectory(
            at: corpus, includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles])) ?? []
        let decoder = JSONDecoder()
        history = directories.compactMap { directory in
            let metadataURL = directory.appendingPathComponent("metadata.json")
            guard let data = try? Data(contentsOf: metadataURL),
                let metadata = try? decoder.decode(NativeRecordingMetadata.self, from: data)
            else { return nil }
            guard metadata.speechDetected != false else { return nil }
            return NativeHistoryItem(
                metadata: metadata, directoryURL: directory, metadataURL: metadataURL)
        }.sorted { $0.metadata.createdAtUnixMs > $1.metadata.createdAtUnixMs }
        usage = Self.computeUsage(history)
    }

    static func computeUsage(_ history: [NativeHistoryItem]) -> NativeUsageStats {
        var stats = NativeUsageStats(recordings: history.count)
        var activeDays = Set<Date>()
        for item in history {
            let words = item.displayText.split(whereSeparator: { $0.isWhitespace }).count
            stats.words += words
            stats.speakingMilliseconds += item.metadata.audioDurationMs ?? 0
            stats.dictionaryFixes += item.metadata.dictionaryCorrections.reduce(0) {
                $0 + Int($1.count)
            }
            if Calendar.current.isDateInToday(item.date) {
                stats.wordsToday += words
                stats.recordingsToday += 1
            }
            activeDays.insert(Calendar.current.startOfDay(for: item.date))
        }
        stats.activeDays = activeDays.count
        return stats
    }

    private func writeJSON<T: Encodable>(_ value: T, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(value)
        try fileManager.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url, options: .atomic)
    }

    /// settings.json is shared with the Rust side, which owns keys this struct
    /// does not model (`profile_prefix`); a save must not drop them.
    private func writeSettingsKeepingUnknownKeys(_ updated: NativeSettings) throws {
        let encoded = try JSONSerialization.jsonObject(with: JSONEncoder().encode(updated))
        var object =
            (try? Data(contentsOf: settingsURL)).flatMap {
                try? JSONSerialization.jsonObject(with: $0) as? [String: Any]
            } ?? [:]
        for (key, value) in encoded as? [String: Any] ?? [:] {
            object[key] = value
        }
        try fileManager.createDirectory(
            at: settingsURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try writeJSONObject(object, to: settingsURL)
    }

    private func writeJSONObject(_ object: [String: Any], to url: URL) throws {
        let data = try JSONSerialization.data(
            withJSONObject: object, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        try data.write(to: url, options: .atomic)
    }

    private var settingsURL: URL { supportDirectory.appendingPathComponent("settings.json") }
    private var dictionaryURL: URL { supportDirectory.appendingPathComponent("dictionary.json") }
    private var macrosURL: URL { supportDirectory.appendingPathComponent(MacroBook.fileName) }
    private var corpusURL: URL {
        supportDirectory.appendingPathComponent("Corpus", isDirectory: true)
    }
}
