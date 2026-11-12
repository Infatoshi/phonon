import AppKit
import Foundation

// Per-app meeting rules and the one global choice (SPEC "Meeting notes", "Rules").
// What a detected call does: settings.json `meeting_mode` (ask | rules | off) and, under
// `rules`, `meeting_rules` ({key: "always" | "never"}). A key is the app's bundle id
// (lowercased) or, for a call in a browser tab, the conferencing site it is on
// (meet.google.com, zoom.us), never a display name.

/// Settings › Notes › "When a call starts". `ask` is the default and what every install
/// that had detection on before maps to.
enum MeetingMode: String, CaseIterable, Codable {
    /// Ask before recording any call (Record / Not now).
    case ask
    /// Follow the per-app rules; a call in an app without one asks whether to remember it.
    case rules
    /// No detection at all. Manual start still works.
    case off

    var title: String {
        switch self {
        case .ask: return "Ask every time"
        case .rules: return "Follow my rules"
        case .off: return "Don't detect"
        }
    }
}

enum MeetingRule: String, CaseIterable, Codable {
    case always
    case never

    var title: String {
        switch self {
        case .always: return "Always record"
        case .never: return "Never"
        }
    }
}

/// What a detected call leads to.
enum MeetingDecision: Equatable {
    case record
    case ask
    case ignore
}

/// The four answers of the per-app question.
enum MeetingChoice: Equatable {
    /// Record now and every later call in this app or site.
    case always
    /// Record now, remember nothing.
    case once
    /// Not now: no recording, quiet for this app for a while (MeetingPromptPolicy).
    case notNow
    /// Never ask or record for this app or site.
    case never

    /// The rule the answer stores, if any.
    var rule: MeetingRule? {
        switch self {
        case .always: return .always
        case .never: return .never
        case .once, .notNow: return nil
        }
    }

    var records: Bool { self == .always || self == .once }
}

/// One rule change, kept so the Undo button can put the previous value back.
struct MeetingRuleChange: Equatable {
    let key: String
    let previous: MeetingRule?
    let new: MeetingRule?
}

enum MeetingRules {
    /// The rule key of a detection verdict (`MeetingDetector.Verdict.meeting`): a native call
    /// or chat app is its bundle id; a conference tab (`app` is then the page host) is the
    /// conferencing site it matches, so us02web.zoom.us and app.zoom.us share zoom.us.
    static func key(app: String, bundleID: String) -> String {
        guard MeetingAppIcon.isSite(bundleID: bundleID) else { return bundleID.lowercased() }
        return siteKey(host: app)
    }

    static func siteKey(host: String) -> String {
        let host = host.lowercased()
        let match = MeetingDetector.conferenceHosts
            .filter { host == $0 || host.hasSuffix("." + $0) }
            .max { $0.count < $1.count }
        return match ?? host
    }

    /// The whole decision: the global mode first, then the app's rule.
    static func decide(mode: MeetingMode, rule: MeetingRule?) -> MeetingDecision {
        switch mode {
        case .off: return .ignore
        case .ask: return .ask
        case .rules:
            switch rule {
            case .always: return .record
            case .never: return .ignore
            case nil: return .ask
            }
        }
    }

    /// What `meeting_auto_start: true` used to do (record every detected call without
    /// asking), as rules: every app and site the detector knows, set to always.
    static func autoStartRules() -> [String: MeetingRule] {
        var rules = [String: MeetingRule]()
        for id in MeetingDetector.callApps.keys { rules[id.lowercased()] = .always }
        for id in MeetingDetector.chatAppsWithCalls.keys { rules[id.lowercased()] = .always }
        for host in MeetingDetector.conferenceHosts { rules[host] = .always }
        return rules
    }

    /// Readable names for the sites (the apps use their own names).
    static let siteNames = [
        "meet.google.com": "Google Meet", "zoom.us": "Zoom (web)", "zoomgov.com": "Zoom (web)",
        "teams.microsoft.com": "Microsoft Teams (web)", "teams.live.com": "Microsoft Teams (web)",
        "webex.com": "Webex (web)", "gotomeeting.com": "GoTo Meeting", "bluejeans.com": "BlueJeans",
        "whereby.com": "Whereby", "chime.aws": "Amazon Chime", "meetings.dialpad.com": "Dialpad",
        "slack.com": "Slack huddles (web)", "discord.com": "Discord (web)", "around.co": "Around (web)",
        "meet.jit.si": "Jitsi Meet",
    ]

    /// The name a rule key shows as: a known app or site, an installed app, else the key.
    static func displayName(
        key: String,
        installedName: (String) -> String? = { id in
            NSWorkspace.shared.urlForApplication(withBundleIdentifier: id).map {
                FileManager.default.displayName(atPath: $0.path)
                    .replacingOccurrences(of: ".app", with: "")
            }
        }
    ) -> String {
        if let site = siteNames[key] { return site }
        let apps = MeetingDetector.callApps.merging(MeetingDetector.chatAppsWithCalls) { a, _ in a }
        if let known = apps.first(where: { $0.key.lowercased() == key })?.value { return known }
        return installedName(key) ?? key
    }

    /// The name the question uses for a verdict.
    static func displayName(app: String, bundleID: String) -> String {
        guard MeetingAppIcon.isSite(bundleID: bundleID) else { return app }
        return siteNames[siteKey(host: app)] ?? app
    }
}

/// The rules as stored: `[key: rule]`. Setting returns the change, which `undo` reverts.
struct MeetingRuleBook: Equatable, Codable {
    private(set) var rules: [String: MeetingRule]

    init(_ rules: [String: MeetingRule] = [:]) {
        self.rules = rules
    }

    func rule(for key: String) -> MeetingRule? {
        rules[key.lowercased()]
    }

    @discardableResult
    mutating func set(_ rule: MeetingRule?, for key: String) -> MeetingRuleChange {
        let key = key.lowercased()
        let change = MeetingRuleChange(key: key, previous: rules[key], new: rule)
        rules[key] = rule
        return change
    }

    /// Put the value from before `change` back, unless the rule was changed again since
    /// (a Settings edit after the prompt wins).
    mutating func undo(_ change: MeetingRuleChange) {
        guard rules[change.key] == change.new else { return }
        rules[change.key] = change.previous
    }

    /// Rows for Settings, by name.
    func sorted(name: (String) -> String) -> [(key: String, rule: MeetingRule)] {
        rules.map { (key: $0.key, rule: $0.value) }
            .sorted { name($0.key).localizedCaseInsensitiveCompare(name($1.key)) == .orderedAscending }
    }

    /// settings.json form; unknown rule values are dropped on read.
    static func decode(_ raw: [String: String]) -> MeetingRuleBook {
        var book = MeetingRuleBook()
        for (key, value) in raw {
            guard let rule = MeetingRule(rawValue: value), !key.isEmpty else { continue }
            book.set(rule, for: key)
        }
        return book
    }

    var encoded: [String: String] { rules.mapValues(\.rawValue) }

    init(from decoder: Decoder) throws {
        let raw = (try? decoder.singleValueContainer().decode([String: String].self)) ?? [:]
        self = Self.decode(raw)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(encoded)
    }
}
