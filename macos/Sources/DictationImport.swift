// Import from the user's other dictation apps (SPEC "Import from other
// dictation apps").
//
// Detection is cheap and touches no protected folder: an app counts as present
// when its bundle is installed or its store exists under ~/Library. The scan
// reads each store read-only (SQLite databases through a copy, JSON files and
// preferences directly) and returns what Phonon can use: vocabulary, replacements,
// snippets, past dictations, the dictation shortcut and the chosen microphone.
// Nothing leaves the Mac and nothing is ever written to another app's files.

import AppKit
import CoreAudio
import Foundation
import SQLite3

// MARK: - Apps

enum DictationApp: String, CaseIterable, Identifiable, Codable {
    case wisprFlow = "wispr-flow"
    case aquaVoice = "aqua-voice"
    case superwhisper
    case voiceInk = "voiceink"
    case macWhisper = "macwhisper"
    case fluidVoice = "fluidvoice"

    var id: String { rawValue }

    var name: String {
        switch self {
        case .wisprFlow: return "Wispr Flow"
        case .aquaVoice: return "Aqua Voice"
        case .superwhisper: return "superwhisper"
        case .voiceInk: return "VoiceInk"
        case .macWhisper: return "MacWhisper"
        case .fluidVoice: return "FluidVoice"
        }
    }

    var bundleIDs: [String] {
        switch self {
        case .wisprFlow: return ["com.electron.wispr-flow"]
        case .aquaVoice: return ["com.electron.aqua-voice"]
        case .superwhisper: return ["com.superduper.superwhisper", "com.superduper.superwhisper-setapp"]
        case .voiceInk: return ["com.prakashjoshipax.VoiceInk"]
        case .macWhisper: return ["com.goodsnooze.MacWhisper"]
        case .fluidVoice: return ["com.FluidApp.app"]
        }
    }

    /// Stores under `~/Library/Application Support` whose presence means the app
    /// was used here even if it has since been deleted.
    var supportFolders: [String] {
        switch self {
        case .wisprFlow: return ["Wispr Flow"]
        case .aquaVoice: return ["Aqua Voice"]
        case .superwhisper: return ["superwhisper"]
        case .voiceInk: return ["com.prakashjoshipax.VoiceInk"]
        case .macWhisper: return ["MacWhisper"]
        case .fluidVoice: return ["FluidVoice"]
        }
    }
}

// MARK: - What a scan finds

/// One past dictation from another app. Written to `Imported/<app>.jsonl`.
struct ImportedDictation: Codable, Equatable {
    var id: String
    var source: String
    var timestamp: Date
    /// What the speech recognizer heard.
    var rawText: String
    /// What the app typed after its own cleanup.
    var refinedText: String
    /// What the user changed it to, when the app observed an edit.
    var editedText: String?
    /// Bundle id or app name it went into, when the app recorded one.
    var app: String?

    enum CodingKeys: String, CodingKey {
        case id, source, timestamp, app
        case rawText = "raw_text"
        case refinedText = "refined_text"
        case editedText = "edited_text"
    }

    /// The text the user ended up with.
    var finalText: String { editedText ?? refinedText }
}

struct DictationAppFindings: Identifiable, Equatable {
    let app: DictationApp
    var terms: [NativeDictionaryEntry] = []
    var macros: [PhononMacro] = []
    var dictations: [ImportedDictation] = []
    /// Dictations left out because they went into a private app or site.
    var skippedPrivate = 0
    /// settings.json `shortcut_mode` closest to the app's dictation key.
    var shortcutMode: String?
    var shortcutLabel: String?
    /// A CoreAudio input name (or a fragment of one) the user picked there.
    var microphone: String?
    /// Stores Phonon looked in that held something, for the "read from" line.
    var readPaths: [String] = []

    var id: String { app.rawValue }
    var wordCount: Int { terms.filter { $0.replacement == nil }.count }
    var replacementCount: Int { terms.filter { $0.replacement != nil }.count }

    var isEmpty: Bool {
        terms.isEmpty && macros.isEmpty && dictations.isEmpty && shortcutMode == nil
            && microphone == nil
    }

    /// "277 words · 8 replacements · 3 snippets · 1,223 dictations".
    var summary: [String] {
        var parts: [String] = []
        func count(_ n: Int, _ one: String, _ many: String) {
            guard n > 0 else { return }
            parts.append("\(n.formatted()) \(n == 1 ? one : many)")
        }
        count(wordCount, "word", "words")
        count(replacementCount, "replacement", "replacements")
        count(macros.count, "snippet", "snippets")
        count(dictations.count, "dictation", "dictations")
        if let shortcutLabel { parts.append("shortcut \(shortcutLabel)") }
        if let microphone { parts.append("mic \(microphone)") }
        return parts
    }
}

/// What an import changed, for the closing line of the sheet.
struct DictationImportSummary: Equatable {
    var addedTerms = 0
    var addedMacros = 0
    var savedDictations = 0
    var shortcutMode: String?
    var microphone: String?

    var isEmpty: Bool {
        addedTerms == 0 && addedMacros == 0 && savedDictations == 0 && shortcutMode == nil
            && microphone == nil
    }
}

// MARK: - Environment

/// Everything the scan reads from outside its files, injectable for tests.
struct DictationImportEnvironment {
    var home: URL
    /// A preferences domain as a dictionary (another app's `defaults`).
    var defaults: (String) -> [String: Any]?
    var isInstalled: (String) -> Bool
    /// A CoreAudio device UID to its name, for apps that store the UID.
    var microphoneName: (String) -> String?
    /// Apps and sites whose dictations are never imported (settings.json
    /// `screen_context_denylist`, plus `ContextPolicy.alwaysPrivate`).
    var denylist: [String]
    /// Where database copies are made.
    var scratch: URL

    var support: URL { home.appendingPathComponent("Library/Application Support", isDirectory: true) }

    static func live(denylist: [String]) -> DictationImportEnvironment {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return DictationImportEnvironment(
            home: home,
            defaults: { domain in
                if let domain = UserDefaults.standard.persistentDomain(forName: domain) {
                    return domain
                }
                let url = home.appendingPathComponent("Library/Preferences/\(domain).plist")
                guard let data = try? Data(contentsOf: url) else { return nil }
                return (try? PropertyListSerialization.propertyList(from: data, format: nil))
                    as? [String: Any]
            },
            isInstalled: { bundleID in
                NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundleID) != nil
            },
            microphoneName: CoreAudioUID.name(forUID:),
            denylist: denylist,
            scratch: FileManager.default.temporaryDirectory)
    }
}

enum CoreAudioUID {
    static func name(forUID uid: String) -> String? {
        var cfUID = uid as CFString
        var device = AudioDeviceID(0)
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyTranslateUIDToDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        let status = withUnsafeMutablePointer(to: &cfUID) { pointer in
            AudioObjectGetPropertyData(
                AudioObjectID(kAudioObjectSystemObject), &address,
                UInt32(MemoryLayout<CFString>.size), pointer, &size, &device)
        }
        guard status == noErr, device != 0 else { return nil }
        var nameAddress = AudioObjectPropertyAddress(
            mSelector: kAudioObjectPropertyName,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var name: Unmanaged<CFString>?
        var nameSize = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        guard AudioObjectGetPropertyData(device, &nameAddress, 0, nil, &nameSize, &name) == noErr,
            let value = name?.takeRetainedValue()
        else { return nil }
        return value as String
    }
}

// MARK: - Scanner

enum DictationImportScanner {
    /// Apps installed or with a store on this Mac, in table order. Reads no
    /// file contents and nothing under ~/Documents, so it never prompts.
    static func detect(_ env: DictationImportEnvironment) -> [DictationApp] {
        let fileManager = FileManager.default
        return DictationApp.allCases.filter { app in
            app.bundleIDs.contains(where: env.isInstalled)
                || app.supportFolders.contains {
                    fileManager.fileExists(
                        atPath: env.support.appendingPathComponent($0, isDirectory: true).path)
                }
                || app.bundleIDs.contains { !(env.defaults($0) ?? [:]).isEmpty }
        }
    }

    /// Read everything Phonon can use from the given apps. superwhisper keeps
    /// its files in ~/Documents by default, so macOS may ask once.
    static func scan(_ apps: [DictationApp], env: DictationImportEnvironment) -> [DictationAppFindings] {
        apps.map { scan($0, env: env) }
    }

    static func scan(_ app: DictationApp, env: DictationImportEnvironment) -> DictationAppFindings {
        switch app {
        case .wisprFlow: return WisprFlowReader.read(env)
        case .aquaVoice: return AquaVoiceReader.read(env)
        case .superwhisper: return SuperwhisperReader.read(env)
        case .voiceInk: return VoiceInkReader.read(env)
        case .macWhisper: return MacWhisperReader.read(env)
        case .fluidVoice: return FluidVoiceReader.read(env)
        }
    }

    // MARK: Shared helpers

    /// Phonon's `shortcut_mode` for the chords another app records with, or nil
    /// when none of them is a key Phonon offers.
    static func shortcutMode(for chords: [CompetitorChord]) -> String? {
        let kinds = Set(chords.map(\.kind))
        switch (kinds.contains(.fn), kinds.contains(.option), kinds.contains(.controlSpace)) {
        case (true, _, true): return "fn_and_control_space"
        case (true, _, false): return "fn"
        case (false, true, true): return "both"
        case (false, true, false): return "right_option"
        case (false, false, true): return "control_space"
        default: return nil
        }
    }

    static func shortcutLabel(for mode: String) -> String {
        switch mode {
        case "fn": return "Globe (fn)"
        case "fn_and_control_space": return "Globe (fn) + Control Space"
        case "right_option": return "Right Option"
        case "both": return "Right Option + Control Space"
        case "control_space": return "Control Space"
        default: return mode
        }
    }

    static func applyShortcut(_ chords: [CompetitorChord], to findings: inout DictationAppFindings) {
        guard let mode = shortcutMode(for: chords) else { return }
        findings.shortcutMode = mode
        findings.shortcutLabel = shortcutLabel(for: mode)
    }

    /// A microphone label as another app shows it, reduced to a fragment of the
    /// CoreAudio name: Chromium's "(046d:0ab7)" and Wispr's "(USB)" tags go,
    /// and labels that mean "whatever macOS picks" give nil.
    static func microphoneFragment(_ label: String?) -> String? {
        guard var name = label?.trimmingCharacters(in: .whitespacesAndNewlines), !name.isEmpty
        else { return nil }
        let lowered = name.lowercased()
        if lowered.hasPrefix("default") || lowered.hasPrefix("auto-detect")
            || lowered.hasPrefix("system default") || lowered.contains("(recommended)")
        {
            return nil
        }
        while let open = name.lastIndex(of: "("), name.hasSuffix(")") {
            name = String(name[..<open]).trimmingCharacters(in: .whitespaces)
        }
        name = name.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
        return name.isEmpty ? nil : name
    }

    static func isPrivate(app: String?, url: String?, env: DictationImportEnvironment) -> Bool {
        let host = url.flatMap { URL(string: $0)?.host ?? ContextPolicy.normalizeEntry($0) }
        return ContextPolicy.isDenied(bundleID: app, host: host, denylist: env.denylist)
    }

    /// Wispr Flow's `stripFormattingMarkupTags` (phonon-core `pairs::strip_markup`).
    static func stripMarkup(_ text: String?) -> String {
        var output = text ?? ""
        for tag in ["<ol>", "</ol>", "<li>", "</li>", "<ul>", "</ul>"] {
            output = output.replacingOccurrences(of: tag, with: "")
        }
        return output.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func clean(_ text: String?) -> String {
        (text ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func json(at url: URL) -> Any? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONSerialization.jsonObject(with: data)
    }

    /// Values in preferences that hold JSON as a string or as data.
    static func jsonValue(_ value: Any?) -> Any? {
        if let data = value as? Data { return try? JSONSerialization.jsonObject(with: data) }
        if let string = value as? String, let data = string.data(using: .utf8) {
            return try? JSONSerialization.jsonObject(with: data)
        }
        return value
    }

    /// Core Data and `Date.timeIntervalSinceReferenceDate` seconds.
    static func referenceDate(_ value: Any?) -> Date? {
        if let seconds = value as? Double { return Date(timeIntervalSinceReferenceDate: seconds) }
        if let seconds = value as? Int64 { return Date(timeIntervalSinceReferenceDate: Double(seconds)) }
        if let seconds = value as? Int { return Date(timeIntervalSinceReferenceDate: Double(seconds)) }
        return nil
    }

    /// `2026-10-12 16:52:42.760 +00:00` (Wispr, Sequelize), `2026-10-12
    /// 16:52:42.760` (GRDB, UTC), ISO 8601 with or without fractions.
    static func parseDate(_ value: Any?, localIfNoZone: Bool = false) -> Date? {
        guard let text = (value as? String)?.trimmingCharacters(in: .whitespaces), !text.isEmpty
        else { return nil }
        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = iso.date(from: text) { return date }
        iso.formatOptions = [.withInternetDateTime]
        if let date = iso.date(from: text) { return date }
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        for format in ["yyyy-MM-dd HH:mm:ss.SSS xxx", "yyyy-MM-dd HH:mm:ss xxx"] {
            formatter.dateFormat = format
            if let date = formatter.date(from: text) { return date }
        }
        formatter.timeZone = localIfNoZone ? .current : TimeZone(identifier: "UTC")
        for format in [
            "yyyy-MM-dd HH:mm:ss.SSS", "yyyy-MM-dd HH:mm:ss", "yyyy-MM-dd'T'HH:mm:ss.SSS",
            "yyyy-MM-dd'T'HH:mm:ss",
        ] {
            formatter.dateFormat = format
            if let date = formatter.date(from: text) { return date }
        }
        return nil
    }

    /// Keep a dictation unless it is empty or went somewhere private.
    static func add(
        _ dictation: ImportedDictation, url: String? = nil, to findings: inout DictationAppFindings,
        env: DictationImportEnvironment
    ) {
        guard !dictation.finalText.isEmpty || !dictation.rawText.isEmpty else { return }
        if isPrivate(app: dictation.app, url: url, env: env) {
            findings.skippedPrivate += 1
            return
        }
        findings.dictations.append(dictation)
    }

    static func term(
        _ phrase: String?, replacement: String? = nil, spokenForms: [String] = [],
        app: DictationApp, starred: Bool = false, usageCount: UInt64 = 0
    ) -> NativeDictionaryEntry? {
        let phrase = clean(phrase)
        guard !phrase.isEmpty else { return nil }
        let replacement = clean(replacement)
        return NativeDictionaryEntry(
            phrase: phrase, replacement: replacement.isEmpty ? nil : replacement,
            spokenForms: spokenForms.map { clean($0) }.filter { !$0.isEmpty && $0 != phrase },
            source: app.rawValue, starred: starred, usageCount: usageCount)
    }
}

// MARK: - SQLite through a copy

/// Read a SQLite database through a copy (with its `-wal` and `-shm`) in a
/// scratch directory, so the owning app's files are never opened, locked or
/// checkpointed. On APFS the copy is a clone and costs nothing.
enum SQLiteSnapshot {
    enum Failure: Error, Equatable {
        case missing
        case open(String)
        case query(String)
    }

    static func rows(
        _ database: URL, queries: [String], scratch: URL
    ) throws -> [[[String: Any]]] {
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: database.path) else { throw Failure.missing }
        let directory = scratch.appendingPathComponent(
            "phonon-import-\(UUID().uuidString)", isDirectory: true)
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: directory) }
        let copy = directory.appendingPathComponent(database.lastPathComponent)
        try fileManager.copyItem(at: database, to: copy)
        for suffix in ["-wal", "-shm"] {
            let side = URL(fileURLWithPath: database.path + suffix)
            if fileManager.fileExists(atPath: side.path) {
                try fileManager.copyItem(at: side, to: URL(fileURLWithPath: copy.path + suffix))
            }
        }
        var handle: OpaquePointer?
        guard sqlite3_open_v2(copy.path, &handle, SQLITE_OPEN_READWRITE, nil) == SQLITE_OK,
            let db = handle
        else {
            let message = handle.map { String(cString: sqlite3_errmsg($0)) } ?? "open failed"
            sqlite3_close(handle)
            throw Failure.open(message)
        }
        defer { sqlite3_close(db) }
        return try queries.map { try run(db, $0) }
    }

    static func rows(_ database: URL, query: String, scratch: URL) throws -> [[String: Any]] {
        try rows(database, queries: [query], scratch: scratch)[0]
    }

    private static func run(_ db: OpaquePointer, _ query: String) throws -> [[String: Any]] {
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, query, -1, &statement, nil) == SQLITE_OK, let statement else {
            throw Failure.query(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(statement) }
        var result: [[String: Any]] = []
        while true {
            let step = sqlite3_step(statement)
            if step == SQLITE_DONE { break }
            guard step == SQLITE_ROW else { throw Failure.query(String(cString: sqlite3_errmsg(db))) }
            var row: [String: Any] = [:]
            for column in 0..<sqlite3_column_count(statement) {
                let name = String(cString: sqlite3_column_name(statement, column))
                switch sqlite3_column_type(statement, column) {
                case SQLITE_INTEGER: row[name] = sqlite3_column_int64(statement, column)
                case SQLITE_FLOAT: row[name] = sqlite3_column_double(statement, column)
                case SQLITE_TEXT:
                    if let text = sqlite3_column_text(statement, column) {
                        row[name] = String(cString: text)
                    }
                default: break
                }
            }
            result.append(row)
        }
        return result
    }

    static func tables(_ database: URL, scratch: URL) -> Set<String> {
        let rows = (try? rows(
            database, query: "SELECT name FROM sqlite_master WHERE type = 'table'",
            scratch: scratch)) ?? []
        return Set(rows.compactMap { $0["name"] as? String })
    }
}

// MARK: - Wispr Flow

/// `~/Library/Application Support/Wispr Flow/`: `flow.sqlite` (tables
/// `Dictionary` and `History`) and `config.json` (`prefs.user.shortcuts`,
/// `overrideAudioDeviceId` into `rankedAudioDevices`).
enum WisprFlowReader {
    static func read(_ env: DictationImportEnvironment) -> DictationAppFindings {
        var findings = DictationAppFindings(app: .wisprFlow)
        let folder = env.support.appendingPathComponent("Wispr Flow", isDirectory: true)
        let database = folder.appendingPathComponent("flow.sqlite")
        if let results = try? SQLiteSnapshot.rows(
            database,
            queries: [
                "SELECT phrase, replacement, isSnippet, isStarred, frequencyUsed FROM Dictionary "
                    + "WHERE isDeleted = 0 ORDER BY phrase COLLATE NOCASE",
                "SELECT transcriptEntityId AS id, asrText, formattedText, editedText, timestamp, "
                    + "app, url FROM History WHERE isArchived = 0 ORDER BY timestamp",
            ], scratch: env.scratch)
        {
            findings.readPaths.append(database.path)
            for row in results[0] {
                let phrase = row["phrase"] as? String
                let replacement = row["replacement"] as? String
                if (row["isSnippet"] as? Int64 ?? 0) != 0 {
                    let trigger = DictationImportScanner.clean(phrase)
                    let expansion = DictationImportScanner.clean(replacement)
                    if !trigger.isEmpty, !expansion.isEmpty {
                        findings.macros.append(PhononMacro(trigger: trigger, expansion: expansion))
                    }
                } else if let entry = DictationImportScanner.term(
                    phrase, replacement: replacement, app: .wisprFlow,
                    starred: (row["isStarred"] as? Int64 ?? 0) != 0,
                    usageCount: UInt64(max(0, row["frequencyUsed"] as? Int64 ?? 0)))
                {
                    findings.terms.append(entry)
                }
            }
            for row in results[1] {
                guard let id = row["id"] as? String,
                    let date = DictationImportScanner.parseDate(row["timestamp"])
                else { continue }
                let raw = DictationImportScanner.stripMarkup(row["asrText"] as? String)
                var refined = DictationImportScanner.stripMarkup(row["formattedText"] as? String)
                if refined.isEmpty { refined = raw }
                let edited = DictationImportScanner.stripMarkup(row["editedText"] as? String)
                DictationImportScanner.add(
                    ImportedDictation(
                        id: id, source: DictationApp.wisprFlow.rawValue, timestamp: date,
                        rawText: raw, refinedText: refined,
                        editedText: edited.isEmpty || edited == refined ? nil : edited,
                        app: row["app"] as? String),
                    url: row["url"] as? String, to: &findings, env: env)
            }
        }
        let config = folder.appendingPathComponent("config.json")
        if let json = DictationImportScanner.json(at: config) {
            findings.readPaths.append(config.path)
            DictationImportScanner.applyShortcut(dictationChords(config: json), to: &findings)
            findings.microphone = microphone(config: json)
        }
        return findings
    }

    /// `prefs.user.shortcuts`: `{"63": "ptt", "49+63": "popo", "53": "dismiss"}`;
    /// only push-to-talk and hands-free start a dictation.
    static func dictationChords(config: Any) -> [CompetitorChord] {
        guard let shortcuts = user(config)?["shortcuts"] as? [String: Any] else { return [] }
        return shortcuts.keys.sorted().compactMap { key in
            guard let action = shortcuts[key] as? String, action == "ptt" || action == "popo"
            else { return nil }
            let codes = key.split(separator: "+").compactMap {
                Int($0.trimmingCharacters(in: .whitespaces))
            }
            return codes.isEmpty ? nil : CompetitorChord.fromKeyCodes(codes)
        }
    }

    /// The microphone the user picked in Wispr (`overrideAudioDeviceId`), by
    /// its name in `rankedAudioDevices`.
    static func microphone(config: Any) -> String? {
        guard let user = user(config), let chosen = user["overrideAudioDeviceId"] as? String,
            let ranked = user["rankedAudioDevices"] as? [[String: Any]],
            let device = ranked.first(where: { ($0["deviceId"] as? String) == chosen })
        else { return nil }
        return DictationImportScanner.microphoneFragment(device["name"] as? String)
    }

    private static func user(_ config: Any) -> [String: Any]? {
        ((config as? [String: Any])?["prefs"] as? [String: Any])?["user"] as? [String: Any]
    }
}

// MARK: - Aqua Voice

/// `~/Library/Application Support/Aqua Voice/settings.json`: `dictionary`
/// (strings), `replacements` (`{from, to}`), `history` (`rawText`, `content`,
/// `timestamp`, up to 100), `hotkeys`, `nativeMicrophone` / `jsMicrophone`.
enum AquaVoiceReader {
    static func read(_ env: DictationImportEnvironment) -> DictationAppFindings {
        var findings = DictationAppFindings(app: .aquaVoice)
        let url = env.support.appendingPathComponent("Aqua Voice/settings.json")
        guard let root = DictationImportScanner.json(at: url) as? [String: Any] else {
            return findings
        }
        findings.readPaths.append(url.path)
        for word in root["dictionary"] as? [Any] ?? [] {
            if let entry = DictationImportScanner.term(word as? String, app: .aquaVoice) {
                findings.terms.append(entry)
            }
        }
        for item in root["replacements"] as? [[String: Any]] ?? [] {
            if let entry = DictationImportScanner.term(
                item["from"] as? String, replacement: item["to"] as? String, app: .aquaVoice),
                entry.replacement != nil
            {
                findings.terms.append(entry)
            }
        }
        for (index, item) in (root["history"] as? [[String: Any]] ?? []).enumerated() {
            guard let date = DictationImportScanner.parseDate(item["timestamp"]) else { continue }
            let raw = DictationImportScanner.clean(item["rawText"] as? String)
            let content = DictationImportScanner.clean(item["content"] as? String)
            let id = (item["sessionId"]).map { "\($0)" } ?? "aqua-\(index)"
            DictationImportScanner.add(
                ImportedDictation(
                    id: id, source: DictationApp.aquaVoice.rawValue, timestamp: date,
                    rawText: raw, refinedText: content.isEmpty ? raw : content),
                to: &findings, env: env)
        }
        let hotkeys = (root["hotkeys"] as? [[String: Any]] ?? []).compactMap {
            entry -> CompetitorChord? in
            guard let action = entry["action"] as? String, action == "activate" || action == "lock",
                let keys = entry["keys"] as? String, !keys.isEmpty
            else { return nil }
            return CompetitorChord.fromTokens(keys.split(separator: "+").map(String.init))
        }
        DictationImportScanner.applyShortcut(hotkeys, to: &findings)
        let native = (root["useNativeAudio"] as? Bool) ?? false
        let microphone =
            (root[native ? "nativeMicrophone" : "jsMicrophone"] as? [String: Any])?["name"]
            as? String
        findings.microphone = DictationImportScanner.microphoneFragment(microphone)
        return findings
    }
}

// MARK: - superwhisper

/// Preferences `com.superduper.superwhisper` (`appFolderDirectory`, default
/// ~/Documents; `KeyboardShortcuts_toggleRecording`), then under
/// `<folder>/superwhisper/`: `settings/settings.json` (`vocabulary`,
/// `replacements` `{original, with}`) and `recordings/<epoch>/meta.json`
/// (`rawResult`, `result`, `llmResult`, `datetime`, `recordingDevice`).
enum SuperwhisperReader {
    static let domains = DictationApp.superwhisper.bundleIDs

    static func read(_ env: DictationImportEnvironment) -> DictationAppFindings {
        var findings = DictationAppFindings(app: .superwhisper)
        let prefs = domains.lazy.compactMap(env.defaults).first ?? [:]
        let base = (prefs["appFolderDirectory"] as? String).map {
            URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath, isDirectory: true)
        } ?? env.home.appendingPathComponent("Documents", isDirectory: true)
        let folder = base.appendingPathComponent("superwhisper", isDirectory: true)
        let settingsURL = folder.appendingPathComponent("settings/settings.json")
        if let settings = DictationImportScanner.json(at: settingsURL) as? [String: Any] {
            findings.readPaths.append(settingsURL.path)
            for word in settings["vocabulary"] as? [Any] ?? [] {
                if let entry = DictationImportScanner.term(word as? String, app: .superwhisper) {
                    findings.terms.append(entry)
                }
            }
            for item in settings["replacements"] as? [[String: Any]] ?? [] {
                if let entry = DictationImportScanner.term(
                    item["original"] as? String, replacement: item["with"] as? String,
                    app: .superwhisper), entry.replacement != nil
                {
                    findings.terms.append(entry)
                }
            }
        }
        let recordings = folder.appendingPathComponent("recordings", isDirectory: true)
        let names = ((try? FileManager.default.contentsOfDirectory(atPath: recordings.path)) ?? [])
            .sorted()
        var lastDevice: String?
        for name in names {
            let meta = recordings.appendingPathComponent(name).appendingPathComponent("meta.json")
            guard let item = DictationImportScanner.json(at: meta) as? [String: Any] else { continue }
            let date = Double(name).map { Date(timeIntervalSince1970: $0) }
                ?? DictationImportScanner.parseDate(item["datetime"], localIfNoZone: true)
            guard let date else { continue }
            let raw = DictationImportScanner.clean(item["rawResult"] as? String)
            var refined = DictationImportScanner.clean(item["llmResult"] as? String)
            if refined.isEmpty { refined = DictationImportScanner.clean(item["result"] as? String) }
            if refined.isEmpty { refined = raw }
            let context = (item["promptContext"] as? [String: Any])?["applicationContext"]
                as? [String: Any]
            let app = (context?["bundleIdentifier"] as? String)
                ?? (context?["name"] as? String).flatMap { $0.isEmpty ? nil : $0 }
            DictationImportScanner.add(
                ImportedDictation(
                    id: name, source: DictationApp.superwhisper.rawValue, timestamp: date,
                    rawText: raw, refinedText: refined, app: app),
                to: &findings, env: env)
            if let device = item["recordingDevice"] as? String, !device.isEmpty {
                lastDevice = device
            }
        }
        if !names.isEmpty { findings.readPaths.append(recordings.path) }
        findings.microphone = DictationImportScanner.microphoneFragment(lastDevice)
        if case let .chords(chords) = CompetitorHotkeyReader.voiceInk(
            shortcut: DictationImportScanner.jsonValue(prefs["KeyboardShortcuts_toggleRecording"]))
        {
            DictationImportScanner.applyShortcut(chords, to: &findings)
        }
        return findings
    }
}

// MARK: - VoiceInk

/// `~/Library/Application Support/com.prakashjoshipax.VoiceInk/`: SwiftData
/// stores `dictionary.store` (2.x) and `default.store` (`ZVOCABULARYWORD.ZWORD`,
/// `ZWORDREPLACEMENT` `ZORIGINALTEXT` (comma-separated) to `ZREPLACEMENTTEXT`,
/// `ZTRANSCRIPTION` `ZTEXT` / `ZENHANCEDTEXT` / `ZTIMESTAMP`). Preferences
/// `com.prakashjoshipax.VoiceInk`: `CustomRecordingShortcut_primary` or
/// `Shortcut_primaryRecording`, `selectedAudioDeviceUID`; 1.x kept
/// `CustomVocabularyItems` and `wordReplacements` there too.
enum VoiceInkReader {
    static let domain = "com.prakashjoshipax.VoiceInk"

    static func read(_ env: DictationImportEnvironment) -> DictationAppFindings {
        var findings = DictationAppFindings(app: .voiceInk)
        let folder = env.support.appendingPathComponent(domain, isDirectory: true)
        for store in ["dictionary.store", "default.store"] {
            let url = folder.appendingPathComponent(store)
            let tables = SQLiteSnapshot.tables(url, scratch: env.scratch)
            guard !tables.isEmpty else { continue }
            findings.readPaths.append(url.path)
            let queries = [
                tables.contains("ZVOCABULARYWORD")
                    ? "SELECT ZWORD AS word FROM ZVOCABULARYWORD" : "SELECT 1 WHERE 0",
                tables.contains("ZWORDREPLACEMENT")
                    ? "SELECT ZORIGINALTEXT AS original, ZREPLACEMENTTEXT AS replacement, "
                        + "ZISENABLED AS enabled FROM ZWORDREPLACEMENT" : "SELECT 1 WHERE 0",
                // dictionary.store carries the same model, but history lives in default.store.
                tables.contains("ZTRANSCRIPTION") && store == "default.store"
                    ? "SELECT Z_PK AS pk, ZTEXT AS text, ZENHANCEDTEXT AS enhanced, "
                        + "ZTIMESTAMP AS timestamp FROM ZTRANSCRIPTION ORDER BY ZTIMESTAMP"
                    : "SELECT 1 WHERE 0",
            ]
            guard let results = try? SQLiteSnapshot.rows(url, queries: queries, scratch: env.scratch)
            else { continue }
            for row in results[0] {
                if let entry = DictationImportScanner.term(row["word"] as? String, app: .voiceInk) {
                    findings.terms.append(entry)
                }
            }
            for row in results[1] where (row["enabled"] as? Int64 ?? 1) != 0 {
                findings.terms += replacements(
                    original: row["original"] as? String,
                    replacement: row["replacement"] as? String)
            }
            for row in results[2] {
                guard let date = DictationImportScanner.referenceDate(row["timestamp"]) else {
                    continue
                }
                let raw = DictationImportScanner.clean(row["text"] as? String)
                let enhanced = DictationImportScanner.clean(row["enhanced"] as? String)
                DictationImportScanner.add(
                    ImportedDictation(
                        id: "\(store)-\(row["pk"] as? Int64 ?? 0)",
                        source: DictationApp.voiceInk.rawValue, timestamp: date,
                        rawText: raw, refinedText: enhanced.isEmpty ? raw : enhanced),
                    to: &findings, env: env)
            }
        }
        let prefs = env.defaults(domain) ?? [:]
        if let items = DictationImportScanner.jsonValue(prefs["CustomVocabularyItems"]) as? [[String: Any]] {
            for item in items {
                if let entry = DictationImportScanner.term(item["word"] as? String, app: .voiceInk) {
                    findings.terms.append(entry)
                }
            }
        }
        if let legacy = DictationImportScanner.jsonValue(prefs["wordReplacements"]) as? [String: String] {
            for (original, replacement) in legacy.sorted(by: { $0.key < $1.key }) {
                findings.terms += replacements(original: original, replacement: replacement)
            }
        }
        for key in ["CustomRecordingShortcut_primary", "Shortcut_primaryRecording"] {
            if case let .chords(chords) = CompetitorHotkeyReader.voiceInk(
                shortcut: DictationImportScanner.jsonValue(prefs[key])), !chords.isEmpty
            {
                DictationImportScanner.applyShortcut(chords, to: &findings)
                break
            }
        }
        if let uid = prefs["selectedAudioDeviceUID"] as? String, !uid.isEmpty {
            findings.microphone = env.microphoneName(uid)
        }
        findings.terms = unique(findings.terms)
        return findings
    }

    /// VoiceInk lets one replacement cover several spellings: "a, b -> c".
    static func replacements(original: String?, replacement: String?) -> [NativeDictionaryEntry] {
        (original ?? "").split(separator: ",").compactMap {
            DictationImportScanner.term(String($0), replacement: replacement, app: .voiceInk)
        }.filter { $0.replacement != nil }
    }

    /// 2.x keeps a copy of the 1.x dictionary in both stores.
    private static func unique(_ terms: [NativeDictionaryEntry]) -> [NativeDictionaryEntry] {
        var seen = Set<String>()
        return terms.filter { seen.insert($0.id).inserted }
    }
}

// MARK: - MacWhisper

/// `~/Library/Application Support/MacWhisper/Database/main.sqlite`, table
/// `dictation` (`transcribedText`, `processedText`, `dateCreated`,
/// `targetAppBundleID`, `dateDeleted`). Its replacement list and dictation key
/// are not in a location Phonon has verified, so only history is read.
enum MacWhisperReader {
    static func read(_ env: DictationImportEnvironment) -> DictationAppFindings {
        var findings = DictationAppFindings(app: .macWhisper)
        let url = env.support.appendingPathComponent("MacWhisper/Database/main.sqlite")
        guard let rows = try? SQLiteSnapshot.rows(
            url,
            query: "SELECT hex(id) AS id, transcribedText, processedText, dateCreated, "
                + "targetAppBundleID FROM dictation WHERE dateDeleted IS NULL ORDER BY dateCreated",
            scratch: env.scratch)
        else { return findings }
        findings.readPaths.append(url.path)
        for row in rows {
            guard let date = DictationImportScanner.parseDate(row["dateCreated"])
                ?? DictationImportScanner.referenceDate(row["dateCreated"])
            else { continue }
            let raw = DictationImportScanner.clean(row["transcribedText"] as? String)
            let processed = DictationImportScanner.clean(row["processedText"] as? String)
            DictationImportScanner.add(
                ImportedDictation(
                    id: (row["id"] as? String) ?? UUID().uuidString,
                    source: DictationApp.macWhisper.rawValue, timestamp: date, rawText: raw,
                    refinedText: processed.isEmpty ? raw : processed,
                    app: row["targetAppBundleID"] as? String),
                to: &findings, env: env)
        }
        return findings
    }
}

// MARK: - FluidVoice

/// `~/Library/Application Support/FluidVoice/parakeet_custom_vocabulary.json`
/// (`terms[]` with `text` and `aliases`, which become spoken forms) and
/// preferences `com.FluidApp.app`: `TranscriptionHistoryEntries` (JSON:
/// `rawText`, `processedText`, `timestamp`, `appName`), `HotkeyShortcutKey`.
enum FluidVoiceReader {
    static let domain = "com.FluidApp.app"

    static func read(_ env: DictationImportEnvironment) -> DictationAppFindings {
        var findings = DictationAppFindings(app: .fluidVoice)
        let url = env.support.appendingPathComponent("FluidVoice/parakeet_custom_vocabulary.json")
        if let root = DictationImportScanner.json(at: url) as? [String: Any] {
            findings.readPaths.append(url.path)
            for item in root["terms"] as? [[String: Any]] ?? [] {
                if let entry = DictationImportScanner.term(
                    item["text"] as? String,
                    spokenForms: (item["aliases"] as? [String]) ?? [], app: .fluidVoice)
                {
                    findings.terms.append(entry)
                }
            }
        }
        let prefs = env.defaults(domain) ?? [:]
        for item in DictationImportScanner.jsonValue(prefs["TranscriptionHistoryEntries"])
            as? [[String: Any]] ?? []
        {
            guard let date = DictationImportScanner.referenceDate(item["timestamp"]) else { continue }
            let raw = DictationImportScanner.clean(item["rawText"] as? String)
            let processed = DictationImportScanner.clean(item["processedText"] as? String)
            DictationImportScanner.add(
                ImportedDictation(
                    id: (item["id"] as? String) ?? UUID().uuidString,
                    source: DictationApp.fluidVoice.rawValue, timestamp: date, rawText: raw,
                    refinedText: processed.isEmpty ? raw : processed,
                    app: item["appName"] as? String),
                to: &findings, env: env)
        }
        if case let .chords(chords) = CompetitorHotkeyReader.voiceInk(
            shortcut: DictationImportScanner.jsonValue(prefs["HotkeyShortcutKey"]))
        {
            DictationImportScanner.applyShortcut(chords, to: &findings)
        }
        return findings
    }
}

// MARK: - Writing what was imported

enum DictationImportFiles {
    /// `Imported/<app>.jsonl`: one dictation per line, raw and final text, for
    /// examples and corrections. Rewritten on every import, so importing twice
    /// does not double anything.
    static func historyURL(_ app: DictationApp, support: URL) -> URL {
        support.appendingPathComponent("Imported", isDirectory: true)
            .appendingPathComponent("\(app.rawValue).jsonl")
    }

    /// `profile/extract/<app>.txt`: the user's final text, one dictation per
    /// line, which is the vocabulary miner's input (SPEC "User vocabulary
    /// onboarding", step 2).
    static func extractURL(_ app: DictationApp, support: URL) -> URL {
        support.appendingPathComponent("profile/extract", isDirectory: true)
            .appendingPathComponent("\(app.rawValue).txt")
    }

    static func write(_ dictations: [ImportedDictation], app: DictationApp, support: URL) throws {
        let fileManager = FileManager.default
        let history = historyURL(app, support: support)
        let extract = extractURL(app, support: support)
        try fileManager.createDirectory(
            at: history.deletingLastPathComponent(), withIntermediateDirectories: true)
        try fileManager.createDirectory(
            at: extract.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        var jsonl = Data()
        var lines: [String] = []
        for dictation in dictations.sorted(by: { $0.timestamp < $1.timestamp }) {
            jsonl.append(try encoder.encode(dictation))
            jsonl.append(0x0A)
            let line = dictation.finalText.split(whereSeparator: { $0.isWhitespace })
                .joined(separator: " ")
            if !line.isEmpty { lines.append(line) }
        }
        try jsonl.write(to: history, options: .atomic)
        try Data((lines.joined(separator: "\n") + (lines.isEmpty ? "" : "\n")).utf8)
            .write(to: extract, options: .atomic)
    }
}
