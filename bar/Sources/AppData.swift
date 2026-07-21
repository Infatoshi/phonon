import AppKit
import ApplicationServices
import AVFoundation
import Foundation
import ScreenCaptureKit
import ServiceManagement

struct NativeSettings: Codable, Equatable {
    var schemaVersion: Int
    var streaming: Bool
    var localHistory: Bool
    var screenContext: Bool
    var microphonePriority: [String]
    var instantMic: Bool
    var shortcutMode: String

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case streaming
        case localHistory = "local_history"
        case screenContext = "screen_context"
        case microphonePriority = "microphone_priority"
        case instantMic = "instant_mic"
        case shortcutMode = "shortcut_mode"
    }

    init(
        schemaVersion: Int = 1,
        streaming: Bool = true,
        localHistory: Bool = true,
        screenContext: Bool = true,
        microphonePriority: [String] = ["Yeti", "MacBook Pro Microphone"],
        instantMic: Bool = true,
        shortcutMode: String = "both"
    ) {
        self.schemaVersion = schemaVersion
        self.streaming = streaming
        self.localHistory = localHistory
        self.screenContext = screenContext
        self.microphonePriority = microphonePriority
        self.instantMic = instantMic
        self.shortcutMode = shortcutMode
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try values.decodeIfPresent(Int.self, forKey: .schemaVersion) ?? 1
        streaming = try values.decodeIfPresent(Bool.self, forKey: .streaming) ?? true
        localHistory = try values.decodeIfPresent(Bool.self, forKey: .localHistory) ?? true
        screenContext = try values.decodeIfPresent(Bool.self, forKey: .screenContext) ?? true
        microphonePriority = try values.decodeIfPresent([String].self, forKey: .microphonePriority)
            ?? ["Yeti", "MacBook Pro Microphone"]
        instantMic = try values.decodeIfPresent(Bool.self, forKey: .instantMic) ?? true
        shortcutMode = try values.decodeIfPresent(String.self, forKey: .shortcutMode) ?? "both"
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

    enum CodingKeys: String, CodingKey {
        case id, source, microphone, llm
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

@MainActor
final class NativeAppStore: ObservableObject {
    @Published private(set) var settings = NativeSettings()
    @Published private(set) var dictionaryEntries: [NativeDictionaryEntry] = []
    @Published private(set) var history: [NativeHistoryItem] = []
    @Published private(set) var usage = NativeUsageStats()
    @Published var engineReady = false
    @Published var engineMessage = "Loading local models"
    @Published var selectedMicrophone = "Resolving…"
    @Published var inputMonitoringAvailable = false
    @Published var lastError: String?
    var onDictionaryChanged: (() -> Void)?
    var onMicrophonePermissionGranted: (() -> Void)?

    private let fileManager: FileManager
    private let supportDirectory: URL

    init(fileManager: FileManager = .default, supportDirectory: URL? = nil) {
        self.fileManager = fileManager
        self.supportDirectory = supportDirectory
            ?? PhononDataPaths.supportDirectory(fileManager: fileManager)
        reloadAll()
    }

    func reloadAll() {
        loadSettings()
        loadDictionary()
        loadHistory()
    }

    func updateSettings(_ mutate: (inout NativeSettings) -> Void) {
        var updated = settings
        mutate(&updated)
        updated.schemaVersion = 1
        settings = updated
        do {
            try writeJSON(updated, to: settingsURL)
            lastError = nil
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
            lastError = "Could not save intended transcription: \(error.localizedDescription)"
        }
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

    func refreshPermissions() {
        objectWillChange.send()
    }

    private func loadSettings() {
        let url = settingsURL
        guard let data = try? Data(contentsOf: url),
            let decoded = try? JSONDecoder().decode(NativeSettings.self, from: data)
        else {
            settings = NativeSettings()
            try? writeJSON(settings, to: url)
            return
        }
        settings = decoded
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

    private func writeJSONObject(_ object: [String: Any], to url: URL) throws {
        let data = try JSONSerialization.data(
            withJSONObject: object, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        try data.write(to: url, options: .atomic)
    }

    private var settingsURL: URL { supportDirectory.appendingPathComponent("settings.json") }
    private var dictionaryURL: URL { supportDirectory.appendingPathComponent("dictionary.json") }
    private var corpusURL: URL {
        supportDirectory.appendingPathComponent("Corpus", isDirectory: true)
    }
}
