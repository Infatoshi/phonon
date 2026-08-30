import AppKit
import AVFoundation
import SwiftUI

enum EmberTheme {
    static let background = Color(red: 0.067, green: 0.047, blue: 0.039)
    static let sidebar = Color(red: 0.094, green: 0.063, blue: 0.049)
    static let surface = Color(red: 0.129, green: 0.086, blue: 0.067)
    static let surfaceRaised = Color(red: 0.165, green: 0.106, blue: 0.075)
    static let border = Color(red: 0.31, green: 0.176, blue: 0.11)
    static let text = Color(red: 1.0, green: 0.957, blue: 0.925)
    static let muted = Color(red: 0.72, green: 0.61, blue: 0.55)
    static let accent = Color(red: 1.0, green: 0.35, blue: 0.12)
    static let accentSoft = Color(red: 0.235, green: 0.102, blue: 0.045)
    static let warm = Color(red: 0.965, green: 0.65, blue: 0.30)
    static let healthy = Color(red: 0.45, green: 0.78, blue: 0.60)

    static let nsBackground = NSColor(
        calibratedRed: 0.067, green: 0.047, blue: 0.039, alpha: 1)
}

enum NativeAppPage: String, CaseIterable, Identifiable {
    case home = "Home"
    case history = "History"
    case dictionary = "Dictionary"
    case settings = "Settings"

    var id: String { rawValue }
    var symbol: String {
        switch self {
        case .home: return "house"
        case .history: return "clock.arrow.circlepath"
        case .dictionary: return "text.book.closed"
        case .settings: return "gearshape"
        }
    }
}

struct PhononMainView: View {
    @ObservedObject var store: NativeAppStore
    let onSettingsChanged: () -> Void
    let onShowModelStatus: () -> Void
    @State private var page: NativeAppPage = .home
    @State private var showPrivacyChoice = false

    init(
        store: NativeAppStore,
        initialPage: NativeAppPage = .home,
        onSettingsChanged: @escaping () -> Void,
        onShowModelStatus: @escaping () -> Void
    ) {
        self.store = store
        self.onSettingsChanged = onSettingsChanged
        self.onShowModelStatus = onShowModelStatus
        _page = State(initialValue: initialPage)
    }

    var body: some View {
        HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 10) {
                    Image(systemName: "waveform.circle.fill")
                        .font(.title2)
                        .foregroundStyle(EmberTheme.accent)
                    Text("Phonon").font(.headline)
                }
                .padding(.horizontal, 14)
                .padding(.bottom, 14)

                ForEach(NativeAppPage.allCases) { candidate in
                    Button {
                        page = candidate
                    } label: {
                        Label(candidate.rawValue, systemImage: candidate.symbol)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, 11)
                            .padding(.vertical, 8)
                            .contentShape(Rectangle())
                            .background(
                                page == candidate ? EmberTheme.accentSoft : .clear,
                                in: RoundedRectangle(cornerRadius: 7)
                            )
                            .overlay {
                                if page == candidate {
                                    RoundedRectangle(cornerRadius: 7)
                                        .stroke(EmberTheme.border.opacity(0.7), lineWidth: 1)
                                }
                            }
                    }
                    .buttonStyle(.plain)
                }
                Spacer()
            }
            .padding(10)
            .frame(width: 190)
            .background(EmberTheme.sidebar)

            Divider()

            detailView
            .frame(minWidth: 700, maxWidth: .infinity, maxHeight: .infinity)
            .background(EmberTheme.background)
        }
        .frame(width: 980, height: 680)
        .foregroundStyle(EmberTheme.text)
        .background(EmberTheme.background)
        .tint(EmberTheme.accent)
        .preferredColorScheme(.dark)
        .alert(
            "Phonon",
            isPresented: Binding(
                get: { store.lastError != nil },
                set: { if !$0 { store.lastError = nil } }
            )
        ) {
            Button("OK") { store.lastError = nil }
        } message: {
            Text(store.lastError ?? "Unknown error")
        }
        .sheet(item: $store.permissionGuide) { guide in
            PermissionGuideView(guide: guide) {
                store.permissionGuide = nil
                NSApp.terminate(nil)
            }
        }
        .sheet(isPresented: $showPrivacyChoice) {
            PrivacyChoiceView(store: store) {
                showPrivacyChoice = false
                onSettingsChanged()
            }
        }
        .onAppear {
            showPrivacyChoice = store.needsPrivacyChoice
            store.pruneExpiredRecordings()
        }
    }

    private var detailView: AnyView {
        switch page {
        case .home:
            return AnyView(HomeView(store: store, onSettingsChanged: onSettingsChanged))
        case .history:
            return AnyView(HistoryView(store: store))
        case .dictionary:
            return AnyView(DictionaryView(store: store))
        case .settings:
            return AnyView(
                SettingsView(
                    store: store,
                    onSettingsChanged: onSettingsChanged,
                    onShowModelStatus: onShowModelStatus
                ))
        }
    }
}

struct HomeView: View {
    @ObservedObject var store: NativeAppStore
    let onSettingsChanged: () -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                HStack(spacing: 14) {
                    StatTile(
                        value: "\(store.usage.wordsPerMinute)",
                        label: "voice WPM",
                        symbol: "speedometer"
                    )
                    StatTile(
                        value: "\(store.usage.wordsToday)",
                        label: "words today",
                        symbol: "sun.max.fill"
                    )
                    StatTile(
                        value: "\(store.usage.words)",
                        label: "total words",
                        symbol: "text.word.spacing"
                    )
                }

                HStack(alignment: .top, spacing: 14) {
                    MicrophonePriorityCard(
                        store: store,
                        onSettingsChanged: onSettingsChanged
                    )
                    PermissionSummary(store: store)
                }

                Card {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Your activity").font(.headline)
                        HStack(spacing: 28) {
                            MetricColumn(
                                label: "Dictations", value: "\(store.usage.recordings)")
                            MetricColumn(
                                label: "Speaking time",
                                value: Self.duration(store.usage.speakingMilliseconds))
                            MetricColumn(
                                label: "Average length",
                                value: "\(store.usage.averageWordsPerRecording) words")
                            MetricColumn(
                                label: "Dictionary repairs",
                                value: "\(store.usage.dictionaryFixes)")
                            MetricColumn(
                                label: "Active days", value: "\(store.usage.activeDays)")
                        }
                    }
                }
            }
            .padding(28)
        }
        .navigationTitle("Home")
        .onAppear { store.refreshPermissions() }
    }

    private static func duration(_ milliseconds: UInt64) -> String {
        let totalSeconds = milliseconds / 1_000
        if totalSeconds < 60 { return "\(totalSeconds)s" }
        return "\(totalSeconds / 60)m \(totalSeconds % 60)s"
    }
}

struct HistoryView: View {
    @ObservedObject var store: NativeAppStore
    @State private var query = ""
    @State private var selectedID: String?
    @State private var intendedText = ""
    @State private var pendingDelete: NativeHistoryItem?

    private var filtered: [NativeHistoryItem] {
        guard !query.isEmpty else { return store.history }
        return store.history.filter { item in
            [item.displayText, item.metadata.rawTranscript, item.metadata.finalTranscript]
                .contains { $0.localizedCaseInsensitiveContains(query) }
        }
    }

    private var selected: NativeHistoryItem? {
        filtered.first(where: { $0.id == selectedID }) ?? filtered.first
    }

    var body: some View {
        VStack(spacing: 0) {
            HSplitView {
                VStack(spacing: 0) {
                    TextField("Search transcriptions", text: $query)
                        .textFieldStyle(.roundedBorder)
                        .padding(12)
                    List(filtered, selection: $selectedID) { item in
                        VStack(alignment: .leading, spacing: 5) {
                            Text(item.displayText.isEmpty ? "No speech" : item.displayText)
                                .lineLimit(2)
                            HStack {
                                Text(relativeDate(item.date))
                                if item.metadata.speechDetected == false { Text("No speech") }
                            }
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        }
                        .padding(.vertical, 4)
                        .tag(item.id)
                    }
                    .scrollContentBackground(.hidden)
                    .background(EmberTheme.background)
                }
                .frame(minWidth: 280, idealWidth: 330)

                Group {
                    if let item = selected {
                        ScrollView {
                            VStack(alignment: .leading, spacing: 18) {
                                HStack {
                                    VStack(alignment: .leading, spacing: 3) {
                                        Text(item.date.formatted(date: .abbreviated, time: .shortened))
                                            .font(.headline)
                                        Text(item.metadata.microphone ?? item.metadata.source)
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }
                                    Spacer()
                                    Button("Copy") {
                                        NSPasteboard.general.clearContents()
                                        NSPasteboard.general.setString(item.displayText, forType: .string)
                                    }
                                    Button(role: .destructive) { pendingDelete = item } label: {
                                        Image(systemName: "trash")
                                    }
                                }

                                TranscriptSection(title: "Final", text: item.metadata.finalTranscript)
                                if item.metadata.rawTranscript != item.metadata.finalTranscript {
                                    TranscriptSection(title: "Raw ASR", text: item.metadata.rawTranscript)
                                }

                                VStack(alignment: .leading, spacing: 8) {
                                    Text("Intended transcription").font(.headline)
                                    TextEditor(text: $intendedText)
                                        .font(.body)
                                        .frame(minHeight: 90)
                                        .padding(6)
                                        .background(EmberTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 8))
                                    HStack {
                                        Text("Saving this creates ground truth for future evaluation.")
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                        Spacer()
                                        Button("Save intended") {
                                            store.saveIntendedTranscript(itemID: item.id, text: intendedText)
                                        }
                                        .buttonStyle(.borderedProminent)
                                    }
                                }

                                if let llm = item.metadata.llm {
                                    HStack(spacing: 18) {
                                        if let value = llm.ttftMs { Label("TTFT \(value, specifier: "%.0f") ms", systemImage: "bolt") }
                                        if let value = llm.tokensPerSecond { Label("\(value, specifier: "%.0f") tok/s", systemImage: "speedometer") }
                                    }
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                }
                            }
                            .padding(24)
                        }
                        .id(item.id)
                        .onAppear { intendedText = item.metadata.intendedTranscript ?? item.displayText }
                        .onChange(of: item.id) { _, _ in
                            intendedText = item.metadata.intendedTranscript ?? item.displayText
                        }
                    } else {
                        ContentUnavailableView("No matching dictations", systemImage: "magnifyingglass")
                    }
                }
                .frame(minWidth: 430)
            }
        }
        .navigationTitle("History")
        .onAppear {
            if selectedID == nil { selectedID = filtered.first?.id }
        }
        .confirmationDialog(
            "Move this recording to Trash?", isPresented: Binding(
                get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } }
            ), titleVisibility: .visible
        ) {
            Button("Move to Trash", role: .destructive) {
                if let pendingDelete { store.trashRecording(itemID: pendingDelete.id) }
                pendingDelete = nil
            }
            Button("Cancel", role: .cancel) { pendingDelete = nil }
        }
    }
}

private func relativeDate(_ date: Date) -> String {
    let formatter = RelativeDateTimeFormatter()
    formatter.unitsStyle = .abbreviated
    return formatter.localizedString(for: date, relativeTo: Date())
}

struct DictionaryView: View {
    @ObservedObject var store: NativeAppStore
    @State private var query = ""
    @State private var editingID: String?
    @State private var phrase = ""
    @State private var replacement = ""
    @State private var spokenForms = ""
    @State private var pendingDelete: NativeDictionaryEntry?

    private var filtered: [NativeDictionaryEntry] {
        guard !query.isEmpty else { return store.dictionaryEntries }
        return store.dictionaryEntries.filter {
            $0.phrase.localizedCaseInsensitiveContains(query)
                || ($0.replacement?.localizedCaseInsensitiveContains(query) ?? false)
                || $0.spokenForms.contains { $0.localizedCaseInsensitiveContains(query) }
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            HSplitView {
                VStack(spacing: 0) {
                    TextField("Search terms", text: $query)
                        .textFieldStyle(.roundedBorder)
                        .padding(12)
                    List(filtered) { entry in
                        HStack {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(entry.replacement ?? entry.phrase).fontWeight(.medium)
                                if let replacement = entry.replacement {
                                    Text("\(entry.phrase) → \(replacement)")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                } else if !entry.spokenForms.isEmpty {
                                    Text(entry.spokenForms.joined(separator: ", "))
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                }
                            }
                            Spacer()
                            Button { edit(entry) } label: { Image(systemName: "pencil") }
                                .buttonStyle(.plain)
                            Button { pendingDelete = entry } label: { Image(systemName: "trash") }
                                .buttonStyle(.plain)
                                .foregroundStyle(.secondary)
                        }
                        .padding(.vertical, 4)
                    }
                    .scrollContentBackground(.hidden)
                    .background(EmberTheme.background)
                }
                .frame(minWidth: 360, idealWidth: 430)

                ScrollView {
                    VStack(alignment: .leading, spacing: 18) {
                        Text(editingID == nil ? "Add a term" : "Edit term")
                            .font(.title2.weight(.semibold))
                        LabeledContent("Written term") {
                            TextField("Blackwell", text: $phrase).frame(width: 280)
                        }
                        LabeledContent("Replace spoken phrase") {
                            TextField("Optional exact output", text: $replacement).frame(width: 280)
                        }
                        LabeledContent("Other spoken forms") {
                            TextField("Comma separated", text: $spokenForms).frame(width: 280)
                        }
                        Text("Leave replacement empty for a spelling term. Add a replacement when a spoken form should deterministically become different written text.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        HStack {
                            if editingID != nil {
                                Button("Cancel") { clearEditor() }
                            }
                            Spacer()
                            Button(editingID == nil ? "Add term" : "Save changes") {
                                store.upsertDictionary(
                                    originalID: editingID,
                                    phrase: phrase,
                                    replacement: replacement,
                                    spokenForms: spokenForms.split(separator: ",").map(String.init)
                                )
                                clearEditor()
                            }
                            .buttonStyle(.borderedProminent)
                            .disabled(phrase.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                        }
                    }
                    .padding(28)
                }
                .frame(minWidth: 400)
            }
        }
        .navigationTitle("Dictionary")
        .confirmationDialog(
            "Remove this dictionary entry?", isPresented: Binding(
                get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } }
            ), titleVisibility: .visible
        ) {
            Button("Remove", role: .destructive) {
                if let pendingDelete { store.removeDictionary(id: pendingDelete.id) }
                pendingDelete = nil
            }
            Button("Cancel", role: .cancel) { pendingDelete = nil }
        }
    }

    private func edit(_ entry: NativeDictionaryEntry) {
        editingID = entry.id
        phrase = entry.phrase
        replacement = entry.replacement ?? ""
        spokenForms = entry.spokenForms.joined(separator: ", ")
    }

    private func clearEditor() {
        editingID = nil
        phrase = ""
        replacement = ""
        spokenForms = ""
    }
}

struct SettingsView: View {
    @ObservedObject var store: NativeAppStore
    let onSettingsChanged: () -> Void
    let onShowModelStatus: () -> Void
    @State private var confirmClearHistory = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                SettingsSection("Dictation") {
                    ToggleRow(
                        title: "Streaming",
                        detail: "Produce acoustic partials while you speak.",
                        isOn: settingBinding(\.streaming)
                    )
                    Divider()
                    ToggleRow(
                        title: "Screen context",
                        detail: "Run local OCR and use only relevant dictionary terms.",
                        isOn: settingBinding(\.screenContext)
                    )
                    Divider()
                    ToggleRow(
                        title: "Local history",
                        detail: "Retain paired WAV and metadata after insertion.",
                        isOn: settingBinding(\.localHistory)
                    )
                    Divider()
                    ToggleRow(
                        title: "Instant microphone",
                        detail: "Keep CoreAudio warm for near-instant key-to-recording latency.",
                        isOn: settingBinding(\.instantMic)
                    )
                    Divider()
                    ToggleRow(
                        title: "Recording cues",
                        detail: "Play a short sweep when a recording opens and closes.",
                        isOn: settingBinding(\.soundFeedback)
                    )
                }

                SettingsSection("Backup") {
                    VStack(alignment: .leading, spacing: 3) {
                        Text(backupStatus)
                        Text(
                            "Uninstalling removes everything under ~/Library, so a copy "
                            + "of the dictionary, settings and history is kept in "
                            + "~/.phonon. Phonon offers it back if it ever starts empty."
                        )
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    }
                    Divider()
                    HStack {
                        VStack(alignment: .leading, spacing: 3) {
                            Text("Export everything")
                            Text("Copy the dictionary, settings, history and every recording.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button("Export…", action: exportEverything)
                    }
                    Divider()
                    HStack {
                        VStack(alignment: .leading, spacing: 3) {
                            Text("Forget the backup")
                            Text("Delete the copy in ~/.phonon. Your live data is untouched.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button("Show in Finder") { store.revealBackupInFinder() }
                            .disabled(store.mirrorManifest == nil)
                        Button("Delete") { store.deleteBackup() }
                            .disabled(store.mirrorManifest == nil)
                    }
                }
                SettingsSection("Stored recordings") {
                    Picker("Keep recordings for", selection: Binding(
                        get: { store.settings.historyRetentionDays },
                        set: { value in
                            store.updateSettings { $0.historyRetentionDays = value }
                            store.pruneExpiredRecordings()
                            onSettingsChanged()
                        }
                    )) {
                        Text("Until I delete them").tag(NativeSettings.keepRecordingsForever)
                        Text("7 days").tag(7)
                        Text("30 days").tag(30)
                        Text("90 days").tag(90)
                    }
                    .pickerStyle(.menu)
                    Divider()
                    HStack {
                        VStack(alignment: .leading, spacing: 3) {
                            Text("Clear all history")
                            Text("Move every stored recording and transcript to the Trash.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button("Clear") { confirmClearHistory = true }
                            .disabled(store.history.isEmpty)
                    }
                }

                SettingsSection("Shortcut") {
                    Picker("Active shortcut", selection: Binding(
                        get: { store.settings.shortcutMode },
                        set: { value in
                            store.updateSettings { $0.shortcutMode = value }
                            onSettingsChanged()
                        }
                    )) {
                        Text("Globe (fn) hold").tag("fn")
                        Text("Globe (fn) hold + Control Space toggle")
                            .tag("fn_and_control_space")
                        Text("Right Option hold").tag("right_option")
                        Text("Right Option hold + Control Space toggle").tag("both")
                        Text("Control Space toggle").tag("control_space")
                    }
                    .pickerStyle(.menu)
                    if store.settings.shortcutMode.hasPrefix("fn") {
                        Text(
                            "Hold Globe to talk; release to insert. Double-tap to keep "
                                + "recording, then tap once to stop."
                        )
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        Text(
                            "macOS also acts on the Globe key. Set System Settings › "
                                + "Keyboard › \"Press 🌐 key to\" to \"Do Nothing\" so it only "
                                + "records."
                        )
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    }
                }

                SettingsSection("System") {
                    ToggleRow(
                        title: "Launch at login",
                        detail: "Keep Phonon warm in the menu bar after signing in.",
                        isOn: Binding(
                            get: { store.launchAtLoginEnabled },
                            set: { store.setLaunchAtLogin($0) }
                        )
                    )
                    Divider()
                    HStack {
                        VStack(alignment: .leading, spacing: 3) {
                            Text("Model status")
                            Text("Inspect weights, smoke tests, and decode throughput.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button("Open", action: onShowModelStatus)
                    }
                }

            }
            .padding(28)
        }
        .navigationTitle("Settings")
        .onAppear { store.refreshPermissions() }
        .alert("Clear all history?", isPresented: $confirmClearHistory) {
            Button("Cancel", role: .cancel) {}
            Button("Move to Trash", role: .destructive) { store.clearAllHistory() }
        } message: {
            Text(
                "\(store.history.count) recordings and their transcripts go to the Trash. This cannot be undone from Phonon."
            )
        }
    }

    private var backupStatus: String {
        guard let manifest = store.mirrorManifest else {
            return "No backup yet. One is written the next time anything changes."
        }
        let stamp = DateFormatter.localizedString(
            from: manifest.savedAt, dateStyle: .medium, timeStyle: .short)
        return "Backed up \(stamp): \(manifest.dictionaryEntries) dictionary entries, "
            + "\(manifest.historyItems) recordings."
    }

    private func exportEverything() {
        let panel = NSSavePanel()
        panel.title = "Export Phonon data"
        panel.nameFieldStringValue = "Phonon Data"
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let url = panel.url else { return }
        store.exportEverything(to: url)
    }
    private func settingBinding(_ keyPath: WritableKeyPath<NativeSettings, Bool>) -> Binding<Bool> {
        Binding(
            get: { store.settings[keyPath: keyPath] },
            set: { value in
                store.updateSettings { $0[keyPath: keyPath] = value }
                onSettingsChanged()
            }
        )
    }

}

struct PermissionSummary: View {
    @ObservedObject var store: NativeAppStore

    var body: some View {
        Card {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Permissions and health").font(.headline)
                    Spacer()
                    Button("Refresh") { store.refreshPermissions() }
                        .buttonStyle(.borderless)
                }
                PermissionRow(
                    name: "Microphone", granted: store.microphonePermission,
                    statusText: store.microphoneStatusText,
                    actionTitle: store.microphoneActionTitle,
                    action: { store.performMicrophonePermissionAction() })
                PermissionRow(
                    name: "Accessibility", granted: store.accessibilityPermission,
                    action: { openPrivacy(.accessibility) })
                PermissionRow(
                    name: "Input Monitoring", granted: store.inputMonitoringAvailable,
                    actionTitle: store.inputMonitoringActionTitle,
                    action: { store.performInputMonitoringPermissionAction() })
                PermissionRow(
                    name: "Screen Recording", granted: store.screenRecordingPermission,
                    actionTitle: store.screenRecordingActionTitle,
                    action: { store.performScreenRecordingPermissionAction() })
            }
        }
    }

    private func openPrivacy(_ pane: PrivacyPane) {
        NSWorkspace.shared.open(pane.settingsURL)
    }
}

/// First run only. Both switches retain data, so neither is on until asked.
struct PrivacyChoiceView: View {
    @ObservedObject var store: NativeAppStore
    let onDone: () -> Void
    @State private var localHistory = false
    @State private var screenContext = false

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("What Phonon may keep")
                .font(.title2.bold())
            Text(
                "Dictation runs entirely on this Mac either way. These two features store or read more than the transcript, so they start off."
            )
            .foregroundStyle(.secondary)

            VStack(spacing: 0) {
                ToggleRow(
                    title: "Keep recordings and transcripts",
                    detail:
                        "Save the paired WAV and text after insertion so History, the dictionary, and accuracy work have something to learn from.",
                    isOn: $localHistory
                )
                Divider()
                ToggleRow(
                    title: "Read the active window",
                    detail:
                        "Run local OCR on the frontmost window to spell technical terms correctly. Nothing leaves the Mac.",
                    isOn: $screenContext
                )
            }
            .padding(14)
            .background(EmberTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 10))

            Text("Both can be changed any time in Settings, and History has a Clear all button.")
                .font(.caption)
                .foregroundStyle(.secondary)

            HStack {
                Spacer()
                Button("Continue") {
                    store.recordPrivacyChoice(
                        localHistory: localHistory, screenContext: screenContext)
                    onDone()
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(24)
        .frame(width: 540)
        .background(EmberTheme.background)
        .foregroundStyle(EmberTheme.text)
        .preferredColorScheme(.dark)
    }
}

struct PermissionGuideView: View {
    let guide: PermissionGuide
    let onDone: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text(guide.title)
                .font(.title2.bold())
            Text(guide.detail)
                .foregroundStyle(.secondary)

            HStack(spacing: 14) {
                Image(nsImage: NSWorkspace.shared.icon(forFile: Bundle.main.bundleURL.path))
                    .resizable()
                    .frame(width: 58, height: 58)
                VStack(alignment: .leading, spacing: 4) {
                    Text("Phonon.app").font(.headline)
                    Text("Installed in Applications and ready to add in System Settings.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(14)
            .background(EmberTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 10))

            Text(guide.manualInstructions)
                .font(.headline)
            Text("After enabling Phonon, quit it completely before reopening so macOS applies the permission.")
                .font(.callout)

            HStack {
                Button("Open Applications") {
                    NSWorkspace.shared.open(
                        URL(fileURLWithPath: "/Applications", isDirectory: true))
                }
                Spacer()
                Button("Open System Settings") {
                    NSWorkspace.shared.open(guide.pane.settingsURL)
                }
                .buttonStyle(.borderedProminent)
                Button("Quit Phonon", action: onDone)
            }
        }
        .padding(24)
        .frame(width: 540)
        .background(EmberTheme.background)
        .foregroundStyle(EmberTheme.text)
        .preferredColorScheme(.dark)
    }
}

enum PrivacyPane: String, CaseIterable {
    case microphone = "Privacy_Microphone"
    case accessibility = "Privacy_Accessibility"
    case inputMonitoring = "Privacy_ListenEvent"
    case screenRecording = "Privacy_ScreenCapture"

    var settingsURL: URL {
        URL(string: "x-apple.systempreferences:com.apple.preference.security?\(rawValue)")!
    }
}

struct Card<Content: View>: View {
    @ViewBuilder let content: Content
    var body: some View {
        content
            .padding(18)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(EmberTheme.surface, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .stroke(EmberTheme.border.opacity(0.72), lineWidth: 1)
            )
    }
}

struct StatTile: View {
    let value: String
    let label: String
    let symbol: String

    var body: some View {
        Card {
            HStack(spacing: 14) {
                Image(systemName: symbol)
                    .font(.title2)
                    .foregroundStyle(EmberTheme.warm)
                    .frame(width: 32)
                VStack(alignment: .leading, spacing: 2) {
                    Text(value)
                        .font(.title2.weight(.semibold).monospacedDigit())
                        .lineLimit(1)
                    Text(label)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }
}

struct MetricColumn: View {
    let label: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(value)
                .font(.body.weight(.medium).monospacedDigit())
                .lineLimit(1)
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct MicrophonePriorityCard: View {
    @ObservedObject var store: NativeAppStore
    let onSettingsChanged: () -> Void

    /// Three is enough to cover a headset, a desk microphone and the built-in one,
    /// and a shorter list is easier to reason about than an open-ended one.
    private static let maximumRanked = 3

    private var ranked: [String] { store.settings.microphonePriority }

    private var addable: [String] {
        store.availableMicrophones.filter { candidate in
            !ranked.contains { $0.localizedCaseInsensitiveCompare(candidate) == .orderedSame }
        }
    }

    private func isConnected(_ microphone: String) -> Bool {
        store.availableMicrophones.contains {
            $0.localizedCaseInsensitiveContains(microphone)
        }
    }

    var body: some View {
        Card {
            VStack(alignment: .leading, spacing: 12) {
                Text("Microphone priority").font(.headline)
                Text(
                    "Phonon follows the microphone selected in System Settings, except a Bluetooth headset mic, which would drop the headset to call quality. Rank microphones here to override that."
                )
                    .font(.caption)
                    .foregroundStyle(.secondary)

                if ranked.isEmpty {
                    Text("Nothing ranked, so Phonon follows the system input.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                ForEach(Array(ranked.enumerated()), id: \.offset) { index, microphone in
                    let connected = isConnected(microphone)
                    let active = microphone.localizedCaseInsensitiveCompare(
                        store.selectedMicrophone) == .orderedSame
                    HStack(spacing: 8) {
                        Text("\(index + 1)")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                            .frame(width: 16)
                        Image(systemName: connected ? "mic.fill" : "mic.slash")
                            .foregroundStyle(
                                active
                                    ? EmberTheme.healthy
                                    : (connected ? EmberTheme.muted : .secondary))
                        Text(microphone)
                            .lineLimit(1)
                            .foregroundStyle(connected ? .primary : .secondary)
                        if active {
                            Text("in use")
                                .font(.caption2)
                                .foregroundStyle(EmberTheme.healthy)
                        } else if !connected {
                            Text("not connected")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button {
                            moveMicrophone(from: index, by: -1)
                        } label: {
                            Image(systemName: "chevron.up")
                        }
                        .buttonStyle(.plain)
                        .disabled(index == 0)
                        Button {
                            moveMicrophone(from: index, by: 1)
                        } label: {
                            Image(systemName: "chevron.down")
                        }
                        .buttonStyle(.plain)
                        .disabled(index == ranked.count - 1)
                        Button {
                            removeMicrophone(at: index)
                        } label: {
                            Image(systemName: "xmark")
                        }
                        .buttonStyle(.plain)
                    }
                }

                // Only offer microphones CoreAudio actually reports. Typing a name
                // by hand used to let a ranking silently never match, for example
                // "Yeti" against a device really called "Yeti Stereo Microphone".
                if ranked.count < Self.maximumRanked {
                    Menu {
                        if addable.isEmpty {
                            Text("Every microphone found is already ranked")
                        }
                        ForEach(addable, id: \.self) { microphone in
                            Button(microphone) { addMicrophone(microphone) }
                        }
                    } label: {
                        Label("Add microphone", systemImage: "plus")
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()
                    .disabled(addable.isEmpty)
                } else {
                    Text("Remove one to rank a different microphone.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .onAppear { store.refreshAvailableMicrophones() }
    }

    private func moveMicrophone(from index: Int, by offset: Int) {
        let destination = index + offset
        guard store.settings.microphonePriority.indices.contains(destination) else { return }
        store.updateSettings {
            $0.microphonePriority.swapAt(index, destination)
        }
        onSettingsChanged()
    }

    private func removeMicrophone(at index: Int) {
        guard store.settings.microphonePriority.indices.contains(index) else { return }
        store.updateSettings {
            $0.microphonePriority.remove(at: index)
        }
        onSettingsChanged()
    }

    private func addMicrophone(_ name: String) {
        guard store.settings.microphonePriority.count < Self.maximumRanked,
              !store.settings.microphonePriority.contains(where: {
                  $0.localizedCaseInsensitiveCompare(name) == .orderedSame
              }) else { return }
        store.updateSettings { $0.microphonePriority.append(name) }
        onSettingsChanged()
    }
}

struct SettingsSection<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content
    init(_ title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title.uppercased())
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Card { VStack(spacing: 12) { content } }
        }
    }
}

struct ToggleRow: View {
    let title: String
    let detail: String
    @Binding var isOn: Bool
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                Text(detail).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            Toggle("", isOn: $isOn).labelsHidden()
        }
    }
}

struct PermissionRow: View {
    let name: String
    let granted: Bool
    var statusText: String?
    var actionTitle: String?
    let action: () -> Void
    var body: some View {
        HStack {
            Image(systemName: granted ? "checkmark.circle.fill" : "exclamationmark.circle.fill")
                .foregroundStyle(granted ? EmberTheme.healthy : EmberTheme.warm)
            Text(name)
            Spacer()
            Text(statusText ?? (granted ? "Granted" : "Needs access"))
                .font(.caption)
                .foregroundStyle(.secondary)
            Button(actionTitle ?? "Settings", action: action)
                .buttonStyle(.borderless)
        }
    }
}

struct MetricRow: View {
    let label: String
    let value: String
    var body: some View {
        HStack {
            Text(label).foregroundStyle(.secondary)
            Spacer()
            Text(value).font(.body.monospacedDigit())
        }
    }
}

struct TranscriptSection: View {
    let title: String
    let text: String
    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            Text(title).font(.headline)
            Text(text.isEmpty ? "No text" : text)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(12)
                .background(EmberTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 8))
        }
    }
}
