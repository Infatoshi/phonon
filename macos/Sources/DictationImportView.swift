import AppKit
import SwiftUI

/// Opens the import sheet: from first run (after the privacy choices, with the
/// apps already detected) or from Settings.
struct DictationImportRequest: Identifiable {
    let id = UUID()
    var apps: [DictationApp]
    var firstRun: Bool
}

/// "Import from your other dictation apps" (SPEC "Import from other dictation
/// apps"): what was detected, what each app holds, one Import button.
struct DictationImportView: View {
    @ObservedObject var store: NativeAppStore
    let request: DictationImportRequest
    let onSettingsChanged: () -> Void
    let onDone: () -> Void

    enum Phase: Equatable {
        case offer
        case scanning
        case review([DictationAppFindings])
        case done(DictationImportSummary)
    }

    @State private var phase: Phase
    @State private var selected: Set<DictationApp> = []

    init(
        store: NativeAppStore, request: DictationImportRequest,
        onSettingsChanged: @escaping () -> Void, onDone: @escaping () -> Void
    ) {
        self.store = store
        self.request = request
        self.onSettingsChanged = onSettingsChanged
        self.onDone = onDone
        _phase = State(initialValue: request.firstRun ? .offer : .scanning)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            switch phase {
            case .offer: offer
            case .scanning: scanning
            case let .review(findings): review(findings)
            case let .done(summary): done(summary)
            }
        }
        .padding(24)
        .frame(width: 560)
        .background(EmberTheme.background)
        .foregroundStyle(EmberTheme.text)
        .tint(EmberTheme.ember)
        .preferredColorScheme(.dark)
        .task {
            if phase == .scanning { await scan() }
        }
    }

    // MARK: Phases

    @ViewBuilder private var offer: some View {
        Text("Bring your words with you").font(.title2.bold())
        Text(
            "Phonon found \(Self.list(request.apps.map(\.name))) on this Mac. It can bring over "
                + "your dictionary, snippets, past dictations, shortcut and microphone."
        )
        .foregroundStyle(.secondary)
        VStack(alignment: .leading, spacing: 10) {
            ForEach(request.apps) { app in
                HStack(spacing: 10) {
                    AppIconView(bundleID: app.bundleIDs.first, size: 22)
                    Text(app.name)
                }
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(EmberTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 10))
        privacyNote
        HStack {
            Button("Not now", action: onDone)
            Spacer()
            Button("Import from your other dictation apps") {
                phase = .scanning
                Task { await scan() }
            }
            .buttonStyle(.borderedProminent)
        }
    }

    @ViewBuilder private var scanning: some View {
        Text("Looking at your other dictation apps").font(.title2.bold())
        HStack(spacing: 10) {
            ProgressView().controlSize(.small)
            Text("Reading their dictionaries and history on this Mac…")
                .foregroundStyle(.secondary)
        }
        if request.apps.contains(.superwhisper) || !request.firstRun {
            Text("superwhisper keeps its files in Documents, so macOS may ask once.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    @ViewBuilder private func review(_ findings: [DictationAppFindings]) -> some View {
        let useful = findings.filter { !$0.isEmpty }
        Text(useful.isEmpty ? "Nothing to import" : "Found on this Mac").font(.title2.bold())
        if useful.isEmpty {
            Text(
                findings.isEmpty
                    ? "Phonon did not find Wispr Flow, Aqua Voice, superwhisper, VoiceInk, "
                        + "MacWhisper or FluidVoice on this Mac."
                    : "\(Self.list(findings.map(\.app.name))) had no words, snippets or "
                        + "dictations Phonon can use."
            )
            .foregroundStyle(.secondary)
            HStack {
                Spacer()
                Button("Done", action: onDone).buttonStyle(.borderedProminent)
            }
        } else {
            VStack(spacing: 0) {
                ForEach(Array(useful.enumerated()), id: \.element.id) { index, finding in
                    if index > 0 { Divider() }
                    FindingRow(finding: finding, isOn: binding(for: finding.app))
                        .padding(.vertical, 8)
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 6)
            .background(EmberTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 10))
            Text(
                "New words join your dictionary next to the ones you have; nothing is replaced. "
                    + "Past dictations are kept in Phonon's folder to learn your vocabulary from. "
                    + "The shortcut and microphone are set only if they differ from Phonon's."
            )
            .font(.caption).foregroundStyle(.secondary)
            privacyNote
            HStack {
                Button("Cancel", action: onDone)
                Spacer()
                Button("Import") { apply(useful) }
                    .buttonStyle(.borderedProminent)
                    .disabled(selected.isEmpty)
            }
        }
    }

    @ViewBuilder private func done(_ summary: DictationImportSummary) -> some View {
        Text(summary.isEmpty ? "Already up to date" : "Imported").font(.title2.bold())
        VStack(alignment: .leading, spacing: 6) {
            ForEach(Self.doneLines(summary), id: \.self) { line in
                Label(line, systemImage: "checkmark")
                    .foregroundStyle(EmberTheme.text)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(EmberTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 10))
        Text("Import again any time from Settings › System.")
            .font(.caption).foregroundStyle(.secondary)
        HStack {
            Spacer()
            Button("Done", action: onDone).buttonStyle(.borderedProminent)
        }
    }

    private var privacyNote: some View {
        Text("Their files are only read, never changed, and nothing leaves this Mac.")
            .font(.caption).foregroundStyle(.secondary)
    }

    // MARK: Actions

    private func scan() async {
        let apps = request.apps.isEmpty ? store.detectDictationApps() : request.apps
        let findings = await store.scanDictationApps(apps)
        selected = Set(findings.filter { !$0.isEmpty }.map(\.app))
        phase = .review(findings)
    }

    private func apply(_ findings: [DictationAppFindings]) {
        let summary = store.applyDictationImport(findings.filter { selected.contains($0.app) })
        if summary.shortcutMode != nil || summary.microphone != nil { onSettingsChanged() }
        phase = .done(summary)
    }

    private func binding(for app: DictationApp) -> Binding<Bool> {
        Binding(
            get: { selected.contains(app) },
            set: { on in
                if on { selected.insert(app) } else { selected.remove(app) }
            })
    }

    // MARK: Text

    static func list(_ names: [String]) -> String {
        switch names.count {
        case 0: return "no dictation apps"
        case 1: return names[0]
        case 2: return "\(names[0]) and \(names[1])"
        default: return names.dropLast().joined(separator: ", ") + " and " + names.last!
        }
    }

    static func doneLines(_ summary: DictationImportSummary) -> [String] {
        var lines: [String] = []
        if summary.addedTerms > 0 {
            lines.append("\(summary.addedTerms.formatted()) new dictionary entries")
        }
        if summary.addedMacros > 0 {
            lines.append("\(summary.addedMacros.formatted()) snippets added as macros")
        }
        if summary.savedDictations > 0 {
            lines.append(
                "\(summary.savedDictations.formatted()) past dictations kept for your vocabulary")
        }
        if let mode = summary.shortcutMode {
            lines.append("Shortcut set to \(DictationImportScanner.shortcutLabel(for: mode))")
        }
        if let microphone = summary.microphone {
            lines.append("\(microphone) is now your first microphone")
        }
        if lines.isEmpty { lines.append("Everything found was already in Phonon") }
        return lines
    }
}

private struct FindingRow: View {
    let finding: DictationAppFindings
    @Binding var isOn: Bool

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            AppIconView(bundleID: finding.app.bundleIDs.first, size: 22)
            VStack(alignment: .leading, spacing: 3) {
                Text(finding.app.name)
                Text(finding.summary.joined(separator: " · "))
                    .font(.caption).foregroundStyle(.secondary)
                if finding.skippedPrivate > 0 {
                    Text(
                        "\(finding.skippedPrivate.formatted()) dictations into private apps "
                            + "or sites are left out."
                    )
                    .font(.caption).foregroundStyle(EmberTheme.muted)
                }
            }
            .help(finding.readPaths.joined(separator: "\n"))
            Spacer()
            Toggle(finding.app.name, isOn: $isOn)
                .labelsHidden()
                .toggleStyle(EmberSwitchStyle())
                .accessibilityLabel("Import from \(finding.app.name)")
        }
    }
}
