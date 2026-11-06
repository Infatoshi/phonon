import PhononCore
import SwiftUI

struct HomeView: View {
    @EnvironmentObject var engine: DictationEngine
    @State private var showSettings = false
    @State private var copiedID: String?

    var body: some View {
        NavigationStack {
            List {
                Section {
                    ModelStatusRow()
                    SessionRow()
                }
                Section {
                    RecordPanel()
                        .listRowInsets(EdgeInsets(top: 20, leading: 16, bottom: 20, trailing: 16))
                }
                if !engine.history.isEmpty {
                    Section("Recent") {
                        ForEach(engine.history) { d in
                            HistoryRow(dictation: d, copied: copiedID == d.id)
                                .contentShape(Rectangle())
                                .onTapGesture {
                                    UIPasteboard.general.string = d.text
                                    copiedID = d.id
                                }
                        }
                    }
                }
            }
            .navigationTitle("Phonon")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button { showSettings = true } label: { Image(systemName: "gearshape") }
                        .accessibilityLabel("Settings")
                }
            }
            .sheet(isPresented: $showSettings) {
                NavigationStack { SettingsView() }.environmentObject(engine)
            }
        }
    }
}

struct ModelStatusRow: View {
    @EnvironmentObject var engine: DictationEngine

    var body: some View {
        switch engine.modelPhase {
        case .ready:
            Label {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Models ready")
                    if let cold = engine.coldStart, cold.asrSeconds + cold.refinerSeconds > 0 {
                        Text("Loaded in \(cold.asrSeconds + cold.refinerSeconds, specifier: "%.1f") s")
                            .font(.footnote).foregroundStyle(.secondary)
                    }
                }
            } icon: { Image(systemName: "checkmark.circle") }
        case .loading(let what):
            Label { Text(what + "…") } icon: { ProgressView() }
        case .downloading(let fraction):
            DownloadRow(fraction: fraction)
        case .missing:
            Button { engine.downloadModels() } label: {
                Label("Download models (\(ByteCountFormatter.string(fromByteCount: ModelManifest.totalBytes, countStyle: .file)))", systemImage: "arrow.down.circle")
            }
        case .failed(let message):
            VStack(alignment: .leading, spacing: 8) {
                Label(message, systemImage: "exclamationmark.triangle")
                Button("Resume download") { engine.downloadModels() }
            }
        }
    }
}

struct DownloadRow: View {
    @EnvironmentObject var engine: DictationEngine
    let fraction: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Downloading models")
                Spacer()
                Text(fraction, format: .percent.precision(.fractionLength(0)))
                    .monospacedDigit().foregroundStyle(.secondary)
            }
            ProgressView(value: fraction)
            if let p = engine.downloadProgress {
                Text("\(ByteCountFormatter.string(fromByteCount: p.bytesDone, countStyle: .file)) of \(ByteCountFormatter.string(fromByteCount: p.bytesTotal, countStyle: .file)). Keep Phonon open; it resumes where it stopped.")
                    .font(.footnote).foregroundStyle(.secondary)
            }
            Button("Pause") { engine.pauseDownload() }.font(.footnote)
        }
        .padding(.vertical, 4)
    }
}

struct SessionRow: View {
    @EnvironmentObject var engine: DictationEngine

    var body: some View {
        HStack {
            Label {
                Text(title)
            } icon: {
                Image(systemName: engine.phase == .off ? "mic.slash" : "mic")
            }
            Spacer()
            if engine.phase == .ready {
                Button("Close mic") { engine.endSession() }.font(.callout)
            }
        }
    }

    private var title: String {
        switch engine.phase {
        case .off: return "Mic closed. The keyboard opens Phonon to start."
        case .ready: return "Mic ready for the keyboard"
        case .recording: return "Recording"
        case .transcribing: return "Transcribing"
        case .refining: return "Cleaning up"
        case .failed: return engine.notice ?? "Microphone unavailable"
        }
    }
}

struct RecordPanel: View {
    @EnvironmentObject var engine: DictationEngine

    var body: some View {
        VStack(spacing: 16) {
            RecordButton()
            if let notice = engine.notice, engine.phase != .failed {
                Text(notice).font(.footnote).foregroundStyle(.secondary).multilineTextAlignment(.center)
            }
            if let latest = engine.latest {
                VStack(alignment: .leading, spacing: 8) {
                    Text(latest.text)
                        .font(.body)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    HStack {
                        Text(meta(latest)).font(.caption).foregroundStyle(.secondary)
                        Spacer()
                        Button {
                            UIPasteboard.general.string = latest.text
                        } label: { Label("Copy", systemImage: "doc.on.doc") }
                            .font(.callout)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity)
    }

    private func meta(_ d: Dictation) -> String {
        var parts = [String(format: "%.1f s audio", d.audioSeconds)]
        if d.asrSeconds > 0 { parts.append(String(format: "ASR %.1f s", d.asrSeconds)) }
        if d.refined { parts.append(String(format: "cleanup %.1f s", d.refineSeconds)) } else { parts.append("raw") }
        if d.guardFired { parts.append("guard kept raw") }
        if d.recovered { parts.append("recovered") }
        return parts.joined(separator: " · ")
    }
}

struct RecordButton: View {
    @EnvironmentObject var engine: DictationEngine

    var body: some View {
        let recording = engine.phase == .recording
        let busy = engine.phase == .transcribing || engine.phase == .refining
        VStack(spacing: 10) {
            Button {
                if recording { engine.stopRecording() } else { Task { await engine.startRecording() } }
            } label: {
                ZStack {
                    Circle()
                        .fill(recording ? Palette.recording : Palette.accent)
                        .frame(width: 76, height: 76)
                        .scaleEffect(recording ? 1 + CGFloat(engine.level) * 0.12 : 1)
                        .animation(.spring(response: 0.25, dampingFraction: 1), value: engine.level)
                    if busy {
                        ProgressView().tint(.white)
                    } else {
                        Image(systemName: recording ? "stop.fill" : "mic.fill")
                            .font(.system(size: 28, weight: .semibold))
                            .foregroundStyle(.white)
                    }
                }
            }
            .buttonStyle(PressScale())
            .disabled(busy || engine.modelPhase != .ready)
            .accessibilityLabel(recording ? "Stop dictation" : "Start dictation")
            Text(label).font(.subheadline).foregroundStyle(.secondary).monospacedDigit()
        }
    }

    private var label: String {
        switch engine.phase {
        case .recording: return "Recording. Tap to stop."
        case .transcribing: return "Transcribing…"
        case .refining: return "Cleaning up…"
        default: return engine.modelPhase == .ready ? "Tap to dictate here" : "Models not ready"
        }
    }
}

struct PressScale: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.95 : 1)
            .animation(.spring(response: 0.2, dampingFraction: 1), value: configuration.isPressed)
    }
}

struct HistoryRow: View {
    let dictation: Dictation
    let copied: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(dictation.text).lineLimit(3)
            HStack(spacing: 6) {
                Text(dictation.createdAt, style: .time)
                if dictation.recovered { Text("· recovered") }
                if !dictation.refined { Text("· raw") }
                Spacer()
                if copied { Label("Copied", systemImage: "checkmark").labelStyle(.titleAndIcon) }
            }
            .font(.caption).foregroundStyle(.secondary)
        }
        .padding(.vertical, 2)
    }
}

/// Shown when a keyboard tap opened the app: recording has started; go back to type.
struct ListeningView: View {
    @EnvironmentObject var engine: DictationEngine
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 28) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "arrow.up.left").font(.title2.weight(.semibold)).foregroundStyle(Palette.accent)
                Text("Go back with the app name at the top left. Phonon keeps listening; stop from the keyboard.")
                    .font(.headline)
                Spacer(minLength: 0)
            }
            .padding(.top, 8)
            Spacer()
            RecordButton()
            if let started = engine.recordingStartedAt {
                Text(started, style: .timer).font(.title3).monospacedDigit().foregroundStyle(.secondary)
            }
            Spacer()
            HStack(spacing: 20) {
                if engine.phase == .recording {
                    Button("Cancel", role: .destructive) { engine.cancelRecording() }
                }
                Button("Done") { dismiss() }
            }
            .font(.body)
        }
        .padding(24)
        .onChange(of: engine.phase) { old, new in
            // Close once the dictation is done; not on the way into recording.
            if (old == .transcribing || old == .refining) && (new == .ready || new == .off) { dismiss() }
        }
    }
}
