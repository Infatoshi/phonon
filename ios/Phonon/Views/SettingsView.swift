import BackgroundTasks
import PhononCore
import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var engine: DictationEngine
    @Environment(\.dismiss) private var dismiss
    @State private var confirmDelete = false
    @State private var newTerm = ""

    var body: some View {
        Form {
            Section {
                Picker("Writing style", selection: $engine.settings.writingMode) {
                    Text("Automatic, casual in chats").tag(WritingMode?.none)
                    ForEach(WritingMode.allCases, id: \.self) { mode in
                        Text("\(mode.label), \(mode.detail.lowercased())").tag(WritingMode?.some(mode))
                    }
                }
                Toggle("Clean up with the refiner", isOn: $engine.settings.refinerEnabled)
                Toggle("Insert automatically", isOn: $engine.settings.autoInsert)
            } footer: {
                Text("With cleanup off, the keyboard inserts the raw Parakeet transcript.")
            }

            Section {
                Picker("Keep mic ready", selection: $engine.settings.warmMinutes) {
                    Text("1 minute").tag(1)
                    Text("5 minutes").tag(5)
                    Text("15 minutes").tag(15)
                    Text("1 hour").tag(60)
                }
            } footer: {
                Text("While the mic is ready, the keyboard dictates without opening Phonon. iOS shows the orange mic dot during that time; nothing is kept until you tap the mic.")
            }

            Section {
                ForEach(engine.settings.vocabulary, id: \.self) { Text($0) }
                    .onDelete { engine.settings.vocabulary.remove(atOffsets: $0) }
                HStack {
                    TextField("Add a term, e.g. vLLM", text: $newTerm)
                        .textInputAutocapitalization(.never).autocorrectionDisabled()
                        .onSubmit(addTerm)
                    Button("Add", action: addTerm).disabled(newTerm.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            } header: {
                Text("Vocabulary")
            } footer: {
                Text("Names and terms the refiner should spell your way. Most important first.")
            }

            Section("Models") {
                ModelStatusRow()
                if engine.modelsDownloaded {
                    Button("Delete models", role: .destructive) { confirmDelete = true }
                }
            }

            Section("Diagnostics") {
                NavigationLink("Self-test and device limits") { DiagnosticsView() }
            }

            Section {
                NavigationLink("Licenses") { LicensesView() }
                Button("Clear recent dictations", role: .destructive) { engine.clearHistory() }
            } footer: {
                Text("Phonon has no analytics and no account. Its only network use is the one-time model download from huggingface.co.")
            }
        }
        .navigationTitle("Settings")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        .confirmationDialog("Delete both models? Dictation stops working until you download them again.",
                            isPresented: $confirmDelete, titleVisibility: .visible) {
            Button("Delete models", role: .destructive) { engine.deleteModels() }
        }
    }

    private func addTerm() {
        let term = newTerm.trimmingCharacters(in: .whitespaces)
        guard !term.isEmpty, !engine.settings.vocabulary.contains(term) else { return }
        engine.settings.vocabulary.append(term)
        newTerm = ""
    }
}

struct DiagnosticsView: View {
    @EnvironmentObject var engine: DictationEngine
    @State private var rows: [DictationEngine.SelfTestRow] = []
    @State private var running = false
    @State private var reportPath: String?

    var body: some View {
        List {
            Section {
                Button(running ? "Running…" : "Run self-test") { run() }
                    .disabled(running || engine.modelPhase != .ready)
            } footer: {
                Text("Runs the bundled speech samples through both models on this phone and saves a report to Files > On My iPhone > Phonon.")
            }
            Section("This device") {
                LabeledContent("Memory available now", value: ByteCountFormatter.string(fromByteCount: Int64(os_proc_available_memory()), countStyle: .memory))
                LabeledContent("Refiner threads", value: "\(engine.refinerThreads)")
                LabeledContent("Background GPU for long tasks", value: backgroundGPU)
                if let cold = engine.coldStart {
                    LabeledContent("Cold start, speech", value: String(format: "%.1f s", cold.asrSeconds))
                    LabeledContent("Cold start, cleanup", value: String(format: "%.1f s", cold.refinerSeconds))
                }
            }
            ForEach(rows) { r in
                Section(r.name) {
                    Text(r.refined)
                    Text("Raw: \(r.raw)").font(.footnote).foregroundStyle(.secondary)
                    Text("Expected: \(r.expected)").font(.footnote).foregroundStyle(.secondary)
                    Text(String(format: "%.1f s audio · ASR %.2f s · cleanup %.2f s · prefill %.0f tok/s · decode %.1f tok/s",
                                r.audioSeconds, r.asrSeconds, r.refineSeconds, r.prefillTokPerSec, r.decodeTokPerSec))
                        .font(.caption).monospacedDigit().foregroundStyle(.secondary)
                }
            }
            if let reportPath {
                Text("Saved \(reportPath)").font(.caption).foregroundStyle(.secondary)
            }
        }
        .navigationTitle("Diagnostics")
    }

    private var backgroundGPU: String {
        if #available(iOS 26.0, *) {
            return BGTaskScheduler.supportedResources.contains(.gpu) ? "Supported" : "Not on this device"
        }
        return "Needs iOS 26"
    }

    private func run() {
        running = true
        Task {
            rows = await engine.runSelfTest()
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let url = docs.appendingPathComponent("selftest-\(Int(Date().timeIntervalSince1970)).json")
            let report = SelfTestReport(device: UIDevice.current.model, system: UIDevice.current.systemVersion,
                                        backgroundGPU: backgroundGPU, coldStart: engine.coldStart, rows: rows)
            let enc = JSONEncoder()
            enc.outputFormatting = [.prettyPrinted, .sortedKeys]
            if let data = try? enc.encode(report) {
                try? data.write(to: url)
                reportPath = url.lastPathComponent
            }
            print("PHONON_SELFTEST " + (String(data: (try? JSONEncoder().encode(report)) ?? Data(), encoding: .utf8) ?? ""))
            running = false
        }
    }
}

struct SelfTestReport: Codable {
    var device: String
    var system: String
    var backgroundGPU: String
    var coldStart: DictationEngine.ColdStart?
    var rows: [DictationEngine.SelfTestRow]
}

struct LicensesView: View {
    var body: some View {
        ScrollView {
            Text(licenses).font(.footnote).padding()
                .frame(maxWidth: .infinity, alignment: .leading)
                .textSelection(.enabled)
        }
        .navigationTitle("Licenses")
    }

    private var licenses: String {
        guard let url = Bundle.main.url(forResource: "LICENSES", withExtension: "md"),
              let text = try? String(contentsOf: url, encoding: .utf8) else { return "LICENSES.md missing from the bundle." }
        return text
    }
}
