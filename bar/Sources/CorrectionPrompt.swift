import SwiftUI

/// The instructions the correction model runs under, and the pieces Settings > Advanced >
/// Correction prompt needs to edit and try them.
///
/// The model was trained on `trained`; a different wording can change what it fixes, so the
/// editor always offers a way back. The engine reads the saved prompt from settings.json
/// (`refiner_prompt`, empty for the trained one) whenever the file changes.
enum CorrectionPrompt {
    /// Byte-identical to `REFINER_SYSTEM_PROMPT` in sidecar/polish_server.py (a test holds
    /// the two together).
    static let trained =
        "Rewrite the raw dictation transcript into the exact text the speaker intended. "
        + "Fix recognition errors and technical terms, keep the speaker's wording and casing style, "
        + "do not add or drop content."

    /// The engine and the sidecar refuse a longer prompt and use the trained one.
    static let maxLength = 4_000

    /// A dictation as a recogniser often delivers it, full of the mistakes a correction
    /// should clear: missing capitals and punctuation, homophones, doubled words, grammar,
    /// product names and numbers.
    static let sample =
        "hey john i wanted to follow up on the meeting from yesterday. we talked about the new "
        + "feature and i think its going to be great. there are a few things that we needs to fix "
        + "before the release tho. first the login page is not working on safari and the the error "
        + "messages are confusing. second we should of tested the api more, its returning a 500 "
        + "error when you send an empty request. can you ask sarah if she can take a look at it. "
        + "their going to need to push the changes to get hub by friday. also your welcome to join "
        + "the standup tomorrow at 10 am, let me know if you cant make it. thanks alot"

    /// What the saved setting is for an edited prompt: empty (the trained wording) when the
    /// text is blank or is the trained wording itself.
    static func setting(for draft: String) -> String {
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        return text == trained ? "" : text
    }

    /// The prompt that runs for a saved setting.
    static func effective(_ saved: String) -> String {
        let text = saved.trimmingCharacters(in: .whitespacesAndNewlines)
        return text.isEmpty || text.count > maxLength ? trained : text
    }

    /// The engine's `polish` command for one try: no app, so nothing is read from or typed
    /// into one, and the prompt in `prompt` instead of the saved one.
    static func testCommand(id: String, text: String, prompt: String, mode: AppWritingMode)
        -> [String: Any]
    {
        [
            "cmd": "polish", "id": id, "text": text,
            "refiner_prompt": setting(for: prompt), "writing_mode": mode.rawValue,
        ]
    }
}

/// One correction at a time for the Settings test box. The engine answers a `polish`
/// command with a `result` (kind `polish`) or an `error` carrying the id it was given.
@MainActor
final class PromptTester: ObservableObject {
    static let shared = PromptTester()
    static let idPrefix = "prompt-test-"

    enum State: Equatable {
        case idle
        case running
        case done(text: String, seconds: Double?)
        case failed(String)
    }

    @Published private(set) var state: State = .idle
    /// Sends one JSON command to the engine. Set by the app once the engine runs.
    var send: (([String: Any]) -> Void)?
    /// Whether the correction model is loaded.
    var isReady: () -> Bool = { false }

    private var counter = 0
    private var currentID: String?
    private var timeout: Task<Void, Never>?
    private let timeoutSeconds: Double

    init(timeoutSeconds: Double = 90) {
        self.timeoutSeconds = timeoutSeconds
    }

    var running: Bool { state == .running }

    func run(text: String, prompt: String, mode: AppWritingMode) {
        let input = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !input.isEmpty else {
            state = .failed("Type some text to correct first.")
            return
        }
        guard prompt.count <= CorrectionPrompt.maxLength else {
            state = .failed("The prompt is over \(CorrectionPrompt.maxLength) characters.")
            return
        }
        guard let send, isReady() else {
            state = .failed("The correction model is still loading. Try again in a moment.")
            return
        }
        counter += 1
        let id = Self.idPrefix + String(counter)
        currentID = id
        state = .running
        send(CorrectionPrompt.testCommand(id: id, text: input, prompt: prompt, mode: mode))
        timeout?.cancel()
        let seconds = timeoutSeconds
        timeout = Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            guard !Task.isCancelled else { return }
            self?.expire(id)
        }
    }

    /// An engine `result` event; true when it was this tester's.
    func handleResult(id: String?, text: String?, seconds: Double?) -> Bool {
        guard let id, id.hasPrefix(Self.idPrefix) else { return false }
        guard id == currentID else { return true }  // an answer to a try that was replaced
        finish(.done(text: text ?? "", seconds: seconds))
        return true
    }

    /// An engine `error` event; true when it was this tester's.
    func handleError(id: String?, message: String) -> Bool {
        guard let id, id.hasPrefix(Self.idPrefix) else { return false }
        guard id == currentID else { return true }
        finish(.failed(message))
        return true
    }

    func reset() {
        timeout?.cancel()
        currentID = nil
        state = .idle
    }

    private func expire(_ id: String) {
        guard currentID == id, state == .running else { return }
        finish(.failed("The correction model did not answer."))
    }

    private func finish(_ next: State) {
        timeout?.cancel()
        currentID = nil
        state = next
    }
}

/// Settings > Advanced > Correction prompt: the prompt the correction model runs under and a
/// box to try it on a sample.
struct CorrectionPromptEditor: View {
    @ObservedObject var store: NativeAppStore
    let onChanged: () -> Void
    @ObservedObject private var tester = PromptTester.shared
    @State private var draft = ""
    @State private var sample = CorrectionPrompt.sample
    @State private var mode = AppWritingMode.minimal
    @State private var loaded = false

    private var saved: String { CorrectionPrompt.effective(store.settings.refinerPrompt) }
    private var usingOwn: Bool { CorrectionPrompt.setting(for: saved).isEmpty == false }
    private var dirty: Bool {
        draft.trimmingCharacters(in: .whitespacesAndNewlines) != saved
    }
    private var tooLong: Bool { draft.count > CorrectionPrompt.maxLength }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(
                "The instructions the correction model follows, for the live text and the final "
                    + "text. The model was trained on the default wording, so a rewrite can fix "
                    + "more or less than you expect: try it below before you rely on it. "
                    + "A saved prompt applies from the next dictation."
            )
            .font(.system(size: 12))
            .foregroundStyle(EmberTheme.mute)
            .fixedSize(horizontal: false, vertical: true)

            HStack {
                Text(usingOwn ? "Using your prompt" : "Using the trained default")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(usingOwn ? EmberTheme.ember2 : EmberTheme.mute)
                Spacer()
                Text("\(draft.count) / \(CorrectionPrompt.maxLength)")
                    .font(.mono(11))
                    .foregroundStyle(tooLong ? EmberTheme.warm : EmberTheme.dim)
            }
            PromptTextBox(text: $draft, minHeight: 110)

            HStack(spacing: 8) {
                Button("Save") {
                    store.updateSettings {
                        $0.refinerPrompt = CorrectionPrompt.setting(for: draft)
                    }
                    draft = CorrectionPrompt.effective(store.settings.refinerPrompt)
                    onChanged()
                }
                .buttonStyle(PillButtonStyle(primary: true))
                .disabled(!dirty || tooLong || draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                Button("Discard edits") { draft = saved }
                    .buttonStyle(PillButtonStyle())
                    .disabled(!dirty)
                Button("Use the default") {
                    draft = CorrectionPrompt.trained
                    store.updateSettings { $0.refinerPrompt = "" }
                    onChanged()
                }
                .buttonStyle(PillButtonStyle())
                .disabled(!usingOwn && draft == CorrectionPrompt.trained)
            }

            Rectangle().fill(EmberTheme.line).frame(height: 1).padding(.vertical, 2)
            Text("Try it")
            Text(
                "Runs the correction model on the text below with the prompt above, saved or not. "
                    + "Nothing is typed into any app."
            )
            .font(.system(size: 12))
            .foregroundStyle(EmberTheme.mute)
            .fixedSize(horizontal: false, vertical: true)

            HStack(spacing: 10) {
                Text("Writing mode").font(.system(size: 12)).foregroundStyle(EmberTheme.mute)
                ChoiceChips(
                    options: AppWritingMode.allCases.map { ($0, $0.title) }, selection: mode
                ) { mode = $0 }
            }
            PromptTextBox(text: $sample, minHeight: 130, monospaced: false)
            HStack(spacing: 8) {
                Button("Run test") {
                    tester.run(text: sample, prompt: draft, mode: mode)
                }
                .buttonStyle(PillButtonStyle(primary: true))
                .disabled(tester.running || tooLong)
                Button("Restore the sample") { sample = CorrectionPrompt.sample }
                    .buttonStyle(PillButtonStyle())
                    .disabled(sample == CorrectionPrompt.sample)
                if tester.running {
                    ProgressView().controlSize(.small)
                }
            }
            result
        }
        .onAppear {
            guard !loaded else { return }
            loaded = true
            draft = saved
        }
    }

    @ViewBuilder private var result: some View {
        switch tester.state {
        case .idle, .running:
            EmptyView()
        case .failed(let message):
            Text(message)
                .font(.system(size: 12))
                .foregroundStyle(EmberTheme.warm)
                .fixedSize(horizontal: false, vertical: true)
        case .done(let text, let seconds):
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text(summary(text: text, seconds: seconds))
                        .font(.system(size: 12))
                        .foregroundStyle(EmberTheme.mute)
                    Spacer()
                    CopyIconButton(text: text)
                }
                Text(text.isEmpty ? "(no text came back)" : text)
                    .font(.system(size: 13))
                    .foregroundStyle(EmberTheme.ink)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(10)
                    .background(
                        EmberTheme.panel, in: RoundedRectangle(cornerRadius: 8, style: .continuous)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .stroke(EmberTheme.panelBorder, lineWidth: 1))
            }
        }
    }

    private func summary(text: String, seconds: Double?) -> String {
        let before = sample.split(whereSeparator: \.isWhitespace).count
        let after = text.split(whereSeparator: \.isWhitespace).count
        let same = text.trimmingCharacters(in: .whitespacesAndNewlines)
            == sample.trimmingCharacters(in: .whitespacesAndNewlines)
        var parts = [same ? "Nothing changed" : "Corrected text", "\(before) → \(after) words"]
        if let seconds { parts.append(String(format: "%.1f s", seconds)) }
        return parts.joined(separator: " · ")
    }
}

/// A multi-line field in the settings' dark style.
private struct PromptTextBox: View {
    @Binding var text: String
    var minHeight: CGFloat
    var monospaced = true

    var body: some View {
        TextEditor(text: $text)
            .font(monospaced ? .mono(12) : .system(size: 13))
            .foregroundStyle(EmberTheme.ink)
            .scrollContentBackground(.hidden)
            .padding(6)
            .frame(minHeight: minHeight)
            .background(
                EmberTheme.panel, in: RoundedRectangle(cornerRadius: 8, style: .continuous)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(EmberTheme.panelBorder, lineWidth: 1))
    }
}
