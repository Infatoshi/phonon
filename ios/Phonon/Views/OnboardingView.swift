import PhononCore
import SwiftUI

struct OnboardingView: View {
    @EnvironmentObject var engine: DictationEngine
    let done: () -> Void
    /// `-PhononOnboardingStep N` opens a given step (screenshots and checks).
    @State private var step = min(4, max(0, UserDefaults.standard.integer(forKey: "PhononOnboardingStep")))
    @State private var micAllowed = false
    @State private var tryText = ""
    @FocusState private var tryFocused: Bool

    private let steps = ["Phonon", "Microphone", "Models", "Keyboard", "Try it"]

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 6) {
                ForEach(steps.indices, id: \.self) { i in
                    Capsule().fill(i <= step ? Palette.accent : Color.secondary.opacity(0.25)).frame(height: 4)
                }
            }
            .padding(.horizontal, 24).padding(.top, 12)
            .accessibilityLabel("Step \(step + 1) of \(steps.count)")

            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    Text(steps[step]).font(.largeTitle.bold()).padding(.top, 32)
                    content
                }
                .padding(24)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            footer.padding(24)
        }
        .onAppear { micAllowed = engine.microphoneAllowed }
        .animation(.spring(response: 0.35, dampingFraction: 1), value: step)
    }

    @ViewBuilder private var content: some View {
        switch step {
        case 0:
            Para("Dictate into any app with the Phonon keyboard. Speech recognition and cleanup both run on this iPhone; your voice and text never leave it.")
            Para("The keyboard cannot use the microphone itself, so the Phonon app listens and the keyboard types the result. The first time, a tap on the keyboard's mic opens Phonon for a moment; after that the mic stays ready for a few minutes and the keyboard works without leaving your app.")
        case 1:
            Para("Phonon needs the microphone to hear you. Audio is processed on the phone and deleted once the text is saved.")
            if micAllowed {
                Label("Microphone allowed", systemImage: "checkmark.circle").foregroundStyle(.secondary)
            } else if engine.microphoneDenied {
                Para("Microphone access is off. Turn it on in Settings > Phonon > Microphone.")
                Button("Open Settings") { openSettings() }
            }
        case 2:
            Para("Two models run on the phone: Parakeet TDT 0.6B v2 for speech recognition and the Phonon refiner 1.7B to fix recognition errors, punctuation and technical terms. They download once from Hugging Face, pinned to exact versions and checked byte for byte.")
            ModelStatusRow()
            Text("\(ByteCountFormatter.string(fromByteCount: ModelManifest.totalBytes, countStyle: .file)) total. Use Wi-Fi. You can continue while it downloads.")
                .font(.footnote).foregroundStyle(.secondary)
        case 3:
            VStack(alignment: .leading, spacing: 12) {
                NumberedStep(n: 1, text: "Open Settings > Phonon > Keyboards.")
                NumberedStep(n: 2, text: "Turn on Phonon, then Allow Full Access.")
                NumberedStep(n: 3, text: "In any app, hold the globe key and pick Phonon.")
            }
            Button("Open Phonon settings") { openSettings() }
            Text("Why Full Access").font(.headline).padding(.top, 8)
            Para("iOS gives a keyboard a shared folder with its app only when Full Access is on. Phonon uses that folder to hand the finished text from the app to the keyboard. The keyboard has no network code and keeps no record of what you type.")
            if engine.store.keyboardState.updatedAt > .distantPast {
                Label("Keyboard connected", systemImage: "checkmark.circle").foregroundStyle(.secondary)
            }
        default:
            Para("Tap the field, switch to the Phonon keyboard with the globe key, then tap its mic.")
            TextField("Dictate here", text: $tryText, axis: .vertical)
                .lineLimit(3...8)
                .padding(12)
                .background(RoundedRectangle(cornerRadius: 12).fill(Color(.secondarySystemBackground)))
                .focused($tryFocused)
                .onAppear { tryFocused = UserDefaults.standard.bool(forKey: "PhononFocusTryField") }
        }
    }

    @ViewBuilder private var footer: some View {
        HStack {
            if step > 0 {
                Button("Back") { step -= 1 }
            }
            Spacer()
            switch step {
            case 1 where !micAllowed:
                Button("Allow microphone") {
                    Task {
                        micAllowed = await engine.requestMicrophone()
                        if micAllowed { step += 1 }
                    }
                }
                .buttonStyle(.borderedProminent)
            case 2 where engine.modelPhase == .missing:
                Button("Download") { engine.downloadModels() }.buttonStyle(.borderedProminent)
            case steps.count - 1:
                Button("Done") { done() }.buttonStyle(.borderedProminent)
            default:
                Button("Continue") { step += 1 }.buttonStyle(.borderedProminent)
            }
        }
        .controlSize(.large)
    }

    private func openSettings() {
        if let url = URL(string: UIApplication.openSettingsURLString) { UIApplication.shared.open(url) }
    }
}

struct Para: View {
    let text: String
    init(_ text: String) { self.text = text }
    var body: some View { Text(text).font(.body).fixedSize(horizontal: false, vertical: true) }
}

struct NumberedStep: View {
    let n: Int
    let text: String
    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Text("\(n)").font(.body.monospacedDigit().weight(.semibold)).foregroundStyle(Palette.accent)
            Text(text)
        }
    }
}
