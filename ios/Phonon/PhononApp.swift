import PhononCore
import SwiftUI

@main
struct PhononApp: App {
    @StateObject private var engine = DictationEngine()
    @AppStorage("onboarded") private var onboarded = false
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            Group {
                if onboarded {
                    HomeView()
                } else {
                    OnboardingView(done: { onboarded = true })
                }
            }
            .environmentObject(engine)
            .tint(Palette.accent)
            .fullScreenCover(isPresented: $engine.openedFromKeyboard) {
                ListeningView().environmentObject(engine)
            }
            .onOpenURL { url in handle(url) }
            .task { engine.bootstrap() }
        }
        .onChange(of: scenePhase) { _, phase in
            if phase == .active { engine.bootstrap() }
        }
    }

    /// `phonon://dictate`: the keyboard was tapped while no microphone session was open.
    private func handle(_ url: URL) {
        guard url.scheme == PhononIDs.urlScheme, url.host == "dictate" else { return }
        engine.openedFromKeyboard = true
        // The tap's command is in the App Group; acting on it starts the recording.
        engine.handleKeyboardCommand()
    }
}

enum Palette {
    /// Okabe-Ito blue, lighter sky blue in dark mode: one accent, distinct in lightness too.
    static let accent = Color(UIColor { $0.userInterfaceStyle == .dark
        ? UIColor(red: 0.34, green: 0.71, blue: 0.91, alpha: 1)
        : UIColor(red: 0.0, green: 0.45, blue: 0.70, alpha: 1) })
    /// Okabe-Ito vermillion for recording; always paired with a stop square and a label.
    static let recording = Color(UIColor { $0.userInterfaceStyle == .dark
        ? UIColor(red: 0.95, green: 0.45, blue: 0.20, alpha: 1)
        : UIColor(red: 0.84, green: 0.37, blue: 0.0, alpha: 1) })
}
