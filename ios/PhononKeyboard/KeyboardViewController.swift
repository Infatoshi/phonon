import PhononCore
import SwiftUI
import UIKit

/// The Phonon keyboard: dictation controls plus the few keys a dictation keyboard needs
/// (globe, space, delete, return). It never touches the microphone; the app does.
final class KeyboardViewController: UIInputViewController {
    private let model = KeyboardModel()
    private var host: UIHostingController<KeyboardView>?

    override func viewDidLoad() {
        super.viewDidLoad()
        model.controller = self
        let view = KeyboardView(model: model)
        let host = UIHostingController(rootView: view)
        host.view.backgroundColor = .clear
        host.view.translatesAutoresizingMaskIntoConstraints = false
        addChild(host)
        self.view.addSubview(host.view)
        NSLayoutConstraint.activate([
            host.view.leadingAnchor.constraint(equalTo: self.view.leadingAnchor),
            host.view.trailingAnchor.constraint(equalTo: self.view.trailingAnchor),
            host.view.topAnchor.constraint(equalTo: self.view.topAnchor),
            host.view.bottomAnchor.constraint(equalTo: self.view.bottomAnchor),
            host.view.heightAnchor.constraint(equalToConstant: 236),
        ])
        host.didMove(toParent: self)
        self.host = host
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        model.showsGlobe = needsInputModeSwitchKey
        model.attach(fullAccess: hasFullAccess)
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        model.detach()
    }

    // MARK: Actions the model calls

    func insert(_ text: String) {
        let proxy = textDocumentProxy
        var out = text
        // Separate from the word before the cursor, as typing would.
        if let before = proxy.documentContextBeforeInput, let last = before.last, !last.isWhitespace,
           let first = out.first, !first.isPunctuation {
            out = " " + out
        }
        proxy.insertText(out)
    }

    func deleteBackward() { textDocumentProxy.deleteBackward() }
    func insertSpace() { textDocumentProxy.insertText(" ") }
    func insertReturn() { textDocumentProxy.insertText("\n") }
    func nextKeyboard() { advanceToNextInputMode() }

    var returnKeyTitle: String {
        switch textDocumentProxy.returnKeyType {
        case .send: return "send"
        case .search, .google, .yahoo: return "search"
        case .go: return "go"
        case .done: return "done"
        case .next: return "next"
        default: return "return"
        }
    }

    /// Host field hint for the automatic writing mode.
    var hostHint: String {
        automaticMode(keyboardType: textDocumentProxy.keyboardType.map { $0.rawValue },
                      textContentType: textDocumentProxy.textContentType??.rawValue).rawValue
    }

    /// Opens Phonon. A keyboard has no `UIApplication.shared`, so walk the responder chain to
    /// the application object and send it `openURL:options:completionHandler:`.
    func openApp(_ url: URL) {
        let selector = NSSelectorFromString("openURL:options:completionHandler:")
        var responder: UIResponder? = self
        while let r = responder {
            if let app = r as? UIApplication, app.responds(to: selector) {
                typealias OpenURL = @convention(c) (NSObject, Selector, NSURL, NSDictionary, AnyObject?) -> Void
                let fn = unsafeBitCast(app.method(for: selector), to: OpenURL.self)
                fn(app, selector, url as NSURL, NSDictionary(), nil)
                return
            }
            responder = r.next
        }
    }
}

@MainActor
final class KeyboardModel: ObservableObject {
    @Published var phase: SessionPhase = .off
    @Published var fullAccess = true
    @Published var startedAt: Date?
    @Published var level: Float = 0
    @Published var status = "Tap to dictate"
    @Published var lastText: String?
    @Published var showsGlobe = true
    @Published var waiting = false

    weak var controller: KeyboardViewController?
    private var store: SharedStore?
    private var seq = 0
    /// Only dictations finished after the keyboard asked for them are inserted automatically.
    private var awaitingSince: Date? {
        get { store?.keyboardState.awaitingSince }
        set {
            guard let store else { return }
            var k = store.keyboardState
            k.awaitingSince = newValue
            store.writeKeyboardState(k)
        }
    }
    private var ackTimer: Task<Void, Never>?

    func attach(fullAccess: Bool) {
        self.fullAccess = fullAccess
        guard fullAccess, let store = SharedStore.appGroup() else {
            status = "Allow Full Access to dictate"
            return
        }
        self.store = store
        seq = max(store.command?.seq ?? 0, store.appState.ackSeq)
        var k = store.keyboardState
        k.updatedAt = Date()
        store.writeKeyboardState(k)
        DarwinNote.observe(PhononIDs.stateNote) { [weak self] in
            Task { @MainActor in self?.refresh() }
        }
        refresh()
    }

    func detach() {
        DarwinNote.stopObserving(PhononIDs.stateNote)
    }

    func refresh() {
        guard let store else { return }
        let state = store.appState
        // A state older than a few seconds while "recording" means the app died mid-way.
        phase = state.phase
        startedAt = state.recordingStartedAt
        level = state.level
        lastText = store.history.first?.text
        if state.ackSeq >= seq { waiting = false }
        status = statusText(state)

        if let latest = state.latest, let since = awaitingSince, latest.createdAt >= since,
           store.settings.autoInsert, !store.keyboardState.inserted.contains(latest.id) {
            controller?.insert(latest.text)
            var k = store.keyboardState
            k.markInserted(latest.id)
            k.awaitingSince = nil
            store.writeKeyboardState(k)
            status = latest.refined ? "Inserted" : "Inserted raw transcript"
        }
    }

    private func statusText(_ s: AppState) -> String {
        switch s.phase {
        case .recording: return "Listening"
        case .transcribing: return "Transcribing…"
        case .refining: return "Cleaning up…"
        case .failed: return s.message ?? "Open Phonon to fix the microphone"
        case .ready, .off:
            if !s.modelsReady && s.phase == .ready { return "Models loading in Phonon" }
            if let m = s.message, awaitingSince != nil { return m }
            return "Tap to dictate"
        }
    }

    func micTapped() {
        guard fullAccess, let store else {
            controller?.openApp(URL(string: "phonon://setup")!)
            return
        }
        if phase == .recording {
            send(.stop)
            return
        }
        guard phase != .transcribing, phase != .refining else { return }
        awaitingSince = Date()
        send(.start)
        // If the app does not acknowledge quickly it is not running with an open mic:
        // open it, and it starts recording from the same command.
        let mySeq = seq
        ackTimer?.cancel()
        ackTimer = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 700_000_000)
            guard !Task.isCancelled, store.appState.ackSeq < mySeq else { return }
            status = "Opening Phonon…"
            controller?.openApp(URL(string: "phonon://dictate?seq=\(mySeq)")!)
        }
    }

    func cancelTapped() {
        awaitingSince = nil
        send(.cancel)
    }

    func insertLast() {
        guard let store, let last = store.history.first else { return }
        controller?.insert(last.text)
        var k = store.keyboardState
        k.markInserted(last.id)
        store.writeKeyboardState(k)
        status = "Inserted last dictation"
    }

    private func send(_ kind: KeyboardCommand.Kind) {
        guard let store else { return }
        seq += 1
        waiting = true
        store.writeCommand(KeyboardCommand(seq: seq, kind: kind, hostHint: controller?.hostHint))
        DarwinNote.post(PhononIDs.commandNote)
    }
}
