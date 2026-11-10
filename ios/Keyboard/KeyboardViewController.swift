import SwiftUI
import UIKit

/// The Phonon keyboard: a dictation toolbar above a full keyboard (Shared/KeyboardUI.swift). The mic never runs here
/// (iOS forbids it in keyboards); the toolbar signals the Phonon app, which holds a live mic session, and inserts the
/// text it returns.
final class KeyboardViewController: UIInputViewController, KeyboardHost {
    private let model = KeyboardModel()
    private var seenSeq = 0
    private var waiting = false
    private var pendingMic = false  // mic pressed, waiting for the app's pong before listening
    private var pongTimer: Timer?
    private var lastInsert: String?
    private var heightConstraint: NSLayoutConstraint?

    override func viewDidLoad() {
        super.viewDidLoad()
        model.host = self
        model.mode = WritingMode(rawValue: IPC.string(IPC.modeKey) ?? "") ?? .minimal
        seenSeq = Int(IPC.double(IPC.resultSeq))
        let host = UIHostingController(rootView: KeyboardView(model: model))
        host.view.backgroundColor = .clear
        host.view.translatesAutoresizingMaskIntoConstraints = false
        addChild(host)
        view.addSubview(host.view)
        NSLayoutConstraint.activate([
            host.view.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            host.view.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            host.view.topAnchor.constraint(equalTo: view.topAnchor),
            host.view.bottomAnchor.constraint(equalTo: view.bottomAnchor),
        ])
        host.didMove(toParent: self)
        let h = view.heightAnchor.constraint(equalToConstant: 280)
        h.priority = .defaultHigh
        h.isActive = true
        heightConstraint = h

        IPC.observe(IPC.result) { [weak self] in self?.takeResult() }
        IPC.observe(IPC.pong) { [weak self] in self?.gotPong() }
        IPC.observe(IPC.level) { [weak self] in
            guard let self, self.model.phase == .listening else { return }
            self.model.push(level: CGFloat(IPC.double(IPC.levelValue)))
        }
        IPC.observe(IPC.state) { [weak self] in
            guard let self, self.model.phase != .needsAccess else { return }
            let s = IPC.string(IPC.stateText) ?? ""
            if self.waiting, s == "Refining" || s == "Transcribing" { self.model.phase = .working }
        }
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        model.visible = true
        model.showGlobe = needsInputModeSwitchKey
        model.mode = WritingMode(rawValue: IPC.string(IPC.modeKey) ?? "") ?? model.mode
        checkSession()
    }

    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        model.visible = false  // stops the idle fluid's redraw
    }

    override func textDidChange(_ textInput: UITextInput?) {
        super.textDidChange(textInput)
        model.contextVersion += 1
    }

    // MARK: session

    private func checkSession() {
        guard hasFullAccess else {
            model.phase = .needsAccess
            model.status = "Settings > Phonon > Keyboards > Allow Full Access"
            return
        }
        IPC.set(IPC.now(), IPC.keyboardSeen)
        IPC.post(IPC.ping)
        pongTimer?.invalidate()
        pongTimer = Timer.scheduledTimer(withTimeInterval: 0.4, repeats: false) { [weak self] _ in
            guard let self, self.model.phase == .needsSession || self.model.phase == .needsAccess else { return }
            self.model.phase = .needsSession
            self.model.status = "Opens Phonon once to keep the mic ready"
        }
        if IPC.sessionAlive { gotPong() }
    }

    private func gotPong() {
        pongTimer?.invalidate()
        if pendingMic {
            pendingMic = false
            model.status = ""
            beginListening()
            return
        }
        if model.phase == .needsSession || model.phase == .needsAccess {
            model.phase = .ready
            model.status = ""
        }
    }

    // MARK: KeyboardHost

    func insert(_ text: String) { textDocumentProxy.insertText(text) }
    func deleteBackward() { textDocumentProxy.deleteBackward() }
    func nextKeyboard() { advanceToNextInputMode() }
    var contextBefore: String { textDocumentProxy.documentContextBeforeInput ?? "" }

    /// Keyboards cannot open URLs directly; the responder chain reaches the host's UIApplication.
    func openApp() {
        if !hasFullAccess {
            model.status = "Turn on Allow Full Access first"
            return
        }
        var parts = URLComponents(string: "phonon://start")
        let (host, via) = hostBundleID()
        // `via` says which lookup answered, so the app's Documents/return.jsonl shows it from a device run
        parts?.queryItems = [URLQueryItem(name: "via", value: via)]
        if let host { parts?.queryItems?.append(URLQueryItem(name: "host", value: host)) }
        guard let url = parts?.url else { return }
        var r: UIResponder? = self
        while let next = r {
            if let app = next as? UIApplication {
                app.open(url, options: [:], completionHandler: nil)
                return
            }
            r = next.next
        }
        model.status = "Open the Phonon app and tap Start"
    }

    /// The app the user is typing in, from UIKit private API: UIViewController's _hostApplicationBundleIdentifier
    /// (present in iOS 26.5 and 26.7.1 UIKitCore), then the older _hostBundleID (iOS 26.3 and earlier).
    /// Returns the id (nil when neither answers) and which selector answered, for the app's return log.
    /// On the iOS 26.7.1 device both selectors answer nil (via=none on every real dictation, 2026-11-06), so the
    /// fallback reads the host's audit token (_hostAuditToken, private) and asks Security for its code-signing
    /// identifier, which is the bundle id for App Store apps. `via` also carries a short probe trail for the log.
    private func hostBundleID() -> (String?, String) {
        var trail: [String] = []
        for name in ["_hostApplicationBundleIdentifier", "_hostBundleID"] {
            let sel = NSSelectorFromString(name)
            var c: UIViewController? = self
            for depth in 0..<4 {
                guard let vc = c else { break }
                if vc.responds(to: sel) {
                    let raw = vc.perform(sel)?.takeUnretainedValue()
                    if let id = raw as? String, id.contains("."), id != Bundle.main.bundleIdentifier {
                        return (id, "\(name)@\(depth)")
                    }
                    let shown = raw.map { "\(type(of: $0)):\(String(String(describing: $0).prefix(48)))" } ?? "nil"
                    trail.append("\(name)@\(depth)=\(shown)")
                }
                c = vc.parent
            }
        }
        let tokSel = NSSelectorFromString("_hostAuditToken")
        var c: UIViewController? = self
        for depth in 0..<4 {
            guard let vc = c else { break }
            if vc.responds(to: tokSel) {
                typealias F = @convention(c) (AnyObject, Selector) -> audit_token_t
                let tok = unsafeBitCast(vc.method(for: tokSel), to: F.self)(vc, tokSel)
                let pid = tok.val.5
                let (id, why) = Self.signingIdentifier(tok)
                if let id, id.contains("."), id != Bundle.main.bundleIdentifier {
                    return (id, "_hostAuditToken@\(depth)")
                }
                trail.append("tok@\(depth):pid=\(pid):\(why)")
                // SecTask signing id is denied in the keyboard sandbox on 26.7.1 (nosig0); try the pid instead
                let (rid, rwhy) = Self.bundleForPid(pid_t(bitPattern: pid))
                if let rid, rid.contains("."), rid != Bundle.main.bundleIdentifier {
                    return (rid, "pid@\(depth):\(rwhy)")
                }
                trail.append("pid:\(rwhy)")
                break
            }
            c = vc.parent
        }
        var chain: [String] = []
        c = self
        while let vc = c, chain.count < 4 { chain.append(String(describing: type(of: vc))); c = vc.parent }
        trail.append("chain=" + chain.joined(separator: ">"))
        return (nil, "none|" + trail.joined(separator: "|"))
    }

    /// pid -> bundle id: RunningBoard's RBSProcessHandle (private), then proc_pidpath's ".app" folder name as a last
    /// hint (that is the app's name, not its id, so it is only logged). Each step reports why it failed.
    private static func bundleForPid(_ pid: pid_t) -> (String?, String) {
        var why: [String] = []
        if let idCls = NSClassFromString("RBSProcessIdentifier") as AnyObject?,
           let hCls = NSClassFromString("RBSProcessHandle") as AnyObject? {
            let mk = NSSelectorFromString("identifierWithPid:")
            let get = NSSelectorFromString("handleForIdentifier:error:")
            if idCls.responds(to: mk), hCls.responds(to: get) {
                typealias Mk = @convention(c) (AnyObject, Selector, Int32) -> AnyObject?
                typealias Get = @convention(c) (AnyObject, Selector, AnyObject, UnsafeMutablePointer<NSError?>?) -> AnyObject?
                if let ident = unsafeBitCast(idCls.method(for: mk), to: Mk.self)(idCls, mk, pid) {
                    var err: NSError?
                    if let handle = unsafeBitCast(hCls.method(for: get), to: Get.self)(hCls, get, ident, &err) as? NSObject {
                        for path in ["bundle.identifier", "identity.embeddedApplicationIdentifier"] {
                            if let id = handle.value(forKeyPath: path) as? String, id.contains(".") { return (id, "rbs:\(path)") }
                        }
                        why.append("rbs:noid")
                    } else {
                        why.append("rbs:err\(err?.code ?? 0)")
                    }
                } else { why.append("rbs:noident") }
            } else { why.append("rbs:nosel") }
        } else { why.append("rbs:nocls") }
        if let sym = dlsym(UnsafeMutableRawPointer(bitPattern: -2), "proc_pidpath") {
            typealias P = @convention(c) (Int32, UnsafeMutableRawPointer, UInt32) -> Int32
            var buf = [UInt8](repeating: 0, count: 4096)
            let n = buf.withUnsafeMutableBytes { unsafeBitCast(sym, to: P.self)(pid, $0.baseAddress!, 4096) }
            if n > 0 {
                let path = String(decoding: buf.prefix(Int(n)), as: UTF8.self)
                let app = path.split(separator: "/").first { $0.hasSuffix(".app") }.map(String.init) ?? "?"
                why.append("path:\(app)")
            } else { why.append("path:err\(errno)") }
        } else { why.append("path:nosym") }
        return (nil, why.joined(separator: ","))
    }

    /// SecTaskCreateWithAuditToken + SecTaskCopySigningIdentifier, looked up at runtime (not in the iOS SDK headers).
    private static func signingIdentifier(_ tok: audit_token_t) -> (String?, String) {
        guard let h = dlopen("/System/Library/Frameworks/Security.framework/Security", RTLD_NOW),
              let createSym = dlsym(h, "SecTaskCreateWithAuditToken"),
              let copySym = dlsym(h, "SecTaskCopySigningIdentifier")
        else { return (nil, "nosym") }
        typealias Create = @convention(c) (CFAllocator?, audit_token_t) -> Unmanaged<CFTypeRef>?
        typealias Copy = @convention(c) (CFTypeRef, UnsafeMutablePointer<Unmanaged<CFError>?>?) -> Unmanaged<CFString>?
        guard let task = unsafeBitCast(createSym, to: Create.self)(nil, tok)?.takeRetainedValue() else {
            return (nil, "notask")
        }
        var err: Unmanaged<CFError>?
        guard let sig = unsafeBitCast(copySym, to: Copy.self)(task, &err)?.takeRetainedValue() else {
            let code = err.map { CFErrorGetCode($0.takeRetainedValue()) } ?? 0
            return (nil, "nosig\(code)")
        }
        return (sig as String, "ok")
    }

    func micDown() {
        if IPC.sessionAlive {
            beginListening()
            return
        }
        // the heartbeat can be stale even with the session up (e.g. right after a reinstall), so ask the app directly
        // over a Darwin ping before bouncing to it; only open Phonon if nobody answers within 0.4 s
        pendingMic = true
        IPC.post(IPC.ping)
        pongTimer?.invalidate()
        pongTimer = Timer.scheduledTimer(withTimeInterval: 0.4, repeats: false) { [weak self] _ in
            guard let self, self.pendingMic else { return }
            self.pendingMic = false
            self.model.phase = .needsSession
            self.openApp()
        }
    }

    private func beginListening() {
        waiting = true
        model.levels = Array(repeating: 0, count: model.levels.count)
        model.phase = .listening
        IPC.set(model.mode.rawValue, IPC.modeKey)
        IPC.post(IPC.start)
    }

    func micUp() {
        if pendingMic && !model.latched {  // a hold released before the app answered: nothing was recorded
            pendingMic = false
            pongTimer?.invalidate()
            model.phase = .ready
            return
        }
        guard waiting else { return }
        IPC.set(IPC.now(), IPC.stopAt)
        IPC.post(IPC.stop)
        model.phase = .working
    }

    func cancelDictation() {
        guard waiting else { return }
        waiting = false
        model.latched = false
        IPC.post(IPC.cancel)
        model.phase = .ready
        model.status = "Cancelled"
    }

    func undoDictation() {
        guard let last = lastInsert else { return }
        let before = contextBefore
        guard before.hasSuffix(last) else {
            lastInsert = nil
            model.canUndo = false
            return
        }
        for _ in 0..<last.count { textDocumentProxy.deleteBackward() }
        lastInsert = nil
        model.canUndo = false
        model.status = "Removed the last dictation"
    }

    private func takeResult() {
        let seq = Int(IPC.double(IPC.resultSeq))
        guard seq != seenSeq else { return }
        seenSeq = seq
        guard waiting else { return }
        waiting = false
        model.latched = false
        model.phase = IPC.sessionAlive ? .ready : .needsSession
        var text = IPC.string(IPC.resultText) ?? ""
        // A newline (numbered steps, an email's greeting) reaches the host as a Return press. Where Return
        // submits (Send, Go, Search, Done...) it would send half the dictation, so those fields get one line.
        let submits: [UIReturnKeyType] = [.go, .google, .join, .route, .search, .send, .yahoo, .done,
                                          .emergencyCall, .continue]
        if let key = textDocumentProxy.returnKeyType, submits.contains(key) {
            text = text.replacingOccurrences(of: "\n", with: " ")
        }
        guard !text.isEmpty else {
            model.status = "Didn't catch that"
            return
        }
        let before = contextBefore
        let lead = before.isEmpty || before.hasSuffix(" ") || before.hasSuffix("\n") ? "" : " "
        let insert = lead + text + " "
        textDocumentProxy.insertText(insert)
        model.contextVersion += 1
        lastInsert = insert
        model.canUndo = true
        let ms = Int((IPC.now() - IPC.double(IPC.stopAt)) * 1000)
        model.status = "\(text.split(separator: " ").count) words · \(String(format: "%.1f", Double(ms) / 1000)) s"
    }
}
