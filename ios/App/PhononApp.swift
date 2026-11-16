import SwiftUI
import UIKit

@main
struct PhononApp: App {
    @UIApplicationDelegateAdaptor(SourceAppDelegate.self) private var appDelegate
    @StateObject private var session = Session()
    private let args = ProcessInfo.processInfo.arguments

    init() { BackToCaller.observe() }

    var body: some Scene {
        WindowGroup {
            Group {
                if args.contains("--preview-keyboard") {
                    KeyboardPreview(listening: args.contains("--listening"), needsSession: args.contains("--off"),
                                    working: args.contains("--working"))
                } else {
                    HomeView(session: session)
                }
            }
            .preferredColorScheme(.dark)
            .tint(Ember.ember)
            .task {
                if args.contains("--selftest") { session.selftest() }
                if args.contains("--probe-return") {
                    BackToCaller.probe()
                    if args.contains("--probe-open") {  // open Settings by bundle id, as the return does for a host
                        try? await Task.sleep(for: .seconds(2))
                        let r = BackToCaller.openBundleForProbe("com.apple.Preferences")
                        BackToCaller.logProbe(["event": "probe_open", "ok": r,
                                               "state": UIApplication.shared.applicationState.rawValue])
                        try? await Task.sleep(for: .seconds(1.5))
                        BackToCaller.logProbe(["event": "probe_open_after", "state": UIApplication.shared.applicationState.rawValue])
                    }
                }
            }
            .onOpenURL { url in
                // the keyboard opens phonon://start when the session is not running: start the mic, then go straight
                // back to the app the user was typing in
                guard url.host == "start" else { return }
                // capture the breadcrumb's action now: it belongs to this activation and can be gone after the
                // async mic start
                let hint = HostLookup.parse(url)
                let target = BackToCaller.capture(hint: hint)
                Task {
                    await session.start()
                    guard session.live else { return }  // still down (a call holds the mic): stay here
                    IPC.post(IPC.pong)  // tell a keyboard that is already running: the mic is up, dictate now
                    session.backHint = !(await BackToCaller.go(target))
                }
            }
        }
    }
}

/// Scene callbacks that carry the opener of a URL (UIOpenURLContext.options.sourceApplication), which SwiftUI's
/// onOpenURL does not show. When the keyboard's open is attributed to the host app (not to the keyboard), that id is
/// the app to return to. Logged to return.jsonl as `source` either way, so a device run says whether it answers.
final class SourceAppDelegate: NSObject, UIApplicationDelegate {
    func application(_ application: UIApplication, configurationForConnecting connectingSceneSession: UISceneSession,
                     options: UIScene.ConnectionOptions) -> UISceneConfiguration {
        let config = UISceneConfiguration(name: nil, sessionRole: connectingSceneSession.role)
        config.delegateClass = SourceSceneDelegate.self
        return config
    }
}

final class SourceSceneDelegate: NSObject, UIWindowSceneDelegate {
    func scene(_ scene: UIScene, willConnectTo session: UISceneSession, options: UIScene.ConnectionOptions) {
        Self.note(options.urlContexts, from: "connect")
    }

    func scene(_ scene: UIScene, openURLContexts URLContexts: Set<UIOpenURLContext>) {
        Self.note(URLContexts, from: "open")
    }

    private static func note(_ contexts: Set<UIOpenURLContext>, from: String) {
        for c in contexts where c.url.host == "start" {
            BackToCaller.recordSource(c.options.sourceApplication, from: from)
        }
    }
}

/// When another app opens Phonon, iOS shows a "◀ App" breadcrumb in the status bar. Public API has no way to follow
/// it, so this sends the breadcrumb's own action (UIKit private: _systemNavigationAction, iOS 14+), checked at runtime.
/// Returns false when it is unavailable; the caller then shows a hint to tap the breadcrumb.
/// App Store review risk: private UIKit selector. Fine for TestFlight/dev builds; revisit before a store release.
/// Fallback when the breadcrumb action is refused: reopen the host by bundle id (LSApplicationWorkspace, private).
/// Every step is logged to Documents/return.jsonl so a device test shows which path worked.
enum BackToCaller {
    struct Target {
        var bundle: String?
        var action: NSObject?
        var destination: UInt = 0
    }

    private static let own = Bundle.main.bundleIdentifier ?? "sh.phonon.ios"

    private static func valid(_ id: String?) -> String? {
        guard let id, id.contains("."), id.count <= 255, id != own, !id.hasPrefix(own + ".") else { return nil }
        return id
    }

    /// The breadcrumb action as UIKit announced it (UIApplicationSystemNavigationActionChangedNotification, private):
    /// it can arrive after onOpenURL and be cleared again by the time the app polls for it.
    @MainActor private static var announced: NSObject?

    @MainActor private static func current() -> NSObject? {
        let app = UIApplication.shared
        let key = NSSelectorFromString("_systemNavigationAction")
        guard app.responds(to: key) else { return nil }
        return app.perform(key)?.takeUnretainedValue() as? NSObject
    }

    /// The sourceApplication of the newest phonon://start, with when it arrived.
    nonisolated(unsafe) private static var source: (id: String?, at: Double)?

    static func recordSource(_ id: String?, from: String) {
        source = (id, Date().timeIntervalSince1970)
        log(["event": "source", "from": from, "source": id ?? "nil"])
    }

    /// Call once at launch.
    @MainActor static func observe() {
        NotificationCenter.default.addObserver(
            forName: Notification.Name("UIApplicationSystemNavigationActionChangedNotification"), object: nil, queue: .main
        ) { note in
            MainActor.assumeIsolated {
                if let action = current() { announced = action }
                log(["event": "announced", "action": announced.map { String(describing: type(of: $0)) } ?? "nil",
                     "info": String(describing: note.userInfo ?? [:])])
            }
        }
    }

    /// Call as soon as the URL arrives. `host` is the keyboard's guess at the app it was typing in, if it had one;
    /// `via` is which keyboard lookup produced it.
    @MainActor static func capture(hint: HostLookup.Hint) -> Target {
        var t = capture(host: hint.host, via: hint.via)
        if t.bundle == nil, let src = source, Date().timeIntervalSince1970 - src.at < 5, let id = valid(src.id) {
            t.bundle = id
            log(["event": "source_used", "bundle": id])
        }
        guard t.bundle == nil, let pid = hint.pid else { return t }
        // the keyboard could not name the host (iOS 26.7 hides it): resolve its pid here, then its process name
        let (rbs, why) = HostLookup.bundleForPid(pid)
        let pname = hint.processName ?? HostLookup.processName(pid: pid)
        let known = HostLookup.bundle(forProcessName: pname)
        t.bundle = valid(rbs) ?? valid(known)
        log(["event": "resolve", "pid": Int(pid), "rbs": rbs ?? "", "why": why, "pname": pname ?? "",
             "pname_from_keyboard": hint.processName != nil, "table": known ?? "", "bundle": t.bundle ?? ""])
        return t
    }

    @MainActor static func capture(host: String?, via: String? = nil) -> Target {
        var t = Target(bundle: valid(host))
        let app = UIApplication.shared
        var dests: [UInt] = []
        var names: [String] = []
        let live = current()
        let found = live ?? announced
        var nav = live != nil ? "live" : (announced != nil ? "announced" : "nil")
        if let found, found.responds(to: NSSelectorFromString("destinations")) {
            nav += " destinations=\(String(describing: found.value(forKey: "destinations")))"
        }
        if let action = found,
           action.responds(to: NSSelectorFromString("sendResponseForDestination:")),
           let list = action.value(forKey: "destinations") as? [NSNumber] {
            dests = list.map(\.uintValue)
            let bundleSel = NSSelectorFromString("bundleIdForDestination:")
            for d in dests {
                var name: String?
                if action.responds(to: bundleSel) {
                    typealias F = @convention(c) (AnyObject, Selector, UInt) -> AnyObject?
                    name = unsafeBitCast(action.method(for: bundleSel), to: F.self)(action, bundleSel, d) as? String
                }
                names.append(name ?? "?")
                // prefer the destination whose app matches the keyboard's host; otherwise the only one there is
                if t.action == nil, valid(name) != nil, t.bundle == nil || t.bundle == name || dests.count == 1 {
                    t.action = action
                    t.destination = d
                    t.bundle = t.bundle ?? name
                }
            }
            if t.action == nil, dests.count == 1 { t.action = action; t.destination = dests[0] }
        }
        log(["event": "capture", "host": host ?? "", "via": via ?? "", "destinations": dests.map { Int($0) }, "bundles": names,
             "bundle": t.bundle ?? "", "action": t.action != nil, "nav": nav, "state": app.applicationState.rawValue])
        return t
    }

    /// Tries the breadcrumb action, then the bundle id, up to four times while Phonon is still in front.
    /// Returns true once Phonon has actually left the foreground.
    @MainActor static func go(_ target: Target) async -> Bool {
        let app = UIApplication.shared
        var t = target
        for (i, delay) in [0.15, 0.35, 0.8, 1.5].enumerated() {
            try? await Task.sleep(for: .seconds(delay))
            if app.applicationState == .background { log(["event": "left", "attempt": i]); return true }
            guard app.applicationState == .active else { continue }
            // the breadcrumb's action can be filled in only after the activation settles; look again
            if t.action == nil {
                let again = capture(host: t.bundle)
                if again.action != nil || t.bundle == nil { t = again }
            }
            var via = "none"
            var accepted = false
            if let action = t.action {
                let send = NSSelectorFromString("sendResponseForDestination:")
                typealias F = @convention(c) (AnyObject, Selector, UInt) -> Bool
                accepted = unsafeBitCast(action.method(for: send), to: F.self)(action, send, t.destination)
                via = "breadcrumb"
            }
            if !accepted, let bundle = t.bundle { accepted = openBundle(bundle); via = "workspace" }
            log(["event": "attempt", "attempt": i, "via": via, "accepted": accepted, "state": app.applicationState.rawValue])
        }
        try? await Task.sleep(for: .seconds(0.5))
        // inactive is not gone: a permission alert over Phonon also makes it inactive
        let left = app.applicationState == .background
        log(["event": left ? "left" : "stayed", "state": app.applicationState.rawValue])
        return left
    }

    /// Launched with --probe-return: what this iOS lets the app do toward returning to another app, written to
    /// Documents/return.jsonl. Side-effect free: the bundle-id open is aimed at Phonon itself.
    @MainActor static func probe() {
        let pid = getpid()
        let (own, ownWhy) = HostLookup.bundleForPid(pid)
        let (init1, init1Why) = HostLookup.bundleForPid(1)
        let ws = NSClassFromString("LSApplicationWorkspace") as AnyObject?
        log(["event": "probe", "ios": UIDevice.current.systemVersion,
             "workspace_class": ws != nil,
             "workspace_open_sel": ws?.responds(to: NSSelectorFromString("defaultWorkspace")) ?? false,
             "state": UIApplication.shared.applicationState.rawValue,
             "own_name": HostLookup.processName(pid: pid) ?? "nil", "pid1_name": HostLookup.processName(pid: 1) ?? "nil",
             "own_rbs": own ?? "nil", "own_rbs_why": ownWhy, "pid1_rbs": init1 ?? "nil", "pid1_rbs_why": init1Why,
             "nav_action": current() != nil])
        // --probe-pid N: can the app name another process by its pid (what the start link carries)?
        let a = ProcessInfo.processInfo.arguments
        if let i = a.firstIndex(of: "--probe-pid"), i + 1 < a.count, let other = Int32(a[i + 1]) {
            let (id, why) = HostLookup.bundleForPid(other)
            log(["event": "probe_pid", "pid": Int(other), "name": HostLookup.processName(pid: other) ?? "nil",
                 "rbs": id ?? "nil", "why": why])
        }
    }

    @MainActor static func openBundleForProbe(_ id: String) -> Bool { openBundle(id) }
    static func logProbe(_ row: [String: Any]) { log(row) }

    private static func openBundle(_ bundle: String) -> Bool {
        guard let cls = NSClassFromString("LSApplicationWorkspace") as AnyObject?,
              cls.responds(to: NSSelectorFromString("defaultWorkspace")),
              let ws = cls.perform(NSSelectorFromString("defaultWorkspace"))?.takeUnretainedValue() as? NSObject
        else { return false }
        let open = NSSelectorFromString("openApplicationWithBundleID:")
        guard ws.responds(to: open) else { return false }
        typealias F = @convention(c) (AnyObject, Selector, NSString) -> Bool
        return unsafeBitCast(ws.method(for: open), to: F.self)(ws, open, bundle as NSString)
    }

    private static func log(_ row: [String: Any]) {
        var row = row
        row["t"] = Date().timeIntervalSince1970
        guard let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first,
              let data = try? JSONSerialization.data(withJSONObject: row) else { return }
        let url = dir.appendingPathComponent("return.jsonl")
        if !FileManager.default.fileExists(atPath: url.path) { FileManager.default.createFile(atPath: url.path, contents: nil) }
        guard let h = try? FileHandle(forWritingTo: url) else { return }
        h.seekToEndOfFile()
        h.write(data + Data("\n".utf8))
        try? h.close()
    }
}

// MARK: home

struct HomeView: View {
    @ObservedObject var session: Session
    @Environment(\.scenePhase) private var phase
    @State private var keyboardAdded = false
    @State private var fullAccess = false
    @State private var copied: UUID?
    @State private var showDictionary = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                sessionCard
                if !(keyboardAdded && fullAccess && session.parakeetReady) { setupCard }
                modeCard
                dictionaryCard
                historyCard
                modelsFooter
            }
            .padding(.horizontal, 18)
            .padding(.top, 8)
            .padding(.bottom, 32)
        }
        .scrollIndicators(.hidden)
        .background(Ember.window.ignoresSafeArea())
        .foregroundStyle(Ember.ink)
        .onAppear(perform: refreshSetup)
        .onChange(of: phase) { _, p in if p == .active { refreshSetup() } }
        .sheet(isPresented: $showDictionary) {
            DictionaryView(entries: session.dictionary.entries,
                           onAdd: { session.addToDictionary(phrase: $0, replacement: $1) },
                           onRemove: { session.removeFromDictionary($0) },
                           onClose: { showDictionary = false })
                .preferredColorScheme(.dark)
                .tint(Ember.ember)
        }
    }

    // MARK: dictionary

    private var dictionaryCard: some View {
        Card(title: "Dictionary") {
            Button {
                showDictionary = true
            } label: {
                HStack(spacing: 12) {
                    Image(systemName: "character.book.closed").frame(width: 22).foregroundStyle(Ember.ember)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(session.dictionary.isEmpty ? "Add your words"
                             : "\(session.dictionary.entries.count) word\(session.dictionary.entries.count == 1 ? "" : "s")")
                            .font(.system(size: 15, weight: .medium))
                        Text("Names and jargon Phonon should spell your way.")
                            .font(.system(size: 12)).foregroundStyle(Ember.mute)
                    }
                    Spacer()
                    Image(systemName: "chevron.right").font(.system(size: 13, weight: .semibold)).foregroundStyle(Ember.mute)
                }
                .padding(.vertical, 4)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
        }
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            Text("ϕ").font(.system(size: 30, weight: .semibold)).foregroundStyle(Ember.ember)
            Text("Phonon").font(.system(size: 30, weight: .semibold))
            Spacer()
            Text("on-device").font(.system(size: 12, weight: .medium, design: .monospaced))
                .foregroundStyle(Ember.mute)
                .padding(.horizontal, 8).padding(.vertical, 4)
                .background(Capsule().stroke(Ember.line))
        }
        .padding(.top, 8)
    }

    // MARK: session

    private var sessionCard: some View {
        Card {
            HStack(spacing: 16) {
                PulseDot(live: session.live, busy: session.status == "Listening")
                VStack(alignment: .leading, spacing: 3) {
                    Text(session.running ? sessionTitle : "Phonon is off")
                        .font(.system(size: 18, weight: .semibold))
                    Text(session.running
                        ? "Switch to any app, pick the Phonon keyboard, and talk."
                        : "Start once. The keyboard can then dictate in any app without bouncing back here.")
                        .font(.system(size: 13)).foregroundStyle(Ember.mute)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
            }
            Button {
                if session.running { session.stopSession() } else { Task { await session.start() } }
            } label: {
                Text(session.running ? "Stop" : "Start Phonon")
                    .font(.system(size: 16, weight: .semibold))
                    .frame(maxWidth: .infinity)
                    .frame(height: 48)
                    .background(RoundedRectangle(cornerRadius: 12).fill(session.running ? Ember.panelRaised : Ember.ember))
                    .foregroundStyle(session.running ? Ember.soft : .white)
            }
            .buttonStyle(.plain)
            .padding(.top, 4)
            if session.backHint && session.running {
                // only shown when auto-return failed: make the way back impossible to miss
                VStack(alignment: .leading, spacing: 6) {
                    Label("Phonon is listening. Go back to your app", systemImage: "arrow.uturn.left")
                        .font(.system(size: 17, weight: .semibold))
                    Text("Tap ◀ in the top-left corner, or swipe right along the bottom edge.")
                        .font(.system(size: 15))
                        .foregroundStyle(Ember.soft)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(14)
                .background(RoundedRectangle(cornerRadius: 12).fill(Ember.ember.opacity(0.18)))
                .overlay(RoundedRectangle(cornerRadius: 12).stroke(Ember.ember, lineWidth: 1.5))
            }
            if session.status.hasPrefix("Microphone") {
                Text(session.status).font(.footnote).foregroundStyle(Ember.warn)
            }
        }
    }

    private var sessionTitle: String {
        switch session.status {
        case "Listening": return "Listening"
        case "Transcribing", "Refining": return "Writing it down"
        default: return "Ready to dictate"
        }
    }

    // MARK: setup

    private var setupCard: some View {
        Card(title: "Set up the keyboard") {
            SetupRow(n: 1, title: "Add the Phonon keyboard", detail: "Settings > Phonon > Keyboards", done: keyboardAdded,
                     action: keyboardAdded ? nil : ("Open Settings", openSettings))
            // both steps open the same Settings page, so only the first open step carries the button
            SetupRow(n: 2, title: "Allow Full Access", detail: "Lets the keyboard talk to this app. Nothing leaves your iPhone.",
                     done: fullAccess, action: (fullAccess || !keyboardAdded) ? nil : ("Open Settings", openSettings))
            SetupRow(n: 3, title: "Speech model", detail: session.parakeetReady ? "Parakeet, on device"
                        : (session.running ? session.asrStatus.replacingOccurrences(of: "Speech model: ", with: "")
                            : "Downloads once (0.6 GB) when you start Phonon"),
                     done: session.parakeetReady, action: nil)
        }
    }

    private func refreshSetup() {
        keyboardAdded = UITextInputMode.activeInputModes.contains { mode in
            (mode.value(forKey: "identifier") as? String)?.hasPrefix("sh.phonon.ios") == true
        }
        fullAccess = IPC.double(IPC.keyboardSeen) > 0
    }

    private func openSettings() {
        if let url = URL(string: UIApplication.openSettingsURLString) { UIApplication.shared.open(url) }
    }

    // MARK: writing mode

    private var modeCard: some View {
        Card(title: "Writing mode") {
            ForEach(WritingMode.allCases) { m in
                Button {
                    UISelectionFeedbackGenerator().selectionChanged()
                    session.mode = m.rawValue
                } label: {
                    HStack(spacing: 12) {
                        Image(systemName: m.symbol).frame(width: 22)
                            .foregroundStyle(session.mode == m.rawValue ? Ember.ember : Ember.mute)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(m.title).font(.system(size: 15, weight: .medium))
                            Text(m.blurb).font(.system(size: 12)).foregroundStyle(Ember.mute)
                        }
                        Spacer()
                        if session.mode == m.rawValue {
                            Image(systemName: "checkmark").font(.system(size: 13, weight: .bold)).foregroundStyle(Ember.ember)
                        }
                    }
                    .padding(.vertical, 6)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
            Toggle(isOn: $session.useRefiner) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Clean up with the Phonon model").font(.system(size: 15, weight: .medium))
                    Text(RefinerSupport.onThisDevice ? "Off: you get the raw transcript." : RefinerSupport.offNote)
                        .font(.system(size: 12)).foregroundStyle(Ember.mute)
                }
            }
            .disabled(!RefinerSupport.onThisDevice)
            .padding(.top, 4)
        }
    }

    // MARK: history

    private var historyCard: some View {
        Card(title: "Recent") {
            if session.history.isEmpty {
                Text("Your dictations show up here. They stay on this iPhone.")
                    .font(.system(size: 13)).foregroundStyle(Ember.mute)
            } else {
                ForEach(session.history.prefix(12)) { d in
                    VStack(alignment: .leading, spacing: 4) {
                        Text(d.text).font(.system(size: 14)).lineLimit(4)
                        HStack(spacing: 8) {
                            Text(d.at, style: .time)
                            Text(d.mode.capitalized)
                            if let ms = d.ms, ms >= 0 { Text(String(format: "%.1f s", Double(ms) / 1000)) }
                            Spacer()
                            Button {
                                UIPasteboard.general.string = d.text
                                copied = d.id
                            } label: {
                                Label(copied == d.id ? "Copied" : "Copy", systemImage: copied == d.id ? "checkmark" : "doc.on.doc")
                                    .labelStyle(.titleAndIcon)
                            }
                            .buttonStyle(.plain)
                            .foregroundStyle(Ember.soft)
                        }
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(Ember.mute)
                    }
                    .padding(.vertical, 6)
                    if d.id != session.history.prefix(12).last?.id { Divider().overlay(Ember.line) }
                }
            }
        }
    }

    private var modelsFooter: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(session.asrStatus)
            Text(session.refinerStatus)
        }
        .font(.system(size: 11, design: .monospaced))
        .foregroundStyle(Ember.dim)
        .padding(.horizontal, 4)
    }
}

// MARK: pieces (Card is in DictionaryView.swift, so the unit tests can render that screen)

struct SetupRow: View {
    let n: Int
    let title: String
    let detail: String
    let done: Bool
    let action: (String, () -> Void)?

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            ZStack {
                Circle().fill(done ? Ember.done.opacity(0.18) : Ember.panelRaised)
                if done {
                    Image(systemName: "checkmark").font(.system(size: 11, weight: .bold)).foregroundStyle(Ember.done)
                } else {
                    Text("\(n)").font(.system(size: 12, weight: .semibold)).foregroundStyle(Ember.soft)
                }
            }
            .frame(width: 24, height: 24)
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.system(size: 15, weight: .medium)).foregroundStyle(done ? Ember.mute : Ember.ink)
                Text(detail).font(.system(size: 12)).foregroundStyle(Ember.mute).fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
            if let action {
                Button(action.0, action: action.1)
                    .font(.system(size: 13, weight: .semibold))
                    .buttonStyle(.plain)
                    .foregroundStyle(Ember.ember2)
            }
        }
        .padding(.vertical, 3)
    }
}

/// Session indicator: a still dot when off, a slow ember pulse when live, faster while listening.
struct PulseDot: View {
    let live: Bool
    let busy: Bool
    @State private var on = false

    var body: some View {
        ZStack {
            Circle().fill((live ? Ember.ember : Ember.dim).opacity(0.18)).frame(width: 46, height: 46)
                .scaleEffect(live && on ? 1.15 : 0.9)
            Circle().fill(live ? Ember.ember : Ember.dim).frame(width: 14, height: 14)
        }
        .animation(live ? .easeInOut(duration: busy ? 0.45 : 1.4).repeatForever(autoreverses: true) : .default, value: on)
        .onAppear { on = true }
        .onChange(of: live) { _, _ in on.toggle() }
    }
}

// MARK: keyboard preview (simulator screenshots: --preview-keyboard [--listening] [--off])

private final class PreviewHost: KeyboardHost {
    var text = "Hey, can you"
    func insert(_ t: String) { text += t }
    func deleteBackward() { if !text.isEmpty { text.removeLast() } }
    func nextKeyboard() {}
    var contextBefore: String { text }
    func micDown() {}
    func micUp() {}
    func cancelDictation() {}
    func undoDictation() {}
    func openApp() {}
    func startViaApp() {}
}

struct KeyboardPreview: View {
    let listening: Bool
    let needsSession: Bool
    var working = false
    @StateObject private var model = KeyboardModel()
    @State private var host = PreviewHost()

    var body: some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 8) {
                Text("Messages").font(.system(size: 13, weight: .semibold)).foregroundStyle(Ember.mute)
                Text(host.text + "|").font(.system(size: 17))
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(RoundedRectangle(cornerRadius: 18).fill(Color(hex: 0x1F1F22)))
            }
            .padding(16)
            Spacer()
            KeyboardView(model: model)
                .frame(height: 280)
        }
        .background(Color.black.ignoresSafeArea())
        .onAppear {
            model.host = host
            model.contextVersion += 1
            model.showGlobe = true
            if needsSession {
                model.phase = .needsSession
                model.status = "Opens Phonon once to keep the mic ready"
            } else if working {
                model.phase = .working
            } else if listening {
                model.phase = .listening
                model.levels = (0..<28).map { i in CGFloat(0.15 + 0.75 * abs(sin(Double(i) * 0.55)) * (i > 6 ? 1 : 0.3)) }
            } else {
                model.phase = .ready
                model.canUndo = true
                model.status = "14 words · 0.9 s"
            }
        }
    }
}
