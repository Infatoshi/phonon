import AppKit
import SwiftUI

// The meeting popup and the capsule's recording state (DEVLOG 2026-11-04, mock
// ui_pitch/app.html "Meeting starts"). The popup shows only the meeting app's
// logo with "Not now" and "Record", just above the capsule; while a meeting
// records, a red light blinks inside the capsule and hovering it shows a round
// stop button over the light; a click stops and "Saved to Notes" shows.

/// What the meeting UI shows, as one value: the popup above the capsule, the
/// recording app, and whether the pointer is over the capsule. MeetingController
/// owns it; the popup panel and the capsule (PillState) follow it.
struct MeetingFlow: Equatable {
    enum Popup: Equatable {
        case hidden
        /// "Ask every time": Not now / Record.
        case asking(app: String, bundleID: String)
        /// "Remember per app" (the default), an app without a rule: "Record this and future <App> calls
        /// automatically?" Always / Just this once / Not now / Never.
        case askingRule(app: String, bundleID: String)
        /// A rule was just set from the question: Undo for a few seconds. `recorded` is
        /// whether that answer also started this recording (Always).
        case undo(app: String, bundleID: String, change: MeetingRuleChange, recorded: Bool)
        case saved
        /// The one-time meeting questions for an install that predates them.
        case questions
    }

    private(set) var popup = Popup.hidden
    private(set) var recordingApp: String?
    private(set) var hovering = false

    var isRecording: Bool { recordingApp != nil }
    /// The capsule takes the pointer while a meeting records (for the stop button) and
    /// while the pointer is over it (for the start button); otherwise every click passes
    /// through to the app below.
    var capsuleAcceptsMouse: Bool { isRecording || hovering }
    var showsStopButton: Bool { isRecording && hovering }
    /// Hovering the idle capsule offers to start the note taker by hand.
    var showsStartButton: Bool { !isRecording && hovering }

    mutating func ask(app: String, bundleID: String, remember: Bool = false) {
        guard !isRecording else { return }
        popup = remember
            ? .askingRule(app: app, bundleID: bundleID) : .asking(app: app, bundleID: bundleID)
    }

    /// "Not now": the app to remember as declined, if a question was up.
    mutating func notNow() -> String? {
        switch popup {
        case let .asking(app, _), let .askingRule(app, _):
            popup = .hidden
            return app
        default:
            return nil
        }
    }

    /// An answer to the per-app question: the call it was about, if that question was up.
    mutating func choose(_ choice: MeetingChoice) -> (app: String, bundleID: String)? {
        guard case let .askingRule(app, bundleID) = popup else { return nil }
        popup = .hidden
        return (app, bundleID)
    }

    /// Show Undo for a rule just set from the question.
    mutating func offerUndo(app: String, bundleID: String, change: MeetingRuleChange, recorded: Bool) {
        popup = .undo(app: app, bundleID: bundleID, change: change, recorded: recorded)
    }

    /// Undo: the change to revert, and the question is up again for the same call.
    mutating func undo() -> (app: String, bundleID: String, change: MeetingRuleChange, recorded: Bool)? {
        guard case let .undo(app, bundleID, change, recorded) = popup else { return nil }
        popup = .askingRule(app: app, bundleID: bundleID)
        return (app, bundleID, change, recorded)
    }

    /// The one-time questions, only when nothing else is up.
    mutating func askQuestions() -> Bool {
        guard !isRecording, popup == .hidden else { return false }
        popup = .questions
        return true
    }

    /// The questions were answered or skipped.
    mutating func answeredQuestions() -> Bool {
        guard popup == .questions else { return false }
        popup = .hidden
        return true
    }

    /// "Record": the call to start, if a question was up.
    mutating func record() -> (app: String, bundleID: String)? {
        guard case let .asking(app, bundleID) = popup else { return nil }
        popup = .hidden
        return (app, bundleID)
    }

    /// Recording began (from "Record", a rule, or by hand). An Undo card stays up.
    mutating func started(app: String) {
        recordingApp = app
        hovering = false
        switch popup {
        case .asking, .askingRule, .questions: popup = .hidden
        default: break
        }
    }

    mutating func hover(_ inside: Bool) {
        hovering = inside
    }

    /// Recording ended. A stop the user asked for confirms with "Saved to Notes".
    mutating func stopped(byUser: Bool) {
        guard isRecording else { return }
        recordingApp = nil
        hovering = false
        popup = byUser ? .saved : .hidden
    }

    /// The question or confirmation timed out (not a decline).
    mutating func expire() {
        popup = .hidden
    }
}

// MARK: - Icon

/// The meeting app's own icon, from NSWorkspace. A native call app (Zoom,
/// Teams, FaceTime, Slack, Discord...) gives its bundle's icon. A call in a
/// browser tab gives the site's installed app (zoom.us -> Zoom), else an
/// installed web app for the site (a Chrome/Edge/Brave "Google Meet" shim),
/// else the browser holding the call; a camera symbol if none resolves.
enum MeetingAppIcon {
    enum Source: Equatable {
        case app(URL)
        case symbol
    }

    /// Native apps for conferencing sites, by host suffix.
    static let siteApps: [(host: String, bundleIDs: [String])] = [
        ("zoom.us", ["us.zoom.xos"]),
        ("zoomgov.com", ["us.zoom.xos"]),
        ("teams.microsoft.com", ["com.microsoft.teams2", "com.microsoft.teams"]),
        ("teams.live.com", ["com.microsoft.teams2", "com.microsoft.teams"]),
        ("slack.com", ["com.tinyspeck.slackmacgap"]),
        ("discord.com", ["com.hnc.Discord"]),
        ("webex.com", ["com.cisco.webexmeetingsapp", "Cisco-Systems.Spark"]),
        ("around.co", ["co.around.Around"]),
    ]

    /// Where browsers install web apps ("Install Google Meet" in Chrome).
    static func webAppFolders(home: URL = FileManager.default.homeDirectoryForCurrentUser)
        -> [URL]
    {
        ["Chrome Apps.localized", "Edge Apps.localized", "Brave Browser Apps.localized",
         "Chromium Apps.localized"]
            .map { home.appendingPathComponent("Applications/\($0)", isDirectory: true) }
    }

    /// Whether the verdict came from a conference tab (its app is the site host).
    static func isSite(bundleID: String) -> Bool {
        MeetingDetector.callApps[bundleID] == nil
            && MeetingDetector.chatAppsWithCalls[bundleID] == nil
    }

    static func source(
        app: String, bundleID: String,
        appURL: (String) -> URL? = { NSWorkspace.shared.urlForApplication(withBundleIdentifier: $0) },
        webApp: (String) -> URL? = { MeetingAppIcon.webAppURL(host: $0) }
    ) -> Source {
        if !isSite(bundleID: bundleID) {
            return appURL(bundleID).map(Source.app) ?? .symbol
        }
        let host = app.lowercased()
        for (suffix, ids) in siteApps where host == suffix || host.hasSuffix("." + suffix) {
            if let url = ids.lazy.compactMap(appURL).first { return .app(url) }
        }
        if let url = webApp(host) { return .app(url) }
        return appURL(bundleID).map(Source.app) ?? .symbol
    }

    /// An installed web app shim whose start URL is on `host`.
    static func webAppURL(host: String, folders: [URL] = MeetingAppIcon.webAppFolders())
        -> URL?
    {
        let files = FileManager.default
        for folder in folders {
            guard let apps = try? files.contentsOfDirectory(
                at: folder, includingPropertiesForKeys: nil)
            else { continue }
            for app in apps where app.pathExtension == "app" {
                let plist = app.appendingPathComponent("Contents/Info.plist")
                guard let info = NSDictionary(contentsOf: plist),
                    let start = info["CrAppModeShortcutURL"] as? String,
                    let startHost = ContextPolicy.host(of: URL(string: start))
                else { continue }
                if startHost == host || startHost.hasSuffix("." + host) { return app }
            }
        }
        return nil
    }

    static func image(for source: Source) -> NSImage {
        switch source {
        case let .app(url):
            return NSWorkspace.shared.icon(forFile: url.path)
        case .symbol:
            let image = NSImage(
                systemSymbolName: "video.fill", accessibilityDescription: "Meeting")
            return image ?? NSImage(size: NSSize(width: 28, height: 28))
        }
    }

    static func image(app: String, bundleID: String) -> NSImage {
        image(for: source(app: app, bundleID: bundleID))
    }
}

// MARK: - Popup

enum MeetingPopupStyle {
    static let surface = Color(red: 0x0d / 255, green: 0x09 / 255, blue: 0x07 / 255).opacity(0.95)
    static let border = Color(red: 0x3b / 255, green: 0x25 / 255, blue: 0x16 / 255)
    static let ghostText = Color(red: 0xcf / 255, green: 0xc3 / 255, blue: 0xb6 / 255)
    static let ghostHover = Color(red: 0x22 / 255, green: 0x1a / 255, blue: 0x14 / 255)
    static let record = Color(red: 0xc8 / 255, green: 0x32 / 255, blue: 0x1f / 255)
    static let recordHover = Color(red: 0xdc / 255, green: 0x3a / 255, blue: 0x25 / 255)
    static let savedBorder = Color(red: 0x2c / 255, green: 0x3b / 255, blue: 0x2f / 255)
    static let savedText = Color(red: 0xbf / 255, green: 0xe8 / 255, blue: 0xcb / 255)
}

@MainActor
final class MeetingPopupModel: ObservableObject {
    enum Content {
        case ask(NSImage)
        /// The per-app question: icon and the app's or site's name.
        case askRule(NSImage, String)
        /// What was just remembered, with Undo.
        case undo(String)
        case saved
        /// The one-time meeting questions.
        case questions

        var wide: Bool {
            switch self {
            case .ask, .saved: return false
            case .askRule, .undo, .questions: return true
            }
        }
    }

    @Published var content = Content.saved
    /// Drives the slide-up and fade-in.
    @Published var visible = false
    var onNotNow: () -> Void = {}
    var onRecord: () -> Void = {}
    var onChoose: (MeetingChoice) -> Void = { _ in }
    var onUndo: () -> Void = {}
    /// The questions card: a mode, or nil for Skip.
    var onAnswer: (MeetingMode?) -> Void = { _ in }
}

struct MeetingPopupView: View {
    @ObservedObject var model: MeetingPopupModel

    var body: some View {
        let size = MeetingPopupPanel.size(wide: model.content.wide)
        card
            .shadow(color: .black.opacity(0.72), radius: 17, y: 14)
            .opacity(model.visible ? 1 : 0)
            .offset(y: model.visible ? 0 : 8)
            .animation(.easeOut(duration: 0.16), value: model.visible)
            .padding(.top, MeetingPopupPanel.topMargin)
            .padding(.bottom, MeetingPopupPanel.bottomMargin)
            .frame(width: size.width, height: size.height, alignment: .bottom)
    }

    private func surface<Content: View>(_ content: Content) -> some View {
        content
            .background(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(MeetingPopupStyle.surface))
            .overlay(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .strokeBorder(MeetingPopupStyle.border, lineWidth: 1))
    }

    private func appIcon(_ icon: NSImage) -> some View {
        Image(nsImage: icon)
            .resizable()
            .interpolation(.high)
            .aspectRatio(contentMode: .fit)
            .frame(width: 28, height: 28)
            .accessibilityLabel("Meeting app")
    }

    @ViewBuilder private var card: some View {
        switch model.content {
        case let .askRule(icon, name):
            surface(
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 10) {
                        appIcon(icon)
                        Text("Record this and future \(name) calls automatically?")
                            .font(.system(size: 13))
                            .foregroundColor(MeetingPopupStyle.savedText.opacity(0.95))
                            .lineLimit(2)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    HStack(spacing: 4) {
                        Button(action: { model.onChoose(.always) }) {
                            HStack(spacing: 7) {
                                Circle().fill(Color.white).frame(width: 8, height: 8)
                                Text("Always")
                            }
                        }
                        .buttonStyle(MeetingRecordButtonStyle())
                        Button("Just this once") { model.onChoose(.once) }
                            .buttonStyle(MeetingGhostButtonStyle())
                        Button("Not now") { model.onChoose(.notNow) }
                            .buttonStyle(MeetingGhostButtonStyle())
                        Button("Never") { model.onChoose(.never) }
                            .buttonStyle(MeetingGhostButtonStyle())
                    }
                }
                .padding(EdgeInsets(top: 10, leading: 10, bottom: 8, trailing: 8)))
        case let .undo(text):
            surface(
                HStack(spacing: 10) {
                    Image(systemName: "checkmark")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundColor(MeetingPopupStyle.savedText)
                    Text(text)
                        .font(.system(size: 13))
                        .foregroundColor(MeetingPopupStyle.savedText)
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                    Button("Undo", action: model.onUndo)
                        .buttonStyle(MeetingGhostButtonStyle())
                }
                .padding(EdgeInsets(top: 6, leading: 12, bottom: 6, trailing: 6)))
        case .questions:
            surface(
                VStack(alignment: .leading, spacing: 8) {
                    Text("Phonon can notice calls and take notes on this Mac. When a call starts:")
                        .font(.system(size: 13))
                        .foregroundColor(MeetingPopupStyle.savedText.opacity(0.95))
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                    HStack(spacing: 4) {
                        Button(MeetingMode.perApp.title) { model.onAnswer(.perApp) }
                            .buttonStyle(MeetingRecordButtonStyle())
                        Button(MeetingMode.everyCall.title) { model.onAnswer(.everyCall) }
                            .buttonStyle(MeetingGhostButtonStyle())
                        Button(MeetingMode.off.title) { model.onAnswer(.off) }
                            .buttonStyle(MeetingGhostButtonStyle())
                        Button("Skip") { model.onAnswer(nil) }
                            .buttonStyle(MeetingGhostButtonStyle())
                    }
                }
                .padding(EdgeInsets(top: 10, leading: 12, bottom: 8, trailing: 8)))
        case let .ask(icon):
            HStack(spacing: 10) {
                Image(nsImage: icon)
                    .resizable()
                    .interpolation(.high)
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 28, height: 28)
                    .accessibilityLabel("Meeting app")
                Button("Not now", action: model.onNotNow)
                    .buttonStyle(MeetingGhostButtonStyle())
                Button(action: model.onRecord) {
                    HStack(spacing: 7) {
                        Circle().fill(Color.white).frame(width: 8, height: 8)
                        Text("Record")
                    }
                }
                .buttonStyle(MeetingRecordButtonStyle())
            }
            .padding(EdgeInsets(top: 8, leading: 10, bottom: 8, trailing: 8))
            .background(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(MeetingPopupStyle.surface))
            .overlay(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .strokeBorder(MeetingPopupStyle.border, lineWidth: 1))
        case .saved:
            HStack(spacing: 8) {
                Image(systemName: "checkmark")
                    .font(.system(size: 11, weight: .bold))
                Text("Saved to Notes")
                    .font(.system(size: 13))
            }
            .foregroundColor(MeetingPopupStyle.savedText)
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(MeetingPopupStyle.surface))
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .strokeBorder(MeetingPopupStyle.savedBorder, lineWidth: 1))
        }
    }
}

private struct MeetingGhostButtonStyle: ButtonStyle {
    @State private var hover = false

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 13))
            .foregroundColor(MeetingPopupStyle.ghostText)
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background(
                RoundedRectangle(cornerRadius: 9, style: .continuous)
                    .fill(
                        hover || configuration.isPressed
                            ? MeetingPopupStyle.ghostHover : Color.clear))
            .contentShape(Rectangle())
            .onHover { hover = $0 }
    }
}

private struct MeetingRecordButtonStyle: ButtonStyle {
    @State private var hover = false

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 13))
            .foregroundColor(.white)
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .background(
                RoundedRectangle(cornerRadius: 9, style: .continuous)
                    .fill(
                        hover || configuration.isPressed
                            ? MeetingPopupStyle.recordHover : MeetingPopupStyle.record))
            .contentShape(Rectangle())
            .onHover { hover = $0 }
    }
}

/// Borderless and non-activating, never key or main: clicking "Record" or
/// "Not now" leaves the user's app in front with its focus. The transparent
/// margin is room for the card's shadow and slide.
final class MeetingPopupPanel: NSPanel {
    static let size = NSSize(width: 320, height: 96)
    /// The per-app question, Undo and the meeting questions: two rows, four buttons.
    static let wideSize = NSSize(width: 480, height: 140)

    static func size(wide: Bool) -> NSSize { wide ? wideSize : size }
    static let topMargin: CGFloat = 8
    /// Below the card: its shadow (0 14px 34px in the mock).
    static let bottomMargin: CGFloat = 40
    /// The card's bottom edge above the screen bottom: the capsule's bottom
    /// (PanelGeometry.bottomInset) plus 30, as in the mock.
    static let cardBottom: CGFloat = PanelGeometry.bottomInset + 30

    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }

    init() {
        super.init(
            contentRect: NSRect(origin: .zero, size: Self.size),
            styleMask: [.borderless, .nonactivatingPanel, .fullSizeContentView],
            backing: .buffered, defer: true)
        isFloatingPanel = true
        level = .floating
        hidesOnDeactivate = false
        becomesKeyOnlyIfNeeded = true
        collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        backgroundColor = .clear
        isOpaque = false
        hasShadow = false
        appearance = NSAppearance(named: .darkAqua)
    }

    static func frame(screen: NSRect, size: NSSize = MeetingPopupPanel.size) -> NSRect {
        NSRect(
            x: screen.midX - size.width / 2, y: screen.minY + cardBottom - bottomMargin,
            width: size.width, height: size.height)
    }
}

/// A hosting view that takes the first click without activating Phonon.
final class FirstMouseHostingView<Content: View>: NSHostingView<Content> {
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
}

@MainActor
final class MeetingPopupController {
    /// A question stays this long (as the old card), the confirmation this long.
    static let questionSeconds: TimeInterval = SuggestionPanelController.questionSeconds
    static let savedSeconds: TimeInterval = 1.6
    /// How long Undo stays after a rule is set from the question.
    static let undoSeconds: TimeInterval = 6
    /// The one-time meeting questions stay longer: nothing about them is urgent.
    static let questionsSeconds: TimeInterval = 30

    private var panel: MeetingPopupPanel?
    let model = MeetingPopupModel()
    private var expiry: DispatchWorkItem?
    private var generation = 0

    private func clearActions() {
        model.onNotNow = {}
        model.onRecord = {}
        model.onChoose = { _ in }
        model.onUndo = {}
        model.onAnswer = { _ in }
    }

    func ask(icon: NSImage, onNotNow: @escaping () -> Void, onRecord: @escaping () -> Void,
        onExpire: @escaping () -> Void)
    {
        clearActions()
        model.content = .ask(icon)
        model.onNotNow = onNotNow
        model.onRecord = onRecord
        show(for: Self.questionSeconds, onExpire: onExpire)
    }

    /// "Record this and future <name> calls automatically?"
    func askRule(
        icon: NSImage, name: String, onChoose: @escaping (MeetingChoice) -> Void,
        onExpire: @escaping () -> Void
    ) {
        clearActions()
        model.content = .askRule(icon, name)
        model.onChoose = onChoose
        show(for: Self.questionSeconds, onExpire: onExpire)
    }

    func undo(text: String, onUndo: @escaping () -> Void, onExpire: @escaping () -> Void) {
        clearActions()
        model.content = .undo(text)
        model.onUndo = onUndo
        show(for: Self.undoSeconds, onExpire: onExpire)
    }

    func questions(onAnswer: @escaping (MeetingMode?) -> Void, onExpire: @escaping () -> Void) {
        clearActions()
        model.content = .questions
        model.onAnswer = onAnswer
        show(for: Self.questionsSeconds, onExpire: onExpire)
    }

    func saved(onExpire: @escaping () -> Void) {
        clearActions()
        model.content = .saved
        show(for: Self.savedSeconds, onExpire: onExpire)
    }

    func dismiss() {
        expiry?.cancel()
        expiry = nil
        generation += 1
        let current = generation
        model.visible = false
        // Let the fade run, unless something new showed meanwhile.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.18) { [weak self] in
            guard let self, self.generation == current else { return }
            self.panel?.orderOut(nil)
        }
    }

    private func show(for seconds: TimeInterval, onExpire: @escaping () -> Void) {
        generation += 1
        let panel = self.panel ?? makePanel()
        self.panel = panel
        let size = MeetingPopupPanel.size(wide: model.content.wide)
        panel.setFrame(
            MeetingPopupPanel.frame(screen: PillPanel.screenFrame(), size: size), display: false)
        panel.contentView?.frame = NSRect(origin: .zero, size: size)
        if !panel.isVisible {
            model.visible = false
            panel.orderFrontRegardless()  // never makeKey: it must not take focus
        }
        // Start hidden for one pass, then slide up and fade in.
        DispatchQueue.main.async { [weak self] in self?.model.visible = true }
        expiry?.cancel()
        let work = DispatchWorkItem { [weak self] in
            self?.dismiss()
            onExpire()
        }
        expiry = work
        DispatchQueue.main.asyncAfter(deadline: .now() + seconds, execute: work)
    }

    private func makePanel() -> MeetingPopupPanel {
        let panel = MeetingPopupPanel()
        let host = FirstMouseHostingView(rootView: MeetingPopupView(model: model))
        host.frame = NSRect(origin: .zero, size: MeetingPopupPanel.size)
        panel.contentView = host
        return panel
    }
}

// MARK: - Capsule recording state

/// The capsule's red recording light: #ff3b2f, about 5 pt, a soft glow,
/// blinking on a 1.2 s ease-in-out between full and 25%.
enum RecordLight {
    static let color = Color(red: 1, green: 0x3b / 255, blue: 0x2f / 255)
    static let diameter: CGFloat = 5
    static let period: TimeInterval = 1.2
    static let stopDiameter: CGFloat = 22

    /// Full at 0, 25% at half the period, full again at the period.
    static func opacity(at seconds: TimeInterval) -> Double {
        let phase = seconds.truncatingRemainder(dividingBy: period) / period
        let leg = phase < 0.5 ? phase * 2 : (1 - phase) * 2
        let eased = (1 - cos(Double.pi * leg)) / 2
        return 1 - 0.75 * eased
    }

    /// The glow follows the light (6 px blur at full, 2 px dim in the mock).
    static func glowRadius(opacity: Double) -> CGFloat {
        CGFloat(1 + 2 * (opacity - 0.25) / 0.75)
    }
}

/// The round stop button that replaces the light on hover: red, white rounded
/// square inside, scaled in over about 160 ms.
struct CapsuleStopButton: View {
    var body: some View {
        Circle()
            .fill(RecordLight.color)
            .overlay(
                RoundedRectangle(cornerRadius: 2, style: .continuous)
                    .fill(Color.white)
                    .frame(width: 8, height: 8))
            .frame(width: RecordLight.stopDiameter, height: RecordLight.stopDiameter)
            .shadow(color: RecordLight.color.opacity(0.53), radius: 5, y: 2)
            .accessibilityLabel("Stop recording")
    }
}

/// The round start button the idle capsule shows on hover: dark, ember ring, a
/// white play triangle. It starts the note taker with nothing detected.
struct CapsuleStartButton: View {
    var body: some View {
        Circle()
            .fill(Color(red: 0x1a / 255, green: 0x12 / 255, blue: 0x0d / 255))
            .overlay(Circle().strokeBorder(RecordLight.color.opacity(0.85), lineWidth: 1.5))
            .overlay(
                Image(systemName: "play.fill")
                    .font(.system(size: 9, weight: .bold))
                    .foregroundColor(.white)
                    .offset(x: 1))
            .frame(width: RecordLight.stopDiameter, height: RecordLight.stopDiameter)
            .shadow(color: .black.opacity(0.5), radius: 4, y: 2)
            .accessibilityLabel("Start note taker")
            .help("Start the note taker: record your mic and this Mac's audio")
    }
}

/// The capsule panel's content: room around the 80 x 16 capsule for the stop
/// button, which is larger than the capsule. The capsule's own spot on screen
/// does not move (PillPanelLayout).
struct PillPanelContent: View {
    @ObservedObject var state: PillState
    var onStop: () -> Void = {}
    var onStart: () -> Void = {}

    var body: some View {
        PillView(state: state, onStop: onStop, onStart: onStart)
            .padding(PillPanelLayout.insets)
    }
}

enum PillPanelLayout {
    static let capsule = NSSize(width: 80, height: 16)
    static let insets = EdgeInsets(top: 12, leading: 8, bottom: 12, trailing: 8)
    static let size = NSSize(
        width: capsule.width + insets.leading + insets.trailing,
        height: capsule.height + insets.top + insets.bottom)

    /// The panel frame that keeps the capsule at its resting spot.
    static func frame(screen: NSRect) -> NSRect {
        let resting = PanelGeometry.restingFrame(screen: screen, size: capsule)
        return NSRect(
            x: resting.minX - insets.leading, y: resting.minY - insets.bottom,
            width: size.width, height: size.height)
    }
}

/// Notices the pointer over the idle capsule, whose panel ignores the mouse (clicks pass
/// through it) until then: a global mouse-moved monitor, installed only while the idle
/// capsule offers its start button, and a 10 Hz position check only while the pointer is
/// inside (the panel then takes the mouse, so the global monitor no longer sees it).
@MainActor
final class CapsuleHoverWatch {
    /// The capsule panel's frame on screen.
    var area: () -> NSRect = { .zero }
    var onChange: (Bool) -> Void = { _ in }
    private var monitor: Any?
    private var poll: Timer?
    private(set) var inside = false

    func setArmed(_ armed: Bool) {
        if armed {
            guard monitor == nil else { return }
            monitor = NSEvent.addGlobalMonitorForEvents(matching: [.mouseMoved]) { [weak self] _ in
                MainActor.assumeIsolated { self?.check() }
            }
            check()
        } else {
            if let monitor { NSEvent.removeMonitor(monitor) }
            monitor = nil
            set(false)
        }
    }

    private func check() {
        set(area().contains(NSEvent.mouseLocation))
    }

    private func set(_ now: Bool) {
        guard now != inside else { return }
        inside = now
        if now {
            poll = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
                MainActor.assumeIsolated { self?.check() }
            }
        } else {
            poll?.invalidate()
            poll = nil
        }
        onChange(now)
    }
}

/// The capsule's hosting view: reports the pointer entering and leaving with an
/// always-active tracking area (Phonon is never the active app) and takes the
/// first click, so the stop button works without activating Phonon. The panel
/// ignores the mouse entirely unless a meeting records.
final class PillHostingView<Content: View>: NSHostingView<Content> {
    var onHover: (Bool) -> Void = { _ in }
    private var area: NSTrackingArea?

    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        if let area { removeTrackingArea(area) }
        let area = NSTrackingArea(
            rect: .zero, options: [.mouseEnteredAndExited, .activeAlways, .inVisibleRect],
            owner: self, userInfo: nil)
        addTrackingArea(area)
        self.area = area
    }

    override func mouseEntered(with event: NSEvent) {
        super.mouseEntered(with: event)
        onHover(true)
    }

    override func mouseExited(with event: NSEvent) {
        super.mouseExited(with: event)
        onHover(false)
    }
}
