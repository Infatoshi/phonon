import AppKit
import SwiftUI

/// The card above the capsule for a dictation that went wrong, after Wispr's notices:
/// - `notPasted`: nowhere to paste (no window, or the desktop, was focused). The text
///   shows with a big copy button and is on the clipboard already (Wispr's "void
///   dictation": "Select a textbox first, then dictate", Copy).
/// - `needsAccessibility`: the paste was refused for want of the permission; same text
///   and copy, plus the way to the setting.
/// - `notHeard`: the dictation came back empty (Wispr's "We didn't hear you"), naming
///   the mic.
/// - `micSilent`: shown while the key is held when the mic delivers only digital silence
///   (Wispr's "no audio during dictation").
/// Every card has a close button (issue #22).
enum Notice: Equatable {
    case notPasted(String)
    case needsAccessibility(String)
    case notHeard(mic: String)
    case micSilent(mic: String)

    var text: String? {
        switch self {
        case .notPasted(let text), .needsAccessibility(let text): return text
        case .notHeard, .micSilent: return nil
        }
    }
}

final class NotPastedModel: ObservableObject {
    @Published var notice = Notice.notPasted("")
    /// Drives the slide-up and fade-in.
    @Published var visible = false
    var onHover: (Bool) -> Void = { _ in }
    var onClose: () -> Void = {}

    var text: String {
        get { notice.text ?? "" }
        set { notice = .notPasted(newValue) }
    }
}

struct NotPastedView: View {
    @ObservedObject var model: NotPastedModel

    var body: some View {
        card
            .shadow(color: .black.opacity(0.67), radius: 12, y: 10)
            .opacity(model.visible ? 1 : 0)
            .offset(y: model.visible ? 0 : 8)
            .animation(.easeOut(duration: 0.16), value: model.visible)
            .onHover { model.onHover($0) }
            .padding(.top, NotPastedPanel.topMargin)
            .padding(.bottom, NotPastedPanel.bottomMargin)
            .frame(
                width: NotPastedPanel.size.width, height: NotPastedPanel.size.height,
                alignment: .bottom)
    }

    private var lines: (title: String, detail: String) {
        switch model.notice {
        case .notPasted:
            return (
                "Nothing was focused, so it's saved here.",
                "It's on your clipboard too: click a text box and press ⌘V."
            )
        case .needsAccessibility:
            return (
                "Phonon needs Accessibility to paste.",
                "It's on your clipboard: press ⌘V, or turn Phonon on in Accessibility."
            )
        case .notHeard(let mic):
            return ("Phonon didn't hear anything.", "Listening on \(mic). Hold the key while you talk.")
        case .micSilent(let mic):
            return ("Your mic isn't sending any sound.", "\(mic) is silent. Check it isn't muted.")
        }
    }

    private var card: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let text = model.notice.text {
                Text(text)
                    .font(.system(size: LiveTextMetrics.fontSize))
                    .foregroundColor(LiveRGB.ink.color)
                    .lineLimit(4)
                    .lineSpacing(LiveTextMetrics.lineHeight - LiveTextMetrics.fontSize * 1.2)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack(spacing: 12) {
                if model.notice.text == nil {
                    Image(systemName: "mic.slash")
                        .font(.system(size: 18, weight: .medium))
                        .foregroundColor(EmberTheme.ember2)
                        .frame(width: 24)
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text(lines.title)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundColor(LiveRGB.ink.color.opacity(0.85))
                    Text(lines.detail)
                        .font(.system(size: 12))
                        .foregroundColor(LiveRGB.muted.color)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
                switch model.notice {
                case .notPasted(let text):
                    BigCopyButton(text: text)
                case .needsAccessibility(let text):
                    VStack(alignment: .trailing, spacing: 6) {
                        BigCopyButton(text: text)
                        Button("Open Accessibility") { Self.open(Self.accessibilityPane) }
                            .buttonStyle(NoticeLinkButtonStyle())
                    }
                case .notHeard, .micSilent:
                    Button("Sound settings") { Self.open(Self.soundInputPane) }
                        .buttonStyle(NoticeLinkButtonStyle())
                }
            }
        }
        // Room on the right for the close button, clear of the text and copy.
        .padding(.trailing, 14)
        .padding(.horizontal, LiveTextMetrics.paddingH + LiveTextMetrics.border + 2)
        .padding(.vertical, 12)
        .frame(width: LiveTextMetrics.cardWidth, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: LiveTextMetrics.cornerRadius, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [LiveRGB(0x1a120c).color, LiveRGB(0x0d0907).color],
                        startPoint: .top, endPoint: .bottom))
        )
        .overlay(
            RoundedRectangle(cornerRadius: LiveTextMetrics.cornerRadius, style: .continuous)
                .strokeBorder(LiveRGB(0x4a2a16).color, lineWidth: LiveTextMetrics.border)
        )
        .overlay(alignment: .topTrailing) {
            NoticeCloseButton { model.onClose() }
                .padding(.top, 6)
                .padding(.trailing, 6)
        }
    }

    static let accessibilityPane =
        "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
    static let soundInputPane = "x-apple.systempreferences:com.apple.Sound-Settings.extension?input"

    static func open(_ url: String) {
        if let url = URL(string: url) { NSWorkspace.shared.open(url) }
    }
}

/// The card's close button (issue #22): an x in the top-right corner.
struct NoticeCloseButton: View {
    let action: () -> Void
    @State private var hover = false

    var body: some View {
        Button(action: action) {
            Image(systemName: "xmark")
                .font(.system(size: 10, weight: .bold))
                .foregroundColor(LiveRGB.ink.color.opacity(hover ? 0.95 : 0.6))
                .frame(width: 22, height: 22)
                .background(Circle().fill(Color.white.opacity(hover ? 0.14 : 0.06)))
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .onHover { hover = $0 }
        .accessibilityLabel("Close")
        .help("Close")
    }
}

/// A quiet outlined button for the card's secondary action.
private struct NoticeLinkButtonStyle: ButtonStyle {
    @State private var hover = false

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 13, weight: .medium))
            .foregroundColor(LiveRGB.ink.color)
            .padding(.horizontal, 12)
            .frame(height: 30)
            .background(
                RoundedRectangle(cornerRadius: 9, style: .continuous)
                    .fill(Color.white.opacity(hover || configuration.isPressed ? 0.12 : 0.06)))
            .overlay(
                RoundedRectangle(cornerRadius: 9, style: .continuous)
                    .strokeBorder(LiveRGB(0x4a2a16).color, lineWidth: 1))
            .contentShape(Rectangle())
            .onHover { hover = $0 }
    }
}

/// A large labelled copy button: "Copy" with the icon, which springs into a green
/// "Copied" check, so it reads at a glance when a dictation did not land.
struct BigCopyButton: View {
    let text: String
    @State private var done = false
    @State private var hover = false
    @State private var resetTask: Task<Void, Never>?

    var body: some View {
        Button {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(text, forType: .string)
            withAnimation(.spring(response: 0.22, dampingFraction: 0.62)) { done = true }
            resetTask?.cancel()
            resetTask = Task { @MainActor in
                try? await Task.sleep(for: .milliseconds(1600))
                guard !Task.isCancelled else { return }
                withAnimation(.easeOut(duration: 0.2)) { done = false }
            }
        } label: {
            HStack(spacing: 8) {
                ZStack {
                    Image(systemName: "doc.on.doc")
                        .opacity(done ? 0 : 1)
                        .scaleEffect(done ? 0.4 : 1)
                    Image(systemName: "checkmark")
                        .font(.system(size: 14, weight: .bold))
                        .opacity(done ? 1 : 0)
                        .scaleEffect(done ? 1 : 0.4)
                        .rotationEffect(.degrees(done ? 0 : -20))
                }
                .frame(width: 18)
                Text(done ? "Copied" : "Copy")
                    .contentTransition(.opacity)
            }
            .font(.system(size: 15, weight: .semibold))
            .foregroundColor(done ? Color(hex: 0x0d2615) : .white)
            .padding(.horizontal, 18)
            .frame(height: 38)
            .frame(minWidth: 112)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(done ? EmberTheme.done : (hover ? EmberTheme.ember2 : EmberTheme.ember))
            )
            .scaleEffect(hover && !done ? 1.03 : 1)
            .animation(.easeOut(duration: 0.12), value: hover)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hover = $0 }
        .accessibilityLabel(done ? "Copied" : "Copy dictation")
    }
}

/// Borderless and non-activating, never key or main, like the meeting popup: clicking
/// copy leaves the user's app in front.
final class NotPastedPanel: NSPanel {
    static let topMargin: CGFloat = 8
    static let bottomMargin: CGFloat = 24
    static let size = NSSize(width: LiveTextMetrics.cardWidth + 48, height: 220)

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

    /// Bottom-anchored just above the capsule, where the meeting popup sits.
    static func frame(screen: NSRect) -> NSRect {
        NSRect(
            x: screen.midX - size.width / 2,
            y: screen.minY + PanelGeometry.bottomInset + LiveTextPanel.capsuleHeight
                + LiveTextPanel.gap + 2 - bottomMargin,
            width: size.width, height: size.height)
    }
}

@MainActor
final class NotPastedController {
    /// The card stays this long, and while the pointer is on it.
    static let seconds: TimeInterval = 8

    private var panel: NotPastedPanel?
    let model = NotPastedModel()
    private var expiry: DispatchWorkItem?
    private var generation = 0
    private var hovering = false

    init() {
        model.onHover = { [weak self] inside in
            guard let self else { return }
            self.hovering = inside
            if !inside { self.armExpiry() }
        }
        model.onClose = { [weak self] in
            guard let self else { return }
            self.hovering = false
            self.dismiss()
        }
    }

    /// What is showing, or nil once dismissed.
    private(set) var current: Notice?

    func show(_ text: String) { show(.notPasted(text)) }

    /// Shows `notice`; `stays` keeps it up until `dismiss` (the mic-silent warning lasts
    /// while the key is held).
    func show(_ notice: Notice, stays: Bool = false) {
        generation += 1
        current = notice
        model.notice = notice
        let panel = self.panel ?? makePanel()
        self.panel = panel
        panel.setFrame(NotPastedPanel.frame(screen: PillPanel.screenFrame()), display: false)
        if !panel.isVisible {
            model.visible = false
            panel.orderFrontRegardless()  // never makeKey: it must not take focus
        }
        DispatchQueue.main.async { [weak self] in self?.model.visible = true }
        if stays { expiry?.cancel() } else { armExpiry() }
    }

    func dismiss() {
        expiry?.cancel()
        expiry = nil
        current = nil
        generation += 1
        let current = generation
        model.visible = false
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.18) { [weak self] in
            guard let self, self.generation == current else { return }
            self.panel?.orderOut(nil)
        }
    }

    private func armExpiry() {
        expiry?.cancel()
        if case .micSilent = current { return }  // lasts until the key comes up
        let work = DispatchWorkItem { [weak self] in
            guard let self else { return }
            if self.hovering { return }  // leaving the card re-arms it
            self.dismiss()
        }
        expiry = work
        DispatchQueue.main.asyncAfter(deadline: .now() + Self.seconds, execute: work)
    }

    private func makePanel() -> NotPastedPanel {
        let panel = NotPastedPanel()
        let host = FirstMouseHostingView(rootView: NotPastedView(model: model))
        host.frame = NSRect(origin: .zero, size: NotPastedPanel.size)
        panel.contentView = host
        return panel
    }
}
