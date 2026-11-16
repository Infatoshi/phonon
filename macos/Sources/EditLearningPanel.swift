import AppKit
import SwiftUI
import os

/// An AX element handed to a background read. AX elements are plain CF
/// references that the AX API accepts from any thread.
struct AXElementRef: @unchecked Sendable {
    let element: AXUIElement
}

/// The field as it was just before Phonon pasted into it.
struct PrePasteSnapshot: @unchecked Sendable {
    let element: AXUIElement
    let pid: pid_t
    let value: String
    let selection: NSRange
}

/// Watches the field a dictation was pasted into, reports what the user kept
/// to the engine, and asks the user about the engine's proposals. Nothing it
/// reads is written to disk: the engine stores the record only in the
/// dictation's corpus entry and only when local history is on, and the
/// dictionary changes only when the user presses Add.
@MainActor
final class EditLearningController {
    /// The target app handles Command-V on its own schedule; look for the
    /// landed text at these offsets (seconds) before giving up.
    static let landingChecks: [TimeInterval] = [0.15, 0.4, 0.9, 1.6]
    static let pollInterval: TimeInterval = 0.4
    /// A key read is coalesced this long so a typing burst costs one read.
    static let keyReadDelay: TimeInterval = 0.08

    /// Shared with the event tap, which forwards key-downs only while a field
    /// is being watched.
    private let keyWatch: OSAllocatedUnfairLock<Bool>

    private let store: NativeAppStore
    private let send: ([String: Any]) -> Void
    private var watch: Watch?
    private var landingToken = 0
    private var lastCategory: KeystrokeCategory?
    private var keyReadPending = false
    private var filter = EditSuggestionFilter()
    private let panel = SuggestionPanelController()
    private var queue: [(suggestion: EditSuggestion, passID: String, category: String)] = []
    /// App category per pass, from the engine's polish result.
    private var categories: [String: String] = [:]
    private var prefetched: (passID: String, snapshot: PrePasteSnapshot)?
    private var readInFlight = false

    private struct Watch {
        let passID: String
        let element: AXUIElement
        var tracker: EditObservationTracker
        let record: Bool
        var timer: Timer?
    }

    init(
        store: NativeAppStore, keyWatch: OSAllocatedUnfairLock<Bool>,
        send: @escaping ([String: Any]) -> Void
    ) {
        self.store = store
        self.keyWatch = keyWatch
        self.send = send
    }

    private static func now() -> TimeInterval { ProcessInfo.processInfo.systemUptime }

    func noteCategory(_ category: String, passID: String) {
        categories[passID] = category
        if categories.count > 64 { categories.removeAll() }
    }

    /// Keep the field read at key-down (by the dictation-target task, bound to
    /// the dictation's app and checked against the private list) until the
    /// paste. Reading at key-down means it has long finished by the time the
    /// correction comes back, so no dictation goes unwatched for a slow read.
    func store(_ snapshot: PrePasteSnapshot?, passID: String) {
        prefetched = snapshot.map { (passID, $0) }
    }

    /// The field read for this pass, if any.
    func takeSnapshot(passID: String) -> PrePasteSnapshot? {
        defer { prefetched = nil }
        guard let prefetched, prefetched.passID == passID else { return nil }
        return prefetched.snapshot
    }

    func didPaste(_ snapshot: PrePasteSnapshot, passID: String, pasted: String, record: Bool) {
        end(reason: .nextDictation)
        landingToken += 1
        checkLanding(
            snapshot, passID: passID, pasted: pasted, record: record, attempt: 0,
            token: landingToken)
    }

    /// The next dictation (or quitting) ends any watch in progress.
    func interrupt(reason: ObservationEndReason) {
        landingToken += 1
        end(reason: reason)
    }

    private func checkLanding(
        _ snapshot: PrePasteSnapshot, passID: String, pasted: String, record: Bool,
        attempt: Int, token: Int
    ) {
        guard attempt < Self.landingChecks.count else { return }
        let delay = Self.landingChecks[attempt] - (attempt > 0 ? Self.landingChecks[attempt - 1] : 0)
        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self, self.landingToken == token, self.watch == nil else { return }
            Task { [weak self] in
                let read = await Self.readOffMain(snapshot.element)
                guard let self, self.landingToken == token, self.watch == nil else { return }
                self.landed(
                    read, snapshot: snapshot, passID: passID, pasted: pasted, record: record,
                    attempt: attempt, token: token)
            }
        }
    }

    /// Accessibility reads can take tens of milliseconds on a big Chrome or
    /// Electron field; they never run on the main thread.
    private static func readOffMain(_ element: AXUIElement) async -> (value: String?, gone: Bool) {
        let ref = AXElementRef(element: element)
        return await Task.detached(priority: .userInitiated) {
            TextboxAccess.readValue(ref.element)
        }.value
    }

    private func landed(
        _ read: (value: String?, gone: Bool), snapshot: PrePasteSnapshot, passID: String,
        pasted: String, record: Bool, attempt: Int, token: Int
    ) {
        if read.gone { return }
        let retry = {
            self.checkLanding(
                snapshot, passID: passID, pasted: pasted, record: record,
                attempt: attempt + 1, token: token)
        }
        guard let contents = read.value else {
            retry()
            return
        }
        switch EditObservationTracker.start(
            pasted: pasted, contentsBefore: snapshot.value,
            selectionBefore: snapshot.selection, contentsAfter: contents, now: Self.now())
        {
        case let .success(tracker):
            begin(Watch(
                passID: passID, element: snapshot.element, tracker: tracker,
                record: record, timer: nil))
        case .failure(.textboxTooLong):
            return
        case .failure:
            retry()
        }
    }

    private func begin(_ newWatch: Watch) {
        var started = newWatch
        started.timer = Timer.scheduledTimer(withTimeInterval: Self.pollInterval, repeats: true) {
            [weak self] _ in
            MainActor.assumeIsolated { self?.poll() }
        }
        watch = started
        lastCategory = nil
        keyWatch.withLock { $0 = true }
    }

    private func poll() {
        guard let current = watch, !readInFlight else { return }
        readInFlight = true
        let passID = current.passID
        Task { [weak self] in
            let read = await Self.readOffMain(current.element)
            guard let self else { return }
            self.readInFlight = false
            guard var watch = self.watch, watch.passID == passID else { return }
            if read.gone {
                self.end(reason: .textboxGone)
                return
            }
            let reason = watch.tracker.observe(contents: read.value, now: Self.now())
            self.watch = watch
            if let reason { self.end(reason: reason) }
        }
    }

    /// A key-down from the event tap, reduced to its category.
    func noteKey(keyCode: Int, command: Bool, option: Bool, shift: Bool) {
        guard watch != nil else { return }
        let category = KeystrokeCategory.classify(
            keyCode: keyCode, command: command, option: option, shift: shift,
            previous: lastCategory)
        lastCategory = category
        watch?.tracker.noteKeystroke(category, at: Self.now())
        guard !keyReadPending else { return }
        keyReadPending = true
        DispatchQueue.main.asyncAfter(deadline: .now() + Self.keyReadDelay) { [weak self] in
            self?.keyReadPending = false
            self?.poll()
        }
    }

    private func end(reason: ObservationEndReason) {
        guard let finished = watch else { return }
        finished.timer?.invalidate()
        watch = nil
        keyWatch.withLock { $0 = false }
        let result = finished.tracker.finish(reason: reason, now: Self.now())
        send(result.engineCommand(id: finished.passID, record: finished.record))
    }

    // MARK: Proposals

    func handleEngineResult(_ event: [String: Any]) {
        guard let passID = event["id"] as? String else { return }
        let category = event["app_category"] as? String ?? categories[passID] ?? "other"
        let approved = Set(store.settings.stylePreferences[category] ?? [])
        let admitted = filter.admit(
            EditSuggestion.parse(event), category: category, approvedStyles: approved)
        queue.append(contentsOf: admitted.map { ($0, passID, category) })
        showNext()
    }

    private func showNext() {
        guard !panel.isShowing, !queue.isEmpty else { return }
        let (suggestion, passID, category) = queue.removeFirst()
        panel.present(
            title: EditSuggestionCopy.title(suggestion, category: category),
            detail: EditSuggestionCopy.detail(suggestion),
            approveLabel: EditSuggestionCopy.approveLabel(suggestion),
            onApprove: { [weak self] in
                self?.approve(suggestion, passID: passID, category: category)
            },
            onReject: { [weak self] in
                guard let self else { return }
                self.filter.reject(suggestion)
                self.decide(suggestion, passID: passID, "rejected")
                self.panel.dismiss()
                self.showNext()
            },
            onExpire: { [weak self] in self?.showNext() })
    }

    private func approve(_ suggestion: EditSuggestion, passID: String, category: String) {
        decide(suggestion, passID: passID, "approved")
        switch suggestion.kind {
        case let .term(term, heardAs, _, _):
            let undo = store.approveLearnedTerm(term: term, heardAs: heardAs)
            panel.confirm(
                "Added \u{201C}\(term)\u{201D}",
                onUndo: undo.map { undo in
                    { [weak self] in
                        guard let self else { return }
                        self.store.undoLearnedTerm(undo)
                        self.decide(suggestion, passID: passID, "undone")
                        self.panel.dismiss()
                        self.showNext()
                    }
                },
                onExpire: { [weak self] in self?.showNext() })
        case let .style(preference):
            store.approveStylePreference(preference, category: category)
            panel.confirm(
                "Saved for \(EditSuggestionCopy.placeName(category))",
                onUndo: { [weak self] in
                    guard let self else { return }
                    self.store.removeStylePreference(preference, category: category)
                    self.decide(suggestion, passID: passID, "undone")
                    self.panel.dismiss()
                    self.showNext()
                },
                onExpire: { [weak self] in self?.showNext() })
        }
    }

    private func decide(_ suggestion: EditSuggestion, passID: String, _ decision: String) {
        send(["cmd": "edit_decision", "id": passID, "index": suggestion.index, "decision": decision])
    }
}

// MARK: - Panel

@MainActor
final class SuggestionPanelModel: ObservableObject {
    @Published var title = ""
    @Published var detail = ""
    @Published var approveLabel = "Add"
    @Published var confirmation: String?
    @Published var canUndo = false
    var onApprove: () -> Void = {}
    var onReject: () -> Void = {}
    var onUndo: () -> Void = {}
}

struct SuggestionCard: View {
    @ObservedObject var model: SuggestionPanelModel

    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(model.confirmation ?? model.title)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(.white)
                    .lineLimit(2)
                if model.confirmation == nil {
                    Text(model.detail)
                        .font(.system(size: 11))
                        .foregroundColor(.white.opacity(0.6))
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 8)
            if model.confirmation != nil {
                if model.canUndo {
                    pill("Undo", prominent: false, action: model.onUndo)
                }
            } else {
                pill("Not now", prominent: false, action: model.onReject)
                pill(model.approveLabel, prominent: true, action: model.onApprove)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .frame(width: SuggestionPanelGeometry.size.width, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color.black.opacity(0.94)))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(Color.white.opacity(0.14), lineWidth: 1))
    }

    private func pill(_ label: String, prominent: Bool, action: @escaping () -> Void)
        -> some View
    {
        Button(action: action) {
            Text(label)
                .font(.system(size: 12, weight: .semibold))
                .foregroundColor(prominent ? .black : .white)
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(
                    Capsule().fill(prominent ? Color.white : Color.white.opacity(0.12)))
        }
        .buttonStyle(.plain)
    }
}

/// A floating, non-activating card above the dictation capsule. It never
/// takes focus, so the user's typing keeps going to their own app.
final class SuggestionPanel: NSPanel {
    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }

    init() {
        super.init(
            contentRect: NSRect(origin: .zero, size: SuggestionPanelGeometry.size),
            styleMask: [.borderless, .nonactivatingPanel, .fullSizeContentView],
            backing: .buffered, defer: true)
        isFloatingPanel = true
        level = .floating
        hidesOnDeactivate = false
        collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        backgroundColor = .clear
        isOpaque = false
        hasShadow = true
    }
}

@MainActor
final class SuggestionPanelController {
    /// A question stays this long, a confirmation with Undo this long.
    static let questionSeconds: TimeInterval = 12
    static let confirmationSeconds: TimeInterval = 6

    private var panel: SuggestionPanel?
    private let model = SuggestionPanelModel()
    private var expiry: DispatchWorkItem?
    private(set) var isShowing = false
    /// Points above the default spot, so two cards never overlap.
    private let raise: CGFloat

    init(raisedBy raise: CGFloat = 0) {
        self.raise = raise
    }

    func present(
        title: String, detail: String, approveLabel: String,
        onApprove: @escaping () -> Void, onReject: @escaping () -> Void,
        onExpire: @escaping () -> Void
    ) {
        model.title = title
        model.detail = detail
        model.approveLabel = approveLabel
        model.confirmation = nil
        model.canUndo = false
        model.onApprove = onApprove
        model.onReject = onReject
        show(for: Self.questionSeconds, onExpire: onExpire)
    }

    func confirm(_ text: String, onUndo: (() -> Void)?, onExpire: @escaping () -> Void) {
        model.confirmation = text
        model.canUndo = onUndo != nil
        model.onUndo = onUndo ?? {}
        show(for: Self.confirmationSeconds, onExpire: onExpire)
    }

    func dismiss() {
        expiry?.cancel()
        expiry = nil
        panel?.orderOut(nil)
        isShowing = false
    }

    private func show(for seconds: TimeInterval, onExpire: @escaping () -> Void) {
        let panel = self.panel ?? makePanel()
        self.panel = panel
        let raise = self.raise
        let fit = {
            panel.contentView?.layoutSubtreeIfNeeded()
            let fitting = panel.contentView?.fittingSize.height
            panel.setFrame(
                SuggestionPanelGeometry.frame(
                    screen: PillPanel.screenFrame(), height: fitting, raisedBy: raise),
                display: true)
        }
        fit()
        panel.orderFrontRegardless()
        // SwiftUI applies the new text on the next pass; fit again after it.
        DispatchQueue.main.async { fit() }
        isShowing = true
        expiry?.cancel()
        let work = DispatchWorkItem { [weak self] in
            self?.dismiss()
            onExpire()
        }
        expiry = work
        DispatchQueue.main.asyncAfter(deadline: .now() + seconds, execute: work)
    }

    private func makePanel() -> SuggestionPanel {
        let panel = SuggestionPanel()
        let host = NSHostingView(rootView: SuggestionCard(model: model))
        host.frame = NSRect(origin: .zero, size: SuggestionPanelGeometry.size)
        panel.contentView = host
        return panel
    }
}
