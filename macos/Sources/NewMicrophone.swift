import AppKit
import SwiftUI

// The new-microphone prompt (SPEC "Microphone"). When an input Phonon has not
// seen before appears (a USB mic plugged in, AirPods connected), a small card
// above the capsule asks whether to rank it first, below the current first, or
// not at all. The answer goes into `microphone_priority`, the same list the
// Settings card edits. Each device (CoreAudio UID) is asked about once.

enum MicrophonePlacement: Equatable {
    /// Top of the priority list, above the current first.
    case first
    /// Below the current first: used only while that one is missing.
    case below
    /// Leave the list alone. The device stays seen and is not asked about again.
    case ignore
}

/// One question, with everything the card says. Built from a snapshot of the
/// devices and the ranking; the answer is applied to the ranking as it is then.
struct NewMicrophoneQuestion: Equatable {
    let uid: String
    let name: String
    let isBluetooth: Bool
    /// What the new device would go above or below: the first ranked name, or
    /// with nothing ranked, the microphone Phonon would use without it.
    let current: String?
    /// The ranked name that "Use first" pushes off a full list.
    let dropped: String?
    let offersBelow: Bool

    /// The prominent button. A Bluetooth headset defaults below the built-in
    /// mic: opening its mic drops the headset to call-quality audio.
    var recommended: MicrophonePlacement {
        isBluetooth && offersBelow ? .below : .first
    }

    var title: String { "New microphone: \(name). Use it for dictation?" }

    var detail: String {
        var parts: [String] = []
        if isBluetooth {
            parts.append(
                "Bluetooth headset: while you dictate through its mic, its sound drops to call quality.")
            if let current, offersBelow {
                parts.append("Add below keeps \(current) first.")
            }
        } else if let current {
            parts.append(
                offersBelow
                    ? "Use first ranks it above \(current); Add below uses it only when \(current) is missing."
                    : "Use first ranks it above \(current).")
        } else {
            parts.append("Phonon will use it whenever it is connected.")
        }
        if let dropped {
            parts.append(
                "\(dropped) leaves the list (\(NewMicrophonePolicy.maximumRanked) at most).")
        }
        return parts.joined(separator: " ")
    }
}

enum NewMicrophonePolicy {
    /// Shared with the Settings card: a headset, a desk mic and the built-in one.
    static let maximumRanked = 3

    struct Scan: Equatable {
        /// The seen set to store.
        let seen: [String]
        /// Devices to ask about, in CoreAudio order.
        let newcomers: [AudioInputDevice]
    }

    /// Compare what is connected with what has been seen. With no seen set yet
    /// (first launch with this feature) everything present is recorded and
    /// nothing is asked. Virtual and aggregate inputs, and devices a ranked name
    /// already matches, become seen without a question.
    static func scan(devices: [AudioInputDevice], seen: [String]?, priorities: [String]) -> Scan {
        let present = devices.filter { !$0.uid.isEmpty }
        guard let seen else {
            var seeded: [String] = []
            for device in present where !seeded.contains(device.uid) { seeded.append(device.uid) }
            return Scan(seen: seeded, newcomers: [])
        }
        var updated = seen
        var newcomers: [AudioInputDevice] = []
        for device in present where !updated.contains(device.uid) {
            updated.append(device.uid)
            if device.isVirtual || isRanked(device.name, priorities: priorities) { continue }
            newcomers.append(device)
        }
        return Scan(seen: updated, newcomers: newcomers)
    }

    /// Same match as `MicrophonePriorityResolver`: a ranked name is a substring.
    static func isRanked(_ name: String, priorities: [String]) -> Bool {
        priorities.contains { name.localizedCaseInsensitiveContains($0) }
    }

    static func question(
        for device: AudioInputDevice, devices: [AudioInputDevice], priorities: [String]
    ) -> NewMicrophoneQuestion {
        let others = devices.filter { $0.uid != device.uid && !$0.isVirtual }
        let current =
            priorities.first
            ?? MicrophonePriorityResolver.resolve(devices: others, priorities: [])?.name
        let full = priorities.count >= maximumRanked
        return NewMicrophoneQuestion(
            uid: device.uid, name: device.name, isBluetooth: device.isBluetooth,
            current: current, dropped: full ? priorities.last : nil,
            offersBelow: current != nil && !full)
    }

    /// The ranking after the answer. "Use first" with nothing ranked ranks only
    /// the new device, so Phonon still follows the system input while it is
    /// away. "Add below" with nothing ranked ranks `current` above it, since
    /// below nothing would make it first.
    static func apply(
        _ placement: MicrophonePlacement, name: String, current: String?, priorities: [String]
    ) -> [String] {
        let same = { (entry: String) in entry.localizedCaseInsensitiveCompare(name) == .orderedSame }
        switch placement {
        case .ignore:
            return priorities
        case .first:
            var list = priorities.filter { !same($0) }
            list.insert(name, at: 0)
            return Array(list.prefix(maximumRanked))
        case .below:
            if priorities.contains(where: same) { return priorities }
            if priorities.isEmpty {
                guard let current, !same(current) else { return [name] }
                return [current, name]
            }
            guard priorities.count < maximumRanked else { return priorities }
            return priorities + [name]
        }
    }
}

// MARK: - Controller

@MainActor
final class NewMicrophoneController {
    /// CoreAudio announces a device in several steps (the device, then its
    /// streams); scan once the burst is over.
    static let settle: TimeInterval = 1.5
    static let busyRetry: TimeInterval = 1
    static let questionSeconds: TimeInterval = 20

    private let store: NativeAppStore
    private let devices: () -> [AudioInputDevice]
    /// Recording or inserting: the card waits so it never lands mid-sentence.
    var isBusy: () -> Bool = { false }
    /// The ranking changed; the recorder should re-resolve.
    var onRankingChanged: () -> Void = {}
    private let panel = MicrophonePromptPanelController()
    private var queue: [AudioInputDevice] = []
    private var pendingScan: DispatchWorkItem?
    private var pendingPresent: DispatchWorkItem?
    private var asking = false
    private var started = false

    init(store: NativeAppStore, devices: @escaping () -> [AudioInputDevice] = CoreAudioInputDevices.all) {
        self.store = store
        self.devices = devices
    }

    func start() {
        guard !started else { return }
        started = true
        CoreAudioInputDevices.observeDeviceListChanges { [weak self] in
            MainActor.assumeIsolated { self?.scheduleScan() }
        }
        scan()
    }

    private func scheduleScan() {
        pendingScan?.cancel()
        let work = DispatchWorkItem { [weak self] in
            MainActor.assumeIsolated { self?.scan() }
        }
        pendingScan = work
        DispatchQueue.main.asyncAfter(deadline: .now() + Self.settle, execute: work)
    }

    func scan() {
        let result = NewMicrophonePolicy.scan(
            devices: devices(), seen: store.settings.microphonesSeen,
            priorities: store.settings.microphonePriority)
        if result.seen != store.settings.microphonesSeen {
            store.updateSettings { $0.microphonesSeen = result.seen }
        }
        for device in result.newcomers where !queue.contains(where: { $0.uid == device.uid }) {
            NSLog("phonon mic: new microphone \(device.name)")
            queue.append(device)
        }
        presentNext()
    }

    private func presentNext() {
        guard !asking, !queue.isEmpty else { return }
        if isBusy() {
            pendingPresent?.cancel()
            let work = DispatchWorkItem { [weak self] in
                MainActor.assumeIsolated { self?.presentNext() }
            }
            pendingPresent = work
            DispatchQueue.main.asyncAfter(deadline: .now() + Self.busyRetry, execute: work)
            return
        }
        let present = devices()
        // A device unplugged while it waited is not asked about.
        queue.removeAll { queued in !present.contains { $0.uid == queued.uid } }
        guard !queue.isEmpty else { return }
        let device = queue.removeFirst()
        let question = NewMicrophonePolicy.question(
            for: device, devices: present, priorities: store.settings.microphonePriority)
        asking = true
        panel.present(
            question, seconds: Self.questionSeconds,
            onAnswer: { [weak self] placement in self?.answer(placement, to: question) },
            onExpire: { [weak self] in
                guard let self else { return }
                NSLog("phonon mic: no answer for \(question.name); left unranked")
                self.asking = false
                self.presentNext()
            })
    }

    private func answer(_ placement: MicrophonePlacement, to question: NewMicrophoneQuestion) {
        panel.dismiss()
        asking = false
        let before = store.settings.microphonePriority
        let after = NewMicrophonePolicy.apply(
            placement, name: question.name, current: question.current, priorities: before)
        NSLog("phonon mic: \(question.name) answered \(placement), ranking \(after)")
        if after != before {
            store.updateSettings { $0.microphonePriority = after }
            store.refreshAvailableMicrophones()
            onRankingChanged()
        }
        presentNext()
    }
}

// MARK: - Panel

@MainActor
final class MicrophonePromptModel: ObservableObject {
    @Published var question: NewMicrophoneQuestion?
    var onAnswer: (MicrophonePlacement) -> Void = { _ in }
}

struct MicrophonePromptCard: View {
    @ObservedObject var model: MicrophonePromptModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let question = model.question {
                VStack(alignment: .leading, spacing: 2) {
                    Text(question.title)
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundColor(.white)
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                    Text(question.detail)
                        .font(.system(size: 11))
                        .foregroundColor(.white.opacity(0.6))
                        .lineLimit(3)
                        .fixedSize(horizontal: false, vertical: true)
                }
                HStack(spacing: 6) {
                    Spacer(minLength: 0)
                    pill("Ignore", prominent: false) { model.onAnswer(.ignore) }
                    if question.offersBelow {
                        pill("Add below", prominent: question.recommended == .below) {
                            model.onAnswer(.below)
                        }
                    }
                    pill("Use first", prominent: question.recommended == .first) {
                        model.onAnswer(.first)
                    }
                }
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

/// The same floating, non-activating panel as the edit-learning card, raised
/// above that card's spot so the two never overlap.
@MainActor
final class MicrophonePromptPanelController {
    static let raise: CGFloat = SuggestionPanelGeometry.size.height + 12

    private var panel: SuggestionPanel?
    private let model = MicrophonePromptModel()
    private var expiry: DispatchWorkItem?

    func present(
        _ question: NewMicrophoneQuestion, seconds: TimeInterval,
        onAnswer: @escaping (MicrophonePlacement) -> Void, onExpire: @escaping () -> Void
    ) {
        model.question = question
        model.onAnswer = onAnswer
        let panel = self.panel ?? makePanel()
        self.panel = panel
        let fit = {
            panel.contentView?.layoutSubtreeIfNeeded()
            let fitting = panel.contentView?.fittingSize.height
            panel.setFrame(
                SuggestionPanelGeometry.frame(
                    screen: PillPanel.screenFrame(), height: fitting, raisedBy: Self.raise),
                display: true)
        }
        fit()
        panel.orderFrontRegardless()  // never makeKey: it must not take focus
        DispatchQueue.main.async { fit() }
        expiry?.cancel()
        let work = DispatchWorkItem { [weak self] in
            MainActor.assumeIsolated {
                self?.dismiss()
                onExpire()
            }
        }
        expiry = work
        DispatchQueue.main.asyncAfter(deadline: .now() + seconds, execute: work)
    }

    func dismiss() {
        expiry?.cancel()
        expiry = nil
        panel?.orderOut(nil)
        model.onAnswer = { _ in }
    }

    private func makePanel() -> SuggestionPanel {
        let panel = SuggestionPanel()
        let host = FirstMouseHostingView(rootView: MicrophonePromptCard(model: model))
        host.frame = NSRect(origin: .zero, size: SuggestionPanelGeometry.size)
        panel.contentView = host
        return panel
    }
}
