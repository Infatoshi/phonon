import AppKit
import Foundation

/// Watches for calls and, with the user's yes, records meeting notes audio. What a
/// detected call does follows Settings › Notes (`meeting_mode`, MeetingRules.swift):
/// remember per app (default: an "always" rule records without a click, "never" stays
/// silent, an app without a rule asks whether to remember), ask every time, or no detection. The note taker can also be started by hand at any time.
@MainActor
final class MeetingController {
    nonisolated static let detectionInterval: TimeInterval = 4

    private let store: NativeAppStore
    private var timer: Timer?
    private var prompts = MeetingPromptPolicy()
    private var endPolicy: MeetingEndPolicy?
    private var recorder: MeetingRecorder?
    /// The folder of the meeting being recorded.
    private var recordingDirectory: URL?
    private var recordingApp: String?
    /// The process whose microphone use defines the call being recorded.
    private var recordingBundleID: String?
    /// The logo-only popup just above the capsule, and "Saved to Notes".
    private let popup = MeetingPopupController()
    private var checking = false
    /// What the popup and the capsule show (MeetingPopup.swift).
    private(set) var flow = MeetingFlow() {
        didSet { if flow != oldValue { onFlowChanged?(flow) } }
    }
    /// Tells the menu bar item whether a meeting is being recorded.
    var onRecordingChanged: ((String?) -> Void)?
    /// Tells the capsule: recording light, stop button, pointer.
    var onFlowChanged: ((MeetingFlow) -> Void)?

    init(store: NativeAppStore) {
        self.store = store
    }

    var isRecording: Bool { recorder != nil }

    /// Host time a dictation's capture started, while one is under way.
    private var dictationStartNs: UInt64?

    /// A dictation's capture started (host clock, ns since boot). Its stretch of the mic track
    /// is left out of the meeting's "me" transcript (GOAL N9).
    func dictationStarted(atNs ns: UInt64 = HostClock.now()) {
        dictationStartNs = ns
    }

    /// The capture ended (committed, discarded or failed to start).
    func dictationEnded(atNs ns: UInt64 = HostClock.now()) {
        guard let start = dictationStartNs else { return }
        dictationStartNs = nil
        recorder?.noteDictation(MeetingDictation(startHostNs: start, endHostNs: max(start, ns)))
    }

    /// Start or stop watching to match the setting.
    func refresh() {
        if store.settings.meetingCapture {
            guard timer == nil else { return }
            timer = Timer.scheduledTimer(withTimeInterval: Self.detectionInterval, repeats: true) {
                [weak self] _ in
                MainActor.assumeIsolated { self?.check() }
            }
        } else {
            timer?.invalidate()
            timer = nil
        }
    }

    private static func now() -> TimeInterval { ProcessInfo.processInfo.systemUptime }

    private func check() {
        guard !checking else { return }
        checking = true
        let denylist = store.settings.screenContextDenylist
        let bundleID = NSWorkspace.shared.frontmostApplication?.bundleIdentifier
        Task { [weak self] in
            let evidence = await Task.detached(priority: .utility) {
                let audio = MicrophoneHolders.snapshot()
                let holders = audio.input
                // The frontmost tab's URL is read only when the frontmost app
                // is a browser, not private, and itself holds the microphone.
                // Only the focused element's ancestors are walked, to the
                // top-level page (never an iframe's), and only http(s) counts.
                var url: URL?
                if MeetingDetector.shouldReadTabURL(
                    holders: holders, frontmostBundleID: bundleID, denylist: denylist)
                {
                    url = BrowserPage.focusedPage().url
                }
                return MeetingDetector.Evidence(
                    microphoneHolders: holders, frontmostBundleID: bundleID, frontmostURL: url,
                    denylist: denylist, audioOutputs: audio.output)
            }.value
            guard let self else { return }
            self.checking = false
            self.handle(MeetingDetector.evaluate(evidence), evidence: evidence)
        }
    }

    private func handle(_ verdict: MeetingDetector.Verdict, evidence: MeetingDetector.Evidence) {
        let now = Self.now()
        // A call asked about stays asked about until its app lets go of the mic for good.
        prompts.observe(holders: evidence.microphoneHolders, now: now)
        if let recordingBundleID, var policy = endPolicy {
            // Still in the call while that process holds the microphone,
            // whatever app is in front now.
            let inCall = MeetingDetector.stillInCall(
                recordingBundleID: recordingBundleID, holders: evidence.microphoneHolders)
            let over = policy.observe(stillInCall: inCall, now: now)
            endPolicy = policy
            if over { stop(reason: "call_ended") }
            return
        }
        guard recorder == nil, case let .meeting(meeting, bundleID) = verdict else { return }
        let settings = store.settings
        let key = MeetingRules.key(app: meeting, bundleID: bundleID)
        let decision = MeetingRules.decide(
            mode: settings.meetingMode, rule: settings.meetingRules.rule(for: key))
        // Asked once per call; a call already recorded resumes (as a new meeting) on its own.
        switch prompts.action(meeting: meeting, bundleID: bundleID, decision: decision, now: now) {
        case .ignore:
            return
        case .record:
            start(meeting: meeting, bundleID: bundleID)
        case .ask:
            ask(meeting: meeting, bundleID: bundleID, remember: settings.meetingMode == .perApp)
        }
    }

    /// The question above the capsule: the per-app question under "Remember per app" (the
    /// default), or Record / Not now under "Ask every time".
    private func ask(meeting: String, bundleID: String, remember: Bool) {
        // The one-time questions give way to a real call.
        if flow.answeredQuestions() { popup.dismiss() }
        flow.ask(app: meeting, bundleID: bundleID, remember: remember)
        let icon = MeetingAppIcon.image(app: meeting, bundleID: bundleID)
        if remember {
            popup.askRule(
                icon: icon, name: MeetingRules.displayName(app: meeting, bundleID: bundleID),
                onChoose: { [weak self] choice in self?.choose(choice) },
                onExpire: { [weak self] in self?.flow.expire() })
        } else {
            popup.ask(
                icon: icon,
                onNotNow: { [weak self] in self?.notNow() },
                onRecord: { [weak self] in self?.record() },
                onExpire: { [weak self] in self?.flow.expire() })
        }
    }

    /// "Not now": not again for this app for a while (MeetingPromptPolicy).
    func notNow() {
        guard let meeting = flow.notNow() else { return }
        prompts.declined(meeting, now: Self.now())
        popup.dismiss()
    }

    /// An answer to "Record this and future <App> calls automatically?". Always and Never
    /// store the rule and show Undo; Just this once records without remembering.
    func choose(_ choice: MeetingChoice) {
        if choice == .notNow {
            notNow()
            return
        }
        guard let call = flow.choose(choice) else { return }
        popup.dismiss()
        var change: MeetingRuleChange?
        if let rule = choice.rule {
            change = store.setMeetingRule(rule, for: MeetingRules.key(app: call.app, bundleID: call.bundleID))
        }
        if choice.records { start(meeting: call.app, bundleID: call.bundleID) }
        guard let change else { return }
        flow.offerUndo(app: call.app, bundleID: call.bundleID, change: change, recorded: choice.records)
        let name = MeetingRules.displayName(app: call.app, bundleID: call.bundleID)
        popup.undo(
            text: choice == .always
                ? "\(name) calls will be recorded automatically."
                : "Phonon won't ask about \(name) calls again.",
            onUndo: { [weak self] in self?.undo() },
            onExpire: { [weak self] in self?.flow.expire() })
    }

    /// Undo on the card: the rule goes back to what it was, a recording that answer
    /// started (a few seconds) is stopped and deleted, and the question is up again.
    func undo() {
        guard let undone = flow.undo() else { return }
        store.undoMeetingRule(undone.change)
        // Not recorded again on its own in this call unless the question is answered again.
        holdOff(meeting: undone.app, bundleID: undone.bundleID)
        if undone.recorded, recorder != nil, !recordingManual {
            stop(reason: "undone", discard: true)
            // `stop` hid the popup; the question comes back for the same call.
            flow.ask(app: undone.app, bundleID: undone.bundleID, remember: true)
        }
        popup.askRule(
            icon: MeetingAppIcon.image(app: undone.app, bundleID: undone.bundleID),
            name: MeetingRules.displayName(app: undone.app, bundleID: undone.bundleID),
            onChoose: { [weak self] choice in self?.choose(choice) },
            onExpire: { [weak self] in self?.flow.expire() })
    }

    /// The one-time meeting questions for an install that predates them, on the same
    /// non-activating card (it never takes the keyboard). Only while nothing else is up.
    func offerQuestions() {
        guard !store.settings.meetingQuestionsAsked, store.settings.privacyChoiceMade,
            flow.askQuestions()
        else { return }
        popup.questions(
            onAnswer: { [weak self] mode in self?.answerQuestions(mode) },
            onExpire: { [weak self] in
                // Unanswered counts as skipped: asked once, the default kept.
                guard let self, self.flow.answeredQuestions() else { return }
                self.store.recordMeetingQuestions(mode: nil)
            })
    }

    private func answerQuestions(_ mode: MeetingMode?) {
        guard flow.answeredQuestions() else { return }
        popup.dismiss()
        store.recordMeetingQuestions(mode: mode)
        onSettingsChanged?()
    }

    /// Asks the app to re-apply settings (the mode changed from a card).
    var onSettingsChanged: (() -> Void)?

    /// Start the note taker with nothing detected (the capsule's start button, the Notes
    /// page, the ϕ menu). It records the mic and this Mac's audio until stopped by hand,
    /// whatever the detection setting.
    func startManually() {
        guard recorder == nil else { return }
        if case .asking = flow.popup { popup.dismiss() }
        if case .askingRule = flow.popup { popup.dismiss() }
        if flow.answeredQuestions() { popup.dismiss() }
        start(meeting: Self.manualName, bundleID: nil)
    }

    nonisolated static let manualName = "Note taker"

    /// Settings changed: turning detection off stops a detected call's recording, never
    /// one started by hand.
    func settingsChanged() {
        if !store.settings.meetingCapture, recorder != nil, !recordingManual {
            stop(reason: "turned_off")
        }
        refresh()
    }

    /// "Record" on the popup.
    func record() {
        guard let call = flow.record() else { return }
        popup.dismiss()
        start(meeting: call.app, bundleID: call.bundleID)
    }

    /// The pointer entered or left the capsule.
    func hover(_ inside: Bool) {
        flow.hover(inside)
    }

    /// Whether the recording under way was started by hand (no call to follow).
    private(set) var recordingManual = false

    /// The capsule's stop button or the menu's Stop item: the same stop as
    /// before, then "Saved to Notes" above the capsule.
    func stopFromUser() {
        guard recorder != nil else { return }
        // Stopped by the user: not restarted on its own in this call.
        if let recordingApp { holdOff(meeting: recordingApp, bundleID: recordingBundleID) }
        stop(reason: "user_stopped")
    }

    /// Record the call now. The metadata names the app holding the
    /// microphone (not whatever is in front); its window title and, for a
    /// browser, the conference URL are read off the main actor and filled in
    /// when known, so recording never waits on accessibility.
    /// `bundleID` nil is a start by hand: no call app, nothing read from any window, and
    /// it runs until stopped (no call to end it).
    private func start(meeting: String, bundleID: String?) {
        guard recorder == nil, bundleID == nil || store.settings.meetingCapture else { return }
        let app = bundleID.flatMap {
            NSRunningApplication.runningApplications(withBundleIdentifier: $0).first
        }
        let pid = app?.processIdentifier
        let appName = app?.localizedName ?? meeting
        let micDevice = CoreAudioInputDevices.resolve(
            priorities: store.settings.microphonePriority)?.name
        let denylist = store.settings.screenContextDenylist
        let id = "meeting_\(Int(Date().timeIntervalSince1970 * 1_000))"
        let directory = PhononDataPaths.supportDirectory()
            .appendingPathComponent("Meetings", isDirectory: true)
            .appendingPathComponent(id, isDirectory: true)
        let metadata = MeetingMetadata(
            id: id,
            startedAt: ISO8601DateFormatter().string(from: Date()),
            startedHostNs: DispatchTime.now().uptimeNanoseconds,
            appBundleID: bundleID,
            appName: appName,
            windowTitle: nil,
            conferenceURL: nil,
            micDevice: micDevice)
        if let bundleID { prompts.accepted(meeting: meeting, bundleID: bundleID, now: Self.now()) }
        begin(meeting: meeting, bundleID: bundleID, directory: directory, metadata: metadata)
        guard let recorder, let bundleID else { return }
        Task.detached(priority: .userInitiated) {
            // Read at the start: title and page of the window holding the
            // call app's focus, dropped together if either shows a private
            // site or (for a browser) the page is unknown.
            let info = MeetingWindowInfo.read(bundleID: bundleID, pid: pid, denylist: denylist)
            recorder.setWindowInfo(title: info.title, conferenceURL: info.conferenceURL)
        }
    }

    private func begin(
        meeting: String, bundleID: String?, directory: URL, metadata: MeetingMetadata
    ) {
        do {
            let recorder = try MeetingRecorder(directory: directory, metadata: metadata)
            recorder.onWriteFailed = { [weak self] error in
                Task { @MainActor in self?.writeFailed(recorder, meeting: meeting, bundleID: bundleID, error: error) }
            }
            self.recorder = recorder
            recordingDirectory = directory
            store.recordingMeetingID = metadata.id
            recordingApp = meeting
            recordingBundleID = bundleID
            recordingManual = bundleID == nil
            // A start by hand has no call whose end stops it.
            endPolicy = bundleID == nil ? nil : MeetingEndPolicy(now: Self.now())
            flow.started(app: meeting)
            onRecordingChanged?(meeting)
            Task { [weak self] in
                do {
                    try await recorder.start()
                } catch {
                    NSLog("phonon meeting: could not start: \(error.localizedDescription)")
                    // Close this recorder's files whatever happened since;
                    // stop the controller only if it is still this recording
                    // (a newer one may have started after a quick stop).
                    await recorder.stop(reason: "start_failed")
                    guard let self else { return }
                    self.holdOff(meeting: meeting, bundleID: bundleID)
                    guard Self.isSame(self.recorder, recorder) else { return }
                    self.stop(reason: "start_failed")
                }
            }
        } catch {
            NSLog("phonon meeting: \(error.localizedDescription)")
            holdOff(meeting: meeting, bundleID: bundleID)
        }
    }

    /// A write failed (a full disk): stop cleanly, keep what was written (it is transcribed like
    /// any stopped meeting), do not start again in this call, and tell the user.
    private func writeFailed(_ failed: MeetingRecorder, meeting: String, bundleID: String?, error: Error) {
        guard Self.isSame(recorder, failed) else { return }
        holdOff(meeting: meeting, bundleID: bundleID)
        stop(reason: "write_failed")
        let message = MeetingWriteFailure.message(for: error)
        onRecordingFailed?(message.title, message.detail)
    }

    /// Shows the user why a recording stopped (title, detail).
    var onRecordingFailed: ((String, String) -> Void)?

    /// A recording that could not start is not retried on every poll of the same call.
    private func holdOff(meeting: String, bundleID: String?) {
        guard let bundleID else { return }
        prompts.holdOff(meeting: meeting, bundleID: bundleID, now: Self.now())
    }

    /// Whether `current` is the very recorder `candidate` (identity, not
    /// equality).
    nonisolated static func isSame(_ current: AnyObject?, _ candidate: AnyObject) -> Bool {
        current.map { $0 === candidate } ?? false
    }

    /// `discard` (Undo of the "Always" that started it): the few seconds recorded are
    /// deleted instead of transcribed.
    func stop(reason: String, discard: Bool = false) {
        guard let recorder else { return }
        recordingManual = false
        // A dictation still under way is cut at the stop, so it is in the final metadata.
        if let start = dictationStartNs {
            recorder.noteDictation(MeetingDictation(startHostNs: start, endHostNs: max(start, HostClock.now())))
            dictationStartNs = HostClock.now()
        }
        self.recorder = nil
        let directory = recordingDirectory
        recordingDirectory = nil
        // Before any async work: a quit while the files close still leaves the meeting marked
        // for transcription at the next launch (the marker goes when transcript.json is written).
        if let directory, !discard { MeetingTranscriber.markPending(directory: directory) }
        // `store.recordingMeetingID` stays set until the files are closed (no Transcribe button
        // against metadata still missing its last chunks).
        endPolicy = nil
        recordingApp = nil
        recordingBundleID = nil
        let byUser = reason == "user_stopped"
        flow.stopped(byUser: byUser)
        onRecordingChanged?(nil)
        if byUser {
            popup.saved(onExpire: { [weak self] in self?.flow.expire() })
        }
        let store = store
        Task {
            await recorder.stop(reason: reason)
            guard let directory else { return }
            if discard {
                try? FileManager.default.removeItem(at: directory)
                store.meetingRecordingDiscarded(directory: directory)
                return
            }
            // The chunks are closed and metadata.json final: queue it and show it on Notes.
            store.meetingRecordingFinished(directory: directory)
        }
    }

    /// On quit: close the open chunks synchronously so they stay readable, and mark the
    /// meeting for transcription at the next launch.
    func stopNow(reason: String) {
        timer?.invalidate()
        timer = nil
        guard let recorder else { return }
        self.recorder = nil
        if let start = dictationStartNs {
            recorder.noteDictation(MeetingDictation(startHostNs: start, endHostNs: max(start, HostClock.now())))
        }
        recorder.closeFiles(reason: reason)
        if let recordingDirectory { MeetingTranscriber.markPending(directory: recordingDirectory) }
        recordingDirectory = nil
    }
}
