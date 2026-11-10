import AppKit
import Foundation

/// Watches for calls and, with the user's yes, records meeting notes audio.
/// Off unless Settings > Meetings is on; never records without a click unless
/// the user also turned on auto-start.
@MainActor
final class MeetingController {
    static let detectionInterval: TimeInterval = 4

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
                let holders = MicrophoneHolders.bundleIDs()
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
                    denylist: denylist)
            }.value
            guard let self else { return }
            self.checking = false
            self.handle(MeetingDetector.evaluate(evidence), evidence: evidence)
        }
    }

    private func handle(_ verdict: MeetingDetector.Verdict, evidence: MeetingDetector.Evidence) {
        let now = Self.now()
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
        guard recorder == nil, case let .meeting(meeting, bundleID) = verdict,
            prompts.shouldAsk(meeting: meeting, now: now)
        else {
            if case .none = verdict { _ = prompts.shouldAsk(meeting: nil, now: now) }
            return
        }
        if store.settings.meetingAutoStart {
            start(meeting: meeting, bundleID: bundleID)
            return
        }
        flow.ask(app: meeting, bundleID: bundleID)
        popup.ask(
            icon: MeetingAppIcon.image(app: meeting, bundleID: bundleID),
            onNotNow: { [weak self] in self?.notNow() },
            onRecord: { [weak self] in self?.record() },
            onExpire: { [weak self] in self?.flow.expire() })
    }

    /// "Not now": not again for this app for a while (MeetingPromptPolicy).
    func notNow() {
        guard let meeting = flow.notNow() else { return }
        prompts.declined(meeting, now: Self.now())
        popup.dismiss()
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

    /// The capsule's stop button or the menu's Stop item: the same stop as
    /// before, then "Saved to Notes" above the capsule.
    func stopFromUser() {
        guard recorder != nil else { return }
        stop(reason: "user_stopped")
    }

    /// Record the call now. The metadata names the app holding the
    /// microphone (not whatever is in front); its window title and, for a
    /// browser, the conference URL are read off the main actor and filled in
    /// when known, so recording never waits on accessibility.
    private func start(meeting: String, bundleID: String) {
        guard recorder == nil, store.settings.meetingCapture else { return }
        let app = NSRunningApplication.runningApplications(withBundleIdentifier: bundleID).first
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
        begin(meeting: meeting, bundleID: bundleID, directory: directory, metadata: metadata)
        guard let recorder else { return }
        Task.detached(priority: .userInitiated) {
            // Read at the start: title and page of the window holding the
            // call app's focus, dropped together if either shows a private
            // site or (for a browser) the page is unknown.
            let info = MeetingWindowInfo.read(bundleID: bundleID, pid: pid, denylist: denylist)
            recorder.setWindowInfo(title: info.title, conferenceURL: info.conferenceURL)
        }
    }

    private func begin(
        meeting: String, bundleID: String, directory: URL, metadata: MeetingMetadata
    ) {
        do {
            let recorder = try MeetingRecorder(directory: directory, metadata: metadata)
            self.recorder = recorder
            recordingDirectory = directory
            store.recordingMeetingID = metadata.id
            recordingApp = meeting
            recordingBundleID = bundleID
            endPolicy = MeetingEndPolicy(now: Self.now())
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
                    guard let self, Self.isSame(self.recorder, recorder) else { return }
                    self.stop(reason: "start_failed")
                }
            }
        } catch {
            NSLog("phonon meeting: \(error.localizedDescription)")
        }
    }

    /// Whether `current` is the very recorder `candidate` (identity, not
    /// equality).
    nonisolated static func isSame(_ current: AnyObject?, _ candidate: AnyObject) -> Bool {
        current.map { $0 === candidate } ?? false
    }

    func stop(reason: String) {
        guard let recorder else { return }
        self.recorder = nil
        let directory = recordingDirectory
        recordingDirectory = nil
        // Before any async work: a quit while the files close still leaves the meeting marked
        // for transcription at the next launch (the marker goes when transcript.json is written).
        if let directory { MeetingTranscriber.markPending(directory: directory) }
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
            // The chunks are closed and metadata.json final: queue it and show it on Notes.
            if let directory { store.meetingRecordingFinished(directory: directory) }
        }
    }

    /// On quit: close the open chunks synchronously so they stay readable, and mark the
    /// meeting for transcription at the next launch.
    func stopNow(reason: String) {
        timer?.invalidate()
        timer = nil
        guard let recorder else { return }
        self.recorder = nil
        recorder.closeFiles(reason: reason)
        if let recordingDirectory { MeetingTranscriber.markPending(directory: recordingDirectory) }
        recordingDirectory = nil
    }
}
