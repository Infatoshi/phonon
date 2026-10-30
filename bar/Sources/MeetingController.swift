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
    private var recordingApp: String?
    /// The process whose microphone use defines the call being recorded.
    private var recordingBundleID: String?
    private let panel = SuggestionPanelController(raisedBy: 70)
    private var checking = false
    /// Tells the menu bar item whether a meeting is being recorded.
    var onRecordingChanged: ((String?) -> Void)?

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
        panel.present(
            title: "Meeting detected in \(meeting): take notes?",
            detail: "Phonon records your microphone and the call audio on this Mac.",
            approveLabel: "Take notes",
            onApprove: { [weak self] in
                self?.panel.dismiss()
                self?.start(meeting: meeting, bundleID: bundleID)
            },
            onReject: { [weak self] in
                self?.prompts.declined(meeting, now: Self.now())
                self?.panel.dismiss()
            },
            onExpire: {})
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
            recordingApp = meeting
            recordingBundleID = bundleID
            endPolicy = MeetingEndPolicy(now: Self.now())
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
        endPolicy = nil
        recordingApp = nil
        recordingBundleID = nil
        onRecordingChanged?(nil)
        Task { await recorder.stop(reason: reason) }
    }

    /// On quit: close the open chunks synchronously so they stay readable.
    func stopNow(reason: String) {
        timer?.invalidate()
        timer = nil
        guard let recorder else { return }
        self.recorder = nil
        recorder.closeFiles(reason: reason)
    }
}
