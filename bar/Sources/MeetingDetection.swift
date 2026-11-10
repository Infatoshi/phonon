import AppKit
import CoreAudio
import Foundation

// Meeting detection, the groundwork for meeting capture. Nothing here records
// or shows anything; the recorder waits on the capture format.
//
// After Wispr Flow 1.6 (`src/main/meetings`): `conference-url-detector`
// (HTTPS URLs on known conferencing hosts, Slack only under /huddle/, Discord
// only under /channels/), and `meeting-autodetector`, which watches which
// processes hold the microphone. Wispr also fingerprints in-call controls of
// Zoom and Teams through Accessibility (`meeting-app-tells`); Phonon relies on
// "a meeting app or conference tab is up and another process holds the
// microphone", which needs no per-app signatures.

enum MeetingDetector {
    static let conferenceHosts = [
        "meet.google.com", "zoom.us", "zoomgov.com", "teams.microsoft.com", "teams.live.com",
        "webex.com", "gotomeeting.com", "bluejeans.com", "whereby.com", "chime.aws",
        "meetings.dialpad.com", "slack.com", "discord.com", "around.co", "meet.jit.si",
    ]

    /// Native apps whose only job is calls: holding the microphone means a call.
    static let callApps = [
        "us.zoom.xos": "Zoom",
        "com.microsoft.teams2": "Microsoft Teams",
        "com.microsoft.teams": "Microsoft Teams",
        "com.cisco.webexmeetingsapp": "Webex",
        "Cisco-Systems.Spark": "Webex",
        "com.apple.FaceTime": "FaceTime",
        "com.ringcentral.glip": "RingCentral",
        "co.around.Around": "Around",
    ]

    /// Chat apps that also call (Slack huddles): they hold the microphone for
    /// a call only while one is up, so they count when frontmost.
    static let chatAppsWithCalls = [
        "com.tinyspeck.slackmacgap": "Slack",
        "com.hnc.Discord": "Discord",
        "net.whatsapp.WhatsApp": "WhatsApp",
        "ru.keepcoder.Telegram": "Telegram",
    ]

    /// Wispr's `conference-url-detector`.
    static func isConferenceURL(_ url: URL) -> Bool {
        guard url.scheme?.lowercased() == "https", let host = url.host?.lowercased() else {
            return false
        }
        guard conferenceHosts.contains(where: { host == $0 || host.hasSuffix("." + $0) }) else {
            return false
        }
        if host == "slack.com" || host.hasSuffix(".slack.com") {
            return url.path.contains("/huddle/")
        }
        if host == "discord.com" || host.hasSuffix(".discord.com") {
            return url.path.hasPrefix("/channels/")
        }
        return true
    }

    /// Processes that capture or play audio for an app, by the app they belong to (after Wispr
    /// Flow 1.6's `meeting-service-names`): Chromium browsers and Electron apps (Slack, Discord)
    /// capture in a `.helper` process, Safari in WebKit's GPU process, FaceTime in
    /// avconferenced and Teams 2 in its module host. Without this the detector compared the
    /// helper with the app and missed every browser call, FaceTime and Slack huddles.
    static let processOwners = [
        "com.apple.webkit.gpu": "com.apple.safari",
        "com.apple.avconferenced": "com.apple.facetime",
        // a daemon with no bundle id is named by its executable (`MicrophoneHolders`)
        "avconferenced": "com.apple.facetime",
        "com.microsoft.teams2.modulehost": "com.microsoft.teams2",
    ]

    /// The app a capturing or playing process belongs to, lowercased: bundle ids are compared
    /// without case (Wispr's table is lowercase; Arc's app is company.thebrowser.Browser and its
    /// helper company.thebrowser.browser.helper).
    static func owner(of bundleID: String) -> String {
        let id = bundleID.lowercased()
        if let owner = processOwners[id] { return owner }
        // Zoom's other processes (us.zoom.CptHost shares the screen and its sound, ZoomPhone,
        // the meeting hosts) are Zoom; its helpers are named ZoomCefHelper, not `.helper`
        if id.hasPrefix("us.zoom.") { return "us.zoom.xos" }
        if let range = id.range(of: ".helper"), range.lowerBound > id.startIndex,
            range.upperBound == id.endIndex || id[range.upperBound] == "." || id[range.upperBound] == " "
        {
            return String(id[..<range.lowerBound])
        }
        return id
    }

    private static func lookup(_ table: [String: String], _ owner: String) -> (bundleID: String, name: String)? {
        table.first { $0.key.lowercased() == owner }.map { ($0.key, $0.value) }
    }

    struct Evidence {
        /// Bundle ids of the processes that hold the microphone, Phonon
        /// excluded.
        var microphoneHolders: [String]
        var frontmostBundleID: String?
        /// URL of the frontmost browser tab, if any.
        var frontmostURL: URL?
        /// The private list. A call in a private app or site is still offered
        /// (the user asked for Discord calls), but nothing is read from it:
        /// no tab URL, no window title, no page in the metadata.
        var denylist: [String]
        /// Bundle ids of the processes playing audio, Phonon excluded. A chat app counts as in
        /// a call only while it also plays (the far side): recording a voice message only
        /// captures.
        var audioOutputs: [String] = []
    }

    /// `app` is what the prompt names; `bundleID` is the process that holds
    /// the microphone for the call, which is what the recording follows.
    enum Verdict: Equatable {
        case none
        case meeting(app: String, bundleID: String)
    }

    /// Only bundle ids and the tab URL of a non-private browser feed this, so
    /// a private call app can be offered without reading anything from it.
    /// Whether `app` (or a process of it) is among `processes`.
    static func holds(_ processes: [String], app: String) -> Bool {
        let owner = owner(of: app)
        return processes.contains { Self.owner(of: $0) == owner }
    }

    static func evaluate(_ evidence: Evidence) -> Verdict {
        let holders = evidence.microphoneHolders
        guard !holders.isEmpty else { return .none }
        let owners = holders.map(owner(of:))
        // `bundleID` is the app's own id (its pid and name are looked up by it), not its helper's
        for owner in owners {
            if let call = lookup(callApps, owner) { return .meeting(app: call.name, bundleID: call.bundleID) }
        }
        if let front = evidence.frontmostBundleID, owners.contains(owner(of: front)),
            let chat = lookup(chatAppsWithCalls, owner(of: front)),
            holds(evidence.audioOutputs, app: front)
        {
            return .meeting(app: chat.name, bundleID: front)
        }
        if let url = evidence.frontmostURL, isConferenceURL(url),
            let front = evidence.frontmostBundleID, owners.contains(owner(of: front))
        {
            return .meeting(app: ContextPolicy.host(of: url) ?? "browser", bundleID: front)
        }
        return .none
    }

    /// Whether to read the frontmost tab's URL at all: only for a browser
    /// that is not private and is itself holding the microphone. Nothing is
    /// read from any other app.
    static func shouldReadTabURL(
        holders: [String], frontmostBundleID: String?, denylist: [String]
    ) -> Bool {
        guard let front = frontmostBundleID, holds(holders, app: front),
            ContextPolicy.isBrowser(front),
            !ContextPolicy.isDenied(bundleID: front, host: nil, denylist: denylist)
        else { return false }
        return true
    }

    /// What of the call app's window may go into metadata.json, read at the
    /// moment recording starts: title and URL come from the same window and
    /// are both dropped if either shows a private site (the user may have
    /// switched that browser window to a private tab before clicking).
    static func recordableWindowInfo(
        title: String?, url: URL?, bundleID: String, denylist: [String]
    ) -> (title: String?, conferenceURL: String?) {
        let host = BrowserPage.pageHost(url)
        // Fail closed: a browser tab whose top-level http(s) page cannot be
        // identified records nothing about itself.
        if ContextPolicy.isBrowser(bundleID) && host == nil {
            return (nil, nil)
        }
        if let host, ContextPolicy.isDenied(bundleID: nil, host: host, denylist: denylist) {
            return (nil, nil)
        }
        // A title naming a private site (the user's list or the built-in
        // password vaults) is not written either.
        if let title, ContextPolicy.isBrowser(bundleID),
            ContextPolicy.hostEntries(ContextPolicy.alwaysPrivate + denylist).contains(where: {
                ContextPolicy.titleNamesSite(title, host: $0)
            })
        {
            return (nil, nil)
        }
        let conference = url.flatMap { isConferenceURL($0) ? $0.absoluteString : nil }
        return (title, conference)
    }

    /// Once recording, the call is still on while the process that held the
    /// microphone still holds it, wherever the user's focus has gone.
    static func stillInCall(recordingBundleID: String, holders: [String]) -> Bool {
        holds(holders, app: recordingBundleID)
    }
}

/// What metadata.json may say about the call's window, read once when
/// recording starts (recording does not wait for it). A browser gives the
/// title and top-level page of the window holding its focus, with the
/// window's own page when the focus is in the toolbar (fail closed); a
/// native call app gives its focused window's title, and no page search
/// runs for it. Blocking AX; call off the main thread.
enum MeetingWindowInfo {
    static func read(
        bundleID: String, pid: pid_t?, denylist: [String],
        isBrowser: (String?) -> Bool = ContextPolicy.isBrowser,
        browserPage: (pid_t) -> (url: URL?, title: String?) = BrowserPage.focusedPage(pid:),
        appWindowTitle: (pid_t) -> String? = TextboxAccess.focusedWindowTitle(pid:)
    ) -> (title: String?, conferenceURL: String?) {
        guard let pid else { return (nil, nil) }
        // A private app (Discord, Signal) is recorded on request but never read.
        if !isBrowser(bundleID),
            ContextPolicy.isDenied(bundleID: bundleID, host: nil, denylist: denylist)
        {
            return (nil, nil)
        }
        if isBrowser(bundleID) {
            let page = browserPage(pid)
            return MeetingDetector.recordableWindowInfo(
                title: page.title, url: page.url, bundleID: bundleID, denylist: denylist)
        }
        return MeetingDetector.recordableWindowInfo(
            title: appWindowTitle(pid), url: nil, bundleID: bundleID, denylist: denylist)
    }
}

/// When to ask "Meeting detected: take notes?". Once per call; after "Not
/// now", not again for the same app for a while.
struct MeetingPromptPolicy {
    static let quietAfterDecline: TimeInterval = 30 * 60

    private var declined = [String: TimeInterval]()
    private var askedForCurrentCall: String?

    /// `meeting` is the current verdict's app, nil when no call is up.
    mutating func shouldAsk(meeting: String?, now: TimeInterval) -> Bool {
        guard let meeting else {
            askedForCurrentCall = nil
            return false
        }
        guard askedForCurrentCall != meeting else { return false }
        if let at = declined[meeting], now - at < Self.quietAfterDecline { return false }
        askedForCurrentCall = meeting
        return true
    }

    mutating func declined(_ meeting: String, now: TimeInterval) {
        declined[meeting] = now
    }
}

/// Which processes are capturing audio input right now (CoreAudio process
/// objects, macOS 14+). Phonon's own warm microphone is left out, so "the mic
/// is in use" means someone else is using it.
enum MicrophoneHolders {
    static func bundleIDs(excludingPID ownPID: pid_t = getpid()) -> [String] {
        snapshot(excludingPID: ownPID).input
    }

    /// Bundle ids of the processes capturing audio (`input`) and playing it (`output`).
    static func snapshot(excludingPID ownPID: pid_t = getpid()) -> (input: [String], output: [String]) {
        var input: [String] = []
        var output: [String] = []
        for object in processObjects() {
            let capturing = isRunning(object, kAudioProcessPropertyIsRunningInput)
            let playing = isRunning(object, kAudioProcessPropertyIsRunningOutput)
            guard capturing || playing, let pid = processID(object), pid != ownPID,
                let id = bundleID(object) ?? NSRunningApplication(processIdentifier: pid)?.bundleIdentifier
                    ?? executableName(pid)
            else { continue }
            if capturing { input.append(id) }
            if playing { output.append(id) }
        }
        return (input, output)
    }

    /// A daemon (avconferenced for FaceTime) has no bundle id and no NSRunningApplication:
    /// its executable's name, so it is not dropped.
    private static func executableName(_ pid: pid_t) -> String? {
        var buffer = [CChar](repeating: 0, count: 4 * Int(MAXPATHLEN))
        guard proc_pidpath(pid, &buffer, UInt32(buffer.count)) > 0 else { return nil }
        let name = URL(fileURLWithPath: String(cString: buffer)).lastPathComponent
        return name.isEmpty ? nil : name
    }

    private static func processObjects() -> [AudioObjectID] {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyProcessObjectList,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var size: UInt32 = 0
        guard
            AudioObjectGetPropertyDataSize(
                AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size) == noErr
        else { return [] }
        var objects = [AudioObjectID](
            repeating: 0, count: Int(size) / MemoryLayout<AudioObjectID>.size)
        guard
            AudioObjectGetPropertyData(
                AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &objects)
                == noErr
        else { return [] }
        return objects
    }

    private static func isRunning(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) -> Bool {
        var address = AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var running: UInt32 = 0
        var size = UInt32(MemoryLayout<UInt32>.size)
        return AudioObjectGetPropertyData(object, &address, 0, nil, &size, &running) == noErr
            && running != 0
    }

    private static func processID(_ object: AudioObjectID) -> pid_t? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioProcessPropertyPID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var pid: pid_t = 0
        var size = UInt32(MemoryLayout<pid_t>.size)
        guard AudioObjectGetPropertyData(object, &address, 0, nil, &size, &pid) == noErr else {
            return nil
        }
        return pid
    }

    private static func bundleID(_ object: AudioObjectID) -> String? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioProcessPropertyBundleID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var value: Unmanaged<CFString>?
        var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        guard AudioObjectGetPropertyData(object, &address, 0, nil, &size, &value) == noErr,
            let value
        else { return nil }
        let id = value.takeRetainedValue() as String
        return id.isEmpty ? nil : id
    }
}
