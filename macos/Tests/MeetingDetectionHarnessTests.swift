import XCTest

@testable import PhononBar

/// Deterministic harness for call detection (GOAL N1 and N2), logic level only: scripted
/// snapshots of what the detector sees each poll (which processes hold the microphone, which
/// play audio, the frontmost app and, for a browser, its tab), fed through the real
/// `MeetingDetector`, `MeetingPromptPolicy` and `MeetingEndPolicy` at the controller's 4 s
/// cadence. Nothing here touches CoreAudio, Accessibility or the screen.
///
/// Which process holds the microphone for each app is taken from Wispr Flow 1.6's shipping
/// table (src/main/meetings/meeting-service-names.js: Chromium browsers capture in their
/// `.helper` process, Safari in com.apple.WebKit.GPU, FaceTime in com.apple.avconferenced,
/// Slack in its helper, Teams 2 in its module host). It was not observed on this Mac; a live
/// call per app is still the owner's check.
final class MeetingDetectionHarnessTests: XCTestCase {
    /// What the detector sees for a stretch of time.
    struct Moment {
        var from: TimeInterval
        var to: TimeInterval
        var holders: [String] = []
        var outputs: [String] = []
        var front: String?
        var url: String?
    }

    struct Scenario {
        var name: String
        var moments: [Moment]
        /// For a call: when the user joined (the mic came on) and when the call ended.
        var call: ClosedRange<TimeInterval>?
    }

    struct Outcome {
        var prompts: [TimeInterval] = []
        var app: String?
        /// For a call: when the simulated recording (Record clicked at the prompt) stopped.
        var recordingStopped: TimeInterval?
    }

    static let denylist = ContextPolicy.defaultDenylist

    static func evidence(_ moment: Moment?) -> MeetingDetector.Evidence {
        let holders = moment?.holders ?? []
        let front = moment?.front
        // the controller reads the tab only when the detector allows it
        let url =
            MeetingDetector.shouldReadTabURL(holders: holders, frontmostBundleID: front, denylist: denylist)
            ? moment?.url.flatMap(URL.init(string:)) : nil
        return MeetingDetector.Evidence(
            microphoneHolders: holders, frontmostBundleID: front, frontmostURL: url, denylist: denylist,
            audioOutputs: moment?.outputs ?? [])
    }

    /// The controller's loop: a poll every 4 s from `phase`; asks when the verdict says so and
    /// the prompt policy agrees; once recording, follows the recorded process until it has not
    /// held the microphone for the grace period.
    static func run(_ scenario: Scenario, phase: TimeInterval, recordOnPrompt: Bool = true) -> Outcome {
        var outcome = Outcome()
        var prompts = MeetingPromptPolicy()
        var recording: (bundleID: String, end: MeetingEndPolicy)?
        let last = scenario.moments.map(\.to).max() ?? 0
        var now = phase
        while now <= last + 20 {
            let moment = scenario.moments.first { $0.from <= now && now < $0.to }
            let evidence = evidence(moment)
            prompts.observe(holders: evidence.microphoneHolders, now: now)
            if var current = recording {
                let inCall = MeetingDetector.stillInCall(
                    recordingBundleID: current.bundleID, holders: evidence.microphoneHolders)
                if current.end.observe(stillInCall: inCall, now: now) {
                    outcome.recordingStopped = now
                    recording = nil
                } else {
                    recording = current
                }
            } else {
                let verdict = MeetingDetector.evaluate(evidence)
                if case let .meeting(app, bundleID) = verdict {
                    if prompts.shouldAsk(meeting: app, bundleID: bundleID, now: now) {
                        outcome.prompts.append(now)
                        outcome.app = outcome.app ?? app
                        if recordOnPrompt, scenario.call != nil, outcome.recordingStopped == nil {
                            recording = (bundleID, MeetingEndPolicy(now: now))
                        }
                    }
                }
            }
            now += MeetingController.detectionInterval
        }
        return outcome
    }

    // MARK: The 30 scripted calls (N1)

    static let chrome = "com.google.Chrome", chromeMic = "com.google.Chrome.helper"
    static let safari = "com.apple.Safari", safariMic = "com.apple.WebKit.GPU"
    static let arc = "company.thebrowser.Browser", arcMic = "company.thebrowser.browser.helper"
    static let edge = "com.microsoft.edgemac", edgeMic = "com.microsoft.edgemac.helper"
    static let zoom = "us.zoom.xos"
    static let teams = "com.microsoft.teams2", teamsMic = "com.microsoft.teams2.modulehost"
    static let facetime = "com.apple.FaceTime", facetimeMic = "com.apple.avconferenced"
    static let slack = "com.tinyspeck.slackmacgap", slackMic = "com.tinyspeck.slackmacgap.helper"
    static let notes = "com.apple.Notes", calendar = "com.apple.iCal"

    /// A call held by `mic` (also playing the far side), the user in `front` and later in
    /// `then` (from 20 s on), for `minutes`.
    static func call(
        _ name: String, mic: String, front: String?, url: String? = nil, then: String? = nil,
        minutes: Double = 3
    ) -> Scenario {
        let end = minutes * 60
        var moments = [Moment(from: 0, to: 20, holders: [mic], outputs: [mic], front: front, url: url)]
        moments.append(
            Moment(
                from: 20, to: end, holders: [mic], outputs: [mic], front: then ?? front,
                url: then == nil ? url : nil))
        return Scenario(name: name, moments: moments, call: 0...end)
    }

    static let meet = "https://meet.google.com/abc-defg-hij"
    static let huddle = "https://app.slack.com/huddle/T0123ABC/C0456DEF"

    static let calls: [Scenario] = [
        call("Zoom app, in front", mic: zoom, front: zoom),
        call("Zoom app, then notes", mic: zoom, front: zoom, then: notes),
        call("Zoom app joined from the calendar", mic: zoom, front: calendar, then: zoom),
        call("Zoom app, 45 minutes", mic: zoom, front: zoom, minutes: 45),
        call("Meet in Chrome", mic: chromeMic, front: chrome, url: meet),
        call("Meet in Chrome, then notes", mic: chromeMic, front: chrome, url: meet, then: notes),
        call("Meet in Chrome, 30 minutes", mic: chromeMic, front: chrome, url: meet, minutes: 30),
        call("Meet in Safari", mic: safariMic, front: safari, url: meet),
        call("Meet in Safari, then notes", mic: safariMic, front: safari, url: meet, then: notes),
        call("Meet in Safari, 30 minutes", mic: safariMic, front: safari, url: meet, minutes: 30),
        call("Meet in Arc", mic: arcMic, front: arc, url: meet),
        call("Meet in Arc, then notes", mic: arcMic, front: arc, url: meet, then: notes),
        call("Meet in Arc, 30 minutes", mic: arcMic, front: arc, url: meet, minutes: 30),
        call("Zoom web client in Chrome", mic: chromeMic, front: chrome, url: "https://app.zoom.us/wc/123456789/join"),
        call("Teams app (module host)", mic: teamsMic, front: teams),
        call("Teams app (main process)", mic: teams, front: teams),
        call("Teams app, then notes", mic: teamsMic, front: teams, then: notes),
        call("Teams in Edge", mic: edgeMic, front: edge, url: "https://teams.microsoft.com/v2/"),
        call("Teams in Chrome", mic: chromeMic, front: chrome, url: "https://teams.live.com/v2/"),
        call("FaceTime", mic: facetimeMic, front: facetime),
        call("FaceTime, then notes", mic: facetimeMic, front: facetime, then: notes),
        call("FaceTime (app process)", mic: facetime, front: facetime),
        call("Slack huddle in the app", mic: slackMic, front: slack),
        call("Slack huddle in the app, then Chrome", mic: slackMic, front: slack, then: chrome),
        call("Slack huddle in the app (main process)", mic: slack, front: slack),
        call("Slack huddle in the app, 20 minutes", mic: slackMic, front: slack, minutes: 20),
        call("Slack huddle in Chrome", mic: chromeMic, front: chrome, url: huddle),
        call("Slack huddle in Chrome, then notes", mic: chromeMic, front: chrome, url: huddle, then: notes),
        call("Slack huddle in Arc", mic: arcMic, front: arc, url: huddle),
        call("Slack huddle in Safari", mic: safariMic, front: safari, url: huddle),
    ]

    struct CallReport {
        var noticed = 0
        var noticedWithin10 = 0
        var recordedToTheEnd = 0
        var misses: [String] = []
        var cutShort: [String] = []
        var worstSeconds: TimeInterval = 0
    }

    /// Each call at four poll phases (0, 1, 2 and 3 s after the mic comes on); a call counts as
    /// noticed within 10 s only if it is at every phase.
    static func callReport() -> CallReport {
        var report = CallReport()
        for scenario in calls {
            let call = scenario.call!
            let runs = [0.0, 1, 2, 3].map { run(scenario, phase: $0) }
            let delays = runs.map { $0.prompts.first.map { $0 - call.lowerBound } }
            if delays.allSatisfy({ $0 != nil }) { report.noticed += 1 }
            if delays.allSatisfy({ ($0 ?? .infinity) <= 10 }) {
                report.noticedWithin10 += 1
                report.worstSeconds = max(report.worstSeconds, delays.compactMap { $0 }.max() ?? 0)
            } else {
                report.misses.append(scenario.name)
            }
            // recorded from the prompt to the end: stopped within grace + one poll after it ended
            let toTheEnd = runs.allSatisfy { outcome in
                guard let stopped = outcome.recordingStopped else { return false }
                return stopped >= call.upperBound
                    && stopped <= call.upperBound + MeetingEndPolicy.grace + MeetingController.detectionInterval
            }
            if toTheEnd { report.recordedToTheEnd += 1 } else if delays.contains(where: { $0 != nil }) {
                report.cutShort.append(scenario.name)
            }
        }
        return report
    }

    // MARK: Eight hours of ordinary microphone use (N2)

    static let whatsapp = "net.whatsapp.WhatsApp", telegram = "ru.keepcoder.Telegram"
    static let messages = "com.apple.MobileSMS", voiceMemos = "com.apple.VoiceMemos"
    static let wispr = "com.electron.wispr-flow", appleDictation = "com.apple.SpeechRecognitionCore.speechrecognitiond"
    static let textEdit = "com.apple.TextEdit", loom = "com.loom.desktop", quickTime = "com.apple.QuickTimePlayerX"

    /// One working day with no call: each use is placed after the previous one with a gap.
    static func ordinaryDay() -> (Scenario, [String: Int]) {
        var moments: [Moment] = []
        var counts: [String: Int] = [:]
        var at: TimeInterval = 0
        func use(
            _ kind: String, seconds: TimeInterval, holders: [String] = [], outputs: [String] = [], front: String?,
            url: String? = nil, gap: TimeInterval = 120
        ) {
            moments.append(Moment(from: at, to: at + seconds, holders: holders, outputs: outputs, front: front, url: url))
            at += seconds + gap
            counts[kind, default: 0] += 1
        }
        for index in 0..<60 {  // Phonon's own dictation: its process is never a holder
            use("Phonon dictation", seconds: 8 + Double(index % 5) * 6, front: [slack, chrome, notes][index % 3], gap: 120)
        }
        for index in 0..<20 {
            use(
                "Wispr Flow dictation", seconds: 12, holders: [wispr], front: [slack, chrome, textEdit, whatsapp][index % 4],
                url: index % 4 == 1 ? "https://docs.google.com/document/d/x" : nil)
        }
        for _ in 0..<10 { use("Apple Dictation", seconds: 15, holders: [appleDictation], front: textEdit) }
        for index in 0..<8 {
            use("WhatsApp voice message", seconds: 5 + Double(index) * 6, holders: [whatsapp], front: whatsapp)
        }
        for index in 0..<6 {
            use("Telegram voice message", seconds: 8 + Double(index) * 7, holders: [telegram], front: telegram)
        }
        for index in 0..<4 { use("Slack voice clip (helper)", seconds: 20 + Double(index) * 10, holders: [slackMic], front: slack) }
        for index in 0..<4 { use("Slack voice clip (main process)", seconds: 15 + Double(index) * 10, holders: [slack], front: slack) }
        for _ in 0..<4 { use("Messages audio message", seconds: 12, holders: [messages], front: messages) }
        for index in 0..<3 { use("Voice Memos", seconds: 60 + Double(index) * 120, holders: [voiceMemos], front: voiceMemos) }
        use("YouTube in Chrome", seconds: 2 * 3_600, outputs: [chromeMic], front: chrome, url: "https://www.youtube.com/watch?v=x")
        use("Netflix in Safari", seconds: 1_800, outputs: [safariMic], front: safari, url: "https://www.netflix.com/watch/1")
        use("a recorded Zoom call played in QuickTime", seconds: 900, outputs: [quickTime], front: quickTime)
        for _ in 0..<5 {
            use(
                "ChatGPT voice in Chrome", seconds: 120, holders: [chromeMic], outputs: [chromeMic], front: chrome,
                url: "https://chatgpt.com/c/x")
        }
        for _ in 0..<2 {
            use("voice search in Chrome", seconds: 6, holders: [chromeMic], front: chrome, url: "https://www.google.com/")
        }
        use(
            "WhatsApp Web voice message in Chrome", seconds: 25, holders: [chromeMic], front: chrome,
            url: "https://web.whatsapp.com/")
        use(
            "Slack web voice clip in Chrome", seconds: 30, holders: [chromeMic], front: chrome,
            url: "https://app.slack.com/client/T0123ABC/C0456DEF")
        use(
            "dictating into Docs with a Meet tab open (no call)", seconds: 40, holders: [wispr], front: chrome,
            url: "https://docs.google.com/document/d/x")
        use("Loom recording", seconds: 180, holders: [loom], outputs: [], front: loom)
        use("QuickTime audio recording", seconds: 120, holders: [quickTime], front: quickTime)
        return (Scenario(name: "ordinary day", moments: moments, call: nil), counts)
    }

    struct DayReport {
        var hours: Double
        var uses: Int
        var falsePrompts: [String: Int] = [:]
        var total: Int { falsePrompts.values.reduce(0, +) }
    }

    static func dayReport() -> DayReport {
        let (day, counts) = ordinaryDay()
        let hours = (day.moments.map(\.to).max() ?? 0) / 3_600
        var report = DayReport(hours: hours, uses: counts.values.reduce(0, +))
        // each use on its own, so one prompt is credited to the use that caused it
        for moment in day.moments {
            let single = Scenario(name: "", moments: [Moment(
                from: 0, to: moment.to - moment.from, holders: moment.holders, outputs: moment.outputs,
                front: moment.front, url: moment.url)], call: nil)
            for phase in [0.0, 2] {
                let outcome = run(single, phase: phase, recordOnPrompt: false)
                if !outcome.prompts.isEmpty {
                    report.falsePrompts[kind(of: moment, in: day, counts: counts), default: 0] += 1
                    break
                }
            }
        }
        return report
    }

    private static func kind(of moment: Moment, in day: Scenario, counts: [String: Int]) -> String {
        "\(moment.front ?? "-") holding \(moment.holders.joined(separator: ","))"
    }

    func testThirtyScriptedCallsAndAnOrdinaryDay() {
        let calls = Self.callReport()
        let day = Self.dayReport()
        print(
            "N1 logic: \(calls.noticedWithin10) of \(Self.calls.count) calls noticed within 10 s at every poll phase "
                + "(worst \(calls.worstSeconds) s), \(calls.noticed) noticed at all, \(calls.recordedToTheEnd) recorded "
                + "to the end; missed: \(calls.misses); cut short: \(calls.cutShort)")
        print(
            "N2 logic: \(day.total) false prompts in \(String(format: "%.1f", day.hours)) h of \(day.uses) ordinary uses: "
                + "\(day.falsePrompts.sorted { $0.key < $1.key })")
        XCTAssertEqual(Self.calls.count, 30)
        XCTAssertGreaterThanOrEqual(day.hours, 8)
        // the bars, at the logic level: at least 29 of 30 within 10 s, no false prompt
        XCTAssertGreaterThanOrEqual(calls.noticedWithin10, 29)
        XCTAssertEqual(calls.recordedToTheEnd, 30)
        XCTAssertEqual(day.total, 0)
    }

    /// Known limits, outside the 30 and the day: reported, not hidden.
    func testKnownLimits() {
        // a browser call is seen only while its tab is in front: leaving it within 2 s of the mic
        // coming on (no lobby) is missed at half the poll phases
        let quick = Scenario(
            name: "Meet in Chrome, left after 2 s",
            moments: [
                Moment(from: 0, to: 2, holders: [Self.chromeMic], outputs: [Self.chromeMic], front: Self.chrome, url: Self.meet),
                Moment(from: 2, to: 180, holders: [Self.chromeMic], outputs: [Self.chromeMic], front: Self.notes),
            ], call: 0...180)
        let seen = [0.0, 1, 2, 3].filter { !Self.run(quick, phase: $0).prompts.isEmpty }.count
        XCTAssertEqual(seen, 2)
        // a native call app's own microphone test (Zoom's settings) is offered like a call
        let test = Scenario(
            name: "Zoom microphone test", moments: [Moment(from: 0, to: 15, holders: [Self.zoom], front: Self.zoom)],
            call: nil)
        XCTAssertEqual(Self.run(test, phase: 0, recordOnPrompt: false).prompts.count, 1)
        print("detection limits: a Meet tab left within 2 s is seen at \(seen) of 4 poll phases; Zoom's mic test prompts")
    }
}
