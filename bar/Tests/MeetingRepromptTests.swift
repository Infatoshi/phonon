import XCTest

@testable import PhononBar

/// The owner's report: hours into one Discord call, Phonon kept asking whether to take notes.
/// One call gets at most one question, whatever the answer; only a call that has really ended
/// (its app off the microphone for `MeetingPromptPolicy.callEndGap`) lets the next one ask.
/// Scripted timelines fed through the real detector, prompt policy and rules at the
/// controller's 4 s cadence, the way `MeetingController.handle` uses them.
final class MeetingRepromptTests: XCTestCase {
    typealias Moment = MeetingDetectionHarnessTests.Moment

    static let discord = "com.hnc.Discord", discordMic = "com.hnc.Discord.helper"
    static let notes = "com.apple.Notes", chrome = "com.google.Chrome"

    enum Answer { case notNow, noAnswer }

    /// Prompt times. `answer` is what the owner does with each question; `rules` are the
    /// per-app rules by `MeetingRules.key`.
    static func prompts(
        _ moments: [Moment], mode: MeetingMode = .perApp, rules: [String: MeetingRule] = [:],
        answer: Answer = .notNow
    ) -> [TimeInterval] {
        var policy = MeetingPromptPolicy()
        var asked: [TimeInterval] = []
        let last = moments.map(\.to).max() ?? 0
        var now: TimeInterval = 0
        while now <= last + 20 {
            let moment = moments.first { $0.from <= now && now < $0.to }
            let evidence = MeetingDetectionHarnessTests.evidence(moment)
            policy.observe(holders: evidence.microphoneHolders, now: now)
            if case let .meeting(app, bundleID) = MeetingDetector.evaluate(evidence),
                policy.shouldAsk(meeting: app, bundleID: bundleID, now: now)
            {
                let rule = rules[MeetingRules.key(app: app, bundleID: bundleID)]
                if MeetingRules.decide(mode: mode, rule: rule) == .ask {
                    asked.append(now)
                    if answer == .notNow { policy.declined(app, now: now) }
                }
            }
            now += MeetingController.detectionInterval
        }
        return asked
    }

    /// Discord in its own app, on the mic and playing, the owner in front.
    static func inCall(_ from: TimeInterval, _ to: TimeInterval, front: String? = discord, mic: String = discordMic,
        playing: Bool = true) -> Moment
    {
        Moment(from: from, to: to, holders: [mic], outputs: playing ? [discordMic] : [], front: front)
    }

    static func gap(_ from: TimeInterval, _ to: TimeInterval, front: String? = discord) -> Moment {
        Moment(from: from, to: to, holders: [], outputs: [], front: front)
    }

    static let hour: TimeInterval = 3_600

    /// Three hours of one Discord call: the owner looks away every few minutes (the verdict
    /// drops to none), mutes (Discord lets go of the mic for 30 s), the far side goes quiet,
    /// dictates with Phonon mid-call (Phonon is never a holder), switches headset (the mic is
    /// released for 6 s and taken again by Discord's main process), and the detector sees
    /// nothing at all for 100 s. Then 10 minutes with no call, and a new 30 minute call.
    static func longCallThenANewOne() -> (moments: [Moment], end: TimeInterval) {
        var moments: [Moment] = []
        var at: TimeInterval = 0
        func add(_ moment: Moment) {
            moments.append(moment)
            at = moment.to
        }
        let end = 3 * hour
        var block = 0
        while at < end {
            switch block % 6 {
            case 0: add(inCall(at, at + 300))
            case 1: add(inCall(at, at + 240, front: notes))  // looking at notes
            case 2: add(gap(at, at + 30))  // muted: the mic released
            case 3: add(inCall(at, at + 300, playing: false))  // far side quiet
            case 4: add(inCall(at, at + 300, front: chrome))  // browsing, no call in the tab
            default: add(inCall(at, at + 270))
            }
            block += 1
            if moments.count == 12 {  // about 1 h in: dictating into Notes with Phonon
                add(inCall(at, at + 25, front: notes))
            }
            if moments.count == 20 {  // headset switch: 6 s off the mic, back in the main process
                add(gap(at, at + 6))
                add(inCall(at, at + 300, mic: discord))
            }
            if moments.count == 26 {  // a detector dropout shorter than the end gap
                add(Moment(from: at, to: at + 100))
            }
        }
        moments[moments.count - 1].to = end
        let next = end + 600
        moments.append(gap(end, next, front: notes))
        moments.append(inCall(next, next + 1_800))
        return (moments, end)
    }

    func testOneLongCallAsksOnceThenANewCallAsksAgain() {
        let (moments, end) = Self.longCallThenANewOne()
        XCTAssertTrue(moments.contains { $0.holders.isEmpty && $0.to - $0.from == 100 }, "has the dropout")
        XCTAssertTrue(moments.contains { $0.holders == [Self.discord] }, "has the device switch")
        for mode in [MeetingMode.perApp, .everyCall] {
            let asked = Self.prompts(moments, mode: mode)
            XCTAssertEqual(asked.filter { $0 < end }.count, 1, "\(mode): one question in the 3 h call")
            XCTAssertEqual(asked.first, 0)
            XCTAssertEqual(asked.count, 2, "\(mode): the new call after it ended is asked about")
            XCTAssertEqual(asked.last, end + 600)
        }
    }

    func testAQuestionLeftToTimeOutIsNotAskedAgainInTheSameCall() {
        let (moments, end) = Self.longCallThenANewOne()
        let asked = Self.prompts(moments.filter { $0.from < end }, answer: .noAnswer)
        XCTAssertEqual(asked, [0])
    }

    func testNeverForTheAppNeverAsks() {
        let (moments, _) = Self.longCallThenANewOne()
        let asked = Self.prompts(moments, rules: [Self.discord.lowercased(): .never])
        XCTAssertEqual(asked, [])
        // "Ask every time" asks whatever the rules say, still once per call
        XCTAssertEqual(
            Self.prompts(moments, mode: .everyCall, rules: [Self.discord.lowercased(): .never]).count, 2)
    }

    // MARK: Recording again after the end grace (an "always" call must not stay unrecorded)

    struct Run {
        var prompts: [TimeInterval] = []
        var starts: [TimeInterval] = []
        var stops: [TimeInterval] = []
    }

    /// `MeetingController.handle` with a recorder: the end policy stops a recording 10 s off
    /// the mic; while nothing records, `MeetingPromptPolicy.action` says whether to ask, record
    /// or do nothing. A question is answered with Record (`recordOnAsk`); `userStopsAt` presses
    /// Stop at that time.
    static func record(
        _ moments: [Moment], mode: MeetingMode = .perApp, rules: [String: MeetingRule] = [:],
        recordOnAsk: Bool = true, userStopsAt: TimeInterval? = nil
    ) -> Run {
        var policy = MeetingPromptPolicy()
        var run = Run()
        var recording: (app: String, bundleID: String, end: MeetingEndPolicy)?
        let last = moments.map(\.to).max() ?? 0
        var now: TimeInterval = 0
        func start(_ app: String, _ bundleID: String) {
            policy.accepted(meeting: app, bundleID: bundleID, now: now)
            recording = (app, bundleID, MeetingEndPolicy(now: now))
            run.starts.append(now)
        }
        while now <= last + 20 {
            let moment = moments.first { $0.from <= now && now < $0.to }
            let evidence = MeetingDetectionHarnessTests.evidence(moment)
            policy.observe(holders: evidence.microphoneHolders, now: now)
            if let stopAt = userStopsAt, let current = recording, now >= stopAt, run.stops.isEmpty {
                policy.holdOff(meeting: current.app, bundleID: current.bundleID, now: now)
                recording = nil
                run.stops.append(now)
            }
            if var current = recording {
                let inCall = MeetingDetector.stillInCall(
                    recordingBundleID: current.bundleID, holders: evidence.microphoneHolders)
                if current.end.observe(stillInCall: inCall, now: now) {
                    recording = nil
                    run.stops.append(now)
                } else {
                    recording = current
                }
            } else if case let .meeting(app, bundleID) = MeetingDetector.evaluate(evidence) {
                let rule = rules[MeetingRules.key(app: app, bundleID: bundleID)]
                switch policy.action(
                    meeting: app, bundleID: bundleID, decision: MeetingRules.decide(mode: mode, rule: rule), now: now)
                {
                case .ignore: break
                case .record: start(app, bundleID)
                case .ask:
                    run.prompts.append(now)
                    if recordOnAsk { start(app, bundleID) }
                }
            }
            now += MeetingController.detectionInterval
        }
        return run
    }

    static let always = [discord.lowercased(): MeetingRule.always]

    /// Two Discord calls 30 s apart (the end grace is 10 s, the call end gap 2 minutes), and a
    /// 30 s mute inside the second: an "always" rule records each stretch, never asking.
    static let backToBack = [
        inCall(0, 600), gap(600, 632), inCall(632, 1_200), gap(1_200, 1_232), inCall(1_232, 1_800),
    ]

    func testAlwaysRuleRecordsAgainAfterTheMicWasFree() {
        let run = Self.record(Self.backToBack, rules: Self.always)
        XCTAssertEqual(run.prompts, [])
        XCTAssertEqual(run.starts, [0, 632, 1_232], "each stretch on the mic is recorded")
        XCTAssertEqual(run.stops.count, 3)
    }

    func testARecordingTheUserAcceptedResumesWithoutANewQuestion() {
        for mode in [MeetingMode.perApp, .everyCall] {
            let run = Self.record(Self.backToBack, mode: mode)
            XCTAssertEqual(run.prompts, [0], "\(mode): asked once")
            XCTAssertEqual(run.starts, [0, 632, 1_232], "\(mode)")
        }
        // Not recorded at the question: nothing starts later in the call either.
        XCTAssertEqual(Self.record(Self.backToBack, recordOnAsk: false).starts, [])
        // "Never" stays silent.
        XCTAssertEqual(Self.record(Self.backToBack, rules: [Self.discord.lowercased(): .never]).starts, [])
    }

    func testAStopByTheUserHoldsForTheRestOfTheCall() {
        let run = Self.record(Self.backToBack, rules: Self.always, userStopsAt: 300)
        XCTAssertEqual(run.starts, [0], "not restarted after Stop while the call lasts")
        // After the call really ends (off the mic for the end gap), the next call records again.
        let gapEnd = 1_800 + MeetingPromptPolicy.callEndGap + 20
        let later = Self.record(
            Self.backToBack + [Self.gap(1_800, gapEnd), Self.inCall(gapEnd, gapEnd + 300)],
            rules: Self.always, userStopsAt: 300)
        XCTAssertEqual(later.starts, [0, gapEnd])
        XCTAssertEqual(later.prompts, [])
    }

    func testTheLongCallStillAsksOnceWhenRecordingResumes() {
        let (moments, end) = Self.longCallThenANewOne()
        let run = Self.record(moments)
        XCTAssertEqual(run.prompts, [0, end + 600], "one question per call")
        XCTAssertGreaterThan(run.starts.filter { $0 < end }.count, 1, "the mute gaps resume the recording")
    }

    /// The end gap is what separates calls: just under it is the same call, just over it a new one.
    func testTheEndGapSeparatesCalls() {
        let gap = MeetingPromptPolicy.callEndGap
        let same = [Self.inCall(0, 600), Self.gap(600, 600 + gap - 8), Self.inCall(600 + gap - 8, 1_200 + gap)]
        XCTAssertEqual(Self.prompts(same, answer: .noAnswer).count, 1)
        let apart = [Self.inCall(0, 600), Self.gap(600, 600 + gap + 8), Self.inCall(600 + gap + 8, 1_200 + gap)]
        XCTAssertEqual(Self.prompts(apart, answer: .noAnswer).count, 2)
    }
}
