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

    /// The end gap is what separates calls: just under it is the same call, just over it a new one.
    func testTheEndGapSeparatesCalls() {
        let gap = MeetingPromptPolicy.callEndGap
        let same = [Self.inCall(0, 600), Self.gap(600, 600 + gap - 8), Self.inCall(600 + gap - 8, 1_200 + gap)]
        XCTAssertEqual(Self.prompts(same, answer: .noAnswer).count, 1)
        let apart = [Self.inCall(0, 600), Self.gap(600, 600 + gap + 8), Self.inCall(600 + gap + 8, 1_200 + gap)]
        XCTAssertEqual(Self.prompts(apart, answer: .noAnswer).count, 2)
    }
}
