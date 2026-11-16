import XCTest

@testable import PhononBar

/// Per-app meeting rules (MeetingRules.swift): the rule key of a detected call, the
/// decision (mode + rule -> record / ask / ignore), the rule store and its persistence,
/// Undo, the questions card, and the migration from `meeting_capture` + `meeting_auto_start`.
@MainActor
final class MeetingRulesTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("phonon-meeting-rules-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func decode(_ json: String) throws -> NativeSettings {
        try JSONDecoder().decode(NativeSettings.self, from: Data(json.utf8))
    }

    private var settingsURL: URL { directory.appendingPathComponent("settings.json") }

    private func storedJSON() throws -> [String: Any] {
        try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: settingsURL)) as? [String: Any])
    }

    // MARK: Keys

    func testRulesKeyOnBundleIDsAndConferencingSitesNotNames() {
        // Native apps: their bundle id, lowercased.
        XCTAssertEqual(MeetingRules.key(app: "Zoom", bundleID: "us.zoom.xos"), "us.zoom.xos")
        XCTAssertEqual(
            MeetingRules.key(app: "Webex", bundleID: "Cisco-Systems.Spark"), "cisco-systems.spark")
        XCTAssertEqual(MeetingRules.key(app: "Discord", bundleID: "com.hnc.Discord"), "com.hnc.discord")
        // A tab: the conferencing site, whatever browser and subdomain.
        XCTAssertEqual(
            MeetingRules.key(app: "meet.google.com", bundleID: "com.google.Chrome"), "meet.google.com")
        XCTAssertEqual(
            MeetingRules.key(app: "meet.google.com", bundleID: "com.apple.Safari"), "meet.google.com")
        XCTAssertEqual(MeetingRules.key(app: "us02web.zoom.us", bundleID: "com.google.Chrome"), "zoom.us")
        XCTAssertEqual(MeetingRules.key(app: "app.slack.com", bundleID: "company.thebrowser.Browser"), "slack.com")
        XCTAssertEqual(MeetingRules.key(app: "teams.live.com", bundleID: "com.microsoft.edgemac"), "teams.live.com")

        // The keys of what the detector actually returns.
        let meet = MeetingDetector.evaluate(
            MeetingDetector.Evidence(
                microphoneHolders: ["com.google.Chrome.helper"], frontmostBundleID: "com.google.Chrome",
                frontmostURL: URL(string: "https://meet.google.com/abc-defg-hij"), denylist: []))
        guard case let .meeting(app, bundleID) = meet else { return XCTFail("Meet not detected") }
        XCTAssertEqual(MeetingRules.key(app: app, bundleID: bundleID), "meet.google.com")
        let discord = MeetingDetector.evaluate(
            MeetingDetector.Evidence(
                microphoneHolders: ["com.hnc.Discord.helper"], frontmostBundleID: "com.hnc.Discord",
                frontmostURL: nil, denylist: ["com.hnc.Discord"], audioOutputs: ["com.hnc.Discord.helper"]))
        guard case let .meeting(app2, bundleID2) = discord else { return XCTFail("Discord not detected") }
        XCTAssertEqual(MeetingRules.key(app: app2, bundleID: bundleID2), "com.hnc.discord")
    }

    func testDisplayNames() {
        let none = { (_: String) -> String? in nil }
        XCTAssertEqual(MeetingRules.displayName(key: "meet.google.com", installedName: none), "Google Meet")
        XCTAssertEqual(MeetingRules.displayName(key: "us.zoom.xos", installedName: none), "Zoom")
        XCTAssertEqual(MeetingRules.displayName(key: "com.hnc.discord", installedName: none), "Discord")
        XCTAssertEqual(
            MeetingRules.displayName(key: "org.example.call", installedName: { _ in "Example" }), "Example")
        XCTAssertEqual(MeetingRules.displayName(key: "org.example.call", installedName: none), "org.example.call")
        XCTAssertEqual(
            MeetingRules.displayName(app: "us02web.zoom.us", bundleID: "com.google.Chrome"), "Zoom (web)")
        XCTAssertEqual(MeetingRules.displayName(app: "Slack", bundleID: "com.tinyspeck.slackmacgap"), "Slack")
    }

    // MARK: Decision

    func testTheDecisionFollowsTheModeThenTheRule() {
        let table: [(MeetingMode, MeetingRule?, MeetingDecision)] = [
            (.perApp, nil, .ask), (.perApp, .always, .record), (.perApp, .never, .ignore),
            (.everyCall, nil, .ask), (.everyCall, .always, .ask), (.everyCall, .never, .ask),
            (.off, nil, .ignore), (.off, .always, .ignore), (.off, .never, .ignore),
        ]
        for (mode, rule, expected) in table {
            XCTAssertEqual(
                MeetingRules.decide(mode: mode, rule: rule), expected, "\(mode) \(String(describing: rule))")
        }
    }

    func testYesToMeetNoToDiscord() {
        var book = MeetingRuleBook()
        book.set(.always, for: MeetingRules.key(app: "meet.google.com", bundleID: "com.google.Chrome"))
        book.set(.never, for: MeetingRules.key(app: "Discord", bundleID: "com.hnc.Discord"))
        func decision(_ app: String, _ bundleID: String) -> MeetingDecision {
            MeetingRules.decide(
                mode: .standard, rule: book.rule(for: MeetingRules.key(app: app, bundleID: bundleID)))
        }
        XCTAssertEqual(decision("meet.google.com", "com.apple.Safari"), .record, "any browser")
        XCTAssertEqual(decision("Discord", "com.hnc.Discord"), .ignore)
        XCTAssertEqual(decision("Zoom", "us.zoom.xos"), .ask, "no rule: asks")
        XCTAssertEqual(decision("discord.com", "com.google.Chrome"), .ask, "the site is its own rule")
    }

    // MARK: Store

    func testTheRuleStoreSurvivesARelaunchAndDropsUnknownValues() throws {
        let store = NativeAppStore(supportDirectory: directory)
        store.updateSettings { $0.meetingMode = .perApp }
        store.setMeetingRule(.always, for: "meet.google.com")
        store.setMeetingRule(.never, for: "com.hnc.Discord")
        XCTAssertEqual(store.settings.meetingRules.rule(for: "com.hnc.discord"), .never, "keys ignore case")

        let json = try storedJSON()
        XCTAssertEqual(json["meeting_mode"] as? String, "rules")
        XCTAssertEqual(
            json["meeting_rules"] as? [String: String],
            ["meet.google.com": "always", "com.hnc.discord": "never"])

        let relaunched = NativeAppStore(supportDirectory: directory)
        XCTAssertEqual(relaunched.settings.meetingMode, .perApp)
        XCTAssertEqual(relaunched.settings.meetingRules.rule(for: "meet.google.com"), .always)
        XCTAssertEqual(relaunched.settings.meetingRules.rule(for: "com.hnc.discord"), .never)

        relaunched.setMeetingRule(nil, for: "meet.google.com")
        XCTAssertNil(NativeAppStore(supportDirectory: directory).settings.meetingRules.rule(for: "meet.google.com"))

        let edited = try decode(
            #"{"schema_version":3,"meeting_mode":"rules","meeting_rules":{"zoom.us":"sometimes","US.ZOOM.XOS":"always","":"never"}}"#)
        XCTAssertEqual(edited.meetingRules.rules, ["us.zoom.xos": .always])
        XCTAssertEqual(try decode(#"{"meeting_mode":"rules","meeting_rules":"oops"}"#).meetingRules.rules, [:])
    }

    // MARK: Undo

    func testUndoPutsThePreviousRuleBack() {
        var book = MeetingRuleBook(["zoom.us": .never])
        let first = book.set(.always, for: "meet.google.com")
        XCTAssertEqual(first, MeetingRuleChange(key: "meet.google.com", previous: nil, new: .always))
        let second = book.set(.always, for: "zoom.us")
        book.undo(first)
        XCTAssertNil(book.rule(for: "meet.google.com"))
        book.undo(second)
        XCTAssertEqual(book.rule(for: "zoom.us"), .never, "back to the rule from before")

        // Changed again since (in Settings): Undo leaves the newer value.
        let third = book.set(.always, for: "us.zoom.xos")
        book.set(.never, for: "us.zoom.xos")
        book.undo(third)
        XCTAssertEqual(book.rule(for: "us.zoom.xos"), .never)

        let store = NativeAppStore(supportDirectory: directory)
        let change = store.setMeetingRule(.always, for: "com.apple.FaceTime")
        store.undoMeetingRule(change)
        XCTAssertNil(NativeAppStore(supportDirectory: directory).settings.meetingRules.rule(for: "com.apple.facetime"))
    }

    func testTheQuestionUndoCardAndBackToTheQuestion() {
        var flow = MeetingFlow()
        flow.ask(app: "meet.google.com", bundleID: "com.google.Chrome", remember: true)
        XCTAssertEqual(flow.popup, .askingRule(app: "meet.google.com", bundleID: "com.google.Chrome"))
        XCTAssertNil(flow.record(), "the per-app question has no plain Record")

        // Always: the rule is stored, recording starts, Undo stays up over the recording.
        let call = flow.choose(.always)
        XCTAssertEqual(call?.app, "meet.google.com")
        XCTAssertNil(flow.choose(.always), "one answer per question")
        let change = MeetingRuleChange(key: "meet.google.com", previous: nil, new: .always)
        flow.offerUndo(app: "meet.google.com", bundleID: "com.google.Chrome", change: change, recorded: true)
        flow.started(app: "meet.google.com")
        XCTAssertEqual(
            flow.popup,
            .undo(app: "meet.google.com", bundleID: "com.google.Chrome", change: change, recorded: true))

        let undone = flow.undo()
        XCTAssertEqual(undone?.change, change)
        XCTAssertEqual(undone?.recorded, true)
        XCTAssertEqual(flow.popup, .askingRule(app: "meet.google.com", bundleID: "com.google.Chrome"))
        XCTAssertNil(flow.undo(), "one undo")

        // Never: nothing records; Undo; then Not now declines this call as before.
        _ = flow.choose(.never)
        let never = MeetingRuleChange(key: "meet.google.com", previous: nil, new: .never)
        flow.offerUndo(app: "meet.google.com", bundleID: "com.google.Chrome", change: never, recorded: false)
        XCTAssertEqual(flow.undo()?.recorded, false)
        XCTAssertEqual(flow.notNow(), "meet.google.com")
        XCTAssertEqual(flow.popup, .hidden)

        // The Undo card times out like any other.
        flow.offerUndo(app: "Zoom", bundleID: "us.zoom.xos", change: never, recorded: false)
        flow.expire()
        XCTAssertNil(flow.undo())

        XCTAssertEqual(MeetingChoice.always.rule, .always)
        XCTAssertEqual(MeetingChoice.never.rule, .never)
        XCTAssertNil(MeetingChoice.once.rule)
        XCTAssertNil(MeetingChoice.notNow.rule)
        XCTAssertTrue(MeetingChoice.once.records)
        XCTAssertFalse(MeetingChoice.never.records)
    }

    // MARK: Questions

    func testTheQuestionsCardShowsOnlyWhenNothingElseIs() {
        var flow = MeetingFlow()
        XCTAssertTrue(flow.askQuestions())
        XCTAssertEqual(flow.popup, .questions)
        XCTAssertTrue(flow.answeredQuestions())
        XCTAssertFalse(flow.answeredQuestions())

        flow.ask(app: "Zoom", bundleID: "us.zoom.xos")
        XCTAssertFalse(flow.askQuestions(), "a call's question wins")
        _ = flow.notNow()
        flow.started(app: MeetingController.manualName)
        XCTAssertFalse(flow.askQuestions(), "never while recording")
        flow.stopped(byUser: false)

        // A start by hand takes the questions card down.
        XCTAssertTrue(flow.askQuestions())
        flow.started(app: MeetingController.manualName)
        XCTAssertEqual(flow.popup, .hidden)
    }

    func testTheQuestionsAreAskedOnceAndSkipKeepsTheDefault() throws {
        // A fresh install answers on the first-run sheet; an install from before has not.
        XCTAssertFalse(NativeSettings().meetingQuestionsAsked)
        XCTAssertFalse(try decode(#"{"schema_version":3,"privacy_choice_made":true}"#).meetingQuestionsAsked)

        let store = NativeAppStore(supportDirectory: directory)
        store.recordMeetingQuestions(mode: nil)
        XCTAssertTrue(store.settings.meetingQuestionsAsked)
        XCTAssertEqual(store.settings.meetingMode, .perApp, "Skip keeps the default")
        store.recordMeetingQuestions(mode: .everyCall)
        let relaunched = NativeAppStore(supportDirectory: directory)
        XCTAssertTrue(relaunched.settings.meetingQuestionsAsked)
        XCTAssertEqual(relaunched.settings.meetingMode, .everyCall)
        XCTAssertEqual(try storedJSON()["meeting_mode"] as? String, "every_call")
    }

    func testTheOptionNamesSayWhatTheyDo() {
        XCTAssertEqual(MeetingMode.allCases, [.perApp, .everyCall, .off], "the default comes first")
        XCTAssertEqual(MeetingMode.standard, .perApp)
        XCTAssertEqual(MeetingMode.allCases.map(\.title), ["Remember per app", "Ask every time", "Don't detect calls"])
    }

    // MARK: Migration

    func testOldSettingsMapOntoTheMode() throws {
        // The default-on prompt becomes the per-app default (no rules, so every app asks);
        // detection off is "off".
        XCTAssertEqual(try decode(#"{"schema_version":3,"meeting_capture":true}"#).meetingMode, .perApp)
        let prompted = try decode(#"{"schema_version":3,"meeting_capture":true,"meeting_auto_start":false}"#)
        XCTAssertEqual(prompted.meetingMode, .perApp)
        XCTAssertTrue(prompted.meetingRules.rules.isEmpty, "nothing records without a click")
        XCTAssertEqual(try decode(#"{"schema_version":3,"meeting_capture":false}"#).meetingMode, .off)
        XCTAssertEqual(
            try decode(#"{"schema_version":3,"meeting_capture":false,"meeting_auto_start":true}"#)
                .meetingMode, .off)

        // Auto-start recorded every detected call: per-app mode with every known app and site
        // on always, so it still records them without a click.
        let auto = try decode(#"{"schema_version":3,"meeting_capture":true,"meeting_auto_start":true}"#)
        XCTAssertEqual(auto.meetingMode, .perApp)
        for key in ["us.zoom.xos", "com.apple.facetime", "com.tinyspeck.slackmacgap", "com.hnc.discord",
            "meet.google.com", "zoom.us", "teams.microsoft.com"]
        {
            XCTAssertEqual(auto.meetingRules.rule(for: key), .always, key)
            XCTAssertEqual(MeetingRules.decide(mode: auto.meetingMode, rule: auto.meetingRules.rule(for: key)), .record)
        }
        XCTAssertEqual(
            auto.meetingRules.rules.count,
            MeetingDetector.callApps.count + MeetingDetector.chatAppsWithCalls.count
                + MeetingDetector.conferenceHosts.count)

        // Once written, `meeting_mode` wins over the old keys; unknown values ask every time.
        XCTAssertEqual(
            try decode(#"{"schema_version":3,"meeting_mode":"off","meeting_capture":true,"meeting_auto_start":true}"#)
                .meetingMode, .off)
        let kept = try decode(
            #"{"schema_version":3,"meeting_mode":"rules","meeting_rules":{"zoom.us":"never"},"meeting_auto_start":true}"#)
        XCTAssertEqual(kept.meetingMode, .perApp)
        XCTAssertEqual(kept.meetingRules.rules, ["zoom.us": .never], "no seeding over the owner's rules")
        XCTAssertEqual(try decode(#"{"meeting_mode":"every_call"}"#).meetingMode, .everyCall)
        XCTAssertEqual(try decode(#"{"meeting_mode":"sometimes"}"#).meetingMode, .everyCall)
        // `ask`, the default of the build that introduced the key, is the per-app default now.
        let previous = try decode(#"{"schema_version":3,"meeting_mode":"ask","meeting_rules":{"zoom.us":"never"}}"#)
        XCTAssertEqual(previous.meetingMode, .perApp)
        XCTAssertEqual(previous.meetingRules.rules, ["zoom.us": .never])
    }

    /// The build that introduced `meeting_mode` knew only ask | rules | off and read anything
    /// else as ask; the raw values stay compatible with that.
    func testTheStoredModeStaysSafeForTheBuildBefore() {
        XCTAssertEqual(MeetingMode.perApp.rawValue, "rules", "same meaning in the build before")
        XCTAssertEqual(MeetingMode.off.rawValue, "off")
        XCTAssertFalse(["ask", "rules", "off"].contains(MeetingMode.everyCall.rawValue),
            "unknown there, so it asks every time there too")
    }

    func testAnUpgradedFileKeepsWorkingForOlderBuilds() throws {
        try Data(#"{"schema_version":3,"meeting_capture":true,"meeting_auto_start":true,"extra":1}"#.utf8)
            .write(to: settingsURL)
        let store = NativeAppStore(supportDirectory: directory)
        XCTAssertEqual(store.settings.meetingMode, .perApp)
        store.updateSettings { $0.meetingMode = .everyCall }

        let json = try storedJSON()
        XCTAssertEqual(json["meeting_mode"] as? String, "every_call")
        // An older build reads these: detection on, and it asks (never records unasked).
        XCTAssertEqual(json["meeting_capture"] as? Bool, true)
        XCTAssertEqual(json["meeting_auto_start"] as? Bool, false)
        XCTAssertEqual(json["extra"] as? Int, 1, "unknown keys are kept")
        XCTAssertEqual(NativeAppStore(supportDirectory: directory).settings.meetingMode, .everyCall)
        store.updateSettings { $0.meetingMode = .perApp }
        XCTAssertEqual(try storedJSON()["meeting_auto_start"] as? Bool, false, "always rules stay ours")

        store.updateSettings { $0.meetingCapture = false }
        XCTAssertEqual(store.settings.meetingMode, .off)
        XCTAssertEqual(try storedJSON()["meeting_capture"] as? Bool, false)
        store.updateSettings { $0.meetingCapture = true }
        XCTAssertEqual(store.settings.meetingMode, .perApp, "turning detection back on is the default")
    }
}
