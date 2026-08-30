import XCTest

@testable import PhononBar

final class CompetingAppsTests: XCTestCase {
    // MARK: Table

    func testTableLooksUpMainAndHelperBundles() {
        XCTAssertEqual(CompetingAppTable.app(forBundleID: "com.electron.wispr-flow")?.name, "Wispr Flow")
        XCTAssertEqual(
            CompetingAppTable.app(forBundleID: "com.superduper.superwhisper-setapp")?.name,
            "superwhisper")
        XCTAssertNil(CompetingAppTable.app(forBundleID: "com.apple.Safari"))
        XCTAssertTrue(CompetingAppTable.isHelper(bundleID: "com.electron.wispr-flow.accessibility-mac-app"))
        XCTAssertFalse(CompetingAppTable.isHelper(bundleID: "com.electron.wispr-flow"))
        XCTAssertTrue(CompetingAppTable.watchedBundleIDs.contains("now.typeless.desktop"))
    }

    func testTableHasNoDuplicateBundleIDs() {
        let all = CompetingAppTable.apps.flatMap { $0.bundleIDs + $0.helperBundleIDs }
        XCTAssertEqual(all.count, Set(all).count)
    }

    // MARK: Chords

    func testKeyCodeChordsClassify() {
        XCTAssertEqual(CompetitorChord.fromKeyCodes([63]).kind, .fn)
        XCTAssertEqual(CompetitorChord.fromKeyCodes([49, 63]).kind, .fn)
        XCTAssertEqual(CompetitorChord.fromKeyCodes([61]).kind, .option)
        XCTAssertEqual(CompetitorChord.fromKeyCodes([58, 49]).kind, .option)
        XCTAssertEqual(CompetitorChord.fromKeyCodes([59, 49]).kind, .controlSpace)
        // Control+fn (Wispr "lens") does not fire Phonon's fn-only hold.
        XCTAssertEqual(CompetitorChord.fromKeyCodes([59, 63]).kind, .other)
        // Option+1 involves another key.
        XCTAssertEqual(CompetitorChord.fromKeyCodes([18, 58]).kind, .other)
        XCTAssertEqual(CompetitorChord.fromKeyCodes([53]).kind, .other)
    }

    func testTokenChordsClassify() {
        XCTAssertEqual(CompetitorChord.fromTokens(["Fn"]).kind, .fn)
        XCTAssertEqual(CompetitorChord.fromTokens(["Fn", "Space"]).kind, .fn)
        XCTAssertEqual(CompetitorChord.fromTokens(["Alt", "Space"]).kind, .option)
        XCTAssertEqual(CompetitorChord.fromTokens(["Control", "Space"]).kind, .controlSpace)
        XCTAssertEqual(CompetitorChord.fromTokens(["Meta", "Control", "KeyV"]).kind, .other)
        XCTAssertEqual(CompetitorChord.fromTokens(["Escape"]).kind, .other)
    }

    func testModifierBitChords() {
        XCTAssertEqual(CompetitorChord.fromKeyCode(63, modifierFlags: 0).kind, .fn)
        XCTAssertEqual(CompetitorChord.fromKeyCode(49, modifierFlags: 1 << 18).kind, .controlSpace)
        XCTAssertEqual(CompetitorChord.fromKeyCode(49, modifierFlags: 1 << 19).kind, .option)
        XCTAssertEqual(CompetitorChord.fromKeyCode(49, carbonModifiers: 4096).kind, .controlSpace)
        XCTAssertEqual(CompetitorChord.fromKeyCode(49, carbonModifiers: 2048).kind, .option)
        XCTAssertEqual(CompetitorChord.fromKeyCode(9, carbonModifiers: 256).kind, .other)
    }

    // MARK: Readers

    private func json(_ text: String) -> Any {
        try! JSONSerialization.jsonObject(with: Data(text.utf8))
    }

    func testWisprConfigReader() {
        let config = json(
            #"""
            {"prefs":{"user":{"openAtLogin":true,"shortcuts":{
              "53":"dismiss","63":"ptt","49+63":"popo","59+63":"lens","18+58":"polish"}}}}
            """#)
        guard case .chords(let chords) = CompetitorHotkeyReader.wispr(config: config) else {
            return XCTFail("expected chords")
        }
        // Keys are read in sorted order: 18+58, 49+63, 53, 59+63, 63.
        XCTAssertEqual(chords.map(\.kind), [.other, .fn, .other, .other, .fn])
        XCTAssertEqual(CompetitorHotkeyReader.wisprOpensAtLogin(config: config), true)
        XCTAssertEqual(CompetitorHotkeyReader.wispr(config: json(#"{"prefs":{}}"#)), .unreadable)
        XCTAssertEqual(CompetitorHotkeyReader.wispr(config: ["nope"]), .unreadable)
    }

    func testAquaSettingsReader() {
        let settings = json(
            #"""
            {"hotkeys":[{"keys":"Fn","action":"activate"},{"keys":"Fn+Space","action":"lock"},
             {"keys":"Meta+Control+KeyV","action":"paste_last_transcript"},
             {"keys":"Escape","action":"cancel"},{"keys":"","action":"open_note"}]}
            """#)
        guard case .chords(let chords) = CompetitorHotkeyReader.aqua(settings: settings) else {
            return XCTFail("expected chords")
        }
        XCTAssertEqual(chords.map(\.kind), [.fn, .fn, .other, .other])
        XCTAssertEqual(CompetitorHotkeyReader.aqua(settings: json(#"{"version":1}"#)), .unreadable)
    }

    func testVoiceInkDefaultsReader() {
        XCTAssertEqual(
            CompetitorHotkeyReader.voiceInk(shortcut: #"{"keyCode":63,"modifierFlags":0}"#),
            .chords([CompetitorChord.fromKeyCodes([63])]))
        XCTAssertEqual(
            CompetitorHotkeyReader.voiceInk(
                shortcut: Data(#"{"carbonKeyCode":49,"carbonModifiers":4096}"#.utf8)),
            .chords([CompetitorChord.fromKeyCode(49, carbonModifiers: 4096)]))
        XCTAssertEqual(
            CompetitorHotkeyReader.voiceInk(shortcut: ["keyCode": 61, "modifierFlags": 0]),
            .chords([CompetitorChord.fromKeyCodes([61])]))
        XCTAssertEqual(CompetitorHotkeyReader.voiceInk(shortcut: nil), .unreadable)
        XCTAssertEqual(CompetitorHotkeyReader.voiceInk(shortcut: "not json"), .unreadable)
        XCTAssertEqual(CompetitorHotkeyReader.voiceInk(shortcut: #"{"other":1}"#), .unreadable)
    }

    func testReadFromMissingHomeIsUnreadable() {
        let home = URL(fileURLWithPath: "/nonexistent/phonon-test-home")
        XCTAssertEqual(CompetitorHotkeyReader.read(.wisprConfig, home: home), .unreadable)
        XCTAssertEqual(CompetitorHotkeyReader.read(.aquaSettings, home: home), .unreadable)
        XCTAssertEqual(CompetitorHotkeyReader.read(.unknown, home: home), .unreadable)
        XCTAssertNil(CompetitorHotkeyReader.wisprOpensAtLogin(home: home))
    }

    // MARK: Decision

    func testConflictMatchesPhononShortcutMode() {
        let fn = CompetitorHotkey.chords([CompetitorChord.fromKeyCodes([63])])
        let option = CompetitorHotkey.chords([CompetitorChord.fromKeyCodes([61])])
        let controlSpace = CompetitorHotkey.chords([CompetitorChord.fromKeyCodes([59, 49])])
        let escape = CompetitorHotkey.chords([CompetitorChord.fromKeyCodes([53])])

        // Wispr on fn while Phonon is on the default Right Option + Control Space: no prompt.
        XCTAssertNil(CompetitorPolicy.conflict(hotkey: fn, phononMode: "both"))
        XCTAssertEqual(
            CompetitorPolicy.conflict(hotkey: fn, phononMode: "fn"),
            .sameHotkey(CompetitorChord.fromKeyCodes([63])))
        XCTAssertNotNil(CompetitorPolicy.conflict(hotkey: fn, phononMode: "fn_and_control_space"))
        XCTAssertNotNil(CompetitorPolicy.conflict(hotkey: option, phononMode: "both"))
        XCTAssertNotNil(CompetitorPolicy.conflict(hotkey: option, phononMode: "right_option"))
        XCTAssertNil(CompetitorPolicy.conflict(hotkey: option, phononMode: "fn"))
        XCTAssertNotNil(CompetitorPolicy.conflict(hotkey: controlSpace, phononMode: "both"))
        XCTAssertNotNil(CompetitorPolicy.conflict(hotkey: controlSpace, phononMode: "control_space"))
        XCTAssertNil(CompetitorPolicy.conflict(hotkey: controlSpace, phononMode: "right_option"))
        XCTAssertNil(CompetitorPolicy.conflict(hotkey: escape, phononMode: "both"))
        // Unknown shortcut: a clash cannot be ruled out, so ask.
        XCTAssertEqual(CompetitorPolicy.conflict(hotkey: .unreadable, phononMode: "both"), .unreadable)
    }

    func testPromptGate() {
        let id = "com.electron.wispr-flow"
        XCTAssertTrue(
            CompetitorPolicy.shouldPrompt(bundleID: id, muted: [], promptCount: 0, relaunch: false))
        XCTAssertFalse(
            CompetitorPolicy.shouldPrompt(bundleID: id, muted: [id], promptCount: 0, relaunch: false))
        XCTAssertFalse(
            CompetitorPolicy.shouldPrompt(bundleID: id, muted: [id], promptCount: 0, relaunch: true))
        // Once per session, unless the app relaunched: then one more time only.
        XCTAssertFalse(
            CompetitorPolicy.shouldPrompt(bundleID: id, muted: [], promptCount: 1, relaunch: false))
        XCTAssertTrue(
            CompetitorPolicy.shouldPrompt(bundleID: id, muted: [], promptCount: 1, relaunch: true))
        XCTAssertFalse(
            CompetitorPolicy.shouldPrompt(bundleID: id, muted: [], promptCount: 2, relaunch: true))
        XCTAssertTrue(
            CompetitorPolicy.shouldPrompt(
                bundleID: id, muted: ["com.electron.aqua-voice"], promptCount: 0, relaunch: false))
    }

    func testHelperMatchNeedsNameAndPath() {
        let helper = CompetitorHelperProcess(name: "AquaMacOSBridge", pathFragment: "/Aqua Voice.app/")
        XCTAssertTrue(
            CompetitorPolicy.helperMatches(
                helper, processName: "AquaMacOSBridge",
                executablePath: "/Applications/Aqua Voice.app/Contents/MacOS/AquaMacOSBridge"))
        XCTAssertFalse(
            CompetitorPolicy.helperMatches(
                helper, processName: "AquaMacOSBridge", executablePath: "/tmp/AquaMacOSBridge"))
        XCTAssertFalse(
            CompetitorPolicy.helperMatches(
                helper, processName: "AquaMacOSBridge2",
                executablePath: "/Applications/Aqua Voice.app/Contents/MacOS/AquaMacOSBridge2"))
    }

    func testNoTableHelperMatchesAnUnrelatedProcess() {
        for helper in CompetingAppTable.apps.flatMap(\.helperProcesses) {
            XCTAssertTrue(CompetitorProcesses.pids(matching: helper).allSatisfy { $0 > 0 })
            XCTAssertFalse(
                CompetitorPolicy.helperMatches(
                    helper, processName: "xctest", executablePath: "/usr/bin/xctest"))
        }
    }

    // MARK: Prompt text

    @MainActor
    func testInformativeTextNamesAppHotkeyAndLoginReturn() throws {
        let wispr = try XCTUnwrap(CompetingAppTable.app(forBundleID: "com.electron.wispr-flow"))
        let finding = CompetitorFinding(
            app: wispr, reason: .sameHotkey(CompetitorChord.fromKeyCodes([63])), running: [])
        let text = CompetitorCoordinator.informativeText(for: [finding], manual: false)
        XCTAssertTrue(text.contains("Wispr Flow is running and also records on Globe (fn)"))
        XCTAssertTrue(text.contains("opens at login by default"))
        XCTAssertTrue(text.contains("never changes the app's login items"))

        let typeless = try XCTUnwrap(CompetingAppTable.app(forBundleID: "now.typeless.desktop"))
        let unknown = CompetitorFinding(app: typeless, reason: .unreadable, running: [])
        XCTAssertTrue(
            CompetitorCoordinator.informativeText(for: [unknown], manual: false)
                .contains("could not read its shortcut"))
        XCTAssertFalse(
            CompetitorCoordinator.informativeText(for: [unknown], manual: true)
                .contains("could not read its shortcut"))
    }

    // MARK: Settings persistence

    @MainActor
    func testMutedCompetitorsPersistInSettings() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "phonon-competitors-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let store = NativeAppStore(supportDirectory: directory)
        XCTAssertEqual(store.settings.competitorQuitMuted, [])
        store.updateSettings { $0.competitorQuitMuted = ["com.electron.wispr-flow"] }

        let data = try Data(contentsOf: directory.appendingPathComponent("settings.json"))
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["competitor_quit_muted"] as? [String], ["com.electron.wispr-flow"])

        let reloaded = NativeAppStore(supportDirectory: directory)
        XCTAssertEqual(reloaded.settings.competitorQuitMuted, ["com.electron.wispr-flow"])
    }
}
