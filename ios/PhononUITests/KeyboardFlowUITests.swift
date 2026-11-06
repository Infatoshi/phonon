import XCTest

/// The whole keyboard flow in the simulator, the way he will use it on the phone: Phonon
/// keyboard added with Full Access in Settings, selected with the globe key in Messages, mic
/// tapped, Phonon opened by the keyboard, back to Messages, stop tapped, text inserted.
/// The microphone is a fixture recording (`-PhononFixtureAudio`, debug builds only). Stills
/// and a step log go to PHONON_SHOTS_DIR when set (pass it as TEST_RUNNER_PHONON_SHOTS_DIR).
final class KeyboardFlowUITests: XCTestCase {
    private let env = ProcessInfo.processInfo.environment
    private var out: URL? { env["PHONON_SHOTS_DIR"].map { URL(fileURLWithPath: $0) } }
    private var log: [[String: String]] = []
    private let started = Date()

    func testDictateIntoMessagesThroughThePhononKeyboard() throws {
        continueAfterFailure = false
        let fixture = env["PHONON_FIXTURE"] ?? "01-plain"
        let expected = (env["PHONON_EXPECT"] ?? "dinner tomorrow").lowercased()

        // Phonon running, models loaded, the fixture as its microphone.
        let phonon = XCUIApplication()
        phonon.launchArguments += ["-onboarded", "YES", "-PhononFixtureAudio", fixture]
        phonon.launch()
        XCTAssertTrue(phonon.staticTexts["Models ready"].waitForExistence(timeout: 120),
                      dump(phonon, "phonon-models"))
        step("phonon launched, models ready")
        // Closed, so the first mic tap has to open it, as after a reboot or a long idle.
        // PHONON_KEEP_RUNNING=1 instead leaves it finishing planted unfinished dictations,
        // so the keyboard's start arrives while recovery is busy.
        let race = env["PHONON_KEEP_RUNNING"] == "1"
        if !race {
            phonon.terminate()
            step("Phonon closed")
        }

        try enableKeyboard()

        // Messages, a new message, the body field.
        let messages = XCUIApplication(bundleIdentifier: "com.apple.MobileSMS")
        messages.terminate()
        messages.launch()
        for label in ["Continue", "Not Now", "OK"] where messages.buttons[label].waitForExistence(timeout: 2) {
            messages.buttons[label].tap()
        }
        // A conversation: the simulator's Messages lists two seeded ones and, on iOS 26, shows
        // no compose button in the accessibility tree.
        let compose = messages.buttons.matching(NSPredicate(format: "label CONTAINS[c] 'compose' OR identifier CONTAINS[c] 'compose'")).firstMatch
        let thread = messages.collectionViews["ConversationList"].cells.firstMatch
        if thread.waitForExistence(timeout: 10) { thread.tap() } else {
            XCTAssertTrue(compose.waitForExistence(timeout: 5), dump(messages, "messages-compose"))
            compose.tap()
        }
        let body = messages.descendants(matching: .any).matching(identifier: "messageBodyField").firstMatch
        XCTAssertTrue(body.waitForExistence(timeout: 10), dump(messages, "messages-body"))
        body.tap()
        step("Messages body field focused")

        // The globe key until the Phonon keyboard shows.
        var mic = keyboardButton(messages, "Start dictation")
        for _ in 0..<5 where !mic.waitForExistence(timeout: 2) {
            if messages.buttons["Continue"].exists { messages.buttons["Continue"].tap() }
            let globe = messages.buttons["Next keyboard"]
            XCTAssertTrue(globe.waitForExistence(timeout: 5), dump(messages, "messages-globe"))
            globe.tap()
            mic = keyboardButton(messages, "Start dictation")
        }
        XCTAssertTrue(mic.waitForExistence(timeout: 5), dump(messages, "messages-phonon-keyboard"))
        step("Phonon keyboard selected with the globe key")
        Thread.sleep(forTimeInterval: 1.5)
        shot("sim-keyboard-messages-light")
        XCUIDevice.shared.appearance = .dark
        Thread.sleep(forTimeInterval: 2)
        shot("sim-keyboard-messages-dark")
        XCUIDevice.shared.appearance = .light
        Thread.sleep(forTimeInterval: 1.5)

        // Mic: no warm session yet, so the keyboard opens Phonon, which starts recording.
        // A draft can stay in the seeded thread between runs: only new text counts.
        let before = (body.value as? String) ?? ""
        step("field before: \(before)")
        mic.tap()
        step("mic tapped")
        if !race {
            XCTAssertTrue(phonon.wait(for: .runningForeground, timeout: 20), dump(messages, "handoff"))
            step("Phonon opened by the keyboard")
            Thread.sleep(forTimeInterval: 1.5)
            shot("sim-keyboard-handoff-light")
        }
        Thread.sleep(forTimeInterval: Double(env["PHONON_FIXTURE_SECONDS"] ?? "4") ?? 4)

        // Back to Messages, as the app-name breadcrumb would, and stop from the keyboard.
        messages.activate()
        let stop = keyboardButton(messages, "Stop dictation")
        XCTAssertTrue(stop.waitForExistence(timeout: 30), dump(messages, "stop-button"))
        step("back in Messages, keyboard shows recording")
        if race { Thread.sleep(forTimeInterval: Double(env["PHONON_FIXTURE_SECONDS"] ?? "4") ?? 4) }
        shot(race ? "sim-keyboard-race-recording-light" : "sim-keyboard-recording-light")
        stop.tap()
        step("stop tapped")

        let inserted = NSPredicate { _, _ in
            let now = (body.value as? String) ?? ""
            return now != before && now.lowercased().contains(expected)
        }
        let wait = XCTNSPredicateExpectation(predicate: inserted, object: nil)
        let result = XCTWaiter().wait(for: [wait], timeout: 120)
        step("text in Messages: \((body.value as? String) ?? "")")
        Thread.sleep(forTimeInterval: 1)
        shot(race ? "sim-keyboard-race-inserted-light" : "sim-keyboard-inserted-light")
        XCTAssertEqual(result, .completed, dump(messages, "inserted"))

        // Second dictation: the session is still open, so it records without leaving Messages.
        // First let Phonon finish any dictation a kill left behind (it runs after this one).
        let busy = NSPredicate { _, _ in
            !messages.staticTexts.matching(NSPredicate(format: "label IN {'Transcribing…', 'Cleaning up…'}")).firstMatch.exists
        }
        _ = XCTWaiter().wait(for: [XCTNSPredicateExpectation(predicate: busy, object: nil)], timeout: 120)
        let first = (body.value as? String) ?? ""
        let mic2 = keyboardButton(messages, "Start dictation")
        XCTAssertTrue(mic2.waitForExistence(timeout: 10), dump(messages, "mic-again"))
        mic2.tap()
        step("second mic tap")
        XCTAssertTrue(stop.waitForExistence(timeout: 10), dump(messages, "stop-again"))
        XCTAssertEqual(messages.state, .runningForeground, "the second dictation left Messages")
        step("second dictation recording, Messages still in front")
        Thread.sleep(forTimeInterval: Double(env["PHONON_FIXTURE_SECONDS"] ?? "4") ?? 4)
        stop.tap()
        step("second stop tapped")
        let again = NSPredicate { _, _ in
            let now = ((body.value as? String) ?? "").lowercased()
            return now.components(separatedBy: expected).count > first.lowercased().components(separatedBy: expected).count
        }
        let result2 = waitUntil(again)
        step("text in Messages: \((body.value as? String) ?? "")")
        shot(race ? "sim-keyboard-race-inserted-twice-light" : "sim-keyboard-inserted-twice-light")
        XCTAssertEqual(result2, .completed, dump(messages, "inserted-again"))
    }

    private func waitUntil(_ predicate: NSPredicate) -> XCTWaiter.Result {
        XCTWaiter().wait(for: [XCTNSPredicateExpectation(predicate: predicate, object: nil)], timeout: 120)
    }

    /// Settings > General > Keyboard > Keyboards: add Phonon, then Allow Full Access.
    private func enableKeyboard() throws {
        let settings = XCUIApplication(bundleIdentifier: "com.apple.Preferences")
        settings.terminate()
        settings.launch()
        try tapLabel(settings, "General")
        try tapLabel(settings, "Keyboard")
        try tapLabel(settings, "Keyboards", prefix: true)
        let row = settings.cells.matching(NSPredicate(format: "label BEGINSWITH 'Phonon'")).firstMatch
        if !row.waitForExistence(timeout: 3) {
            try tapLabel(settings, "Add New Keyboard", prefix: true)
            try tapLabel(settings, "Phonon")
            step("Phonon keyboard added in Settings")
        }
        XCTAssertTrue(row.waitForExistence(timeout: 5), dump(settings, "settings-row"))
        row.tap()
        let full = settings.switches.matching(NSPredicate(format: "label BEGINSWITH 'Allow Full Access'")).firstMatch
        XCTAssertTrue(full.waitForExistence(timeout: 5), dump(settings, "settings-full-access"))
        if (full.value as? String) != "1" {
            full.coordinate(withNormalizedOffset: CGVector(dx: 0.93, dy: 0.5)).tap()
            let allow = settings.alerts.buttons["Allow"]
            if allow.waitForExistence(timeout: 5) { allow.tap() }
        }
        XCTAssertEqual(full.value as? String, "1", dump(settings, "settings-full-access-on"))
        step("Full Access on")
        shot("sim-settings-full-access-light")
    }

    private func tapLabel(_ app: XCUIApplication, _ label: String, prefix: Bool = false) throws {
        let predicate = prefix ? NSPredicate(format: "label BEGINSWITH %@", label) : NSPredicate(format: "label == %@", label)
        let element = app.cells.matching(predicate).firstMatch.exists ? app.cells.matching(predicate).firstMatch
            : app.descendants(matching: .any).matching(predicate).firstMatch
        var tries = 0
        while !(element.waitForExistence(timeout: 2) && element.isHittable), tries < 4 {
            app.swipeUp()
            tries += 1
        }
        XCTAssertTrue(element.exists, dump(app, "tap-\(label)"))
        element.tap()
    }

    /// A key of the Phonon keyboard: the extension's views show up in the host app's tree.
    private func keyboardButton(_ host: XCUIApplication, _ label: String) -> XCUIElement {
        host.buttons[label]
    }

    override func tearDown() {
        writeLog()
        super.tearDown()
    }

    private func step(_ text: String) {
        log.append(["t": String(format: "%.1f", Date().timeIntervalSince(started)), "step": text])
    }

    private func shot(_ name: String) {
        let png = XCUIScreen.main.screenshot().pngRepresentation
        let attachment = XCTAttachment(data: png, uniformTypeIdentifier: "public.png")
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
        if let out { try? png.write(to: out.appendingPathComponent("\(name).png")) }
    }

    /// The accessibility tree at a failure, so a broken step says what was on screen.
    private func dump(_ app: XCUIApplication, _ name: String) -> String {
        let tree = app.debugDescription
        if let out { try? tree.write(to: out.appendingPathComponent("tree-\(name).txt"), atomically: true, encoding: .utf8) }
        return "step \(name) failed; tree in tree-\(name).txt"
    }

    private func writeLog() {
        guard let out, let data = try? JSONSerialization.data(withJSONObject: log, options: [.prettyPrinted]) else { return }
        try? data.write(to: out.appendingPathComponent("keyboard-flow-steps.json"))
    }
}
