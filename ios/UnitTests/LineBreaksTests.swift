import XCTest

/// The Rust module's tests (crates/phonon-llm/src/linebreak.rs), on the same reference cases; mirrors
/// android/app/src/test/java/ai/phonon/keyboard/LineBreaksTest.kt, plus the Rust chat_join test (the Swift
/// port has chatJoin; Android does not).
final class LineBreaksTests: XCTestCase {
    func testMatchesThePythonReference() throws {
        let cases = try Cases.load("linebreak_cases")
        XCTAssertGreaterThanOrEqual(cases.count, 30)
        var changed = 0
        for c in cases {
            let id = c["id"] as? String ?? "?"
            let mode = try XCTUnwrap(c["mode"] as? String)
            let input = try XCTUnwrap(c["input"] as? String)
            let expected = try XCTUnwrap(c["expected"] as? String)
            if input != expected { changed += 1 }
            XCTAssertEqual(LineBreaks.apply(input, mode: mode), expected, "\(id) (\(mode))")
        }
        XCTAssertGreaterThanOrEqual(changed, 20, "most cases exercise a change")
    }

    func testCasualStaysInlineAndPlainTextIsUntouched() {
        XCTAssertEqual(LineBreaks.apply("1. eggs 2. milk", mode: "casual"), "1. eggs 2. milk")
        for mode in ["minimal", "casual", "formal", "prompt"] {
            XCTAssertEqual(LineBreaks.apply("Ship it after lunch.", mode: mode), "Ship it after lunch.", mode)
        }
        XCTAssertEqual(LineBreaks.apply("Steps 2. and 3. are optional", mode: "prompt"), "Steps 2. and 3. are optional")
    }

    func testStepsAndLetters() {
        XCTAssertEqual(LineBreaks.apply("Do this: 1. open the file 2. fix the bug 3. run tests", mode: "prompt"),
                       "Do this:\n1. open the file\n2. fix the bug\n3. run tests")
        XCTAssertEqual(LineBreaks.apply("Hi Sam, the build is green. Best regards, Chris", mode: "formal"),
                       "Hi Sam,\nThe build is green.\nBest regards,\nChris")
        XCTAssertEqual(LineBreaks.apply("Hi Sam, the build is green. Thanks, Chris", mode: "minimal"),
                       "Hi Sam, the build is green. Thanks, Chris")
        XCTAssertEqual(LineBreaks.apply("1. ship it 2. tag it", mode: "prompt"), "1. ship it\n2. tag it")
    }

    func testChatJoinUndoesOnlyGreetingAndSignOffLines() {
        XCTAssertEqual(LineBreaks.chatJoin("Hello team,\n\nThe build is green.\n\nBest regards,\nChris"),
                       "Hello team, the build is green. Best regards, Chris")
        XCTAssertEqual(LineBreaks.chatJoin("Dear Ms. Chen,\nThe build is green.\nWarm regards, Chris Lee"),
                       "Dear Ms. Chen, the build is green. Warm regards, Chris Lee")
        XCTAssertEqual(LineBreaks.chatJoin("Good morning, Sam.\n\nShip it today.\n\nThanks!"),
                       "Good morning, Sam. Ship it today. Thanks!")
        for (input, want) in [
            ("Good morning everyone,\n\nHere is the recipe.", "Good morning everyone, here is the recipe."),
            ("Hi Sam,\n\nBerlin is booked.", "Hi Sam, Berlin is booked."),
            ("Hi Sam,\nI'm in.", "Hi Sam, I'm in."),
            ("Hi team,\nIT is down.", "Hi team, IT is down."),
            ("Hi team,\nThere's cake.", "Hi team, there's cake."),
            ("Hi team,\nPlease review.", "Hi team, Please review."),
            ("Good morning, Sam.\nThe plan holds.", "Good morning, Sam. The plan holds."),
        ] {
            XCTAssertEqual(LineBreaks.chatJoin(input), want, input)
        }
        XCTAssertEqual(LineBreaks.chatJoin("Hi all,\n\nSteps:\n1. Grate it.\n2. Serve it.\n\nBest regards,\nChris"),
                       "Hi all, Steps:\n1. Grate it.\n2. Serve it.\nBest regards, Chris")
        for t in [
            "The build is green.\n\nShip it today.",
            "Hi Sam, the build is green. Thanks, Chris",
            "Steps:\n1. ship it\n2. tag it",
            "Hi,\n1. ship it\n2. tag it",
            "Ship it.\nThanks for the review of the release notes",
            "Hi Sam,",
        ] {
            let want = t.hasPrefix("Hi,\n") ? "Hi, " + t.dropFirst(4) : t
            XCTAssertEqual(LineBreaks.chatJoin(t), want, t)
        }
    }
}
