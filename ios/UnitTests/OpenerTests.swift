import XCTest

/// The Rust module's tests (crates/phonon-llm/src/opener.rs), on the same reference cases; mirrors
/// android/app/src/test/java/ai/phonon/keyboard/OpenerTest.kt.
final class OpenerTests: XCTestCase {
    /// Prompt mode, the one most dictations use; the word rules are the same in every mode that restores.
    private func restore(_ raw: String, _ refined: String) -> String {
        Opener.restore(raw: raw, refined: refined, mode: "prompt")
    }

    func testMatchesTheReferenceCases() throws {
        let cases = try Cases.load("opener_cases")
        XCTAssertEqual(cases.count, 36, "every shared case")
        var ran = 0
        for c in cases {
            let id = c["id"] as? String ?? "?"
            let mode = try XCTUnwrap(c["mode"] as? String)
            let raw = try XCTUnwrap(c["raw"] as? String)
            let refined = try XCTUnwrap(c["refined"] as? String)
            let expected = try XCTUnwrap(c["expected"] as? String)
            XCTAssertEqual(Opener.restore(raw: raw, refined: refined, mode: mode), expected, "\(id) (\(mode))")
            ran += 1
        }
        XCTAssertEqual(ran, 36)
    }

    func testCasualMatchesTheRefinedCase() {
        XCTAssertEqual(Opener.restore(raw: "okay see you at five", refined: "see you at five", mode: "casual"),
                       "okay, see you at five")
        XCTAssertEqual(Opener.restore(raw: "and Sam is coming", refined: "sam is coming", mode: "casual"),
                       "and sam is coming")
        XCTAssertEqual(Opener.restore(raw: "okay see you at five", refined: "See you at five.", mode: "casual"),
                       "Okay, see you at five.")
        XCTAssertEqual(Opener.restore(raw: "so see you at five", refined: "see you at five", mode: "casual"),
                       "see you at five")
    }

    func testFormalVerbatimAndUnknownModesNeverRestore() {
        for mode in ["formal", "verbatim", "raw", "", "unknown"] {
            XCTAssertEqual(Opener.restore(raw: "okay the build is green", refined: "The build is green.", mode: mode),
                           "The build is green.", mode)
        }
    }

    func testEachOpenerAndOkBecomesOkay() {
        XCTAssertEqual(restore("okay the build is green", "The build is green."), "Okay, the build is green.")
        XCTAssertEqual(restore("Well, it depends on the load", "It depends on the load."), "Well, it depends on the load.")
        XCTAssertEqual(restore("And then we ship it", "Then we ship it."), "And then we ship it.")
        XCTAssertEqual(restore("Now, let's look at the logs", "Let's look at the logs."), "Now, let's look at the logs.")
        XCTAssertEqual(restore("OK, run the tests", "Run the tests."), "Okay, run the tests.")
        XCTAssertEqual(restore("ok. Next sentence", "Next sentence."), "Okay, next sentence.")
        XCTAssertEqual(restore("okay run the tests", "OK, run the tests."), "OK, run the tests.")
    }

    func testFirstWordCapitalKeptWhereItBelongs() {
        XCTAssertEqual(restore("Okay, Sam said the build is green", "Sam said the build is green."),
                       "Okay, Sam said the build is green.")
        XCTAssertEqual(restore("okay I think so", "I think so."), "Okay, I think so.")
        XCTAssertEqual(restore("and I'm done", "I'm done."), "And I'm done.")
        XCTAssertEqual(restore("well API keys rotate", "API keys rotate."), "Well, API keys rotate.")
        XCTAssertEqual(restore("okay iPhone builds pass", "iPhone builds pass."), "Okay, iPhone builds pass.")
        XCTAssertEqual(restore("now GitHub is down", "GitHub is down."), "Now, GitHub is down.")
        XCTAssertEqual(restore("Okay. The build is green", "The build is green."), "Okay, the build is green.")
    }

    func testSoOtherWordsEmptyAndOpenerOnlyInputs() {
        for (raw, refined) in [
            ("So the build is green", "The build is green."),
            ("Okayish results today", "Results today."),
            ("Nowhere to go", "There is nowhere to go."),
            ("Well it depends", "well, it depends."),
        ] {
            XCTAssertEqual(restore(raw, refined), refined, raw)
        }
        XCTAssertEqual(restore("Okay, ship it", ""), "")
        XCTAssertEqual(restore("Okay, ship it", "  "), "  ")
        XCTAssertEqual(restore("", "Ship it."), "Ship it.")
        XCTAssertEqual(restore("Okay.", "Okay."), "Okay.")
        XCTAssertEqual(restore("Okay.", "Sure."), "Sure.")
        XCTAssertEqual(restore("and", "Hmm."), "Hmm.")
        XCTAssertEqual(restore("okay ship it", "okay ship it"), "okay ship it")
    }

    func testRunsBeforeLineBreaks() {
        XCTAssertEqual(
            LineBreaks.apply(restore("okay do this one open the file two fix the bug",
                                     "Do this: 1. open the file 2. fix the bug"), mode: "prompt"),
            "Okay, do this:\n1. open the file\n2. fix the bug")
    }
}
