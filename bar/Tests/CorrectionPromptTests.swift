import XCTest

@testable import PhononBar

@MainActor
final class CorrectionPromptTests: XCTestCase {
    func testTheTrainedWordingIsTheDefaultWhetherSavedAsEmptyOrAsItself() {
        XCTAssertEqual(CorrectionPrompt.setting(for: ""), "")
        XCTAssertEqual(CorrectionPrompt.setting(for: "  \n"), "")
        XCTAssertEqual(CorrectionPrompt.setting(for: CorrectionPrompt.trained + "\n"), "")
        XCTAssertEqual(CorrectionPrompt.setting(for: "  Fix grammar. "), "Fix grammar.")
        XCTAssertEqual(CorrectionPrompt.effective(""), CorrectionPrompt.trained)
        XCTAssertEqual(CorrectionPrompt.effective("  Fix grammar. "), "Fix grammar.")
        XCTAssertEqual(
            CorrectionPrompt.effective(String(repeating: "x", count: CorrectionPrompt.maxLength + 1)),
            CorrectionPrompt.trained)
    }

    func testTheSampleHasTheMistakesACorrectionShouldClear() {
        let sample = CorrectionPrompt.sample
        for mistake in ["its going", "should of", "the the", "get hub", "your welcome", "thanks alot"] {
            XCTAssertTrue(sample.contains(mistake), mistake)
        }
        XCTAssertLessThan(sample.count, CorrectionPrompt.maxLength)
    }

    func testATryCarriesItsOwnPromptAndModeToThePolishCommand() {
        let command = CorrectionPrompt.testCommand(
            id: "prompt-test-1", text: "hello there", prompt: " Fix it. ", mode: .formal)
        XCTAssertEqual(command["cmd"] as? String, "polish")
        XCTAssertEqual(command["id"] as? String, "prompt-test-1")
        XCTAssertEqual(command["text"] as? String, "hello there")
        XCTAssertEqual(command["refiner_prompt"] as? String, "Fix it.")
        XCTAssertEqual(command["writing_mode"] as? String, "formal")
        // the trained wording is sent as empty: "the default", not the saved prompt
        let trained = CorrectionPrompt.testCommand(
            id: "x", text: "a", prompt: CorrectionPrompt.trained, mode: .minimal)
        XCTAssertEqual(trained["refiner_prompt"] as? String, "")
    }

    func testSettingsKeepTheSavedPromptAcrossAReload() throws {
        var settings = NativeSettings()
        settings.refinerPrompt = "Fix grammar."
        let data = try JSONEncoder().encode(settings)
        XCTAssertEqual(try JSONDecoder().decode(NativeSettings.self, from: data).refinerPrompt, "Fix grammar.")
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(json["refiner_prompt"] as? String, "Fix grammar.")
        let old = try JSONDecoder().decode(NativeSettings.self, from: Data(#"{"schema_version":2}"#.utf8))
        XCTAssertEqual(old.refinerPrompt, "")
    }

    func testTesterSendsOneCommandAndAcceptsOnlyItsOwnAnswer() {
        let tester = PromptTester()
        var sent: [[String: Any]] = []
        tester.send = { sent.append($0) }
        tester.isReady = { true }

        tester.run(text: " hi there ", prompt: "P", mode: .casual)
        XCTAssertTrue(tester.running)
        XCTAssertEqual(sent.count, 1)
        let id = sent[0]["id"] as? String ?? ""
        XCTAssertTrue(id.hasPrefix(PromptTester.idPrefix))
        XCTAssertEqual(sent[0]["text"] as? String, "hi there")

        // somebody else's answers are not this tester's
        XCTAssertFalse(tester.handleResult(id: "p3", text: "x", seconds: 1))
        XCTAssertFalse(tester.handleError(id: nil, message: "boom"))
        XCTAssertTrue(tester.running)
        // an answer to a try that was already replaced is swallowed, not shown
        XCTAssertTrue(tester.handleResult(id: PromptTester.idPrefix + "99", text: "late", seconds: 1))
        XCTAssertTrue(tester.running)

        XCTAssertTrue(tester.handleResult(id: id, text: "Hi there.", seconds: 0.4))
        XCTAssertEqual(tester.state, .done(text: "Hi there.", seconds: 0.4))
    }

    func testTesterReportsAnEngineErrorAndAModelThatIsNotReady() {
        let tester = PromptTester()
        var sent: [[String: Any]] = []
        tester.send = { sent.append($0) }
        tester.isReady = { false }
        tester.run(text: "hello", prompt: "", mode: .minimal)
        XCTAssertTrue(sent.isEmpty)
        guard case .failed(let message) = tester.state else { return XCTFail("\(tester.state)") }
        XCTAssertTrue(message.contains("loading"))

        tester.isReady = { true }
        tester.run(text: "hello", prompt: "", mode: .minimal)
        let id = sent[0]["id"] as? String ?? ""
        XCTAssertTrue(tester.handleError(id: id, message: "the correction model is not ready yet"))
        XCTAssertEqual(tester.state, .failed("the correction model is not ready yet"))

        tester.run(text: "   ", prompt: "", mode: .minimal)
        XCTAssertEqual(tester.state, .failed("Type some text to correct first."))
        XCTAssertEqual(sent.count, 1)
    }

    func testTesterGivesUpWhenTheEngineNeverAnswers() async throws {
        let tester = PromptTester(timeoutSeconds: 0.05)
        tester.send = { _ in }
        tester.isReady = { true }
        tester.run(text: "hello", prompt: "", mode: .minimal)
        try await Task.sleep(nanoseconds: 300_000_000)
        XCTAssertEqual(tester.state, .failed("The correction model did not answer."))
    }
}
