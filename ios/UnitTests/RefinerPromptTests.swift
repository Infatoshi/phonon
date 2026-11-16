import XCTest

/// The refiner's prompt against fixtures/refiner_prompt_cases.json, which the training-side builders produced from
/// synthetic inputs: byte for byte, for every case the iPhone can produce (it has no screen text). Without a
/// vocabulary or text before the cursor, the head is the exact text the iPhone used before.
final class RefinerPromptTests: XCTestCase {
    private let system = "Rewrite the raw dictation transcript into the exact text the speaker intended. Fix recognition "
        + "errors and technical terms, keep the speaker's wording and casing style, do not add or drop content."

    private func fixtures() throws -> [String: Any] {
        let bundle = Bundle(for: Self.self)
        let url = try XCTUnwrap(bundle.url(forResource: "refiner_prompt_cases", withExtension: "json"),
                                "refiner_prompt_cases.json is not in the test bundle")
        return try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    }

    func testNoVocabularyIsTheOldHead() {
        for (mode, line) in [("minimal", "Writing mode: minimal (notes and documents)."),
                             ("casual", "Writing mode: casual (a chat message)."),
                             ("formal", "Writing mode: formal (an email or document)."),
                             ("prompt", "Writing mode: prompt (an instruction to an AI assistant)."),
                             ("unknown", "Writing mode: minimal (notes and documents).")] {
            let want = "<|startoftext|><|im_start|>system\n\(system)\n\(line)<|im_end|>\n<|im_start|>user\n"
            XCTAssertEqual(RefinerPrompt.headText(mode: mode), want, mode)
            XCTAssertEqual(RefinerPrompt.headText(mode: mode, vocab: []), want, mode)
        }
    }

    func testTheFixedStringsAreTheTrainingOnes() throws {
        let f = try fixtures()
        XCTAssertEqual(f["instruction"] as? String, RefinerPrompt.systemPrompt)
        XCTAssertEqual(f["modes"] as? [String: String], RefinerPrompt.modeLines)
        XCTAssertEqual(f["vocabulary_prefix"] as? String, RefinerPrompt.vocabPrefix)
        XCTAssertEqual(f["screen_max_words"] as? Int, RefinerPrompt.screenMaxWords)
        XCTAssertEqual(f["before_max_words"] as? Int, RefinerPrompt.beforeMaxWords)
    }

    func testPromptsAreByteIdenticalToTheTrainingBuilder() throws {
        let cases = try XCTUnwrap(try fixtures()["cases"] as? [[String: Any]])
        XCTAssertGreaterThanOrEqual(cases.count, 20)
        for c in cases {
            let name = c["name"] as? String ?? "?"
            let mode = try XCTUnwrap(c["mode"] as? String)
            let vocab = try XCTUnwrap(c["vocabulary"] as? [String])
            let screen = c["screen_text"] as? String
            let before = c["before_cursor"] as? String
            let transcript = try XCTUnwrap(c["transcript"] as? String)
            XCTAssertEqual(RefinerPrompt.system(mode: mode, vocab: vocab, screen: screen, before: before),
                           c["system"] as? String, "system: \(name)")
            let user = RefinerPrompt.userTurn(transcript, before: before)
            XCTAssertEqual(user, c["user"] as? String, "user: \(name)")
            XCTAssertEqual(RefinerPrompt.headText(mode: mode, vocab: vocab, screen: screen, before: before)
                           + RefinerPrompt.tailText(user), c["prompt"] as? String, "prompt: \(name)")
        }
    }

    func testThe31JoinRuleCasesOfTheTrainingRule() throws {
        let cases = try XCTUnwrap(try fixtures()["join_rule_cases"] as? [[Any]])
        XCTAssertEqual(cases.count, 31)
        for c in cases {
            let before = c[0] as? String
            let raw = try XCTUnwrap(c[1] as? String)
            XCTAssertEqual(RefinerPrompt.joinRule(before: before, raw: raw), c[2] as? String,
                           "before \(String(describing: before)) raw \(raw)")
        }
    }
}
