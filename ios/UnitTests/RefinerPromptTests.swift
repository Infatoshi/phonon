import XCTest

/// The refiner's prompt head: without a vocabulary it is the exact text the iPhone used before the dictionary, and
/// with one it matches Android's refiner_jni.cpp headText and the Mac's refiner_system_prompt.
final class RefinerPromptTests: XCTestCase {
    private let system = "Rewrite the raw dictation transcript into the exact text the speaker intended. Fix recognition "
        + "errors and technical terms, keep the speaker's wording and casing style, do not add or drop content."

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

    func testVocabularyLineFollowsTheModeLine() {
        XCTAssertEqual(RefinerPrompt.headText(mode: "prompt", vocab: ["vLLM", "Kubernetes"]),
                       "<|startoftext|><|im_start|>system\n\(system)\nWriting mode: prompt (an instruction to an AI "
                       + "assistant).\nVocabulary the speaker may use: vLLM, Kubernetes<|im_end|>\n<|im_start|>user\n")
    }
}
