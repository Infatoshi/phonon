import Foundation

/// The refiner's fixed prompt head (Refiner.swift tokenizes it and caches the model state after it), kept free of
/// llama so the unit tests check it. Same text as the Mac (sidecar/polish_server.py refiner_system_prompt) and
/// Android (refiner_jni.cpp headText): the system prompt, the writing-mode line, then with a dictionary the line
/// "Vocabulary the speaker may use: a, b, c". No vocabulary: byte-identical to the head before the dictionary.
enum RefinerPrompt {
    static let systemPrompt =
        "Rewrite the raw dictation transcript into the exact text the speaker intended. Fix recognition errors "
        + "and technical terms, keep the speaker's wording and casing style, do not add or drop content."
    static let modeLines = [
        "minimal": "Writing mode: minimal (notes and documents).",
        "casual": "Writing mode: casual (a chat message).",
        "formal": "Writing mode: formal (an email or document).",
        "prompt": "Writing mode: prompt (an instruction to an AI assistant).",
    ]
    /// polish_server.py REFINER_VOCAB_PREFIX
    static let vocabPrefix = "Vocabulary the speaker may use: "

    static func headText(mode: String, vocab: [String] = []) -> String {
        var system = systemPrompt + "\n" + (modeLines[mode] ?? modeLines["minimal"]!)
        if !vocab.isEmpty { system += "\n" + vocabPrefix + vocab.joined(separator: ", ") }
        return "<|startoftext|><|im_start|>system\n\(system)<|im_end|>\n<|im_start|>user\n"
    }
}
