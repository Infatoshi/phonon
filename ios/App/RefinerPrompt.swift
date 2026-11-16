import Foundation

/// The refiner's prompt (Refiner.swift tokenizes the head and caches the model state after it), kept free of llama so
/// the unit tests check it. One definition per language of the prompt the refiner was trained on (phonon-research,
/// the ctx refiner builders): the instruction, the writing-mode line, then with a dictionary the line
/// "Vocabulary the speaker may use: a, b, c", then with text before the cursor "Before cursor: <its last 40 words>"
/// (the iPhone has no screen text, so no "On screen:" line, as in the training rows without one). With a
/// Before-cursor line the transcript goes through the join rule. RefinerPromptTests checks every case of
/// fixtures/refiner_prompt_cases.json (made by the training-side builders) byte for byte; the Mac sidecar
/// (sidecar/refiner_prompt.py), Windows and Linux (crates/phonon-llm refiner_prompt.rs) and Android
/// (RefinerPrompt.kt) check the same file.
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
    static let vocabPrefix = "Vocabulary the speaker may use: "
    static let screenPrefix = "On screen: "
    static let beforePrefix = "Before cursor: "
    /// Training screens were at most 350 words; the text before the cursor the last 40.
    static let screenMaxWords = 350
    static let beforeMaxWords = 40

    /// Python's `str.split()` whitespace: Unicode White_Space plus U+001C...U+001F.
    static func isSpace(_ s: Unicode.Scalar) -> Bool {
        s.properties.isWhitespace || (0x1C...0x1F).contains(s.value)
    }

    static func words(_ text: String) -> [String] {
        var out: [String] = []
        var word = String.UnicodeScalarView()
        for s in text.unicodeScalars {
            if isSpace(s) {
                if !word.isEmpty { out.append(String(word)); word = String.UnicodeScalarView() }
            } else {
                word.append(s)
            }
        }
        if !word.isEmpty { out.append(String(word)) }
        return out
    }

    /// The On-screen line's text: the first 350 words, whitespace collapsed.
    static func screenText(_ text: String?) -> String {
        words(text ?? "").prefix(screenMaxWords).joined(separator: " ")
    }

    /// The Before-cursor line's text: the last 40 words, whitespace collapsed.
    static func beforeText(_ text: String?) -> String {
        words(text ?? "").suffix(beforeMaxWords).joined(separator: " ")
    }

    enum Ending { case word, comma, sentenceEnd, colon, newline, other }

    /// How the text before the cursor ends: trailing spaces and tabs are ignored, a line break after them is a
    /// newline ending, then trailing closing quotes and brackets are ignored.
    static func endClass(_ before: String) -> Ending {
        var s = Array(before.unicodeScalars)
        while let last = s.last, last == " " || last == "\t" { s.removeLast() }
        if let last = s.last, last == "\n" || last == "\r" { return .newline }
        while let last = s.last, isSpace(last) { s.removeLast() }
        while let last = s.last, "'\")]".unicodeScalars.contains(last) { s.removeLast() }
        guard let c = s.last else { return .other }
        switch c {
        case ".", "?", "!": return .sentenceEnd
        case ",", ";": return .comma
        case ":": return .colon
        default: return isAlnum(c) ? .word : .other
        }
    }

    static func isAlnum(_ s: Unicode.Scalar) -> Bool {
        s.properties.isAlphabetic || s.properties.numericType != nil
    }

    /// The transcript after text before the cursor that does not end a sentence (a word, "," or ";"): its first
    /// letter lower-cased, unless the first word is "I" / I'm / I'll ..., or has another capital after its first
    /// letter (API, GitHub, OK). Anything else, and no Before-cursor text (nil), leaves it unchanged.
    static func joinRule(before: String?, raw: String) -> String {
        guard let before, [Ending.word, Ending.comma].contains(endClass(before)) else { return raw }
        let scalars = Array(raw.unicodeScalars)
        guard let lead = scalars.firstIndex(where: { !isSpace($0) }) else { return raw }
        let c = scalars[lead]
        guard c.properties.isAlphabetic, c.properties.isUppercase else { return raw }
        var token = Array(scalars[lead...].prefix(while: { !isSpace($0) }))
        // Python's re.sub(r"[^\w'’]+$", "", token): drop trailing characters that are not word characters or quotes
        while let last = token.last, !(isAlnum(last) || last == "_" || last == "'" || last == "\u{2019}") {
            token.removeLast()
        }
        let core = String(String.UnicodeScalarView(token))
        if core == "I" || core.hasPrefix("I'") || core.hasPrefix("I\u{2019}") { return raw }
        if token.dropFirst().contains(where: { $0.properties.isUppercase }) { return raw }
        var out = String.UnicodeScalarView(scalars[..<lead])
        out.append(contentsOf: c.properties.lowercaseMapping.unicodeScalars)
        out.append(contentsOf: scalars[(lead + 1)...])
        return String(out)
    }

    /// The system turn: instruction, mode line, then the optional vocabulary, On-screen and Before-cursor lines.
    static func system(mode: String, vocab: [String] = [], screen: String? = nil, before: String? = nil) -> String {
        var system = systemPrompt + "\n" + (modeLines[mode] ?? modeLines["minimal"]!)
        if !vocab.isEmpty { system += "\n" + vocabPrefix + vocab.joined(separator: ", ") }
        let shown = screenText(screen)
        if !shown.isEmpty { system += "\n" + screenPrefix + shown }
        let typed = beforeText(before)
        if !typed.isEmpty { system += "\n" + beforePrefix + typed }
        return system
    }

    /// The user turn: the trimmed transcript, through the join rule when there is a Before-cursor line.
    static func userTurn(_ transcript: String, before: String? = nil) -> String {
        let scalars = Array(transcript.unicodeScalars)
        let start = scalars.firstIndex(where: { !isSpace($0) }) ?? scalars.count
        let end = (scalars.lastIndex(where: { !isSpace($0) }) ?? -1) + 1
        let raw = start < end ? String(String.UnicodeScalarView(scalars[start..<end])) : ""
        return beforeText(before).isEmpty ? raw : joinRule(before: before, raw: raw)
    }

    /// Everything before the user's words; the model state after it is cached.
    static func headText(mode: String, vocab: [String] = [], screen: String? = nil, before: String? = nil) -> String {
        "<|startoftext|><|im_start|>system\n\(system(mode: mode, vocab: vocab, screen: screen, before: before))"
            + "<|im_end|>\n<|im_start|>user\n"
    }

    /// Everything after the head: the user turn and the assistant header with thinking off (the model's chat
    /// template, enable_thinking false).
    static func tailText(_ user: String) -> String {
        "\(user)<|im_end|>\n<|im_start|>assistant\n<think></think>"
    }
}
