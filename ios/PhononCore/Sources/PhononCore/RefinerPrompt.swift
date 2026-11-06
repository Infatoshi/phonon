import Foundation

/// The refiner contract, ported from `sidecar/polish_server.py` (REFINER_SYSTEM_PROMPT,
/// refiner_system_prompt, refiner_token_budget, guard_output, clean_output, restore_terms).
/// The wording is the trained prompt: a different string is a different model, so these
/// constants must stay byte-identical to the Python source. `RefinerPromptTests` pins them.
public enum RefinerPrompt {
    public static let systemPrompt =
        "Rewrite the raw dictation transcript into the exact text the speaker intended. "
        + "Fix recognition errors and technical terms, keep the speaker's wording and casing style, "
        + "do not add or drop content."
    public static let vocabularyPrefix = "Vocabulary the speaker may use: "

    public static let outputTokenCeiling = 2048
    public static let guardTokMult = 1.5
    public static let guardTokAdd = 32
    public static let guardMaxRatio = 2.0
    public static let guardMinRatio = 0.4
    public static let guardMinWords = 8

    /// Rendered with the refiner's own chat template (`chat_template.jinja` at the pinned
    /// revision, `enable_thinking=False`, `add_generation_prompt=True`), checked against
    /// transformers' `apply_chat_template` output in `RefinerPromptTests`.
    public static func render(system: String, user: String) -> String {
        "<|startoftext|><|im_start|>system\n\(system)<|im_end|>\n"
            + "<|im_start|>user\n\(user)<|im_end|>\n"
            + "<|im_start|>assistant\n<think></think>"
    }

    public static func system(vocabulary: [String], modeLine: String) -> String {
        var system = systemPrompt
        if !modeLine.isEmpty { system += "\n" + modeLine }
        if vocabulary.isEmpty { return system }
        return system + "\n" + vocabularyPrefix + vocabulary.joined(separator: ", ")
    }

    /// The adaptive vocabulary cap the refiner was evaluated with: min(80, max(30, 3 x words)).
    public static func capVocabulary(_ terms: [String], transcript: String) -> [String] {
        var seen = Set<String>()
        var vocab: [String] = []
        for raw in terms {
            let term = raw.trimmingCharacters(in: .whitespaces)
            if !term.isEmpty, seen.insert(term).inserted { vocab.append(term) }
        }
        let words = transcript.split(whereSeparator: { $0.isWhitespace }).count
        return Array(vocab.prefix(min(80, max(30, 3 * words))))
    }

    public static func tokenBudget(inputTokens: Int) -> Int {
        max(8, min(outputTokenCeiling, Int(guardTokMult * Double(inputTokens)) + guardTokAdd))
    }

    /// The spoken text instead of a runaway or a collapsed output. Returns (text, guardFired).
    public static func guardOutput(spoken: String, output: String) -> (String, Bool) {
        let spokenWords = spoken.split(whereSeparator: { $0.isWhitespace }).count
        let outputWords = output.split(whereSeparator: { $0.isWhitespace }).count
        if spokenWords == 0 { return (output, false) }
        if Double(outputWords) > guardMaxRatio * Double(spokenWords) { return (spoken, true) }
        if spokenWords > guardMinWords && Double(outputWords) < guardMinRatio * Double(spokenWords) {
            return (spoken, true)
        }
        return (output, false)
    }

    /// `LEAKED_MARKER_PATTERN = <\|?/?(?:channel|turn|think)\|?>?` removed, then stripped.
    public static func clean(_ text: String) -> String {
        let pattern = #"<\|?/?(?:channel|turn|think)\|?>?"#
        let stripped = text.replacingOccurrences(of: pattern, with: "", options: .regularExpression)
        return stripped.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Put back a one-word dictionary term the speaker said where the refiner wrote another
    /// term-shaped word in its place (one-for-one swaps only). Port of `restore_terms`.
    public static func restoreTerms(spoken: String, output: String, vocabulary: [String]) -> String {
        var terms: [String: String] = [:]
        for t in vocabulary {
            let term = t.trimmingCharacters(in: .whitespaces)
            if !term.isEmpty && !term.contains(" ") { terms[norm(term)] = term }
        }
        if terms.isEmpty { return output }
        let said = spoken.split(whereSeparator: { $0.isWhitespace }).map { norm(String($0)) }
        let saidSet = Set(said)
        let ns = output as NSString
        let regex = try! NSRegularExpression(pattern: #"\S+"#)
        let spans = regex.matches(in: output, range: NSRange(location: 0, length: ns.length)).map(\.range)
        let words = spans.map { ns.substring(with: $0) }
        let normed = words.map(norm)
        var fixes: [Int: String] = [:]
        for op in SequenceDiff.opcodes(said, normed) where op.tag == .replace && op.a.count == op.b.count {
            for (i, j) in zip(op.a, op.b) {
                guard let term = terms[said[i]] else { continue }
                let (lead, core, trail) = wordParts(words[j])
                if !saidSet.contains(normed[j]), normed[j] != norm(term), termShaped(core) {
                    fixes[j] = lead + term + trail
                }
            }
        }
        if fixes.isEmpty { return output }
        var out = ""
        var last = 0
        for (j, r) in spans.enumerated() {
            if let fix = fixes[j] {
                out += ns.substring(with: NSRange(location: last, length: r.location - last)) + fix
                last = r.location + r.length
            }
        }
        return out + ns.substring(from: last)
    }

    static func norm(_ word: String) -> String {
        String(word.lowercased().unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) }.map(Character.init))
    }

    /// `TERM_SHAPE = [A-Z].*[A-Z0-9]|.*\d.*|.+-.+` as a full match.
    static func termShaped(_ core: String) -> Bool {
        core.range(of: #"^(?:[A-Z].*[A-Z0-9]|.*\d.*|.+-.+)$"#, options: .regularExpression) != nil
    }

    /// `WORD_PARTS = ^(\W*)(.*?)(\W*)$`
    static func wordParts(_ word: String) -> (String, String, String) {
        let isWord: (Character) -> Bool = { $0.isLetter || $0.isNumber || $0 == "_" }
        let chars = Array(word)
        var start = 0
        while start < chars.count && !isWord(chars[start]) { start += 1 }
        var end = chars.count
        while end > start && !isWord(chars[end - 1]) { end -= 1 }
        return (String(chars[0..<start]), String(chars[start..<end]), String(chars[end...]))
    }
}

/// Writing modes from the refiner's `phonon_refiner.json` (`writing_modes`).
public enum WritingMode: String, CaseIterable, Codable, Sendable {
    case casual, formal, prompt, minimal

    public var line: String {
        switch self {
        case .minimal: return "Writing mode: minimal (notes and documents)."
        case .casual: return "Writing mode: casual (a chat message)."
        case .formal: return "Writing mode: formal (an email or document)."
        case .prompt: return "Writing mode: prompt (an instruction to an AI assistant)."
        }
    }

    public var label: String {
        switch self {
        case .casual: return "Casual"
        case .formal: return "Formal"
        case .prompt: return "Prompt"
        case .minimal: return "Minimal"
        }
    }

    public var detail: String {
        switch self {
        case .casual: return "Chat messages"
        case .formal: return "Email and documents"
        case .prompt: return "Instructions to an AI assistant"
        case .minimal: return "Notes, fewest changes"
        }
    }
}

/// difflib.SequenceMatcher(autojunk=False).get_opcodes(), enough of it for restore_terms.
enum SequenceDiff {
    enum Tag { case equal, replace, delete, insert }
    struct Op { let tag: Tag; let a: Range<Int>; let b: Range<Int> }

    static func opcodes(_ a: [String], _ b: [String]) -> [Op] {
        var blocks = matchingBlocks(a, b, 0, a.count, 0, b.count)
        blocks.append((a.count, b.count, 0))
        var ops: [Op] = []
        var i = 0, j = 0
        for (ai, bj, size) in blocks {
            let tag: Tag?
            if i < ai && j < bj { tag = .replace } else if i < ai { tag = .delete } else if j < bj { tag = .insert } else { tag = nil }
            if let tag { ops.append(Op(tag: tag, a: i..<ai, b: j..<bj)) }
            i = ai + size
            j = bj + size
            if size > 0 { ops.append(Op(tag: .equal, a: ai..<i, b: bj..<j)) }
        }
        return ops
    }

    private static func matchingBlocks(_ a: [String], _ b: [String], _ alo: Int, _ ahi: Int, _ blo: Int, _ bhi: Int) -> [(Int, Int, Int)] {
        var b2j: [String: [Int]] = [:]
        for (j, x) in b.enumerated() { b2j[x, default: []].append(j) }
        var queue = [(alo, ahi, blo, bhi)]
        var blocks: [(Int, Int, Int)] = []
        while let (alo, ahi, blo, bhi) = queue.popLast() {
            let (i, j, k) = longestMatch(a, b2j, alo, ahi, blo, bhi)
            if k > 0 {
                blocks.append((i, j, k))
                if alo < i && blo < j { queue.append((alo, i, blo, j)) }
                if i + k < ahi && j + k < bhi { queue.append((i + k, ahi, j + k, bhi)) }
            }
        }
        blocks.sort { ($0.0, $0.1) < ($1.0, $1.1) }
        var merged: [(Int, Int, Int)] = []
        for blk in blocks {
            if let last = merged.last, last.0 + last.2 == blk.0, last.1 + last.2 == blk.1 {
                merged[merged.count - 1] = (last.0, last.1, last.2 + blk.2)
            } else {
                merged.append(blk)
            }
        }
        return merged
    }

    private static func longestMatch(_ a: [String], _ b2j: [String: [Int]], _ alo: Int, _ ahi: Int, _ blo: Int, _ bhi: Int) -> (Int, Int, Int) {
        var besti = alo, bestj = blo, bestsize = 0
        var j2len: [Int: Int] = [:]
        for i in alo..<ahi {
            var newj2len: [Int: Int] = [:]
            for j in b2j[a[i]] ?? [] {
                if j < blo { continue }
                if j >= bhi { break }
                let k = (j2len[j - 1] ?? 0) + 1
                newj2len[j] = k
                if k > bestsize { besti = i - k + 1; bestj = j - k + 1; bestsize = k }
            }
            j2len = newj2len
        }
        return (besti, bestj, bestsize)
    }
}
