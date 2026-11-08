import Foundation

/// Deterministic line breaks for the refiner's output: a literal port of crates/phonon-llm/src/linebreak.rs
/// (itself a port of the reference fmt_v0/code/linebreak_rule.py), checked against the same
/// linebreak_cases.json.
///
/// The refiner never emits a newline (its training targets were joined with spaces), so dictated steps come
/// out as "1. a 2. b" on one line and an email's greeting and sign-off run into the body. This restores:
/// - lists (every mode but casual): two or more inline "N." / "N)" markers numbered 1, 2, 3... in order each
///   start a new line; text before marker 1 stays as the lead-in line;
/// - formal only: a leading greeting ("Hi Sam,") and a trailing sign-off ("Best regards, Chris") go on their
///   own lines.
///
/// Works on Unicode scalars, as the Rust works on chars, so the two agree on every input.
enum LineBreaks {
    typealias Scalars = [Unicode.Scalar]

    /// `line_breaks(text, mode)` of the Rust; `apply(text, mode)` of the reference.
    static func apply(_ text: String, mode: String) -> String {
        var t = mode != "casual" ? lists(text) : text
        if mode == "formal" {
            t = signOff(greeting(t))
        }
        return t
    }

    // MARK: - helpers

    private static func ws(_ c: Unicode.Scalar) -> Bool { c.properties.isWhitespace }
    private static func isDigit(_ c: Unicode.Scalar) -> Bool { c.value >= 0x30 && c.value <= 0x39 }
    private static func isASCIIAlpha(_ c: Unicode.Scalar) -> Bool {
        (c.value >= 0x41 && c.value <= 0x5A) || (c.value >= 0x61 && c.value <= 0x7A)
    }
    /// Rust `char::is_alphanumeric() || c == '_'`, the regex `\b` word class.
    private static func isWord(_ c: Unicode.Scalar) -> Bool {
        c.properties.isAlphabetic || c.properties.numericType != nil || c == "_"
    }
    private static func lowerASCII(_ c: Unicode.Scalar) -> Unicode.Scalar {
        (c.value >= 0x41 && c.value <= 0x5A) ? Unicode.Scalar(c.value + 32)! : c
    }
    /// Whether `c[at...]` starts with the lowercase ASCII `word`, ignoring ASCII case.
    private static func hasPrefix(_ c: Scalars, at: Int, _ word: Scalars) -> Bool {
        guard at + word.count <= c.count else { return false }
        for (k, w) in word.enumerated() where lowerASCII(c[at + k]) != w { return false }
        return true
    }
    private static func str(_ s: ArraySlice<Unicode.Scalar>) -> String {
        var view = String.UnicodeScalarView()
        view.append(contentsOf: s)
        return String(view)
    }
    private static func trimEnd(_ s: ArraySlice<Unicode.Scalar>) -> ArraySlice<Unicode.Scalar> {
        var end = s.endIndex
        while end > s.startIndex && ws(s[end - 1]) { end -= 1 }
        return s[s.startIndex..<end]
    }
    private static func trim(_ s: ArraySlice<Unicode.Scalar>) -> ArraySlice<Unicode.Scalar> {
        let tail = trimEnd(s)
        var start = tail.startIndex
        while start < tail.endIndex && ws(tail[start]) { start += 1 }
        return tail[start..<tail.endIndex]
    }

    // MARK: - chat join

    /// Closing phrases a chat message's sign-off line may hold (`chat_join`).
    private static let chatSignOffs: [String] = [
        "best regards", "kind regards", "warm regards", "warmest regards", "all the best", "best wishes",
        "many thanks", "thank you", "thanks", "regards", "best", "cheers", "sincerely", "talk soon", "warmly",
    ]

    /// `chat_join(text)` of the Rust: in a chat app's formal mode, the newlines a refiner (RLQv4U on) wrote
    /// after a leading greeting line and before and inside a trailing sign-off become single spaces; after a
    /// list the sign-off keeps one line of its own. Every other newline and every word stays. The keyboard
    /// cannot see which app it types into, so nothing on iOS calls this yet; it is here so the ports agree.
    static func chatJoin(_ text: String) -> String {
        var t = text
        if let nl = t.firstIndex(of: "\n") {
            let head = rtrim(String(t[..<nl]))
            let body = ltrim(String(t[nl...]))
            if isGreetingLine(head) && !body.isEmpty { t = head + " " + afterComma(head, body) }
        }
        let lines = rtrim(t).components(separatedBy: "\n")
        for k in [2, 1] where lines.count > k {
            let block = lines[(lines.count - k)...].map { rtrim(ltrim($0)) }
            let fits = k == 2 ? isClosing(block[0]) && isName(block[1]) : isSignOffLine(block[0])
            if !fits { continue }
            let body = rtrim(lines[..<(lines.count - k)].joined(separator: "\n"))
            guard let end = body.unicodeScalars.last, end == "." || end == "!" || end == "?" else { continue }
            let last = body.components(separatedBy: "\n").last ?? body
            let sep = markers(Array(last.unicodeScalars)).first?.0 == 0 ? "\n" : " "
            t = body + sep + block.joined(separator: " ")
            break
        }
        return t
    }

    private static func ltrim(_ s: String) -> String {
        let c = Array(s.unicodeScalars)
        var i = 0
        while i < c.count && ws(c[i]) { i += 1 }
        return str(c[i...])
    }
    private static func rtrim(_ s: String) -> String { str(trimEnd(Array(s.unicodeScalars)[...])) }

    /// Common sentence openers that read lowercase after a greeting's comma (`JOINED_OPENERS`).
    private static let joinedOpeners: Set<String> = [
        "here", "this", "the", "we", "just", "so", "it", "that", "there", "quick", "can", "could", "would",
    ]

    /// `after_comma`: the body's first word lowercased when the greeting ends in a comma and the word is a
    /// capitalized opener ("It", not "IT"); otherwise the body unchanged.
    private static func afterComma(_ head: String, _ body: String) -> String {
        let c = Array(body.unicodeScalars)
        var n = 0
        while n < c.count && c[n].properties.isAlphabetic { n += 1 }
        let word = c[0..<n]
        let capitalized = (word.first?.properties.isUppercase ?? false)
            && word.dropFirst().allSatisfy { $0.properties.isLowercase }
        let lower = str(word).lowercased()
        guard head.hasSuffix(","), capitalized, joinedOpeners.contains(lower) else { return body }
        return lower + str(c[n...])
    }

    /// A greeting word, at most six words, ending in ",", "!" or ".".
    private static func isGreetingLine(_ line: String) -> Bool {
        let words = line.split(whereSeparator: { $0.unicodeScalars.allSatisfy(ws) }).count
        guard (1...6).contains(words), let end = line.unicodeScalars.last,
              end == "," || end == "!" || end == "." else { return false }
        let c = Array(line.unicodeScalars)
        return greetingWords.contains { w in hasPrefix(c, at: 0, w) && !(w.count < c.count && isWord(c[w.count])) }
    }

    /// A closing phrase alone, bare or with its punctuation.
    private static func isClosing(_ line: String) -> Bool {
        var core = line
        while let e = core.unicodeScalars.last, e == "," || e == "!" || e == "." { core.unicodeScalars.removeLast() }
        let lower = Array(core.unicodeScalars).map(lowerASCII)
        return chatSignOffs.contains { lower == Array($0.unicodeScalars) }
    }

    /// One to three capitalized name words.
    private static func isName(_ line: String) -> Bool {
        let words = line.split(whereSeparator: { $0.unicodeScalars.allSatisfy(ws) })
        return (1...3).contains(words.count) && words.allSatisfy { w in
            guard let f = w.unicodeScalars.first, f.properties.isUppercase else { return false }
            return w.unicodeScalars.allSatisfy { $0.properties.isAlphabetic || $0 == "." || $0 == "-" || $0 == "'" }
        }
    }

    /// "Thanks!", "Best regards," or "Best regards, Chris".
    private static func isSignOffLine(_ line: String) -> Bool {
        if isClosing(line) { return true }
        guard let comma = line.firstIndex(of: ",") else { return false }
        return isClosing(String(line[..<comma])) && isName(ltrim(String(line[line.index(after: comma)...])))
    }

    // MARK: - lists

    /// Inline numbered markers in order become one per line (the reference's `lists`).
    static func lists(_ text: String) -> String {
        let t = Array(text.unicodeScalars)
        var seq: [Int] = []
        var want = 1
        for (start, number) in markers(t) where number == want {
            seq.append(start)
            want += 1
        }
        if seq.count < 2 { return text }
        var out: [String] = []
        var last = 0
        for start in seq {
            out.append(str(trimEnd(t[last..<start])))
            last = start
        }
        out.append(str(trim(t[last...])))
        let items = out.dropFirst().joined(separator: "\n")
        return out[0].isEmpty ? items : out[0] + "\n" + items
    }

    /// `(?:(?<=\s)|^)(\d{1,2})[.)]\s+(?=\S)`, non-overlapping, left to right: (scalar offset of the digits,
    /// their value).
    static func markers(_ c: Scalars) -> [(Int, Int)] {
        var found: [(Int, Int)] = []
        var i = 0
        while i < c.count {
            let preceded = i == 0 || ws(c[i - 1])
            if preceded && isDigit(c[i]) {
                var j = i
                while j < c.count && j - i < 2 && isDigit(c[j]) { j += 1 }
                if j < c.count && (c[j] == "." || c[j] == ")") {
                    var k = j + 1
                    while k < c.count && ws(c[k]) { k += 1 }
                    if k > j + 1 && k < c.count {
                        found.append((i, Int(str(c[i..<j])) ?? 0))
                        i = k
                        continue
                    }
                }
            }
            i += 1
        }
        return found
    }

    // MARK: - greeting

    private static let greetingWords: [Scalars] = [
        "hi", "hello", "hey", "dear", "good morning", "good afternoon", "good evening",
    ].map { Array($0.unicodeScalars) }

    /// `^((?:hi|hello|hey|dear|good (?:morning|afternoon|evening))\b[^.!?\n]{0,40}?[,!])\s+(?=\S)` with re.I:
    /// (end of group 1, end of the match), scalar offsets.
    static func greetingMatch(_ t: Scalars) -> (Int, Int)? {
        for word in greetingWords {
            guard hasPrefix(t, at: 0, word) else { continue }
            let rest = word.count
            if rest < t.count && isWord(t[rest]) { continue }  // \b
            var taken = 0
            var p = rest
            while p < t.count {
                let c = t[p]
                if c == "," || c == "!" {
                    let groupEnd = p + 1
                    var q = groupEnd
                    while q < t.count && ws(t[q]) { q += 1 }
                    if q > groupEnd && q < t.count { return (groupEnd, q) }
                    if c == "!" { break }  // "!" is outside [^.!?\n]
                } else if c == "." || c == "?" || c == "\n" {
                    break
                }
                if taken == 40 { break }
                taken += 1
                p += 1
            }
        }
        return nil
    }

    /// Formal: the greeting on its own line, and the body after it capitalized.
    static func greeting(_ text: String) -> String {
        let s = Array(text.unicodeScalars)
        var t = s
        if let (groupEnd, end) = greetingMatch(s) {
            t = Array(s[0..<groupEnd]) + ["\n"] + Array(s[end...])
        }
        if let nl = t.firstIndex(of: "\n") {
            let head = Array(t[0..<nl])
            let body = Array(t[(nl + 1)...])
            if greetingMatch(head + Array(" x".unicodeScalars)) != nil,
               let first = body.first, first.properties.isLowercase {
                return str(head[...]) + "\n" + String(first).uppercased() + str(body.dropFirst())
            }
        }
        return str(t[...])
    }

    // MARK: - sign-off

    private static let signOffPhrases: [Scalars] = [
        "thanks", "thank you", "best regards", "kind regards", "best", "regards", "cheers", "sincerely",
        "talk soon", "all the best", "warmly",
    ].map { Array($0.unicodeScalars) }

    /// `(?<=[.!?])\s+((?:phrases)(?:,)?\s+[A-Z][a-z]+\.?)\s*$` with re.I, leftmost match: (start of the
    /// whitespace, start and end of group 1).
    static func signOffMatch(_ c: Scalars) -> (Int, Int, Int)? {
        guard c.count > 1 else { return nil }
        for i in 1..<c.count {
            let prev = c[i - 1]
            guard prev == "." || prev == "!" || prev == "?", ws(c[i]) else { continue }
            var p = i
            while p < c.count && ws(c[p]) { p += 1 }
            if p == c.count { continue }
            for phrase in signOffPhrases {
                guard hasPrefix(c, at: p, phrase) else { continue }
                var r = p + phrase.count
                if r < c.count && c[r] == "," { r += 1 }
                var q = r
                while q < c.count && ws(c[q]) { q += 1 }
                if q == r { continue }
                // [A-Z][a-z]+ under re.I: two or more ASCII letters
                var l = q
                while l < c.count && isASCIIAlpha(c[l]) { l += 1 }
                if l - q < 2 { continue }
                var e = l
                if e < c.count && c[e] == "." { e += 1 }
                if !c[e...].allSatisfy(ws) { continue }
                return (i, p, e)
            }
        }
        return nil
    }

    /// Formal: the sign-off (and the name after its comma) on new lines.
    static func signOff(_ text: String) -> String {
        let c = Array(text.unicodeScalars)
        guard let (space, start, end) = signOffMatch(c) else { return text }
        var group = str(c[start..<end])
        if group.contains(","), let comma = group.range(of: ", ") {
            group.replaceSubrange(comma, with: ",\n")
        }
        return str(c[0..<space]) + "\n" + group
    }
}
