import Foundation

/// The user's dictionary on the iPhone: a literal port of the desktop's DictionaryFile (crates/phonon-core/src/data.rs:
/// relevant_entries, apply_exact_replacements, phonetic_matches, prepare_polish_input) and of the refiner side of it
/// (sidecar/polish_server.py: refiner_vocabulary, transcript_payload, restore_terms), checked against the same
/// crates/phonon-core/src/dictionary_cases.json as the Rust, the Python and the Android port
/// (android/app/src/main/java/ai/phonon/keyboard/Dictionary.kt; ios/UnitTests/DictionaryTests.swift).
///
/// Per dictation, as on the Mac: the transcript gets the exact replacements and the phonetic pre-corrections
/// (`prepare`), the refiner gets the corrected transcript and the vocabulary line of the relevant terms, and its
/// output gets the one-word term swaps undone (`restoreTerms`) and the exact replacements again (`applyExact`).
/// An empty dictionary changes nothing anywhere. The phone has no screen text, so screen_confirmed_terms is empty.
///
/// Named UserDictionary so it does not shadow Swift's Dictionary. The Rust works on chars and bytes and the Python
/// on code points, so everything here is on Unicode scalars or UTF-8 bytes, never on Swift Characters (whose
/// comparisons fold canonical equivalents).
struct UserDictionary: Sendable {
    struct Entry: Sendable, Equatable, Hashable {
        var phrase: String
        var replacement: String? = nil
        var spokenForms: [String] = []
        var source: String = ""
        var starred: Bool = false
        var usageCount: Int64 = 0

        var canonical: String { replacement ?? phrase }
        var forms: [String] { [phrase] + spokenForms }
    }

    /// What the refiner gets: the corrected transcript and the vocabulary line terms (empty: no line).
    struct Prepared: Sendable, Equatable {
        let text: String
        let vocab: [String]
        var vocabLine: String { vocab.joined(separator: ", ") }
    }

    let entries: [Entry]
    let words: WordList?

    init(entries: [Entry] = [], words: WordList? = nil) {
        self.entries = entries
        self.words = words
    }

    static let maxWindow = 3
    static let fileName = "dictionary.json"

    var isEmpty: Bool { entries.isEmpty }

    /// The refiner's input for `transcript`; an empty dictionary returns it unchanged with no vocabulary.
    func prepare(_ transcript: String) -> Prepared {
        if entries.isEmpty { return Prepared(text: transcript, vocab: []) }
        let env = envelope(transcript)
        return Prepared(text: Self.transcriptPayload(env), vocab: Self.refinerVocabulary(env))
    }

    // MARK: - data.rs DictionaryFile

    /// `relevant_entries`.
    func relevantEntries(_ transcript: String) -> [Entry] {
        let normalized = Self.normalize(transcript)
        let tw = Self.spaceWords(normalized)
        var windowKeys = Set<String>()
        if !tw.isEmpty {
            for len in 1...min(3, tw.count) {
                for s in 0...(tw.count - len) { windowKeys.insert(Self.phoneticKey(tw[s..<(s + len)].joined(separator: " "))) }
            }
        }
        var matches: [(entry: Entry, score: Int64, order: Int)] = []
        for (order, entry) in entries.enumerated() {
            var best: Int64 = 0
            for raw in entry.forms {
                let form = Self.normalize(raw)
                if form.isEmpty { continue }
                if Self.containsPhrase(normalized, form) { best = max(best, 1_000_000); continue }
                let fw = Self.spaceWords(form)
                if fw.count == 1 {
                    for w in tw where Self.plausiblySameWord(w, fw[0]) { best = max(best, 500_000) }
                    let key = Self.phoneticKey(form)
                    if key.utf8.count >= 3 && windowKeys.contains(key) { best = max(best, 750_000) }
                }
            }
            if best > 0 { matches.append((entry, best + entry.usageCount, order)) }
        }
        // Rust sort_by is stable: the original order breaks the last ties
        return matches.sorted { a, b in
            if a.score != b.score { return a.score > b.score }
            if a.entry.starred != b.entry.starred { return a.entry.starred }
            let c = Self.compareScalars(a.entry.phrase, b.entry.phrase)
            if c != 0 { return c < 0 }
            return a.order < b.order
        }.map(\.entry)
    }

    /// `apply_exact_replacements(text).text`.
    func applyExact(_ text: String) -> String {
        var out = text
        for e in entries {
            guard let rep = e.replacement else { continue }
            for f in e.forms { out = Self.replaceAsciiPhrase(out, f, rep).text }
        }
        return out
    }

    /// `prepare_polish_input(transcript)` with no screen text: the desktop's dictionary envelope.
    func envelope(_ transcript: String) -> String {
        if entries.isEmpty { return transcript }
        let likely = relevantEntries(transcript)
        let exact = applyExact(transcript)
        let phonetic = phoneticMatches(exact)
        var pre = exact
        for (spoken, canonical) in phonetic { pre = Self.replaceAsciiPhrase(pre, spoken, canonical).text }
        let terms = likely.filter { $0.replacement == nil }.map(\.canonical)
        let likelyTerms = likely.map(\.canonical)
        let replacements = likely.compactMap { e in e.replacement.map { "\(e.phrase) => \($0)" } }
        return "<phonon_dictionary>\ncanonical_terms: \(terms.joined(separator: ", "))\n"
            + "likely_terms: \(likelyTerms.joined(separator: ", "))\n"
            + "screen_confirmed_terms: \nphonetic_matches: \(phonetic.map { "\($0.0) => \($0.1)" }.joined(separator: "; "))\n"
            + "exact_replacements: \(replacements.joined(separator: "; "))\n</phonon_dictionary>\n"
            + "<transcript>\n\(Self.rustTrim(pre))\n</transcript>"
    }

    private final class Slot {
        var canonical: String
        var counts: Set<Int>
        init(_ canonical: String, _ n: Int) {
            self.canonical = canonical
            counts = [n]
        }
    }

    /// `phonetic_matches`: (spoken window, canonical) pairs, longest windows first.
    func phoneticMatches(_ transcript: String) -> [(String, String)] {
        var byKey: [String: Slot] = [:]
        var ambiguous = Set<String>()
        for e in entries {
            for form in e.forms {
                let n = Self.spaceWords(Self.normalize(form)).count
                if n == 0 || n > Self.maxWindow { continue }
                let key = Self.phoneticKey(form)
                if key.utf8.count < 3 { continue }
                let canonical = e.canonical
                if ambiguous.contains(key) { continue }
                if let known = byKey[key] {
                    if Self.sameTerm(known.canonical, canonical) {
                        known.counts.insert(n)
                        if Self.capitalCount(canonical) > Self.capitalCount(known.canonical) { known.canonical = canonical }
                    } else {
                        byKey[key] = nil
                        ambiguous.insert(key)
                    }
                } else {
                    byKey[key] = Slot(canonical, n)
                }
            }
        }
        let w = Self.spaceWords(Self.normalize(transcript))
        var occupied = [Bool](repeating: false, count: w.count)
        var out: [(String, String)] = []
        var len = min(Self.maxWindow, w.count)
        while len >= 1 {
            for start in 0...(w.count - len) {
                if (start..<(start + len)).contains(where: { occupied[$0] }) { continue }
                let window = Array(w[start..<(start + len)])
                let spoken = window.joined(separator: " ")
                guard let slot = byKey[Self.phoneticKey(spoken)] else { continue }
                if !slot.counts.contains(where: { abs(len - $0) <= 1 }) { continue }
                if window.contains(where: { Self.phoneticKey($0).isEmpty }) { continue }
                if Self.normalize(slot.canonical) == spoken { continue }
                if isOrdinaryEnglish(spoken) { continue }
                let absorbing = slot.counts.allSatisfy { len > $0 }
                if absorbing && window.contains(where: isRealWord) { continue }
                for i in start..<(start + len) { occupied[i] = true }
                out.append((spoken, slot.canonical))
            }
            len -= 1
        }
        return out
    }

    private func isOrdinaryEnglish(_ phrase: String) -> Bool {
        guard let list = words, !list.isEmpty else { return true }
        var any = false
        for word in Self.rustWords(phrase) {
            any = true
            if !Self.isOrdinaryEnglishWord(list, word) { return false }
        }
        return any
    }

    private func isRealWord(_ word: String) -> Bool {
        if word.unicodeScalars.count <= 2 { return true }
        guard let list = words, !list.isEmpty else { return true }
        return list.contains(word.lowercased())
    }

    private static func isOrdinaryEnglishWord(_ list: WordList, _ raw: String) -> Bool {
        let word = raw.lowercased()
        let w = Array(word.unicodeScalars)
        if w.count <= 2 { return true }
        if list.contains(word) { return true }
        for suffix in ["s", "es", "ed", "d", "ing", "ly", "er", "ers", "est"] {
            let suf = Array(suffix.unicodeScalars)
            guard w.count >= suf.count, Array(w.suffix(suf.count)) == suf else { continue }
            let stemScalars = Array(w.dropLast(suf.count))
            let stem = string(stemScalars)
            if stem.utf8.count < 3 { continue }
            if list.contains(stem) { return true }
            // running -> runn -> run, stopped -> stopp -> stop (the stem minus its last char)
            let shortened = string(Array(stemScalars.dropLast()))
            if shortened.utf8.count >= 3 && list.contains(shortened) { return true }
            if list.contains(stem + "e") { return true }
            if stemScalars.last == "i" {
                let withoutI = string(Array(stemScalars.dropLast()))
                if withoutI.utf8.count >= 3 && list.contains(withoutI + "y") { return true }
            }
        }
        return false
    }

    // MARK: - editing and storage (the desktop's dictionary.json schema)

    /// A copy with `entry` added (phrase trimmed, replacement dropped when empty or equal to the phrase), or with the
    /// entry of the same phrase and replacement (ignoring case) replaced, as DictionaryFile::merge keys them.
    func with(_ entry: Entry) -> UserDictionary {
        let phrase = entry.phrase.trimmingCharacters(in: .whitespacesAndNewlines)
        if phrase.isEmpty { return self }
        let rep = entry.replacement?.trimmingCharacters(in: .whitespacesAndNewlines)
        let replacement = (rep?.isEmpty ?? true) || rep == phrase ? nil : rep
        var seen = Set<String>()
        let forms = entry.spokenForms.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty && seen.insert($0.lowercased()).inserted }
        var clean = entry
        clean.phrase = phrase
        clean.replacement = replacement
        clean.spokenForms = forms
        let key = (phrase.lowercased(), replacement?.lowercased())
        let rest = entries.filter {
            ($0.phrase.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(), $0.replacement?.lowercased()) != key
        }
        return UserDictionary(entries: Self.sortForSave(rest + [clean]), words: words)
    }

    func without(_ entry: Entry) -> UserDictionary {
        UserDictionary(entries: entries.filter { $0 != entry }, words: words)
    }

    /// The same entries with a word list (the list is only needed once there is an entry).
    func withWords(_ list: WordList?) -> UserDictionary { UserDictionary(entries: entries, words: list) }

    /// The desktop's save order: phrase (lowercase), then replacement.
    static func sortForSave(_ list: [Entry]) -> [Entry] {
        list.enumerated().sorted { a, b in
            let c = compareScalars(a.element.phrase.lowercased(), b.element.phrase.lowercased())
            if c != 0 { return c < 0 }
            switch (a.element.replacement, b.element.replacement) {
            case let (x?, y?) where x != y: return compareScalars(x, y) < 0
            case (nil, _?): return true
            case (_?, nil): return false
            default: return a.offset < b.offset
            }
        }.map(\.element)
    }

    func toJSON(now: Date = Date()) -> Data {
        let list: [[String: Any]] = entries.map { e in
            var o: [String: Any] = ["phrase": e.phrase, "spoken_forms": e.spokenForms, "source": e.source,
                                    "starred": e.starred, "usage_count": e.usageCount]
            if let r = e.replacement { o["replacement"] = r }
            return o
        }
        let root: [String: Any] = ["schema_version": 1, "updated_at_unix_ms": Int64(now.timeIntervalSince1970 * 1000),
                                   "entries": list]
        let data = (try? JSONSerialization.data(withJSONObject: root,
                                                options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])) ?? Data()
        return data + Data("\n".utf8)
    }

    static func parseEntries(_ list: [Any]) -> [Entry] {
        list.compactMap { item in
            guard let o = item as? [String: Any], let phrase = o["phrase"] as? String else { return nil }
            return Entry(phrase: phrase, replacement: o["replacement"] as? String,
                         spokenForms: (o["spoken_forms"] as? [Any])?.compactMap { $0 as? String } ?? [],
                         source: o["source"] as? String ?? "", starred: o["starred"] as? Bool ?? false,
                         usageCount: (o["usage_count"] as? NSNumber)?.int64Value ?? 0)
        }
    }

    static func parse(_ data: Data, words: WordList? = nil) throws -> UserDictionary {
        let root = try JSONSerialization.jsonObject(with: data)
        guard let o = root as? [String: Any] else { throw CocoaError(.fileReadCorruptFile) }
        return UserDictionary(entries: parseEntries(o["entries"] as? [Any] ?? []), words: words)
    }

    /// Loads `dir`/dictionary.json, or an empty dictionary when there is none or it does not parse. The word list
    /// is only loaded when there is an entry.
    static func load(dir: URL, words: () -> WordList?) -> UserDictionary {
        let url = dir.appendingPathComponent(fileName)
        guard let data = try? Data(contentsOf: url), let d = try? parse(data) else { return UserDictionary() }
        return d.isEmpty ? d : d.withWords(words())
    }

    /// Writes `d` to `dir`/dictionary.json atomically.
    static func save(dir: URL, _ d: UserDictionary) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try d.toJSON().write(to: dir.appendingPathComponent(fileName), options: .atomic)
    }

    // MARK: - data.rs helpers

    private static func string(_ s: [Unicode.Scalar]) -> String {
        var v = String.UnicodeScalarView()
        v.append(contentsOf: s)
        return String(v)
    }

    /// Rust `char::is_alphanumeric`: Alphabetic, or a number (Nd, Nl, No).
    static func isAlnum(_ c: Unicode.Scalar) -> Bool {
        if c.properties.isAlphabetic { return true }
        switch c.properties.generalCategory {
        case .decimalNumber, .letterNumber, .otherNumber: return true
        default: return false
        }
    }

    private static func asciiLower(_ c: Unicode.Scalar) -> Unicode.Scalar {
        (c.value >= 0x41 && c.value <= 0x5A) ? Unicode.Scalar(c.value + 32)! : c
    }

    private static func asciiLowerBytes(_ b: [UInt8]) -> [UInt8] { b.map { ($0 >= 0x41 && $0 <= 0x5A) ? $0 + 32 : $0 } }

    private static func isAsciiAlnum(_ b: UInt8) -> Bool {
        (b >= 0x30 && b <= 0x39) || (b >= 0x41 && b <= 0x5A) || (b >= 0x61 && b <= 0x7A)
    }

    /// Rust `char::is_whitespace`: the White_Space property.
    private static func isRustWhitespace(_ c: Unicode.Scalar) -> Bool { c.properties.isWhitespace }

    /// Rust str ordering: by code point.
    static func compareScalars(_ a: String, _ b: String) -> Int {
        let x = Array(a.unicodeScalars), y = Array(b.unicodeScalars)
        for (p, q) in zip(x, y) where p != q { return p.value < q.value ? -1 : 1 }
        return x.count == y.count ? 0 : (x.count < y.count ? -1 : 1)
    }

    /// First byte offset of `needle` in `hay` at or after `from`.
    private static func find(_ hay: [UInt8], _ needle: [UInt8], from: Int) -> Int? {
        guard !needle.isEmpty, hay.count >= needle.count, from <= hay.count - needle.count else { return nil }
        var i = from
        let first = needle[0]
        while i <= hay.count - needle.count {
            if hay[i] == first {
                var k = 1
                while k < needle.count && hay[i + k] == needle[k] { k += 1 }
                if k == needle.count { return i }
            }
            i += 1
        }
        return nil
    }

    /// Words of a string already normalized (split on ' ', empties dropped).
    static func spaceWords(_ s: String) -> [String] {
        s.unicodeScalars.split(separator: " ").map { string(Array($0)) }
    }

    /// Rust `split_whitespace`.
    static func rustWords(_ s: String) -> [String] {
        s.unicodeScalars.split(whereSeparator: isRustWhitespace).map { string(Array($0)) }
    }

    /// Rust `str::trim`.
    static func rustTrim(_ s: String) -> String {
        let c = Array(s.unicodeScalars)
        var a = 0, b = c.count
        while a < b && isRustWhitespace(c[a]) { a += 1 }
        while b > a && isRustWhitespace(c[b - 1]) { b -= 1 }
        return string(Array(c[a..<b]))
    }

    /// `normalize`: alphanumerics (ASCII lowercased), everything else a single space between words.
    static func normalize(_ value: String) -> String {
        let mapped = value.unicodeScalars.map { isAlnum($0) ? asciiLower($0) : " " }
        return mapped.split(separator: " ").map { string(Array($0)) }.joined(separator: " ")
    }

    static func containsPhrase(_ text: String, _ phrase: String) -> Bool {
        find(Array(" \(text) ".utf8), Array(" \(phrase) ".utf8), from: 0) != nil
    }

    static func levenshtein(_ left: String, _ right: String) -> Int {
        let l = Array(left.unicodeScalars), r = Array(right.unicodeScalars)
        var prev = Array(0...r.count)
        for (row, lc) in l.enumerated() {
            var cur = [Int](repeating: 0, count: r.count + 1)
            cur[0] = row + 1
            for col in 0..<r.count {
                let cost = lc != r[col] ? 1 : 0
                cur[col + 1] = min(cur[col] + 1, prev[col + 1] + 1, prev[col] + cost)
            }
            prev = cur
        }
        return prev[r.count]
    }

    static func plausiblySameWord(_ left: String, _ right: String) -> Bool {
        if Array(left.unicodeScalars) == Array(right.unicodeScalars) { return true }
        let longest = max(left.unicodeScalars.count, right.unicodeScalars.count)
        if longest < 5 || left.unicodeScalars.first != right.unicodeScalars.first { return false }
        let allowed = longest >= 6 ? 2 : 1
        return levenshtein(left, right) <= allowed
    }

    /// `phonetic_key`: consonant skeleton (c and q as k, vowels and y dropped, doubled letters inside a word
    /// collapsed, never across words).
    static func phoneticKey(_ value: String) -> String {
        var out = String.UnicodeScalarView()
        for word in normalize(value).unicodeScalars.split(separator: " ") {
            var prev: Unicode.Scalar?
            for var c in word {
                if !isAlnum(c) { continue }
                if c == "c" || c == "q" { c = "k" }
                if "aeiouy".unicodeScalars.contains(c) { continue }
                if prev != c { out.append(c) }
                prev = c
            }
        }
        return String(out)
    }

    private static func squash(_ s: String) -> String {
        s.unicodeScalars.filter(isAlnum).map { String($0).lowercased() }.joined()
    }

    static func sameTerm(_ a: String, _ b: String) -> Bool {
        Array(squash(a).unicodeScalars) == Array(squash(b).unicodeScalars)
    }

    static func capitalCount(_ s: String) -> Int { s.unicodeScalars.filter { $0.properties.isUppercase }.count }

    /// `replace_ascii_phrase`: every ASCII-case-insensitive, ASCII-word-bounded `from` in `text` becomes `to`.
    static func replaceAsciiPhrase(_ text: String, _ from: String, _ to: String) -> (text: String, count: Int) {
        let needle = Array(rustTrim(from).utf8)
        let lowerNeedle = asciiLowerBytes(needle)
        if needle.isEmpty || lowerNeedle == asciiLowerBytes(Array(to.utf8)) { return (text, 0) }
        let t = Array(text.utf8)
        let lowerText = asciiLowerBytes(t)
        var out: [UInt8] = []
        out.reserveCapacity(t.count)
        var cursor = 0
        var count = 0
        while let start = find(lowerText, lowerNeedle, from: cursor) {
            let end = start + lowerNeedle.count
            let left = start == 0 || !isAsciiAlnum(t[start - 1])
            let right = end == t.count || !isAsciiAlnum(t[end])
            if left && right {
                out += t[cursor..<start]
                out += Array(to.utf8)
                count += 1
            } else {
                out += t[cursor..<end]
            }
            cursor = end
        }
        out += t[cursor...]
        return (String(decoding: out, as: UTF8.self), count)
    }

    // MARK: - polish_server.py

    private static let vocabLines: [(String, Bool)] = [
        ("screen_confirmed_terms", false), ("phonetic_matches", true), ("exact_replacements", true),
        ("likely_terms", false), ("canonical_terms", false),
    ]

    /// Python `str.isspace`: White_Space plus the separators U+001C...U+001F.
    private static func isPyWhitespace(_ c: Unicode.Scalar) -> Bool {
        c.properties.isWhitespace || (c.value >= 0x1C && c.value <= 0x1F)
    }

    /// Python `str.isalnum`: a letter (Lu, Ll, Lt, Lm, Lo) or anything with a numeric value.
    private static func isPyAlnum(_ c: Unicode.Scalar) -> Bool {
        switch c.properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter: return true
        default: return c.properties.numericType != nil
        }
    }

    /// Python re `\w` on str.
    private static func isPyWord(_ c: Unicode.Scalar) -> Bool { isPyAlnum(c) || c == "_" }

    /// Python `str.strip()`.
    static func pyStrip(_ s: String) -> String {
        let c = Array(s.unicodeScalars)
        var a = 0, b = c.count
        while a < b && isPyWhitespace(c[a]) { a += 1 }
        while b > a && isPyWhitespace(c[b - 1]) { b -= 1 }
        return string(Array(c[a..<b]))
    }

    /// Python `str.split()`.
    static func pySplit(_ s: String) -> [String] {
        s.unicodeScalars.split(whereSeparator: isPyWhitespace).map { string(Array($0)) }
    }

    /// Python `str.split(sep)` (keeps empty pieces).
    private static func pySplit(_ s: String, on sep: String) -> [String] {
        let t = Array(s.utf8), n = Array(sep.utf8)
        var out: [String] = []
        var cursor = 0
        while let at = find(t, n, from: cursor) {
            out.append(String(decoding: t[cursor..<at], as: UTF8.self))
            cursor = at + n.count
        }
        out.append(String(decoding: t[cursor...], as: UTF8.self))
        return out
    }

    /// Python `str.splitlines()`.
    private static func pySplitLines(_ s: String) -> [String] {
        let breaks: Set<UInt32> = [0x0A, 0x0B, 0x0C, 0x0D, 0x1C, 0x1D, 0x1E, 0x85, 0x2028, 0x2029]
        let c = Array(s.unicodeScalars)
        var out: [String] = []
        var start = 0, i = 0
        while i < c.count {
            if breaks.contains(c[i].value) {
                out.append(string(Array(c[start..<i])))
                if c[i] == "\r" && i + 1 < c.count && c[i + 1] == "\n" { i += 1 }
                start = i + 1
            }
            i += 1
        }
        if start < c.count { out.append(string(Array(c[start...]))) }
        return out
    }

    /// The text between the first `open` and the next `close` (a lazy `(.*?)` with DOTALL), or nil.
    private static func between(_ text: String, _ open: String, _ close: String) -> String? {
        let t = Array(text.utf8)
        guard let a = find(t, Array(open.utf8), from: 0) else { return nil }
        let from = a + open.utf8.count
        guard let b = find(t, Array(close.utf8), from: from) else { return nil }
        return String(decoding: t[from..<b], as: UTF8.self)
    }

    /// `transcript_payload`.
    static func transcriptPayload(_ text: String) -> String {
        pyStrip(between(text, "<transcript>", "</transcript>") ?? text)
    }

    /// `refiner_vocabulary`: distinct terms of the envelope, most confident first, capped at
    /// min(80, max(30, 3 x words)).
    static func refinerVocabulary(_ text: String) -> [String] {
        guard let body = between(text, "<phonon_dictionary>", "</phonon_dictionary>") else { return [] }
        var fields: [String: String] = [:]
        for line in pySplitLines(body) {
            let parts = pySplit(line, on: ":")
            if parts.count > 1 { fields[pyStrip(parts[0])] = pyStrip(parts.dropFirst().joined(separator: ":")) }
        }
        var terms: [String] = []
        for (key, pairs) in vocabLines {
            let value = fields[key] ?? ""
            if pairs {
                for pair in pySplit(value, on: ";") {
                    let sides = pySplit(pair, on: "=>")
                    terms.append(sides.count > 1 ? sides.dropFirst().joined(separator: "=>") : "")
                }
            } else {
                terms += pySplit(value, on: ",")
            }
        }
        var seen = Set<[Unicode.Scalar]>()
        var vocab: [String] = []
        for t in terms.map(pyStrip) where !t.isEmpty && seen.insert(Array(t.unicodeScalars)).inserted { vocab.append(t) }
        let n = pySplit(transcriptPayload(text)).count
        return Array(vocab.prefix(min(80, max(30, 3 * n))))
    }

    /// Python `_norm`: lowercased, letters and numbers only.
    private static func pyNorm(_ word: String) -> String {
        string(word.lowercased().unicodeScalars.filter(isPyAlnum))
    }

    /// `TERM_SHAPE.fullmatch(core)`: `[A-Z].*[A-Z0-9]|.*\d.*|.+-.+` (no newline inside a word).
    private static func termShaped(_ c: [Unicode.Scalar]) -> Bool {
        func upper(_ s: Unicode.Scalar) -> Bool { s.value >= 0x41 && s.value <= 0x5A }
        func digit(_ s: Unicode.Scalar) -> Bool { s.value >= 0x30 && s.value <= 0x39 }
        if c.contains("\n") { return false }
        if c.count >= 2, upper(c[0]), let l = c.last, upper(l) || digit(l) { return true }
        if c.contains(where: { $0.properties.generalCategory == .decimalNumber }) { return true }
        return c.count >= 3 && c[1..<(c.count - 1)].contains("-")
    }

    /// `WORD_PARTS.match(word).groups()`: `^(\W*)(.*?)(\W*)$`.
    private static func wordParts(_ w: [Unicode.Scalar]) -> ([Unicode.Scalar], [Unicode.Scalar], [Unicode.Scalar]) {
        var a = 0
        while a < w.count && !isPyWord(w[a]) { a += 1 }
        var b = w.count
        while b > a && !isPyWord(w[b - 1]) { b -= 1 }
        return (Array(w[..<a]), Array(w[a..<b]), Array(w[b...]))
    }

    /// `restore_terms`: a one-word dictionary term the speaker said goes back where the refiner wrote another
    /// term-shaped word in its place (one-for-one swaps only).
    static func restoreTerms(spoken: String, output: String, vocab: [String]) -> String {
        var terms: [String: String] = [:]
        for t in vocab {
            let s = pyStrip(t)
            if !s.isEmpty && !s.unicodeScalars.contains(" ") { terms[pyNorm(t)] = s }
        }
        if terms.isEmpty { return output }
        let said = pySplit(spoken).map(pyNorm)
        let saidSet = Set(said)
        let o = Array(output.unicodeScalars)
        var spans: [Range<Int>] = []
        var i = 0
        while i < o.count {
            if isPyWhitespace(o[i]) { i += 1; continue }
            var j = i
            while j < o.count && !isPyWhitespace(o[j]) { j += 1 }
            spans.append(i..<j)
            i = j
        }
        let words = spans.map { Array(o[$0]) }
        let normed = words.map { pyNorm(string($0)) }
        var fixes: [Int: [Unicode.Scalar]] = [:]
        for op in SequenceMatcher(said, normed).opcodes() {
            if op.tag != "replace" || op.i2 - op.i1 != op.j2 - op.j1 { continue }
            for k in 0..<(op.i2 - op.i1) {
                let i = op.i1 + k, j = op.j1 + k
                guard let term = terms[said[i]] else { continue }
                let (lead, core, trail) = wordParts(words[j])
                if !saidSet.contains(normed[j]) && normed[j] != pyNorm(term) && termShaped(core) {
                    fixes[j] = lead + Array(term.unicodeScalars) + trail
                }
            }
        }
        if fixes.isEmpty { return output }
        var out: [Unicode.Scalar] = []
        var last = 0
        for (j, r) in spans.enumerated() {
            guard let fix = fixes[j] else { continue }
            out += o[last..<r.lowerBound]
            out += fix
            last = r.upperBound
        }
        out += o[last...]
        return string(out)
    }
}

/// difflib.SequenceMatcher(a=..., b=..., autojunk=False) without a junk function: get_opcodes only.
struct SequenceMatcher {
    struct Opcode: Equatable {
        let tag: String
        let i1: Int, i2: Int, j1: Int, j2: Int
    }

    private let a: [String]
    private let b: [String]
    private var b2j: [String: [Int]] = [:]

    init(_ a: [String], _ b: [String]) {
        self.a = a
        self.b = b
        for (j, s) in b.enumerated() { b2j[s, default: []].append(j) }
    }

    private func longest(_ alo: Int, _ ahi: Int, _ blo: Int, _ bhi: Int) -> (Int, Int, Int) {
        var bi = alo, bj = blo, size = 0
        var j2len: [Int: Int] = [:]
        for i in alo..<ahi {
            var next: [Int: Int] = [:]
            for j in b2j[a[i]] ?? [] {
                if j < blo { continue }
                if j >= bhi { break }
                let k = (j2len[j - 1] ?? 0) + 1
                next[j] = k
                if k > size { bi = i - k + 1; bj = j - k + 1; size = k }
            }
            j2len = next
        }
        while bi > alo && bj > blo && a[bi - 1] == b[bj - 1] { bi -= 1; bj -= 1; size += 1 }
        while bi + size < ahi && bj + size < bhi && a[bi + size] == b[bj + size] { size += 1 }
        return (bi, bj, size)
    }

    private func matchingBlocks() -> [(Int, Int, Int)] {
        var queue = [(0, a.count, 0, b.count)]
        var blocks: [(Int, Int, Int)] = []
        while let q = queue.popLast() {
            let (alo, ahi, blo, bhi) = q
            let (i, j, k) = longest(alo, ahi, blo, bhi)
            if k > 0 {
                blocks.append((i, j, k))
                if alo < i && blo < j { queue.append((alo, i, blo, j)) }
                if i + k < ahi && j + k < bhi { queue.append((i + k, ahi, j + k, bhi)) }
            }
        }
        blocks.sort { ($0.0, $0.1, $0.2) < ($1.0, $1.1, $1.2) }
        var out: [(Int, Int, Int)] = []
        var i1 = 0, j1 = 0, k1 = 0
        for (i2, j2, k2) in blocks {
            if i1 + k1 == i2 && j1 + k1 == j2 {
                k1 += k2
            } else {
                if k1 > 0 { out.append((i1, j1, k1)) }
                (i1, j1, k1) = (i2, j2, k2)
            }
        }
        if k1 > 0 { out.append((i1, j1, k1)) }
        out.append((a.count, b.count, 0))
        return out
    }

    func opcodes() -> [Opcode] {
        var i = 0, j = 0
        var out: [Opcode] = []
        for (ai, bj, size) in matchingBlocks() {
            let tag = i < ai && j < bj ? "replace" : i < ai ? "delete" : j < bj ? "insert" : ""
            if !tag.isEmpty { out.append(Opcode(tag: tag, i1: i, i2: ai, j1: j, j2: bj)) }
            i = ai + size
            j = bj + size
            if size > 0 { out.append(Opcode(tag: "equal", i1: ai, i2: i, j1: bj, j2: j)) }
        }
        return out
    }
}

/// The English word list (assets/english_words.txt: lowercase a-z, one per line, byte-sorted) as one byte array
/// with line offsets, binary searched.
final class WordList: Sendable {
    private let bytes: [UInt8]
    private let starts: [Int]

    init(_ data: Data) {
        let b = [UInt8](data)
        var s: [Int] = []
        s.reserveCapacity(200_000)
        var at = 0
        while at < b.count {
            var end = at
            while end < b.count && b[end] != 0x0A { end += 1 }
            if end > at { s.append(at) }
            at = end + 1
        }
        bytes = b
        starts = s
    }

    convenience init?(contentsOf url: URL) {
        guard let d = try? Data(contentsOf: url, options: .mappedIfSafe) else { return nil }
        self.init(d)
    }

    /// The list bundled with the app (Phonon target resource english_words.txt).
    static func bundled(_ bundle: Bundle = .main) -> WordList? {
        bundle.url(forResource: "english_words", withExtension: "txt").flatMap { WordList(contentsOf: $0) }
    }

    var isEmpty: Bool { starts.isEmpty }
    var count: Int { starts.count }

    private func end(_ i: Int) -> Int {
        var e = starts[i]
        while e < bytes.count && bytes[e] != 0x0A && bytes[e] != 0x0D { e += 1 }
        return e
    }

    func contains(_ word: String) -> Bool {
        let w = Array(word.utf8)
        if w.isEmpty || w.contains(where: { $0 < 0x61 || $0 > 0x7A }) { return false }
        var lo = 0, hi = starts.count - 1
        while lo <= hi {
            let mid = (lo + hi) >> 1
            let s = starts[mid], e = end(mid)
            var c = 0, k = 0
            while c == 0 && k < w.count && s + k < e { c = Int(bytes[s + k]) - Int(w[k]); k += 1 }
            if c == 0 { c = (e - s) - w.count }
            if c < 0 { lo = mid + 1 } else if c > 0 { hi = mid - 1 } else { return true }
        }
        return false
    }
}
