import Foundation

/// Opener restore: a literal port of crates/phonon-llm/src/opener.rs, checked against the same opener_cases.json.
///
/// When the raw transcript starts with "Okay"/"Ok", "Well", "And" or "Now" and the refiner's text does not, the
/// opener goes back in front ("Okay, ", "Well, ", "Now, ", "And ") and the old first word is lowercased unless it
/// is "I", an I-contraction, all caps, internally capitalised, or capitalised mid-sentence in the transcript (a
/// proper noun). "So" is left alone. Runs on the refiner's text before the line breaks.
///
/// Per writing mode (`MODES` of the Rust): prompt and minimal restore; casual restores, in lowercase when the
/// refined text starts lowercase; formal and any other mode never restore.
///
/// Works on Unicode scalars, as the Rust works on chars.
enum Opener {
    typealias Scalars = [Unicode.Scalar]

    private enum Rule { case never, restore, matchCase }

    /// `MINIMAL_RESTORES` of the Rust.
    private static let minimalRestores = true

    /// `MODES` of the Rust; a mode not listed never restores.
    private static let modes: [String: Rule] = [
        "prompt": .restore, "casual": .matchCase, "minimal": minimalRestores ? .restore : .never, "formal": .never,
    ]

    private static let openers: [String: String] = [
        "okay": "Okay, ", "ok": "Okay, ", "well": "Well, ", "and": "And ", "now": "Now, ",
    ]

    /// `restore(raw, refined, mode)` of the Rust.
    static func restore(raw: String, refined: String, mode: String) -> String {
        let rule = modes[mode] ?? .never
        let body = Scalars(refined.unicodeScalars).drop(while: ws)
        if rule == .never || body.isEmpty { return refined }
        // a one-word output is a collapse; leave it for the output guard
        if words(Scalars(body)).count <= 1 { return refined }
        var rawWords = words(Scalars(raw.unicodeScalars))
        guard !rawWords.isEmpty else { return refined }
        let first = rawWords.removeFirst()
        let opener = lower(Scalars(first.reversed().drop(while: { $0 == "," || $0 == "." }).reversed()))
        guard let prefix = openers[opener] else { return refined }
        if !rawWords.contains(where: { $0.contains(where: alnum) }) { return refined }
        let refinedFirst = words(Scalars(body)).first ?? []
        let refinedOpener = lower(core(refinedFirst))
        let isOkay: (String) -> Bool = { $0 == "okay" || $0 == "ok" }
        if refinedOpener == opener || (isOkay(opener) && isOkay(refinedOpener)) { return refined }
        // casual style written lowercase: the opener goes in lowercase, the rest as it is
        if rule == .matchCase, let c = body.first, c.properties.isLowercase {
            return prefix.lowercased() + String(String.UnicodeScalarView(body))
        }
        var rest = String(String.UnicodeScalarView(body))
        if !keepsCapital(refinedFirst, raw: raw), let c = body.first {
            rest = c.properties.lowercaseMapping + String(String.UnicodeScalarView(body.dropFirst()))
        }
        return prefix + rest
    }

    // MARK: - helpers

    private static func ws(_ c: Unicode.Scalar) -> Bool { c.properties.isWhitespace }
    private static func alnum(_ c: Unicode.Scalar) -> Bool { c.properties.isAlphabetic || c.properties.numericType != nil }

    private static func words(_ s: Scalars) -> [Scalars] {
        s.split(whereSeparator: ws).map { Scalars($0) }
    }

    private static func lower(_ s: Scalars) -> String {
        s.map { $0.properties.lowercaseMapping }.joined()
    }

    /// The word without leading or trailing punctuation ("Sam's," -> "Sam's").
    private static func core(_ w: Scalars) -> Scalars {
        Scalars(w.drop(while: { !alnum($0) }).reversed().drop(while: { !alnum($0) }).reversed())
    }

    /// `keeps_capital(word, raw)` of the Rust.
    private static func keepsCapital(_ w: Scalars, raw: String) -> Bool {
        let word = core(w)
        let text = String(String.UnicodeScalarView(word))
        if text == "I" || text.hasPrefix("I'") || text.hasPrefix("I\u{2019}") { return true }
        let letters = word.filter { $0.properties.isAlphabetic }
        if letters.count >= 2 && letters.allSatisfy({ $0.properties.isUppercase }) { return true }
        if word.dropFirst().contains(where: { $0.properties.isUppercase }) { return true }
        guard let first = word.first, first.properties.isUppercase else { return false }
        let tokens = words(Scalars(raw.unicodeScalars))
        guard tokens.count >= 2 else { return false }
        let closers: Set<Unicode.Scalar> = ["\"", "'", ")", "\u{201D}", "\u{2019}"]
        for i in 1..<tokens.count {
            let prev = tokens[i - 1].reversed().drop(while: { closers.contains($0) })
            let sentenceEnd = prev.first.map { $0 == "." || $0 == "?" || $0 == "!" } ?? false
            if !sentenceEnd && core(tokens[i]) == word { return true }
        }
        return false
    }
}
