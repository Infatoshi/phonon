package ai.phonon.keyboard

/**
 * Opener restore: a literal port of crates/phonon-llm/src/opener.rs (also ported to ios/App/Opener.swift),
 * checked against the same opener_cases.json (app/src/test). Runs on the refiner's text before LineBreaks.
 *
 * When the raw transcript starts with "Okay"/"Ok", "Well", "And" or "Now" and the refiner's text does not, the
 * opener goes back in front ("Okay, ", "Well, ", "Now, ", "And ") and the old first word is lowercased unless it
 * is "I", an I-contraction, all caps, internally capitalised, or capitalised mid-sentence in the transcript (a
 * proper noun). "So" is left alone.
 *
 * Per writing mode (`MODES` of the Rust): prompt and minimal restore; casual restores, in lowercase when the
 * refined text starts lowercase; formal and any other mode never restore.
 *
 * Works on code points with Rust's character classes (Unicode White_Space, Alphabetic, numeric, Lowercase,
 * Uppercase), as LineBreaks does.
 */
object Opener {
    private enum class Rule { NEVER, RESTORE, MATCH_CASE }

    /** `MINIMAL_RESTORES` of the Rust. */
    const val MINIMAL_RESTORES = true

    /** `MODES` of the Rust; a mode not listed never restores. */
    private val modes = mapOf(
        "prompt" to Rule.RESTORE,
        "casual" to Rule.MATCH_CASE,
        "minimal" to if (MINIMAL_RESTORES) Rule.RESTORE else Rule.NEVER,
        "formal" to Rule.NEVER,
    )

    private val openers = mapOf(
        "okay" to "Okay, ", "ok" to "Okay, ", "well" to "Well, ", "and" to "And ", "now" to "Now, ",
    )

    /** `restore(raw, refined, mode)` of the Rust: `refined` with the opener `raw` starts with restored, or unchanged. */
    fun restore(raw: String, refined: String, mode: String): String {
        val rule = modes[mode] ?: Rule.NEVER
        val r = cps(refined)
        var s = 0
        while (s < r.size && ws(r[s])) s++
        val body = r.copyOfRange(s, r.size)
        if (rule == Rule.NEVER || body.isEmpty()) return refined
        // A one-word output is a collapse the output guard swaps for the raw transcript; an opener in front would
        // make it two words and hide it from the guard.
        val bodyWords = words(body)
        if (bodyWords.size <= 1) return refined
        val rawWords = words(cps(raw))
        if (rawWords.isEmpty()) return refined
        // "Okay", "Okay," or "Okay." (the opener as its own sentence)
        val first = rawWords[0]
        var e = first.size
        while (e > 0 && (first[e - 1] == ','.code || first[e - 1] == '.'.code)) e--
        val opener = str(first, 0, e).lowercase()
        val prefix = openers[opener] ?: return refined
        // a transcript that is only the opener has nothing to restore it in front of
        if (rawWords.drop(1).none { w -> w.any(::alnum) }) return refined
        val refinedFirst = bodyWords[0]
        val refinedOpener = str(core(refinedFirst)).lowercase()
        val isOkay = { w: String -> w == "okay" || w == "ok" }
        if (refinedOpener == opener || (isOkay(opener) && isOkay(refinedOpener))) return refined
        // casual style written lowercase: the opener goes in lowercase, the rest as it is
        if (rule == Rule.MATCH_CASE && Character.isLowerCase(body[0])) return prefix.lowercase() + str(body)
        val rest = if (keepsCapital(refinedFirst, raw)) str(body)
        else String(body, 0, 1).lowercase() + str(body, 1)
        return prefix + rest
    }

    // ---- helpers

    /** Rust `char::is_whitespace` (Unicode White_Space). */
    private fun ws(c: Int): Boolean = when (c) {
        in 0x09..0x0D, 0x20, 0x85, 0xA0, 0x1680, in 0x2000..0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 -> true
        else -> false
    }
    private fun isNumeric(c: Int) = when (Character.getType(c).toByte()) {
        Character.DECIMAL_DIGIT_NUMBER, Character.LETTER_NUMBER, Character.OTHER_NUMBER -> true
        else -> false
    }
    /** Rust `char::is_alphanumeric`. */
    private fun alnum(c: Int) = Character.isAlphabetic(c) || isNumeric(c)
    private fun cps(s: String): IntArray = s.codePoints().toArray()
    private fun str(c: IntArray, from: Int = 0, to: Int = c.size) = String(c, from, to - from)

    /** Rust `split_whitespace`. */
    private fun words(c: IntArray): List<IntArray> {
        val out = ArrayList<IntArray>()
        var i = 0
        while (i < c.size) {
            while (i < c.size && ws(c[i])) i++
            if (i == c.size) break
            val start = i
            while (i < c.size && !ws(c[i])) i++
            out.add(c.copyOfRange(start, i))
        }
        return out
    }

    /** The word without leading or trailing punctuation ("Sam's," -> "Sam's"). */
    private fun core(w: IntArray): IntArray {
        var s = 0
        var e = w.size
        while (s < e && !alnum(w[s])) s++
        while (e > s && !alnum(w[e - 1])) e--
        return w.copyOfRange(s, e)
    }

    /** `keeps_capital(word, raw)` of the Rust. */
    private fun keepsCapital(w: IntArray, raw: String): Boolean {
        val word = core(w)
        val text = str(word)
        if (text == "I" || text.startsWith("I'") || text.startsWith("I’")) return true
        val letters = word.filter { Character.isAlphabetic(it) }
        if (letters.size >= 2 && letters.all { Character.isUpperCase(it) }) return true
        if (word.drop(1).any { Character.isUpperCase(it) }) return true
        if (word.isEmpty() || !Character.isUpperCase(word[0])) return false
        val tokens = words(cps(raw))
        val closers = setOf('"'.code, '\''.code, ')'.code, 0x201D, 0x2019)
        for (i in 1 until tokens.size) {
            val prev = tokens[i - 1]
            var e = prev.size
            while (e > 0 && prev[e - 1] in closers) e--
            val sentenceEnd = e > 0 && (prev[e - 1] == '.'.code || prev[e - 1] == '?'.code || prev[e - 1] == '!'.code)
            if (!sentenceEnd && core(tokens[i]).contentEquals(word)) return true
        }
        return false
    }
}
