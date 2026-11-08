package ai.phonon.keyboard

/**
 * Deterministic line breaks for the refiner's output: a literal port of crates/phonon-llm/src/linebreak.rs
 * (itself a port of the reference fmt_v0/code/linebreak_rule.py), checked against the same
 * linebreak_cases.json (app/src/test).
 *
 * The refiner never emits a newline (its training targets were joined with spaces), so dictated steps come
 * out as "1. a 2. b" on one line and an email's greeting and sign-off run into the body. This restores:
 * - lists (every mode but casual): two or more inline "N." / "N)" markers numbered 1, 2, 3... in order each
 *   start a new line; text before marker 1 stays as the lead-in line;
 * - formal only: a leading greeting ("Hi Sam,") and a trailing sign-off ("Best regards, Chris") go on their
 *   own lines.
 *
 * Works on code points with Rust's character classes (Unicode White_Space, Alphabetic, numeric, Lowercase),
 * so it agrees with the Rust on every input; Kotlin's own isWhitespace differs (no U+00A0, adds U+001C-1F).
 */
object LineBreaks {
    /** `line_breaks(text, mode)` of the Rust; `apply(text, mode)` of the reference. */
    fun apply(text: String, mode: String): String {
        var t = if (mode != "casual") lists(text) else text
        if (mode == "formal") t = signOff(greeting(t))
        return t
    }

    // ---- helpers

    private fun ws(c: Int): Boolean = when (c) {
        in 0x09..0x0D, 0x20, 0x85, 0xA0, 0x1680, in 0x2000..0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 -> true
        else -> false
    }
    private fun isDigit(c: Int) = c in 0x30..0x39
    private fun isAsciiAlpha(c: Int) = c in 0x41..0x5A || c in 0x61..0x7A
    private fun isNumeric(c: Int) = when (Character.getType(c).toByte()) {
        Character.DECIMAL_DIGIT_NUMBER, Character.LETTER_NUMBER, Character.OTHER_NUMBER -> true
        else -> false
    }
    /** Rust `char::is_alphanumeric() || c == '_'`, the regex `\b` word class. */
    private fun isWord(c: Int) = Character.isAlphabetic(c) || isNumeric(c) || c == '_'.code
    private fun lowerAscii(c: Int) = if (c in 0x41..0x5A) c + 32 else c
    private fun cps(s: String): IntArray = s.codePoints().toArray()
    private fun str(c: IntArray, from: Int = 0, to: Int = c.size) = String(c, from, to - from)
    private fun hasPrefix(c: IntArray, at: Int, word: IntArray): Boolean {
        if (at + word.size > c.size) return false
        for (k in word.indices) if (lowerAscii(c[at + k]) != word[k]) return false
        return true
    }
    private fun trimEnd(c: IntArray, from: Int, to: Int): Int {
        var e = to
        while (e > from && ws(c[e - 1])) e--
        return e
    }

    // ---- lists

    /** Inline numbered markers in order become one per line (the reference's `lists`). */
    fun lists(text: String): String {
        val t = cps(text)
        val seq = ArrayList<Int>()
        var want = 1
        for ((start, number) in markers(t)) {
            if (number == want) {
                seq.add(start)
                want++
            }
        }
        if (seq.size < 2) return text
        val out = ArrayList<String>()
        var last = 0
        for (start in seq) {
            out.add(str(t, last, trimEnd(t, last, start)))
            last = start
        }
        var s = last
        val e = trimEnd(t, last, t.size)
        while (s < e && ws(t[s])) s++
        out.add(str(t, s, e))
        val items = out.drop(1).joinToString("\n")
        return if (out[0].isEmpty()) items else out[0] + "\n" + items
    }

    /** `(?:(?<=\s)|^)(\d{1,2})[.)]\s+(?=\S)`, non-overlapping, left to right: (offset, value). */
    fun markers(c: IntArray): List<Pair<Int, Int>> {
        val found = ArrayList<Pair<Int, Int>>()
        var i = 0
        while (i < c.size) {
            val preceded = i == 0 || ws(c[i - 1])
            if (preceded && isDigit(c[i])) {
                var j = i
                while (j < c.size && j - i < 2 && isDigit(c[j])) j++
                if (j < c.size && (c[j] == '.'.code || c[j] == ')'.code)) {
                    var k = j + 1
                    while (k < c.size && ws(c[k])) k++
                    if (k > j + 1 && k < c.size) {
                        found.add(i to (str(c, i, j).toIntOrNull() ?: 0))
                        i = k
                        continue
                    }
                }
            }
            i++
        }
        return found
    }

    // ---- greeting

    private val greetingWords = listOf(
        "hi", "hello", "hey", "dear", "good morning", "good afternoon", "good evening",
    ).map { cps(it) }

    /**
     * `^((?:hi|hello|hey|dear|good (?:morning|afternoon|evening))\b[^.!?\n]{0,40}?[,!])\s+(?=\S)` with re.I:
     * (end of group 1, end of the match).
     */
    fun greetingMatch(t: IntArray): Pair<Int, Int>? {
        for (word in greetingWords) {
            if (!hasPrefix(t, 0, word)) continue
            val rest = word.size
            if (rest < t.size && isWord(t[rest])) continue  // \b
            var taken = 0
            var p = rest
            while (p < t.size) {
                val c = t[p]
                if (c == ','.code || c == '!'.code) {
                    val groupEnd = p + 1
                    var q = groupEnd
                    while (q < t.size && ws(t[q])) q++
                    if (q > groupEnd && q < t.size) return groupEnd to q
                    if (c == '!'.code) break  // "!" is outside [^.!?\n]
                } else if (c == '.'.code || c == '?'.code || c == '\n'.code) {
                    break
                }
                if (taken == 40) break
                taken++
                p++
            }
        }
        return null
    }

    /** Formal: the greeting on its own line, and the body after it capitalized. */
    fun greeting(text: String): String {
        val s = cps(text)
        var t = s
        greetingMatch(s)?.let { (groupEnd, end) ->
            t = s.copyOfRange(0, groupEnd) + intArrayOf('\n'.code) + s.copyOfRange(end, s.size)
        }
        val nl = t.indexOf('\n'.code)
        if (nl >= 0) {
            val head = t.copyOfRange(0, nl)
            val body = t.copyOfRange(nl + 1, t.size)
            if (greetingMatch(head + cps(" x")) != null && body.isNotEmpty() && Character.isLowerCase(body[0])) {
                return str(head) + "\n" + String(intArrayOf(body[0]), 0, 1).uppercase() + str(body, 1)
            }
        }
        return str(t)
    }

    // ---- sign-off

    private val signOffPhrases = listOf(
        "thanks", "thank you", "best regards", "kind regards", "best", "regards", "cheers", "sincerely",
        "talk soon", "all the best", "warmly",
    ).map { cps(it) }

    /**
     * `(?<=[.!?])\s+((?:phrases)(?:,)?\s+[A-Z][a-z]+\.?)\s*$` with re.I, leftmost match: (start of the
     * whitespace, start and end of group 1).
     */
    fun signOffMatch(c: IntArray): Triple<Int, Int, Int>? {
        for (i in 1 until c.size) {
            val prev = c[i - 1]
            if (!(prev == '.'.code || prev == '!'.code || prev == '?'.code) || !ws(c[i])) continue
            var p = i
            while (p < c.size && ws(c[p])) p++
            if (p == c.size) continue
            for (phrase in signOffPhrases) {
                if (!hasPrefix(c, p, phrase)) continue
                var r = p + phrase.size
                if (r < c.size && c[r] == ','.code) r++
                var q = r
                while (q < c.size && ws(c[q])) q++
                if (q == r) continue
                // [A-Z][a-z]+ under re.I: two or more ASCII letters
                var l = q
                while (l < c.size && isAsciiAlpha(c[l])) l++
                if (l - q < 2) continue
                var e = l
                if (e < c.size && c[e] == '.'.code) e++
                if ((e until c.size).any { !ws(c[it]) }) continue
                return Triple(i, p, e)
            }
        }
        return null
    }

    /** Formal: the sign-off (and the name after its comma) on new lines. */
    fun signOff(text: String): String {
        val c = cps(text)
        val (space, start, end) = signOffMatch(c) ?: return text
        var group = str(c, start, end)
        if (group.contains(',')) group = group.replaceFirst(", ", ",\n")
        return str(c, 0, space) + "\n" + group
    }
}
