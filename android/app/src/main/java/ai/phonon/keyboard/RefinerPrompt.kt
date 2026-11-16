package ai.phonon.keyboard

/**
 * The refiner's prompt, one definition for Android (refiner_jni.cpp only adds the fixed instruction and mode line in
 * front of [systemTail] and the chat markup around it). The refiner gets the prompt it was trained on (phonon-research,
 * the ctx refiner builders): the instruction, the writing-mode line, then with a dictionary
 * "Vocabulary the speaker may use: a, b, c", then with text before the cursor "Before cursor: <its last 40 words>"
 * (Android has no screen text, so no "On screen:" line, as in the training rows without one). With a Before-cursor
 * line the transcript goes through [joinRule]; the model's output is never changed by it. RefinerPromptTest checks
 * every case of fixtures/refiner_prompt_cases.json (made by the training-side builders) byte for byte, as the Mac
 * sidecar, the Rust crates and the iPhone do.
 */
object RefinerPrompt {
    const val INSTRUCTION = "Rewrite the raw dictation transcript into the exact text the speaker intended. Fix " +
        "recognition errors and technical terms, keep the speaker's wording and casing style, do not add or drop content."
    val MODE_LINES = linkedMapOf(
        "minimal" to "Writing mode: minimal (notes and documents).",
        "casual" to "Writing mode: casual (a chat message).",
        "formal" to "Writing mode: formal (an email or document).",
        "prompt" to "Writing mode: prompt (an instruction to an AI assistant).",
    )
    const val VOCAB_PREFIX = "Vocabulary the speaker may use: "
    const val SCREEN_PREFIX = "On screen: "
    const val BEFORE_PREFIX = "Before cursor: "
    /** Training screens were at most 350 words; the text before the cursor the last 40. */
    const val SCREEN_MAX_WORDS = 350
    const val BEFORE_MAX_WORDS = 40
    /** Characters of the field read before the cursor: enough for 40 words. */
    const val BEFORE_READ_CHARS = 600

    fun modeLine(mode: String): String = MODE_LINES[mode] ?: MODE_LINES.getValue("minimal")

    /** Python's `str.split()` whitespace. */
    private fun isSpace(cp: Int): Boolean =
        Character.isWhitespace(cp) || Character.isSpaceChar(cp) || cp == 0x85 || cp in 0x1c..0x1f

    private fun isAlnum(cp: Int): Boolean = Character.isLetter(cp) || when (Character.getType(cp)) {
        Character.DECIMAL_DIGIT_NUMBER.toInt(), Character.LETTER_NUMBER.toInt(), Character.OTHER_NUMBER.toInt() -> true
        else -> false
    }

    private fun codePoints(s: String): IntArray {
        val out = ArrayList<Int>(s.length)
        var i = 0
        while (i < s.length) {
            val cp = Character.codePointAt(s, i)
            out += cp
            i += Character.charCount(cp)
        }
        return out.toIntArray()
    }

    private fun string(cps: List<Int>): String = buildString { cps.forEach { appendCodePoint(it) } }

    fun words(text: String?): List<String> {
        val out = mutableListOf<String>()
        val word = mutableListOf<Int>()
        for (cp in codePoints(text ?: "")) {
            if (isSpace(cp)) {
                if (word.isNotEmpty()) { out += string(word); word.clear() }
            } else word += cp
        }
        if (word.isNotEmpty()) out += string(word)
        return out
    }

    /** The On-screen line's text: the first 350 words, whitespace collapsed. */
    fun screenText(text: String?): String = words(text).take(SCREEN_MAX_WORDS).joinToString(" ")

    /** The Before-cursor line's text: the last 40 words, whitespace collapsed. */
    fun beforeText(text: String?): String = words(text).takeLast(BEFORE_MAX_WORDS).joinToString(" ")

    /** How the text before the cursor ends: word / comma / sentence-end / colon / newline / other. Trailing spaces and
     *  tabs are ignored, a line break after them is a newline ending, then closing quotes and brackets are ignored. */
    fun endClass(before: String): String {
        val s = codePoints(before).toMutableList()
        while (s.isNotEmpty() && (s.last() == ' '.code || s.last() == '\t'.code)) s.removeAt(s.lastIndex)
        if (s.isNotEmpty() && (s.last() == '\n'.code || s.last() == '\r'.code)) return "newline"
        while (s.isNotEmpty() && isSpace(s.last())) s.removeAt(s.lastIndex)
        while (s.isNotEmpty() && s.last() in "'\")]".map { it.code }) s.removeAt(s.lastIndex)
        val c = s.lastOrNull() ?: return "other"
        return when {
            c == '.'.code || c == '?'.code || c == '!'.code -> "sentence-end"
            c == ','.code || c == ';'.code -> "comma"
            c == ':'.code -> "colon"
            isAlnum(c) -> "word"
            else -> "other"
        }
    }

    /** The transcript after text before the cursor that does not end a sentence (a word, "," or ";"): its first
     *  letter lower-cased, unless the first word is "I" / I'm / I'll ..., or has another capital after its first
     *  letter (API, GitHub, OK). Anything else, and no Before-cursor text (null), leaves it unchanged. */
    fun joinRule(before: String?, raw: String): String {
        if (before == null || endClass(before) !in setOf("word", "comma")) return raw
        val cps = codePoints(raw)
        val lead = cps.indexOfFirst { !isSpace(it) }
        if (lead < 0) return raw
        val c = cps[lead]
        if (!(Character.isLetter(c) && Character.isUpperCase(c))) return raw
        val token = cps.drop(lead).takeWhile { !isSpace(it) }.toMutableList()
        // Python's re.sub(r"[^\w'’]+$", "", token)
        while (token.isNotEmpty()) {
            val t = token.last()
            if (isAlnum(t) || t == '_'.code || t == '\''.code || t == 0x2019) break
            token.removeAt(token.lastIndex)
        }
        val core = string(token)
        if (core == "I" || core.startsWith("I'") || core.startsWith("I’")) return raw
        if (token.drop(1).any { Character.isUpperCase(it) }) return raw
        return string(cps.take(lead)) + string(listOf(Character.toLowerCase(c))) + string(cps.drop(lead + 1))
    }

    /** The system turn after the mode line, each line led by "\n" ("" for none): what refiner_jni.cpp headText appends. */
    fun systemTail(vocab: List<String>, screen: String? = null, before: String? = null): String {
        val b = StringBuilder()
        if (vocab.isNotEmpty()) b.append("\n").append(VOCAB_PREFIX).append(vocab.joinToString(", "))
        val shown = screenText(screen)
        if (shown.isNotEmpty()) b.append("\n").append(SCREEN_PREFIX).append(shown)
        val typed = beforeText(before)
        if (typed.isNotEmpty()) b.append("\n").append(BEFORE_PREFIX).append(typed)
        return b.toString()
    }

    /** The whole system turn. */
    fun system(mode: String, vocab: List<String>, screen: String? = null, before: String? = null): String =
        INSTRUCTION + "\n" + modeLine(mode) + systemTail(vocab, screen, before)

    /** The user turn: the trimmed transcript, through the join rule when there is a Before-cursor line. */
    fun userTurn(transcript: String, before: String? = null): String {
        val cps = codePoints(transcript)
        val start = cps.indexOfFirst { !isSpace(it) }
        val raw = if (start < 0) "" else string(cps.slice(start..cps.indexOfLast { !isSpace(it) }))
        return if (beforeText(before).isEmpty()) raw else joinRule(before, raw)
    }

    /** refiner_jni.cpp headText and the tail it decodes after it, for the tests. */
    fun headText(mode: String, tail: String): String =
        "<|startoftext|><|im_start|>system\n" + INSTRUCTION + "\n" + modeLine(mode) + tail + "<|im_end|>\n<|im_start|>user\n"

    fun tailText(user: String): String = user + "<|im_end|>\n<|im_start|>assistant\n<think></think>"
}
