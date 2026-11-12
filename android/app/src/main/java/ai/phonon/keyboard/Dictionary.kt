package ai.phonon.keyboard

import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * The user's dictionary on the phone: a literal port of the desktop's DictionaryFile (crates/phonon-core/src/data.rs:
 * relevant_entries, apply_exact_replacements, phonetic_matches, prepare_polish_input) and of the refiner side of it
 * (sidecar/polish_server.py: refiner_vocabulary, restore_terms), checked against the same
 * crates/phonon-core/src/dictionary_cases.json as the Rust and Python (app/src/test DictionaryTest; the iPhone port
 * ios/App/Dictionary.swift uses it too).
 *
 * Per dictation, as on the Mac: the transcript gets the exact replacements and the phonetic pre-corrections
 * ([prepare]), the refiner gets the corrected transcript and the vocabulary line of the relevant terms, and its
 * output gets the one-word term swaps undone ([restoreTerms]) and the exact replacements again ([applyExact]).
 * An empty dictionary changes nothing anywhere. The phone has no screen text, so screen_confirmed_terms is empty.
 *
 * Strings are handled by Unicode code point where the Rust works on chars.
 */
class Dictionary(val entries: List<Entry>, private val words: WordList?) {
    data class Entry(
        val phrase: String,
        val replacement: String? = null,
        val spokenForms: List<String> = emptyList(),
        val source: String = "",
        val starred: Boolean = false,
        val usageCount: Long = 0,
    ) {
        val canonical: String get() = replacement ?: phrase
        val forms: List<String> get() = listOf(phrase) + spokenForms
    }

    /** What the refiner gets: the corrected transcript and the vocabulary line terms (empty: no line). */
    data class Prepared(val text: String, val vocab: List<String>) {
        val vocabLine: String get() = vocab.joinToString(", ")
    }

    val isEmpty: Boolean get() = entries.isEmpty()

    /** The refiner's input for [transcript]; an empty dictionary returns it unchanged with no vocabulary. */
    fun prepare(transcript: String): Prepared {
        if (entries.isEmpty()) return Prepared(transcript, emptyList())
        val env = envelope(transcript)
        return Prepared(transcriptPayload(env), refinerVocabulary(env))
    }

    // ---------- data.rs DictionaryFile ----------

    /** `relevant_entries`. */
    fun relevantEntries(transcript: String): List<Entry> {
        val normalized = normalize(transcript)
        val tw = normalized.split(' ').filter { it.isNotEmpty() }
        val windowKeys = HashSet<String>()
        for (len in 1..minOf(3, tw.size)) for (s in 0..tw.size - len) windowKeys += phoneticKey(tw.subList(s, s + len).joinToString(" "))
        val matches = ArrayList<Pair<Entry, Long>>()
        for (entry in entries) {
            var best = 0L
            for (raw in entry.forms) {
                val form = normalize(raw)
                if (form.isEmpty()) continue
                if (containsPhrase(normalized, form)) { best = maxOf(best, 1_000_000L); continue }
                val fw = form.split(' ').filter { it.isNotEmpty() }
                if (fw.size == 1) {
                    for (w in tw) if (plausiblySameWord(w, fw[0])) best = maxOf(best, 500_000L)
                    val key = phoneticKey(form)
                    if (utf8Len(key) >= 3 && key in windowKeys) best = maxOf(best, 750_000L)
                }
            }
            if (best > 0) matches += entry to best + entry.usageCount
        }
        return matches.sortedWith { (a, sa), (b, sb) ->
            when {
                sa != sb -> sb.compareTo(sa)
                a.starred != b.starred -> if (b.starred) 1 else -1
                else -> compareCodePoints(a.phrase, b.phrase)
            }
        }.map { it.first }
    }

    /** `apply_exact_replacements(text).text`. */
    fun applyExact(text: String): String {
        var out = text
        for (e in entries) {
            val rep = e.replacement ?: continue
            for (f in e.forms) out = replaceAsciiPhrase(out, f, rep).first
        }
        return out
    }

    /** `prepare_polish_input(transcript)` with no screen text: the desktop's dictionary envelope. */
    fun envelope(transcript: String): String {
        if (entries.isEmpty()) return transcript
        val likely = relevantEntries(transcript)
        val exact = applyExact(transcript)
        val phonetic = phoneticMatches(exact)
        var pre = exact
        for ((spoken, canonical) in phonetic) pre = replaceAsciiPhrase(pre, spoken, canonical).first
        val terms = likely.filter { it.replacement == null }.map { it.canonical }
        val likelyTerms = likely.map { it.canonical }
        val replacements = likely.filter { it.replacement != null }.map { "${it.phrase} => ${it.replacement}" }
        return "<phonon_dictionary>\ncanonical_terms: ${terms.joinToString(", ")}\nlikely_terms: ${likelyTerms.joinToString(", ")}\n" +
            "screen_confirmed_terms: \nphonetic_matches: ${phonetic.joinToString("; ") { "${it.first} => ${it.second}" }}\n" +
            "exact_replacements: ${replacements.joinToString("; ")}\n</phonon_dictionary>\n<transcript>\n${rustTrim(pre)}\n</transcript>"
    }

    /** `phonetic_matches`: (spoken window, canonical) pairs, longest windows first. */
    fun phoneticMatches(transcript: String): List<Pair<String, String>> {
        class Slot(var canonical: String, val counts: MutableSet<Int>)
        val byKey = HashMap<String, Slot?>()
        for (e in entries) for (form in e.forms) {
            val n = normalize(form).split(' ').count { it.isNotEmpty() }
            if (n == 0 || n > MAX_WINDOW) continue
            val key = phoneticKey(form)
            if (utf8Len(key) < 3) continue
            val canonical = e.canonical
            if (byKey.containsKey(key)) {
                val known = byKey[key]
                if (known != null && sameTerm(known.canonical, canonical)) {
                    known.counts += n
                    if (capitalCount(canonical) > capitalCount(known.canonical)) known.canonical = canonical
                } else {
                    byKey[key] = null
                }
            } else {
                byKey[key] = Slot(canonical, mutableSetOf(n))
            }
        }
        val w = normalize(transcript).split(' ').filter { it.isNotEmpty() }
        val occupied = BooleanArray(w.size)
        val out = ArrayList<Pair<String, String>>()
        for (len in minOf(MAX_WINDOW, w.size) downTo 1) {
            for (start in 0..w.size - len) {
                if ((start until start + len).any { occupied[it] }) continue
                val window = w.subList(start, start + len)
                val spoken = window.joinToString(" ")
                val slot = byKey[phoneticKey(spoken)] ?: continue
                if (slot.counts.none { kotlin.math.abs(len - it) <= 1 }) continue
                if (window.any { phoneticKey(it).isEmpty() }) continue
                if (normalize(slot.canonical) == spoken) continue
                if (isOrdinaryEnglish(spoken)) continue
                val absorbing = slot.counts.all { len > it }
                if (absorbing && window.any { isRealWord(it) }) continue
                for (i in start until start + len) occupied[i] = true
                out += spoken to slot.canonical
            }
        }
        return out
    }

    private fun isOrdinaryEnglish(phrase: String): Boolean {
        val list = words
        if (list == null || list.isEmpty) return true
        var any = false
        for (word in rustWords(phrase)) {
            any = true
            if (!isOrdinaryEnglishWord(list, word)) return false
        }
        return any
    }

    private fun isRealWord(word: String): Boolean {
        if (cpLen(word) <= 2) return true
        val list = words
        return list == null || list.isEmpty || list.contains(word.lowercase())
    }

    private fun isOrdinaryEnglishWord(list: WordList, raw: String): Boolean {
        val word = raw.lowercase()
        if (cpLen(word) <= 2) return true
        if (list.contains(word)) return true
        for (suffix in listOf("s", "es", "ed", "d", "ing", "ly", "er", "ers", "est")) {
            if (!word.endsWith(suffix)) continue
            val stem = word.substring(0, word.length - suffix.length)
            if (utf8Len(stem) < 3) continue
            if (list.contains(stem)) return true
            // running -> runn -> run, stopped -> stopp -> stop (the stem minus its last char)
            val shortened = stem.substring(0, stem.length - Character.charCount(stem.codePointBefore(stem.length)))
            if (utf8Len(shortened) >= 3 && list.contains(shortened)) return true
            if (list.contains(stem + "e")) return true
            if (stem.endsWith("i")) {
                val withoutI = stem.substring(0, stem.length - 1)
                if (utf8Len(withoutI) >= 3 && list.contains(withoutI + "y")) return true
            }
        }
        return false
    }

    // ---------- storage (the desktop's dictionary.json schema) ----------

    fun toJson(): String {
        val arr = JSONArray()
        for (e in entries) {
            val o = JSONObject().put("phrase", e.phrase)
            if (e.replacement != null) o.put("replacement", e.replacement)
            o.put("spoken_forms", JSONArray(e.spokenForms)).put("source", e.source).put("starred", e.starred)
                .put("usage_count", e.usageCount)
            arr.put(o)
        }
        return JSONObject().put("schema_version", 1).put("updated_at_unix_ms", System.currentTimeMillis())
            .put("entries", arr).toString(2) + "\n"
    }

    /** A copy with [entry] added (phrase trimmed, replacement dropped when empty or equal to the phrase), or with the
     *  entry of the same phrase and replacement (ignoring case) replaced, as DictionaryFile::merge keys them. */
    fun with(entry: Entry): Dictionary {
        val phrase = entry.phrase.trim()
        if (phrase.isEmpty()) return this
        val rep = entry.replacement?.trim()?.takeIf { it.isNotEmpty() && it != phrase }
        val clean = entry.copy(phrase = phrase, replacement = rep,
            spokenForms = entry.spokenForms.map { it.trim() }.filter { it.isNotEmpty() }.distinctBy { it.lowercase() })
        val key = phrase.lowercase() to rep?.lowercase()
        val rest = entries.filter { (it.phrase.trim().lowercase() to it.replacement?.lowercase()) != key }
        return Dictionary(sortForSave(rest + clean), words)
    }

    fun without(entry: Entry): Dictionary = Dictionary(entries.filter { it != entry }, words)

    companion object {
        const val MAX_WINDOW = 3
        const val FILE_NAME = "dictionary.json"

        fun parse(json: String, words: WordList?): Dictionary {
            val arr = JSONObject(json).optJSONArray("entries") ?: JSONArray()
            return Dictionary(parseEntries(arr), words)
        }

        fun parseEntries(arr: JSONArray): List<Entry> = (0 until arr.length()).map { i ->
            val o = arr.getJSONObject(i)
            val forms = o.optJSONArray("spoken_forms")
            Entry(
                phrase = o.getString("phrase"),
                replacement = if (o.has("replacement") && !o.isNull("replacement")) o.getString("replacement") else null,
                spokenForms = if (forms == null) emptyList() else (0 until forms.length()).map { forms.getString(it) },
                source = o.optString("source", ""),
                starred = o.optBoolean("starred", false),
                usageCount = o.optLong("usage_count", 0L),
            )
        }

        /** The desktop's save order: phrase (lowercase), then replacement. */
        fun sortForSave(list: List<Entry>): List<Entry> = list.sortedWith { a, b ->
            val c = compareCodePoints(a.phrase.lowercase(), b.phrase.lowercase())
            if (c != 0) c else compareValues(a.replacement, b.replacement)
        }

        // ---------- data.rs helpers ----------

        private fun isAlnum(cp: Int): Boolean = Character.isAlphabetic(cp) || when (Character.getType(cp)) {
            Character.DECIMAL_DIGIT_NUMBER.toInt(), Character.LETTER_NUMBER.toInt(), Character.OTHER_NUMBER.toInt() -> true
            else -> false
        }

        private fun asciiLower(cp: Int): Int = if (cp in 'A'.code..'Z'.code) cp + 32 else cp
        private fun asciiLower(s: String): String {
            val sb = StringBuilder(s.length)
            for (ch in s) sb.append(if (ch in 'A'..'Z') ch + 32 else ch)
            return sb.toString()
        }

        private fun isAsciiAlnum(ch: Char) = ch in 'a'..'z' || ch in 'A'..'Z' || ch in '0'..'9'
        private fun isRustWhitespace(cp: Int) = Character.isWhitespace(cp) || Character.isSpaceChar(cp) || cp == 0x85

        fun cpLen(s: String) = s.codePointCount(0, s.length)
        fun utf8Len(s: String) = s.toByteArray(Charsets.UTF_8).size

        fun compareCodePoints(a: String, b: String): Int {
            var i = 0
            var j = 0
            while (i < a.length && j < b.length) {
                val x = a.codePointAt(i)
                val y = b.codePointAt(j)
                if (x != y) return x.compareTo(y)
                i += Character.charCount(x)
                j += Character.charCount(y)
            }
            return (a.length - i).compareTo(b.length - j)
        }

        /** Rust `split_whitespace`. */
        fun rustWords(s: String): List<String> {
            val out = ArrayList<String>()
            val sb = StringBuilder()
            var i = 0
            while (i < s.length) {
                val cp = s.codePointAt(i)
                if (isRustWhitespace(cp)) {
                    if (sb.isNotEmpty()) { out += sb.toString(); sb.setLength(0) }
                } else sb.appendCodePoint(cp)
                i += Character.charCount(cp)
            }
            if (sb.isNotEmpty()) out += sb.toString()
            return out
        }

        /** Rust `str::trim`. */
        fun rustTrim(s: String): String {
            var a = 0
            var b = s.length
            while (a < b && isRustWhitespace(s.codePointAt(a))) a += Character.charCount(s.codePointAt(a))
            while (b > a && isRustWhitespace(s.codePointBefore(b))) b -= Character.charCount(s.codePointBefore(b))
            return s.substring(a, b)
        }

        /** `normalize`: alphanumerics (ASCII lowercased), everything else a single space between words. */
        fun normalize(value: String): String {
            val sb = StringBuilder(value.length)
            var i = 0
            while (i < value.length) {
                val cp = value.codePointAt(i)
                sb.appendCodePoint(if (isAlnum(cp)) asciiLower(cp) else ' '.code)
                i += Character.charCount(cp)
            }
            return sb.split(' ').filter { it.isNotEmpty() }.joinToString(" ")
        }

        fun containsPhrase(text: String, phrase: String) = " $text ".contains(" $phrase ")

        fun levenshtein(left: String, right: String): Int {
            val l = left.codePoints().toArray()
            val r = right.codePoints().toArray()
            var prev = IntArray(r.size + 1) { it }
            for ((row, lc) in l.withIndex()) {
                val cur = IntArray(r.size + 1)
                cur[0] = row + 1
                for (col in r.indices) {
                    val cost = if (lc != r[col]) 1 else 0
                    cur[col + 1] = minOf(cur[col] + 1, prev[col + 1] + 1, prev[col] + cost)
                }
                prev = cur
            }
            return prev[r.size]
        }

        fun plausiblySameWord(left: String, right: String): Boolean {
            if (left == right) return true
            val longest = maxOf(cpLen(left), cpLen(right))
            val lf = if (left.isEmpty()) -1 else left.codePointAt(0)
            val rf = if (right.isEmpty()) -1 else right.codePointAt(0)
            if (longest < 5 || lf != rf) return false
            val allowed = if (longest >= 6) 2 else 1
            return levenshtein(left, right) <= allowed
        }

        /** `phonetic_key`: consonant skeleton (c and q as k, vowels and y dropped, doubled letters inside a word
         *  collapsed, never across words). */
        fun phoneticKey(value: String): String {
            val sb = StringBuilder()
            for (word in normalize(value).split(' ')) {
                if (word.isEmpty()) continue
                var prev = -1
                var i = 0
                while (i < word.length) {
                    var cp = word.codePointAt(i)
                    i += Character.charCount(cp)
                    if (!isAlnum(cp)) continue
                    if (cp == 'c'.code || cp == 'q'.code) cp = 'k'.code
                    if (cp == 'a'.code || cp == 'e'.code || cp == 'i'.code || cp == 'o'.code || cp == 'u'.code || cp == 'y'.code) continue
                    if (prev != cp) sb.appendCodePoint(cp)
                    prev = cp
                }
            }
            return sb.toString()
        }

        private fun squash(s: String): String {
            val sb = StringBuilder()
            var i = 0
            while (i < s.length) {
                val cp = s.codePointAt(i)
                if (isAlnum(cp)) sb.append(String(Character.toChars(cp)).lowercase())
                i += Character.charCount(cp)
            }
            return sb.toString()
        }

        fun sameTerm(a: String, b: String) = squash(a) == squash(b)
        fun capitalCount(s: String) = s.codePoints().filter { Character.isUpperCase(it) }.count().toInt()

        /** `replace_ascii_phrase`: every ASCII-case-insensitive, ASCII-word-bounded [from] in [text] becomes [to]. */
        fun replaceAsciiPhrase(text: String, from: String, to: String): Pair<String, Long> {
            val needle = rustTrim(from)
            if (needle.isEmpty() || asciiLower(needle) == asciiLower(to)) return text to 0L
            val lowerText = asciiLower(text)
            val lowerNeedle = asciiLower(needle)
            val out = StringBuilder(text.length)
            var cursor = 0
            var count = 0L
            while (true) {
                val start = lowerText.indexOf(lowerNeedle, cursor)
                if (start < 0) break
                val end = start + lowerNeedle.length
                val left = start == 0 || !isAsciiAlnum(text[start - 1])
                val right = end == text.length || !isAsciiAlnum(text[end])
                if (left && right) {
                    out.append(text, cursor, start).append(to)
                    count++
                } else {
                    out.append(text, cursor, end)
                }
                cursor = end
            }
            out.append(text, cursor, text.length)
            return out.toString() to count
        }

        // ---------- polish_server.py ----------

        private val DICTIONARY = Regex("<phonon_dictionary>(.*?)</phonon_dictionary>", RegexOption.DOT_MATCHES_ALL)
        private val TRANSCRIPT = Regex("<transcript>(.*?)</transcript>", RegexOption.DOT_MATCHES_ALL)
        private val VOCAB_LINES = listOf(
            "screen_confirmed_terms" to false, "phonetic_matches" to true, "exact_replacements" to true,
            "likely_terms" to false, "canonical_terms" to false,
        )

        /** Python `str.strip()` / `str.split()` whitespace. */
        private fun isPyWhitespace(ch: Char) = ch.isWhitespace() || ch in "\u001c\u001d\u001e\u001f\u0085"
        fun pyStrip(s: String): String = s.trim { isPyWhitespace(it) }
        fun pySplit(s: String): List<String> {
            val out = ArrayList<String>()
            val sb = StringBuilder()
            for (ch in s) {
                if (isPyWhitespace(ch)) { if (sb.isNotEmpty()) { out += sb.toString(); sb.setLength(0) } } else sb.append(ch)
            }
            if (sb.isNotEmpty()) out += sb.toString()
            return out
        }

        /** `transcript_payload`. */
        fun transcriptPayload(text: String): String = pyStrip(TRANSCRIPT.find(text)?.groupValues?.get(1) ?: text)

        /** `refiner_vocabulary`: distinct terms of the envelope, most confident first, capped at
         *  min(80, max(30, 3 x words)). */
        fun refinerVocabulary(text: String): List<String> {
            val body = DICTIONARY.find(text)?.groupValues?.get(1) ?: return emptyList()
            val fields = HashMap<String, String>()
            for (line in body.split(Regex("\r\n|\n|\r"))) {
                val at = line.indexOf(':')
                if (at >= 0) fields[pyStrip(line.substring(0, at))] = pyStrip(line.substring(at + 1))
            }
            val terms = ArrayList<String>()
            for ((key, pairs) in VOCAB_LINES) {
                val value = fields[key] ?: ""
                if (pairs) {
                    for (pair in value.split(";")) {
                        val at = pair.indexOf("=>")
                        terms += if (at >= 0) pair.substring(at + 2) else ""
                    }
                } else {
                    terms += value.split(",")
                }
            }
            val seen = HashSet<String>()
            val vocab = ArrayList<String>()
            for (t in terms.map(::pyStrip)) if (t.isNotEmpty() && seen.add(t)) vocab += t
            val n = pySplit(transcriptPayload(text)).size
            return vocab.take(minOf(80, maxOf(30, 3 * n)))
        }

        /** Python's Unicode \d and \W: the JVM needs UNICODE_CHARACTER_CLASS; Android's ICU regex is Unicode
         *  already and rejects the flag (IllegalArgumentException), so it is dropped there. */
        private fun unicodePattern(p: String, flags: Int = 0): java.util.regex.Pattern = try {
            java.util.regex.Pattern.compile(p, flags or java.util.regex.Pattern.UNICODE_CHARACTER_CLASS)
        } catch (e: IllegalArgumentException) {
            java.util.regex.Pattern.compile(p, flags)
        }

        private val TERM_SHAPE = unicodePattern("[A-Z].*[A-Z0-9]|.*\\d.*|.+-.+")
        private val WORD_PARTS = unicodePattern("^(\\W*)(.*?)(\\W*)$", java.util.regex.Pattern.DOTALL)
        private val NON_SPACE = Regex("\\S+")

        private fun pyNorm(word: String): String {
            val sb = StringBuilder()
            val lower = word.lowercase()
            var i = 0
            while (i < lower.length) {
                val cp = lower.codePointAt(i)
                if (isAlnum(cp)) sb.appendCodePoint(cp)
                i += Character.charCount(cp)
            }
            return sb.toString()
        }

        /** `restore_terms`: a one-word dictionary term the speaker said goes back where the refiner wrote another
         *  term-shaped word in its place (one-for-one swaps only). */
        fun restoreTerms(spoken: String, output: String, vocab: List<String>): String {
            val terms = HashMap<String, String>()
            for (t in vocab) {
                val s = pyStrip(t)
                if (s.isNotEmpty() && ' ' !in s) terms[pyNorm(t)] = s
            }
            if (terms.isEmpty()) return output
            val said = pySplit(spoken).map(::pyNorm)
            val saidSet = said.toHashSet()
            val spans = NON_SPACE.findAll(output).map { it.range }.toList()
            val words = spans.map { output.substring(it.first, it.last + 1) }
            val normed = words.map(::pyNorm)
            val fixes = HashMap<Int, String>()
            for (op in SequenceMatcher(said, normed).opcodes()) {
                if (op.tag != "replace" || op.i2 - op.i1 != op.j2 - op.j1) continue
                for (k in 0 until op.i2 - op.i1) {
                    val i = op.i1 + k
                    val j = op.j1 + k
                    val term = terms[said[i]] ?: continue
                    val m = WORD_PARTS.matcher(words[j])
                    if (!m.matches()) continue
                    val core = m.group(2)
                    if (normed[j] !in saidSet && normed[j] != pyNorm(term) && TERM_SHAPE.matcher(core).matches()) {
                        fixes[j] = m.group(1) + term + m.group(3)
                    }
                }
            }
            if (fixes.isEmpty()) return output
            val sb = StringBuilder()
            var last = 0
            for ((j, r) in spans.withIndex()) {
                val fix = fixes[j] ?: continue
                sb.append(output, last, r.first).append(fix)
                last = r.last + 1
            }
            return sb.append(output, last, output.length).toString()
        }

        /** Loads [dir]/dictionary.json, or an empty dictionary when there is none or it does not parse. */
        fun load(dir: File, words: () -> WordList?): Dictionary {
            val f = File(dir, FILE_NAME)
            if (!f.isFile) return Dictionary(emptyList(), null)
            return try {
                val d = parse(f.readText(), null)
                if (d.isEmpty) d else Dictionary(d.entries, words())
            } catch (e: Exception) {
                Dictionary(emptyList(), null)
            }
        }

        /** Writes [d] to [dir]/dictionary.json atomically. */
        fun save(dir: File, d: Dictionary) {
            val f = File(dir, FILE_NAME)
            val tmp = File(dir, "$FILE_NAME.tmp")
            tmp.writeText(d.toJson())
            if (!tmp.renameTo(f)) { f.delete(); tmp.renameTo(f) }
        }
    }
}

/** difflib.SequenceMatcher(a=..., b=..., autojunk=False) without a junk function: get_opcodes only. */
class SequenceMatcher(private val a: List<String>, private val b: List<String>) {
    data class Opcode(val tag: String, val i1: Int, val i2: Int, val j1: Int, val j2: Int)

    private val b2j = HashMap<String, MutableList<Int>>().also { m -> b.forEachIndexed { j, s -> m.getOrPut(s) { ArrayList() } += j } }

    private fun longest(alo: Int, ahi: Int, blo: Int, bhi: Int): Triple<Int, Int, Int> {
        var bi = alo
        var bj = blo
        var size = 0
        var j2len = HashMap<Int, Int>()
        for (i in alo until ahi) {
            val next = HashMap<Int, Int>()
            for (j in b2j[a[i]] ?: emptyList<Int>()) {
                if (j < blo) continue
                if (j >= bhi) break
                val k = (j2len[j - 1] ?: 0) + 1
                next[j] = k
                if (k > size) { bi = i - k + 1; bj = j - k + 1; size = k }
            }
            j2len = next
        }
        while (bi > alo && bj > blo && a[bi - 1] == b[bj - 1]) { bi--; bj--; size++ }
        while (bi + size < ahi && bj + size < bhi && a[bi + size] == b[bj + size]) size++
        return Triple(bi, bj, size)
    }

    private fun matchingBlocks(): List<Triple<Int, Int, Int>> {
        val queue = ArrayDeque<IntArray>()
        queue.addLast(intArrayOf(0, a.size, 0, b.size))
        val blocks = ArrayList<Triple<Int, Int, Int>>()
        while (queue.isNotEmpty()) {
            val (alo, ahi, blo, bhi) = queue.removeLast()
            val (i, j, k) = longest(alo, ahi, blo, bhi)
            if (k > 0) {
                blocks += Triple(i, j, k)
                if (alo < i && blo < j) queue.addLast(intArrayOf(alo, i, blo, j))
                if (i + k < ahi && j + k < bhi) queue.addLast(intArrayOf(i + k, ahi, j + k, bhi))
            }
        }
        blocks.sortWith(compareBy<Triple<Int, Int, Int>>({ it.first }, { it.second }, { it.third }))
        val out = ArrayList<Triple<Int, Int, Int>>()
        var i1 = 0
        var j1 = 0
        var k1 = 0
        for ((i2, j2, k2) in blocks) {
            if (i1 + k1 == i2 && j1 + k1 == j2) k1 += k2
            else {
                if (k1 > 0) out += Triple(i1, j1, k1)
                i1 = i2; j1 = j2; k1 = k2
            }
        }
        if (k1 > 0) out += Triple(i1, j1, k1)
        out += Triple(a.size, b.size, 0)
        return out
    }

    fun opcodes(): List<Opcode> {
        var i = 0
        var j = 0
        val out = ArrayList<Opcode>()
        for ((ai, bj, size) in matchingBlocks()) {
            val tag = when {
                i < ai && j < bj -> "replace"
                i < ai -> "delete"
                j < bj -> "insert"
                else -> ""
            }
            if (tag.isNotEmpty()) out += Opcode(tag, i, ai, j, bj)
            i = ai + size
            j = bj + size
            if (size > 0) out += Opcode("equal", ai, i, bj, j)
        }
        return out
    }
}

/** The English word list (assets/english_words.txt: lowercase a-z, one per line, byte-sorted) as one byte array
 *  with line offsets, binary searched: about 2.6 MB instead of a 20 MB hash set. */
class WordList(private val bytes: ByteArray) {
    private val starts: IntArray

    init {
        val s = ArrayList<Int>(200_000)
        var at = 0
        while (at < bytes.size) {
            var end = at
            while (end < bytes.size && bytes[end] != '\n'.code.toByte()) end++
            if (end > at) s += at
            at = end + 1
        }
        starts = s.toIntArray()
    }

    val isEmpty: Boolean get() = starts.isEmpty()
    val size: Int get() = starts.size

    private fun end(i: Int): Int {
        var e = starts[i]
        while (e < bytes.size && bytes[e] != '\n'.code.toByte() && bytes[e] != '\r'.code.toByte()) e++
        return e
    }

    fun contains(word: String): Boolean {
        if (word.isEmpty() || word.any { it !in 'a'..'z' }) return false
        val w = word.toByteArray(Charsets.US_ASCII)
        var lo = 0
        var hi = starts.size - 1
        while (lo <= hi) {
            val mid = (lo + hi) ushr 1
            val s = starts[mid]
            val e = end(mid)
            var c = 0
            var k = 0
            while (c == 0 && k < w.size && s + k < e) { c = (bytes[s + k].toInt() and 0xff) - (w[k].toInt() and 0xff); k++ }
            if (c == 0) c = (e - s) - w.size
            when {
                c < 0 -> lo = mid + 1
                c > 0 -> hi = mid - 1
                else -> return true
            }
        }
        return false
    }
}
