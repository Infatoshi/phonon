package ai.phonon.keyboard

import java.io.File
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** The desktop dictionary (crates/phonon-core data.rs) and its refiner side (sidecar/polish_server.py), on the
 *  shared reference cases crates/phonon-core/src/dictionary_cases.json. */
class DictionaryTest {
    // Gradle runs unit tests from the module directory (android/app).
    private fun repoFile(rel: String): File =
        listOf("../../$rel", "../$rel").map(::File).first { it.isFile }

    private val cases = JSONObject(repoFile("crates/phonon-core/src/dictionary_cases.json").readText())
    private val words = WordList(repoFile("assets/english_words.txt").readBytes())

    private fun dict(c: JSONObject) = Dictionary(Dictionary.parseEntries(c.getJSONArray("entries")), words)
    private fun strings(a: JSONArray) = (0 until a.length()).map { a.getString(it) }

    @Test
    fun envelopesMatchTheRust() {
        val list = cases.getJSONArray("envelope")
        assertTrue(list.length() >= 20)
        for (i in 0 until list.length()) {
            val c = list.getJSONObject(i)
            assertEquals(c.getString("id"), c.getString("envelope"), dict(c).envelope(c.getString("transcript")))
        }
    }

    @Test
    fun vocabularyMatchesThePython() {
        val list = cases.getJSONArray("envelope")
        for (i in 0 until list.length()) {
            val c = list.getJSONObject(i)
            val expected = strings(c.getJSONArray("vocab"))
            assertEquals(c.getString("id"), expected, Dictionary.refinerVocabulary(c.getString("envelope")))
            val p = dict(c).prepare(c.getString("transcript"))
            assertEquals(c.getString("id"), expected, p.vocab)
        }
    }

    @Test
    fun exactReplacementsMatchTheRust() {
        val list = cases.getJSONArray("exact")
        assertTrue(list.length() >= 5)
        for (i in 0 until list.length()) {
            val c = list.getJSONObject(i)
            assertEquals(c.getString("id"), c.getString("expected"), dict(c).applyExact(c.getString("text")))
        }
    }

    @Test
    fun restoreTermsMatchesThePython() {
        val list = cases.getJSONArray("restore")
        assertTrue(list.length() >= 8)
        for (i in 0 until list.length()) {
            val c = list.getJSONObject(i)
            assertEquals(c.getString("id"), c.getString("expected"),
                Dictionary.restoreTerms(c.getString("spoken"), c.getString("output"), strings(c.getJSONArray("vocab"))))
        }
    }

    @Test
    fun emptyDictionaryChangesNothing() {
        val d = Dictionary(emptyList(), null)
        val p = d.prepare("  um so the plan is fine ")
        assertEquals("  um so the plan is fine ", p.text)
        assertTrue(p.vocab.isEmpty())
        assertEquals("", p.vocabLine)
        assertEquals("Ship it.", d.applyExact("Ship it."))
    }

    @Test
    fun wordListLooksUpExactWords() {
        assertTrue(words.size > 190_000)
        assertTrue(words.contains("aardvark"))
        assertTrue(words.contains("please"))
        assertFalse(words.contains("pietorch"))
        assertFalse(words.contains("Please"))
        assertFalse(words.contains(""))
    }

    @Test
    fun savesAndLoadsTheDesktopSchema() {
        val dir = createTempDirLike()
        var d = Dictionary(emptyList(), words)
        d = d.with(Dictionary.Entry(phrase = " vee llm ", replacement = "vLLM", source = "phone"))
        d = d.with(Dictionary.Entry(phrase = "Kubernetes", spokenForms = listOf("cooper netties", "Cooper Netties")))
        d = d.with(Dictionary.Entry(phrase = "VEE LLM", replacement = "vllm"))  // same key as the first: replaces it
        assertEquals(2, d.entries.size)
        Dictionary.save(dir, d)
        val back = Dictionary.load(dir) { words }
        assertEquals(d.entries, back.entries)
        assertEquals(listOf("cooper netties"), back.entries.first { it.phrase == "Kubernetes" }.spokenForms)
        val json = JSONObject(File(dir, "dictionary.json").readText())
        assertEquals(1, json.getInt("schema_version"))
        assertEquals("vllm", json.getJSONArray("entries").getJSONObject(1).getString("replacement"))
        assertEquals(1, back.without(back.entries[0]).entries.size)
        dir.deleteRecursively()
    }

    private fun createTempDirLike(): File = kotlin.io.path.createTempDirectory("dict").toFile()
}
