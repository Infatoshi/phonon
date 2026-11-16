package ai.phonon.keyboard

import java.io.File
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** The refiner's prompt against fixtures/refiner_prompt_cases.json, which the training-side builders produced from
 *  synthetic inputs, byte for byte; and refiner_jni.cpp's fixed strings and markup against the same file. */
class RefinerPromptTest {
    // Gradle runs unit tests from the module directory (android/app).
    private fun repoFile(rel: String): File =
        listOf("../../$rel", "../$rel").map(::File).first { it.isFile }

    private val fixtures = JSONObject(repoFile("fixtures/refiner_prompt_cases.json").readText())

    private fun opt(o: JSONObject, key: String): String? = if (o.isNull(key)) null else o.getString(key)

    @Test
    fun theFixedStringsAreTheTrainingOnes() {
        assertEquals(fixtures.getString("instruction"), RefinerPrompt.INSTRUCTION)
        assertEquals(fixtures.getString("vocabulary_prefix"), RefinerPrompt.VOCAB_PREFIX)
        assertEquals(fixtures.getInt("screen_max_words"), RefinerPrompt.SCREEN_MAX_WORDS)
        assertEquals(fixtures.getInt("before_max_words"), RefinerPrompt.BEFORE_MAX_WORDS)
        val modes = fixtures.getJSONObject("modes")
        assertEquals(modes.length(), RefinerPrompt.MODE_LINES.size)
        for ((mode, line) in RefinerPrompt.MODE_LINES) assertEquals(modes.getString(mode), line)
    }

    @Test
    fun promptsAreByteIdenticalToTheTrainingBuilder() {
        val cases = fixtures.getJSONArray("cases")
        assertTrue(cases.length() >= 20)
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            val name = c.getString("name")
            val mode = c.getString("mode")
            val v = c.getJSONArray("vocabulary")
            val vocab = (0 until v.length()).map { v.getString(it) }
            val screen = opt(c, "screen_text")
            val before = opt(c, "before_cursor")
            assertEquals(name, c.getString("system"), RefinerPrompt.system(mode, vocab, screen, before))
            val user = RefinerPrompt.userTurn(c.getString("transcript"), before)
            assertEquals(name, c.getString("user"), user)
            assertEquals(name, c.getString("prompt"),
                RefinerPrompt.headText(mode, RefinerPrompt.systemTail(vocab, screen, before)) + RefinerPrompt.tailText(user))
        }
    }

    @Test
    fun the31JoinRuleCasesOfTheTrainingRule() {
        val cases = fixtures.getJSONArray("join_rule_cases")
        assertEquals(31, cases.length())
        for (i in 0 until cases.length()) {
            val c = cases.getJSONArray(i)
            val before = if (c.isNull(0)) null else c.getString(0)
            assertEquals("before=$before raw=${c.getString(1)}", c.getString(2), RefinerPrompt.joinRule(before, c.getString(1)))
        }
    }

    /** refiner_jni.cpp builds the head and tail the Kotlin side mirrors: same instruction, mode lines and markup. */
    @Test
    fun theJniPromptMatches() {
        val cpp = repoFile("android/app/src/main/cpp/refiner_jni.cpp").readText()
        val literal = Regex("""kSystemPrompt =\s*((?:"[^"]*"\s*)+);""").find(cpp)!!.groupValues[1]
        assertEquals(RefinerPrompt.INSTRUCTION, Regex("\"([^\"]*)\"").findAll(literal).joinToString("") { it.groupValues[1] })
        for (line in RefinerPrompt.MODE_LINES.values) assertTrue(line, cpp.contains("\"$line\""))
        assertTrue(cpp.contains("std::string system = std::string(kSystemPrompt) + \"\\n\" + modeLine(mode) + tail;"))
        assertTrue(cpp.contains("return \"<|startoftext|><|im_start|>system\\n\" + system + \"<|im_end|>\\n<|im_start|>user\\n\";"))
        assertTrue(cpp.contains("raw + \"<|im_end|>\\n<|im_start|>assistant\\n<think></think>\""))
    }
}
