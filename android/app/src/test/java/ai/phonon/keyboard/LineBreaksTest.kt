package ai.phonon.keyboard

import java.io.File
import org.json.JSONArray
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** The Rust module's tests (crates/phonon-llm/src/linebreak.rs), on the same reference cases. */
class LineBreaksTest {
    private fun cases(): JSONArray {
        // Gradle runs unit tests from the module directory (android/app).
        val f = listOf("../../crates/phonon-llm/src/linebreak_cases.json", "../crates/phonon-llm/src/linebreak_cases.json")
            .map(::File).first { it.isFile }
        return JSONArray(f.readText())
    }

    @Test
    fun matchesThePythonReference() {
        val cases = cases()
        assertTrue(cases.length() >= 30)
        var changed = 0
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            val input = c.getString("input")
            val expected = c.getString("expected")
            if (input != expected) changed++
            assertEquals("${c.get("id")} (${c.getString("mode")})", expected, LineBreaks.apply(input, c.getString("mode")))
        }
        assertTrue("most cases exercise a change", changed >= 20)
    }

    @Test
    fun casualStaysInlineAndPlainTextIsUntouched() {
        assertEquals("1. eggs 2. milk", LineBreaks.apply("1. eggs 2. milk", "casual"))
        for (mode in listOf("minimal", "casual", "formal", "prompt")) {
            assertEquals("Ship it after lunch.", LineBreaks.apply("Ship it after lunch.", mode))
        }
        assertEquals("Steps 2. and 3. are optional", LineBreaks.apply("Steps 2. and 3. are optional", "prompt"))
    }

    @Test
    fun stepsAndLetters() {
        assertEquals(
            "Do this:\n1. open the file\n2. fix the bug\n3. run tests",
            LineBreaks.apply("Do this: 1. open the file 2. fix the bug 3. run tests", "prompt"),
        )
        assertEquals(
            "Hi Sam,\nThe build is green.\nBest regards,\nChris",
            LineBreaks.apply("Hi Sam, the build is green. Best regards, Chris", "formal"),
        )
        assertEquals(
            "Hi Sam, the build is green. Thanks, Chris",
            LineBreaks.apply("Hi Sam, the build is green. Thanks, Chris", "minimal"),
        )
        assertEquals("1. ship it\n2. tag it", LineBreaks.apply("1. ship it 2. tag it", "prompt"))
    }
}
