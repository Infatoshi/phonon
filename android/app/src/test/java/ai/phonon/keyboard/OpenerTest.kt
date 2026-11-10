package ai.phonon.keyboard

import java.io.File
import org.json.JSONArray
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** The Rust module's tests (crates/phonon-llm/src/opener.rs), on the same reference cases. */
class OpenerTest {
    private fun cases(): JSONArray {
        // Gradle runs unit tests from the module directory (android/app).
        val f = listOf("../../crates/phonon-llm/src/opener_cases.json", "../crates/phonon-llm/src/opener_cases.json")
            .map(::File).first { it.isFile }
        return JSONArray(f.readText())
    }

    /** Prompt mode, the one most dictations use; the word rules are the same in every mode that restores. */
    private fun restore(raw: String, refined: String) = Opener.restore(raw, refined, "prompt")

    @Test
    fun matchesTheReferenceCases() {
        val cases = cases()
        assertEquals("every shared case", 36, cases.length())
        var ran = 0
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            val mode = c.getString("mode")
            if (mode == "minimal" && !Opener.MINIMAL_RESTORES) continue
            assertEquals("${c.get("id")} ($mode)", c.getString("expected"),
                Opener.restore(c.getString("raw"), c.getString("refined"), mode))
            ran++
        }
        assertEquals(36, ran)
    }

    @Test
    fun casualMatchesTheRefinedCase() {
        assertEquals("okay, see you at five", Opener.restore("okay see you at five", "see you at five", "casual"))
        assertEquals("and sam is coming", Opener.restore("and Sam is coming", "sam is coming", "casual"))
        assertEquals("Okay, see you at five.", Opener.restore("okay see you at five", "See you at five.", "casual"))
        assertEquals("see you at five", Opener.restore("so see you at five", "see you at five", "casual"))
    }

    @Test
    fun formalVerbatimAndUnknownModesNeverRestore() {
        for (mode in listOf("formal", "verbatim", "raw", "", "unknown")) {
            assertEquals(mode, "The build is green.", Opener.restore("okay the build is green", "The build is green.", mode))
        }
    }

    @Test
    fun eachOpenerAndOkBecomesOkay() {
        assertEquals("Okay, the build is green.", restore("okay the build is green", "The build is green."))
        assertEquals("Well, it depends on the load.", restore("Well, it depends on the load", "It depends on the load."))
        assertEquals("And then we ship it.", restore("And then we ship it", "Then we ship it."))
        assertEquals("Now, let's look at the logs.", restore("Now, let's look at the logs", "Let's look at the logs."))
        assertEquals("Okay, run the tests.", restore("OK, run the tests", "Run the tests."))
        assertEquals("Okay, next sentence.", restore("ok. Next sentence", "Next sentence."))
        assertEquals("OK, run the tests.", restore("okay run the tests", "OK, run the tests."))
    }

    @Test
    fun firstWordCapitalKeptWhereItBelongs() {
        assertEquals("Okay, Sam said the build is green.", restore("Okay, Sam said the build is green", "Sam said the build is green."))
        assertEquals("Okay, I think so.", restore("okay I think so", "I think so."))
        assertEquals("And I'm done.", restore("and I'm done", "I'm done."))
        assertEquals("Well, API keys rotate.", restore("well API keys rotate", "API keys rotate."))
        assertEquals("Okay, iPhone builds pass.", restore("okay iPhone builds pass", "iPhone builds pass."))
        assertEquals("Now, GitHub is down.", restore("now GitHub is down", "GitHub is down."))
        assertEquals("Okay, the build is green.", restore("Okay. The build is green", "The build is green."))
    }

    @Test
    fun soOtherWordsEmptyAndOpenerOnlyInputs() {
        for ((raw, refined) in listOf(
            "So the build is green" to "The build is green.",
            "Okayish results today" to "Results today.",
            "Nowhere to go" to "There is nowhere to go.",
            "Well it depends" to "well, it depends.",
        )) assertEquals(raw, refined, restore(raw, refined))
        assertEquals("", restore("Okay, ship it", ""))
        assertEquals("  ", restore("Okay, ship it", "  "))
        assertEquals("Ship it.", restore("", "Ship it."))
        assertEquals("Okay.", restore("Okay.", "Okay."))
        assertEquals("Sure.", restore("Okay.", "Sure."))
        assertEquals("Hmm.", restore("and", "Hmm."))
        assertEquals("okay ship it", restore("okay ship it", "okay ship it"))
    }

    @Test
    fun runsBeforeLineBreaks() {
        assertEquals("Okay, do this:\n1. open the file\n2. fix the bug",
            LineBreaks.apply(restore("okay do this one open the file two fix the bug", "Do this: 1. open the file 2. fix the bug"), "prompt"))
    }
}
