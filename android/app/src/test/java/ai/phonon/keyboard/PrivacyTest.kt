package ai.phonon.keyboard

import android.text.InputType
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** Privacy: what a log line may say about the user's text, and which fields count as password fields. */
class PrivacyTest {
    private val text = InputType.TYPE_CLASS_TEXT
    private val number = InputType.TYPE_CLASS_NUMBER

    @Test
    fun passwordFieldsOfEveryKind() {
        assertTrue(Privacy.isPassword(text or InputType.TYPE_TEXT_VARIATION_PASSWORD))
        assertTrue(Privacy.isPassword(text or InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD))
        assertTrue(Privacy.isPassword(text or InputType.TYPE_TEXT_VARIATION_WEB_PASSWORD))
        assertTrue(Privacy.isPassword(number or InputType.TYPE_NUMBER_VARIATION_PASSWORD))
        assertTrue(Privacy.isPassword(text or InputType.TYPE_TEXT_VARIATION_PASSWORD or InputType.TYPE_TEXT_FLAG_MULTI_LINE))
    }

    @Test
    fun ordinaryFieldsAreNotPasswords() {
        assertFalse(Privacy.isPassword(InputType.TYPE_NULL))
        assertFalse(Privacy.isPassword(text))
        assertFalse(Privacy.isPassword(text or InputType.TYPE_TEXT_FLAG_MULTI_LINE))
        assertFalse(Privacy.isPassword(text or InputType.TYPE_TEXT_VARIATION_EMAIL_ADDRESS))
        // numeric password's variation bits are text URI's: the class decides
        assertFalse(Privacy.isPassword(text or InputType.TYPE_TEXT_VARIATION_URI))
        assertFalse(Privacy.isPassword(number))
    }

    @Test
    fun releaseFieldNamesNeitherAppNorInputType() {
        val type = text or InputType.TYPE_TEXT_VARIATION_PASSWORD
        assertEquals("password=true", Privacy.field("com.example.bank", type, debug = false))
        assertEquals("password=false", Privacy.field("com.example.bank", text, debug = false))
        assertEquals("app=com.example.bank inputType=0x81 password=true", Privacy.field("com.example.bank", type, debug = true))
    }

    @Test
    fun keyThatTypesTextIsNeverNamed() {
        assertEquals("char", Privacy.keyName("q", typesText = true))
        assertEquals("char", Privacy.keyName("7", typesText = true))
        assertEquals("123", Privacy.keyName("123", typesText = false))
    }

    @Test
    fun refineStatsCarryLengthsNotText() {
        val json = """{"text":"my pin is 4821","raw":"my pin is four eight two one","mode":"minimal",""" +
            """"prompt_tokens":40,"gen_tokens":6,"decode_ms":120,"guard":false,"tok_hash":"00ff00ff00ff00ff"}"""
        val s = Privacy.refineStats(json)
        for (secret in listOf("4821", "four", "pin", "00ff00ff00ff00ff")) assertFalse("$secret in $s", s.contains(secret))
        val o = JSONObject(s)
        assertEquals(14, o.getInt("text_chars"))
        assertEquals(28, o.getInt("raw_chars"))
        assertEquals(120, o.getInt("decode_ms"))
        assertEquals("minimal", o.getString("mode"))
        assertFalse(o.has("text") || o.has("raw") || o.has("tok_hash"))
        assertTrue(s.startsWith("{"))  // ImeE2ETest waits for " WARMUP {"
        assertEquals("null", Privacy.refineStats(null))
        assertEquals("unparsed_chars=11", Privacy.refineStats("not json pw"))
    }
}
