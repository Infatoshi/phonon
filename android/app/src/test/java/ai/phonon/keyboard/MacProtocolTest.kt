package ai.phonon.keyboard

import android.text.InputType
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** The messages of the Mac's WebSocket API (docs/websocket-api.md) and what the voice input makes of them. */
class MacProtocolTest {
    @Test
    fun helloCarriesTheTokenAndAName() {
        val o = JSONObject(MacProtocol.hello("tok-123", "phonon-android"))
        assertEquals("hello", o.getString("type"))
        assertEquals("tok-123", o.getString("token"))
        assertEquals("phonon-android", o.getString("client"))
    }

    @Test
    fun startNamesTheAppAndTheTextAroundTheCursor() {
        val o = JSONObject(MacProtocol.start("d1", "com.Slack", "Thanks! ", "", "", live = true))
        assertEquals("start", o.getString("type"))
        assertEquals("d1", o.getString("id"))
        assertTrue(o.getBoolean("live"))
        assertEquals("com.Slack", o.getJSONObject("app").getString("package"))
        val textbox = o.getJSONObject("textbox")
        assertEquals("Thanks! ", textbox.getString("before_text"))
        assertEquals("", textbox.getString("selected_text"))
        assertEquals("", textbox.getString("after_text"))
    }

    @Test
    fun startLeavesOutWhatIsNotKnown() {
        val o = JSONObject(MacProtocol.start("d2", null, null, null, null, live = false))
        assertFalse(o.has("app"))
        assertFalse(o.has("textbox"))
        assertFalse(o.getBoolean("live"))
        assertFalse(JSONObject(MacProtocol.start("d3", "", null, null, null, false)).has("app"))
    }

    @Test
    fun theTextBoxKeepsTheCharactersNearestTheCursor() {
        val before = "a".repeat(2500) + "END"
        val after = "START" + "b".repeat(2500)
        val textbox = JSONObject(MacProtocol.start("d1", null, before, null, after, false)).getJSONObject("textbox")
        assertEquals(MacProtocol.TEXTBOX_CHARS, textbox.getString("before_text").length)
        assertTrue(textbox.getString("before_text").endsWith("END"))
        assertEquals(MacProtocol.TEXTBOX_CHARS, textbox.getString("after_text").length)
        assertTrue(textbox.getString("after_text").startsWith("START"))
    }

    @Test
    fun controlMessages() {
        // key order is up to the JSON library: compare fields
        fun fields(json: String) = JSONObject(json).let { o -> o.keys().asSequence().associateWith { o.get(it) } }
        assertEquals(mapOf("type" to "finish", "id" to "d1"), fields(MacProtocol.finish("d1")))
        assertEquals(mapOf("type" to "cancel", "id" to "d1"), fields(MacProtocol.cancel("d1")))
        assertEquals(mapOf("type" to "ping"), fields(MacProtocol.ping()))
    }

    @Test
    fun repliesParse() {
        assertEquals(MacProtocol.Reply.Ready(1, true, true, false), MacProtocol.parse(
            """{"type":"ready","protocol":1,"version":"1.0.0","asr_ready":true,"llm_ready":true,""" +
                """"stream_refine":true,"busy":false,"sample_rate":16000,"max_seconds":1200,"later":"ignored"}"""))
        assertEquals(MacProtocol.Reply.Started("d1"), MacProtocol.parse("""{"type":"started","id":"d1"}"""))
        assertEquals(MacProtocol.Reply.Preview("d1", "we deploy", corrected = false),
            MacProtocol.parse("""{"type":"partial","id":"d1","text":"we deploy"}"""))
        assertEquals(MacProtocol.Reply.Preview("d1", "We deploy", corrected = true),
            MacProtocol.parse("""{"type":"draft","id":"d1","text":"We deploy","complete":true}"""))
        assertEquals(MacProtocol.Reply.Final("d1", "We deploy.", "we deploy", true, null), MacProtocol.parse(
            """{"type":"final","id":"d1","text":"We deploy.","raw":"we deploy","refined":true,"app_category":"work_chat"}"""))
        assertEquals(MacProtocol.Reply.Final("d1", "", "", false, "no_speech"), MacProtocol.parse(
            """{"type":"final","id":"d1","text":"","raw":"","refined":false,"reason":"no_speech"}"""))
        assertEquals(MacProtocol.Reply.Error(null, "unauthorized", "bad token"),
            MacProtocol.parse("""{"type":"error","id":null,"code":"unauthorized","msg":"bad token"}"""))
        assertEquals(MacProtocol.Reply.Error("d1", "busy", "x"),
            MacProtocol.parse("""{"type":"error","id":"d1","code":"busy","msg":"x"}"""))
        assertEquals(MacProtocol.Reply.Warning("d1", "lost a piece"),
            MacProtocol.parse("""{"type":"warning","id":"d1","msg":"lost a piece"}"""))
        assertEquals(MacProtocol.Reply.Cancelled("d1"), MacProtocol.parse("""{"type":"cancelled","id":"d1"}"""))
        assertEquals(MacProtocol.Reply.Pong, MacProtocol.parse("""{"type":"pong"}"""))
        // what a later Mac may add, and what is not JSON at all
        assertEquals(MacProtocol.Reply.Other, MacProtocol.parse("""{"type":"something_new","id":"d1"}"""))
        assertEquals(MacProtocol.Reply.Other, MacProtocol.parse("not json"))
    }

    @Test
    fun theMacSaysWhenItCannotTakeADictation() {
        assertNull(MacProtocol.unavailable(MacProtocol.Reply.Ready(1, true, true, false)))
        assertEquals("the Mac is dictating", MacProtocol.unavailable(MacProtocol.Reply.Ready(1, true, true, true)))
        assertEquals("the Mac is still loading its models",
            MacProtocol.unavailable(MacProtocol.Reply.Ready(1, true, false, false)))
        assertEquals("the Mac speaks protocol 2, this app 1",
            MacProtocol.unavailable(MacProtocol.Reply.Ready(2, true, true, false)))
        assertEquals("a dictation started on the Mac",
            MacProtocol.explain(MacProtocol.Reply.Error("d1", "preempted", "")))
        assertTrue(MacProtocol.explain(MacProtocol.Reply.Error(null, "unauthorized", "")).contains("token"))
        assertTrue(MacProtocol.explain(MacProtocol.Reply.Error("d1", "something_new", "boom")).contains("boom"))
    }

    @Test
    fun passwordFieldsAreRecognised() {
        val text = InputType.TYPE_CLASS_TEXT
        assertTrue(MacProtocol.isPassword(text or InputType.TYPE_TEXT_VARIATION_PASSWORD))
        assertTrue(MacProtocol.isPassword(text or InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD))
        assertTrue(MacProtocol.isPassword(text or InputType.TYPE_TEXT_VARIATION_WEB_PASSWORD))
        assertTrue(MacProtocol.isPassword(InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_VARIATION_PASSWORD))
        assertFalse(MacProtocol.isPassword(text or InputType.TYPE_TEXT_VARIATION_EMAIL_ADDRESS))
        assertFalse(MacProtocol.isPassword(text or InputType.TYPE_TEXT_FLAG_MULTI_LINE))
        assertFalse(MacProtocol.isPassword(0))
    }

    @Test
    fun insertedTextFitsTheField() {
        val multi = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
        val single = InputType.TYPE_CLASS_TEXT
        val email = "Hi Sarah,\n\nThanks.\n\nBerni"
        assertEquals(email, MacProtocol.textToInsert(email, multi, null))
        assertEquals("Hi Sarah, Thanks. Berni", MacProtocol.textToInsert(email, single, null))
        // a space only where the text would run into the word before the cursor
        assertEquals(" See you at 5.", MacProtocol.textToInsert("See you at 5.", single, "ok"))
        assertEquals("See you at 5.", MacProtocol.textToInsert("See you at 5.", single, "ok "))
        assertEquals("See you at 5.", MacProtocol.textToInsert("See you at 5.", single, ""))
        assertEquals("See you at 5.", MacProtocol.textToInsert("See you at 5.", single, null))
        assertEquals(", and you?", MacProtocol.textToInsert(", and you?", single, "fine"))
        assertEquals("", MacProtocol.textToInsert("", multi, "word"))
    }

    @Test
    fun addressesBecomeWebSocketUrls() {
        assertEquals("ws://192.168.1.20:8377/", MacProtocol.normalizeUrl("192.168.1.20"))
        assertEquals("ws://192.168.1.20:9000/", MacProtocol.normalizeUrl(" 192.168.1.20:9000 "))
        assertEquals("ws://100.64.0.7:8377/", MacProtocol.normalizeUrl("ws://100.64.0.7:8377"))
        assertEquals("ws://mac.tail1234.ts.net:8377/", MacProtocol.normalizeUrl("mac.tail1234.ts.net"))
        assertEquals("ws://[fd7a:115c::1]:8377/", MacProtocol.normalizeUrl("[fd7a:115c::1]"))
        assertEquals("wss://mac.example.ts.net/", MacProtocol.normalizeUrl("wss://mac.example.ts.net/"))
        assertNull(MacProtocol.normalizeUrl(""))
        assertNull(MacProtocol.normalizeUrl("   "))
        assertNull(MacProtocol.normalizeUrl("ws://"))
    }

    @Test
    fun theMacsPairingCodeFillsTheSettings() {
        // exactly what the Mac's QR code holds (bar: WebSocketAPI.pairingURI)
        assertEquals("ws://192.168.0.52:8377/" to "fzqxz6qcsyh3cptvew5atmfs", MacProtocol.parsePairing(
            "phonon://pair?url=ws%3A%2F%2F192.168.0.52%3A8377&token=fzqxz6qcsyh3cptvew5atmfs"))
        // order, case and an unencoded url do not matter
        assertEquals("ws://100.64.0.7:8377/" to "fzqxz6qcsyh3cptvew5atmfs", MacProtocol.parsePairing(
            " PHONON://pair?token=fzqxz6qcsyh3cptvew5atmfs&url=ws://100.64.0.7:8377 "))
        // anything else changes nothing
        assertNull(MacProtocol.parsePairing("https://example.com/?url=ws://1.2.3.4&token=fzqxz6qcsyh3cptvew5atmfs"))
        assertNull(MacProtocol.parsePairing("phonon://pair?url=ws%3A%2F%2F192.168.0.52%3A8377"))
        assertNull(MacProtocol.parsePairing("phonon://pair?url=ws%3A%2F%2F192.168.0.52%3A8377&token=short"))
        assertNull(MacProtocol.parsePairing("phonon://pair?token=fzqxz6qcsyh3cptvew5atmfs&url="))
        assertNull(MacProtocol.parsePairing("phonon://pair?url=%ZZ&token=fzqxz6qcsyh3cptvew5atmfs"))
        assertNull(MacProtocol.parsePairing("not a link"))
    }
}
