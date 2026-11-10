package ai.phonon.keyboard

import android.text.InputType
import org.json.JSONObject

/**
 * Phonon's WebSocket API, protocol 1 (docs/websocket-api.md): the messages this phone sends to a Mac running
 * Phonon and what it makes of the replies. Pure functions over org.json, so the JVM unit tests cover them.
 */
object MacProtocol {
    const val PROTOCOL = 1
    const val SAMPLE_RATE = 16000
    /** Characters of the field kept on each side of the cursor (the Mac keeps no more). */
    const val TEXTBOX_CHARS = 2000

    fun hello(token: String, client: String): String =
        JSONObject().put("type", "hello").put("token", token).put("client", client).toString()

    /**
     * `start` for dictation [id] into the app [packageName]. [before]/[selected]/[after] are the field's text
     * around the cursor; null leaves the text box out (a password field, or no field).
     */
    fun start(id: String, packageName: String?, before: String?, selected: String?, after: String?,
              live: Boolean): String {
        val o = JSONObject().put("type", "start").put("id", id).put("live", live)
        if (!packageName.isNullOrEmpty()) o.put("app", JSONObject().put("package", packageName))
        if (before != null || selected != null || after != null) {
            o.put("textbox", JSONObject()
                .put("before_text", (before ?: "").takeLast(TEXTBOX_CHARS))
                .put("selected_text", (selected ?: "").take(TEXTBOX_CHARS))
                .put("after_text", (after ?: "").take(TEXTBOX_CHARS)))
        }
        return o.toString()
    }

    fun finish(id: String): String = JSONObject().put("type", "finish").put("id", id).toString()
    fun cancel(id: String): String = JSONObject().put("type", "cancel").put("id", id).toString()
    fun ping(): String = JSONObject().put("type", "ping").toString()

    /** A message from the Mac. Unknown types and fields are ignored, as the protocol asks. */
    sealed class Reply {
        data class Ready(val protocol: Int, val asrReady: Boolean, val llmReady: Boolean, val busy: Boolean) : Reply()
        data class Started(val id: String) : Reply()
        /** `partial` (raw) or `draft` (corrected): the text so far, a preview only. */
        data class Preview(val id: String, val text: String, val corrected: Boolean) : Reply()
        data class Final(val id: String, val text: String, val raw: String, val refined: Boolean,
                         val reason: String?) : Reply()
        data class Cancelled(val id: String) : Reply()
        data class Warning(val id: String?, val msg: String) : Reply()
        data class Error(val id: String?, val code: String, val msg: String) : Reply()
        object Pong : Reply()
        object Other : Reply()
    }

    fun parse(text: String): Reply {
        val o = try { JSONObject(text) } catch (e: Exception) { return Reply.Other }
        val id = if (o.isNull("id")) null else o.optString("id")
        return when (o.optString("type")) {
            "ready" -> Reply.Ready(o.optInt("protocol", 0), o.optBoolean("asr_ready"), o.optBoolean("llm_ready"),
                o.optBoolean("busy"))
            "started" -> Reply.Started(id ?: "")
            "partial" -> Reply.Preview(id ?: "", o.optString("text"), corrected = false)
            "draft" -> Reply.Preview(id ?: "", o.optString("text"), corrected = true)
            "final" -> Reply.Final(id ?: "", o.optString("text"), o.optString("raw"), o.optBoolean("refined"),
                if (o.isNull("reason")) null else o.optString("reason").ifEmpty { null })
            "cancelled" -> Reply.Cancelled(id ?: "")
            "warning" -> Reply.Warning(id, o.optString("msg"))
            "error" -> Reply.Error(id, o.optString("code", "engine"), o.optString("msg"))
            "pong" -> Reply.Pong
            else -> Reply.Other
        }
    }

    /**
     * Why the Mac cannot take a dictation, from its `ready`, or null when it can. Any such reason, an `error`, or a
     * dead connection means the phone transcribes the recording itself.
     */
    fun unavailable(ready: Reply.Ready): String? = when {
        ready.protocol != PROTOCOL -> "the Mac speaks protocol ${ready.protocol}, this app $PROTOCOL"
        !ready.asrReady || !ready.llmReady -> "the Mac is still loading its models"
        ready.busy -> "the Mac is dictating"
        else -> null
    }

    /** A short line for the user about an `error` from the Mac. */
    fun explain(error: Reply.Error): String = when (error.code) {
        "unauthorized" -> "the Mac refused the token; copy it again from Phonon's Settings"
        "busy" -> "the Mac is dictating"
        "preempted" -> "a dictation started on the Mac"
        "not_ready" -> "the Mac is still loading its models"
        "too_long" -> "the dictation was too long for the Mac"
        "timeout" -> "the Mac took too long"
        else -> "the Mac failed: ${error.msg.take(80)}"
    }

    /** Password fields never send their text to the Mac. */
    fun isPassword(inputType: Int): Boolean {
        val variation = inputType and InputType.TYPE_MASK_VARIATION
        return when (inputType and InputType.TYPE_MASK_CLASS) {
            InputType.TYPE_CLASS_TEXT -> variation == InputType.TYPE_TEXT_VARIATION_PASSWORD ||
                variation == InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD ||
                variation == InputType.TYPE_TEXT_VARIATION_WEB_PASSWORD
            InputType.TYPE_CLASS_NUMBER -> variation == InputType.TYPE_NUMBER_VARIATION_PASSWORD
            else -> false
        }
    }

    /**
     * The text to commit: one line for a single-line field (some apps take a newline as Enter and would send half
     * the message), and a space before it when it would otherwise run into the word before the cursor.
     */
    fun textToInsert(text: String, inputType: Int, before: String?): String {
        val multiLine = (inputType and InputType.TYPE_TEXT_FLAG_MULTI_LINE) != 0
        var out = if (multiLine) text else text.replace(Regex("\\s*\\n+\\s*"), " ")
        out = out.trim(' ')
        if (out.isEmpty()) return ""
        val last = before?.lastOrNull()
        if (last != null && !last.isWhitespace() && !out.first().isWhitespace() && out.first() !in ".,;:!?)") {
            out = " $out"
        }
        return out
    }

    /**
     * The Mac's pairing code (Settings › Advanced › WebSocket API on the Mac):
     * `phonon://pair?url=ws%3A%2F%2F192.168.0.52%3A8377&token=…`. Returns (url, token), or null for anything
     * else, so a stray QR code never changes the settings.
     */
    fun parsePairing(text: String): Pair<String, String>? {
        val trimmed = text.trim()
        val prefix = "phonon://pair?"
        if (!trimmed.startsWith(prefix, ignoreCase = true)) return null
        val params = trimmed.substring(prefix.length).split('&').mapNotNull { part ->
            val key = part.substringBefore('=', "")
            if (key.isEmpty()) null
            else key to try {
                java.net.URLDecoder.decode(part.substringAfter('='), "UTF-8")
            } catch (e: IllegalArgumentException) {
                return null
            }
        }.toMap()
        val url = params["url"]?.let(::normalizeUrl) ?: return null
        val token = params["token"]?.trim()?.takeIf { it.length >= 16 && it.none(Char::isWhitespace) } ?: return null
        return url to token
    }

    /** `ws://host:port/` from what the user typed: a bare host, host:port, or a full ws:// URL. */
    fun normalizeUrl(input: String): String? {
        var s = input.trim()
        if (s.isEmpty()) return null
        if (s.startsWith("wss://", ignoreCase = true)) return s
        if (!s.startsWith("ws://", ignoreCase = true)) s = "ws://$s"
        val rest = s.substring(5)
        val hostPort = rest.substringBefore('/')
        if (hostPort.isEmpty() || hostPort.any { it.isWhitespace() }) return null
        val withPort = if (hostPort.startsWith("[")) {
            if (hostPort.contains("]:")) hostPort else "$hostPort:8377"
        } else if (hostPort.contains(':')) hostPort else "$hostPort:8377"
        return "ws://$withPort/"
    }
}
