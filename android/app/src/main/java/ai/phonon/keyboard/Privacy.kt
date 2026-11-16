package ai.phonon.keyboard

import android.content.Context
import android.content.pm.ApplicationInfo
import android.text.InputType
import org.json.JSONException
import org.json.JSONObject

/**
 * What the keyboard may tell logcat, which adb (and, on some devices, apps granted log access) can read. Log lines
 * carry content-free facts only: event names, counts, lengths, durations, error classes. Typed keys, dictated or
 * corrected text, text around the cursor and dictionary entries never go to the log in any build; the host app's
 * package name and the field's input type only in debuggable builds.
 */
object Privacy {
    /** A password field of any kind (text, visible, web, numeric PIN): its dictation skips the dictionary and the
     *  refiner and is committed as heard. Numeric password shares its variation bits with text URI, hence the class. */
    fun isPassword(inputType: Int): Boolean {
        val v = inputType and InputType.TYPE_MASK_VARIATION
        return when (inputType and InputType.TYPE_MASK_CLASS) {
            InputType.TYPE_CLASS_TEXT -> v == InputType.TYPE_TEXT_VARIATION_PASSWORD ||
                v == InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD || v == InputType.TYPE_TEXT_VARIATION_WEB_PASSWORD
            InputType.TYPE_CLASS_NUMBER -> v == InputType.TYPE_NUMBER_VARIATION_PASSWORD
            else -> false
        }
    }

    fun debuggable(c: Context) = (c.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE) != 0

    /** The field for a log line: whether it is a password field; the package and input type only when [debug]
     *  (ImeE2ETest matches "app=PKG " and the input type on debug builds). */
    fun field(pkg: String?, inputType: Int, debug: Boolean): String {
        val p = "password=${isPassword(inputType)}"
        return if (debug) "app=$pkg inputType=0x${Integer.toHexString(inputType)} $p" else p
    }

    /** A key press for the log: a key that types text is "char", never which one; other keys by name. */
    fun keyName(name: String, typesText: Boolean) = if (typesText) "char" else name

    /** The refiner's JSON with its text replaced by lengths ("text" and "raw" become "text_chars" and "raw_chars")
     *  and "tok_hash", a fingerprint of the output, dropped; the timings and counts stay. */
    fun refineStats(json: String?): String {
        if (json == null) return "null"
        val o = try { JSONObject(json) } catch (e: JSONException) { return "unparsed_chars=${json.length}" }
        for (k in listOf("text", "raw")) if (o.has(k)) {
            o.put("${k}_chars", o.optString(k).length)
            o.remove(k)
        }
        o.remove("tok_hash")
        return o.toString()
    }
}
