package ai.phonon.keyboard

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.os.ParcelFileDescriptor
import android.os.SystemClock
import android.util.Log
import android.view.View
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.EditText
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * The refiner waits for Wi-Fi (Firebase Test Lab; run it alone: ModelFetch is one per process). The network is
 * forced to read as metered, so: the recogniser still downloads and raw dictation works; the refiner moves no byte
 * for [HOLD_S] seconds and the launcher shows its mobile-data button; one tap on that button starts the refiner,
 * which then loads and refines. Needs a phone above the RAM gate. Lines go to logcat under "Phonon" with prefix WFT.
 */
class WifiOnlyE2ETest {
    private val inst = InstrumentationRegistry.getInstrumentation()
    private val ctx: Context = inst.targetContext
    private val pkg = ctx.packageName
    private val ext: File = ctx.getExternalFilesDir(null)!!

    private fun log(s: String) {
        Log.i(TAG, "WFT $s")
        println("WFT $s")
    }

    private fun sh(cmd: String): String {
        val pfd = inst.uiAutomation.executeShellCommand(cmd)
        return ParcelFileDescriptor.AutoCloseInputStream(pfd).use { it.readBytes().toString(Charsets.UTF_8) }
    }

    private fun phononLog(): List<String> = sh("logcat -d -v brief -s Phonon:I").lines()

    private fun waitLog(seconds: Int, pred: (List<String>) -> Boolean): List<String> {
        val end = SystemClock.elapsedRealtime() + seconds * 1000L
        var lines = phononLog()
        while (!pred(lines) && SystemClock.elapsedRealtime() < end) {
            SystemClock.sleep(1000)
            lines = phononLog()
        }
        return lines
    }

    private fun dictate(field: EditText): String {
        inst.runOnMainSync {
            field.setText("")
            field.requestFocus()
            (ctx.getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager).showSoftInput(field, 0)
        }
        SystemClock.sleep(1500)
        val before = phononLog().count { it.contains(" E2E source=") }
        ctx.sendBroadcast(Intent(PhononIME.ACTION_WAV).setPackage(pkg).putExtra("path", File(ext, "s1.wav").path))
        val lines = waitLog(180) { l -> l.count { it.contains(" E2E source=") } > before }
        log("E2E ${lines.lastOrNull { it.contains(" E2E source=") }?.substringAfter("E2E ")?.substringBefore(" text=")}")
        var s = ""
        inst.runOnMainSync { s = field.text.toString() }
        return s
    }

    @Test
    fun refinerWaitsForWifiUntilTheUserAllowsMobileData() {
        File(ext, "parakeet").deleteRecursively()
        File(ext, "refiner.gguf").delete()
        File(ctx.filesDir, "models").deleteRecursively()
        ctx.getSharedPreferences("phonon", Context.MODE_PRIVATE).edit().clear().commit()
        val stage = inst.uiAutomation.executeShellCommand("cat /sdcard/phonon/s1.wav")
        ParcelFileDescriptor.AutoCloseInputStream(stage).use { i -> File(ext, "s1.wav").outputStream().use { i.copyTo(it) } }
        log("DEVICE memtotal_mb=${Models.memTotalMb()}")
        ModelFetch.forceMetered = true

        sh("pm grant $pkg android.permission.RECORD_AUDIO")
        sh("ime enable $pkg/.PhononIME"); sh("ime set $pkg/.PhononIME")
        sh("logcat -c")
        val t0 = SystemClock.elapsedRealtime()
        val activity = inst.startActivitySync(
            Intent(ctx, TestActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) as Activity
        lateinit var field: EditText
        lateinit var button: Button
        inst.runOnMainSync {
            field = activity.findViewById(R.id.field)
            button = activity.findViewById(R.id.cellular)
        }

        val waiting = waitLog(900) { l -> l.any { it.contains("refiner_wait_unmetered") } && l.any { it.contains("ASR_WARMUP") } }
        log("WAITING ms=${SystemClock.elapsedRealtime() - t0} wait_logged=${waiting.any { it.contains("refiner_wait_unmetered") }}")
        val raw = dictate(field)
        log("RAW_DICTATION ok=${raw.contains("quarterly report", ignoreCase = true)} chars=${raw.length}")

        SystemClock.sleep(HOLD_S * 1000L)
        val part = File(ctx.filesDir, "models/refiner.gguf.part")
        val started = phononLog().any { it.contains("DOWNLOAD start file=refiner.gguf") }
        var shown = false
        inst.runOnMainSync { shown = button.visibility == View.VISIBLE }
        log("HELD seconds=$HOLD_S refiner_part_bytes=${part.length()} refiner_started=$started button_visible=$shown " +
            "state_waiting=${ModelFetch.state.waitingForWifi}")

        inst.runOnMainSync { button.performClick() }  // the user's one tap
        val tTap = SystemClock.elapsedRealtime()
        val loaded = waitLog(1200) { l -> l.any { it.contains("REFINER_LOAD") } }
        log("AFTER_TAP refiner_load_ms=${SystemClock.elapsedRealtime() - tTap} " +
            "ok=${loaded.any { it.contains("REFINER_LOAD ok=true") }} " +
            "cellular_pref=${ctx.getSharedPreferences("phonon", Context.MODE_PRIVATE).getBoolean("refiner_on_cellular", false)}")
        SystemClock.sleep(3000)
        val refined = dictate(field)
        log("REFINED_DICTATION ok=${refined.contains("quarterly report", ignoreCase = true)} chars=${refined.length}")
        log("DONE")

        assertTrue("raw dictation while waiting", raw.contains("quarterly report", ignoreCase = true))
        assertFalse("the refiner started on a metered network", started)
        assertEquals("refiner bytes on a metered network", 0L, part.length())
        assertTrue("the mobile-data button was not shown", shown)
        assertTrue("the refiner did not load after the tap", loaded.any { it.contains("REFINER_LOAD ok=true") })
        assertTrue("dictation after the refiner", refined.contains("quarterly report", ignoreCase = true))
    }

    companion object {
        const val HOLD_S = 20
    }
}
