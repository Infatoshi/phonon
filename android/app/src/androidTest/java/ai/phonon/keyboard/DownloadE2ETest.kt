package ai.phonon.keyboard

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.ParcelFileDescriptor
import android.os.SystemClock
import android.util.Log
import android.view.inputmethod.InputMethodManager
import android.widget.EditText
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/**
 * Cold install to first dictation on a real phone (Firebase Test Lab), with the models fetched from Hugging Face
 * by the app itself. Only the WAVs arrive with --other-files (under /sdcard/phonon/); no model is sideloaded.
 *  - removes any model files, then seeds the first [SEED] bytes of the encoder's weights as a ".part" with a direct
 *    Range request, so the app's own download must resume it through Hugging Face's redirect (logged http=206);
 *  - opens the launcher activity and the keyboard, as a user would, and times both to the first dictation
 *    (recogniser downloaded, verified, loaded) and to the refiner (or its RAM-gate skip);
 *  - dictates s1.wav after each and reads the field back.
 * Lines go to logcat under tag "Phonon" with the prefix DLT.
 */
class DownloadE2ETest {
    private val inst = InstrumentationRegistry.getInstrumentation()
    private val ctx: Context = inst.targetContext
    private val pkg = ctx.packageName
    private val ime = "$pkg/.PhononIME"
    private val ext: File = ctx.getExternalFilesDir(null)!!

    private fun log(s: String) {
        Log.i(TAG, "DLT $s")
        println("DLT $s")
    }

    private fun sh(cmd: String): String {
        val pfd = inst.uiAutomation.executeShellCommand(cmd)
        return ParcelFileDescriptor.AutoCloseInputStream(pfd).use { it.readBytes().toString(Charsets.UTF_8) }
    }

    private fun stage(src: String, dst: File) {
        dst.parentFile!!.mkdirs()
        val pfd = inst.uiAutomation.executeShellCommand("cat $src")
        ParcelFileDescriptor.AutoCloseInputStream(pfd).use { input -> dst.outputStream().use { input.copyTo(it, 1 shl 20) } }
    }

    private fun phononLog(): List<String> = sh("logcat -d -v brief -s Phonon:I").lines()  // I and above; a second "Phonon:E" spec would replace it

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

    /** The first [bytes] of [f] into its ".part", fetched directly (not by ModelStore). */
    private fun seedPart(store: ModelStore, f: ModelFile, bytes: Long) {
        val part = File(store.root, f.path + ".part").apply { parentFile!!.mkdirs() }
        // Test Lab's network drops connections now and then (seen twice on 2026-11-08): retry the seed itself
        for (attempt in 1..4) {
            try {
                val c = URL(f.url).openConnection() as HttpURLConnection
                c.setRequestProperty("Range", "bytes=0-${bytes - 1}")
                c.inputStream.use { input -> part.outputStream().use { input.copyTo(it, 1 shl 16) } }
                log("SEED file=${f.path} http=${c.responseCode} bytes=${part.length()} attempt=$attempt")
                if (part.length() == bytes) return
            } catch (e: java.io.IOException) {
                log("SEED retry attempt=$attempt error=${e.message}")
                SystemClock.sleep(3000L * attempt)
            }
        }
        error("could not seed ${f.path}")
    }

    @Test
    fun coldInstallToFirstDictation() {
        log("DEVICE manufacturer=${Build.MANUFACTURER} model=${Build.MODEL} sdk=${Build.VERSION.SDK_INT} " +
            "memtotal_mb=${Models.memTotalMb()}")
        // cold: no sideloaded models, no downloaded ones
        File(ext, "parakeet").deleteRecursively()
        File(ext, "refiner.gguf").delete()
        val store = ModelStore(File(ctx.filesDir, "models"))
        store.root.deleteRecursively()
        stage("/sdcard/phonon/s1.wav", File(ext, "s1.wav"))
        seedPart(store, Models.ASR.first { it.path.endsWith(".data") }, SEED)

        sh("pm grant $pkg android.permission.RECORD_AUDIO")
        log("IME enable: ${sh("ime enable $ime").trim()} | set: ${sh("ime set $ime").trim()}")
        sh("logcat -c")

        val t0 = SystemClock.elapsedRealtime()
        val activity = inst.startActivitySync(
            Intent(ctx, TestActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) as Activity
        lateinit var field: EditText
        inst.runOnMainSync { field = activity.findViewById(R.id.field) }
        inst.runOnMainSync {
            field.requestFocus()
            (ctx.getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager).showSoftInput(field, 0)
        }

        val asr = waitLog(1500) { l -> l.any { it.contains("ASR_WARMUP") || it.contains("ASR_LOAD ok=false") || it.contains("MODELS failed") } }
        val tAsr = SystemClock.elapsedRealtime() - t0
        asr.filter { Regex("MODELS|DOWNLOAD|ASR_").containsMatchIn(it) }.forEach { log("LOG ${it.substringAfter("Phonon").take(300)}") }
        val text1 = dictate(field)
        val tFirst = SystemClock.elapsedRealtime() - t0
        val ok1 = text1.contains("quarterly report", ignoreCase = true)
        log("FIRST_DICTATION asr_ready_ms=$tAsr committed_ms=$tFirst ok=$ok1 chars=${text1.length}")

        val ref = waitLog(1500) { l -> l.any { it.contains("REFINER_LOAD") || it.contains("MODELS failed") } }
        val tRef = SystemClock.elapsedRealtime() - t0
        ref.filter { Regex("MODELS|DOWNLOAD|REFINER_LOAD|TUNE").containsMatchIn(it) }.forEach { log("LOG ${it.substringAfter("Phonon").take(300)}") }
        SystemClock.sleep(3000)  // the refiner's warm-up
        val text2 = dictate(field)
        val ok2 = text2.contains("quarterly report", ignoreCase = true)
        log("REFINER_DICTATION ready_ms=$tRef loaded=${ref.any { it.contains("REFINER_LOAD ok=true") }} ok=$ok2 chars=${text2.length}")
        for (f in Models.ASR + Models.REFINER) log("FILE ${f.path} ready=${store.isReady(f)} bytes=${store.file(f).length()}")
        log("DONE")
        assertTrue("first dictation did not reach the field", ok1)
        assertTrue("dictation after the refiner did not reach the field", ok2)
        // above the RAM gate on Test Lab's network (unmetered) the refiner must actually arrive and load
        if (Models.memTotalMb() >= Models.REFINER_MIN_TOTAL_MB)
            assertTrue("the refiner did not load", ref.any { it.contains("REFINER_LOAD ok=true") })
        assertTrue("a model step failed", ref.none { it.contains("MODELS failed") })
    }

    companion object {
        const val SEED = 50L shl 20
    }
}
