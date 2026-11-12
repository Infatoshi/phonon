package ai.phonon.keyboard

import android.os.Build
import android.os.ParcelFileDescriptor
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertNotNull
import org.junit.Test
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/** The pinned public refiner (Infatoshi/phonon-refiner-1.7b @2b0ab74, Q4_K_S-e4), as on the desktop builds. */
const val REFINER_URL = "https://huggingface.co/Infatoshi/phonon-refiner-1.7b/resolve/" +
    "2b0ab74bee7ab7cf364695bbe44f64ce4c47439e/phonon-refiner-1.7b-Q4_K_S-e4.gguf"
const val REFINER_BYTES = 988_453_696L

/**
 * The refiner benchmark on a real phone (Firebase Test Lab): the same 4 transcripts x 2 as the iOS self-test
 * (ios/App/Session.swift selftest) and scripts/m3_refine.sh. rep 0 prefills the fixed head after a mode switch,
 * rep 1 reuses it. Every line goes to logcat under tag "Phonon" with the prefix BENCH.
 *
 * Pass instrumentation arg `model=https://...` (for example REFINER_URL) to download the model on the phone; that
 * replaces Test Lab's --other-files push, which costs about 98 s per phone. Otherwise
 * the model arrives through `--other-files /sdcard/phonon/refiner.gguf=...`. The app cannot read /sdcard without a
 * storage permission, so the test streams it through a shell `cat` (UiAutomation runs as the shell user) into the
 * app's files dir. Override the source path, thread count and weight repacking with instrumentation args `model`, `threads` and `repack` (0 = off).
 */
class RefinerBenchTest {
    private val cases = listOf(
        "minimal" to "um so the first thing we do is call could a malloc to to allocate memory on the GPU",
        "prompt" to "okay so I want you to add a retry to the fetch function like three retries no actually make it five with exponential backoff and um also log every failure to the console",
        "casual" to "yeah I think we should uh grab dinner around seven if that works for you",
        "formal" to "hi sarah just wanted to let you know the the deploy is gonna slip to thursday because the tests are failing on the arm runners",
    )

    private fun log(s: String) {
        Log.i(TAG, "BENCH $s")
        println("BENCH $s")
    }

    private fun cpuSummary(): String {
        val lines = File("/proc/cpuinfo").readLines()
        val features = lines.firstOrNull { it.startsWith("Features") }?.substringAfter(":")?.trim() ?: "?"
        val parts = lines.filter { it.startsWith("CPU part") }.map { it.substringAfter(":").trim() }
        val maxFreq = (0 until Runtime.getRuntime().availableProcessors()).map { i ->
            runCatching { File("/sys/devices/system/cpu/cpu$i/cpufreq/cpuinfo_max_freq").readText().trim() }.getOrDefault("?")
        }
        return "cores=${Runtime.getRuntime().availableProcessors()} parts=${parts.joinToString(",")} " +
            "max_khz=${maxFreq.joinToString(",")} features=$features"
    }

    /** Download over HTTPS into the app's files dir (debug builds carry INTERNET), checking the pinned size. */
    private fun download(url: String, dst: File): File {
        val t = SystemClock.elapsedRealtime()
        val conn = URL(url).openConnection() as HttpURLConnection
        conn.connectTimeout = 15_000
        conn.readTimeout = 60_000
        conn.inputStream.use { input -> dst.outputStream().use { input.copyTo(it, 1 shl 20) } }
        val ms = SystemClock.elapsedRealtime() - t
        log("DOWNLOAD url=$url bytes=${dst.length()} ms=$ms mb_s=${"%.1f".format(dst.length() / 1e6 / (ms / 1000.0))}")
        check(dst.length() == REFINER_BYTES) { "downloaded ${dst.length()} bytes, expected $REFINER_BYTES" }
        return dst
    }

    private fun stageModel(src: String): File {
        val inst = InstrumentationRegistry.getInstrumentation()
        val dst = File(inst.targetContext.filesDir, "refiner.gguf")
        if (src.startsWith("https://")) return download(src, dst)
        val direct = File(src)
        if (direct.canRead()) return direct
        val t = SystemClock.elapsedRealtime()
        val pfd = inst.uiAutomation.executeShellCommand("cat $src")
        ParcelFileDescriptor.AutoCloseInputStream(pfd).use { input ->
            dst.outputStream().use { input.copyTo(it, 1 shl 20) }
        }
        log("STAGE src=$src bytes=${dst.length()} ms=${SystemClock.elapsedRealtime() - t}")
        return dst
    }

    @Test
    fun refinerBench() {
        val args = InstrumentationRegistry.getArguments()
        val threads = args.getString("threads")?.toIntOrNull() ?: 4
        val repack = args.getString("repack") != "0"
        val src = args.getString("model") ?: "/sdcard/phonon/refiner.gguf"
        log("DEVICE manufacturer=${Build.MANUFACTURER} model=${Build.MODEL} device=${Build.DEVICE} " +
            "soc=${if (Build.VERSION.SDK_INT >= 31) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}" else Build.HARDWARE} " +
            "sdk=${Build.VERSION.SDK_INT}")
        log("CPU ${cpuSummary()}")
        val model = stageModel(src)
        log("MODEL path=${model.path} bytes=${model.length()}")
        ModelPin.check(model)
        val t0 = SystemClock.elapsedRealtime()
        val r = Refiner.load(model.path, threads, repack)
        val loadMs = SystemClock.elapsedRealtime() - t0
        log("LOAD ok=${r != null} load_ms=$loadMs threads=$threads repack=$repack ${Mem.snapshot()}")
        assertNotNull("refiner failed to load", r)
        for ((mode, raw) in cases) {
            for (rep in 0..1) {
                val t = SystemClock.elapsedRealtime()
                val json = r!!.refine(raw, mode)
                log("REFINE rep=$rep total_ms=${SystemClock.elapsedRealtime() - t} ${Mem.snapshot()} $json")
            }
        }
        log("DONE ${Mem.snapshot()}")
    }
}
