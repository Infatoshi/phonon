package ai.phonon.keyboard

import android.os.Build
import android.os.ParcelFileDescriptor
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Test
import java.io.File

/**
 * Before/after bench of the keyboard's refiner thread handling on a real phone (Firebase Test Lab), in one process:
 * each arm loads the model the way the keyboard does, runs the keyboard's warm-up refine, then the RefinerBenchTest
 * transcripts (4 modes x 2 reps; rep 0 after a mode switch re-prefills the head, rep 1 reuses it), then sits idle 2 s
 * to read the process CPU time an idle keyboard would burn, then frees the model.
 *
 * Instrumentation args:
 *   arms        comma list, default "legacy,auto,auto". legacy = the pre-tuning path (no threadpool, 4 threads
 *               unpinned, repack on, no RAM gate); auto = Refiner.loadAuto; "auto/k4/poll0" passes the spec
 *               "k=4,poll=0". The tune cache is deleted at the start, so the first auto arm probes K and later
 *               ones read the cache, as on a phone's first and later keyboard starts.
 *   arm_budget_s  stop an arm's remaining refines once it has run this long (default 150), so a collapsed
 *               configuration cannot eat the Test Lab timeout.
 * Logcat tag Phonon, prefix TB.
 */
class RefinerTuneBenchTest {
    private val cases = listOf(
        "minimal" to "um so the first thing we do is call could a malloc to to allocate memory on the GPU",
        "prompt" to "okay so I want you to add a retry to the fetch function like three retries no actually make it five with exponential backoff and um also log every failure to the console",
        "casual" to "yeah I think we should uh grab dinner around seven if that works for you",
        "formal" to "hi sarah just wanted to let you know the the deploy is gonna slip to thursday because the tests are failing on the arm runners",
    )

    private fun log(s: String) {
        Log.i(TAG, "TB $s")
    }

    private fun meminfo(): String {
        val m = File("/proc/meminfo").readLines().associate { l -> l.substringBefore(":") to l.substringAfter(":").trim() }
        val mb = { k: String -> m[k]?.split(" ")?.firstOrNull()?.toLongOrNull()?.div(1024) }
        return "memtotal_mb=${mb("MemTotal")} memavail_mb=${mb("MemAvailable")}"
    }

    private fun stageModel(src: String): File {
        val inst = InstrumentationRegistry.getInstrumentation()
        val direct = File(src)
        if (direct.canRead()) return direct
        val dst = File(inst.targetContext.filesDir, "refiner.gguf")
        val t = SystemClock.elapsedRealtime()
        val pfd = inst.uiAutomation.executeShellCommand("cat $src")
        ParcelFileDescriptor.AutoCloseInputStream(pfd).use { input -> dst.outputStream().use { input.copyTo(it, 1 shl 20) } }
        log("STAGE bytes=${dst.length()} ms=${SystemClock.elapsedRealtime() - t}")
        return dst
    }

    private fun idleCpu(arm: Int, ms: Long) {
        val c0 = Refiner.nativeCpuMs()
        SystemClock.sleep(ms)
        log("IDLE arm=$arm wall_ms=$ms cpu_ms=${Refiner.nativeCpuMs() - c0}")
    }

    @Test
    fun tuneBench() {
        val args = InstrumentationRegistry.getArguments()
        val arms = (args.getString("arms") ?: "legacy,auto,auto").split(",").map { it.trim() }.filter { it.isNotEmpty() }
        val budgetMs = (args.getString("arm_budget_s")?.toLongOrNull() ?: 150L) * 1000
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        log("DEVICE model=${Build.MODEL} device=${Build.DEVICE} " +
            "soc=${if (Build.VERSION.SDK_INT >= 31) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}" else Build.HARDWARE} " +
            "sdk=${Build.VERSION.SDK_INT} cores=${Runtime.getRuntime().availableProcessors()} ${meminfo()} arms=${arms.joinToString(",")}")
        val model = stageModel(args.getString("model") ?: "/sdcard/phonon/refiner.gguf")
        val tune = File(ctx.filesDir, "refiner_tune.txt")
        tune.delete()
        idleCpu(-1, 2000)
        for ((i, arm) in arms.withIndex()) {
            // "auto/kb6/poll0" -> spec "kb=6,poll=0" (no '=' in the arm names: gcloud's KEY=VALUE parsing)
            val spec = arm.split("/").drop(1).joinToString(",") { it.replace(Regex("^([a-z]+)(\\d+)$"), "$1=$2") }
            val t0 = SystemClock.elapsedRealtime()
            val r = if (arm.startsWith("legacy")) Refiner.load(model.path, 4, true) else Refiner.loadAuto(model.path, tune.path, spec)
            val loadMs = SystemClock.elapsedRealtime() - t0
            log("LOAD arm=$i name=$arm ok=${r != null} load_ms=$loadMs ${Mem.snapshot()} ${meminfo()}")
            if (r == null) continue
            var t = SystemClock.elapsedRealtime()
            // "/heads1" arms: the pre-2026-11-12 warm-up (minimal head only), for the before/after of the per-mode
            // head cache; otherwise the keyboard's warm-up (Refiner.warmUp: every mode's head cached)
            val warm = if (arm.contains("/heads1")) r.refine("warm up", "minimal") else r.warmUp()
            log("WARMUP arm=$i total_ms=${SystemClock.elapsedRealtime() - t} $warm")
            val armStart = SystemClock.elapsedRealtime()
            loop@ for ((mode, raw) in cases) {
                for (rep in 0..1) {
                    if (SystemClock.elapsedRealtime() - armStart > budgetMs) {
                        log("BUDGET arm=$i stopped after ${SystemClock.elapsedRealtime() - armStart} ms")
                        break@loop
                    }
                    t = SystemClock.elapsedRealtime()
                    val json = r.refine(raw, mode)
                    log("REFINE arm=$i name=$arm mode=$mode rep=$rep total_ms=${SystemClock.elapsedRealtime() - t} ${Mem.snapshot()} $json")
                }
            }
            idleCpu(i, 2000)
            r.free()
            log("FREED arm=$i ${Mem.snapshot()}")
        }
        log("DONE tune=${if (tune.exists()) tune.readText().trim() else "-"}")
    }
}
