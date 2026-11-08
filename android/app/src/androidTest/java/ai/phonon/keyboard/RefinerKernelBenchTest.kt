package ai.phonon.keyboard

import android.os.Build
import android.os.ParcelFileDescriptor
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Test
import java.io.File

/** JNI side: app/src/main/cpp/kbench_jni.cpp (libphonon_kbench.so, benchmark only). */
object KernelBench {
    init { System.loadLibrary("phonon_kbench") }
    @JvmStatic external fun nativeInit(variant: String): String
    @JvmStatic external fun nativeMemBw(mb: Int, threads: Int, cpus: IntArray, reps: Int): String
    @JvmStatic external fun nativePeak(kind: String, cpus: IntArray, ms: Int): String
    @JvmStatic external fun nativeLoad(path: String, repack: Boolean): Long
    @JvmStatic external fun nativeFree(h: Long)
    @JvmStatic external fun nativeRun(h: Long, mode: String, threads: Int, cpus: IntArray, poll: Int, ubatch: Int,
                                      ndecode: Int, reps: Int, profile: Boolean): String
}

/**
 * Per-chip kernel investigation (DEVLOG 2026-11-06 "mobile refiner kernels"): memory bandwidth and int8 peak
 * microbenchmarks, then the refiner's prefill and decode under thread configurations, repack on/off, optional CPU
 * variant override, and a per-op profile. Logcat tag "Phonon", prefix "KB". Instrumentation args (all optional):
 *   kb_cfg     comma list of configs, each (disp|tp|top)N[pP][uU]: disp = the keyboard's path (no threadpool),
 *              tp = persistent threadpool unpinned, top = persistent threadpool pinned to the N highest-capacity
 *              cores; pP = poll level (default 50), uU = n_ubatch (default 512). Default disp2,disp4,disp6,disp8,tp4,top2,top4,top6
 *   kb_repack  order of repack settings, default "0,1"
 *   kb_decode  decode steps per rep (24), kb_reps reps (2)
 *   kb_prof    comma list of configs to profile per op with repack on (default top1; "none" to skip). Multi-
 *              thread profiles are not trustworthy: on the emulator a 2-thread context with the eval callback set ran
 *              decode at 2.8 s/token even with the callback answering "no" (1 thread: 45 ms, same as unprofiled).
 *   kb_bw, kb_peak  "0" to skip the microbenchmarks
 *   kb_variants  extra CPU variants to rerun kb_vcfg (default "disp4,top4") with repack on, e.g. android_armv8.6_1
 *   kb_rounds  microbenchmark rounds (default 1; round 2+ run after the refiner configs); kb_model=0 skips the refiner
 *   kb_bwmb, kb_bwreps, kb_peakms  bandwidth buffer MB (256), timed passes (8), ms per peak measurement (150)
 *   kb_perf=1 / kb_perfetto=1  simpleperf / Perfetto probes during a decode with kb_perfcfg (default tp4) after the
 *              first refiner pass. "topbig" in kb_cfg / kb_perfcfg = pinned to every core with >= half the top capacity.
 */
class RefinerKernelBenchTest {
    private fun log(s: String) {
        Log.i(TAG, "KB $s")
    }

    private data class Core(val cpu: Int, val cap: Long, val khz: Long, val part: String)

    private fun cores(): List<Core> {
        val n = Runtime.getRuntime().availableProcessors()
        val parts = File("/proc/cpuinfo").readLines().filter { it.startsWith("CPU part") }.map { it.substringAfter(":").trim() }
        return (0 until n).map { i ->
            val d = "/sys/devices/system/cpu/cpu$i"
            val khz = runCatching { File("$d/cpufreq/cpuinfo_max_freq").readText().trim().toLong() }.getOrDefault(0L)
            val cap = runCatching { File("$d/cpu_capacity").readText().trim().toLong() }.getOrDefault(khz / 1000)
            Core(i, cap, khz, parts.getOrElse(i) { "?" })
        }
    }

    private fun meminfo(): String {
        val m = File("/proc/meminfo").readLines().associate { l -> l.substringBefore(":") to l.substringAfter(":").trim() }
        val mb = { k: String -> m[k]?.split(" ")?.firstOrNull()?.toLongOrNull()?.div(1024) }
        return "memtotal_mb=${mb("MemTotal")} memavail_mb=${mb("MemAvailable")} swap_mb=${mb("SwapTotal")}"
    }

    private fun stageModel(src: String): File {
        val inst = InstrumentationRegistry.getInstrumentation()
        val dst = File(inst.targetContext.filesDir, "refiner.gguf")
        if (dst.length() == REFINER_BYTES) return dst
        val direct = File(src)
        if (direct.canRead()) return direct
        val t = SystemClock.elapsedRealtime()
        val pfd = inst.uiAutomation.executeShellCommand("cat $src")
        ParcelFileDescriptor.AutoCloseInputStream(pfd).use { input -> dst.outputStream().use { input.copyTo(it, 1 shl 20) } }
        log("STAGE bytes=${dst.length()} ms=${SystemClock.elapsedRealtime() - t}")
        return dst
    }

    /** Runs a shell command as the shell user (UiAutomation); returns stdout + stderr (stderr only on API 31+). */
    private fun sh(cmd: String): String {
        val ua = InstrumentationRegistry.getInstrumentation().uiAutomation
        return if (Build.VERSION.SDK_INT >= 31) {
            val p = ua.executeShellCommandRwe(cmd)
            p[1].close()
            val err = StringBuilder()
            val t = Thread { err.append(ParcelFileDescriptor.AutoCloseInputStream(p[2]).bufferedReader().readText()) }
            t.start()
            val out = ParcelFileDescriptor.AutoCloseInputStream(p[0]).bufferedReader().readText()
            t.join()
            out + (if (err.isNotEmpty()) "\n[stderr] $err" else "")
        } else {
            ParcelFileDescriptor.AutoCloseInputStream(ua.executeShellCommand(cmd)).bufferedReader().readText()
        }
    }

    private fun logBlock(tag: String, s: String, max: Int = 40) {
        s.lines().map { it.trimEnd() }.filter { it.isNotBlank() }.take(max).forEach { log("$tag $it") }
    }

    /**
     * What simpleperf can do on a Test Lab phone without root: the debug app is debuggable, so `--app` (run-as) is
     * the documented path. Counts hardware events and samples symbols while the refiner decodes.
     */
    private fun perfProbe(h: Long, c: Cfg, n: Int, top: (Int) -> IntArray) {
        log("PERF env paranoid=${sh("cat /proc/sys/kernel/perf_event_paranoid").trim()} " +
            "perf_harden=${sh("getprop security.perf_harden").trim()} which=${sh("which simpleperf").trim()}")
        logBlock("PERF list", sh("simpleperf list hw"), 20)
        logBlock("PERF list_raw", sh("simpleperf list raw"), 60)
        val cpus = if (c.pinned) top(c.n) else IntArray(0)
        fun during(cmd: String, tag: String) {
            var out = ""
            val t = Thread { out = sh(cmd) }
            t.start()
            Thread.sleep(400)
            val r = KernelBench.nativeRun(h, c.mode, c.n, cpus, c.poll, c.ub, 96, 1, false)
            t.join()
            log("PERF $tag run cfg=${c.name} $r")
            logBlock("PERF $tag", out, 45)
        }
        during("simpleperf stat --app ai.phonon.keyboard --duration 3 " +
            "-e cpu-cycles,instructions,cache-references,cache-misses,stalled-cycles-backend", "stat")
        during("simpleperf stat --app ai.phonon.keyboard --duration 3 " +
            "-e raw-l2d-cache-refill,raw-l3d-cache-refill,raw-bus-access,raw-mem-access,raw-ll-cache-miss-rd", "stat_raw")
        during("simpleperf record --app ai.phonon.keyboard --duration 3 -f 4000 -o /data/local/tmp/kb_perf.data", "record")
        logBlock("PERF report", sh("simpleperf report -i /data/local/tmp/kb_perf.data --sort symbol --percent-limit 0.8"), 45)
        logBlock("PERF report_thread", sh("simpleperf report -i /data/local/tmp/kb_perf.data --sort tid,comm --percent-limit 1"), 20)
        copyOut("PERF", "/data/local/tmp/kb_perf.data")
    }

    /**
     * Perfetto from the shell user (no root): lightweight mode, atrace/ftrace categories sched + freq + idle, while the
     * refiner decodes. The trace is copied to /sdcard/phonon_kb/ so Test Lab can pull it (--directories-to-pull).
     */
    private fun perfettoProbe(h: Long, c: Cfg, top: (Int) -> IntArray) {
        val out = "/data/misc/perfetto-traces/kb_trace.pftrace"
        sh("rm -f $out")
        log("PERFETTO env pid=${android.os.Process.myPid()} traced=${sh("getprop persist.traced.enable").trim()} " +
            "which=${sh("which perfetto").trim()}")
        val cpus = if (c.pinned) top(c.n) else IntArray(0)
        var res = ""
        val t = Thread { res = sh("perfetto -o $out -t 6s sched freq idle") }
        t.start()
        Thread.sleep(1000)
        val r = KernelBench.nativeRun(h, c.mode, c.n, cpus, c.poll, c.ub, 48, 1, false)
        t.join()
        log("PERFETTO run cfg=${c.name} cpus=${cpus.joinToString(",")} $r")
        logBlock("PERFETTO out", res, 20)
        logBlock("PERFETTO ls", sh("ls -la $out"), 5)
        copyOut("PERFETTO", out)
    }

    /** UiAutomation shell commands are not run through a shell (no &&), so one call per command. */
    private fun copyOut(tag: String, path: String) {
        sh("mkdir -p /sdcard/phonon_kb")
        logBlock("$tag copy", sh("cp $path /sdcard/phonon_kb/") + sh("ls -la /sdcard/phonon_kb"), 10)
    }

    private data class Cfg(val name: String, val mode: String, val n: Int, val pinned: Boolean, val poll: Int, val ub: Int)

    private fun parseCfg(s: String): Cfg? {
        val m = Regex("^(disp|tp|top)(\\d+)(?:p(\\d+))?(?:u(\\d+))?$").find(s.trim()) ?: return null
        val (kind, n, p, u) = m.destructured
        return Cfg(s.trim(), if (kind == "disp") "disp" else "tp", n.toInt(), kind == "top", p.toIntOrNull() ?: 50,
            u.toIntOrNull() ?: 512)
    }

    @Test
    fun kernelBench() {
        val args = InstrumentationRegistry.getArguments()
        val arg = { k: String, d: String -> args.getString(k) ?: d }
        val cs = cores()
        val byCap = cs.sortedWith(compareByDescending<Core> { it.cap }.thenByDescending { it.khz })
        log("DEVICE model=${Build.MODEL} device=${Build.DEVICE} " +
            "soc=${if (Build.VERSION.SDK_INT >= 31) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}" else Build.HARDWARE} " +
            "sdk=${Build.VERSION.SDK_INT} ${meminfo()}")
        log("CORES " + cs.joinToString(" ") { "${it.cpu}:${it.part}:${it.cap}:${it.khz / 1000}" })
        log("INIT ${KernelBench.nativeInit("")}")
        val n = cs.size
        val top = { k: Int -> byCap.take(k).map { it.cpu }.toIntArray() }
        val clusters = cs.groupBy { it.cap }.toSortedMap(compareByDescending { it })
        // "big" = every core with at least half the top capacity (all clusters except the efficiency cores)
        val maxCap = cs.maxOf { it.cap }
        val big = byCap.filter { it.cap * 2 >= maxCap }.map { it.cpu }.toIntArray()
        log("CLUSTERS big=${big.joinToString(",")} " + clusters.entries.joinToString(" ") { (cap, cl) ->
            "cap$cap=${cl.joinToString(",") { it.cpu.toString() }}:${cl.first().part}:${cl.maxOf { it.khz } / 1000}" })

        // kb_rounds=2 repeats the microbenchmarks at the end of the run (round 2), after the refiner configs.
        fun microbench(round: Int) {
            if (arg("kb_bw", "1") != "0") {
                val mb = arg("kb_bwmb", "256").toInt()
                val bwReps = arg("kb_bwreps", "8").toInt()
                val bw = mutableListOf<Pair<String, Pair<Int, IntArray>>>()
                for ((cap, cl) in clusters) bw += "1t_cap$cap" to (1 to intArrayOf(cl.first().cpu))
                for ((cap, cl) in clusters) if (cl.size > 1) bw += "cl_cap$cap" to (cl.size to cl.map { it.cpu }.toIntArray())
                for (k in listOf(2, 4, 6)) if (k < n) bw += "top$k" to (k to top(k))
                if (big.size < n) bw += "big${big.size}" to (big.size to big)
                bw += "all$n" to (n to top(n))
                bw += "unpinned4" to (4 to IntArray(0))
                for ((name, cfg) in bw) {
                    log("BW $name threads=${cfg.first} cpus=${cfg.second.joinToString(",")} mb=$mb gbps_best_med=${
                        KernelBench.nativeMemBw(mb, cfg.first, cfg.second, bwReps)} round=$round")
                }
            }
            if (arg("kb_peak", "1") != "0") {
                val ms = arg("kb_peakms", "150").toInt()
                for (kind in listOf("sdot", "smmla", "fma", "fp16")) {
                    for ((cap, cl) in clusters) {
                        log("PEAK $kind 1t_cap$cap cpu=${cl.first().cpu} gops=${KernelBench.nativePeak(kind, intArrayOf(cl.first().cpu), ms)} round=$round")
                        if (cl.size > 1) log("PEAK $kind cl_cap$cap cpus=${cl.joinToString(",") { it.cpu.toString() }} gops=${
                            KernelBench.nativePeak(kind, cl.map { it.cpu }.toIntArray(), ms)} round=$round")
                    }
                    if (big.size < n) log("PEAK $kind big${big.size} gops=${KernelBench.nativePeak(kind, big, ms)} round=$round")
                    log("PEAK $kind all$n gops=${KernelBench.nativePeak(kind, top(n), ms)} round=$round")
                }
            }
        }
        microbench(1)
        val rounds = arg("kb_rounds", "1").toInt()
        if (arg("kb_model", "1") == "0") {
            for (r in 2..rounds) microbench(r)
            log("DONE")
            return
        }

        val model = stageModel(arg("model", "/sdcard/phonon/refiner.gguf"))
        val cfgs = arg("kb_cfg", "disp2,disp4,disp6,disp8,tp4,top2,top4,top6").replace("topbig", "top${big.size}")
            .split(",").distinct().mapNotNull { parseCfg(it) }.filter { it.n <= n }
        var probesDone = false
        val nDecode = arg("kb_decode", "24").toInt()
        val reps = arg("kb_reps", "2").toInt()
        val prof = arg("kb_prof", "top1")

        fun runAll(tag: String, repack: Boolean, list: List<Cfg>, profile: Boolean) {
            val t0 = SystemClock.elapsedRealtime()
            val h = KernelBench.nativeLoad(model.path, repack)
            log("LOAD $tag repack=$repack ok=${h != 0L} ms=${SystemClock.elapsedRealtime() - t0} ${Mem.snapshot()} ${meminfo()}")
            if (h == 0L) return
            for (c in list) {
                val cpus = if (c.pinned) top(c.n) else IntArray(0)
                val r = KernelBench.nativeRun(h, c.mode, c.n, cpus, c.poll, c.ub, nDecode, reps, false)
                log("RUN $tag repack=${if (repack) 1 else 0} cfg=${c.name} cpus=${cpus.joinToString(",")} $r ${Mem.snapshot()}")
            }
            if (profile) {
                for (c in prof.split(",").mapNotNull { parseCfg(it) }.filter { it.n <= n }) {
                    val cpus = if (c.pinned) top(c.n) else IntArray(0)
                    val r = KernelBench.nativeRun(h, c.mode, c.n, cpus, c.poll, c.ub, 8, 1, true)
                    val parts = r.split(" |")
                    log("PROF $tag repack=${if (repack) 1 else 0} cfg=${c.name} ${parts[0]}")
                    for (p in parts.drop(1)) {
                        val items = p.split(" ;")
                        log("PROF ${items[0].trim()}")
                        for (it in items.drop(1)) log("PROF   ${items[0].trim().substringBefore(" ")} $it")
                    }
                }
            }
            if (!probesDone && tag == "best") {
                probesDone = true
                val pc = parseCfg(arg("kb_perfcfg", "tp4").replace("topbig", "top${big.size}"))!!
                if (arg("kb_perf", "0") == "1") perfProbe(h, pc, n, top)
                if (arg("kb_perfetto", "0") == "1") perfettoProbe(h, pc, top)
            }
            KernelBench.nativeFree(h)
            log("FREED ${Mem.snapshot()}")
        }

        for (rp in arg("kb_repack", "0,1").split(",")) runAll("best", rp.trim() == "1", cfgs, rp.trim() == "1")
        val vcfg = arg("kb_vcfg", "disp4,top4").split(",").mapNotNull { parseCfg(it) }.filter { it.n <= n }
        for (v in arg("kb_variants", "").split(",").map { it.trim() }.filter { it.isNotEmpty() }) {
            log("INIT ${KernelBench.nativeInit(v)}")
            runAll(v, true, vcfg, false)
        }
        for (r in 2..rounds) microbench(r)
        log("DONE")
    }
}
