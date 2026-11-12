package ai.phonon.keyboard

import android.accessibilityservice.AccessibilityServiceInfo
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.ParcelFileDescriptor
import android.os.SystemClock
import android.text.InputType
import android.util.Log
import android.view.InputDevice
import android.view.MotionEvent
import android.view.accessibility.AccessibilityNodeInfo
import android.view.accessibility.AccessibilityWindowInfo
import android.view.inputmethod.InputMethodManager
import android.widget.EditText
import android.widget.LinearLayout
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * The keyboard end to end on a real phone (Firebase Test Lab): the shipping PhononIME is enabled and selected
 * through the shell, the models arrive with --other-files under /sdcard/phonon/, and recorded WAVs stand in for a
 * hold of the talk bar (the IME's WAV broadcast; a Test Lab phone hears nothing). Covers:
 *  - load and warm-up of both models (the IME's own ASR_LOAD / ASR_WARMUP / REFINER_LOAD / WARMUP lines);
 *  - the RAM gate (REFINER_LOAD ok=false below ~4 GB: the raw transcript is committed);
 *  - dictation into a multi-line and a single-line field, read back from the field;
 *  - the line-break stage: numbered steps keep their newlines in the multi-line field and none in the single-line;
 *  - the opener stage: a prompt-mode "okay ..." commits starting with the opener (refiner loaded only);
 *  - dictation into another app's field (Messages composer, then the contact editor), read back by accessibility.
 * Every line goes to logcat under tag "Phonon" with the prefix E2ET.
 */
class ImeE2ETest {
    private val inst = InstrumentationRegistry.getInstrumentation()
    private val ctx: Context = inst.targetContext
    private val pkg = ctx.packageName
    private val ime = "$pkg/.PhononIME"
    private val dir: File = ctx.getExternalFilesDir(null)!!

    private fun log(s: String) {
        Log.i(TAG, "E2ET $s")
        println("E2ET $s")
    }

    private fun sh(cmd: String): String {
        val pfd = inst.uiAutomation.executeShellCommand(cmd)
        return ParcelFileDescriptor.AutoCloseInputStream(pfd).use { it.readBytes().toString(Charsets.UTF_8) }
    }

    private fun stage(src: String, dst: File) {
        dst.parentFile!!.mkdirs()
        val t = SystemClock.elapsedRealtime()
        val pfd = inst.uiAutomation.executeShellCommand("cat $src")
        ParcelFileDescriptor.AutoCloseInputStream(pfd).use { input -> dst.outputStream().use { input.copyTo(it, 1 shl 20) } }
        log("STAGE $src bytes=${dst.length()} ms=${SystemClock.elapsedRealtime() - t}")
    }

    private fun meminfo(): String = File("/proc/meminfo").readLines()
        .filter { it.startsWith("MemTotal") || it.startsWith("MemAvailable") }
        .joinToString(" ") { it.replace(Regex("\\s+"), " ") }

    /** With the `smaps=1` argument: where this process's memory sits (the test runs in the IME's process).
     *  One line per model-file mapping (encoder .onnx / .onnx.data, decoder, joiner): size, Rss, Private_Dirty;
     *  totals by kind (model files, other files, anon by name); then the largest anonymous mappings by Rss. */
    private fun smaps(label: String) {
        if (InstrumentationRegistry.getArguments().getString("smaps") != "1") return
        data class Map(val name: String, var size: Long = 0, var rss: Long = 0, var dirty: Long = 0)
        val maps = mutableListOf<Map>()
        val header = Regex("^[0-9a-f]+-[0-9a-f]+ ")
        for (line in File("/proc/self/smaps").readLines()) {
            if (header.containsMatchIn(line)) {
                val parts = line.trim().split(Regex("\\s+"), limit = 6)
                maps += Map(if (parts.size > 5) parts[5] else "[anon]")
                continue
            }
            val m = maps.lastOrNull() ?: continue
            val kb = line.substringAfter(':').trim().substringBefore(' ').toLongOrNull() ?: continue
            when (line.substringBefore(':')) {
                "Size" -> m.size += kb
                "Rss" -> m.rss += kb
                "Private_Dirty" -> m.dirty += kb
            }
        }
        val model = Regex("parakeet/(encoder|decoder|joiner)")
        maps.filter { model.containsMatchIn(it.name) }.groupBy { it.name }.forEach { (n, ms) ->
            log("SMAPS $label file=${n.substringAfterLast('/')} maps=${ms.size} size_mb=${ms.sumOf { it.size } / 1024} " +
                "rss_mb=${ms.sumOf { it.rss } / 1024} private_dirty_mb=${ms.sumOf { it.dirty } / 1024}")
        }
        fun kind(n: String) = when {
            model.containsMatchIn(n) -> "model_files"
            n.startsWith("/") -> "other_files"
            n.startsWith("[anon:") -> n.substringBefore(']') + "]"
            else -> n
        }
        maps.groupBy { kind(it.name) }.map { (k, ms) -> Triple(k, ms.sumOf { it.rss } / 1024, ms.sumOf { it.dirty } / 1024) }
            .sortedByDescending { it.second }.take(12)
            .forEach { (k, rss, dirty) -> log("SMAPS $label kind=$k rss_mb=$rss private_dirty_mb=$dirty") }
        maps.filter { !it.name.startsWith("/") }.sortedByDescending { it.rss }.take(10).forEach {
            log("SMAPS $label anon_top name=${it.name} size_mb=${it.size / 1024} rss_mb=${it.rss / 1024} private_dirty_mb=${it.dirty / 1024}")
        }
    }

    private fun phononLog(): List<String> = sh("logcat -d -v brief -s Phonon:I").lines()

    /** Wait until the IME's log has a line matching [pred], up to [seconds]. */
    private fun waitLog(seconds: Int, pred: (List<String>) -> Boolean): List<String> {
        val end = SystemClock.elapsedRealtime() + seconds * 1000L
        var lines = phononLog()
        while (!pred(lines) && SystemClock.elapsedRealtime() < end) {
            SystemClock.sleep(500)
            lines = phononLog()
        }
        return lines
    }

    private fun e2eCount() = phononLog().count { it.contains(" E2E source=") }

    /** One injected dictation; returns the IME's E2E line, or null on timeout. */
    private fun dictateWav(name: String, timeoutS: Int = 120): String? {
        val before = e2eCount()
        ctx.sendBroadcast(Intent(PhononIME.ACTION_WAV).setPackage(pkg).putExtra("path", File(dir, name).path))
        val lines = waitLog(timeoutS) { l -> l.count { it.contains(" E2E source=") } > before }
        val e2e = lines.filter { it.contains(" E2E source=") }
        return if (e2e.size > before) e2e.last() else null
    }

    private fun refineCommit(text: String, mode: String, timeoutS: Int = 120): String? {
        val before = e2eCount()
        ctx.sendBroadcast(Intent("ai.phonon.keyboard.REFINE").setPackage(pkg)
            .putExtra("text", text).putExtra("mode", mode).putExtra("commit", true))
        val lines = waitLog(timeoutS) { l -> l.count { it.contains(" E2E source=") } > before }
        val e2e = lines.filter { it.contains(" E2E source=") }
        return if (e2e.size > before) e2e.last() else null
    }

    private fun focus(field: EditText) {
        inst.runOnMainSync {
            field.setText("")
            field.requestFocus()
            (ctx.getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager).showSoftInput(field, 0)
        }
        SystemClock.sleep(1500)
    }

    private fun fieldText(field: EditText): String {
        var s = ""
        inst.runOnMainSync { s = field.text.toString() }
        return s
    }

    private fun esc(s: String) = s.replace("\n", "\\n")

    private fun findEditable(node: AccessibilityNodeInfo?, focused: Boolean): AccessibilityNodeInfo? {
        if (node == null) return null
        if (node.isEditable && (!focused || node.isFocused)) return node
        for (i in 0 until node.childCount) findEditable(node.getChild(i), focused)?.let { return it }
        return null
    }

    private fun findFocusedEditable(node: AccessibilityNodeInfo?) = findEditable(node, true)

    /** The top application window's root (all interactive windows, not only the "active" one, which can be a
     *  transparent trampoline: Messages on the Samsung phones gave an empty tree). */
    private fun appRoot(): AccessibilityNodeInfo? {
        val ua = inst.uiAutomation
        val info = ua.serviceInfo
        if (info.flags and AccessibilityServiceInfo.FLAG_RETRIEVE_INTERACTIVE_WINDOWS == 0) {
            info.flags = info.flags or AccessibilityServiceInfo.FLAG_RETRIEVE_INTERACTIVE_WINDOWS
            ua.serviceInfo = info
        }
        val apps = ua.windows.filter { it.type == AccessibilityWindowInfo.TYPE_APPLICATION }
        return (apps.firstOrNull { it.isFocused } ?: apps.firstOrNull())?.root ?: ua.rootInActiveWindow
    }

    private val dismiss = listOf("Phone", "Device", "Device & Google", "Don't Show Again", "OK", "Got it", "Skip",
        "Not now", "No thanks", "Continue", "Start chat", "Close sheet", "Take me to Gmail")

    /** The clickable node (or its clickable ancestor) whose text or description is in [dismiss], by list order. */
    private fun clickTarget(root: AccessibilityNodeInfo?): AccessibilityNodeInfo? {
        if (root == null) return null
        val all = mutableListOf<AccessibilityNodeInfo>()
        fun walk(n: AccessibilityNodeInfo?, d: Int) {
            if (n == null || d > 25) return
            all.add(n)
            for (i in 0 until n.childCount) walk(n.getChild(i), d + 1)
        }
        walk(root, 0)
        for (want in dismiss) {
            val hit = all.firstOrNull { (it.text ?: it.contentDescription)?.toString()?.trim().equals(want, ignoreCase = true) }
                ?: continue
            var c: AccessibilityNodeInfo? = hit
            while (c != null && !c.isClickable) c = c.parent
            return c ?: hit
        }
        return null
    }

    /** Short outline of the window for a log line: class and text of the first nodes. */
    private fun outline(node: AccessibilityNodeInfo?, out: MutableList<String> = mutableListOf(), depth: Int = 0): List<String> {
        if (node == null || out.size >= 25 || depth > 12) return out
        val t = (node.text ?: node.contentDescription)?.toString()?.take(30)
        if (t != null || node.isEditable) out.add("${node.className?.toString()?.substringAfterLast('.')}${if (node.isEditable) "*" else ""}:$t")
        for (i in 0 until node.childCount) outline(node.getChild(i), out, depth + 1)
        return out
    }

    /** Another app's field: launch it with `am start [amArgs]` (the shell's start: MIUI aborts this app's own
     *  starts as background starts), let it settle, dictate, read the focused field back. */
    private fun otherApp(label: String, amArgs: String) {
        val res = sh("am start -W $amArgs")
        log("APP $label start=${res.lines().firstOrNull { it.startsWith("Status") || it.startsWith("Error") }?.trim()}")
        SystemClock.sleep(8000)
        val home = sh("cmd package resolve-activity --brief -c android.intent.category.HOME -a android.intent.action.MAIN")
            .lines().lastOrNull { it.contains('/') }?.substringBefore('/')?.trim()
        var root = appRoot()
        var relaunched = false
        for (attempt in 1..6) {
            if (findFocusedEditable(root) != null) break
            if (root?.packageName?.toString() == home) {  // never press anything on the launcher
                log("APP $label not_started foreground=launcher($home)"); break
            }
            // a welcome tour that was skipped leaves the app on its main screen: ask for the editor again, once
            if (attempt > 2 && !relaunched && clickTarget(root) == null && findEditable(root, false) == null) {
                relaunched = true
                log("APP $label relaunch")
                sh("am start -W $amArgs"); SystemClock.sleep(5000); root = appRoot(); continue
            }
            val field = findEditable(root, false)
            if (field != null) {
                // tap the first text field, as a user would
                field.performAction(AccessibilityNodeInfo.ACTION_FOCUS)
                field.performAction(AccessibilityNodeInfo.ACTION_CLICK)
            } else {
                // a first-run sheet or picker in front of the editor: take the local/default choice
                val button = clickTarget(root)
                log("APP $label attempt=$attempt pressing=\"${button?.let { it.text ?: it.contentDescription }}\" " +
                    "outline=${outline(root).joinToString(" | ")}")
                button?.performAction(AccessibilityNodeInfo.ACTION_CLICK)
            }
            SystemClock.sleep(2500)
            root = appRoot()
        }
        val before = findFocusedEditable(root)
        val shown = sh("dumpsys input_method").lines().firstOrNull { it.contains("mInputShown") }?.trim()
        log("APP $label foreground=${root?.packageName} focused_editable=${before != null} imm=$shown " +
            "root_children=${root?.childCount} focus=${sh("dumpsys window").lines().firstOrNull { it.contains("mCurrentFocus") }?.trim()}")
        if (before == null) log("APP $label outline=${outline(root).joinToString(" | ")}")
        shot("$label-settled")
        // an app whose fields the accessibility tree does not show (Google Messages gave an empty tree on both Test
        // Lab phones): tap where a compose box sits, above the navigation bar, as a user would, until the keyboard
        // starts on a text field of that app
        val app = root?.packageName?.toString()
        val tapped = if (before == null && app != null) tapBottomField(label, app) else false
        val e2e = dictateWav("s1.wav")
        SystemClock.sleep(1000)
        shot("$label-dictated")
        val after = findFocusedEditable(appRoot())
        // the field as the keyboard reads it back through its own input connection (works where the tree is empty)
        val back = phononLog().lastOrNull { it.contains("E2E_READBACK") }
        log("APP $label e2e=${e2e?.substringAfter("E2E ")} tapped_bottom=$tapped field=\"${esc(after?.text?.toString() ?: "")}\" " +
            "readback=${back?.substringAfter("E2E_READBACK ")?.take(200)}")
        val ok = e2e?.contains("committed=true") == true && back?.contains("app=$app ") == true && back.contains("landed=true")
        log("APP_RESULT $label app=$app ok=$ok")
        sh("input keyevent KEYCODE_BACK"); SystemClock.sleep(500)
        sh("input keyevent KEYCODE_BACK"); SystemClock.sleep(500)
        sh("input keyevent KEYCODE_HOME"); SystemClock.sleep(1000)
    }

    /** Screenshot to the app's files/shots (pulled by Test Lab with --directories-to-pull). */
    private fun shot(name: String) {
        try {
            val b = inst.uiAutomation.takeScreenshot() ?: return
            val f = File(dir, "shots/$name.png")
            f.parentFile!!.mkdirs()
            f.outputStream().use { b.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, it) }
            log("SHOT ${f.path}")
        } catch (e: Throwable) {
            log("SHOT $name failed=${e.javaClass.simpleName}")
        }
    }

    /** Tap up the screen from just above the navigation bar until the IME starts on a text field of [app]. */
    private fun tapBottomField(label: String, app: String): Boolean {
        val m = ctx.resources.displayMetrics
        val wm = ctx.getSystemService(Context.WINDOW_SERVICE) as android.view.WindowManager
        val bounds = wm.maximumWindowMetrics.bounds
        val navId = ctx.resources.getIdentifier("navigation_bar_height", "dimen", "android")
        val nav = if (navId > 0) ctx.resources.getDimensionPixelSize(navId) else (48 * m.density).toInt()
        for (dpUp in listOf(32, 48, 64, 80, 100, 124)) {
            val y = bounds.height() - nav - (dpUp * m.density).toInt()
            val x = bounds.width() * 0.45f
            val n = phononLog().count { it.contains("INPUT_VIEW app=$app") && !it.contains("inputType=0x0") }
            sh("input tap ${x.toInt()} $y")
            val ok = waitLog(3) { l -> l.count { it.contains("INPUT_VIEW app=$app") && !it.contains("inputType=0x0") } > n }
                .count { it.contains("INPUT_VIEW app=$app") && !it.contains("inputType=0x0") } > n
            log("APP $label tap_bottom x=${x.toInt()} y=$y dp_up=$dpUp input_view=$ok")
            if (ok) { SystemClock.sleep(800); return true }
        }
        return false
    }

    private fun status(key: String): String =
        File("/proc/self/status").readLines().firstOrNull { it.startsWith("$key:") }?.substringAfter(':')?.trim() ?: "?"

    /**
     * Instrumentation arg `sustain=N` (GOAL A10, A6, A5): N dictations of long.wav (10.9 s of speech) into the
     * multi-line field, one every `sustain_period_s` seconds (default 20: 30 in 10 minutes), each logged with the
     * IME's key-up-to-commit time, its refiner timings, the thermal status and battery level. The test runs in the
     * keyboard's own process, so the CPU mask logged is the keyboard's. Then the test ends (the feature checks are
     * the default run's job).
     */
    private fun sustained(field: EditText, n: Int) {
        val period = (InstrumentationRegistry.getArguments().getString("sustain_period_s")?.toLongOrNull() ?: 20L) * 1000
        val pm = ctx.getSystemService(Context.POWER_SERVICE) as android.os.PowerManager
        val bm = ctx.getSystemService(Context.BATTERY_SERVICE) as android.os.BatteryManager
        fun battery() = "battery_pct=${bm.getIntProperty(android.os.BatteryManager.BATTERY_PROPERTY_CAPACITY)} " +
            "charge_uah=${bm.getIntProperty(android.os.BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)} " +
            "plugged=${sh("dumpsys battery").lines().firstOrNull { it.contains("AC powered") || it.contains("USB powered") }?.trim()}"
        log("SUSTAIN start n=$n period_ms=$period cpus_allowed=${status("Cpus_allowed_list")} ${battery()} " +
            "thermal=${pm.currentThermalStatus}")
        // the other modes' heads are primed after the warm-up; let that finish so dictation 1 is like the rest
        waitLog(60) { l -> l.any { it.contains("PRIME_MODE mode=prompt") } }
        val start = SystemClock.elapsedRealtime()
        for (i in 0 until n) {
            focus(field)
            val e2e = dictateWav("long.wav")
            val pipe = phononLog().lastOrNull { it.contains("PIPELINE source=wav") }
            log("SUSTAIN i=$i t_s=${(SystemClock.elapsedRealtime() - start) / 1000} " +
                "e2e=${e2e?.substringAfter("E2E ")?.substringBefore(" text=")} thermal=${pm.currentThermalStatus} " +
                "${battery()} pipe=${pipe?.substringAfter("PIPELINE ")?.replace(Regex("raw=\".*?\" "), "")?.take(500)}")
            val next = start + (i + 1) * period
            val wait = next - SystemClock.elapsedRealtime()
            if (wait > 0) SystemClock.sleep(wait)
        }
        log("SUSTAIN done cpus_allowed=${status("Cpus_allowed_list")} ${battery()} ${Mem.snapshot()} ${meminfo()}")
    }

    @Test
    fun keyboardEndToEnd() {
        log("DEVICE manufacturer=${Build.MANUFACTURER} model=${Build.MODEL} device=${Build.DEVICE} " +
            "soc=${if (Build.VERSION.SDK_INT >= 31) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}" else Build.HARDWARE} " +
            "sdk=${Build.VERSION.SDK_INT} cores=${Runtime.getRuntime().availableProcessors()}")
        log("MEM start ${meminfo()}")

        // models and fixtures into the app's external files dir, where the IME looks
        for (f in listOf("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"))
            stage("/sdcard/phonon/parakeet/$f", File(dir, "parakeet/$f"))
        // an encoder with external weights (mapped by onnxruntime instead of copied to the heap) brings its .data file
        File(dir, "parakeet/encoder.int8.onnx.data").delete()
        if (sh("ls /sdcard/phonon/parakeet/encoder.int8.onnx.data").contains("encoder.int8.onnx.data"))
            stage("/sdcard/phonon/parakeet/encoder.int8.onnx.data", File(dir, "parakeet/encoder.int8.onnx.data"))
        // instrumentation arg norefiner=1: no refiner file, so the IME runs as it does below the RAM gate
        if (InstrumentationRegistry.getArguments().getString("norefiner") == "1") {
            File(dir, "refiner.gguf").delete(); log("NOREFINER refiner.gguf removed")
        } else stage("/sdcard/phonon/refiner.gguf", File(dir, "refiner.gguf"))
        for (w in listOf("s1.wav", "steps.wav", "long.wav")) stage("/sdcard/phonon/$w", File(dir, w))

        sh("pm grant $pkg android.permission.RECORD_AUDIO")
        log("IME enable: ${sh("ime enable $ime").trim()} | set: ${sh("ime set $ime").trim()}")
        log("IME default=${sh("settings get secure default_input_method").trim()}")
        sh("logcat -c")

        // started by the shell, not by this process: MIUI refuses an app's own start while it has no window ("Abort
        // background activity starts", Xiaomi 14 in Test Lab 2026-11-11), and startActivitySync then timed out
        val monitor = inst.addMonitor(TestActivity::class.java.name, null, false)
        log("LAUNCH ${sh("am start -W -n $pkg/.TestActivity").lines().firstOrNull { it.startsWith("Status") }?.trim()}")
        val activity = monitor.waitForActivityWithTimeout(60_000) as Activity?
            ?: throw AssertionError("TestActivity did not start")
        inst.removeMonitor(monitor)
        inst.waitForIdleSync()
        lateinit var multi: EditText
        lateinit var single: EditText
        inst.runOnMainSync {
            multi = activity.findViewById(R.id.field)
            single = EditText(activity).apply { inputType = InputType.TYPE_CLASS_TEXT; hint = "single line" }
            (multi.parent as LinearLayout).addView(single)
        }
        log("FIELDS multi_inputType=0x${Integer.toHexString(multi.inputType)} single_inputType=0x${Integer.toHexString(single.inputType)}")

        // the IME's service is created when the field asks for it; both models then load and warm up
        val t0 = SystemClock.elapsedRealtime()
        focus(multi)
        val ready = waitLog(600) { l ->
            l.any { it.contains("ASR_WARMUP") || it.contains("ASR_LOAD ok=false") } &&
                l.any { it.contains("REFINER_LOAD ok=false") || it.contains(" WARMUP {") || it.contains("no /") }
        }
        log("READY wall_ms=${SystemClock.elapsedRealtime() - t0} (from showSoftInput)")
        ready.filter { Regex("IME_CREATE|ASR_LOAD|ASR_WARMUP|REFINER_LOAD|WARMUP|TUNE|GATE|gate").containsMatchIn(it) }
            .forEach { log("IMELOG ${it.substringAfter("Phonon").take(400)}") }
        val refiner = ready.any { it.contains("REFINER_LOAD ok=true") }
        log("MEM ready ${meminfo()} refiner_loaded=$refiner")
        smaps("ready")
        val sustain = InstrumentationRegistry.getArguments().getString("sustain")?.toIntOrNull() ?: 0
        if (sustain > 0) { sustained(multi, sustain); return }

        // dictation into the multi-line field
        var firstOk = false
        for (w in listOf("s1.wav", "long.wav", "steps.wav", "s1.wav")) {
            focus(multi)
            val e2e = dictateWav(w)
            val text = fieldText(multi)
            if (w == "s1.wav" && text.contains("quarterly report", ignoreCase = true)) firstOk = true
            log("DICTATE field=multi wav=$w e2e=${e2e?.substringAfter("E2E ")?.substringBefore(" text=")} " +
                "newlines=${text.count { it == '\n' }} field=\"${esc(text)}\"")
        }
        phononLog().filter { it.contains("PIPELINE") }.forEach { log("PIPE ${it.substringAfter("PIPELINE ").take(600)}") }

        // the single-line field gets one line
        focus(single)
        val e2eS = dictateWav("steps.wav")
        val textS = fieldText(single)
        log("DICTATE field=single wav=steps.wav e2e=${e2eS?.substringAfter("E2E ")?.substringBefore(" text=")} " +
            "newlines=${textS.count { it == '\n' }} field=\"${esc(textS)}\"")

        // line breaks with a transcript that already carries numbered markers (refiner keeps them in minimal mode)
        val stepsRaw = "please follow these steps 1. check the data schema 2. write the training script 3. run the hyperparameter search"
        for ((name, field) in listOf("multi" to multi, "single" to single)) {
            focus(field)
            val e2e = if (refiner) refineCommit(stepsRaw, "minimal") else null
            val text = fieldText(field)
            log("REFINE field=$name refiner=$refiner e2e=${e2e?.substringAfter("E2E ")?.substringBefore(" text=")} " +
                "newlines=${text.count { it == '\n' }} field=\"${esc(text)}\"")
        }
        // opener restore: a prompt-mode dictation that opens with "okay" commits with the opener in front, whether
        // the refiner kept it or the opener stage put it back (the guard's raw fallback starts with it too)
        var openerOk = true
        if (refiner) {
            focus(multi)
            val e2e = refineCommit("okay so the deploy is going to slip to thursday because the tests are failing", "prompt")
            val text = fieldText(multi)
            openerOk = Regex("^(okay|ok)\\b", RegexOption.IGNORE_CASE).containsMatchIn(text)
            log("OPENER ok=$openerOk e2e=${e2e?.substringAfter("E2E ")?.substringBefore(" text=")} field=\"${esc(text)}\"")
            phononLog().lastOrNull { it.contains("REFINE total_ms=") }?.let { log("OPENER refiner_text=${it.substringAfter("{\"text\":").take(200)}") }
        }
        val bar = talkBar(multi)
        log("MEM after_dictation ${meminfo()} ${Mem.snapshot()}")
        smaps("after_dictation")

        // other apps
        // Google Messages: Test Lab phones have no SIM, so the conversation shows "Please insert a SIM card to
        // continue" where the compose box would be (screenshots, 2026-11-12) and there is no field: committed=false
        // is the right answer there. Gmail is no way round it: past its tour it asks for an email account to be added.
        otherApp("messages", "-a android.intent.action.SENDTO -d smsto:5550100")
        otherApp("contacts", "-a android.intent.action.INSERT -t vnd.android.cursor.dir/contact")
        // no field at all (home screen): the IME must not report the dictation as committed
        sh("input keyevent KEYCODE_HOME"); SystemClock.sleep(2000)
        log("NOFIELD e2e=${dictateWav("s1.wav")?.substringAfter("E2E ")?.substringBefore(" text=")}")
        log("DONE ${meminfo()} ${Mem.snapshot()}")
        assertTrue("the first dictation did not reach the field", firstOk)
        assertTrue("the opener did not survive to the field", openerOk)
        assertTrue("talk bar checks failed: $bar", bar.values.all { it })
    }

    // ---------- the talk bar and keys, pressed with real touches ----------

    /** Centre of the IME key [name] from its latest KEYPOS line (screen pixels), waiting up to [seconds]. */
    private fun keyPos(name: String, seconds: Int = 10): Pair<Float, Float>? {
        val end = SystemClock.elapsedRealtime() + seconds * 1000L
        while (true) {
            val line = phononLog().lastOrNull { it.contains(" KEYPOS ") }
            val tok = line?.split(' ')?.firstOrNull { it.startsWith("$name=") }?.substring(name.length + 1)
            val xy = tok?.split(',')?.mapNotNull { it.trim().toFloatOrNull() }
            if (xy != null && xy.size == 2) return xy[0] to xy[1]
            if (SystemClock.elapsedRealtime() > end) return null
            SystemClock.sleep(300)
        }
    }

    private fun touch(p: Pair<Float, Float>, action: Int, downAt: Long) {
        val ev = MotionEvent.obtain(downAt, SystemClock.uptimeMillis(), action, p.first, p.second, 0)
        ev.source = InputDevice.SOURCE_TOUCHSCREEN
        inst.uiAutomation.injectInputEvent(ev, true)
        ev.recycle()
    }

    private fun downs(name: String) = phononLog().count { it.contains("TOUCH_DOWN key=$name") }

    /** Press [name] for [holdMs] (a tap by default); false if the key is not on the keyboard or no press arrived.
     *  A press the IME did not log (TOUCH_DOWN) is lifted and sent again: the Xiaomi 14 dropped the first injected
     *  touch after the keyboard was asked to show (Test Lab 2026-11-12). */
    private fun press(name: String, holdMs: Long = 60): Boolean {
        for (attempt in 1..3) {
            val p = keyPos(name) ?: run { log("BAR no key $name"); return false }
            val before = downs(name)
            val down = SystemClock.uptimeMillis()
            touch(p, MotionEvent.ACTION_DOWN, down)
            // a tap is lifted on time and checked after (reading logcat can take longer than a 0.35 s tap);
            // a hold is checked while held
            val tap = holdMs < 1000
            if (tap) { SystemClock.sleep(holdMs); touch(p, MotionEvent.ACTION_UP, down) }
            val end = SystemClock.elapsedRealtime() + 1500
            var arrived = false
            while (!arrived && SystemClock.elapsedRealtime() < end) {
                SystemClock.sleep(100)
                arrived = downs(name) > before
            }
            if (!arrived) {
                if (!tap) touch(p, MotionEvent.ACTION_CANCEL, down)
                log("BAR press $name attempt=$attempt not delivered")
                SystemClock.sleep(500)
                continue
            }
            if (!tap) {
                SystemClock.sleep(maxOf(0L, holdMs - (SystemClock.uptimeMillis() - down)))
                touch(p, MotionEvent.ACTION_UP, down)
            }
            SystemClock.sleep(400)
            return true
        }
        return false
    }

    private fun waitE2E(before: Int, seconds: Int): String? {
        val lines = waitLog(seconds) { l -> l.count { it.contains(" E2E source=") } > before }
        return lines.filter { it.contains(" E2E source=") }.drop(before).lastOrNull()
    }

    /**
     * The talk bar and keys with injected touches, recorded speech standing in for the mic (the IME's debug
     * mic_feed.wav, played at real time through its recording path):
     *  - hold: press the bar for the length of s1.wav plus 0.6 s, let go: the dictation reaches the field;
     *  - latch: a 0.12 s tap starts listening, a second tap after the speech finishes it;
     *  - cancel: tap to listen, then the cancel button: nothing is committed;
     *  - digit: "123", then "7", then "ABC": the field reads "7".
     */
    private fun talkBar(field: EditText): Map<String, Boolean> {
        val r = linkedMapOf<String, Boolean>()
        val feed = File(dir, "mic_feed.wav")
        File(dir, "s1.wav").copyTo(feed, overwrite = true)
        val speechMs = (File(dir, "s1.wav").length() - 44) / 32
        fun heard(t: String) = t.contains("quarterly report", ignoreCase = true)

        focus(field)
        var before = e2eCount()
        press("mic", speechMs + 600)
        var e2e = waitE2E(before, 120)
        var text = fieldText(field)
        r["hold"] = e2e?.contains("source=mic") == true && heard(text)
        log("BAR hold ok=${r["hold"]} e2e=${e2e?.substringAfter("E2E ")?.substringBefore(" text=")} field=\"${esc(text)}\"")

        focus(field)
        before = e2eCount()
        press("mic", 120)
        SystemClock.sleep(speechMs + 600)
        val listening = phononLog().count { it.contains("MIC_UP") }
        press("mic")  // the bar has moved right of the cancel button: keyPos reads the new centre
        e2e = waitE2E(before, 120)
        text = fieldText(field)
        r["latch"] = e2e?.contains("source=mic") == true && heard(text)
        log("BAR latch ok=${r["latch"]} mic_up_before_second_tap=$listening e2e=${e2e?.substringAfter("E2E ")?.substringBefore(" text=")} field=\"${esc(text)}\"")

        focus(field)
        before = e2eCount()
        press("mic", 120)
        SystemClock.sleep(1500)
        val tapped = press("cancel")
        val cancelled = waitLog(10) { l -> l.any { it.contains("CANCEL phase=") } }.any { it.contains("CANCEL phase=") }
        SystemClock.sleep(5000)
        text = fieldText(field)
        r["cancel"] = tapped && cancelled && e2eCount() == before && text.isEmpty()
        log("BAR cancel ok=${r["cancel"]} tapped=$tapped cancel_logged=$cancelled e2e_after=${e2eCount() - before} field=\"${esc(text)}\"")

        focus(field)
        val digit = press("123") && press("7") && press("ABC")
        text = fieldText(field)
        r["digit"] = digit && text == "7"
        log("BAR digit ok=${r["digit"]} field=\"${esc(text)}\"")

        feed.delete()
        log("BAR_SUMMARY pass=${r.values.count { it }}/${r.size} ${r.entries.joinToString(" ") { "${it.key}=${it.value}" }}")
        return r
    }
}
