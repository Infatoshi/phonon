package ai.phonon.keyboard

import android.Manifest
import android.annotation.SuppressLint
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.inputmethodservice.InputMethodService
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.os.SystemClock
import android.text.InputType
import android.util.Log
import android.view.Gravity
import android.view.KeyEvent
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import java.io.File
import kotlin.concurrent.thread

/**
 * Phonon keyboard: QWERTY plus a hold-to-talk bar. The IME records the mic itself, transcribes with sherpa-onnx
 * Parakeet and refines with the LFM3-1.7B GGUF through llama.cpp, all in this process, then commits the text.
 * The models are downloaded on first use (ModelFetch); each loads as soon as its files are verified.
 *
 * Debug broadcasts (adb shell am broadcast -p ai.phonon.keyboard -a ACTION ...):
 *   ai.phonon.keyboard.REFINE  --es text "..." [--es mode minimal] [--ez commit true]
 *   ai.phonon.keyboard.WAV     --es path /sdcard/Android/data/ai.phonon.keyboard/files/x.wav  (stands in for a hold)
 *   ai.phonon.keyboard.STATUS
 */
class PhononIME : InputMethodService() {
    private val main = Handler(Looper.getMainLooper())
    private lateinit var worker: Handler
    @Volatile private var refiner: Refiner? = null
    @Volatile private var asr: Asr? = null
    private var mode = "minimal"
    private var shift = false
    private var status: TextView? = null
    private val keyViews = mutableListOf<Pair<String, View>>()

    // mic capture
    @Volatile private var recording = false
    private var recThread: Thread? = null
    private val pcm = java.io.ByteArrayOutputStream()

    private val receiver = object : BroadcastReceiver() {
        override fun onReceive(c: Context, i: Intent) {
            when (i.action) {
                ACTION_REFINE -> {
                    val text = i.getStringExtra("text") ?: return
                    val m = i.getStringExtra("mode") ?: mode
                    val commit = i.getBooleanExtra("commit", false)
                    worker.post {
                        val r = refiner ?: run { Log.w(TAG, "REFINE refiner not loaded"); return@post }
                        val t = SystemClock.elapsedRealtime()
                        val json = r.refine(text, m)
                        val ms = SystemClock.elapsedRealtime() - t
                        Log.i(TAG, "REFINE total_ms=$ms ${Mem.snapshot()} $json")
                        if (commit) main.post { commitResult(refinedText(json), t, "refine") }
                    }
                }
                ACTION_WAV -> {
                    val path = i.getStringExtra("path") ?: return
                    val tUp = SystemClock.elapsedRealtime()  // the simulated key-up
                    worker.post {
                        val samples = try { Wav.read(path) } catch (e: Throwable) { Log.e(TAG, "WAV read", e); return@post }
                        process(samples, tUp, "wav")
                    }
                }
                ACTION_STATUS -> Log.i(TAG, "STATUS asr=${asr != null} refiner=${refiner != null} ${Mem.snapshot()}")
            }
        }
    }

    override fun onCreate() {
        super.onCreate()
        worker = Handler(HandlerThread("phonon-engines").apply { start() }.looper)
        val f = IntentFilter().apply { addAction(ACTION_REFINE); addAction(ACTION_WAV); addAction(ACTION_STATUS) }
        if (Build.VERSION.SDK_INT >= 33) registerReceiver(receiver, f, RECEIVER_EXPORTED) else registerReceiver(receiver, f)
        Log.i(TAG, "IME_CREATE ${Mem.snapshot()}")
        ModelFetch.addListener(onModels)
        ModelFetch.start(this)
    }

    private val asrRequested = java.util.concurrent.atomic.AtomicBoolean(false)
    private val refinerRequested = java.util.concurrent.atomic.AtomicBoolean(false)

    /** Called on the download thread (and once at registration): load each model as soon as its files are ready. */
    private val onModels: (ModelFetch.State) -> Unit = { s ->
        if (s.asrDir != null && asrRequested.compareAndSet(false, true)) worker.post { loadAsr(s.asrDir) }
        if (s.done && refinerRequested.compareAndSet(false, true)) worker.post { loadRefiner(s.refinerPath) }
        if (!s.done || s.error != null) main.post { setStatus(modelStatus()) }
    }

    private fun loadAsr(dir: String) {
        Log.i(TAG, "ASR_DIR $dir")
        asr = Asr.load(dir, THREADS)
        asr?.let {  // warm-up, as on iOS and the Mac: the first decode pays ONNX Runtime's one-time setup
            val t = SystemClock.elapsedRealtime()
            it.transcribe(FloatArray(16000))
            Log.i(TAG, "ASR_WARMUP ms=${SystemClock.elapsedRealtime() - t}")
        }
        main.post { setStatus(if (asr != null) modelStatus() else "speech model failed to load") }
    }

    private fun loadRefiner(gguf: String?) {
        // null below the RAM gate: process() then commits the raw transcript
        if (gguf != null) refiner = Refiner.loadAuto(gguf, File(filesDir, "refiner_tune.txt").path)
        else Log.i(TAG, "REFINER_LOAD ok=false skipped=no_file (below the RAM gate or not sideloaded)")
        if (refiner != null) {  // warm-up, as on iOS: caches the minimal-mode prompt head
            Log.i(TAG, "WARMUP ${refiner!!.refine("warm up", "minimal")}")
        }
        main.post { setStatus("ready: asr=${asr != null} refiner=${refiner != null}") }
    }

    override fun onDestroy() {
        ModelFetch.removeListener(onModels)
        unregisterReceiver(receiver)
        super.onDestroy()
    }

    private fun setStatus(s: String) { status?.text = s }

    /** The keyboard's status line while models arrive: a download in progress or failed stays visible after the
     *  recogniser is ready, so a refiner that never arrives does not look like success. */
    private fun modelStatus(): String {
        val s = ModelFetch.state
        return when {
            refiner != null -> "ready"
            asr == null -> s.message
            s.done && s.error == null -> "dictation ready"
            else -> "dictation ready; ${s.message}"
        }
    }

    /** Each time the keyboard opens: retries a download that failed (a running or finished one is left alone). */
    override fun onStartInputView(info: android.view.inputmethod.EditorInfo?, restarting: Boolean) {
        super.onStartInputView(info, restarting)
        if (ModelFetch.state.error != null) ModelFetch.start(this)
    }

    /** Runs on the worker thread: ASR, refine, then commit on the main thread. */
    private fun process(samples: FloatArray, tUp: Long, source: String) {
        val a = asr ?: run {
            Log.w(TAG, "ASR not loaded")
            main.post { setStatus("the speech model is not ready yet: ${ModelFetch.state.message}") }
            return
        }
        main.post { setStatus("transcribing...") }
        val t0 = SystemClock.elapsedRealtime()
        val raw = a.transcribe(samples)
        val t1 = SystemClock.elapsedRealtime()
        val r = refiner
        val json = if (r != null && raw.isNotEmpty()) r.refine(raw, mode) else null
        val t2 = SystemClock.elapsedRealtime()
        // Refiner gated off by RAM (r == null): the raw transcript still gets the line-break stage, since Parakeet
        // already writes "1. ... 2. ..." for dictated steps (Galaxy A16 5G, Test Lab 2026-11-08).
        val text = json?.let { refinedText(it) } ?: if (r == null) LineBreaks.apply(raw, mode) else raw
        Log.i(TAG, "PIPELINE source=$source audio_ms=${samples.size / 16} asr_ms=${t1 - t0} refine_ms=${t2 - t1} " +
            "raw=\"$raw\" refined=${json ?: "null"} ${Mem.snapshot()}")
        main.post { commitResult(text, tUp, source) }
    }

    /** The refiner's text through the shared output stage, in the writing mode the refiner ran with: the opener
     *  the refiner dropped back in front (crates/phonon-llm opener), then numbered steps one per line, and in
     *  formal mode the greeting and sign-off on their own lines (linebreak). A length-guard fallback is the raw
     *  transcript and stays as it is, as on the Mac and iPhone. */
    private fun refinedText(json: String): String {
        val o = org.json.JSONObject(json)
        val text = o.getString("text")
        if (o.optBoolean("guard", false)) return text
        val m = o.optString("mode", mode)
        return LineBreaks.apply(Opener.restore(o.optString("raw", ""), text, m), m)
    }

    private fun commitResult(result: String, tUp: Long, source: String) {
        val ic = currentInputConnection
        // A single-line field (search box, many chat inputs) has no use for a newline, and some apps treat
        // one as Enter, which would send half the dictation: those fields get one line.
        val multiLine = ((currentInputEditorInfo?.inputType ?: 0) and InputType.TYPE_TEXT_FLAG_MULTI_LINE) != 0
        val text = if (multiLine) result else result.replace('\n', ' ')
        // commitText returns true even with no real field behind it (a dialog over the app, a window without an
        // editor: the system then binds a TYPE_NULL editor), so "committed" also needs a bound text field. The text
        // is still sent: some real editors are TYPE_NULL (terminal apps read the commit as key input).
        val bound = currentInputStarted && (currentInputEditorInfo?.inputType ?: InputType.TYPE_NULL) != InputType.TYPE_NULL
        val ok = (ic?.commitText(text, 1) ?: false) && bound
        Log.i(TAG, "E2E source=$source key_up_to_commit_ms=${SystemClock.elapsedRealtime() - tUp} committed=$ok " +
            "bound=$bound text=\"$text\"")
        setStatus(if (ok) "committed" else "no field")
    }

    @SuppressLint("MissingPermission")
    private fun startRecording() {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            setStatus("mic permission missing: open the Phonon app once"); return
        }
        pcm.reset()
        recording = true
        setStatus("listening...")
        recThread = thread(name = "phonon-mic") {
            val min = AudioRecord.getMinBufferSize(16000, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
            val rec = AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, 16000,
                AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, maxOf(min, 3200) * 2)
            val buf = ByteArray(3200)
            rec.startRecording()
            while (recording) {
                val n = rec.read(buf, 0, buf.size)
                if (n > 0) synchronized(pcm) { pcm.write(buf, 0, n) }
            }
            rec.stop(); rec.release()
        }
    }

    private fun stopRecording() {
        val tUp = SystemClock.elapsedRealtime()
        recording = false
        recThread?.join(500)
        val bytes = synchronized(pcm) { pcm.toByteArray() }
        val samples = FloatArray(bytes.size / 2) {
            ((bytes[2 * it].toInt() and 0xff) or (bytes[2 * it + 1].toInt() shl 8)).toShort() / 32768f
        }
        var peak = 0f
        for (s in samples) peak = maxOf(peak, kotlin.math.abs(s))
        Log.i(TAG, "MIC_UP samples=${samples.size} peak=$peak")
        worker.post { process(samples, tUp, "mic") }
    }

    // ---------- keyboard view ----------

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreateInputView(): View {
        val dp = resources.displayMetrics.density
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.rgb(28, 28, 30))
            setPadding((4 * dp).toInt(), (4 * dp).toInt(), (4 * dp).toInt(), (8 * dp).toInt())
        }
        status = TextView(this).apply {
            setTextColor(Color.LTGRAY); textSize = 12f; gravity = Gravity.CENTER
            text = modelStatus()
            contentDescription = "status"
            // while the refiner waits for Wi-Fi, one tap allows mobile data (remembered)
            setOnClickListener { if (ModelFetch.state.waitingForWifi) ModelFetch.allowCellular(this@PhononIME) }
        }
        root.addView(status)
        val mic = TextView(this).apply {
            text = "Hold to talk"; contentDescription = "mic"; gravity = Gravity.CENTER
            setTextColor(Color.WHITE); textSize = 16f
            background = pill(Color.rgb(10, 132, 255), dp)
            setOnTouchListener { v, e ->
                when (e.action) {
                    MotionEvent.ACTION_DOWN -> { v.alpha = 0.6f; startRecording() }
                    MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> { v.alpha = 1f; if (recording) stopRecording() }
                }
                true
            }
        }
        root.addView(mic, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, (48 * dp).toInt()).apply {
            setMargins((4 * dp).toInt(), (4 * dp).toInt(), (4 * dp).toInt(), (6 * dp).toInt())
        })
        keyViews.clear()
        keyViews += "mic" to mic
        val rows = listOf("qwertyuiop", "asdfghjkl", "^zxcvbnm<", "_,␣.⏎")
        for (row in rows) {
            val line = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER }
            for (ch in row) {
                val label = when (ch) { '^' -> "⇧"; '<' -> "⌫"; '␣' -> "space"; '⏎' -> "⏎"; '_' -> ""; else -> ch.toString() }
                if (ch == '_') { line.addView(View(this), LinearLayout.LayoutParams(0, 1, 1f)); continue }
                val key = TextView(this).apply {
                    text = label; gravity = Gravity.CENTER; setTextColor(Color.WHITE); textSize = 18f
                    contentDescription = "key_$label"
                    background = pill(Color.rgb(72, 72, 74), dp)
                    setOnClickListener { onKey(ch) }
                }
                val w = if (ch == '␣') 5f else if (ch == '^' || ch == '<' || ch == '⏎') 1.5f else 1f
                line.addView(key, LinearLayout.LayoutParams(0, (46 * dp).toInt(), w).apply {
                    setMargins((3 * dp).toInt(), (4 * dp).toInt(), (3 * dp).toInt(), (4 * dp).toInt())
                })
                keyViews += label to key
            }
            root.addView(line)
        }
        root.viewTreeObserver.addOnGlobalLayoutListener {
            // key centres in screen pixels, so adb `input tap` can press real keys headlessly
            val loc = IntArray(2)
            val sb = StringBuilder("KEYPOS")
            for ((l, v) in keyViews) {
                v.getLocationOnScreen(loc)
                sb.append(" ").append(if (l == "space") "space" else l).append("=")
                    .append(loc[0] + v.width / 2).append(",").append(loc[1] + v.height / 2)
            }
            Log.i(TAG, sb.toString())
        }
        return root
    }

    private fun pill(c: Int, dp: Float) = GradientDrawable().apply { setColor(c); cornerRadius = 8 * dp }

    private fun onKey(ch: Char) {
        val ic = currentInputConnection ?: return
        when (ch) {
            '^' -> { shift = !shift; return }
            '<' -> ic.deleteSurroundingText(1, 0)
            '␣' -> ic.commitText(" ", 1)
            '⏎' -> ic.sendKeyEvent(KeyEvent(KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_ENTER)).also {
                ic.sendKeyEvent(KeyEvent(KeyEvent.ACTION_UP, KeyEvent.KEYCODE_ENTER))
            }
            else -> {
                ic.commitText(if (shift) ch.uppercase() else ch.toString(), 1)
                shift = false
            }
        }
        Log.i(TAG, "KEY $ch")
    }

    companion object {
        const val ACTION_REFINE = "ai.phonon.keyboard.REFINE"
        const val ACTION_WAV = "ai.phonon.keyboard.WAV"
        const val ACTION_STATUS = "ai.phonon.keyboard.STATUS"
        const val THREADS = 4
    }
}
