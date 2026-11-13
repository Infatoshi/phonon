package ai.phonon.keyboard

import android.Manifest
import android.annotation.SuppressLint
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
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
import android.view.HapticFeedbackConstants
import android.view.KeyEvent
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.EditorInfo
import android.widget.LinearLayout
import android.widget.TextView
import java.io.File
import kotlin.concurrent.thread

/**
 * Phonon keyboard: the talk bar (TalkBar: hold to talk, or tap to latch) over letter, number and symbol keys
 * (KeyPanel), the iPhone keyboard's design. The IME records the mic itself, transcribes with sherpa-onnx
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
    /** The writing mode chosen in the app (WritingModes, default minimal), read for every dictation. */
    private val mode: String get() = WritingModes.get(this)
    private var bar: TalkBar? = null
    private var cancel: View? = null
    private var keys: KeyPanel? = null
    private var micView: View? = null

    /** The talk bar's state, as on the iPhone: at rest, listening (held, or latched on by a tap) or writing. */
    private enum class Phase { READY, LISTENING, WORKING }
    private var phase = Phase.READY
    private var latched = false
    /** Bumped by cancel: a dictation already in the engines whose generation is stale is not committed. */
    @Volatile private var generation = 0
    private var flashText: String? = null
    @Volatile private var asrError: String? = null
    private var flashId = 0

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
                        if (refiner == null) { Log.w(TAG, "REFINE refiner not loaded"); return@post }
                        val t = SystemClock.elapsedRealtime()
                        val (out, json) = correct(text, m)
                        val ms = SystemClock.elapsedRealtime() - t
                        Log.i(TAG, "REFINE total_ms=$ms ${Mem.snapshot()} out=\"$out\" $json")
                        if (commit) main.post { commitResult(out, t, "refine") }
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
        if (!s.done || s.error != null) main.post { setStatus(modelStatus() ?: "ready") }
    }

    private fun loadAsr(dir: String) {
        Log.i(TAG, "ASR_DIR $dir")
        asr = Asr.load(dir, THREADS)
        asr?.let {  // warm-up, as on iOS and the Mac: the first decode pays ONNX Runtime's one-time setup
            val t = SystemClock.elapsedRealtime()
            it.transcribe(FloatArray(16000))
            Log.i(TAG, "ASR_WARMUP ms=${SystemClock.elapsedRealtime() - t}")
        }
        if (asr == null) asrError = "The speech model failed to load"
        main.post { setStatus(modelStatus() ?: "ready") }
    }

    private fun loadRefiner(gguf: String?) {
        // null below the RAM gate: process() then commits the raw transcript
        if (gguf != null) refiner = Refiner.loadAuto(gguf, File(filesDir, "refiner_tune.txt").path)
        else Log.i(TAG, "REFINER_LOAD ok=false skipped=no_file (below the RAM gate or not sideloaded)")
        if (refiner != null) {  // warm-up, as on iOS: caches the minimal-mode prompt head
            val t = SystemClock.elapsedRealtime()
            val json = refiner!!.refine("warm up", "minimal")
            Log.i(TAG, "WARMUP $json")  // ImeE2ETest waits for " WARMUP {"
            Log.i(TAG, "WARMUP_MS ${SystemClock.elapsedRealtime() - t}")
        }
        main.post { setStatus("ready: asr=${asr != null} refiner=${refiner != null}") }
        primeModes(Refiner.MODES.filter { it != "minimal" })
    }

    /** The other writing modes' prompt heads, cached one per worker message (Refiner.warmUp's second half), so a
     *  dictation that arrives meanwhile waits for at most one head (about 1.5 s on a Galaxy A16, Test Lab
     *  2026-11-12) and the keyboard reads ready as early as before. */
    private fun primeModes(left: List<String>) {
        if (left.isEmpty()) return
        worker.post {
            val r = refiner ?: return@post
            val t = SystemClock.elapsedRealtime()
            val ms = r.prime(left[0])
            Log.i(TAG, "PRIME_MODE mode=${left[0]} ms=$ms wall_ms=${SystemClock.elapsedRealtime() - t}")
            primeModes(left.drop(1))
        }
    }

    override fun onDestroy() {
        ModelFetch.removeListener(onModels)
        unregisterReceiver(receiver)
        super.onDestroy()
    }

    /** Models changed: the resting bar's label follows them (the status itself is in the log). */
    private fun setStatus(s: String) {
        Log.i(TAG, "STATUS_LINE $s")
        paintBar()
    }

    /** The bar's label while models arrive: a download in progress or failed stays visible after the recogniser is
     *  ready, so a refiner that never arrives does not look like success. Null once both models are loaded (or the
     *  refiner is gated off by RAM and the download is finished). */
    private fun modelStatus(): String? {
        val s = ModelFetch.state
        return when {
            refiner != null -> null
            asr == null -> asrError ?: s.message
            s.done && s.error == null -> null
            else -> "Hold to talk · ${s.message}"
        }
    }

    /** Each time the keyboard opens: retries a download that failed (a running or finished one is left alone). */
    override fun onStartInputView(info: android.view.inputmethod.EditorInfo?, restarting: Boolean) {
        super.onStartInputView(info, restarting)
        Log.i(TAG, "INPUT_VIEW app=${info?.packageName} inputType=0x${Integer.toHexString(info?.inputType ?: 0)}")
        if (ModelFetch.state.error != null) ModelFetch.start(this)
        keys?.reset()
        bar?.animate = !lowPower()
        paintBar()
    }

    override fun onFinishInputView(finishingInput: Boolean) {
        super.onFinishInputView(finishingInput)
        bar?.animate = false
        if (recording) cancelDictation(null)  // the keyboard went away mid-dictation
    }

    private fun lowPower() = (getSystemService(Context.POWER_SERVICE) as android.os.PowerManager).isPowerSaveMode

    /** The bar for the current phase: the Mac capsule while listening or writing, else the resting fluid with the
     *  model status (dimmed while the speech model is missing) or a status flashed for about 2 s. */
    private fun paintBar() {
        val b = bar ?: return
        b.mode = when (phase) {
            Phase.LISTENING -> TalkBar.Mode.LISTENING
            Phase.WORKING -> TalkBar.Mode.WORKING
            Phase.READY -> TalkBar.Mode.REST
        }
        b.dim = asr == null
        b.label = if (phase == Phase.READY) (modelStatus() ?: flashText) else null
        cancel?.visibility = if (phase == Phase.READY) View.GONE else View.VISIBLE
        micView?.contentDescription = when (phase) {
            Phase.LISTENING -> if (latched) "Finish dictation (tap)" else "Finish dictation (let go)"
            Phase.WORKING -> "Writing it down"
            Phase.READY -> modelStatus() ?: "Dictate: hold to talk, or tap"
        }
    }

    private fun setPhase(p: Phase) {
        phase = p
        if (p != Phase.READY) { flashId++; flashText = null }
        paintBar()
    }

    /** A status over the resting bar for about 2 s ("Didn't catch that", "Cancelled", the word count). */
    private fun flash(s: String) {
        Log.i(TAG, "FLASH $s")
        val id = ++flashId
        flashText = s
        paintBar()
        main.postDelayed({ if (flashId == id) { flashText = null; paintBar() } }, 2000)
    }

    /** Runs on the worker thread: ASR, refine, then commit on the main thread. */
    private fun process(samples: FloatArray, tUp: Long, source: String, gen: Int = generation) {
        val a = asr ?: run {
            Log.w(TAG, "ASR not loaded")
            main.post { setPhase(Phase.READY); flash("The speech model is not ready yet") }
            return
        }
        main.post { if (gen == generation) setPhase(Phase.WORKING) }
        val t0 = SystemClock.elapsedRealtime()
        val raw = a.transcribe(samples)
        val t1 = SystemClock.elapsedRealtime()
        val (text, json) = correct(raw, mode)
        val t2 = SystemClock.elapsedRealtime()
        Log.i(TAG, "PIPELINE source=$source audio_ms=${samples.size / 16} asr_ms=${t1 - t0} refine_ms=${t2 - t1} " +
            "raw=\"$raw\" refined=${json ?: "null"} ${Mem.snapshot()}")
        main.post {
            if (gen != generation) { Log.i(TAG, "E2E source=$source cancelled"); return@post }
            commitResult(text, tUp, source)
        }
    }

    @Volatile private var dict: Dictionary = Dictionary(emptyList(), null)
    private var dictStamp = -1L
    private var words: WordList? = null

    /** The user's dictionary (DictionaryActivity writes it), reread when the file changes; the English word list
     *  the phonetic guesses need is read from the APK the first time a dictionary has entries. */
    private fun dictionary(): Dictionary {
        val f = File(filesDir, Dictionary.FILE_NAME)
        val stamp = if (f.isFile) f.lastModified() xor f.length() else 0L
        if (stamp != dictStamp) {
            dictStamp = stamp
            dict = Dictionary.load(filesDir) {
                words ?: try {
                    WordList(assets.open("english_words.txt").use { it.readBytes() }).also { words = it }
                } catch (e: Exception) {
                    Log.w(TAG, "DICT no word list", e); null
                }
            }
            Log.i(TAG, "DICT entries=${dict.entries.size}")
        }
        return dict
    }

    /**
     * Transcript to committed text (runs on the worker): the dictionary's exact replacements and phonetic
     * corrections, the refiner with the dictionary's vocabulary line, then the shared output stage. Returns the text
     * and the refiner's JSON (null without a refiner). An empty dictionary leaves every step as it was.
     */
    private fun correct(raw: String, m: String): Pair<String, String?> {
        val d = dictionary()
        val prepared = d.prepare(raw)
        val r = refiner
        // Refiner gated off by RAM (r == null): the transcript still gets the line-break stage, since Parakeet
        // already writes "1. ... 2. ..." for dictated steps (Galaxy A16 5G, Test Lab 2026-11-08).
        if (r == null) return LineBreaks.apply(d.applyExact(prepared.text), m) to null
        if (raw.isEmpty()) return raw to null
        val json = r.refine(prepared.text, m, prepared.vocabLine)
        return refinedText(json, raw, prepared, d) to json
    }

    /** The refiner's text through the shared output stage, in the writing mode the refiner ran with: a one-word
     *  dictionary term the refiner swapped put back (polish_server restore_terms), the opener the refiner dropped
     *  back in front (crates/phonon-llm opener), the dictionary's exact replacements, then numbered steps one per
     *  line, and in formal mode the greeting and sign-off on their own lines (linebreak). A length-guard fallback is
     *  the refiner's input and stays as it is, as on the Mac and iPhone. */
    private fun refinedText(json: String, raw: String, prepared: Dictionary.Prepared, d: Dictionary): String {
        val o = org.json.JSONObject(json)
        val text = o.getString("text")
        if (o.optBoolean("guard", false)) return text
        val m = o.optString("mode", mode)
        val restored = Dictionary.restoreTerms(prepared.text, text, prepared.vocab)
        return LineBreaks.apply(d.applyExact(Opener.restore(raw, restored, m)), m)
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
        if (ok) {
            // what the field now holds before the cursor, read through the same connection: proof the text landed in
            // apps whose fields a test cannot read (ImeE2ETest's Messages check)
            val back = ic?.getTextBeforeCursor(text.length + 40, 0)?.toString()
            Log.i(TAG, "E2E_READBACK app=${currentInputEditorInfo?.packageName} landed=${back?.endsWith(text) == true} " +
                "before=\"${back?.replace("\n", "\\n")}\"")
        }
        setPhase(Phase.READY)
        val words = text.split(Regex("\\s+")).count { it.isNotEmpty() }
        flash(when {
            words == 0 -> "Didn't catch that"
            !ok -> "No text field here"
            else -> "$words ${if (words == 1) "word" else "words"} · ${"%.1f".format((SystemClock.elapsedRealtime() - tUp) / 1000.0)} s"
        })
    }

    @SuppressLint("MissingPermission")
    private fun startRecording(): Boolean {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            flash("Open Phonon once to allow the microphone"); return false
        }
        pcm.reset()
        recording = true
        setPhase(Phase.LISTENING)
        val feed = micFeed()
        if (feed != null) {
            // a debug build with a mic_feed.wav: the bar's real touch and recording path, with recorded speech in
            // place of the microphone at real time (a Test Lab phone hears nothing; ImeE2ETest's talk bar checks)
            Log.i(TAG, "MIC_FEED samples=${feed.size}")
            recThread = thread(name = "phonon-mic") {
                val buf = ByteArray(1600)
                var at = 0
                var next = SystemClock.elapsedRealtime()
                while (recording) {
                    for (i in 0 until 800) {
                        val s = if (at < feed.size) (feed[at++] * 32767).toInt().coerceIn(-32768, 32767) else 0
                        buf[2 * i] = s.toByte(); buf[2 * i + 1] = (s shr 8).toByte()
                    }
                    next += 50
                    SystemClock.sleep(maxOf(0L, next - SystemClock.elapsedRealtime()))
                    if (!recording) break
                    synchronized(pcm) { pcm.write(buf, 0, buf.size) }
                    val level = level(buf, buf.size)
                    main.post { if (phase == Phase.LISTENING) bar?.level = level }
                }
            }
            return true
        }
        val gen = generation
        recThread = thread(name = "phonon-mic") {
            val min = AudioRecord.getMinBufferSize(16000, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
            // during a phone call (or with another app holding the mic) the recorder can fail to open, fail to
            // start, or start returning error codes: each ends the dictation with a message, never a spin or crash
            val rec = try {
                AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, 16000,
                    AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, maxOf(min, 3200) * 2)
            } catch (e: Exception) { Log.w(TAG, "MIC_ERROR create", e); null }
            if (rec == null || rec.state != AudioRecord.STATE_INITIALIZED) {
                Log.w(TAG, "MIC_ERROR state=${rec?.state}")
                rec?.release()
                micFailed(gen)
                return@thread
            }
            val buf = ByteArray(1600)  // 50 ms: the level the capsule lights with updates 20 times a second
            try {
                rec.startRecording()
                val err = if (rec.recordingState != AudioRecord.RECORDSTATE_RECORDING) -3
                else MicLoop.run(buf, { recording }, { rec.read(it, 0, it.size) }) { b, n ->
                    synchronized(pcm) { pcm.write(b, 0, n) }
                    val level = level(b, n)
                    main.post { if (phase == Phase.LISTENING) bar?.level = level }
                }
                if (err != null) { Log.w(TAG, "MIC_ERROR read=$err"); micFailed(gen) }
            } catch (e: IllegalStateException) {
                Log.w(TAG, "MIC_ERROR start", e)
                micFailed(gen)
            } finally {
                try { rec.stop() } catch (_: IllegalStateException) {}
                rec.release()
            }
        }
        return true
    }

    /** The mic failed under this dictation: with half a second or more already heard, write that down as if the
     *  user let go (a call arriving mid-sentence keeps the sentence); otherwise drop it and say why on the bar. */
    private fun micFailed(gen: Int) {
        main.post {
            if (!recording || gen != generation) return@post
            if (synchronized(pcm) { pcm.size() } >= 16000) stopRecording() else cancelDictation(MicLoop.BUSY)
        }
    }

    /** Debug builds only: recorded speech that stands in for the microphone, if the test put one in place. */
    private fun micFeed(): FloatArray? {
        if ((applicationInfo.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE) == 0) return null
        val f = File(getExternalFilesDir(null) ?: return null, "mic_feed.wav")
        return if (f.isFile) try { Wav.read(f.path) } catch (e: Exception) { Log.w(TAG, "MIC_FEED", e); null } else null
    }

    /** RMS of the chunk in dBFS, -55..-15 mapped to 0..1 (the iPhone keyboard's meter). */
    private fun level(buf: ByteArray, n: Int): Double {
        var sum = 0.0
        var i = 0
        while (i + 1 < n) {
            val s = ((buf[i].toInt() and 0xff) or (buf[i + 1].toInt() shl 8)).toShort() / 32768.0
            sum += s * s; i += 2
        }
        val rms = kotlin.math.sqrt(sum / maxOf(1, n / 2))
        val db = 20 * kotlin.math.log10(maxOf(rms, 1e-6))
        return ((db + 55) / 40).coerceIn(0.0, 1.0)
    }

    /** Drop the recording, or a dictation still in the engines (its result is not committed). */
    private fun cancelDictation(message: String?) {
        generation++
        if (recording) { recording = false; recThread?.join(500) }
        latched = false
        Log.i(TAG, "CANCEL phase=$phase")
        setPhase(Phase.READY)
        message?.let { flash(it) }
    }

    private fun stopRecording() {
        val tUp = SystemClock.elapsedRealtime()
        recording = false
        latched = false
        setPhase(Phase.WORKING)
        recThread?.join(500)
        val bytes = synchronized(pcm) { pcm.toByteArray() }
        val samples = FloatArray(bytes.size / 2) {
            ((bytes[2 * it].toInt() and 0xff) or (bytes[2 * it + 1].toInt() shl 8)).toShort() / 32768f
        }
        var peak = 0f
        for (s in samples) peak = maxOf(peak, kotlin.math.abs(s))
        Log.i(TAG, "MIC_UP samples=${samples.size} peak=$peak")
        val gen = generation
        worker.post { process(samples, tUp, "mic", gen) }
    }

    // ---------- keyboard view ----------

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreateInputView(): View {
        val dp = resources.displayMetrics.density
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(KeyPanel.BG)  // the iPhone keyboard's ember palette, in light and dark alike
            setPadding(0, 0, 0, (8 * dp).toInt())
            // Android 15 (targetSdk 35) draws the keyboard window edge to edge: without the navigation bar's inset
            // the gesture handle and the IME switcher sat on top of the bottom key row
            setOnApplyWindowInsetsListener { v, insets ->
                val nav = if (Build.VERSION.SDK_INT >= 30)
                    insets.getInsets(android.view.WindowInsets.Type.navigationBars()).bottom
                else @Suppress("DEPRECATION") insets.systemWindowInsetBottom
                v.setPadding(v.paddingLeft, v.paddingTop, v.paddingRight, (8 * dp).toInt() + nav)
                insets
            }
        }
        // the toolbar: a cancel button while listening or writing, then the talk bar (KeyboardUI.swift toolbar)
        val toolbar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
            setPadding((6 * dp).toInt(), (4 * dp).toInt(), (6 * dp).toInt(), 0)
        }
        val x = TextView(this).apply {
            text = "✕"; gravity = Gravity.CENTER; setTextColor(KeyPanel.SOFT); textSize = 14f
            contentDescription = "Cancel dictation"
            background = GradientDrawable().apply { shape = GradientDrawable.OVAL; setColor(KeyPanel.RAISED) }
            visibility = View.GONE
            setOnTouchListener { _, e ->
                if (e.actionMasked == MotionEvent.ACTION_DOWN) Log.i(TAG, "TOUCH_DOWN key=cancel")
                false
            }
            setOnClickListener { v ->
                v.performHapticFeedback(HapticFeedbackConstants.KEYBOARD_TAP)
                cancelDictation("Cancelled")
            }
        }
        cancel = x
        toolbar.addView(x, LinearLayout.LayoutParams((40 * dp).toInt(), (40 * dp).toInt()).apply {
            marginEnd = (8 * dp).toInt()
        })
        val talk = TalkBar(this)
        talk.isClickable = true
        talk.setOnTouchListener { v, e -> talkTouch(v, e); true }
        bar = talk
        micView = talk
        toolbar.addView(talk, LinearLayout.LayoutParams(0, (42 * dp).toInt(), 1f))
        root.addView(toolbar, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, (48 * dp).toInt() + (4 * dp).toInt()))
        val panel = KeyPanel(this, keyHost).apply { setPadding((3 * dp).toInt(), (10 * dp).toInt(), (3 * dp).toInt(), 0) }
        keys = panel
        root.addView(panel)
        panel.onLayout = { logKeys() }
        root.viewTreeObserver.addOnGlobalLayoutListener { logKeys() }
        paintBar()
        return root
    }

    private var lastKeyPos = ""

    /** Key centres in screen pixels, so adb `input tap` can press real keys headlessly (scripts/m2_type.sh, m4_mic.sh). */
    private fun logKeys() {
        val panel = keys ?: return
        val loc = IntArray(2)
        val sb = StringBuilder("KEYPOS")
        val all = listOfNotNull(micView?.let { "mic" to it }, cancel?.takeIf { it.visibility == View.VISIBLE }?.let { "cancel" to it }) +
            panel.keyViews
        for ((l, v) in all) {
            v.getLocationOnScreen(loc)
            sb.append(" ").append(l).append("=").append(loc[0] + v.width / 2).append(",").append(loc[1] + v.height / 2)
        }
        val s = sb.toString()
        if (s != lastKeyPos) { lastKeyPos = s; Log.i(TAG, s) }
    }

    private var pressedAt = 0L
    private enum class Press { NONE, STARTING, STOPPING }
    private var press = Press.NONE

    /** Hold to talk and let go to finish, or tap once to latch it on and tap again to finish (KeyboardUI.swift
     *  talkBar): a release within 0.35 s of the press latches. Before the speech model is ready a tap does what the
     *  label asks (allow mobile data while the text model waits for Wi-Fi). */
    private fun talkTouch(v: View, e: MotionEvent) {
        when (e.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                Log.i(TAG, "TOUCH_DOWN key=mic phase=$phase")
                pressedAt = SystemClock.uptimeMillis()
                press = Press.NONE
                when {
                    phase == Phase.LISTENING -> press = Press.STOPPING
                    phase == Phase.WORKING -> {}
                    ModelFetch.state.waitingForWifi && asr == null -> ModelFetch.allowCellular(this)
                    asr == null -> flash(modelStatus() ?: "The speech model is not ready yet")
                    else -> {
                        if (ModelFetch.state.waitingForWifi) ModelFetch.allowCellular(this)
                        v.performHapticFeedback(HapticFeedbackConstants.VIRTUAL_KEY)
                        latched = false
                        if (startRecording()) press = Press.STARTING
                    }
                }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                val held = SystemClock.uptimeMillis() - pressedAt
                when (press) {
                    Press.STOPPING -> if (recording) stopRecording()
                    Press.STARTING -> if (held < 350 && e.actionMasked == MotionEvent.ACTION_UP) {
                        latched = true  // a quick tap: stay on until the next tap
                        paintBar()
                    } else if (recording) stopRecording()
                    Press.NONE -> {}
                }
                press = Press.NONE
            }
        }
    }

    private val keyHost = object : KeyPanel.Host {
        override fun type(s: String) {
            currentInputConnection?.commitText(s, 1)
            Log.i(TAG, "KEY $s")
        }

        override fun deleteBackward() {
            val ic = currentInputConnection ?: return
            // a selection goes as a whole; otherwise one character (a surrogate pair counts as one)
            if (!ic.getSelectedText(0).isNullOrEmpty()) ic.commitText("", 1)
            else {
                val b = ic.getTextBeforeCursor(2, 0)
                val n = if (b != null && b.length == 2 && Character.isSurrogatePair(b[0], b[1])) 2 else 1
                ic.deleteSurroundingText(n, 0)
            }
            Log.i(TAG, "KEY ⌫")
        }

        override fun enter() {
            val ic = currentInputConnection ?: return
            val info = currentInputEditorInfo
            val action = (info?.imeOptions ?: 0) and EditorInfo.IME_MASK_ACTION
            val multiLine = ((info?.inputType ?: 0) and InputType.TYPE_TEXT_FLAG_MULTI_LINE) != 0
            val noEnterAction = ((info?.imeOptions ?: 0) and EditorInfo.IME_FLAG_NO_ENTER_ACTION) != 0
            // a single-line field with an action (search, send, go) gets the action; otherwise a newline
            if (!multiLine && !noEnterAction && action != EditorInfo.IME_ACTION_NONE && action != EditorInfo.IME_ACTION_UNSPECIFIED) {
                ic.performEditorAction(action)
            } else {
                ic.sendKeyEvent(KeyEvent(KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_ENTER))
                ic.sendKeyEvent(KeyEvent(KeyEvent.ACTION_UP, KeyEvent.KEYCODE_ENTER))
            }
            Log.i(TAG, "KEY ⏎")
        }

        override fun nextKeyboard() { switchToNextInputMethod(false) }

        override fun showGlobe() = shouldOfferSwitchingToNextInputMethod()

        override fun before() = currentInputConnection?.getTextBeforeCursor(4, 0)?.toString() ?: ""

        override fun autoCaps(): Boolean {
            val t = currentInputEditorInfo?.inputType ?: return false
            if ((t and InputType.TYPE_MASK_CLASS) != InputType.TYPE_CLASS_TEXT) return false
            val v = t and InputType.TYPE_MASK_VARIATION
            return v !in setOf(InputType.TYPE_TEXT_VARIATION_EMAIL_ADDRESS, InputType.TYPE_TEXT_VARIATION_WEB_EMAIL_ADDRESS,
                InputType.TYPE_TEXT_VARIATION_URI, InputType.TYPE_TEXT_VARIATION_PASSWORD,
                InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD, InputType.TYPE_TEXT_VARIATION_WEB_PASSWORD)
        }
    }

    companion object {
        const val ACTION_REFINE = "ai.phonon.keyboard.REFINE"
        const val ACTION_WAV = "ai.phonon.keyboard.WAV"
        const val ACTION_STATUS = "ai.phonon.keyboard.STATUS"
        const val THREADS = 4
    }
}
