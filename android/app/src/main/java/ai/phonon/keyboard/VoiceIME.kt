package ai.phonon.keyboard

import android.Manifest
import android.annotation.SuppressLint
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.os.Build
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.inputmethodservice.InputMethodService
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.EditorInfo
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.io.ByteArrayOutputStream
import java.io.File
import kotlin.concurrent.thread

/**
 * Phonon voice input: a voice-only input method for another keyboard's mic key. FUTO Keyboard (with its built-in
 * voice input turned off), AOSP-based keyboards and Android itself switch to an input method whose subtype mode is
 * "voice" (res/xml/voice_method.xml); this is one.
 *
 * It starts listening as soon as it opens. The audio goes to a Mac running Phonon (MacDictation, the WebSocket API)
 * as it is recorded; the Mac's corrected text is typed into the field and the previous keyboard comes back. When no
 * Mac is set up, or it cannot take the dictation, this phone transcribes the recording itself with the speech
 * model the Phonon keyboard downloads (raw transcript, no correction), if that model is on the phone.
 *
 * Debug builds only, and only from adb (adb shell am broadcast -p ai.phonon.keyboard -a ACTION ...):
 *   ai.phonon.keyboard.VOICE_CONFIG --es url ws://10.0.2.2:8377 --es token ...   the Mac, as the settings screen sets it
 *   ai.phonon.keyboard.VOICE_WAV --es path /sdcard/Android/data/ai.phonon.keyboard/files/x.wav
 *     the open dictation hears this 16 kHz mono WAV instead of the microphone, then Done
 *   ai.phonon.keyboard.VOICE_STATUS   logs whether Android lists this input method as a voice shortcut
 */
class VoiceIME : InputMethodService() {
    private val main = Handler(Looper.getMainLooper())
    private lateinit var worker: Handler
    private lateinit var settings: MacSettings

    private var status: TextView? = null
    private var preview: TextView? = null
    private var doneButton: TextView? = null
    private var retryButton: TextView? = null

    @Volatile private var recording = false
    private var recThread: Thread? = null
    /** The whole recording (16 kHz PCM16 LE), for this phone's own transcription and a retry. Main thread. */
    private val pcm = ByteArrayOutputStream()
    private var mac: MacDictation? = null
    /** Why the Mac is out of this dictation, once it is. */
    private var macProblem: String? = null
    private var active = false
    /** The text is typed and the previous keyboard is coming back: a restart of the field must not listen again. */
    private var leaving = false
    private var stopped = false
    /** Bumped for every dictation: callbacks of an older one are ignored. */
    private var generation = 0

    private val debugReceiver = object : BroadcastReceiver() {
        override fun onReceive(c: Context, i: Intent) {
            when (i.action) {
                ACTION_CONFIG -> {
                    i.getStringExtra("url")?.let { settings.url = it }
                    i.getStringExtra("token")?.let { settings.token = it }
                    Log.i(TAG, "VOICE_CONFIG url=${settings.url} configured=${settings.configured}")
                }
                ACTION_WAV -> {
                    val path = i.getStringExtra("path") ?: return
                    val samples = try { Wav.read(path) } catch (e: Throwable) { Log.e(TAG, "VOICE_WAV", e); return }
                    if (!active || stopped) { Log.w(TAG, "VOICE_WAV no open dictation"); return }
                    stopRecording()
                    val bytes = ByteArray(samples.size * 2)
                    for (k in samples.indices) {
                        val v = (samples[k] * 32768f).toInt().coerceIn(-32768, 32767)
                        bytes[2 * k] = v.toByte(); bytes[2 * k + 1] = (v shr 8).toByte()
                    }
                    val gen = generation
                    main.post {
                        if (gen != generation) return@post
                        for (off in bytes.indices step CHUNK_BYTES) {
                            val chunk = bytes.copyOfRange(off, minOf(off + CHUNK_BYTES, bytes.size))
                            pcm.write(chunk)
                            mac?.audio(chunk)
                        }
                        Log.i(TAG, "VOICE_WAV fed_ms=${samples.size / 16}")
                        stop()
                    }
                }
                ACTION_STATUS -> {
                    val imm = getSystemService(INPUT_METHOD_SERVICE) as android.view.inputmethod.InputMethodManager
                    val shortcuts = imm.shortcutInputMethodsAndSubtypes.entries.joinToString { (imi, subtypes) ->
                        imi.id + subtypes.joinToString(prefix = "[", postfix = "]") { it.mode }
                    }
                    Log.i(TAG, "VOICE_STATUS shortcuts=$shortcuts configured=${settings.configured} " +
                        "asr_here=${LocalSpeech.asrDir(this@VoiceIME) != null}")
                }
            }
        }
    }

    override fun onCreate() {
        super.onCreate()
        settings = MacSettings(this)
        worker = Handler(HandlerThread("phonon-voice").apply { start() }.looper)
        if (applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0) {
            // Only a sender holding DUMP (adb's shell, the system) may use them: another app on the phone must not
            // point dictations at a server of its choosing.
            val f = IntentFilter().apply { addAction(ACTION_CONFIG); addAction(ACTION_WAV); addAction(ACTION_STATUS) }
            if (Build.VERSION.SDK_INT >= 33) registerReceiver(debugReceiver, f, Manifest.permission.DUMP, null, RECEIVER_EXPORTED)
            else registerReceiver(debugReceiver, f, Manifest.permission.DUMP, null)
        }
    }

    override fun onDestroy() {
        if (applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0) unregisterReceiver(debugReceiver)
        abandon()
        worker.looper.quitSafely()
        super.onDestroy()
    }

    // ---------- view ----------

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreateInputView(): View {
        val dp = resources.displayMetrics.density
        fun px(v: Int) = (v * dp).toInt()
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.rgb(28, 28, 30))
            setPadding(px(12), px(10), px(12), px(14))
        }
        status = TextView(this).apply {
            setTextColor(Color.LTGRAY); textSize = 13f; gravity = Gravity.CENTER
            contentDescription = "voice_status"
        }
        root.addView(status)
        preview = TextView(this).apply {
            setTextColor(Color.WHITE); textSize = 16f
            setPadding(px(4), px(8), px(4), px(8))
            contentDescription = "voice_preview"
        }
        root.addView(ScrollView(this).apply { addView(preview) },
            LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, px(110)))
        val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER }
        fun button(label: String, description: String, color: Int, weight: Float, onClick: () -> Unit) =
            TextView(this).apply {
                text = label; contentDescription = description; gravity = Gravity.CENTER
                setTextColor(Color.WHITE); textSize = 16f
                background = GradientDrawable().apply { setColor(color); cornerRadius = 10 * dp }
                setOnClickListener { onClick() }
                row.addView(this, LinearLayout.LayoutParams(0, px(52), weight).apply {
                    setMargins(px(4), px(4), px(4), px(4))
                })
            }
        button("✕", "voice_cancel", Color.rgb(72, 72, 74), 1f) { abandon(); leave() }
        retryButton = button("Retry on Mac", "voice_retry", Color.rgb(72, 72, 74), 2f) { retryOnMac() }
        doneButton = button("Done", "voice_done", Color.rgb(10, 132, 255), 3f) { stop() }
        button("⌨", "voice_keyboard", Color.rgb(72, 72, 74), 1f) { abandon(); leave() }
        root.addView(row)
        retryButton?.visibility = View.GONE
        return root
    }

    override fun onStartInputView(info: EditorInfo?, restarting: Boolean) {
        super.onStartInputView(info, restarting)
        if (!active && !leaving) begin(info)
    }

    override fun onFinishInputView(finishingInput: Boolean) {
        // The user went to another field or app: nothing will be typed.
        abandon()
        super.onFinishInputView(finishingInput)
    }

    /** Gone from the screen (the previous keyboard is back): the next time it opens, it listens again. A field that
     *  restarts after the commit, while this window is still up, does not count. */
    override fun onWindowHidden() {
        leaving = false
        super.onWindowHidden()
    }

    private fun setStatus(text: String) { status?.text = text }
    private fun setPreview(text: String) { preview?.text = text }

    // ---------- one dictation ----------

    private fun begin(info: EditorInfo?) {
        val gen = ++generation
        active = true
        stopped = false
        macProblem = null
        pcm.reset()
        setPreview("")
        retryButton?.visibility = View.GONE
        doneButton?.visibility = View.VISIBLE
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            active = false
            doneButton?.visibility = View.GONE
            setStatus("Phonon needs the microphone: open the Phonon app once to allow it")
            return
        }
        val inputType = info?.inputType ?: 0
        // The text around the cursor helps the Mac continue it; never a password field's.
        val textbox = if (MacProtocol.isPassword(inputType)) null else currentInputConnection?.let { ic ->
            Triple(
                ic.getTextBeforeCursor(MacProtocol.TEXTBOX_CHARS, 0)?.toString(),
                ic.getSelectedText(0)?.toString(),
                ic.getTextAfterCursor(MacProtocol.TEXTBOX_CHARS, 0)?.toString(),
            )
        }
        if (settings.configured) {
            mac = MacDictation(settings, info?.packageName, textbox, macListener(gen)).also { it.connect() }
            setStatus("Listening… connecting to the Mac")
        } else {
            macProblem = "no Mac set up"
            setStatus(if (ModelFetch.localEnabled(this)) "Listening… (no Mac set up: this phone transcribes)"
                else "No Mac set up, and on-device fallback is off: open Phonon voice settings")
        }
        Log.i(TAG, "VOICE_BEGIN pkg=${info?.packageName} mac=${settings.configured} textbox=${textbox != null}")
        startRecording(gen)
    }

    private fun macListener(gen: Int) = object : MacDictation.Listener {
        override fun onConnected() {
            if (gen != generation) return
            setStatus(if (stopped) "Correcting on the Mac…" else "Listening… (Mac)")
        }

        override fun onPreview(text: String, corrected: Boolean) {
            if (gen == generation && !stopped) setPreview(text)
        }

        override fun onFinal(text: String, refined: Boolean) {
            if (gen != generation) return
            Log.i(TAG, "VOICE_FINAL source=mac refined=$refined chars=${text.length}")
            mac = null
            deliver(text, if (text.isEmpty()) "Nothing heard" else null)
        }

        override fun onFailed(reason: String) {
            if (gen != generation) return
            Log.i(TAG, "VOICE_MAC_FAILED $reason")
            mac = null
            macProblem = reason
            if (stopped) transcribeHere() else setStatus("Listening… ($reason: this phone transcribes)")
        }
    }

    @SuppressLint("MissingPermission")
    private fun startRecording(gen: Int) {
        recording = true
        recThread = thread(name = "phonon-voice-mic") {
            val min = AudioRecord.getMinBufferSize(MacProtocol.SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT)
            val rec = AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, MacProtocol.SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, maxOf(min, CHUNK_BYTES) * 2)
            if (rec.state != AudioRecord.STATE_INITIALIZED) {
                rec.release()
                main.post { if (gen == generation) { setStatus("The microphone is busy"); abandon() } }
                return@thread
            }
            rec.startRecording()
            val buf = ByteArray(CHUNK_BYTES)
            while (recording) {
                val n = rec.read(buf, 0, buf.size)
                if (n > 0) {
                    val chunk = buf.copyOf(n)
                    main.post {
                        if (gen != generation) return@post
                        pcm.write(chunk)
                        mac?.audio(chunk)
                    }
                }
            }
            rec.stop(); rec.release()
        }
    }

    private fun stopRecording() {
        recording = false
        recThread?.join(500)
        recThread = null
    }

    /** Done: the audio is complete; the Mac (or this phone) makes the text. */
    private fun stop() {
        if (!active || stopped) return
        stopRecording()
        // The mic thread's last chunks are posted to the main thread: let them land before finishing.
        val gen = generation
        main.post {
            if (gen != generation) return@post
            stopped = true
            doneButton?.visibility = View.GONE
            val m = mac
            if (m != null) {
                setStatus("Correcting on the Mac…")
                m.finish()
            } else {
                transcribeHere()
            }
        }
    }

    private fun retryOnMac() {
        if (!settings.configured || pcm.size() == 0) return
        val gen = ++generation
        active = true
        stopped = true
        retryButton?.visibility = View.GONE
        setStatus("Sending to the Mac…")
        val info = currentInputEditorInfo
        mac = MacDictation(settings, info?.packageName, null, macListener(gen)).also {
            it.connect()
            it.audio(pcm.toByteArray())
            it.finish()
        }
    }

    /** The Mac is out of it: Parakeet on this phone, if its files are here. */
    private fun transcribeHere() {
        val gen = generation
        val dir = LocalSpeech.asrDir(this)
        if (dir == null) {
            val why = macProblem ?: "The Mac is unavailable"
            setStatus(if (ModelFetch.localEnabled(this)) "$why. No speech model on this phone: open Phonon to download it."
                else "${why.replaceFirstChar { it.uppercase() }}. On-device fallback is off.")
            retryButton?.visibility = if (settings.configured) View.VISIBLE else View.GONE
            return
        }
        setStatus("Transcribing on this phone (${macProblem ?: "Mac unavailable"})…")
        val bytes = pcm.toByteArray()
        worker.post {
            val t = SystemClock.elapsedRealtime()
            val asr = LocalSpeech.asr(dir)
            val samples = FloatArray(bytes.size / 2) {
                ((bytes[2 * it].toInt() and 0xff) or (bytes[2 * it + 1].toInt() shl 8)).toShort() / 32768f
            }
            val raw = asr?.transcribe(samples).orEmpty()
            val text = if (raw.isEmpty()) "" else LineBreaks.apply(raw, "minimal")
            Log.i(TAG, "VOICE_FINAL source=phone ms=${SystemClock.elapsedRealtime() - t} chars=${text.length} asr=${asr != null}")
            main.post {
                if (gen != generation) return@post
                when {
                    asr == null -> {
                        setStatus("The speech model on this phone failed to load")
                        retryButton?.visibility = if (settings.configured) View.VISIBLE else View.GONE
                    }
                    else -> deliver(text, if (text.isEmpty()) "Nothing heard" else null)
                }
            }
        }
    }

    /** Type [text] into the field and give the keyboard back; [note] shows briefly when nothing is typed. */
    private fun deliver(text: String, note: String?) {
        active = false
        val ic = currentInputConnection
        val info = currentInputEditorInfo
        val insert = MacProtocol.textToInsert(text, info?.inputType ?: 0, ic?.getTextBeforeCursor(1, 0)?.toString())
        if (insert.isNotEmpty() && ic != null) ic.commitText(insert, 1)
        if (note != null && insert.isEmpty()) {
            setStatus(note)
            main.postDelayed({ if (!active) leave() }, 900)
        } else {
            leave()
        }
    }

    /** Stop listening and drop the dictation, typing nothing. */
    private fun abandon() {
        generation++
        active = false
        stopRecording()
        mac?.cancel()
        mac = null
    }

    /** Back to the keyboard the user came from. */
    private fun leave() {
        leaving = true
        if (!switchToPreviousInputMethod()) {
            leaving = false
            val imm = getSystemService(INPUT_METHOD_SERVICE) as android.view.inputmethod.InputMethodManager
            imm.showInputMethodPicker()
        }
    }

    companion object {
        /** 100 ms of 16 kHz PCM16. */
        const val CHUNK_BYTES = 3200
        const val ACTION_CONFIG = "ai.phonon.keyboard.VOICE_CONFIG"
        const val ACTION_WAV = "ai.phonon.keyboard.VOICE_WAV"
        const val ACTION_STATUS = "ai.phonon.keyboard.VOICE_STATUS"
    }
}

/** This phone's own recogniser, shared by every dictation of the process; loaded on first use. */
object LocalSpeech {
    @Volatile private var loaded: Pair<String, Asr?>? = null

    /** The Parakeet directory when its files are complete (sideloaded or downloaded by the keyboard), else null. */
    fun asrDir(context: android.content.Context): String? {
        if (!ModelFetch.localEnabled(context)) return null
        val ext = context.getExternalFilesDir(null)
        if (ext != null && File(ext, "parakeet/encoder.int8.onnx").exists()) return File(ext, "parakeet").path
        val store = ModelStore(File(context.filesDir, "models"))
        return if (store.allReady(Models.ASR)) File(store.root, "parakeet").path else null
    }

    @Synchronized fun asr(dir: String): Asr? {
        loaded?.let { (d, a) -> if (d == dir) return a }
        val a = Asr.load(dir, 4)
        loaded = dir to a
        return a
    }
}
