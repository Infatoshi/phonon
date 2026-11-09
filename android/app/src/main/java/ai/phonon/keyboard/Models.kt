package ai.phonon.keyboard

import android.content.Context
import android.os.SystemClock
import android.util.Log
import java.io.File
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.concurrent.thread

/**
 * The models the keyboard downloads, pinned to Hugging Face commits with their sha256 and size.
 *
 * Parakeet: Parakeet TDT 0.6B v2 with Phonon's N8avg LoRA merged, sherpa-onnx int8 layout, the encoder's weights
 * in a separate encoder.int8.onnx.data file (onnxruntime maps it instead of reading it all into the heap).
 * Refiner: the same GGUF the Mac and Windows builds use.
 */
object Models {
    private const val ASR_BASE =
        "https://huggingface.co/Infatoshi/phonon-asr-parakeet-tdt-0.6b-v2-onnx/resolve/76fbda42bfcf7be26cca2488f8fe38ea4328f755/"
    private const val REFINER_URL =
        "https://huggingface.co/Infatoshi/phonon-refiner-1.7b/resolve/b20b28075046af94056ab5d5bff89a7abc0f6f50/" +
            "phonon-refiner-1.7b-Q4_K_S-e4.gguf"

    val ASR = listOf(
        ModelFile("parakeet/encoder.int8.onnx", ASR_BASE + "encoder.int8.onnx",
            "5c0a15da8929acd0ddd559471a201bd8ede3bbe2a7f2df6f606d68b8b73f52bb", 42_405_802L),
        ModelFile("parakeet/encoder.int8.onnx.data", ASR_BASE + "encoder.int8.onnx.data",
            "f91c1c8d1c8269c9d21d4d23e7590b924243ce6036a1ff787e0e1beb515dabca", 609_820_416L),
        ModelFile("parakeet/decoder.int8.onnx", ASR_BASE + "decoder.int8.onnx",
            "b6bb64963457237b900e496ee9994b59294526439fbcc1fecf705b31a15c6b4e", 7_257_753L),
        ModelFile("parakeet/joiner.int8.onnx", ASR_BASE + "joiner.int8.onnx",
            "7946164367946e7f9f29a122407c3252b680dbae9a51343eb2488d057c3c43d2", 1_739_080L),
        ModelFile("parakeet/tokens.txt", ASR_BASE + "tokens.txt",
            "ec182b70dd42113aff6c5372c75cac58c952443eb22322f57bbd7f53977d497d", 9_384L),
    )
    val REFINER = ModelFile("refiner.gguf", REFINER_URL,
        "0439996bda5886b92c86db49249be7d7ad2175ccf35bf2630c09433684e502e5", 988_453_696L)

    /** The refiner loads only on phones with MemTotal >= 4000 MB (refiner_jni.cpp's RAM gate); below that the
     *  988 MB download would never be used. The gate's MemAvailable half is checked at load, not here. */
    const val REFINER_MIN_TOTAL_MB = 4000L

    fun memTotalMb(): Long = File("/proc/meminfo").useLines { lines ->
        lines.firstOrNull { it.startsWith("MemTotal:") }?.split(Regex("\\s+"))?.get(1)?.toLongOrNull()?.div(1024) ?: 0L
    }
}

/**
 * Fetches the models once per process, for whichever comes first: the launcher activity or the keyboard. The
 * recogniser comes first, so dictation works (raw transcript) before the larger refiner arrives.
 *
 * Models sideloaded into the app's external files dir (parakeet/encoder.int8.onnx there, as the Test Lab tests
 * stage them) take precedence and nothing is downloaded.
 */
object ModelFetch {
    data class State(
        val asrDir: String? = null,      // set when the recogniser's files are ready
        val refinerPath: String? = null, // set when the refiner is ready
        val refinerSkipped: Boolean = false, // below the RAM gate: never downloaded
        val message: String = "preparing models...",
        val done: Boolean = false,
        val error: String? = null,
    )

    @Volatile var state = State()
        private set
    private val listeners = CopyOnWriteArrayList<(State) -> Unit>()
    private var worker: Thread? = null

    fun addListener(l: (State) -> Unit) { listeners += l; l(state) }
    fun removeListener(l: (State) -> Unit) { listeners -= l }

    private fun publish(s: State) {
        state = s
        for (l in listeners) l(s)
    }

    /** Start (or restart after a failure) the download; a running one is left alone. */
    @Synchronized fun start(ctx: Context) {
        if (worker?.isAlive == true || state.done) return
        val app = ctx.applicationContext
        if (state.error != null) publish(state.copy(error = null, message = "retrying the download..."))
        worker = thread(name = "phonon-download") { run(app) }
    }

    private fun run(ctx: Context) {
        val ext = ctx.getExternalFilesDir(null)
        if (ext != null && File(ext, "parakeet/encoder.int8.onnx").exists()) {
            val gguf = File(ext, "refiner.gguf")
            Log.i(TAG, "MODELS source=sideload dir=${ext.path} refiner=${gguf.exists()}")
            publish(State(asrDir = File(ext, "parakeet").path, refinerPath = gguf.takeIf { it.exists() }?.path,
                refinerSkipped = !gguf.exists(), message = "models ready", done = true))
            return
        }
        val store = ModelStore(File(ctx.filesDir, "models"), log = { Log.i(TAG, it) })
        val memMb = Models.memTotalMb()
        val wantRefiner = memMb >= Models.REFINER_MIN_TOTAL_MB
        val all = Models.ASR + if (wantRefiner) listOf(Models.REFINER) else emptyList()
        val t0 = SystemClock.elapsedRealtime()
        Log.i(TAG, "MODELS source=download dir=${store.root.path} memtotal_mb=$memMb refiner=$wantRefiner " +
            "remaining_mb=${store.remaining(all) shr 20}")
        try {
            var lastPct = -1
            val report = { label: String -> { done: Long, total: Long ->
                val pct = (done * 100 / maxOf(total, 1)).toInt()
                if (pct != lastPct) {
                    lastPct = pct
                    publish(state.copy(message = "downloading $label $pct% (${done shr 20} of ${total shr 20} MB)"))
                }
            } }
            store.ensure(Models.ASR, progress = report("speech model"))
            val asrDir = File(store.root, "parakeet").path
            Log.i(TAG, "MODELS asr_ready ms=${SystemClock.elapsedRealtime() - t0}")
            publish(state.copy(asrDir = asrDir, message = "speech model ready"))
            if (wantRefiner) {
                lastPct = -1
                store.ensure(listOf(Models.REFINER), progress = report("text model"))
                Log.i(TAG, "MODELS refiner_ready ms=${SystemClock.elapsedRealtime() - t0}")
                publish(state.copy(refinerPath = store.file(Models.REFINER).path, message = "models ready", done = true))
            } else {
                publish(state.copy(refinerSkipped = true, message = "models ready", done = true))
            }
        } catch (e: Exception) {
            Log.e(TAG, "MODELS failed: ${e.message}")
            publish(state.copy(message = "download failed: ${e.message} (open the keyboard again to retry)", error = e.message))
        }
    }
}
