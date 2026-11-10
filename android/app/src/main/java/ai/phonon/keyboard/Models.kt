package ai.phonon.keyboard

import android.content.Context
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.os.Build
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
 * The refiner (988 MB) downloads only on an unmetered network unless the user allows mobile data once (one tap on
 * the keyboard's status line or the launcher's button, remembered); the recogniser downloads on any network.
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
        val waitingForWifi: Boolean = false, // the refiner waits for an unmetered network or the user's override
    )

    @Volatile var state = State()
        private set
    private val listeners = CopyOnWriteArrayList<(State) -> Unit>()
    private var worker: Thread? = null
    private val gate = NetworkGate()
    private var watching = false
    private const val PREFS = "phonon"
    private const val CELLULAR_KEY = "refiner_on_cellular"

    /** Tests only: treat every network as metered (true) or unmetered (false) instead of asking Android. */
    @Volatile var forceMetered: Boolean? = null

    /** The user's one tap: download the refiner on mobile data from now on. */
    fun allowCellular(ctx: Context) {
        ctx.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putBoolean(CELLULAR_KEY, true).apply()
        Log.i(TAG, "MODELS cellular_allowed=1")
        gate.allowCellular()
    }

    /** Feeds [gate] from the default network's NOT_METERED capability (or [forceMetered]). */
    @Synchronized private fun watchNetwork(ctx: Context) {
        if (ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getBoolean(CELLULAR_KEY, false)) gate.allowCellular()
        forceMetered?.let { gate.setUnmetered(!it); Log.i(TAG, "MODELS network forced unmetered=${!it}"); return }
        if (watching) return
        val cm = ctx.getSystemService(ConnectivityManager::class.java)
        fun unmetered(c: NetworkCapabilities?) = c != null && (c.hasCapability(NetworkCapabilities.NET_CAPABILITY_NOT_METERED) ||
            (Build.VERSION.SDK_INT >= 30 && c.hasCapability(NetworkCapabilities.NET_CAPABILITY_TEMPORARILY_NOT_METERED)))
        gate.setUnmetered(unmetered(cm.getNetworkCapabilities(cm.activeNetwork)))
        cm.registerDefaultNetworkCallback(object : ConnectivityManager.NetworkCallback() {
            override fun onCapabilitiesChanged(n: Network, c: NetworkCapabilities) {
                val u = unmetered(c)
                if (u != last) Log.i(TAG, "MODELS network unmetered=$u")
                last = u
                gate.setUnmetered(u)
            }
            var last = gate.allowed()
            override fun onLost(n: Network) { last = false; gate.setUnmetered(false) }
        })
        watching = true  // only once registered: a failure above is retried on the next start
        Log.i(TAG, "MODELS network unmetered=${gate.allowed()} (or cellular allowed)")
    }

    fun addListener(l: (State) -> Unit) { listeners += l; l(state) }
    fun removeListener(l: (State) -> Unit) { listeners -= l }

    private fun publish(s: State) {
        state = s
        for (l in listeners) l(s)
    }

    private const val LOCAL_KEY = "local_models"
    /** The user turned the on-device models off: the download stops and nothing is fetched. */
    @Volatile private var stopRequested = false

    /** Setting "Fall back to the on-device model" (default on). Off: no on-device models are kept or fetched. */
    fun localEnabled(ctx: Context): Boolean =
        ctx.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getBoolean(LOCAL_KEY, true)

    /** Turn the on-device models on again and fetch them. */
    fun enableLocal(ctx: Context) {
        ctx.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putBoolean(LOCAL_KEY, true).apply()
        synchronized(this) {
            stopRequested = false
            if (!state.done) publish(State())
        }
        start(ctx)
    }

    /**
     * Turn the on-device models off: stop a download in progress and delete every model file this app holds
     * (downloaded and sideloaded). Blocking (it waits for the download thread): call it off the main thread.
     * Engines already loaded keep their memory until the process ends. Returns the bytes freed.
     */
    fun disableLocal(ctx: Context): Long {
        val app = ctx.applicationContext
        app.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putBoolean(LOCAL_KEY, false).commit()
        val running = synchronized(this) {
            stopRequested = true
            worker
        }
        running?.interrupt()
        running?.join(15_000)
        val targets = listOfNotNull(File(app.filesDir, "models"),
            app.getExternalFilesDir(null)?.let { File(it, "parakeet") },
            app.getExternalFilesDir(null)?.let { File(it, "refiner.gguf") })
        val freed = targets.sumOf { t -> t.walkBottomUp().filter { it.isFile }.sumOf { it.length() } }
        targets.forEach { it.deleteRecursively() }
        Log.i(TAG, "MODELS local_off freed_mb=${freed shr 20}")
        synchronized(this) { publish(State(message = OFF_MESSAGE)) }
        return freed
    }

    const val OFF_MESSAGE = "on-device models are off (Phonon voice settings)"

    /** Start (or restart after a failure) the download; a running one is left alone. */
    @Synchronized fun start(ctx: Context) {
        if (!localEnabled(ctx)) {
            if (state.message != OFF_MESSAGE) publish(State(message = OFF_MESSAGE))
            return
        }
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
            store.ensure(Models.ASR, allowed = { !stopRequested }, progress = report("speech model"))
            val asrDir = File(store.root, "parakeet").path
            Log.i(TAG, "MODELS asr_ready ms=${SystemClock.elapsedRealtime() - t0}")
            publish(state.copy(asrDir = asrDir, message = "speech model ready"))
            if (wantRefiner) {
                lastPct = -1
                watchNetwork(ctx)
                val textReport = report("text model")
                store.ensureWhenAllowed(listOf(Models.REFINER), gate, onWaiting = {
                    Log.i(TAG, "MODELS refiner_wait_unmetered")
                    publish(state.copy(waitingForWifi = true,
                        message = "text model waits for Wi-Fi; tap here to download it on mobile data (988 MB)"))
                }, onAllowed = {
                    publish(state.copy(waitingForWifi = false, message = "downloading text model..."))
                }, progress = textReport, stop = { stopRequested })
                Log.i(TAG, "MODELS refiner_ready ms=${SystemClock.elapsedRealtime() - t0}")
                publish(state.copy(refinerPath = store.file(Models.REFINER).path, message = "models ready", done = true))
            } else {
                publish(state.copy(refinerSkipped = true, message = "models ready", done = true))
            }
        } catch (e: Exception) {
            if (stopRequested) {  // the user turned the models off: disableLocal says so
                Log.i(TAG, "MODELS stopped")
                return
            }
            Log.e(TAG, "MODELS failed: ${e.message}")
            publish(state.copy(message = "download failed: ${e.message} (open the keyboard again to retry)", error = e.message,
                waitingForWifi = false))
        }
    }
}
