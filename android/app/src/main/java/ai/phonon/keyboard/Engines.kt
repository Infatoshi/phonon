package ai.phonon.keyboard

import android.os.Debug
import android.os.SystemClock
import android.util.Log
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.OfflineTransducerModelConfig
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

const val TAG = "Phonon"

object Mem {
    /** VmRSS and VmHWM of this process in MB, plus the native heap. */
    fun snapshot(): String {
        var rss = "?"
        var hwm = "?"
        File("/proc/self/status").forEachLine {
            if (it.startsWith("VmRSS:")) rss = it.split(Regex("\\s+"))[1]
            if (it.startsWith("VmHWM:")) hwm = it.split(Regex("\\s+"))[1]
        }
        val heap = Debug.getNativeHeapAllocatedSize() / (1 shl 20)
        return "rss_mb=${rss.toLongOrNull()?.div(1024)} hwm_mb=${hwm.toLongOrNull()?.div(1024)} native_heap_mb=$heap"
    }
}

/** The JNI side is app/src/main/cpp/refiner_jni.cpp, a port of ios/App/Refiner.swift. */
class Refiner private constructor(private var handle: Long) {
    /** [vocab]: the dictionary's vocabulary line terms ("A, B, C"), or "" for none (the prompt is then unchanged). */
    fun refine(raw: String, mode: String, vocab: String = ""): String =
        String(nativeRefine(handle, raw, mode, vocab), Charsets.UTF_8)

    /** Prefills and caches the prompt head of [mode]; ms taken, 0 if it was already cached, -1 on failure. */
    fun prime(mode: String, vocab: String = ""): Long = nativePrime(handle, mode, vocab)

    /**
     * The keyboard's warm-up after a load, also run by the benches: one short refine in minimal mode (pages in the
     * weights, as on iOS), then the prompt head of every other writing mode, so no dictation re-prefills its head
     * after a mode switch (GOAL 13.8). Returns the minimal refine's JSON.
     */
    fun warmUp(): String {
        val json = refine("warm up", "minimal")
        for (m in MODES) if (m != "minimal") prime(m)
        return json
    }

    fun free() {
        if (handle != 0L) nativeFree(handle)
        handle = 0L
    }

    companion object {
        init { System.loadLibrary("phonon_refiner") }

        @JvmStatic external fun nativeLoad(path: String, threads: Int, repack: Boolean, autoTune: Boolean,
                                           tunePath: String?, spec: String?): Long
        /** The writing modes the refiner was trained with (refiner_jni.cpp modeLine). */
        val MODES = listOf("minimal", "casual", "formal", "prompt")

        @JvmStatic external fun nativeRefine(handle: Long, raw: String, mode: String, vocab: String): ByteArray
        @JvmStatic external fun nativePrime(handle: Long, mode: String, vocab: String): Long
        @JvmStatic external fun nativeFree(handle: Long)
        @JvmStatic external fun nativeCpuMs(): Long

        /** The pre-tuning path (no threadpool, `threads` unpinned, no RAM gate); kept for the before/after bench. */
        fun load(path: String, threads: Int, repack: Boolean = true): Refiner? {
            val t = SystemClock.elapsedRealtime()
            val h = nativeLoad(path, threads, repack, false, null, null)
            Log.i(TAG, "REFINER_LOAD ok=${h != 0L} ms=${SystemClock.elapsedRealtime() - t} threads=$threads repack=$repack ${Mem.snapshot()}")
            return if (h != 0L) Refiner(h) else null
        }

        /**
         * The keyboard's path: RAM gate (null below ~4 GB or with < 1.5 GB available: commit the raw transcript),
         * a persistent threadpool pinned to the top-K cores, K probed at first load and cached in [tunePath].
         * [spec] holds bench overrides ("" in the keyboard).
         */
        fun loadAuto(path: String, tunePath: String, spec: String = ""): Refiner? {
            val t = SystemClock.elapsedRealtime()
            val h = nativeLoad(path, 0, true, true, tunePath, spec)
            Log.i(TAG, "REFINER_LOAD ok=${h != 0L} ms=${SystemClock.elapsedRealtime() - t} auto=1 spec=$spec ${Mem.snapshot()}")
            return if (h != 0L) Refiner(h) else null
        }
    }
}

/**
 * Long recordings go to Parakeet in passes. The exported int8 encoder takes at most 400 s (its position table is
 * 5,000 frames of 80 ms; 400.2 s throws), and long single passes also lose words: LibriSpeech test-clean, 915 s,
 * one pass per <= L s: WER 1.31 (30 s), 1.35 (60 s), 2.03 (120 s), 2.92 (240 s), 2.83 (360 s); 1.35 one utterance
 * at a time (x86 CPU, 2026-11-08). Same rule as the desktop build (crates/phonon-win/src/asr.rs `passes`).
 */
object Passes {
    const val MAX_SECONDS = 60

    /** [from, to) ranges of at most MAX_SECONDS covering [n] samples, each cut at the quietest 100 ms in the last
     *  quarter of its window ([energy] gives a range's energy). */
    fun split(n: Int, rate: Int, energy: (Int, Int) -> Double): List<Pair<Int, Int>> {
        val max = MAX_SECONDS * rate
        val frame = rate / 10
        val search = max / 4
        val out = mutableListOf<Pair<Int, Int>>()
        var start = 0
        while (n - start > max) {
            val limit = start + max
            var cut = limit
            var quietest = Double.MAX_VALUE
            var at = limit - search
            while (at + frame <= limit) {
                val e = energy(at, at + frame)
                if (e < quietest) { quietest = e; cut = at + frame / 2 }
                at += frame
            }
            out += start to cut
            start = cut
        }
        out += start to n
        return out
    }
}

/** sherpa-onnx with NeMo Parakeet TDT 0.6B v2 int8. */
class Asr private constructor(private val rec: OfflineRecognizer) {
    /** 16 kHz mono samples to text, in passes of at most [Passes.MAX_SECONDS] (see [Passes]). */
    fun transcribe(samples: FloatArray): String =
        Passes.split(samples.size, 16000) { from, to -> energy(samples, from, to) }
            .map { (from, to) -> pass(samples.copyOfRange(from, to)) }
            .filter { it.isNotEmpty() }
            .joinToString(" ")

    private fun pass(samples: FloatArray): String {
        val s = rec.createStream()
        s.acceptWaveform(samples, 16000)
        rec.decode(s)
        val text = rec.getResult(s).text.trim()
        s.release()
        return text
    }

    private fun energy(samples: FloatArray, from: Int, to: Int): Double {
        var e = 0.0
        for (i in from until to) e += samples[i].toDouble() * samples[i]
        return e
    }

    companion object {
        fun load(dir: String, threads: Int): Asr? {
            val t = SystemClock.elapsedRealtime()
            return try {
                val cfg = OfflineRecognizerConfig(
                    featConfig = FeatureConfig(sampleRate = 16000, featureDim = 128),
                    modelConfig = OfflineModelConfig(
                        transducer = OfflineTransducerModelConfig(
                            encoder = "$dir/encoder.int8.onnx",
                            decoder = "$dir/decoder.int8.onnx",
                            joiner = "$dir/joiner.int8.onnx",
                        ),
                        tokens = "$dir/tokens.txt",
                        modelType = "nemo_transducer",
                        numThreads = threads,
                    ),
                )
                val a = Asr(OfflineRecognizer(null, cfg))
                Log.i(TAG, "ASR_LOAD ok=true ms=${SystemClock.elapsedRealtime() - t} ${Mem.snapshot()}")
                a
            } catch (e: Throwable) {
                Log.e(TAG, "ASR_LOAD ok=false", e)
                null
            }
        }
    }
}

object Wav {
    /** 16 kHz mono PCM16 WAV to floats in [-1, 1]. */
    fun read(path: String): FloatArray {
        val bytes = RandomAccessFile(path, "r").use { f -> ByteArray(f.length().toInt()).also { f.readFully(it) } }
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        var pos = 12
        var rate = 0
        var channels = 0
        while (pos + 8 <= bytes.size) {
            val id = String(bytes, pos, 4, Charsets.US_ASCII)
            val len = bb.getInt(pos + 4)
            if (id == "fmt ") {
                channels = bb.getShort(pos + 10).toInt()
                rate = bb.getInt(pos + 12)
            } else if (id == "data") {
                require(rate == 16000 && channels == 1) { "need 16 kHz mono, got $rate Hz x$channels" }
                val n = minOf(len, bytes.size - pos - 8) / 2
                return FloatArray(n) { bb.getShort(pos + 8 + 2 * it) / 32768f }
            }
            pos += 8 + len + (len and 1)
        }
        error("no data chunk in $path")
    }
}
