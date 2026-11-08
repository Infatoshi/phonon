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
    fun refine(raw: String, mode: String): String = String(nativeRefine(handle, raw, mode), Charsets.UTF_8)

    fun free() {
        if (handle != 0L) nativeFree(handle)
        handle = 0L
    }

    companion object {
        init { System.loadLibrary("phonon_refiner") }

        @JvmStatic external fun nativeLoad(path: String, threads: Int, repack: Boolean, autoTune: Boolean,
                                           tunePath: String?, spec: String?): Long
        @JvmStatic external fun nativeRefine(handle: Long, raw: String, mode: String): ByteArray
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

/** sherpa-onnx with NeMo Parakeet TDT 0.6B v2 int8. */
class Asr private constructor(private val rec: OfflineRecognizer) {
    fun transcribe(samples: FloatArray): String {
        val s = rec.createStream()
        s.acceptWaveform(samples, 16000)
        rec.decode(s)
        val text = rec.getResult(s).text.trim()
        s.release()
        return text
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
