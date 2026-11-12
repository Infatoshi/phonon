package ai.phonon.keyboard

import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File

/**
 * The refiner a device test measures must be the one the keyboard downloads (Models.REFINER, pinned by sha256). On
 * 2026-11-12 the Test Lab input refiner.gguf turned out to be an older model of the same size, so every lab refiner
 * number before then ran a model no user gets. [check] logs the staged file's sha256 (logcat tag Phonon, "MODEL") and
 * fails the test when it is not the pin, unless the instrumentation arg `model_any=1` says another model is intended.
 */
object ModelPin {
    fun check(file: File, label: String = "refiner") {
        val t = SystemClock.elapsedRealtime()
        val sha = ModelStore.sha256(file)
        val pin = Models.REFINER.sha256
        val any = InstrumentationRegistry.getArguments().getString("model_any") == "1"
        Log.i(TAG, "MODEL $label path=${file.name} bytes=${file.length()} sha256=$sha pinned=${sha == pin} " +
            "model_any=$any ms=${SystemClock.elapsedRealtime() - t}")
        if (sha != pin && !any)
            throw AssertionError("$label ${file.name} sha256 $sha is not the shipped refiner $pin (pass model_any=1 to run it anyway)")
    }

    /** Logs the sha256 of a file that is deliberately not the pinned model (e.g. the shortlist build). */
    fun log(file: File, label: String) {
        Log.i(TAG, "MODEL $label path=${file.name} bytes=${file.length()} sha256=${ModelStore.sha256(file)}")
    }

    private const val TAG = "Phonon"
}
