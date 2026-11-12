package ai.phonon.keyboard

import android.content.Context

/** Dark status and navigation bar icons when a DayNight activity is light (they were white on the light
 *  background), light icons in dark mode. */
@Suppress("DEPRECATION")
fun android.app.Activity.lightSystemBars() {
    val dark = (resources.configuration.uiMode and android.content.res.Configuration.UI_MODE_NIGHT_MASK) ==
        android.content.res.Configuration.UI_MODE_NIGHT_YES
    if (android.os.Build.VERSION.SDK_INT >= 30) {
        val both = android.view.WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS or
            android.view.WindowInsetsController.APPEARANCE_LIGHT_NAVIGATION_BARS
        window.insetsController?.setSystemBarsAppearance(if (dark) 0 else both, both)
    } else {
        val flags = android.view.View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR or android.view.View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR
        val v = window.decorView
        v.systemUiVisibility = if (dark) v.systemUiVisibility and flags.inv() else v.systemUiVisibility or flags
    }
}

/** The writing modes the refiner was trained with, as the iPhone's WritingMode (ios/Shared/Theme.swift): same keys,
 *  titles and descriptions. The choice is stored under the key "mode" (the iPhone's IPC.modeKey); default minimal. */
object WritingModes {
    data class Mode(val key: String, val title: String, val blurb: String)

    val ALL = listOf(
        Mode("minimal", "Minimal", "Notes and documents. Your words, cleaned up."),
        Mode("casual", "Casual", "Chat messages. Relaxed, lowercase is fine."),
        Mode("formal", "Formal", "Email and documents. Full sentences."),
        Mode("prompt", "Prompt", "Instructions to an AI. Clear and direct."),
    )

    private const val PREFS = "phonon"
    const val KEY = "mode"

    fun get(c: Context): String {
        val m = c.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString(KEY, null)
        return if (ALL.any { it.key == m }) m!! else "minimal"
    }

    fun set(c: Context, key: String) {
        c.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putString(KEY, key).apply()
    }
}
