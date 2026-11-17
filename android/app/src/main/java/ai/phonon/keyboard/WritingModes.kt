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
 *  resolved automatically from the focused app's package name. */
object WritingModes {
    fun forPackage(packageName: String?): String {
        val pkg = packageName?.trim()?.lowercase()?.takeIf { it.isNotEmpty() } ?: return "minimal"
        return when {
            pkg in PROMPT_PACKAGES -> "prompt"
            pkg in FORMAL_PACKAGES -> "formal"
            pkg in CASUAL_PACKAGES -> "casual"
            else -> "minimal"
        }
    }

    private val CASUAL_PACKAGES = setOf(
        "com.google.android.apps.messaging",
        "com.samsung.android.messaging",
        "com.whatsapp",
        "com.whatsapp.w4b",
        "com.slack",
        "com.discord",
        "org.telegram.messenger",
        "org.thoughtcrime.securesms",
        "com.facebook.orca",
        "com.microsoft.teams",
        "com.instagram.android",
        "com.beeper.android",
        "jp.naver.line.android",
    )

    private val FORMAL_PACKAGES = setOf(
        "com.google.android.gm",
        "com.microsoft.office.outlook",
        "com.readdle.spark",
        "ch.protonmail.android",
        "com.superhuman.mail",
        "com.yahoo.mobile.client.android.mail",
    )

    private val PROMPT_PACKAGES = setOf(
        "com.openai.chatgpt",
        "com.anthropic.claude",
        "ai.perplexity.app.android",
        "com.google.android.apps.bard",
        "com.microsoft.copilot",
        "com.termux",
    )
}
