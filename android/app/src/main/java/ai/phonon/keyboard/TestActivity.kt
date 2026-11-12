package ai.phonon.keyboard

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.provider.Settings
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

/**
 * The app's home, the iPhone app's wording and structure (ios/App/PhononApp.swift ContentView): the Phonon header;
 * "Set up the keyboard" until the keyboard is on and chosen, the mic allowed and the speech model in place; a field to
 * try the keyboard in (the iPhone's session card has no Android counterpart: this keyboard records the mic itself);
 * the writing mode; the dictionary; the models' status. In the Mac's ember palette when the phone is in dark mode, a
 * light version of it otherwise. It asks for the mic once (a keyboard cannot prompt) and starts the model download.
 */
class TestActivity : Activity() {
    private lateinit var c: Palette
    private var models: TextView? = null
    private var cellular: Button? = null
    private lateinit var setupCard: View
    private lateinit var setupRows: LinearLayout
    private lateinit var modeRows: LinearLayout
    private lateinit var dictTitle: TextView
    private val dp get() = resources.displayMetrics.density

    private val onModels: (ModelFetch.State) -> Unit = { s ->
        runOnUiThread {
            models?.text = s.message
            cellular?.visibility = if (s.waitingForWifi) View.VISIBLE else View.GONE
            refreshSetup()
        }
    }

    /** The Mac's ember palette (ios/Shared/Theme.swift Ember) and a light version of it. */
    private class Palette(dark: Boolean) {
        val window = if (dark) 0xFF12100E.toInt() else 0xFFF7F3EE.toInt()
        val panel = if (dark) 0xFF1A1512.toInt() else 0xFFFFFFFF.toInt()
        val raised = if (dark) 0xFF231C17.toInt() else 0xFFF0E9E2.toInt()
        val line = if (dark) 0xFF2A221C.toInt() else 0xFFE6DCD2.toInt()
        val ink = if (dark) 0xFFEEE6DC.toInt() else 0xFF1E1914.toInt()
        val soft = if (dark) 0xFFCDBFB2.toInt() else 0xFF4A4038.toInt()
        val mute = if (dark) 0xFF8F8377.toInt() else 0xFF7A6E63.toInt()
        val dim = if (dark) 0xFF6F6559.toInt() else 0xFF948A80.toInt()
        val ember = if (dark) 0xFFE8692C.toInt() else 0xFFD2561A.toInt()
        val ember2 = if (dark) 0xFFF08A4B.toInt() else 0xFFC04A12.toInt()
        val done = if (dark) 0xFF8FE0A8.toInt() else 0xFF2E8B4E.toInt()
    }

    override fun onCreate(b: Bundle?) {
        super.onCreate(b)
        actionBar?.hide()
        c = Palette((resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK) == Configuration.UI_MODE_NIGHT_YES)
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1)
        }
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(px(18), px(8), px(18), px(32))
        }
        root.addView(header())

        setupRows = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        setupCard = card("Set up the keyboard", setupRows)
        root.addView(setupCard, gap())

        // a field to try the keyboard in; ImeE2ETest adds a single-line field next to it in this same LinearLayout
        val tryRows = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        tryRows.addView(EditText(this).apply {
            id = R.id.field
            hint = "Type, or hold the bar to talk"
            minLines = 3
            gravity = Gravity.TOP or Gravity.START
            setTextColor(c.ink); setHintTextColor(c.dim); textSize = 15f
            background = rounded(c.raised, 10f)
            setPadding(px(12), px(10), px(12), px(10))
        })
        tryRows.addView(TextView(this).apply {
            text = "Not the Phonon keyboard? Switch keyboards"
            setTextColor(c.ember2); textSize = 13f; typeface = Typeface.DEFAULT_BOLD
            setPadding(0, px(10), 0, px(2))
            setOnClickListener { (getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager).showInputMethodPicker() }
        })
        root.addView(card("Try it", tryRows), gap())

        modeRows = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; id = R.id.modes }
        root.addView(card("Writing mode", modeRows), gap())
        renderModes()

        root.addView(card("Dictionary", dictionaryRow()), gap())

        // the models' status (the iPhone's footer), with the mobile-data override while the text model waits for Wi-Fi
        root.addView(LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(px(4), 0, px(4), 0)
            models = TextView(this@TestActivity).apply {
                id = R.id.models
                typeface = Typeface.MONOSPACE; textSize = 11f; setTextColor(c.dim)
            }
            addView(models)
            cellular = Button(this@TestActivity).apply {
                id = R.id.cellular
                text = "Download the text model on mobile data (988 MB)"
                isAllCaps = false
                visibility = View.GONE
                setOnClickListener { ModelFetch.allowCellular(this@TestActivity) }
            }
            addView(cellular)
        }, gap())

        val scroll = ScrollView(this).apply { addView(root); setBackgroundColor(c.window) }
        // targetSdk 35 draws edge to edge, and without the inset frame the first rows sat under the status bar
        setContentView(FrameLayout(this).apply {
            fitsSystemWindows = true
            setBackgroundColor(c.window)
            addView(scroll)
        })
        window.decorView.setBackgroundColor(c.window)
        lightSystemBars()
        ModelFetch.addListener(onModels)
        ModelFetch.start(this)
    }

    override fun onResume() {
        super.onResume()
        refreshSetup()
        refreshDictionary()
    }

    /** The keyboard picker closes without pausing this activity: re-check when focus comes back. */
    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) refreshSetup()
    }

    override fun onRequestPermissionsResult(code: Int, perms: Array<out String>, results: IntArray) {
        super.onRequestPermissionsResult(code, perms, results)
        refreshSetup()
    }

    override fun onDestroy() {
        ModelFetch.removeListener(onModels)
        super.onDestroy()
    }

    // ---------- header ----------

    private fun header() = LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
        setPadding(0, px(8), 0, 0)
        addView(TextView(this@TestActivity).apply {
            text = "ϕ"; textSize = 30f; setTextColor(c.ember); typeface = Typeface.DEFAULT_BOLD
        })
        addView(TextView(this@TestActivity).apply {
            text = "Phonon"; textSize = 30f; setTextColor(c.ink); typeface = Typeface.DEFAULT_BOLD
            setPadding(px(8), 0, 0, 0)
        })
        addView(View(this@TestActivity), LinearLayout.LayoutParams(0, 1, 1f))
        addView(TextView(this@TestActivity).apply {
            text = "on-device"; textSize = 12f; setTextColor(c.mute); typeface = Typeface.MONOSPACE
            setPadding(px(8), px(4), px(8), px(4))
            background = GradientDrawable().apply { cornerRadius = 100 * dp; setStroke(px(1), c.line) }
        })
    }

    // ---------- setup ----------

    private fun keyboardOn(): Boolean =
        (getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager).enabledInputMethodList.any { it.packageName == packageName }

    private fun keyboardChosen(): Boolean =
        Settings.Secure.getString(contentResolver, Settings.Secure.DEFAULT_INPUT_METHOD)?.startsWith("$packageName/") == true

    private fun micAllowed() = checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED

    private fun refreshSetup() {
        if (!::setupRows.isInitialized) return
        val on = keyboardOn()
        val chosen = keyboardChosen()
        val mic = micAllowed()
        val s = ModelFetch.state
        val speech = s.asrDir != null
        setupCard.visibility = if (on && chosen && mic && speech) View.GONE else View.VISIBLE
        setupRows.removeAllViews()
        setupRows.addView(setupRow(1, "Add the Phonon keyboard", "Settings > Keyboard > On-screen keyboard", on,
            if (on) null else "Open Settings" to { startActivity(Intent(Settings.ACTION_INPUT_METHOD_SETTINGS)) }))
        setupRows.addView(setupRow(2, "Switch to Phonon", "Pick it as the keyboard you type with.", chosen,
            if (chosen || !on) null else "Choose" to {
                (getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager).showInputMethodPicker()
            }))
        setupRows.addView(setupRow(3, "Allow the microphone", "The keyboard hears you only while you hold or tap the bar. Nothing leaves your phone.",
            mic, if (mic) null else "Allow" to { micAction() }))
        setupRows.addView(setupRow(4, "Speech model", if (speech) "Parakeet, on device" else s.message, speech, null))
    }

    /** Asks again while Android still shows the prompt; after "Don't allow" twice only the app's settings can. */
    private fun micAction() {
        if (shouldShowRequestPermissionRationale(Manifest.permission.RECORD_AUDIO) || !asked) {
            asked = true
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1)
        } else {
            startActivity(Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, android.net.Uri.fromParts("package", packageName, null)))
        }
    }
    private var asked = false

    private fun setupRow(n: Int, title: String, detail: String, done: Boolean, action: Pair<String, () -> Unit>?) =
        LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(0, px(3), 0, px(3))
            addView(TextView(this@TestActivity).apply {
                text = if (done) "✓" else "$n"
                gravity = Gravity.CENTER; textSize = 12f; typeface = Typeface.DEFAULT_BOLD
                setTextColor(if (done) c.done else c.soft)
                background = GradientDrawable().apply {
                    shape = GradientDrawable.OVAL
                    setColor(if (done) (c.done and 0x00FFFFFF) or 0x2E000000 else c.raised)
                }
            }, LinearLayout.LayoutParams(px(24), px(24)).apply { marginEnd = px(12); topMargin = px(1) })
            addView(LinearLayout(this@TestActivity).apply {
                orientation = LinearLayout.VERTICAL
                addView(text(title, 15f, if (done) c.mute else c.ink, bold = true))
                addView(text(detail, 12f, c.mute))
            }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            if (action != null) addView(TextView(this@TestActivity).apply {
                text = action.first; textSize = 13f; typeface = Typeface.DEFAULT_BOLD; setTextColor(c.ember2)
                setPadding(px(8), px(2), 0, px(8))
                setOnClickListener { action.second() }
            })
        }

    // ---------- writing mode ----------

    private fun renderModes() {
        modeRows.removeAllViews()
        val current = WritingModes.get(this)
        for ((i, m) in WritingModes.ALL.withIndex()) {
            val on = m.key == current
            modeRows.addView(LinearLayout(this).apply {
                id = 1000 + i
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(0, px(6), 0, px(6))
                isSelected = on
                contentDescription = "${m.title}. ${m.blurb}${if (on) " Selected." else ""}"
                addView(LinearLayout(this@TestActivity).apply {
                    orientation = LinearLayout.VERTICAL
                    addView(text(m.title, 15f, c.ink, bold = true))
                    addView(text(m.blurb, 12f, c.mute))
                }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
                addView(TextView(this@TestActivity).apply {
                    text = if (on) "✓" else ""; textSize = 15f; typeface = Typeface.DEFAULT_BOLD; setTextColor(c.ember)
                    gravity = Gravity.CENTER
                }, LinearLayout.LayoutParams(px(24), ViewGroup.LayoutParams.WRAP_CONTENT))
                setOnClickListener {
                    WritingModes.set(this@TestActivity, m.key)
                    renderModes()
                }
            })
        }
    }

    // ---------- dictionary ----------

    private fun dictionaryRow() = LinearLayout(this).apply {
        id = R.id.dictionary
        orientation = LinearLayout.HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
        setPadding(0, px(4), 0, px(4))
        dictTitle = text("Add your words", 15f, c.ink, bold = true)
        addView(LinearLayout(this@TestActivity).apply {
            orientation = LinearLayout.VERTICAL
            addView(dictTitle)
            addView(text("Names and jargon Phonon should spell your way.", 12f, c.mute))
        }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        addView(text("›", 22f, c.mute))
        setOnClickListener { startActivity(Intent(this@TestActivity, DictionaryActivity::class.java)) }
    }

    private fun refreshDictionary() {
        if (!::dictTitle.isInitialized) return
        val n = Dictionary.load(filesDir) { null }.entries.size
        dictTitle.text = if (n == 0) "Add your words" else "$n word${if (n == 1) "" else "s"}"
    }

    // ---------- pieces ----------

    /** The iPhone's Card: an uppercase caption over a rounded panel with a hairline border. */
    private fun card(title: String, content: View) = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(px(16), px(16), px(16), px(16))
        background = rounded(c.panel, 16f).apply { setStroke(px(1), c.line) }
        addView(text(title.uppercase(), 11f, c.mute, bold = true).apply {
            letterSpacing = 0.07f
            setPadding(0, 0, 0, px(10))
        })
        addView(content)
    }

    private fun text(s: String, size: Float, color: Int, bold: Boolean = false) = TextView(this).apply {
        text = s; textSize = size; setTextColor(color)
        if (bold) typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
    }

    private fun rounded(color: Int, radius: Float) = GradientDrawable().apply { setColor(color); cornerRadius = radius * dp }

    private fun gap() = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
        .apply { topMargin = px(16) }

    private fun px(v: Int) = (v * dp).toInt()
}
