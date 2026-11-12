package ai.phonon.keyboard

import android.Manifest
import android.app.Activity
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView

/** Launcher: asks for the mic once (the IME cannot prompt), starts the model download with its progress shown,
 *  and gives the IME a text field to type into. */
class TestActivity : Activity() {
    private var progress: TextView? = null
    private var cellular: Button? = null
    private val onModels: (ModelFetch.State) -> Unit = { s ->
        runOnUiThread {
            progress?.text = s.message
            cellular?.visibility = if (s.waitingForWifi) View.VISIBLE else View.GONE
        }
    }

    override fun onCreate(b: Bundle?) {
        super.onCreate(b)
        requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(32, 32, 32, 32) }
        root.addView(TextView(this).apply { text = "Phonon keyboard test field" })
        progress = TextView(this).apply { id = R.id.models }
        root.addView(progress)
        cellular = Button(this).apply {
            id = R.id.cellular
            text = "Download the text model on mobile data (988 MB)"
            visibility = View.GONE
            setOnClickListener { ModelFetch.allowCellular(this@TestActivity) }
        }
        root.addView(cellular)
        // the writing mode, as in the iPhone app (WritingMode in ios/Shared/Theme.swift): the keyboard reads it
        // for every dictation (PhononIME.currentMode)
        root.addView(TextView(this).apply { text = "Writing style"; setPadding(0, 24, 0, 0) })
        val current = WritingModes.get(this)
        root.addView(android.widget.RadioGroup(this).apply {
            id = R.id.modes
            for ((i, m) in WritingModes.ALL.withIndex()) {
                addView(android.widget.RadioButton(this@TestActivity).apply {
                    id = 1000 + i
                    text = "${m.title}: ${m.blurb}"
                    isChecked = m.key == current
                })
            }
            setOnCheckedChangeListener { _, checked ->
                WritingModes.set(this@TestActivity, WritingModes.ALL[checked - 1000].key)
            }
        })
        root.addView(Button(this).apply {
            id = R.id.dictionary
            text = "Dictionary"
            setOnClickListener { startActivity(android.content.Intent(this@TestActivity, DictionaryActivity::class.java)) }
        })
        root.addView(EditText(this).apply {
            id = R.id.field
            hint = "type or hold to talk"
            minLines = 3
        })
        // targetSdk 35 draws edge to edge, and without the inset frame the first rows sat under the title bar
        setContentView(android.widget.FrameLayout(this).apply { fitsSystemWindows = true; addView(root) })
        lightSystemBars()
        ModelFetch.addListener(onModels)
        ModelFetch.start(this)
    }

    override fun onDestroy() {
        ModelFetch.removeListener(onModels)
        super.onDestroy()
    }
}
