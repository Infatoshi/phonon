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
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(32, 96, 32, 32) }
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
        root.addView(EditText(this).apply {
            id = R.id.field
            hint = "type or hold to talk"
            minLines = 3
        })
        setContentView(root)
        ModelFetch.addListener(onModels)
        ModelFetch.start(this)
    }

    override fun onDestroy() {
        ModelFetch.removeListener(onModels)
        super.onDestroy()
    }
}
