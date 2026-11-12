package ai.phonon.keyboard

import android.app.Activity
import android.os.Bundle
import android.text.InputType
import android.util.TypedValue
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

/**
 * The dictionary editor: words Phonon should spell the user's way, and phrases with what to write instead (the
 * desktop's dictionary entries: a term, or a phrase with a replacement). Saved as dictionary.json in the app's
 * files dir in the desktop's schema; the keyboard rereads it when it changes (PhononIME.dictionary).
 */
class DictionaryActivity : Activity() {
    private lateinit var list: LinearLayout
    private var dict = Dictionary(emptyList(), null)

    private fun dp(v: Float) = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, resources.displayMetrics).toInt()

    override fun onCreate(b: Bundle?) {
        super.onCreate(b)
        dict = Dictionary.load(filesDir) { null }
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20f), dp(12f), dp(20f), dp(20f))
        }
        root.addView(TextView(this).apply {
            text = "Words Phonon should spell your way, like names and technical terms. For a word it keeps " +
                "getting wrong, add what it hears and what to write instead."
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 15f)
            alpha = 0.75f
            setPadding(0, dp(6f), 0, dp(14f))
        })
        val word = EditText(this).apply {
            id = R.id.dict_word
            hint = "Word or phrase"
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS
            isSingleLine = true
        }
        val instead = EditText(this).apply {
            id = R.id.dict_instead
            hint = "Write instead (optional)"
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS
            isSingleLine = true
        }
        root.addView(word, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        root.addView(instead, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        root.addView(Button(this).apply {
            id = R.id.dict_add
            text = "Add"
            setOnClickListener {
                val w = word.text.toString().trim()
                if (w.isEmpty()) return@setOnClickListener
                val r = instead.text.toString().trim().ifEmpty { null }
                update(dict.with(Dictionary.Entry(phrase = w, replacement = r, source = "phone")))
                word.setText("")
                instead.setText("")
            }
        }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
            gravity = Gravity.END
        })
        list = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(0, dp(12f), 0, 0) }
        root.addView(ScrollView(this).apply { addView(list) },
            LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        // targetSdk 35 draws edge to edge: the frame takes the system bar insets as padding, the column keeps its own
        setContentView(android.widget.FrameLayout(this).apply { fitsSystemWindows = true; addView(root) })
        lightSystemBars()
        render()
    }

    private fun update(d: Dictionary) {
        dict = d
        Dictionary.save(filesDir, d)
        render()
    }

    private fun render() {
        list.removeAllViews()
        if (dict.entries.isEmpty()) {
            list.addView(TextView(this).apply {
                text = "No words yet."
                alpha = 0.6f
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 15f)
            })
            return
        }
        for (e in dict.entries) {
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(0, dp(4f), 0, dp(4f))
            }
            row.addView(TextView(this).apply {
                text = if (e.replacement != null) "${e.phrase}  →  ${e.replacement}" else e.phrase
                setTextAppearance(android.R.style.TextAppearance_DeviceDefault)  // primary text colour, not the dim default
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 17f)
            }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            row.addView(Button(this, null, android.R.attr.borderlessButtonStyle).apply {
                text = "Remove"
                contentDescription = "Remove ${e.phrase}"
                setOnClickListener { update(dict.without(e)) }
            })
            list.addView(row)
        }
    }
}
