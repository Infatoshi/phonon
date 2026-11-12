package ai.phonon.keyboard

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.LayerDrawable
import android.graphics.drawable.StateListDrawable
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.view.Gravity
import android.view.HapticFeedbackConstants
import android.view.MotionEvent
import android.view.View
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView

/**
 * The keys under the talk bar, the iPhone keyboard's layout and colours (ios/Shared/KeyboardUI.swift keys, Key):
 * letters with shift (double tap locks caps), a numbers layer and a symbols layer ("123" / "#+=" / "ABC"), a globe key
 * to the next keyboard when the system asks for one, space (double space types ". "), return and delete (repeats
 * while held). Text goes through [Host]; the panel only lays out keys and tracks shift and the layer.
 */
@SuppressLint("ViewConstructor", "ClickableViewAccessibility")
class KeyPanel(context: Context, private val host: Host) : LinearLayout(context) {
    interface Host {
        fun type(s: String)
        fun deleteBackward()
        fun enter()
        fun nextKeyboard()
        fun showGlobe(): Boolean
        /** Text before the cursor (a few characters), for auto-capitals and double space. */
        fun before(): String
        /** False in fields that take no capitals (email, web address, password). */
        fun autoCaps(): Boolean
    }

    private enum class Layer { LETTERS, NUMBERS, SYMBOLS }

    private val dp = resources.displayMetrics.density
    private var layer = Layer.LETTERS
    private var shift = true  // starts capitalised
    private var capsLock = false
    private var lastShiftTap = 0L
    private var lastSpace = 0L
    private val letterKeys = mutableListOf<Pair<Char, TextView>>()
    private var shiftKey: TextView? = null
    /** Key name to view, for the IME's KEYPOS log (adb `input tap` presses real keys headlessly). */
    val keyViews = mutableListOf<Pair<String, View>>()
    var onLayout: (() -> Unit)? = null

    init {
        orientation = VERTICAL
        build()
    }

    /** Rebuild for the current field (the globe key comes and goes) and set shift from the text before the cursor. */
    fun reset() {
        layer = Layer.LETTERS
        capsLock = false
        build()
        autoShift()
    }

    private fun build() {
        removeAllViews()
        letterKeys.clear()
        keyViews.clear()
        shiftKey = null
        when (layer) {
            Layer.LETTERS -> {
                row(0f, "qwertyuiop".map { letter(it) })
                row(0.5f, "asdfghjkl".map { letter(it) })
                val sh = special(shiftLabel(), "shift") { tapShift() }
                shiftKey = sh
                paintShift()
                row3(sh, "zxcvbnm".map { letter(it) to 1f })
            }
            Layer.NUMBERS, Layer.SYMBOLS -> {
                val rows = if (layer == Layer.NUMBERS) listOf("1234567890", "-/:;()$&@\"", ".,?!'")
                    else listOf("[]{}#%^*+=", "_\\|~<>€£¥•", ".,?!'")
                row(0f, rows[0].map { char(it) })
                row(0f, rows[1].map { char(it) })
                val toggle = special(if (layer == Layer.NUMBERS) "#+=" else "123", "layer") {
                    layer = if (layer == Layer.NUMBERS) Layer.SYMBOLS else Layer.NUMBERS
                    build()
                }
                row3(toggle, rows[2].map { char(it) to 1.4f })
            }
        }
        val bottom = mutableListOf<Pair<View, Float>>()
        bottom += special(if (layer == Layer.LETTERS) "123" else "ABC", "mode") {
            layer = if (layer == Layer.LETTERS) Layer.NUMBERS else Layer.LETTERS
            build()
            if (layer == Layer.LETTERS) autoShift()
        } to 1.35f
        var spaceW = 10f - 1.35f - 2.1f
        if (host.showGlobe()) {
            bottom += globe() to 1.1f
            spaceW -= 1.1f
        }
        bottom += key("space", "space", special = false, small = true, name = "space") { space() } to spaceW
        bottom += special("return", "return") { host.enter(); autoShift() } to 2.1f
        addRow(bottom, 0f, last = true)
    }

    // ---------- rows ----------

    private fun row(inset: Float, keys: List<View>) = addRow(keys.map { it to 1f }, inset)

    /** Shift or the layer toggle, the middle keys between flexible gaps, then delete (the iPhone's third row). */
    private fun row3(left: View, middle: List<Pair<View, Float>>) {
        val del = deleteKey()
        val used = 1.35f * 2 + middle.sumOf { it.second.toDouble() }.toFloat()
        val gap = maxOf(0f, (10f - used) / 2)
        addRow(listOf(left to 1.35f) + middle + listOf(del to 1.35f), 0f, split = gap)
    }

    /** One row: [inset] key units of space at each end, [split] units after the first and before the last key. */
    private fun addRow(keys: List<Pair<View, Float>>, inset: Float, split: Float = 0f, last: Boolean = false) {
        val line = LinearLayout(context).apply { orientation = HORIZONTAL; weightSum = 10f }
        if (inset > 0) line.addView(View(context), LayoutParams(0, 1, inset))
        keys.forEachIndexed { i, (v, w) ->
            if (split > 0 && i == keys.size - 1) line.addView(View(context), LayoutParams(0, 1, split))
            line.addView(v, LayoutParams(0, (KEY_H * dp).toInt(), w).apply {
                setMargins((GAP / 2 * dp).toInt(), 0, (GAP / 2 * dp).toInt(), 0)
            })
            if (split > 0 && i == 0) line.addView(View(context), LayoutParams(0, 1, split))
        }
        if (inset > 0) line.addView(View(context), LayoutParams(0, 1, inset))
        addView(line, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply {
            if (!last) bottomMargin = (ROW_GAP * dp).toInt()
        })
    }

    // ---------- keys ----------

    private fun letter(c: Char): View {
        val v = key(c.toString(), c.toString(), special = false, name = c.toString()) {
            val s = if (shift || capsLock) c.uppercaseChar().toString() else c.toString()
            host.type(s)
            if (shift && !capsLock) { shift = false; paintShift() }
        }
        letterKeys += c to v
        return v
    }

    private fun char(c: Char): View = key(c.toString(), c.toString(), special = false, name = c.toString()) {
        host.type(c.toString())
        if (c == '.' || c == '?' || c == '!') autoShift()
    }

    private fun special(label: String, desc: String, action: () -> Unit) =
        key(label, desc, special = true, small = label.length > 1, name = label, action = action)

    private fun key(label: String, desc: String, special: Boolean, small: Boolean = label.length > 1, name: String,
                    action: () -> Unit): TextView = TextView(context).apply {
        text = label
        gravity = Gravity.CENTER
        setTextColor(INK)
        textSize = if (small) 15f else 23f
        includeFontPadding = false
        contentDescription = "key_$desc"
        background = keyBackground(if (special) SPECIAL else KEY)
        isSoundEffectsEnabled = false
        setOnTouchListener { v, e ->
            when (e.actionMasked) {
                MotionEvent.ACTION_DOWN -> { v.isPressed = true; v.performHapticFeedback(HapticFeedbackConstants.KEYBOARD_TAP) }
                MotionEvent.ACTION_UP -> { v.isPressed = false; action() }
                MotionEvent.ACTION_CANCEL -> v.isPressed = false
            }
            true
        }
        keyViews += name to this
    }

    private val repeat = Handler(Looper.getMainLooper())

    /** Delete: once on press, then every 80 ms after 0.45 s held (the iPhone's timings). */
    private fun deleteKey(): View = TextView(context).apply {
        text = "⌫"; gravity = Gravity.CENTER; setTextColor(INK); textSize = 20f; includeFontPadding = false
        contentDescription = "key_delete"
        background = keyBackground(SPECIAL)
        val tick = object : Runnable {
            override fun run() { del(); repeat.postDelayed(this, 80) }
        }
        setOnTouchListener { v, e ->
            when (e.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    v.isPressed = true
                    v.performHapticFeedback(HapticFeedbackConstants.KEYBOARD_TAP)
                    del()
                    repeat.postDelayed(tick, 450)
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> { v.isPressed = false; repeat.removeCallbacks(tick) }
            }
            true
        }
        keyViews += "⌫" to this
    }

    private fun del() {
        host.deleteBackward()
        autoShift()
    }

    private fun globe(): View = FrameLayout(context).apply {
        contentDescription = "key_globe"
        background = keyBackground(SPECIAL)
        addView(GlobeGlyph(context), FrameLayout.LayoutParams((18 * dp).toInt(), (18 * dp).toInt(), Gravity.CENTER))
        setOnClickListener { host.nextKeyboard() }
        keyViews += "globe" to this
    }

    // ---------- shift, space ----------

    private fun shiftLabel() = if (capsLock) "⇪" else "⇧"

    private fun tapShift() {
        val now = SystemClock.uptimeMillis()
        when {
            now - lastShiftTap < 300 -> { capsLock = true; shift = true }
            capsLock -> { capsLock = false; shift = false }
            else -> shift = !shift
        }
        lastShiftTap = now
        paintShift()
    }

    private fun paintShift() {
        val on = shift || capsLock
        shiftKey?.apply {
            text = shiftLabel()
            textSize = 25f
            setTextColor(if (on) BG else INK)
            background = keyBackground(if (on) INK else SPECIAL)
        }
        for ((c, v) in letterKeys) v.text = if (on) c.uppercaseChar().toString() else c.toString()
    }

    private fun space() {
        val before = host.before()
        val now = SystemClock.uptimeMillis()
        // double space: ". " like the system keyboard
        if (now - lastSpace < 450 && before.endsWith(" ") && before.length >= 2 && before[before.length - 2].isLetterOrDigit()) {
            host.deleteBackward()
            host.type(". ")
            lastSpace = 0
            autoShift()
            return
        }
        host.type(" ")
        lastSpace = now
        autoShift()
    }

    /** Capitalise at the start of the field and after a sentence end, as the system keyboard does. */
    fun autoShift() {
        if (capsLock || layer != Layer.LETTERS) return
        val before = host.before()
        val t = before.trimEnd(' ')
        shift = host.autoCaps() && (before.isEmpty() || before.endsWith("\n") ||
            (before.endsWith(" ") && t.isNotEmpty() && t.last() in ".?!"))
        paintShift()
    }

    override fun onLayout(changed: Boolean, l: Int, t: Int, r: Int, b: Int) {
        super.onLayout(changed, l, t, r, b)
        onLayout?.invoke()  // a layer switch keeps the panel's size: log on every pass (the IME drops repeats)
    }

    private fun keyBackground(fill: Int): StateListDrawable {
        fun layer(c: Int) = LayerDrawable(arrayOf(
            GradientDrawable().apply { setColor(Color.argb(115, 0, 0, 0)); cornerRadius = 6 * dp },
            GradientDrawable().apply { setColor(c); cornerRadius = 6 * dp },
        )).apply { setLayerInset(1, 0, 0, 0, dp.toInt().coerceAtLeast(1)) }  // the 1 pt shadow under each key
        return StateListDrawable().apply {
            addState(intArrayOf(android.R.attr.state_pressed), layer(PRESSED))
            addState(intArrayOf(), layer(fill))
        }
    }

    /** The globe glyph (SF Symbols "globe"): a circle, its central meridian ellipse and two parallels. */
    private class GlobeGlyph(context: Context) : View(context) {
        private val p = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE; color = INK; strokeWidth = 1.4f * resources.displayMetrics.density
        }
        override fun onDraw(c: Canvas) {
            val s = minOf(width, height).toFloat()
            val r = s / 2 - p.strokeWidth
            val cx = width / 2f
            val cy = height / 2f
            c.drawCircle(cx, cy, r, p)
            c.drawOval(cx - r * 0.42f, cy - r, cx + r * 0.42f, cy + r, p)
            c.drawLine(cx - r, cy, cx + r, cy, p)
            val y = r * 0.5f
            val x = r * 0.866f
            c.drawLine(cx - x, cy - y, cx + x, cy - y, p)
            c.drawLine(cx - x, cy + y, cx + x, cy + y, p)
        }
    }

    companion object {
        const val KEY_H = 43f
        const val GAP = 6f
        const val ROW_GAP = 11f
        val BG = 0xFF16120F.toInt()
        val KEY = 0xFF3A312A.toInt()
        val SPECIAL = 0xFF2A231E.toInt()
        val PRESSED = 0xFF4A3F36.toInt()
        val INK = 0xFFEEE6DC.toInt()
        val SOFT = 0xFFCDBFB2.toInt()
        val RAISED = 0xFF231C17.toInt()
    }
}
