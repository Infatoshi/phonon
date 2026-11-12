package ai.phonon.keyboard

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RadialGradient
import android.graphics.RectF
import android.graphics.Shader
import android.os.SystemClock
import android.view.View
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * The keyboard's talk bar, the iPhone's design (ios/Shared/KeyboardUI.swift talkBar, IdleFluid, MacPill):
 *  - at rest a dark capsule with grey light drifting inside (no label), redrawn at most 20 times a second while the
 *    keyboard is on screen and animations are on;
 *  - listening: the Mac app's dictation capsule (bar/Sources/PhononBar.swift PillView) at 2.6x, its ember centre
 *    lit and widened by the voice level;
 *  - working: the same capsule, smaller, breathing on a 1.6 s cycle;
 *  - [label] (not ready yet, or a status for about 2 s) is drawn over a dimmed resting fluid.
 * Touch handling (hold to talk, tap to latch) is the IME's; this view only draws.
 */
class TalkBar(context: Context) : View(context) {
    enum class Mode { REST, LISTENING, WORKING }

    var mode = Mode.REST
        set(v) { field = v; invalidate() }
    /** 0..1 voice level while listening (smoothed here). */
    var level = 0.0
        set(v) { field = v; invalidate() }
    var label: String? = null
        set(v) { field = v; invalidate() }
    var dim = false
        set(v) { field = v; invalidate() }
    /** False with the keyboard hidden or animations off (Settings > Accessibility > Remove animations). */
    var animate = true
        set(v) { field = v; invalidate() }

    private val dp = resources.displayMetrics.density
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG)
    private val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeWidth = 0.5f * dp; color = Color.argb(26, 255, 255, 255)
    }
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFCDBFB2.toInt(); textSize = 13f * resources.displayMetrics.scaledDensity
        textAlign = Paint.Align.CENTER; isFakeBoldText = false
    }
    private val rect = RectF()
    private val clip = Path()
    private var shownLevel = 0.0

    private data class Blob(val x: Double, val ax: Double, val ay: Double, val sx: Double, val sy: Double,
                            val phase: Double, val r: Double, val white: Double)
    private val blobs = listOf(
        Blob(0.22, 0.14, 0.30, 0.47, 0.71, 0.0, 1.25, 0.20),
        Blob(0.55, 0.18, 0.35, 0.41, 0.53, 2.1, 1.45, 0.15),
        Blob(0.82, 0.12, 0.28, 0.67, 0.43, 4.2, 1.10, 0.18),
        Blob(0.40, 0.22, 0.40, 0.83, 0.61, 1.3, 0.80, 0.10),
    )

    private fun animationsOn() = ValueAnimator.areAnimatorsEnabled()

    override fun onDraw(c: Canvas) {
        val w = width.toFloat()
        val h = height.toFloat()
        val t = if (animate && animationsOn()) (SystemClock.uptimeMillis() % 3_600_000L) / 1000.0 else 0.0
        if (mode == Mode.REST) {
            drawFluid(c, w, h, t)
            label?.let { c.drawText(ellipsize(it, w - 32 * dp), w / 2, h / 2 - (text.ascent() + text.descent()) / 2, text) }
            if (animate && animationsOn()) postInvalidateDelayed(50)
            return
        }
        // Mac capsule at 2.6x, centred
        val energy = if (mode == Mode.LISTENING) {
            shownLevel += (min(1.0, max(0.08, level)) - shownLevel) * 0.35
            shownLevel
        } else {
            0.18 + (sin(t * Math.PI * 2 / 1.6) + 1) / 2 * 0.18
        }
        val (sw, sh) = if (mode == Mode.LISTENING) (66 + 14 * energy) to (13 + 3 * energy) else (38 + 10 * energy) to (10 + 3 * energy)
        val scale = 2.6f * dp
        val pw = (sw * scale).toFloat()
        val ph = (sh * scale).toFloat()
        rect.set(w / 2 - pw / 2, h / 2 - ph / 2, w / 2 + pw / 2, h / 2 + ph / 2)
        val warmth = 0.50 + energy * 0.38
        val glow = 0.22 + energy * 0.12
        // the warm glow under the capsule (the Mac's shadow: radius 4, y 1, at the same 2.6x)
        fill.shader = null
        fill.color = Color.BLACK
        fill.setShadowLayer(4 * scale, 0f, scale, Color.argb((glow * 255).toInt(), 140, 41, 10))
        c.drawRoundRect(rect, ph / 2, ph / 2, fill)
        fill.clearShadowLayer()
        fill.shader = LinearGradient(rect.left, 0f, rect.right, 0f,
            intArrayOf(Color.BLACK, Color.argb((warmth * 255).toInt(), 107, 31, 9), Color.BLACK), null, Shader.TileMode.CLAMP)
        c.drawRoundRect(rect, ph / 2, ph / 2, fill)
        fill.shader = null
        if (mode == Mode.WORKING && animate && animationsOn()) postInvalidateDelayed(33)
        if (mode == Mode.LISTENING && kotlin.math.abs(shownLevel - level) > 0.01) postInvalidateDelayed(33)
    }

    private fun drawFluid(c: Canvas, w: Float, h: Float, t: Double) {
        rect.set(0f, 0f, w, h)
        clip.reset()
        clip.addRoundRect(rect, h / 2, h / 2, Path.Direction.CW)
        c.save()
        c.clipPath(clip)
        fill.shader = null
        fill.color = 0xFF1D1D1F.toInt()
        c.drawRect(rect, fill)
        val k = if (dim) 0.6 else 1.0
        for (b in blobs) {
            val cx = (w * (b.x + b.ax * sin(t * b.sx + b.phase))).toFloat()
            val cy = (h * (0.5 + b.ay * cos(t * b.sy + b.phase))).toFloat()
            val r = (h * b.r).toFloat()
            c.save()
            c.translate(cx, cy)
            c.scale(2.2f, 1f)  // stretched sideways so the blobs read as a flowing band
            fill.shader = RadialGradient(0f, 0f, r, Color.argb((b.white * k * 255).toInt(), 255, 255, 255),
                Color.argb(0, 255, 255, 255), Shader.TileMode.CLAMP)
            c.drawCircle(0f, 0f, r, fill)
            c.restore()
        }
        fill.shader = null
        c.restore()
        rect.inset(stroke.strokeWidth / 2, stroke.strokeWidth / 2)
        c.drawRoundRect(rect, h / 2, h / 2, stroke)
    }

    private fun ellipsize(s: String, max: Float): String {
        if (text.measureText(s) <= max) return s
        var e = s
        while (e.length > 1 && text.measureText("$e…") > max) e = e.dropLast(1)
        return "$e…"
    }
}
