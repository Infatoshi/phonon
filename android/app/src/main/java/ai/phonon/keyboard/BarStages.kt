package ai.phonon.keyboard

import kotlin.math.exp
import kotlin.math.sqrt

/**
 * The talk bar's four stages (ignite, listening, processing, pasted), as pure maths so JVM tests can check them.
 * The same numbers as the Mac and iPhone bars and the Windows and Linux tray icon (crates/phonon-win/src/indicator.rs).
 *
 * The bar is a horizontal gradient, black at 0.5-w and 0.5+w, one warm light at 0.5:
 *  1. ignite, the instant the finger goes down: full strength, falling to the live voice level in 0.18 s;
 *  2. listening: w = 0.5, strength follows the smoothed voice level;
 *  3. processing, from let-go until the text is delivered: over 1.2 s w narrows from 0.5 to 0.08 and the light
 *     warms and brightens, then the tight core holds, still;
 *  4. pasted: over 0.6 s w opens from where it was back to 0.5 (sqrt of progress) while a light orange fades out;
 *  5. ended with nothing pasted: no burst, the core fades out in 0.2 s;
 *  6. reduced motion (animator duration scale 0): the width never moves; processing is a steady medium core and
 *     pasted a 0.3 s fade.
 */
object BarStages {
    /** The Mac capsule's ember (0.42, 0.12, 0.035). */
    val BASE = doubleArrayOf(107.0, 31.0, 9.0)
    /** What the light warms to while gathering. */
    val BRIGHT = doubleArrayOf(227.0, 101.0, 29.0)
    /** The burst's light orange. */
    val ORANGE = doubleArrayOf(255.0, 170.0, 100.0)

    const val IGNITE_SECONDS = 0.18
    const val GATHER_SECONDS = 1.2
    const val BURST_SECONDS = 0.6
    const val DROP_SECONDS = 0.2
    const val REDUCED_FADE_SECONDS = 0.3
    const val TIGHT_WIDTH = 0.08
    const val WIDE_WIDTH = 0.5
    /** Reduced motion's processing core: steady, between wide and tight. */
    const val REDUCED_WIDTH = 0.25
    const val LEVEL_FLOOR = 0.08

    /** Listening's light is this opaque over black in silence, plus [LISTEN_STRENGTH_RANGE] times the voice
     *  energy (0..1): the Mac capsule's 0.50 + energy * 0.38 (bar/Sources/PhononBar.swift warmth). */
    const val LISTEN_STRENGTH = 0.50
    const val LISTEN_STRENGTH_RANGE = 0.38
    /** Full strength, energy 1 (the ignite flash and the loudest voice): 0.50 + 0.38 = 0.88 opacity over black.
     *  It stays under 1.0 so the gather, which ends brighter at [GATHER_END_STRENGTH], still reads as a change. */
    const val FULL_STRENGTH = LISTEN_STRENGTH + LISTEN_STRENGTH_RANGE
    /** The gather's strength at let-go (the Mac's 0.5 + 0.5 * heat 0.3) and once the core is tight. */
    const val GATHER_START_STRENGTH = 0.65
    const val GATHER_END_STRENGTH = 1.0
    /** Reduced motion's steady processing core. */
    const val REDUCED_STRENGTH = 0.80
    const val REDUCED_HEAT = 0.5

    /** Where in a dictation the bar is. The ending stages carry what they fade from. */
    sealed class Stage {
        /** Finger down until let-go. Elapsed counts from the press. */
        object Listening : Stage()
        /** Let-go until the text is delivered or the dictation ends. Elapsed counts from the let-go. */
        object Processing : Stage()
        /** The text went in; [fromWidth] is the gather's width then. */
        data class Pasted(val fromWidth: Double) : Stage()
        /** Ended with nothing pasted (empty, error, cancelled). */
        data class Dropped(val fromWidth: Double, val fromStrength: Double, val fromHeat: Double) : Stage()
    }

    /** One frame. The centre colour is lerp(BASE, BRIGHT, heat) * strength over black, plus ORANGE * burst. */
    data class Look(
        /** Half the width of the lit part, as a fraction of the whole (0.5 = the full width). */
        val halfWidth: Double,
        /** Opacity of the warm light over black. */
        val strength: Double,
        /** 0 = the base ember, 1 = the bright processing colour. */
        val heat: Double,
        /** Opacity of the burst's orange light. */
        val burst: Double,
    )

    private fun lerp(a: Double, b: Double, t: Double) = a + (b - a) * t
    private fun unit(x: Double) = x.coerceIn(0.0, 1.0)

    /** The stage maths: stage, seconds since it began, smoothed voice level (0..1), reduced motion. */
    fun look(stage: Stage, elapsed: Double, level: Double, reduced: Boolean): Look {
        val e = maxOf(0.0, elapsed)
        val lv = level.coerceIn(LEVEL_FLOOR, 1.0)
        return when (stage) {
            Stage.Listening -> {
                val ignite = unit(1.0 - e / IGNITE_SECONDS)
                // full strength at the press, the live level 0.18 s later, linearly between
                Look(WIDE_WIDTH, lerp(LISTEN_STRENGTH + LISTEN_STRENGTH_RANGE * lv, FULL_STRENGTH, ignite), 0.0, 0.0)
            }
            Stage.Processing ->
                if (reduced) Look(REDUCED_WIDTH, REDUCED_STRENGTH, REDUCED_HEAT, 0.0)
                else {
                    val k = unit(e / GATHER_SECONDS)
                    Look(lerp(WIDE_WIDTH, TIGHT_WIDTH, k), lerp(GATHER_START_STRENGTH, GATHER_END_STRENGTH, k), k, 0.0)
                }
            is Stage.Pasted ->
                if (reduced) Look(stage.fromWidth, 0.0, 0.0, 1.0 - unit(e / REDUCED_FADE_SECONDS))
                else {
                    val p = unit(e / BURST_SECONDS)
                    Look(lerp(stage.fromWidth, WIDE_WIDTH, sqrt(p)), 0.0, 0.0, 1.0 - p)
                }
            is Stage.Dropped ->
                Look(stage.fromWidth, stage.fromStrength * (1.0 - unit(e / DROP_SECONDS)), stage.fromHeat, 0.0)
        }
    }

    /** Whether a stage that ends (pasted, dropped) has finished and the bar is at rest again. */
    fun finished(stage: Stage, elapsed: Double, reduced: Boolean): Boolean = when (stage) {
        is Stage.Pasted -> elapsed >= (if (reduced) REDUCED_FADE_SECONDS else BURST_SECONDS)
        is Stage.Dropped -> elapsed >= DROP_SECONDS
        Stage.Listening, Stage.Processing -> false
    }

    /** Fast attack, slow release, floor 0.08. */
    class Smoother {
        var value = LEVEL_FLOOR
            private set

        fun step(target: Double, dt: Double): Double {
            val t = target.coerceIn(LEVEL_FLOOR, 1.0)
            val tau = if (t > value) ATTACK_SECONDS else RELEASE_SECONDS
            value = (value + (t - value) * (1.0 - exp(-maxOf(0.0, dt) / tau))).coerceIn(LEVEL_FLOOR, 1.0)
            return value
        }

        companion object {
            const val ATTACK_SECONDS = 0.04
            const val RELEASE_SECONDS = 0.30
        }
    }

    /** The bar's state, in seconds on any steady clock. Each event says what it does out of turn: nothing. */
    class Machine {
        private var stage: Stage? = null
        private var since = 0.0

        val isRest get() = stage == null
        val current get() = stage

        /** Finger down or tap. Lights at once, whatever the bar was doing. */
        fun press(now: Double) {
            if (stage != Stage.Listening) { stage = Stage.Listening; since = now }
        }

        /** Let-go or stop: listening becomes processing. */
        fun release(now: Double) {
            if (stage == Stage.Listening) { stage = Stage.Processing; since = now }
        }

        /** The text went in: burst from wherever the gather is. */
        fun pasted(now: Double, reduced: Boolean) {
            if (stage == Stage.Processing) {
                stage = Stage.Pasted(look(Stage.Processing, now - since, LEVEL_FLOOR, reduced).halfWidth)
                since = now
            }
        }

        /** The dictation ended with nothing pasted: no burst, the core fades out. */
        fun dropped(now: Double, level: Double, reduced: Boolean) {
            val s = stage
            if (s != null && (s == Stage.Listening || s == Stage.Processing)) {
                val l = look(s, now - since, level, reduced)
                stage = Stage.Dropped(l.halfWidth, l.strength, l.heat)
                since = now
            }
        }

        /** The frame to draw now, or null at rest. A stage that ends hands back to rest here. */
        fun frame(now: Double, level: Double, reduced: Boolean): Look? {
            val s = stage ?: return null
            val elapsed = now - since
            if (finished(s, elapsed, reduced)) { stage = null; return null }
            return look(s, elapsed, level, reduced)
        }

        /** Whether the next frame can differ from this one (false once the gather has finished and holds). */
        fun moving(now: Double, reduced: Boolean): Boolean = when (stage) {
            null -> false
            Stage.Processing -> !reduced && now - since < GATHER_SECONDS
            else -> true
        }
    }

    /** The gradient's centre colour as 0..255 RGB. */
    fun centre(look: Look): IntArray = IntArray(3) {
        ((lerp(BASE[it], BRIGHT[it], look.heat) * look.strength + ORANGE[it] * look.burst)).coerceIn(0.0, 255.0).toInt()
    }
}
