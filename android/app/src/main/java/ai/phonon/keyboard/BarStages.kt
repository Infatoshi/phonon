package ai.phonon.keyboard

import kotlin.math.exp

/**
 * The talk bar's stages (ignite, listening, steady warm processing, then a crisp fade back to rest), as pure maths
 * so JVM tests can check them. The same numbers as the Mac and iPhone bars and the Windows and Linux tray icon
 * (crates/phonon-win/src/indicator.rs).
 *
 * The bar is a full-width horizontal gradient, black at both ends, one warm ember light at 0.5 (`halfWidth` is always
 * 0.5, with no sliding-together core):
 *  1. ignite, the instant the finger goes down: full strength, falling to the live voice level in 0.10 s;
 *  2. listening: w = 0.5, strength follows the smoothed voice level;
 *  3. processing, from let-go until the text is delivered: steady warm ember at full width (0.65 strength, 0% redraw);
 *  4. ended (text pasted or not): no pulse, fades straight out at full width in 0.10 s with quadratic ease-out.
 */
object BarStages {
    /** The capsule's warm ember (0.42, 0.12, 0.035). */
    val BASE = doubleArrayOf(107.0, 31.0, 9.0)

    const val IGNITE_SECONDS = 0.10
    const val DROP_SECONDS = 0.10
    const val WIDE_WIDTH = 0.5
    /** Width never moves in either normal or reduced-motion mode. */
    const val REDUCED_WIDTH = WIDE_WIDTH
    const val LEVEL_FLOOR = 0.08

    /** Listening's light is this opaque over black in silence, plus [LISTEN_STRENGTH_RANGE] times the voice
     *  energy (0..1): the Mac capsule's 0.50 + energy * 0.38 (macos/Sources/PhononBar.swift warmth). */
    const val LISTEN_STRENGTH = 0.50
    const val LISTEN_STRENGTH_RANGE = 0.38
    /** Full strength, energy 1 (the ignite flash and the loudest voice): 0.50 + 0.38 = 0.88 opacity over black. */
    const val FULL_STRENGTH = LISTEN_STRENGTH + LISTEN_STRENGTH_RANGE
    /** Steady warm ember strength while processing (let-go until text arrives). */
    const val PROCESSING_STRENGTH = 0.65

    /** Where in a dictation the bar is. The ending stages carry what they fade from. */
    sealed class Stage {
        /** Finger down until let-go. Elapsed counts from the press. */
        object Listening : Stage()
        /** Let-go until the text is delivered or the dictation ends. Elapsed counts from the let-go. */
        object Processing : Stage()
        /** Ended, text pasted or not (empty, error, cancelled). */
        data class Dropped(val fromWidth: Double, val fromStrength: Double, val fromHeat: Double) : Stage()
    }

    /** One frame. The centre colour is BASE * strength over black. */
    data class Look(
        /** Half the width of the lit part, as a fraction of the whole (always 0.5 = full width). */
        val halfWidth: Double,
        /** Opacity of the warm light over black. */
        val strength: Double,
        /** Kept at 0.0: single warm ember colour. */
        val heat: Double,
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
                // full strength at the press, the live level 0.10 s later, linearly between
                Look(WIDE_WIDTH, lerp(LISTEN_STRENGTH + LISTEN_STRENGTH_RANGE * lv, FULL_STRENGTH, ignite), 0.0)
            }
            Stage.Processing ->
                Look(WIDE_WIDTH, PROCESSING_STRENGTH, 0.0)
            is Stage.Dropped -> {
                val p = unit(e / DROP_SECONDS)
                val remain = (1.0 - p) * (1.0 - p)
                Look(stage.fromWidth, stage.fromStrength * remain, stage.fromHeat)
            }
        }
    }

    /** Whether the ending stage has finished and the bar is at rest again. */
    fun finished(stage: Stage, elapsed: Double): Boolean = when (stage) {
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

        /** The dictation ended (text pasted or not): no pulse, fades straight out from where it is. */
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
            if (finished(s, elapsed)) { stage = null; return null }
            return look(s, elapsed, level, reduced)
        }

        /** Whether the next frame can differ from this one (false while processing holds steady). */
        fun moving(now: Double, reduced: Boolean): Boolean = when (stage) {
            null, Stage.Processing -> false
            else -> true
        }
    }

    /** The gradient's centre colour as 0..255 RGB. */
    fun centre(look: Look): IntArray = IntArray(3) {
        (BASE[it] * look.strength).coerceIn(0.0, 255.0).toInt()
    }
}
