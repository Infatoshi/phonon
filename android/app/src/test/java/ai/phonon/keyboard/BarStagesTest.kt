package ai.phonon.keyboard

import ai.phonon.keyboard.BarStages.Stage
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class BarStagesTest {
    private val eps = 1e-4

    /** Listening's strength at a voice energy (0..1). */
    private fun listening(energy: Double) = BarStages.LISTEN_STRENGTH + BarStages.LISTEN_STRENGTH_RANGE * energy

    @Test
    fun fullStrengthIsTheMacCapsulesListeningMaximum() {
        // 0.50 + 0.38 opacity over black, the loudest voice and the ignite alike
        val loudest = BarStages.look(Stage.Listening, 5.0, 1.0, false).strength
        assertEquals(0.88, loudest, eps)
        assertEquals(BarStages.FULL_STRENGTH, loudest, eps)
        assertEquals(BarStages.PROCESSING_STRENGTH, BarStages.look(Stage.Processing, 0.0, 0.5, false).strength, eps)
    }

    @Test
    fun igniteIsFullStrengthAtThePressBeforeAnyLevelArrives() {
        val atPress = BarStages.look(Stage.Listening, 0.0, 0.0, false)
        assertEquals(BarStages.FULL_STRENGTH, atPress.strength, eps)
        assertEquals(atPress, BarStages.look(Stage.Listening, 0.0, 0.3, false))
        assertEquals(BarStages.WIDE_WIDTH, atPress.halfWidth, eps)
    }

    @Test
    fun igniteDecaysLinearlyToTheLiveLevelIn010Seconds() {
        val level = 0.4
        // half way through the ignite: half full energy, half the live level
        assertEquals(listening(0.5 + 0.5 * level), BarStages.look(Stage.Listening, 0.05, level, false).strength, eps)
        assertEquals(listening(level), BarStages.look(Stage.Listening, 0.10, level, false).strength, eps)
        assertEquals(listening(level), BarStages.look(Stage.Listening, 2.0, level, false).strength, eps)
        assertEquals(listening(BarStages.LEVEL_FLOOR), BarStages.look(Stage.Listening, 5.0, 0.0, false).strength, eps)
    }

    @Test
    fun processingIsSteadyFullWidthWarmEmber() {
        val start = BarStages.look(Stage.Processing, 0.0, 0.5, false)
        assertEquals(BarStages.WIDE_WIDTH, start.halfWidth, eps)
        assertEquals(BarStages.PROCESSING_STRENGTH, start.strength, eps)
        assertEquals(0.0, start.heat, eps)
        for (t in listOf(0.02, 0.1, 0.6, 1.2, 5.0, 60.0)) {
            assertEquals("no sliding or breathing at $t", start, BarStages.look(Stage.Processing, t, 0.9, false))
            assertEquals("identical under reduced motion at $t", start, BarStages.look(Stage.Processing, t, 0.9, true))
        }
    }

    @Test
    fun fastDictationEndsAndFadesCleanlyAtFullWidth() {
        val m = BarStages.Machine()
        m.press(0.0); m.release(1.0); m.dropped(1.02, 0.5, false)
        val s = m.current as Stage.Dropped
        assertEquals(BarStages.WIDE_WIDTH, s.fromWidth, eps)
        assertEquals(BarStages.PROCESSING_STRENGTH, s.fromStrength, eps)
        assertEquals(BarStages.WIDE_WIDTH, m.frame(1.02, 0.5, false)!!.halfWidth, eps)
        assertNotNull(m.frame(1.11, 0.5, false))
        assertNull(m.frame(1.13, 0.5, false))
        assertTrue(m.isRest)
    }

    @Test
    fun theEndFadesWithNoPulseOrWidthChange() {
        val m = BarStages.Machine()
        m.press(0.0); m.release(1.0); m.dropped(3.0, 0.5, false)
        val first = m.frame(3.0, 0.5, false)!!
        assertEquals(BarStages.WIDE_WIDTH, first.halfWidth, eps)
        assertEquals(BarStages.PROCESSING_STRENGTH, first.strength, eps)
        for (t in listOf(3.02, 3.05, 3.09)) {
            val f = m.frame(t, 0.5, false)!!
            assertEquals("no colour change", 0.0, f.heat, 0.0)
            assertTrue("never brighter", f.strength <= first.strength)
            assertEquals("width stays full", BarStages.WIDE_WIDTH, f.halfWidth, eps)
        }
        assertEquals(BarStages.PROCESSING_STRENGTH * 0.25,
            BarStages.look(m.current!!, BarStages.DROP_SECONDS / 2, 0.5, false).strength, eps)
        assertNull(m.frame(3.11, 0.5, false))
        assertTrue(m.isRest)
    }

    @Test
    fun aFailureWhileListeningFadesToo() {
        val m = BarStages.Machine()
        m.press(0.0); m.dropped(0.05, 0.3, false)
        assertTrue(m.current is Stage.Dropped)
        assertNull(m.frame(0.16, 0.3, false))
    }

    @Test
    fun widthNeverMovesUnderNormalOrReducedMotion() {
        for (reduced in listOf(false, true)) {
            for (t in listOf(0.0, 0.05, 0.3, 1.2, 4.0)) {
                assertEquals(BarStages.WIDE_WIDTH, BarStages.look(Stage.Processing, t, 0.5, reduced).halfWidth, eps)
                assertEquals(BarStages.WIDE_WIDTH, BarStages.look(Stage.Listening, t, 0.5, reduced).halfWidth, eps)
            }
            val m = BarStages.Machine()
            m.press(0.0); m.release(1.0); m.dropped(2.0, 0.5, reduced)
            val s = m.current!!
            for (t in listOf(0.0, 0.05, 0.09)) assertEquals(BarStages.WIDE_WIDTH, BarStages.look(s, t, 0.5, reduced).halfWidth, eps)
            assertTrue(BarStages.finished(s, BarStages.DROP_SECONDS))
            assertFalse(BarStages.finished(s, BarStages.DROP_SECONDS - 0.01))
        }
    }

    @Test
    fun eventsOutOfOrderDoNothing() {
        val m = BarStages.Machine()
        m.release(0.0); m.dropped(0.0, 0.5, false)
        assertTrue(m.isRest)
        m.press(1.0); m.release(2.0)
        m.press(2.5) // a new dictation while the last still processes
        assertEquals(Stage.Listening, m.current)
    }

    /** TalkBar's WORKING entry for a bar created mid-dictation: press then release on a resting machine. */
    @Test
    fun aBarCreatedWhileWritingStartsSteadyProcessingNotTheRestingFluid() {
        val m = BarStages.Machine()
        m.press(5.0); m.release(5.0)
        assertEquals(Stage.Processing, m.current)
        val f = m.frame(5.0, 0.5, false)!!
        assertEquals(BarStages.WIDE_WIDTH, f.halfWidth, eps)
        assertEquals(BarStages.PROCESSING_STRENGTH, f.strength, eps)
    }

    @Test
    fun steadyProcessingNeedsNoRedraw() {
        val m = BarStages.Machine()
        assertFalse(m.moving(0.0, false))
        m.press(0.0)
        assertTrue(m.moving(0.05, false))
        m.release(1.0)
        assertFalse(m.moving(1.0, false))
        assertFalse(m.moving(1.5, false))
        assertFalse(m.moving(1.0, true))
    }

    @Test
    fun theSmootherAttacksFastAndReleasesSlowly() {
        val s = BarStages.Smoother()
        val up = s.step(1.0, 0.04)
        assertTrue("$up", up > 0.6)
        val down = s.step(0.0, 0.04)
        assertTrue("$down after $up", down > 0.55 && down < up)
        repeat(200) { s.step(0.0, 0.05) }
        assertEquals(BarStages.LEVEL_FLOOR, s.value, 1e-3)
    }

    @Test
    fun theCentreColourIsTheWarmEmber() {
        val ember = BarStages.centre(BarStages.Look(BarStages.WIDE_WIDTH, 1.0, 0.0))
        assertEquals(listOf(107, 31, 9), ember.toList())
    }
}
