package ai.phonon.keyboard

import ai.phonon.keyboard.BarStages.Stage
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
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
        // the held core is brighter still, so the let-go reads as a change
        assertTrue(BarStages.look(Stage.Processing, BarStages.GATHER_SECONDS, 0.5, false).strength > loudest)
    }

    @Test
    fun igniteIsFullStrengthAtThePressBeforeAnyLevelArrives() {
        val atPress = BarStages.look(Stage.Listening, 0.0, 0.0, false)
        assertEquals(BarStages.FULL_STRENGTH, atPress.strength, eps)
        assertEquals(atPress, BarStages.look(Stage.Listening, 0.0, 0.3, false))
        assertEquals(BarStages.WIDE_WIDTH, atPress.halfWidth, eps)
    }

    @Test
    fun igniteDecaysLinearlyToTheLiveLevelIn018Seconds() {
        val level = 0.4
        // half way through the ignite: half full energy, half the live level
        assertEquals(listening(0.5 + 0.5 * level), BarStages.look(Stage.Listening, 0.09, level, false).strength, eps)
        assertEquals(listening(level), BarStages.look(Stage.Listening, 0.18, level, false).strength, eps)
        assertEquals(listening(level), BarStages.look(Stage.Listening, 2.0, level, false).strength, eps)
        assertEquals(listening(BarStages.LEVEL_FLOOR), BarStages.look(Stage.Listening, 5.0, 0.0, false).strength, eps)
    }

    @Test
    fun gatherReaches008At12SecondsAndHolds() {
        val start = BarStages.look(Stage.Processing, 0.0, 0.5, false)
        assertEquals(0.5, start.halfWidth, eps)
        assertEquals(BarStages.GATHER_START_STRENGTH, start.strength, eps)
        assertEquals(0.29, BarStages.look(Stage.Processing, 0.6, 0.5, false).halfWidth, eps)
        val end = BarStages.look(Stage.Processing, 1.2, 0.5, false)
        assertEquals(0.08, end.halfWidth, eps)
        assertEquals(BarStages.GATHER_END_STRENGTH, end.strength, eps)
        assertEquals(1.0, end.heat, eps)
        for (t in listOf(1.3, 5.0, 60.0)) assertEquals("no loop, no breathing at $t", end, BarStages.look(Stage.Processing, t, 0.9, false))
    }

    @Test
    fun anEndBeforeTheGatherFinishesFadesFromWhereItIs() {
        val m = BarStages.Machine()
        m.press(0.0); m.release(1.0); m.dropped(1.6, 0.5, false) // half way: w = 0.29
        val s = m.current as Stage.Dropped
        assertEquals(0.29, s.fromWidth, eps)
        assertEquals(0.29, m.frame(1.6, 0.5, false)!!.halfWidth, eps)
        assertNull(m.frame(1.8, 0.5, false))
        assertTrue(m.isRest)
    }

    @Test
    fun theEndFadesTheCoreWithNoPulse() {
        val m = BarStages.Machine()
        m.press(0.0); m.release(1.0); m.dropped(3.0, 0.5, false)
        val first = m.frame(3.0, 0.5, false)!!
        assertEquals(BarStages.TIGHT_WIDTH, first.halfWidth, eps)
        assertEquals(BarStages.GATHER_END_STRENGTH, first.strength, eps)
        for (t in listOf(3.05, 3.1, 3.19)) {
            val f = m.frame(t, 0.5, false)!!
            assertEquals("no colour change", first.heat, f.heat, 0.0)
            assertTrue("never brighter", f.strength <= first.strength)
            assertEquals("the core only fades, it does not open", BarStages.TIGHT_WIDTH, f.halfWidth, eps)
        }
        assertEquals(BarStages.GATHER_END_STRENGTH / 2,
            BarStages.look(m.current!!, BarStages.DROP_SECONDS / 2, 0.5, false).strength, eps)
        assertNull(m.frame(3.2, 0.5, false))
        assertTrue(m.isRest)
    }

    @Test
    fun aFailureWhileListeningFadesToo() {
        val m = BarStages.Machine()
        m.press(0.0); m.dropped(0.05, 0.3, false)
        assertTrue(m.current is Stage.Dropped)
        assertNull(m.frame(0.30, 0.3, false))
    }

    @Test
    fun reducedMotionKeepsTheWidthConstant() {
        for (t in listOf(0.0, 0.3, 1.2, 4.0))
            assertEquals(BarStages.REDUCED_WIDTH, BarStages.look(Stage.Processing, t, 0.5, true).halfWidth, eps)
        val m = BarStages.Machine()
        m.press(0.0); m.release(1.0); m.dropped(2.0, 0.5, true)
        val s = m.current!!
        for (t in listOf(0.0, 0.1, 0.19)) assertEquals(BarStages.REDUCED_WIDTH, BarStages.look(s, t, 0.5, true).halfWidth, eps)
        assertTrue(BarStages.finished(s, 0.2))
        assertFalse(BarStages.finished(s, 0.19))
        // reduced motion's steady core needs no redraw once it is lit
        val held = BarStages.Machine()
        held.press(0.0); held.release(1.0)
        assertFalse(held.moving(1.1, true))
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
    fun aBarCreatedWhileWritingStartsTheGatherNotTheRestingFluid() {
        val m = BarStages.Machine()
        m.press(5.0); m.release(5.0)
        assertEquals(Stage.Processing, m.current)
        val f = m.frame(5.0, 0.5, false)!!
        assertEquals(BarStages.WIDE_WIDTH, f.halfWidth, eps)
        assertEquals(BarStages.GATHER_START_STRENGTH, f.strength, eps)
    }

    @Test
    fun holdingTheGatheredCoreNeedsNoRedraw() {
        val m = BarStages.Machine()
        assertFalse(m.moving(0.0, false))
        m.press(0.0)
        assertTrue(m.moving(0.1, false))
        m.release(1.0)
        assertTrue(m.moving(1.5, false))
        assertFalse(m.moving(2.3, false))
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
    fun theCentreColourIsTheEmberAtRestAndTheBrightCoreWhenGathered() {
        val ember = BarStages.centre(BarStages.Look(0.5, 1.0, 0.0))
        assertEquals(listOf(107, 31, 9), ember.toList())
        val core = BarStages.centre(BarStages.Look(0.08, 1.0, 1.0))
        assertEquals(listOf(227, 101, 29), core.toList())
    }
}
