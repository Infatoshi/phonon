package ai.phonon.keyboard

import ai.phonon.keyboard.BarStages.Stage
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class BarStagesTest {
    private val eps = 1e-4

    @Test
    fun igniteIsFullStrengthAtThePressBeforeAnyLevelArrives() {
        val atPress = BarStages.look(Stage.Listening, 0.0, 0.0, false)
        assertEquals(0.88, atPress.strength, eps)
        assertEquals(atPress, BarStages.look(Stage.Listening, 0.0, 0.3, false))
        assertEquals(BarStages.WIDE_WIDTH, atPress.halfWidth, eps)
    }

    @Test
    fun igniteDecaysLinearlyToTheLiveLevelIn018Seconds() {
        val level = 0.4
        val live = 0.50 + 0.38 * level
        assertEquals(0.50 + 0.38 * (0.5 + 0.5 * level), BarStages.look(Stage.Listening, 0.09, level, false).strength, eps)
        assertEquals(live, BarStages.look(Stage.Listening, 0.18, level, false).strength, eps)
        assertEquals(live, BarStages.look(Stage.Listening, 2.0, level, false).strength, eps)
        assertEquals(0.50 + 0.38 * BarStages.LEVEL_FLOOR, BarStages.look(Stage.Listening, 5.0, 0.0, false).strength, eps)
    }

    @Test
    fun gatherReaches008At12SecondsAndHolds() {
        val start = BarStages.look(Stage.Processing, 0.0, 0.5, false)
        assertEquals(0.5, start.halfWidth, eps)
        assertEquals(0.65, start.strength, eps)
        assertEquals(0.29, BarStages.look(Stage.Processing, 0.6, 0.5, false).halfWidth, eps)
        val end = BarStages.look(Stage.Processing, 1.2, 0.5, false)
        assertEquals(0.08, end.halfWidth, eps)
        assertEquals(1.0, end.strength, eps)
        assertEquals(1.0, end.heat, eps)
        for (t in listOf(1.3, 5.0, 60.0)) assertEquals("no loop, no breathing at $t", end, BarStages.look(Stage.Processing, t, 0.9, false))
    }

    @Test
    fun burstFollowsSqrtAndEndsAtZero() {
        val s = Stage.Pasted(0.08)
        val start = BarStages.look(s, 0.0, 0.5, false)
        assertEquals(0.08, start.halfWidth, eps)
        assertEquals(1.0, start.burst, eps)
        val quarter = BarStages.look(s, 0.15, 0.5, false) // p = 0.25
        assertEquals(0.08 + 0.42 * 0.5, quarter.halfWidth, eps)
        assertEquals(0.75, quarter.burst, eps)
        val end = BarStages.look(s, 0.6, 0.5, false)
        assertEquals(0.5, end.halfWidth, eps)
        assertEquals(0.0, end.burst, eps)
        assertTrue(BarStages.finished(s, 0.6, false))
        assertFalse(BarStages.finished(s, 0.59, false))
    }

    @Test
    fun aPasteBeforeTheGatherEndsBurstsFromWhereItIs() {
        val m = BarStages.Machine()
        m.press(0.0); m.release(1.0); m.pasted(1.6, false) // half way: w = 0.29
        val s = m.current as Stage.Pasted
        assertEquals(0.29, s.fromWidth, eps)
        assertEquals(0.29, m.frame(1.6, 0.5, false)!!.halfWidth, eps)
        val last = m.frame(2.19, 0.5, false)!!
        assertTrue(last.halfWidth > 0.49 && last.burst < 0.02)
        assertNull(m.frame(2.2, 0.5, false))
        assertTrue(m.isRest)
    }

    @Test
    fun endingWithNothingPastedSkipsTheBurst() {
        val m = BarStages.Machine()
        m.press(0.0); m.release(1.0); m.dropped(3.0, 0.5, false)
        val first = m.frame(3.0, 0.5, false)!!
        assertEquals(0.08, first.halfWidth, eps)
        assertEquals(1.0, first.strength, eps)
        for (t in listOf(3.05, 3.1, 3.19)) {
            val f = m.frame(t, 0.5, false)!!
            assertEquals(0.0, f.burst, 0.0)
            assertEquals("the core only fades, it does not open", 0.08, f.halfWidth, eps)
        }
        assertEquals(0.5, BarStages.look(m.current!!, 0.1, 0.5, false).strength, eps)
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
        m.press(0.0); m.release(1.0); m.pasted(2.0, true)
        val s = m.current!!
        for (t in listOf(0.0, 0.1, 0.29)) assertEquals(BarStages.REDUCED_WIDTH, BarStages.look(s, t, 0.5, true).halfWidth, eps)
        assertEquals(0.5, BarStages.look(s, 0.15, 0.5, true).burst, eps)
        assertTrue(BarStages.finished(s, 0.3, true))
        assertFalse(BarStages.finished(s, 0.29, true))
        // reduced motion's steady core needs no redraw once it is lit
        val held = BarStages.Machine()
        held.press(0.0); held.release(1.0)
        assertFalse(held.moving(1.1, true))
    }

    @Test
    fun eventsOutOfOrderDoNothing() {
        val m = BarStages.Machine()
        m.pasted(0.0, false); m.release(0.0); m.dropped(0.0, 0.5, false)
        assertTrue(m.isRest)
        m.press(1.0); m.pasted(1.1, false) // not processing yet
        assertEquals(Stage.Listening, m.current)
        m.release(2.0)
        m.press(2.5) // a new dictation while the last still processes
        assertEquals(Stage.Listening, m.current)
        m.pasted(3.0, false) // the old one's late paste must not light the new one
        assertEquals(Stage.Listening, m.current)
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
    fun theCentreColourIsTheEmberAtRestAndOrangeInTheBurst() {
        val ember = BarStages.centre(BarStages.Look(0.5, 1.0, 0.0, 0.0))
        assertEquals(listOf(107, 31, 9), ember.toList())
        val core = BarStages.centre(BarStages.Look(0.08, 1.0, 1.0, 0.0))
        assertEquals(listOf(227, 101, 29), core.toList())
        val burst = BarStages.centre(BarStages.Look(0.3, 0.0, 0.0, 1.0))
        assertEquals(listOf(255, 170, 100), burst.toList())
    }
}
