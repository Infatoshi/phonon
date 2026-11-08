package ai.phonon.keyboard

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PassesTest {
    private val rate = 16000

    @Test
    fun shortAudioIsOnePass() {
        assertEquals(listOf(0 to rate * 45), Passes.split(rate * 45, rate) { _, _ -> 1.0 })
        assertEquals(listOf(0 to 0), Passes.split(0, rate) { _, _ -> 1.0 })
    }

    @Test
    fun longAudioIsCutInThePausesAndEveryPassFits() {
        // 150 s of "speech" with 200 ms pauses at 55 s and 110 s
        val n = rate * 150
        val pauses = listOf(55 * rate, 110 * rate)
        val loud = { from: Int, to: Int ->
            (from until to).count { i -> pauses.none { p -> i >= p && i < p + rate / 5 } }.toDouble()
        }
        val passes = Passes.split(n, rate, loud)
        assertEquals(3, passes.size)
        assertEquals(0, passes.first().first)
        assertEquals(n, passes.last().second)
        passes.zipWithNext().forEach { (a, b) -> assertEquals(a.second, b.first) }
        passes.forEach { (a, b) -> assertTrue(b - a <= Passes.MAX_SECONDS * rate) }
        assertTrue(passes[0].second in pauses[0] until pauses[0] + rate / 5)
        assertTrue(passes[1].second in pauses[1] until pauses[1] + rate / 5)
    }

    @Test
    fun noPauseStillFits() {
        val passes = Passes.split(rate * 1000, rate) { _, _ -> 1.0 }
        passes.forEach { (a, b) -> assertTrue(b - a <= Passes.MAX_SECONDS * rate) }
        assertEquals(rate * 1000, passes.last().second)
    }
}
