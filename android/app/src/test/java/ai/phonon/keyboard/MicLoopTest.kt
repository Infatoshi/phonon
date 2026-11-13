package ai.phonon.keyboard

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class MicLoopTest {
    private val buf = ByteArray(8)

    /** A phone call takes the mic: read returns ERROR_INVALID_OPERATION forever. The loop must end, not spin. */
    @Test
    fun errorCodeEndsTheLoop() {
        var reads = 0
        val got = mutableListOf<Int>()
        val err = MicLoop.run(buf, { true }, { reads++; if (reads <= 2) 4 else -3 }) { _, n -> got.add(n) }
        assertEquals(-3, err)
        assertEquals(3, reads)
        assertEquals(listOf(4, 4), got)
    }

    @Test
    fun deadObjectEndsTheLoop() {
        assertEquals(-6, MicLoop.run(buf, { true }, { -6 }) { _, _ -> error("no data expected") })
    }

    @Test
    fun stopsWhenRecordingEndsAndSkipsEmptyReads() {
        var reads = 0
        val got = mutableListOf<Int>()
        val err = MicLoop.run(buf, { reads < 4 }, { reads++; if (reads % 2 == 0) 0 else 8 }) { _, n -> got.add(n) }
        assertNull(err)
        assertEquals(listOf(8, 8), got)
    }
}
