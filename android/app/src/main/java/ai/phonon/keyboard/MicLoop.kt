package ai.phonon.keyboard

/** The microphone read loop, apart from AudioRecord so JVM tests can drive it. AudioRecord.read returns a byte
 *  count, or a negative error code (ERROR -1, ERROR_BAD_VALUE -2, ERROR_INVALID_OPERATION -3, ERROR_DEAD_OBJECT -6:
 *  the mic was taken, e.g. by a phone call). An error ends the loop; looping on it spun the keyboard forever. */
object MicLoop {
    /** Reads into `buf` while `keepGoing()`, handing each chunk to `onData`. Returns the read error code that ended
     *  it, or null when `keepGoing()` turned false. */
    fun run(buf: ByteArray, keepGoing: () -> Boolean, read: (ByteArray) -> Int, onData: (ByteArray, Int) -> Unit): Int? {
        while (keepGoing()) {
            val n = read(buf)
            if (n < 0) return n
            if (n > 0) onData(buf, n)
        }
        return null
    }

    /** What the keyboard says when the mic stops working mid-dictation or cannot open. */
    const val BUSY = "Microphone unavailable (a call or another app may be using it)"
}
