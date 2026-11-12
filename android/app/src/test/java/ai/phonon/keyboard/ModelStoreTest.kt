package ai.phonon.keyboard

import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test
import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import java.nio.file.Files
import java.security.MessageDigest
import kotlin.concurrent.thread
import kotlin.random.Random

/** ModelStore against a local server shaped like Hugging Face: /resolve/x redirects (302) to /cdn/x, which
 *  honours Range with 206 unless told not to, and can drop the connection part way through. */
class ModelStoreTest {
    private val data = Random(7).nextBytes(3_000_000)
    private val sha = MessageDigest.getInstance("SHA-256").digest(data).joinToString("") { "%02x".format(it) }
    private lateinit var server: ServerSocket
    private lateinit var root: File
    private var honourRange = true
    private var dropAfter = -1          // bytes to send before closing, on the next request only
    private val ranges = java.util.Collections.synchronizedList(mutableListOf<String?>())
    @Volatile private var served = 0L

    @Before
    fun up() {
        root = Files.createTempDirectory("models").toFile()
        server = ServerSocket(0, 50, InetAddress.getByName("127.0.0.1"))
        thread(isDaemon = true) {
            while (!server.isClosed) {
                val sock = try { server.accept() } catch (_: Exception) { break }
                sock.use { serve(it) }
            }
        }
    }

    /** One request per connection (Connection: close), as HttpURLConnection is happy to do. */
    private fun serve(sock: Socket) {
        val input = BufferedReader(InputStreamReader(sock.getInputStream(), Charsets.ISO_8859_1))
        val path = input.readLine()?.split(' ')?.getOrNull(1) ?: return
        var range: String? = null
        while (true) {
            val h = input.readLine() ?: break
            if (h.isEmpty()) break
            if (h.startsWith("Range:", ignoreCase = true)) range = h.substringAfter(':').trim()
        }
        val out = sock.getOutputStream()
        fun head(status: String, vararg headers: String) =
            out.write(("HTTP/1.1 $status\r\n" + headers.joinToString("") { "$it\r\n" } + "Connection: close\r\n\r\n")
                .toByteArray(Charsets.ISO_8859_1))
        if (path.startsWith("/resolve/")) {
            head("302 Found", "Location: /cdn/" + path.substringAfterLast('/'), "Content-Length: 0")
            return
        }
        ranges += range
        val from = if (honourRange && range != null) range.substringAfter("bytes=").substringBefore('-').toInt() else 0
        if (from > 0) head("206 Partial Content", "Content-Range: bytes $from-${data.size - 1}/${data.size}",
            "Content-Length: ${data.size - from}")
        else head("200 OK", "Content-Length: ${data.size}")
        val n = if (dropAfter >= 0) minOf(dropAfter, data.size - from) else data.size - from
        dropAfter = -1
        served += n  // counted before the write: the client can finish reading before this thread resumes
        try {
            out.write(data, from, n)
            out.flush()
        } catch (_: Exception) {
        }
    }

    @After
    fun down() {
        server.close()
        root.deleteRecursively()
    }

    private fun model(hash: String = sha) =
        ModelFile("parakeet/encoder.int8.onnx.data", "http://127.0.0.1:${server.localPort}/resolve/x.data", hash, data.size.toLong())

    private fun store() = ModelStore(root, sleep = {})

    @Test
    fun freshDownloadIsVerifiedAndMarkedReady() {
        val f = model()
        val s = store()
        assertFalse(s.isReady(f))
        var last = 0L to 0L
        s.ensure(listOf(f)) { d, t -> last = d to t }
        assertTrue(s.isReady(f))
        assertArrayEquals(data, s.file(f).readBytes())
        assertFalse(File(root, f.path + ".part").exists())
        assertEquals(data.size.toLong() to data.size.toLong(), last)
        // a second ensure fetches nothing
        served = 0
        s.ensure(listOf(f))
        assertEquals(0L, served)
    }

    @Test
    fun aPartialFileResumesThroughTheRedirectWithARange() {
        val f = model()
        File(root, "parakeet").mkdirs()
        File(root, f.path + ".part").writeBytes(data.copyOf(1_000_000))
        store().ensure(listOf(f))
        assertEquals(listOf<String?>("bytes=1000000-"), ranges)
        assertEquals(2_000_000L, served)
        assertArrayEquals(data, store().file(f).readBytes())
    }

    @Test
    fun aServerThatIgnoresRangeRestartsTheFile() {
        honourRange = false
        val f = model()
        File(root, "parakeet").mkdirs()
        File(root, f.path + ".part").writeBytes(data.copyOf(1_000_000))
        store().ensure(listOf(f))
        assertEquals(data.size.toLong(), served)
        assertArrayEquals(data, store().file(f).readBytes())
    }

    @Test
    fun aDroppedConnectionResumesOnTheNextAttempt() {
        dropAfter = 1_234_567
        val f = model()
        store().ensure(listOf(f))
        assertEquals(listOf(null, "bytes=1234567-"), ranges)
        assertArrayEquals(data, store().file(f).readBytes())
    }

    @Test
    fun aCorruptPartIsDiscardedAndRefetched() {
        val f = model()
        File(root, "parakeet").mkdirs()
        File(root, f.path + ".part").writeBytes(ByteArray(1_000_000))  // zeros, not the file's first MB
        store().ensure(listOf(f))
        assertEquals(listOf("bytes=1000000-", null), ranges)
        assertArrayEquals(data, store().file(f).readBytes())
        assertTrue(store().isReady(f))
    }

    @Test
    fun aWrongHashFailsAndLeavesNothingReady() {
        val f = model("0".repeat(64))
        try {
            store().ensure(listOf(f), attempts = 2)
            fail("expected a failure")
        } catch (e: ModelStore.Failure) {
            assertTrue(e.message!!.contains("sha256 mismatch"))
        }
        assertFalse(store().isReady(f))
        assertFalse(store().file(f).exists())
    }

    @Test
    fun aMeteredNetworkPausesMidFileAndTheResumeFinishesIt() {
        val f = model()
        var checks = 0
        try {
            store().ensure(listOf(f), allowed = { checks++ < 10 })  // the request plus 9 chunks of 64 KB
            fail("expected a pause")
        } catch (e: ModelStore.Paused) {
            assertEquals(f.path, e.path)
        }
        val kept = File(root, f.path + ".part").length()
        assertTrue("kept $kept", kept in 1..9L * 65536)  // 9 reads of at most 64 KB each
        assertFalse(store().isReady(f))
        store().ensure(listOf(f))
        assertEquals(listOf(null, "bytes=$kept-"), ranges)
        assertArrayEquals(data, store().file(f).readBytes())
    }

    @Test
    fun theGatedDownloadWaitsWithoutTransferringUntilAllowed() {
        val f = model()
        val gate = NetworkGate(unmetered = false)
        var waited = 0
        val t = thread { store().ensureWhenAllowed(listOf(f), gate, onWaiting = { waited++ }) }
        Thread.sleep(500)
        assertTrue(t.isAlive)
        assertEquals(1, waited)
        assertEquals(0L, served)
        gate.allowCellular()  // the user's tap
        t.join(10_000)
        assertFalse(t.isAlive)
        assertTrue(store().isReady(f))
        assertArrayEquals(data, store().file(f).readBytes())
    }

    @Test
    fun anUnmeteredNetworkOpensTheGateAndAMeteredOneClosesIt() {
        val gate = NetworkGate()
        assertFalse(gate.allowed())
        gate.setUnmetered(true)
        assertTrue(gate.allowed())
        gate.setUnmetered(false)
        assertFalse(gate.allowed())
        gate.allowCellular()
        assertTrue(gate.allowed())
        gate.setUnmetered(false)
        assertTrue(gate.allowed())  // the override outlasts network changes
    }

    @Test
    fun aCompletePartFinishesWithoutTheNetwork() {
        val f = model()
        File(root, "parakeet").mkdirs()
        File(root, f.path + ".part").writeBytes(data)  // e.g. killed during the hash, or paused on the last chunk
        var waited = 0
        store().ensureWhenAllowed(listOf(f), NetworkGate(), onWaiting = { waited++ })
        assertEquals(0, waited)
        assertEquals(0L, served)
        assertTrue(store().isReady(f))
    }

    @Test
    fun theGateOpeningIsReported() {
        val f = model()
        val gate = NetworkGate()
        var allowed = 0
        val t = thread { store().ensureWhenAllowed(listOf(f), gate, onWaiting = {}, onAllowed = { allowed++ }) }
        Thread.sleep(300)
        assertEquals(0, allowed)
        gate.setUnmetered(true)
        t.join(10_000)
        assertEquals(1, allowed)
        assertTrue(store().isReady(f))
    }

    @Test
    fun aReadyFileNeedsNoNetwork() {
        val f = model()
        store().ensure(listOf(f))
        served = 0
        var waited = 0
        store().ensureWhenAllowed(listOf(f), NetworkGate(), onWaiting = { waited++ })
        assertEquals(0, waited)
        assertEquals(0L, served)
    }

    @Test
    fun theManifestIsPinnedAndHashed() {
        for (f in Models.ASR + Models.REFINER) {
            assertTrue(f.path, Regex("^[0-9a-f]{64}$").matches(f.sha256))
            assertTrue(f.path, f.size > 0)
            assertTrue(f.url, Regex("^https://huggingface\\.co/[^/]+/[^/]+/resolve/[0-9a-f]{40}/[^/]+$").matches(f.url))
            assertEquals(f.path.substringAfterLast('/'), f.url.substringAfterLast('/').let {
                if (f == Models.REFINER) "refiner.gguf" else it
            })
        }
    }

    @Test
    fun aFileWithoutItsMarkerIsNotReady() {
        val f = model()
        store().ensure(listOf(f))
        File(root, f.path + ".ok").delete()
        assertFalse(store().isReady(f))
        assertEquals(data.size.toLong(), store().remaining(listOf(f)))
    }

    @Test
    fun sizesReadAsPeopleSayThem() {
        // the two models together: Parakeet (661 MB) and the refiner (988 MB)
        assertEquals("1.6 GB", formatSize(661_232_435L + 988_453_696L))
        assertEquals("661 MB", formatSize(661_232_435L))
        assertEquals("1.6 MB", formatSize(1_572_864L))
        assertEquals("4 KB", formatSize(4_096L))
    }
}
