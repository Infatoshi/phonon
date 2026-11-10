package ai.phonon.keyboard

import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/** One model file: where it goes under the store's root, where it comes from, and what it must hash to. */
data class ModelFile(val path: String, val url: String, val sha256: String, val size: Long)

/**
 * Downloads model files into [root] with resume and sha256 checks. Plain JVM (no Android classes), so the unit
 * tests run it against a local HTTP server.
 *
 * A file is ready when it exists at its full size next to a "<name>.ok" marker holding "<sha256> <size>", written
 * only after the whole file hashed correctly; readiness never rehashes 650 MB on each start. A download goes to
 * "<name>.part" and resumes from its length with an HTTP Range request (Hugging Face answers 206 through its CDN
 * redirect); a server that ignores Range (200) restarts the file. The hash is checked once the part is complete:
 * a mismatch deletes it and the next attempt starts over.
 */
class ModelStore(
    val root: File,
    private val log: (String) -> Unit = {},
    private val open: (URL) -> HttpURLConnection = { it.openConnection() as HttpURLConnection },
    private val sleep: (Long) -> Unit = { Thread.sleep(it) },
) {
    fun file(f: ModelFile) = File(root, f.path)
    private fun part(f: ModelFile) = File(root, f.path + ".part")
    private fun marker(f: ModelFile) = File(root, f.path + ".ok")

    fun isReady(f: ModelFile): Boolean {
        val m = marker(f)
        return file(f).length() == f.size && m.exists() && m.readText().trim() == "${f.sha256} ${f.size}"
    }

    fun allReady(files: List<ModelFile>) = files.all { isReady(it) }

    /** Bytes still to fetch for [files] (a resumable part counts as fetched; a complete part leaves 0). */
    fun remaining(files: List<ModelFile>) = files.filter { !isReady(it) }.sumOf { it.size - minOf(part(it).length(), it.size) }

    class Failure(message: String) : Exception(message)

    /** [ensure]'s `allowed` turned false (the network became metered): the part stays for a later resume. */
    class Paused(val path: String) : Exception("paused: $path")

    /**
     * Fetch every file of [files] that is not ready. [progress] gets (bytes done, bytes total) over the whole list.
     * Throws [Failure] when the disk lacks space or a file still fails after [attempts] tries, and [Paused] as soon
     * as [allowed] returns false (checked before each request and every 64 KB; a pause is not a failed attempt).
     */
    fun ensure(files: List<ModelFile>, attempts: Int = 8, allowed: () -> Boolean = { true },
               progress: (Long, Long) -> Unit = { _, _ -> }) {
        root.mkdirs()
        val need = remaining(files)
        if (need > 0 && root.usableSpace < need + MARGIN) {
            throw Failure("not enough storage: need ${need shr 20} MB, ${root.usableSpace shr 20} MB free")
        }
        val total = files.sumOf { it.size }
        var before = files.filter { isReady(it) }.sumOf { it.size }
        for (f in files) {
            if (isReady(f)) continue
            fetch(f, attempts, allowed) { have -> progress(before + have, total) }
            before += f.size
            progress(before, total)
        }
    }

    private fun fetch(f: ModelFile, attempts: Int, allowed: () -> Boolean, progress: (Long) -> Unit) {
        file(f).parentFile?.mkdirs()
        marker(f).delete()
        var last: Exception? = null
        for (attempt in 1..attempts) {
            try {
                download(f, allowed, progress)
                val p = part(f)
                val got = sha256(p)
                if (got != f.sha256) {
                    p.delete()
                    throw IOException("sha256 mismatch for ${f.path}: got $got")
                }
                val dst = file(f)
                dst.delete()
                if (!p.renameTo(dst)) throw IOException("rename failed for ${f.path}")
                marker(f).writeText("${f.sha256} ${f.size}\n")
                log("DOWNLOAD ok file=${f.path} bytes=${f.size} attempt=$attempt")
                return
            } catch (e: IOException) {
                last = e
                log("DOWNLOAD retry file=${f.path} attempt=$attempt have=${part(f).length()} error=${e.message}")
                if (attempt < attempts) sleep(minOf(30_000L, 1000L shl attempt))
            }
        }
        throw Failure("download failed for ${f.path}: ${last?.message}")
    }

    /** Bring "<name>.part" to the full size, resuming from its current length. */
    private fun download(f: ModelFile, allowed: () -> Boolean, progress: (Long) -> Unit) {
        val p = part(f)
        var have = p.length()
        if (have > f.size) { p.delete(); have = 0 }
        if (have == f.size) return
        if (!allowed()) throw Paused(f.path)
        val c = connect(f.url, have)
        try {
            val code = c.responseCode
            val append = when {
                code == 206 && have > 0 -> {
                    val start = c.getHeaderField("Content-Range")?.substringAfter("bytes ")?.substringBefore('-')?.toLongOrNull()
                    if (start != have) throw IOException("bad Content-Range ${c.getHeaderField("Content-Range")} for offset $have")
                    true
                }
                code == 200 || code == 206 -> { have = 0; false }
                code == 416 && have > 0 -> { p.delete(); throw IOException("HTTP 416 at offset $have") }
                else -> throw IOException("HTTP $code for ${f.path}")
            }
            log("DOWNLOAD start file=${f.path} from=$have size=${f.size} http=$code")
            c.inputStream.use { input ->
                FileOutputStream(p, append).use { out ->
                    val buf = ByteArray(1 shl 16)
                    var lastReport = 0L
                    while (true) {
                        if (!allowed()) {  // everything written so far stays in the part
                            log("DOWNLOAD paused file=${f.path} have=$have")
                            throw Paused(f.path)
                        }
                        val n = input.read(buf)
                        if (n < 0) break
                        if (have + n > f.size) throw IOException("more bytes than expected for ${f.path}")
                        out.write(buf, 0, n)
                        have += n
                        if (have - lastReport >= REPORT_EVERY) { progress(have); lastReport = have }
                    }
                }
            }
            progress(have)
            if (have != f.size) throw IOException("short read for ${f.path}: $have of ${f.size}")
        } finally {
            c.disconnect()
        }
    }

    /** Open [url] with a Range from [offset], following redirects by hand so the Range header goes along. */
    private fun connect(url: String, offset: Long): HttpURLConnection {
        var u = URL(url)
        repeat(MAX_REDIRECTS) {
            val c = open(u)
            c.instanceFollowRedirects = false
            c.connectTimeout = 20_000
            c.readTimeout = 60_000
            c.setRequestProperty("User-Agent", "phonon-android")
            if (offset > 0) c.setRequestProperty("Range", "bytes=$offset-")
            val code = c.responseCode
            if (code in 300..399 && code != 304) {
                val loc = c.getHeaderField("Location") ?: throw IOException("HTTP $code without Location")
                c.disconnect()
                u = URL(u, loc)
            } else {
                return c
            }
        }
        throw IOException("too many redirects for $url")
    }

    companion object {
        const val MARGIN = 64L shl 20
        const val REPORT_EVERY = 4L shl 20
        const val MAX_REDIRECTS = 10

        fun sha256(file: File): String {
            val md = MessageDigest.getInstance("SHA-256")
            FileInputStream(file).use { input ->
                val buf = ByteArray(1 shl 20)
                while (true) {
                    val n = input.read(buf)
                    if (n < 0) break
                    md.update(buf, 0, n)
                }
            }
            return md.digest().joinToString("") { "%02x".format(it) }
        }
    }
}

/**
 * Whether a large download may use the current network: an unmetered one (Wi-Fi, Ethernet), or any network once the
 * user has allowed mobile data. Plain JVM: the Android side feeds [setUnmetered] from a network callback.
 */
class NetworkGate(@Volatile private var unmetered: Boolean = false, @Volatile private var cellularAllowed: Boolean = false) {
    private val lock = Object()

    fun allowed() = unmetered || cellularAllowed

    fun setUnmetered(value: Boolean) = synchronized(lock) { unmetered = value; lock.notifyAll() }

    fun allowCellular() = synchronized(lock) { cellularAllowed = true; lock.notifyAll() }

    /** Blocks until [allowed]. */
    fun awaitAllowed() = synchronized(lock) { while (!allowed()) lock.wait() }
}

/**
 * [ModelStore.ensure] that only transfers while [gate] allows: it waits for the gate first (calling [onWaiting]),
 * and a download paused by the network turning metered resumes from its part once the gate opens again.
 */
fun ModelStore.ensureWhenAllowed(files: List<ModelFile>, gate: NetworkGate, onWaiting: () -> Unit,
                                 onAllowed: () -> Unit = {}, progress: (Long, Long) -> Unit = { _, _ -> },
                                 stop: () -> Boolean = { false }) {
    while (true) {
        if (stop()) throw InterruptedException("stopped")
        if (!allowed(files, gate)) {
            onWaiting()
            gate.awaitAllowed()
            onAllowed()
        }
        try {
            ensure(files, allowed = { !stop() && gate.allowed() }, progress = progress)
            return
        } catch (_: ModelStore.Paused) {
        }
    }
}

/** Nothing left to transfer (every file ready or its part complete: hashing and the rename are local) needs no network. */
private fun ModelStore.allowed(files: List<ModelFile>, gate: NetworkGate) = gate.allowed() || remaining(files) == 0L

/** A size for people: "850 MB", "1.6 GB", in the decimal units Android's storage settings use. */
fun formatSize(bytes: Long): String {
    val mb = bytes / 1_000_000.0
    return when {
        mb >= 1000 -> String.format(java.util.Locale.US, "%.1f GB", mb / 1000)
        mb >= 10 -> "${mb.toLong()} MB"
        mb >= 0.1 -> String.format(java.util.Locale.US, "%.1f MB", mb)
        else -> "${bytes / 1000} KB"
    }
}
