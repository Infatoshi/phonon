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

    /** Bytes still to fetch for [files] (a resumable part counts as fetched). */
    fun remaining(files: List<ModelFile>) = files.filter { !isReady(it) }.sumOf { it.size - minOf(part(it).length(), it.size) }

    class Failure(message: String) : Exception(message)

    /**
     * Fetch every file of [files] that is not ready. [progress] gets (bytes done, bytes total) over the whole list.
     * Throws [Failure] when the disk lacks space or a file still fails after [attempts] tries.
     */
    fun ensure(files: List<ModelFile>, attempts: Int = 8, progress: (Long, Long) -> Unit = { _, _ -> }) {
        root.mkdirs()
        val need = remaining(files)
        if (need > 0 && root.usableSpace < need + MARGIN) {
            throw Failure("not enough storage: need ${need shr 20} MB, ${root.usableSpace shr 20} MB free")
        }
        val total = files.sumOf { it.size }
        var before = files.filter { isReady(it) }.sumOf { it.size }
        for (f in files) {
            if (isReady(f)) continue
            fetch(f, attempts) { have -> progress(before + have, total) }
            before += f.size
            progress(before, total)
        }
    }

    private fun fetch(f: ModelFile, attempts: Int, progress: (Long) -> Unit) {
        file(f).parentFile?.mkdirs()
        marker(f).delete()
        var last: Exception? = null
        for (attempt in 1..attempts) {
            try {
                download(f, progress)
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
    private fun download(f: ModelFile, progress: (Long) -> Unit) {
        val p = part(f)
        var have = p.length()
        if (have > f.size) { p.delete(); have = 0 }
        if (have == f.size) return
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
