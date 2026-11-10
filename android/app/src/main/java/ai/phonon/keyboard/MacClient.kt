package ai.phonon.keyboard

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString.Companion.toByteString
import java.util.concurrent.TimeUnit

/** Where the Mac is and its token, in this app's private preferences. The token is never logged. */
class MacSettings(context: Context) {
    private val prefs = context.applicationContext.getSharedPreferences("phonon_mac", Context.MODE_PRIVATE)

    var url: String
        get() = prefs.getString("url", "") ?: ""
        set(value) { prefs.edit().putString("url", value).apply() }
    var token: String
        get() = prefs.getString("token", "") ?: ""
        set(value) { prefs.edit().putString("token", value).apply() }
    /** Ask the Mac for its corrected text while speaking (`draft`), shown above the button. */
    var live: Boolean
        get() = prefs.getBoolean("live", true)
        set(value) { prefs.edit().putBoolean("live", value).apply() }

    val configured: Boolean get() = MacProtocol.normalizeUrl(url) != null && token.isNotBlank()
}

/**
 * One dictation through the Mac (docs/websocket-api.md). It connects, says hello, sends `start` and the audio
 * captured so far, then streams the rest as it arrives; [finish] asks for the result. Every callback runs on the
 * main thread, and exactly one of [Listener.onFinal] or [Listener.onFailed] is called, once.
 */
class MacDictation(
    private val settings: MacSettings,
    private val packageName: String?,
    private val textbox: Triple<String?, String?, String?>?,
    private val listener: Listener,
) {
    interface Listener {
        fun onConnected() {}
        fun onPreview(text: String, corrected: Boolean) {}
        fun onFinal(text: String, refined: Boolean)
        /** The Mac could not produce the text; [reason] is for the user. */
        fun onFailed(reason: String)
    }

    private val main = Handler(Looper.getMainLooper())
    private val id = "d" + System.currentTimeMillis().toString(36)
    private var socket: WebSocket? = null
    /** Audio before `start` went out; null once it has been sent. Main thread only. */
    private var pending: java.io.ByteArrayOutputStream? = java.io.ByteArrayOutputStream()
    private var started = false
    private var finishRequested = false
    private var ended = false
    private val ping = object : Runnable {
        override fun run() {
            socket?.send(MacProtocol.ping())
            main.postDelayed(this, PING_MS)
        }
    }
    private val connectTimeout = Runnable { fail("could not reach the Mac") }
    private var resultTimeout: Runnable? = null

    fun connect() {
        val url = MacProtocol.normalizeUrl(settings.url) ?: return fail("no Mac address set")
        main.postDelayed(connectTimeout, CONNECT_MS)
        val request = Request.Builder().url(url.replaceFirst("ws://", "http://").replaceFirst("wss://", "https://")).build()
        socket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                webSocket.send(MacProtocol.hello(settings.token, CLIENT_NAME))
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                val reply = MacProtocol.parse(text)
                main.post { handle(reply) }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.i(TAG, "MAC_FAIL ${t.javaClass.simpleName}: ${t.message}")
                main.post { fail("could not reach the Mac") }
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                main.post { fail("the Mac closed the connection") }
            }
        })
    }

    /** 16 kHz mono PCM16 little-endian, as recorded. Main thread. */
    fun audio(pcm: ByteArray) {
        if (ended) return
        val buffer = pending
        if (buffer != null) buffer.write(pcm) else socket?.send(pcm.toByteString())
    }

    /** The user is done speaking. Main thread. */
    fun finish() {
        if (ended) return
        finishRequested = true
        if (started) sendFinish()
    }

    /** Drop the dictation (the user cancelled, or the phone took over). Main thread. */
    fun cancel() {
        if (ended) return
        ended = true
        if (started) socket?.send(MacProtocol.cancel(id))
        close()
    }

    private fun sendFinish() {
        socket?.send(MacProtocol.finish(id))
        val timeout = Runnable { fail("the Mac took too long") }
        resultTimeout = timeout
        main.postDelayed(timeout, RESULT_MS)
    }

    private fun handle(reply: MacProtocol.Reply) {
        if (ended) return
        when (reply) {
            is MacProtocol.Reply.Ready -> {
                MacProtocol.unavailable(reply)?.let { return fail(it) }
                main.removeCallbacks(connectTimeout)
                main.postDelayed(ping, PING_MS)
                val (before, selected, after) = textbox ?: Triple(null, null, null)
                val ws = socket ?: return
                ws.send(MacProtocol.start(id, packageName, before, selected, after, settings.live))
                // Messages on the connection are handled in order: the audio follows `start` at once.
                pending?.toByteArray()?.takeIf { it.isNotEmpty() }?.let { ws.send(it.toByteString()) }
                pending = null
                listener.onConnected()
            }
            is MacProtocol.Reply.Started -> if (reply.id == id) {
                started = true
                if (finishRequested) sendFinish()
            }
            is MacProtocol.Reply.Preview -> if (reply.id == id && !finishRequested) {
                listener.onPreview(reply.text, reply.corrected)
            }
            is MacProtocol.Reply.Final -> if (reply.id == id) {
                ended = true
                close()
                listener.onFinal(reply.text, reply.refined)
            }
            is MacProtocol.Reply.Error -> if (reply.id == null || reply.id == id) fail(MacProtocol.explain(reply))
            else -> {}
        }
    }

    private fun fail(reason: String) {
        if (ended) return
        ended = true
        close()
        listener.onFailed(reason)
    }

    private fun close() {
        main.removeCallbacks(ping)
        main.removeCallbacks(connectTimeout)
        resultTimeout?.let { main.removeCallbacks(it) }
        socket?.close(1000, null)
        socket = null
        pending = null
    }

    companion object {
        const val CONNECT_MS = 4_000L
        const val RESULT_MS = 120_000L
        const val PING_MS = 30_000L
        const val CLIENT_NAME = "phonon-android"

        /** One client for every dictation: its connection pool and threads are shared. */
        private val client: OkHttpClient by lazy {
            OkHttpClient.Builder()
                .connectTimeout(CONNECT_MS, TimeUnit.MILLISECONDS)
                .readTimeout(0, TimeUnit.MILLISECONDS)
                .build()
        }

        /** Connect, say hello and report the Mac's `ready`, for the settings screen's Test button. */
        fun test(settings: MacSettings, done: (String) -> Unit) {
            val main = Handler(Looper.getMainLooper())
            val url = MacProtocol.normalizeUrl(settings.url) ?: return done("Enter the Mac's address first.")
            val request = Request.Builder().url(url.replaceFirst("ws://", "http://").replaceFirst("wss://", "https://")).build()
            var answered = false
            fun answer(text: String) = main.post { if (!answered) { answered = true; done(text) } }
            client.newWebSocket(request, object : WebSocketListener() {
                override fun onOpen(webSocket: WebSocket, response: Response) {
                    webSocket.send(MacProtocol.hello(settings.token, CLIENT_NAME))
                }

                override fun onMessage(webSocket: WebSocket, text: String) {
                    when (val reply = MacProtocol.parse(text)) {
                        is MacProtocol.Reply.Ready -> answer(MacProtocol.unavailable(reply)?.let { "Connected, but $it." }
                            ?: "Connected. The Mac is ready.")
                        is MacProtocol.Reply.Error -> answer(MacProtocol.explain(reply).replaceFirstChar { it.uppercase() } + ".")
                        else -> return
                    }
                    webSocket.close(1000, null)
                }

                override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                    answer("Could not connect: ${t.message ?: t.javaClass.simpleName}")
                }
            })
        }
    }
}
