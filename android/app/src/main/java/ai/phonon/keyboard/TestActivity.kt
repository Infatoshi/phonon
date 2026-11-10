package ai.phonon.keyboard

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.provider.Settings
import android.text.InputType
import android.view.View
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import com.google.zxing.client.android.Intents
import com.google.zxing.integration.android.IntentIntegrator

/** Launcher: asks for the mic once (the IME cannot prompt), starts the model download with its progress shown,
 *  sets up the Mac for Phonon voice (VoiceIME), and gives the IME a text field to type into. */
class TestActivity : Activity() {
    private lateinit var settings: MacSettings
    private lateinit var url: EditText
    private lateinit var token: EditText
    private lateinit var result: TextView
    private var progress: TextView? = null
    private var cellular: Button? = null
    private val onModels: (ModelFetch.State) -> Unit = { s ->
        runOnUiThread {
            progress?.text = s.message
            cellular?.visibility = if (s.waitingForWifi) View.VISIBLE else View.GONE
        }
    }

    override fun onCreate(b: Bundle?) {
        super.onCreate(b)
        requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1)
        settings = MacSettings(this)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(32, 96, 32, 32) }

        root.addView(heading("Phonon voice: dictate through your Mac"))
        root.addView(note("On the Mac: Phonon › Settings › Advanced › WebSocket API. Turn it on, then scan its " +
            "pairing code, or copy the address and token here."))
        root.addView(Button(this).apply {
            text = "Scan pairing code"
            setOnClickListener {
                IntentIntegrator(this@TestActivity)
                    .setDesiredBarcodeFormats(IntentIntegrator.QR_CODE)
                    .setPrompt("Point at the code in Phonon's Settings on the Mac")
                    .setOrientationLocked(false)
                    .setBeepEnabled(false)
                    .initiateScan()
            }
        })
        url = EditText(this).apply {
            id = R.id.mac_url
            hint = "ws://192.168.1.20:8377"
            setText(settings.url)
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
            isSingleLine = true
        }
        root.addView(url)
        token = EditText(this).apply {
            id = R.id.mac_token
            hint = "token"
            setText(settings.token)
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
            isSingleLine = true
            // isSingleLine resets the transformation: mask the token again
            transformationMethod = android.text.method.PasswordTransformationMethod.getInstance()
        }
        root.addView(token)
        val live = CheckBox(this).apply {
            text = "Show the corrected text while speaking"
            isChecked = settings.live
            setOnCheckedChangeListener { _, on -> settings.live = on }
        }
        root.addView(live)
        val fallbackNote = TextView(this)
        root.addView(CheckBox(this).apply {
            text = "Fall back to the on-device model when the Mac can't take a dictation"
            isChecked = ModelFetch.localEnabled(this@TestActivity)
            setOnCheckedChangeListener { box, on ->
                if (on) {
                    ModelFetch.enableLocal(this@TestActivity)
                    fallbackNote.text = "Downloading the on-device models…"
                    return@setOnCheckedChangeListener
                }
                android.app.AlertDialog.Builder(this@TestActivity)
                    .setTitle("Remove the on-device models?")
                    .setMessage("Phonon voice then only works while your Mac can take dictations, and the " +
                        "Phonon keyboard's own dictation stops working. Turning this back on downloads them again.")
                    .setPositiveButton("Remove") { _, _ ->
                        fallbackNote.text = "Removing…"
                        kotlin.concurrent.thread(name = "phonon-remove-models") {
                            val freed = ModelFetch.disableLocal(this@TestActivity)
                            runOnUiThread {
                                fallbackNote.text = if (freed > 0) {
                                    "Removed the on-device models from this phone, reclaimed ${formatSize(freed)}."
                                } else {
                                    "On-device fallback is off. No models were on this phone."
                                }
                            }
                        }
                    }
                    .setNegativeButton("Keep") { _, _ -> box.isChecked = true }
                    .setOnCancelListener { box.isChecked = true }
                    .show()
            }
        })
        root.addView(fallbackNote)
        result = TextView(this).apply { id = R.id.mac_result }
        root.addView(Button(this).apply {
            id = R.id.mac_test
            text = "Save and test"
            setOnClickListener { saveAndTest() }
        })
        root.addView(result)
        root.addView(note("Then let your keyboard use it:\n" +
            "1. Turn on \"Phonon voice\" below.\n" +
            "2. FUTO Keyboard › Settings › Voice input: turn on \"Disable built-in voice input\".\n" +
            "3. Tap the keyboard's mic key: Phonon voice opens and listens. Tap Done; the text is typed and your " +
            "keyboard comes back.\n" +
            "Without a Mac (or when it is busy) the speech model on this phone transcribes instead, once it has " +
            "downloaded."))
        root.addView(Button(this).apply {
            text = "Turn on Phonon voice (input method settings)"
            setOnClickListener { startActivity(Intent(Settings.ACTION_INPUT_METHOD_SETTINGS)) }
        })

        root.addView(heading("On-device models"))
        progress = TextView(this).apply { id = R.id.models }
        root.addView(progress)
        cellular = Button(this).apply {
            id = R.id.cellular
            text = "Download the text model on mobile data (988 MB)"
            visibility = View.GONE
            setOnClickListener { ModelFetch.allowCellular(this@TestActivity) }
        }
        root.addView(cellular)
        root.addView(heading("Test field"))
        root.addView(EditText(this).apply {
            id = R.id.field
            hint = "type or hold to talk"
            minLines = 3
        })
        setContentView(ScrollView(this).apply { addView(root) })
        ModelFetch.addListener(onModels)
        ModelFetch.start(this)
        fillFromLink(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        fillFromLink(intent)
    }

    private fun saveAndTest() {
        settings.url = MacProtocol.normalizeUrl(url.text.toString()) ?: url.text.toString().trim()
        settings.token = token.text.toString().trim()
        url.setText(settings.url)
        result.text = "Connecting…"
        MacDictation.test(settings) { result.text = it }
    }

    /** A scan from "Scan pairing code": the user asked for it, so it is saved and tested at once. */
    @Deprecated("Activity result API; IntentIntegrator needs a plain Activity")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        val scan = IntentIntegrator.parseActivityResult(requestCode, resultCode, data)
        if (scan == null) {
            @Suppress("DEPRECATION") super.onActivityResult(requestCode, resultCode, data)
            return
        }
        val text = scan.contents ?: run {
            if (data?.getStringExtra(Intents.Scan.MISSING_CAMERA_PERMISSION) != null) {
                result.text = "Phonon needs the camera to scan the code; or copy the address and token."
            }
            return
        }
        val pairing = MacProtocol.parsePairing(text) ?: run {
            result.text = "That is not a Phonon pairing code."
            return
        }
        url.setText(pairing.first)
        token.setText(pairing.second)
        saveAndTest()
    }

    /** A phonon://pair link (a camera app opened the Mac's code): fill the fields and let the user save them, since
     *  a link can come from anywhere. */
    private fun fillFromLink(intent: Intent?) {
        val data = intent?.dataString ?: return
        val pairing = MacProtocol.parsePairing(data) ?: return
        url.setText(pairing.first)
        token.setText(pairing.second)
        result.text = "Pairing code read: check the address, then tap Save and test."
    }

    private fun heading(text: String) = TextView(this).apply {
        this.text = text
        textSize = 18f
        setPadding(0, 32, 0, 8)
    }

    private fun note(text: String) = TextView(this).apply {
        this.text = text
        setPadding(0, 8, 0, 8)
    }

    override fun onDestroy() {
        ModelFetch.removeListener(onModels)
        super.onDestroy()
    }
}
