# Phonon WebSocket API

Protocol version 1.

The WebSocket API lets another device dictate through a Mac running Phonon. The device
records the audio and names the app it is typing into. The Mac transcribes the audio,
corrects it with the same models, dictionary and writing modes it uses for its own
dictation, and sends back the text to insert. A phone keyboard's voice button is the
intended client, but anything that can open a WebSocket can use it.

This document is the whole contract. You don't need the Phonon source to write a client.
[`scripts/ws-api-client.py`](../scripts/ws-api-client.py) is a working client in about
200 lines of standard-library Python, if you want something to compare against.

- [Setting up the Mac](#setting-up-the-mac) (and the [pairing code](#pairing-code))
- [Connecting](#connecting)
- [Messages at a glance](#messages-at-a-glance)
- [Dictating](#dictating)
- [Refining text](#refining-text)
- [Context: app, text box, screen](#context-app-text-box-screen)
- [Results](#results)
- [Errors](#errors)
- [Sharing the Mac](#sharing-the-mac)
- [Keeping the connection alive](#keeping-the-connection-alive)
- [Limits](#limits)
- [Security](#security)
- [Writing a client](#writing-a-client)
- [Example sessions](#example-sessions)
- [Compatibility](#compatibility)

## Setting up the Mac

1. Open Phonon › Settings › Advanced › **WebSocket API** and turn it on.
2. The section then shows:
   - **Address**: one `ws://` URL for each of the Mac's network addresses. On a
     Tailscale tailnet, use the `100.x.y.z` one.
   - **Token**: the secret every client must present. Copy it to the device.
   - **Port**: 8377 by default.
   - A status line: "Listening on port 8377", or why the API is not listening.
3. **New token** replaces the token. Clients connected with the old one are disconnected.

### Pairing code

Settings also shows a QR code, so a phone can be set up without typing. Tick one of the
addresses to choose which one it carries; a tailnet address is ticked by default. The code holds
a link:

```
phonon://pair?url=ws%3A%2F%2F192.168.0.52%3A8377&token=fzqxz6qcsyh3cptvew5atmfs
```

`url` is the `ws://` address and `token` the token, each percent-encoded. A client that reads
it, with its own scanner or by registering the `phonon://pair` link, should treat it like
typed-in settings:

- Accept only `phonon://pair` links whose `url` is a `ws://` or `wss://` address and whose
  token has 16 characters or more.
- Ask before saving one that arrived as a link from elsewhere. A link can come from anyone,
  and saving it would send the user's dictations to whoever made it.

The API is available only while Phonon is running and its models are loaded, and only
while the Mac is awake.

Optional, by hand in `~/Library/Application Support/Phonon/settings.json`:
`"websocket_api_bind": "100.101.102.103"` makes Phonon listen on that one address of the
Mac instead of all of them. Use the tailnet address to keep the API off the local network,
or `127.0.0.1` when you publish it with `tailscale serve`.

## Connecting

Phonon speaks plain WebSocket (RFC 6455) over TCP, with no TLS: `ws://host:8377/`.

- **Path:** any path works.
- **Subprotocols:** none.
- **Origin:** not checked.
- **Extensions:** none are negotiated, so don't compress frames (no `permessage-deflate`).
- **Handshake failures:**
  - A request that is not a WebSocket upgrade is refused with `400 Bad Request`.
  - When 8 connections are already open, the server answers `503 Service Unavailable`.

Frames:

| Frame | Carries |
|---|---|
| Text | One JSON object with a `"type"` field |
| Binary | Audio for the open dictation (see [Audio format](#audio-format)) |
| Ping | Answered with a pong carrying the same payload |
| Close | Answered with a close, then the server hangs up |

Fragmented messages are accepted. As the RFC requires, client frames must be masked.

### Authenticating

The first message must be `hello`, and it must arrive within 5 seconds of the handshake:

```json
{"type": "hello", "token": "the-token-from-settings", "client": "my-keyboard 0.3"}
```

- `token` (string, required): the token shown in Settings.
- `client` (string, optional): a name for the Mac's log. Up to 256 characters are kept.

If the token is right, the Mac answers with `ready`:

```json
{"type": "ready", "protocol": 1, "version": "1.0.0", "asr_ready": true, "llm_ready": true,
 "stream_refine": true, "busy": false, "sample_rate": 16000, "max_seconds": 1200}
```

| Field | Meaning |
|---|---|
| `protocol` | The protocol version, `1` |
| `version` | The engine's version (informational) |
| `asr_ready` | Speech recognition is loaded |
| `llm_ready` | The correction model is loaded |
| `stream_refine` | The Mac transcribes and corrects while you speak, so `partial` and `draft` messages are frequent and the wait after `finish` is short |
| `busy` | A dictation is running right now (see [Sharing the Mac](#sharing-the-mac)) |
| `sample_rate` | Always `16000`: the audio format the Mac expects |
| `max_seconds` | The longest dictation, `1200` |

If the token is wrong, or the first message is not `hello`, the Mac sends one error and
closes the connection:

```json
{"type": "error", "id": null, "code": "unauthorized", "msg": "the first message must be hello with this Mac's token"}
```

A binary frame before `hello`, or no `hello` within 5 seconds, closes the connection
without a message.

## Messages at a glance

Client to Mac:

| `type` | Fields | Purpose |
|---|---|---|
| `hello` | `token`, `client`? | Authenticate; must be first |
| `start` | `id`, `app`?, `textbox`?, `screen_text`?, `live`?, `sample_rate`? | Begin a dictation; audio frames follow |
| *(binary)* | PCM audio | Audio for the open dictation |
| `context` | `id`, `app`?, `textbox`?, `screen_text`? | Update the open dictation's context |
| `finish` | `id` | End of audio: transcribe and correct |
| `cancel` | `id` | Drop a dictation or a refine |
| `refine` | `id`, `text`, `app`?, `textbox`?, `screen_text`? | Correct text you already have, no audio |
| `status` | | Ask for a fresh `ready` |
| `ping` | | Ask for a `pong` |

Mac to client:

| `type` | Fields | Meaning |
|---|---|---|
| `ready` | see above | Reply to `hello` and to `status` |
| `started` | `id` | The dictation is open; send audio |
| `partial` | `id`, `text` | The raw transcript so far |
| `draft` | `id`, `text`, `complete` | The corrected text so far (only with `"live": true`) |
| `final` | `id`, `text`, `raw`, `refined`, `app_category`?, `reason`? | The result; the request is over |
| `cancelled` | `id` | Reply to `cancel`; the request is over |
| `warning` | `id`, `msg` | Something went wrong, but the dictation goes on |
| `error` | `id`, `code`, `msg` | The request failed and is over, unless it was a `context` |
| `pong` | | Reply to `ping` |

**Ids.** You choose each request's `id`: any non-empty string, of which up to 256
characters are kept. Ids belong to your connection, so two clients can both use `"d1"`. You
may reuse an id once its request is over.

**One answer per request.** Every `start`, `finish` and `refine` ends with exactly one
`final`, `cancelled` or `error` carrying its `id`. An `error` answering a `context` does
not end the dictation it names.

**Order.** Messages on one connection arrive in the order the Mac sent them.

## Dictating

```
client                              Mac
  │ start {id:"d1", app, live:true} │
  │────────────────────────────────▶│
  │                started {id:"d1"}│
  │◀────────────────────────────────│
  │ binary audio … (as you speak)   │
  │────────────────────────────────▶│
  │      partial / draft {id:"d1"} …│   while you speak
  │◀────────────────────────────────│
  │ finish {id:"d1"}                │   key released
  │────────────────────────────────▶│
  │         final {id:"d1", text}   │   typically 0.1–1 s later
  │◀────────────────────────────────│
```

### `start`

```json
{"type": "start", "id": "d1",
 "app": {"package": "com.Slack", "name": "Slack"},
 "textbox": {"before_text": "Thanks! ", "selected_text": "", "after_text": ""},
 "live": true}
```

| Field | Type | Meaning |
|---|---|---|
| `id` | string, required | Names this dictation |
| `app` | object | The app the text is for; see [Context](#context-app-text-box-screen) |
| `textbox` | object | The text around the cursor |
| `screen_text` | string | Text visible on screen |
| `live` | bool, default `false` | Send `draft` messages while you speak |
| `sample_rate` | number | If present, it must be `16000`; any other value is refused |

Send `start` the moment recording begins, and stream the audio as you capture it: the Mac
does most of the work while you are still speaking. Audio sent before `started` arrives is
still accepted, because messages on a connection are handled in order.

`start` is refused when:

| Code | Why |
|---|---|
| `not_ready` | The models are still loading. Wait for `ready` with both `asr_ready` and `llm_ready` true (send `status` to ask again). |
| `busy` | Another dictation is running; see [Sharing the Mac](#sharing-the-mac). |
| `bad_request` | No `id`, or a `sample_rate` other than 16000. |

### Audio format

Binary frames hold raw PCM:

- 16,000 samples per second, mono
- signed 16-bit integers, little-endian
- no header (not a WAV file)

Frames may be any size, and samples may even be split across frames, because the Mac only
concatenates them. 20–100 ms per frame (640–3,200 bytes) is a good size. Send audio at the
pace you capture it. Sending faster, for example from a file, also works; you just get
`partial` messages later.

Audio arriving while no dictation of yours is open, or after `finish`, is silently
dropped.

Phonon gets its best results from the raw microphone signal. On Android, record with
`MediaRecorder.AudioSource.VOICE_RECOGNITION`. Don't apply your own gain, noise suppression
or compression.

### `finish`

```json
{"type": "finish", "id": "d1"}
```

This says the audio is complete. The Mac transcribes the remainder and corrects the whole
dictation, then sends `final`.

- **Under 80 ms of audio:** the answer is at once
  `{"type":"final","id":"d1","text":"","raw":"","refined":false,"reason":"too_short"}`.
- **No answer within 60 seconds** for the transcript, or 60 more for the correction: the
  dictation ends with `error` code `timeout`.

### `cancel`

```json
{"type": "cancel", "id": "d1"}
```

This drops the dictation or refine with that id, at any stage, and its recording is
deleted. The Mac always answers `{"type":"cancelled","id":"d1"}`, even for an id it doesn't
know.

### Live text: `partial` and `draft`

While you speak, the Mac sends:

- `partial`: the raw transcript so far, as recognised. Always sent.
- `draft`: the corrected text so far, after the same correction the final text gets. Sent
  only when `start` had `"live": true`.

```json
{"type": "partial", "id": "d1", "text": "Hey team, quick update on the deploy. We ship the"}
{"type": "draft", "id": "d1", "text": "Hey team,\n\nQuick update on the deploy. We ship the", "complete": true}
```

Each message replaces the one before it entirely; it is not a delta. Later messages may
rewrite earlier words, for example "ship" becoming "shipped". `complete: false` means that
draft stopped at the model's output limit and is shorter than what was said.

Treat both as a preview only. Insert only the `final` text: it can differ from the last
draft.

How often they come depends on the Mac:

- With `stream_refine` on, about every 0.4 s once you are speaking.
- With it off, `partial` messages come less often and `draft` messages not at all.

## Refining text

`refine` corrects text you already have, such as text typed by hand or produced by a
recogniser on the device:

```json
{"type": "refine", "id": "r1",
 "text": "hi sarah um i wanted to follow up on the invoice can you send it again thanks berni",
 "app": {"package": "com.google.android.gm"}}
```

The answer is a `final`:

```json
{"type": "final", "id": "r1", "refined": true, "app_category": "email",
 "raw": "hi sarah um i wanted to follow up on the invoice can you send it again thanks berni",
 "text": "Hi Sarah,\n\nI wanted to follow up on the invoice. Can you send it again?\n\nThanks,\nBerni"}
```

- `text` is required, non-empty, and at most 20,000 characters.
- `app`, `textbox` and `screen_text` work as they do for `start`.
- Refines don't conflict with dictation: one can run while a dictation (yours or the Mac's)
  is open.
- Up to 16 refines can be waiting at once; more are refused with `busy`.
- A refine needs only the correction model, so it works as soon as `llm_ready` is true.
- Refined text is not saved on the Mac.

## Context: app, text box, screen

Context is optional, but it is what makes the result fit where it is going.

### `app`: where the text goes

```json
{"package": "com.Slack", "name": "Slack", "url_host": "mail.google.com"}
```

| Field | Meaning |
|---|---|
| `package` | The target app's id: an Android package name or an Apple bundle id (`bundle_id` is accepted as a synonym) |
| `name` | Its display name (informational) |
| `url_host` | For a browser, the host of the page; it takes precedence over the app |

Each field keeps up to 256 characters. The Mac sorts the app into a category, and the
category picks the writing mode:

| `app_category` | Examples | Writing mode |
|---|---|---|
| `chat` | WhatsApp, Discord, Telegram, Signal, Messages | casual |
| `work_chat` | Slack, Teams, Google Chat | formal |
| `email` | Gmail, Outlook, Proton Mail | formal, with the greeting and sign-off on their own lines |
| `ai_prompt` | ChatGPT, Claude, Gemini, Perplexity | prompt |
| `code` | Termux, GitHub, terminals and editors | prompt |
| `docs` | Google Docs, Keep, Word, OneNote | minimal |
| `other` | anything else | minimal |

The modes:

- **casual:** relaxed punctuation and capitals, as people write in chat.
- **formal:** full sentences.
- **prompt:** structured instructions.
- **minimal:** fixes only what was misheard and removes filler.

If the user has picked a writing mode for an app on the Mac, that choice wins, matched on
the same id.

### `textbox`: the text around the cursor

```json
{"before_text": "Thanks for the notes. ", "selected_text": "", "after_text": ""}
```

This helps the correction continue the text that is already there, for example in its
capitalisation and punctuation. Up to 2,000 characters are kept on each side, the ones
nearest the cursor. Don't send it from password fields or other sensitive inputs.

### `screen_text`: text on screen

This is plain text visible on screen, newline-separated, up to 20,000 characters. The Mac
uses it only to confirm the spelling of terms in the user's dictionary that match what was
said. The screen text itself is not given to the correction model.

### Updating context

Context known only after recording started, such as a browser's page host, can follow in a
`context` message:

```json
{"type": "context", "id": "d1", "app": {"package": "com.android.chrome", "url_host": "mail.google.com"}}
```

Fields you omit are left as they were. A `context` for an id that is not your open
dictation is answered with `error` code `bad_request`, and it does not end anything.

## Results

```json
{"type": "final", "id": "d1", "refined": true, "app_category": "work_chat",
 "raw": "Hey team, quick update on the deploy. Um, we shipped it.",
 "text": "Hey team, quick update on the deploy. We shipped it."}
```

| Field | Meaning |
|---|---|
| `text` | The text to insert. It may be empty. |
| `raw` | What was recognised, before correction (for a refine, the text you sent) |
| `refined` | `true` if `text` is the corrected text |
| `reason` | Present when `refined` is `false` |
| `app_category` | The category used (see the table above), or `null` without an `app` |

`refined: false` reasons:

| `reason` | `text` |
|---|---|
| `too_short` | `""`: less than 80 ms of audio |
| `no_speech` | `""`: nothing was said |
| `refine_failed` | The raw transcript, because correction failed or was refused. Insert it. |

**Inserting the text:**

- `text` may contain newlines. Email layout puts the greeting and sign-off on their own
  lines, and numbered steps are one per line. For a single-line field, replace each `\n`
  with a space.
- `text` has no leading or trailing space. When the cursor follows a word, add a space
  before inserting.
- An empty `text` means insert nothing.

## Errors

```json
{"type": "error", "id": "d1", "code": "busy", "msg": "another dictation is running"}
```

`msg` is for people and logs; don't parse it. `id` is `null` when the error belongs to no
request, for example a malformed frame.

| `code` | When | What to do |
|---|---|---|
| `unauthorized` | Bad token, or `hello` not first; the connection then closes | Ask the user for the token |
| `bad_request` | Missing `id`, unknown `type`, a frame that is not a JSON object, unknown id for `context` or `finish`, empty or oversized refine text, wrong `sample_rate` | Fix the client |
| `not_ready` | Models still loading | Retry after `ready` reports both ready |
| `busy` | Another dictation is running, or 16 refines are waiting | Use a fallback, or retry later |
| `preempted` | A dictation started on the Mac while yours was recording | Your dictation is gone; use a fallback or tell the user |
| `too_long` | Over 1,200 seconds of audio | The dictation is dropped |
| `timeout` | No result within 60 s | The dictation is dropped |
| `engine` | The Mac's engine failed | The request is over; retrying may work |

A `warning` doesn't end anything:

```json
{"type": "warning", "id": "d1", "msg": "…"}
```

It means a piece of the live stream was lost. The whole recording is still transcribed at
`finish`, so keep sending audio.

Treat any unknown `code` like `engine`.

## Sharing the Mac

The Mac runs one dictation at a time, and its owner's own dictation comes first:

- **The Mac is dictating.** `start` is refused with `busy`. The Mac is free again as soon
  as that dictation's text is out, or at most 15 seconds after the Mac's last activity if
  its dictation was abandoned.
- **Another client is dictating.** `start` is refused with `busy`.
- **The Mac starts dictating while you record.** Your dictation ends with `preempted`.
- **The Mac starts dictating after your `finish`.** Your dictation still completes.
- **Refines** are not affected by any of this.

`ready.busy` (from `hello` or `status`) tells you in advance. A client with its own
recogniser should fall back to it on `busy`, `preempted`, `not_ready` or a failed
connection, rather than make the user wait.

## Keeping the connection alive

- **Idle timeout.** The Mac drops a connection that sends nothing for 90 seconds. Send
  `{"type":"ping"}` (answered with `{"type":"pong"}`) or a WebSocket ping frame every 30
  seconds.
- **When the connection closes,** for any reason, an open dictation is dropped and its
  recording deleted, and waiting refines are forgotten. Reconnect and start again.
- **Engine restarts.** The connection also closes when the Mac's engine restarts, Phonon
  quits, the API is turned off, or the token is changed.
- **Reconnecting.** Keep the connection open while the user is likely to dictate (while
  your keyboard is shown, say), so the press of the mic button doesn't wait for a TCP and
  WebSocket handshake. Reconnect with backoff (for example 1, 2, 5, 10 s) when it drops.

## Limits

| Limit | Value |
|---|---|
| Open connections | 8 |
| Time to send `hello` | 5 s |
| `hello` message size | 4 KB |
| Any later message | 1 MB |
| Idle time before disconnect | 90 s |
| Dictation length | 1,200 s (38.4 MB of audio) |
| Shortest dictation transcribed | 80 ms |
| Waiting refines | 16 |
| `refine` text, `screen_text` | 20,000 characters |
| `textbox` | 2,000 characters on each side |
| `id`, `client`, `app` fields | 256 characters (longer is cut) |
| Result timeout | 60 s to transcribe, then 60 s to correct |

## Security

- **The token is the only access control.** Anyone who has it and can reach the port can
  dictate through the Mac and see its corrections. Store it the way you would store a
  password, for example in Android's private app storage or the iOS Keychain. Never log
  it.
- **The connection is not encrypted.** Audio, context and text cross the network in the
  clear. Use it on a network you trust, or over a VPN such as Tailscale, where the tunnel
  is encrypted.
- **Where the Mac's token lives.** It is in Phonon's `settings.json`
  (`websocket_api_token`), so Phonon's backup and exports carry it too. **New token**
  revokes it.
- **What the Mac keeps.** With the Mac's Local history setting on, each dictation is kept
  on the Mac: the audio, the transcript, the result, and the app and text-box context.
  It appears in the Mac's history with source `websocket`. With it off, nothing is kept.
  Refined text is never kept.
- **Don't send sensitive context.** Don't send `textbox` or `screen_text` from password
  fields or apps the user marks private.

## Writing a client

A minimal client:

1. Open `ws://host:port/`. Send `hello` with the token, and wait for `ready`.
2. When the user starts dictating, send `start` with a new `id`, the target app's id, and
   the text before the cursor.
3. Stream microphone audio as 16 kHz mono 16-bit little-endian PCM in binary frames.
4. Show `partial` (or, with `live`, `draft`) text as a preview if you want to.
5. When the user stops, send `finish`.
6. Wait for the `final`, `error` or `cancelled` with that `id`.
7. On `final`, insert `text`, adapting newlines and a leading space to the field.
8. Ping every 30 s while connected. Fall back to an on-device recogniser on `busy`,
   `preempted`, `not_ready` or a dead connection.

Checklist:

- [ ] Client frames are masked; no compression is negotiated.
- [ ] The token comes from the user and is stored privately.
- [ ] Each request gets a fresh `id`, and replies are matched by `id`.
- [ ] Unknown message types and unknown fields are ignored.
- [ ] Audio is converted to 16 kHz mono PCM16 little-endian before sending.
- [ ] Every request has a way out: a `final`, `cancelled` or `error`, or your own timeout
      (2 minutes is generous).
- [ ] Empty `text` inserts nothing; newlines are adapted to single-line fields.
- [ ] No `textbox` or `screen_text` is sent from password fields.

### Converting audio

- **Android:** `AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, 16000,
  AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, …)` delivers the format
  directly. Send the bytes it fills.
- **iOS / macOS:** take `AVAudioEngine`'s input format and convert with an
  `AVAudioConverter` to `AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16000,
  channels: 1, interleaved: true)`.
- **Browsers:** an `AudioWorklet` receives Float32 at the context's rate. Resample to
  16 kHz, then write `Math.max(-1, Math.min(1, x)) * 32767` into an `Int16Array`, which is
  little-endian on every common platform.
- **Files:** `ffmpeg -i in.m4a -ac 1 -ar 16000 -f s16le out.pcm` writes raw PCM, or
  `-acodec pcm_s16le out.wav` for a WAV whose 44-byte header you skip.

### Testing against a Mac

```bash
export PHONON_WS_TOKEN=…   # from Settings
say -o /tmp/hello.wav --data-format=LEI16@16000 "Hey team, the deploy went out this morning"
scripts/ws-api-client.py ws://192.168.1.20:8377 /tmp/hello.wav --app com.Slack --live
scripts/ws-api-client.py ws://192.168.1.20:8377 --refine "i will be there at five" --app com.whatsapp
```

The script prints every message the Mac sends, as JSON lines.

## Example sessions

A dictation into Slack, with live drafts. `→` is client to Mac and `←` is Mac to client;
binary frames are shown as `[audio]`.

```
→ {"type":"hello","token":"…","client":"keyboard"}
← {"type":"ready","protocol":1,"version":"1.0.0","asr_ready":true,"llm_ready":true,"stream_refine":true,"busy":false,"sample_rate":16000,"max_seconds":1200}
→ {"type":"start","id":"d1","app":{"package":"com.Slack"},"live":true}
← {"type":"started","id":"d1"}
→ [audio] [audio] [audio] …
← {"type":"partial","id":"d1","text":"Hey team."}
← {"type":"draft","id":"d1","text":"Hey team.","complete":true}
→ [audio] [audio] …
← {"type":"partial","id":"d1","text":"Hey team, quick update on the deploy. We shipped the new Kubernetes config this morning, um, and the latency dropped by about 30%."}
← {"type":"draft","id":"d1","text":"Hey team,\n\nQuick update on the deploy. We shipped the new Kubernetes config this morning, and the latency dropped by about 30%.","complete":true}
→ {"type":"finish","id":"d1"}
← {"type":"final","id":"d1","refined":true,"app_category":"work_chat","raw":"Hey team, quick update on the deploy. We shipped the new Kubernetes config this morning, um, and the latency dropped by about 30%.","text":"Hey team, quick update on the deploy. We shipped the new Kubernetes config this morning, and the latency dropped by about 30%."}
```

Here the drafts laid the message out like an email, but the final text did not. That is
why only `final` is inserted.

The Mac takes over while the phone is recording:

```
→ {"type":"start","id":"d2"}
← {"type":"started","id":"d2"}
→ [audio] …
← {"type":"error","id":"d2","code":"preempted","msg":"a dictation started on the Mac"}
→ {"type":"start","id":"d3"}
← {"type":"error","id":"d3","code":"busy","msg":"another dictation is running"}
```

## Compatibility

- **New fields and messages.** Later versions may add fields to any message and new
  message types. Ignore what you don't recognise.
- **Breaking changes** raise `protocol` in `ready`. Check it, and refuse to run against a
  version you don't know.
- **Unknown types are errors.** The Mac answers a message `type` it doesn't know with
  `bad_request`, so don't send new types speculatively.
