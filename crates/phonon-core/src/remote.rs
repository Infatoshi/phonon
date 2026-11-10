//! The WebSocket API: another device (a phone keyboard's voice button) dictates through
//! this Mac's warm engines. Off unless settings.json says `"websocket_api": true`.
//!
//! settings.json keys (the engine re-reads the file when it changes, so the Settings
//! switch applies without a restart):
//!   "websocket_api": true            — listen
//!   "websocket_api_token": "…"       — the token every client must present, 16 characters
//!                                      or more; the bar makes it. Without one nothing listens.
//!   "websocket_api_port": 8377       — optional
//!   "websocket_api_bind": "0.0.0.0"  — optional; an address of this Mac to listen on
//! `{"cmd":"websocket_api","token":"…"}` on stdin overrides the file's token for this run
//! (memory only); the bar does not use it.
//!
//! Protocol 1. Text frames are JSON objects with a `type`; binary frames are audio for the
//! open dictation: 16 kHz mono 16-bit little-endian PCM, in pieces of any size.
//!
//! Client to Mac:
//!   {"type":"hello","token":"…","client":"android"}   first message, within 5 s
//!   {"type":"start","id":"d1","app":{"package":"com.Slack","name":"Slack","url_host":"…"},
//!    "textbox":{"before_text":"…","selected_text":"…","after_text":"…"},
//!    "screen_text":"…","live":true}                   begin a dictation; audio follows
//!   {"type":"context","id":"d1","app":{…},"textbox":{…},"screen_text":"…"}   while it is open
//!   {"type":"finish","id":"d1"}                       end of the audio: transcribe and refine
//!   {"type":"cancel","id":"d1"}                       throw it away
//!   {"type":"refine","id":"r1","text":"…","app":{…},"textbox":{…}}   refine text, no audio
//!   {"type":"status"} / {"type":"ping"}
//!
//! Mac to client:
//!   {"type":"ready","protocol":1,"version":"…","asr_ready":true,"llm_ready":true,
//!    "stream_refine":true,"busy":false,"sample_rate":16000,"max_seconds":1200}
//!   {"type":"started","id":"d1"}
//!   {"type":"partial","id":"d1","text":"…"}           the transcript so far
//!   {"type":"draft","id":"d1","text":"…","complete":true}   the corrected text so far (`live`)
//!   {"type":"final","id":"d1","text":"…","raw":"…","refined":true,"app_category":"work_chat"}
//!     the text to insert. `refined` false: `text` is the raw transcript, and `reason` says
//!     why ("no_speech", "too_short", "refine_failed")
//!   {"type":"cancelled","id":"d1"} / {"type":"pong"}
//!   {"type":"warning","id":"d1","msg":"…"}            the dictation goes on (a piece of the
//!     live stream was lost; the whole recording is still transcribed at `finish`)
//!   {"type":"error","id":"d1","code":"…","msg":"…"}   codes: unauthorized, bad_request,
//!     not_ready, busy (a dictation is running here or on the Mac), preempted (a dictation
//!     started on the Mac), too_long, timeout, engine
//! A request ends with exactly one `final`, `cancelled` or `error` carrying its id; an
//! `error` for a `context` leaves its dictation open.
//!
//! The engine runs one dictation at a time, so the API does too: `start` is refused with
//! `busy` while the Mac or another client dictates, and a dictation started on the Mac ends
//! a client's open one with `preempted`. A client that hears nothing for 90 s is dropped;
//! send a ping every 30 s.
//!
//! Each finished dictation is a corpus entry (source "websocket"), kept or removed by the
//! same `local_history` setting as the Mac's own.

use crate::{ws, EngineEvent};
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufReader, Write};
use std::net::{IpAddr, Ipv4Addr, Shutdown, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

pub const PROTOCOL: u32 = 1;
pub const DEFAULT_PORT: u16 = 8377;
const MIN_TOKEN_CHARS: usize = 16;
const SAMPLE_RATE: u32 = 16_000;
/// Audio reaches the speech sidecar in 0.4 s pieces, as from the bar.
const CHUNK_BYTES: usize = SAMPLE_RATE as usize * 2 * 4 / 10;
/// 20 minutes, the bar's own limit.
const MAX_SECONDS: usize = 1200;
const MAX_PCM_BYTES: usize = SAMPLE_RATE as usize * 2 * MAX_SECONDS;
/// Under 80 ms there is nothing to transcribe (the bar's rule for a silent stub).
const MIN_PCM_BYTES: usize = SAMPLE_RATE as usize * 2 * 8 / 100;
const MAX_TEXT_CHARS: usize = 20_000;
const MAX_FIELD_CHARS: usize = 256;
const MAX_REFINES: usize = 16;
const MAX_CONNECTIONS: usize = 8;
const HELLO_TIMEOUT: Duration = Duration::from_secs(5);
const IDLE_TIMEOUT: Duration = Duration::from_secs(90);
const WRITE_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_HELLO_BYTES: usize = 4 * 1024;
const MAX_MESSAGE_BYTES: usize = 1024 * 1024;
/// From `finish` to the final text, and again from the transcript to the refined text.
const RESULT_TIMEOUT: Duration = Duration::from_secs(60);
/// A bar dictation with no command for this long is taken as over.
const BAR_STALE: Duration = Duration::from_secs(15);
const SETTINGS_CHECK: Duration = Duration::from_secs(1);
const LISTEN_RETRY: Duration = Duration::from_secs(5);
/// Engine request ids of this API; their events never reach the bar.
const ID_PREFIX: &str = "ws-";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub bind: IpAddr,
    pub port: u16,
    pub token: String,
}

/// The API's configuration in a parsed settings.json: None when it is off, an error when
/// it is on but cannot be served as written. `token` is the one the bar sent, which comes
/// before a `websocket_api_token` in the file.
pub fn config_from_settings(settings: &Value, token: Option<&str>) -> Result<Option<Config>> {
    if settings.get("websocket_api").and_then(Value::as_bool) != Some(true) {
        return Ok(None);
    }
    let token = token
        .or_else(|| settings.get("websocket_api_token").and_then(Value::as_str))
        .unwrap_or("")
        .trim();
    if token.is_empty() {
        bail!("on, but no token has been set");
    }
    if token.chars().count() < MIN_TOKEN_CHARS {
        bail!("the token must have {MIN_TOKEN_CHARS} characters or more");
    }
    let port = match settings.get("websocket_api_port") {
        None | Some(Value::Null) => DEFAULT_PORT,
        Some(value) => value
            .as_u64()
            .and_then(|port| u16::try_from(port).ok())
            .filter(|port| *port != 0)
            .context("websocket_api_port is not a port number")?,
    };
    let bind = match settings.get("websocket_api_bind").and_then(Value::as_str) {
        None | Some("") => IpAddr::V4(Ipv4Addr::UNSPECIFIED),
        Some(text) => text
            .trim()
            .parse()
            .with_context(|| format!("websocket_api_bind is not an IP address: {text:?}"))?,
    };
    Ok(Some(Config {
        bind,
        port,
        token: token.to_string(),
    }))
}

/// What the API asks of the engine. `Engine` implements it; the tests use a recorder.
pub(crate) trait Host {
    /// (speech ready, correction ready, streaming refine on)
    fn ready(&self) -> (bool, bool, bool);
    fn stream_start(&mut self, id: &str, app: Option<&Value>, live: bool) -> Result<()>;
    fn stream_chunk(&mut self, id: &str, pcm16_base64: &str) -> Result<()>;
    fn stream_app(&mut self, id: &str, app: &Value);
    fn stream_screen(&mut self, id: &str, text: &str);
    /// The transcript so far of the open stream `id`, when the engine keeps one.
    fn stream_text(&self, id: &str) -> Option<String>;
    fn stream_finish(&mut self, id: &str) -> Result<()>;
    fn stream_stop(&mut self, id: &str) -> Result<()>;
    fn transcribe(&mut self, path: &Path, id: &str) -> Result<()>;
    fn polish(
        &mut self,
        text: &str,
        screen_text: &str,
        id: &str,
        app: Option<&Value>,
        textbox: Option<&Value>,
        keep_context: bool,
    ) -> Result<()>;
}

impl Host for crate::Engine {
    fn ready(&self) -> (bool, bool, bool) {
        (self.asr_ready, self.llm_ready, self.stream_refine)
    }

    fn stream_start(&mut self, id: &str, app: Option<&Value>, live: bool) -> Result<()> {
        crate::Engine::stream_start(self, Some(id), app)?;
        // Drafts for this dictation whatever settings.json `live_text` says: that key
        // decides what the Mac shows, the client asked for its own.
        if live && self.segment_streams.contains_key(id) {
            self.live_text_stream = Some(id.to_string());
        }
        Ok(())
    }

    fn stream_chunk(&mut self, id: &str, pcm16_base64: &str) -> Result<()> {
        crate::Engine::stream_chunk(self, pcm16_base64, Some(id))
    }

    fn stream_app(&mut self, id: &str, app: &Value) {
        crate::Engine::stream_app(self, id, app);
    }

    fn stream_screen(&mut self, id: &str, text: &str) {
        self.stream_screen_context(id, text);
    }

    fn stream_text(&self, id: &str) -> Option<String> {
        self.segment_streams.get(id)?.last().cloned()
    }

    fn stream_finish(&mut self, id: &str) -> Result<()> {
        crate::Engine::stream_finish(self, Some(id))
    }

    fn stream_stop(&mut self, id: &str) -> Result<()> {
        self.segment_streams.remove(id);
        crate::Engine::stream_stop(self, Some(id))
    }

    fn transcribe(&mut self, path: &Path, id: &str) -> Result<()> {
        self.transcribe_with_source(path, Some(id), "websocket")
    }

    fn polish(
        &mut self,
        text: &str,
        screen_text: &str,
        id: &str,
        app: Option<&Value>,
        textbox: Option<&Value>,
        keep_context: bool,
    ) -> Result<()> {
        self.polish_request(text, screen_text, Some(id), app, textbox, keep_context)
    }
}

/// What a connection thread tells the engine loop.
pub(crate) enum Inbound {
    /// The handshake and `hello` passed; `out` takes encoded frames for the client.
    Open {
        conn: u64,
        out: mpsc::Sender<Vec<u8>>,
        client: String,
    },
    Text {
        conn: u64,
        value: Value,
    },
    Audio {
        conn: u64,
        pcm: Vec<u8>,
    },
    Closed {
        conn: u64,
    },
}

type Streams = Arc<Mutex<HashMap<u64, TcpStream>>>;

/// The listener and its connections. Dropping it closes them all.
struct Server {
    addr: SocketAddr,
    stop: Arc<AtomicBool>,
    streams: Streams,
    /// Checked at each `hello`; a new token takes effect without listening anew.
    token: Arc<Mutex<String>>,
    listener: Option<thread::JoinHandle<()>>,
}

static NEXT_CONNECTION: AtomicU64 = AtomicU64::new(1);

impl Server {
    fn start(
        config: &Config,
        tx: mpsc::Sender<Inbound>,
        waker: mpsc::Sender<String>,
    ) -> Result<Self> {
        let listener = TcpListener::bind(SocketAddr::new(config.bind, config.port))
            .with_context(|| format!("listen on {}:{}", config.bind, config.port))?;
        let addr = listener.local_addr()?;
        let stop = Arc::new(AtomicBool::new(false));
        let streams: Streams = Arc::new(Mutex::new(HashMap::new()));
        let token = Arc::new(Mutex::new(config.token.clone()));
        let (shared_token, stopped, live) = (token.clone(), stop.clone(), streams.clone());
        let listener = thread::Builder::new()
            .name("phonon-ws-listen".into())
            .spawn(move || {
                for stream in listener.incoming() {
                    if stopped.load(Ordering::SeqCst) {
                        break;
                    }
                    let Ok(mut stream) = stream else {
                        thread::sleep(Duration::from_millis(50));
                        continue;
                    };
                    let _ = stream.set_write_timeout(Some(WRITE_TIMEOUT));
                    let conn = NEXT_CONNECTION.fetch_add(1, Ordering::Relaxed);
                    {
                        let mut live = live.lock().unwrap_or_else(|e| e.into_inner());
                        let clone = stream.try_clone();
                        match clone {
                            Ok(clone) if live.len() < MAX_CONNECTIONS => {
                                live.insert(conn, clone);
                            }
                            _ => {
                                let _ = stream.write_all(
                                    ws::http_error("503 Service Unavailable").as_bytes(),
                                );
                                continue;
                            }
                        }
                    }
                    let (token, tx, waker, streams) = (
                        shared_token.clone(),
                        tx.clone(),
                        waker.clone(),
                        live.clone(),
                    );
                    let spawned = thread::Builder::new()
                        .name(format!("phonon-ws-{conn}"))
                        .spawn(move || {
                            let opened = serve_connection(conn, stream, &token, &tx, &waker);
                            // Not shut down here: the writer thread does that once the
                            // last frame (the answer to a close) is out.
                            streams
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .remove(&conn);
                            if opened {
                                let _ = tx.send(Inbound::Closed { conn });
                                let _ = waker.send(String::new());
                            }
                        });
                    if spawned.is_err() {
                        live.lock().unwrap_or_else(|e| e.into_inner()).remove(&conn);
                    }
                }
            })
            .context("start the WebSocket listener thread")?;
        Ok(Self {
            addr,
            stop,
            streams,
            token,
            listener: Some(listener),
        })
    }

    /// A new token: connections made with the old one are closed, and only the new one is
    /// accepted from now on. The socket keeps listening, so no client finds the port shut.
    fn set_token(&self, token: &str) {
        *self.token.lock().unwrap_or_else(|e| e.into_inner()) = token.to_string();
        for (_, stream) in self
            .streams
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .drain()
        {
            let _ = stream.shutdown(Shutdown::Both);
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        // `incoming()` blocks: one connection to ourselves lets the thread see the flag.
        let wake = match self.addr.ip() {
            ip if ip.is_unspecified() && ip.is_ipv4() => {
                SocketAddr::new(Ipv4Addr::LOCALHOST.into(), self.addr.port())
            }
            ip if ip.is_unspecified() => {
                SocketAddr::new(std::net::Ipv6Addr::LOCALHOST.into(), self.addr.port())
            }
            _ => self.addr,
        };
        let woken = TcpStream::connect_timeout(&wake, Duration::from_millis(500)).is_ok();
        if let Some(listener) = self.listener.take().filter(|_| woken) {
            // The port is free again once the thread is gone (off and on again reuses it).
            let _ = listener.join();
        }
        for (_, stream) in self
            .streams
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .drain()
        {
            let _ = stream.shutdown(Shutdown::Both);
        }
    }
}

/// Equal without stopping at the first difference (the lengths may be told apart).
fn token_matches(given: &str, expected: &str) -> bool {
    let (given, expected) = (given.as_bytes(), expected.as_bytes());
    let mut difference = given.len() ^ expected.len();
    for (i, byte) in expected.iter().enumerate() {
        difference |= usize::from(byte ^ given.get(i).copied().unwrap_or(0));
    }
    difference == 0
}

fn error_frame(id: Option<&str>, code: &str, msg: &str) -> Vec<u8> {
    ws::encode_text(&json!({"type": "error", "id": id, "code": code, "msg": msg}).to_string())
}

/// One client, on its own thread: handshake, `hello`, then every message to the engine
/// loop. Returns whether the engine loop was told about the connection.
fn serve_connection(
    conn: u64,
    mut stream: TcpStream,
    token: &Mutex<String>,
    tx: &mpsc::Sender<Inbound>,
    waker: &mpsc::Sender<String>,
) -> bool {
    let _ = stream.set_nodelay(true);
    let _ = stream.set_read_timeout(Some(HELLO_TIMEOUT));
    let Ok(read_half) = stream.try_clone() else {
        return false;
    };
    let mut reader = BufReader::new(read_half);
    let key = match ws::read_handshake(&mut reader) {
        Ok(key) => key,
        Err(_) => {
            let _ = stream.write_all(ws::http_error("400 Bad Request").as_bytes());
            return false;
        }
    };
    if stream
        .write_all(ws::handshake_response(&key).as_bytes())
        .is_err()
    {
        return false;
    }
    let mut messages = ws::MessageReader::new(reader, MAX_HELLO_BYTES);
    let hello = loop {
        match messages.read() {
            Ok(ws::Message::Text(text)) => break text,
            Ok(ws::Message::Ping(payload)) => {
                if stream
                    .write_all(&ws::encode(ws::OP_PONG, &payload))
                    .is_err()
                {
                    return false;
                }
            }
            Ok(ws::Message::Pong) => {}
            _ => return false,
        }
    };
    let hello: Value = serde_json::from_str(&hello).unwrap_or(Value::Null);
    let given = hello.get("token").and_then(Value::as_str).unwrap_or("");
    let expected = token.lock().unwrap_or_else(|e| e.into_inner()).clone();
    if hello.get("type").and_then(Value::as_str) != Some("hello")
        || !token_matches(given, &expected)
    {
        let _ = stream.write_all(&error_frame(
            None,
            "unauthorized",
            "the first message must be hello with this Mac's token",
        ));
        let _ = stream.write_all(&ws::encode(ws::OP_CLOSE, &[]));
        return false;
    }
    let client = clip(
        hello.get("client").and_then(Value::as_str).unwrap_or(""),
        MAX_FIELD_CHARS,
    );
    let _ = stream.set_read_timeout(Some(IDLE_TIMEOUT));
    messages.set_max_message(MAX_MESSAGE_BYTES);

    // Frames are written on a thread of their own, so a client that stops reading holds
    // up neither this thread nor the engine loop.
    let (out, frames) = mpsc::channel::<Vec<u8>>();
    let writer = thread::Builder::new()
        .name(format!("phonon-ws-{conn}-out"))
        .spawn(move || {
            for frame in frames {
                if stream.write_all(&frame).is_err() {
                    break;
                }
            }
            let _ = stream.shutdown(Shutdown::Both);
        });
    if writer.is_err() {
        return false;
    }
    if tx
        .send(Inbound::Open {
            conn,
            out: out.clone(),
            client,
        })
        .is_err()
    {
        return false;
    }
    let _ = waker.send(String::new());
    loop {
        let inbound = match messages.read() {
            Ok(ws::Message::Text(text)) => match serde_json::from_str::<Value>(&text) {
                Ok(value) if value.is_object() => Inbound::Text { conn, value },
                _ => {
                    let _ = out.send(error_frame(
                        None,
                        "bad_request",
                        "a text frame must be a JSON object",
                    ));
                    continue;
                }
            },
            Ok(ws::Message::Binary(pcm)) => Inbound::Audio { conn, pcm },
            Ok(ws::Message::Ping(payload)) => {
                let _ = out.send(ws::encode(ws::OP_PONG, &payload));
                continue;
            }
            Ok(ws::Message::Pong) => continue,
            Ok(ws::Message::Close) => {
                let _ = out.send(ws::encode(ws::OP_CLOSE, &[]));
                break;
            }
            Err(_) => break,
        };
        if tx.send(inbound).is_err() {
            break;
        }
        let _ = waker.send(String::new());
    }
    true
}

fn clip(text: &str, max_chars: usize) -> String {
    text.chars().take(max_chars).collect()
}

/// A client's `app` as the engine's `app` object. `package` (Android) and `bundle_id` are
/// the same field: `app_context::categorize` knows both kinds of name.
fn engine_app(app: &Value) -> Option<Value> {
    let field = |key: &str| {
        clip(
            app.get(key).and_then(Value::as_str).unwrap_or("").trim(),
            MAX_FIELD_CHARS,
        )
    };
    let id = match field("package") {
        package if package.is_empty() => field("bundle_id"),
        package => package,
    };
    let host = field("url_host");
    if id.is_empty() && host.is_empty() {
        return None;
    }
    let mut out = json!({"bundle_id": id, "name": field("name")});
    if !host.is_empty() {
        out["url_host"] = json!(host);
    }
    Some(out)
}

/// The bar's last check on a refined text (`PreviewRevisionPolicy.safeFinalText`): the raw
/// transcript stands in for an empty one, a leaked dictionary envelope or a runaway.
fn safe_final_text(raw: &str, refined: &str) -> String {
    let refined = refined.trim();
    if refined.is_empty() {
        return raw.to_string();
    }
    let words = |text: &str| text.split_whitespace().count();
    let envelope = refined.contains("<phonon_dictionary")
        || refined.contains("</phonon_dictionary>")
        || refined.contains("canonical_terms:");
    let runaway = words(refined) > words(raw) * 2 + 8
        || refined.chars().count() > raw.chars().count() * 4 + 64;
    if envelope || runaway {
        raw.to_string()
    } else {
        refined.to_string()
    }
}

fn write_wav_pcm16(path: &Path, pcm: &[u8]) -> Result<()> {
    let parent = path.parent().context("WAV path has no parent")?;
    std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    let mut bytes = Vec::with_capacity(44 + pcm.len());
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + pcm.len() as u32).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
    bytes.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    bytes.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
    bytes.extend_from_slice(pcm);
    std::fs::write(path, bytes).with_context(|| format!("write {}", path.display()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Audio is arriving.
    Recording,
    /// `finish` came: waiting for the transcript.
    Transcribing,
    /// Waiting for the refined text.
    Refining,
}

/// The one open dictation.
struct Session {
    conn: u64,
    client_id: String,
    engine_id: String,
    phase: Phase,
    /// All the audio so far; `sent` bytes of it have gone to the speech sidecar.
    pcm: Vec<u8>,
    sent: usize,
    app: Option<Value>,
    textbox: Option<Value>,
    screen_text: String,
    live: bool,
    raw: String,
    last_partial: String,
    wav: Option<PathBuf>,
    deadline: Option<Instant>,
    /// Milliseconds of audio, once `finish` came.
    audio_ms: u64,
}

/// A `refine` in flight.
struct Refine {
    conn: u64,
    client_id: String,
    text: String,
}

struct Client {
    out: mpsc::Sender<Vec<u8>>,
}

/// The engine loop's side of the API: owns the listener, the clients and the one open
/// dictation. Everything here runs on the engine loop's thread.
pub struct Remote {
    tx: mpsc::Sender<Inbound>,
    rx: mpsc::Receiver<Inbound>,
    waker: mpsc::Sender<String>,
    /// settings.json to follow; None in tests that drive the state directly.
    settings_path: Option<PathBuf>,
    settings_stamp: Option<(SystemTime, u64)>,
    settings_checked: Option<Instant>,
    desired: Option<Config>,
    server: Option<Server>,
    listen_retry: Option<Instant>,
    last_problem: String,
    /// The token the bar sent (`{"cmd":"websocket_api"}`), in memory only.
    pushed_token: Option<String>,
    /// The listener state last told to the bar: (address, problem).
    reported: (Option<SocketAddr>, String),
    /// settings.json `local_history`: keep each dictation's corpus entry.
    keep_history: bool,
    corpus: PathBuf,
    clients: HashMap<u64, Client>,
    session: Option<Session>,
    refines: HashMap<String, Refine>,
    next_request: u64,
    /// The bar's dictation in flight (its request id) and its last command.
    bar: Option<(String, Instant)>,
    /// The open dictation as last told to the bar (`websocket_dictation` status events).
    reported_dictation: Option<String>,
    /// The last dictation that ended with text: (engine id, words, milliseconds of audio).
    completed: Option<(String, usize, u64)>,
}

impl Remote {
    /// Follows the user's settings.json. `waker` is the engine loop's command channel: an
    /// empty line makes the loop turn.
    pub fn new(waker: mpsc::Sender<String>) -> Self {
        let mut remote = Self::with_corpus(
            waker,
            crate::data::corpus_dir().unwrap_or_else(|_| std::env::temp_dir()),
        );
        remote.settings_path = crate::data::settings_path().ok();
        remote
    }

    fn with_corpus(waker: mpsc::Sender<String>, corpus: PathBuf) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            tx,
            rx,
            waker,
            settings_path: None,
            settings_stamp: None,
            settings_checked: None,
            desired: None,
            server: None,
            listen_retry: None,
            last_problem: String::new(),
            pushed_token: None,
            reported: (None, String::new()),
            keep_history: false,
            corpus,
            clients: HashMap::new(),
            session: None,
            refines: HashMap::new(),
            next_request: 0,
            bar: None,
            reported_dictation: None,
            completed: None,
        }
    }

    /// The address the API listens on, when it does.
    pub fn listening(&self) -> Option<SocketAddr> {
        self.server.as_ref().map(|server| server.addr)
    }

    /// `{"cmd":"websocket_api","token":"…"}` from the bar; no token clears it.
    pub fn set_token(&mut self, token: Option<&str>) {
        let token = token
            .map(str::trim)
            .filter(|token| !token.is_empty())
            .map(str::to_string);
        if token != self.pushed_token {
            self.pushed_token = token;
            // read the settings again on the next turn, with this token
            self.settings_stamp = None;
            self.settings_checked = None;
        }
    }

    /// A `status` event for the bar when the listener started, stopped or failed since the
    /// last one: `{"type":"status","kind":"websocket_api","listening":true,"address":"…"}`,
    /// or `"listening":false` with the reason in `msg` when it should be on.
    pub fn take_status(&mut self) -> Option<Value> {
        let state = (self.listening(), self.last_problem.clone());
        if state == self.reported {
            return None;
        }
        self.reported = state;
        Some(json!({
            "type": "status",
            "kind": "websocket_api",
            "listening": self.reported.0.is_some(),
            "address": self.reported.0.map(|address| address.to_string()),
            "msg": self.reported.1,
        }))
    }

    /// `status` events for the bar about the clients' dictations, when one started or ended
    /// since the last call: `{"type":"status","kind":"websocket_dictation","state":"started"}`,
    /// then `{"…","state":"ended","words":42,"audio_ms":9800}` (`words` 0 when nothing was
    /// typed: no speech, cancelled, failed). The bar draws them on Home.
    pub fn take_dictation_events(&mut self) -> Vec<Value> {
        let current = self
            .session
            .as_ref()
            .map(|session| session.engine_id.clone());
        if current == self.reported_dictation {
            return Vec::new();
        }
        let mut out = Vec::new();
        if let Some(ended) = self.reported_dictation.take() {
            let (words, audio_ms) = match self.completed.take() {
                Some((id, words, audio_ms)) if id == ended => (words, audio_ms),
                _ => (0, 0),
            };
            out.push(
                json!({"type": "status", "kind": "websocket_dictation", "state": "ended",
                            "words": words, "audio_ms": audio_ms}),
            );
        }
        if current.is_some() {
            out.push(json!({"type": "status", "kind": "websocket_dictation", "state": "started"}));
        }
        self.reported_dictation = current;
        out
    }

    fn problem(&mut self, message: String) {
        if message != self.last_problem {
            eprintln!("phonon engine: WebSocket API: {message}");
            self.last_problem = message;
        }
    }

    /// Start, stop or move the listener to match `config`.
    fn apply(&mut self, config: Option<Config>) {
        // Only the token changed: keep listening, close the connections made with the old one.
        if let (Some(server), Some(new), Some(old)) = (&self.server, &config, &self.desired) {
            if new.bind == old.bind && new.port == old.port && new.token != old.token {
                server.set_token(&new.token);
                eprintln!("phonon engine: WebSocket API token changed");
                self.desired = config;
                return;
            }
        }
        if config != self.desired {
            self.server = None;
            self.listen_retry = None;
            if config.is_none() && self.desired.is_some() {
                eprintln!("phonon engine: WebSocket API off");
            }
            self.desired = config;
        }
        let Some(config) = self.desired.clone() else {
            return;
        };
        if self.server.is_some() || self.listen_retry.is_some_and(|at| Instant::now() < at) {
            return;
        }
        match Server::start(&config, self.tx.clone(), self.waker.clone()) {
            Ok(server) => {
                eprintln!("phonon engine: WebSocket API listening on {}", server.addr);
                self.last_problem.clear();
                self.listen_retry = None;
                self.server = Some(server);
            }
            Err(error) => {
                // The port may still belong to an engine that is shutting down.
                self.listen_retry = Some(Instant::now() + LISTEN_RETRY);
                self.problem(format!("{error:#}"));
            }
        }
    }

    /// Re-read settings.json when it changed (checked once a second).
    fn follow_settings(&mut self) {
        let Some(path) = self.settings_path.clone() else {
            return;
        };
        if self
            .settings_checked
            .is_some_and(|at| at.elapsed() < SETTINGS_CHECK)
        {
            return;
        }
        self.settings_checked = Some(Instant::now());
        let stamp = std::fs::metadata(&path)
            .ok()
            .and_then(|meta| Some((meta.modified().ok()?, meta.len())));
        if stamp != self.settings_stamp || self.settings_stamp.is_none() {
            self.settings_stamp = stamp;
            let settings = std::fs::read(&path)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
                .unwrap_or(Value::Null);
            self.keep_history =
                settings.get("local_history").and_then(Value::as_bool) == Some(true);
            let config = match config_from_settings(&settings, self.pushed_token.as_deref()) {
                Ok(config) => {
                    self.last_problem.clear();
                    config
                }
                Err(error) => {
                    self.problem(format!("{error:#}"));
                    None
                }
            };
            self.apply(config);
        } else {
            self.apply(self.desired.clone());
        }
    }

    fn send(&self, conn: u64, message: Value) {
        if let Some(client) = self.clients.get(&conn) {
            let _ = client.out.send(ws::encode_text(&message.to_string()));
        }
    }

    fn send_error(&self, conn: u64, id: Option<&str>, code: &str, msg: &str) {
        if let Some(client) = self.clients.get(&conn) {
            let _ = client.out.send(error_frame(id, code, msg));
        }
    }

    fn bar_busy(&self) -> bool {
        self.bar
            .as_ref()
            .is_some_and(|(_, last)| last.elapsed() < BAR_STALE)
    }

    fn status(&self, host: &impl Host, kind: &str) -> Value {
        let (asr_ready, llm_ready, stream_refine) = host.ready();
        json!({
            "type": kind,
            "protocol": PROTOCOL,
            "version": env!("CARGO_PKG_VERSION"),
            "asr_ready": asr_ready,
            "llm_ready": llm_ready,
            "stream_refine": stream_refine,
            "busy": self.session.is_some() || self.bar_busy(),
            "sample_rate": SAMPLE_RATE,
            "max_seconds": MAX_SECONDS,
        })
    }

    fn next_engine_id(&mut self, conn: u64) -> String {
        self.next_request += 1;
        format!("{ID_PREFIX}{conn}-{}", self.next_request)
    }

    /// Once per turn of the engine loop, before `Engine::poll`.
    pub(crate) fn tick(&mut self, host: &mut impl Host) {
        self.follow_settings();
        while let Ok(inbound) = self.rx.try_recv() {
            match inbound {
                Inbound::Open { conn, out, client } => {
                    eprintln!("phonon engine: WebSocket API client {conn} connected ({client})");
                    self.clients.insert(conn, Client { out });
                    self.send(conn, self.status(host, "ready"));
                }
                Inbound::Text { conn, value } => self.on_text(host, conn, &value),
                Inbound::Audio { conn, pcm } => self.on_audio(host, conn, &pcm),
                Inbound::Closed { conn } => {
                    eprintln!("phonon engine: WebSocket API client {conn} disconnected");
                    self.clients.remove(&conn);
                    if self.session.as_ref().is_some_and(|s| s.conn == conn) {
                        self.abandon(host);
                    }
                    self.refines.retain(|_, refine| refine.conn != conn);
                }
            }
        }
        self.forward_partial(host);
        if self
            .session
            .as_ref()
            .and_then(|session| session.deadline)
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            if let Some(session) = self.session.take() {
                self.send_error(
                    session.conn,
                    Some(&session.client_id),
                    "timeout",
                    "the engine did not answer in time",
                );
                self.finish_recording(&session, false);
            }
        }
    }

    fn on_text(&mut self, host: &mut impl Host, conn: u64, value: &Value) {
        let id = value
            .get("id")
            .and_then(Value::as_str)
            .map(|id| clip(id, MAX_FIELD_CHARS))
            .filter(|id| !id.is_empty());
        let kind = value.get("type").and_then(Value::as_str).unwrap_or("");
        match (kind, id.as_deref()) {
            ("ping", _) => self.send(conn, json!({"type": "pong"})),
            ("status", _) => self.send(conn, self.status(host, "ready")),
            ("hello", _) => {} // already said
            ("start", Some(id)) => self.on_start(host, conn, id, value),
            ("context", Some(id)) => self.on_context(host, conn, id, value),
            ("finish", Some(id)) => self.on_finish(host, conn, id),
            ("cancel", Some(id)) => self.on_cancel(host, conn, id),
            ("refine", Some(id)) => self.on_refine(host, conn, id, value),
            ("start" | "context" | "finish" | "cancel" | "refine", None) => {
                self.send_error(conn, None, "bad_request", "this message needs an id")
            }
            (_, id) => self.send_error(conn, id, "bad_request", &format!("unknown type: {kind:?}")),
        }
    }

    fn on_start(&mut self, host: &mut impl Host, conn: u64, id: &str, value: &Value) {
        if let Some(rate) = value.get("sample_rate").filter(|rate| !rate.is_null()) {
            if rate.as_u64() != Some(u64::from(SAMPLE_RATE)) {
                return self.send_error(
                    conn,
                    Some(id),
                    "bad_request",
                    "audio must be 16000 Hz mono 16-bit PCM",
                );
            }
        }
        let (asr_ready, llm_ready, _) = host.ready();
        if !asr_ready || !llm_ready {
            return self.send_error(conn, Some(id), "not_ready", "the models are still loading");
        }
        if self.session.is_some() || self.bar_busy() {
            return self.send_error(conn, Some(id), "busy", "another dictation is running");
        }
        let engine_id = self.next_engine_id(conn);
        let app = value.get("app").and_then(engine_app);
        let live = value.get("live").and_then(Value::as_bool) == Some(true);
        if let Err(error) = host.stream_start(&engine_id, app.as_ref(), live) {
            return self.send_error(conn, Some(id), "engine", &format!("{error:#}"));
        }
        let screen_text = clip(
            value
                .get("screen_text")
                .and_then(Value::as_str)
                .unwrap_or(""),
            MAX_TEXT_CHARS,
        );
        if !screen_text.is_empty() {
            host.stream_screen(&engine_id, &screen_text);
        }
        self.session = Some(Session {
            conn,
            client_id: id.to_string(),
            engine_id,
            phase: Phase::Recording,
            pcm: Vec::new(),
            sent: 0,
            app,
            textbox: value.get("textbox").filter(|v| v.is_object()).cloned(),
            screen_text,
            live,
            raw: String::new(),
            last_partial: String::new(),
            wav: None,
            deadline: None,
            audio_ms: 0,
        });
        self.send(conn, json!({"type": "started", "id": id}));
    }

    /// The open dictation, if it is `conn`'s and the client calls it `id`.
    fn session_of(&mut self, conn: u64, id: &str) -> Option<&mut Session> {
        self.session
            .as_mut()
            .filter(|session| session.conn == conn && session.client_id == id)
    }

    fn on_audio(&mut self, host: &mut impl Host, conn: u64, pcm: &[u8]) {
        let Some(session) = self
            .session
            .as_mut()
            .filter(|session| session.conn == conn && session.phase == Phase::Recording)
        else {
            return; // audio after finish, or with no dictation open: nothing to add it to
        };
        if session.pcm.len() + pcm.len() > MAX_PCM_BYTES {
            let id = session.client_id.clone();
            self.send_error(
                conn,
                Some(&id),
                "too_long",
                &format!("a dictation holds at most {MAX_SECONDS} seconds"),
            );
            return self.abandon(host);
        }
        session.pcm.extend_from_slice(pcm);
        while session.pcm.len() - session.sent >= CHUNK_BYTES {
            let chunk = &session.pcm[session.sent..session.sent + CHUNK_BYTES];
            // A chunk the sidecar missed is still in the WAV the final transcript is checked against.
            let _ = host.stream_chunk(&session.engine_id, &ws::base64(chunk));
            session.sent += CHUNK_BYTES;
        }
    }

    fn on_context(&mut self, host: &mut impl Host, conn: u64, id: &str, value: &Value) {
        let Some(session) = self.session_of(conn, id) else {
            return self.send_error(
                conn,
                Some(id),
                "bad_request",
                "no open dictation with this id",
            );
        };
        if let Some(app) = value.get("app").and_then(engine_app) {
            host.stream_app(&session.engine_id, &app);
            session.app = Some(app);
        }
        if let Some(textbox) = value.get("textbox").filter(|v| v.is_object()) {
            session.textbox = Some(textbox.clone());
        }
        if let Some(text) = value.get("screen_text").and_then(Value::as_str) {
            session.screen_text = clip(text, MAX_TEXT_CHARS);
            host.stream_screen(&session.engine_id, &session.screen_text);
        }
    }

    fn on_finish(&mut self, host: &mut impl Host, conn: u64, id: &str) {
        let corpus = self.corpus.clone();
        let Some(session) = self
            .session_of(conn, id)
            .filter(|session| session.phase == Phase::Recording)
        else {
            return self.send_error(
                conn,
                Some(id),
                "bad_request",
                "no open dictation with this id",
            );
        };
        let whole = session.pcm.len() & !1; // a trailing half sample is dropped
        if whole < MIN_PCM_BYTES {
            let _ = host.stream_stop(&session.engine_id);
            self.session = None;
            return self.send(
                conn,
                json!({"type": "final", "id": id, "text": "", "raw": "", "refined": false,
                       "reason": "too_short"}),
            );
        }
        if whole > session.sent {
            let _ = host.stream_chunk(
                &session.engine_id,
                &ws::base64(&session.pcm[session.sent..whole]),
            );
            session.sent = whole;
        }
        // The engine's speech gate and the corpus entry read the recording from a file.
        let millis = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let wav = corpus
            .join(format!("ws_{millis}_{}_{conn}", std::process::id()))
            .join("audio.wav");
        let engine_id = session.engine_id.clone();
        let result = write_wav_pcm16(&wav, &session.pcm[..whole])
            .and_then(|()| host.stream_finish(&engine_id))
            .and_then(|()| host.transcribe(&wav, &engine_id));
        session.wav = Some(wav);
        session.audio_ms = (whole / 32) as u64; // 16 kHz PCM16: 32 bytes a millisecond
        session.pcm = Vec::new();
        match result {
            Ok(()) => {
                session.phase = Phase::Transcribing;
                session.deadline = Some(Instant::now() + RESULT_TIMEOUT);
            }
            Err(error) => {
                self.send_error(conn, Some(id), "engine", &format!("{error:#}"));
                self.abandon(host);
            }
        }
    }

    fn on_cancel(&mut self, host: &mut impl Host, conn: u64, id: &str) {
        if self.session_of(conn, id).is_some() {
            self.abandon(host);
        } else {
            self.refines
                .retain(|_, refine| !(refine.conn == conn && refine.client_id == id));
        }
        self.send(conn, json!({"type": "cancelled", "id": id}));
    }

    fn on_refine(&mut self, host: &mut impl Host, conn: u64, id: &str, value: &Value) {
        let text = value
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim();
        if text.is_empty() || text.chars().count() > MAX_TEXT_CHARS {
            return self.send_error(
                conn,
                Some(id),
                "bad_request",
                &format!("refine needs text of at most {MAX_TEXT_CHARS} characters"),
            );
        }
        if !host.ready().1 {
            return self.send_error(conn, Some(id), "not_ready", "the models are still loading");
        }
        if self.refines.len() >= MAX_REFINES {
            return self.send_error(conn, Some(id), "busy", "too many refinements are waiting");
        }
        let engine_id = self.next_engine_id(conn);
        let app = value.get("app").and_then(engine_app);
        let screen_text = clip(
            value
                .get("screen_text")
                .and_then(Value::as_str)
                .unwrap_or(""),
            MAX_TEXT_CHARS,
        );
        // Typed text has no recording, so there is no corpus entry to keep its context in.
        match host.polish(
            text,
            &screen_text,
            &engine_id,
            app.as_ref(),
            value.get("textbox").filter(|v| v.is_object()),
            false,
        ) {
            Ok(()) => {
                self.refines.insert(
                    engine_id,
                    Refine {
                        conn,
                        client_id: id.to_string(),
                        text: text.to_string(),
                    },
                );
            }
            Err(error) => self.send_error(conn, Some(id), "engine", &format!("{error:#}")),
        }
    }

    /// Drop the open dictation without a result: stop its stream and remove its recording.
    fn abandon(&mut self, host: &mut impl Host) {
        let Some(session) = self.session.take() else {
            return;
        };
        if session.phase == Phase::Recording {
            let _ = host.stream_stop(&session.engine_id);
        }
        self.finish_recording(&session, false);
    }

    /// The end of a dictation's recording: its corpus entry stays only with local history
    /// on and a result worth keeping, as for the bar's own (`finishRecordingRetention`).
    fn finish_recording(&self, session: &Session, keep: bool) {
        let Some(directory) = session.wav.as_deref().and_then(Path::parent) else {
            return;
        };
        let ours = directory.parent() == Some(self.corpus.as_path())
            && directory
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("ws_"));
        if ours && !(keep && self.keep_history) {
            let _ = std::fs::remove_dir_all(directory);
        }
    }

    /// The transcript so far of a streaming-refine dictation, when it moved.
    fn forward_partial(&mut self, host: &impl Host) {
        let Some(session) = self
            .session
            .as_mut()
            .filter(|session| session.phase == Phase::Recording)
        else {
            return;
        };
        let Some(text) = host
            .stream_text(&session.engine_id)
            .filter(|text| !text.is_empty() && *text != session.last_partial)
        else {
            return;
        };
        session.last_partial = text.clone();
        let (conn, id) = (session.conn, session.client_id.clone());
        self.send(conn, json!({"type": "partial", "id": id, "text": text}));
    }

    /// An engine event. True when it belongs to this API (the bar must not see it).
    pub(crate) fn take_event(&mut self, host: &mut impl Host, event: &EngineEvent) -> bool {
        let id = match event {
            EngineEvent::AsrResult { id, .. }
            | EngineEvent::PolishResult { id, .. }
            | EngineEvent::Error { id, .. } => id.as_deref(),
            EngineEvent::DraftResult { id, .. } => Some(id.as_str()),
            _ => None,
        };
        let Some(id) = id.filter(|id| id.starts_with(ID_PREFIX)) else {
            return false;
        };
        // The loop turns once more: what this starts may already have an answer queued.
        let _ = self.waker.send(String::new());
        if let Some(refine) = self.refines.remove(id) {
            match event {
                EngineEvent::PolishResult {
                    text, app_category, ..
                } => self.send(
                    refine.conn,
                    json!({"type": "final", "id": refine.client_id,
                           "text": safe_final_text(&refine.text, text), "raw": refine.text,
                           "refined": true,
                           "app_category": app_category.map(|category| category.as_str())}),
                ),
                EngineEvent::Error { msg, .. } => {
                    self.send_error(refine.conn, Some(&refine.client_id), "engine", msg)
                }
                _ => {
                    self.refines.insert(id.to_string(), refine);
                }
            }
            return true;
        }
        let Some(session) = self
            .session
            .as_mut()
            .filter(|session| session.engine_id == id)
        else {
            return true; // late news of a dictation that is over
        };
        let (conn, client_id) = (session.conn, session.client_id.clone());
        match event {
            EngineEvent::AsrResult { text, partial, .. } if *partial => {
                if session.phase == Phase::Recording && *text != session.last_partial {
                    session.last_partial = text.clone();
                    self.send(
                        conn,
                        json!({"type": "partial", "id": client_id, "text": text}),
                    );
                }
            }
            EngineEvent::AsrResult { text, .. } => {
                if session.phase != Phase::Transcribing {
                    return true;
                }
                let raw = text.trim().to_string();
                if raw.is_empty() {
                    self.send(
                        conn,
                        json!({"type": "final", "id": client_id, "text": "", "raw": "",
                               "refined": false, "reason": "no_speech"}),
                    );
                    if let Some(session) = self.session.take() {
                        self.finish_recording(&session, false);
                    }
                    return true;
                }
                session.raw = raw.clone();
                let keep = self.keep_history;
                match host.polish(
                    &raw,
                    &session.screen_text,
                    id,
                    session.app.as_ref(),
                    session.textbox.as_ref(),
                    keep,
                ) {
                    Ok(()) => {
                        session.phase = Phase::Refining;
                        session.deadline = Some(Instant::now() + RESULT_TIMEOUT);
                    }
                    Err(_) => self.end_unrefined(),
                }
            }
            EngineEvent::PolishResult {
                text, app_category, ..
            } => {
                if session.phase != Phase::Refining {
                    return true;
                }
                let Some(session) = self.session.take() else {
                    return true;
                };
                let text = safe_final_text(&session.raw, text);
                self.completed = Some((
                    session.engine_id.clone(),
                    text.split_whitespace().count(),
                    session.audio_ms,
                ));
                self.send(
                    conn,
                    json!({"type": "final", "id": client_id,
                           "text": text, "raw": session.raw,
                           "refined": true,
                           "app_category": app_category.map(|category| category.as_str())}),
                );
                self.finish_recording(&session, true);
            }
            EngineEvent::DraftResult { text, complete, .. } => {
                if session.live && session.phase != Phase::Refining {
                    self.send(
                        conn,
                        json!({"type": "draft", "id": client_id, "text": text,
                               "complete": complete}),
                    );
                }
            }
            EngineEvent::Error { msg, .. } => match session.phase {
                // The stream may have lost a piece; the recording is transcribed at finish.
                Phase::Recording => self.send(
                    conn,
                    json!({"type": "warning", "id": client_id, "msg": msg}),
                ),
                Phase::Transcribing => {
                    self.send_error(conn, Some(&client_id), "engine", msg);
                    if let Some(session) = self.session.take() {
                        self.finish_recording(&session, false);
                    }
                }
                Phase::Refining => self.end_unrefined(),
            },
            _ => {}
        }
        true
    }

    /// The refiner failed after the transcript arrived: the transcript is the result.
    fn end_unrefined(&mut self) {
        let Some(session) = self.session.take() else {
            return;
        };
        self.completed = Some((
            session.engine_id.clone(),
            session.raw.split_whitespace().count(),
            session.audio_ms,
        ));
        self.send(
            session.conn,
            json!({"type": "final", "id": session.client_id, "text": session.raw,
                   "raw": session.raw, "refined": false, "reason": "refine_failed"}),
        );
        self.finish_recording(&session, true);
    }

    /// A command the bar sent. Its dictations come first: one starting ends a client's
    /// open dictation, and while one runs a client's `start` is refused.
    pub(crate) fn note_bar_command(&mut self, command: &Value) {
        let name = command.get("cmd").and_then(Value::as_str).unwrap_or("");
        if name == "dictation_end" {
            // the bar dropped its dictation (too short, discarded): the engine is free
            self.bar = None;
            return;
        }
        let dictation = matches!(
            name,
            "stream_start"
                | "stream_chunk"
                | "stream_finish"
                | "stream_stop"
                | "transcribe"
                | "polish"
        );
        let Some(id) = command
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| dictation && !id.is_empty())
        else {
            return;
        };
        self.bar = Some((id.to_string(), Instant::now()));
        if name != "stream_start" {
            return;
        }
        // The engine has just dropped the open stream for the bar's. A dictation that is
        // already being transcribed or refined still finishes.
        if self
            .session
            .as_ref()
            .is_some_and(|session| session.phase == Phase::Recording)
        {
            if let Some(session) = self.session.take() {
                self.send_error(
                    session.conn,
                    Some(&session.client_id),
                    "preempted",
                    "a dictation started on the Mac",
                );
            }
        }
    }

    /// An event on its way to the bar: the end of its dictation frees the engine.
    pub(crate) fn note_bar_event(&mut self, event: &EngineEvent) {
        let ended = match event {
            EngineEvent::PolishResult { id, .. } | EngineEvent::Error { id, .. } => id.as_deref(),
            EngineEvent::AsrResult {
                id, text, partial, ..
            } if !partial && text.trim().is_empty() => id.as_deref(),
            _ => None,
        };
        if ended.is_some() && ended == self.bar.as_ref().map(|(id, _)| id.as_str()) {
            self.bar = None;
        }
    }
}

#[cfg(test)]
mod tests;
