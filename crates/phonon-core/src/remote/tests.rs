use super::*;
use crate::app_context::AppCategory;
use std::io::Read;
use std::sync::atomic::AtomicUsize;

/// Stands in for the engine: records what the API asks and, with `auto`, answers a
/// `transcribe` and a `polish` the way the engine's events would.
#[derive(Default)]
struct MockHost {
    not_ready: bool,
    stream_refine: bool,
    calls: Vec<String>,
    /// The audio of every `stream_chunk`, still base64.
    chunks: Vec<String>,
    stream_text: Option<String>,
    fail_polish: bool,
    auto: bool,
    transcript: String,
    events: Vec<EngineEvent>,
}

/// A flat JSON object with its keys in order, whatever order serde_json keeps them in
/// (that depends on a feature another crate of the workspace may turn on).
fn sorted(value: Option<&Value>) -> String {
    let Some(object) = value.and_then(Value::as_object) else {
        return String::new();
    };
    let mut fields: Vec<String> = object
        .iter()
        .map(|(key, value)| format!("{key:?}:{value}"))
        .collect();
    fields.sort();
    format!("{{{}}}", fields.join(","))
}

impl Host for MockHost {
    fn ready(&self) -> (bool, bool, bool) {
        (!self.not_ready, !self.not_ready, self.stream_refine)
    }
    fn stream_start(&mut self, id: &str, app: Option<&Value>, live: bool) -> Result<()> {
        let app = sorted(app);
        self.calls
            .push(format!("stream_start {id} live={live} app={app}"));
        Ok(())
    }
    fn stream_chunk(&mut self, id: &str, pcm16_base64: &str) -> Result<()> {
        self.calls.push(format!("stream_chunk {id}"));
        self.chunks.push(pcm16_base64.to_string());
        Ok(())
    }
    fn stream_app(&mut self, id: &str, app: &Value) {
        self.calls
            .push(format!("stream_app {id} {}", sorted(Some(app))));
    }
    fn stream_screen(&mut self, id: &str, text: &str) {
        self.calls.push(format!("stream_screen {id} {text}"));
    }
    fn stream_text(&self, _id: &str) -> Option<String> {
        self.stream_text.clone()
    }
    fn stream_finish(&mut self, id: &str) -> Result<()> {
        self.calls.push(format!("stream_finish {id}"));
        Ok(())
    }
    fn stream_stop(&mut self, id: &str) -> Result<()> {
        self.calls.push(format!("stream_stop {id}"));
        Ok(())
    }
    fn transcribe(&mut self, path: &Path, id: &str) -> Result<()> {
        let bytes = std::fs::metadata(path).map(|meta| meta.len()).unwrap_or(0);
        self.calls
            .push(format!("transcribe {id} wav_bytes={bytes}"));
        if self.auto {
            self.events.push(asr(id, &self.transcript.clone(), false));
        }
        Ok(())
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
        if self.fail_polish {
            bail!("the correction model is not ready yet");
        }
        let (app, textbox) = (sorted(app), sorted(textbox));
        self.calls.push(format!(
            "polish {id} text={text} screen={screen_text} app={app} textbox={textbox} keep={keep_context}"
        ));
        if self.auto {
            self.events.push(polished(id, &format!("Refined: {text}")));
        }
        Ok(())
    }
}

fn asr(id: &str, text: &str, partial: bool) -> EngineEvent {
    EngineEvent::AsrResult {
        id: Some(id.to_string()),
        text: text.to_string(),
        seconds: 0.1,
        partial,
    }
}

fn polished(id: &str, text: &str) -> EngineEvent {
    EngineEvent::PolishResult {
        id: Some(id.to_string()),
        text: text.to_string(),
        latency_ms: 50.0,
        ttft_ms: 20.0,
        tok_s: 200.0,
        app_category: Some(AppCategory::WorkChat),
    }
}

fn engine_error(id: &str) -> EngineEvent {
    EngineEvent::Error {
        id: Some(id.to_string()),
        msg: "sidecar trouble".into(),
    }
}

static NEXT_DIRECTORY: AtomicUsize = AtomicUsize::new(0);

fn temp_dir(label: &str) -> PathBuf {
    let directory = std::env::temp_dir().join(format!(
        "phonon-ws-{label}-{}-{}",
        std::process::id(),
        NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

/// The JSON of one text frame from the server.
fn decode(frame: &[u8]) -> Value {
    assert_eq!(frame[0], 0x80 | ws::OP_TEXT, "a final text frame");
    let payload = match frame[1] {
        126 => &frame[4..],
        127 => &frame[10..],
        _ => &frame[2..],
    };
    serde_json::from_slice(payload).unwrap()
}

/// A `Remote` with a mock engine and clients that are plain channels (no sockets).
struct Harness {
    remote: Remote,
    host: MockHost,
    corpus: PathBuf,
    _wake: mpsc::Receiver<String>,
}

impl Harness {
    fn new() -> Self {
        let (waker, wake) = mpsc::channel();
        let corpus = temp_dir("corpus");
        Self {
            remote: Remote::with_corpus(waker, corpus.clone()),
            host: MockHost::default(),
            corpus,
            _wake: wake,
        }
    }

    fn tick(&mut self) {
        self.remote.tick(&mut self.host);
    }

    /// Connect client `conn` and return what it receives, after its `ready`.
    fn connect(&mut self, conn: u64) -> mpsc::Receiver<Vec<u8>> {
        let (out, frames) = mpsc::channel();
        self.remote
            .tx
            .send(Inbound::Open {
                conn,
                out,
                client: "test".into(),
            })
            .unwrap();
        self.tick();
        assert_eq!(next(&frames)["type"], "ready");
        frames
    }

    fn text(&mut self, conn: u64, value: Value) {
        self.remote.tx.send(Inbound::Text { conn, value }).unwrap();
        self.tick();
    }

    fn audio(&mut self, conn: u64, pcm: Vec<u8>) {
        self.remote.tx.send(Inbound::Audio { conn, pcm }).unwrap();
        self.tick();
    }

    fn event(&mut self, event: EngineEvent) -> bool {
        self.remote.take_event(&mut self.host, &event)
    }

    /// The corpus entries on disk.
    fn recordings(&self) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(&self.corpus)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    /// Start a dictation as client 1 and send one second of audio.
    fn record(&mut self, frames: &mpsc::Receiver<Vec<u8>>) {
        self.text(1, json!({"type": "start", "id": "d1"}));
        assert_eq!(next(frames), json!({"type": "started", "id": "d1"}));
        self.audio(1, vec![1; 32_000]);
    }
}

fn next(frames: &mpsc::Receiver<Vec<u8>>) -> Value {
    decode(&frames.try_recv().expect("a message for the client"))
}

fn nothing(frames: &mpsc::Receiver<Vec<u8>>) {
    assert!(
        frames.try_recv().is_err(),
        "no message for the client was expected"
    );
}

#[test]
fn settings_decide_whether_and_where_to_listen() {
    let token = "0123456789abcdef0123456789abcdef";
    assert_eq!(config_from_settings(&json!({}), None).unwrap(), None);
    assert_eq!(config_from_settings(&Value::Null, None).unwrap(), None);
    assert_eq!(
        config_from_settings(
            &json!({"websocket_api": false, "websocket_api_token": token}),
            Some(token)
        )
        .unwrap(),
        None
    );
    assert_eq!(
        config_from_settings(
            &json!({"websocket_api": true, "websocket_api_token": token}),
            None
        )
        .unwrap(),
        Some(Config {
            bind: "0.0.0.0".parse().unwrap(),
            port: DEFAULT_PORT,
            token: token.into()
        })
    );
    assert_eq!(
        config_from_settings(
            &json!({
                "websocket_api": true, "websocket_api_token": token,
                "websocket_api_port": 9000, "websocket_api_bind": "100.101.102.103"
            }),
            None
        )
        .unwrap(),
        Some(Config {
            bind: "100.101.102.103".parse().unwrap(),
            port: 9000,
            token: token.into()
        })
    );
    // the token the bar sent comes before one in the file, and is enough alone
    assert_eq!(
        config_from_settings(
            &json!({"websocket_api": true, "websocket_api_token": token}),
            Some("from-the-keychain-0123456789")
        )
        .unwrap()
        .unwrap()
        .token,
        "from-the-keychain-0123456789"
    );
    assert!(
        config_from_settings(&json!({"websocket_api": true}), Some(token))
            .unwrap()
            .is_some()
    );
    // on, but not servable as written: an error, never a listener without a token
    for broken in [
        json!({"websocket_api": true}),
        json!({"websocket_api": true, "websocket_api_token": "short"}),
        json!({"websocket_api": true, "websocket_api_token": token, "websocket_api_port": 70000}),
        json!({"websocket_api": true, "websocket_api_token": token, "websocket_api_port": 0}),
        json!({"websocket_api": true, "websocket_api_token": token, "websocket_api_bind": "mac.local"}),
    ] {
        assert!(config_from_settings(&broken, None).is_err(), "{broken}");
    }
}

#[test]
fn tokens_compare_whole() {
    assert!(token_matches("abcdef", "abcdef"));
    assert!(!token_matches("abcdeg", "abcdef"));
    assert!(!token_matches("abcde", "abcdef"));
    assert!(!token_matches("abcdefg", "abcdef"));
    assert!(!token_matches("", "abcdef"));
}

#[test]
fn a_client_app_becomes_the_engine_app() {
    assert_eq!(
        engine_app(&json!({"package": "com.Slack", "name": "Slack"})),
        Some(json!({"bundle_id": "com.Slack", "name": "Slack"}))
    );
    assert_eq!(
        engine_app(&json!({"bundle_id": "com.android.chrome", "url_host": "mail.google.com"})),
        Some(json!({"bundle_id": "com.android.chrome", "name": "", "url_host": "mail.google.com"}))
    );
    assert_eq!(engine_app(&json!({"name": "Nameless"})), None);
    assert_eq!(engine_app(&json!("com.Slack")), None);
    // and the engine sorts an Android package into its category
    let app = engine_app(&json!({"package": "com.Slack"})).unwrap();
    let context = crate::app_context::AppContext::from_request(&app).unwrap();
    assert_eq!(context.category, AppCategory::WorkChat);
}

#[test]
fn the_final_text_falls_back_to_the_transcript() {
    assert_eq!(
        safe_final_text("hello there", " Hello there. "),
        "Hello there."
    );
    assert_eq!(safe_final_text("hello there", "  "), "hello there");
    assert_eq!(
        safe_final_text("hello", "<phonon_dictionary>x</phonon_dictionary> hello"),
        "hello"
    );
    let runaway = "word ".repeat(40);
    assert_eq!(safe_final_text("hello there", &runaway), "hello there");
}

#[test]
fn the_recording_is_a_mono_16k_wav_the_speech_gate_reads() {
    let path = temp_dir("wav").join("ws_1").join("audio.wav");
    let pcm: Vec<u8> = (0..32_000u32).map(|i| (i % 251) as u8).collect();
    write_wav_pcm16(&path, &pcm).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(bytes.len(), 44 + pcm.len());
    assert_eq!(&bytes[..4], b"RIFF");
    assert_eq!(
        u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
        16_000
    );
    assert_eq!(&bytes[44..], &pcm[..]);
    // the engine's own reader accepts it
    assert!(phonon_audio::wav_has_speech(&path).is_ok());
}

#[test]
fn a_dictation_runs_from_start_to_the_refined_text() {
    let mut h = Harness::new();
    h.remote.keep_history = true;
    let frames = h.connect(1);
    h.text(
        1,
        json!({"type": "start", "id": "d1", "live": true, "screen_text": "Kubernetes",
               "app": {"package": "com.Slack", "name": "Slack"},
               "textbox": {"before_text": "Hi ", "selected_text": "", "after_text": ""}}),
    );
    assert_eq!(next(&frames), json!({"type": "started", "id": "d1"}));
    assert_eq!(
        h.host.calls,
        vec![
            r#"stream_start ws-1-1 live=true app={"bundle_id":"com.Slack","name":"Slack"}"#,
            "stream_screen ws-1-1 Kubernetes"
        ]
    );

    // One second in uneven pieces: the sidecar gets 0.4 s chunks, the rest waits.
    let pcm: Vec<u8> = (0..32_000u32).map(|i| (i % 251) as u8).collect();
    for piece in pcm.chunks(5_000) {
        h.audio(1, piece.to_vec());
    }
    assert_eq!(
        h.host.chunks,
        vec![ws::base64(&pcm[..12_800]), ws::base64(&pcm[12_800..25_600])]
    );

    h.text(1, json!({"type": "finish", "id": "d1"}));
    assert_eq!(h.host.chunks[2], ws::base64(&pcm[25_600..]), "the tail");
    assert_eq!(
        &h.host.calls[5..],
        ["stream_finish ws-1-1", "transcribe ws-1-1 wav_bytes=32044"]
    );
    nothing(&frames);

    // the engine's events for this request go to the client, not to the bar
    assert!(h.event(asr("ws-1-1", " we deploy on kubernetes ", false)));
    assert_eq!(
        h.host.calls.last().unwrap(),
        "polish ws-1-1 text=we deploy on kubernetes screen=Kubernetes \
         app={\"bundle_id\":\"com.Slack\",\"name\":\"Slack\"} \
         textbox={\"after_text\":\"\",\"before_text\":\"Hi \",\"selected_text\":\"\"} keep=true"
    );
    assert!(h.event(polished("ws-1-1", "We deploy on Kubernetes.")));
    assert_eq!(
        next(&frames),
        json!({"type": "final", "id": "d1", "text": "We deploy on Kubernetes.",
               "raw": "we deploy on kubernetes", "refined": true, "app_category": "work_chat"})
    );
    assert_eq!(h.recordings().len(), 1, "local history on: the entry stays");
    assert!(h.remote.session.is_none());

    // and the next dictation can start
    h.text(1, json!({"type": "start", "id": "d2"}));
    assert_eq!(next(&frames), json!({"type": "started", "id": "d2"}));
}

#[test]
fn without_local_history_no_recording_is_left() {
    let mut h = Harness::new();
    let frames = h.connect(1);
    h.record(&frames);
    h.text(1, json!({"type": "finish", "id": "d1"}));
    assert_eq!(h.recordings().len(), 1, "the engine reads it from a file");
    h.event(asr("ws-1-1", "hello", false));
    assert!(h.host.calls.last().unwrap().ends_with("keep=false"));
    h.event(polished("ws-1-1", "Hello."));
    assert_eq!(next(&frames)["text"], "Hello.");
    assert!(h.recordings().is_empty());
}

#[test]
fn transcripts_and_drafts_reach_the_client_while_it_talks() {
    let mut h = Harness::new();
    let frames = h.connect(1);
    h.text(1, json!({"type": "start", "id": "d1", "live": true}));
    next(&frames);
    // streaming refine: the engine holds the transcript so far
    h.host.stream_text = Some("we deploy".into());
    h.tick();
    assert_eq!(
        next(&frames),
        json!({"type": "partial", "id": "d1", "text": "we deploy"})
    );
    h.tick();
    nothing(&frames); // unchanged: said once
                      // without it: the preview stream's partial results
    assert!(h.event(asr("ws-1-1", "we deploy on", true)));
    assert_eq!(next(&frames)["text"], "we deploy on");
    assert!(h.event(EngineEvent::DraftResult {
        id: "ws-1-1".into(),
        text: "We deploy on".into(),
        source: "we deploy on".into(),
        complete: true,
    }));
    assert_eq!(
        next(&frames),
        json!({"type": "draft", "id": "d1", "text": "We deploy on", "complete": true})
    );
}

#[test]
fn drafts_stay_home_unless_asked_for() {
    let mut h = Harness::new();
    let frames = h.connect(1);
    h.record(&frames);
    assert!(h.event(EngineEvent::DraftResult {
        id: "ws-1-1".into(),
        text: "Draft".into(),
        source: "draft".into(),
        complete: true,
    }));
    nothing(&frames);
}

#[test]
fn only_this_api_s_events_are_taken() {
    let mut h = Harness::new();
    let _frames = h.connect(1);
    assert!(!h.event(asr("p7", "the bar's own", false)));
    assert!(!h.event(polished("p7", "The bar's own.")));
    assert!(!h.event(EngineEvent::Error {
        id: None,
        msg: "x".into()
    }));
    assert!(!h.event(EngineEvent::AllReady {
        stream_refine: true
    }));
    // late news of a dictation that is over is swallowed, never shown to the bar
    assert!(h.event(polished("ws-9-9", "late")));
}

#[test]
fn silence_and_stubs_end_without_text() {
    let mut h = Harness::new();
    let frames = h.connect(1);
    h.text(1, json!({"type": "start", "id": "d1"}));
    next(&frames);
    h.audio(1, vec![0; 1_000]); // 31 ms
    h.text(1, json!({"type": "finish", "id": "d1"}));
    assert_eq!(
        next(&frames),
        json!({"type": "final", "id": "d1", "text": "", "raw": "", "refined": false,
               "reason": "too_short"})
    );
    assert_eq!(h.host.calls.last().unwrap(), "stream_stop ws-1-1");
    assert!(h.recordings().is_empty());

    h.remote.keep_history = true;
    h.text(1, json!({"type": "start", "id": "d2"}));
    next(&frames);
    h.audio(1, vec![0; 32_000]);
    h.text(1, json!({"type": "finish", "id": "d2"}));
    h.event(asr("ws-1-2", "  ", false));
    assert_eq!(next(&frames)["reason"], "no_speech");
    assert!(h.recordings().is_empty(), "nothing was said: nothing kept");
    assert!(h.remote.session.is_none());
}

#[test]
fn a_failed_refine_still_gives_the_transcript() {
    let mut h = Harness::new();
    let frames = h.connect(1);
    h.record(&frames);
    h.text(1, json!({"type": "finish", "id": "d1"}));
    h.host.fail_polish = true;
    h.event(asr("ws-1-1", "hello there", false));
    assert_eq!(
        next(&frames),
        json!({"type": "final", "id": "d1", "text": "hello there", "raw": "hello there",
               "refined": false, "reason": "refine_failed"})
    );

    // and so does a refiner error after the request went out
    h.host.fail_polish = false;
    h.text(1, json!({"type": "start", "id": "d2"}));
    next(&frames);
    h.audio(1, vec![1; 32_000]);
    h.text(1, json!({"type": "finish", "id": "d2"}));
    h.event(asr("ws-1-2", "second try", false));
    h.event(engine_error("ws-1-2"));
    let last = next(&frames);
    assert_eq!(last["text"], "second try");
    assert_eq!(last["refined"], false);
}

#[test]
fn engine_errors_end_a_dictation_only_after_finish() {
    let mut h = Harness::new();
    let frames = h.connect(1);
    h.record(&frames);
    h.event(engine_error("ws-1-1"));
    assert_eq!(
        next(&frames),
        json!({"type": "warning", "id": "d1", "msg": "sidecar trouble"})
    );
    assert!(h.remote.session.is_some(), "the recording is still good");
    h.text(1, json!({"type": "finish", "id": "d1"}));
    h.event(engine_error("ws-1-1"));
    assert_eq!(next(&frames)["code"], "engine");
    assert!(h.remote.session.is_none());
    assert!(h.recordings().is_empty());
}

#[test]
fn an_answer_that_never_comes_times_out() {
    let mut h = Harness::new();
    let frames = h.connect(1);
    h.record(&frames);
    h.text(1, json!({"type": "finish", "id": "d1"}));
    h.tick();
    nothing(&frames);
    h.remote.session.as_mut().unwrap().deadline = Some(Instant::now());
    h.tick();
    assert_eq!(next(&frames)["code"], "timeout");
    assert!(h.remote.session.is_none());
    assert!(h.recordings().is_empty());
}

#[test]
fn one_dictation_at_a_time() {
    let mut h = Harness::new();
    let first = h.connect(1);
    let second = h.connect(2);
    h.record(&first);
    h.text(2, json!({"type": "start", "id": "x"}));
    assert_eq!(next(&second)["code"], "busy");
    h.text(1, json!({"type": "start", "id": "again"}));
    assert_eq!(next(&first)["code"], "busy");
    // audio and commands from the other client do not touch it
    h.audio(2, vec![2; 32_000]);
    h.text(2, json!({"type": "finish", "id": "d1"}));
    assert_eq!(next(&second)["code"], "bad_request");
    assert_eq!(h.remote.session.as_ref().unwrap().pcm.len(), 32_000);
    h.text(1, json!({"type": "status"}));
    assert_eq!(next(&first)["busy"], true);
}

#[test]
fn the_mac_s_own_dictation_comes_first() {
    let mut h = Harness::new();
    let frames = h.connect(1);
    // the bar is dictating: a client waits
    h.remote
        .note_bar_command(&json!({"cmd": "stream_start", "id": "p4"}));
    h.remote
        .note_bar_command(&json!({"cmd": "stream_chunk", "id": "p4", "pcm16": ""}));
    h.text(1, json!({"type": "start", "id": "d1"}));
    assert_eq!(next(&frames)["code"], "busy");
    // someone else's result does not free it; its own does
    h.remote.note_bar_event(&polished("p3", "old"));
    assert!(h.remote.bar_busy());
    h.remote.note_bar_event(&polished("p4", "done"));
    assert!(!h.remote.bar_busy());
    h.record(&frames);

    // a dictation starting on the Mac takes the engine from the client
    h.remote
        .note_bar_command(&json!({"cmd": "stream_start", "id": "p5"}));
    assert_eq!(
        next(&frames),
        json!({"type": "error", "id": "d1", "code": "preempted",
               "msg": "a dictation started on the Mac"})
    );
    assert!(h.remote.session.is_none());
    assert!(
        !h.host
            .calls
            .iter()
            .any(|call| call.starts_with("stream_stop")),
        "the open stream is the bar's now: it must not be stopped"
    );
    // an empty transcript ends the bar's dictation too
    h.remote.note_bar_event(&asr("p5", "", false));
    assert!(!h.remote.bar_busy());
    // a dictation the bar dropped frees it too
    h.remote
        .note_bar_command(&json!({"cmd": "stream_start", "id": "p6"}));
    assert!(h.remote.bar_busy());
    h.remote.note_bar_command(&json!({"cmd": "dictation_end"}));
    assert!(!h.remote.bar_busy());
    // commands that are not a dictation do not hold the engine
    h.remote
        .note_bar_command(&json!({"cmd": "reload_dictionary"}));
    h.remote.note_bar_command(&json!({"cmd": "prewarm"}));
    assert!(!h.remote.bar_busy());
}

#[test]
fn a_dictation_past_its_audio_is_not_preempted() {
    let mut h = Harness::new();
    let frames = h.connect(1);
    h.record(&frames);
    h.text(1, json!({"type": "finish", "id": "d1"}));
    h.remote
        .note_bar_command(&json!({"cmd": "stream_start", "id": "p5"}));
    nothing(&frames);
    h.event(asr("ws-1-1", "hello", false));
    h.event(polished("ws-1-1", "Hello."));
    assert_eq!(next(&frames)["text"], "Hello.");
}

#[test]
fn context_can_follow_the_start() {
    let mut h = Harness::new();
    let frames = h.connect(1);
    h.record(&frames);
    h.text(
        1,
        json!({"type": "context", "id": "d1", "app": {"package": "com.discord"},
               "screen_text": "Blackwell", "textbox": {"before_text": "yo "}}),
    );
    assert_eq!(
        &h.host.calls[h.host.calls.len() - 2..],
        [
            r#"stream_app ws-1-1 {"bundle_id":"com.discord","name":""}"#,
            "stream_screen ws-1-1 Blackwell"
        ]
    );
    h.text(1, json!({"type": "finish", "id": "d1"}));
    h.event(asr("ws-1-1", "hey", false));
    let polish = h.host.calls.last().unwrap();
    assert!(polish.contains("screen=Blackwell"), "{polish}");
    assert!(polish.contains("com.discord"), "{polish}");
    assert!(polish.contains("before_text"), "{polish}");
    h.text(
        1,
        json!({"type": "context", "id": "nope", "screen_text": "x"}),
    );
    assert_eq!(next(&frames)["code"], "bad_request");
}

#[test]
fn cancel_and_disconnect_drop_the_dictation() {
    let mut h = Harness::new();
    let frames = h.connect(1);
    h.record(&frames);
    h.text(1, json!({"type": "cancel", "id": "d1"}));
    assert_eq!(next(&frames), json!({"type": "cancelled", "id": "d1"}));
    assert_eq!(h.host.calls.last().unwrap(), "stream_stop ws-1-1");
    assert!(h.remote.session.is_none());

    // gone mid-way through the transcription: the recording goes too
    h.remote.keep_history = true;
    h.text(1, json!({"type": "start", "id": "d2"}));
    next(&frames);
    h.audio(1, vec![1; 32_000]);
    h.text(1, json!({"type": "finish", "id": "d2"}));
    assert_eq!(h.recordings().len(), 1);
    h.remote.tx.send(Inbound::Closed { conn: 1 }).unwrap();
    h.tick();
    assert!(h.remote.session.is_none());
    assert!(h.remote.clients.is_empty());
    assert!(h.recordings().is_empty());
    // its late results are swallowed
    assert!(h.event(asr("ws-1-2", "hello", false)));
    assert!(!h.host.calls.last().unwrap().starts_with("polish"));
}

#[test]
fn text_alone_can_be_refined() {
    let mut h = Harness::new();
    let frames = h.connect(1);
    h.text(
        1,
        json!({"type": "refine", "id": "r1", "text": " i will be there at five ",
               "app": {"package": "com.google.android.gm"}}),
    );
    assert_eq!(
        h.host.calls,
        vec![
            "polish ws-1-1 text=i will be there at five screen= \
             app={\"bundle_id\":\"com.google.android.gm\",\"name\":\"\"} textbox= keep=false"
        ]
    );
    // a dictation can run meanwhile; each result finds its request
    h.record(&frames);
    assert!(h.event(polished("ws-1-1", "I will be there at five.")));
    assert_eq!(
        next(&frames),
        json!({"type": "final", "id": "r1", "text": "I will be there at five.",
               "raw": "i will be there at five", "refined": true, "app_category": "work_chat"})
    );
    assert!(h.remote.refines.is_empty());
    assert!(h.remote.session.is_some());

    h.text(1, json!({"type": "refine", "id": "r2", "text": "  "}));
    assert_eq!(next(&frames)["code"], "bad_request");
    h.text(1, json!({"type": "refine", "id": "r3", "text": "x"}));
    h.event(engine_error("ws-1-3"));
    assert_eq!(next(&frames)["code"], "engine");
}

#[test]
fn requests_are_checked() {
    let mut h = Harness::new();
    let frames = h.connect(1);
    h.text(1, json!({"type": "ping"}));
    assert_eq!(next(&frames), json!({"type": "pong"}));
    h.text(1, json!({"type": "start"}));
    assert_eq!(next(&frames)["code"], "bad_request");
    h.text(1, json!({"type": "dance", "id": "d1"}));
    assert_eq!(next(&frames)["code"], "bad_request");
    h.text(
        1,
        json!({"type": "start", "id": "d1", "sample_rate": 48_000}),
    );
    assert_eq!(next(&frames)["code"], "bad_request");
    h.text(1, json!({"type": "finish", "id": "d1"}));
    assert_eq!(next(&frames)["code"], "bad_request");
    h.audio(1, vec![0; 100]); // no dictation open: dropped
    nothing(&frames);
    assert!(h.host.calls.is_empty());

    h.host.not_ready = true;
    h.text(1, json!({"type": "start", "id": "d1"}));
    assert_eq!(next(&frames)["code"], "not_ready");
    h.text(1, json!({"type": "refine", "id": "r1", "text": "x"}));
    assert_eq!(next(&frames)["code"], "not_ready");
    h.text(1, json!({"type": "status"}));
    assert_eq!(next(&frames)["asr_ready"], false);
}

#[test]
fn a_dictation_cannot_grow_without_end() {
    let mut h = Harness::new();
    let frames = h.connect(1);
    h.record(&frames);
    h.remote.session.as_mut().unwrap().pcm = vec![0; MAX_PCM_BYTES - 10];
    h.remote.session.as_mut().unwrap().sent = MAX_PCM_BYTES - 10;
    h.audio(1, vec![0; 100]);
    assert_eq!(next(&frames)["code"], "too_long");
    assert!(h.remote.session.is_none());
}

#[test]
fn the_settings_file_turns_the_listener_on_and_off() {
    let mut h = Harness::new();
    let directory = temp_dir("settings");
    let path = directory.join("settings.json");
    h.remote.settings_path = Some(path.clone());
    let follow = |h: &mut Harness| {
        h.remote.settings_checked = None; // not a second later: now
        h.tick();
    };
    follow(&mut h);
    assert!(h.remote.listening().is_none(), "no file: off");

    // A free port below the range the system hands out on its own: the other tests'
    // sockets land up there, and one of them taking this port between two steps of this
    // test would fail it for no fault of the listener.
    let port = (0..200u16)
        .map(|step| 21_000 + (std::process::id() % 20_000) as u16 + step)
        .find(|port| TcpListener::bind(("127.0.0.1", *port)).is_ok())
        .expect("a free port");
    let write = |extra: Value| {
        let mut settings = json!({
            "schema_version": 2, "websocket_api_port": port,
            "websocket_api_bind": "127.0.0.1",
            "websocket_api_token": "0123456789abcdef0123456789abcdef",
        });
        for (key, value) in extra.as_object().unwrap() {
            settings[key] = value.clone();
        }
        std::fs::write(&path, settings.to_string()).unwrap();
    };
    write(json!({"websocket_api": false}));
    follow(&mut h);
    assert!(h.remote.listening().is_none());

    write(json!({"websocket_api": true, "local_history": true}));
    follow(&mut h);
    assert_eq!(h.remote.listening().unwrap().port(), port);
    assert!(h.remote.keep_history);
    assert!(TcpStream::connect(("127.0.0.1", port)).is_ok());

    write(json!({"websocket_api": false, "local_history": false, "pad": 1}));
    follow(&mut h);
    assert!(h.remote.listening().is_none());
    assert!(!h.remote.keep_history);
    // the port is free again
    assert!(TcpListener::bind(("127.0.0.1", port)).is_ok());

    // on without a token: stays off, and the bar is told why
    while h.remote.take_status().is_some() {}
    std::fs::write(
        &path,
        json!({"websocket_api": true, "websocket_api_port": port,
               "websocket_api_bind": "127.0.0.1"})
        .to_string(),
    )
    .unwrap();
    follow(&mut h);
    assert!(h.remote.listening().is_none());
    assert_eq!(
        h.remote.take_status().unwrap(),
        json!({"type": "status", "kind": "websocket_api", "listening": false,
               "address": null, "msg": "on, but no token has been set"})
    );
    assert_eq!(h.remote.take_status(), None, "said once");

    // the bar sends the token from the Keychain: it listens, without a second's wait
    h.remote.set_token(Some("from-the-keychain-0123456789"));
    h.tick();
    assert_eq!(h.remote.listening().unwrap().port(), port);
    assert_eq!(
        h.remote.take_status().unwrap(),
        json!({"type": "status", "kind": "websocket_api", "listening": true,
               "address": format!("127.0.0.1:{port}"), "msg": ""})
    );
    let mut client = TestClient::connect(
        h.remote.listening().unwrap(),
        "from-the-keychain-0123456789",
    );
    for _ in 0..200 {
        h.tick();
        if !h.remote.clients.is_empty() {
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(client.message()["type"], "ready");

    // a new token: the old one's connections are closed, and it no longer opens the door
    h.remote.set_token(Some("a-newer-token-0123456789abc"));
    h.tick();
    assert!(client.frame().is_err());
    let mut stale = TestClient::connect(
        h.remote.listening().unwrap(),
        "from-the-keychain-0123456789",
    );
    assert_eq!(stale.message()["code"], "unauthorized");

    // the token withdrawn: nothing listens
    h.remote.set_token(None);
    h.tick();
    assert!(h.remote.listening().is_none());
    assert_eq!(h.remote.take_status().unwrap()["listening"], false);
}

// ---- over a real socket ----

/// A WebSocket client, as much of one as these tests need.
struct TestClient {
    stream: TcpStream,
}

impl TestClient {
    /// Open a socket and upgrade it. Err when the server refuses the handshake.
    fn open(addr: SocketAddr) -> Result<Self> {
        let mut stream = TcpStream::connect(addr)?;
        stream.set_read_timeout(Some(Duration::from_secs(5)))?;
        stream.write_all(
            b"GET / HTTP/1.1\r\nHost: test\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\
              Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n",
        )?;
        let mut head = Vec::new();
        let mut byte = [0u8; 1];
        while !head.ends_with(b"\r\n\r\n") {
            stream.read_exact(&mut byte)?;
            head.push(byte[0]);
        }
        let head = String::from_utf8_lossy(&head).into_owned();
        if !head.starts_with("HTTP/1.1 101") {
            bail!("refused: {head}");
        }
        assert!(head.contains("Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo="));
        Ok(Self { stream })
    }

    fn connect(addr: SocketAddr, token: &str) -> Self {
        let mut client = Self::open(addr).unwrap();
        client.send(json!({"type": "hello", "token": token, "client": "test"}));
        client
    }

    fn send(&mut self, value: Value) {
        let frame = ws::encode_masked(
            ws::OP_TEXT,
            true,
            value.to_string().as_bytes(),
            [7, 3, 9, 1],
        );
        self.stream.write_all(&frame).unwrap();
    }

    fn send_audio(&mut self, pcm: &[u8]) {
        let frame = ws::encode_masked(ws::OP_BINARY, true, pcm, [4, 4, 2, 8]);
        self.stream.write_all(&frame).unwrap();
    }

    /// The next frame: its opcode and payload. Err when the connection is closed.
    fn frame(&mut self) -> Result<(u8, Vec<u8>)> {
        let mut head = [0u8; 2];
        self.stream.read_exact(&mut head)?;
        assert_eq!(head[1] & 0x80, 0, "server frames are not masked");
        let len = match head[1] {
            126 => {
                let mut bytes = [0u8; 2];
                self.stream.read_exact(&mut bytes)?;
                u16::from_be_bytes(bytes) as usize
            }
            127 => {
                let mut bytes = [0u8; 8];
                self.stream.read_exact(&mut bytes)?;
                u64::from_be_bytes(bytes) as usize
            }
            short => short as usize,
        };
        let mut payload = vec![0u8; len];
        self.stream.read_exact(&mut payload)?;
        Ok((head[0] & 0x0F, payload))
    }

    fn message(&mut self) -> Value {
        let (opcode, payload) = self.frame().expect("a message from the server");
        assert_eq!(opcode, ws::OP_TEXT);
        serde_json::from_slice(&payload).unwrap()
    }
}

/// A `Remote` listening on a loopback port, turned by a thread the way the engine loop
/// turns it: tick, then the mock engine's events.
struct Listening {
    addr: SocketAddr,
    stop: Arc<AtomicBool>,
    pump: Option<thread::JoinHandle<(Remote, MockHost)>>,
}

const TOKEN: &str = "test-token-0123456789abcdef";

fn listen(host: MockHost) -> Listening {
    let (waker, wake) = mpsc::channel::<String>();
    let mut remote = Remote::with_corpus(waker, temp_dir("corpus"));
    remote.apply(Some(Config {
        bind: Ipv4Addr::LOCALHOST.into(),
        port: 0,
        token: TOKEN.into(),
    }));
    let addr = remote.listening().expect("listening");
    let stop = Arc::new(AtomicBool::new(false));
    let stopped = stop.clone();
    let pump = thread::spawn(move || {
        let mut host = host;
        while !stopped.load(Ordering::SeqCst) {
            // the engine loop blocks on this same channel
            let _ = wake.recv_timeout(Duration::from_millis(20));
            remote.tick(&mut host);
            for event in std::mem::take(&mut host.events) {
                remote.take_event(&mut host, &event);
            }
        }
        (remote, host)
    });
    Listening {
        addr,
        stop,
        pump: Some(pump),
    }
}

impl Listening {
    fn finish(mut self) -> (Remote, MockHost) {
        self.stop.store(true, Ordering::SeqCst);
        self.pump.take().unwrap().join().unwrap()
    }
}

#[test]
fn a_client_dictates_over_a_socket() {
    let server = listen(MockHost {
        auto: true,
        transcript: "we deploy on kubernetes".into(),
        ..MockHost::default()
    });
    let mut client = TestClient::connect(server.addr, TOKEN);
    let ready = client.message();
    assert_eq!(ready["type"], "ready");
    assert_eq!(ready["protocol"], 1);
    assert_eq!(ready["busy"], false);

    client.send(json!({"type": "start", "id": "d1", "app": {"package": "com.Slack"}}));
    assert_eq!(client.message(), json!({"type": "started", "id": "d1"}));
    let pcm: Vec<u8> = (0..48_000u32).map(|i| (i % 249) as u8).collect();
    for piece in pcm.chunks(3_200) {
        client.send_audio(piece);
    }
    client.send(json!({"type": "finish", "id": "d1"}));
    assert_eq!(
        client.message(),
        json!({"type": "final", "id": "d1", "text": "Refined: we deploy on kubernetes",
               "raw": "we deploy on kubernetes", "refined": true, "app_category": "work_chat"})
    );

    // a WebSocket ping is answered by the connection itself
    client
        .stream
        .write_all(&ws::encode_masked(ws::OP_PING, true, b"hi", [1, 2, 3, 4]))
        .unwrap();
    assert_eq!(client.frame().unwrap(), (ws::OP_PONG, b"hi".to_vec()));
    client.send(json!({"type": "ping"}));
    assert_eq!(client.message(), json!({"type": "pong"}));
    // a frame that is not a JSON object is answered, and the connection lives on
    client
        .stream
        .write_all(&ws::encode_masked(ws::OP_TEXT, true, b"[1]", [1, 2, 3, 4]))
        .unwrap();
    assert_eq!(client.message()["code"], "bad_request");

    // closing is answered and noticed
    client
        .stream
        .write_all(&ws::encode_masked(ws::OP_CLOSE, true, &[], [1, 2, 3, 4]))
        .unwrap();
    assert_eq!(client.frame().unwrap().0, ws::OP_CLOSE);
    assert!(client.frame().is_err(), "the server hangs up");
    thread::sleep(Duration::from_millis(100));

    let (remote, host) = server.finish();
    assert!(remote.clients.is_empty());
    // every byte arrived, in order, in 0.4 s chunks and a tail
    let sent: String = host.chunks.concat();
    let expected: String = pcm.chunks(12_800).map(ws::base64).collect();
    assert_eq!(sent, expected);
}

#[test]
fn the_token_is_the_way_in() {
    let server = listen(MockHost::default());
    // a wrong token: told so, then hung up on
    let mut wrong = TestClient::connect(server.addr, "not-the-token-0123456789");
    let refusal = wrong.message();
    assert_eq!(refusal["code"], "unauthorized");
    assert_eq!(wrong.frame().unwrap().0, ws::OP_CLOSE);
    assert!(wrong.frame().is_err());

    // a first message that is not hello: the same
    let mut eager = TestClient::open(server.addr).unwrap();
    eager.send(json!({"type": "start", "id": "d1"}));
    assert_eq!(eager.message()["code"], "unauthorized");

    // audio first: dropped without a word
    let mut noise = TestClient::open(server.addr).unwrap();
    noise.send_audio(&[0; 64]);
    assert!(noise.frame().is_err());

    // plain HTTP: refused before any WebSocket
    let mut plain = TcpStream::connect(server.addr).unwrap();
    plain
        .write_all(b"GET / HTTP/1.1\r\nHost: test\r\n\r\n")
        .unwrap();
    let mut answer = String::new();
    let _ = plain.read_to_string(&mut answer);
    assert!(answer.starts_with("HTTP/1.1 400"), "{answer}");

    // and the right one still works afterwards
    let mut right = TestClient::connect(server.addr, TOKEN);
    assert_eq!(right.message()["type"], "ready");
    thread::sleep(Duration::from_millis(50));
    let (remote, host) = server.finish();
    assert_eq!(
        remote.clients.len(),
        1,
        "only the client that knew the token"
    );
    assert!(host.calls.is_empty(), "nothing reached the engine");
}

#[test]
fn turning_the_api_off_hangs_up() {
    let (waker, _wake) = mpsc::channel::<String>();
    let mut remote = Remote::with_corpus(waker, temp_dir("corpus"));
    let mut host = MockHost::default();
    let config = Config {
        bind: Ipv4Addr::LOCALHOST.into(),
        port: 0,
        token: TOKEN.into(),
    };
    remote.apply(Some(config));
    let addr = remote.listening().unwrap();
    let mut client = TestClient::connect(addr, TOKEN);
    // the hello is read on the connection's thread; wait for it to reach the loop
    for _ in 0..200 {
        remote.tick(&mut host);
        if !remote.clients.is_empty() {
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(client.message()["type"], "ready");
    remote.apply(None);
    assert!(remote.listening().is_none());
    assert!(client.frame().is_err(), "the connection is closed");
    assert!(TestClient::open(addr).is_err(), "and nothing listens");
}

#[test]
fn the_bar_hears_when_a_client_dictates() {
    let mut h = Harness::new();
    let frames = h.connect(1);
    assert!(h.remote.take_dictation_events().is_empty(), "nothing yet");
    h.record(&frames);
    assert_eq!(
        h.remote.take_dictation_events(),
        vec![json!({"type": "status", "kind": "websocket_dictation", "state": "started"})]
    );
    assert!(h.remote.take_dictation_events().is_empty(), "said once");
    h.text(1, json!({"type": "finish", "id": "d1"}));
    assert!(h.remote.take_dictation_events().is_empty(), "still running");
    h.event(asr("ws-1-1", "we deploy on friday", false));
    h.event(polished("ws-1-1", "We deploy on Friday."));
    next(&frames);
    assert_eq!(
        h.remote.take_dictation_events(),
        vec![
            json!({"type": "status", "kind": "websocket_dictation", "state": "ended",
                    "words": 4, "audio_ms": 1000})
        ]
    );

    // one that ends without text counts no words; the next one starts fresh
    h.text(1, json!({"type": "start", "id": "d2"}));
    next(&frames);
    h.remote.take_dictation_events();
    h.text(1, json!({"type": "cancel", "id": "d2"}));
    h.text(1, json!({"type": "start", "id": "d3"}));
    next(&frames);
    next(&frames);
    assert_eq!(
        h.remote.take_dictation_events(),
        vec![
            json!({"type": "status", "kind": "websocket_dictation", "state": "ended",
                   "words": 0, "audio_ms": 0}),
            json!({"type": "status", "kind": "websocket_dictation", "state": "started"}),
        ]
    );
}
