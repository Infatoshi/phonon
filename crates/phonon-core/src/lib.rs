//! Shared warm engines: Parakeet ASR + local correction model over JSONL.
//!
//! Protocol (stdin/stdout, one JSON object per line):
//!   {"cmd":"status"}
//!   {"cmd":"transcribe","path":"/tmp/x.wav","id":"..."}
//!   {"cmd":"polish","text":"...","id":"...","context":"<ocr>","app":{"bundle_id":"...","name":"...","url_host":"..."},"textbox":{...}}
//!   {"cmd":"prewarm"}  — dictation started; no event in reply
//!   {"cmd":"stream_start","id":"...","app":{...}} / {"cmd":"stream_chunk","id":"...","pcm16":"<base64>"} /
//!   {"cmd":"stream_stop","id":"..."}  — live audio; with streaming refine a segment stream
//!   {"cmd":"stream_finish","id":"..."}  — key up of a segment stream with all its audio sent:
//!   the release transcription starts now; the `transcribe` that follows releases the result
//!   {"cmd":"screen_context","id":"...","text":"..."}  — screen text for the open stream
//!   {"cmd":"stream_app","id":"...","app":{...}}  — the open stream's target app, once resolved
//!   {"cmd":"reload_dictionary"}
//!   {"cmd":"byok","base_url":"https://…/v1","model":"…","api_key":"…","timeout_ms":6000}
//!   — the user's own OpenAI-compatible endpoint refines instead of the local refiner,
//!   which stays the fallback; without base_url/model it is off. Memory only.
//!   {"cmd":"websocket_api","token":"…"}  — a token for the WebSocket API that overrides
//!   settings.json `websocket_api_token` for this run (`remote.rs`). Memory only.
//!   {"cmd":"edit_observed","id":"...","pasted":"...","edited":"...","end_reason":"...","record":false}
//!   {"cmd":"edit_decision","id":"...","index":0,"decision":"approved|rejected|undone"}
//!   {"cmd":"meeting_span","id":"...","chunks":[{"path":"...","start_sample":0}],"start_sample":0,"end_sample":0}
//!   — one span of a recorded meeting's track, run when no dictation is under way (meeting.rs)
//!   {"cmd":"dictation_end","id":"..."}  — a dictation ended with no WAV or stream to finish
//!   (discarded, too short, the mic failed): releases the hold `prewarm` / `stream_start` took
//!   {"cmd":"shutdown"}
//!
//! Events (stdout):
//!   {"type":"meeting_span","id":"...","sentences":[{"start":0.5,"end":1.2,"text":"..."}],"quiet":false,
//!    "audio_seconds":60,"seconds":0.4,"model":"..."}  — or {"type":"meeting_span","id":"...","error":"..."}
//!   {"type":"dictation","quiet":true}  — dictation stopped (true) or started (false) holding
//!   meeting work back (meeting.rs holds); sent at the first poll and on every change
//!   {"type":"stream","name":"asr|llm","state":"loading|ready|error","pct":0.5,"msg":"...","load_ms":123}
//!   {"type":"ready","stream_refine":true}  — all required streams ready; streaming refine on
//!   (the bar then sends the last partial chunk at key up and {"cmd":"screen_context"})
//!   {"type":"result","kind":"asr|polish","id":"...","text":"...","seconds":0.1,"latency_ms":70,"ttft_ms":30,"tok_s":200}
//!   {"type":"result","kind":"edit","id":"...","outcome":"correction","suggestions":[...],"recorded":false}
//!   {"type":"result","kind":"draft","id":"...","text":"...","source":"...","complete":true}  — live
//!   text (settings.json `live_text`): the open stream's corrected text so far, display only
//!   {"type":"status","kind":"websocket_api","listening":true,"address":"0.0.0.0:8377","msg":""}
//!   — the WebSocket API started, stopped or could not start (`msg` says why)
//!   {"type":"status","kind":"websocket_dictation","state":"started|ended","words":42,"audio_ms":9800}
//!   — a WebSocket client's dictation started or ended (`words` and `audio_ms` on `ended`)
//!   {"type":"error","id":"...","msg":"..."}

pub mod app_context;
pub mod data;
pub mod edits;
pub mod meeting;
pub mod pairs;
mod paths;
pub mod remote;
pub mod ws;

use anyhow::{bail, Context, Result};
pub use paths::project_root;
use phonon_asr::{AsrEvent, AsrSidecar};
use phonon_audio::wav_has_speech;
use phonon_llm::{polish_available, PolishEvent, PolishSidecar};
use serde_json::json;
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

const STARTUP_ASR_BATCH_ID: &str = "__phonon_startup_asr_batch__";
const STARTUP_ASR_STREAM_ID: &str = "__phonon_startup_asr_stream__";
const STARTUP_LLM_PRIME_ID: &str = "__phonon_startup_llm_prime__";
const STARTUP_LLM_AUDIO_ID: &str = "__phonon_startup_llm_audio__";
const STARTUP_LLM_ID: &str = "__phonon_startup_llm__";
const STARTUP_LLM_PRIME_TEXT: &str = "hello fluid voice startup prime";
const STARTUP_LLM_TEXT: &str = "Do you know the difference between Koo Bloss and Koo DNN?";

#[derive(Debug, Clone)]
pub enum EngineEvent {
    Stream {
        name: String,
        state: String,
        pct: f64,
        msg: String,
        load_ms: Option<f64>,
    },
    /// Both stacks are ready. `stream_refine`: the bar should send the last partial chunk
    /// at key up and the screen text while recording (streaming refine is on).
    AllReady {
        stream_refine: bool,
    },
    AsrResult {
        id: Option<String>,
        text: String,
        seconds: f64,
        partial: bool,
    },
    PolishResult {
        id: Option<String>,
        text: String,
        latency_ms: f64,
        ttft_ms: f64,
        tok_s: f64,
        app_category: Option<app_context::AppCategory>,
    },
    /// Live text: the corrected text so far of the open stream `id`, from a background
    /// refine. Display only; the polish result is still what gets inserted.
    /// `text` went through the same correction path as a polish result (`finish_correction`);
    /// `source` is the transcript so far it refines, for the bar's own final check.
    DraftResult {
        id: String,
        text: String,
        source: String,
        complete: bool,
    },
    /// A meeting span answered: the sidecar's reply (`sentences`, `quiet`, `audio_seconds`,
    /// `seconds`, `model`), or `{"error": msg}`. Never a dictation error.
    MeetingSpan {
        id: String,
        body: serde_json::Value,
    },
    /// Dictation started or stopped holding meeting work back (`meeting::QuietWatch`):
    /// `{"type":"dictation","quiet":..}`. The bar pauses its meeting cleanup while not quiet.
    DictationQuiet {
        quiet: bool,
    },
    Error {
        id: Option<String>,
        msg: String,
    },
}

pub struct Engine {
    asr: AsrSidecar,
    polisher: Option<PolishSidecar>,
    asr_ready: bool,
    llm_ready: bool,
    root: std::path::PathBuf,
    started: Instant,
    /// FIFO of polish requests (serve-json returns in order).
    polish_requests: Vec<PolishRequest>,
    /// The user's own OpenAI-compatible endpoint (`{"cmd":"byok"}`), off when None.
    byok: Option<phonon_llm::byok::Endpoint>,
    /// Endpoint requests in flight, by polish request id.
    byok_jobs: HashMap<String, ByokJob>,
    dictionary: data::DictionaryFile,
    styles: app_context::StylePreferenceCache,
    /// Capitalizations of ordinary words seen this session, in memory only.
    repeated_edits: edits::RepeatedEdits,
    recording_ids: HashMap<String, String>,
    pending_events: Vec<EngineEvent>,
    all_ready_emitted: bool,
    llm_primed: bool,
    llm_primed_at: Option<Instant>,
    startup_asr_text: Option<String>,
    startup_audio_demo_sent: bool,
    startup_audio_demo_passed: bool,
    startup_dictionary_demo_sent: bool,
    /// Refine while the speaker talks: segment streams in the ASR sidecar, each closed
    /// segment's transcript so far committed to the correction sidecar. settings.json
    /// `stream_refine` (default on), overridden by PHONON_STREAM_REFINE=0/1.
    stream_refine: bool,
    /// Transcripts so far received from the open segment stream, by request id.
    segment_streams: HashMap<String, Vec<String>>,
    /// Screen text captured at key down for the open stream, so background commits are built
    /// from the same dictionary envelope as the final polish.
    stream_screen_text: Option<(String, String)>,
    /// Segment streams the bar finished at key up (`stream_finish`) before its `transcribe`
    /// with the WAV: the release ASR runs while the WAV is written, and the result goes out
    /// once the WAV's speech gate has spoken.
    finishing: HashMap<String, StreamFinish>,
    /// The dictionary envelope of the open stream's last commit: (transcript, screen text,
    /// envelope). Building one costs about 0.18 ms per word (M4 Max, 40 ms at 228 words);
    /// when the speaker stops inside the last chunk the final polish asks for the same one.
    stream_envelope: Option<(String, String, String)>,
    /// The open stream's target app category (`stream_start` / `stream_app`), so background
    /// commits carry the writing mode the final polish will use; and the category and
    /// writing mode the last commit went out with.
    stream_category: Option<(String, app_context::AppCategory)>,
    /// The open stream's target bundle id, for a per-app writing mode.
    stream_bundle: Option<(String, String)>,
    stream_commit_category: (Option<app_context::AppCategory>, Option<String>),
    /// The open stream whose background drafts go out as `draft` results: settings.json
    /// `live_text` as of its `stream_start` (re-read only when the file changed), so a
    /// flipped setting applies from the next dictation without an engine restart.
    live_text_stream: Option<String>,
    /// Dictation as the meeting queue sees it, and the meeting spans waiting for the speech
    /// model (`meeting.rs`): one at a time, never while a dictation is under way.
    dictation: meeting::DictationActivity,
    meeting_spans: meeting::MeetingSpans,
    /// Last dictation state sent to the bar (`{"type":"dictation","quiet":..}`).
    dictation_quiet: meeting::QuietWatch,
}

/// Words a recognizer must produce for a clip the energy gate called silent to go out anyway.
const MIN_WORDS_PAST_GATE: usize = 3;

/// The two halves of a streamed dictation's final transcript (see `Engine::finishing`).
#[derive(Debug, Default)]
struct StreamFinish {
    text: Option<(String, f64)>,
    speech: Option<bool>,
    /// The sidecar could not finish the stream: the WAV is transcribed the usual way instead.
    failed: bool,
    /// The WAV, once the bar's `transcribe` has named it.
    path: Option<std::path::PathBuf>,
}

impl StreamFinish {
    /// What to send now (text, seconds), if anything, and whether the entry is finished.
    /// Both the ASR text and the gate are needed. When the energy gate finds no speech, the
    /// recognizer's words decide: `MIN_WORDS_PAST_GATE` or more go out, anything shorter
    /// (the "thank you" a recognizer makes of silence) becomes an empty result. The gate
    /// alone dropped quiet speech: 13 of the owner's 49 rejected clips over 1 s held 3-28
    /// words, while the rejected clips without speech gave Parakeet 0-1 words (2026-11-03).
    fn resolve(&mut self) -> (Option<(String, f64)>, bool) {
        match (self.speech, self.text.take()) {
            (Some(false), Some(text)) => {
                let spoken = text.0.split_whitespace().count() >= MIN_WORDS_PAST_GATE;
                (Some(if spoken { text } else { (String::new(), 0.0) }), true)
            }
            (Some(true), Some(text)) => (Some(text), true),
            (_, text) => {
                self.text = text;
                (None, false)
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
struct PolishRequest {
    id: Option<String>,
    source_text: String,
    screen_context_terms: Vec<String>,
    /// Where the dictation goes, for the corpus and the reply to the bar.
    app: Option<app_context::AppContext>,
    /// Style preferences the user approved for that app's category.
    style: Vec<String>,
    /// The text around the cursor, when the user turned that on.
    textbox: Option<app_context::TextboxContext>,
    /// Local history is on: the app and text-box context may be kept in the
    /// corpus entry. Off, nothing read from the screen or a field is written.
    keep_context: bool,
    /// The writing mode the text is laid out for (line breaks): the user's pick for
    /// the app, else the app category's mode.
    mode: Option<String>,
}

/// Dictionary terms sent to the user's own endpoint with a transcript.
const BYOK_VOCABULARY_CAP: usize = 80;

/// A request to the user's own endpoint (issue #24) running beside the local refiner.
struct ByokJob {
    deadline: Instant,
    reply: std::sync::mpsc::Receiver<Result<String>>,
}

/// The endpoint's text for request `id`, waiting until its deadline at most;
/// None (use the local result) on an error, a timeout or no request.
fn take_byok_result(jobs: &mut HashMap<String, ByokJob>, id: Option<&str>) -> Option<String> {
    let job = jobs.remove(id?)?;
    let wait = job.deadline.saturating_duration_since(Instant::now());
    match job.reply.recv_timeout(wait) {
        Ok(Ok(text)) => Some(text),
        Ok(Err(error)) => {
            eprintln!("phonon engine: own endpoint failed, local result used: {error:#}");
            None
        }
        Err(_) => {
            eprintln!("phonon engine: own endpoint timed out, local result used");
            None
        }
    }
}

/// The endpoint's text survives the output guard (it is not swapped back for the raw
/// transcript); otherwise the local refiner's result is the better fallback.
fn byok_text_passes(dictionary: &data::DictionaryFile, source: &str, remote: &str) -> bool {
    let guarded = safe_polish_output(dictionary, source, remote).text;
    let raw = dictionary.apply_exact_replacements(source).text;
    guarded != raw || dictionary.apply_exact_replacements(remote.trim()).text == raw
}

/// Line breaks of the inserted text: numbered lists one per line in every mode but
/// casual; the greeting and sign-off on their own lines only in email (`email_layout`),
/// never in Slack or Teams although they map to formal. A chat app in formal mode also
/// gets the greeting and sign-off newlines the refiner wrote itself joined back
/// (`linebreak::chat_join`); lists stay.
fn final_layout(text: &str, mode: &str, category: Option<app_context::AppCategory>) -> String {
    use app_context::AppCategory;
    if category == Some(AppCategory::Email) {
        app_context::email_layout(&phonon_llm::linebreak::lists(text))
    } else if mode == "formal"
        && matches!(category, Some(AppCategory::WorkChat | AppCategory::Chat))
    {
        phonon_llm::linebreak::layout(&phonon_llm::linebreak::chat_join(text), mode, false)
    } else {
        phonon_llm::linebreak::layout(text, mode, false)
    }
}

/// The refiner's writing mode for an app category (`polish_server.WRITING_MODE_BY_CATEGORY`).
fn category_mode(category: Option<app_context::AppCategory>) -> &'static str {
    match category.map(|category| category.as_str()) {
        Some("chat") => "casual",
        Some("work_chat") | Some("email") => "formal",
        Some("ai_prompt") | Some("code") => "prompt",
        _ => "minimal",
    }
}

impl Engine {
    pub fn start(root: &Path) -> Result<Self> {
        if !polish_available(root) {
            bail!("the required local correction runtime is missing from this install");
        }
        // Decided once per engine run and handed to both sidecars, which read it from
        // PHONON_STREAM_REFINE in their environment: the environment, else settings.json,
        // else on for Macs with 16 GB of memory or more.
        let stream_refine = stream_refine_setting(
            std::env::var("PHONON_STREAM_REFINE").ok(),
            data::SettingsFile::load()
                .ok()
                .and_then(|settings| settings.stream_refine)
                .unwrap_or_else(|| stream_refine_default(physical_memory_bytes())),
        );
        eprintln!(
            "phonon engine: streaming refine {}",
            if stream_refine { "on" } else { "off" }
        );
        let mut asr = AsrSidecar::spawn_with(root, stream_refine)?;
        let dictionary = data::DictionaryFile::load()?;
        if let Err(error) = asr.set_terms(&dictionary.asr_terms()) {
            eprintln!("phonon engine: ASR dictionary terms not sent: {error:#}");
        }
        let mut polish_config = data::polish_config()?;
        polish_config.stream_refine = stream_refine;
        let polisher = PolishSidecar::spawn_with(root, &polish_config)?
            .context("the required local correction runtime could not start")?;

        Ok(Self {
            asr,
            polisher: Some(polisher),
            asr_ready: false,
            llm_ready: false,
            root: root.to_path_buf(),
            started: Instant::now(),
            polish_requests: Vec::new(),
            byok: None,
            byok_jobs: HashMap::new(),
            dictionary,
            styles: app_context::StylePreferenceCache::new(None),
            repeated_edits: edits::RepeatedEdits::default(),
            recording_ids: HashMap::new(),
            pending_events: Vec::new(),
            all_ready_emitted: false,
            llm_primed: false,
            llm_primed_at: None,
            startup_asr_text: None,
            startup_audio_demo_sent: false,
            startup_audio_demo_passed: false,
            startup_dictionary_demo_sent: false,
            stream_refine,
            segment_streams: HashMap::new(),
            stream_screen_text: None,
            finishing: HashMap::new(),
            stream_envelope: None,
            stream_category: None,
            stream_bundle: None,
            stream_commit_category: (None, None),
            live_text_stream: None,
            dictation: meeting::DictationActivity::default(),
            meeting_spans: meeting::MeetingSpans::default(),
            dictation_quiet: meeting::QuietWatch::default(),
        })
    }

    pub fn stacks_ready(&self) -> bool {
        self.asr_ready && self.llm_ready
    }

    /// Have both sidecars nudge `sender` (an empty line) whenever they queue an event, so a
    /// serve loop can block on its command channel instead of sleeping between polls.
    pub fn set_waker(&self, sender: mpsc::Sender<String>) {
        self.asr.set_waker(sender.clone());
        if let Some(polisher) = self.polisher.as_ref() {
            polisher.set_waker(sender);
        }
    }

    pub fn poll(&mut self) -> Vec<EngineEvent> {
        let mut out = std::mem::take(&mut self.pending_events);
        for event in self.asr.poll() {
            match event {
                AsrEvent::Status { pct, msg } => out.push(EngineEvent::Stream {
                    name: "asr".into(),
                    state: "loading".into(),
                    pct,
                    msg,
                    load_ms: None,
                }),
                AsrEvent::Ready { load_ms, .. } => {
                    out.push(EngineEvent::Stream {
                        name: "asr".into(),
                        state: "loading".into(),
                        pct: 0.78,
                        msg: "Checking speech recognition with a test clip".into(),
                        load_ms: Some(load_ms),
                    });
                    let fixture = self.root.join("assets/startup.wav");
                    if let Err(error) = self.asr.transcribe(&fixture, Some(STARTUP_ASR_BATCH_ID)) {
                        out.push(EngineEvent::Stream {
                            name: "asr".into(),
                            state: "error".into(),
                            pct: 0.78,
                            msg: format!("startup demo failed: {error:#}"),
                            load_ms: None,
                        });
                    }
                }
                AsrEvent::Span { id, body } => {
                    if self.meeting_spans.finish(id.as_deref()) {
                        out.push(EngineEvent::MeetingSpan {
                            id: id.unwrap_or_default(),
                            body,
                        });
                    }
                }
                AsrEvent::Result {
                    id,
                    text,
                    seconds,
                    partial,
                } => {
                    if let Some(request_id) = id.as_deref().filter(|_| !partial) {
                        self.dictation.asr_done(request_id, Instant::now());
                    }
                    if id.as_deref() == Some(STARTUP_ASR_BATCH_ID) {
                        if asr_smoke_passed(&text) {
                            self.startup_asr_text = Some(text.clone());
                            out.push(EngineEvent::Stream {
                                name: "asr".into(),
                                state: "loading".into(),
                                pct: 0.9,
                                msg: "Checking live speech recognition".into(),
                                load_ms: None,
                            });
                            let fixture = self.root.join("assets/startup.wav");
                            if let Err(error) = self
                                .asr
                                .warmup_stream(&fixture, Some(STARTUP_ASR_STREAM_ID))
                            {
                                out.push(EngineEvent::Stream {
                                    name: "asr".into(),
                                    state: "error".into(),
                                    pct: 0.9,
                                    msg: format!("stream demo failed: {error:#}"),
                                    load_ms: None,
                                });
                            }
                        } else {
                            out.push(EngineEvent::Stream {
                                name: "asr".into(),
                                state: "error".into(),
                                pct: 0.9,
                                msg: format!("demo transcript mismatch: {text:?}"),
                                load_ms: None,
                            });
                        }
                    } else if id.as_deref() == Some(STARTUP_ASR_STREAM_ID) {
                        if asr_smoke_passed(&text) {
                            self.asr_ready = true;
                            out.push(EngineEvent::Stream {
                                name: "asr".into(),
                                state: "ready".into(),
                                pct: 1.0,
                                msg: "Speech recognition ready".into(),
                                load_ms: Some(self.started.elapsed().as_secs_f64() * 1000.0),
                            });
                        } else {
                            out.push(EngineEvent::Stream {
                                name: "asr".into(),
                                state: "error".into(),
                                pct: 0.95,
                                msg: format!("stream demo transcript mismatch: {text:?}"),
                                load_ms: None,
                            });
                        }
                    } else if let Some(finish) = id
                        .as_deref()
                        .filter(|_| !partial)
                        .and_then(|request_id| self.finishing.get_mut(request_id))
                    {
                        finish.text = Some((text, seconds));
                        let request_id = id.clone().unwrap_or_default();
                        out.extend(self.finished_stream_result(&request_id));
                    } else {
                        if !partial {
                            if let Some(recording_id) = id
                                .as_ref()
                                .and_then(|request_id| self.recording_ids.get(request_id))
                            {
                                if let Err(error) = update_recording_raw(recording_id, &text) {
                                    out.push(EngineEvent::Error {
                                        id: id.clone(),
                                        msg: format!("save raw transcript: {error:#}"),
                                    });
                                }
                            }
                        }
                        out.push(EngineEvent::AsrResult {
                            id,
                            text,
                            seconds,
                            partial,
                        });
                    }
                }
                AsrEvent::Segment { id, index, text } => {
                    if let Some(error) = self.commit_segment(id.as_deref(), index, text) {
                        out.push(EngineEvent::Error { id, msg: error });
                    }
                }
                AsrEvent::Error { id, msg } => {
                    if self.meeting_spans.finish(id.as_deref()) {
                        // the id only: the message goes to the bar, which shows it on Notes
                        eprintln!(
                            "phonon engine: meeting span {} failed",
                            id.as_deref().unwrap_or("?")
                        );
                        out.push(EngineEvent::MeetingSpan {
                            id: id.unwrap_or_default(),
                            body: json!({"error": msg}),
                        });
                        continue;
                    }
                    if let Some(request_id) = id.as_deref() {
                        self.dictation.asr_done(request_id, Instant::now());
                    }
                    if let Some(request_id) =
                        id.as_deref().filter(|i| self.finishing.contains_key(*i))
                    {
                        // A stream finished at key up failed in the sidecar: transcribe the
                        // WAV instead (now, or when the bar's transcribe names it).
                        let request_id = request_id.to_string();
                        if let Some(error) = self.fall_back_from_stream(&request_id) {
                            out.push(EngineEvent::Error {
                                id: Some(request_id),
                                msg: error,
                            });
                        }
                    }
                    out.push(EngineEvent::Error { id: None, msg });
                }
            }
        }
        if let Some(polisher) = self.polisher.as_mut() {
            for event in polisher.poll() {
                match event {
                    PolishEvent::Loading { msg, progress } => {
                        out.push(EngineEvent::Stream {
                            name: "llm".into(),
                            state: "loading".into(),
                            pct: llm_loading_pct(progress),
                            msg,
                            load_ms: None,
                        });
                    }
                    PolishEvent::Ready { msg, load_ms } => {
                        out.push(EngineEvent::Stream {
                            name: "llm".into(),
                            state: "loading".into(),
                            pct: 0.78,
                            msg: llm_checking_message(&msg),
                            load_ms: Some(load_ms),
                        });
                        self.polish_requests.push(PolishRequest {
                            id: Some(STARTUP_LLM_PRIME_ID.into()),
                            source_text: STARTUP_LLM_PRIME_TEXT.into(),
                            ..Default::default()
                        });
                        if let Err(error) = polisher.polish(STARTUP_LLM_PRIME_TEXT) {
                            self.polish_requests.pop();
                            out.push(EngineEvent::Error {
                                id: Some(STARTUP_LLM_PRIME_ID.into()),
                                msg: format!("LLM startup prime failed: {error:#}"),
                            });
                        }
                    }
                    PolishEvent::Result {
                        text,
                        latency_ms,
                        ttft_ms,
                        tok_s,
                    } => {
                        let request = if self.polish_requests.is_empty() {
                            PolishRequest::default()
                        } else {
                            self.polish_requests.remove(0)
                        };
                        let id = request.id;
                        if id.as_deref() == Some(STARTUP_LLM_PRIME_ID) {
                            if llm_prime_passed(&text) {
                                self.llm_primed = true;
                                self.llm_primed_at = Some(Instant::now());
                                out.push(EngineEvent::Stream {
                                    name: "llm".into(),
                                    state: "loading".into(),
                                    pct: 0.9,
                                    msg: "Waiting for speech recognition".into(),
                                    load_ms: None,
                                });
                            } else {
                                out.push(EngineEvent::Stream {
                                    name: "llm".into(),
                                    state: "error".into(),
                                    pct: 0.85,
                                    msg: format!("prime output mismatch: {text:?}"),
                                    load_ms: None,
                                });
                            }
                        } else if id.as_deref() == Some(STARTUP_LLM_AUDIO_ID) {
                            let spoken = self.startup_asr_text.clone().unwrap_or_default();
                            if llm_audio_smoke_passed(&spoken, &text) {
                                self.startup_audio_demo_passed = true;
                                out.push(EngineEvent::Stream {
                                    name: "llm".into(),
                                    state: "loading".into(),
                                    pct: 0.94,
                                    msg: "Test clip corrected".into(),
                                    load_ms: None,
                                });
                            } else {
                                out.push(EngineEvent::Stream {
                                    name: "llm".into(),
                                    state: "error".into(),
                                    pct: 0.94,
                                    msg: format!("end-to-end audio demo mismatch: {text:?}"),
                                    load_ms: None,
                                });
                            }
                        } else if id.as_deref() == Some(STARTUP_LLM_ID) {
                            if llm_smoke_passed(&text) {
                                self.llm_ready = true;
                                let startup_ms = self.started.elapsed().as_secs_f64() * 1000.0;
                                out.push(EngineEvent::Stream {
                                    name: "llm".into(),
                                    state: "ready".into(),
                                    pct: 1.0,
                                    msg: "Correction model ready".into(),
                                    load_ms: Some(startup_ms),
                                });
                            } else {
                                out.push(EngineEvent::Stream {
                                    name: "llm".into(),
                                    state: "error".into(),
                                    pct: 0.9,
                                    msg: format!("demo output mismatch: {text:?}"),
                                    load_ms: None,
                                });
                            }
                        } else {
                            // The user's own endpoint, when set and in time; the guard in
                            // finish_correction still applies to its text.
                            let text = match take_byok_result(&mut self.byok_jobs, id.as_deref()) {
                                Some(remote)
                                    if byok_text_passes(
                                        &self.dictionary,
                                        &request.source_text,
                                        &remote,
                                    ) =>
                                {
                                    remote
                                }
                                _ => text,
                            };
                            let mut corrected = finish_correction(
                                &self.dictionary,
                                &request.source_text,
                                &text,
                                &request.style,
                                // no mode, no opener restore
                                request.mode.as_deref().unwrap_or(""),
                            );
                            let app_category = request.app.as_ref().map(|app| app.category);
                            corrected.text = final_layout(
                                &corrected.text,
                                request.mode.as_deref().unwrap_or("minimal"),
                                app_category,
                            );
                            if let Some(recording_id) = id
                                .as_ref()
                                .and_then(|request_id| self.recording_ids.get(request_id))
                            {
                                if let Err(error) = update_recording_final(
                                    recording_id,
                                    &corrected.text,
                                    corrected.applied.clone(),
                                    request.screen_context_terms,
                                    data::LlmMetadata {
                                        latency_ms,
                                        ttft_ms,
                                        tokens_per_second: tok_s,
                                    },
                                    request.keep_context.then_some(request.app).flatten(),
                                    request.keep_context.then_some(request.textbox).flatten(),
                                ) {
                                    out.push(EngineEvent::Error {
                                        id: id.clone(),
                                        msg: format!("save recording metadata: {error:#}"),
                                    });
                                }
                            }
                            self.dictation.touch(Instant::now());
                            out.push(EngineEvent::PolishResult {
                                id,
                                text: corrected.text,
                                latency_ms,
                                ttft_ms,
                                tok_s,
                                app_category,
                            });
                        }
                    }
                    PolishEvent::Draft {
                        id,
                        seq,
                        text,
                        complete,
                    } => {
                        // Live text, for the open stream only: a draft from a finished or
                        // replaced dictation, or of an unknown commit, is dropped.
                        let Some(source) = draft_source(
                            self.live_text_stream.as_deref(),
                            &self.segment_streams,
                            &id,
                            seq,
                        ) else {
                            continue;
                        };
                        let style = match &self.stream_category {
                            Some((stream_id, category)) if *stream_id == id => {
                                self.styles.styles_for(*category)
                            }
                            _ => Vec::new(),
                        };
                        // the draft's writing mode, resolved as the final polish resolves it
                        let bundle = match &self.stream_bundle {
                            Some((stream_id, bundle)) if *stream_id == id => Some(bundle.clone()),
                            _ => None,
                        };
                        let category = match &self.stream_category {
                            Some((stream_id, category)) if *stream_id == id => Some(*category),
                            _ => None,
                        };
                        let mode = self
                            .styles
                            .writing_mode_for(bundle.as_deref())
                            .unwrap_or_else(|| category_mode(category).to_string());
                        let corrected =
                            finish_correction(&self.dictionary, &source, &text, &style, &mode);
                        out.push(EngineEvent::DraftResult {
                            id,
                            text: corrected.text,
                            source,
                            complete,
                        });
                    }
                    PolishEvent::Error { msg } => {
                        // drop matching id if any
                        let id = if self.polish_requests.is_empty() {
                            None
                        } else {
                            self.polish_requests.remove(0).id
                        };
                        out.push(EngineEvent::Error { id, msg });
                    }
                }
            }
            let prime_settled = self
                .llm_primed_at
                .is_some_and(|primed_at| primed_at.elapsed() >= Duration::from_millis(250));
            if self.llm_primed && prime_settled && self.asr_ready && !self.startup_audio_demo_sent {
                self.startup_audio_demo_sent = true;
                let startup_text = self
                    .startup_asr_text
                    .clone()
                    .unwrap_or_else(|| "Hello Fluid Voice.".into());
                out.push(EngineEvent::Stream {
                    name: "llm".into(),
                    state: "loading".into(),
                    pct: 0.92,
                    msg: "Correcting the test clip".into(),
                    load_ms: None,
                });
                self.polish_requests.push(PolishRequest {
                    id: Some(STARTUP_LLM_AUDIO_ID.into()),
                    source_text: startup_text.clone(),
                    ..Default::default()
                });
                let startup_input = self.dictionary.prepare_polish_input(&startup_text);
                if let Err(error) = polisher.polish(&startup_input) {
                    self.polish_requests.pop();
                    out.push(EngineEvent::Error {
                        id: Some(STARTUP_LLM_AUDIO_ID.into()),
                        msg: format!("end-to-end audio startup demo failed: {error:#}"),
                    });
                }
            }
            if self.startup_audio_demo_passed && !self.startup_dictionary_demo_sent {
                self.startup_dictionary_demo_sent = true;
                out.push(EngineEvent::Stream {
                    name: "llm".into(),
                    state: "loading".into(),
                    pct: 0.94,
                    msg: "Checking corrections with a test sentence".into(),
                    load_ms: None,
                });
                self.polish_requests.push(PolishRequest {
                    id: Some(STARTUP_LLM_ID.into()),
                    source_text: STARTUP_LLM_TEXT.into(),
                    ..Default::default()
                });
                let startup_input =
                    startup_demo_dictionary().prepare_polish_input(STARTUP_LLM_TEXT);
                if let Err(error) = polisher.polish(&startup_input) {
                    self.polish_requests.pop();
                    out.push(EngineEvent::Error {
                        id: Some(STARTUP_LLM_ID.into()),
                        msg: format!("LLM dictionary demo failed: {error:#}"),
                    });
                }
            }
        }

        if self.stacks_ready() && !self.all_ready_emitted {
            self.all_ready_emitted = true;
            out.push(EngineEvent::AllReady {
                stream_refine: self.stream_refine,
            });
        }
        out.extend(self.pump_meeting_spans());
        // After the pump, which mirrors the engine's dictation state into the holds.
        if let Some(quiet) = self
            .dictation_quiet
            .update(self.dictation.quiet(Instant::now()))
        {
            out.push(EngineEvent::DictationQuiet { quiet });
        }
        out
    }

    /// `{"cmd":"meeting_span"}`: queue one span of a meeting track for the speech model.
    pub fn queue_meeting_span(&mut self, request: meeting::SpanRequest) -> Vec<EngineEvent> {
        self.meeting_spans.push(request);
        self.pump_meeting_spans()
    }

    /// Hand the sidecar the next meeting span when the model is free: both stacks ready, no
    /// meeting span in it, and no dictation under way (no stream, no transcript or correction
    /// owed, no key held, none in the last `meeting::QUIET_AFTER_DICTATION`).
    fn pump_meeting_spans(&mut self) -> Vec<EngineEvent> {
        // Engine state that means a dictation is under way, as holds that go stale (meeting.rs):
        // a stream the bar abandoned or a result that never came cannot pin the queue for good.
        let now = Instant::now();
        self.dictation
            .mirror("stream", !self.segment_streams.is_empty(), now);
        self.dictation
            .mirror("finishing", !self.finishing.is_empty(), now);
        self.dictation
            .mirror("polish", !self.polish_requests.is_empty(), now);
        let idle = self.stacks_ready() && self.dictation.quiet(now);
        let Some(request) = self.meeting_spans.next(idle) else {
            return Vec::new();
        };
        match self.asr.transcribe_span(
            &request.id,
            &request.chunks,
            request.start_sample,
            request.end_sample,
        ) {
            Ok(()) => Vec::new(),
            Err(error) => {
                self.meeting_spans.finish(Some(&request.id));
                vec![EngineEvent::MeetingSpan {
                    id: request.id,
                    body: json!({"error": format!("send span: {error:#}")}),
                }]
            }
        }
    }

    pub fn transcribe(&mut self, path: &Path, id: Option<&str>) -> Result<()> {
        self.transcribe_with_source(path, id, "terminal")
    }

    pub fn transcribe_with_source(
        &mut self,
        path: &Path,
        id: Option<&str>,
        source: &str,
    ) -> Result<()> {
        if let Some(id) = id.filter(|id| !id.starts_with("__phonon_")) {
            self.dictation.key_up(Instant::now());
            let metadata = data::register_recording(path, source)?;
            self.recording_ids
                .insert(id.to_string(), metadata.id.clone());
            let speech_detected = wav_has_speech(path)?;
            data::set_speech_detected(&metadata.id, speech_detected)?;
            if let Some(finish) = self.finishing.get_mut(id) {
                if finish.failed {
                    // the sidecar could not finish the stream: the usual path below
                    self.finishing.remove(id);
                } else {
                    // The release ASR already runs (stream_finish at key up); the gate
                    // decides what of it goes out.
                    finish.speech = Some(speech_detected);
                    finish.path = Some(path.to_path_buf());
                    let events = self.finished_stream_result(id);
                    self.pending_events.extend(events);
                    return Ok(());
                }
            }
            if !speech_detected {
                // Transcribe anyway; as for a stream, the recognizer's words decide
                // (`StreamFinish::resolve`), and a failure gives an empty result.
                self.finishing.insert(
                    id.to_string(),
                    StreamFinish {
                        speech: Some(false),
                        path: Some(path.to_path_buf()),
                        ..StreamFinish::default()
                    },
                );
            }
        }
        // Asked of the sidecar from here on (a stream finished at key up already was, above):
        // meeting spans wait for its answer.
        if let Some(request_id) = id.filter(|id| !id.starts_with("__phonon_")) {
            self.dictation.asr_sent(request_id, Instant::now());
        }
        if let Some(stream_id) = id.filter(|id| self.segment_streams.contains_key(*id)) {
            // The closed segments are transcribed already; only the open one is left.
            self.segment_streams.remove(stream_id);
            return self.asr.stream_finish(Some(path), Some(stream_id));
        }
        self.asr.transcribe(path, id)
    }

    /// Key up of a streamed dictation, before the WAV exists: the bar has sent the last
    /// audio, so the sidecar can transcribe the release window now. `transcribe` with the
    /// WAV follows (corpus entry, speech gate) and releases the result. For any other
    /// stream this is `stream_stop`.
    pub fn stream_finish(&mut self, id: Option<&str>) -> Result<()> {
        self.dictation.key_up(Instant::now());
        let Some(stream_id) = id.filter(|id| self.segment_streams.contains_key(*id)) else {
            return self.asr.stream_stop(id);
        };
        self.segment_streams.remove(stream_id);
        self.finishing
            .insert(stream_id.to_string(), StreamFinish::default());
        self.asr.stream_finish(None, Some(stream_id))
    }

    /// The sidecar failed a stream finished at key up: transcribe its WAV the usual way, now if
    /// the bar has named it (and the gate found speech), else when it does.
    fn fall_back_from_stream(&mut self, id: &str) -> Option<String> {
        let finish = self.finishing.get_mut(id)?;
        if finish.speech == Some(false) {
            // no speech by the gate and no recognizer words to overrule it
            self.finishing.remove(id);
            self.pending_events.push(EngineEvent::AsrResult {
                id: Some(id.to_string()),
                text: String::new(),
                seconds: 0.0,
                partial: false,
            });
            return None;
        }
        match (finish.speech, finish.path.clone()) {
            (Some(true), Some(path)) => {
                self.finishing.remove(id);
                self.asr
                    .transcribe(&path, Some(id))
                    .err()
                    .map(|error| format!("transcribe after a failed stream: {error:#}"))
            }
            _ => {
                finish.failed = true;
                None
            }
        }
    }

    /// The final transcript of a stream finished at key up, once both its ASR text and
    /// its WAV's speech gate are in (an empty one as soon as the gate finds no speech).
    fn finished_stream_result(&mut self, id: &str) -> Vec<EngineEvent> {
        let Some(finish) = self.finishing.get_mut(id) else {
            return Vec::new();
        };
        let (answer, done) = finish.resolve();
        if done {
            self.finishing.remove(id);
        }
        let Some((text, seconds)) = answer else {
            return Vec::new();
        };
        let mut out = Vec::new();
        if let Some(recording_id) = self.recording_ids.get(id).filter(|_| !text.is_empty()) {
            if let Err(error) = update_recording_raw(recording_id, &text) {
                out.push(EngineEvent::Error {
                    id: Some(id.to_string()),
                    msg: format!("save raw transcript: {error:#}"),
                });
            }
        }
        out.push(EngineEvent::AsrResult {
            id: Some(id.to_string()),
            text,
            seconds,
            partial: false,
        });
        out
    }

    /// `app`: the bar's `app` object for the dictation's target as known at key down.
    pub fn stream_start(
        &mut self,
        id: Option<&str>,
        app: Option<&serde_json::Value>,
    ) -> Result<()> {
        self.dictation.key_down(Instant::now());
        if let (true, Some(stream_id)) = (self.stream_refine && self.llm_ready, id) {
            self.segment_streams.clear();
            self.segment_streams
                .insert(stream_id.to_string(), Vec::new());
            self.stream_screen_text = None;
            // a dictation the bar finished but never sent a WAV for (empty capture)
            self.finishing.clear();
            self.stream_envelope = None;
            self.stream_category = None;
            self.stream_bundle = None;
            self.stream_commit_category = (None, None);
            self.live_text_stream = self.styles.live_text().then(|| stream_id.to_string());
            if let Some(app) = app {
                self.stream_app(stream_id, app);
            }
            if let Some(polisher) = self.polisher.as_mut() {
                polisher.stream_reset()?;
            }
            return self.asr.stream_start_segments(id);
        }
        self.live_text_stream = None;
        self.asr.stream_start(id)
    }

    /// Screen text for the stream `id` (captured at key down); later commits use it.
    pub fn stream_screen_context(&mut self, id: &str, text: &str) {
        if self.segment_streams.contains_key(id) {
            self.stream_screen_text = Some((id.to_string(), text.to_string()));
        }
    }

    /// The target app of the stream `id` once the bar resolved it (a browser tab's host
    /// arrives after key down); later commits carry its category.
    pub fn stream_app(&mut self, id: &str, app: &serde_json::Value) {
        if self.segment_streams.contains_key(id) {
            let app = app_context::AppContext::from_request(app);
            self.stream_category = app.as_ref().map(|app| (id.to_string(), app.category));
            self.stream_bundle = app.map(|app| (id.to_string(), app.bundle_id));
        }
    }

    /// The `context` of a background commit for the stream `id`: the app category only,
    /// the one key the refiner's prompt depends on.
    fn stream_commit_context(&self, id: Option<&str>) -> Option<app_context::AppCategory> {
        match (&self.stream_category, id) {
            (Some((stream_id, category)), Some(id)) if stream_id == id => Some(*category),
            _ => None,
        }
    }

    /// The transcript so far of an open segment stream (a segment closed at a pause, or a
    /// speculative round reaching the present): send it to the correction sidecar to refine
    /// in the background. Returns an error message to report.
    fn commit_segment(&mut self, id: Option<&str>, index: usize, text: String) -> Option<String> {
        let seen = self.segment_streams.get_mut(id?)?;
        if seen.len() != index {
            return Some(format!("stream text {index} arrived after {}", seen.len()));
        }
        let text = text.trim().to_string();
        seen.push(text.clone());
        if text.is_empty() {
            return None;
        }
        let screen = match (&self.stream_screen_text, id) {
            (Some((screen_id, screen)), Some(id)) if screen_id == id => screen.clone(),
            _ => String::new(),
        };
        let category = self.stream_commit_context(id);
        let bundle = match (&self.stream_bundle, id) {
            (Some((stream_id, bundle)), Some(id)) if stream_id == id => Some(bundle.clone()),
            _ => None,
        };
        let mode = self.styles.writing_mode_for(bundle.as_deref());
        if matches!(&self.stream_envelope, Some((t, s, _)) if *t == text && *s == screen)
            && (category, mode.clone()) == self.stream_commit_category
        {
            return None; // a round that heard nothing new: the sidecar has this one already
        }
        let input = self
            .dictionary
            .prepare_polish_input_with_context(&text, &screen);
        let live = id.filter(|id| self.live_text_stream.as_deref() == Some(*id));
        let context = app_context::with_writing_mode(
            stream_commit_json(category, live.map(|id| (id, index))),
            mode.as_deref(),
        );
        let polisher = self.polisher.as_mut()?;
        let error = polisher
            .stream_commit(&input, context.as_ref())
            .err()
            .map(|error| format!("stream commit: {error:#}"));
        self.stream_envelope = Some((text, screen, input));
        self.stream_commit_category = (category, mode);
        error
    }

    /// A reloaded dictionary; envelopes built from the old one are dropped.
    pub fn set_dictionary(&mut self, dictionary: data::DictionaryFile) {
        if let Err(error) = self.asr.set_terms(&dictionary.asr_terms()) {
            eprintln!("phonon engine: ASR dictionary terms not sent: {error:#}");
        }
        self.dictionary = dictionary;
        self.stream_envelope = None;
    }

    /// The dictionary envelope for a polish: the open stream's last one when the transcript
    /// and screen text are the same, else built now.
    fn polish_envelope(&self, text: &str, screen_text: &str) -> String {
        match &self.stream_envelope {
            Some((last_text, last_screen, input))
                if last_text == text && last_screen == screen_text =>
            {
                input.clone()
            }
            _ => self
                .dictionary
                .prepare_polish_input_with_context(text, screen_text),
        }
    }

    pub fn stream_chunk(&mut self, pcm16: &str, id: Option<&str>) -> Result<()> {
        self.asr.stream_chunk(pcm16, id)
    }

    pub fn stream_stop(&mut self, id: Option<&str>) -> Result<()> {
        self.dictation.key_up(Instant::now());
        self.asr.stream_stop(id)
    }

    pub fn polish(&mut self, text: &str, id: Option<&str>) -> Result<()> {
        self.polish_with_context(text, "", id)
    }

    pub fn polish_with_context(
        &mut self,
        text: &str,
        screen_text: &str,
        id: Option<&str>,
    ) -> Result<()> {
        self.polish_request(text, screen_text, id, None, None, false)
    }

    /// A polish request that knows where it is going. `app` is the bar's
    /// `app` object (bundle id, name, page host); `textbox` is the text around
    /// the cursor when the user turned text-box context on. Both reach the
    /// sidecar as `context`; the approved style preferences for the app's
    /// category are applied to the result. `keep_context` (local history on)
    /// lets the corpus entry keep them.
    pub fn polish_request(
        &mut self,
        text: &str,
        screen_text: &str,
        id: Option<&str>,
        app: Option<&serde_json::Value>,
        textbox: Option<&serde_json::Value>,
        keep_context: bool,
    ) -> Result<()> {
        if self.polisher.is_none() {
            bail!("the required local correction model is unavailable");
        }
        if !self.llm_ready {
            bail!("the correction model is not ready yet");
        }
        let input = self.polish_envelope(text, screen_text);
        let app = app.and_then(app_context::AppContext::from_request);
        // Re-parsed only when settings.json changes, so an approval applies
        // to the very next dictation without a read per preview.
        let style = match app.as_ref() {
            Some(app) => self.styles.styles_for(app.category),
            None => Vec::new(),
        };
        let textbox = textbox.and_then(app_context::TextboxContext::from_request);
        let mode = self
            .styles
            .writing_mode_for(app.as_ref().map(|app| app.bundle_id.as_str()));
        let context = app_context::with_writing_mode(
            app_context::polish_context(app.as_ref(), &style, textbox.as_ref()),
            mode.as_deref(),
        );
        let layout_mode = mode
            .clone()
            .unwrap_or_else(|| category_mode(app.as_ref().map(|app| app.category)).to_string());
        self.polish_requests.push(PolishRequest {
            id: id.map(str::to_string),
            source_text: text.to_string(),
            screen_context_terms: self.dictionary.screen_confirmed_terms(text, screen_text),
            app,
            style,
            textbox,
            keep_context,
            mode: Some(layout_mode.clone()),
        });
        let Some(polisher) = self.polisher.as_mut() else {
            bail!("the required local correction model is unavailable");
        };
        if let Err(error) = polisher.polish_with_context(&input, context.as_ref()) {
            self.polish_requests.pop();
            return Err(error);
        }
        // The user's own endpoint refines the same transcript beside the local refiner;
        // the local result stands in whenever it fails or is late.
        if let (Some(endpoint), Some(id)) = (self.byok.clone(), id) {
            let mut vocabulary = self
                .polish_requests
                .last()
                .map(|r| r.screen_context_terms.clone())
                .unwrap_or_default();
            for term in self.dictionary.asr_terms() {
                if vocabulary.len() >= BYOK_VOCABULARY_CAP {
                    break;
                }
                if !vocabulary.contains(&term) {
                    vocabulary.push(term);
                }
            }
            let system = phonon_llm::byok::system_prompt(Some(&layout_mode), &vocabulary);
            let transcript = text.to_string();
            let (tx, rx) = mpsc::channel();
            let deadline = Instant::now() + endpoint.timeout;
            thread::spawn(move || {
                let _ = tx.send(phonon_llm::byok::complete(&endpoint, &system, &transcript));
            });
            self.byok_jobs.insert(
                id.to_string(),
                ByokJob {
                    deadline,
                    reply: rx,
                },
            );
        }
        Ok(())
    }

    /// `{"cmd":"byok"}`: set or clear the user's own endpoint (issue #24). The bar
    /// reads the key from the Keychain and sends it here; it is kept in memory only.
    pub fn set_byok(&mut self, endpoint: Option<phonon_llm::byok::Endpoint>) {
        self.byok = endpoint;
        self.byok_jobs.clear();
    }

    pub fn polish_available(&self) -> bool {
        self.polisher.is_some()
    }

    /// Dictation started. Best effort: a sidecar that is not ready simply
    /// misses the hint.
    /// `{"cmd":"dictation_end"}`: the bar ended a dictation without a WAV or a stream to finish
    /// (a discarded or too-short one, a microphone that failed to start). Releases the key hold
    /// that `prewarm` / `stream_start` took; nothing reaches the sidecars.
    pub fn dictation_end(&mut self) {
        self.dictation.key_up(Instant::now());
    }

    pub fn prewarm_polish(&mut self) -> Result<()> {
        self.dictation.key_down(Instant::now());
        match self.polisher.as_mut() {
            Some(polisher) if self.llm_ready => polisher.prewarm(),
            _ => Ok(()),
        }
    }

    fn recording_metadata_path(&self, id: Option<&str>) -> Option<std::path::PathBuf> {
        let recording_id = id.and_then(|id| self.recording_ids.get(id))?;
        data::corpus_dir()
            .ok()
            .map(|corpus| corpus.join(recording_id).join("metadata.json"))
    }

    /// Classify what the user did to a pasted dictation and, when the bar says
    /// local history is on, keep the record in that dictation's corpus entry.
    pub fn edit_observed(
        &mut self,
        id: Option<&str>,
        request: &edits::ObservedEdit,
    ) -> Result<serde_json::Value> {
        let mut analysis = request.analyze(&self.dictionary);
        if edits::end_reason_is_learnable(&request.end_reason) {
            self.repeated_edits.promote(&mut analysis);
        }
        let recorded = match (request.record, self.recording_metadata_path(id)) {
            (true, Some(path)) => edits::write_record(&path, &request.record(&analysis))?,
            _ => false,
        };
        Ok(edits::observed_event(id, &analysis, recorded))
    }

    pub fn edit_decision(&self, id: Option<&str>, index: usize, decision: &str) -> Result<bool> {
        match self.recording_metadata_path(id) {
            Some(path) => edits::write_decision(&path, index, decision),
            None => Ok(false),
        }
    }

    pub fn shutdown(mut self) {
        self.asr.shutdown();
        if let Some(polisher) = self.polisher.as_mut() {
            polisher.shutdown();
        }
    }
}

fn update_recording_raw(recording_id: &str, raw_text: &str) -> Result<()> {
    let mut recording = data::load_recording_by_id(recording_id)?;
    recording.raw_transcript = raw_text.trim().to_string();
    data::save_recording(&recording)
}

fn update_recording_final(
    recording_id: &str,
    final_text: &str,
    corrections: Vec<data::AppliedCorrection>,
    screen_context_terms: Vec<String>,
    llm: data::LlmMetadata,
    app: Option<app_context::AppContext>,
    textbox: Option<app_context::TextboxContext>,
) -> Result<()> {
    let mut recording = data::load_recording_by_id(recording_id)?;
    recording.final_transcript = final_text.trim().to_string();
    recording.dictionary_corrections = corrections;
    recording.screen_context_terms = screen_context_terms;
    recording.llm = Some(llm);
    if app.is_some() {
        recording.app_context = app;
    }
    if textbox.is_some() {
        recording.textbox_context = textbox;
    }
    data::save_recording(&recording)
}

fn safe_polish_output(
    dictionary: &data::DictionaryFile,
    source: &str,
    polished: &str,
) -> data::CorrectionResult {
    data::safe_polish_output(dictionary, source, polished)
}

/// The one path from a refiner output to the text the user sees: the final polish result
/// and the live-text drafts both go through it, so the overlay shows what would be pasted.
/// `mode`: the writing mode the refiner ran under ("" when unknown: no opener restore,
/// `phonon_llm::opener::MODES`).
fn finish_correction(
    dictionary: &data::DictionaryFile,
    source: &str,
    polished: &str,
    style: &[String],
    mode: &str,
) -> data::CorrectionResult {
    // The opener the refiner dropped ("Okay, ...") goes back first, before the guard: the
    // guard's deletion check would otherwise read the dropped opener as a lost word and
    // swap the whole correction for the raw transcript.
    let polished =
        phonon_llm::opener::restore(source, data::extract_transcript_payload(polished), mode);
    let mut corrected = safe_polish_output(dictionary, source, &polished);
    corrected.text = app_context::apply_style(&corrected.text, style, dictionary);
    corrected
}

/// The transcript so far a live-text draft refines, when the draft belongs to the open
/// live-text stream and names a commit of it.
fn draft_source(
    live_stream: Option<&str>,
    streams: &HashMap<String, Vec<String>>,
    id: &str,
    seq: Option<usize>,
) -> Option<String> {
    if live_stream != Some(id) {
        return None;
    }
    streams.get(id)?.get(seq?).cloned()
}

/// The `context` of a background commit: the app category (the one key the refiner's
/// prompt depends on) and, for a live-text stream, the tag its draft comes back under.
fn stream_commit_json(
    category: Option<app_context::AppCategory>,
    live: Option<(&str, usize)>,
) -> Option<serde_json::Value> {
    let mut context = category.map(|category| json!({"app_category": category}));
    if let Some((id, seq)) = live {
        let context = context.get_or_insert_with(|| json!({}));
        context["draft_id"] = json!(id);
        context["draft_seq"] = json!(seq);
    }
    context
}

fn normalized_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|character| character.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect()
        })
        .collect()
}

/// Streaming refine with nothing set: on with 16 GB of memory or more (a "16 GB" Mac reports
/// exactly 16 GiB), off below or when the size is unknown.
fn stream_refine_default(memory_bytes: Option<u64>) -> bool {
    memory_bytes.is_some_and(|bytes| bytes >= 16 << 30)
}

/// Physical memory (macOS `hw.memsize`); None where sysctl is missing or says nothing.
fn physical_memory_bytes() -> Option<u64> {
    let out = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

/// Streaming refine: PHONON_STREAM_REFINE ("1"/"true"/"on" or "0"/"false"/"off") when set,
/// else the settings.json value (or the memory default when that is absent).
fn stream_refine_setting(env: Option<String>, setting: bool) -> bool {
    match env
        .as_deref()
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("1" | "true" | "on") => true,
        Some("0" | "false" | "off") => false,
        _ => setting,
    }
}

fn asr_smoke_passed(text: &str) -> bool {
    let words = normalized_words(text);
    words.iter().any(|word| word == "hello")
        && words
            .iter()
            .any(|word| word == "voice" || word == "fluidvoice" || word == "boys")
        && words
            .iter()
            .any(|word| word == "fluid" || word == "flamid" || word == "fluidvoice")
}

/// The two terms the dictionary demo needs, independent of what the user has
/// taught. A fresh install has an empty dictionary, so asserting against the
/// real one made the startup gate unpassable on a clean Mac; the demo has to
/// prove the retrieval path works, not that this user says "cuBLAS".
fn startup_demo_dictionary() -> data::DictionaryFile {
    let entry = |phrase: &str| data::DictionaryEntry {
        phrase: phrase.to_string(),
        replacement: None,
        spoken_forms: Vec::new(),
        source: "startup-demo".into(),
        starred: false,
        usage_count: 0,
    };
    data::DictionaryFile {
        entries: vec![entry("cuBLAS"), entry("cuDNN")],
        ..Default::default()
    }
}

fn llm_smoke_passed(text: &str) -> bool {
    let words = normalized_words(data::extract_transcript_payload(text));
    ["difference", "cublas", "cudnn"]
        .iter()
        .all(|expected| words.iter().any(|word| word == expected))
}

/// The audio demo proves the recorded WAV reached the correction stage and came
/// back as a usable sentence. It deliberately does not assert which words the
/// dictionary chose: a term like `FluidAudio` is a legitimate rewrite of the
/// fixture, and the gate must not depend on the user's installed vocabulary.
fn llm_audio_smoke_passed(asr_text: &str, corrected: &str) -> bool {
    let spoken = normalized_words(asr_text);
    let corrected = normalized_words(data::extract_transcript_payload(corrected));
    if spoken.is_empty() || corrected.is_empty() {
        return false;
    }
    // Catches both the one-word collapse and a runaway expansion.
    let floor = spoken.len().saturating_sub(1).max(1);
    corrected.len() >= floor && corrected.len() <= spoken.len() + 2
}

/// Where the correction stream sits while its sidecar loads: 5% at start, up
/// to 65% as the first-launch download lands, below the 78% that `Ready`
/// (weights loaded) reports. A stage without a byte count holds at 5%, and the
/// bar keeps the highest value it has shown, so a later stage never rewinds it.
fn llm_loading_pct(progress: Option<f64>) -> f64 {
    0.05 + 0.6 * progress.unwrap_or(0.0).clamp(0.0, 1.0)
}

/// The startup line once the correction weights are loaded. The sidecar's own
/// ready message is developer detail; the one fact a user needs from it is
/// that this Mac fell back to the slower MLX path.
fn llm_checking_message(ready_message: &str) -> String {
    if ready_message.contains("native engine unavailable: no engine binary") {
        // A source build without scripts/build-engine.sh.
        "Checking the correction model (MLX; this build has no native engine)".into()
    } else if ready_message.contains("native engine unavailable") {
        "Checking the correction model (MLX fallback; the native engine is unavailable on this Mac)"
            .into()
    } else {
        "Checking the correction model".into()
    }
}

/// The prime is the warmup request, not a vocabulary test. A refiner may
/// rewrite the product name in it as a technical term (the public refiner on
/// MLX returned "hello FluentBit startup prime"), so it passes on the audio
/// demo's rule (no collapse, no runaway) when most of its words survive.
/// Keyword retention is the dictionary demo's job.
fn llm_prime_passed(text: &str) -> bool {
    let spoken = normalized_words(STARTUP_LLM_PRIME_TEXT);
    let corrected = normalized_words(data::extract_transcript_payload(text));
    let kept = spoken
        .iter()
        .filter(|word| corrected.contains(word))
        .count();
    llm_audio_smoke_passed(STARTUP_LLM_PRIME_TEXT, text) && kept * 2 >= spoken.len()
}

/// Long-lived CLI: `phonon engine` — JSONL on stdin/stdout for the floating bar.
pub fn run_engine_serve() -> Result<()> {
    // The bar appends this process's stderr to ~/Library/Logs/Phonon/engine.log; the
    // sidecars' stderr goes there too, labeled, so a failure leaves evidence.
    phonon_asr::pass_stderr_through(true);
    eprintln!(
        "phonon engine {} (pid {}) starting",
        env!("CARGO_PKG_VERSION"),
        std::process::id()
    );
    let root = project_root();
    let mut eng = Engine::start(&root)?;

    emit(&json!({
        "type": "stream",
        "name": "asr",
        "state": "loading",
        "pct": 0.05,
        "msg": "Loading the speech model"
    }));
    emit(&json!({
        "type": "stream",
        "name": "llm",
        "state": "loading",
        "pct": 0.05,
        "msg": "Loading the correction model"
    }));

    // Commands and sidecar wake-ups share one channel, so the loop blocks until either arrives
    // instead of sleeping 15 ms between polls (up to 15 ms on every hop of a dictation: ASR
    // result out, polish command in, polish result out). A wake-up is an empty line.
    let (cmd_tx, cmd_rx) = mpsc::channel::<String>();
    eng.set_waker(cmd_tx.clone());
    // The WebSocket API (settings.json `websocket_api`, off by default): another device
    // dictates through these engines. Its requests arrive on the same wake-ups.
    let mut remote = remote::Remote::new(cmd_tx.clone());
    thread::spawn(move || {
        let stdin = std::io::stdin();
        let mut lock = stdin.lock();
        let mut input = Vec::new();
        loop {
            input.clear();
            // Bytes, not read_line: a line that is not valid UTF-8 is an error there, which
            // used to read as the end of input and stop the engine.
            match lock.read_until(b'\n', &mut input) {
                Ok(0) | Err(_) => {
                    // The wakers keep the channel open, so say so explicitly.
                    let _ = cmd_tx.send(STDIN_CLOSED.to_string());
                    break;
                }
                Ok(_) => {
                    let line = String::from_utf8_lossy(&input).into_owned();
                    if cmd_tx.send(line).is_err() {
                        break;
                    }
                }
            }
        }
    });

    loop {
        remote.tick(&mut eng);
        if let Some(status) = remote.take_status() {
            emit(&status);
        }
        for event in remote.take_dictation_events() {
            emit(&event);
        }
        for ev in eng.poll() {
            // A WebSocket client's results go to that client, never to the bar.
            if remote.take_event(&mut eng, &ev) {
                continue;
            }
            remote.note_bar_event(&ev);
            emit_engine_event(&ev);
        }
        match cmd_rx.recv_timeout(Duration::from_millis(250)) {
            Ok(cmd_line) if cmd_line == STDIN_CLOSED => {
                eprintln!("phonon engine: stdin closed, shutting down");
                eng.shutdown();
                return Ok(());
            }
            Ok(cmd_line) => {
                let cmd_line = cmd_line.trim().to_string();
                if cmd_line.is_empty() {
                    continue;
                }
                let v: serde_json::Value = match serde_json::from_str(&cmd_line) {
                    Ok(v) => v,
                    Err(e) => {
                        emit(&json!({"type":"error","msg":format!("bad json: {e}")}));
                        continue;
                    }
                };
                let cmd = v.get("cmd").and_then(|x| x.as_str()).unwrap_or("");
                remote.note_bar_command(&v);
                match cmd {
                    "status" => {
                        emit(&json!({
                            "type": "status",
                            "asr_ready": eng.asr_ready,
                            "llm_ready": eng.llm_ready,
                            "stacks_ready": eng.stacks_ready(),
                        }));
                    }
                    "transcribe" => {
                        let path = v.get("path").and_then(|x| x.as_str()).unwrap_or("");
                        let id = v.get("id").and_then(|x| x.as_str());
                        let source = v.get("source").and_then(|x| x.as_str()).unwrap_or("bar");
                        if let Err(e) = eng.transcribe_with_source(Path::new(path), id, source) {
                            emit(&json!({"type":"error","id":id,"msg":format!("{e:#}")}));
                        }
                    }
                    "stream_start" => {
                        let id = v.get("id").and_then(|x| x.as_str());
                        if let Err(e) = eng.stream_start(id, v.get("app")) {
                            emit(&json!({"type":"error","id":id,"msg":format!("{e:#}")}));
                        }
                    }
                    "stream_app" => {
                        let id = v.get("id").and_then(|x| x.as_str()).unwrap_or("");
                        if let Some(app) = v.get("app") {
                            eng.stream_app(id, app);
                        }
                    }
                    "stream_chunk" => {
                        let pcm16 = v.get("pcm16").and_then(|x| x.as_str()).unwrap_or("");
                        let id = v.get("id").and_then(|x| x.as_str());
                        if let Err(e) = eng.stream_chunk(pcm16, id) {
                            emit(&json!({"type":"error","id":id,"msg":format!("{e:#}")}));
                        }
                    }
                    "stream_stop" => {
                        let id = v.get("id").and_then(|x| x.as_str());
                        if let Err(e) = eng.stream_stop(id) {
                            emit(&json!({"type":"error","id":id,"msg":format!("{e:#}")}));
                        }
                    }
                    "stream_finish" => {
                        let id = v.get("id").and_then(|x| x.as_str());
                        if let Err(e) = eng.stream_finish(id) {
                            emit(&json!({"type":"error","id":id,"msg":format!("{e:#}")}));
                        }
                    }
                    "screen_context" => {
                        let id = v.get("id").and_then(|x| x.as_str()).unwrap_or("");
                        let text = v.get("text").and_then(|x| x.as_str()).unwrap_or("");
                        eng.stream_screen_context(id, text);
                    }
                    "polish" => {
                        let text = v.get("text").and_then(|x| x.as_str()).unwrap_or("");
                        let context = v.get("context").and_then(|x| x.as_str()).unwrap_or("");
                        let id = v.get("id").and_then(|x| x.as_str());
                        if let Err(e) = eng.polish_request(
                            text,
                            context,
                            id,
                            v.get("app"),
                            v.get("textbox"),
                            v.get("local_history")
                                .and_then(|x| x.as_bool())
                                .unwrap_or(false),
                        ) {
                            emit(&json!({"type":"error","id":id,"msg":format!("{e:#}")}));
                        }
                    }
                    "dictation_end" => eng.dictation_end(),
                    "meeting_span" => match meeting::SpanRequest::from_command(&v) {
                        Ok(request) => {
                            for ev in eng.queue_meeting_span(request) {
                                emit_engine_event(&ev);
                            }
                        }
                        Err(msg) => {
                            let id = v.get("id").and_then(|x| x.as_str()).unwrap_or("");
                            emit_engine_event(&EngineEvent::MeetingSpan {
                                id: id.to_string(),
                                body: json!({ "error": msg }),
                            });
                        }
                    },
                    "byok" => {
                        let endpoint = phonon_llm::byok::Endpoint::from_command(&v);
                        let on = endpoint.is_some();
                        eng.set_byok(endpoint);
                        emit(&json!({"type":"status","kind":"byok","on":on}));
                    }
                    "websocket_api" => {
                        remote.set_token(v.get("token").and_then(|x| x.as_str()));
                    }
                    "prewarm" => {
                        if let Err(e) = eng.prewarm_polish() {
                            emit(&json!({"type":"error","msg":format!("{e:#}")}));
                        }
                    }
                    "edit_observed" => {
                        let id = v.get("id").and_then(|x| x.as_str());
                        let outcome = serde_json::from_value::<edits::ObservedEdit>(v.clone())
                            .map_err(anyhow::Error::from)
                            .and_then(|request| eng.edit_observed(id, &request));
                        match outcome {
                            Ok(event) => emit(&event),
                            Err(e) => {
                                emit(&json!({"type":"error","id":id,"msg":format!("edit: {e:#}")}))
                            }
                        }
                    }
                    "edit_decision" => {
                        let id = v.get("id").and_then(|x| x.as_str());
                        let index = v.get("index").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                        let decision = v.get("decision").and_then(|x| x.as_str()).unwrap_or("");
                        if let Err(e) = eng.edit_decision(id, index, decision) {
                            emit(
                                &json!({"type":"error","id":id,"msg":format!("edit decision: {e:#}")}),
                            );
                        }
                    }
                    "reload_dictionary" => match data::DictionaryFile::load() {
                        Ok(dictionary) => {
                            let entries = dictionary.entries.len();
                            eng.set_dictionary(dictionary);
                            emit(&json!({
                                "type": "status",
                                "kind": "dictionary",
                                "msg": format!("dictionary reloaded: {entries} entries")
                            }));
                        }
                        Err(error) => emit(&json!({
                            "type": "error",
                            "msg": format!("reload dictionary: {error:#}")
                        })),
                    },
                    "shutdown" => {
                        eprintln!("phonon engine: shutdown requested");
                        eng.shutdown();
                        emit(&json!({"type":"status","msg":"bye"}));
                        return Ok(());
                    }
                    _ => emit(&json!({"type":"error","msg":format!("unknown cmd: {cmd}")})),
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                eng.shutdown();
                return Ok(());
            }
        }
    }
}

const STDIN_CLOSED: &str = "\u{0}stdin closed";

fn emit(v: &serde_json::Value) {
    // Not println!: that panics when the bar has already gone (a quit closes our stdout).
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{v}");
    let _ = out.flush();
}

fn emit_engine_event(ev: &EngineEvent) {
    match ev {
        EngineEvent::Stream {
            name,
            state,
            pct,
            msg,
            load_ms,
        } => emit(&json!({
            "type": "stream",
            "name": name,
            "state": state,
            "pct": pct,
            "msg": msg,
            "load_ms": load_ms,
        })),
        EngineEvent::AllReady { stream_refine } => {
            emit(&json!({"type": "ready", "stream_refine": stream_refine}))
        }
        EngineEvent::AsrResult {
            id,
            text,
            seconds,
            partial,
        } => emit(&json!({
            "type": "result",
            "kind": "asr",
            "id": id,
            "text": text,
            "seconds": seconds,
            "partial": partial,
        })),
        EngineEvent::PolishResult {
            id,
            text,
            latency_ms,
            ttft_ms,
            tok_s,
            app_category,
        } => emit(&json!({
            "type": "result",
            "kind": "polish",
            "id": id,
            "text": text,
            "latency_ms": latency_ms,
            "ttft_ms": ttft_ms,
            "tok_s": tok_s,
            "app_category": app_category,
        })),
        EngineEvent::DraftResult { .. } => emit(&draft_json(ev).unwrap_or_default()),
        EngineEvent::MeetingSpan { .. } => emit(&meeting_span_json(ev).unwrap_or_default()),
        EngineEvent::DictationQuiet { quiet } => emit(&json!({
            "type": "dictation",
            "quiet": quiet,
        })),
        EngineEvent::Error { id, msg } => emit(&json!({
            "type": "error",
            "id": id,
            "msg": msg,
        })),
    }
}

/// `{"type":"result","kind":"draft",...}` for a live-text draft; None for any other event.
fn draft_json(ev: &EngineEvent) -> Option<serde_json::Value> {
    let EngineEvent::DraftResult {
        id,
        text,
        source,
        complete,
    } = ev
    else {
        return None;
    };
    Some(json!({
        "type": "result",
        "kind": "draft",
        "id": id,
        "text": text,
        "source": source,
        "complete": complete,
    }))
}

/// `{"type":"meeting_span","id":..,"sentences":[..],..}` (or `"error"`) for a meeting span;
/// None for any other event. A type of its own, so the bar never reads it as a dictation's.
fn meeting_span_json(ev: &EngineEvent) -> Option<serde_json::Value> {
    let EngineEvent::MeetingSpan { id, body } = ev else {
        return None;
    };
    let mut out = match body {
        serde_json::Value::Object(map) => serde_json::Value::Object(map.clone()),
        _ => json!({}),
    };
    out["type"] = json!("meeting_span");
    out["id"] = json!(id);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::{
        asr_smoke_passed, llm_audio_smoke_passed, llm_checking_message, llm_loading_pct,
        llm_prime_passed, llm_smoke_passed, physical_memory_bytes, safe_polish_output,
        stream_refine_default, stream_refine_setting, StreamFinish,
    };
    use super::{draft_json, draft_source, finish_correction, stream_commit_json, EngineEvent};
    use crate::app_context::AppCategory;
    use crate::data::{DictionaryEntry, DictionaryFile};
    use serde_json::json;
    use std::collections::HashMap;

    #[test]
    fn a_meeting_span_goes_out_under_its_own_type() {
        use super::meeting_span_json;
        let answered = EngineEvent::MeetingSpan {
            id: "m1:mic:0".into(),
            body: json!({"type":"span","id":"m1:mic:0","sentences":[{"start":0.5,"end":1.0,"text":"Hi."}],
                         "quiet":false,"seconds":0.2}),
        };
        let out = meeting_span_json(&answered).unwrap();
        assert_eq!(out["type"], "meeting_span");
        assert_eq!(out["id"], "m1:mic:0");
        assert_eq!(out["sentences"][0]["end"], 1.0);
        let failed = EngineEvent::MeetingSpan {
            id: "m1:call:3".into(),
            body: json!({"error": "transcribe span failed"}),
        };
        let out = meeting_span_json(&failed).unwrap();
        assert_eq!(
            (out["type"].as_str(), out["error"].as_str()),
            (Some("meeting_span"), Some("transcribe span failed"))
        );
        let other = EngineEvent::Error {
            id: None,
            msg: "x".into(),
        };
        assert!(meeting_span_json(&other).is_none());
    }

    /// Issue #24: the endpoint's text wins only when it arrives in time and passes
    /// the guard; an error, a timeout, a hung server or a runaway reply all fall back
    /// to the local result.
    #[test]
    fn own_endpoint_falls_back_to_the_local_result() {
        use super::{byok_text_passes, take_byok_result, ByokJob};
        use std::time::{Duration, Instant};
        let mut jobs: HashMap<String, ByokJob> = HashMap::new();
        let job = |result: Option<anyhow::Result<String>>, wait_ms: u64| {
            let (tx, rx) = std::sync::mpsc::channel();
            if let Some(result) = result {
                tx.send(result).unwrap();
            } else {
                std::mem::forget(tx); // never answers
            }
            ByokJob {
                deadline: Instant::now() + Duration::from_millis(wait_ms),
                reply: rx,
            }
        };
        jobs.insert("ok".into(), job(Some(Ok("Remote text.".into())), 500));
        jobs.insert(
            "err".into(),
            job(Some(Err(anyhow::anyhow!("HTTP 500"))), 500),
        );
        jobs.insert("late".into(), job(None, 200));
        assert_eq!(
            take_byok_result(&mut jobs, Some("ok")).as_deref(),
            Some("Remote text.")
        );
        assert_eq!(take_byok_result(&mut jobs, Some("err")), None);
        let started = Instant::now();
        assert_eq!(take_byok_result(&mut jobs, Some("late")), None);
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "waits only to the deadline"
        );
        assert_eq!(take_byok_result(&mut jobs, Some("unknown")), None);
        assert_eq!(take_byok_result(&mut jobs, None), None);
        assert!(jobs.is_empty());

        // a real HTTP endpoint that accepts and never answers: the local result stands
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = phonon_llm::byok::Endpoint {
            base_url: format!("http://{}/v1", listener.local_addr().unwrap()),
            model: "m".into(),
            api_key: String::new(),
            timeout: Duration::from_millis(600),
        };
        let (tx, rx) = std::sync::mpsc::channel();
        let deadline = Instant::now() + endpoint.timeout;
        std::thread::spawn(move || {
            let _ = tx.send(phonon_llm::byok::complete(&endpoint, "S", "t"));
        });
        jobs.insert(
            "hung".into(),
            ByokJob {
                deadline,
                reply: rx,
            },
        );
        assert_eq!(take_byok_result(&mut jobs, Some("hung")), None);
        drop(listener);

        let dictionary = DictionaryFile::default();
        let source = "run v llm on black well tonight and log the throughput";
        assert!(byok_text_passes(
            &dictionary,
            source,
            "Run vLLM on Blackwell tonight and log the throughput."
        ));
        assert!(!byok_text_passes(&dictionary, source, ""), "empty reply");
        assert!(
            !byok_text_passes(
                &dictionary,
                source,
                &"Sure! Here is a long answer. ".repeat(20)
            ),
            "an answer instead of a correction"
        );
        assert!(
            byok_text_passes(&dictionary, "ship it", "ship it"),
            "unchanged is fine"
        );
    }

    #[test]
    fn greeting_and_sign_off_lines_only_in_email() {
        use super::final_layout;
        let text = "Hi Sarah, the build is green. Thanks, Chris";
        // Slack and Teams map to formal: the message stays one line
        assert_eq!(
            final_layout(text, "formal", Some(AppCategory::WorkChat)),
            text
        );
        // email: greeting and sign-off on their own lines
        let email = final_layout(text, "formal", Some(AppCategory::Email));
        assert!(email.starts_with("Hi Sarah,\n"), "{email:?}");
        assert!(email.contains("\nThanks"), "{email:?}");
        // numbered lists break in both, as before
        let steps = "Hi Sarah, two things: 1. ship it 2. tag it";
        assert_eq!(
            final_layout(steps, "formal", Some(AppCategory::WorkChat)),
            "Hi Sarah, two things:\n1. ship it\n2. tag it"
        );
        assert!(final_layout(steps, "formal", Some(AppCategory::Email))
            .contains("\n1. ship it\n2. tag it"));
        assert_eq!(
            final_layout("1. eggs 2. milk", "casual", Some(AppCategory::Chat)),
            "1. eggs 2. milk"
        );
    }

    #[test]
    fn model_written_greeting_and_sign_off_lines_join_only_in_chat() {
        use super::final_layout;
        // RLQv4U's formal output on a public fmt_v0 email probe
        let model = "Hello team,\n\nThe dependencies are pinned. Please let me know.\n\nBest regards,\nChris";
        assert_eq!(
            final_layout(model, "formal", Some(AppCategory::WorkChat)),
            "Hello team, the dependencies are pinned. Please let me know. Best regards, Chris"
        );
        // the mail app keeps the email layout; other categories are untouched
        let email = final_layout(model, "formal", Some(AppCategory::Email));
        assert!(email.starts_with("Hello team,\n"), "{email:?}");
        assert!(email.contains("\nChris"), "{email:?}");
        for category in [None, Some(AppCategory::Docs), Some(AppCategory::Other)] {
            assert_eq!(final_layout(model, "formal", category), model);
        }
        // a list inside the chat message keeps its lines
        let steps = "Hi Sarah,\n\nTwo things:\n1. ship it\n2. tag it\n\nThanks,\nChris";
        assert_eq!(
            final_layout(steps, "formal", Some(AppCategory::WorkChat)),
            "Hi Sarah, Two things:\n1. ship it\n2. tag it\n\nThanks,\nChris"
        );
    }

    #[test]
    fn layout_modes_follow_the_category_table() {
        use super::category_mode;
        assert_eq!(category_mode(Some(AppCategory::Chat)), "casual");
        assert_eq!(category_mode(Some(AppCategory::WorkChat)), "formal");
        assert_eq!(category_mode(Some(AppCategory::Email)), "formal");
        assert_eq!(category_mode(Some(AppCategory::Code)), "prompt");
        assert_eq!(category_mode(Some(AppCategory::AiPrompt)), "prompt");
        assert_eq!(category_mode(Some(AppCategory::Docs)), "minimal");
        assert_eq!(category_mode(None), "minimal");
    }

    #[test]
    fn a_live_text_draft_maps_to_a_draft_result_event() {
        let event = EngineEvent::DraftResult {
            id: "p7".into(),
            text: "Run vLLM.".into(),
            source: "run v llm".into(),
            complete: false,
        };
        assert_eq!(
            draft_json(&event),
            Some(json!({"type": "result", "kind": "draft", "id": "p7",
                        "text": "Run vLLM.", "source": "run v llm", "complete": false}))
        );
        let error = EngineEvent::Error {
            id: None,
            msg: "x".into(),
        };
        assert_eq!(draft_json(&error), None);
    }

    #[test]
    fn drafts_go_out_only_for_the_open_live_text_stream() {
        let streams = HashMap::from([("p2".to_string(), vec!["a".into(), "a b".into()])]);
        assert_eq!(
            draft_source(Some("p2"), &streams, "p2", Some(1)),
            Some("a b".into())
        );
        assert_eq!(draft_source(None, &streams, "p2", Some(1)), None); // setting off
        assert_eq!(draft_source(Some("p2"), &streams, "p1", Some(1)), None); // old stream
        assert_eq!(draft_source(Some("p3"), &streams, "p3", Some(0)), None); // finished
        assert_eq!(draft_source(Some("p2"), &streams, "p2", Some(5)), None); // unknown commit
        assert_eq!(draft_source(Some("p2"), &streams, "p2", None), None);
    }

    #[test]
    fn a_commit_asks_for_its_draft_only_with_live_text() {
        assert_eq!(stream_commit_json(None, None), None); // the default path is unchanged
        assert_eq!(
            stream_commit_json(Some(AppCategory::Chat), None),
            Some(json!({"app_category": "chat"}))
        );
        assert_eq!(
            stream_commit_json(Some(AppCategory::Chat), Some(("p2", 3))),
            Some(json!({"app_category": "chat", "draft_id": "p2", "draft_seq": 3}))
        );
        assert_eq!(
            stream_commit_json(None, Some(("p2", 0))),
            Some(json!({"draft_id": "p2", "draft_seq": 0}))
        );
    }

    #[test]
    fn drafts_and_the_final_polish_share_one_correction_path() {
        let dictionary = DictionaryFile {
            entries: vec![DictionaryEntry {
                phrase: "vee llm".into(),
                replacement: Some("vLLM".into()),
                spoken_forms: Vec::new(),
                source: "test".into(),
                starred: false,
                usage_count: 0,
            }],
            ..DictionaryFile::default()
        };
        let style = vec!["lowercase_start".to_string()];
        let corrected = finish_correction(
            &dictionary,
            "run vee llm now",
            "Run vee llm now.",
            &style,
            "minimal",
        );
        let mut expected = safe_polish_output(&dictionary, "run vee llm now", "Run vee llm now.");
        expected.text = crate::app_context::apply_style(&expected.text, &style, &dictionary);
        assert_eq!(corrected.text, expected.text);
        assert!(corrected.text.contains("vLLM"));
    }

    #[test]
    fn the_correction_path_restores_a_dropped_opener() {
        use super::final_layout;
        let dictionary = DictionaryFile::default();
        let corrected = finish_correction(
            &dictionary,
            "okay so the build is green",
            "So the build is green.",
            &[],
            "prompt",
        );
        assert_eq!(corrected.text, "Okay, so the build is green.");
        // then the line breaks, as in the final polish
        let corrected = finish_correction(
            &dictionary,
            "and the steps are 1. pull 2. build",
            "The steps are 1. Pull 2. Build.",
            &[],
            "prompt",
        );
        assert_eq!(
            final_layout(&corrected.text, "prompt", None),
            "And the steps are\n1. Pull\n2. Build."
        );
        // formal, or a request without a mode: the refiner's text as it is (with a fix
        // beside the dropped opener, or the guard swaps in the raw transcript)
        for mode in ["formal", ""] {
            let corrected = finish_correction(
                &dictionary,
                "okay the bild is green and the tests pass",
                "The build is green and the tests pass.",
                &[],
                mode,
            );
            assert_eq!(
                corrected.text, "The build is green and the tests pass.",
                "{mode:?}"
            );
        }
    }

    #[test]
    fn a_stream_finished_at_key_up_waits_for_both_its_text_and_the_gate() {
        // text first, then the gate finds speech
        let mut finish = StreamFinish {
            text: Some(("hello there".into(), 0.05)),
            ..StreamFinish::default()
        };
        assert_eq!(finish.resolve(), (None, false));
        finish.speech = Some(true);
        assert_eq!(finish.resolve(), (Some(("hello there".into(), 0.05)), true));
        // gate first, then the text
        let mut finish = StreamFinish {
            speech: Some(true),
            ..StreamFinish::default()
        };
        assert_eq!(finish.resolve(), (None, false));
        finish.text = Some(("hi".into(), 0.04));
        assert_eq!(finish.resolve(), (Some(("hi".into(), 0.04)), true));
        // no speech by the gate: the recognizer's words decide, so a short hallucination
        // of silence is dropped
        let mut finish = StreamFinish {
            speech: Some(false),
            ..StreamFinish::default()
        };
        assert_eq!(finish.resolve(), (None, false));
        finish.text = Some(("thank you".into(), 0.04));
        assert_eq!(finish.resolve(), (Some((String::new(), 0.0)), true));
        // ... and quiet speech the gate missed still goes out (a clip of 2026-11-03)
        let mut finish = StreamFinish {
            speech: Some(false),
            ..StreamFinish::default()
        };
        let quiet = "So basically plan, then execute, and use sub-agents where necessary.";
        finish.text = Some((quiet.into(), 0.03));
        assert_eq!(finish.resolve(), (Some((quiet.into(), 0.03)), true));
    }

    #[test]
    fn streaming_refine_defaults_on_from_16_gb() {
        assert!(!stream_refine_default(Some(8 << 30)));
        assert!(!stream_refine_default(Some((16 << 30) - (1 << 20))));
        assert!(stream_refine_default(Some(16 << 30)));
        assert!(stream_refine_default(Some(36 << 30)));
        assert!(!stream_refine_default(None));
        if cfg!(target_os = "macos") {
            assert!(physical_memory_bytes().is_some_and(|bytes| bytes >= 4 << 30));
        }
    }

    #[test]
    fn stream_refine_follows_the_setting_unless_the_environment_says() {
        assert!(stream_refine_setting(None, true));
        assert!(!stream_refine_setting(None, false));
        assert!(stream_refine_setting(Some(" On ".into()), false));
        assert!(!stream_refine_setting(Some("0".into()), true));
        assert!(!stream_refine_setting(Some("false".into()), true));
        assert!(stream_refine_setting(Some("maybe".into()), true));
    }

    #[test]
    fn startup_smoke_validation_rejects_empty_or_collapsed_results() {
        assert!(asr_smoke_passed("Hello, Fluid Voice."));
        assert!(asr_smoke_passed("Hello Flamid Voice."));
        assert!(!asr_smoke_passed("Hello."));
        assert!(llm_audio_smoke_passed(
            "Hello Flamid Voice.",
            "Hello FluidAudio Voice."
        ));
        assert!(!llm_audio_smoke_passed("Hello Flamid Voice.", "Hello."));
        assert!(!llm_audio_smoke_passed("Hello Flamid Voice.", ""));
        assert!(!llm_audio_smoke_passed(
            "Hello Flamid Voice.",
            "Hello Fluid Voice, and here is a great deal of invented extra content."
        ));
        assert!(!llm_audio_smoke_passed(
            "Hello Flamid Voice.",
            "<phonon_dictionary>\ncanonical_terms: Hello, Fluid Voice"
        ));
        assert!(llm_smoke_passed(
            "Do you know the difference between cuBLAS and cuDNN?"
        ));
        assert!(llm_smoke_passed(
            "<phonon_dictionary>ignored</phonon_dictionary>\n<transcript>Do you know the difference between cuBLAS and cuDNN?</transcript>"
        ));
        assert!(!llm_smoke_passed("Test."));
        assert!(llm_prime_passed("Hello Fluid Voice startup prime."));
        assert!(!llm_prime_passed("Hello."));
        // A refiner's legitimate rewrite of the product name still primes.
        assert!(llm_prime_passed("hello FluentBit startup prime"));
        // Unrelated text of the right length, and a runaway, do not.
        assert!(!llm_prime_passed("the quick brown fox jumps"));
        assert!(!llm_prime_passed(
            "hello fluid voice startup prime and then a long explanation follows here"
        ));
    }

    #[test]
    fn llm_loading_progress_stays_below_weights_loaded() {
        assert_eq!(llm_loading_pct(None), 0.05);
        assert!((llm_loading_pct(Some(0.5)) - 0.35).abs() < 1e-9);
        assert!(llm_loading_pct(Some(7.0)) < 0.78);
        assert_eq!(llm_loading_pct(Some(-1.0)), 0.05);
    }

    #[test]
    fn llm_checking_message_names_the_mlx_fallback() {
        assert_eq!(
            llm_checking_message("x loaded in 1.0s; native refiner engine"),
            "Checking the correction model"
        );
        assert!(llm_checking_message(
            "x loaded in 1.0s; MLX refiner (native engine unavailable: probe request failed)"
        )
        .contains("MLX fallback"));
        assert!(llm_checking_message(
            "x loaded in 1.0s; MLX refiner (native engine unavailable: no engine binary)"
        )
        .contains("no native engine"));
    }

    #[test]
    fn final_polish_rejects_one_word_collapse_and_applies_dictionary() {
        let dictionary = DictionaryFile {
            entries: vec![DictionaryEntry {
                phrase: "black well".into(),
                replacement: Some("Blackwell".into()),
                spoken_forms: Vec::new(),
                source: "test".into(),
                starred: true,
                usage_count: 0,
            }],
            ..Default::default()
        };
        let source = "Please benchmark black well with the longer prompt.";
        let output = safe_polish_output(&dictionary, source, "benchmark");
        assert_eq!(
            output.text,
            "Please benchmark Blackwell with the longer prompt."
        );
        assert_eq!(output.applied.len(), 1);
    }
}
