//! Shared warm engines: Parakeet ASR + local correction model over JSONL.
//!
//! Protocol (stdin/stdout, one JSON object per line):
//!   {"cmd":"status"}
//!   {"cmd":"transcribe","path":"/tmp/x.wav","id":"..."}
//!   {"cmd":"polish","text":"...","id":"..."}
//!   {"cmd":"reload_dictionary"}
//!   {"cmd":"shutdown"}
//!
//! Events (stdout):
//!   {"type":"stream","name":"asr|llm","state":"loading|ready|error","pct":0.5,"msg":"...","load_ms":123}
//!   {"type":"ready"}  — all required streams ready
//!   {"type":"result","kind":"asr|polish","id":"...","text":"...","seconds":0.1,"latency_ms":70,"ttft_ms":30,"tok_s":200}
//!   {"type":"error","id":"...","msg":"..."}

pub mod data;
mod paths;

use anyhow::{bail, Context, Result};
pub use paths::project_root;
use phonon_asr::{AsrEvent, AsrSidecar};
use phonon_audio::wav_has_speech;
use phonon_llm::{polish_available, PolishEvent, PolishSidecar};
use serde_json::json;
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::mpsc::{self, TryRecvError};
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
    AllReady,
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
    dictionary: data::DictionaryFile,
    recording_ids: HashMap<String, String>,
    pending_events: Vec<EngineEvent>,
    all_ready_emitted: bool,
    llm_primed: bool,
    llm_primed_at: Option<Instant>,
    startup_asr_text: Option<String>,
    startup_audio_demo_sent: bool,
    startup_audio_demo_passed: bool,
    startup_dictionary_demo_sent: bool,
}

#[derive(Debug, Clone)]
struct PolishRequest {
    id: Option<String>,
    source_text: String,
    screen_context_terms: Vec<String>,
}

impl Engine {
    pub fn start(root: &Path) -> Result<Self> {
        if !polish_available(root) {
            bail!("the required local correction runtime is missing from this install");
        }
        let asr = AsrSidecar::spawn(root)?;
        let polisher = PolishSidecar::spawn_with(root, &data::polish_config()?)?
            .context("the required local correction runtime could not start")?;

        Ok(Self {
            asr,
            polisher: Some(polisher),
            asr_ready: false,
            llm_ready: false,
            root: root.to_path_buf(),
            started: Instant::now(),
            polish_requests: Vec::new(),
            dictionary: data::DictionaryFile::load()?,
            recording_ids: HashMap::new(),
            pending_events: Vec::new(),
            all_ready_emitted: false,
            llm_primed: false,
            llm_primed_at: None,
            startup_asr_text: None,
            startup_audio_demo_sent: false,
            startup_audio_demo_passed: false,
            startup_dictionary_demo_sent: false,
        })
    }

    pub fn stacks_ready(&self) -> bool {
        self.asr_ready && self.llm_ready
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
                AsrEvent::Ready { model, load_ms } => {
                    out.push(EngineEvent::Stream {
                        name: "asr".into(),
                        state: "loading".into(),
                        pct: 0.78,
                        msg: format!("{model} loaded; running demo pass"),
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
                AsrEvent::Result {
                    id,
                    text,
                    seconds,
                    partial,
                } => {
                    if id.as_deref() == Some(STARTUP_ASR_BATCH_ID) {
                        if asr_smoke_passed(&text) {
                            self.startup_asr_text = Some(text.clone());
                            out.push(EngineEvent::Stream {
                                name: "asr".into(),
                                state: "loading".into(),
                                pct: 0.9,
                                msg: format!("batch demo passed: {text}; warming stream"),
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
                                msg: format!("batch + streaming demos passed: {text}"),
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
                AsrEvent::Error { msg } => out.push(EngineEvent::Error { id: None, msg }),
            }
        }
        if let Some(polisher) = self.polisher.as_mut() {
            for event in polisher.poll() {
                match event {
                    PolishEvent::Ready { msg, load_ms } => {
                        out.push(EngineEvent::Stream {
                            name: "llm".into(),
                            state: "loading".into(),
                            pct: 0.78,
                            msg: format!("weights loaded · {msg}; compiling demo request"),
                            load_ms: Some(load_ms),
                        });
                        self.polish_requests.push(PolishRequest {
                            id: Some(STARTUP_LLM_PRIME_ID.into()),
                            source_text: STARTUP_LLM_PRIME_TEXT.into(),
                            screen_context_terms: Vec::new(),
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
                            PolishRequest {
                                id: None,
                                source_text: String::new(),
                                screen_context_terms: Vec::new(),
                            }
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
                                    msg: "base graph primed; waiting for speech warmup".into(),
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
                                    msg: "startup.wav passed through ASR and correction".into(),
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
                                    msg: format!(
                                        "demo passed · TTFT {ttft_ms:.0}ms · {tok_s:.0} tok/s"
                                    ),
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
                            let corrected =
                                safe_polish_output(&self.dictionary, &request.source_text, &text);
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
                                ) {
                                    out.push(EngineEvent::Error {
                                        id: id.clone(),
                                        msg: format!("save recording metadata: {error:#}"),
                                    });
                                }
                            }
                            out.push(EngineEvent::PolishResult {
                                id,
                                text: corrected.text,
                                latency_ms,
                                ttft_ms,
                                tok_s,
                            });
                        }
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
                    msg: "running startup.wav through the correction model".into(),
                    load_ms: None,
                });
                self.polish_requests.push(PolishRequest {
                    id: Some(STARTUP_LLM_AUDIO_ID.into()),
                    source_text: startup_text.clone(),
                    screen_context_terms: Vec::new(),
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
                    msg: "running full dictionary correction demo".into(),
                    load_ms: None,
                });
                self.polish_requests.push(PolishRequest {
                    id: Some(STARTUP_LLM_ID.into()),
                    source_text: STARTUP_LLM_TEXT.into(),
                    screen_context_terms: Vec::new(),
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
            out.push(EngineEvent::AllReady);
        }
        out
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
            let metadata = data::register_recording(path, source)?;
            self.recording_ids
                .insert(id.to_string(), metadata.id.clone());
            let speech_detected = wav_has_speech(path)?;
            data::set_speech_detected(&metadata.id, speech_detected)?;
            if !speech_detected {
                self.pending_events.push(EngineEvent::AsrResult {
                    id: Some(id.to_string()),
                    text: String::new(),
                    seconds: 0.0,
                    partial: false,
                });
                return Ok(());
            }
        }
        self.asr.transcribe(path, id)
    }

    pub fn stream_start(&mut self, id: Option<&str>) -> Result<()> {
        self.asr.stream_start(id)
    }

    pub fn stream_chunk(&mut self, pcm16: &str, id: Option<&str>) -> Result<()> {
        self.asr.stream_chunk(pcm16, id)
    }

    pub fn stream_stop(&mut self, id: Option<&str>) -> Result<()> {
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
        let Some(polisher) = self.polisher.as_mut() else {
            bail!("the required local correction model is unavailable");
        };
        if !self.llm_ready {
            bail!("the correction model is not ready yet");
        }
        let input = self
            .dictionary
            .prepare_polish_input_with_context(text, screen_text);
        self.polish_requests.push(PolishRequest {
            id: id.map(str::to_string),
            source_text: text.to_string(),
            screen_context_terms: self.dictionary.screen_confirmed_terms(text, screen_text),
        });
        if let Err(error) = polisher.polish(&input) {
            self.polish_requests.pop();
            return Err(error);
        }
        Ok(())
    }

    pub fn polish_available(&self) -> bool {
        self.polisher.is_some()
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
) -> Result<()> {
    let mut recording = data::load_recording_by_id(recording_id)?;
    recording.final_transcript = final_text.trim().to_string();
    recording.dictionary_corrections = corrections;
    recording.screen_context_terms = screen_context_terms;
    recording.llm = Some(llm);
    data::save_recording(&recording)
}

fn safe_polish_output(
    dictionary: &data::DictionaryFile,
    source: &str,
    polished: &str,
) -> data::CorrectionResult {
    data::safe_polish_output(dictionary, source, polished)
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

fn llm_prime_passed(text: &str) -> bool {
    let words = normalized_words(data::extract_transcript_payload(text));
    words.iter().any(|word| word == "fluid")
        && words.iter().any(|word| word == "voice")
        && words.iter().any(|word| word == "prime")
}

/// Long-lived CLI: `phonon engine` — JSONL on stdin/stdout for the floating bar.
pub fn run_engine_serve() -> Result<()> {
    let root = project_root();
    let mut eng = Engine::start(&root)?;

    emit(&json!({
        "type": "stream",
        "name": "asr",
        "state": "loading",
        "pct": 0.05,
        "msg": "starting parakeet"
    }));
    emit(&json!({
        "type": "stream",
        "name": "llm",
        "state": "loading",
        "pct": 0.05,
        "msg": "starting the correction model"
    }));

    let (cmd_tx, cmd_rx) = mpsc::channel::<String>();
    thread::spawn(move || {
        let stdin = std::io::stdin();
        let mut lock = stdin.lock();
        let mut input = String::new();
        loop {
            input.clear();
            match lock.read_line(&mut input) {
                Ok(0) => break,
                Ok(_) => {
                    if cmd_tx.send(input.clone()).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });

    loop {
        for ev in eng.poll() {
            emit_engine_event(&ev);
        }
        match cmd_rx.try_recv() {
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
                        if let Err(e) = eng.stream_start(id) {
                            emit(&json!({"type":"error","id":id,"msg":format!("{e:#}")}));
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
                    "polish" => {
                        let text = v.get("text").and_then(|x| x.as_str()).unwrap_or("");
                        let context = v.get("context").and_then(|x| x.as_str()).unwrap_or("");
                        let id = v.get("id").and_then(|x| x.as_str());
                        if let Err(e) = eng.polish_with_context(text, context, id) {
                            emit(&json!({"type":"error","id":id,"msg":format!("{e:#}")}));
                        }
                    }
                    "reload_dictionary" => match data::DictionaryFile::load() {
                        Ok(dictionary) => {
                            let entries = dictionary.entries.len();
                            eng.dictionary = dictionary;
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
                        eng.shutdown();
                        emit(&json!({"type":"status","msg":"bye"}));
                        return Ok(());
                    }
                    _ => emit(&json!({"type":"error","msg":format!("unknown cmd: {cmd}")})),
                }
            }
            Err(TryRecvError::Empty) => {
                thread::sleep(Duration::from_millis(15));
            }
            Err(TryRecvError::Disconnected) => {
                eng.shutdown();
                return Ok(());
            }
        }
    }
}

fn emit(v: &serde_json::Value) {
    println!("{v}");
    let _ = std::io::stdout().flush();
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
        EngineEvent::AllReady => emit(&json!({"type": "ready"})),
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
        } => emit(&json!({
            "type": "result",
            "kind": "polish",
            "id": id,
            "text": text,
            "latency_ms": latency_ms,
            "ttft_ms": ttft_ms,
            "tok_s": tok_s,
        })),
        EngineEvent::Error { id, msg } => emit(&json!({
            "type": "error",
            "id": id,
            "msg": msg,
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        asr_smoke_passed, llm_audio_smoke_passed, llm_prime_passed, llm_smoke_passed,
        safe_polish_output,
    };
    use crate::data::{DictionaryEntry, DictionaryFile};

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
