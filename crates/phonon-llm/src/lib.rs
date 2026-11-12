//! Transcript correction over a pinned local MLX model (`sidecar/polish_server.py`).
//!
//! Protocol:
//!   stdin JSONL:
//!     {"command":"warmup"}
//!     {"command":"run","inputText":"…"}
//!     {"command":"prewarm"}  (reply: a non-ready status line, skipped)
//!     {"command":"status"}
//!     {"command":"shutdown"}
//!   stdout JSONL responses with diagnostics.
//!
//! `ServeJson` is the synchronous handle over the same protocol, used by
//! evaluation and benchmark tooling. `ServeJson::spawn_fluid` and `cold_run`
//! reach the old FluidVoice helper instead; they exist only so `phonon bench` /
//! `phonon profile` can compare against that baseline locally.

pub mod byok;
pub mod linebreak;
pub mod numpass;
pub mod opener;
mod paths;

use anyhow::{bail, Context, Result};
pub use paths::{
    bundled_engine, fluid1_available, fluid1_paths, fluid_drafter_dir, fluid_helper,
    fluid_model_dir, polish_available, POLISH_MODEL_ID, POLISH_MODEL_REVISION,
    POLISH_RUNTIME_REQUIREMENT,
};
use paths::{polish_prompt, polish_script};
use phonon_asr::{StderrTail, Waker};
use serde::Deserialize;
use serde_json::json;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Deserialize)]
pub struct ServeJsonResp {
    pub ok: bool,
    #[serde(default, rename = "outputText")]
    pub output_text: Option<String>,
    pub error: Option<String>,
    pub status: Option<ServeStatus>,
    pub diagnostics: Option<ServeDiagnostics>,
    /// The reply to a background `stream_commit` (never a correction result).
    #[serde(default)]
    pub stream: Option<StreamReply>,
}

/// The live-text part of a `stream_commit` reply; present only when the commit's context
/// carried a `draft_id`.
#[derive(Debug, Clone, Deserialize)]
pub struct StreamReply {
    #[serde(default, rename = "draftId")]
    pub draft_id: Option<String>,
    /// The commit's place in its stream (the engine's `draft_seq`), echoed back.
    #[serde(default, rename = "draftSeq")]
    pub draft_seq: Option<usize>,
    #[serde(default, rename = "draftText")]
    pub draft_text: Option<String>,
    #[serde(default, rename = "draftComplete")]
    pub draft_complete: Option<bool>,
}

impl ServeJsonResp {
    /// The background draft of a live-text commit, if this reply is one.
    pub fn draft(&self) -> Option<PolishEvent> {
        let stream = self.stream.as_ref()?;
        Some(PolishEvent::Draft {
            id: stream.draft_id.clone()?,
            seq: stream.draft_seq,
            text: stream.draft_text.clone()?.trim().to_string(),
            complete: stream.draft_complete.unwrap_or(false),
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServeStatus {
    #[allow(dead_code)]
    pub state: Option<String>,
    pub message: Option<String>,
    /// Fraction of a loading stage done (the first-launch download), 0 to 1.
    #[serde(default)]
    pub progress: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServeDiagnostics {
    #[serde(rename = "latencyMilliseconds")]
    pub latency_ms: Option<f64>,
    #[serde(rename = "timeToFirstTokenMilliseconds")]
    pub ttft_ms: Option<f64>,
    #[serde(rename = "tokensPerSecond")]
    pub tokens_per_second: Option<f64>,
    #[serde(rename = "generatedTokenCount")]
    pub generated_tokens: Option<u64>,
    #[serde(rename = "inputCharacterCount")]
    #[allow(dead_code)]
    pub input_chars: Option<u64>,
    #[serde(rename = "outputCharacterCount")]
    #[allow(dead_code)]
    pub output_chars: Option<u64>,
    #[serde(rename = "stageTimingsMilliseconds")]
    pub stage_timings: Option<serde_json::Value>,
    #[serde(rename = "promptTokenCount")]
    pub prompt_tokens: Option<u64>,
    /// Prompt tokens served from the sidecar's prefix cache (0 on a miss).
    #[serde(rename = "cachedPromptTokenCount")]
    pub cached_prompt_tokens: Option<u64>,
    /// Tokens the speaker profile block occupies in the prompt (0 without one).
    #[serde(rename = "profileTokenCount")]
    pub profile_tokens: Option<u64>,
}

/// How the correction sidecar builds its prompt prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolishConfig {
    /// Directory holding `user.md` and `vocab.md` (SPEC, vocabulary onboarding
    /// step 5). `None` leaves the prompt exactly as shipped without a profile.
    pub profile_dir: Option<PathBuf>,
    /// Token bound for the profile block; `None` is the sidecar default (1500).
    pub profile_token_budget: Option<u32>,
    /// Keep the KV state of the stable prompt prefix between requests. Off
    /// only for developer comparisons; output text does not depend on it.
    pub prefix_cache: bool,
    /// A local MLX model directory used instead of the pinned download: the
    /// user's own trained refiner (`phonon_refiner.json` inside switches the
    /// sidecar to the refiner prompt and guard). `None` runs the pinned model.
    pub model_dir: Option<PathBuf>,
    /// Refine while the speaker talks (SPEC "Streaming refine"); the sidecar reads it
    /// from PHONON_STREAM_REFINE.
    pub stream_refine: bool,
}

impl Default for PolishConfig {
    fn default() -> Self {
        Self {
            profile_dir: None,
            profile_token_budget: None,
            prefix_cache: true,
            model_dir: None,
            stream_refine: false,
        }
    }
}

impl PolishConfig {
    /// Developer overrides, read by the eval tooling; unset in normal use.
    ///   PHONON_PROFILE_DIR=<dir>         profile directory (empty disables)
    ///   PHONON_PROFILE_PREFIX=0|1        force the profile block off or on
    ///   PHONON_PROFILE_TOKEN_BUDGET=<n>  bound for the profile block
    ///   PHONON_POLISH_PREFIX_CACHE=0     prefill the whole prompt every call
    pub fn with_env_overrides(mut self, default_profile_dir: Option<&Path>) -> Self {
        if let Ok(dir) = std::env::var("PHONON_PROFILE_DIR") {
            self.profile_dir = (!dir.is_empty()).then(|| PathBuf::from(dir));
        }
        match std::env::var("PHONON_PROFILE_PREFIX").as_deref() {
            Ok("0") | Ok("false") | Ok("off") => self.profile_dir = None,
            Ok("1") | Ok("true") | Ok("on") if self.profile_dir.is_none() => {
                self.profile_dir = default_profile_dir.map(Path::to_path_buf);
            }
            _ => {}
        }
        if let Some(budget) = std::env::var("PHONON_PROFILE_TOKEN_BUDGET")
            .ok()
            .and_then(|value| value.parse().ok())
        {
            self.profile_token_budget = Some(budget);
        }
        if matches!(
            std::env::var("PHONON_POLISH_PREFIX_CACHE").as_deref(),
            Ok("0") | Ok("false") | Ok("off")
        ) {
            self.prefix_cache = false;
        }
        self
    }

    /// Environment overrides only; the engine passes settings in instead.
    pub fn from_env() -> Self {
        Self::default().with_env_overrides(None)
    }

    /// The sidecar's `--model` and `--revision`: the user's own refiner
    /// directory when one is configured (used as it is, including the engine
    /// binary its `phonon_engine.json` names), otherwise the pinned public
    /// refiner download.
    pub fn model_and_revision(&self) -> (String, String) {
        match &self.model_dir {
            Some(dir) => (dir.display().to_string(), "local".to_string()),
            None => (
                POLISH_MODEL_ID.to_string(),
                POLISH_MODEL_REVISION.to_string(),
            ),
        }
    }
}

/// Synchronous serve-json handle for bench / tooling.
pub struct ServeJson {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    /// Drained continuously: the sidecar reports download progress on stderr,
    /// and an undrained pipe would eventually block it mid-load.
    stderr: Option<StderrTail>,
}

impl ServeJson {
    /// The shipped correction stage, so evaluation measures what users run.
    /// Honours `PHONON_POLISH_BACKEND=fluid` for developer comparisons.
    pub fn spawn(root: &Path) -> Result<Self> {
        Self::spawn_with(root, &PolishConfig::from_env())
    }

    pub fn spawn_with(root: &Path, config: &PolishConfig) -> Result<Self> {
        let cmd = polish_command(root, config)?.context("correction stage files missing")?;
        Self::attach(cmd)
    }

    /// The fluid-1 helper, for developer comparison only.
    pub fn spawn_fluid(root: &Path, use_mtp: bool) -> Result<Self> {
        Self::attach(fluid_command(root, use_mtp)?)
    }

    fn attach(mut cmd: Command) -> Result<Self> {
        let mut child = cmd.spawn().context("spawn serve-json")?;
        let stdin = child.stdin.take().context("serve-json stdin")?;
        let stdout = BufReader::new(child.stdout.take().context("serve-json stdout")?);
        let stderr = child
            .stderr
            .take()
            .map(|stderr| StderrTail::capture(stderr, "serve-json"));
        Ok(Self {
            child,
            stdin,
            stdout,
            stderr,
        })
    }

    pub fn request(&mut self, payload: &serde_json::Value) -> Result<ServeJsonResp> {
        writeln!(self.stdin, "{payload}")?;
        self.stdin.flush()?;
        self.read_response()
    }

    fn read_response(&mut self) -> Result<ServeJsonResp> {
        let mut line = String::new();
        // serve-json is single-threaded; one response line per request.
        // Cap wait so a hung helper fails the bench instead of locking us.
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            if Instant::now() > deadline {
                bail!("serve-json timed out waiting for response");
            }
            line.clear();
            let n = self.stdout.read_line(&mut line)?;
            if n == 0 {
                match self.stderr.as_ref().map(StderrTail::text) {
                    Some(tail) if !tail.trim().is_empty() => {
                        bail!("serve-json exited: {}", tail.trim())
                    }
                    _ => bail!("serve-json EOF"),
                }
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            return serde_json::from_str(trimmed)
                .with_context(|| format!("bad serve-json line: {trimmed}"));
        }
    }

    /// Warmup streams progress: several `loading` lines before `ready`. Consume
    /// them all, or every later response arrives one request behind.
    pub fn warmup(&mut self) -> Result<(Duration, ServeJsonResp)> {
        let t0 = Instant::now();
        let mut resp = self.request(&json!({"command": "warmup"}))?;
        while resp.ok && resp.status.as_ref().and_then(|s| s.state.as_deref()) == Some("loading") {
            resp = self.read_response()?;
        }
        Ok((t0.elapsed(), resp))
    }

    pub fn run_text(&mut self, text: &str) -> Result<(Duration, ServeJsonResp)> {
        let t0 = Instant::now();
        let resp = self.request(&json!({"command": "run", "inputText": text}))?;
        Ok((t0.elapsed(), resp))
    }

    pub fn shutdown(mut self) {
        let _ = writeln!(self.stdin, "{}", json!({"command": "shutdown"}));
        let _ = self.stdin.flush();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for ServeJson {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// One-shot cold `run` (new process, full model load). Parses latency lines from stdout.
pub fn cold_run(root: &Path, text: &str, use_mtp: bool) -> Result<ColdRunResult> {
    let (helper, model, drafter) = fluid1_paths().context("fluid-1 not installed")?;
    let prompt = polish_prompt(root);

    let mut cmd = Command::new(&helper);
    cmd.arg("run")
        .arg("--model-dir")
        .arg(&model)
        .arg("--text")
        .arg(text)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if use_mtp {
        if let Some(d) = drafter {
            cmd.arg("--mtp-drafter-dir")
                .arg(d)
                .arg("--draft-block-size")
                .arg("6");
        }
    }
    if prompt.is_file() {
        cmd.arg("--system-prompt-file").arg(&prompt);
    }

    let t0 = Instant::now();
    let out = cmd.output().context("cold fluid-intelligence-mlx run")?;
    let wall = t0.elapsed();
    if !out.status.success() {
        bail!(
            "cold run failed: {}",
            String::from_utf8_lossy(&out.stderr)
                .chars()
                .take(300)
                .collect::<String>()
        );
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut latency_ms = None;
    let mut ttft_ms = None;
    let mut tok_s = None;
    let mut gen_tokens = None;
    let mut output_lines = Vec::new();
    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("mtp_") {
            continue;
        }
        if let Some(v) = line.strip_prefix("latency_ms=") {
            latency_ms = v.parse().ok();
        } else if let Some(v) = line.strip_prefix("ttft_ms=") {
            ttft_ms = v.parse().ok();
        } else if let Some(v) = line.strip_prefix("tokens_per_second=") {
            tok_s = v.parse().ok();
        } else if let Some(v) = line.strip_prefix("generated_tokens=") {
            gen_tokens = v.parse().ok();
        } else if !line.contains('=') {
            output_lines.push(line.to_string());
        }
    }
    Ok(ColdRunResult {
        wall_ms: wall.as_secs_f64() * 1000.0,
        latency_ms,
        ttft_ms,
        tokens_per_second: tok_s,
        generated_tokens: gen_tokens,
        output: output_lines.last().cloned().unwrap_or_default(),
    })
}

#[derive(Debug, Clone)]
pub struct ColdRunResult {
    pub wall_ms: f64,
    pub latency_ms: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub tokens_per_second: Option<f64>,
    pub generated_tokens: Option<u64>,
    pub output: String,
}

/// Derived timing fields from a warm run response.
#[derive(Debug, Clone)]
pub struct RunTiming {
    pub wall_ms: f64,
    #[allow(dead_code)]
    pub latency_ms: f64,
    pub ttft_ms: f64,
    pub decode_ms: f64,
    pub itl_ms: f64,
    pub tokens: u64,
    pub tok_s: f64,
    #[allow(dead_code)]
    pub output: String,
    pub stage_timings: Option<serde_json::Value>,
}

pub fn timing_from_resp(wall: Duration, resp: &ServeJsonResp) -> Result<RunTiming> {
    if !resp.ok {
        bail!(
            "polish failed: {}",
            resp.error.as_deref().unwrap_or("unknown")
        );
    }
    let d = resp.diagnostics.as_ref();
    let latency_ms = d
        .and_then(|x| x.latency_ms)
        .unwrap_or(wall.as_secs_f64() * 1000.0);
    let ttft_ms = d.and_then(|x| x.ttft_ms).unwrap_or(0.0);
    let tokens = d.and_then(|x| x.generated_tokens).unwrap_or(0);
    let tok_s = d.and_then(|x| x.tokens_per_second).unwrap_or(0.0);
    let decode_ms = (latency_ms - ttft_ms).max(0.0);
    let itl_ms = if tokens > 1 {
        decode_ms / (tokens as f64 - 1.0)
    } else {
        0.0
    };
    Ok(RunTiming {
        wall_ms: wall.as_secs_f64() * 1000.0,
        latency_ms,
        ttft_ms,
        decode_ms,
        itl_ms,
        tokens,
        tok_s,
        output: resp.output_text.as_deref().unwrap_or("").trim().to_string(),
        stage_timings: d.and_then(|x| x.stage_timings.clone()),
    })
}

#[derive(Debug, Clone)]
pub enum PolishEvent {
    /// A startup stage the sidecar reports before `Ready`: download bytes,
    /// engine start, MLX fallback. Never a readiness signal.
    Loading {
        msg: String,
        progress: Option<f64>,
    },
    Ready {
        msg: String,
        load_ms: f64,
    },
    Result {
        text: String,
        latency_ms: f64,
        ttft_ms: f64,
        tok_s: f64,
    },
    /// Live text: the corrected text so far of the dictation `id` (a background refine of
    /// the transcript so far). `complete` is false when it stopped at the output budget.
    /// Display only: never paired with a polish request.
    Draft {
        id: String,
        seq: Option<usize>,
        text: String,
        complete: bool,
    },
    Error {
        msg: String,
    },
}

pub struct PolishSidecar {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<PolishEvent>,
    waker: Waker,
}

/// The sidecar's `--model` and `--revision` after PHONON_POLISH_MODEL / PHONON_POLISH_REVISION.
/// A hub model needs an exact 40-hex commit (the sidecar also needs its hashes in
/// sidecar/model_pins.json); a branch or a hub override without a revision fails here instead of
/// downloading whatever the branch points at. A local directory needs no revision.
fn polish_model_override(
    default: (String, String),
    model: Option<String>,
    revision: Option<String>,
) -> Result<(String, String)> {
    if model.is_none() && revision.is_none() {
        return Ok(default);
    }
    let model_overridden = model.is_some();
    let model = model.unwrap_or(default.0);
    if Path::new(&model).is_dir() {
        return Ok((model, revision.unwrap_or(default.1)));
    }
    let revision = match revision {
        Some(revision) => revision,
        None if model_overridden => {
            bail!("PHONON_POLISH_MODEL={model} needs PHONON_POLISH_REVISION set to a 40-hex commit, or a local folder")
        }
        None => default.1,
    };
    if !is_commit(&revision) {
        bail!("polish model {model}: revision {revision:?} is not a 40-hex commit");
    }
    Ok((model, revision))
}

fn is_commit(revision: &str) -> bool {
    revision.len() == 40
        && revision
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// The shipped correction stage. `Ok(None)` means its own files are absent.
/// `PHONON_POLISH_BACKEND=fluid` swaps in the old helper on identical inputs;
/// it is a developer comparison switch and is never set in a shipped build.
fn polish_command(root: &Path, config: &PolishConfig) -> Result<Option<Command>> {
    let script = polish_script(root);
    // Overridable for developer model comparisons only; unset in shipped builds.
    let prompt = std::env::var("PHONON_POLISH_PROMPT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| polish_prompt(root));
    if !(script.is_file() && prompt.is_file()) {
        return Ok(None);
    }
    if std::env::var("PHONON_POLISH_BACKEND").as_deref() == Ok("fluid") {
        return fluid_command(root, true).map(Some);
    }
    let uv = phonon_asr::resolve_uv().context("uv not found; install it with Homebrew")?;
    // Overridable for developer model comparisons only; unset in shipped builds.
    // A local directory has no revision; the sidecar uses it as it is.
    let (model, revision) = polish_model_override(
        config.model_and_revision(),
        std::env::var("PHONON_POLISH_MODEL").ok(),
        std::env::var("PHONON_POLISH_REVISION").ok(),
    )?;
    let mut command = Command::new(&uv);
    command
        .args(phonon_asr::sidecar_run_args(&script)?)
        .args(["--model", &model, "--revision", &revision])
        .arg("--system-prompt-file")
        .arg(&prompt)
        .env(
            "PHONON_STREAM_REFINE",
            if config.stream_refine { "1" } else { "0" },
        )
        .current_dir(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(profile_dir) = &config.profile_dir {
        command.arg("--profile-dir").arg(profile_dir);
    }
    if let Some(budget) = config.profile_token_budget {
        command
            .arg("--profile-token-budget")
            .arg(budget.to_string());
    }
    if !config.prefix_cache {
        command.arg("--no-prefix-cache");
    }
    if let Some((binary, shaders)) = bundled_engine(root) {
        command
            .arg("--engine-binary")
            .arg(binary)
            .arg("--engine-shaders")
            .arg(shaders);
    }
    // Developer comparison switch for correctors trained on the bare transcript
    // (no dictionary envelope); never set in a shipped build.
    if std::env::var("PHONON_POLISH_TRANSCRIPT_ONLY").as_deref() == Ok("1") {
        command.arg("--transcript-only");
    }
    phonon_asr::apply_offline_policy(&mut command, &uv, &script);
    Ok(Some(command))
}

/// fluid-1 helper, for developer comparison only. See `PHONON_POLISH_BACKEND`.
fn fluid_command(root: &Path, use_mtp: bool) -> Result<Command> {
    let (helper, model, drafter) =
        fluid1_paths().context("fluid-1 baseline requested but not installed")?;
    let mut command = Command::new(helper);
    command
        .arg("serve-json")
        .arg("--model-dir")
        .arg(model)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    if use_mtp {
        if let Some(drafter) = drafter {
            command
                .arg("--mtp-drafter-dir")
                .arg(drafter)
                .arg("--draft-block-size")
                .arg("6");
        }
    }
    command.arg("--system-prompt-file").arg(polish_prompt(root));
    Ok(command)
}

impl PolishSidecar {
    pub fn spawn(root: &Path) -> Result<Option<Self>> {
        Self::spawn_with(root, &PolishConfig::from_env())
    }

    pub fn spawn_with(root: &Path, config: &PolishConfig) -> Result<Option<Self>> {
        let Some(mut command) = polish_command(root, config)? else {
            return Ok(None);
        };
        Self::attach(command.spawn().context("spawn polish")?).map(Some)
    }

    fn attach(mut child: Child) -> Result<Self> {
        let stdout = child.stdout.take().context("polish stdout")?;
        let mut stdin = child.stdin.take().context("polish stdin")?;
        let stderr = child
            .stderr
            .take()
            .map(|stderr| StderrTail::capture(stderr, "polish"));
        let started = Instant::now();
        writeln!(stdin, "{}", json!({"command":"warmup"}))?;
        stdin.flush()?;
        let (tx, rx) = mpsc::channel();
        let waker = Waker::default();
        let thread_waker = waker.clone();
        thread::spawn(move || {
            let mut ready = false;
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                let Ok(message) = serde_json::from_str::<ServeJsonResp>(&line) else {
                    continue;
                };
                let event = if !message.ok {
                    PolishEvent::Error {
                        msg: message.error.unwrap_or_else(|| "polish error".into()),
                    }
                } else if let Some(draft) = message.draft() {
                    draft
                } else if let Some(text) = message.output_text {
                    let diagnostics = message.diagnostics.as_ref();
                    PolishEvent::Result {
                        text: text.trim().to_string(),
                        latency_ms: diagnostics.and_then(|x| x.latency_ms).unwrap_or(0.0),
                        ttft_ms: diagnostics.and_then(|x| x.ttft_ms).unwrap_or(0.0),
                        tok_s: diagnostics.and_then(|x| x.tokens_per_second).unwrap_or(0.0),
                    }
                } else if message.status.as_ref().and_then(|x| x.state.as_deref()) == Some("ready")
                {
                    ready = true;
                    PolishEvent::Ready {
                        msg: message.status.and_then(|x| x.message).unwrap_or_default(),
                        load_ms: started.elapsed().as_secs_f64() * 1000.0,
                    }
                } else if let Some(status) = message
                    .status
                    .filter(|x| !ready && x.state.as_deref() == Some("loading"))
                {
                    PolishEvent::Loading {
                        msg: status.message.unwrap_or_default(),
                        progress: status.progress,
                    }
                } else {
                    continue;
                };
                if tx.send(event).is_err() {
                    return;
                }
                thread_waker.wake();
            }
            // stdout closed. If that happened before `ready`, the process died
            // during startup and nobody would ever hear about it otherwise.
            if !ready {
                let msg = stderr
                    .as_ref()
                    .map(|tail| tail.exit_message("correction sidecar"))
                    .unwrap_or_else(|| "correction sidecar exited before becoming ready".into());
                let _ = tx.send(PolishEvent::Error { msg });
                thread_waker.wake();
            }
        });
        Ok(Self {
            child,
            stdin,
            rx,
            waker,
        })
    }

    /// Nudge `sender` (an empty line) whenever an event is queued.
    pub fn set_waker(&self, sender: std::sync::mpsc::Sender<String>) {
        self.waker.set(sender);
    }

    pub fn poll(&self) -> Vec<PolishEvent> {
        let mut events = Vec::new();
        while let Ok(event) = self.rx.try_recv() {
            events.push(event);
        }
        events
    }

    pub fn polish(&mut self, text: &str) -> Result<()> {
        self.polish_with_context(text, None)
    }

    /// `context` (app category, site, approved style preferences, text-box
    /// text) rides along for a correction model that reads it; the sidecar
    /// ignores keys it does not use.
    pub fn polish_with_context(
        &mut self,
        text: &str,
        context: Option<&serde_json::Value>,
    ) -> Result<()> {
        writeln!(self.stdin, "{}", run_request(text, context))?;
        self.stdin.flush()?;
        Ok(())
    }

    /// Dictation started: let the sidecar wake its model while the user is
    /// still talking. The reply is a status line the reader thread skips, so
    /// it never pairs with a queued polish request.
    pub fn prewarm(&mut self) -> Result<()> {
        writeln!(self.stdin, "{}", json!({"command":"prewarm"}))?;
        self.stdin.flush()?;
        Ok(())
    }

    /// The transcript so far of a dictation still in progress (same envelope as `polish`).
    /// A refiner sidecar refines it in the background so the `polish` at key release has
    /// its prompt cached and a draft to verify; the answer is not a correction result.
    /// `context` is the dictation's app context, so the background prompt carries the
    /// same writing mode as the `polish` that follows.
    pub fn stream_commit(&mut self, text: &str, context: Option<&serde_json::Value>) -> Result<()> {
        writeln!(self.stdin, "{}", stream_commit_request(text, context))?;
        self.stdin.flush()?;
        Ok(())
    }

    /// Forget any background refine state (a new dictation starts).
    pub fn stream_reset(&mut self) -> Result<()> {
        writeln!(self.stdin, "{}", json!({"command":"stream_reset"}))?;
        self.stdin.flush()?;
        Ok(())
    }

    pub fn shutdown(&mut self) {
        let _ = writeln!(self.stdin, "{}", json!({"command":"shutdown"}));
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for PolishSidecar {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

/// One `run` line for the serve-json sidecar. The `context` key is present
/// only when there is context, so a helper that predates it sees the old
/// request unchanged.
pub fn run_request(text: &str, context: Option<&serde_json::Value>) -> serde_json::Value {
    let mut request = json!({"command": "run", "inputText": text});
    if let Some(context) = context.filter(|value| !value.is_null()) {
        request["context"] = context.clone();
    }
    request
}

/// One `stream_commit` line: a `run` line by another name, context included.
pub fn stream_commit_request(text: &str, context: Option<&serde_json::Value>) -> serde_json::Value {
    let mut request = run_request(text, context);
    request["command"] = json!("stream_commit");
    request
}

#[cfg(test)]
mod run_request_tests {
    use super::{run_request, stream_commit_request};

    #[test]
    fn a_stream_commit_carries_the_same_context_as_its_run() {
        let context = json!({"app_category": "ai_prompt"});
        assert_eq!(
            stream_commit_request("hi", Some(&context)),
            json!({"command": "stream_commit", "inputText": "hi", "context": context})
        );
        assert_eq!(
            stream_commit_request("hi", None),
            json!({"command": "stream_commit", "inputText": "hi"})
        );
    }
    use serde_json::json;

    #[test]
    fn only_a_live_text_commit_reply_is_a_draft() {
        let parse = |line: &str| serde_json::from_str::<super::ServeJsonResp>(line).unwrap();
        // the default path: a commit reply without draft keys stays silent
        let plain = parse(r#"{"ok":true,"stream":{"commits":2,"aborted":false}}"#);
        assert!(plain.draft().is_none());
        assert!(parse(r#"{"ok":true,"outputText":"Hi."}"#).draft().is_none());
        let draft = parse(
            r#"{"ok":true,"stream":{"commits":3,"draftId":"p4","draftSeq":2,"draftText":" So far. ","draftComplete":false}}"#,
        );
        match draft.draft() {
            Some(super::PolishEvent::Draft {
                id,
                seq,
                text,
                complete,
            }) => {
                assert_eq!(
                    (id.as_str(), seq, text.as_str()),
                    ("p4", Some(2), "So far.")
                );
                assert!(!complete);
            }
            other => panic!("expected a draft, got {other:?}"),
        }
    }

    #[test]
    fn context_is_added_only_when_present() {
        assert_eq!(
            run_request("hi", None),
            json!({"command": "run", "inputText": "hi"})
        );
        let context = json!({"app_category": "chat"});
        let request = run_request("hi", Some(&context));
        assert_eq!(request["context"]["app_category"], "chat");
        assert_eq!(request["inputText"], "hi");
        assert!(run_request("hi", Some(&serde_json::Value::Null))
            .get("context")
            .is_none());
    }
}

#[cfg(test)]
mod model_precedence_tests {
    use super::{polish_model_override, PolishConfig, POLISH_MODEL_ID, POLISH_MODEL_REVISION};
    use std::path::PathBuf;

    #[test]
    fn a_configured_user_refiner_wins_over_the_public_download() {
        let personal =
            PathBuf::from("/Users/someone/Library/Application Support/Phonon/models/refiner");
        let config = PolishConfig {
            model_dir: Some(personal.clone()),
            ..PolishConfig::default()
        };
        assert_eq!(
            config.model_and_revision(),
            (personal.display().to_string(), "local".to_string())
        );
        assert_eq!(
            PolishConfig::default().model_and_revision(),
            (
                POLISH_MODEL_ID.to_string(),
                POLISH_MODEL_REVISION.to_string()
            )
        );
    }

    fn pinned() -> (String, String) {
        (
            POLISH_MODEL_ID.to_string(),
            POLISH_MODEL_REVISION.to_string(),
        )
    }

    #[test]
    fn overrides_need_a_commit_or_a_local_folder() {
        let gemma = "mlx-community/gemma-4-e2b-it-4bit".to_string();
        let commit = "238767527555cb75a05732a84dff5d6ba0dd6809".to_string();
        assert_eq!(
            polish_model_override(pinned(), None, None).unwrap(),
            pinned()
        );
        assert_eq!(
            polish_model_override(pinned(), Some(gemma.clone()), Some(commit.clone())).unwrap(),
            (gemma.clone(), commit)
        );
        // a hub override without a revision would otherwise inherit the refiner's pin, or a branch
        assert!(polish_model_override(pinned(), Some(gemma.clone()), None).is_err());
        assert!(polish_model_override(pinned(), Some(gemma), Some("main".into())).is_err());
        assert!(polish_model_override(pinned(), None, Some("main".into())).is_err());
        let dir = std::env::temp_dir();
        let local = dir.display().to_string();
        assert_eq!(
            polish_model_override(pinned(), Some(local.clone()), None).unwrap(),
            (local, POLISH_MODEL_REVISION.to_string())
        );
    }

    /// sidecar/model_pins.json holds the hashes the sidecar checks; it must pin the same commit.
    #[test]
    fn the_sidecar_pin_file_agrees_with_the_pinned_revision() {
        let pins: serde_json::Value =
            serde_json::from_str(include_str!("../../../sidecar/model_pins.json")).unwrap();
        assert_eq!(
            pins["models"][POLISH_MODEL_ID]["revision"],
            POLISH_MODEL_REVISION
        );
        assert!(pins["models"][POLISH_MODEL_ID]["files"]
            .as_object()
            .is_some_and(|files| !files.is_empty()));
    }
}
