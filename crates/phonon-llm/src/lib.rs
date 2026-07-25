//! Transcript correction over a pinned local MLX model (`sidecar/polish_server.py`).
//!
//! Protocol:
//!   stdin JSONL:
//!     {"command":"warmup"}
//!     {"command":"run","inputText":"…"}
//!     {"command":"status"}
//!     {"command":"shutdown"}
//!   stdout JSONL responses with diagnostics.
//!
//! `ServeJson` is the synchronous handle over the same protocol, used by
//! evaluation and benchmark tooling. `ServeJson::spawn_fluid` and `cold_run`
//! reach the old FluidVoice helper instead; they exist only so `phonon bench` /
//! `phonon profile` can compare against that baseline locally.

mod paths;

use anyhow::{bail, Context, Result};
pub use paths::{
    fluid1_available, fluid1_paths, fluid_drafter_dir, fluid_helper, fluid_model_dir,
    polish_available, POLISH_MODEL_ID, POLISH_MODEL_REVISION, POLISH_RUNTIME_REQUIREMENT,
};
use paths::{polish_prompt, polish_script};
use phonon_asr::StderrTail;
use serde::Deserialize;
use serde_json::json;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
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
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServeStatus {
    #[allow(dead_code)]
    pub state: Option<String>,
    pub message: Option<String>,
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
        let cmd = polish_command(root)?.context("correction stage files missing")?;
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
        let stderr = child.stderr.take().map(StderrTail::capture);
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
    Error {
        msg: String,
    },
}

pub struct PolishSidecar {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<PolishEvent>,
}

/// The shipped correction stage. `Ok(None)` means its own files are absent.
/// `PHONON_POLISH_BACKEND=fluid` swaps in the old helper on identical inputs;
/// it is a developer comparison switch and is never set in a shipped build.
fn polish_command(root: &Path) -> Result<Option<Command>> {
    let script = polish_script(root);
    let prompt = polish_prompt(root);
    if !(script.is_file() && prompt.is_file()) {
        return Ok(None);
    }
    if std::env::var("PHONON_POLISH_BACKEND").as_deref() == Ok("fluid") {
        return fluid_command(root, true).map(Some);
    }
    let uv = phonon_asr::resolve_uv().context("uv not found; install it with Homebrew")?;
    // Overridable for developer model comparisons only; unset in shipped builds.
    let model =
        std::env::var("PHONON_POLISH_MODEL").unwrap_or_else(|_| POLISH_MODEL_ID.to_string());
    let revision = std::env::var("PHONON_POLISH_REVISION")
        .unwrap_or_else(|_| POLISH_MODEL_REVISION.to_string());
    let mut command = Command::new(uv);
    command
        .args([
            "run",
            "--python",
            phonon_asr::PYTHON_REQUIREMENT,
            "--with",
            POLISH_RUNTIME_REQUIREMENT,
            "python",
        ])
        .arg(&script)
        .args(["--model", &model, "--revision", &revision])
        .arg("--system-prompt-file")
        .arg(&prompt)
        .current_dir(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
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
        let Some(mut command) = polish_command(root)? else {
            return Ok(None);
        };
        Self::attach(command.spawn().context("spawn polish")?).map(Some)
    }

    fn attach(mut child: Child) -> Result<Self> {
        let stdout = child.stdout.take().context("polish stdout")?;
        let mut stdin = child.stdin.take().context("polish stdin")?;
        let stderr = child.stderr.take().map(StderrTail::capture);
        let started = Instant::now();
        writeln!(stdin, "{}", json!({"command":"warmup"}))?;
        stdin.flush()?;
        let (tx, rx) = mpsc::channel();
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
                } else {
                    continue;
                };
                if tx.send(event).is_err() {
                    return;
                }
            }
            // stdout closed. If that happened before `ready`, the process died
            // during startup and nobody would ever hear about it otherwise.
            if !ready {
                let msg = stderr
                    .as_ref()
                    .map(|tail| tail.exit_message("correction sidecar"))
                    .unwrap_or_else(|| "correction sidecar exited before becoming ready".into());
                let _ = tx.send(PolishEvent::Error { msg });
            }
        });
        Ok(Self { child, stdin, rx })
    }

    pub fn poll(&self) -> Vec<PolishEvent> {
        let mut events = Vec::new();
        while let Ok(event) = self.rx.try_recv() {
            events.push(event);
        }
        events
    }

    pub fn polish(&mut self, text: &str) -> Result<()> {
        writeln!(self.stdin, "{}", json!({"command":"run", "inputText":text}))?;
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
