use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::json;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

/// Where a sidecar reader thread nudges its owner after queueing an event, so an event loop can
/// block instead of sleeping between polls. The nudge is an empty line on the owner's command
/// channel, which the engine loop already skips.
#[derive(Clone, Default)]
pub struct Waker(Arc<Mutex<Option<Sender<String>>>>);

impl Waker {
    pub fn set(&self, sender: Sender<String>) {
        if let Ok(mut slot) = self.0.lock() {
            *slot = Some(sender);
        }
    }

    pub fn wake(&self) {
        if let Ok(slot) = self.0.lock() {
            if let Some(sender) = slot.as_ref() {
                let _ = sender.send(String::new());
            }
        }
    }
}

/// Pinned ASR model and runtime. Exact revisions so an upstream release cannot
/// change what shipped users run.
pub const ASR_MODEL_ID: &str = "mlx-community/parakeet-tdt-0.6b-v2";
pub const ASR_MODEL_REVISION: &str = "8ae155301e23d820d82aa60d24817c900e69e487";
pub const ASR_RUNTIME_REQUIREMENT: &str = "parakeet-mlx==0.5.2";

/// Pinned interpreter for both sidecars. Without this, uv falls back to whatever
/// `python3` it finds, which on a Mac with no developer tooling is the system
/// CPython 3.9 that MLX publishes no wheels for. An open-ended range would also
/// drift onto each new release before MLX supports it.
pub const PYTHON_REQUIREMENT: &str = "3.12";

/// Last lines a dying sidecar wrote, so its exit can be reported instead of
/// hanging the loader at whatever percentage it reached.
pub struct StderrTail(std::sync::Arc<std::sync::Mutex<Vec<String>>>);

static STDERR_PASS_THROUGH: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Copy every sidecar stderr line to this process's stderr (see `StderrTail::capture`).
pub fn pass_stderr_through(on: bool) {
    STDERR_PASS_THROUGH.store(on, std::sync::atomic::Ordering::Relaxed);
}

impl StderrTail {
    const KEEP: usize = 12;

    /// Drains `stderr` on a background thread, keeping only the tail. With
    /// `pass_stderr_through(true)` (the engine's serve mode) every line is also written to this
    /// process's own stderr as `[label] line`, which the bar appends to
    /// ~/Library/Logs/Phonon/engine.log, so a sidecar's last words outlive it.
    pub fn capture(stderr: std::process::ChildStderr, label: &'static str) -> Self {
        let lines = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = std::sync::Arc::clone(&lines);
        thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if STDERR_PASS_THROUGH.load(std::sync::atomic::Ordering::Relaxed) {
                    eprintln!("[{label}] {line}");
                }
                let Ok(mut kept) = sink.lock() else { return };
                if kept.len() == Self::KEEP {
                    kept.remove(0);
                }
                kept.push(line);
            }
        });
        Self(lines)
    }

    pub fn text(&self) -> String {
        self.0
            .lock()
            .map(|lines| lines.join("\n"))
            .unwrap_or_default()
    }

    /// Message for a sidecar whose stdout closed before it was ready.
    pub fn exit_message(&self, what: &str) -> String {
        let tail = self.text();
        let tail = tail.trim();
        if tail.is_empty() {
            format!("{what} exited before becoming ready")
        } else {
            format!("{what} exited before becoming ready: {tail}")
        }
    }
}

#[derive(Debug, Deserialize)]
struct SidecarMsg {
    #[serde(rename = "type")]
    kind: String,
    pct: Option<f64>,
    msg: Option<String>,
    model: Option<String>,
    text: Option<String>,
    seconds: Option<f64>,
    id: Option<String>,
    #[serde(default)]
    partial: bool,
    index: Option<usize>,
}

#[derive(Debug, Clone)]
pub enum AsrEvent {
    Status {
        pct: f64,
        msg: String,
    },
    Ready {
        model: String,
        load_ms: f64,
    },
    Result {
        id: Option<String>,
        text: String,
        seconds: f64,
        partial: bool,
    },
    /// A speech segment closed by a pause during a segment stream, transcribed on its own.
    Segment {
        id: Option<String>,
        index: usize,
        text: String,
    },
    Error {
        /// The request the error belongs to, when the sidecar names one.
        id: Option<String>,
        msg: String,
    },
}

pub struct AsrSidecar {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<AsrEvent>,
    waker: Waker,
}

impl AsrSidecar {
    pub fn spawn(root: &Path) -> Result<Self> {
        Self::spawn_with(root, false)
    }

    /// `stream_refine`: segment streaming's ASR side on (SPEC "Streaming refine"); the
    /// sidecar reads it from PHONON_STREAM_REFINE.
    pub fn spawn_with(root: &Path, stream_refine: bool) -> Result<Self> {
        let script = root.join("sidecar/asr_server.py");
        if !script.is_file() {
            bail!("missing {}", script.display());
        }
        let uv = resolve_uv().context("uv not found; install it with Homebrew")?;
        let started = Instant::now();
        let mut command = Command::new(&uv);
        command
            .args([
                "run",
                "--python",
                PYTHON_REQUIREMENT,
                "--with",
                ASR_RUNTIME_REQUIREMENT,
                "python",
            ])
            .arg(&script)
            .args(["--model", ASR_MODEL_ID, "--revision", ASR_MODEL_REVISION])
            .env(
                "PHONON_STREAM_REFINE",
                if stream_refine { "1" } else { "0" },
            )
            .current_dir(root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        apply_offline_policy(&mut command, &uv, ASR_RUNTIME_REQUIREMENT);
        let mut child = command.spawn().context("spawn ASR")?;
        let stdout = child.stdout.take().context("asr stdout")?;
        let stdin = child.stdin.take().context("asr stdin")?;
        let stderr = StderrTail::capture(child.stderr.take().context("asr stderr")?, "asr");
        let (tx, rx) = mpsc::channel();
        let waker = Waker::default();
        let thread_waker = waker.clone();
        thread::spawn(move || {
            let mut ready = false;
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                let Ok(msg) = serde_json::from_str::<SidecarMsg>(&line) else {
                    continue;
                };
                let event = match msg.kind.as_str() {
                    "status" => AsrEvent::Status {
                        pct: msg.pct.unwrap_or(0.0),
                        msg: msg.msg.unwrap_or_default(),
                    },
                    "ready" => {
                        ready = true;
                        AsrEvent::Ready {
                            model: msg.model.unwrap_or_else(|| "parakeet".into()),
                            load_ms: started.elapsed().as_secs_f64() * 1000.0,
                        }
                    }
                    "result" => AsrEvent::Result {
                        id: msg.id,
                        text: msg.text.unwrap_or_default(),
                        seconds: msg.seconds.unwrap_or(0.0),
                        partial: msg.partial,
                    },
                    "segment" => AsrEvent::Segment {
                        id: msg.id,
                        index: msg.index.unwrap_or(0),
                        text: msg.text.unwrap_or_default(),
                    },
                    "error" => AsrEvent::Error {
                        id: msg.id,
                        msg: msg.msg.unwrap_or_else(|| "asr error".into()),
                    },
                    _ => continue,
                };
                if tx.send(event).is_err() {
                    return;
                }
                thread_waker.wake();
            }
            // stdout closed. If that happened before `ready`, the process died
            // during startup and nobody would ever hear about it otherwise.
            if !ready {
                let _ = tx.send(AsrEvent::Error {
                    id: None,
                    msg: stderr.exit_message("ASR sidecar"),
                });
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
    pub fn set_waker(&self, sender: Sender<String>) {
        self.waker.set(sender);
    }

    pub fn poll(&self) -> Vec<AsrEvent> {
        let mut events = Vec::new();
        while let Ok(event) = self.rx.try_recv() {
            events.push(event);
        }
        events
    }

    pub fn send(&mut self, value: serde_json::Value) -> Result<()> {
        writeln!(self.stdin, "{value}")?;
        self.stdin.flush()?;
        Ok(())
    }

    pub fn transcribe(&mut self, path: &Path, id: Option<&str>) -> Result<()> {
        self.send(json!({"cmd":"transcribe", "path":path, "id":id}))
    }

    pub fn stream_start(&mut self, id: Option<&str>) -> Result<()> {
        self.send(json!({"cmd":"stream_start", "id":id}))
    }

    pub fn stream_chunk(&mut self, pcm16: &str, id: Option<&str>) -> Result<()> {
        self.send(json!({"cmd":"stream_chunk", "pcm16":pcm16, "id":id}))
    }

    pub fn stream_stop(&mut self, id: Option<&str>) -> Result<()> {
        self.send(json!({"cmd":"stream_stop", "id":id}))
    }

    /// Start a segment stream: chunks are cut at pauses and each closed segment is
    /// transcribed while the speaker keeps talking (AsrEvent::Segment).
    pub fn stream_start_segments(&mut self, id: Option<&str>) -> Result<()> {
        self.send(json!({"cmd":"stream_start", "id":id, "segments": true}))
    }

    /// Finish a segment stream at key release: the sidecar appends the part of `path` it has
    /// not received (none when `path` is None: the stream holds all the audio), transcribes
    /// the open segment, and answers with the whole transcript as a final AsrEvent::Result.
    pub fn stream_finish(&mut self, path: Option<&Path>, id: Option<&str>) -> Result<()> {
        self.send(json!({"cmd":"stream_finish", "path":path, "id":id}))
    }

    pub fn warmup_stream(&mut self, path: &Path, id: Option<&str>) -> Result<()> {
        self.send(json!({"cmd":"warmup_stream", "path":path, "id":id}))
    }

    pub fn shutdown(&mut self) {
        let _ = self.send(json!({"cmd":"shutdown"}));
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub fn resolve_uv() -> Option<std::path::PathBuf> {
    let bundled = std::env::current_exe().ok().and_then(|executable| {
        executable
            .parent()
            .map(|directory| directory.join("uv"))
            .filter(|path| path.is_file())
    });
    bundled.or_else(|| which::which("uv").ok()).or_else(|| {
        ["/opt/homebrew/bin/uv", "/usr/local/bin/uv"]
            .into_iter()
            .map(std::path::PathBuf::from)
            .find(|path| path.is_file())
    })
}

/// True when uv can assemble the sidecar environment for `requirement` from
/// its cache alone. Without this, uv revalidates its PyPI index entry once the
/// entry goes stale, and with no network that fails after three retries
/// (about 33 s) even though every wheel is already on disk. Costs 50-150 ms.
pub fn uv_offline_ready(uv: &Path, requirement: &str) -> bool {
    Command::new(uv)
        .args([
            "run",
            "--offline",
            "--python",
            PYTHON_REQUIREMENT,
            "--with",
            requirement,
            "python",
            "-c",
            "",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// Pin a sidecar launch to uv's cache when the cache is enough, so a Mac with
/// no network or a captive portal starts dictation instead of waiting on
/// PyPI. A first launch with nothing cached still resolves online.
pub fn apply_offline_policy(command: &mut Command, uv: &Path, requirement: &str) {
    if uv_offline_ready(uv, requirement) {
        command.env("UV_OFFLINE", "1");
    }
}

impl Drop for AsrSidecar {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

#[cfg(test)]
mod tests {
    use super::Waker;
    use std::sync::mpsc;

    #[test]
    fn waker_nudges_with_an_empty_line_once_set() {
        let waker = Waker::default();
        waker.wake(); // nothing set yet: no panic, nothing sent
        let (tx, rx) = mpsc::channel();
        waker.clone().set(tx);
        waker.wake();
        assert_eq!(rx.try_recv().as_deref(), Ok(""));
        assert!(rx.try_recv().is_err());
    }
}
