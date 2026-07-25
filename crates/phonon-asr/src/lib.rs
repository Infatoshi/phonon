use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::json;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Instant;

/// Pinned ASR model and runtime. Exact revisions so an upstream release cannot
/// change what shipped users run.
pub const ASR_MODEL_ID: &str = "mlx-community/parakeet-tdt-0.6b-v2";
pub const ASR_MODEL_REVISION: &str = "8ae155301e23d820d82aa60d24817c900e69e487";
pub const ASR_RUNTIME_REQUIREMENT: &str = "parakeet-mlx==0.5.2";

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
    Error {
        msg: String,
    },
}

pub struct AsrSidecar {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<AsrEvent>,
}

impl AsrSidecar {
    pub fn spawn(root: &Path) -> Result<Self> {
        let script = root.join("sidecar/asr_server.py");
        if !script.is_file() {
            bail!("missing {}", script.display());
        }
        let uv = resolve_uv().context("uv not found; install it with Homebrew")?;
        let started = Instant::now();
        let mut child = Command::new(uv)
            .args(["run", "--with", ASR_RUNTIME_REQUIREMENT, "python"])
            .arg(&script)
            .args(["--model", ASR_MODEL_ID, "--revision", ASR_MODEL_REVISION])
            .current_dir(root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("spawn ASR")?;
        let stdout = child.stdout.take().context("asr stdout")?;
        let stdin = child.stdin.take().context("asr stdin")?;
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                let Ok(msg) = serde_json::from_str::<SidecarMsg>(&line) else {
                    continue;
                };
                let event = match msg.kind.as_str() {
                    "status" => AsrEvent::Status {
                        pct: msg.pct.unwrap_or(0.0),
                        msg: msg.msg.unwrap_or_default(),
                    },
                    "ready" => AsrEvent::Ready {
                        model: msg.model.unwrap_or_else(|| "parakeet".into()),
                        load_ms: started.elapsed().as_secs_f64() * 1000.0,
                    },
                    "result" => AsrEvent::Result {
                        id: msg.id,
                        text: msg.text.unwrap_or_default(),
                        seconds: msg.seconds.unwrap_or(0.0),
                        partial: msg.partial,
                    },
                    "error" => AsrEvent::Error {
                        msg: msg.msg.unwrap_or_else(|| "asr error".into()),
                    },
                    _ => continue,
                };
                if tx.send(event).is_err() {
                    break;
                }
            }
        });
        Ok(Self { child, stdin, rx })
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

impl Drop for AsrSidecar {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}
