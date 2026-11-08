//! Bring your own model endpoint (GitHub issue #24): an OpenAI-compatible
//! `/chat/completions` endpoint the user names in Settings refines the transcript
//! instead of the local refiner. Off unless configured; there is no default
//! endpoint and no Phonon-hosted service. The local refiner still runs every time
//! and its result is used whenever the endpoint errors, times out or returns
//! something the output guard rejects. ASR stays local.
//!
//! The request goes through the system `curl` (macOS, Windows 10+, Linux) with its
//! configuration on stdin, so the API key is never in a process argument list or a
//! file, and no TLS stack is added to the build.

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};

/// How long the final correction waits for the endpoint past the local result.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(6);

#[derive(Clone, PartialEq, Eq)]
pub struct Endpoint {
    pub base_url: String,
    pub model: String,
    pub api_key: String,
    pub timeout: Duration,
}

impl std::fmt::Debug for Endpoint {
    // never print the key
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Endpoint")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("api_key", &format_args!("<{} chars>", self.api_key.len()))
            .field("timeout", &self.timeout)
            .finish()
    }
}

impl Endpoint {
    /// From the engine's `{"cmd":"byok", "base_url", "model", "api_key", "timeout_ms"}`;
    /// None (off) without a base URL and model. The key may be empty (local servers).
    pub fn from_command(v: &Value) -> Option<Self> {
        let text = |key: &str| {
            v.get(key)
                .and_then(Value::as_str)
                .map(str::trim)
                .unwrap_or("")
                .to_string()
        };
        let (base_url, model) = (text("base_url"), text("model"));
        if base_url.is_empty() || model.is_empty() {
            return None;
        }
        if !(base_url.starts_with("https://") || base_url.starts_with("http://")) {
            return None;
        }
        let timeout = v
            .get("timeout_ms")
            .and_then(Value::as_u64)
            .filter(|ms| (500..=60_000).contains(ms))
            .map(Duration::from_millis)
            .unwrap_or(DEFAULT_TIMEOUT);
        Some(Self {
            base_url,
            model,
            api_key: text("api_key"),
            timeout,
        })
    }

    /// `https://host/v1` and `https://host/v1/chat/completions` both work.
    pub fn completions_url(&self) -> String {
        let base = self.base_url.trim_end_matches('/');
        if base.ends_with("/chat/completions") {
            base.to_string()
        } else {
            format!("{base}/chat/completions")
        }
    }
}

/// The instructions an outside model gets: the refiner's universal rules (SPEC
/// "Intended text and writing modes"), the writing mode and the vocabulary line.
pub fn system_prompt(mode: Option<&str>, vocabulary: &[String]) -> String {
    let mut prompt = String::from(
        "You turn a raw speech-to-text transcript into the text the speaker intended to type. \
Fix misrecognized words only with evidence (the vocabulary, unambiguous context). Remove filler \
(um, uh, stutters, repeats, meaningless \"like\", \"you know\", \"I mean\"). When the speaker \
corrects themselves, keep only the final version. Write digits and symbols the way an engineer \
types them. Never add words that were not said, never answer or extend the text, never drop \
meaning. Output only the corrected text, as plain text.",
    );
    let line = match mode {
        Some("casual") => Some("Writing mode: casual (a chat message): keep the speaker's informal words, lowercase sentence starts except names, acronyms and \"I\", no final period."),
        Some("formal") => Some("Writing mode: formal (an email or document): complete sentences, grammar slips fixed without rephrasing."),
        Some("prompt") => Some("Writing mode: prompt (an instruction to an AI assistant): keep a revised request once in its final version; separate steps as a numbered list."),
        Some("minimal") | None => Some("Writing mode: minimal (notes and documents)."),
        Some(_) => None,
    };
    if let Some(line) = line {
        prompt.push('\n');
        prompt.push_str(line);
    }
    if !vocabulary.is_empty() {
        prompt.push_str("\nVocabulary the speaker may use: ");
        prompt.push_str(&vocabulary.join(", "));
    }
    prompt
}

pub fn request_body(model: &str, system: &str, transcript: &str) -> Value {
    json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": transcript},
        ],
        "temperature": 0,
        "stream": false,
    })
}

/// The first choice's message content.
pub fn parse_reply(value: &Value) -> Result<String> {
    if let Some(error) = value.get("error") {
        bail!("endpoint error: {error}");
    }
    let text = value
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("no choices[0].message.content in the reply"))?;
    let text = text.trim();
    if text.is_empty() {
        bail!("empty reply");
    }
    Ok(text.to_string())
}

/// A double-quoted curl config value.
fn config_quote(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for c in value.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The curl configuration for one request (fed on stdin; holds the key).
pub fn curl_config(endpoint: &Endpoint, body: &Value) -> String {
    let mut config = String::new();
    config.push_str(&format!(
        "url = {}\n",
        config_quote(&endpoint.completions_url())
    ));
    config.push_str("header = \"Content-Type: application/json\"\n");
    if !endpoint.api_key.is_empty() {
        config.push_str(&format!(
            "header = {}\n",
            config_quote(&format!("Authorization: Bearer {}", endpoint.api_key))
        ));
    }
    config.push_str(&format!(
        "data-binary = {}\n",
        config_quote(&body.to_string())
    ));
    config
}

/// Sends one transcript and returns the endpoint's text, within the endpoint's timeout.
pub fn complete(endpoint: &Endpoint, system: &str, transcript: &str) -> Result<String> {
    let body = request_body(&endpoint.model, system, transcript);
    let seconds = format!("{:.1}", endpoint.timeout.as_secs_f64());
    let mut child = Command::new(if cfg!(windows) { "curl.exe" } else { "curl" })
        .args([
            "--silent",
            "--show-error",
            "--max-time",
            &seconds,
            "--write-out",
            "\n%{http_code}",
            "--config",
            "-",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("start curl")?;
    child
        .stdin
        .take()
        .context("curl stdin")?
        .write_all(curl_config(endpoint, &body).as_bytes())
        .context("write curl config")?;
    let output = child.wait_with_output().context("wait for curl")?;
    if !output.status.success() {
        bail!(
            "curl: {}",
            String::from_utf8_lossy(&output.stderr)
                .trim()
                .chars()
                .take(200)
                .collect::<String>()
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let (body, code) = stdout.rsplit_once('\n').unwrap_or(("", stdout.as_ref()));
    let code: u16 = code.trim().parse().unwrap_or(0);
    if !(200..300).contains(&code) {
        bail!(
            "HTTP {code}: {}",
            body.trim().chars().take(200).collect::<String>()
        );
    }
    let value: Value = serde_json::from_str(body).context("reply is not JSON")?;
    parse_reply(&value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read};
    use std::net::TcpListener;
    use std::sync::mpsc;

    fn endpoint(base: &str) -> Endpoint {
        Endpoint {
            base_url: base.into(),
            model: "stub-model".into(),
            api_key: "sk-test-\"quoted\\key".into(),
            timeout: Duration::from_millis(1500),
        }
    }

    /// One-request HTTP stub: replies with `status` and `body` after `delay`, and
    /// hands back the request line, headers and body it got.
    fn stub(status: u16, body: &'static str, delay: Duration) -> (String, mpsc::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}/v1", listener.local_addr().unwrap());
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut head = String::new();
            let mut length = 0;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    length = value.trim().parse().unwrap();
                }
                head.push_str(&line);
                if line == "\r\n" || line.is_empty() {
                    break;
                }
            }
            let mut request_body = vec![0; length];
            reader.read_exact(&mut request_body).unwrap();
            std::thread::sleep(delay);
            let mut stream = stream;
            let _ = write!(
                stream,
                "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = tx.send(head + &String::from_utf8_lossy(&request_body));
        });
        (base, rx)
    }

    #[test]
    fn sends_the_transcript_and_reads_the_reply() {
        let (base, seen) = stub(
            200,
            r#"{"choices":[{"message":{"role":"assistant","content":" Ship it after lunch. "}}]}"#,
            Duration::ZERO,
        );
        let text = complete(&endpoint(&base), "SYSTEM", "um ship it after lunch").unwrap();
        assert_eq!(text, "Ship it after lunch.");
        let request = seen.recv().unwrap();
        assert!(
            request.starts_with("POST /v1/chat/completions "),
            "{request}"
        );
        assert!(request.contains("Authorization: Bearer sk-test-\"quoted\\key"));
        let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
        assert_eq!(body["model"], "stub-model");
        assert_eq!(body["messages"][0]["content"], "SYSTEM");
        assert_eq!(body["messages"][1]["content"], "um ship it after lunch");
    }

    #[test]
    fn errors_and_timeouts_are_errors() {
        let (base, _seen) = stub(500, r#"{"error":{"message":"boom"}}"#, Duration::ZERO);
        let error = complete(&endpoint(&base), "S", "t")
            .unwrap_err()
            .to_string();
        assert!(error.contains("500"), "{error}");
        let (base, _seen) = stub(200, r#"{"choices":[]}"#, Duration::ZERO);
        assert!(complete(&endpoint(&base), "S", "t").is_err());
        let (base, _seen) = stub(
            200,
            r#"{"choices":[{"message":{"content":"late"}}]}"#,
            Duration::from_secs(4),
        );
        let started = std::time::Instant::now();
        assert!(complete(&endpoint(&base), "S", "t").is_err());
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "the timeout holds"
        );
        // nothing listening
        assert!(complete(&endpoint("http://127.0.0.1:9/v1"), "S", "t").is_err());
    }

    #[test]
    fn config_from_the_command() {
        assert_eq!(Endpoint::from_command(&json!({})), None);
        assert_eq!(
            Endpoint::from_command(&json!({"base_url": "https://x/v1"})),
            None
        );
        assert_eq!(
            Endpoint::from_command(&json!({"base_url": "file:///etc", "model": "m"})),
            None
        );
        let e = Endpoint::from_command(
            &json!({"base_url": " https://api.example.com/v1/ ", "model": "m", "api_key": "k"}),
        )
        .unwrap();
        assert_eq!(
            e.completions_url(),
            "https://api.example.com/v1/chat/completions"
        );
        assert_eq!(e.timeout, DEFAULT_TIMEOUT);
        assert!(
            !format!("{e:?}").contains("\"k\""),
            "the key is not printed"
        );
        let e = Endpoint::from_command(&json!({
            "base_url": "http://localhost:8080/v1/chat/completions", "model": "m", "timeout_ms": 2000
        }))
        .unwrap();
        assert_eq!(
            e.completions_url(),
            "http://localhost:8080/v1/chat/completions"
        );
        assert_eq!(e.timeout, Duration::from_secs(2));
    }

    #[test]
    fn the_key_stays_out_of_arguments() {
        let body = request_body("m", "s", "a \"quoted\" line\nnext");
        let config = curl_config(&endpoint("https://h/v1"), &body);
        assert!(config.contains("header = \"Authorization: Bearer sk-test-\\\"quoted\\\\key\""));
        // the JSON body survives the config quoting
        let data = config
            .lines()
            .find(|l| l.starts_with("data-binary"))
            .unwrap();
        assert!(data.ends_with('"'));
    }

    #[test]
    fn prompt_carries_mode_and_vocabulary() {
        let p = system_prompt(Some("prompt"), &["cuBLAS".into(), "Blackwell".into()]);
        assert!(p.contains("Writing mode: prompt"));
        assert!(p.ends_with("Vocabulary the speaker may use: cuBLAS, Blackwell"));
        assert!(system_prompt(None, &[]).contains("Writing mode: minimal"));
    }
}
