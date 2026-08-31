//! Transcript correction through llama.cpp.
//!
//! The macOS build runs Gemma 4 E2B on MLX inside `sidecar/polish_server.py`.
//! Windows runs the same model, Google's quantisation-aware q4_0 GGUF, inside a
//! resident `llama-server.exe`. The prompt, the temperature, and the output
//! budget rule are the same, so both platforms correct a transcript the same way.
//!
//! Correction is mandatory. A recogniser result that never reaches this stage is
//! a failure, not a degraded mode.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};

use crate::fetch;
use crate::http;
use crate::manifest;

/// The correction prompt, shared with the macOS sidecar. Embedded so the shipped
/// executable carries no loose files.
pub const SYSTEM_PROMPT: &str = include_str!("../../../prompts/polish_v2.txt");

/// Bounds on generated tokens, copied from `sidecar/polish_server.py`.
pub const OUTPUT_TOKEN_CEILING: u32 = 256;
pub const OUTPUT_TOKEN_FLOOR: u32 = 48;

/// The model may take this long to load before the server is called dead.
const READY_TIMEOUT: Duration = Duration::from_secs(600);

/// Wrap a transcript the way the prompt expects.
pub fn user_message(text: &str) -> String {
    format!("<transcript>\n{}\n</transcript>", text.trim())
}

/// The text inside `<transcript>`, or the whole string when the tag is absent.
/// Mirrors `transcript_payload` in the macOS sidecar.
pub fn transcript_payload(text: &str) -> &str {
    let Some(open) = text.find("<transcript>") else {
        return text.trim();
    };
    let rest = &text[open + "<transcript>".len()..];
    match rest.find("</transcript>") {
        Some(close) => rest[..close].trim(),
        None => rest.trim(),
    }
}

/// Strip channel and turn markers a chat template can leak into the answer.
/// Mirrors `clean_output` in the macOS sidecar, whose pattern is
/// `<\|?/?(?:channel|turn|think)\|?>?`.
pub fn clean_output(text: &str) -> String {
    const MARKERS: [&str; 3] = ["channel", "turn", "think"];
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut index = 0;
    while index < chars.len() {
        if chars[index] == '<' {
            if let Some(width) = marker_width(&chars[index..], &MARKERS) {
                index += width;
                continue;
            }
        }
        out.push(chars[index]);
        index += 1;
    }
    out.trim().to_string()
}

/// How many characters the marker at the start of `rest` spans, if it is one.
fn marker_width(rest: &[char], markers: &[&str]) -> Option<usize> {
    let mut at = 1; // the '<' itself
    let eat = |at: &mut usize, wanted: char| {
        if rest.get(*at) == Some(&wanted) {
            *at += 1;
        }
    };
    eat(&mut at, '|');
    eat(&mut at, '/');
    let word = markers.iter().find(|marker| {
        marker
            .chars()
            .enumerate()
            .all(|(offset, wanted)| rest.get(at + offset) == Some(&wanted))
    })?;
    at += word.chars().count();
    eat(&mut at, '|');
    eat(&mut at, '>');
    Some(at)
}

/// How many tokens the correction may generate, given the spoken length.
/// A clean-up pass returns roughly its input, with headroom for expanded
/// orthography. Same rule as the macOS sidecar.
pub fn output_budget(spoken_tokens: u32) -> u32 {
    let budget = (spoken_tokens as f64 * 1.8).ceil() as u32 + 24;
    budget.clamp(OUTPUT_TOKEN_FLOOR, OUTPUT_TOKEN_CEILING)
}

/// A rough token count for text, used only when the server cannot tokenise.
/// English averages a little under four characters per token.
pub fn estimate_tokens(text: &str) -> u32 {
    let chars = text.trim().chars().count() as u32;
    chars.div_ceil(4).max(1)
}

/// How many threads llama.cpp should use. One core stays free for the tray, the
/// hook, and whatever the user is typing into.
pub fn thread_count(logical_cpus: usize) -> usize {
    logical_cpus.saturating_sub(1).max(1)
}

/// A free loopback port. The server binds it a moment later; nothing else on the
/// machine is looking for a port in that instant.
fn free_port() -> Result<u16> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").context("reserve a local port")?;
    Ok(listener.local_addr()?.port())
}

/// The answer inside one chat-completion reply.
///
/// A reply whose `content` is empty but whose `reasoning_content` is not means
/// the model reasoned instead of answering. Say so: it is a different fault
/// from a server that returned nothing at all.
pub fn parse_reply(value: &serde_json::Value) -> Result<String> {
    let choice = value
        .get("choices")
        .and_then(|choices| choices.get(0))
        .ok_or_else(|| anyhow!("correction reply has no choices: {value}"))?;
    let message = choice
        .get("message")
        .ok_or_else(|| anyhow!("correction reply has no message: {value}"))?;
    let content = message
        .get("content")
        .and_then(|content| content.as_str())
        .unwrap_or_default();
    if !content.trim().is_empty() {
        return Ok(content.to_string());
    }
    let reasoning = message
        .get("reasoning_content")
        .and_then(|reasoning| reasoning.as_str())
        .unwrap_or_default();
    let finish = choice
        .get("finish_reason")
        .and_then(|reason| reason.as_str())
        .unwrap_or("unknown");
    if reasoning.trim().is_empty() {
        bail!("correction returned an empty answer (finish_reason {finish})");
    }
    bail!(
        "correction reasoned instead of answering (finish_reason {finish}): {}",
        reasoning.trim()
    )
}

/// A running `llama-server` with the correction model loaded.
pub struct Corrector {
    child: Child,
    base: String,
    system_prompt: String,
}

impl Corrector {
    /// Start the server and wait until the model answers. Call after
    /// `fetch::ensure_all`.
    pub fn start() -> Result<Self> {
        let server = fetch::sentinel_path(&manifest::LLAMA_RUNTIME);
        if !server.is_file() {
            bail!(
                "{} is missing; run the first-run download",
                server.display()
            );
        }
        let model: PathBuf =
            fetch::component_dir(&manifest::POLISH_MODEL).join(manifest::POLISH_GGUF);
        if !model.is_file() {
            bail!("{} is missing", model.display());
        }
        let port = free_port()?;
        let threads = thread_count(
            std::thread::available_parallelism()
                .map(|count| count.get())
                .unwrap_or(2),
        );

        let mut command = Command::new(&server);
        command
            .arg("-m")
            .arg(&model)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("-c")
            .arg("4096")
            .arg("-t")
            .arg(threads.to_string())
            .arg("--jinja")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(dir) = server.parent() {
            command.current_dir(dir);
        }
        let mut child = command
            .spawn()
            .with_context(|| format!("start {}", server.display()))?;

        let base = format!("http://127.0.0.1:{port}");
        let started = Instant::now();
        loop {
            if let Some(status) = child.try_wait()? {
                let mut stderr = String::new();
                if let Some(mut pipe) = child.stderr.take() {
                    let _ = pipe.read_to_string(&mut stderr);
                }
                bail!(
                    "llama-server exited with {status} before it was ready: {}",
                    stderr
                        .lines()
                        .rev()
                        .take(12)
                        .collect::<Vec<_>>()
                        .join(" | ")
                );
            }
            if http::agent()
                .get(&format!("{base}/health"))
                .timeout(Duration::from_secs(5))
                .call()
                .is_ok()
            {
                break;
            }
            if started.elapsed() > READY_TIMEOUT {
                let _ = child.kill();
                bail!("llama-server did not become ready within {READY_TIMEOUT:?}");
            }
            std::thread::sleep(Duration::from_millis(500));
        }

        Ok(Self {
            child,
            base,
            system_prompt: SYSTEM_PROMPT.trim().to_string(),
        })
    }

    /// Token count from the loaded model's own tokeniser, or an estimate when the
    /// endpoint is unavailable.
    fn count_tokens(&self, text: &str) -> u32 {
        let Ok(response) = http::agent()
            .post(&format!("{}/tokenize", self.base))
            .timeout(Duration::from_secs(10))
            .send_json(serde_json::json!({ "content": text }))
        else {
            return estimate_tokens(text);
        };
        let Ok(value) = response.into_json::<serde_json::Value>() else {
            return estimate_tokens(text);
        };
        value
            .get("tokens")
            .and_then(|tokens| tokens.as_array())
            .map(|tokens| tokens.len() as u32)
            .unwrap_or_else(|| estimate_tokens(text))
    }

    /// Ask the model once. `messages` is sent as it is.
    fn chat(&self, messages: serde_json::Value, max_tokens: u32) -> Result<String> {
        let response = http::agent()
            .post(&format!("{}/v1/chat/completions", self.base))
            .timeout(Duration::from_secs(300))
            .send_json(serde_json::json!({
                "messages": messages,
                "temperature": 0.0,
                "top_k": 1,
                "max_tokens": max_tokens,
                "stream": false,
                "cache_prompt": true,
                // Gemma 4 opens a thought channel on its own. llama.cpp turns
                // the template's thinking on by default, and the model then
                // spends the whole budget reasoning and answers nothing. A
                // formatter has nothing to reason about, so turn it off. This is
                // what the closed-thought prefill does in the macOS sidecar.
                "chat_template_kwargs": { "enable_thinking": false },
            }))?;
        let value: serde_json::Value = response.into_json().context("read correction reply")?;
        parse_reply(&value)
    }

    /// Correct one transcript.
    pub fn correct(&self, raw: &str) -> Result<String> {
        let user = user_message(raw);
        let max_tokens = output_budget(self.count_tokens(transcript_payload(&user)));

        let separate = serde_json::json!([
            { "role": "system", "content": self.system_prompt },
            { "role": "user", "content": user },
        ]);
        let reply = match self.chat(separate, max_tokens) {
            Ok(reply) => reply,
            // Some Gemma chat templates reject a separate system turn. Fold the
            // prompt into the user turn and ask again; the text is identical.
            Err(_) => {
                let merged = serde_json::json!([{
                    "role": "user",
                    "content": format!("{}\n\n{}", self.system_prompt, user),
                }]);
                self.chat(merged, max_tokens)
                    .context("correction request failed")?
            }
        };

        let cleaned = clean_output(transcript_payload(&reply));
        if cleaned.is_empty() {
            bail!("correction returned nothing for {raw:?}; the reply was {reply:?}");
        }
        Ok(cleaned)
    }
}

impl Drop for Corrector {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_answer_out_of_a_reply() {
        let reply = serde_json::json!({
            "choices": [{ "finish_reason": "stop",
                          "message": { "content": "Hello, Fluid Voice." } }]
        });
        assert_eq!(parse_reply(&reply).unwrap(), "Hello, Fluid Voice.");
    }

    /// The first end-to-end run on Windows failed exactly this way, so the
    /// message has to name the cause rather than say "nothing came back".
    #[test]
    fn reasoning_without_an_answer_is_named() {
        let reply = serde_json::json!({
            "choices": [{ "finish_reason": "length",
                          "message": { "content": "",
                                       "reasoning_content": "Thinking Process: ..." } }]
        });
        let error = parse_reply(&reply).unwrap_err().to_string();
        assert!(error.contains("reasoned instead of answering"), "{error}");
        assert!(error.contains("length"), "{error}");
    }

    #[test]
    fn an_empty_reply_is_an_error() {
        let reply = serde_json::json!({
            "choices": [{ "finish_reason": "stop", "message": { "content": "  " } }]
        });
        assert!(parse_reply(&reply)
            .unwrap_err()
            .to_string()
            .contains("empty answer"));
        assert!(parse_reply(&serde_json::json!({})).is_err());
    }

    #[test]
    fn the_prompt_ships_inside_the_executable() {
        assert!(SYSTEM_PROMPT.contains("second-pass formatter"));
        assert!(SYSTEM_PROMPT.contains("<transcript>"));
    }

    #[test]
    fn wraps_the_transcript() {
        assert_eq!(
            user_message("  hello there  "),
            "<transcript>\nhello there\n</transcript>"
        );
    }

    #[test]
    fn reads_the_payload_back_out() {
        assert_eq!(transcript_payload("<transcript>\nhi\n</transcript>"), "hi");
        assert_eq!(transcript_payload("plain text"), "plain text");
        assert_eq!(transcript_payload("<transcript>\nunclosed"), "unclosed");
    }

    #[test]
    fn strips_leaked_channel_markers() {
        assert_eq!(clean_output("<|channel|>Hello there."), "Hello there.");
        assert_eq!(
            clean_output("<channel>thought</channel>Done."),
            "thoughtDone."
        );
        assert_eq!(clean_output("  Plain answer.  "), "Plain answer.");
        assert_eq!(clean_output("<|think|>ok<|/think|>"), "ok");
    }

    /// A less-than sign that is not a marker must survive.
    #[test]
    fn keeps_ordinary_angle_brackets() {
        assert_eq!(
            clean_output("use a < b in the loop"),
            "use a < b in the loop"
        );
        assert_eq!(clean_output("<div> stays"), "<div> stays");
    }

    #[test]
    fn budget_follows_the_macos_rule() {
        // Floor.
        assert_eq!(output_budget(0), OUTPUT_TOKEN_FLOOR);
        assert_eq!(output_budget(10), OUTPUT_TOKEN_FLOOR);
        // ceil(20 * 1.8) + 24 = 60.
        assert_eq!(output_budget(20), 60);
        // Ceiling.
        assert_eq!(output_budget(400), OUTPUT_TOKEN_CEILING);
    }

    #[test]
    fn the_estimate_is_never_zero() {
        assert_eq!(estimate_tokens(""), 1);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
    }

    #[test]
    fn correction_threads_leave_one_core_free() {
        assert_eq!(thread_count(1), 1);
        assert_eq!(thread_count(2), 1);
        assert_eq!(thread_count(16), 15);
    }
}
