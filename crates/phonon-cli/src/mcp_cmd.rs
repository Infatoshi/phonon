//! `phonon mcp`: a read-only Model Context Protocol server over stdio.
//!
//! One JSON-RPC 2.0 message per line on stdin, one response per line on stdout. Logs go to
//! stderr only, and nothing on disk is ever written: meetings are read without running speech
//! recognition, and dictation history and the dictionary are only read.

use crate::transcript_cmd::{
    clamp_limit, list_meetings, now_unix_ms, parse_offset_seconds, parse_time_filter,
    query_history, search_meetings, show_meeting, DataPaths, HistoryQuery, TranscriptFormat,
    HISTORY_DEFAULT, HISTORY_MAX, MEETINGS_LIST_DEFAULT, MEETINGS_LIST_MAX,
    MEETINGS_SEARCH_DEFAULT, MEETINGS_SEARCH_MAX,
};
use anyhow::{anyhow, bail, Result};
use serde_json::{json, Map, Value};
use std::io::{BufRead, Write};
use std::path::Path;

pub const PROTOCOL_VERSION: &str = "2024-11-05";

const PARSE_ERROR: i64 = -32700;
const INVALID_REQUEST: i64 = -32600;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;

pub fn run_mcp() -> Result<()> {
    let paths = DataPaths::from_env()?;
    eprintln!(
        "phonon mcp: read-only server on stdio (meetings: {})",
        paths.meetings.display()
    );
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if let Some(response) = handle_line(&line, &paths) {
            writeln!(stdout, "{response}")?;
            stdout.flush()?;
        }
    }
    Ok(())
}

/// Handles one JSON-RPC line against the Phonon data folder `data_dir`; `None` for
/// notifications and blank lines. The stdio server uses the same handler with
/// [`DataPaths::from_env`]; this entry point lets tests drive it on fixture folders.
#[cfg_attr(not(test), allow(dead_code))]
pub fn handle_jsonrpc_line(line: &str, data_dir: &Path) -> Option<String> {
    handle_line(line, &DataPaths::from_data_dir(data_dir))
}

fn error_response(id: Value, code: i64, message: impl Into<String>) -> String {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message.into() },
    })
    .to_string()
}

fn handle_line(line: &str, paths: &DataPaths) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    let message: Value = match serde_json::from_str(trimmed) {
        Ok(v) => v,
        Err(err) => {
            return Some(error_response(
                Value::Null,
                PARSE_ERROR,
                format!("parse error: {err}"),
            ))
        }
    };
    let Some(obj) = message.as_object() else {
        return Some(error_response(
            Value::Null,
            INVALID_REQUEST,
            "expected a JSON-RPC request object",
        ));
    };
    let id = obj.get("id").cloned();
    let Some(method) = obj.get("method").and_then(Value::as_str) else {
        // Responses from the client, or malformed requests.
        return id.map(|id| error_response(id, INVALID_REQUEST, "missing method"));
    };
    let Some(id) = id else {
        // Notifications (`notifications/initialized`, `notifications/cancelled`, ...) get no reply.
        return None;
    };
    let params = obj.get("params").cloned().unwrap_or(Value::Null);
    let result = match method {
        "initialize" => Ok(json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": { "tools": {} },
            "serverInfo": { "name": "phonon", "version": env!("CARGO_PKG_VERSION") },
            "instructions": "Read-only access to Phonon's local meeting transcripts and dictation history.",
        })),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({ "tools": tool_definitions() })),
        "tools/call" => call_tool(&params, paths),
        other => Err((METHOD_NOT_FOUND, format!("method not found: {other}"))),
    };
    Some(match result {
        Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }).to_string(),
        Err((code, message)) => error_response(id, code, message),
    })
}

fn time_props() -> Value {
    json!({
        "since": {
            "type": "string",
            "description": "Start of the time window: 3d, 36h, 90m, 2w, today, yesterday, now, YYYY-MM-DD, RFC3339 or Unix seconds/ms."
        },
        "until": {
            "type": "string",
            "description": "End of the time window, same forms as since."
        }
    })
}

fn limit_prop(default: usize, max: usize) -> Value {
    json!({ "type": "integer", "minimum": 1, "maximum": max, "default": default })
}

fn schema(mut props: Map<String, Value>, extra: Value, required: &[&str]) -> Value {
    if let Value::Object(extra) = extra {
        props.extend(extra);
    }
    json!({
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": false,
    })
}

fn time_map() -> Map<String, Value> {
    match time_props() {
        Value::Object(map) => map,
        _ => Map::new(),
    }
}

fn offset_prop(description: &str) -> Value {
    json!({
        "type": ["string", "number"],
        "description": description,
    })
}

pub fn tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "name": "list_meetings",
            "description": "List recorded meetings, newest first, with title, app, start time, duration and turn count.",
            "inputSchema": schema(time_map(), json!({
                "limit": limit_prop(MEETINGS_LIST_DEFAULT, MEETINGS_LIST_MAX),
            }), &[]),
        }),
        json!({
            "name": "get_transcript",
            "description": "Get one meeting transcript with speaker turns (You = microphone, Call = other participants).",
            "inputSchema": schema(Map::new(), json!({
                "meeting_id": {
                    "type": "string",
                    "default": "latest",
                    "description": "\"latest\", an exact meeting id, or an unambiguous id prefix of at least 6 characters."
                },
                "from": offset_prop("Only turns from this offset: mm:ss, h:mm:ss, seconds, 90s or 1.5m."),
                "to": offset_prop("Only turns up to this offset, same forms as from."),
                "format": {
                    "type": "string",
                    "enum": ["md", "json", "srt", "vtt"],
                    "default": "md"
                }
            }), &[]),
        }),
        json!({
            "name": "search_transcripts",
            "description": "Search meeting transcripts (stemmed BM25 with fuzzy and dictionary-alias matching; \"quoted phrases\" must match exactly).",
            "inputSchema": schema(time_map(), json!({
                "query": { "type": "string", "minLength": 1 },
                "limit": limit_prop(MEETINGS_SEARCH_DEFAULT, MEETINGS_SEARCH_MAX),
            }), &["query"]),
        }),
        json!({
            "name": "list_history",
            "description": "List dictation history, newest first, optionally filtered by app or substring.",
            "inputSchema": schema(time_map(), json!({
                "app": { "type": "string", "description": "Case-insensitive match on app name, bundle id or web host." },
                "grep": { "type": "string", "description": "Case-insensitive substring the text must contain." },
                "limit": limit_prop(HISTORY_DEFAULT, HISTORY_MAX),
                "raw": { "type": "boolean", "default": false, "description": "Use raw speech recognition text instead of the corrected text." },
            }), &[]),
        }),
        json!({
            "name": "search_history",
            "description": "Search dictation history by relevance (stemmed BM25 with fuzzy and dictionary-alias matching).",
            "inputSchema": schema(time_map(), json!({
                "query": { "type": "string", "minLength": 1 },
                "app": { "type": "string", "description": "Case-insensitive match on app name, bundle id or web host." },
                "limit": limit_prop(HISTORY_DEFAULT, HISTORY_MAX),
                "raw": { "type": "boolean", "default": false },
            }), &["query"]),
        }),
    ]
}

type RpcResult = std::result::Result<Value, (i64, String)>;

fn call_tool(params: &Value, paths: &DataPaths) -> RpcResult {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or((INVALID_PARAMS, "tools/call needs a tool name".to_string()))?;
    let empty = Map::new();
    let args = match params.get("arguments") {
        None | Some(Value::Null) => &empty,
        Some(Value::Object(map)) => map,
        Some(_) => {
            return Ok(tool_error("arguments must be an object"));
        }
    };
    let outcome = match name {
        "list_meetings" => tool_list_meetings(args, paths),
        "get_transcript" => tool_get_transcript(args, paths),
        "search_transcripts" => tool_search_transcripts(args, paths),
        "list_history" => tool_history(args, paths, false),
        "search_history" => tool_history(args, paths, true),
        other => return Err((INVALID_PARAMS, format!("unknown tool: {other}"))),
    };
    Ok(match outcome {
        Ok(text) => json!({ "content": [{ "type": "text", "text": text }], "isError": false }),
        Err(err) => tool_error(&format!("{err:#}")),
    })
}

fn tool_error(message: &str) -> Value {
    json!({ "content": [{ "type": "text", "text": message }], "isError": true })
}

fn arg_str(args: &Map<String, Value>, key: &str) -> Result<Option<String>> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(_) => bail!("{key} must be a string"),
    }
}

fn arg_bool(args: &Map<String, Value>, key: &str) -> Result<bool> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(false),
        Some(Value::Bool(b)) => Ok(*b),
        Some(_) => bail!("{key} must be a boolean"),
    }
}

fn arg_limit(args: &Map<String, Value>, default: usize, max: usize) -> Result<usize> {
    let raw = match args.get("limit") {
        None | Some(Value::Null) => None,
        Some(v) => {
            let n = v
                .as_u64()
                .filter(|n| *n >= 1)
                .ok_or_else(|| anyhow!("limit must be a positive integer"))?;
            Some(usize::try_from(n).unwrap_or(max))
        }
    };
    Ok(clamp_limit(raw, default, max))
}

fn arg_offset(args: &Map<String, Value>, key: &str) -> Result<Option<f64>> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => parse_offset_seconds(s).map(Some),
        Some(Value::Number(n)) => {
            let v = n.as_f64().filter(|v| v.is_finite() && *v >= 0.0);
            v.map(Some)
                .ok_or_else(|| anyhow!("{key} must be a non-negative number of seconds"))
        }
        Some(_) => bail!("{key} must be an offset string or a number of seconds"),
    }
}

fn arg_window(args: &Map<String, Value>) -> Result<(Option<i64>, Option<i64>)> {
    let now = now_unix_ms();
    let since = arg_str(args, "since")?
        .map(|s| parse_time_filter(&s, now))
        .transpose()?;
    let until = arg_str(args, "until")?
        .map(|s| parse_time_filter(&s, now))
        .transpose()?;
    if let (Some(a), Some(b)) = (since, until) {
        if a > b {
            bail!("since is after until");
        }
    }
    Ok((since, until))
}

fn required_query(args: &Map<String, Value>) -> Result<String> {
    arg_str(args, "query")?
        .filter(|q| !q.trim().is_empty())
        .ok_or_else(|| anyhow!("query is required"))
}

fn tool_list_meetings(args: &Map<String, Value>, paths: &DataPaths) -> Result<String> {
    let (since, until) = arg_window(args)?;
    let limit = arg_limit(args, MEETINGS_LIST_DEFAULT, MEETINGS_LIST_MAX)?;
    let meetings = list_meetings(&paths.meetings, since, until, limit)?;
    Ok(serde_json::to_string_pretty(
        &json!({ "meetings": meetings }),
    )?)
}

fn tool_get_transcript(args: &Map<String, Value>, paths: &DataPaths) -> Result<String> {
    let meeting = arg_str(args, "meeting_id")?.unwrap_or_else(|| "latest".into());
    let from = arg_offset(args, "from")?;
    let to = arg_offset(args, "to")?;
    let format = match arg_str(args, "format")?.as_deref() {
        None => TranscriptFormat::Markdown,
        Some(f @ ("md" | "json" | "srt" | "vtt")) => TranscriptFormat::parse(f)?,
        Some(other) => bail!("format must be one of md, json, srt, vtt (got {other:?})"),
    };
    show_meeting(&paths.meetings, &meeting, from, to, format)
}

fn tool_search_transcripts(args: &Map<String, Value>, paths: &DataPaths) -> Result<String> {
    let query = required_query(args)?;
    let (since, until) = arg_window(args)?;
    let limit = arg_limit(args, MEETINGS_SEARCH_DEFAULT, MEETINGS_SEARCH_MAX)?;
    let hits = search_meetings(paths, &query, since, until, limit)?;
    Ok(serde_json::to_string_pretty(
        &json!({ "query": query, "results": hits }),
    )?)
}

fn tool_history(args: &Map<String, Value>, paths: &DataPaths, search: bool) -> Result<String> {
    let query = if search {
        Some(required_query(args)?)
    } else {
        None
    };
    let (since, until) = arg_window(args)?;
    let entries = query_history(
        paths,
        &HistoryQuery {
            since,
            until,
            app: arg_str(args, "app")?,
            grep: if search { None } else { arg_str(args, "grep")? },
            search: query.clone(),
            raw: arg_bool(args, "raw")?,
            limit: arg_limit(args, HISTORY_DEFAULT, HISTORY_MAX)?,
        },
    )?;
    let mut body = json!({ "entries": entries });
    if let Some(q) = query {
        body["query"] = json!(q);
    }
    Ok(serde_json::to_string_pretty(&body)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "phonon-mcp-unit-{tag}-{}-{nanos}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn fixture() -> TempDir {
        let tmp = TempDir::new("fixture");
        let meeting = tmp.0.join("Meetings/meeting_1794959123259");
        fs::create_dir_all(&meeting).unwrap();
        fs::write(
            meeting.join("metadata.json"),
            r#"{"id":"meeting_1794959123259","started_at":"2026-11-17T19:00:00Z","ended_at":"2026-11-17T19:05:00Z","app_name":"Zoom"}"#,
        )
        .unwrap();
        fs::write(
            meeting.join("transcript.json"),
            r#"{"model":"parakeet","turns":[
                {"start_seconds":1.0,"end_seconds":4.0,"speaker":"call","text":"Can you ship the release?"},
                {"start_seconds":5.0,"end_seconds":8.0,"speaker":"me","text":"Shipping it <today> & tomorrow."}
            ]}"#,
        )
        .unwrap();
        let rec = tmp.0.join("Corpus/rec1");
        fs::create_dir_all(&rec).unwrap();
        fs::write(
            rec.join("metadata.json"),
            r#"{"schema_version":1,"id":"rec1","created_at_unix_ms":1794959000000,"source":"live","audio_file":"audio.wav","raw_transcript":"fonon is great","final_transcript":"Phonon is great","app_context":{"bundle_id":"com.tinyspeck.slackmacgap","app_name":"Slack","category":"chat"}}"#,
        )
        .unwrap();
        tmp
    }

    fn call(dir: &Path, id: i64, method: &str, params: Value) -> Value {
        let line = json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}).to_string();
        let out = handle_jsonrpc_line(&line, dir).expect("response");
        serde_json::from_str(&out).unwrap()
    }

    #[test]
    fn initialize_list_and_call_tools() {
        let tmp = fixture();
        let init = call(&tmp.0, 1, "initialize", json!({}));
        assert_eq!(init["result"]["protocolVersion"], PROTOCOL_VERSION);
        assert_eq!(init["result"]["serverInfo"]["name"], "phonon");
        assert!(handle_jsonrpc_line(
            r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
            &tmp.0
        )
        .is_none());
        let tools = call(&tmp.0, 2, "tools/list", json!({}));
        assert_eq!(tools["result"]["tools"].as_array().unwrap().len(), 5);

        let t = call(
            &tmp.0,
            3,
            "tools/call",
            json!({"name":"get_transcript","arguments":{"format":"vtt"}}),
        );
        let text = t["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.starts_with("WEBVTT"));
        assert!(text.contains("&lt;today&gt; &amp; tomorrow"));

        let bad = call(
            &tmp.0,
            4,
            "tools/call",
            json!({"name":"get_transcript","arguments":{"meeting_id":"nope_missing"}}),
        );
        assert_eq!(bad["result"]["isError"], true);

        let unknown = call(&tmp.0, 5, "bogus/method", json!({}));
        assert_eq!(unknown["error"]["code"], METHOD_NOT_FOUND);
        let parse = handle_jsonrpc_line("{not json", &tmp.0).unwrap();
        assert!(parse.contains("-32700"));
    }
}
