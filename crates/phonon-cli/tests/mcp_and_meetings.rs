//! `phonon meetings`, `phonon history`, `phonon transcript` formats and `phonon mcp`, end to end
//! on a fixture data folder. HOME points at a temporary folder and PHONON_DATA_DIR at the Phonon
//! data folder inside it, so the commands read the fixtures whichever variable they honour.

use serde_json::{json, Value};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

struct Fixture {
    home: PathBuf,
}

impl Fixture {
    fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let home = std::env::temp_dir().join(format!(
            "phonon-mcp-meetings-{tag}-{}-{nanos}",
            std::process::id()
        ));
        let fixture = Self { home };
        fixture.populate();
        fixture
    }

    fn data(&self) -> PathBuf {
        self.home.join("Library/Application Support/Phonon")
    }

    fn meeting(&self, id: &str, started: &str, title: &str, turns: Value) {
        let dir = self.data().join("Meetings").join(id);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("metadata.json"),
            json!({
                "id": id,
                "started_at": started,
                "ended_at": started,
                "app_name": "Zoom",
                "window_title": title,
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            dir.join("transcript.json"),
            json!({ "schema_version": 1, "model": "parakeet", "turns": turns }).to_string(),
        )
        .unwrap();
    }

    fn dictation(&self, id: &str, ms: i64, app: &str, raw: &str, fin: &str) {
        let dir = self.data().join("Corpus").join(id);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("metadata.json"),
            json!({
                "schema_version": 1,
                "id": id,
                "created_at_unix_ms": ms,
                "source": "live",
                "audio_file": "audio.wav",
                "raw_transcript": raw,
                "final_transcript": fin,
                "app_context": {"bundle_id": format!("com.example.{}", app.to_lowercase()), "app_name": app, "category": "chat"}
            })
            .to_string(),
        )
        .unwrap();
    }

    fn populate(&self) {
        self.meeting(
            "meeting_1794942000000",
            "2026-11-17T19:00:00Z",
            "Release planning",
            json!([
                {"start_seconds": 2.0, "end_seconds": 6.0, "speaker": "call", "text": "When do we ship the release?"},
                {"start_seconds": 65.0, "end_seconds": 70.0, "speaker": "me", "text": "We are shipping Phonon on Friday."},
                {"start_seconds": 6783.0, "end_seconds": 6790.0, "speaker": "call", "text": "Late <note> & wrap --> up"}
            ]),
        );
        self.meeting(
            "meeting_1795000000000",
            "2026-11-18T11:06:40Z",
            "Design review",
            json!([
                {"start_seconds": 1.0, "end_seconds": 3.0, "speaker": "me", "text": "Let us review the fonon logo."},
                {"start_seconds": 4.0, "end_seconds": 9.0, "speaker": "call", "text": "The design review looks good."}
            ]),
        );
        self.meeting(
            "meeting_1790000000000",
            "2026-09-21T14:13:20Z",
            "Old sync",
            json!([
                {"start_seconds": 1.0, "end_seconds": 3.0, "speaker": "me", "text": "Nothing about releases here, just a release."}
            ]),
        );
        self.dictation(
            "rec_a",
            1_794_900_000_000,
            "Slack",
            "ship the fonon build",
            "Ship the Phonon build.",
        );
        self.dictation(
            "rec_b",
            1_794_950_000_000,
            "Mail",
            "hello team",
            "Hello team, quick update.",
        );
        self.dictation(
            "rec_c",
            1_795_000_000_000,
            "Slack",
            "design review at noon",
            "Design review at noon.",
        );
        fs::write(
            self.data().join("dictionary.json"),
            json!({
                "schema_version": 1,
                "updated_at_unix_ms": 1,
                "entries": [{"phrase": "Phonon", "spoken_forms": ["fonon", "phone on"]}]
            })
            .to_string(),
        )
        .unwrap();
    }

    fn command(&self) -> Command {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_phonon"));
        cmd.env("HOME", &self.home)
            .env("PHONON_DATA_DIR", self.data())
            .env_remove("PHONON_MEETINGS_DIR");
        cmd
    }

    fn run(&self, args: &[&str]) -> Output {
        let out = self.command().args(args).output().unwrap();
        assert!(
            out.status.success(),
            "phonon {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        out
    }

    fn stdout(&self, args: &[&str]) -> String {
        String::from_utf8(self.run(args).stdout).unwrap()
    }

    fn json(&self, args: &[&str]) -> Value {
        serde_json::from_str(&self.stdout(args)).unwrap()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.home);
    }
}

fn snapshot(dir: &Path) -> Vec<(String, u64)> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in fs::read_dir(&d).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                let len = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                out.push((path.display().to_string(), len));
            }
        }
    }
    out.sort();
    out
}

#[test]
fn meetings_list_show_and_search() {
    let f = Fixture::new("meetings");
    let before = snapshot(&f.data());

    let list = f.json(&["meetings", "list", "--json"]);
    let ids: Vec<&str> = list["meetings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(
        ids,
        [
            "meeting_1795000000000",
            "meeting_1794942000000",
            "meeting_1790000000000"
        ]
    );
    assert_eq!(list["meetings"][1]["turns"], 3);

    let limited = f.json(&["meetings", "list", "--limit", "1", "--json"]);
    assert_eq!(limited["meetings"].as_array().unwrap().len(), 1);
    let window = f.json(&[
        "meetings",
        "list",
        "--since",
        "2026-11-01",
        "--until",
        "2026-11-17T23:00:00Z",
        "--json",
    ]);
    assert_eq!(window["meetings"].as_array().unwrap().len(), 1);
    assert_eq!(window["meetings"][0]["id"], "meeting_1794942000000");
    let text = f.stdout(&["meetings", "list"]);
    assert!(text.contains("Release planning"), "{text}");

    // Show: latest, prefix, formats and ranges.
    let md = f.stdout(&["meetings", "show"]);
    assert!(md.contains("Design review"));
    let md = f.stdout(&[
        "meetings", "show", "1794942", "--md", "--from", "1:00", "--to", "113:03",
    ]);
    assert!(md.starts_with("# Release planning"));
    assert!(md.contains("**[1:05] You:** We are shipping Phonon on Friday."));
    assert!(md.contains("**[1:53:03] Call:**"));
    assert!(!md.contains("When do we ship"));

    let v = f.json(&[
        "meetings",
        "show",
        "meeting_1794942000000",
        "--json",
        "--to",
        "90s",
    ]);
    assert_eq!(v["version"], 1);
    assert_eq!(v["meeting"]["title"], "Release planning");
    assert_eq!(v["range"]["to_seconds"], 90.0);
    assert_eq!(v["turns"].as_array().unwrap().len(), 2);

    let srt = f.stdout(&["meetings", "show", "1794942", "--srt", "--from", "113:00"]);
    assert_eq!(
        srt,
        "1\n01:53:03,000 --> 01:53:10,000\nCall: Late <note> & wrap -> up\n\n"
    );
    let vtt = f.stdout(&[
        "meetings", "show", "1794942", "--vtt", "--from", "1m", "--to", "1.5m",
    ]);
    assert_eq!(
        vtt,
        "WEBVTT\n\n00:01:05.000 --> 00:01:10.000\n<v You>We are shipping Phonon on Friday.</v>\n\n"
    );

    let ambiguous = f
        .command()
        .args(["meetings", "show", "17949"])
        .output()
        .unwrap();
    assert!(!ambiguous.status.success());
    let ambiguous = f
        .command()
        .args(["meetings", "show", "meeting_179"])
        .output()
        .unwrap();
    assert!(!ambiguous.status.success());
    assert!(String::from_utf8_lossy(&ambiguous.stderr).contains("ambiguous"));

    // Search: tiered ranking, aliases, phrases.
    let s = f.json(&["meetings", "search", "ship release", "--json"]);
    let results = s["results"].as_array().unwrap();
    assert_eq!(results[0]["meeting_id"], "meeting_1794942000000");
    assert_eq!(results[0]["match"], "all");
    assert!(results[0]["snippet"].as_str().unwrap().contains("**"));

    let alias = f.json(&["meetings", "search", "phonon", "--json"]);
    let alias_ids: Vec<&str> = alias["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["meeting_id"].as_str().unwrap())
        .collect();
    assert!(
        alias_ids.contains(&"meeting_1795000000000"),
        "fonon alias: {alias}"
    );
    assert!(alias_ids.contains(&"meeting_1794942000000"));

    let phrase = f.json(&["meetings", "search", "\"design review\"", "--json"]);
    assert_eq!(phrase["results"].as_array().unwrap().len(), 1);
    assert_eq!(phrase["results"][0]["meeting_id"], "meeting_1795000000000");

    let since = f.json(&[
        "meetings",
        "search",
        "release",
        "--since",
        "2026-11-01",
        "--json",
    ]);
    assert!(since["results"]
        .as_array()
        .unwrap()
        .iter()
        .all(|r| r["meeting_id"] != "meeting_1790000000000"));

    assert_eq!(
        before,
        snapshot(&f.data()),
        "read-only commands wrote to disk"
    );
}

#[test]
fn transcript_command_keeps_legacy_flags_and_adds_formats() {
    let f = Fixture::new("transcript");
    let legacy = f.json(&["transcript", "--meeting", "meeting_1794942000000", "--json"]);
    assert_eq!(legacy["id"], "meeting_1794942000000");
    assert_eq!(legacy["status"], "complete");
    assert!(
        legacy.get("version").is_none(),
        "legacy JSON shape unchanged"
    );

    let text = f.stdout(&["transcript", "--minutes", "1"]);
    assert!(text.contains("Meeting: Design review"));

    let vtt = f.stdout(&[
        "transcript",
        "--meeting",
        "meeting_1794942000000",
        "--vtt",
        "--from",
        "113:00",
    ]);
    assert!(vtt.contains("<v Call>Late &lt;note&gt; &amp; wrap --&gt; up</v>"));
    let md = f.stdout(&["transcript", "--md", "--to", "0:03"]);
    assert!(md.contains("**[0:01] You:** Let us review the fonon logo."));
    assert!(!md.contains("looks good"));

    // Completed meetings need no transcription, so no live artifacts are written.
    assert!(!f.data().join("Meetings/events.jsonl").exists());

    let conflict = f
        .command()
        .args(["transcript", "--json", "--srt"])
        .output()
        .unwrap();
    assert!(!conflict.status.success());
    let bad = f
        .command()
        .args(["transcript", "--from=-1:00"])
        .output()
        .unwrap();
    assert!(!bad.status.success());
}

#[test]
fn history_filters_and_search() {
    let f = Fixture::new("history");
    let before = snapshot(&f.data());

    let all = f.json(&["history", "--json"]);
    let ids: Vec<&str> = all["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["rec_c", "rec_b", "rec_a"]);
    assert_eq!(all["entries"][2]["text"], "Ship the Phonon build.");

    let slack = f.json(&["history", "--app", "slack", "--json"]);
    assert_eq!(slack["entries"].as_array().unwrap().len(), 2);
    let raw = f.json(&["history", "--raw", "--grep", "FONON", "--json"]);
    assert_eq!(raw["entries"].as_array().unwrap().len(), 1);
    assert_eq!(raw["entries"][0]["text"], "ship the fonon build");
    let window = f.json(&[
        "history",
        "--since",
        "1794940000000",
        "--until",
        "1794960000",
        "--json",
    ]);
    assert_eq!(window["entries"].as_array().unwrap().len(), 1);
    assert_eq!(window["entries"][0]["id"], "rec_b");
    let limited = f.json(&["history", "--limit", "1", "--json"]);
    assert_eq!(limited["entries"].as_array().unwrap().len(), 1);

    let search = f.json(&["history", "--search", "fonon", "--json"]);
    assert_eq!(search["entries"][0]["id"], "rec_a", "alias fonon -> Phonon");
    assert!(search["entries"][0]["snippet"]
        .as_str()
        .unwrap()
        .contains("**Phonon**"));

    let text = f.stdout(&["history", "--app", "mail"]);
    assert!(text.contains("Hello team"), "{text}");

    assert_eq!(before, snapshot(&f.data()));
}

fn mcp_session(f: &Fixture, requests: &[Value]) -> (Vec<Value>, String) {
    let mut child = f
        .command()
        .arg("mcp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    {
        let stdin = child.stdin.as_mut().unwrap();
        for req in requests {
            writeln!(stdin, "{req}").unwrap();
        }
        writeln!(stdin, "this is not json").unwrap();
    }
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).unwrap();
    let responses = stdout
        .lines()
        .map(|l| serde_json::from_str::<Value>(l).expect("every stdout line is JSON"))
        .collect();
    (responses, String::from_utf8_lossy(&out.stderr).into_owned())
}

fn tool_call(id: i64, name: &str, arguments: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "method": "tools/call", "params": {"name": name, "arguments": arguments}})
}

fn text_of(v: &Value) -> &str {
    v["result"]["content"][0]["text"].as_str().unwrap()
}

#[test]
fn mcp_round_trip_is_read_only() {
    let f = Fixture::new("mcp");
    let before = snapshot(&f.data());
    let requests = vec![
        json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}}),
        json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json!({"jsonrpc": "2.0", "id": 2, "method": "ping"}),
        json!({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}),
        tool_call(4, "list_meetings", json!({"limit": 2})),
        tool_call(
            5,
            "get_transcript",
            json!({"meeting_id": "1794942", "from": "1:00", "to": 70, "format": "json"}),
        ),
        tool_call(6, "search_transcripts", json!({"query": "ship release"})),
        tool_call(7, "list_history", json!({"app": "slack", "limit": 5})),
        tool_call(8, "search_history", json!({"query": "phonon", "raw": true})),
        tool_call(9, "get_transcript", json!({})),
        // Invalid parameters: tool errors, not crashes.
        tool_call(
            10,
            "get_transcript",
            json!({"meeting_id": "no_such_meeting"}),
        ),
        tool_call(11, "get_transcript", json!({"format": "docx"})),
        tool_call(12, "search_transcripts", json!({})),
        tool_call(13, "list_meetings", json!({"limit": "ten"})),
        tool_call(14, "list_history", json!({"since": "whenever"})),
        tool_call(15, "nope", json!({})),
        json!({"jsonrpc": "2.0", "id": 16, "method": "resources/read"}),
        tool_call(17, "get_transcript", json!({"from": "-5"})),
    ];
    let (responses, stderr) = mcp_session(&f, &requests);
    // 17 requests with ids + 1 parse error; the notification gets no reply.
    assert_eq!(responses.len(), 18, "stderr: {stderr}");
    let by_id = |id: i64| {
        responses
            .iter()
            .find(|r| r["id"] == json!(id))
            .unwrap_or_else(|| panic!("no response for id {id}"))
    };

    let init = by_id(1);
    assert_eq!(init["jsonrpc"], "2.0");
    assert_eq!(init["result"]["protocolVersion"], "2024-11-05");
    assert_eq!(init["result"]["capabilities"], json!({"tools": {}}));
    assert_eq!(init["result"]["serverInfo"]["name"], "phonon");
    assert_eq!(
        init["result"]["serverInfo"]["version"],
        env!("CARGO_PKG_VERSION")
    );
    assert_eq!(by_id(2)["result"], json!({}));

    let tools = by_id(3)["result"]["tools"].as_array().unwrap().clone();
    let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert_eq!(
        names,
        [
            "list_meetings",
            "get_transcript",
            "search_transcripts",
            "list_history",
            "search_history"
        ]
    );
    for t in &tools {
        assert_eq!(t["inputSchema"]["type"], "object");
        assert!(t["description"].as_str().unwrap().len() > 10);
    }
    assert_eq!(
        tools[1]["inputSchema"]["properties"]["format"]["enum"],
        json!(["md", "json", "srt", "vtt"])
    );
    assert_eq!(tools[2]["inputSchema"]["required"], json!(["query"]));

    let list: Value = serde_json::from_str(text_of(by_id(4))).unwrap();
    assert_eq!(list["meetings"].as_array().unwrap().len(), 2);
    assert_eq!(by_id(4)["result"]["isError"], false);

    let transcript: Value = serde_json::from_str(text_of(by_id(5))).unwrap();
    assert_eq!(transcript["version"], 1);
    assert_eq!(transcript["meeting"]["id"], "meeting_1794942000000");
    assert_eq!(transcript["turns"].as_array().unwrap().len(), 1);
    assert_eq!(transcript["range"]["from_seconds"], 60.0);

    let search: Value = serde_json::from_str(text_of(by_id(6))).unwrap();
    assert_eq!(search["results"][0]["meeting_id"], "meeting_1794942000000");

    let history: Value = serde_json::from_str(text_of(by_id(7))).unwrap();
    assert_eq!(history["entries"].as_array().unwrap().len(), 2);
    let hsearch: Value = serde_json::from_str(text_of(by_id(8))).unwrap();
    assert_eq!(hsearch["entries"][0]["id"], "rec_a", "raw fonon via alias");

    assert!(
        text_of(by_id(9)).starts_with("# Design review"),
        "default md of latest"
    );

    for id in [10, 11, 12, 13, 14, 17] {
        assert_eq!(
            by_id(id)["result"]["isError"],
            true,
            "id {id}: {}",
            by_id(id)
        );
    }
    assert_eq!(by_id(15)["error"]["code"], -32602);
    assert_eq!(by_id(16)["error"]["code"], -32601);
    let parse_error = responses.iter().find(|r| r["id"].is_null()).unwrap();
    assert_eq!(parse_error["error"]["code"], -32700);

    assert_eq!(before, snapshot(&f.data()), "mcp wrote to disk");
}
