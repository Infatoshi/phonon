//! `phonon transcript` end to end, on a fixture `Meetings` directory under a temporary HOME.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

struct TempHome(PathBuf);

impl TempHome {
    fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let home = std::env::temp_dir().join(format!(
            "phonon-transcript-e2e-{tag}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&home).unwrap();
        Self(home)
    }
}

impl Drop for TempHome {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn cli_dumps_latest_meeting_transcript_text_and_json() {
    let temp = TempHome::new("cli");
    let meetings = temp.0.join("Library/Application Support/Phonon/Meetings");
    let meeting = meetings.join("meeting_1794959123259");
    fs::create_dir_all(&meeting).unwrap();
    fs::write(
        meeting.join("metadata.json"),
        r#"{
            "id": "meeting_1794959123259",
            "started_at": "2026-11-17T19:00:00Z",
            "ended_at": "2026-11-17T19:05:00Z",
            "app_name": "Discord"
        }"#,
    )
    .unwrap();
    fs::write(
        meeting.join("transcript.json"),
        r#"{
            "schema_version": 1,
            "model": "parakeet",
            "created_at": "2026-11-17T19:05:01Z",
            "turns": [
                {"start_seconds": 5.0, "end_seconds": 10.0, "speaker": "call", "text": "Old topic."},
                {"start_seconds": 250.0, "end_seconds": 265.0, "speaker": "me", "text": "Latest update."}
            ]
        }"#,
    )
    .unwrap();

    let text_out = Command::new(env!("CARGO_BIN_EXE_phonon"))
        .env("HOME", &temp.0)
        .args(["transcript", "--minutes", "1"])
        .output()
        .unwrap();
    assert!(
        text_out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&text_out.stderr)
    );
    let stdout = String::from_utf8_lossy(&text_out.stdout);
    assert!(stdout.contains("Meeting: Discord call"));
    assert!(stdout.contains("4:10 You: Latest update."));
    assert!(!stdout.contains("Old topic."));

    let json_out = Command::new(env!("CARGO_BIN_EXE_phonon"))
        .env("HOME", &temp.0)
        .args(["transcript", "--minutes", "1", "--json"])
        .output()
        .unwrap();
    assert!(json_out.status.success());
    let parsed: serde_json::Value = serde_json::from_slice(&json_out.stdout).unwrap();
    assert_eq!(parsed["id"], "meeting_1794959123259");
    assert_eq!(parsed["turns"].as_array().unwrap().len(), 1);
    assert_eq!(parsed["turns"][0]["speaker"], "me");
}
