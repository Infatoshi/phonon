//! Meeting transcription's share of the one resident speech model (SPEC "Meeting notes").
//!
//! The bar plans a recorded meeting into bounded spans of one track (about a minute each) and
//! sends them one at a time as `{"cmd":"meeting_span"}`. The engine holds them here and hands
//! the ASR sidecar at most one, and only while no dictation is under way: a key held down, a
//! stream open, a transcript or a correction still owed, or the last of these under
//! `QUIET_AFTER_DICTATION` ago. A dictation that starts while a span is in the sidecar waits for
//! that span alone (well under a second for a minute of audio on Metal Parakeet); the next one
//! waits for the dictation to finish.

use std::collections::{HashSet, VecDeque};
use std::time::{Duration, Instant};

/// How long after the last dictation event a meeting span may start: dictations come in runs.
pub const QUIET_AFTER_DICTATION: Duration = Duration::from_millis(1_000);

/// A dictation that never reports its end (a lost key up, a result that never came) stops
/// holding meeting work back after this long.
pub const DICTATION_STALE_AFTER: Duration = Duration::from_secs(120);

/// What the engine knows of dictation, for deciding when a meeting span may run.
#[derive(Debug, Default)]
pub struct DictationActivity {
    held_since: Option<Instant>,
    asr_pending: HashSet<String>,
    last: Option<Instant>,
}

impl DictationActivity {
    /// Key down (`prewarm`, `stream_start`).
    pub fn key_down(&mut self, now: Instant) {
        self.held_since = Some(now);
        self.last = Some(now);
    }

    /// Key up (`stream_finish`, `stream_stop`, the bar's `transcribe`).
    pub fn key_up(&mut self, now: Instant) {
        self.held_since = None;
        self.last = Some(now);
    }

    /// A dictation transcript was asked of the sidecar.
    pub fn asr_sent(&mut self, id: &str, now: Instant) {
        self.asr_pending.insert(id.to_string());
        self.last = Some(now);
    }

    /// The transcript (or an error) for `id` came back.
    pub fn asr_done(&mut self, id: &str, now: Instant) {
        if self.asr_pending.remove(id) {
            self.last = Some(now);
        }
    }

    /// Any other dictation event (a correction asked for or delivered).
    pub fn touch(&mut self, now: Instant) {
        self.last = Some(now);
    }

    /// No dictation is under way and none ended in the last `QUIET_AFTER_DICTATION`.
    pub fn quiet(&self, now: Instant) -> bool {
        let since_last = self
            .last
            .map_or(Duration::MAX, |last| now.saturating_duration_since(last));
        let held = self
            .held_since
            .is_some_and(|since| now.saturating_duration_since(since) < DICTATION_STALE_AFTER);
        let owed = !self.asr_pending.is_empty() && since_last < DICTATION_STALE_AFTER;
        !held && !owed && since_last >= QUIET_AFTER_DICTATION
    }
}

/// One span as the bar sent it.
#[derive(Debug, Clone, PartialEq)]
pub struct SpanRequest {
    pub id: String,
    pub chunks: serde_json::Value,
    pub start_sample: i64,
    pub end_sample: i64,
}

impl SpanRequest {
    /// `{"cmd":"meeting_span","id":..,"chunks":[{"path","start_sample"}],"start_sample":..,"end_sample":..}`
    pub fn from_command(v: &serde_json::Value) -> Result<Self, String> {
        let id = v
            .get("id")
            .and_then(|x| x.as_str())
            .filter(|id| !id.is_empty())
            .ok_or("meeting_span needs an id")?;
        let chunks = v
            .get("chunks")
            .filter(|c| c.as_array().is_some_and(|a| !a.is_empty()))
            .ok_or("meeting_span needs chunks")?;
        let start = v.get("start_sample").and_then(|x| x.as_i64()).unwrap_or(0);
        let end = v.get("end_sample").and_then(|x| x.as_i64()).unwrap_or(0);
        if end <= start {
            return Err("meeting_span is empty".into());
        }
        Ok(Self {
            id: id.to_string(),
            chunks: chunks.clone(),
            start_sample: start,
            end_sample: end,
        })
    }
}

/// Spans waiting for the model, and the one in the sidecar.
#[derive(Debug, Default)]
pub struct MeetingSpans {
    queue: VecDeque<SpanRequest>,
    in_flight: Option<String>,
}

impl MeetingSpans {
    /// Queue a span; one already queued under the same id is replaced (a bar that re-sent it).
    pub fn push(&mut self, request: SpanRequest) {
        if self.in_flight.as_deref() == Some(request.id.as_str()) {
            return; // already in the sidecar; its answer will come
        }
        self.queue.retain(|queued| queued.id != request.id);
        self.queue.push_back(request);
    }

    /// The next span to send, when nothing is in the sidecar and `idle` says the model is free.
    pub fn next(&mut self, idle: bool) -> Option<SpanRequest> {
        if self.in_flight.is_some() || !idle {
            return None;
        }
        let request = self.queue.pop_front()?;
        self.in_flight = Some(request.id.clone());
        Some(request)
    }

    /// The sidecar answered (or failed) `id`; true when it was the span in flight.
    pub fn finish(&mut self, id: Option<&str>) -> bool {
        if id.is_some() && self.in_flight.as_deref() == id {
            self.in_flight = None;
            return true;
        }
        false
    }

    pub fn in_flight(&self) -> Option<&str> {
        self.in_flight.as_deref()
    }

    pub fn queued(&self) -> usize {
        self.queue.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn span(id: &str) -> SpanRequest {
        SpanRequest {
            id: id.into(),
            chunks: json!([{"path": "/m/mic/0000.wav", "start_sample": 0}]),
            start_sample: 0,
            end_sample: 48_000,
        }
    }

    #[test]
    fn no_dictation_ever_is_quiet() {
        assert!(DictationActivity::default().quiet(Instant::now()));
    }

    #[test]
    fn a_held_key_and_an_owed_transcript_hold_meeting_work_back() {
        let t0 = Instant::now();
        let mut activity = DictationActivity::default();
        activity.key_down(t0);
        assert!(!activity.quiet(t0 + Duration::from_secs(30)));
        activity.key_up(t0 + Duration::from_secs(31));
        activity.asr_sent("p1", t0 + Duration::from_secs(31));
        assert!(!activity.quiet(t0 + Duration::from_secs(40)));
        activity.asr_done("p1", t0 + Duration::from_secs(41));
        // the run of dictation may go on: a second of quiet first
        assert!(!activity.quiet(t0 + Duration::from_millis(41_500)));
        assert!(activity.quiet(t0 + Duration::from_millis(42_000)));
    }

    #[test]
    fn a_correction_restarts_the_quiet_window() {
        let t0 = Instant::now();
        let mut activity = DictationActivity::default();
        activity.touch(t0);
        assert!(!activity.quiet(t0 + Duration::from_millis(999)));
        assert!(activity.quiet(t0 + QUIET_AFTER_DICTATION));
    }

    #[test]
    fn a_dictation_that_never_ends_stops_blocking_after_the_stale_limit() {
        let t0 = Instant::now();
        let mut activity = DictationActivity::default();
        activity.key_down(t0);
        assert!(!activity.quiet(t0 + DICTATION_STALE_AFTER - Duration::from_secs(1)));
        assert!(activity.quiet(t0 + DICTATION_STALE_AFTER));
        let mut owed = DictationActivity::default();
        owed.asr_sent("lost", t0);
        assert!(!owed.quiet(t0 + Duration::from_secs(60)));
        assert!(owed.quiet(t0 + DICTATION_STALE_AFTER));
    }

    #[test]
    fn an_unknown_transcript_does_not_count_as_activity() {
        let t0 = Instant::now();
        let mut activity = DictationActivity::default();
        activity.asr_done("meeting", t0);
        assert!(activity.quiet(t0));
    }

    #[test]
    fn one_span_at_a_time_and_only_when_idle() {
        let mut spans = MeetingSpans::default();
        spans.push(span("a"));
        spans.push(span("b"));
        assert_eq!(spans.next(false), None);
        assert_eq!(spans.next(true).map(|s| s.id), Some("a".into()));
        assert_eq!(spans.next(true), None, "a is still in the sidecar");
        assert!(!spans.finish(Some("b")));
        assert!(!spans.finish(None));
        assert!(spans.finish(Some("a")));
        assert_eq!(spans.next(true).map(|s| s.id), Some("b".into()));
        assert!(spans.finish(Some("b")));
        assert_eq!(spans.next(true), None);
    }

    #[test]
    fn a_re_sent_span_is_not_queued_twice() {
        let mut spans = MeetingSpans::default();
        spans.push(span("a"));
        spans.push(span("a"));
        assert_eq!(spans.queued(), 1);
        let _ = spans.next(true);
        spans.push(span("a"));
        assert_eq!(spans.queued(), 0);
        assert_eq!(spans.in_flight(), Some("a"));
    }

    #[test]
    fn span_commands_are_checked() {
        let ok = json!({"cmd":"meeting_span","id":"m1:mic:0","chunks":[{"path":"/x.wav","start_sample":0}],
                        "start_sample":10,"end_sample":20});
        let request = SpanRequest::from_command(&ok).unwrap();
        assert_eq!((request.start_sample, request.end_sample), (10, 20));
        for bad in [
            json!({"chunks":[{"path":"/x.wav"}],"start_sample":0,"end_sample":5}),
            json!({"id":"a","chunks":[],"start_sample":0,"end_sample":5}),
            json!({"id":"a","chunks":[{"path":"/x.wav"}],"start_sample":5,"end_sample":5}),
        ] {
            assert!(SpanRequest::from_command(&bad).is_err(), "{bad}");
        }
    }
}
