//! Meeting transcription's share of the one resident speech model (SPEC "Meeting notes").
//!
//! The bar plans a recorded meeting into bounded spans of one track (about a minute each) and
//! sends them one at a time as `{"cmd":"meeting_span"}`. The engine holds them here and hands
//! the ASR sidecar at most one, and only while no dictation is under way: a key held down, a
//! stream open, a transcript or a correction still owed, or the last of these under
//! `QUIET_AFTER_DICTATION` ago. Each of these is a hold that goes stale after
//! `DICTATION_STALE_AFTER`, so a dictation end the engine never hears about delays meeting
//! work instead of stopping it. A dictation that starts while a span is in the sidecar waits for
//! that span alone (well under a second for a minute of audio on Metal Parakeet); the next one
//! waits for the dictation to finish.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// How long after the last dictation event a meeting span may start: dictations come in runs.
pub const QUIET_AFTER_DICTATION: Duration = Duration::from_millis(1_000);

/// Every hold stops holding meeting work back this long after it began, released or not: a
/// dictation whose end never reaches the engine (a lost key up, a stream the bar abandoned, a
/// result that never came) delays meeting transcripts, never stops them.
pub const DICTATION_STALE_AFTER: Duration = Duration::from_secs(120);

const KEY: &str = "key";

/// What the engine knows of dictation, for deciding when a meeting span may run: named holds,
/// each with the time it began, and the time of the last dictation event.
#[derive(Debug, Default)]
pub struct DictationActivity {
    holds: HashMap<String, Instant>,
    last: Option<Instant>,
}

impl DictationActivity {
    fn hold(&mut self, name: String, now: Instant) {
        // a new hold restarts its clock, so one leaked earlier never shortens a real one
        self.holds.insert(name, now);
        self.last = Some(now);
    }

    fn release(&mut self, name: &str, now: Instant) {
        if self.holds.remove(name).is_some() {
            self.last = Some(now);
        }
    }

    /// Key down (`prewarm`, `stream_start`).
    pub fn key_down(&mut self, now: Instant) {
        self.hold(KEY.into(), now);
    }

    /// Key up (`stream_finish`, `stream_stop`, `dictation_end`, the bar's `transcribe`).
    pub fn key_up(&mut self, now: Instant) {
        self.release(KEY, now);
        self.last = Some(now);
    }

    /// A dictation transcript was asked of the sidecar.
    pub fn asr_sent(&mut self, id: &str, now: Instant) {
        self.hold(format!("asr:{id}"), now);
    }

    /// The transcript (or an error) for `id` came back.
    pub fn asr_done(&mut self, id: &str, now: Instant) {
        self.release(&format!("asr:{id}"), now);
    }

    /// Mirror a piece of engine state (`stream`: a segment stream open, `finishing`, `polish`:
    /// corrections owed) as a hold. A hold already in place keeps the time it began, so state
    /// that never clears goes stale like any other hold.
    pub fn mirror(&mut self, name: &str, held: bool, now: Instant) {
        match (held, self.holds.contains_key(name)) {
            (true, false) => self.hold(name.to_string(), now),
            (false, true) => self.release(name, now),
            _ => {}
        }
    }

    /// Any other dictation event (a correction delivered).
    pub fn touch(&mut self, now: Instant) {
        self.last = Some(now);
    }

    /// Holds still in force (under `DICTATION_STALE_AFTER` old).
    pub fn live_holds(&self, now: Instant) -> usize {
        self.holds
            .values()
            .filter(|since| now.saturating_duration_since(**since) < DICTATION_STALE_AFTER)
            .count()
    }

    /// No hold in force, and no dictation event in the last `QUIET_AFTER_DICTATION`.
    pub fn quiet(&self, now: Instant) -> bool {
        let since_last = self
            .last
            .map_or(Duration::MAX, |last| now.saturating_duration_since(last));
        self.live_holds(now) == 0 && since_last >= QUIET_AFTER_DICTATION
    }
}

/// Tells the bar when dictation starts and stops holding meeting work back, for work the
/// engine does not run itself: the bar's meeting cleanup pauses its CPU process while a
/// dictation is under way (`{"type":"dictation","quiet":..}`). `update` returns the new
/// state only when it changed, and the first state ever seen, so the bar knows it from
/// the engine's first poll.
#[derive(Debug, Default)]
pub struct QuietWatch {
    last: Option<bool>,
}

impl QuietWatch {
    pub fn update(&mut self, quiet: bool) -> Option<bool> {
        if self.last == Some(quiet) {
            return None;
        }
        self.last = Some(quiet);
        Some(quiet)
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
    fn a_segment_stream_never_closed_goes_stale_however_often_it_is_seen() {
        // stream_start, then the bar abandons the dictation: no stream_finish or stream_stop
        // ever comes, and the engine's segment_streams keeps the entry (mirrored every pump)
        let t0 = Instant::now();
        let mut activity = DictationActivity::default();
        activity.key_down(t0);
        let mut at = t0;
        while at < t0 + DICTATION_STALE_AFTER {
            activity.mirror("stream", true, at);
            assert!(!activity.quiet(at));
            at += Duration::from_millis(250);
        }
        activity.mirror("stream", true, at);
        assert!(activity.quiet(at), "both holds are stale by now");
        assert_eq!(activity.live_holds(at), 0);
    }

    #[test]
    fn owed_corrections_and_a_finishing_stream_hold_until_they_clear_or_go_stale() {
        let t0 = Instant::now();
        let mut activity = DictationActivity::default();
        activity.mirror("polish", true, t0);
        activity.mirror("finishing", true, t0 + Duration::from_secs(1));
        assert!(!activity.quiet(t0 + Duration::from_secs(10)));
        activity.mirror("polish", false, t0 + Duration::from_secs(10));
        assert_eq!(activity.live_holds(t0 + Duration::from_secs(10)), 1);
        activity.mirror("finishing", false, t0 + Duration::from_secs(11));
        // released: the quiet window runs from the release
        assert!(!activity.quiet(t0 + Duration::from_millis(11_500)));
        assert!(activity.quiet(t0 + Duration::from_secs(12)));
        // a correction stuck for good stops blocking after the stale limit
        let mut stuck = DictationActivity::default();
        stuck.mirror("polish", true, t0);
        stuck.mirror("polish", true, t0 + Duration::from_secs(100));
        assert!(!stuck.quiet(t0 + Duration::from_secs(119)));
        assert!(stuck.quiet(t0 + DICTATION_STALE_AFTER));
    }

    #[test]
    fn a_key_down_after_a_leaked_one_holds_for_its_full_time() {
        let t0 = Instant::now();
        let mut activity = DictationActivity::default();
        activity.key_down(t0); // never released (a tap the bar dropped)
        let later = t0 + Duration::from_secs(300);
        assert!(activity.quiet(later));
        activity.key_down(later);
        assert!(!activity.quiet(later + Duration::from_secs(60)));
        activity.key_up(later + Duration::from_secs(61));
        assert!(activity.quiet(later + Duration::from_secs(62)));
    }

    #[test]
    fn an_unknown_transcript_does_not_count_as_activity() {
        let t0 = Instant::now();
        let mut activity = DictationActivity::default();
        activity.asr_done("meeting", t0);
        assert!(activity.quiet(t0));
    }

    #[test]
    fn the_quiet_watch_reports_the_first_state_and_then_only_changes() {
        let t0 = Instant::now();
        let mut activity = DictationActivity::default();
        let mut watch = QuietWatch::default();
        assert_eq!(watch.update(activity.quiet(t0)), Some(true), "first poll");
        assert_eq!(watch.update(activity.quiet(t0)), None);
        activity.key_down(t0);
        assert_eq!(
            watch.update(activity.quiet(t0)),
            Some(false),
            "key down pauses"
        );
        assert_eq!(
            watch.update(activity.quiet(t0 + Duration::from_secs(5))),
            None
        );
        activity.key_up(t0 + Duration::from_secs(6));
        // still held back for the quiet window after the key comes up
        assert_eq!(
            watch.update(activity.quiet(t0 + Duration::from_millis(6_500))),
            None
        );
        assert_eq!(
            watch.update(activity.quiet(t0 + Duration::from_secs(7))),
            Some(true),
            "a second after the last dictation event"
        );
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
