//! Inspect or transcribe a live or recent meeting transcript (`phonon transcript`).

use anyhow::{bail, Context, Result};
use phonon_asr::{AsrEvent, AsrSidecar};
use phonon_core::data::{app_support_dir, DictionaryFile};
use phonon_core::project_root;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

const TARGET_SPAN_SECONDS: f64 = 60.0;
const MAX_SPAN_SECONDS: f64 = 75.0;
const DEFAULT_SAMPLE_RATE: usize = 16_000;
const ASR_READY_TIMEOUT: Duration = Duration::from_secs(180);
const ASR_SPAN_TIMEOUT: Duration = Duration::from_secs(120);

/// Arguments for `phonon transcript`.
#[derive(Debug, Clone)]
pub struct TranscriptArgs {
    pub minutes: Option<f64>,
    pub meeting: Option<String>,
    pub json: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Speaker {
    Me,
    Call,
}

impl Speaker {
    pub fn label(self) -> &'static str {
        match self {
            Self::Me => "You",
            Self::Call => "Call",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Turn {
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub speaker: Speaker,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TimedSentence {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

impl TimedSentence {
    fn shifted(&self, delta_seconds: f64) -> Self {
        Self {
            start: self.start + delta_seconds,
            end: self.end + delta_seconds,
            text: self.text.clone(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct MeetingChunk {
    #[serde(default)]
    file: String,
    start_sample: i64,
    sample_count: i64,
}

#[derive(Debug, Deserialize)]
struct MeetingMetadata {
    id: String,
    #[serde(default)]
    started_at: String,
    #[serde(default)]
    ended_at: Option<String>,
    #[serde(default)]
    app_name: Option<String>,
    #[serde(default)]
    window_title: Option<String>,
    #[serde(default)]
    conference_url: Option<String>,
    #[serde(default = "default_sample_rate")]
    sample_rate: usize,
    #[serde(default)]
    mic_chunks: Vec<MeetingChunk>,
    #[serde(default)]
    system_chunks: Vec<MeetingChunk>,
    #[serde(default)]
    mic_start_offset_samples: Option<i64>,
    #[serde(default)]
    system_start_offset_samples: Option<i64>,
}

fn default_sample_rate() -> usize {
    DEFAULT_SAMPLE_RATE
}

#[derive(Debug, Deserialize)]
struct MeetingTranscriptFile {
    #[serde(default)]
    model: String,
    #[serde(default)]
    turns: Vec<Turn>,
}

#[derive(Debug, Default, Deserialize)]
struct TrackProgress {
    #[serde(default)]
    cursor: i64,
    #[serde(default)]
    sentences: Vec<TimedSentence>,
}

#[derive(Debug, Default, Deserialize)]
struct MeetingTranscriptProgress {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    mic: TrackProgress,
    #[serde(default)]
    system: TrackProgress,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct TranscriptOutput {
    pub id: String,
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub app: Option<String>,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conference_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub duration_seconds: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window_minutes: Option<f64>,
    pub source: String,
    pub turns: Vec<Turn>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkInput {
    pub path: PathBuf,
    pub start_sample: i64,
}

/// Transcribes a single track span into sentences whose timestamps are relative to `start_sample`.
pub trait SpanTranscriber {
    fn model_name(&self) -> Option<&str>;

    fn transcribe_span(
        &mut self,
        id: &str,
        chunks: &[ChunkInput],
        start_sample: i64,
        end_sample: i64,
    ) -> Result<Vec<TimedSentence>>;
}

/// Lazily spawns the ASR sidecar on the first span that needs live/catch-up transcription.
struct LazyAsrTranscriber {
    root: PathBuf,
    asr: Option<AsrSidecar>,
    wakeup: Option<mpsc::Receiver<String>>,
    model: Option<String>,
}

impl LazyAsrTranscriber {
    fn new(root: PathBuf) -> Self {
        Self {
            root,
            asr: None,
            wakeup: None,
            model: None,
        }
    }

    fn ensure_ready(&mut self) -> Result<()> {
        if self.asr.is_some() {
            return Ok(());
        }
        let mut asr = AsrSidecar::spawn(&self.root)?;
        let (tx, rx) = mpsc::channel();
        asr.set_waker(tx);

        let deadline = Instant::now() + ASR_READY_TIMEOUT;
        loop {
            for event in asr.poll() {
                match event {
                    AsrEvent::Ready { model, .. } => {
                        if !model.is_empty() {
                            self.model = Some(model);
                        }
                        if let Ok(dict) = DictionaryFile::load() {
                            let terms = dict.asr_terms();
                            if !terms.is_empty() {
                                let _ = asr.set_terms(&terms);
                            }
                        }
                        self.asr = Some(asr);
                        self.wakeup = Some(rx);
                        return Ok(());
                    }
                    AsrEvent::Error { msg, .. } => {
                        asr.shutdown();
                        bail!("ASR sidecar failed: {msg}");
                    }
                    _ => {}
                }
            }
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                asr.shutdown();
                bail!("timed out waiting for ASR sidecar to become ready");
            };
            let _ = rx.recv_timeout(remaining.min(Duration::from_millis(250)));
        }
    }
}

impl Drop for LazyAsrTranscriber {
    fn drop(&mut self) {
        if let Some(mut asr) = self.asr.take() {
            asr.shutdown();
        }
    }
}

impl SpanTranscriber for LazyAsrTranscriber {
    fn model_name(&self) -> Option<&str> {
        self.model.as_deref()
    }

    fn transcribe_span(
        &mut self,
        id: &str,
        chunks: &[ChunkInput],
        start_sample: i64,
        end_sample: i64,
    ) -> Result<Vec<TimedSentence>> {
        self.ensure_ready()?;
        let asr = self.asr.as_mut().context("ASR sidecar missing")?;
        let rx = self.wakeup.as_ref().context("ASR waker missing")?;

        let chunk_json = serde_json::Value::Array(
            chunks
                .iter()
                .map(|c| {
                    json!({
                        "path": c.path,
                        "start_sample": c.start_sample,
                    })
                })
                .collect(),
        );
        asr.transcribe_span(id, &chunk_json, start_sample, end_sample)?;

        let deadline = Instant::now() + ASR_SPAN_TIMEOUT;
        loop {
            for event in asr.poll() {
                match event {
                    AsrEvent::Span {
                        id: event_id, body, ..
                    } if event_id.as_deref() == Some(id) => {
                        if let Some(model) = body.get("model").and_then(|v| v.as_str()) {
                            if !model.is_empty() && model != "unknown" {
                                self.model = Some(model.to_string());
                            }
                        }
                        let raw_sentences =
                            body.get("sentences").cloned().unwrap_or_else(|| json!([]));
                        let sentences: Vec<TimedSentence> = serde_json::from_value(raw_sentences)
                            .context("decode ASR span sentences")?;
                        return Ok(sentences);
                    }
                    AsrEvent::Error { id: event_id, msg }
                        if event_id.is_none() || event_id.as_deref() == Some(id) =>
                    {
                        bail!("ASR span {id} failed: {msg}");
                    }
                    _ => {}
                }
            }
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                bail!("timed out waiting for ASR span {id}");
            };
            let _ = rx.recv_timeout(remaining.min(Duration::from_millis(250)));
        }
    }
}

pub fn run_transcript(args: TranscriptArgs) -> Result<()> {
    let meetings_root = default_meetings_root()?;
    let mut transcriber = LazyAsrTranscriber::new(project_root());
    let output = load_transcript(&meetings_root, &args, &mut transcriber)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        print!("{}", format_plain_text(&output));
    }
    Ok(())
}

fn default_meetings_root() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("PHONON_MEETINGS_DIR") {
        let trimmed = PathBuf::from(dir);
        if !trimmed.as_os_str().is_empty() {
            return Ok(trimmed);
        }
    }
    Ok(app_support_dir()?.join("Meetings"))
}

pub fn load_transcript(
    meetings_root: &Path,
    args: &TranscriptArgs,
    transcriber: &mut dyn SpanTranscriber,
) -> Result<TranscriptOutput> {
    if let Some(minutes) = args.minutes {
        if !minutes.is_finite() || minutes <= 0.0 {
            bail!("--minutes must be a positive number");
        }
    }
    let dir = resolve_meeting_dir(meetings_root, args.meeting.as_deref())?;
    let id = dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("meeting")
        .to_string();

    let metadata = read_json::<MeetingMetadata>(&dir.join("metadata.json"))?;
    let finished_json = read_json::<MeetingTranscriptFile>(&dir.join("transcript.json"))?;
    let plain_txt = fs::read_to_string(dir.join("transcript.txt")).ok();

    let rate = metadata
        .as_ref()
        .map(|m| m.sample_rate.max(1))
        .unwrap_or(DEFAULT_SAMPLE_RATE);
    let (mic_offset_samples, sys_offset_samples) =
        metadata.as_ref().map(timeline_offsets).unwrap_or((0, 0));
    let meta_duration = metadata
        .as_ref()
        .map(|m| metadata_duration_seconds(m, mic_offset_samples, sys_offset_samples, rate))
        .unwrap_or(0.0);

    // Fast path 1: completed transcript.json covers a finished meeting (or has no metadata).
    if let Some(transcript_file) = &finished_json {
        let live_recording = metadata.as_ref().is_some_and(|m| m.ended_at.is_none());
        if !live_recording {
            let max_turn = transcript_file
                .turns
                .iter()
                .map(|t| t.end_seconds)
                .fold(0.0_f64, f64::max);
            let duration = meta_duration.max(max_turn);
            let turns = filter_turns_by_minutes(&transcript_file.turns, duration, args.minutes);
            return Ok(build_output(
                &id,
                metadata.as_ref(),
                "complete",
                non_empty(&transcript_file.model),
                duration,
                args.minutes,
                "transcript.json",
                turns,
            ));
        }
    }

    // Fast path 2: plain transcript.txt only (no metadata.json / chunks on disk).
    if metadata.is_none() {
        if let Some(txt) = &plain_txt {
            let parsed = parse_transcript_txt(txt);
            let max_turn = parsed
                .turns
                .iter()
                .map(|t| t.end_seconds)
                .fold(0.0_f64, f64::max);
            let duration = parsed.duration_seconds.unwrap_or(max_turn).max(max_turn);
            let turns = filter_turns_by_minutes(&parsed.turns, duration, args.minutes);
            return Ok(TranscriptOutput {
                id,
                title: parsed.title.unwrap_or_else(|| "Meeting".into()),
                app: parsed.app,
                status: parsed.status.unwrap_or_else(|| "complete".into()),
                started_at: parsed.started_at,
                ended_at: parsed.ended_at,
                conference_url: parsed.conference_url,
                model: parsed.model,
                duration_seconds: duration,
                window_minutes: args.minutes,
                source: "transcript.txt".into(),
                turns,
            });
        }
        bail!("meeting {id} has neither metadata.json nor transcript files");
    }

    let meta = metadata.as_ref().expect("checked above");
    let progress = read_json::<MeetingTranscriptProgress>(&dir.join("transcript.partial.json"))?
        .unwrap_or_default();
    let cutoff_secs = args.minutes.map(|m| (meta_duration - m * 60.0).max(0.0));

    let mic_sentences = collect_track_sentences(
        &dir,
        "mic",
        &meta.mic_chunks,
        &progress.mic,
        mic_offset_samples,
        rate,
        cutoff_secs,
        transcriber,
    )?;
    let system_sentences = collect_track_sentences(
        &dir,
        "system",
        &meta.system_chunks,
        &progress.system,
        sys_offset_samples,
        rate,
        cutoff_secs,
        transcriber,
    )?;

    let mic_shifted: Vec<TimedSentence> = mic_sentences
        .iter()
        .map(|s| s.shifted(mic_offset_samples as f64 / rate as f64))
        .collect();
    let sys_shifted: Vec<TimedSentence> = system_sentences
        .iter()
        .map(|s| s.shifted(sys_offset_samples as f64 / rate as f64))
        .collect();
    let merged = merge_turns(&mic_shifted, &sys_shifted);
    let max_turn = merged.iter().map(|t| t.end_seconds).fold(0.0_f64, f64::max);
    let duration = meta_duration.max(max_turn);
    let turns = filter_turns_by_minutes(&merged, duration, args.minutes);

    let status = if meta.ended_at.is_some() {
        "transcribing"
    } else {
        "recording"
    };
    let model = transcriber
        .model_name()
        .map(str::to_string)
        .or_else(|| progress.model.as_deref().and_then(non_empty));
    let source = if progress.mic.cursor > 0 || progress.system.cursor > 0 {
        "partial+wav"
    } else {
        "wav"
    };

    Ok(build_output(
        &id,
        Some(meta),
        status,
        model,
        duration,
        args.minutes,
        source,
        turns,
    ))
}

#[allow(clippy::too_many_arguments)]
fn collect_track_sentences(
    meeting_dir: &Path,
    track_dir: &str,
    chunks: &[MeetingChunk],
    progress: &TrackProgress,
    track_offset_samples: i64,
    rate: usize,
    cutoff_timeline_secs: Option<f64>,
    transcriber: &mut dyn SpanTranscriber,
) -> Result<Vec<TimedSentence>> {
    let mut sentences = progress.sentences.clone();
    // Skip chunks strictly before the requested window so tail queries on long calls stay fast.
    let window_track_start = match cutoff_timeline_secs {
        Some(cutoff) => {
            let track_secs = (cutoff - track_offset_samples as f64 / rate as f64 - 1.0).max(0.0);
            (track_secs * rate as f64).floor() as i64
        }
        None => 0,
    };
    let mut cursor = progress.cursor.max(window_track_start);
    let spans = plan_uncovered_spans(meeting_dir, track_dir, chunks, cursor, rate);
    for (span_idx, span) in spans.into_iter().enumerate() {
        if span.end_sample <= cursor {
            continue;
        }
        let span_id = format!("{track_dir}-{span_idx}");
        let span_sentences = transcriber.transcribe_span(
            &span_id,
            &span.chunks,
            span.start_sample,
            span.end_sample,
        )?;
        let shift_secs = span.start_sample as f64 / rate as f64;
        for sentence in span_sentences {
            let trimmed = sentence.text.trim();
            if trimmed.is_empty() {
                continue;
            }
            sentences.push(TimedSentence {
                start: sentence.start + shift_secs,
                end: sentence.end + shift_secs,
                text: trimmed.to_string(),
            });
        }
        cursor = span.end_sample;
    }
    Ok(sentences)
}

#[derive(Debug)]
struct PlannedSpan {
    start_sample: i64,
    end_sample: i64,
    chunks: Vec<ChunkInput>,
}

fn plan_uncovered_spans(
    meeting_dir: &Path,
    track_dir: &str,
    chunks: &[MeetingChunk],
    cursor: i64,
    rate: usize,
) -> Vec<PlannedSpan> {
    let mut resolved: Vec<(i64, i64, PathBuf)> = chunks
        .iter()
        .filter(|c| c.sample_count > 0 && c.start_sample + c.sample_count > cursor)
        .filter_map(|c| {
            let path = resolve_chunk_path(meeting_dir, track_dir, &c.file)?;
            Some((c.start_sample, c.sample_count, path))
        })
        .collect();
    resolved.sort_by_key(|(start, _, _)| *start);

    let rate_f = rate.max(1) as f64;
    let target_samples = (TARGET_SPAN_SECONDS * rate_f).round() as i64;
    let max_samples = (MAX_SPAN_SECONDS * rate_f).round() as i64;

    let mut spans = Vec::new();
    let mut idx = 0;
    while idx < resolved.len() {
        let mut span_start = cursor.max(resolved[idx].0);
        while idx < resolved.len() && resolved[idx].0 + resolved[idx].1 <= span_start {
            idx += 1;
        }
        if idx >= resolved.len() {
            break;
        }
        span_start = span_start.max(resolved[idx].0);

        let mut run_idx = idx;
        let mut run_end = resolved[run_idx].0 + resolved[run_idx].1;
        while run_idx + 1 < resolved.len() && resolved[run_idx + 1].0 == run_end {
            run_idx += 1;
            run_end = resolved[run_idx].0 + resolved[run_idx].1;
        }

        let mut pos = span_start;
        while pos < run_end {
            let span_end = if run_end - pos > max_samples {
                (pos + target_samples).min(run_end)
            } else {
                run_end
            };
            let span_chunks: Vec<ChunkInput> = resolved[idx..=run_idx]
                .iter()
                .filter(|(s, len, _)| *s + *len > pos && *s < span_end)
                .map(|(s, _, p)| ChunkInput {
                    path: p.clone(),
                    start_sample: *s,
                })
                .collect();
            if !span_chunks.is_empty() && span_end > pos {
                spans.push(PlannedSpan {
                    start_sample: pos,
                    end_sample: span_end,
                    chunks: span_chunks,
                });
            }
            pos = span_end;
        }
        idx = run_idx + 1;
    }
    spans
}

fn resolve_chunk_path(meeting_dir: &Path, track_dir: &str, file: &str) -> Option<PathBuf> {
    let direct = meeting_dir.join(file);
    if direct.is_file() {
        return Some(direct);
    }
    let nested = meeting_dir.join(track_dir).join(file);
    if nested.is_file() {
        return Some(nested);
    }
    None
}

fn resolve_meeting_dir(meetings_root: &Path, meeting: Option<&str>) -> Result<PathBuf> {
    if !meetings_root.is_dir() {
        bail!("meetings folder not found at {}", meetings_root.display());
    }
    if let Some(raw_id) = meeting {
        let trimmed = raw_id.trim();
        if trimmed.is_empty()
            || trimmed == "."
            || trimmed == ".."
            || trimmed.contains('/')
            || trimmed.contains('\\')
        {
            bail!("invalid meeting id: {raw_id:?}");
        }
        let exact = meetings_root.join(trimmed);
        if is_meeting_dir(&exact) {
            return Ok(exact);
        }
        if !trimmed.starts_with("meeting_") {
            let prefixed = meetings_root.join(format!("meeting_{trimmed}"));
            if is_meeting_dir(&prefixed) {
                return Ok(prefixed);
            }
        }
        bail!("meeting {trimmed} not found in {}", meetings_root.display());
    }

    let mut seen = HashSet::new();
    let mut candidates: Vec<(i64, String, PathBuf)> = Vec::new();
    for entry in fs::read_dir(meetings_root)
        .with_context(|| format!("read {}", meetings_root.display()))?
        .flatten()
    {
        let path = entry.path();
        if !is_meeting_dir(&path) {
            continue;
        }
        if let Ok(canon) = path.canonicalize() {
            if !seen.insert(canon) {
                continue;
            }
        }
        let id = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        let rank = meeting_sort_timestamp(&path, &id);
        candidates.push((rank, id, path));
    }

    let latest_link = meetings_root.join("latest.txt");
    if let Ok(target) = fs::read_link(&latest_link) {
        let resolved = if target.is_absolute() {
            target
        } else {
            meetings_root.join(target)
        };
        if let Some(parent) = resolved.parent() {
            if is_meeting_dir(parent) {
                let is_new = parent
                    .canonicalize()
                    .map(|c| seen.insert(c))
                    .unwrap_or(true);
                if is_new {
                    let id = parent
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or_default()
                        .to_string();
                    let rank = meeting_sort_timestamp(parent, &id);
                    candidates.push((rank, id, parent.to_path_buf()));
                }
            }
        }
    }

    candidates.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
    candidates
        .into_iter()
        .next()
        .map(|(_, _, path)| path)
        .with_context(|| format!("no meetings found in {}", meetings_root.display()))
}

fn is_meeting_dir(path: &Path) -> bool {
    path.is_dir()
        && (path.join("metadata.json").is_file()
            || path.join("transcript.json").is_file()
            || path.join("transcript.txt").is_file())
}

fn meeting_sort_timestamp(dir: &Path, id: &str) -> i64 {
    if let Ok(Some(meta)) = read_json::<MeetingMetadata>(&dir.join("metadata.json")) {
        if let Some(ts) = parse_rfc3339_unix_secs(&meta.started_at) {
            return ts * 1_000;
        }
    }
    if let Some(rest) = id.strip_prefix("meeting_") {
        if let Ok(ms) = rest.parse::<i64>() {
            return ms;
        }
    }
    0
}

/// Parses RFC3339 timestamps (`YYYY-MM-DDTHH:MM:SS[.fff][Z|+HH:MM|-HH:MM]`) into Unix seconds.
fn parse_rfc3339_unix_secs(raw: &str) -> Option<i64> {
    let s = raw.trim();
    if s.len() < 20 {
        return None;
    }
    let year: i64 = s.get(0..4)?.parse().ok()?;
    let month: i64 = s.get(5..7)?.parse().ok()?;
    let day: i64 = s.get(8..10)?.parse().ok()?;
    let hour: i64 = s.get(11..13)?.parse().ok()?;
    let min: i64 = s.get(14..16)?.parse().ok()?;
    let sec: i64 = s.get(17..19)?.parse().ok()?;
    let mut rest = s.get(19..)?;
    if let Some(after_dot) = rest.strip_prefix('.') {
        let digits = after_dot.bytes().take_while(|b| b.is_ascii_digit()).count();
        rest = &after_dot[digits..];
    }
    let offset_secs: i64 = if rest == "Z" || rest == "z" {
        0
    } else if rest.len() == 6 && (rest.starts_with('+') || rest.starts_with('-')) {
        let sign = if rest.starts_with('+') { 1 } else { -1 };
        let oh: i64 = rest.get(1..3)?.parse().ok()?;
        let om: i64 = rest.get(4..6)?.parse().ok()?;
        sign * (oh * 3600 + om * 60)
    } else {
        return None;
    };

    let days = days_from_civil(year, month, day)?;
    Some(days * 86_400 + hour * 3600 + min * 60 + sec - offset_secs)
}

fn days_from_civil(year: i64, month: i64, day: i64) -> Option<i64> {
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    let y = year - if month <= 2 { 1 } else { 0 };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let m = month + if month > 2 { -3 } else { 9 };
    let doy = (153 * m + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(era * 146_097 + doe - 719_468)
}

fn timeline_offsets(metadata: &MeetingMetadata) -> (i64, i64) {
    let mic = metadata.mic_start_offset_samples.unwrap_or(0);
    let system = metadata.system_start_offset_samples.unwrap_or(0);
    let base = mic.min(system);
    (mic - base, system - base)
}

fn metadata_duration_seconds(
    metadata: &MeetingMetadata,
    mic_offset: i64,
    system_offset: i64,
    rate: usize,
) -> f64 {
    let mic_end = metadata
        .mic_chunks
        .iter()
        .map(|c| c.start_sample + c.sample_count)
        .max()
        .unwrap_or(0);
    let sys_end = metadata
        .system_chunks
        .iter()
        .map(|c| c.start_sample + c.sample_count)
        .max()
        .unwrap_or(0);
    let mic_secs = if metadata.mic_chunks.is_empty() {
        0.0
    } else {
        (mic_offset + mic_end) as f64 / rate as f64
    };
    let sys_secs = if metadata.system_chunks.is_empty() {
        0.0
    } else {
        (system_offset + sys_end) as f64 / rate as f64
    };
    mic_secs.max(sys_secs)
}

fn merge_turns(mic: &[TimedSentence], call: &[TimedSentence]) -> Vec<Turn> {
    let mut tagged: Vec<(usize, Speaker, &TimedSentence)> = mic
        .iter()
        .enumerate()
        .map(|(i, s)| (i, Speaker::Me, s))
        .chain(
            call.iter()
                .enumerate()
                .map(|(i, s)| (mic.len() + i, Speaker::Call, s)),
        )
        .collect();
    tagged.sort_by(|a, b| a.2.start.total_cmp(&b.2.start).then_with(|| a.0.cmp(&b.0)));

    let mut turns: Vec<Turn> = Vec::new();
    for (_, speaker, sentence) in tagged {
        let text = sentence.text.trim();
        if text.is_empty() {
            continue;
        }
        if let Some(last) = turns.last_mut() {
            if last.speaker == speaker {
                last.text.push(' ');
                last.text.push_str(text);
                last.end_seconds = last.end_seconds.max(sentence.end);
                continue;
            }
        }
        turns.push(Turn {
            start_seconds: sentence.start,
            end_seconds: sentence.end,
            speaker,
            text: text.to_string(),
        });
    }
    turns
}

fn filter_turns_by_minutes(
    turns: &[Turn],
    total_duration_seconds: f64,
    minutes: Option<f64>,
) -> Vec<Turn> {
    let Some(minutes) = minutes else {
        return turns.to_vec();
    };
    let cutoff = (total_duration_seconds - minutes * 60.0).max(0.0);
    turns
        .iter()
        .filter(|turn| turn.end_seconds >= cutoff)
        .cloned()
        .collect()
}

fn meeting_title(metadata: &MeetingMetadata) -> String {
    if let Some(title) = metadata
        .window_title
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        return title.to_string();
    }
    if let Some(app) = metadata
        .app_name
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        return if app == "Manual note" {
            app.to_string()
        } else {
            format!("{app} call")
        };
    }
    "Meeting".to_string()
}

#[allow(clippy::too_many_arguments)]
fn build_output(
    id: &str,
    metadata: Option<&MeetingMetadata>,
    status: &str,
    model: Option<String>,
    duration_seconds: f64,
    window_minutes: Option<f64>,
    source: &str,
    turns: Vec<Turn>,
) -> TranscriptOutput {
    let (title, app, started_at, ended_at, conference_url, resolved_id) = match metadata {
        Some(m) => (
            meeting_title(m),
            m.app_name.as_deref().and_then(non_empty),
            non_empty(&m.started_at),
            m.ended_at.as_deref().and_then(non_empty),
            m.conference_url.as_deref().and_then(non_empty),
            if m.id.is_empty() {
                id.to_string()
            } else {
                m.id.clone()
            },
        ),
        None => (
            "Meeting".to_string(),
            None,
            None,
            None,
            None,
            id.to_string(),
        ),
    };
    TranscriptOutput {
        id: resolved_id,
        title,
        app,
        status: status.to_string(),
        started_at,
        ended_at,
        conference_url,
        model,
        duration_seconds,
        window_minutes,
        source: source.to_string(),
        turns,
    }
}

#[derive(Default)]
struct ParsedTxt {
    title: Option<String>,
    app: Option<String>,
    status: Option<String>,
    started_at: Option<String>,
    ended_at: Option<String>,
    conference_url: Option<String>,
    model: Option<String>,
    duration_seconds: Option<f64>,
    turns: Vec<Turn>,
}

fn parse_transcript_txt(content: &str) -> ParsedTxt {
    let mut parsed = ParsedTxt::default();
    let (header_part, body_part) = content.split_once("\n---\n").unwrap_or(("", content));

    let mut headers = BTreeMap::new();
    for line in header_part.lines() {
        if let Some((k, v)) = line.split_once(':') {
            headers.insert(k.trim().to_string(), v.trim().to_string());
        }
    }
    parsed.title = headers.remove("Meeting");
    parsed.app = headers.remove("App");
    parsed.status = headers.remove("Status");
    parsed.started_at = headers.remove("Started");
    parsed.ended_at = headers.remove("Ended");
    parsed.conference_url = headers.remove("URL");
    parsed.model = headers.remove("Model");

    for block in body_part.split("\n\n") {
        let trimmed = block.trim();
        if trimmed.is_empty() || trimmed.starts_with('(') {
            continue;
        }
        let Some((ts_str, rest)) = trimmed.split_once(' ') else {
            continue;
        };
        let Some(start) = parse_clock(ts_str) else {
            continue;
        };
        let (speaker, text) = if let Some(t) = rest.strip_prefix("You:") {
            (Speaker::Me, t.trim())
        } else if let Some(t) = rest.strip_prefix("Call:") {
            (Speaker::Call, t.trim())
        } else {
            continue;
        };
        if !text.is_empty() {
            parsed.turns.push(Turn {
                start_seconds: start,
                end_seconds: start,
                speaker,
                text: text.to_string(),
            });
        }
    }
    parsed.duration_seconds = parsed.turns.iter().map(|t| t.end_seconds).reduce(f64::max);
    parsed
}

fn parse_clock(raw: &str) -> Option<f64> {
    let parts: Vec<&str> = raw.split(':').collect();
    match parts.as_slice() {
        [m, s] => Some(m.parse::<f64>().ok()? * 60.0 + s.parse::<f64>().ok()?),
        [h, m, s] => Some(
            h.parse::<f64>().ok()? * 3600.0
                + m.parse::<f64>().ok()? * 60.0
                + s.parse::<f64>().ok()?,
        ),
        _ => None,
    }
}

pub fn format_clock(seconds: f64) -> String {
    let total = seconds.max(0.0) as u64;
    if total >= 3_600 {
        format!(
            "{}:{:02}:{:02}",
            total / 3_600,
            (total % 3_600) / 60,
            total % 60
        )
    } else {
        format!("{}:{:02}", total / 60, total % 60)
    }
}

pub fn format_plain_text(output: &TranscriptOutput) -> String {
    let mut lines = Vec::new();
    lines.push(format!("Meeting: {}", output.title));
    if let Some(app) = &output.app {
        lines.push(format!("App: {app}"));
    }
    if let Some(started) = &output.started_at {
        lines.push(format!("Started: {started}"));
    }
    if let Some(ended) = &output.ended_at {
        lines.push(format!("Ended: {ended}"));
    }
    if let Some(url) = &output.conference_url {
        lines.push(format!("URL: {url}"));
    }
    lines.push(format!("Status: {}", output.status));
    if let Some(minutes) = output.window_minutes {
        lines.push(format!("Window: last {minutes} min"));
    }
    if let Some(model) = &output.model {
        lines.push(format!("Model: {model}"));
    }

    let body = if output.turns.is_empty() {
        if output.status == "recording" {
            "(no speech detected in window)".to_string()
        } else {
            "(no speech)".to_string()
        }
    } else {
        output
            .turns
            .iter()
            .map(|t| {
                format!(
                    "{} {}: {}",
                    format_clock(t.start_seconds),
                    t.speaker.label(),
                    t.text
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    };

    format!("{}\n\n---\n\n{body}\n", lines.join("\n"))
}

fn non_empty(s: &str) -> Option<String> {
    let trimmed = s.trim();
    (!trimmed.is_empty() && trimmed != "unknown").then(|| trimmed.to_string())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Option<T>> {
    if !path.is_file() {
        return Ok(None);
    }
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let value =
        serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
    Ok(Some(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "phonon-transcript-test-{tag}-{}-{unique}",
                std::process::id()
            ));
            fs::create_dir_all(&path).expect("mkdir");
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[derive(Debug, Clone)]
    struct RecordedSpanCall {
        id: String,
        start_sample: i64,
        chunks: Vec<ChunkInput>,
    }

    #[derive(Default)]
    struct MockTranscriber {
        calls: RefCell<Vec<RecordedSpanCall>>,
        replies: BTreeMap<String, Vec<TimedSentence>>,
    }

    impl SpanTranscriber for MockTranscriber {
        fn model_name(&self) -> Option<&str> {
            Some("parakeet-test")
        }

        fn transcribe_span(
            &mut self,
            id: &str,
            chunks: &[ChunkInput],
            start_sample: i64,
            _end_sample: i64,
        ) -> Result<Vec<TimedSentence>> {
            self.calls.borrow_mut().push(RecordedSpanCall {
                id: id.to_string(),
                start_sample,
                chunks: chunks.to_vec(),
            });
            Ok(self.replies.get(id).cloned().unwrap_or_default())
        }
    }

    fn write_dummy_wav(path: &Path) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("mkdir");
        }
        fs::write(path, b"RIFF0000WAVE").expect("write wav");
    }

    #[test]
    fn reads_finished_transcript_json_with_minutes_filter_without_asr() {
        let tmp = TempDir::new("finished");
        let old_dir = tmp.0.join("meeting_1000");
        let new_dir = tmp.0.join("meeting_2000");
        fs::create_dir_all(&old_dir).unwrap();
        fs::create_dir_all(&new_dir).unwrap();

        fs::write(
            old_dir.join("metadata.json"),
            r#"{"id":"meeting_1000","started_at":"2026-11-17T10:00:00Z","ended_at":"2026-11-17T10:05:00Z","app_name":"Zoom"}"#,
        )
        .unwrap();
        fs::write(
            old_dir.join("transcript.json"),
            r#"{"schema_version":1,"model":"parakeet","created_at":"2026-11-17T10:05:01Z","turns":[{"start_seconds":5.0,"end_seconds":8.0,"speaker":"me","text":"Old call"}]}"#,
        )
        .unwrap();

        fs::write(
            new_dir.join("metadata.json"),
            r#"{"id":"meeting_2000","started_at":"2026-11-17T18:00:00Z","ended_at":"2026-11-17T18:10:00Z","app_name":"Discord","window_title":"Design review"}"#,
        )
        .unwrap();
        fs::write(
            new_dir.join("transcript.json"),
            r#"{
                "schema_version": 1,
                "model": "parakeet",
                "created_at": "2026-11-17T18:10:01Z",
                "turns": [
                    {"start_seconds": 10.0, "end_seconds": 20.0, "speaker": "me", "text": "Early intro."},
                    {"start_seconds": 490.0, "end_seconds": 500.0, "speaker": "call", "text": "Ship the CLI."},
                    {"start_seconds": 520.0, "end_seconds": 530.0, "speaker": "me", "text": "On it now."}
                ]
            }"#,
        )
        .unwrap();

        let mut mock = MockTranscriber::default();
        let out = load_transcript(
            &tmp.0,
            &TranscriptArgs {
                minutes: Some(2.0),
                meeting: None,
                json: false,
            },
            &mut mock,
        )
        .unwrap();

        assert!(
            mock.calls.borrow().is_empty(),
            "must not spawn ASR for completed meeting"
        );
        assert_eq!(out.id, "meeting_2000");
        assert_eq!(out.title, "Design review");
        assert_eq!(out.status, "complete");
        assert_eq!(out.turns.len(), 2);
        assert_eq!(out.turns[0].speaker, Speaker::Call);
        assert_eq!(out.turns[0].text, "Ship the CLI.");
        assert_eq!(out.turns[1].text, "On it now.");

        let rendered = format_plain_text(&out);
        assert!(rendered.contains("Window: last 2 min"));
        assert!(rendered.contains("8:10 Call: Ship the CLI."));
        assert!(rendered.contains("8:40 You: On it now."));
        assert!(!rendered.contains("Early intro."));
    }

    #[test]
    fn live_v103_meeting_transcribes_only_tail_window_chunks_and_merges_speakers() {
        let tmp = TempDir::new("live-wav");
        let dir = tmp.0.join("meeting_1794959123259");
        fs::create_dir_all(&dir).unwrap();

        // 120s total call in 60s chunks (0..60s skipped by --minutes 1, 60..120s transcribed).
        write_dummy_wav(&dir.join("mic/0000.wav"));
        write_dummy_wav(&dir.join("mic/0001.wav"));
        write_dummy_wav(&dir.join("system/0000.wav"));
        write_dummy_wav(&dir.join("system/0001.wav"));

        fs::write(
            dir.join("metadata.json"),
            r#"{
                "id": "meeting_1794959123259",
                "started_at": "2026-11-17T19:00:00Z",
                "app_name": "Discord",
                "sample_rate": 16000,
                "mic_start_offset_samples": 16000,
                "system_start_offset_samples": 0,
                "mic_chunks": [
                    {"index": 0, "file": "0000.wav", "start_sample": 0, "sample_count": 960000},
                    {"index": 1, "file": "0001.wav", "start_sample": 960000, "sample_count": 960000}
                ],
                "system_chunks": [
                    {"index": 0, "file": "0000.wav", "start_sample": 0, "sample_count": 960000},
                    {"index": 1, "file": "0001.wav", "start_sample": 960000, "sample_count": 960000}
                ]
            }"#,
        )
        .unwrap();

        let mut mock = MockTranscriber::default();
        mock.replies.insert(
            "mic-0".into(),
            vec![
                TimedSentence {
                    start: 5.0,
                    end: 7.0,
                    text: "Looks good.".into(),
                },
                TimedSentence {
                    start: 8.0,
                    end: 10.0,
                    text: "Shipping next.".into(),
                },
            ],
        );
        mock.replies.insert(
            "system-0".into(),
            vec![TimedSentence {
                start: 1.0,
                end: 4.0,
                text: "Let us check Discord audio.".into(),
            }],
        );

        let out = load_transcript(
            &tmp.0,
            &TranscriptArgs {
                minutes: Some(1.0),
                meeting: None,
                json: true,
            },
            &mut mock,
        )
        .unwrap();

        let calls = mock.calls.borrow();
        assert_eq!(calls.len(), 2);
        // Mic has +1s start offset + 1s boundary pad (944_000), so it spans end of 0000.wav + 0001.wav.
        assert_eq!(calls[0].id, "mic-0");
        assert_eq!(calls[0].start_sample, 944_000);
        assert_eq!(calls[0].chunks.len(), 2);
        assert!(calls[0].chunks[0].path.ends_with("mic/0000.wav"));
        assert!(calls[0].chunks[1].path.ends_with("mic/0001.wav"));

        // System starts right at 960_000, so chunk 0000.wav is completely skipped.
        assert_eq!(calls[1].id, "system-0");
        assert_eq!(calls[1].start_sample, 960_000);
        assert_eq!(calls[1].chunks.len(), 1);
        assert!(calls[1].chunks[0].path.ends_with("system/0001.wav"));

        assert_eq!(out.status, "recording");
        assert_eq!(out.source, "wav");
        assert_eq!(out.model.as_deref(), Some("parakeet-test"));
        // Call speaks at ~61s, then You merges consecutive sentences starting around ~66s.
        assert_eq!(out.turns.len(), 2);
        assert_eq!(out.turns[0].speaker, Speaker::Call);
        assert_eq!(out.turns[0].text, "Let us check Discord audio.");
        assert_eq!(out.turns[1].speaker, Speaker::Me);
        assert_eq!(out.turns[1].text, "Looks good. Shipping next.");
    }

    #[test]
    fn reuses_partial_transcript_progress_without_retranscribing_covered_chunks() {
        let tmp = TempDir::new("partial-reuse");
        let dir = tmp.0.join("meeting_3000");
        fs::create_dir_all(&dir).unwrap();
        write_dummy_wav(&dir.join("mic/0000.wav"));
        write_dummy_wav(&dir.join("mic/0001.wav"));

        fs::write(
            dir.join("metadata.json"),
            r#"{
                "id": "meeting_3000",
                "started_at": "2026-11-17T19:10:00Z",
                "app_name": "Zoom",
                "sample_rate": 16000,
                "mic_chunks": [
                    {"index": 0, "file": "0000.wav", "start_sample": 0, "sample_count": 320000},
                    {"index": 1, "file": "0001.wav", "start_sample": 320000, "sample_count": 80000}
                ],
                "system_chunks": []
            }"#,
        )
        .unwrap();
        fs::write(
            dir.join("transcript.partial.json"),
            r#"{
                "schema_version": 1,
                "model": "parakeet",
                "mic": {
                    "cursor": 320000,
                    "done": false,
                    "sentences": [{"start": 2.0, "end": 5.0, "text": "Already covered."}]
                },
                "system": {"cursor": 0, "done": false, "sentences": []}
            }"#,
        )
        .unwrap();

        let mut mock = MockTranscriber::default();
        mock.replies.insert(
            "mic-0".into(),
            vec![TimedSentence {
                start: 1.0,
                end: 3.0,
                text: "Tail chunk.".into(),
            }],
        );

        let out = load_transcript(
            &tmp.0,
            &TranscriptArgs {
                minutes: None,
                meeting: Some("3000".into()),
                json: false,
            },
            &mut mock,
        )
        .unwrap();

        let calls = mock.calls.borrow();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].start_sample, 320_000);
        assert_eq!(calls[0].chunks.len(), 1);
        assert_eq!(out.source, "partial+wav");
        assert_eq!(out.turns.len(), 1);
        assert_eq!(out.turns[0].text, "Already covered. Tail chunk.");
    }

    #[test]
    fn reads_plain_transcript_txt_when_only_latest_text_exists() {
        let tmp = TempDir::new("txt-only");
        let dir = tmp.0.join("meeting_4000");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("transcript.txt"),
            "Meeting: Weekly sync\nApp: Meet\nStatus: recording\nModel: parakeet\n\n---\n\n0:05 You: Hello team.\n\n2:30 Call: Ready to ship.\n",
        )
        .unwrap();

        let mut mock = MockTranscriber::default();
        let out = load_transcript(
            &tmp.0,
            &TranscriptArgs {
                minutes: Some(1.0),
                meeting: None,
                json: false,
            },
            &mut mock,
        )
        .unwrap();

        assert!(mock.calls.borrow().is_empty());
        assert_eq!(out.source, "transcript.txt");
        assert_eq!(out.turns.len(), 1);
        assert_eq!(out.turns[0].speaker, Speaker::Call);
        assert_eq!(out.turns[0].text, "Ready to ship.");
    }

    #[test]
    fn errors_clearly_on_missing_or_invalid_meeting_id() {
        let tmp = TempDir::new("missing");
        let mut mock = MockTranscriber::default();
        assert!(load_transcript(
            &tmp.0,
            &TranscriptArgs {
                minutes: None,
                meeting: Some("../etc".into()),
                json: false,
            },
            &mut mock,
        )
        .is_err());
        assert!(load_transcript(
            &tmp.0,
            &TranscriptArgs {
                minutes: Some(0.0),
                meeting: None,
                json: false,
            },
            &mut mock,
        )
        .is_err());
    }
}
