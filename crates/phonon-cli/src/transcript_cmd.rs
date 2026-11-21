//! Inspect or transcribe a live or recent meeting transcript (`phonon transcript`).

use anyhow::{anyhow, bail, Context, Result};
use phonon_asr::{AsrEvent, AsrSidecar};
use phonon_core::data::{app_support_dir, DictionaryFile};
use phonon_core::project_root;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

const TARGET_SPAN_SECONDS: f64 = 60.0;
const MAX_SPAN_SECONDS: f64 = 75.0;
const DEFAULT_SAMPLE_RATE: usize = 16_000;
const ASR_READY_TIMEOUT: Duration = Duration::from_secs(180);
const ASR_SPAN_TIMEOUT: Duration = Duration::from_secs(120);

/// Arguments for `phonon transcript`.
#[derive(Debug, Clone, Default)]
pub struct TranscriptArgs {
    pub minutes: Option<f64>,
    pub meeting: Option<String>,
    pub json: bool,
    /// Only turns overlapping `[from, to]` seconds into the meeting.
    pub from: Option<f64>,
    pub to: Option<f64>,
    /// Output format other than the legacy text/JSON (`--md`, `--srt`, `--vtt`).
    pub format: Option<TranscriptFormat>,
}

/// Rendered transcript formats shared by `phonon transcript`, `phonon meetings show` and `phonon mcp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscriptFormat {
    Text,
    Markdown,
    Json,
    Srt,
    Vtt,
}

impl TranscriptFormat {
    pub fn parse(raw: &str) -> Result<Self> {
        Ok(match raw.trim().to_ascii_lowercase().as_str() {
            "text" | "txt" | "plain" => Self::Text,
            "md" | "markdown" => Self::Markdown,
            "json" => Self::Json,
            "srt" => Self::Srt,
            "vtt" | "webvtt" => Self::Vtt,
            other => bail!("unknown transcript format {other:?} (use md, json, srt or vtt)"),
        })
    }
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

/// Counts the spans actually sent to speech recognition, so only real transcription work
/// updates the live artifacts.
struct CountingTranscriber<'a> {
    inner: &'a mut dyn SpanTranscriber,
    spans: usize,
}

impl SpanTranscriber for CountingTranscriber<'_> {
    fn model_name(&self) -> Option<&str> {
        self.inner.model_name()
    }

    fn transcribe_span(
        &mut self,
        id: &str,
        chunks: &[ChunkInput],
        start_sample: i64,
        end_sample: i64,
    ) -> Result<Vec<TimedSentence>> {
        self.spans += 1;
        self.inner
            .transcribe_span(id, chunks, start_sample, end_sample)
    }
}

/// Never runs speech recognition: read-only queries see only what is already on disk.
struct NoAsr;

impl SpanTranscriber for NoAsr {
    fn model_name(&self) -> Option<&str> {
        None
    }

    fn transcribe_span(
        &mut self,
        _id: &str,
        _chunks: &[ChunkInput],
        _start_sample: i64,
        _end_sample: i64,
    ) -> Result<Vec<TimedSentence>> {
        Ok(Vec::new())
    }
}

pub fn run_transcript(args: TranscriptArgs) -> Result<()> {
    let meetings_root = default_meetings_root()?;
    let mut asr = LazyAsrTranscriber::new(project_root());
    let mut transcriber = CountingTranscriber {
        inner: &mut asr,
        spans: 0,
    };
    let mut output = load_transcript(&meetings_root, &args, &mut transcriber)?;
    if transcriber.spans > 0 {
        // The transcript was brought up to date: refresh the generic live-stream artifacts.
        if let Err(err) = write_live_artifacts(&meetings_root, &output) {
            eprintln!("phonon: could not update live transcript artifacts: {err:#}");
        }
    }
    let range = validate_range(args.from, args.to)?;
    if range.is_some() {
        output.turns = filter_turns_by_range(&output.turns, args.from, args.to);
    }
    let rendered = match args.format {
        Some(format) if format != TranscriptFormat::Json => {
            render_transcript(&output, format, range)?
        }
        _ if args.json => format!("{}\n", serde_json::to_string_pretty(&output)?),
        _ => format_plain_text(&output),
    };
    print!("{rendered}");
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
    load_transcript_at(&dir, args, transcriber)
}

fn load_transcript_at(
    dir: &Path,
    args: &TranscriptArgs,
    transcriber: &mut dyn SpanTranscriber,
) -> Result<TranscriptOutput> {
    let dir = dir.to_path_buf();
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

/// Minimum length of an id prefix accepted as a meeting selector.
const MIN_ID_PREFIX: usize = 6;

fn resolve_meeting_dir(meetings_root: &Path, meeting: Option<&str>) -> Result<PathBuf> {
    if !meetings_root.is_dir() {
        bail!("meetings folder not found at {}", meetings_root.display());
    }
    let meeting = meeting.filter(|m| !m.trim().eq_ignore_ascii_case("latest"));
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
        if trimmed.len() >= MIN_ID_PREFIX {
            let bare = trimmed.strip_prefix("meeting_").unwrap_or(trimmed);
            let matches: Vec<(i64, String, PathBuf)> = list_meeting_dirs(meetings_root)?
                .into_iter()
                .filter(|(_, id, _)| {
                    id.starts_with(trimmed)
                        || (bare.len() >= MIN_ID_PREFIX
                            && id.strip_prefix("meeting_").unwrap_or(id).starts_with(bare))
                })
                .collect();
            match matches.len() {
                1 => return Ok(matches.into_iter().next().expect("one match").2),
                0 => {}
                n => bail!(
                    "meeting prefix {trimmed} is ambiguous ({n} meetings match: {})",
                    matches
                        .iter()
                        .take(5)
                        .map(|(_, id, _)| id.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            }
        }
        bail!("meeting {trimmed} not found in {}", meetings_root.display());
    }

    list_meeting_dirs(meetings_root)?
        .into_iter()
        .next()
        .map(|(_, _, path)| path)
        .with_context(|| format!("no meetings found in {}", meetings_root.display()))
}

/// Every meeting folder (plus the `latest.txt` link target), newest first, as
/// `(sort timestamp ms, id, path)`.
fn list_meeting_dirs(meetings_root: &Path) -> Result<Vec<(i64, String, PathBuf)>> {
    if !meetings_root.is_dir() {
        return Ok(Vec::new());
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
    Ok(candidates)
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

// ---------------------------------------------------------------------------
// Time parsing
// ---------------------------------------------------------------------------

fn parse_plain_number(raw: &str) -> Option<f64> {
    let dots = raw.bytes().filter(|&b| b == b'.').count();
    if raw.is_empty()
        || raw == "."
        || dots > 1
        || !raw.bytes().all(|b| b.is_ascii_digit() || b == b'.')
    {
        return None;
    }
    raw.parse::<f64>()
        .ok()
        .filter(|v| v.is_finite() && *v >= 0.0)
}

fn parse_whole(raw: &str) -> Option<u64> {
    if raw.is_empty() || !raw.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    raw.parse().ok()
}

/// Parses an offset into a meeting: `mm:ss` (minutes may exceed 59, e.g. `113:03`),
/// `h:mm:ss`, plain seconds (`90`, `1.5`), `90s`, `1.5m` or `2h`.
pub fn parse_offset_seconds(s: &str) -> Result<f64> {
    let raw = s.trim().to_ascii_lowercase();
    let invalid = || anyhow!("invalid time offset {s:?} (use mm:ss, h:mm:ss, 90, 90s or 1.5m)");
    let value = if raw.contains(':') {
        let parts: Vec<&str> = raw.split(':').collect();
        match parts.as_slice() {
            [m, sec] => {
                let m = parse_whole(m).ok_or_else(invalid)?;
                let sec = parse_plain_number(sec)
                    .filter(|v| *v < 60.0)
                    .ok_or_else(invalid)?;
                m as f64 * 60.0 + sec
            }
            [h, m, sec] => {
                let h = parse_whole(h).ok_or_else(invalid)?;
                let m = parse_whole(m).filter(|v| *v < 60).ok_or_else(invalid)?;
                let sec = parse_plain_number(sec)
                    .filter(|v| *v < 60.0)
                    .ok_or_else(invalid)?;
                h as f64 * 3600.0 + m as f64 * 60.0 + sec
            }
            _ => return Err(invalid()),
        }
    } else if let Some(n) = raw.strip_suffix('s') {
        parse_plain_number(n).ok_or_else(invalid)?
    } else if let Some(n) = raw.strip_suffix('m') {
        parse_plain_number(n).ok_or_else(invalid)? * 60.0
    } else if let Some(n) = raw.strip_suffix('h') {
        parse_plain_number(n).ok_or_else(invalid)? * 3600.0
    } else {
        parse_plain_number(&raw).ok_or_else(invalid)?
    };
    if !value.is_finite() || value < 0.0 {
        return Err(invalid());
    }
    Ok(value)
}

const DAY_MS: i64 = 86_400_000;

/// Parses a `--since`/`--until` value into Unix milliseconds: `3d`, `36h`, `90m`, `2w` (ago),
/// `today`, `yesterday` (UTC midnight), `now`, `YYYY-MM-DD` (UTC), RFC3339, or Unix
/// seconds/milliseconds.
pub fn parse_time_filter(s: &str, now_unix_ms: i64) -> Result<i64> {
    let raw = s.trim().to_ascii_lowercase();
    let invalid = || {
        anyhow!(
            "invalid time {s:?} (use 3d, 36h, 90m, 2w, today, yesterday, now, YYYY-MM-DD, RFC3339 or Unix time)"
        )
    };
    let today = now_unix_ms.div_euclid(DAY_MS) * DAY_MS;
    match raw.as_str() {
        "" => return Err(invalid()),
        "now" => return Ok(now_unix_ms),
        "today" => return Ok(today),
        "yesterday" => return Ok(today - DAY_MS),
        _ => {}
    }
    if let Some(n) = parse_whole(&raw) {
        let n = i64::try_from(n).map_err(|_| invalid())?;
        // Anything past ~1973 in milliseconds is a millisecond timestamp.
        return Ok(if n >= 100_000_000_000 {
            n
        } else {
            n.checked_mul(1000).ok_or_else(invalid)?
        });
    }
    for (suffix, unit_ms) in [
        ("w", 7 * DAY_MS),
        ("d", DAY_MS),
        ("h", 3_600_000),
        ("m", 60_000),
    ] {
        if let Some(n) = raw.strip_suffix(suffix) {
            if let Some(v) = parse_plain_number(n) {
                let delta = v * unit_ms as f64;
                if !delta.is_finite() || delta > i64::MAX as f64 / 2.0 {
                    return Err(invalid());
                }
                return Ok(now_unix_ms - delta.round() as i64);
            }
        }
    }
    if raw.len() == 10 && raw.as_bytes()[4] == b'-' && raw.as_bytes()[7] == b'-' {
        let y: i64 = raw[0..4].parse().map_err(|_| invalid())?;
        let m: i64 = raw[5..7].parse().map_err(|_| invalid())?;
        let d: i64 = raw[8..10].parse().map_err(|_| invalid())?;
        let days = days_from_civil(y, m, d).ok_or_else(invalid)?;
        return Ok(days * DAY_MS);
    }
    let upper = s.trim().to_ascii_uppercase();
    if let Some(secs) = parse_rfc3339_unix_secs(&upper) {
        return Ok(secs * 1000);
    }
    Err(invalid())
}

fn parse_window(
    since: Option<&str>,
    until: Option<&str>,
    now_unix_ms: i64,
) -> Result<(Option<i64>, Option<i64>)> {
    let since = since
        .map(|s| parse_time_filter(s, now_unix_ms))
        .transpose()?;
    let until = until
        .map(|s| parse_time_filter(s, now_unix_ms))
        .transpose()?;
    if let (Some(a), Some(b)) = (since, until) {
        if a > b {
            bail!("--since is after --until");
        }
    }
    Ok((since, until))
}

pub fn now_unix_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d)
}

/// `2026-11-17T19:00:00Z` (UTC, whole seconds).
pub fn format_unix_ms_rfc3339(ms: i64) -> String {
    let secs = ms.div_euclid(1000);
    let (y, m, d) = civil_from_days(secs.div_euclid(86_400));
    let rem = secs.rem_euclid(86_400);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

// ---------------------------------------------------------------------------
// Ranges and renderers
// ---------------------------------------------------------------------------

/// A `--from`/`--to` window in seconds from the start of the meeting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeRange {
    pub from: Option<f64>,
    pub to: Option<f64>,
}

fn validate_range(from: Option<f64>, to: Option<f64>) -> Result<Option<TimeRange>> {
    for v in [from, to].into_iter().flatten() {
        if !v.is_finite() || v < 0.0 {
            bail!("--from/--to must be non-negative offsets");
        }
    }
    if let (Some(a), Some(b)) = (from, to) {
        if a > b {
            bail!("--from is after --to");
        }
    }
    Ok((from.is_some() || to.is_some()).then_some(TimeRange { from, to }))
}

/// Turns that overlap `[from, to]`.
pub fn filter_turns_by_range(turns: &[Turn], from: Option<f64>, to: Option<f64>) -> Vec<Turn> {
    turns
        .iter()
        .filter(|t| {
            let end = t.end_seconds.max(t.start_seconds);
            from.is_none_or(|f| end >= f) && to.is_none_or(|limit| t.start_seconds <= limit)
        })
        .cloned()
        .collect()
}

pub fn render_transcript(
    output: &TranscriptOutput,
    format: TranscriptFormat,
    range: Option<TimeRange>,
) -> Result<String> {
    Ok(match format {
        TranscriptFormat::Text => format_plain_text(output),
        TranscriptFormat::Markdown => format_markdown(output, range),
        TranscriptFormat::Json => format!(
            "{}\n",
            serde_json::to_string_pretty(&transcript_json_v1(output, range))?
        ),
        TranscriptFormat::Srt => format_srt(&output.turns),
        TranscriptFormat::Vtt => format_vtt(&output.turns),
    })
}

pub fn format_markdown(output: &TranscriptOutput, range: Option<TimeRange>) -> String {
    let mut md = format!("# {}\n\n", output.title);
    md.push_str(&format!("- **ID:** {}\n", output.id));
    if let Some(app) = &output.app {
        md.push_str(&format!("- **App:** {app}\n"));
    }
    if let Some(started) = &output.started_at {
        md.push_str(&format!("- **Started:** {started}\n"));
    }
    if let Some(ended) = &output.ended_at {
        md.push_str(&format!("- **Ended:** {ended}\n"));
    }
    if let Some(url) = &output.conference_url {
        md.push_str(&format!("- **URL:** {url}\n"));
    }
    md.push_str(&format!("- **Status:** {}\n", output.status));
    md.push_str(&format!(
        "- **Duration:** {}\n",
        format_clock(output.duration_seconds)
    ));
    if let Some(model) = &output.model {
        md.push_str(&format!("- **Model:** {model}\n"));
    }
    if let Some(r) = range {
        let from = r.from.map(format_clock).unwrap_or_else(|| "start".into());
        let to = r.to.map(format_clock).unwrap_or_else(|| "end".into());
        md.push_str(&format!("- **Range:** {from} to {to}\n"));
    }
    md.push_str("\n## Transcript\n\n");
    if output.turns.is_empty() {
        md.push_str("_No speech in this range._\n");
    } else {
        for turn in &output.turns {
            md.push_str(&format!(
                "**[{}] {}:** {}\n\n",
                format_clock(turn.start_seconds),
                turn.speaker.label(),
                collapse_whitespace(&turn.text)
            ));
        }
    }
    md
}

/// Versioned JSON (`version: 1`) for `meetings show --json` and the MCP server.
pub fn transcript_json_v1(
    output: &TranscriptOutput,
    range: Option<TimeRange>,
) -> serde_json::Value {
    let mut meeting = serde_json::Map::new();
    meeting.insert("id".into(), json!(output.id));
    meeting.insert("title".into(), json!(output.title));
    let optional = [
        ("app", &output.app),
        ("started_at", &output.started_at),
        ("ended_at", &output.ended_at),
        ("conference_url", &output.conference_url),
        ("model", &output.model),
    ];
    for (key, value) in optional {
        if let Some(v) = value {
            meeting.insert(key.into(), json!(v));
        }
    }
    meeting.insert("status".into(), json!(output.status));
    meeting.insert("duration_seconds".into(), json!(output.duration_seconds));
    meeting.insert("source".into(), json!(output.source));

    let turns: Vec<serde_json::Value> = output
        .turns
        .iter()
        .map(|t| {
            json!({
                "start_seconds": t.start_seconds,
                "end_seconds": t.end_seconds,
                "speaker": t.speaker,
                "speaker_label": t.speaker.label(),
                "text": t.text,
            })
        })
        .collect();
    let mut root = serde_json::Map::new();
    root.insert("version".into(), json!(1));
    root.insert("meeting".into(), serde_json::Value::Object(meeting));
    if let Some(r) = range {
        root.insert(
            "range".into(),
            json!({ "from_seconds": r.from, "to_seconds": r.to }),
        );
    }
    root.insert("turns".into(), serde_json::Value::Array(turns));
    serde_json::Value::Object(root)
}

fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// End of a subtitle cue; turns read from `transcript.txt` have no end time.
fn cue_end(turns: &[Turn], idx: usize) -> f64 {
    let turn = &turns[idx];
    if turn.end_seconds > turn.start_seconds {
        return turn.end_seconds;
    }
    match turns.get(idx + 1) {
        Some(next) if next.start_seconds > turn.start_seconds => {
            next.start_seconds.min(turn.start_seconds + 10.0)
        }
        _ => turn.start_seconds + 3.0,
    }
}

fn subtitle_clock(seconds: f64, separator: char) -> String {
    let total_ms = (seconds.max(0.0) * 1000.0).round() as u64;
    format!(
        "{:02}:{:02}:{:02}{separator}{:03}",
        total_ms / 3_600_000,
        (total_ms % 3_600_000) / 60_000,
        (total_ms % 60_000) / 1000,
        total_ms % 1000
    )
}

/// SubRip subtitles. Blank lines (which end a cue) and `-->` inside text are removed.
pub fn format_srt(turns: &[Turn]) -> String {
    let mut out = String::new();
    let mut index = 0;
    for (i, turn) in turns.iter().enumerate() {
        let text = collapse_whitespace(&turn.text.replace("-->", "->"));
        if text.is_empty() {
            continue;
        }
        index += 1;
        out.push_str(&format!(
            "{index}\n{} --> {}\n{}: {text}\n\n",
            subtitle_clock(turn.start_seconds, ','),
            subtitle_clock(cue_end(turns, i), ','),
            turn.speaker.label()
        ));
    }
    out
}

fn escape_vtt(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// WebVTT with voice spans; `&`, `<` and `>` are escaped and blank lines removed.
pub fn format_vtt(turns: &[Turn]) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for (i, turn) in turns.iter().enumerate() {
        let text = collapse_whitespace(&turn.text);
        if text.is_empty() {
            continue;
        }
        out.push_str(&format!(
            "{} --> {}\n<v {}>{}</v>\n\n",
            subtitle_clock(turn.start_seconds, '.'),
            subtitle_clock(cue_end(turns, i), '.'),
            turn.speaker.label(),
            escape_vtt(&text)
        ));
    }
    out
}

// ---------------------------------------------------------------------------
// Read-only data access (meetings, history)
// ---------------------------------------------------------------------------

/// Where meetings, dictations and the dictionary live.
#[derive(Debug, Clone)]
pub struct DataPaths {
    pub meetings: PathBuf,
    pub corpus: PathBuf,
    pub dictionary: PathBuf,
}

impl DataPaths {
    pub fn from_data_dir(dir: &Path) -> Self {
        Self {
            meetings: dir.join("Meetings"),
            corpus: dir.join("Corpus"),
            dictionary: dir.join("dictionary.json"),
        }
    }

    /// Phonon's data folder (`app_support_dir()`), honouring `PHONON_MEETINGS_DIR`.
    pub fn from_env() -> Result<Self> {
        let mut paths = Self::from_data_dir(&app_support_dir()?);
        paths.meetings = default_meetings_root()?;
        Ok(paths)
    }
}

pub fn clamp_limit(limit: Option<usize>, default: usize, max: usize) -> usize {
    limit.unwrap_or(default).clamp(1, max)
}

/// Loads a meeting without running speech recognition or writing anything.
pub fn load_meeting_readonly(
    meetings_root: &Path,
    selector: Option<&str>,
) -> Result<TranscriptOutput> {
    let dir = resolve_meeting_dir(meetings_root, selector)?;
    load_meeting_dir_readonly(&dir)
}

fn load_meeting_dir_readonly(dir: &Path) -> Result<TranscriptOutput> {
    let mut out = load_transcript_at(dir, &TranscriptArgs::default(), &mut NoAsr)?;
    if out.source == "wav" || out.source == "partial+wav" {
        // A live call: the app streams `transcript.txt`, which may be ahead of the partial file.
        if let Ok(txt) = fs::read_to_string(dir.join("transcript.txt")) {
            let parsed = parse_transcript_txt(&txt);
            if parsed.turns.len() > out.turns.len() {
                let max_turn = parsed
                    .turns
                    .iter()
                    .map(|t| t.end_seconds)
                    .fold(0.0_f64, f64::max);
                out.duration_seconds = out.duration_seconds.max(max_turn);
                out.turns = parsed.turns;
                out.source = "transcript.txt".into();
                if out.model.is_none() {
                    out.model = parsed.model;
                }
            }
        }
    }
    Ok(out)
}

/// Renders one meeting (`latest`, an id, or an unambiguous id prefix) without side effects.
pub fn show_meeting(
    meetings_root: &Path,
    selector: &str,
    from: Option<f64>,
    to: Option<f64>,
    format: TranscriptFormat,
) -> Result<String> {
    let range = validate_range(from, to)?;
    let mut output = load_meeting_readonly(meetings_root, Some(selector))?;
    if range.is_some() {
        output.turns = filter_turns_by_range(&output.turns, from, to);
    }
    render_transcript(&output, format, range)
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct MeetingSummary {
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
    pub started_unix_ms: Option<i64>,
    pub duration_seconds: f64,
    pub turns: usize,
    pub words: usize,
}

fn in_window(rank_ms: i64, since: Option<i64>, until: Option<i64>) -> bool {
    let known = rank_ms > 0;
    since.is_none_or(|s| known && rank_ms >= s) && until.is_none_or(|u| known && rank_ms <= u)
}

/// Meetings in a time window, newest first, read-only.
fn load_meetings_in_window(
    meetings_root: &Path,
    since: Option<i64>,
    until: Option<i64>,
    limit: Option<usize>,
) -> Result<Vec<(i64, TranscriptOutput)>> {
    let mut out = Vec::new();
    for (rank, _id, dir) in list_meeting_dirs(meetings_root)? {
        if !in_window(rank, since, until) {
            continue;
        }
        match load_meeting_dir_readonly(&dir) {
            Ok(meeting) => out.push((rank, meeting)),
            Err(err) => eprintln!("phonon: skipping {}: {err:#}", dir.display()),
        }
        if limit.is_some_and(|l| out.len() >= l) {
            break;
        }
    }
    Ok(out)
}

pub fn list_meetings(
    meetings_root: &Path,
    since: Option<i64>,
    until: Option<i64>,
    limit: usize,
) -> Result<Vec<MeetingSummary>> {
    Ok(
        load_meetings_in_window(meetings_root, since, until, Some(limit))?
            .into_iter()
            .map(|(rank, m)| MeetingSummary {
                words: m
                    .turns
                    .iter()
                    .map(|t| t.text.split_whitespace().count())
                    .sum(),
                turns: m.turns.len(),
                id: m.id,
                title: m.title,
                app: m.app,
                status: m.status,
                started_at: m.started_at,
                ended_at: m.ended_at,
                started_unix_ms: (rank > 0).then_some(rank),
                duration_seconds: m.duration_seconds,
            })
            .collect(),
    )
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SearchTurnRef {
    pub start_seconds: f64,
    pub clock: String,
    pub speaker: Speaker,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct MeetingSearchHit {
    pub meeting_id: String,
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_at: Option<String>,
    pub score: f64,
    /// `all` when every unquoted term matched exactly, else `partial`.
    #[serde(rename = "match")]
    pub match_kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn: Option<SearchTurnRef>,
    pub snippet: String,
}

const SNIPPET_CHARS: usize = 220;

fn round4(v: f64) -> f64 {
    (v * 10_000.0).round() / 10_000.0
}

pub fn search_meetings(
    paths: &DataPaths,
    query: &str,
    since: Option<i64>,
    until: Option<i64>,
    limit: usize,
) -> Result<Vec<MeetingSearchHit>> {
    if query.trim().is_empty() {
        bail!("search query is empty");
    }
    let meetings = load_meetings_in_window(&paths.meetings, since, until, None)?;
    let docs: Vec<String> = meetings
        .iter()
        .map(|(_, m)| {
            let mut doc = m.title.clone();
            for t in &m.turns {
                doc.push('\n');
                doc.push_str(&t.text);
            }
            doc
        })
        .collect();
    let aliases = crate::search::Aliases::load(&paths.dictionary);
    let hits = crate::search::search(&docs, query, &aliases);
    Ok(hits
        .into_iter()
        .take(limit)
        .map(|hit| {
            let meeting = &meetings[hit.index].1;
            let best = meeting
                .turns
                .iter()
                .map(|t| crate::search::match_count(&t.text, &hit.matched_stems))
                .enumerate()
                .filter(|(_, c)| *c > 0)
                .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
                .map(|(i, _)| &meeting.turns[i]);
            let snippet_source = best.map(|t| t.text.as_str()).unwrap_or(&meeting.title);
            MeetingSearchHit {
                meeting_id: meeting.id.clone(),
                title: meeting.title.clone(),
                started_at: meeting.started_at.clone(),
                score: round4(hit.score),
                match_kind: if hit.all_terms { "all" } else { "partial" },
                turn: best.map(|t| SearchTurnRef {
                    start_seconds: t.start_seconds,
                    clock: format_clock(t.start_seconds),
                    speaker: t.speaker,
                }),
                snippet: crate::search::snippet(snippet_source, &hit.matched_stems, SNIPPET_CHARS),
            }
        })
        .collect())
}

#[derive(Debug, Default, Deserialize)]
struct CorpusApp {
    #[serde(default)]
    bundle_id: String,
    #[serde(default)]
    app_name: String,
    #[serde(default)]
    url_host: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CorpusRecord {
    #[serde(default)]
    id: String,
    #[serde(default)]
    created_at_unix_ms: Option<f64>,
    #[serde(default)]
    recorded_at: Option<String>,
    #[serde(default)]
    raw_transcript: String,
    #[serde(default)]
    final_transcript: String,
    #[serde(default)]
    audio_duration_ms: Option<u64>,
    #[serde(default)]
    app_context: Option<CorpusApp>,
}

#[derive(Debug, Clone, Default)]
pub struct HistoryQuery {
    pub since: Option<i64>,
    pub until: Option<i64>,
    pub app: Option<String>,
    pub grep: Option<String>,
    pub search: Option<String>,
    pub raw: bool,
    pub limit: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct HistoryEntry {
    pub id: String,
    pub created_at_unix_ms: i64,
    pub created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub app: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bundle_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url_host: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_duration_ms: Option<u64>,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
}

/// Dictation history from `Corpus/*/metadata.json`, newest first (or by relevance with
/// `search`). Read-only; unreadable recordings are skipped.
pub fn query_history(paths: &DataPaths, query: &HistoryQuery) -> Result<Vec<HistoryEntry>> {
    let mut entries = Vec::new();
    if paths.corpus.is_dir() {
        for entry in fs::read_dir(&paths.corpus)
            .with_context(|| format!("read {}", paths.corpus.display()))?
            .flatten()
        {
            let Ok(Some(rec)) = read_json::<CorpusRecord>(&entry.path().join("metadata.json"))
            else {
                continue;
            };
            let created = rec
                .created_at_unix_ms
                .filter(|v| v.is_finite() && *v > 0.0)
                .map(|v| v as i64)
                .or_else(|| {
                    rec.recorded_at
                        .as_deref()
                        .and_then(parse_rfc3339_unix_secs)
                        .map(|s| s * 1000)
                })
                .unwrap_or(0);
            let text = if query.raw || rec.final_transcript.trim().is_empty() {
                rec.raw_transcript.trim().to_string()
            } else {
                rec.final_transcript.trim().to_string()
            };
            if text.is_empty() || !in_window(created, query.since, query.until) {
                continue;
            }
            let app = rec.app_context.unwrap_or_default();
            if let Some(filter) = query.app.as_deref().map(str::to_lowercase) {
                let haystack = [
                    app.app_name.as_str(),
                    app.bundle_id.as_str(),
                    app.url_host.as_deref().unwrap_or(""),
                ];
                if !haystack.iter().any(|h| h.to_lowercase().contains(&filter)) {
                    continue;
                }
            }
            if let Some(grep) = query.grep.as_deref().map(str::to_lowercase) {
                if !text.to_lowercase().contains(&grep) {
                    continue;
                }
            }
            let id = if rec.id.is_empty() {
                entry.file_name().to_string_lossy().into_owned()
            } else {
                rec.id
            };
            entries.push(HistoryEntry {
                id,
                created_at_unix_ms: created,
                created_at: format_unix_ms_rfc3339(created),
                app: non_empty(&app.app_name),
                bundle_id: non_empty(&app.bundle_id),
                url_host: app.url_host.as_deref().and_then(non_empty),
                audio_duration_ms: rec.audio_duration_ms,
                text,
                score: None,
                snippet: None,
            });
        }
    }
    entries.sort_by(|a, b| {
        b.created_at_unix_ms
            .cmp(&a.created_at_unix_ms)
            .then_with(|| b.id.cmp(&a.id))
    });

    let limit = query.limit.max(1);
    let Some(search) = query.search.as_deref().filter(|s| !s.trim().is_empty()) else {
        entries.truncate(limit);
        return Ok(entries);
    };
    let aliases = crate::search::Aliases::load(&paths.dictionary);
    let texts: Vec<&str> = entries.iter().map(|e| e.text.as_str()).collect();
    let hits = crate::search::search(&texts, search, &aliases);
    Ok(hits
        .into_iter()
        .take(limit)
        .map(|hit| {
            let mut entry = entries[hit.index].clone();
            entry.score = Some(round4(hit.score));
            entry.snippet = Some(crate::search::snippet(
                &entry.text,
                &hit.matched_stems,
                SNIPPET_CHARS,
            ));
            entry
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Text renderers for the CLI
// ---------------------------------------------------------------------------

fn one_line(text: &str, max_chars: usize) -> String {
    let flat = collapse_whitespace(text);
    if flat.chars().count() <= max_chars {
        flat
    } else {
        let cut: String = flat.chars().take(max_chars.saturating_sub(3)).collect();
        format!("{cut}...")
    }
}

pub fn render_meeting_list_text(meetings: &[MeetingSummary]) -> String {
    if meetings.is_empty() {
        return "(no meetings)\n".into();
    }
    let mut out = String::new();
    for m in meetings {
        let started = m
            .started_unix_ms
            .map(format_unix_ms_rfc3339)
            .or_else(|| m.started_at.clone())
            .unwrap_or_else(|| "unknown".into());
        out.push_str(&format!(
            "{}  {}  {:>8}  {:>4} turns  {}  [{}]\n",
            m.id,
            started,
            format_clock(m.duration_seconds),
            m.turns,
            m.title,
            m.status
        ));
    }
    out
}

pub fn render_search_text(hits: &[MeetingSearchHit]) -> String {
    if hits.is_empty() {
        return "(no matches)\n".into();
    }
    let mut out = String::new();
    for h in hits {
        let at = h
            .turn
            .as_ref()
            .map(|t| format!(" @ {} {}", t.clock, t.speaker.label()))
            .unwrap_or_default();
        out.push_str(&format!(
            "{}  {}{at}  (score {:.3}, {})\n    {}\n",
            h.meeting_id, h.title, h.score, h.match_kind, h.snippet
        ));
    }
    out
}

pub fn render_history_text(entries: &[HistoryEntry]) -> String {
    if entries.is_empty() {
        return "(no dictations)\n".into();
    }
    let mut out = String::new();
    for e in entries {
        let app = e.app.as_deref().or(e.bundle_id.as_deref()).unwrap_or("-");
        let body = e.snippet.as_deref().unwrap_or(&e.text);
        out.push_str(&format!(
            "{}  {:<14}  {}\n",
            e.created_at,
            one_line(app, 14),
            one_line(body, 160)
        ));
    }
    out
}

// ---------------------------------------------------------------------------
// CLI entry points
// ---------------------------------------------------------------------------

pub const MEETINGS_LIST_DEFAULT: usize = 20;
pub const MEETINGS_LIST_MAX: usize = 200;
pub const MEETINGS_SEARCH_DEFAULT: usize = 10;
pub const MEETINGS_SEARCH_MAX: usize = 50;
pub const HISTORY_DEFAULT: usize = 50;
pub const HISTORY_MAX: usize = 10_000;

pub fn run_meetings_list(
    since: Option<&str>,
    until: Option<&str>,
    limit: Option<usize>,
    json_out: bool,
) -> Result<()> {
    let paths = DataPaths::from_env()?;
    let (since, until) = parse_window(since, until, now_unix_ms())?;
    let limit = clamp_limit(limit, MEETINGS_LIST_DEFAULT, MEETINGS_LIST_MAX);
    let meetings = list_meetings(&paths.meetings, since, until, limit)?;
    if json_out {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({ "meetings": meetings }))?
        );
    } else {
        print!("{}", render_meeting_list_text(&meetings));
    }
    Ok(())
}

pub fn run_meetings_show(
    selector: &str,
    from: Option<&str>,
    to: Option<&str>,
    format: TranscriptFormat,
) -> Result<()> {
    let paths = DataPaths::from_env()?;
    let from = from.map(parse_offset_seconds).transpose()?;
    let to = to.map(parse_offset_seconds).transpose()?;
    print!(
        "{}",
        show_meeting(&paths.meetings, selector, from, to, format)?
    );
    Ok(())
}

pub fn run_meetings_search(
    query: &str,
    since: Option<&str>,
    until: Option<&str>,
    limit: Option<usize>,
    json_out: bool,
) -> Result<()> {
    let paths = DataPaths::from_env()?;
    let (since, until) = parse_window(since, until, now_unix_ms())?;
    let limit = clamp_limit(limit, MEETINGS_SEARCH_DEFAULT, MEETINGS_SEARCH_MAX);
    let hits = search_meetings(&paths, query, since, until, limit)?;
    if json_out {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({ "query": query, "results": hits }))?
        );
    } else {
        print!("{}", render_search_text(&hits));
    }
    Ok(())
}

#[derive(Debug, Clone, Default)]
pub struct HistoryArgs {
    pub since: Option<String>,
    pub until: Option<String>,
    pub app: Option<String>,
    pub grep: Option<String>,
    pub search: Option<String>,
    pub raw: bool,
    pub limit: Option<usize>,
    pub json: bool,
}

pub fn run_history(args: HistoryArgs) -> Result<()> {
    let paths = DataPaths::from_env()?;
    let (since, until) = parse_window(args.since.as_deref(), args.until.as_deref(), now_unix_ms())?;
    let entries = query_history(
        &paths,
        &HistoryQuery {
            since,
            until,
            app: args.app,
            grep: args.grep,
            search: args.search,
            raw: args.raw,
            limit: clamp_limit(args.limit, HISTORY_DEFAULT, HISTORY_MAX),
        },
    )?;
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({ "entries": entries }))?
        );
    } else {
        print!("{}", render_history_text(&entries));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Live-stream artifacts (write paths only)
// ---------------------------------------------------------------------------

/// After `phonon transcript` brings a meeting up to date: rewrite `Meetings/latest.txt`
/// atomically with the readable transcript and append a `meeting.updated` event to
/// `Meetings/events.jsonl`. A `latest.txt` symlink belongs to the macOS app and is left alone,
/// as is `latest.txt` when a newer meeting exists.
pub fn write_live_artifacts(meetings_root: &Path, output: &TranscriptOutput) -> Result<()> {
    fs::create_dir_all(meetings_root)
        .with_context(|| format!("create {}", meetings_root.display()))?;
    let latest = meetings_root.join("latest.txt");
    let app_owned_link = fs::symlink_metadata(&latest)
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false);
    let newest_id = list_meeting_dirs(meetings_root)?
        .into_iter()
        .next()
        .map(|(_, id, _)| id);
    let is_newest = match &newest_id {
        Some(id) => *id == output.id,
        None => true,
    };
    if !app_owned_link && is_newest {
        let tmp = meetings_root.join(format!(".latest.txt.{}.tmp", std::process::id()));
        fs::write(&tmp, format_plain_text(output))
            .with_context(|| format!("write {}", tmp.display()))?;
        if let Err(err) = fs::rename(&tmp, &latest) {
            let _ = fs::remove_file(&tmp);
            return Err(err).with_context(|| format!("replace {}", latest.display()));
        }
    }
    let event = json!({
        "type": "meeting.updated",
        "id": output.id,
        "turns": output.turns.len(),
        "updated_at": format_unix_ms_rfc3339(now_unix_ms()),
    });
    let events = meetings_root.join("events.jsonl");
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&events)
        .with_context(|| format!("open {}", events.display()))?;
    writeln!(file, "{event}").with_context(|| format!("append {}", events.display()))?;
    Ok(())
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
                ..Default::default()
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
                ..Default::default()
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
                ..Default::default()
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
                ..Default::default()
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
                ..Default::default()
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
                ..Default::default()
            },
            &mut mock,
        )
        .is_err());
    }

    #[test]
    fn parses_offsets() {
        assert_eq!(parse_offset_seconds("1:30").unwrap(), 90.0);
        assert_eq!(parse_offset_seconds("113:03").unwrap(), 6783.0);
        assert_eq!(parse_offset_seconds("1:02:03").unwrap(), 3723.0);
        assert_eq!(parse_offset_seconds("90").unwrap(), 90.0);
        assert_eq!(parse_offset_seconds("1.5").unwrap(), 1.5);
        assert_eq!(parse_offset_seconds("90s").unwrap(), 90.0);
        assert_eq!(parse_offset_seconds("1.5m").unwrap(), 90.0);
        assert_eq!(parse_offset_seconds(" 0:05.5 ").unwrap(), 5.5);
        for bad in [
            "", "-5", "-1:00", "nan", "NaN", "inf", "1e3", "1:60", "1:75:00", "abc", "1:2:3:4",
            "..", "1.2.3", "m", ":30",
        ] {
            assert!(parse_offset_seconds(bad).is_err(), "{bad:?} should fail");
        }
    }

    #[test]
    fn parses_time_filters() {
        // 2026-11-21T15:30:00Z
        let day = 86_400_000;
        let now = days_from_civil(2026, 11, 21).unwrap() * day + 15 * 3_600_000 + 30 * 60_000;
        assert_eq!(format_unix_ms_rfc3339(now), "2026-11-21T15:30:00Z");
        assert_eq!(parse_time_filter("now", now).unwrap(), now);
        assert_eq!(parse_time_filter("3d", now).unwrap(), now - 3 * day);
        assert_eq!(parse_time_filter("36h", now).unwrap(), now - 36 * 3_600_000);
        assert_eq!(parse_time_filter("90m", now).unwrap(), now - 90 * 60_000);
        assert_eq!(parse_time_filter("2w", now).unwrap(), now - 14 * day);
        let today = parse_time_filter("today", now).unwrap();
        assert_eq!(format_unix_ms_rfc3339(today), "2026-11-21T00:00:00Z");
        assert_eq!(parse_time_filter("yesterday", now).unwrap(), today - day);
        assert_eq!(parse_time_filter("2026-11-21", now).unwrap(), today);
        assert_eq!(
            parse_time_filter("2026-11-17T19:00:00Z", now).unwrap(),
            1_794_942_000_000
        );
        assert_eq!(
            parse_time_filter("2026-11-17T13:00:00-06:00", now).unwrap(),
            1_794_942_000_000
        );
        assert_eq!(
            parse_time_filter("1794942000", now).unwrap(),
            1_794_942_000_000
        );
        assert_eq!(
            parse_time_filter("1794942000000", now).unwrap(),
            1_794_942_000_000
        );
        for bad in ["", "-3d", "soon", "2026-13-01", "3x", "nan"] {
            assert!(parse_time_filter(bad, now).is_err(), "{bad:?} should fail");
        }
    }

    fn sample_output() -> TranscriptOutput {
        TranscriptOutput {
            id: "meeting_1".into(),
            title: "Design review".into(),
            app: Some("Zoom".into()),
            status: "complete".into(),
            started_at: Some("2026-11-17T19:00:00Z".into()),
            ended_at: None,
            conference_url: None,
            model: Some("parakeet".into()),
            duration_seconds: 3725.0,
            window_minutes: None,
            source: "transcript.json".into(),
            turns: vec![
                Turn {
                    start_seconds: 1.25,
                    end_seconds: 4.5,
                    speaker: Speaker::Call,
                    text: "A --> B\n\nnext <b>bold</b> & co".into(),
                },
                Turn {
                    start_seconds: 3723.0,
                    end_seconds: 3723.0,
                    speaker: Speaker::Me,
                    text: "Done.".into(),
                },
            ],
        }
    }

    #[test]
    fn formats_srt_and_vtt_with_escaping() {
        let out = sample_output();
        let srt = format_srt(&out.turns);
        assert_eq!(
            srt,
            "1\n00:00:01,250 --> 00:00:04,500\nCall: A -> B next <b>bold</b> & co\n\n\
             2\n01:02:03,000 --> 01:02:06,000\nYou: Done.\n\n"
        );
        let vtt = format_vtt(&out.turns);
        assert_eq!(
            vtt,
            "WEBVTT\n\n00:00:01.250 --> 00:00:04.500\n<v Call>A --&gt; B next &lt;b&gt;bold&lt;/b&gt; &amp; co</v>\n\n\
             01:02:03.000 --> 01:02:06.000\n<v You>Done.</v>\n\n"
        );
    }

    #[test]
    fn formats_markdown_and_versioned_json() {
        let out = sample_output();
        let range = Some(TimeRange {
            from: Some(0.0),
            to: Some(60.0),
        });
        let md = format_markdown(&out, range);
        assert!(md.starts_with("# Design review\n"));
        assert!(md.contains("- **App:** Zoom"));
        assert!(md.contains("- **Range:** 0:00 to 1:00"));
        assert!(md.contains("**[0:01] Call:** A --> B next"));
        let v = transcript_json_v1(&out, range);
        assert_eq!(v["version"], 1);
        assert_eq!(v["meeting"]["id"], "meeting_1");
        assert_eq!(v["range"]["to_seconds"], 60.0);
        assert_eq!(v["turns"][1]["speaker"], "me");
        assert_eq!(v["turns"][1]["speaker_label"], "You");
        assert!(transcript_json_v1(&out, None).get("range").is_none());

        let ranged = filter_turns_by_range(&out.turns, Some(10.0), None);
        assert_eq!(ranged.len(), 1);
        assert_eq!(ranged[0].text, "Done.");
        assert!(validate_range(Some(5.0), Some(1.0)).is_err());
    }

    fn write_meeting(root: &Path, id: &str, started: &str, turns: &str) {
        let dir = root.join(id);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("metadata.json"),
            format!(r#"{{"id":"{id}","started_at":"{started}","ended_at":"{started}","app_name":"Zoom"}}"#),
        )
        .unwrap();
        fs::write(
            dir.join("transcript.json"),
            format!(r#"{{"model":"parakeet","turns":{turns}}}"#),
        )
        .unwrap();
    }

    #[test]
    fn resolves_latest_exact_and_prefix_selectors() {
        let tmp = TempDir::new("prefix");
        let turns = r#"[{"start_seconds":1,"end_seconds":2,"speaker":"me","text":"hi"}]"#;
        write_meeting(
            &tmp.0,
            "meeting_1794959123259",
            "2026-11-17T19:00:00Z",
            turns,
        );
        write_meeting(
            &tmp.0,
            "meeting_1794959999999",
            "2026-11-17T20:00:00Z",
            turns,
        );
        write_meeting(
            &tmp.0,
            "meeting_1700000000000",
            "2023-11-14T22:13:20Z",
            turns,
        );
        let id = |sel: &str| {
            load_meeting_readonly(&tmp.0, Some(sel))
                .map(|o| o.id)
                .map_err(|e| e.to_string())
        };
        assert_eq!(id("latest").unwrap(), "meeting_1794959999999");
        assert_eq!(
            id("meeting_1794959123259").unwrap(),
            "meeting_1794959123259"
        );
        assert_eq!(id("17000000").unwrap(), "meeting_1700000000000");
        assert_eq!(id("1794959123").unwrap(), "meeting_1794959123259");
        assert!(id("179495").unwrap_err().contains("ambiguous"));
        assert!(id("17000").is_err(), "prefix shorter than 6 chars");
        assert!(id("999999").is_err());
    }

    #[test]
    fn live_artifacts_write_latest_and_events_but_keep_app_symlink() {
        let tmp = TempDir::new("artifacts");
        let turns = r#"[{"start_seconds":1,"end_seconds":2,"speaker":"me","text":"hi"}]"#;
        write_meeting(&tmp.0, "meeting_1", "2026-11-17T19:00:00Z", turns);
        let mut out = sample_output();
        out.id = "meeting_1".into();
        write_live_artifacts(&tmp.0, &out).unwrap();
        write_live_artifacts(&tmp.0, &out).unwrap();
        let latest = fs::read_to_string(tmp.0.join("latest.txt")).unwrap();
        assert!(latest.contains("Meeting: Design review"));
        let events = fs::read_to_string(tmp.0.join("events.jsonl")).unwrap();
        let lines: Vec<&str> = events.lines().collect();
        assert_eq!(lines.len(), 2);
        let event: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(event["type"], "meeting.updated");
        assert_eq!(event["id"], "meeting_1");
        assert_eq!(event["turns"], 2);

        #[cfg(unix)]
        {
            fs::remove_file(tmp.0.join("latest.txt")).unwrap();
            std::os::unix::fs::symlink("meeting_1/transcript.json", tmp.0.join("latest.txt"))
                .unwrap();
            write_live_artifacts(&tmp.0, &out).unwrap();
            let meta = fs::symlink_metadata(tmp.0.join("latest.txt")).unwrap();
            assert!(meta.file_type().is_symlink(), "app-owned symlink kept");
        }
    }

    #[test]
    fn read_only_queries_never_write() {
        let tmp = TempDir::new("readonly");
        let meetings = tmp.0.join("Meetings");
        let turns = r#"[{"start_seconds":1,"end_seconds":2,"speaker":"me","text":"ship it"}]"#;
        write_meeting(
            &meetings,
            "meeting_1794959123259",
            "2026-11-17T19:00:00Z",
            turns,
        );
        let paths = DataPaths::from_data_dir(&tmp.0);
        let snapshot = |dir: &Path| {
            let mut names: Vec<String> = fs::read_dir(dir)
                .unwrap()
                .flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect();
            names.sort();
            names
        };
        let before = snapshot(&meetings);
        assert_eq!(list_meetings(&meetings, None, None, 20).unwrap().len(), 1);
        assert!(show_meeting(&meetings, "latest", None, None, TranscriptFormat::Markdown).is_ok());
        assert_eq!(
            search_meetings(&paths, "ship", None, None, 10)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(snapshot(&meetings), before);
        assert!(!meetings.join("events.jsonl").exists());
    }
}
