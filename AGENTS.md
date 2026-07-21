# phonon

Machine: macbook (Apple Silicon). Path: `~/dev/tools/phonon`.

Personal dictation. Three weight streams boot in parallel (asr ∥ fluid-1 ∥ mtp).
Polish always runs after ASR. Multi-pass recording works (unique wav + phase ids).

## Startup contract

The native app exposes only behavior that materially changes dictation:
Streaming, screen context, local history, instant microphone, shortcut,
microphone priority, and launch at login. `phonon` shows one honest startup
percentage loader before the full dictation UI.
Progress advances only when a real startup stage completes.

Application readiness means all of the following have passed:

1. Parakeet weights are mapped and materialized on MLX.
2. Parakeet transcribes `assets/startup.wav` through both batch and live
   streaming paths; both results must contain the fixture keywords. This pays
   both ASR graph/shape costs and checks useful output.
3. fluid-1 and the Gemma MTP drafter finish the helper warmup.
4. The live speculative path polishes a representative dictation request; the
   output must retain the smoke-test keywords. This pays first-request graph and
   shape specialization.

Never emit global `ready`, accept recording as ready, or call startup successful
after weights load alone. The floating bar may queue input during startup, but
the first user request must not become the warmup request. A startup smoke
failure is a readiness failure, not an ASR-only or non-MTP success.

## Profiling contract

`phonon profile kernel` profiles one warmed Fluid+MTP request at the literal MLX
Metal pipeline level. It reports the top named kernels by aggregate CPU dispatch
encoding time plus exact invocation counts. It does not claim per-kernel GPU
duration: Shader Timeline and dispatch timestamp counters are unavailable on this
M4 Max. The profiler instruments an ad-hoc-signed temporary copy of the installed
helper and deletes it afterward; it never modifies the installed FluidVoice app.

`phonon profile model` measures warm model phases. For every process and input
shape, run the helper warmup and one representative real request before collecting
timed samples. Report model-load and startup-demo durations separately from the
warm median. Never mix cold/first-request samples into the warm median. Keep
prompts, MTP settings, input shape, and iteration counts equal when comparing
runs; isolate other Phonon model processes before timing.

`phonon profile e2e` is the user-perceived latency profiler. The floating bar
records every successful real keyboard path from the Right-Option or
Ctrl+Space callback through main-actor dispatch, panel presentation, recorder
start, first streaming chunk, first acoustic partial, first visible preview,
key release, capture finalization, WAV encoding, final ASR, MTP polish, and
completion of Unicode insertion-event posting. Report key-up to insertion as
system latency; report key-down to insertion only alongside dictation duration
so human speech time is never mislabeled as compute latency. Traces live at
`~/Library/Logs/Phonon/e2e.jsonl` and must come from real dictations, not the
startup fixture.

## Commands

```sh
cargo install --path ~/dev/tools/phonon/crates/phonon-cli --force --root ~/.local

phonon                 # native macOS app
phonon bar             # floating NSPanel pill (Wispr-style)
phonon bar --rebuild   # rebuild Swift bar
phonon engine          # warm JSONL backend (bar spawns this)
phonon doctor
phonon bench
phonon profile --help   # discover kernel, model, and end-to-end profilers
phonon profile kernel   # literal MLX Metal kernel top 10 for a warmed request
phonon profile model    # Fluid/MTP warmup, cached prefill, decode breakdown
phonon profile e2e      # real keyboard event → completed insertion profile
phonon dictionary --help # JSON dictionary, Wispr import, hard correction eval
phonon dictionary import-txt # merge dictionary_new_terms.txt idempotently
phonon corpus --help     # paired WAV + metadata + intended transcripts
phonon stats             # local usage totals
```

## Local data contract

All user recordings are durable corpus fixtures under
`~/Library/Application Support/Phonon/Corpus/<id>/audio.wav` with a sibling
`metadata.json`. Metadata contains raw, final, and optional intended transcripts,
dictionary repairs, duration, source, and LLM timings. Never put new recordings
back in `/tmp`. The user dictionary is JSON at
`~/Library/Application Support/Phonon/dictionary.json`; `phonon dictionary
import-wispr` merges active non-snippet entries from Wispr's local SQLite DB while
preserving replacements, provenance, stars, and usage counts.
`phonon dictionary import-txt` merges one canonical term per line from
`dictionary_new_terms.txt` without duplicating existing entries.
Streaming, local-history behavior, and the ordered microphone priority live in
`settings.json`; audio selection reads that list before every recording.

## Screen context contract

With `screen_context: true`, the bar captures all visible displays once when a
recording begins and performs local Vision OCR while the user speaks. Never save
the screenshots or send full OCR text to the model. OCR is only a disambiguation
signal: intersect it with dictionary candidates retrieved from the transcript,
then pass only the confirmed canonical terms to Fluid. Persist only those terms
in corpus metadata. Missing Screen Recording permission must degrade to normal
context-free dictation.

## Speech gate contract

Run the adaptive speech-presence gate on every completed WAV before ASR. A
no-speech clip is still retained and marked `speech_detected: false`, but it must
not reach Parakeet or Fluid and must never insert hallucinated text. The gate
must reject steady background noise and short impulses while accepting sustained
voiced energy; keep all three cases covered by audio-crate tests.

Use `phonon dictionary evaluate` for the warmed Fluid/MTP correction fixture
suite. Add failures as fixtures before changing retrieval, prompts, or exact
replacement behavior.

## Floating bar (from parked Swift Phonon)

Pattern from anvil `~/phonon/macos` `MiniRecorderPanel`:
borderless `NSPanel`, `.nonactivatingPanel`, `.floating`, `canJoinAllSpaces`.

Sources live in `bar/` (SwiftPM). Ref copy of full app under `ref/macos-app/`.

The regular Dock app owns Home, History, Dictionary, Settings,
permissions/health, and model status. It must read and write the canonical JSON
and corpus files directly; do not add a second database or mock state. Dictionary
edits reload the warm engine without restarting model weights. The app uses the
dark ember theme; the menu-bar item and bottom capsule remain lightweight
companions, not the primary application lifecycle.

Hotkey: hold **Right Option (⌥)** push-to-talk (Ctrl+Space toggles).
Needs Accessibility + Input Monitoring + mic.

## Microphone priority

The native app enforces the input-device priority list. Yeti is priority 1 even
while unplugged; MacBook Pro Microphone is priority 2.
Re-scan CoreAudio immediately before every recording so plugging in a Yeti
makes it win the next pass without restarting Phonon. Devices outside the list
are not selected while a listed device is available. In particular, do not use
an AirPods microphone merely because it is the system default: opening it puts
Bluetooth playback into headset mode. The bar menu shows Auto's resolved mic.

The hardened native bundle must be signed with
`com.apple.security.device.audio-input`; without it macOS silently denies the
microphone before Phonon can appear in Privacy & Security. First-time access is
requested explicitly from the native permissions card, while denied or granted
states link directly to the corresponding System Settings pane.

Do not confuse with `~/dev/phonon` (GPUI settings shell only).

## Verification

```sh
cargo fmt --all -- --check
cargo nextest run --workspace
cargo clippy --workspace --all-targets -- -D warnings
PYTHONPATH=. uv run --with pytest --with numpy pytest
uv run ruff check . --fix
swift test --package-path bar
```
