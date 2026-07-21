# Phonon

Open-source voice typing for macOS. Fast, local, and sovereign.

[Website](https://phonon.sh) · [Distribution plan](DISTRIBUTION.md)

## Install

```bash
brew install infatoshi/phonon/phonon
phonon
```

The Homebrew release builds the app locally and downloads the open Parakeet
weights on first launch. A Developer ID-notarized binary release can follow
without changing the open-source install path.

## Pipeline

```
mic → Parakeet ASR → auto fluid-1 polish (+ Gemma MTP) → clipboard / type
```

The current development runtime loads three weight streams in parallel:
**asr ∥ fluid-1 ∥ mtp**. The
single startup loader reaches 100% only after Parakeet transcribes the bundled
fixture and fluid-1 completes a representative request through the MTP drafter.
The first user dictation therefore never pays the generic or first-request
warmup path.

## Build from source

```bash
cargo install --path crates/phonon-cli --force --root ~/.local
phonon bar --rebuild   # once, builds Swift floating pill
```

## Commands

```bash
phonon                 # native macOS app
phonon bar             # explicit native-app launcher
phonon engine          # warm JSONL backend (bar uses this)
phonon doctor
phonon bench
phonon profile --help   # discover kernel, model, and end-to-end profilers
phonon profile kernel   # literal MLX Metal kernel top 10 for a warmed request
phonon profile model    # warm-only LLM prefill / decode sweep
phonon profile e2e      # summarize real keyboard-to-insertion traces
phonon dictionary --help # terms, replacements, Wispr import, correction evaluation
phonon dictionary import-txt # activate dictionary_new_terms.txt
phonon corpus --help     # paired WAV/metadata corpus + intended transcripts
phonon stats             # local words, sessions, speaking time, dictionary fixes
```

## Local data and correction loop

Every native-app recording is retained as a paired corpus item:

```text
~/Library/Application Support/Phonon/
├── dictionary.json
├── settings.json
└── Corpus/
    └── <source>_<timestamp>_<process>/
        ├── audio.wav
        └── metadata.json
```

`metadata.json` keeps the raw ASR transcript, final correction, optional intended
transcript, dictionary changes, source, duration, and LLM timings. Useful loops:

Before ASR, an adaptive audio gate requires sustained speech-like energy above
the clip's measured noise floor. Clips without speech remain in the corpus with
`speech_detected: false`, skip Parakeet and Fluid entirely, and produce no text.

```bash
phonon dictionary import-wispr
phonon dictionary import-txt
phonon dictionary test 'run v llm on black well'
phonon dictionary evaluate
phonon corpus list --search Blackwell
phonon corpus show <id>
phonon corpus set-intended <id> 'intended ground truth'
phonon dictionary learn <id> --from 'black well' --to Blackwell
```

When `screen_context` is enabled in `settings.json`, the floating bar captures
each visible display once at recording start and runs local macOS Vision OCR.
Screenshots are discarded immediately. The engine uses OCR only to rank
dictionary candidates already relevant to the spoken transcript; it sends
confirmed terms, not the full screen text, into the Fluid correction prompt.

### Floating bar

- One bottom-anchored capsule that expands itself; no spawned overlay or inner waveform
- Warm voice-responsive fill while listening; the same capsule breathes while processing
- **hold Option (⌥)** PTT → release → ASR → auto-polish → types into frontmost app
- Ctrl+Space toggle; multi-pass OK

Needs: Mic + Accessibility + Input Monitoring.

### Native app

`phonon` launches a regular macOS Dock app with a dark ember theme. Its native
Home, History, Dictionary, and Settings surfaces read and write the same JSON
and paired corpus used by the engine and CLI. Model status exposes parallel
weight loading, startup smoke tests, TTFT, and speculative throughput. The
menubar `ϕ` remains available as a compact shortcut while the app is running.

The native app uses a configurable microphone priority list. Availability is
refreshed before each recording, so a preferred external microphone can take
over automatically when connected. The bar menu shows the microphone selected
by Auto.

## Layout

```
crates/phonon-audio/   recording and audio file ownership
crates/phonon-asr/     Parakeet sidecar lifecycle + protocol
crates/phonon-llm/     Fluid/MTP lifecycle + benchmark client
crates/phonon-profile/ literal Metal dispatch, LLM phase, and E2E profilers
crates/phonon-core/    pipeline coordination + engine events
crates/phonon-cli/     commands, doctor, bench, bar launcher
bar/                   SwiftPM native Home/History/Dictionary/Settings app + floating pill
sidecar/asr_server.py
prompts/polish_v1.txt
```

## License

Phonon source is GPL-3.0. Downloaded model weights retain their upstream
licenses and are not stored in this repository. See [THIRD_PARTY.md](THIRD_PARTY.md).
