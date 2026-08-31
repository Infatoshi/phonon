# Phonon

Open-source voice typing. Fast, local, and sovereign. macOS is the shipped
platform. Windows is in beta.

[Website](https://phonon.sh) · [Distribution plan](DISTRIBUTION.md)

## Install

```bash
brew install --cask infatoshi/phonon/phonon
phonon
```

The cask installs the Developer ID-signed and notarized app. On first launch
Phonon downloads the open Parakeet and Gemma weights, both pinned to exact
revisions, which is about 5.6 GB and the one slow start. Everything after that
is local and offline. Tagged releases also include the DMG for direct
installation. To build locally instead, use
`brew install --formula infatoshi/phonon/phonon`.

## Windows beta

Beta. Continuous integration builds it on Windows and runs one real dictation
through it on every push. Nobody has run it on a real Windows machine yet.

Download `phonon-win.exe` from the
[latest Windows prerelease](https://github.com/Infatoshi/phonon/releases?q=win-beta)
and double-click it. It needs Windows 10 or 11 on x64. Sixteen gigabytes of
memory is comfortable; eight is the floor.

The build is not code signed. SmartScreen shows **Windows protected your PC**
the first time. Choose **More info**, then **Run anyway**. There is no way
around this without an Extended Validation certificate.

First run downloads about 3.8 GB: the sherpa-onnx tools, the llama.cpp CPU
build, Parakeet as int8 ONNX, and the Gemma correction weights. The tray
tooltip carries the progress. Everything after that is local and offline.

Hold **Right Ctrl** to dictate. Double-tap it to latch, then press it again to
stop. Right-click the tray icon for the menu. Phonon swallows Right Ctrl while
it runs, so Right Ctrl stops acting as a modifier; Left Ctrl is untouched.
`PHONON_WIN_HOTKEY` takes `rightctrl`, `leftctrl`, `rightalt`, `rightshift`,
`capslock`, or `f13`.

```
phonon-win.exe info      # what first run downloads, and what is installed
phonon-win.exe fetch     # download and verify, then stop
phonon-win.exe selftest  # one real pass through both models; the release gate
```

Everything lives in `%LOCALAPPDATA%\Phonon`. Delete that folder to start over.

## Pipeline

```
mic → Parakeet ASR → dictionary retrieval → Gemma correction → clipboard / type
```

The runtime requires and loads two weight streams in parallel:
**asr ∥ llm**. The single startup loader reaches 100% only after Parakeet
transcribes the bundled fixture through batch and streaming ASR, that transcript
survives a round trip through the correction model, and a representative
technical correction succeeds. Phonon does not expose an ASR-only mode.

The correction stage is `sidecar/polish_server.py`: `mlx-community/gemma-4-e2b-it-4bit`
on `mlx-lm`, run locally through `uv`. It is a pipeline stage, not a provider
setting, and there is no way to point it at a remote model.

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

## Surviving an uninstall

Uninstallers match on the app name, the developer name, or the bundle
identifier, then delete every hit under `~/Library/Application Support`,
`Preferences`, `Caches`, `Logs`, and `Saved Application State`. Nothing kept
there can survive one.

So Phonon mirrors the small irreplaceable files to `~/.phonon/backup`, which no
such pattern reaches: the dictionary, settings, history, vocabulary and word
replacements, about 128 KB. The mirror is rewritten whenever any of them change,
and a wiped store can never overwrite a populated backup.

If Phonon ever starts with an empty data folder while that backup exists, it
asks whether to restore it, and offers to delete it instead. It never restores
silently. Settings › Backup shows what is held, reveals it in Finder, deletes
it, and exports everything including the corpus.

Model weights are unaffected. They live in `~/.cache/huggingface`, are named
after the models rather than the app, and no uninstaller touches them.

## Known issues

- Opening Phonon can pull a connected Bluetooth headset into the hands-free
  profile, which degrades its output quality. Reselect the input in System
  Settings › Sound as a workaround.
  See [#9](https://github.com/Infatoshi/phonon/issues/9).

## Local data and correction loop

Nothing is retained until you say so. On first launch the app asks once whether
to keep local history and whether to use active-window context; both start off,
and declining leaves dictation fully working. Settings carries the retention
window (keep until deleted, or 7 / 30 / 90 days) and a clear-everything button.

With local history on, every native-app recording is retained as a paired corpus
item:

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
`speech_detected: false`, skip Parakeet and the correction model entirely, and produce no text.

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
confirmed terms, not the full screen text, into the correction prompt.

### Floating bar

- One bottom-anchored capsule that expands itself; no spawned overlay or inner waveform
- Warm voice-responsive fill while listening; the same capsule breathes while processing
- **hold Globe (fn)** PTT → release → ASR → auto-polish → types into frontmost app
- **double-tap Globe** latches the recording on; one tap stops and inserts
- Ctrl+Space toggle; multi-pass OK

Globe (fn) is the default shortcut. Settings can switch the hold key to Right
Option (⌥), and either hold key can be paired with the Ctrl+Space toggle.
macOS also acts on the Globe key, so set System Settings › Keyboard ›
"Press 🌐 key to" to "Do Nothing".

Recording cues are off by default. Turning them on plays a short rising sweep
when a recording opens and a falling one when it closes. Both are synthesized by
`scripts/make-cue-sounds.py`; no audio is sampled or downloaded.

Needs: Mic + Accessibility + Input Monitoring.

### Native app

`phonon` launches a regular macOS Dock app with a dark ember theme. Its native
Home, History, Dictionary, and Settings surfaces read and write the same JSON
and paired corpus used by the engine and CLI. Model status exposes parallel
weight loading, startup smoke tests, TTFT, and decode throughput. The
menubar `ϕ` remains available as a compact shortcut while the app is running.

The native app uses a configurable microphone priority list. Availability is
refreshed before each recording, so a preferred external microphone can take
over automatically when connected. The bar menu shows the microphone selected
by Auto.

## Layout

```
crates/phonon-audio/   recording and audio file ownership
crates/phonon-asr/     Parakeet sidecar lifecycle + protocol
crates/phonon-llm/     correction sidecar lifecycle + benchmark client
crates/phonon-profile/ literal Metal dispatch, LLM phase, and E2E profilers
crates/phonon-core/    pipeline coordination + engine events
crates/phonon-cli/     commands, doctor, bench, bar launcher
crates/phonon-hotkey/  hold, tap, and double-tap latch, shared by both platforms
crates/phonon-win/     Windows tray app, keyboard hook, WASAPI capture, insertion
bar/                   SwiftPM native Home/History/Dictionary/Settings app + floating pill
sidecar/asr_server.py
sidecar/polish_server.py
assets/english_words.txt
prompts/polish_v2.txt
```

## License

Phonon source is GPL-3.0. Downloaded model weights retain their upstream
licenses and are not stored in this repository. See [THIRD_PARTY.md](THIRD_PARTY.md).
