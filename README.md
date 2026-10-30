# Phonon

Open-source voice typing. Fast. Local. Yours.
macOS is the shipped platform. Windows is in beta.

[Website](https://phonon.sh) · [Releases](https://github.com/Infatoshi/phonon/releases)

## Install

Apple silicon, macOS 14 or later.

```bash
brew install --cask infatoshi/phonon/phonon
phonon
```

The cask installs the Developer ID-signed and notarized app. On first launch
Phonon downloads Parakeet TDT 0.6B v2 for speech recognition and the Phonon
refiner 1.7B for correction, both pinned to exact revisions. That is about
4.9 GB (2.47 GB for Parakeet, 2.40 GB for the refiner) plus the Python packages
`uv` installs, and the one slow start; everything after it is local and offline. Tagged releases
also include the DMG for direct installation. To build from source instead, use
`brew install --formula infatoshi/phonon/phonon`.

## Windows beta

Beta. Continuous integration builds it on Windows and runs one real dictation
through it on every push, and the same flow has run end to end on a Windows
Server 2022 machine: download, first-run setup, dictation into Notepad.

Download `phonon-win.exe` from the
[latest Windows prerelease](https://github.com/Infatoshi/phonon/releases?q=win-beta)
and double-click it. It needs Windows 10 or 11 on x64. Sixteen gigabytes of
memory is comfortable; eight is the floor.

The build is not code signed. SmartScreen shows **Windows protected your PC**
the first time. Choose **More info**, then **Run anyway**. There is no way
around this without an Extended Validation certificate.

First run downloads about 3.8 GB: the sherpa-onnx tools, the llama.cpp CPU
build, Parakeet as int8 ONNX, and the Gemma correction weights. The tray
tooltip carries the progress; Windows hides new tray icons behind the **^**
at the right of the taskbar, so look there. Everything after that is local
and offline.

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
mic → Parakeet speech recognition → dictionary lookup → correction model → typed into the app
```

Both models load in parallel at launch. The startup screen reaches 100% only
after Parakeet transcribes a bundled clip through both the batch and the
streaming path, that transcript goes through the correction model, and a
representative technical correction succeeds. There is no speech-only mode.

The correction model is the Phonon refiner 1.7B, fine-tuned from LiquidAI's
LFM3-1.7B on public audio only; no personal dictation went into it. It runs on
Phonon's native Metal engine (a decode megakernel with a speculative drafter,
prefill through a patched llama.cpp), and falls back to an MLX copy of the same
model when the engine cannot run. On an M4 Max, key-up to typed text takes
about 0.3 s (median, dictations under 30 s), and the engine path is about 1.6x
faster than the earlier MLX refiner on the same weights.
`sidecar/polish_server.py` drives it. Correction is a pipeline stage, not a
provider setting: there is no way to point it at a remote model.

## Build from source

```bash
cargo install --path crates/phonon-cli --force --root ~/.local
phonon bar --rebuild   # once: builds and signs the app from this checkout
```

## Commands

```bash
phonon                       # open the app
phonon doctor                # check the runtime, models and data files
phonon bench                 # time speech recognition and correction
phonon profile model         # warm correction timings by input length
phonon profile e2e           # key-press-to-insertion timings from recent dictations
phonon dictionary --help     # terms, replacements, Wispr Flow import, evaluation
phonon dictionary import-txt # add one term per line from a text file
phonon corpus --help         # stored recordings and their transcripts
phonon stats                 # words, sessions, speaking time, dictionary fixes
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

## Local data

Nothing is retained until you say so. On first launch the app asks once whether
to keep local history and whether to read the screen for context; both start off,
and declining leaves dictation fully working. Settings carries the retention
window (keep until deleted, or 7 / 30 / 90 days) and a clear-everything button.

With local history on, every recording is kept with its metadata:

```text
~/Library/Application Support/Phonon/
├── dictionary.json
├── settings.json
└── Corpus/
    └── <source>_<timestamp>_<process>/
        ├── audio.wav
        └── metadata.json
```

`metadata.json` keeps the raw transcript, the corrected text, what you meant
(when you save it in History), dictionary changes, source, duration, and
correction timings.

Before speech recognition, an adaptive audio gate requires sustained
speech-like energy above the clip's measured noise floor. A clip without speech
stays in the corpus with `speech_detected: false`, skips both models, and
produces no text.

Working with the dictionary and the stored recordings:

```bash
phonon dictionary import-wispr
phonon dictionary import-txt
phonon dictionary test 'run v llm on black well'
phonon dictionary evaluate
phonon corpus list --search Blackwell
phonon corpus show <id>
phonon corpus set-intended <id> 'what you meant to say'
phonon dictionary learn <id> --from 'black well' --to Blackwell
```

With screen context on, Phonon reads the windows visible on each display when
a recording starts, plus any window brought to the front while you speak, with
macOS Vision OCR on the Mac. Screenshots are never saved. The OCR text only
confirms dictionary terms that already match the spoken transcript; those
terms, not the screen text, go into the correction prompt. Apps and sites on
the private list (Settings › Private apps and sites) are never read, and
password managers are always private.

### Floating bar

- One capsule at the bottom of the screen; it widens with your voice while
  listening and breathes while the text is corrected.
- Hold Globe (fn) to talk and release to type into the frontmost app.
- Double-tap Globe to keep recording; tap once to stop and type.

Globe (fn) is the default shortcut. Settings can switch the hold key to Right
Option (⌥), and either hold key can be paired with a Control-Space toggle.
macOS also acts on the Globe key, so set System Settings › Keyboard ›
"Press 🌐 key to" to "Do Nothing".

Recording cues are off by default. Turning them on plays a short rising sweep
when a recording opens and a falling one when it closes. Both are synthesized by
`scripts/make-cue-sounds.py`; no audio is sampled or downloaded.

Phonon needs Microphone, Accessibility and Input Monitoring access, plus Screen
Recording if screen context is on.

### Native app

`phonon` opens a regular macOS Dock app. Its Home, History, Dictionary, and
Settings pages read and write the same JSON files and recordings as the engine
and the CLI. Settings › Model status shows whether both models are loaded. The
`ϕ` in the menu bar stays available as a shortcut while the app runs.

Phonon uses the input selected in System Settings, except a Bluetooth headset
microphone, which would drop the headset to call quality. A priority list on
Home overrides that. Availability is checked before each recording, so a
preferred external microphone takes over as soon as it is connected. The menu
bar menu shows the microphone in use.

## Layout

```
crates/phonon-audio/   recording and audio files
crates/phonon-asr/     Parakeet sidecar lifecycle and protocol
crates/phonon-llm/     correction sidecar lifecycle and benchmark client
crates/phonon-profile/ Metal dispatch, model phase, and key-to-insertion profilers
crates/phonon-core/    pipeline coordination and engine events
crates/phonon-cli/     the phonon command: app launcher, doctor, bench, data tools
crates/phonon-hotkey/  hold, tap, and double-tap latch, shared by macOS and Windows
crates/phonon-win/     Windows tray app, keyboard hook, WASAPI capture, insertion
crates/phonon-mine/    source-identifier extraction for vocabulary mining
bar/                   SwiftPM macOS app: Home, History, Dictionary, Settings, capsule
sidecar/               speech recognition and correction servers (Python, run by uv)
prompts/               correction prompt
assets/                startup clip, English word list, recording cues
ml/                    refiner training and evaluation pipeline (runs on a GPU box)
tools/                 vocabulary-mining research tools
tools/dev/             developer benchmarks that call cloud APIs; not part of the app
website/               phonon.sh
```

## License

Phonon source is GPL-3.0. Model weights are downloaded, not stored in this
repository, and keep their own licenses: Parakeet is CC-BY-4.0, and the Phonon
refiner, as a derivative of LFM3, is under the LFM Open License v1.0, which
allows commercial use only for organizations under $10M in annual revenue. See
[THIRD_PARTY.md](THIRD_PARTY.md).
