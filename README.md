# Phonon

Open-source voice typing. Fast. Local. Yours.
macOS is the shipped platform. Windows is in beta; Linux is a preview.

[Website](https://phonon.sh) · [Releases](https://github.com/Infatoshi/phonon/releases)

## Install

Apple silicon, macOS 14 or later.

```bash
brew install --cask infatoshi/tap/phonon
phonon
```

The cask installs the Developer ID-signed and notarized app. On first launch
Phonon downloads Parakeet TDT 0.6B v2 for speech recognition and the Phonon
refiner 1.7B for correction, both pinned to exact revisions. That is about
4.9 GB (2.47 GB for Parakeet, 2.40 GB for the refiner) plus the Python packages
`uv` installs, and the one slow start; everything after it is local and offline, apart from a
once-a-day check of GitHub for a newer release, which carries nothing about you and can be turned off
in Settings. Tagged releases
also include the DMG for direct installation. To build from source instead, use
`brew install --formula infatoshi/tap/phonon`.

## The recipe

- **Hear:** Parakeet TDT 0.6B v2 transcribes on your Mac, on Phonon's own Metal kernels.
- **Rewrite:** the Phonon refiner 1.7B, fine-tuned from LiquidAI's LFM3-1.7B, turns the transcript into
  what you meant to type: misheard terms fixed, filler and false starts gone, only the final version kept
  when you correct yourself.
- **Data:** 76,960 transcripts of public audio and synthetic dictations, each rewritten by DeepSeek
  following a written spec, in four writing modes (casual, formal, prompt, minimal). No personal dictation
  went into it. The pipeline is in `ml/`.
- **Mode:** picked from the app you dictate into: casual in Discord, formal in Slack and email, prompt in
  AI assistants and terminals, minimal everywhere else.
- **Speed:** both models work while you talk, so at key-up only the last seconds are left. The refiner runs
  on a native Metal engine (a decode megakernel with a speculative drafter). Key-up to pasted text is about
  65 ms on an M4 Max.

## Windows beta

Beta. Continuous integration builds it on Windows and runs one real dictation
through it on every push, and the same flow has run end to end on a Windows
Server 2022 machine: download, first-run setup, dictation into Notepad.

Download `Phonon-Setup-<version>.exe` from the
[latest Windows prerelease](https://github.com/Infatoshi/phonon/releases?q=win-beta)
and double-click it. It installs for your user only (no administrator needed),
adds Phonon to the Start menu and to sign-in, and starts it. If the PC lacks the
Microsoft Visual C++ runtime, which the downloaded speech and correction
runtimes link against, the installer runs Microsoft's redistributable and
Windows asks once for permission. Uninstall from **Settings > Apps**; it asks
whether to keep the downloaded models. It needs Windows 10 or 11 on x64. Sixteen
gigabytes of memory is comfortable; eight is the floor.

The bare `phonon-win.exe` still runs without installing. It then needs the
[Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe)
on its own; Phonon checks before downloading anything and says so if it is missing.

The build is not code signed. SmartScreen shows **Windows protected your PC**
the first time. Choose **More info**, then **Run anyway**. There is no way
around this without an Extended Validation certificate.

First run downloads about 1.6 GB: the sherpa-onnx tools, the llama.cpp build
(CPU, plus Vulkan for a graphics card), Parakeet as int8 ONNX, and Phonon's
refiner (the correction model the Mac app uses). With a Vulkan GPU that decodes
fast enough, correction runs on it; otherwise on the CPU. The tray
tooltip carries the progress; Windows hides new tray icons behind the **^**
at the right of the taskbar, so look there. Everything after that is local
and offline.

Hold **Right Ctrl** to dictate, or tap it to start and tap it again to
stop. Right-click the tray icon for the menu. Phonon swallows Right Ctrl while
it runs, so Right Ctrl stops acting as a modifier; Left Ctrl is untouched.
To use another key, pick it under **Dictation key** in the tray menu (Right
Ctrl, Right Alt, Ctrl+Space, Right Shift, Caps Lock or F13). On a keyboard with
a Copilot key instead of Right Ctrl, Phonon switches to Right Alt (Ctrl+Space
where Right Alt types characters) the first time that key is pressed.
`PHONON_WIN_HOTKEY` overrides the menu and takes `rightctrl`, `leftctrl`, `rightalt`, `rightshift`,
`capslock`, or `f13`.

```
phonon-win.exe info      # what first run downloads, and what is installed
phonon-win.exe fetch     # download and verify, then stop
phonon-win.exe selftest  # one real pass through both models; the release gate
```

Everything lives in `%LOCALAPPDATA%\Phonon`. Delete that folder to start over.

## Linux preview

Ubuntu 24.04 or later and Debian-family systems on x64, GNOME or KDE Plasma, Wayland or X11. Continuous
integration builds `phonon_<version>_amd64.deb` and runs one real dictation through the binary on every push.

```bash
sudo apt install ./phonon_<version>_amd64.deb
phonon run      # or log out and back in: it starts at sign-in
```

The package installs a udev rule that lets the signed-in user read the keyboard and create a virtual one (the
same mechanism Steam uses for controllers), so no group membership or logout is needed. Phonon reads the
dictation key from the kernel, which works under Wayland where apps get no global hotkeys, and types by pasting
with a virtual Control-V, putting your clipboard back afterwards. Terminals paste with Control-Shift-V, so in a
terminal the text stays on the clipboard. Phonon does nothing while the screen is locked.

First run downloads the same 1.6 GB as Windows, into `~/.local/share/phonon`. Hold **Right Ctrl** to dictate;
`PHONON_HOTKEY` takes the Windows key names. Without the package, `phonon setup` prints the one-time rule to
install.

## Pipeline

```
mic → Parakeet speech recognition → dictionary lookup → correction model → typed into the app
```

Both models load in parallel at launch. The startup screen reaches 100% only
after Parakeet transcribes a bundled clip through both the batch and the
streaming path, that transcript goes through the correction model, and a
representative technical correction succeeds. There is no speech-only mode.

The correction model is the Phonon refiner 1.7B, fine-tuned from LiquidAI's
LFM3-1.7B on transcripts of public audio and synthetic transcripts; no personal
dictation went into it. It writes what you meant to type: misheard terms
fixed, filler and false starts gone, and only the final version kept when you
correct yourself. Its writing mode follows the app you dictate into: casual in
personal chat (Discord, Messages, WhatsApp), formal in email and work chat
(Slack, Teams), prompt in AI assistants, terminals and editors, minimal
everywhere else. It runs on
Phonon's native Metal engine (a decode megakernel with a speculative drafter,
prefill through a patched llama.cpp), and falls back to an MLX copy of the same
model when the engine cannot run. Parakeet also runs on Phonon's own Metal
kernels, with MLX as the fallback. On Macs with 16 GB or more, both stages work
while you talk: Parakeet re-transcribes the recording every 0.4 s and the
refiner corrects the transcript so far, so at key-up only the last seconds are
transcribed again. On an M4 Max that took the median from key release to
corrected text from about 215 ms to about 30 ms (40 real dictations, measured in
the engine; pasting adds a few ms), with the same text as transcribing the whole
recording at once. While the key is held, a card above the capsule shows the
corrected text so far and the words just heard after it. It changes only while
you speak, and words more than a dozen back stay put until key-up (Settings
turns it off). Dictations of any length stream: past about 200 words the older
sentences are corrected once and kept, so key-up stays short.
`sidecar/polish_server.py` drives it. Correction is a pipeline stage, not a
provider setting: there is no way to point it at a remote model.

## Build from source

```bash
cargo install --path crates/phonon-cli --force --root ~/.local
phonon bar --rebuild   # once: builds and signs the app from this checkout
```

## Dictating from another device

With Settings › Advanced › WebSocket API on, a phone or any other device on your network or
tailnet can dictate through the Mac: it streams its audio and names the app it types into, and
gets the corrected text back. The protocol is in
[docs/websocket-api.md](docs/websocket-api.md). The Android app's "Phonon voice" input method
is one client: turn off FUTO Keyboard's built-in voice input and its mic key opens it.

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
- Or tap Globe once to keep recording; tap again to stop and type.

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

`phonon` opens a macOS Dock app (Settings › System › Show in Dock off keeps it in
the menu bar only). Its Home, History, Dictionary, and
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
crates/phonon-hotkey/  hold-to-talk and tap-to-toggle, shared by macOS and Windows
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
