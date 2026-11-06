# Phonon for iPhone

Dictate into any app on an iPhone with a keyboard that runs Phonon's two models on the phone:
Parakeet TDT 0.6B v2 turns speech into text, and the Phonon refiner 1.7B fixes recognition
errors, technical terms, restarts and filler the way the Mac app does. No server, no account,
no analytics. The only network request the app ever makes is the one-time model download.

This is an iOS port of [Phonon](https://github.com/Infatoshi/phonon) (GPL-3.0). It lives in
`ios/` of the same repository so the refiner prompt and guard it ports from
`sidecar/polish_server.py` stay next to their source.

## How it works

```
 Messages (or any app)                         Phonon app (background, mic open)
 ┌──────────────────────────┐   Darwin note    ┌─────────────────────────────────────┐
 │ Phonon keyboard          │ ───────────────▶ │ AVAudioEngine, 16 kHz mono           │
 │  mic / stop / cancel     │   command.json   │   │                                  │
 │  insert last             │                  │   ▼                                  │
 │                          │                  │ Parakeet TDT 0.6B v2, Core ML (ANE)  │
 │  insertText(result)  ◀───┼──────────────────│   │                                  │
 └──────────────────────────┘  app-state.json  │   ▼                                  │
             App Group folder on the phone     │ Phonon refiner 1.7B, llama.cpp (CPU) │
                                               └─────────────────────────────────────┘
```

- **The keyboard never records.** iOS gives keyboard extensions no microphone, Full Access or
  not. The keyboard sends start, stop and cancel to the app and inserts what comes back.
- **The app holds the mic for a few minutes.** The first tap opens Phonon, which starts
  recording; you go back with the app name at the top left and keep talking. After that the
  mic session stays open in the background (orange dot), so later taps work without leaving
  your app. Settings sets how long (1, 5, 15 or 60 minutes). Music keeps playing.
- **Both models run while Phonon is in the background.** Parakeet runs on the CPU and Neural
  Engine. The refiner runs on the CPU, because iOS refuses GPU work from background apps.
- **Nothing is lost to a crash.** Audio is written to disk as it is recorded and a journal
  tracks each dictation; on the next launch an unfinished one is finished and shows up under
  Insert last.

## Install on your iPhone

Needs a Mac with Xcode 26, `brew install cmake xcodegen`, and the phone connected by cable,
unlocked, with Developer Mode on (Settings > Privacy & Security > Developer Mode).

```sh
cd ios
scripts/export-ipa.sh                                   # archive, sign, build/Phonon.ipa, sha256
PHONON_INSTALL_APPROVED=yes scripts/install-on-iphone.sh <device-id>   # only when you want it on a phone
```

`export-ipa.sh` builds the native dependencies into `Vendor/` (llama.cpp and FluidAudio, a few
minutes the first time), signs a development build with automatic signing through the team's
App Store Connect API key and prints the .ipa's sha256. It touches no device.
`install-on-iphone.sh` only installs an existing .ipa, and refuses to run without
`PHONON_INSTALL_APPROVED=yes`.

Then on the phone:

1. Open Phonon and follow the five setup screens (microphone, a 1.45 GB model download over
   Wi-Fi, keyboard).
2. Settings > Phonon > Keyboards: turn on Phonon and Allow Full Access.
3. In Messages, hold the globe key, pick Phonon, tap the mic.

## Models

Downloaded at first run from Hugging Face, pinned to exact commits and checked byte for byte
(sha256 per file, `PhononCore/Sources/PhononCore/ModelManifest+Pinned.swift`). A dropped
download resumes from where it stopped.

| Model | Files | Size | License |
| --- | --- | --- | --- |
| [Parakeet TDT 0.6B v2, Core ML](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v2-coreml) @ `ee09c56` | Preprocessor, Encoder, Decoder, JointDecision, vocab | 464 MB | CC BY 4.0 |
| [Phonon refiner 1.7B](https://huggingface.co/Infatoshi/phonon-refiner-1.7b) @ `2b0ab74` | `phonon-refiner-1.7b-Q4_K_S-e4.gguf` | 988 MB | LFM Open License v1.0 |

The Mac app's DFlash2 drafter is not shipped: Phonon's speculative decoding is checked against
greedy decoding, so it changes speed, not text.

## Build and test

```sh
cd ios
scripts/vendor.sh && xcodegen generate
(cd PhononCore && swift test)                       # prompt parity, downloader, IPC, journal
xcodebuild -project Phonon.xcodeproj -scheme Phonon \
  -destination 'platform=iOS Simulator,name=<a simulator>' test   # fixtures through both models
```

The fixture test reads the models from `PHONON_MODELS_DIR` (set in the scheme to
`../../work/models`; point it at any folder made with
`hf download <repo> --revision <sha> --local-dir`).

| Path | What it is |
| --- | --- |
| `Phonon/Engine/DictationEngine.swift` | Mic session, pipeline, keyboard bridge, crash recovery |
| `Phonon/Engine/LlamaRefiner.swift` | The refiner on llama.cpp with the trained prompt and guard |
| `Phonon/Engine/Recorder.swift` | AVAudioEngine capture, interruptions, route changes |
| `PhononKeyboard/` | The keyboard extension |
| `PhononCore/` | Shared code: prompt, manifest, downloader, App Group store, journal |
| `scripts/` | Vendor build, install, stills, manifest and golden generators |

## License

GPL-3.0, like the rest of Phonon. Model weights and linked libraries keep their own licenses;
see [LICENSES.md](LICENSES.md).
