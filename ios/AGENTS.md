# AGENTS.md

Notes for anyone (person or agent) building, testing or changing Phonon for iPhone. Every command
runs from the folder that holds `project.yml`.

## Setup

- A Mac with Xcode 26 and an iOS 26 simulator runtime.
- `brew install cmake xcodegen`.
- `scripts/vendor.sh` builds the native dependencies into `Vendor/` (ignored by git):
  llama.cpp at a pinned commit as `llama.xcframework` with A15 kernels for the device and M1
  kernels for the simulator, and FluidAudio at a pinned commit with Swift 5 language mode. It
  checks both SHAs on every run and stops if a checkout moved. The first run takes a few minutes.
- `xcodegen generate` writes `Phonon.xcodeproj` (ignored by git). Rerun it after any change to
  `project.yml` or after adding a file.

## Test

| Command | What it proves | Needs |
| --- | --- | --- |
| `(cd PhononCore && swift test)` | Refiner prompt, token budget, guard, cleanup and term restore match the Mac app's `polish_server.py` on golden cases; downloader resume and hash checks; App Group store; crash journal. 11 tests. | nothing |
| `scripts/check-offline-asr.sh` | No code path can reach a FluidAudio model download; `ModelDownloader.swift` is the only network code. | nothing |
| `xcodebuild -project Phonon.xcodeproj -scheme Phonon -destination 'id=<sim-udid>' -derivedDataPath build/dd test` | Four spoken fixtures through both models on the simulator, plus the model loader tests. | models in `build/models` |
| `scripts/sim-keyboard-flow.sh <sim-udid> build/models <out-dir> <clip.mp4>` | The keyboard end to end in Messages: add the keyboard with Full Access, pick it with the globe key, mic, handoff to Phonon, stop, text inserted. | models, about 90 s |
| `scripts/sim-stills.sh <sim-udid> build/models <out-dir>` | Light and dark stills of every screen at actual pixels. | models, a build in `build/dd` |

Models for the simulator tests: put both pinned model folders in `build/models`
(`parakeet-tdt-0.6b-v2-coreml` and `phonon-refiner-1.7b`), made with
`hf download <repo> --revision <sha> --local-dir build/models/<folder>`. The repos and revisions
are in `scripts/make-manifest.py`. A symlink to a folder elsewhere works.

Regenerate generated files when their source changes:

- `scripts/make-golden.py <phonon checkout>/sidecar/polish_server.py PhononCore/Tests/PhononCoreTests/golden.json`
  after the Mac app's refiner prompt or guard changes.
- `scripts/make-manifest.py build/models PhononCore/Sources/PhononCore/ModelManifest+Pinned.swift`
  after a model revision changes.
- `scripts/make-fixtures.sh` rebuilds the spoken fixtures with macOS `say`.

## The simulator cache trap

Every install of the app in a simulator compiles the Core ML models again into
`Library/Caches/<bundle id>/com.apple.e5rt.e5bundlecache` inside the app's container, about
2.2 GB per install, and interrupted compiles leave `*.tmp.*` bundles that are never reused. One
simulator reached 40 GB this way. Two guards:

- The app empties that cache on the first launch of a new install and removes other processes'
  partial compiles on every launch (`Phonon/Engine/ModelCache.swift`).
- `scripts/sim-prune-cache.sh <sim-udid>` removes partial compiles and empties the cache when it
  is over 3 GB. The keyboard flow and stills scripts call it at the end; call it yourself after
  `xcodebuild test`.

Check free disk space before long simulator runs.

## Signing

- Simulator builds and tests need no signing.
- To run on a phone, change the bundle id prefix and App Group from `com.abhaymettu` to your own:
  `git grep -l com.abhaymettu | xargs sed -i '' 's/com\.abhaymettu/com.example/g'`, then
  `xcodegen generate`. The App Group in `PhononCore/Sources/PhononCore/Shared.swift` must match
  both entitlements files.
- `PHONON_TEAM=<team id> scripts/export-ipa.sh` archives a development build signed with
  automatic signing through the account in Xcode > Settings > Accounts. It refuses a dirty tree
  and stamps the commit into Info.plist (`PhononCommit`, shown in Settings > Diagnostics > Build).
  The team id is never stored in the repository.
- The app uses the App Groups and Increased Memory Limit capabilities; automatic signing creates
  both. If signing fails with a license error, the account holder has to accept the latest
  Program License Agreement on developer.apple.com first.
- Signing with an App Store Connect API key (`ASC_KEY_PATH`, `ASC_KEY_ID`, `ASC_ISSUER_ID`) created
  the certificate but failed Xcode's App ID step with "Authentication failed" on Xcode 26.6; the
  signed-in account route works.
- `scripts/install-on-iphone.sh` installs an existing .ipa on a connected phone and refuses to run
  without `PHONON_INSTALL_APPROVED=yes`.
- Never commit `.p8` keys, provisioning profiles, team ids or device ids.

## Conventions

- The keyboard extension never touches the microphone; iOS does not allow it. It talks to the app
  through Darwin notifications and JSON files in the App Group folder (`Shared.swift`).
- The refiner prompt and guard in `RefinerPrompt.swift` must stay identical to the Mac app's;
  the golden test enforces it.
- The model downloader is the only network code. Keep `scripts/check-offline-asr.sh` passing.
- Dependencies are pinned by commit in `scripts/vendor.sh`; change the pin and rerun it to update.
