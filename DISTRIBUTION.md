# Distribution

Phonon's public release is installed from its Homebrew tap (a cask for the
signed DMG, and a formula that builds from source) or from the DMG on GitHub
releases. Model weights are not stored in the repository; the app downloads the
open Parakeet weights and Phonon's public refiner on first launch. Tagged
releases provide a Developer ID-signed and notarized DMG.

## Runtime layout

- The app bundle contains the native Swift UI, Rust engine, signed arm64 `uv`
  runtime, both sidecars and their locked Python environments
  (`Contents/Resources/runtime/{asr,polish}`), the native refiner engine (`Contents/Helpers/pe`,
  with its Metal shader sources in `Contents/Resources/engine`), the prompt,
  the English word list, the startup audio fixture, and third-party notices.
- ASR uses the stock `mlx-community/parakeet-tdt-0.6b-v2` model, pinned by
  revision. It is public, ungated, CC-BY-4.0, and approximately 2.5 GB.
- Correction uses Phonon's public refiner, `Infatoshi/phonon-refiner-1.7b`,
  pinned by revision (`crates/phonon-llm/src/paths.rs`). It is fine-tuned from
  LiquidAI/LFM3-1.7B on transcripts of public audio and synthetic transcripts
  (no personal dictations) and is distributed under the LFM Open License
  v1.0. It takes a writing-mode line chosen from the target app (minimal,
  casual, formal, prompt) and lists those lines in `phonon_refiner.json`. One repository holds what the app needs, approximately
  2.4 GB: the engine's GGUF target and DFlash2 drafter (`phonon_engine.json`),
  and the same refiner as MLX weights. First launch therefore downloads roughly
  4.9 GB in total, plus `uv`'s Python packages.
- Those Python packages come from `runtime/asr/uv.lock` and
  `runtime/polish/uv.lock`, which pin every package, transitive dependencies
  included, to an exact version and SHA-256. The sidecars start with
  `uv run --frozen`, so uv installs only what the lock names and rejects any
  file whose hash does not match. The environments live in
  `~/Library/Application Support/Phonon/runtime/`, outside the signed bundle.
  To change a version, edit the requirement in the runtime's
  `pyproject.toml` (and its constant in `phonon-asr` or `phonon-llm`), run
  `uv lock --project runtime/<name>`, and commit both files; CI fails when a
  lock is out of date.
- The sidecar runs the refiner on the bundled engine. When the engine cannot
  start on a Mac, or its first request does not return real text, the sidecar
  loads the MLX weights from the same download instead (`mlx-lm==0.31.3`);
  dictation is the same text path, only slower. The startup window says so.
- A refiner trained on the owner's own dictations, under
  `~/Library/Application Support/Phonon/models/refiner/`, replaces the
  download on that Mac only. It is never published.
- Gemma 4 E2B is no longer downloaded. A developer can still compare against
  it with `PHONON_POLISH_MODEL=mlx-community/gemma-4-e2b-it-4bit` and
  `PHONON_POLISH_REVISION=238767527555cb75a05732a84dff5d6ba0dd6809`.
- Production Phonon does not redistribute FluidVoice or its private MLX helper,
  and does not depend on either. The fluid-1 baseline remains reachable behind
  `PHONON_POLISH_BACKEND=fluid` and `phonon bench --baseline` for local
  comparison only.

This requires no Phonon file-storage service. Hugging Face hosts the weights,
GitHub hosts tagged source releases and the DMG, and Vercel hosts the static
website.

## Release channels

- ASR-only fallback is not a supported release mode. The correction stage is
  mandatory. A correction model that cannot be downloaded or loaded is a
  startup failure, not a degraded mode.
- GitHub releases include a signed, notarized `Phonon.dmg`. It always carries
  the native engine (`release-dmg.sh` fails without it).
- The Homebrew formula builds from source and does not build the engine; it
  corrects on the MLX refiner. Run `scripts/build-engine.sh` before
  `scripts/package-bar.sh` to bundle the engine in a source build.
- The DMG keeps the local model-download design and does not bundle model
  weights or user data.
- From 1.0 the app checks GitHub's releases API once a day and tells the user
  about a newer release (SPEC "Update check"). It announces a release as soon as
  its `Phonon.dmg` is attached, so bump the tap's cask in the same sitting.

## The native refiner engine

`engine/` holds its source (`pe.mm`, `gguf.cpp`, and the Metal shaders it
compiles at startup). It links llama.cpp, MIT, at upstream commit
`39febd2adb947b70da35e95f9351b3bf58ad170a` with
`third_party/llama.cpp/phonon-engine.patch` applied. `scripts/build-engine.sh`
fetches that commit, applies the patch and builds `target/engine/pe` for the
app's minimum macOS (14.0). `package-bar.sh` refuses an engine whose minimum
macOS is newer than the app's: the Command Line Tools default is the build
machine's own version, which dyld refuses on every older Mac.

## Apple distribution requirement

Public direct-download distribution requires an active Apple Developer Program
membership, a Developer ID Application certificate, and notarization. A free
Personal Team development identity is not sufficient.

## Release command

Store notarization credentials once, from an App Store Connect team API key
(App Store Connect, Users and Access, Integrations; it does not expire the way
an app-specific password does):

```bash
xcrun notarytool store-credentials phonon-notary-key \
  --key ~/.appstoreconnect/private_keys/AuthKey_<KEY_ID>.p8 \
  --key-id <KEY_ID> --issuer <ISSUER_ID>
```

Upload the refiner and pin its commit in `crates/phonon-llm/src/paths.rs`,
then run:

```bash
scripts/build-engine.sh
scripts/release-dmg.sh
```

A notarization request answered with HTTP 403 means the account holder has an
updated developer agreement to accept at developer.apple.com.

The script refuses the placeholder refiner revision, builds the app with the
engine, signs nested executables and the bundle with hardened runtime and
secure timestamps, creates and signs `bar/dist/Phonon.dmg`, submits it to
Apple, saves the notarization log, staples the ticket, and verifies the final
artifact with Gatekeeper.
