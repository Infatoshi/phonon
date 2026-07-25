# Distribution

Phonon's public release is installed from source through its Homebrew tap. Model
weights are not stored in the repository; the app downloads the open Parakeet and
Gemma weights on first launch. Tagged releases also provide a Developer ID-signed
and notarized DMG for direct installation.

## Runtime layout

- The app bundle contains the native Swift UI, Rust engine, signed arm64 `uv`
  runtime, both sidecars, the prompt, the English word list, and the startup
  audio fixture.
- ASR uses the stock `mlx-community/parakeet-tdt-0.6b-v2` model, pinned by
  revision. It is public, ungated, CC-BY-4.0, and approximately 2.3 GB on disk.
- Correction uses `mlx-community/gemma-4-e2b-it-4bit`, also pinned by revision,
  approximately 3.3 GB on disk, run locally on `mlx-lm==0.31.3` through the
  bundled `uv`. First launch therefore downloads roughly 5.6 GB in total.
- Production Phonon does not redistribute FluidVoice or its private MLX helper,
  and no longer depends on either. The fluid-1 baseline remains reachable behind
  `PHONON_POLISH_BACKEND=fluid` and `phonon bench --baseline` for local
  comparison only.

This requires no Phonon file-storage service. Hugging Face hosts the upstream
weights, GitHub hosts tagged source releases, and Vercel hosts the static website.

## Release channels

- ASR-only fallback is not a supported release mode. The correction stage is
  mandatory and is now bundled, which clears the blocker that held the previous
  public build.
- GitHub releases include a signed, notarized `Phonon.dmg` for direct download.
- The bundled engine and small runtime resources do not depend on a checkout.
- A correction model that cannot be downloaded or loaded is a startup failure,
  not a degraded mode.
- The DMG retains the same local model-download design and does not bundle model
  weights or user data.

## Apple distribution requirement

Public direct-download distribution requires an active Apple Developer Program
membership, a Developer ID Application certificate, and notarization. A free
Personal Team development identity is not sufficient.

## Release command

Store notarization credentials once with `xcrun notarytool store-credentials`,
then run:

```bash
PHONON_NOTARY_PROFILE=phonon-notary scripts/release-dmg.sh
```

The script builds the app, signs nested executables and the bundle with hardened
runtime and secure timestamps, creates and signs `bar/dist/Phonon.dmg`, submits
it to Apple, saves the notarization log, staples the ticket, and verifies the
final artifact with Gatekeeper.
