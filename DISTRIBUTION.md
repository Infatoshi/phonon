# Distribution

Phonon's public release is installed from source through its Homebrew tap. Model
weights are not stored in the repository; the app downloads the open Parakeet
weights on first launch. Tagged releases also provide a Developer ID-signed and
notarized DMG for direct installation.

## Runtime layout

- The app bundle contains the native Swift UI, Rust engine, signed arm64 `uv`
  runtime, ASR sidecar, prompt, and startup audio fixture.
- ASR uses the stock `mlx-community/parakeet-tdt-0.6b-v2` model. It is public,
  ungated, CC-BY-4.0, and approximately 2.47 GB.
- Fluid-1 correction and its speculative drafter are required. A release must
  bundle or install an open compatible runtime before it can be published.
- Production Phonon does not redistribute FluidVoice or its private MLX helper.

This requires no Phonon file-storage service. Hugging Face hosts the upstream
weights, GitHub hosts tagged source releases, and Vercel hosts the static website.

## Release channels

- The next public build remains blocked until the open correction runtime is
  bundled; ASR-only fallback is not a supported release mode.
- GitHub releases include a signed, notarized `Phonon.dmg` for direct download.
- The bundled engine and small runtime resources do not depend on a checkout.
- Missing Fluid or MTP weights are a startup failure, not a degraded mode.
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
