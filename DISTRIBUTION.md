# Distribution

Phonon's public release is installed from source through its Homebrew tap. Model
weights are not stored in the repository; the app downloads the open Parakeet
weights on first launch. This source-built route works before a Developer ID
binary release is available.

## Runtime layout

- The app bundle contains the native Swift UI, Rust engine, ASR sidecar, prompt,
  and startup audio fixture.
- ASR uses the stock `mlx-community/parakeet-tdt-0.6b-v2` model. It is public,
  ungated, CC-BY-4.0, and approximately 2.47 GB.
- Deterministic dictionary correction is always available. Fluid-1 correction
  is optional until its public runtime integration replaces the development-only
  FluidVoice helper.
- Production Phonon does not redistribute FluidVoice or its private MLX helper.

This requires no Phonon file-storage service. Hugging Face hosts the upstream
weights, GitHub hosts tagged source releases, and Vercel hosts the static website.

## Release channels

- Available now: `brew install infatoshi/phonon/phonon` builds the tagged source,
  installs the self-contained app, and uses the open Parakeet runtime.
- The bundled engine and small runtime resources do not depend on a checkout.
- The open release falls back to deterministic dictionary correction when the
  optional Fluid-1 runtime is unavailable.
- Later: a Developer ID-signed and notarized DMG/cask can provide a prebuilt
  binary while retaining the same local model-download design.

## Apple distribution requirement

Public direct-download distribution requires an active Apple Developer Program
membership, a Developer ID Application certificate, and notarization. A free
Personal Team development identity is not sufficient.
