# Distribution

Phonon is distributed as a small notarized macOS application. Model weights are
not embedded in the DMG. The app downloads and verifies them on first launch,
then stores them under `~/Library/Application Support/Phonon/Models`.

## Runtime layout

- The signed app bundle contains the native Swift UI, the Rust engine, prompts,
  the startup audio fixture, and open inference runtimes.
- ASR uses the stock `mlx-community/parakeet-tdt-0.6b-v2` model. It is public,
  ungated, CC-BY-4.0, and approximately 2.47 GB.
- Correction uses `ALTICDEV/FLUID-1/model-Q4_K_M.gguf`. It is public, ungated,
  AGPL-3.0, and approximately 2.50 GB.
- The Fluid-1 GGUF runs through an open llama.cpp-based runtime. Production
  Phonon must not depend on `/Applications/FluidVoice.app` or its private MLX
  provider helper.
- Downloads are content-addressed and SHA-256 verified before activation.
- Model licenses and attribution are shown before download and retained in the
  installed model directory.

This requires no Phonon file-storage service. Hugging Face hosts the upstream
weights, GitHub Releases hosts versioned Phonon DMGs, and Vercel hosts the
static website.

## Release sequence

1. Bundle the Rust engine and all small runtime resources inside `Phonon.app`.
2. Replace the checkout-relative paths and installed FluidVoice helper lookup.
3. Add first-launch model download, verification, progress, and resumability.
4. Sign with a Developer ID Application certificate.
5. Notarize and staple the app, then build and notarize the DMG.
6. Publish the DMG and checksum to GitHub Releases.
7. Publish a Homebrew cask pointing at the immutable GitHub release asset.

## Apple distribution requirement

Public direct-download distribution requires an active Apple Developer Program
membership, a Developer ID Application certificate, and notarization. A free
Personal Team development identity is not sufficient.
