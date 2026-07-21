# Third-party models and runtimes

Phonon does not commit model weights to this repository. Release builds download
the selected artifacts directly from their upstream hosts and verify SHA-256
before use.

## Parakeet TDT 0.6B v2

- Source: <https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2>
- Upstream model: NVIDIA Parakeet TDT 0.6B v2
- License: CC-BY-4.0
- Primary weight: `model.safetensors`
- Size: 2,471,559,904 bytes
- SHA-256: `b958c37a6baa6874a279108755c8f2818e27bf647d72d54800a234a421341dfe`

## Fluid-1

- Source: <https://huggingface.co/ALTICDEV/FLUID-1>
- License: AGPL-3.0
- Primary weight: `model-Q4_K_M.gguf`
- Size: 2,497,280,160 bytes
- SHA-256: `619dca002c4a2bf683311a6084c75ba31ab610b9b352720f9ead9e9bceac7590`

FluidVoice itself is GPL-3.0, but its release-only `fluid-intelligence-mlx`
provider is not present in the public FluidVoice source. Phonon release builds
must use the public Fluid-1 GGUF through an independently bundled open runtime;
they must not depend on or redistribute FluidVoice's installed private helper.

## llama.cpp

- Source: <https://github.com/ggml-org/llama.cpp>
- License: MIT

The final binary attribution bundle must include the exact notices for every
runtime revision shipped in that release.
