# Licenses

Phonon for iPhone is part of Phonon (https://github.com/Infatoshi/phonon) and is distributed
under the GNU General Public License v3.0, the same license as the rest of this repository
(`LICENSE` at the repository root). Each component below keeps its own license; this file does
not relicense any of them.

## Phonon source (this app and keyboard)
- License: GPL-3.0. Upstream: https://github.com/Infatoshi/phonon. The iOS port lives in `ios/`.
- The refiner prompt, token budget, word-count guard and term restore are ported from
  `sidecar/polish_server.py`.

## Parakeet TDT 0.6B v2 (speech recognition, downloaded at first run)
- NVIDIA, converted to Core ML by FluidInference:
  https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v2-coreml at revision
  ee09c569f73759e6d44c9bd16766f477b2b36d39. Original: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2
- License: Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/. Used unmodified.

## Phonon refiner 1.7B (cleanup, downloaded at first run)
- https://huggingface.co/Infatoshi/phonon-refiner-1.7b at revision
  2b0ab74bee7ab7cf364695bbe44f64ce4c47439e, file phonon-refiner-1.7b-Q4_K_S-e4.gguf.
- A modified version of LFM3-1.7B by Liquid AI, Inc. (https://huggingface.co/LiquidAI/LFM3-1.7B),
  under the LFM Open License v1.0. It permits redistribution of fine-tuned weights; commercial
  use is licensed only to organizations under 10 million US dollars in annual revenue.
- Notice from the model repository:

  > Phonon refiner 1.7B
  > 
  > This work is a modified version of LFM3-1.7B by Liquid AI, Inc.
  > (https://huggingface.co/LiquidAI/LFM3-1.7B), distributed under the LFM Open
  > License v1.0 (see LICENSE).
  > 
  > Changes: the weights were fine-tuned for dictation transcript correction (a
  > LoRA adapter merged into the base weights) and re-quantized. The GGUF target
  > and the MLX weights in this repository are both quantized from the fine-tuned
  > model. The drafter (DFlash2) is a separate small network trained on the
  > outputs and hidden states of the previous release of this model (Phonon
  > 0.2.3, commit 74ea4b6 of this repository).
  > 
  > Modified or new files (by the Phonon project, https://github.com/Infatoshi/phonon):
  > every .gguf and .safetensors file, config.json, model.safetensors.index.json,
  > generation_config.json (repetition_penalty set to 1.0), phonon_refiner.json
  > and phonon_engine.json. tokenizer.json and tokenizer_config.json were
  > re-saved by newer tooling: the same vocabulary, merges and special tokens,
  > with extra metadata keys. chat_template.jinja is the base model's file,
  > unchanged. The base model's custom-code files (configuration_lfm3.py,
  > modeling_lfm3.py, lfm3_conv.py) and the auto_map entry in config.json that
  > names them are not included: these weights are MLX and GGUF, which do not use
  > them.

## FluidAudio (Swift package, linked)
- https://github.com/FluidInference/FluidAudio, tag v0.9.1. License: Apache License 2.0.

## llama.cpp and ggml (linked as llama.xcframework)
- https://github.com/ggml-org/llama.cpp at commit 39febd2adb947b70da35e95f9351b3bf58ad170a,
  the same commit the Mac app pins (`third_party/llama.cpp`). Built unmodified for iOS with its
  `build-xcframework.sh`. License: MIT, Copyright (c) 2023-2026 The ggml authors.
  llama.cpp vendors nlohmann/json (MIT).
