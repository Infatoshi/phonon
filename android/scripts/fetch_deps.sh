#!/usr/bin/env bash
set -euo pipefail
D=/data/android/deps; mkdir -p $D; cd $D
[ -f sherpa-onnx-v2.0.1-android.tar.bz2 ] || curl -fsSLO https://github.com/k2-fsa/sherpa-onnx/releases/download/v2.0.1/sherpa-onnx-v2.0.1-android.tar.bz2
mkdir -p sherpa-android && tar xjf sherpa-onnx-v2.0.1-android.tar.bz2 -C sherpa-android
[ -d sherpa-src ] || git clone -q --depth 1 --branch v2.0.1 https://github.com/k2-fsa/sherpa-onnx.git sherpa-src
[ -f parakeet.tar.bz2 ] || curl -fsSL -o parakeet.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2
tar xjf parakeet.tar.bz2
[ -d llama.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp.git llama.cpp
echo DEPS_DONE
