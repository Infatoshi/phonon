#!/usr/bin/env bash
# Fetches the Android build inputs into /data/android/deps on the Linux build host.
# llama.cpp is pinned to the commit every Android measurement since 2026-11-05 ran on (b12132 + 28 commits), and the
# build uses a patched copy, llama.cpp-phonon (gradle.properties phonon.llamaDir), with the iPhone's shortlist-head
# patch applied: ios/llama-b12132-shortlist.patch, inert unless the model file carries phonon.sl.k, phonon.sl.order
# and output.slx (ios/shortlist_gguf.py). Pass the patch path as $1 when the script is not run from a repo checkout.
set -euo pipefail
PATCH=$(realpath "${1:-$(dirname "$0")/../../ios/llama-b12132-shortlist.patch}")
LLAMA_COMMIT=6023692b1e13a6d4ee6b93f989c4713e474e6165
D=/data/android/deps; mkdir -p $D; cd $D
[ -f sherpa-onnx-v2.0.1-android.tar.bz2 ] || curl -fsSLO https://github.com/k2-fsa/sherpa-onnx/releases/download/v2.0.1/sherpa-onnx-v2.0.1-android.tar.bz2
mkdir -p sherpa-android && tar xjf sherpa-onnx-v2.0.1-android.tar.bz2 -C sherpa-android
[ -d sherpa-src ] || git clone -q --depth 1 --branch v2.0.1 https://github.com/k2-fsa/sherpa-onnx.git sherpa-src
[ -f parakeet.tar.bz2 ] || curl -fsSL -o parakeet.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2
tar xjf parakeet.tar.bz2
if [ ! -d llama.cpp ]; then
  git init -q llama.cpp
  git -C llama.cpp fetch -q --depth 1 https://github.com/ggml-org/llama.cpp.git $LLAMA_COMMIT
  git -C llama.cpp checkout -q FETCH_HEAD
fi
[ "$(git -C llama.cpp rev-parse HEAD)" = $LLAMA_COMMIT ] || { echo "llama.cpp is not at $LLAMA_COMMIT"; exit 1; }
[ -z "$(git -C llama.cpp status --porcelain)" ] || { echo "llama.cpp has local changes"; exit 1; }
rm -rf llama.cpp-phonon.new
mkdir llama.cpp-phonon.new
git -C llama.cpp archive HEAD | tar x -C llama.cpp-phonon.new
(cd llama.cpp-phonon.new && git apply "$PATCH")
# keep the old tree (and the CMake build's timestamps) when nothing changed
if [ -d llama.cpp-phonon ] && diff -rq llama.cpp-phonon llama.cpp-phonon.new > /dev/null; then
  rm -rf llama.cpp-phonon.new
else
  rm -rf llama.cpp-phonon && mv llama.cpp-phonon.new llama.cpp-phonon
fi
echo DEPS_DONE
