#!/bin/bash
# Build ios/Frameworks/llama.xcframework: llama.cpp b12132 plus ios/llama-b12132-phonon.patch, device and simulator
# slices only (about 90 s on an M4 Max). The patch:
#   - device slice built with -DGGML_CPU_ARM_ARCH=armv8.4-a+dotprod+fp16 (every iOS 26 device is A13 or newer;
#     the release zip has no dotprod, so Q4_K ran the scalar fallback)
#   - threadpool workers get the creating thread's QoS class on Apple (stock: QOS_CLASS_DEFAULT, which kept them
#     on E cores about 2/3 of the time while the caller was User Initiated)
#   - repack-path decode matmuls split into about 64-row chunks (stock: 4 per thread)
#   - ggml_cpu_phonon_set(inherit_qos, repack_rows) to switch both back to stock at run time for A/B runs
# usage: ./build_llama.sh [work dir, default $TMPDIR/phonon-llama-b12132]
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
WORK=${1:-${TMPDIR:-/tmp}/phonon-llama-b12132}
if [ ! -d "$WORK/.git" ]; then
  git clone --quiet --depth 1 --branch b12132 https://github.com/ggml-org/llama.cpp "$WORK"
fi
cd "$WORK"
git checkout --quiet -- . && git clean -fdq -e 'build-*'
git apply "$HERE/llama-b12132-phonon.patch"
# experiment branch ios-head-timing only: LFM3 accepts an untied output.weight with fewer rows than n_vocab and pads
# the logits back to n_vocab (timing of a shortlist head; outputs differ from the full head)
git apply "$HERE/llama-b12132-headtiming.patch"
./build-xcframework.sh ios-device ios-sim > "$HERE/build-llama.log" 2>&1 || { tail -30 "$HERE/build-llama.log"; exit 1; }
rm -rf "$HERE/Frameworks/llama.xcframework"
mkdir -p "$HERE/Frameworks"
cp -R build-apple/llama.xcframework "$HERE/Frameworks/"
echo "sdot in device slice: $(otool -tv "$HERE/Frameworks/llama.xcframework/ios-arm64/llama.framework/llama" | grep -c sdot)"
echo "wrote $HERE/Frameworks/llama.xcframework"
