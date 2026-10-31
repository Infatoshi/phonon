#!/bin/bash
# Build the native refiner engine `pe` from engine/ and llama.cpp at the pinned
# upstream commit plus third_party/llama.cpp/phonon-engine.patch, and the Metal
# Parakeet helper `pa` from engine/asr, for the app's minimum macOS. Output:
# target/engine/{pe,mk.metal,df.metal,pa,asr/pa.metal,asr/melfb.npy}, which
# package-bar.sh bundles. Needs git, cmake and ninja.
#
# On a Mac whose Xcode linker is older than its Command Line Tools SDK, run it
# with DEVELOPER_DIR=/Library/Developer/CommandLineTools.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)
llama_commit=39febd2adb947b70da35e95f9351b3bf58ad170a
llama_dir="$project_dir/target/llama.cpp-${llama_commit:0:7}"
out_dir="$project_dir/target/engine"
jobs=${PHONON_BUILD_JOBS:-4}
min_macos=$(/usr/libexec/PlistBuddy -c 'Print :LSMinimumSystemVersion' "$project_dir/bar/Resources/Info.plist")
# Without this the toolchain stamps the build machine's macOS as the minimum,
# and dyld then refuses the engine on every older Mac.
export MACOSX_DEPLOYMENT_TARGET=$min_macos

if [[ ! -f "$llama_dir/.phonon-patched" ]]; then
	rm -rf "$llama_dir"
	mkdir -p "$llama_dir"
	git -C "$llama_dir" init -q
	git -C "$llama_dir" fetch -q --depth 1 https://github.com/ggml-org/llama.cpp "$llama_commit"
	git -C "$llama_dir" checkout -q FETCH_HEAD
	git -C "$llama_dir" apply "$project_dir/third_party/llama.cpp/phonon-engine.patch"
	touch "$llama_dir/.phonon-patched"
fi

# Assertion messages carry source paths; keep the builder's home directory out
# of the shipped binary.
prefix_map="-ffile-prefix-map=$llama_dir=llama.cpp -ffile-prefix-map=$project_dir=phonon"
cmake -S "$llama_dir" -B "$llama_dir/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
	-DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DGGML_METAL=ON -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF \
	-DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF \
	-DCMAKE_C_FLAGS="$prefix_map" -DCMAKE_CXX_FLAGS="$prefix_map" -DCMAKE_OBJC_FLAGS="$prefix_map" \
	-DCMAKE_OSX_DEPLOYMENT_TARGET="$min_macos" >/dev/null
ninja -C "$llama_dir/build" -j"$jobs" llama llama-common >/dev/null

build="$llama_dir/build"
mkdir -p "$out_dir"
# shellcheck disable=SC2086 # prefix_map is two flags
clang++ -std=c++17 -O2 -mcpu=apple-m1 -fobjc-arc $prefix_map \
	-I"$llama_dir/include" -I"$llama_dir/ggml/include" -I"$llama_dir/common" -I"$llama_dir/src" \
	"$project_dir/engine/pe.mm" "$project_dir/engine/gguf.cpp" -o "$out_dir/pe.new" \
	"$build/common/libllama-common.a" "$build/common/libllama-common-base.a" "$build/src/libllama.a" \
	"$build/ggml/src/libggml.a" "$build/ggml/src/ggml-metal/libggml-metal.a" \
	"$build/ggml/src/ggml-blas/libggml-blas.a" "$build/ggml/src/libggml-cpu.a" "$build/ggml/src/libggml-base.a" \
	-framework Metal -framework Foundation -framework Accelerate -framework MetalKit
mv "$out_dir/pe.new" "$out_dir/pe"
cp "$project_dir/engine/mk.metal" "$project_dir/engine/df.metal" "$out_dir/"
printf '%s (minimum macOS %s)\n' "$out_dir/pe" \
	"$(otool -l "$out_dir/pe" | awk '/LC_BUILD_VERSION/ {found = 1} found && $1 == "minos" {print $2; exit}')"

# Metal Parakeet `pa` (engine/asr): the ASR sidecar's encoder + TDT decode on the GPU. Like pe it compiles its
# shader source (pa.metal) at startup; melfb.npy is get_logmel's slaney filterbank (128 x 257) for its GPU log-mel.
# It reads the pinned Parakeet snapshot's model.safetensors directly (no converted weights). Metal and Foundation
# only, no llama.cpp. Output: target/engine/pa and target/engine/asr/{pa.metal,melfb.npy}.
mkdir -p "$out_dir/asr"
# shellcheck disable=SC2086 # prefix_map is two flags
clang++ -std=c++17 -O2 -mcpu=apple-m1 -fobjc-arc $prefix_map "$project_dir/engine/asr/pa.mm" -o "$out_dir/pa.new" \
	-framework Metal -framework Foundation
mv "$out_dir/pa.new" "$out_dir/pa"
cp "$project_dir/engine/asr/pa.metal" "$project_dir/engine/asr/melfb.npy" "$out_dir/asr/"
printf '%s (minimum macOS %s)\n' "$out_dir/pa" \
	"$(otool -l "$out_dir/pa" | awk '/LC_BUILD_VERSION/ {found = 1} found && $1 == "minos" {print $2; exit}')"
