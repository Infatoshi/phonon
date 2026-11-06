#!/bin/sh
# Fetches and builds the two native dependencies into ios/Vendor (not committed):
#  - llama.cpp at the Mac app's pinned commit, built as llama.xcframework (iOS + simulator).
#    The device slice targets A15 and newer (-mcpu=apple-a15: dot product, fp16, int8 matmul).
#    clang's iOS default is the generic ARMv8.0 baseline, which leaves the refiner on the
#    slowest kernels; the app checks the chip at runtime before loading the refiner.
#  - FluidAudio v0.9.1 (commit 915a392), built in Swift 5 language mode: its Swift 6 mode does not compile
#    with Xcode 26.6 (data-race errors in StreamingAsrManager, which Phonon does not use).
# Both are pinned by full commit SHA and checked on every run. Neither has remote Swift package
# dependencies (FluidAudio's Package.swift lists none), so these two SHAs are the whole lock.
# Needs Xcode and cmake (brew install cmake). Takes a few minutes the first time.
set -eu
here="$(cd "$(dirname "$0")/.." && pwd)"
vendor="$here/Vendor"
mkdir -p "$vendor"
LLAMA_COMMIT=39febd2adb947b70da35e95f9351b3bf58ad170a
FLUID_COMMIT=915a392a3b214205fcf4860cc3608bde35891d03 # v0.9.1
if [ ! -f "$vendor/llama.xcframework/.phonon-a15" ]; then
  rm -rf "$vendor/llama.xcframework"
  src="$vendor/src/llama.cpp"
  if [ ! -d "$src" ]; then
    git init -q "$src"
    git -C "$src" remote add origin https://github.com/ggml-org/llama.cpp
  fi
  git -C "$src" fetch -q --depth 1 origin "$LLAMA_COMMIT"
  git -C "$src" checkout -q FETCH_HEAD
  python3 - "$src/build-xcframework.sh" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
def patch(fn, extra):
    global s
    start = s.index(fn + "() {")
    end = s.index("\n}\n", start)
    body = s[start:end]
    body = body.replace('-DCMAKE_C_FLAGS="${COMMON_C_FLAGS}"', '-DCMAKE_C_FLAGS="${COMMON_C_FLAGS} ' + extra + '"')
    body = body.replace('-DCMAKE_CXX_FLAGS="${COMMON_CXX_FLAGS}"', '-DCMAKE_CXX_FLAGS="${COMMON_CXX_FLAGS} ' + extra + '"')
    s = s[:start] + body + s[end:]
patch("build_ios_device", "-mcpu=apple-a15")
patch("build_ios_sim", "-Xarch_arm64 -mcpu=apple-m1")
open(p, "w").write(s)
PY
  (cd "$src" && ./build-xcframework.sh ios-sim ios-device)
  cp -R "$src/build-apple/llama.xcframework" "$vendor/llama.xcframework"
  touch "$vendor/llama.xcframework/.phonon-a15"
fi
if [ ! -d "$vendor/FluidAudio" ]; then
  git init -q "$vendor/FluidAudio"
  git -C "$vendor/FluidAudio" remote add origin https://github.com/FluidInference/FluidAudio
  git -C "$vendor/FluidAudio" fetch -q --depth 1 origin "$FLUID_COMMIT"
  git -C "$vendor/FluidAudio" checkout -q FETCH_HEAD
  printf '\npackage.swiftLanguageModes = [.v5]\n' >> "$vendor/FluidAudio/Package.swift"
fi
pinned() { # dir, sha: the checkout must be exactly the pinned commit
  [ ! -d "$1" ] && return 0
  at="$(git -C "$1" rev-parse HEAD)"
  [ "$at" = "$2" ] || { echo "$1 is at $at, pinned $2. Delete it and rerun."; exit 1; }
}
pinned "$vendor/FluidAudio" "$FLUID_COMMIT"
pinned "$vendor/src/llama.cpp" "$LLAMA_COMMIT"
echo "Vendor ready: $(ls "$vendor")"
