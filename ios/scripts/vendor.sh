#!/bin/sh
# Fetches and builds the two native dependencies into ios/Vendor (not committed):
#  - llama.cpp at the Mac app's pinned commit, built as llama.xcframework (iOS + simulator)
#  - FluidAudio v0.9.1, built in Swift 5 language mode: its Swift 6 mode does not compile
#    with Xcode 26.6 (data-race errors in StreamingAsrManager, which Phonon does not use).
# Needs Xcode and cmake (brew install cmake). Takes a few minutes the first time.
set -eu
here="$(cd "$(dirname "$0")/.." && pwd)"
vendor="$here/Vendor"
mkdir -p "$vendor"
LLAMA_COMMIT=39febd2adb947b70da35e95f9351b3bf58ad170a
FLUID_TAG=v0.9.1
if [ ! -d "$vendor/llama.xcframework" ]; then
  src="$vendor/src/llama.cpp"
  if [ ! -d "$src" ]; then
    git init -q "$src"
    git -C "$src" remote add origin https://github.com/ggml-org/llama.cpp
  fi
  git -C "$src" fetch -q --depth 1 origin "$LLAMA_COMMIT"
  git -C "$src" checkout -q FETCH_HEAD
  (cd "$src" && ./build-xcframework.sh ios-sim ios-device)
  cp -R "$src/build-apple/llama.xcframework" "$vendor/llama.xcframework"
fi
if [ ! -d "$vendor/FluidAudio" ]; then
  git clone -q --depth 1 --branch "$FLUID_TAG" https://github.com/FluidInference/FluidAudio "$vendor/FluidAudio"
  printf '\npackage.swiftLanguageModes = [.v5]\n' >> "$vendor/FluidAudio/Package.swift"
fi
echo "Vendor ready: $(ls "$vendor")"
