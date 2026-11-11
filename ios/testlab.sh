#!/bin/bash
# Build the refiner self-test (Tests/SelftestTests.swift, hosted in Phonon.app, model inside the test bundle) for
# Firebase Test Lab and zip it with its .xctestrun. Prints the zip path.
# usage: PHONON_MODEL=/path/refiner.gguf ./testlab.sh
# then:  gcloud firebase test ios run --configuration=phonon-devicelab --test=build/testlab/PhononSelftest.zip \
#          --device=model=iphone16pro,version=18.3 --xcode-version=26.2 --timeout=15m
set -euo pipefail
cd "$(dirname "$0")"
MODEL=${PHONON_MODEL:?set PHONON_MODEL to the refiner GGUF}
# The refiner needs the patched llama build (ggml_cpu_phonon_set from the threadpool patch, llama_phonon_sl_sample
# from the shortlist patch), not the stock release; a here-string rather than a pipe, so grep -q closing early cannot
# fail the check under pipefail.
SYMS=$(nm -gU Frameworks/llama.xcframework/ios-arm64/llama.framework/llama 2>/dev/null || true)
if ! grep -q ggml_cpu_phonon_set <<< "$SYMS" || ! grep -q llama_phonon_sl_sample <<< "$SYMS"; then
  echo "Frameworks/llama.xcframework is missing or unpatched (no ggml_cpu_phonon_set or llama_phonon_sl_sample); building it with ./build_llama.sh"
  ./build_llama.sh
fi
mkdir -p build/selftest-model
cp -c "$MODEL" build/selftest-model/refiner.gguf 2>/dev/null || cp "$MODEL" build/selftest-model/refiner.gguf
xcodegen generate --quiet
OUT=build/testlab
rm -rf "$OUT"
xcodebuild -project PhononiOS.xcodeproj -scheme PhononSelftest -configuration Debug -sdk iphoneos \
  -destination generic/platform=iOS -derivedDataPath build -allowProvisioningUpdates build-for-testing \
  > build-testlab.log 2>&1 || { grep -E "error:|BUILD FAILED" build-testlab.log | head -20; exit 1; }
mkdir -p "$OUT"
cp -R build/Build/Products/Debug-iphoneos "$OUT/"
cp build/Build/Products/*.xctestrun "$OUT/"
(cd "$OUT" && zip -qr -1 PhononSelftest.zip Debug-iphoneos ./*.xctestrun)
ls -la "$OUT/PhononSelftest.zip"
