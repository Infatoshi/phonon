#!/bin/bash
# Build the refiner self-test (Tests/SelftestTests.swift, hosted in Phonon.app, model inside the test bundle) for
# Firebase Test Lab and zip it with its .xctestrun. Prints the zip path.
# usage: PHONON_MODEL=/path/refiner.gguf ./testlab.sh
# then:  gcloud firebase test ios run --configuration=phonon-devicelab --test=build/testlab/PhononSelftest.zip \
#          --device=model=iphone16pro,version=18.3 --xcode-version=26.2 --timeout=15m
set -euo pipefail
cd "$(dirname "$0")"
MODEL=${PHONON_MODEL:?set PHONON_MODEL to the refiner GGUF}
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
