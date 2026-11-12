#!/usr/bin/env bash
# Runs one instrumentation class on Firebase Test Lab phones with the lab inputs pushed under /sdcard/phonon/.
# usage: testlab.sh STAMP RESULTS_DIR CLASS ENVVARS device...
#   STAMP: folder under $B/phonon-apk/ holding app-debug.apk and app-debug-androidTest.apk
#   CLASS: e.g. RefinerTuneBenchTest or ImeE2ETest; ENVVARS: "^:^arms=auto,auto/k4" style, "" for none
#   device: model=houji,version=35 (one --device each)
# Environment: PHONON_TESTLAB_BUCKET (gs://... results bucket), PHONON_TESTLAB_CONFIG (gcloud configuration),
#   PHONON_TESTLAB_SL=1 to also push refiner-sl.gguf (the shortlist build, for "/sl" bench arms).
# The refiner comes from $B/phonon-inputs-shipped/refiner.gguf, a copy of the file Models.kt pins. Do not use
# $B/phonon-inputs/refiner.gguf: it is an older model of the same size (found 2026-11-12). The tests check the
# staged file's sha256 against Models.REFINER and fail unless it matches (ModelPin.kt, model_any=1 to override).
set -euo pipefail
B=${PHONON_TESTLAB_BUCKET:?set PHONON_TESTLAB_BUCKET}
CFG=${PHONON_TESTLAB_CONFIG:-default}
STAMP=$1; RES=$2; CLASS=$3; ENVV=$4; shift 4
DEVS=(); for d in "$@"; do DEVS+=(--device "$d"); done
I=$B/phonon-inputs
F="/sdcard/phonon/refiner.gguf=$B/phonon-inputs-shipped/refiner.gguf"
[ "${PHONON_TESTLAB_SL:-0}" = 1 ] && F="$F,/sdcard/phonon/refiner-sl.gguf=$B/phonon-inputs-shipped/refiner-sl.gguf"
if [ "$CLASS" = ImeE2ETest ]; then
  for f in encoder.int8.onnx encoder.int8.onnx.data; do F="$F,/sdcard/phonon/parakeet/$f=$I/parakeet-ext/$f"; done
  for f in decoder.int8.onnx joiner.int8.onnx tokens.txt; do F="$F,/sdcard/phonon/parakeet/$f=$I/parakeet/$f"; done
  for f in s1.wav steps.wav long.wav; do F="$F,/sdcard/phonon/$f=$I/wav/$f"; done
fi
echo "refiner input: $(gcloud --configuration="$CFG" storage objects describe "$B/phonon-inputs-shipped/refiner.gguf" \
  --format='value(size,crc32c_hash)')" >&2
EV=(); [ -n "$ENVV" ] && EV=(--environment-variables "$ENVV")
gcloud --configuration="$CFG" firebase test android run --type instrumentation \
  --app "$B/phonon-apk/$STAMP/app-debug.apk" --test "$B/phonon-apk/$STAMP/app-debug-androidTest.apk" \
  --test-targets "class ai.phonon.keyboard.$CLASS" "${EV[@]}" --other-files "$F" \
  --directories-to-pull /sdcard/Android/data/ai.phonon.keyboard/files/shots \
  --timeout 25m --num-flaky-test-attempts 0 --no-record-video --results-dir "$RES" "${DEVS[@]}" --async 2>&1 \
  | grep -oE 'matrix-[a-z0-9]+|ERROR.*' | head -2
