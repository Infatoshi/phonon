#!/bin/sh
# The keyboard flow end to end in a headless simulator (PhononUITests): keyboard added with
# Full Access in Settings, picked with the globe key in Messages, mic, Phonon opened by the
# keyboard, back, stop, text inserted. A fixture recording stands in for the microphone.
# Stills and the step log go to <out-dir>, a screen recording to <clip>.
#   scripts/sim-keyboard-flow.sh <sim-udid> <models-dir> <out-dir> <clip.mp4> [fixture]
# PHONON_RACE=N plants N unfinished dictations (Fixtures/02-restart) in Phonon's journal and
# keeps Phonon running, so the keyboard's start arrives while it is finishing them.
set -eu
udid="$1"; fixture="${5:-01-plain}"
mkdir -p "$3" "$(dirname "$4")"
models="$(cd "$2" && pwd)"; out="$(cd "$3" && pwd)"; clip="$(cd "$(dirname "$4")" && pwd)/$(basename "$4")"
here="$(cd "$(dirname "$0")/.." && pwd)"
dd="$here/../../work/dd-ui"
cd "$here"
xcodegen generate --quiet
xcrun simctl boot "$udid" 2>/dev/null || true
xcrun simctl bootstatus "$udid" -b >/dev/null
xcrun simctl ui "$udid" appearance light
xcodebuild -project Phonon.xcodeproj -scheme PhononKeyboardFlow -destination "id=$udid" \
  -derivedDataPath "$dd" build-for-testing > "$out/keyboard-flow-build.log" 2>&1
xcrun simctl install "$udid" "$dd/Build/Products/Debug-iphonesimulator/Phonon.app"
data="$(xcrun simctl get_app_container "$udid" com.abhaymettu.phonon data)"
dest="$data/Library/Application Support/Models"
mkdir -p "$dest"
for d in parakeet-tdt-0.6b-v2-coreml phonon-refiner-1.7b; do
  [ -d "$dest/$d" ] || cp -cR "$models/$d" "$dest/$d"
done
python3 - "$here/PhononCore/Sources/PhononCore/ModelManifest+Pinned.swift" "$dest/verified.json" <<'PY'
import json, re, sys
s = open(sys.argv[1]).read()
v = {f"{m[1]}/{m[0]}": m[2] for m in re.findall(r'path: "([^"]+)", folder: "([^"]+)", size: \d+, sha256: "([0-9a-f]+)"', s)}
json.dump(v, open(sys.argv[2], "w"))
PY
# The keyboard launches Phonon by URL without launch arguments; this file turns on the
# fixture microphone in that launch (debug builds only). Removed after the run.
mkdir -p "$data/Documents"
printf '%s' "$fixture" > "$data/Documents/phonon-fixture-audio"
if [ -n "${PHONON_RACE:-}" ]; then
  python3 - "$data/Library/Application Support/Recordings" "$here/Fixtures/02-restart.wav" "$PHONON_RACE" <<'PY'
import array, json, os, sys, time, uuid, wave
d, wav, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(d, exist_ok=True)
w = wave.open(wav); pcm = array.array("h", w.readframes(w.getnframes()))
f32 = array.array("f", (x / 32768.0 for x in pcm)).tobytes()
path = os.path.join(d, "journal.json")
entries = json.load(open(path)) if os.path.exists(path) else []
for i in range(n):
    i_d = str(uuid.uuid4()).upper()
    open(os.path.join(d, i_d + ".pcm"), "wb").write(f32)
    entries.append({"id": i_d, "startedAt": time.time() - 978307200 - 60, "stage": "recording", "mode": "casual"})
json.dump(entries, open(path, "w"))
PY
fi
xcrun simctl io "$udid" recordVideo --codec=h264 --force "$clip" > "$out/keyboard-flow-record.log" 2>&1 &
rec=$!
status=0
TEST_RUNNER_PHONON_SHOTS_DIR="$out" TEST_RUNNER_PHONON_FIXTURE="$fixture" \
  TEST_RUNNER_PHONON_KEEP_RUNNING="$([ -n "${PHONON_RACE:-}" ] && echo 1 || echo 0)" \
  xcodebuild -project Phonon.xcodeproj -scheme PhononKeyboardFlow -destination "id=$udid" \
  -derivedDataPath "$dd" -resultBundlePath "$out/keyboard-flow.xcresult" test-without-building \
  > "$out/keyboard-flow-test.log" 2>&1 || status=$?
rm -f "$data/Documents/phonon-fixture-audio"
kill -INT "$rec"
wait "$rec" 2>/dev/null || true
"$here/scripts/sim-prune-cache.sh" "$udid"
grep -E "Test Case .*(passed|failed)|error:" "$out/keyboard-flow-test.log" | head -20
exit $status
