#!/bin/sh
# Native stills from a headless simulator at actual pixels, light and dark.
#   scripts/sim-stills.sh <sim-udid> <models-dir> <out-dir>
# The models are cloned (APFS, no copy) into the app's container exactly where a download
# would put them, so "Models ready" is the real load path, not a mock.
set -eu
udid="$1"; models="$2"; out="$3"
here="$(cd "$(dirname "$0")/.." && pwd)"
app="$(find "$here/../../work/dd/Build/Products/Debug-iphonesimulator" -maxdepth 1 -name Phonon.app | head -1)"
mkdir -p "$out"
xcrun simctl boot "$udid" 2>/dev/null || true
xcrun simctl bootstatus "$udid" -b >/dev/null
xcrun simctl install "$udid" "$app"
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
xcrun simctl status_bar "$udid" override --time 9:41 --batteryState charged --batteryLevel 100 --cellularBars 4 --wifiBars 3 >/dev/null 2>&1 || true
shot() { # name, wait seconds, launch args...
  name="$1"; wait="$2"; shift 2
  for mode in light dark; do
    xcrun simctl ui "$udid" appearance "$mode"
    xcrun simctl terminate "$udid" com.abhaymettu.phonon 2>/dev/null || true
    xcrun simctl launch "$udid" com.abhaymettu.phonon "$@" >/dev/null
    sleep "$wait"
    xcrun simctl io "$udid" screenshot --type=png "$out/sim-$name-$mode.png" >/dev/null 2>&1
  done
}
shot onboarding-1-welcome 3 -onboarded NO -PhononOnboardingStep 0
shot onboarding-3-models 3 -onboarded NO -PhononOnboardingStep 2
shot onboarding-4-keyboard 3 -onboarded NO -PhononOnboardingStep 3
shot home 25 -onboarded YES
shot settings 20 -onboarded YES -PhononScreen settings
shot listening 20 -onboarded YES -PhononScreen listening
shot diagnostics-selftest 140 -onboarded YES -PhononScreen diagnostics -PhononSelfTest YES
ls "$out"
