#!/usr/bin/env bash
# Build, reinstall (keeps the pushed models), restart the IME and wait for the models to load.
set -e
bash /data/android/work/src/scripts/build.sh > /data/android/work/build.log 2>&1 || { tail -40 /data/android/work/build.log; exit 1; }
/data/android/sdk/platform-tools/adb install -r -g /data/android/work/src/app/build/outputs/apk/debug/app-debug.apk
bash /data/android/work/src/scripts/restart_ime.sh
