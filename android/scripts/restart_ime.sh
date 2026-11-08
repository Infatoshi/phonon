#!/usr/bin/env bash
# Kills the Phonon process, reselects it as the IME, focuses the test field and waits for both models to load.
A=/data/android/sdk/platform-tools/adb
$A shell am force-stop ai.phonon.keyboard
$A logcat -c
$A shell ime enable ai.phonon.keyboard/.PhononIME >/dev/null
$A shell ime set ai.phonon.keyboard/.PhononIME
$A shell am start -W -n ai.phonon.keyboard/.TestActivity >/dev/null
sleep 2; $A shell input tap 540 330
for i in $(seq 90); do $A logcat -d -s Phonon | grep -qE "WARMUP|no /|ok=false" && break; sleep 2; done
$A logcat -d | grep -E " Phonon|PhononRefiner|FATAL|AndroidRuntime: (FATAL|java)" | grep -v KEYPOS | cut -c1-1200
