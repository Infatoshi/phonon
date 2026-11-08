#!/usr/bin/env bash
# Milestone 3: the refiner inside the IME process on the iOS self-test transcripts, each mode twice
# (rep 0 prefills the fixed head after a mode switch, rep 1 reuses it), triggered by adb broadcasts.
A=/data/android/sdk/platform-tools/adb
$A logcat -c
send() { $A shell "am broadcast -p ai.phonon.keyboard -a ai.phonon.keyboard.REFINE --es mode $1 --es text '$2'" </dev/null >/dev/null; }
while IFS='|' read -r mode raw; do send "$mode" "$raw"; send "$mode" "$raw"; done <<'CASES'
minimal|um so the first thing we do is call could a malloc to to allocate memory on the GPU
prompt|okay so I want you to add a retry to the fetch function like three retries no actually make it five with exponential backoff and um also log every failure to the console
casual|yeah I think we should uh grab dinner around seven if that works for you
formal|hi sarah just wanted to let you know the the deploy is gonna slip to thursday because the tests are failing on the arm runners
CASES
for i in $(seq 120); do [ "$($A logcat -d -s Phonon | grep -c 'REFINE total_ms')" -ge 8 ] && break; sleep 2; done
$A logcat -d -s Phonon | grep "REFINE total_ms" | sed 's/.*REFINE //'
