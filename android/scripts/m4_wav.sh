#!/usr/bin/env bash
# Milestone 4 (injected audio): a WAV stands in for one hold of the mic key. The broadcast's arrival is the key-up;
# the IME transcribes (sherpa-onnx Parakeet), refines, commits into the focused field, and logs key-up-to-commit.
A=/data/android/sdk/platform-tools/adb
WAV=${1:-d1.wav}; N=${2:-3}
F=/sdcard/Android/data/ai.phonon.keyboard/files
# no `am start -S`: the IME lives in this package's process, and a force-stop drops back to the stock keyboard
$A shell am start -W -n ai.phonon.keyboard/.TestActivity </dev/null >/dev/null
sleep 2; $A shell input tap 540 330 </dev/null; sleep 1
$A logcat -c
for i in $(seq $N); do
  $A shell "am broadcast -p ai.phonon.keyboard -a ai.phonon.keyboard.WAV --es path $F/$WAV" </dev/null >/dev/null
  for j in $(seq 60); do [ "$($A logcat -d -s Phonon | grep -c E2E)" -ge $i ] && break; sleep 1; done
done
$A logcat -d -s Phonon | grep -E "PIPELINE|E2E" | sed 's/.*Phonon  : //' | cut -c1-900
$A shell uiautomator dump /sdcard/ui.xml </dev/null >/dev/null
$A shell cat /sdcard/ui.xml | tr '>' '\n' | grep -o 'text="[^"]*" resource-id="ai.phonon.keyboard:id/field"'
