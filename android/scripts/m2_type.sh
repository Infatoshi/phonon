#!/usr/bin/env bash
# Milestone 2: open the test field, press real keys with `input tap` at the IME's logged key centres, read the field back.
A=/data/android/sdk/platform-tools/adb
WORD=${1:-hello world}
$A logcat -c
$A shell am start -W -n ai.phonon.keyboard/.TestActivity >/dev/null
sleep 3
$A shell input tap 540 330   # the EditText, in case it did not take focus
sleep 2
POS=$($A logcat -d -s Phonon:I | grep KEYPOS | tail -1)
echo "$POS" | cut -c1-400
for ((i=0; i<${#WORD}; i++)); do
  c=${WORD:$i:1}; [ "$c" = " " ] && c=space
  xy=$(echo "$POS" | tr ' ' '\n' | grep "^$c=" | head -1 | cut -d= -f2)
  $A shell input tap ${xy/,/ }
done
sleep 1
$A shell uiautomator dump /sdcard/ui.xml >/dev/null
$A shell cat /sdcard/ui.xml | tr '>' '\n' | grep -o 'text="[^"]*" resource-id="ai.phonon.keyboard:id/field"'
$A logcat -d -s Phonon:I | grep -E "IME_CREATE|KEY " | tail -12
