#!/usr/bin/env bash
# Milestone 4 (real mic path): holds the IME's mic key with a long `input swipe`, routes the emulator's host
# audio capture to a private PipeWire null sink, and plays the WAV into that sink in real time. The commit is timed
# from the real ACTION_UP. The null sink is unloaded at the end; the host's default sink/source are not changed.
A=/data/android/sdk/platform-tools/adb
WAV=${1:-/data/android/deps/d1.wav}
MIC=$($A logcat -d -s Phonon | grep KEYPOS | tail -1 | tr ' ' '\n' | grep '^mic=' | cut -d= -f2)
MIC=${MIC:-540,1741}
MOD=$(pactl load-module module-null-sink sink_name=phonon_feed rate=16000 channels=1 sink_properties=device.description=phonon_feed)
echo "null sink module=$MOD default_source=$(pactl get-default-source)"
$A logcat -c
DUR=$(python3 -c "import wave,sys; w=wave.open(sys.argv[1]); print(int(w.getnframes()/w.getframerate()*1000))" "$WAV")
HOLD=$((DUR + 2500))
$A shell input swipe ${MIC/,/ } ${MIC/,/ } $HOLD </dev/null &
SW=$!
for i in $(seq 20); do SO=$(pactl list source-outputs short | awk '{print $1}' | head -1); [ -n "$SO" ] && break; sleep 0.2; done
echo "emulator source-output=$SO"
[ -n "$SO" ] && pactl move-source-output $SO phonon_feed.monitor
sleep 0.5
paplay --device=phonon_feed "$WAV"
wait $SW
for j in $(seq 60); do $A logcat -d -s Phonon | grep -q E2E && break; sleep 1; done
pactl unload-module $MOD
$A logcat -d -s Phonon | grep -E "MIC_UP|PIPELINE|E2E" | sed 's/.*Phonon  : //' | cut -c1-500
