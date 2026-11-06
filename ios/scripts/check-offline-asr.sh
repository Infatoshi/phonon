#!/bin/sh
# Fails if the app, keyboard or tests can reach a FluidAudio download. FluidAudio downloads in
# AsrModels.load/download/downloadAndLoad/loadFromCache/loadWithAutoRecovery/
# loadWithANEOptimization (through DownloadUtils), in the streaming ASR managers that call
# downloadAndLoad, and in its VAD and diarizer models.
# Phonon loads Parakeet with LocalAsrModels and uses only AsrManager, so:
#  1. no Phonon source calls those entry points or the download helpers;
#  2. FluidAudio's ASR code outside AsrModels.swift and Streaming/ (AsrManager and what it
#     calls) has no download or network call;
#  3. ModelDownloader.swift is the only Phonon file that touches the network.
set -eu
here="$(cd "$(dirname "$0")/.." && pwd)"
cd "$here"
fail=0
# Comment lines are skipped; LocalAsrModels.load is Phonon's own loader.
if grep -rnE '(^|[^A-Za-z])AsrModels\.(load|download|downloadAndLoad|loadFromCache|loadWithAutoRecovery|loadWithANEOptimization)\b|DownloadUtils|AssetDownloader|StreamingAsr|VadManager|Diarizer' \
  Phonon PhononKeyboard PhononCore/Sources PhononTests --include='*.swift' | grep -vE '^[^:]+:[0-9]+:[[:space:]]*//'; then
  echo "FAIL 1: a FluidAudio download entry point is called"; fail=1
else echo "ok 1: no FluidAudio download entry point is called"; fi
asr="Vendor/FluidAudio/Sources/FluidAudio/ASR"
if find "$asr" -name '*.swift' ! -name AsrModels.swift ! -path '*/Streaming/*' ! -path '*/ContextBiasing/*' ! -name CtcModels.swift -print0 \
  | xargs -0 grep -nE 'DownloadUtils|AssetDownloader|URLSession|downloadRepo|AsrModels\.(load|download)'; then
  echo "FAIL 2: FluidAudio ASR code outside AsrModels.swift and Streaming/ downloads"; fail=1
else echo "ok 2: FluidAudio ASR code outside AsrModels.swift and Streaming/ has no download or network call"; fi
net="$(grep -rlE 'URLSession|URLRequest|NWConnection|CFNetwork' Phonon PhononKeyboard PhononCore/Sources --include='*.swift' | sort)"
if [ "$net" != "PhononCore/Sources/PhononCore/ModelDownloader.swift" ]; then
  echo "FAIL 3: network code outside ModelDownloader.swift:"; echo "$net"; fail=1
else echo "ok 3: the only network code is $net"; fi
exit $fail
