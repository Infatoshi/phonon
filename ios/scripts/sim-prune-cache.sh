#!/bin/sh
# Keeps a simulator's Phonon Core ML cache small after test runs: removes partial compiles
# ("*.tmp.*" bundles, which Core ML never reuses) and empties the cache if it is still over 3 GB.
#   scripts/sim-prune-cache.sh <sim-udid>
set -eu
data="$(xcrun simctl get_app_container "$1" com.abhaymettu.phonon data 2>/dev/null)" || exit 0
cache="$data/Library/Caches/com.abhaymettu.phonon/com.apple.e5rt.e5bundlecache"
[ -d "$cache" ] || exit 0
find "$cache" -path '*.tmp.*' -delete
if [ "$(du -sk "$cache" | cut -f1)" -gt 3000000 ]; then find "$cache" -mindepth 1 -delete; fi
echo "Core ML cache after prune: $(du -sh "$cache" | cut -f1), $(find "$cache" -maxdepth 3 -name '*.bundle' | wc -l | tr -d ' ') bundles"
