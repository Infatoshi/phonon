#!/bin/bash
# Run the refiner self-test on the connected iPhone (the app's --selftest) and print Documents/selftest.jsonl.
# usage: ./selftest.sh [threads] [flags...]   flags: --pool 0 (per-token threads), --stock 1 (stock ggml QoS and chunks)
set -uo pipefail
DEVICE=${PHONON_IOS_DEVICE:?set PHONON_IOS_DEVICE to the iPhone UDID}
OUT=${TMPDIR:-/tmp}/phonon_selftest.jsonl
STAMP=$(date +%s)
xcrun devicectl device process launch --terminate-existing --device "$DEVICE" sh.phonon.ios --selftest \
  --threads "${1:-5}" --stamp "$STAMP" "${@:2}" > /dev/null
for _ in $(seq 1 30); do
  sleep 5
  rm -f "$OUT"
  xcrun devicectl device copy from --device "$DEVICE" --source Documents/selftest.jsonl --destination "$OUT" \
    --domain-type appDataContainer --domain-identifier sh.phonon.ios > /dev/null 2>&1 || continue
  [ "$(wc -l < "$OUT")" -ge 9 ] && grep -q "\"stamp\":\"$STAMP\"" "$OUT" && { cat "$OUT"; exit 0; }
done
echo "no fresh selftest.jsonl after 150 s"
exit 1
