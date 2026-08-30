#!/bin/sh
# Run the whole miner. Output dir: $PHONON_MINER_OUT (default ./out).
# Stages needing a model run under uv with the pinned package, offline.
set -eu
cd "$(dirname "$0")"
export PHONON_MINER_OUT="${PHONON_MINER_OUT:-$PWD/out}"
export HF_HUB_OFFLINE=1
UNTIL="${UNTIL:-}"
TOP="${ORACLE_TOP:-6000}"
PY="uv run --offline --python 3.12"

$PY python -m profile_miner extract ${UNTIL:+--until "$UNTIL"}
$PY python -m profile_miner seed
$PY python -m profile_miner candidates
$PY --with parakeet-mlx==0.5.2 python -m profile_miner oracle --top "$TOP"
$PY python -m profile_miner rank
if [ "${GEMMA:-1}" = 1 ]; then
  $PY --with mlx-lm==0.31.3 python -m profile_miner gemma --minutes "${GEMMA_MINUTES:-20}" || echo "gemma pass failed (optional)"
fi
echo "done: $PHONON_MINER_OUT/mined/candidates.json"
