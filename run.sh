#!/usr/bin/env bash
# Deprecated wrapper — prefer: cargo install --path . --force --root ~/.local && phonon
set -euo pipefail
cd "$(dirname "$0")"
exec cargo run --release --quiet -- "$@"
