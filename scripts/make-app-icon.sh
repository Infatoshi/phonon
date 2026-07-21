#!/bin/bash
set -euo pipefail

output_path=${1:?output .icns path required}
work_dir=$(mktemp -d "${TMPDIR:-/tmp}/phonon-icon.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT
iconset="$work_dir/Phonon.iconset"
mkdir -p "$iconset"

swift "$(dirname "$0")/make-app-icon.swift" "$iconset"
iconutil -c icns "$iconset" -o "$output_path"
