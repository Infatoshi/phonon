#!/bin/bash
# Build the Windows installer (target/win-installer/Phonon-Setup-<version>.exe) from a phonon-win.exe.
# usage: scripts/package-win-installer.sh path/to/phonon-win.exe
# Needs makensis (NSIS 3; `apt install nsis`, or set MAKENSIS and NSISDIR to an unpacked copy). Downloads Microsoft's
# Visual C++ redistributable (x64) once into target/win-installer/ and bundles it.
set -euo pipefail
cd "$(dirname "$0")/.."
EXE=$(realpath "${1:?path to phonon-win.exe}")
VERSION=$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)
OUT=target/win-installer; mkdir -p "$OUT"
VC="$OUT/vc_redist.x64.exe"
[ -s "$VC" ] || curl -fsSL -o "$VC" https://aka.ms/vs/17/release/vc_redist.x64.exe
"${MAKENSIS:-makensis}" -V2 -DVERSION="$VERSION" -DEXE="$EXE" -DVCREDIST="$(realpath "$VC")" \
  -DOUT="$(realpath "$OUT")/Phonon-Setup-$VERSION.exe" packaging/windows/phonon.nsi
ls -la "$OUT/Phonon-Setup-$VERSION.exe"
