#!/bin/sh
# Installs an existing build/Phonon.ipa (from scripts/export-ipa.sh) on a connected iPhone.
# It never builds, and it refuses to run without an explicit go-ahead from the phone's owner:
#   PHONON_INSTALL_APPROVED=yes scripts/install-on-iphone.sh [device-id] [path/to/Phonon.ipa]
# Device ids: xcrun devicectl list devices. The phone must be unlocked, trusted, Developer Mode on.
set -eu
[ "${PHONON_INSTALL_APPROVED:-}" = "yes" ] || {
  echo "Refusing: set PHONON_INSTALL_APPROVED=yes only when the phone's owner has asked for this install."; exit 2; }
here="$(cd "$(dirname "$0")/.." && pwd)"
device="${1:-}"
ipa="${2:-$here/build/Phonon.ipa}"
[ -f "$ipa" ] || { echo "No $ipa. Run scripts/export-ipa.sh first."; exit 1; }
[ -n "$device" ] || { echo "Name the device: xcrun devicectl list devices"; exit 1; }
shasum -a 256 "$ipa"
# devicectl installs "an app bundle (with a .app extension)" (its --help), not an .ipa.
tmp="$(mktemp -d)"
ditto -x -k "$ipa" "$tmp"
xcrun devicectl device install app --device "$device" "$tmp/Payload/Phonon.app"
echo "Installed. Open Phonon on the phone to download the models (1.45 GB, Wi-Fi)."
