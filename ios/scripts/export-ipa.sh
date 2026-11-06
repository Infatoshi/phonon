#!/bin/sh
# Archives Phonon, signs it for development with team YOUR_TEAM_ID and exports build/Phonon.ipa,
# then prints its sha256. Touches no device. Xcode creates the Development certificate, App
# IDs, App Group and profiles itself through the App Store Connect API key. The profile covers
# the devices already registered with the team.
#   scripts/export-ipa.sh
set -eu
here="$(cd "$(dirname "$0")/.." && pwd)"
out="${PHONON_OUT:-$here/build}"
key="${ASC_KEY_PATH:-$HOME/.asc/AuthKey_YOUR_KEY_ID.p8}"
key_id="${ASC_KEY_ID:-YOUR_KEY_ID}"
issuer="${ASC_ISSUER_ID:-YOUR_ISSUER_ID}"
auth="-allowProvisioningUpdates -authenticationKeyPath $key -authenticationKeyID $key_id -authenticationKeyIssuerID $issuer"
cd "$here"
"$here/scripts/vendor.sh"
command -v xcodegen >/dev/null || { echo "Install xcodegen: brew install xcodegen"; exit 1; }
xcodegen generate --quiet
mkdir -p "$out"
log="$out/export.log"
# shellcheck disable=SC2086
if ! xcodebuild -project Phonon.xcodeproj -scheme Phonon -configuration Release \
  -destination "generic/platform=iOS" -archivePath "$out/Phonon.xcarchive" $auth archive > "$log" 2>&1; then
  echo "Archive failed. Signing and provisioning errors:"; grep -E 'error:' "$log" | sort -u; exit 1
fi
cat > "$out/export.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>method</key><string>debugging</string>
<key>teamID</key><string>YOUR_TEAM_ID</string>
<key>signingStyle</key><string>automatic</string>
</dict></plist>
PLIST
rm -rf "$out/export"
# shellcheck disable=SC2086
if ! xcodebuild -exportArchive -archivePath "$out/Phonon.xcarchive" -exportPath "$out/export" \
  -exportOptionsPlist "$out/export.plist" $auth >> "$log" 2>&1; then
  echo "Export failed:"; grep -E 'error:' "$log" | sort -u; exit 1
fi
cp "$out/export/Phonon.ipa" "$out/Phonon.ipa"
shasum -a 256 "$out/Phonon.ipa"
echo "Devices in the profile:"
security cms -D -i "$out/Phonon.xcarchive/Products/Applications/Phonon.app/embedded.mobileprovision" 2>/dev/null \
  | plutil -extract ProvisionedDevices json -o - - 2>/dev/null || echo "(none listed)"
