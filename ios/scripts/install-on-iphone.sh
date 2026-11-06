#!/bin/sh
# Builds Phonon, signs it for development with team YOUR_TEAM_ID, writes Phonon.ipa, and
# installs it on a connected iPhone. Xcode creates the Development certificate, the App
# IDs, the App Group and the profiles itself through the App Store Connect API key, and
# registers the phone. Nothing goes to the App Store or TestFlight.
#
#   scripts/install-on-iphone.sh                 # first iPhone devicectl can reach
#   scripts/install-on-iphone.sh <device-udid>   # a specific one (xcrun devicectl list devices)
#
# Needs: the phone unlocked, trusted, Developer Mode on (Settings > Privacy & Security).
set -eu
here="$(cd "$(dirname "$0")/.." && pwd)"
out="${PHONON_OUT:-$here/build}"
key="${ASC_KEY_PATH:-$HOME/.asc/AuthKey_YOUR_KEY_ID.p8}"
key_id="${ASC_KEY_ID:-YOUR_KEY_ID}"
issuer="${ASC_ISSUER_ID:-YOUR_ISSUER_ID}"
auth="-allowProvisioningUpdates -allowProvisioningDeviceRegistration -authenticationKeyPath $key -authenticationKeyID $key_id -authenticationKeyIssuerID $issuer"

cd "$here"
"$here/scripts/vendor.sh"
command -v xcodegen >/dev/null || { echo "Install xcodegen: brew install xcodegen"; exit 1; }
xcodegen generate --quiet

device="${1:-}"
if [ -z "$device" ]; then
  device="$(xcrun devicectl list devices 2>/dev/null | awk '/available/ && /iPhone/ {print $3; exit}')"
fi
[ -n "$device" ] || { echo "No reachable iPhone. Connect it by cable, unlock it, and try again."; exit 1; }
hw="$(xcrun devicectl device info details --device "$device" 2>/dev/null | awk -F': ' '/udid:/ {print $2; exit}')"
echo "Device: $device ${hw:+(udid $hw)}"

mkdir -p "$out"
if [ -n "$hw" ]; then
  # A build for this exact phone registers it with the team, so the profile includes it.
  # shellcheck disable=SC2086
  xcodebuild -project Phonon.xcodeproj -scheme Phonon -configuration Release \
    -destination "id=$hw" $auth build | tail -1
fi
# shellcheck disable=SC2086
xcodebuild -project Phonon.xcodeproj -scheme Phonon -configuration Release \
  -destination "generic/platform=iOS" -archivePath "$out/Phonon.xcarchive" \
  $auth archive | tail -3
cat > "$out/export.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>method</key><string>debugging</string>
<key>teamID</key><string>YOUR_TEAM_ID</string>
<key>signingStyle</key><string>automatic</string>
</dict></plist>
EOF
rm -rf "$out/export"
# shellcheck disable=SC2086
xcodebuild -exportArchive -archivePath "$out/Phonon.xcarchive" -exportPath "$out/export" \
  -exportOptionsPlist "$out/export.plist" $auth | tail -2
cp "$out/export/Phonon.ipa" "$out/Phonon.ipa"
shasum -a 256 "$out/Phonon.ipa"
xcrun devicectl device install app --device "$device" "$out/Phonon.ipa"
echo "Installed. Open Phonon on the phone to download the models (1.45 GB, Wi-Fi)."
