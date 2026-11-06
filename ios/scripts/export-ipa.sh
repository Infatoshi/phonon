#!/bin/sh
# Archives Phonon, signs it for development with team YOUR_TEAM_ID and exports build/Phonon.ipa,
# then prints its sha256. Touches no device. Xcode's automatic signing creates the App IDs, App
# Group and profiles through the developer account signed in under Xcode > Settings > Accounts.
# The profile covers the devices already registered with the team.
#   scripts/export-ipa.sh
# ASC_KEY_PATH, ASC_KEY_ID and ASC_ISSUER_ID sign through an App Store Connect API key instead;
# on 2026-11-06 that route created the certificate but Xcode's App ID step answered
# "Authentication failed", so the signed-in account is the default.
set -eu
here="$(cd "$(dirname "$0")/.." && pwd)"
out="${PHONON_OUT:-$here/build}"
auth="-allowProvisioningUpdates"
if [ -n "${ASC_KEY_PATH:-}" ]; then
  auth="$auth -authenticationKeyPath $ASC_KEY_PATH -authenticationKeyID $ASC_KEY_ID -authenticationKeyIssuerID $ASC_ISSUER_ID"
fi
cd "$here"
command -v xcodegen >/dev/null || { echo "Install xcodegen: brew install xcodegen"; exit 1; }
xcodegen generate --quiet
# The .ipa is tied to one commit: a clean tree after generating (so generated plists match the
# commit too), and the commit in Info.plist (PhononCommit, Settings > Diagnostics > Build).
if [ -n "$(git status --porcelain -- .)" ]; then echo "Commit or stash the changes in ios/ first:"; git status --short -- .; exit 1; fi
commit="$(git rev-parse HEAD)"
"$here/scripts/vendor.sh"
"$here/scripts/check-offline-asr.sh"
mkdir -p "$out"
log="$out/export.log"
# shellcheck disable=SC2086
if ! xcodebuild -project Phonon.xcodeproj -scheme Phonon -configuration Release \
  -destination "generic/platform=iOS" -archivePath "$out/Phonon.xcarchive" PHONON_COMMIT="$commit" $auth archive > "$log" 2>&1; then
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
stamped="$(/usr/libexec/PlistBuddy -c 'Print :PhononCommit' "$out/Phonon.xcarchive/Products/Applications/Phonon.app/Info.plist")"
[ "$stamped" = "$commit" ] || { echo "Info.plist says $stamped, HEAD is $commit"; exit 1; }
echo "Commit: $commit"
# The signed .app alone, for `xcrun devicectl device install app` (it takes a .app, not an .ipa).
(cd "$out/export" && rm -rf app && mkdir app && ditto -x -k Phonon.ipa app && ditto -c -k --keepParent app/Payload/Phonon.app "$out/Phonon-app.zip")
shasum -a 256 "$out/Phonon.ipa" "$out/Phonon-app.zip"
echo "Devices in the profile:"
security cms -D -i "$out/Phonon.xcarchive/Products/Applications/Phonon.app/embedded.mobileprovision" 2>/dev/null \
  | plutil -extract ProvisionedDevices json -o - - 2>/dev/null || echo "(none listed)"
