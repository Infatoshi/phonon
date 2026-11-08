#!/bin/bash
# Build Phonon for iPhone, install it on the connected iPhone and launch it.
# usage: ./run.sh [--no-launch]
set -euo pipefail
cd "$(dirname "$0")"
DEVICE=${PHONON_IOS_DEVICE:?set PHONON_IOS_DEVICE to the iPhone UDID}
: "${PHONON_TEAM_ID:?set PHONON_TEAM_ID to your Apple team ID}"
xcodegen generate --quiet
xcodebuild -project PhononiOS.xcodeproj -scheme Phonon -configuration Debug \
  -destination "platform=iOS,id=$DEVICE" -derivedDataPath build -allowProvisioningUpdates build \
  > build.log 2>&1 || { grep -E "error:|BUILD FAILED" build.log | head -20; exit 1; }
APP=build/Build/Products/Debug-iphoneos/Phonon.app
xcrun devicectl device install app --device "$DEVICE" "$APP" > install.log 2>&1 || { tail -20 install.log; exit 1; }
echo "installed $APP"
if [ "${1:-}" != --no-launch ]; then
  xcrun devicectl device process launch --device "$DEVICE" sh.phonon.ios > /dev/null && echo launched
fi
