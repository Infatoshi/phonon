#!/usr/bin/env bash
# Installs the Android SDK pieces under /data/android on a Linux build host (no sudo).
set -euo pipefail
export ANDROID_HOME=/data/android/sdk
mkdir -p $ANDROID_HOME/cmdline-tools /data/android/dl
cd /data/android/dl
if [ ! -x $ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager ]; then
  curl -fsSLO https://dl.google.com/android/repository/commandlinetools-linux-13114758_latest.zip
  unzip -q -o commandlinetools-linux-13114758_latest.zip -d tmpct
  rm -rf $ANDROID_HOME/cmdline-tools/latest && mv tmpct/cmdline-tools $ANDROID_HOME/cmdline-tools/latest
fi
if [ ! -d /data/android/jdk ]; then
  curl -fsSL -o jdk.tgz "https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jdk/hotspot/normal/eclipse"
  mkdir -p jdktmp && tar xzf jdk.tgz -C jdktmp && mv jdktmp/* /data/android/jdk
fi
export JAVA_HOME=/data/android/jdk
SM=$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager
yes | $SM --licenses >/dev/null || true
$SM --install "platform-tools" "emulator" "platforms;android-35" "build-tools;35.0.0" \
  "system-images;android-35;google_apis;x86_64" "ndk;27.2.12479018" "cmake;3.22.1"
echo SDK_DONE
