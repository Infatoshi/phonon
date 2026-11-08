#!/usr/bin/env bash
# Builds the debug APK on a Linux build host. Source is rsynced to /data/android/work/src from the Mac.
set -euo pipefail
export ANDROID_HOME=/data/android/sdk JAVA_HOME=/data/android/jdk GRADLE_USER_HOME=/data/android/gradle-home
S=/data/android/work/src D=/data/android/deps
G=/data/android/gradle/gradle-8.11.1/bin/gradle
if [ ! -x $G ]; then
  mkdir -p /data/android/gradle && cd /data/android/gradle
  curl -fsSLO https://services.gradle.org/distributions/gradle-8.11.1-bin.zip && unzip -q gradle-8.11.1-bin.zip
fi
K=$S/app/src/main/java/com/k2fsa/sherpa/onnx; mkdir -p $K
for f in OfflineRecognizer OfflineStream FeatureConfig QnnConfig HomophoneReplacerConfig; do
  cp $D/sherpa-src/sherpa-onnx/kotlin-api/$f.kt $K/
done
for abi in x86_64 arm64-v8a; do
  mkdir -p $S/app/src/main/jniLibs/$abi
  cp $D/sherpa-android/jniLibs/$abi/{libsherpa-onnx-jni.so,libonnxruntime.so} $S/app/src/main/jniLibs/$abi/
done
echo "sdk.dir=$ANDROID_HOME" > $S/local.properties
cd $S && $G --no-daemon -q assembleDebug assembleDebugAndroidTest
ls -la $S/app/build/outputs/apk/debug/ $S/app/build/outputs/apk/androidTest/debug/
