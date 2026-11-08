#!/usr/bin/env bash
# Creates (once) and boots the headless phonon AVD on a Linux build host. All state stays under /data/android.
set -euo pipefail
export ANDROID_HOME=/data/android/sdk JAVA_HOME=/data/android/jdk ANDROID_USER_HOME=/data/android/user ANDROID_AVD_HOME=/data/android/avd
mkdir -p $ANDROID_USER_HOME $ANDROID_AVD_HOME
if [ ! -d $ANDROID_AVD_HOME/phonon.avd ]; then
  echo no | $ANDROID_HOME/cmdline-tools/latest/bin/avdmanager create avd -n phonon -k "system-images;android-35;google_apis;x86_64" -d pixel_6
  C=$ANDROID_AVD_HOME/phonon.avd/config.ini
  grep -vE '^(hw.ramSize|hw.cpu.ncore|hw.keyboard|disk.dataPartition.size|hw.audioInput)=' $C > $C.new || true
  printf 'hw.ramSize=8192\nhw.cpu.ncore=4\nhw.keyboard=no\ndisk.dataPartition.size=12G\nhw.audioInput=yes\n' >> $C.new
  mv $C.new $C
fi
exec $ANDROID_HOME/emulator/emulator -avd phonon -no-window -no-snapshot -no-boot-anim -gpu swiftshader_indirect -accel on -allow-host-audio "$@"
