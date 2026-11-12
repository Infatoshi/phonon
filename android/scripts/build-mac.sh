#!/usr/bin/env bash
# Builds the debug APK and runs the unit tests on a Mac, the counterpart of build.sh (Linux build host).
#
# Needs the Android SDK (default ~/Library/Android/sdk) with platforms;android-35, ndk;27.2.12479018 and
# cmake;3.22.1, a JDK 17+ (default: Android Studio's), and git and curl. The native dependencies (sherpa-onnx's
# prebuilt libraries and Kotlin API, a llama.cpp checkout) go to target/android-deps; the build runs on a copy of
# android/ in target/android-work, so the checkout stays free of build output.
#
#   android/scripts/build-mac.sh            # APK: target/android-work/src/app/build/outputs/apk/debug/app-debug.apk
#   PHONON_ANDROID_TASKS="testDebugUnitTest" android/scripts/build-mac.sh
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
repo=$(cd "$here/.." && pwd)
deps=${PHONON_ANDROID_DEPS:-$repo/target/android-deps}
work=$repo/target/android-work
export ANDROID_HOME=${ANDROID_HOME:-$HOME/Library/Android/sdk}
# Gradle 8.11 runs on JDK 17 to 23 (Android Studio's own JBR may be newer): the first such JDK found.
if [ -z "${JAVA_HOME:-}" ]; then
	for candidate in "$HOME"/.gradle/jdks/jdk-2[1-3]*/Contents/Home "$HOME"/.gradle/jdks/jdk-17*/Contents/Home \
		/Library/Java/JavaVirtualMachines/*-2[1-3]*/Contents/Home /Library/Java/JavaVirtualMachines/*-17*/Contents/Home \
		"/Applications/Android Studio.app/Contents/jbr/Contents/Home"; do
		if [ -x "$candidate/bin/java" ]; then
			export JAVA_HOME=$candidate
			break
		fi
	done
fi
gradle_version=8.11.1
tasks=${PHONON_ANDROID_TASKS:-testDebugUnitTest assembleDebug}

mkdir -p "$deps"
cd "$deps"
if [ ! -d sherpa-android/jniLibs ]; then
	[ -f sherpa-onnx-v2.0.1-android.tar.bz2 ] ||
		curl -fsSLO https://github.com/k2-fsa/sherpa-onnx/releases/download/v2.0.1/sherpa-onnx-v2.0.1-android.tar.bz2
	mkdir -p sherpa-android && tar xjf sherpa-onnx-v2.0.1-android.tar.bz2 -C sherpa-android
fi
[ -d sherpa-src ] || git clone -q --depth 1 --branch v2.0.1 https://github.com/k2-fsa/sherpa-onnx.git sherpa-src
# The llama.cpp commit the Mac's native refiner is built from (scripts/build-engine.sh).
llama_commit=39febd2adb947b70da35e95f9351b3bf58ad170a
if [ ! -f llama.cpp/.phonon-commit ] || [ "$(cat llama.cpp/.phonon-commit)" != "$llama_commit" ]; then
	rm -rf llama.cpp && mkdir llama.cpp
	git -C llama.cpp init -q
	git -C llama.cpp fetch -q --depth 1 https://github.com/ggml-org/llama.cpp "$llama_commit"
	git -C llama.cpp checkout -q FETCH_HEAD
	echo "$llama_commit" >llama.cpp/.phonon-commit
fi
gradle=$deps/gradle-$gradle_version/bin/gradle
if [ ! -x "$gradle" ]; then
	curl -fsSLO "https://services.gradle.org/distributions/gradle-$gradle_version-bin.zip"
	unzip -q -o "gradle-$gradle_version-bin.zip"
fi

src=$work/src
mkdir -p "$src"
rsync -a --delete --exclude /app/build --exclude /.gradle --exclude /build --exclude /local.properties \
	--exclude /app/.cxx --exclude /app/src/main/jniLibs --exclude /app/src/main/java/com/k2fsa \
	"$here/" "$src/"
# The unit tests (LineBreaksTest, OpenerTest) read the Rust modules' reference cases from
# ../crates (relative to app/).
mkdir -p "$src/crates/phonon-llm/src"
cp "$repo"/crates/phonon-llm/src/*_cases.json "$src/crates/phonon-llm/src/"
kotlin=$src/app/src/main/java/com/k2fsa/sherpa/onnx
mkdir -p "$kotlin"
for f in OfflineRecognizer OfflineStream FeatureConfig QnnConfig HomophoneReplacerConfig; do
	cp "$deps/sherpa-src/sherpa-onnx/kotlin-api/$f.kt" "$kotlin/"
done
for abi in x86_64 arm64-v8a; do
	mkdir -p "$src/app/src/main/jniLibs/$abi"
	cp "$deps/sherpa-android/jniLibs/$abi/"{libsherpa-onnx-jni.so,libonnxruntime.so} "$src/app/src/main/jniLibs/$abi/"
done
echo "sdk.dir=$ANDROID_HOME" >"$src/local.properties"

cd "$src"
# shellcheck disable=SC2086 # tasks is a list
"$gradle" --no-daemon -q -Pphonon.llamaDir="$deps/llama.cpp" $tasks
ls -la "$src/app/build/outputs/apk/debug/" 2>/dev/null || true
