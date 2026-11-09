#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)
bar_dir="$project_dir/bar"
app_path="$bar_dir/dist/Phonon.app"

# Panic locations carry source paths: map the builder's home, Cargo registry
# and checkout to neutral prefixes (later mappings win), and leave out the
# checkout fallback in phonon_core::project_root, so the shipped binary holds
# no path from the machine that built it.
cargo_home=${CARGO_HOME:-$HOME/.cargo}
export RUSTFLAGS="${RUSTFLAGS:-} --remap-path-prefix=$HOME=/home --remap-path-prefix=$cargo_home=/cargo --remap-path-prefix=$project_dir=/phonon --cfg phonon_bundle"
cargo build --release --package phonon-cli --bin phonon --manifest-path "$project_dir/Cargo.toml"
# The same for PhononBar: the prefix maps cover #file literals and debug info (Swift and the ObjC target), and
# ld -S leaves out the debug map (N_SO/N_OSO stabs naming every source directory and object file), which the
# shipped app does not use. Function symbols stay, so crash reports still name functions.
swift_flags=(-c release --package-path "$bar_dir"
	-Xswiftc -file-prefix-map -Xswiftc "$project_dir=/phonon"
	-Xswiftc -debug-prefix-map -Xswiftc "$project_dir=/phonon"
	-Xcc "-ffile-prefix-map=$project_dir=/phonon"
	-Xlinker -S)
swift build --disable-sandbox "${swift_flags[@]}"
bin_dir=$(swift build --disable-sandbox "${swift_flags[@]}" --show-bin-path)
home_paths=$(LC_ALL=C grep -a -c -F "$HOME" "$bin_dir/PhononBar" || true)
if [[ "$home_paths" != 0 ]]; then
	printf 'PhononBar still holds %s path(s) under %s\n' "$home_paths" "$HOME" >&2
	exit 1
fi

staging_dir=$(mktemp -d "${TMPDIR:-/tmp}/phonon-app.XXXXXX")
trap 'rm -rf "$staging_dir"' EXIT
staged_app="$staging_dir/Phonon.app"
mkdir -p "$staged_app/Contents/MacOS" "$staged_app/Contents/Helpers" \
	"$staged_app/Contents/Resources/sidecar" "$staged_app/Contents/Resources/assets" \
	"$staged_app/Contents/Resources/prompts" "$staged_app/Contents/Resources/licenses/uv"
cp "$bar_dir/Resources/Info.plist" "$staged_app/Contents/Info.plist"
cp "$bin_dir/PhononBar" "$staged_app/Contents/MacOS/PhononBar"
cp "$project_dir/target/release/phonon" "$staged_app/Contents/Helpers/phonon"
uv_bin=${PHONON_UV_BIN:-$(command -v uv || true)}
if [[ -z "$uv_bin" || ! -x "$uv_bin" ]]; then
	printf 'uv executable not found; set PHONON_UV_BIN or install uv.\n' >&2
	exit 1
fi
cp -L "$uv_bin" "$staged_app/Contents/Helpers/uv"
cp "$project_dir/sidecar/asr_server.py" "$staged_app/Contents/Resources/sidecar/asr_server.py"
cp "$project_dir/sidecar/polish_server.py" "$staged_app/Contents/Resources/sidecar/polish_server.py"
# modules the two sidecars import (streaming refine: SPEC "Streaming refine")
for module in linereader fast_tdt stream_asr stream_vad stream_decode stream_draft stream_refiner pa_client asr_terms cache_gc; do
	cp "$project_dir/sidecar/$module.py" "$staged_app/Contents/Resources/sidecar/$module.py"
done
mkdir -p "$staged_app/Contents/Resources/sidecar/models"
cp "$project_dir"/sidecar/models/*.py "$staged_app/Contents/Resources/sidecar/models/"
cp "$project_dir/assets/startup.wav" "$staged_app/Contents/Resources/assets/startup.wav"
cp "$project_dir/assets/record_start.wav" "$staged_app/Contents/Resources/assets/record_start.wav"
cp "$project_dir/assets/record_stop.wav" "$staged_app/Contents/Resources/assets/record_stop.wav"
cp "$project_dir/assets/english_words.txt" "$staged_app/Contents/Resources/assets/english_words.txt"
cp "$project_dir/prompts/polish_v2.txt" "$staged_app/Contents/Resources/prompts/polish_v2.txt"
cp "$project_dir/third_party/uv/LICENSE-APACHE" "$project_dir/third_party/uv/LICENSE-MIT" \
	"$staged_app/Contents/Resources/licenses/uv/"

# The native refiner engine: `pe` plus the Metal shader sources it compiles at
# startup. PHONON_ENGINE_DIR names a directory holding pe, mk.metal and
# df.metal (scripts/build-engine.sh writes target/engine). Without it the app
# still corrects on the MLX refiner, only slower; PHONON_REQUIRE_ENGINE=1 (the
# DMG release) makes a missing engine an error instead.
engine_dir=${PHONON_ENGINE_DIR:-$project_dir/target/engine}
if [[ -x "$engine_dir/pe" && -f "$engine_dir/mk.metal" && -f "$engine_dir/df.metal" ]]; then
	lipo "$engine_dir/pe" -verify_arch arm64
	# dyld refuses a binary whose minimum macOS is newer than the system, and
	# the Command Line Tools SDK defaults it to the build machine's version.
	engine_minos=$(otool -l "$engine_dir/pe" | awk '/LC_BUILD_VERSION/ {found = 1} found && $1 == "minos" {print $2; exit}')
	app_minos=$(/usr/libexec/PlistBuddy -c 'Print :LSMinimumSystemVersion' "$bar_dir/Resources/Info.plist")
	if [[ "$(printf '%s\n%s\n' "$engine_minos" "$app_minos" | sort -V | tail -n 1)" != "$app_minos" ]]; then
		printf 'engine minimum macOS %s is newer than the app minimum %s; rebuild it with scripts/build-engine.sh\n' \
			"$engine_minos" "$app_minos" >&2
		exit 1
	fi
	mkdir -p "$staged_app/Contents/Resources/engine" "$staged_app/Contents/Resources/licenses/llama.cpp"
	cp "$engine_dir/pe" "$staged_app/Contents/Helpers/pe"
	chmod 755 "$staged_app/Contents/Helpers/pe"
	cp "$engine_dir/mk.metal" "$engine_dir/df.metal" "$staged_app/Contents/Resources/engine/"
	cp "$project_dir"/third_party/llama.cpp/LICENSE* "$staged_app/Contents/Resources/licenses/llama.cpp/"
elif [[ ${PHONON_REQUIRE_ENGINE:-0} == 1 ]]; then
	printf 'native refiner engine not found in %s; run scripts/build-engine.sh\n' "$engine_dir" >&2
	exit 1
else
	printf 'note: no native refiner engine in %s; this build corrects on the MLX refiner\n' "$engine_dir" >&2
fi
# Metal Parakeet: `pa` plus its shader source and mel filterbank (scripts/build-engine.sh writes them next to pe).
# The ASR sidecar runs it when present (PHONON_METAL_ASR=0 opts out) and keeps MLX as its fallback, so a build
# without it still transcribes; PHONON_REQUIRE_ENGINE=1 (the DMG release) makes a missing helper an error.
if [[ -x "$engine_dir/pa" && -f "$engine_dir/asr/pa.metal" && -f "$engine_dir/asr/melfb.npy" ]]; then
	lipo "$engine_dir/pa" -verify_arch arm64
	pa_minos=$(otool -l "$engine_dir/pa" | awk '/LC_BUILD_VERSION/ {found = 1} found && $1 == "minos" {print $2; exit}')
	app_minos=$(/usr/libexec/PlistBuddy -c 'Print :LSMinimumSystemVersion' "$bar_dir/Resources/Info.plist")
	if [[ "$(printf '%s\n%s\n' "$pa_minos" "$app_minos" | sort -V | tail -n 1)" != "$app_minos" ]]; then
		printf 'pa minimum macOS %s is newer than the app minimum %s; rebuild it with scripts/build-engine.sh\n' \
			"$pa_minos" "$app_minos" >&2
		exit 1
	fi
	mkdir -p "$staged_app/Contents/Resources/asr"
	cp "$engine_dir/pa" "$staged_app/Contents/Helpers/pa"
	chmod 755 "$staged_app/Contents/Helpers/pa"
	cp "$engine_dir/asr/pa.metal" "$engine_dir/asr/melfb.npy" "$staged_app/Contents/Resources/asr/"
elif [[ ${PHONON_REQUIRE_ENGINE:-0} == 1 ]]; then
	printf 'Metal Parakeet helper not found in %s; run scripts/build-engine.sh\n' "$engine_dir" >&2
	exit 1
else
	printf 'note: no Metal Parakeet helper in %s; this build transcribes on MLX\n' "$engine_dir" >&2
fi
"$script_dir/make-app-icon.sh" "$staged_app/Contents/Resources/Phonon.icns"
chmod 755 "$staged_app/Contents/MacOS/PhononBar" \
	"$staged_app/Contents/Helpers/phonon" "$staged_app/Contents/Helpers/uv"

identity=${PHONON_CODESIGN_IDENTITY:-}
if [[ -z "$identity" ]]; then
	identity=$(security find-identity -v -p codesigning 2>/dev/null |
		sed -n 's/.*"\(Developer ID Application:[^"]*\)".*/\1/p' |
		head -n 1)
fi
if [[ -z "$identity" ]]; then
	identity=$(security find-identity -v -p codesigning 2>/dev/null |
		sed -n 's/.*"\(Apple Development:[^"]*\)".*/\1/p' |
		head -n 1)
fi
if [[ -z "$identity" ]]; then
	identity=-
fi

timestamp_args=(--timestamp=none)
if [[ "$identity" == "Developer ID Application:"* ]]; then
	timestamp_args=(--timestamp)
fi

codesign --force --sign "$identity" --options runtime "${timestamp_args[@]}" \
	"$staged_app/Contents/Helpers/uv"
codesign --force --sign "$identity" --options runtime "${timestamp_args[@]}" \
	"$staged_app/Contents/Helpers/phonon"
if [[ -f "$staged_app/Contents/Helpers/pe" ]]; then
	codesign --force --sign "$identity" --options runtime "${timestamp_args[@]}" \
		"$staged_app/Contents/Helpers/pe"
fi
if [[ -f "$staged_app/Contents/Helpers/pa" ]]; then
	codesign --force --sign "$identity" --options runtime "${timestamp_args[@]}" \
		"$staged_app/Contents/Helpers/pa"
fi
codesign --force --sign "$identity" --identifier com.infatoshi.phonon \
	--options runtime "${timestamp_args[@]}" \
	--entitlements "$bar_dir/Resources/Phonon.entitlements" "$staged_app"
codesign --verify --deep --strict "$staged_app"

mkdir -p "$(dirname "$app_path")"
if [[ -e "$app_path" ]]; then
	rm -rf "$app_path"
fi
mv "$staged_app" "$app_path"
printf '%s\n' "$app_path"
