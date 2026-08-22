#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)
bar_dir="$project_dir/bar"
app_path="$bar_dir/dist/Phonon.app"

cargo build --release --package phonon-cli --bin phonon --manifest-path "$project_dir/Cargo.toml"
swift build --disable-sandbox -c release --package-path "$bar_dir"
bin_dir=$(swift build --disable-sandbox -c release --package-path "$bar_dir" --show-bin-path)

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
cp "$project_dir/assets/startup.wav" "$staged_app/Contents/Resources/assets/startup.wav"
cp "$project_dir/assets/record_start.wav" "$staged_app/Contents/Resources/assets/record_start.wav"
cp "$project_dir/assets/record_stop.wav" "$staged_app/Contents/Resources/assets/record_stop.wav"
cp "$project_dir/assets/english_words.txt" "$staged_app/Contents/Resources/assets/english_words.txt"
cp "$project_dir/prompts/polish_v2.txt" "$staged_app/Contents/Resources/prompts/polish_v2.txt"
cp "$project_dir/third_party/uv/LICENSE-APACHE" "$project_dir/third_party/uv/LICENSE-MIT" \
	"$staged_app/Contents/Resources/licenses/uv/"
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
