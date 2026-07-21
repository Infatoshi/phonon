#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)
bar_dir="$project_dir/bar"
app_path="$bar_dir/dist/Phonon.app"

swift build -c release --package-path "$bar_dir"
bin_dir=$(swift build -c release --package-path "$bar_dir" --show-bin-path)

staging_dir=$(mktemp -d "${TMPDIR:-/tmp}/phonon-app.XXXXXX")
trap 'rm -rf "$staging_dir"' EXIT
staged_app="$staging_dir/Phonon.app"
mkdir -p "$staged_app/Contents/MacOS" "$staged_app/Contents/Resources"
cp "$bar_dir/Resources/Info.plist" "$staged_app/Contents/Info.plist"
cp "$bin_dir/PhononBar" "$staged_app/Contents/MacOS/PhononBar"
"$script_dir/make-app-icon.sh" "$staged_app/Contents/Resources/Phonon.icns"
chmod 755 "$staged_app/Contents/MacOS/PhononBar"

identity=${PHONON_CODESIGN_IDENTITY:-}
if [[ -z "$identity" ]]; then
	identity=$(security find-identity -v -p codesigning 2>/dev/null |
		sed -n 's/.*"\(Apple Development:[^"]*\)".*/\1/p' |
		head -n 1)
fi
if [[ -z "$identity" ]]; then
	identity=-
fi

codesign --force --sign "$identity" --identifier com.infatoshi.phonon \
	--options runtime --timestamp=none \
	--entitlements "$bar_dir/Resources/Phonon.entitlements" "$staged_app"
codesign --verify --deep --strict "$staged_app"

mkdir -p "$(dirname "$app_path")"
if [[ -e "$app_path" ]]; then
	rm -rf "$app_path"
fi
mv "$staged_app" "$app_path"
printf '%s\n' "$app_path"
