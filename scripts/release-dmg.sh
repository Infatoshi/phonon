#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)
bar_dir="$project_dir/bar"
app_path="$bar_dir/dist/Phonon.app"
dmg_path="$bar_dir/dist/Phonon.dmg"
notary_profile=${PHONON_NOTARY_PROFILE:-phonon-notary}

identity=${PHONON_CODESIGN_IDENTITY:-}
if [[ -z "$identity" ]]; then
	identity=$(security find-identity -v -p codesigning 2>/dev/null |
		sed -n 's/.*"\(Developer ID Application:[^"]*\)".*/\1/p' |
		head -n 1)
fi
if [[ -z "$identity" ]]; then
	printf 'Developer ID Application identity not found.\n' >&2
	exit 1
fi

PHONON_CODESIGN_IDENTITY="$identity" "$script_dir/package-bar.sh"

codesign --verify --deep --strict --verbose=4 "$app_path"
codesign -dvv "$app_path" 2>&1 | grep -E 'Authority=Developer ID Application|TeamIdentifier=|Timestamp='
for executable in \
	"$app_path/Contents/MacOS/PhononBar" \
	"$app_path/Contents/Helpers/phonon" \
	"$app_path/Contents/Helpers/uv"; do
	lipo "$executable" -verify_arch arm64
	codesign --verify --strict --verbose=4 "$executable"
done
"$app_path/Contents/Helpers/uv" --version

dmg_root=$(mktemp -d "${TMPDIR:-/tmp}/phonon-dmg.XXXXXX")
notary_result=$(mktemp "${TMPDIR:-/tmp}/phonon-notary-result.XXXXXX.json")
cleanup() {
	rm -rf "$dmg_root"
	rm -f "$notary_result"
}
trap cleanup EXIT

ditto "$app_path" "$dmg_root/Phonon.app"
ln -s /Applications "$dmg_root/Applications"
rm -f "$dmg_path"
hdiutil create -volname Phonon -srcfolder "$dmg_root" -ov -format UDZO "$dmg_path"
codesign --force --sign "$identity" --timestamp "$dmg_path"
codesign --verify --verbose=4 "$dmg_path"

if [[ ${PHONON_SKIP_NOTARIZATION:-0} == 1 ]]; then
	printf 'Signed DMG: %s\n' "$dmg_path"
	exit 0
fi

xcrun notarytool submit "$dmg_path" \
	--keychain-profile "$notary_profile" \
	--wait --timeout 2h --output-format json >"$notary_result"

submission_id=$(jq -r '.id // empty' "$notary_result")
status=$(jq -r '.status // empty' "$notary_result")
if [[ -z "$submission_id" ]]; then
	printf 'Notary response did not include a submission ID.\n' >&2
	jq . "$notary_result" >&2
	exit 1
fi

log_path="$bar_dir/dist/notarization-$submission_id.json"
xcrun notarytool log "$submission_id" "$log_path" \
	--keychain-profile "$notary_profile"

if [[ "$status" != Accepted ]]; then
	printf 'Notarization status: %s\n' "$status" >&2
	jq '.issues' "$log_path" >&2
	exit 1
fi

issue_count=$(jq '[.issues[]? | select(.severity == "warning" or .severity == "error")] | length' "$log_path")
if [[ "$issue_count" != 0 ]]; then
	printf 'Notarization completed with %s warning/error issue(s).\n' "$issue_count" >&2
	jq '.issues' "$log_path" >&2
	exit 1
fi

xcrun stapler staple "$dmg_path"
xcrun stapler validate "$dmg_path"
spctl --assess --type open --context context:primary-signature --verbose=4 "$dmg_path"

mount_info=$(hdiutil attach "$dmg_path" -nobrowse -readonly -plist)
mount_point=$(printf '%s' "$mount_info" | plutil -convert json -o - - |
	jq -r '[."system-entities"[] | ."mount-point"?] | map(select(. != null)) | first')
if [[ -z "$mount_point" || "$mount_point" == null ]]; then
	printf 'Unable to locate mounted DMG.\n' >&2
	exit 1
fi
trap 'hdiutil detach "$mount_point" >/dev/null; cleanup' EXIT
spctl --assess --type exec --verbose=4 "$mount_point/Phonon.app"
hdiutil detach "$mount_point" >/dev/null
trap cleanup EXIT

printf 'Notarized DMG: %s\n' "$dmg_path"
printf 'Submission ID: %s\n' "$submission_id"
printf 'Notarization log: %s\n' "$log_path"
