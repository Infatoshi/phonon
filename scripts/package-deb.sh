#!/bin/bash
# Build the Linux .deb from a release build of phonon-win (run on a Linux x86_64 host).
# usage: scripts/package-deb.sh [path/to/phonon-win]   -> target/deb/phonon_<version>_amd64.deb
# Installs /usr/bin/phonon, the udev rule that lets the signed-in user read keyboards and use /dev/uinput
# (no group change, no logout), uinput autoload, a menu entry, and login autostart. Models download on first run.
# The settings window opens EGL (or GLX), xkbcommon, Wayland and the X11 libraries at run time; the resident app
# needs none of them, so they are Recommends, not Depends (Ubuntu and Debian desktops have them).
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=${1:-target/release/phonon-win}
[ -x "$BIN" ] || { echo "no binary at $BIN (cargo build --release -p phonon-win)"; exit 1; }
# PHONON_DEB_VERSION overrides the package version (a prerelease such as 1.0.1~beta3: apt sorts it above
# 1.0.0 and below 1.0.1, so `apt install ./file.deb` upgrades a beta that carried the same Cargo version).
# The file name uses "-" for "~".
VERSION=${PHONON_DEB_VERSION:-$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)}
FILE_VERSION=${VERSION//\~/-}
# the newest glibc symbol the binary links against (2.39 when built on Ubuntu 24.04)
GLIBC=$(objdump -T "$BIN" | grep -o 'GLIBC_[0-9.]*' | sort -V | tail -1 | cut -d_ -f2)
L=crates/phonon-win/linux
OUT=target/deb; ROOT=$OUT/phonon_${FILE_VERSION}_amd64
rm -rf "$ROOT"; mkdir -p "$ROOT/DEBIAN" "$ROOT/usr/bin" "$ROOT/usr/lib/udev/rules.d" "$ROOT/usr/lib/modules-load.d" \
  "$ROOT/usr/share/applications" "$ROOT/etc/xdg/autostart" "$ROOT/usr/share/doc/phonon"
install -m 0755 "$BIN" "$ROOT/usr/bin/phonon"
install -m 0644 $L/70-phonon.rules "$ROOT/usr/lib/udev/rules.d/70-phonon.rules"
echo uinput > "$ROOT/usr/lib/modules-load.d/phonon-uinput.conf"
install -m 0644 $L/phonon.desktop "$ROOT/usr/share/applications/phonon.desktop"
install -m 0644 $L/phonon.desktop "$ROOT/etc/xdg/autostart/phonon.desktop"
install -m 0644 LICENSE "$ROOT/usr/share/doc/phonon/copyright"
for s in 48 128 256; do
  install -D -m 0644 $L/icons/phonon-$s.png "$ROOT/usr/share/icons/hicolor/${s}x${s}/apps/phonon.png"
done
cat > "$ROOT/DEBIAN/control" <<EOF
Package: phonon
Version: $VERSION
Architecture: amd64
Maintainer: Phonon <https://github.com/Infatoshi/phonon/issues>
Homepage: https://github.com/Infatoshi/phonon
Section: utils
Priority: optional
Depends: libc6 (>= $GLIBC), pipewire-bin | pulseaudio-utils | alsa-utils
Recommends: libnotify-bin, gnome-shell-extension-appindicator, libegl1 | libgl1, libxkbcommon-x11-0, libxcursor1, libxi6, libwayland-egl1
Description: Local dictation: hold Right Ctrl, speak, and the text appears where you type
 Speech recognition (Parakeet) and correction (Phonon's refiner) run on this computer's CPU.
 About 1.6 GB of models download on first start; everything after that is offline.
EOF
cat > "$ROOT/DEBIAN/postinst" <<'EOF'
#!/bin/sh
set -e
if [ "$1" = configure ]; then
  gtk-update-icon-cache -q /usr/share/icons/hicolor 2>/dev/null || true
  modprobe uinput 2>/dev/null || true
  udevadm control --reload 2>/dev/null || true
  udevadm trigger --subsystem-match=input --subsystem-match=misc 2>/dev/null || true
fi
EOF
cat > "$ROOT/DEBIAN/postrm" <<'EOF'
#!/bin/sh
set -e
udevadm control --reload 2>/dev/null || true
EOF
chmod 0755 "$ROOT/DEBIAN/postinst" "$ROOT/DEBIAN/postrm"
dpkg-deb --root-owner-group --build "$ROOT" "$OUT/phonon_${FILE_VERSION}_amd64.deb" >/dev/null
ls -la "$OUT/phonon_${FILE_VERSION}_amd64.deb"
