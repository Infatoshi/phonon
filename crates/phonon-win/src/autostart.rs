//! Launch at login: the operating system's own switch, which this module reads
//! and changes. settings.json `launch_at_login` records the user's last pick.
//!
//! Windows: the installer puts `Phonon.lnk` in the user's Startup folder. On means
//! that shortcut or an `HKCU\...\Run` value named Phonon exists; turning it on writes
//! the Run value when the shortcut is gone, turning it off removes both and sets
//! `HKCU\Software\Phonon\NoLaunchAtLogin`, which tells a later install or upgrade not
//! to put the shortcut back (packaging/windows/phonon.nsi).
//!
//! Linux: the .deb installs `/etc/xdg/autostart/phonon.desktop`. A per-user
//! `~/.config/autostart/phonon.desktop` overrides it (XDG autostart spec): off
//! writes one with `Hidden=true`; on removes it, or, for a copy installed without
//! the package, writes an entry that starts this executable.

// The Linux rules are plain file operations, kept portable so every platform's tests cover them.
#![cfg_attr(windows, allow(dead_code))]

use std::path::Path;
#[cfg(any(windows, target_os = "linux", test))]
use std::path::PathBuf;

use anyhow::{Context, Result};

/// The autostart entry's file name, system-wide and per user.
pub const DESKTOP_FILE: &str = "phonon.desktop";
pub const SYSTEM_AUTOSTART_DIR: &str = "/etc/xdg/autostart";

/// Whether a desktop entry's text keeps it from starting.
fn desktop_disabled(text: &str) -> bool {
    text.lines().any(|line| {
        let line = line.trim();
        let Some((key, value)) = line.split_once('=') else {
            return false;
        };
        let value = value.trim().to_ascii_lowercase();
        match key.trim() {
            "Hidden" => value == "true",
            "X-GNOME-Autostart-enabled" => value == "false",
            _ => false,
        }
    })
}

/// Linux rule: the per-user entry decides when it exists, else the system one.
pub fn desktop_enabled(system: &Path, user: &Path) -> bool {
    match std::fs::read_to_string(user) {
        Ok(text) => !desktop_disabled(&text),
        Err(_) => std::fs::read_to_string(system)
            .map(|text| !desktop_disabled(&text))
            .unwrap_or(false),
    }
}

/// The per-user entry that starts `exec` (with `run`) at login.
fn user_entry(exec: &str, hidden: bool) -> String {
    let mut text = format!(
        "[Desktop Entry]\nType=Application\nName=Phonon\nComment=Local dictation\nExec={exec} run\nIcon=phonon\nTerminal=false\n"
    );
    if hidden {
        text.push_str("Hidden=true\n");
    } else {
        text.push_str("X-GNOME-Autostart-enabled=true\n");
    }
    text
}

/// Linux rule: turn the login start on or off through the per-user entry.
pub fn desktop_set(system: &Path, user: &Path, on: bool, exec: &str) -> Result<()> {
    if on && system.is_file() && !desktop_disabled(&std::fs::read_to_string(system)?) {
        match std::fs::remove_file(user) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error).with_context(|| format!("remove {}", user.display())),
        }
        return Ok(());
    }
    if let Some(dir) = user.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    }
    std::fs::write(user, user_entry(exec, !on)).with_context(|| format!("write {}", user.display()))
}

#[cfg(target_os = "linux")]
fn user_desktop() -> PathBuf {
    let config = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
        .unwrap_or_else(|| PathBuf::from("."));
    config.join("autostart").join(DESKTOP_FILE)
}

/// Whether Phonon starts at login.
#[cfg(target_os = "linux")]
pub fn enabled() -> bool {
    desktop_enabled(
        &Path::new(SYSTEM_AUTOSTART_DIR).join(DESKTOP_FILE),
        &user_desktop(),
    )
}

#[cfg(target_os = "linux")]
pub fn set(on: bool) -> Result<()> {
    // the .deb's /usr/bin/phonon, or wherever a bare download runs from
    let exec = std::env::current_exe()
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "phonon".into());
    desktop_set(
        &Path::new(SYSTEM_AUTOSTART_DIR).join(DESKTOP_FILE),
        &user_desktop(),
        on,
        &exec,
    )?;
    let _ = crate::settings::set(crate::settings::LAUNCH_AT_LOGIN, on.into());
    Ok(())
}

#[cfg(windows)]
mod win {
    use super::*;
    use windows_sys::Win32::Foundation::ERROR_SUCCESS;
    use windows_sys::Win32::System::Registry::{
        RegDeleteKeyValueW, RegGetValueW, RegSetKeyValueW, HKEY_CURRENT_USER, REG_DWORD, REG_SZ,
        RRF_RT_ANY,
    };

    const RUN_KEY: &str = r"Software\Microsoft\Windows\CurrentVersion\Run";
    const PHONON_KEY: &str = r"Software\Phonon";
    const VALUE: &str = "Phonon";
    const NO_LAUNCH: &str = "NoLaunchAtLogin";

    fn wide(text: &str) -> Vec<u16> {
        text.encode_utf16().chain(std::iter::once(0)).collect()
    }

    /// The installer's shortcut in the user's Startup folder.
    pub fn startup_shortcut() -> Option<PathBuf> {
        std::env::var_os("APPDATA").map(|appdata| {
            PathBuf::from(appdata).join(r"Microsoft\Windows\Start Menu\Programs\Startup\Phonon.lnk")
        })
    }

    fn run_value_exists() -> bool {
        let key = wide(RUN_KEY);
        let value = wide(VALUE);
        let mut size = 0u32;
        // SAFETY: NUL-terminated names; a null buffer asks only for the size.
        let status = unsafe {
            RegGetValueW(
                HKEY_CURRENT_USER,
                key.as_ptr(),
                value.as_ptr(),
                RRF_RT_ANY,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut size,
            )
        };
        status == ERROR_SUCCESS
    }

    fn set_value(key: &str, name: &str, kind: u32, data: &[u8]) -> Result<()> {
        let key_w = wide(key);
        let name_w = wide(name);
        // SAFETY: NUL-terminated names and a buffer of the stated length.
        let status = unsafe {
            RegSetKeyValueW(
                HKEY_CURRENT_USER,
                key_w.as_ptr(),
                name_w.as_ptr(),
                kind,
                data.as_ptr().cast(),
                data.len() as u32,
            )
        };
        if status != ERROR_SUCCESS {
            anyhow::bail!("write HKCU\\{key}\\{name}: error {status}");
        }
        Ok(())
    }

    fn delete_value(key: &str, name: &str) {
        let key = wide(key);
        let name = wide(name);
        // SAFETY: NUL-terminated names. A missing value is fine.
        unsafe { RegDeleteKeyValueW(HKEY_CURRENT_USER, key.as_ptr(), name.as_ptr()) };
    }

    pub fn enabled() -> bool {
        startup_shortcut().is_some_and(|path| path.is_file()) || run_value_exists()
    }

    pub fn set(on: bool) -> Result<()> {
        if on {
            delete_value(PHONON_KEY, NO_LAUNCH);
            if !startup_shortcut().is_some_and(|path| path.is_file()) {
                let exe = std::env::current_exe().context("find phonon-win.exe")?;
                let command = format!("\"{}\"", exe.display());
                let data: Vec<u8> = wide(&command)
                    .iter()
                    .flat_map(|unit| unit.to_le_bytes())
                    .collect();
                set_value(RUN_KEY, VALUE, REG_SZ, &data)?;
            }
        } else {
            delete_value(RUN_KEY, VALUE);
            if let Some(path) = startup_shortcut() {
                match std::fs::remove_file(&path) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => {
                        return Err(error).with_context(|| format!("remove {}", path.display()))
                    }
                }
            }
            set_value(PHONON_KEY, NO_LAUNCH, REG_DWORD, &1u32.to_le_bytes())?;
        }
        Ok(())
    }
}

#[cfg(windows)]
pub fn enabled() -> bool {
    win::enabled()
}

#[cfg(windows)]
pub fn set(on: bool) -> Result<()> {
    win::set(on)?;
    let _ = crate::settings::set(crate::settings::LAUNCH_AT_LOGIN, on.into());
    Ok(())
}

#[cfg(not(any(windows, target_os = "linux")))]
pub fn enabled() -> bool {
    false
}

#[cfg(not(any(windows, target_os = "linux")))]
pub fn set(_on: bool) -> Result<()> {
    anyhow::bail!("launch at login is set by the Mac app on this platform")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("phonon-autostart-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    const PACKAGED: &str = include_str!("../linux/phonon.desktop");

    #[test]
    fn the_packaged_entry_starts_phonon_and_a_user_entry_overrides_it() {
        let dir = scratch("deb");
        let system = dir.join("etc/xdg/autostart").join(DESKTOP_FILE);
        let user = dir.join("home/.config/autostart").join(DESKTOP_FILE);
        std::fs::create_dir_all(system.parent().unwrap()).unwrap();
        std::fs::write(&system, PACKAGED).unwrap();
        assert!(
            desktop_enabled(&system, &user),
            "the .deb starts Phonon at login"
        );

        desktop_set(&system, &user, false, "/usr/bin/phonon").unwrap();
        assert!(!desktop_enabled(&system, &user));
        let text = std::fs::read_to_string(&user).unwrap();
        assert!(text.contains("Hidden=true"), "{text}");
        assert_eq!(
            std::fs::read_to_string(&system).unwrap(),
            PACKAGED,
            "the system file is not touched"
        );

        desktop_set(&system, &user, true, "/usr/bin/phonon").unwrap();
        assert!(desktop_enabled(&system, &user));
        assert!(
            !user.exists(),
            "on again: the override goes, the package's entry applies"
        );
        // turning it on twice is fine
        desktop_set(&system, &user, true, "/usr/bin/phonon").unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn without_the_package_on_writes_an_entry_for_this_executable() {
        let dir = scratch("bare");
        let system = dir.join("none").join(DESKTOP_FILE);
        let user = dir.join("autostart").join(DESKTOP_FILE);
        assert!(!desktop_enabled(&system, &user));
        desktop_set(&system, &user, true, "/opt/phonon/phonon-win").unwrap();
        assert!(desktop_enabled(&system, &user));
        let text = std::fs::read_to_string(&user).unwrap();
        assert!(text.contains("Exec=/opt/phonon/phonon-win run"), "{text}");
        desktop_set(&system, &user, false, "/opt/phonon/phonon-win").unwrap();
        assert!(!desktop_enabled(&system, &user));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gnomes_switch_counts_as_off() {
        assert!(desktop_disabled(
            "[Desktop Entry]\nX-GNOME-Autostart-enabled=false\n"
        ));
        assert!(desktop_disabled("[Desktop Entry]\nHidden = True\n"));
        assert!(!desktop_disabled(PACKAGED));
    }
}
