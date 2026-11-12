//! One Phonon per data root. Two running copies both hook the dictation key and
//! both transcribe, and on Windows 10 (22H2 VM, 2026-11-10) neither inserted the
//! text. The second launch finds the first through a lock held for the life of
//! the process (a named mutex on Windows, an flock on Linux), says so and exits.
//! The operating system drops the lock when the process ends, however it ends.

use std::path::Path;

use anyhow::Result;

/// Held for the life of the tray app.
pub struct Guard {
    #[cfg(target_os = "linux")]
    _file: std::fs::File,
    #[cfg(windows)]
    _handle: isize,
}

/// Claim the single-instance lock for the data root `root`. `Ok(None)` means
/// another Phonon already holds it.
pub fn claim(root: &Path) -> Result<Option<Guard>> {
    claim_impl(root, Kind::App)
}

/// Claim the settings window's own lock: one window per data root, apart from
/// the tray app's lock.
pub fn claim_settings_window(root: &Path) -> Result<Option<Guard>> {
    claim_impl(root, Kind::SettingsWindow)
}

#[derive(Clone, Copy)]
enum Kind {
    App,
    SettingsWindow,
}

impl Kind {
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    fn lock_file(self) -> &'static str {
        match self {
            Kind::App => "phonon.lock",
            Kind::SettingsWindow => "settings-window.lock",
        }
    }

    #[cfg_attr(not(any(windows, test)), allow(dead_code))]
    fn mutex_prefix(self) -> &'static str {
        match self {
            Kind::App => "single-instance",
            Kind::SettingsWindow => "settings-window",
        }
    }
}

#[cfg(target_os = "linux")]
fn claim_impl(root: &Path, kind: Kind) -> Result<Option<Guard>> {
    use anyhow::Context;
    use std::os::fd::AsRawFd;
    std::fs::create_dir_all(root).with_context(|| format!("create {}", root.display()))?;
    let path = root.join(kind.lock_file());
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&path)
        .with_context(|| format!("open {}", path.display()))?;
    // SAFETY: flock on a descriptor this function owns.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
        return Ok(Some(Guard { _file: file }));
    }
    let error = std::io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::EWOULDBLOCK) {
        Ok(None)
    } else {
        Err(error).with_context(|| format!("lock {}", path.display()))
    }
}

/// The mutex name for a data root: per session ("Local\"), one per data root so a
/// test copy with its own PHONON_WIN_HOME does not collide with the user's.
#[cfg(test)]
fn mutex_name(root: &Path) -> String {
    mutex_name_for(root, Kind::App)
}

/// The mutex name for one kind of lock in a data root.
#[cfg(any(windows, test))]
fn mutex_name_for(root: &Path, kind: Kind) -> String {
    let tail: String = root
        .to_string_lossy()
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let tail = &tail[tail.len().saturating_sub(180)..];
    format!("Local\\Phonon.{}.{tail}", kind.mutex_prefix())
}

#[cfg(windows)]
fn claim_impl(root: &Path, kind: Kind) -> Result<Option<Guard>> {
    use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, ERROR_ALREADY_EXISTS};
    use windows_sys::Win32::System::Threading::CreateMutexW;
    let name: Vec<u16> = mutex_name_for(root, kind)
        .encode_utf16()
        .chain([0])
        .collect();
    // SAFETY: a NUL-terminated name; the handle is closed or kept below.
    let handle = unsafe { CreateMutexW(std::ptr::null(), 0, name.as_ptr()) };
    if handle.is_null() {
        anyhow::bail!(
            "create the single-instance mutex: {}",
            std::io::Error::last_os_error()
        );
    }
    if unsafe { GetLastError() } == ERROR_ALREADY_EXISTS {
        unsafe { CloseHandle(handle) };
        return Ok(None);
    }
    Ok(Some(Guard {
        _handle: handle as isize,
    }))
}

#[cfg(not(any(windows, target_os = "linux")))]
fn claim_impl(_root: &Path, _kind: Kind) -> Result<Option<Guard>> {
    Ok(Some(Guard {}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn a_second_claim_on_the_same_root_is_refused_until_the_first_ends() {
        let dir = std::env::temp_dir().join(format!("phonon-single-{}", std::process::id()));
        let first = claim(&dir).unwrap();
        assert!(first.is_some());
        // a second open of the lock file is a separate open file description,
        // which flock treats like another process
        assert!(claim(&dir).unwrap().is_none());
        let other = dir.join("other-root");
        assert!(
            claim(&other).unwrap().is_some(),
            "another data root is independent"
        );
        drop(first);
        assert!(
            claim(&dir).unwrap().is_some(),
            "the lock goes with the guard"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The settings window has its own lock: it opens while the tray app runs,
    /// and only one window opens per data root.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_settings_window_lock_is_separate_from_the_apps() {
        let dir = std::env::temp_dir().join(format!("phonon-single-w-{}", std::process::id()));
        let app = claim(&dir).unwrap();
        assert!(app.is_some());
        let window = claim_settings_window(&dir).unwrap();
        assert!(window.is_some(), "the tray app does not block the window");
        assert!(claim_settings_window(&dir).unwrap().is_none());
        drop(window);
        assert!(claim_settings_window(&dir).unwrap().is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_settings_window_mutex_differs_from_the_apps() {
        let root = Path::new(r"C:\Users\qa\AppData\Local\Phonon");
        assert_eq!(mutex_name(root), mutex_name_for(root, Kind::App));
        assert_ne!(
            mutex_name_for(root, Kind::SettingsWindow),
            mutex_name_for(root, Kind::App)
        );
    }

    #[test]
    fn mutex_names_are_per_root_and_legal() {
        let a = mutex_name(Path::new(r"C:\Users\qa\AppData\Local\Phonon"));
        let b = mutex_name(Path::new(r"C:\Users\qa\phonon-test"));
        assert_ne!(a, b);
        assert!(a.starts_with("Local\\Phonon.single-instance."));
        // no backslash after the namespace, and short enough for MAX_PATH
        assert!(!a["Local\\".len()..].contains('\\'));
        assert!(a.len() < 260);
        let long = mutex_name(Path::new(&"x".repeat(1000)));
        assert!(long.len() < 260);
        assert_eq!(
            mutex_name(Path::new(r"C:\USERS\QA")),
            mutex_name(Path::new(r"c:\users\qa"))
        );
    }
}
