//! Where Phonon keeps runtimes, weights, recordings, and logs.
//!
//! Everything lives under one directory so a user can delete it and start over.
//! `PHONON_DATA_DIR` (set and not blank) overrides it, the same variable the Mac
//! app and `phonon-core` honour for an isolated profile, tests and demos. Then
//! `PHONON_WIN_HOME`; continuous integration uses that to put the cache on the
//! runner's fast disk.
//! Linux defaults to `$XDG_DATA_HOME/phonon` or `~/.local/share/phonon`.

use std::ffi::OsString;
use std::path::PathBuf;

/// The Phonon data root.
pub fn data_root() -> PathBuf {
    resolve_root(
        std::env::var_os("PHONON_DATA_DIR"),
        std::env::var_os("PHONON_WIN_HOME"),
    )
}

/// `data_root` from the two overrides, without reading the environment.
fn resolve_root(data_dir: Option<OsString>, win_home: Option<OsString>) -> PathBuf {
    let data_dir =
        data_dir.filter(|dir| !dir.is_empty() && dir.to_str().is_none_or(|s| !s.trim().is_empty()));
    if let Some(explicit) = data_dir.or(win_home) {
        return absolute_root(PathBuf::from(explicit));
    }
    default_root()
}

#[cfg(not(target_os = "linux"))]
fn absolute_root(path: PathBuf) -> PathBuf {
    path
}

#[cfg(target_os = "linux")]
fn absolute_root(path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        std::env::current_dir()
            .expect("read the current directory")
            .join(path)
    }
}

#[cfg(target_os = "linux")]
fn default_root() -> PathBuf {
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share")))
        .unwrap_or_else(|| PathBuf::from("."));
    absolute_root(base.join("phonon"))
}

#[cfg(not(target_os = "linux"))]
fn default_root() -> PathBuf {
    let base = std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("Phonon")
}

/// Where a downloaded file waits before it is verified and installed.
pub fn downloads() -> PathBuf {
    data_root().join("downloads")
}

/// The recordings folder's name in the data root.
pub const RECORDINGS: &str = "recordings";

/// Captured audio, one file per dictation. Kept only while local history is on
/// (see `history::keep_wav`).
pub fn recordings() -> PathBuf {
    data_root().join(RECORDINGS)
}

// `win` is compiled only for Windows, but the hook watchdog's decisions are plain
// arithmetic: run their tests on the Mac and Linux too. On Windows they run as
// `win::watchdog::tests`.
#[cfg(all(test, not(windows)))]
#[path = "win/watchdog.rs"]
mod hook_watchdog;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phonon_data_dir_wins_when_set() {
        let some = |s: &str| Some(OsString::from(s));
        assert_eq!(
            resolve_root(some("/tmp/phonon-data"), some("/tmp/phonon-win-home")),
            PathBuf::from("/tmp/phonon-data")
        );
        // Empty or blank means unset: the next override, then the default.
        assert_eq!(
            resolve_root(some(""), some("/tmp/phonon-win-home")),
            PathBuf::from("/tmp/phonon-win-home")
        );
        assert_eq!(
            resolve_root(some("  "), some("/tmp/phonon-win-home")),
            PathBuf::from("/tmp/phonon-win-home")
        );
        assert_eq!(resolve_root(some(""), None), default_root());
        assert_eq!(resolve_root(None, None), default_root());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn relative_roots_survive_changing_the_child_directory() {
        assert_eq!(
            absolute_root(PathBuf::from("models")),
            std::env::current_dir().unwrap().join("models")
        );
    }

    #[test]
    fn the_override_wins() {
        // Set and read in one process; the other tests never look at this value.
        std::env::set_var("PHONON_WIN_HOME", "/tmp/phonon-win-test-root");
        assert_eq!(data_root(), PathBuf::from("/tmp/phonon-win-test-root"));
        assert_eq!(
            downloads(),
            PathBuf::from("/tmp/phonon-win-test-root/downloads")
        );
        std::env::remove_var("PHONON_WIN_HOME");
    }
}
