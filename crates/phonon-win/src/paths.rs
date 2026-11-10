//! Where Phonon keeps runtimes, weights, recordings, and logs.
//!
//! Everything lives under one directory so a user can delete it and start over.
//! `PHONON_WIN_HOME` overrides it; continuous integration uses that to put the
//! cache on the runner's fast disk.
//! Linux defaults to `$XDG_DATA_HOME/phonon` or `~/.local/share/phonon`.

use std::path::PathBuf;

/// The Phonon data root.
pub fn data_root() -> PathBuf {
    if let Some(explicit) = std::env::var_os("PHONON_WIN_HOME") {
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

#[cfg(test)]
mod tests {
    use super::*;

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
