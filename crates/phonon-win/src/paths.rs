//! Where Phonon keeps runtimes, weights, recordings, and logs on Windows.
//!
//! Everything lives under one directory so a user can delete it and start over.
//! `PHONON_WIN_HOME` overrides it; continuous integration uses that to put the
//! cache on the runner's fast disk.

use std::path::PathBuf;

/// The Phonon data root.
pub fn data_root() -> PathBuf {
    if let Some(explicit) = std::env::var_os("PHONON_WIN_HOME") {
        return PathBuf::from(explicit);
    }
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

/// Captured audio. Every pass keeps its own file.
pub fn recordings() -> PathBuf {
    data_root().join("recordings")
}

#[cfg(test)]
mod tests {
    use super::*;

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
