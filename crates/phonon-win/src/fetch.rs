//! First-run download: fetch, verify, unpack, install.
//!
//! Rules that make this safe to run again after a failure:
//! - A file is written to `downloads/` first and only moved into place once its
//!   SHA-256 matches the manifest.
//! - A component is installed into a temporary directory and renamed at the end,
//!   so a half-written component never looks installed.
//! - An interrupted run leaves the finished components alone.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use sha2::{Digest, Sha256};

use crate::http;
use crate::manifest::{human_bytes, Asset, Component, Unpack};
use crate::paths;

/// Progress the caller shows to the user.
#[derive(Debug, Clone)]
pub struct Progress {
    pub label: &'static str,
    pub file: &'static str,
    pub done_bytes: u64,
    pub total_bytes: u64,
}

impl Progress {
    /// Whole percent of this file, clamped to 100.
    pub fn percent(&self) -> u32 {
        if self.total_bytes == 0 {
            return 100;
        }
        let pct = self.done_bytes.saturating_mul(100) / self.total_bytes;
        pct.min(100) as u32
    }

    /// One short line for a tray tooltip. A component with several files names
    /// the one in flight, so a stall is attributable.
    pub fn line(&self) -> String {
        format!(
            "{} {}% ({} of {}, {})",
            self.label,
            self.percent(),
            human_bytes(self.done_bytes),
            human_bytes(self.total_bytes),
            self.file
        )
    }
}

/// Where a component ends up once it is installed.
pub fn component_dir(component: &Component) -> PathBuf {
    paths::data_root().join(component.dir)
}

/// The component's sentinel file. Callers run tools from this path.
pub fn sentinel_path(component: &Component) -> PathBuf {
    component_dir(component).join(component.sentinel)
}

/// Whether the component is already installed.
pub fn installed(component: &Component) -> bool {
    sentinel_path(component).is_file()
}

/// SHA-256 of a file, lowercase hex.
pub fn file_sha256(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// Download one asset into `downloads/` and verify it. A file already there with
/// the right hash is reused, so a retry does not pay for the bytes again.
fn fetch_asset(
    component: &Component,
    asset: &Asset,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<PathBuf> {
    let dir = paths::downloads();
    fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    let target = dir.join(asset.name);

    if target.is_file() {
        if file_sha256(&target)? == asset.sha256 {
            on_progress(Progress {
                label: component.label,
                file: asset.name,
                done_bytes: asset.bytes,
                total_bytes: asset.bytes,
            });
            return Ok(target);
        }
        // A truncated or stale file is worth nothing; take the bytes again.
        let _ = fs::remove_file(&target);
    }

    let partial = dir.join(format!("{}.part", asset.name));
    let _ = fs::remove_file(&partial);

    let response = http::agent()
        .get(asset.url)
        .call()
        .with_context(|| format!("download {}", asset.url))?;
    let declared: u64 = response
        .header("Content-Length")
        .and_then(|value| value.parse().ok())
        .unwrap_or(asset.bytes);

    let mut reader = response.into_reader();
    let mut file =
        fs::File::create(&partial).with_context(|| format!("create {}", partial.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    let mut done: u64 = 0;
    let mut last_report: u64 = 0;
    loop {
        let read = reader.read(&mut buffer).context("read download body")?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        file.write_all(&buffer[..read])
            .with_context(|| format!("write {}", partial.display()))?;
        done += read as u64;
        // Report about every 4 MB. A per-chunk callback would spend more time in
        // the tray than in the socket.
        if done - last_report >= 4 << 20 || done == declared {
            last_report = done;
            on_progress(Progress {
                label: component.label,
                file: asset.name,
                done_bytes: done,
                total_bytes: declared.max(done),
            });
        }
    }
    file.flush()?;
    drop(file);

    let actual = hex(&hasher.finalize());
    if actual != asset.sha256 {
        let _ = fs::remove_file(&partial);
        bail!(
            "{} failed verification: expected sha256 {}, got {}",
            asset.name,
            asset.sha256,
            actual
        );
    }
    fs::rename(&partial, &target).with_context(|| format!("install {}", target.display()))?;
    Ok(target)
}

/// Expand an archive with the `tar` that ships with Windows 10 and 11. It reads
/// both zip and bzip2 archives, which is every archive in the manifest.
fn expand(archive: &Path, into: &Path) -> Result<()> {
    fs::create_dir_all(into)?;
    let output = std::process::Command::new("tar")
        .arg("-xf")
        .arg(archive)
        .arg("-C")
        .arg(into)
        .output()
        .context("run tar; Windows 10 build 17063 or later provides it")?;
    if !output.status.success() {
        bail!(
            "tar failed on {}: {}",
            archive.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

/// Move every entry of `from` into `to`.
fn move_children(from: &Path, to: &Path) -> Result<()> {
    fs::create_dir_all(to)?;
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        let target = to.join(entry.file_name());
        fs::rename(entry.path(), &target)
            .with_context(|| format!("move into {}", target.display()))?;
    }
    Ok(())
}

/// Fetch and install one component. Returns immediately when it is already there.
pub fn ensure(component: &Component, on_progress: &mut dyn FnMut(Progress)) -> Result<PathBuf> {
    let final_dir = component_dir(component);
    if installed(component) {
        return Ok(final_dir);
    }

    let mut downloaded = Vec::new();
    for asset in component.assets {
        downloaded.push((asset, fetch_asset(component, asset, on_progress)?));
    }

    let parent = final_dir
        .parent()
        .ok_or_else(|| anyhow!("{} has no parent", final_dir.display()))?;
    fs::create_dir_all(parent)?;
    let staging = parent.join(format!(
        ".{}.staging",
        final_dir.file_name().unwrap_or_default().to_string_lossy()
    ));
    let _ = fs::remove_dir_all(&staging);
    fs::create_dir_all(&staging)?;

    match component.unpack {
        Unpack::None => {
            for (asset, path) in &downloaded {
                fs::copy(path, staging.join(asset.name))
                    .with_context(|| format!("place {}", asset.name))?;
            }
        }
        Unpack::Archive { strip_to } => {
            let raw = staging.join(".raw");
            for (_, path) in &downloaded {
                expand(path, &raw)?;
            }
            let root = if strip_to.is_empty() {
                raw.clone()
            } else {
                raw.join(strip_to)
            };
            if !root.is_dir() {
                bail!("{} is not in the archive", root.display());
            }
            move_children(&root, &staging)?;
            let _ = fs::remove_dir_all(&raw);
        }
    }

    if !staging.join(component.sentinel).is_file() {
        bail!(
            "{} is missing after install; expected {}",
            component.label,
            component.sentinel
        );
    }
    // A previous partial install could still be sitting there.
    let _ = fs::remove_dir_all(&final_dir);
    fs::rename(&staging, &final_dir)
        .with_context(|| format!("install into {}", final_dir.display()))?;
    Ok(final_dir)
}

/// Fetch and install everything first run needs.
pub fn ensure_all(on_progress: &mut dyn FnMut(Progress)) -> Result<()> {
    for component in crate::manifest::ALL {
        ensure(component, on_progress)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashes_a_known_file() {
        let dir = std::env::temp_dir().join("phonon-win-hash-test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("abc.txt");
        fs::write(&path, b"abc").unwrap();
        assert_eq!(
            file_sha256(&path).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn percent_never_exceeds_one_hundred() {
        let progress = Progress {
            label: "test",
            file: "f",
            done_bytes: 300,
            total_bytes: 100,
        };
        assert_eq!(progress.percent(), 100);
        let empty = Progress {
            label: "test",
            file: "f",
            done_bytes: 0,
            total_bytes: 0,
        };
        assert_eq!(empty.percent(), 100);
    }

    #[test]
    fn progress_line_reads_naturally() {
        let progress = Progress {
            label: "correction model",
            file: "model.gguf",
            done_bytes: 1_073_741_824,
            total_bytes: 3_221_225_472,
        };
        assert_eq!(
            progress.line(),
            "correction model 33% (1.0 GB of 3.0 GB, model.gguf)"
        );
    }

    /// The unpack path uses the system `tar`, so it can be proved off Windows too.
    #[test]
    fn expands_an_archive_and_strips_its_root() {
        let dir = std::env::temp_dir().join("phonon-win-expand-test");
        let _ = fs::remove_dir_all(&dir);
        let source = dir.join("src/wrapper/bin");
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("tool"), b"binary").unwrap();
        let archive = dir.join("bundle.tar");
        let status = std::process::Command::new("tar")
            .arg("-cf")
            .arg(&archive)
            .arg("-C")
            .arg(dir.join("src"))
            .arg("wrapper")
            .status()
            .unwrap();
        assert!(status.success());

        let out = dir.join("out");
        expand(&archive, &out).unwrap();
        assert!(out.join("wrapper/bin/tool").is_file());

        let flat = dir.join("flat");
        move_children(&out.join("wrapper"), &flat).unwrap();
        assert!(flat.join("bin/tool").is_file());
        let _ = fs::remove_dir_all(&dir);
    }
}
