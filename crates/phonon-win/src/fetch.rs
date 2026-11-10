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

    // file:// serves locally built backends (CI, the selftest); it is checked like a download
    let (mut reader, declared): (Box<dyn Read>, u64) =
        if let Some(path) = asset.url.strip_prefix("file://") {
            let file = fs::File::open(path).with_context(|| format!("open {path}"))?;
            let length = file.metadata()?.len();
            (Box::new(file), length)
        } else {
            let response = http::agent()
                .get(asset.url)
                .call()
                .with_context(|| format!("download {}", asset.url))?;
            let declared = response
                .header("Content-Length")
                .and_then(|value| value.parse().ok())
                .unwrap_or(asset.bytes);
            (response.into_reader(), declared)
        };
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

/// Expand a downloaded asset in-process, choosing its format by name.
fn expand(archive: &Path, into: &Path) -> Result<()> {
    fs::create_dir_all(into)?;
    let file = fs::File::open(archive).with_context(|| format!("open {}", archive.display()))?;
    let name = archive.file_name().unwrap_or_default().to_string_lossy();
    let result = if name.ends_with(".tar.bz2") {
        tar::Archive::new(bzip2::read::MultiBzDecoder::new(file))
            .unpack(into)
            .map_err(anyhow::Error::from)
    } else if name.ends_with(".tar.gz") {
        tar::Archive::new(flate2::read::MultiGzDecoder::new(file))
            .unpack(into)
            .map_err(anyhow::Error::from)
    } else if name.ends_with(".tar") {
        tar::Archive::new(file)
            .unpack(into)
            .map_err(anyhow::Error::from)
    } else if name.ends_with(".zip") {
        // extract validates paths (including symlink destinations) and restores
        // each file's Unix permissions when the entry supplies them.
        zip::ZipArchive::new(file)?
            .extract(into)
            .map_err(anyhow::Error::from)
    } else {
        bail!("unsupported archive format: {}", archive.display());
    };
    result.with_context(|| format!("unpack {}", archive.display()))
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
            // `downloads/` and the component live under the same data root, so a
            // rename moves the weights instead of keeping a second 3 GB copy.
            for (asset, path) in &downloaded {
                fs::rename(path, staging.join(asset.name))
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
    // The component is in place; its verified archives are only dead weight now.
    for (_, path) in &downloaded {
        let _ = fs::remove_file(path);
    }
    Ok(final_dir)
}

/// Remove verified downloads left behind by an install that predates their cleanup.
/// Only names the manifest knows are touched, and only once their component is
/// installed, so an interrupted first run still resumes without paying again.
fn remove_installed_downloads() {
    for component in crate::manifest::ALL {
        if !installed(component) {
            continue;
        }
        for asset in component.assets {
            let _ = fs::remove_file(paths::downloads().join(asset.name));
        }
    }
}

/// Fetch and install everything first run needs.
pub fn ensure_all(on_progress: &mut dyn FnMut(Progress)) -> Result<()> {
    #[cfg(all(target_os = "linux", not(target_arch = "x86_64")))]
    bail!("the prebuilt Linux runtimes require x86_64");
    for component in crate::manifest::ALL {
        ensure(component, on_progress)?;
    }
    remove_installed_downloads();
    // Folders an earlier version installed and this one no longer reads.
    for dir in crate::manifest::SUPERSEDED {
        let path = paths::data_root().join(dir);
        if path.is_dir() {
            let _ = fs::remove_dir_all(&path);
        }
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

    /// Unpacking preserves the runtime layout, executable modes, and library links.
    #[test]
    fn expands_an_archive_and_strips_its_root() {
        let dir = std::env::temp_dir().join("phonon-win-expand-test");
        let _ = fs::remove_dir_all(&dir);
        let source = dir.join("src/wrapper/bin");
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("tool"), b"binary").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::{symlink, PermissionsExt};
            fs::set_permissions(source.join("tool"), fs::Permissions::from_mode(0o755)).unwrap();
            symlink("tool", source.join("tool-link")).unwrap();
        }
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
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_ne!(
                fs::metadata(flat.join("bin/tool"))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o111,
                0
            );
            assert_eq!(
                fs::read_link(flat.join("bin/tool-link")).unwrap(),
                Path::new("tool")
            );
        }
        let _ = fs::remove_dir_all(&dir);
    }

    fn write_archive_fixture(path: &Path, entry_name: &str) {
        let file = fs::File::create(path).unwrap();
        let name = path.file_name().unwrap().to_str().unwrap();
        if name.ends_with(".zip") {
            let mut archive = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated)
                .unix_permissions(0o755);
            archive.start_file(entry_name, options).unwrap();
            archive.write_all(b"binary").unwrap();
            archive.finish().unwrap();
        } else {
            let mut archive = tar::Builder::new(Vec::new());
            let mut header = tar::Header::new_gnu();
            header.set_size(6);
            header.set_mode(0o755);
            // Write the name directly so the fixture can include a malicious
            // path that the tar builder's set_path would refuse to create.
            header.as_mut_bytes()[..entry_name.len()].copy_from_slice(entry_name.as_bytes());
            header.set_cksum();
            archive.append(&header, &b"binary"[..]).unwrap();
            let bytes = archive.into_inner().unwrap();
            if name.ends_with(".tar.bz2") {
                let mut encoder = bzip2::write::BzEncoder::new(file, bzip2::Compression::default());
                encoder.write_all(&bytes).unwrap();
                encoder.finish().unwrap();
            } else {
                let mut encoder =
                    flate2::write::GzEncoder::new(file, flate2::Compression::default());
                encoder.write_all(&bytes).unwrap();
                encoder.finish().unwrap();
            }
        }
    }

    #[test]
    fn expands_compressed_archives_and_strips_their_roots() {
        let dir = std::env::temp_dir().join("phonon-win-compressed-expand-test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        for suffix in ["tar.bz2", "tar.gz", "zip"] {
            let archive = dir.join(format!("bundle.{suffix}"));
            write_archive_fixture(&archive, "wrapper/bin/tool");
            let raw = dir.join(format!("{suffix}/.raw"));
            expand(&archive, &raw).unwrap();
            let flat = dir.join(format!("{suffix}/installed"));
            let strip_to = "wrapper";
            move_children(&raw.join(strip_to), &flat).unwrap();
            assert_eq!(fs::read(flat.join("bin/tool")).unwrap(), b"binary");
            assert!(!flat.join("wrapper").exists());
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                assert_eq!(
                    fs::metadata(flat.join("bin/tool"))
                        .unwrap()
                        .permissions()
                        .mode()
                        & 0o777,
                    0o755
                );
            }
        }
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rejects_archive_path_traversal() {
        let dir = std::env::temp_dir().join("phonon-win-traversal-test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        for suffix in ["tar.bz2", "tar.gz", "zip"] {
            let archive = dir.join(format!("bundle.{suffix}"));
            write_archive_fixture(&archive, "../escaped");
            let out = dir.join("out");
            let result = expand(&archive, &out);
            if suffix == "zip" {
                assert!(result.is_err(), "zip traversal must fail extraction");
            } else {
                // tar rejects traversal entries by skipping them.
                result.unwrap();
            }
            assert!(!dir.join("escaped").exists());
            assert_eq!(fs::read_dir(&out).unwrap().count(), 0);
        }
        fs::remove_dir_all(&dir).unwrap();
    }
}
