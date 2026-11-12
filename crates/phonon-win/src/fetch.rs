//! First-run download: fetch, verify, unpack, install.
//!
//! Rules that make this safe to run again after a failure:
//! - A file is written to `downloads/` first and only moved into place once its
//!   SHA-256 matches the manifest.
//! - A component is installed into a temporary directory and renamed at the end,
//!   so a half-written component never looks installed.
//! - An interrupted run leaves the finished components alone.

use std::fs;
use std::io::{self, Read, Write};
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
    let mut hasher = Sha256::new();
    hash_file_into(&mut hasher, path)?;
    Ok(hex(&hasher.finalize()))
}

fn hash_file_into(hasher: &mut Sha256, path: &Path) -> Result<()> {
    let mut file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(())
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
    download(
        &paths::downloads(),
        asset,
        &mut |done_bytes, total_bytes| {
            on_progress(Progress {
                label: component.label,
                file: asset.name,
                done_bytes,
                total_bytes,
            })
        },
    )
}

/// Fetch `asset` into `dir`. An interrupted transfer leaves `<name>.part` behind and
/// the next call asks the server for the rest. A resumed file that fails the hash is
/// thrown away and fetched once more from zero.
fn download(dir: &Path, asset: &Asset, report: &mut dyn FnMut(u64, u64)) -> Result<PathBuf> {
    fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let target = dir.join(asset.name);

    if target.is_file() {
        if file_sha256(&target)? == asset.sha256 {
            report(asset.bytes, asset.bytes);
            return Ok(target);
        }
        // A truncated or stale file is worth nothing; take the bytes again.
        let _ = fs::remove_file(&target);
    }

    let partial = dir.join(format!("{}.part", asset.name));
    loop {
        let mut have = fs::metadata(&partial).map(|meta| meta.len()).unwrap_or(0);
        if have > asset.bytes {
            let _ = fs::remove_file(&partial);
            have = 0;
        }
        let (actual, resumed) = receive(asset, &partial, have, report)?;
        if actual == asset.sha256 {
            fs::rename(&partial, &target)
                .with_context(|| format!("install {}", target.display()))?;
            return Ok(target);
        }
        let _ = fs::remove_file(&partial);
        if !resumed {
            bail!(
                "{} failed verification: expected sha256 {}, got {}",
                asset.name,
                asset.sha256,
                actual
            );
        }
    }
}

/// Bring `partial` up to the full asset and return its SHA-256, plus whether the
/// bytes already on disk were kept. `have` is the length of `partial` now.
fn receive(
    asset: &Asset,
    partial: &Path,
    have: u64,
    report: &mut dyn FnMut(u64, u64),
) -> Result<(String, bool)> {
    let mut hasher = Sha256::new();
    if have > 0 && have == asset.bytes {
        hash_file_into(&mut hasher, partial)?;
        return Ok((hex(&hasher.finalize()), true));
    }
    let (mut reader, start, declared) = open_body(asset, have)?;
    let mut file = if start > 0 {
        hash_file_into(&mut hasher, partial)?;
        fs::OpenOptions::new().append(true).open(partial)
    } else {
        fs::File::create(partial)
    }
    .with_context(|| format!("open {}", partial.display()))?;
    let mut buffer = vec![0u8; 1 << 20];
    let mut done = start;
    let mut last_report = start;
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
            report(done, declared.max(done));
        }
    }
    file.flush()?;
    Ok((hex(&hasher.finalize()), start > 0))
}

/// Open the asset's bytes from `offset`. Returns the reader, the offset its first
/// byte belongs at (0 when the source sends the whole file), and the full length.
fn open_body(asset: &Asset, offset: u64) -> Result<(Box<dyn Read>, u64, u64)> {
    // file:// serves locally built backends (CI, the selftest); it is checked like a download
    if let Some(path) = asset.url.strip_prefix("file://") {
        let file = fs::File::open(path).with_context(|| format!("open {path}"))?;
        let length = file.metadata()?.len();
        return Ok((Box::new(file), 0, length));
    }
    let mut request = http::agent().get(asset.url);
    if offset > 0 {
        request = request.set("Range", &format!("bytes={offset}-"));
    }
    let response = match request.call() {
        Ok(response) => response,
        // The server cannot serve the rest of this partial; take the whole file.
        Err(ureq::Error::Status(416, _)) if offset > 0 => return open_body(asset, 0),
        Err(error) => return Err(error).with_context(|| format!("download {}", asset.url)),
    };
    if offset > 0 && response.status() == 206 {
        let range = response
            .header("Content-Range")
            .and_then(parse_content_range);
        if range == Some((offset, asset.bytes - 1, asset.bytes)) {
            return Ok((response.into_reader(), offset, asset.bytes));
        }
        // Not the range we asked for: start over rather than splice wrong bytes.
        return open_body(asset, 0);
    }
    let declared = response
        .header("Content-Length")
        .and_then(|value| value.parse().ok())
        .unwrap_or(asset.bytes);
    Ok((response.into_reader(), 0, declared))
}

/// `bytes FIRST-LAST/TOTAL` as numbers; anything else (including an unknown total) is None.
fn parse_content_range(value: &str) -> Option<(u64, u64, u64)> {
    let (span, total) = value.trim().strip_prefix("bytes ")?.split_once('/')?;
    let (first, last) = span.split_once('-')?;
    Some((
        first.trim().parse().ok()?,
        last.trim().parse().ok()?,
        total.trim().parse().ok()?,
    ))
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
    } else if name.ends_with(".tar.xz") {
        // lzma-rs decodes into a writer, not a reader; a backend bundle is tens of MB, so the
        // tar is held in memory for the moment it takes to unpack.
        let mut tarball = Vec::new();
        lzma_rs::xz_decompress(&mut io::BufReader::new(file), &mut tarball)
            .map_err(|error| anyhow!("xz: {error}"))
            .and_then(|()| {
                tar::Archive::new(tarball.as_slice())
                    .unpack(into)
                    .map_err(anyhow::Error::from)
            })
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

    /// How the test server answers a Range request.
    #[derive(Clone, Copy)]
    enum Ranges {
        Honor,
        Ignore,
        WrongStart,
    }

    /// Serve `body` over HTTP on 127.0.0.1 and record each request's Range header.
    fn serve(
        body: Vec<u8>,
        ranges: Ranges,
    ) -> (
        String,
        std::sync::Arc<std::sync::Mutex<Vec<Option<String>>>>,
    ) {
        use std::io::{BufRead, BufReader};
        use std::sync::{Arc, Mutex};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/asset.bin", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = seen.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let mut stream = stream.unwrap();
                let mut range = None;
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap() == 0 || line == "\r\n" {
                        break;
                    }
                    if let Some(value) = line.to_ascii_lowercase().strip_prefix("range: bytes=") {
                        range = Some(value.trim().trim_end_matches('-').to_string());
                    }
                }
                log.lock().unwrap().push(range.clone());
                let total = body.len();
                let start = match (range.and_then(|r| r.parse::<usize>().ok()), ranges) {
                    (Some(start), Ranges::Honor) => Some(start),
                    (Some(start), Ranges::WrongStart) => Some(start - 1),
                    _ => None,
                };
                let head = match start {
                    Some(start) => format!(
                        "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {start}-{}/{total}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        total - 1,
                        total - start
                    ),
                    None => format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {total}\r\nConnection: close\r\n\r\n"
                    ),
                };
                stream.write_all(head.as_bytes()).unwrap();
                stream.write_all(&body[start.unwrap_or(0)..]).unwrap();
            }
        });
        (url, seen)
    }

    fn asset_for(url: String, body: &[u8]) -> Asset {
        let mut hasher = Sha256::new();
        hasher.update(body);
        Asset {
            name: "asset.bin",
            url: Box::leak(url.into_boxed_str()),
            sha256: Box::leak(hex(&hasher.finalize()).into_boxed_str()),
            bytes: body.len() as u64,
        }
    }

    fn fixture_body() -> Vec<u8> {
        (0..100_000u32).map(|i| (i % 251) as u8).collect()
    }

    fn fresh_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(name);
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn resumes_a_partial_with_a_range_request() {
        let body = fixture_body();
        let (url, seen) = serve(body.clone(), Ranges::Honor);
        let asset = asset_for(url, &body);
        let dir = fresh_dir("phonon-win-resume-test");
        fs::write(dir.join("asset.bin.part"), &body[..40_000]).unwrap();
        let mut first_report = None;
        let path = download(&dir, &asset, &mut |done, _| {
            first_report.get_or_insert(done);
        })
        .unwrap();
        assert_eq!(fs::read(&path).unwrap(), body);
        assert_eq!(*seen.lock().unwrap(), vec![Some("40000".to_string())]);
        assert!(first_report.unwrap() > 40_000);
        assert!(!dir.join("asset.bin.part").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn restarts_when_the_server_ignores_range() {
        let body = fixture_body();
        let (url, seen) = serve(body.clone(), Ranges::Ignore);
        let asset = asset_for(url, &body);
        let dir = fresh_dir("phonon-win-ignore-range-test");
        fs::write(dir.join("asset.bin.part"), &body[..40_000]).unwrap();
        let path = download(&dir, &asset, &mut |_, _| {}).unwrap();
        assert_eq!(fs::read(&path).unwrap(), body);
        assert_eq!(*seen.lock().unwrap(), vec![Some("40000".to_string())]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn restarts_when_the_content_range_is_wrong() {
        let body = fixture_body();
        let (url, seen) = serve(body.clone(), Ranges::WrongStart);
        let asset = asset_for(url, &body);
        let dir = fresh_dir("phonon-win-wrong-range-test");
        fs::write(dir.join("asset.bin.part"), &body[..40_000]).unwrap();
        let path = download(&dir, &asset, &mut |_, _| {}).unwrap();
        assert_eq!(fs::read(&path).unwrap(), body);
        assert_eq!(*seen.lock().unwrap(), vec![Some("40000".to_string()), None]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_partial_longer_than_the_asset_starts_over() {
        let body = fixture_body();
        let (url, seen) = serve(body.clone(), Ranges::Honor);
        let asset = asset_for(url, &body);
        let dir = fresh_dir("phonon-win-long-partial-test");
        fs::write(dir.join("asset.bin.part"), vec![0u8; body.len() + 10]).unwrap();
        let path = download(&dir, &asset, &mut |_, _| {}).unwrap();
        assert_eq!(fs::read(&path).unwrap(), body);
        assert_eq!(*seen.lock().unwrap(), vec![None]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_corrupt_partial_fails_the_hash_and_is_fetched_again_from_zero() {
        let body = fixture_body();
        let (url, seen) = serve(body.clone(), Ranges::Honor);
        let asset = asset_for(url, &body);
        let dir = fresh_dir("phonon-win-corrupt-partial-test");
        fs::write(dir.join("asset.bin.part"), vec![0xffu8; 40_000]).unwrap();
        let path = download(&dir, &asset, &mut |_, _| {}).unwrap();
        assert_eq!(fs::read(&path).unwrap(), body);
        assert_eq!(*seen.lock().unwrap(), vec![Some("40000".to_string()), None]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_wrong_file_fails_after_one_fresh_retry() {
        let body = fixture_body();
        let (url, seen) = serve(body.clone(), Ranges::Honor);
        let mut asset = asset_for(url, &body);
        asset.sha256 = "0000000000000000000000000000000000000000000000000000000000000000";
        let dir = fresh_dir("phonon-win-wrong-hash-test");
        fs::write(dir.join("asset.bin.part"), &body[..40_000]).unwrap();
        let error = download(&dir, &asset, &mut |_, _| {}).unwrap_err();
        assert!(error.to_string().contains("failed verification"));
        assert_eq!(*seen.lock().unwrap(), vec![Some("40000".to_string()), None]);
        assert!(!dir.join("asset.bin.part").exists());
        assert!(!dir.join("asset.bin").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn file_urls_still_download_over_a_stale_partial() {
        let body = fixture_body();
        let dir = fresh_dir("phonon-win-file-url-test");
        let source = dir.join("source.bin");
        fs::write(&source, &body).unwrap();
        let asset = asset_for(format!("file://{}", source.display()), &body);
        let downloads = dir.join("downloads");
        fs::create_dir_all(&downloads).unwrap();
        fs::write(downloads.join("asset.bin.part"), vec![0xffu8; 10]).unwrap();
        let path = download(&downloads, &asset, &mut |_, _| {}).unwrap();
        assert_eq!(fs::read(&path).unwrap(), body);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn parses_content_range() {
        assert_eq!(
            parse_content_range("bytes 40000-99999/100000"),
            Some((40_000, 99_999, 100_000))
        );
        assert_eq!(parse_content_range("bytes 0-9/*"), None);
        assert_eq!(parse_content_range("items 0-9/10"), None);
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
            } else if name.ends_with(".tar.xz") {
                let mut file = file;
                lzma_rs::xz_compress(&mut bytes.as_slice(), &mut file).unwrap();
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
        for suffix in ["tar.bz2", "tar.xz", "tar.gz", "zip"] {
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
        for suffix in ["tar.bz2", "tar.xz", "tar.gz", "zip"] {
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
