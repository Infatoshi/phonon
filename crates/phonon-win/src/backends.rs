//! Prebuilt, hardware-specific correction runtimes.
//!
//! Every user gets the CPU runtime (`manifest::LLAMA_RUNTIME`). A machine whose GPU
//! has a tuned build on our side also gets that build: a llama.cpp b12132 runtime
//! with Phonon's CUDA patches (`kernels/cuda/merged`, `kernels/cuda/lean`) compiled
//! for one GPU family only. Nothing is compiled on the user's machine.
//!
//! At start:
//! 1. List the CUDA devices exactly as the runtime will see them: a short-lived
//!    child (`phonon-win cuda-devices`) asks the CUDA driver, under this process's
//!    environment, so `CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER` apply to the
//!    list the same way they apply to llama-server. Phonon's own process never
//!    initialises CUDA.
//! 2. Read the backend manifest and check its ed25519 signature against the keys
//!    compiled in below, before parsing it.
//! 3. Pick the device first (the strongest one with a build), then the build for
//!    that device's compute capability.
//! 4. Install it with the same SHA-256-checked download as every other component,
//!    next to any earlier build of the family, which stays until the new one has
//!    passed the startup check (`Installed::confirm`).
//! 5. Start the server pinned to that device by UUID (`CUDA_VISIBLE_DEVICES=GPU-...`
//!    on the child only), and check that it lists exactly that device.
//!
//! Any step that fails leaves the next path in the fallback order (an earlier build
//! of the family, then Vulkan on Windows, then the CPU), so a missing or broken
//! backend never costs the user dictation.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;

use crate::fetch;
use crate::http;
use crate::manifest::{Asset, Component, Unpack};

/// The manifest the app reads when none is given. None until a signed manifest is
/// published with the release; until then only `selftest --backends URL` uses this
/// path, so the shipped app (Windows and Linux) behaves exactly as before.
pub const DEFAULT_MANIFEST: Option<&str> = None;

/// Public keys (ed25519, hex) whose signature a manifest must carry, in
/// `<manifest URL>.sig` as 128 hex characters. The private key never enters the
/// repository; `phonon-win sign-backends` signs with it. Key 1 created 2026-11-09;
/// its seed is kept with the Android release keystore, not in the repository.
pub const MANIFEST_KEYS: &[&str] =
    &["179736715df6cbb67a77e8c4f57daabc657e491d153a94c64a228cdc35cf2d9c"];

/// Manifest schema this build understands.
const SCHEMA: u32 = 1;

/// A manifest larger than this is not a manifest.
const MANIFEST_LIMIT: u64 = 1 << 20;

/// How long the device listing child may take (a driver can take seconds to start).
const PROBE_TIMEOUT: Duration = Duration::from_secs(20);

/// One CUDA device, in the order the CUDA driver numbers it for this environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaDevice {
    /// "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", the form CUDA_VISIBLE_DEVICES takes.
    pub uuid: String,
    pub name: String,
    /// Compute capability, e.g. (8, 6) for an RTX 3090.
    pub cc: (u32, u32),
    /// "0000:01:00.0", for the log.
    pub pci: String,
}

/// What the app knows about the machine's CUDA devices.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Hardware {
    /// In CUDA's numbering for this environment (CUDA0 first).
    pub cuda: Vec<CudaDevice>,
    /// The newest CUDA version the installed driver supports, as the driver
    /// encodes it (1000 * major + 10 * minor: 13000 is CUDA 13.0). 0 when unknown.
    pub cuda_driver: u32,
}

impl Hardware {
    /// One line for the log.
    pub fn summary(&self) -> String {
        if self.cuda.is_empty() {
            return "no CUDA device".into();
        }
        let gpus: Vec<String> = self
            .cuda
            .iter()
            .map(|g| format!("{} (sm_{}{}, {})", g.name, g.cc.0, g.cc.1, g.pci))
            .collect();
        format!(
            "{}; driver supports CUDA {}",
            gpus.join(", "),
            cuda_version_text(self.cuda_driver)
        )
    }
}

/// "13.0" for 13000, "12.8" for 12080.
pub fn cuda_version_text(encoded: u32) -> String {
    format!("{}.{}", encoded / 1000, (encoded % 1000) / 10)
}

/// One downloadable runtime in the manifest.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct Entry {
    /// Stable name of the hardware family build, e.g. "linux-x64-cuda-sm86".
    pub id: String,
    /// "linux" or "windows".
    pub os: String,
    /// "x86_64" or "aarch64".
    pub arch: String,
    /// "cuda" (the only accelerated kind so far).
    pub kind: String,
    /// The compute capability the build carries machine code for, [major, minor].
    /// A device runs it when the major matches and its minor is at least this one
    /// (CUDA's binary compatibility rule for SASS).
    pub sm: [u32; 2],
    /// Machine code for exactly this compute capability (an arch-specific target
    /// such as sm_120a, which llama.cpp builds for Blackwell): no other minor runs it.
    #[serde(default)]
    pub exact: bool,
    /// Oldest driver that runs it, in the driver's CUDA version encoding.
    pub min_cuda_driver: u32,
    pub url: String,
    pub bytes: u64,
    pub sha256: String,
    /// The server executable inside the archive, relative to its root.
    pub server: String,
}

/// The whole manifest.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub schema: u32,
    pub backends: Vec<Entry>,
}

/// A path inside an archive that cannot escape it, on either OS: relative, no
/// "..", no drive or root ("C:", "/x" and "\x" are all refused on Linux too).
fn safe_relative(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with(['/', '\\'])
        && !path.contains(':')
        && !Path::new(path).is_absolute()
        && !path.split(['/', '\\']).any(|part| part == "..")
}

impl Manifest {
    pub fn parse(text: &str) -> Result<Self> {
        let manifest: Manifest =
            serde_json::from_str(text).context("parse the backend manifest")?;
        if manifest.schema != SCHEMA {
            bail!(
                "backend manifest schema {} is not {SCHEMA}; this build cannot read it",
                manifest.schema
            );
        }
        for entry in &manifest.backends {
            let hex = entry.sha256.len() == 64
                && entry
                    .sha256
                    .chars()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase());
            if !hex {
                bail!("backend {} has no valid sha256", entry.id);
            }
            if entry.bytes == 0 {
                bail!("backend {} has no size", entry.id);
            }
            if !safe_relative(&entry.server) {
                bail!("backend {} names an unsafe server path", entry.id);
            }
            let id_ok = !entry.id.is_empty()
                && entry
                    .id
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
            if !id_ok {
                bail!("backend id {:?} is not a plain name", entry.id);
            }
        }
        Ok(manifest)
    }
}

/// Lowercase or uppercase hex to bytes.
fn unhex(text: &str) -> Option<Vec<u8>> {
    let text = text.trim();
    if !text.len().is_multiple_of(2) {
        return None;
    }
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(text.get(i..i + 2)?, 16).ok())
        .collect()
}

/// Check `signature_hex` (128 hex characters) over the exact manifest bytes against
/// every key in `keys`. Nothing is parsed before this passes.
pub fn verify_manifest(bytes: &[u8], signature_hex: &str, keys: &[&str]) -> Result<()> {
    let signature = unhex(signature_hex)
        .filter(|s| s.len() == 64)
        .ok_or_else(|| anyhow!("the manifest signature is not 128 hex characters"))?;
    let signature = ed25519_compact::Signature::from_slice(&signature)
        .map_err(|e| anyhow!("bad manifest signature: {e}"))?;
    for key in keys {
        let Some(key) = unhex(key) else { continue };
        let Ok(key) = ed25519_compact::PublicKey::from_slice(&key) else {
            continue;
        };
        if key.verify(bytes, &signature).is_ok() {
            return Ok(());
        }
    }
    bail!("the backend manifest is not signed by a Phonon key")
}

/// Sign `bytes` with an ed25519 seed (64 hex characters); returns the signature as
/// hex, and the public key as hex for `MANIFEST_KEYS`.
pub fn sign_manifest(bytes: &[u8], seed_hex: &str) -> Result<(String, String)> {
    let seed = unhex(seed_hex)
        .filter(|s| s.len() == 32)
        .ok_or_else(|| anyhow!("the signing key is not 64 hex characters"))?;
    let seed = ed25519_compact::Seed::from_slice(&seed).map_err(|e| anyhow!("{e}"))?;
    let pair = ed25519_compact::KeyPair::from_seed(seed);
    let hex = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
    Ok((
        hex(pair.sk.sign(bytes, None).as_ref()),
        hex(pair.pk.as_ref()),
    ))
}

/// This build's operating system as the manifest names it.
pub fn this_os() -> &'static str {
    if cfg!(windows) {
        "windows"
    } else {
        "linux"
    }
}

/// Whether `entry` runs on a device of compute capability `cc` with `driver`.
fn runs_on(entry: &Entry, cc: (u32, u32), driver: u32, os: &str, arch: &str) -> bool {
    entry.kind == "cuda"
        && entry.os == os
        && entry.arch == arch
        && entry.sm[0] == cc.0
        && (entry.sm[1] == cc.1 || (!entry.exact && entry.sm[1] < cc.1))
        && driver >= entry.min_cuda_driver
}

/// A device and the build chosen for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Choice<'a> {
    pub device: &'a CudaDevice,
    pub entry: &'a Entry,
}

/// The device first, then its build: the device with the highest compute
/// capability that has a build (CUDA's order breaks ties), and for it the build
/// whose minor is closest to the device's without passing it (equal for `exact`).
pub fn choose<'a>(
    manifest: &'a Manifest,
    hardware: &'a Hardware,
    os: &str,
    arch: &str,
) -> Option<Choice<'a>> {
    let mut devices: Vec<&CudaDevice> = hardware.cuda.iter().collect();
    // stable: equal capabilities keep CUDA's order
    devices.sort_by_key(|device| std::cmp::Reverse(device.cc));
    devices.into_iter().find_map(|device| {
        manifest
            .backends
            .iter()
            .filter(|e| runs_on(e, device.cc, hardware.cuda_driver, os, arch))
            .max_by_key(|e| e.sm[1])
            .map(|entry| Choice { device, entry })
    })
}

/// The listing child's output: "driver <n>", then one
/// "device\t<uuid>\t<major>\t<minor>\t<pci>\t<name>" line per CUDA device in CUDA's
/// order. Malformed lines are skipped.
pub fn parse_probe(text: &str) -> Hardware {
    let mut hardware = Hardware::default();
    for line in text.lines() {
        if let Some(version) = line.strip_prefix("driver ") {
            hardware.cuda_driver = version.trim().parse().unwrap_or(0);
            continue;
        }
        let fields: Vec<&str> = line.splitn(6, '\t').collect();
        if let ["device", uuid, major, minor, pci, name] = fields[..] {
            let (Ok(major), Ok(minor)) = (major.parse(), minor.parse()) else {
                continue;
            };
            if !uuid.starts_with("GPU-") {
                continue;
            }
            hardware.cuda.push(CudaDevice {
                uuid: uuid.to_string(),
                name: name.trim().to_string(),
                cc: (major, minor),
                pci: pci.to_string(),
            });
        }
    }
    hardware
}

/// The CUDA driver's UUID bytes in CUDA_VISIBLE_DEVICES form.
pub fn uuid_text(bytes: &[u8; 16]) -> String {
    let h: Vec<String> = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!(
        "GPU-{}-{}-{}-{}-{}",
        h[0..4].concat(),
        h[4..6].concat(),
        h[6..8].concat(),
        h[8..10].concat(),
        h[10..16].concat()
    )
}

#[cfg(target_os = "linux")]
const CUDA_DRIVER: &str = "libcuda.so.1";
#[cfg(windows)]
const CUDA_DRIVER: &str = "nvcuda.dll";
#[cfg(not(any(target_os = "linux", windows)))]
const CUDA_DRIVER: &str = "";

/// `phonon-win cuda-devices`: print the CUDA devices this environment shows, in the
/// listing format `parse_probe` reads. Runs in its own short-lived process so the
/// app never holds CUDA state. Prints nothing on a machine without the driver.
pub fn print_cuda_devices() -> Result<()> {
    let text = unsafe { cuda_listing() }?;
    print!("{text}");
    Ok(())
}

/// # Safety
/// Calls the CUDA driver API through the documented exports (cuda.h, stable ABI).
unsafe fn cuda_listing() -> Result<String> {
    use std::os::raw::{c_char, c_int, c_uint};
    if CUDA_DRIVER.is_empty() {
        bail!("no CUDA driver on this platform");
    }
    let lib =
        libloading::Library::new(CUDA_DRIVER).with_context(|| format!("load {CUDA_DRIVER}"))?;
    let init: libloading::Symbol<unsafe extern "C" fn(c_uint) -> c_int> = lib.get(b"cuInit\0")?;
    let version: libloading::Symbol<unsafe extern "C" fn(*mut c_int) -> c_int> =
        lib.get(b"cuDriverGetVersion\0")?;
    let count: libloading::Symbol<unsafe extern "C" fn(*mut c_int) -> c_int> =
        lib.get(b"cuDeviceGetCount\0")?;
    let get: libloading::Symbol<unsafe extern "C" fn(*mut c_int, c_int) -> c_int> =
        lib.get(b"cuDeviceGet\0")?;
    let name: libloading::Symbol<unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int> =
        lib.get(b"cuDeviceGetName\0")?;
    let attribute: libloading::Symbol<unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int> =
        lib.get(b"cuDeviceGetAttribute\0")?;
    let uuid: libloading::Symbol<unsafe extern "C" fn(*mut [u8; 16], c_int) -> c_int> = lib
        .get(b"cuDeviceGetUuid_v2\0")
        .or_else(|_| lib.get(b"cuDeviceGetUuid\0"))?;
    // CUdevice_attribute values from cuda.h
    const CC_MAJOR: c_int = 75;
    const CC_MINOR: c_int = 76;
    const PCI_BUS: c_int = 33;
    const PCI_DEVICE: c_int = 34;
    const PCI_DOMAIN: c_int = 50;

    let mut out = String::new();
    let mut driver: c_int = 0;
    if version(&mut driver) == 0 {
        out.push_str(&format!("driver {driver}\n"));
    }
    // CUDA_ERROR_NO_DEVICE (100) is also what an empty CUDA_VISIBLE_DEVICES gives
    if init(0) != 0 {
        return Ok(out);
    }
    let mut n: c_int = 0;
    if count(&mut n) != 0 {
        return Ok(out);
    }
    for ordinal in 0..n {
        let mut device: c_int = 0;
        if get(&mut device, ordinal) != 0 {
            continue;
        }
        let read = |attr: c_int| {
            let mut value: c_int = 0;
            (attribute(&mut value, attr, device) == 0).then_some(value)
        };
        let (Some(major), Some(minor)) = (read(CC_MAJOR), read(CC_MINOR)) else {
            continue;
        };
        let mut id = [0u8; 16];
        if uuid(&mut id, device) != 0 {
            continue;
        }
        let mut label = [0 as c_char; 256];
        let label = if name(label.as_mut_ptr(), label.len() as c_int, device) == 0 {
            std::ffi::CStr::from_ptr(label.as_ptr())
                .to_string_lossy()
                .replace(['\t', '\n'], " ")
        } else {
            format!("CUDA device {ordinal}")
        };
        let pci = format!(
            "{:04x}:{:02x}:{:02x}.0",
            read(PCI_DOMAIN).unwrap_or(0),
            read(PCI_BUS).unwrap_or(0),
            read(PCI_DEVICE).unwrap_or(0)
        );
        out.push_str(&format!(
            "device\t{}\t{major}\t{minor}\t{pci}\t{label}\n",
            uuid_text(&id)
        ));
    }
    Ok(out)
}

/// The listing child: this executable with `cuda-devices`, inheriting the
/// environment (so the user's CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER apply).
fn probe_command() -> Result<Command> {
    let mut command = Command::new(std::env::current_exe().context("find phonon-win")?);
    command
        .arg("cuda-devices")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(crate::asr::CREATE_NO_WINDOW);
    }
    Ok(command)
}

/// The CUDA devices as the runtime will number them. None found (no driver, an
/// empty CUDA_VISIBLE_DEVICES, a hung driver) is an empty list.
pub fn detect() -> Hardware {
    let run = || -> Result<Hardware> {
        let mut child = probe_command()?.spawn()?;
        let started = Instant::now();
        while child.try_wait()?.is_none() {
            if started.elapsed() > PROBE_TIMEOUT {
                let _ = child.kill();
                bail!("listing CUDA devices took over {PROBE_TIMEOUT:?}");
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let output = child.wait_with_output()?;
        Ok(parse_probe(&String::from_utf8_lossy(&output.stdout)))
    };
    run().unwrap_or_else(|error| {
        eprintln!("phonon: no CUDA device information ({error:#})");
        Hardware::default()
    })
}

/// Bytes from an https, http or file URL, at most `MANIFEST_LIMIT`.
fn read_small(url: &str) -> Result<Vec<u8>> {
    use std::io::Read;
    let mut bytes = Vec::new();
    if let Some(path) = url.strip_prefix("file://") {
        std::fs::File::open(path)
            .with_context(|| format!("read {path}"))?
            .take(MANIFEST_LIMIT)
            .read_to_end(&mut bytes)?;
    } else {
        http::agent()
            .get(url)
            .timeout(Duration::from_secs(20))
            .call()
            .with_context(|| format!("download {url}"))?
            .into_reader()
            .take(MANIFEST_LIMIT)
            .read_to_end(&mut bytes)?;
    }
    Ok(bytes)
}

/// The manifest at `url`, after its signature at `url.sig` checks out.
pub fn read_manifest(url: &str) -> Result<Manifest> {
    let bytes = read_small(url)?;
    let signature = read_small(&format!("{url}.sig"))?;
    verify_manifest(&bytes, &String::from_utf8_lossy(&signature), MANIFEST_KEYS)?;
    Manifest::parse(std::str::from_utf8(&bytes).context("the manifest is not UTF-8")?)
}

/// Where every tuned runtime lives.
fn backends_root() -> PathBuf {
    crate::paths::data_root().join("runtime/backends")
}

/// The installed component for an entry. Its directory names the hash, so a new
/// build lands beside the old one and replaces it only once it is confirmed.
/// The strings live for the rest of the process (`Component` borrows them for
/// 'static, like the compiled-in manifest); one manifest is read per start.
fn component(entry: &Entry) -> Component {
    let leak = |s: String| -> &'static str { Box::leak(s.into_boxed_str()) };
    // fetch::expand picks the format by the file name
    let format = if entry.url.ends_with(".zip") {
        "zip"
    } else {
        "tar.gz"
    };
    let asset = Asset {
        name: leak(format!("{}-{}.{format}", entry.id, &entry.sha256[..12])),
        url: leak(entry.url.clone()),
        sha256: leak(entry.sha256.clone()),
        bytes: entry.bytes,
    };
    Component {
        dir: leak(format!(
            "runtime/backends/{}-{}",
            entry.id,
            &entry.sha256[..12]
        )),
        label: "GPU correction runtime",
        assets: Box::leak(Box::new([asset])),
        unpack: Unpack::Archive { strip_to: "" },
        sentinel: leak(entry.server.clone()),
    }
}

/// Whether a directory name is a build of family `id`: "<id>-<12 hex>".
fn is_build_of(name: &str, id: &str) -> bool {
    name.strip_prefix(id)
        .and_then(|rest| rest.strip_prefix('-'))
        .is_some_and(|hash| hash.len() == 12 && hash.chars().all(|c| c.is_ascii_hexdigit()))
}

/// Installed builds of family `id` under `root` that hold `server`, newest first.
fn installed_builds(root: &Path, id: &str, server: &str) -> Vec<PathBuf> {
    let Ok(dirs) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut found: Vec<(std::time::SystemTime, PathBuf)> = dirs
        .flatten()
        .filter(|d| is_build_of(&d.file_name().to_string_lossy(), id))
        .filter(|d| d.path().join(server).is_file())
        .map(|d| {
            let time = d
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::UNIX_EPOCH);
            (time, d.path())
        })
        .collect();
    found.sort_by_key(|(time, _)| std::cmp::Reverse(*time));
    found.into_iter().map(|(_, path)| path).collect()
}

/// Remove every build of family `id` under `root` except `keep`.
fn prune(root: &Path, id: &str, keep: &Path) {
    let Ok(dirs) = std::fs::read_dir(root) else {
        return;
    };
    for dir in dirs.flatten() {
        let path = dir.path();
        if path != keep && is_build_of(&dir.file_name().to_string_lossy(), id) && path.is_dir() {
            let _ = std::fs::remove_dir_all(&path);
        }
    }
}

/// A tuned runtime ready to start, for one device.
#[derive(Debug, Clone)]
pub struct Accelerated {
    pub id: String,
    /// Server executables to try in order: the manifest's build, then earlier
    /// builds of the family still on disk.
    pub servers: Vec<PathBuf>,
    pub device: CudaDevice,
}

impl Accelerated {
    /// Environment for every child of this runtime: only the chosen device, by
    /// UUID, so CUDA0 in the server is the device the build was chosen for.
    pub fn env(&self) -> Vec<(String, String)> {
        vec![("CUDA_VISIBLE_DEVICES".into(), self.device.uuid.clone())]
    }
}

/// A build that started; `confirm` once the startup check passed on it.
#[derive(Debug, Clone)]
pub struct Installed {
    pub id: String,
    pub dir: PathBuf,
}

impl Installed {
    /// The build passed the startup check: remove the family's other builds.
    pub fn confirm(&self) {
        if let Some(root) = self.dir.parent() {
            prune(root, &self.id, &self.dir);
        }
    }
}

/// Whether the pinned server's device listing (`--list-devices`) shows exactly one
/// CUDA device, and it is the chosen card by name
/// ("  CUDA0: NVIDIA GeForce RTX 3090 (24145 MiB, 21037 MiB free)"). The UUID pin
/// is what selects the card; this proves the pin took effect in the runtime.
/// llama-server does not print compute capability or bus id here.
pub fn listing_matches(listing: &str, name: &str) -> bool {
    let devices: Vec<&str> = listing
        .lines()
        .filter_map(|line| line.trim().split_once(':'))
        .filter(|(device, _)| {
            device
                .strip_prefix("CUDA")
                .is_some_and(|i| i.parse::<u32>().is_ok())
        })
        .map(|(_, description)| description.trim())
        .collect();
    matches!(devices[..], [only] if only.strip_prefix(name).is_some_and(|rest| rest.starts_with(" (")))
}

/// Detect, choose and install. Ok(None): no tuned build for this machine. A failed
/// download still returns the family's earlier builds, if any.
pub fn prepare(manifest_url: &str) -> Result<Option<Accelerated>> {
    let hardware = detect();
    eprintln!("phonon: hardware: {}", hardware.summary());
    let manifest = read_manifest(manifest_url)?;
    let Some(choice) = choose(&manifest, &hardware, this_os(), std::env::consts::ARCH) else {
        return Ok(None);
    };
    let (entry, device) = (choice.entry, choice.device.clone());
    eprintln!("phonon: {} for {} ({})", entry.id, device.name, device.uuid);
    let component = component(entry);
    if !fetch::installed(&component) {
        eprintln!(
            "phonon: installing {} ({})",
            entry.id,
            crate::manifest::human_bytes(entry.bytes)
        );
    }
    let mut servers = Vec::new();
    let fresh = match fetch::ensure(&component, &mut |_| {}) {
        Ok(dir) => {
            servers.push(dir.join(&entry.server));
            Some(dir)
        }
        Err(error) => {
            eprintln!("phonon: install {} failed ({error:#})", entry.id);
            None
        }
    };
    for dir in installed_builds(&backends_root(), &entry.id, &entry.server) {
        if Some(&dir) != fresh.as_ref() {
            servers.push(dir.join(&entry.server));
        }
    }
    if servers.is_empty() {
        bail!("no build of {} could be installed", entry.id);
    }
    Ok(Some(Accelerated {
        id: entry.id.clone(),
        servers,
        device,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str, sm: [u32; 2], driver: u32) -> String {
        format!(
            r#"{{"id":"{id}","os":"linux","arch":"x86_64","kind":"cuda","sm":[{},{}],
               "min_cuda_driver":{driver},"url":"file:///b/{id}.tar.gz","bytes":10,
               "sha256":"{}","server":"llama-server"}}"#,
            sm[0],
            sm[1],
            "a".repeat(64)
        )
    }

    fn manifest(entries: &[String]) -> Manifest {
        Manifest::parse(&format!(
            r#"{{"schema":1,"backends":[{}]}}"#,
            entries.join(",")
        ))
        .unwrap()
    }

    fn gpu(uuid: &str, cc: (u32, u32)) -> CudaDevice {
        CudaDevice {
            uuid: uuid.into(),
            name: format!("test sm_{}{}", cc.0, cc.1),
            cc,
            pci: "0000:01:00.0".into(),
        }
    }

    fn hw(gpus: &[(u32, u32)], driver: u32) -> Hardware {
        Hardware {
            cuda: gpus
                .iter()
                .enumerate()
                .map(|(i, &cc)| gpu(&format!("GPU-{i}"), cc))
                .collect(),
            cuda_driver: driver,
        }
    }

    fn families() -> Manifest {
        manifest(&[
            entry("cuda-sm75", [7, 5], 13000),
            entry("cuda-sm86", [8, 6], 13000),
            entry("cuda-sm89", [8, 9], 13000),
            entry("cuda-sm120", [12, 0], 13000),
            entry("cuda-sm61", [6, 1], 12000),
        ])
    }

    fn pick(m: &Manifest, h: &Hardware) -> Option<String> {
        choose(m, h, "linux", "x86_64").map(|c| c.entry.id.clone())
    }

    /// (device uuid, build id)
    fn pick_both(m: &Manifest, h: &Hardware) -> Option<(String, String)> {
        choose(m, h, "linux", "x86_64").map(|c| (c.device.uuid.clone(), c.entry.id.clone()))
    }

    #[test]
    fn an_exact_family_wins() {
        let m = families();
        for (cc, driver, id) in [
            ((8, 6), 13000, "cuda-sm86"),
            ((8, 9), 13000, "cuda-sm89"),
            ((12, 0), 13020, "cuda-sm120"),
            ((6, 1), 12080, "cuda-sm61"),
        ] {
            assert_eq!(pick(&m, &hw(&[cc], driver)).as_deref(), Some(id), "{cc:?}");
        }
    }

    #[test]
    fn a_lower_minor_of_the_same_major_runs() {
        // an A100 (8.0) has no build: 8.6 machine code does not run on 8.0
        let m = families();
        assert_eq!(pick(&m, &hw(&[(8, 0)], 13000)), None);
        // a Jetson Orin (8.7) runs the 8.6 build
        assert_eq!(
            pick(&m, &hw(&[(8, 7)], 13000)).as_deref(),
            Some("cuda-sm86")
        );
        // without an 8.9 entry an RTX 4090 takes 8.6
        let m = manifest(&[entry("cuda-sm86", [8, 6], 13000)]);
        assert_eq!(
            pick(&m, &hw(&[(8, 9)], 13000)).as_deref(),
            Some("cuda-sm86")
        );
    }

    #[test]
    fn an_arch_specific_build_runs_only_on_its_own_minor() {
        // sm_120a machine code: an RTX 5090 (12.0) runs it, a GB10 (12.1) does not
        let e = entry("cuda-sm120a", [12, 0], 13000).replace("\"sm\"", "\"exact\":true,\"sm\"");
        let m = manifest(&[e]);
        assert!(m.backends[0].exact);
        assert_eq!(
            pick(&m, &hw(&[(12, 0)], 13000)).as_deref(),
            Some("cuda-sm120a")
        );
        assert_eq!(pick(&m, &hw(&[(12, 1)], 13000)), None);
    }

    #[test]
    fn another_major_never_runs() {
        let m = manifest(&[entry("cuda-sm86", [8, 6], 13000)]);
        assert_eq!(pick(&m, &hw(&[(12, 0)], 13000)), None);
        assert_eq!(pick(&m, &hw(&[(7, 5)], 13000)), None);
    }

    #[test]
    fn an_old_driver_falls_through() {
        let m = families();
        assert_eq!(pick(&m, &hw(&[(8, 6)], 12080)), None);
        assert_eq!(pick(&m, &hw(&[(8, 6)], 0)), None);
    }

    #[test]
    fn no_gpu_means_no_backend() {
        assert_eq!(pick(&families(), &Hardware::default()), None);
    }

    /// Two cards of different families, in both CUDA orders: the build and the
    /// device it is pinned to are always the same card.
    #[test]
    fn two_gpus_the_build_and_the_device_agree_in_either_order() {
        let m = families();
        let turing = gpu("GPU-aaaa", (7, 5));
        let ampere = gpu("GPU-bbbb", (8, 6));
        for order in [
            vec![turing.clone(), ampere.clone()],
            vec![ampere.clone(), turing.clone()],
        ] {
            let h = Hardware {
                cuda: order,
                cuda_driver: 13000,
            };
            assert_eq!(
                pick_both(&m, &h),
                Some(("GPU-bbbb".into(), "cuda-sm86".into()))
            );
        }
        // only the weaker card has a build: that card, not CUDA0 and not the strongest
        let only75 = manifest(&[entry("cuda-sm75", [7, 5], 13000)]);
        for order in [
            vec![turing.clone(), ampere.clone()],
            vec![ampere.clone(), turing.clone()],
        ] {
            let h = Hardware {
                cuda: order,
                cuda_driver: 13000,
            };
            assert_eq!(
                pick_both(&only75, &h),
                Some(("GPU-aaaa".into(), "cuda-sm75".into()))
            );
        }
        // the strongest has none: the next one does
        let h = hw(&[(12, 0), (7, 5)], 13000);
        assert_eq!(
            pick_both(&only75, &h),
            Some(("GPU-1".into(), "cuda-sm75".into()))
        );
    }

    /// CUDA_VISIBLE_DEVICES is applied by the driver to the listing child, so the
    /// listing holds only the visible cards; the hidden stronger card is never
    /// chosen, and the server is pinned to the chosen card's UUID.
    #[test]
    fn a_card_hidden_by_cuda_visible_devices_is_never_chosen() {
        // both cards, then CUDA_VISIBLE_DEVICES=GPU-aaaa (the Turing card only)
        let all = "driver 13000\n\
            device\tGPU-bbbbbbbb-0000-0000-0000-000000000000\t8\t6\t0000:01:00.0\tNVIDIA GeForce RTX 3090\n\
            device\tGPU-aaaaaaaa-0000-0000-0000-000000000000\t7\t5\t0000:02:00.0\tNVIDIA GeForce RTX 2080 Ti\n";
        let visible = "driver 13000\n\
            device\tGPU-aaaaaaaa-0000-0000-0000-000000000000\t7\t5\t0000:02:00.0\tNVIDIA GeForce RTX 2080 Ti\n";
        let m = families();
        let h = parse_probe(all);
        assert_eq!(pick(&m, &h).as_deref(), Some("cuda-sm86"));
        let h = parse_probe(visible);
        let choice = choose(&m, &h, "linux", "x86_64").unwrap();
        assert_eq!(choice.entry.id, "cuda-sm75");
        let tuned = Accelerated {
            id: choice.entry.id.clone(),
            servers: vec![],
            device: choice.device.clone(),
        };
        assert_eq!(
            tuned.env(),
            vec![(
                "CUDA_VISIBLE_DEVICES".to_string(),
                "GPU-aaaaaaaa-0000-0000-0000-000000000000".to_string()
            )]
        );
        // CUDA_VISIBLE_DEVICES="" (or a driver-less machine): nothing listed
        assert_eq!(pick(&m, &parse_probe("driver 13000\n")), None);
        assert_eq!(pick(&m, &parse_probe("")), None);
    }

    #[test]
    fn the_listing_child_inherits_the_environment() {
        let command = probe_command().unwrap();
        assert!(command.get_envs().next().is_none(), "nothing is overridden");
        assert_eq!(
            command.get_args().collect::<Vec<_>>(),
            vec![std::ffi::OsStr::new("cuda-devices")]
        );
    }

    #[test]
    fn probe_output_parses_and_skips_noise() {
        let text = "driver 13050\n\
            device\tGPU-6a1e8c2b-1111-2222-3333-444455556666\t8\t6\t0000:01:00.0\tNVIDIA GeForce RTX 3090\n\
            device\tnot-a-uuid\t8\t6\t0000:02:00.0\tbroken\n\
            device\tGPU-x\tA\t6\t0000:03:00.0\tbroken\n\
            garbage\n";
        let h = parse_probe(text);
        assert_eq!(h.cuda_driver, 13050);
        assert_eq!(h.cuda.len(), 1);
        assert_eq!(h.cuda[0].cc, (8, 6));
        assert_eq!(h.cuda[0].name, "NVIDIA GeForce RTX 3090");
        assert_eq!(
            h.summary(),
            "NVIDIA GeForce RTX 3090 (sm_86, 0000:01:00.0); driver supports CUDA 13.5"
        );
        assert_eq!(Hardware::default().summary(), "no CUDA device");
    }

    #[test]
    fn uuids_take_the_cuda_visible_devices_form() {
        let bytes = [
            0x6a, 0x1e, 0x8c, 0x2b, 0x11, 0x11, 0x22, 0x22, 0x33, 0x33, 0x44, 0x44, 0x55, 0x55,
            0x66, 0x66,
        ];
        assert_eq!(
            uuid_text(&bytes),
            "GPU-6a1e8c2b-1111-2222-3333-444455556666"
        );
    }

    #[test]
    fn the_pinned_server_must_list_exactly_the_chosen_device() {
        // what llama-server b12132 prints with CUDA_VISIBLE_DEVICES=GPU-<uuid> on a 3090
        let good =
            "Available devices:\n  CUDA0: NVIDIA GeForce RTX 3090 (24145 MiB, 21037 MiB free)\n";
        let rtx3090 = "NVIDIA GeForce RTX 3090";
        assert!(listing_matches(good, rtx3090));
        assert!(!listing_matches(good, "NVIDIA GeForce RTX 2080 Ti"));
        // a listed name that only begins with the chosen one is another card
        assert!(!listing_matches(good, "NVIDIA GeForce RTX 30"));
        let two =
            format!("{good}  CUDA1: NVIDIA GeForce RTX 2080 Ti (11000 MiB, 10000 MiB free)\n");
        assert!(!listing_matches(&two, rtx3090));
        assert!(!listing_matches("Available devices:\n  (none)\n", rtx3090));
    }

    #[test]
    fn another_os_or_arch_is_ignored() {
        let m = families();
        assert_eq!(choose(&m, &hw(&[(8, 6)], 13000), "windows", "x86_64"), None);
        assert_eq!(choose(&m, &hw(&[(8, 6)], 13000), "linux", "aarch64"), None);
    }

    #[test]
    fn bad_manifests_are_refused() {
        assert!(Manifest::parse(r#"{"schema":2,"backends":[]}"#).is_err());
        assert!(Manifest::parse("not json").is_err());
        let short_hash = entry("x", [8, 6], 0).replace(&"a".repeat(64), "abc");
        assert!(Manifest::parse(&format!(r#"{{"schema":1,"backends":[{short_hash}]}}"#)).is_err());
        for server in [
            "../llama-server",
            "/usr/bin/llama-server",
            r"\\server\share\llama-server.exe",
            r"C:\\llama-server.exe",
            "",
        ] {
            let e = entry("x", [8, 6], 0).replace("\"llama-server\"", &format!("{server:?}"));
            assert!(
                Manifest::parse(&format!(r#"{{"schema":1,"backends":[{e}]}}"#)).is_err(),
                "{server}"
            );
        }
        let e = entry("../x", [8, 6], 0);
        assert!(Manifest::parse(&format!(r#"{{"schema":1,"backends":[{e}]}}"#)).is_err());
    }

    #[test]
    fn driver_versions_read_as_cuda_versions() {
        assert_eq!(cuda_version_text(13000), "13.0");
        assert_eq!(cuda_version_text(12080), "12.8");
        assert_eq!(cuda_version_text(13020), "13.2");
    }

    const TEST_SEED: &str = "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60";

    #[test]
    fn only_a_manifest_signed_by_a_listed_key_is_read() {
        let body = br#"{"schema":1,"backends":[]}"#;
        let (signature, public) = sign_manifest(body, TEST_SEED).unwrap();
        assert_eq!(signature.len(), 128);
        assert!(verify_manifest(body, &signature, &[&public]).is_ok());
        assert!(verify_manifest(body, &format!("{signature}\n"), &[&public]).is_ok());
        // one changed byte
        assert!(
            verify_manifest(br#"{"schema":1,"backends":[ ]}"#, &signature, &[&public]).is_err()
        );
        // a key that did not sign it
        assert!(verify_manifest(body, &signature, MANIFEST_KEYS).is_err());
        // malformed signatures
        assert!(verify_manifest(body, "", &[&public]).is_err());
        assert!(verify_manifest(body, &signature[..126], &[&public]).is_err());
        assert!(verify_manifest(body, &"zz".repeat(64), &[&public]).is_err());
    }

    #[test]
    fn the_compiled_in_keys_are_ed25519_public_keys() {
        assert!(!MANIFEST_KEYS.is_empty());
        for key in MANIFEST_KEYS {
            let bytes = unhex(key).expect("hex");
            assert!(
                ed25519_compact::PublicKey::from_slice(&bytes).is_ok(),
                "{key}"
            );
        }
    }

    #[test]
    fn a_new_build_replaces_the_old_one_only_when_confirmed() {
        let root = std::env::temp_dir().join("phonon-win-prune-test");
        let _ = std::fs::remove_dir_all(&root);
        for dir in [
            "cuda-sm86-aaaaaaaaaaaa",
            "cuda-sm86-bbbbbbbbbbbb",
            "cuda-sm89-cccccccccccc",
            "cuda-sm86-extra-dddddddddddd",
        ] {
            std::fs::create_dir_all(root.join(dir)).unwrap();
            std::fs::write(root.join(dir).join("llama-server"), b"").unwrap();
        }
        // both sm86 builds are candidates until one is confirmed; other ids are not
        let builds = installed_builds(&root, "cuda-sm86", "llama-server");
        assert_eq!(builds.len(), 2);
        assert!(builds
            .iter()
            .all(|b| is_build_of(&b.file_name().unwrap().to_string_lossy(), "cuda-sm86")));
        Installed {
            id: "cuda-sm86".into(),
            dir: root.join("cuda-sm86-bbbbbbbbbbbb"),
        }
        .confirm();
        assert!(!root.join("cuda-sm86-aaaaaaaaaaaa").exists());
        assert!(root.join("cuda-sm86-bbbbbbbbbbbb").exists());
        assert!(root.join("cuda-sm89-cccccccccccc").exists());
        assert!(root.join("cuda-sm86-extra-dddddddddddd").exists());
        std::fs::remove_dir_all(&root).unwrap();
    }
}
