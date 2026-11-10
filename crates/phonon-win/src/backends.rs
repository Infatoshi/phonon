//! Prebuilt, hardware-specific correction runtimes.
//!
//! Every user gets the CPU runtime (`manifest::LLAMA_RUNTIME`). A machine whose GPU
//! has a tuned build on our side also gets that build: a llama.cpp b12132 runtime
//! with Phonon's CUDA patch (`kernels/cuda/merged/b12132.patch`) compiled for one GPU
//! family only. Nothing is compiled on the user's machine.
//!
//! At start: read the GPUs (NVML, which ships with every NVIDIA driver and does not
//! create a CUDA context in this process), read the backend manifest, pick the
//! entry for this GPU, and install it with the same SHA-256-checked download as
//! every other component. Any step that fails leaves the next path in the fallback
//! order (Vulkan on Windows, then the CPU), so a missing or broken backend never
//! costs the user dictation.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;

use crate::fetch;
use crate::http;
use crate::manifest::{Asset, Component, Unpack};

/// The manifest the app reads when none is given. None until a signed manifest is
/// published with the release; until then only `selftest --backends URL` uses this
/// path, so the shipped app behaves exactly as before.
pub const DEFAULT_MANIFEST: Option<&str> = None;

/// Manifest schema this build understands.
const SCHEMA: u32 = 1;

/// A manifest larger than this is not a manifest.
const MANIFEST_LIMIT: u64 = 1 << 20;

/// One NVIDIA GPU as the driver reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NvidiaGpu {
    pub name: String,
    /// Compute capability, e.g. (8, 6) for an RTX 3090.
    pub cc: (u32, u32),
}

/// What the app knows about the machine's accelerators.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Hardware {
    pub nvidia: Vec<NvidiaGpu>,
    /// The newest CUDA version the installed driver supports, as the driver
    /// encodes it (1000 * major + 10 * minor: 13000 is CUDA 13.0). 0 when unknown.
    pub cuda_driver: u32,
}

impl Hardware {
    /// One line for the log.
    pub fn summary(&self) -> String {
        if self.nvidia.is_empty() {
            return "no NVIDIA GPU".into();
        }
        let gpus: Vec<String> = self
            .nvidia
            .iter()
            .map(|g| format!("{} (sm_{}{})", g.name, g.cc.0, g.cc.1))
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
            let safe = !entry.server.is_empty()
                && !Path::new(&entry.server).is_absolute()
                && !entry.server.split(['/', '\\']).any(|part| part == "..");
            if !safe {
                bail!("backend {} names an unsafe server path", entry.id);
            }
        }
        Ok(manifest)
    }
}

/// This build's operating system as the manifest names it.
pub fn this_os() -> &'static str {
    if cfg!(windows) {
        "windows"
    } else {
        "linux"
    }
}

/// The best entry for this machine: for each NVIDIA GPU (highest compute
/// capability first), a CUDA build for the same OS and CPU architecture whose
/// major matches, whose minor is not above the device's (equal for an `exact`
/// build), and whose driver requirement the installed driver meets. An exact
/// minor beats a lower one.
pub fn choose<'m>(
    manifest: &'m Manifest,
    hardware: &Hardware,
    os: &str,
    arch: &str,
) -> Option<&'m Entry> {
    let mut gpus: Vec<&NvidiaGpu> = hardware.nvidia.iter().collect();
    gpus.sort_by_key(|gpu| std::cmp::Reverse(gpu.cc));
    gpus.into_iter().find_map(|gpu| {
        manifest
            .backends
            .iter()
            .filter(|e| e.kind == "cuda" && e.os == os && e.arch == arch)
            .filter(|e| e.sm[0] == gpu.cc.0)
            .filter(|e| e.sm[1] == gpu.cc.1 || (!e.exact && e.sm[1] < gpu.cc.1))
            .filter(|e| hardware.cuda_driver >= e.min_cuda_driver)
            .max_by_key(|e| e.sm[1])
    })
}

/// The NVIDIA management library: present wherever the NVIDIA driver is.
#[cfg(target_os = "linux")]
const NVML: &str = "libnvidia-ml.so.1";
#[cfg(windows)]
const NVML: &str = "nvml.dll";
#[cfg(not(any(target_os = "linux", windows)))]
const NVML: &str = "";

/// The machine's NVIDIA GPUs and driver, from NVML. A machine without the NVIDIA
/// driver (or with one too old to report compute capability) has none.
pub fn detect() -> Hardware {
    match unsafe { nvml_detect() } {
        Ok(hardware) => hardware,
        Err(error) => {
            eprintln!("phonon: no NVIDIA GPU information ({error:#})");
            Hardware::default()
        }
    }
}

/// # Safety
/// Calls into the NVIDIA driver through function pointers it exports under the
/// documented names and signatures (nvml.h, stable since R450).
unsafe fn nvml_detect() -> Result<Hardware> {
    use std::os::raw::{c_char, c_int, c_uint, c_void};
    type Device = *mut c_void;
    if NVML.is_empty() {
        bail!("not on this platform");
    }
    let lib = libloading::Library::new(NVML).with_context(|| format!("load {NVML}"))?;
    let init: libloading::Symbol<unsafe extern "C" fn() -> c_int> = lib.get(b"nvmlInit_v2\0")?;
    let shutdown: libloading::Symbol<unsafe extern "C" fn() -> c_int> =
        lib.get(b"nvmlShutdown\0")?;
    let driver: libloading::Symbol<unsafe extern "C" fn(*mut c_int) -> c_int> =
        lib.get(b"nvmlSystemGetCudaDriverVersion\0")?;
    let count: libloading::Symbol<unsafe extern "C" fn(*mut c_uint) -> c_int> =
        lib.get(b"nvmlDeviceGetCount_v2\0")?;
    let handle: libloading::Symbol<unsafe extern "C" fn(c_uint, *mut Device) -> c_int> =
        lib.get(b"nvmlDeviceGetHandleByIndex_v2\0")?;
    let name: libloading::Symbol<unsafe extern "C" fn(Device, *mut c_char, c_uint) -> c_int> =
        lib.get(b"nvmlDeviceGetName\0")?;
    let capability: libloading::Symbol<
        unsafe extern "C" fn(Device, *mut c_int, *mut c_int) -> c_int,
    > = lib.get(b"nvmlDeviceGetCudaComputeCapability\0")?;

    let status = init();
    if status != 0 {
        bail!("nvmlInit_v2 returned {status}");
    }
    let mut hardware = Hardware::default();
    let mut version: c_int = 0;
    if driver(&mut version) == 0 && version > 0 {
        hardware.cuda_driver = version as u32;
    }
    let mut n: c_uint = 0;
    if count(&mut n) == 0 {
        for index in 0..n {
            let mut device: Device = std::ptr::null_mut();
            if handle(index, &mut device) != 0 {
                continue;
            }
            let mut buffer = [0 as c_char; 96];
            let label = if name(device, buffer.as_mut_ptr(), buffer.len() as c_uint) == 0 {
                std::ffi::CStr::from_ptr(buffer.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            } else {
                format!("NVIDIA GPU {index}")
            };
            let (mut major, mut minor): (c_int, c_int) = (0, 0);
            if capability(device, &mut major, &mut minor) == 0 && major > 0 {
                hardware.nvidia.push(NvidiaGpu {
                    name: label,
                    cc: (major as u32, minor.max(0) as u32),
                });
            }
        }
    }
    shutdown();
    Ok(hardware)
}

/// The manifest's bytes from an https, http or file URL.
pub fn read_manifest(url: &str) -> Result<Manifest> {
    let text = if let Some(path) = url.strip_prefix("file://") {
        std::fs::read_to_string(path).with_context(|| format!("read {path}"))?
    } else {
        let response = http::agent()
            .get(url)
            .timeout(std::time::Duration::from_secs(20))
            .call()
            .with_context(|| format!("download {url}"))?;
        let mut text = String::new();
        use std::io::Read;
        response
            .into_reader()
            .take(MANIFEST_LIMIT)
            .read_to_string(&mut text)
            .context("read the backend manifest")?;
        text
    };
    Manifest::parse(&text)
}

/// The installed component for an entry. Its directory names the hash, so a new
/// build lands beside the old one and replaces it only once it is complete.
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

/// Remove the other builds of the same family once `keep` is installed.
fn prune(entry: &Entry, keep: &Path) {
    let Some(parent) = keep.parent() else { return };
    let Ok(dirs) = std::fs::read_dir(parent) else {
        return;
    };
    let prefix = format!("{}-", entry.id);
    for dir in dirs.flatten() {
        let path = dir.path();
        let name = dir.file_name().to_string_lossy().into_owned();
        if path != keep && name.starts_with(&prefix) && path.is_dir() {
            let _ = std::fs::remove_dir_all(&path);
        }
    }
}

/// A tuned runtime ready to start.
#[derive(Debug, Clone)]
pub struct Accelerated {
    pub id: String,
    pub server: PathBuf,
}

/// Detect, choose and install. Ok(None): no tuned build for this machine.
pub fn prepare(manifest_url: &str) -> Result<Option<Accelerated>> {
    let hardware = detect();
    eprintln!("phonon: hardware: {}", hardware.summary());
    let manifest = read_manifest(manifest_url)?;
    let Some(entry) = choose(&manifest, &hardware, this_os(), std::env::consts::ARCH) else {
        return Ok(None);
    };
    let component = component(entry);
    if !fetch::installed(&component) {
        eprintln!(
            "phonon: installing {} ({})",
            entry.id,
            crate::manifest::human_bytes(entry.bytes)
        );
    }
    let dir = fetch::ensure(&component, &mut |_| {})
        .map_err(|error| anyhow!("install {}: {error:#}", entry.id))?;
    prune(entry, &dir);
    Ok(Some(Accelerated {
        id: entry.id.clone(),
        server: dir.join(&entry.server),
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

    fn gpu(cc: (u32, u32)) -> NvidiaGpu {
        NvidiaGpu {
            name: "test".into(),
            cc,
        }
    }

    fn hw(gpus: &[(u32, u32)], driver: u32) -> Hardware {
        Hardware {
            nvidia: gpus.iter().copied().map(gpu).collect(),
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
        choose(m, h, "linux", "x86_64").map(|e| e.id.clone())
    }

    #[test]
    fn an_exact_family_wins() {
        let m = families();
        assert_eq!(
            pick(&m, &hw(&[(8, 6)], 13000)).as_deref(),
            Some("cuda-sm86")
        );
        assert_eq!(
            pick(&m, &hw(&[(8, 9)], 13000)).as_deref(),
            Some("cuda-sm89")
        );
        assert_eq!(
            pick(&m, &hw(&[(12, 0)], 13020)).as_deref(),
            Some("cuda-sm120")
        );
        assert_eq!(
            pick(&m, &hw(&[(6, 1)], 12080)).as_deref(),
            Some("cuda-sm61")
        );
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

    #[test]
    fn the_strongest_gpu_with_a_build_is_used() {
        let m = manifest(&[
            entry("cuda-sm75", [7, 5], 13000),
            entry("cuda-sm86", [8, 6], 13000),
        ]);
        assert_eq!(
            pick(&m, &hw(&[(7, 5), (8, 6)], 13000)).as_deref(),
            Some("cuda-sm86")
        );
        // the strongest has none: the next one does
        assert_eq!(
            pick(&m, &hw(&[(12, 0), (7, 5)], 13000)).as_deref(),
            Some("cuda-sm75")
        );
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
        for server in ["../llama-server", "/usr/bin/llama-server", ""] {
            let e = entry("x", [8, 6], 0).replace("\"llama-server\"", &format!("\"{server}\""));
            assert!(
                Manifest::parse(&format!(r#"{{"schema":1,"backends":[{e}]}}"#)).is_err(),
                "{server}"
            );
        }
    }

    #[test]
    fn driver_versions_read_as_cuda_versions() {
        assert_eq!(cuda_version_text(13000), "13.0");
        assert_eq!(cuda_version_text(12080), "12.8");
        assert_eq!(cuda_version_text(13020), "13.2");
    }

    #[test]
    fn summaries_name_the_family() {
        let h = Hardware {
            nvidia: vec![NvidiaGpu {
                name: "NVIDIA GeForce RTX 3090".into(),
                cc: (8, 6),
            }],
            cuda_driver: 13000,
        };
        assert_eq!(
            h.summary(),
            "NVIDIA GeForce RTX 3090 (sm_86); driver supports CUDA 13.0"
        );
        assert_eq!(Hardware::default().summary(), "no NVIDIA GPU");
    }

    #[test]
    fn a_new_build_replaces_the_old_one_of_its_family() {
        let root = std::env::temp_dir().join("phonon-win-prune-test");
        let _ = std::fs::remove_dir_all(&root);
        for dir in [
            "cuda-sm86-aaaaaaaaaaaa",
            "cuda-sm86-bbbbbbbbbbbb",
            "cuda-sm89-cccccccccccc",
        ] {
            std::fs::create_dir_all(root.join(dir)).unwrap();
        }
        let m = manifest(&[entry("cuda-sm86", [8, 6], 0)]);
        prune(&m.backends[0], &root.join("cuda-sm86-bbbbbbbbbbbb"));
        assert!(!root.join("cuda-sm86-aaaaaaaaaaaa").exists());
        assert!(root.join("cuda-sm86-bbbbbbbbbbbb").exists());
        assert!(root.join("cuda-sm89-cccccccccccc").exists());
        std::fs::remove_dir_all(&root).unwrap();
    }
}
