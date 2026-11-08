//! What Phonon for Windows downloads on first run, and how it proves the bytes.
//!
//! No weights and no third-party runtime ship inside the executable. Every entry
//! below is pinned: a GitHub release tag or a Hugging Face commit, plus the exact
//! SHA-256 of the file. A changed upstream file fails verification instead of
//! silently running something else.

/// One file to fetch and check.
#[derive(Debug, Clone, Copy)]
pub struct Asset {
    /// Where the file lands under the component directory.
    pub name: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub bytes: u64,
}

/// How to unpack a downloaded archive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unpack {
    /// Keep the file as it is.
    None,
    /// Expand the archive, then keep only the directory that holds the binaries.
    /// `strip_to` is the path inside the archive that becomes the component root.
    Archive { strip_to: &'static str },
}

/// A named group of files that lands in one directory.
#[derive(Debug, Clone, Copy)]
pub struct Component {
    /// Directory name under the Phonon data root.
    pub dir: &'static str,
    /// Shown to the user while it downloads.
    pub label: &'static str,
    pub assets: &'static [Asset],
    pub unpack: Unpack,
    /// The file that must exist for the component to count as installed.
    pub sentinel: &'static str,
}

impl Component {
    /// Total bytes to download for this component.
    pub fn download_bytes(&self) -> u64 {
        self.assets.iter().map(|asset| asset.bytes).sum()
    }
}

pub const SHERPA_VERSION: &str = "1.13.6";
/// Same llama.cpp as the iPhone and Android builds; b10726 did not know the refiner's `lfm3` architecture.
pub const LLAMA_BUILD: &str = "b12132";
pub const ASR_MODEL_ID: &str = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8";
pub const ASR_MODEL_REVISION: &str = "1ab9323565ddb038682214b292f588070a538ce2";
/// Phonon's public refiner; the repository is the macOS constant.
pub use phonon_llm::POLISH_MODEL_ID;
/// The revision Windows and Linux fetch: the 1.0 upload, which their builds were tested with. macOS 1.0.1 pins
/// a later revision of the same repository (`phonon_llm::POLISH_MODEL_REVISION`); move this one when the
/// Windows and Linux builds are retested with it.
pub const POLISH_MODEL_REVISION: &str = "2b0ab74bee7ab7cf364695bbe44f64ce4c47439e";
/// The GGUF file name inside the correction model directory.
pub const POLISH_GGUF: &str = "phonon-refiner-1.7b-Q4_K_S-e4.gguf";

/// sherpa-onnx command line tools and the ONNX Runtime they load.
#[cfg(not(target_os = "linux"))]
pub const SHERPA_RUNTIME: Component = Component {
    dir: "runtime/sherpa-onnx-1.13.6",
    label: "speech recognition runtime",
    assets: &[Asset {
        name: "sherpa-onnx-win-x64.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.6/sherpa-onnx-v1.13.6-win-x64-shared-MD-Release-no-tts.tar.bz2",
        sha256: "071d6641efd737a1f60de48c9c4cd596f78d5b0980815e8ad3798c95785d2b26",
        bytes: 18_748_740,
    }],
    unpack: Unpack::Archive {
        strip_to: "sherpa-onnx-v1.13.6-win-x64-shared-MD-Release-no-tts",
    },
    sentinel: "bin/sherpa-onnx-offline.exe",
};

/// llama.cpp CPU build. The x64 CPU zip carries every ggml CPU backend variant,
/// so one artifact covers Sandy Bridge through Zen 4.
#[cfg(not(target_os = "linux"))]
// The Vulkan build: every CPU backend of the CPU build plus ggml-vulkan.dll, loaded
// at run time. Without a Vulkan driver that DLL does not load and the server runs on
// the CPU exactly as the CPU build did; with a discrete GPU, polish.rs offloads to it.
pub const LLAMA_RUNTIME: Component = Component {
    dir: "runtime/llama-b12132-vulkan",
    label: "correction runtime",
    assets: &[Asset {
        name: "llama-win-vulkan-x64.zip",
        url: "https://github.com/ggml-org/llama.cpp/releases/download/b12132/llama-b12132-bin-win-vulkan-x64.zip",
        sha256: "a2bbf7c46695bca2d3baa1919c64975061234dc9ea56a571f2f070704bbd5516",
        bytes: 37_678_209,
    }],
    unpack: Unpack::Archive { strip_to: "" },
    sentinel: "llama-server.exe",
};

/// Linux x86_64 tools, including the shared ONNX Runtime in `lib/`.
#[cfg(target_os = "linux")]
pub const SHERPA_RUNTIME: Component = Component {
    dir: "runtime/sherpa-onnx-1.13.6-linux-x64",
    label: "speech recognition runtime",
    assets: &[Asset {
        name: "sherpa-onnx-linux-x64.tar.bz2",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.6/sherpa-onnx-v1.13.6-linux-x64-shared-no-tts.tar.bz2",
        sha256: "089c01bded9166ed53f9e433b030863f9dd5ec2ad067dca73a8ba96dca25979d",
        bytes: 24_503_640,
    }],
    unpack: Unpack::Archive {
        strip_to: "sherpa-onnx-v1.13.6-linux-x64-shared-no-tts",
    },
    sentinel: "bin/sherpa-onnx-offline",
};

/// The pinned Ubuntu CPU release is a tar.gz, not a zip.
#[cfg(target_os = "linux")]
pub const LLAMA_RUNTIME: Component = Component {
    dir: "runtime/llama-b12132-linux-x64",
    label: "correction runtime",
    assets: &[Asset {
        name: "llama-ubuntu-x64.tar.gz",
        url: "https://github.com/ggml-org/llama.cpp/releases/download/b12132/llama-b12132-bin-ubuntu-x64.tar.gz",
        sha256: "8741ab42ede58f3bc3a20f523c50fff8cf6e750373021b0627603f378f2cd78e",
        bytes: 23_083_703,
    }],
    unpack: Unpack::Archive { strip_to: "llama-b12132" },
    sentinel: "llama-server",
};

/// Parakeet TDT 0.6b v2, the same acoustic model the macOS build runs, exported
/// to ONNX and quantised to int8 for CPU.
pub const ASR_MODEL: Component = Component {
    dir: "models/parakeet-tdt-0.6b-v2-int8",
    label: "speech recognition model",
    assets: &[
        Asset {
            name: "encoder.int8.onnx",
            url: "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/resolve/1ab9323565ddb038682214b292f588070a538ce2/encoder.int8.onnx",
            sha256: "a32b12d17bbbc309d0686fbbcc2987b5e9b8333a7da83fa6b089f0a2acd651ab",
            bytes: 652_184_296,
        },
        Asset {
            name: "decoder.int8.onnx",
            url: "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/resolve/1ab9323565ddb038682214b292f588070a538ce2/decoder.int8.onnx",
            sha256: "b6bb64963457237b900e496ee9994b59294526439fbcc1fecf705b31a15c6b4e",
            bytes: 7_257_753,
        },
        Asset {
            name: "joiner.int8.onnx",
            url: "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/resolve/1ab9323565ddb038682214b292f588070a538ce2/joiner.int8.onnx",
            sha256: "7946164367946e7f9f29a122407c3252b680dbae9a51343eb2488d057c3c43d2",
            bytes: 1_739_080,
        },
        Asset {
            name: "tokens.txt",
            url: "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/resolve/1ab9323565ddb038682214b292f588070a538ce2/tokens.txt",
            sha256: "ec182b70dd42113aff6c5372c75cac58c952443eb22322f57bbd7f53977d497d",
            bytes: 9_384,
        },
    ],
    unpack: Unpack::None,
    sentinel: "encoder.int8.onnx",
};

/// The Phonon refiner (fine-tuned LFM3-1.7B), Q4_K_S with 4-bit embeddings and head: the same file the iPhone
/// and Android builds run.
pub const POLISH_MODEL: Component = Component {
    dir: "models/phonon-refiner-1.7b-2b0ab74",
    label: "correction model",
    assets: &[Asset {
        name: POLISH_GGUF,
        url: "https://huggingface.co/Infatoshi/phonon-refiner-1.7b/resolve/2b0ab74bee7ab7cf364695bbe44f64ce4c47439e/phonon-refiner-1.7b-Q4_K_S-e4.gguf",
        sha256: "059e6e503c6c2ec50cb61a39bd0d556fb6d946bb58e965fb9ddb8b0176dfc6cd",
        bytes: 988_453_696,
    }],
    unpack: Unpack::None,
    sentinel: POLISH_GGUF,
};

/// Everything first run must fetch, in the order it is fetched. Runtimes come
/// first because they are small: a user with a slow link sees progress quickly.
pub const ALL: &[Component] = &[SHERPA_RUNTIME, LLAMA_RUNTIME, ASR_MODEL, POLISH_MODEL];

/// Data folders of earlier versions that `fetch::ensure_all` removes once the
/// current components are in place: the CPU-only llama.cpp runtime (Windows,
/// replaced by the Vulkan build, which carries the same CPU backends).
#[cfg(windows)]
pub const SUPERSEDED: &[&str] = &["runtime/llama-b12132"];
#[cfg(not(windows))]
pub const SUPERSEDED: &[&str] = &[];

/// Total first-run download in bytes.
pub fn total_bytes() -> u64 {
    ALL.iter().map(|c| c.download_bytes()).sum()
}

/// A human-sized description of a byte count.
pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A typo in a hash is not caught by any test that runs offline, but a hash
    /// of the wrong length or the wrong alphabet is.
    #[test]
    fn every_hash_is_a_sha256() {
        for component in ALL {
            for asset in component.assets {
                assert_eq!(
                    asset.sha256.len(),
                    64,
                    "{}/{} hash is not 64 hex characters",
                    component.dir,
                    asset.name
                );
                assert!(
                    asset.sha256.chars().all(|c| c.is_ascii_hexdigit()),
                    "{}/{} hash is not hex",
                    component.dir,
                    asset.name
                );
                assert!(
                    asset.sha256.chars().all(|c| !c.is_ascii_uppercase()),
                    "{}/{} hash must be lowercase",
                    component.dir,
                    asset.name
                );
            }
        }
    }

    /// Every URL must name the pinned revision or tag. An unpinned URL would let
    /// an upstream re-upload change what a user runs.
    /// The refiner URL is a literal (const strings cannot be concatenated); it must name the shared constants.
    #[test]
    fn the_refiner_url_matches_the_shared_pin() {
        let url = POLISH_MODEL.assets[0].url;
        assert!(
            url.contains(&format!(
                "/{POLISH_MODEL_ID}/resolve/{POLISH_MODEL_REVISION}/{POLISH_GGUF}"
            )),
            "{url}"
        );
    }

    #[test]
    fn every_url_is_pinned() {
        for component in ALL {
            for asset in component.assets {
                let pinned = asset.url.contains(ASR_MODEL_REVISION)
                    || asset.url.contains(POLISH_MODEL_REVISION)
                    || asset.url.contains(&format!("/v{SHERPA_VERSION}/"))
                    || asset.url.contains(&format!("/{LLAMA_BUILD}/"));
                assert!(pinned, "{} is not pinned to a revision", asset.url);
                assert!(
                    asset.url.starts_with("https://"),
                    "{} is not https",
                    asset.url
                );
                assert!(asset.bytes > 0, "{} has no recorded size", asset.url);
            }
        }
    }

    #[test]
    fn component_directories_are_distinct() {
        let mut seen = Vec::new();
        for component in ALL {
            assert!(!seen.contains(&component.dir), "{} repeats", component.dir);
            seen.push(component.dir);
        }
    }

    /// The README and the website quote this number. Keep it honest.
    #[test]
    fn first_run_download_is_about_one_point_six_gigabytes() {
        let total = total_bytes();
        assert!(
            (1_500_000_000..1_800_000_000).contains(&total),
            "first-run download is {}, update the documented figure",
            human_bytes(total)
        );
    }

    #[test]
    fn human_bytes_reads_naturally() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(1024), "1.0 KB");
        assert_eq!(human_bytes(18_367_585), "17.5 MB");
        assert_eq!(human_bytes(3_349_516_256), "3.1 GB");
    }
}
