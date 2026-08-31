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
pub const LLAMA_BUILD: &str = "b10726";
pub const ASR_MODEL_ID: &str = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8";
pub const ASR_MODEL_REVISION: &str = "1ab9323565ddb038682214b292f588070a538ce2";
pub const POLISH_MODEL_ID: &str = "google/gemma-4-E2B-it-qat-q4_0-gguf";
pub const POLISH_MODEL_REVISION: &str = "675cff42a74c774d6cb76f76d8eacb49b48c9b93";
/// The GGUF file name inside the correction model directory.
pub const POLISH_GGUF: &str = "gemma-4-E2B_q4_0-it.gguf";

/// sherpa-onnx command line tools and the ONNX Runtime they load.
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
pub const LLAMA_RUNTIME: Component = Component {
    dir: "runtime/llama-b10726",
    label: "correction runtime",
    assets: &[Asset {
        name: "llama-win-cpu-x64.zip",
        url: "https://github.com/ggml-org/llama.cpp/releases/download/b10726/llama-b10726-bin-win-cpu-x64.zip",
        sha256: "92f527b1b1ccf30c7d87fddb583294ef995d1f249d183e79f6ef2c51b5d7c40d",
        bytes: 18_367_585,
    }],
    unpack: Unpack::Archive { strip_to: "" },
    sentinel: "llama-server.exe",
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

/// Gemma 4 E2B instruction tuned, Google's own quantisation-aware q4_0 GGUF.
/// This is the same model family the macOS build corrects with.
pub const POLISH_MODEL: Component = Component {
    dir: "models/gemma-4-E2B-it-qat-q4_0",
    label: "correction model",
    assets: &[Asset {
        name: POLISH_GGUF,
        url: "https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/resolve/675cff42a74c774d6cb76f76d8eacb49b48c9b93/gemma-4-E2B_q4_0-it.gguf",
        sha256: "fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634",
        bytes: 3_349_516_256,
    }],
    unpack: Unpack::None,
    sentinel: POLISH_GGUF,
};

/// Everything first run must fetch, in the order it is fetched. Runtimes come
/// first because they are small: a user with a slow link sees progress quickly.
pub const ALL: &[Component] = &[SHERPA_RUNTIME, LLAMA_RUNTIME, ASR_MODEL, POLISH_MODEL];

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
    fn first_run_download_is_about_four_gigabytes() {
        let total = total_bytes();
        assert!(
            (3_500_000_000..4_500_000_000).contains(&total),
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
