//! Shared filesystem paths for ASR sidecar + fluid-1 polish.

use std::path::PathBuf;

pub fn project_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest.join("../..");
    if workspace.join("sidecar/asr_server.py").is_file() {
        return workspace.canonicalize().unwrap_or(workspace);
    }
    // cargo install still bakes CARGO_MANIFEST_DIR at compile time; fall back
    // to common personal checkout if the binary is relocated later.
    let home = std::env::var_os("HOME").map(PathBuf::from);
    if let Some(h) = home {
        let p = h.join("dev/tools/phonon");
        if p.join("sidecar/asr_server.py").is_file() {
            return p;
        }
    }
    std::env::current_dir().unwrap_or(manifest)
}
