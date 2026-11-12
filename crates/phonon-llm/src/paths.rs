use std::path::{Path, PathBuf};

/// Pinned correction model: Phonon's public refiner, fine-tuned from
/// LiquidAI/LFM3-1.7B on public and synthetic transcripts, with writing
/// modes (`phonon_refiner.json` `writing_modes`). One repository holds the
/// native engine's GGUF target and drafter (`phonon_engine.json`) and the MLX
/// weights the sidecar falls back to. The revision is exact so an upstream
/// re-upload cannot change what shipped users run.
/// RELEASE: the revision is the Hugging Face commit of the upload; set it
/// after the upload and before tagging, together with the file hashes in
/// sidecar/model_pins.json (a test checks the two agree).
pub const POLISH_MODEL_ID: &str = "Infatoshi/phonon-refiner-1.7b";
pub const POLISH_MODEL_REVISION: &str = "b20b28075046af94056ab5d5bff89a7abc0f6f50";
/// Pinned runtime, for the same reason: the inline dependency of
/// sidecar/polish_server.py, installed from its hashed lock
/// (`phonon_asr::sidecar_run_args`); this copy is for display, and a test keeps
/// the three in step.
pub const POLISH_RUNTIME_REQUIREMENT: &str = "mlx-lm==0.31.3";

/// The native refiner engine bundled with the app: `pe` next to the running
/// executable (`Contents/Helpers`) and its Metal shader sources in
/// `<root>/engine` (`Contents/Resources/engine`). `None` outside a packaged
/// app; the sidecar then runs the MLX weights.
pub fn bundled_engine(root: &Path) -> Option<(PathBuf, PathBuf)> {
    let binary = std::env::current_exe().ok()?.parent()?.join("pe");
    let shaders = root.join("engine");
    (binary.is_file() && shaders.join("mk.metal").is_file() && shaders.join("df.metal").is_file())
        .then_some((binary, shaders))
}

pub fn polish_script(root: &Path) -> PathBuf {
    root.join("sidecar/polish_server.py")
}

pub fn polish_prompt(root: &Path) -> PathBuf {
    root.join("prompts/polish_v2.txt")
}

/// Whether the correction stage's own files are present. The stage itself is
/// mandatory; this is not a switch for running without it.
pub fn polish_available(root: &Path) -> bool {
    polish_script(root).is_file() && polish_prompt(root).is_file()
}

// ── FluidVoice baseline (developer comparison only) ─────────────────────────
// Not part of the shipped product. `phonon bench` and `phonon profile` can
// still measure the previous fluid-1 helper when it is installed locally, so
// Gemma-versus-Fluid stays measurable on the same fixtures.

pub fn fluid_helper() -> PathBuf {
    PathBuf::from("/Applications/FluidVoice.app/Contents/Helpers/fluid-intelligence-mlx")
}

pub fn fluid_model_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("HOME").unwrap_or_default()
            + "/Library/Application Support/FluidIntelligence/Models/fluid-1-nvfp4-mlx",
    )
}

pub fn fluid_drafter_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("HOME").unwrap_or_default()
            + "/Library/Application Support/FluidIntelligence/Models/gemma-4-E2B-it-qat-assistant-bf16-mlx-mtp",
    )
}

pub fn fluid1_paths() -> Option<(PathBuf, PathBuf, Option<PathBuf>)> {
    let helper = fluid_helper();
    let model = fluid_model_dir();
    if !(helper.is_file() && model.is_dir()) {
        return None;
    }
    let drafter = fluid_drafter_dir();
    let drafter = drafter.is_dir().then_some(drafter);
    Some((helper, model, drafter))
}

pub fn fluid1_available() -> bool {
    fluid1_paths().is_some()
}
