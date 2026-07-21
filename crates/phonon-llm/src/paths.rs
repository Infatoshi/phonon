use std::path::{Path, PathBuf};

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

pub fn polish_prompt(root: &Path) -> PathBuf {
    root.join("prompts/polish_v1.txt")
}

pub fn fluid1_paths() -> Option<(PathBuf, PathBuf, Option<PathBuf>)> {
    if std::env::var_os("PHONON_DISABLE_LLM").is_some() {
        return None;
    }
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
