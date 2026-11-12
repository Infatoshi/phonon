//! The writing mode the user picked from the tray menu.
//!
//! The refiner rewrites a transcript in one of its four modes (`polish::MODES`).
//! The Mac follows the focused app; Windows and Linux do not know the focused app
//! yet, so the user picks one mode from the tray's "Writing mode" submenu, and
//! every dictation uses it. The pick is `writing_style` in `settings.json` (see
//! settings.rs; `writing_mode.txt`, the file before it, is the fallback). It is read
//! at each dictation, so a pick from the tray or the settings window applies from
//! the next one without a restart.
//!
//! The startup check and the warm-up stay on `polish::DEFAULT_MODE`: they prove
//! the model against a fixed expected answer.

use std::path::Path;

use anyhow::{anyhow, Result};

use crate::polish::{DEFAULT_MODE, MODES};

/// The legacy file name in the data folder.
#[cfg(test)]
const FILE: &str = crate::settings::LEGACY_MODE;

/// The tray menu's names, in `MODES` order (the Mac's titles, bar/Sources/AppData.swift).
pub fn label(mode: &str) -> &'static str {
    match mode {
        "casual" => "Casual",
        "formal" => "Formal",
        "prompt" => "Prompt",
        _ => "Minimal",
    }
}

/// A submenu item's text: the title, and "(default)" on minimal.
pub fn menu_label(mode: &str) -> String {
    if mode == DEFAULT_MODE {
        format!("{} (default)", label(mode))
    } else {
        label(mode).to_string()
    }
}

/// The mode names the menu offers, in `MODES` order.
pub fn names() -> impl Iterator<Item = &'static str> {
    MODES.iter().map(|(name, _)| *name)
}

/// A saved setting as one of the refiner's mode names. Case and surrounding
/// whitespace are ignored; anything else is `None`.
pub fn parse(text: &str) -> Option<&'static str> {
    let wanted = text.trim().to_ascii_lowercase();
    names().find(|name| *name == wanted)
}

/// The mode saved at `path`: minimal when the file is missing, unreadable or
/// names no mode.
pub fn load_from(path: &Path) -> &'static str {
    let Ok(text) = std::fs::read_to_string(path) else {
        return DEFAULT_MODE;
    };
    parse(&text).unwrap_or_else(|| {
        eprintln!(
            "phonon: {} names no writing mode; using {DEFAULT_MODE}",
            path.display()
        );
        DEFAULT_MODE
    })
}

/// Save `mode` at `path` in the legacy format, creating the folder.
#[cfg(test)]
pub fn save_to(path: &Path, mode: &str) -> Result<()> {
    let mode = parse(mode).ok_or_else(|| anyhow!("not a writing mode: {mode:?}"))?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, mode)?;
    Ok(())
}

/// The mode the next dictation uses.
pub fn current() -> &'static str {
    crate::settings::Settings::load().writing_style()
}

/// Use `mode` from the next dictation on, and remember it across restarts.
pub fn choose(mode: &str) -> Result<()> {
    let mode = parse(mode).ok_or_else(|| anyhow!("not a writing mode: {mode:?}"))?;
    crate::settings::set(crate::settings::WRITING_STYLE, mode.into())?;
    eprintln!("phonon: writing mode set to {mode}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polish::{mode_line, prompt};
    use std::path::PathBuf;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "phonon-win-mode-test-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn parses_the_four_modes_and_nothing_else() {
        assert_eq!(parse("minimal"), Some("minimal"));
        assert_eq!(parse("  Casual\n"), Some("casual"));
        assert_eq!(parse("FORMAL"), Some("formal"));
        assert_eq!(parse("prompt\r\n"), Some("prompt"));
        // the Mac's "follow the app" is not offered here yet
        assert_eq!(parse("app"), None);
        assert_eq!(parse(""), None);
        assert_eq!(parse("shouting"), None);
    }

    #[test]
    fn the_menu_lists_every_mode_in_order_with_the_macs_titles() {
        let labels: Vec<_> = names().map(label).collect();
        assert_eq!(labels, ["Minimal", "Casual", "Formal", "Prompt"]);
        assert_eq!(menu_label("minimal"), "Minimal (default)");
        assert_eq!(menu_label("prompt"), "Prompt");
    }

    #[test]
    fn a_missing_or_bad_file_means_minimal() {
        let dir = scratch("missing");
        let path = dir.join(FILE);
        assert_eq!(load_from(&path), "minimal");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&path, "loud").unwrap();
        assert_eq!(load_from(&path), "minimal");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_saved_pick_survives_a_reload_and_replaces_the_last() {
        let dir = scratch("saved");
        let path = dir.join("nested").join(FILE);
        save_to(&path, "Formal").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "formal");
        assert_eq!(load_from(&path), "formal");
        save_to(&path, "casual").unwrap();
        assert_eq!(load_from(&path), "casual");
        assert!(save_to(&path, "app").is_err());
        assert_eq!(load_from(&path), "casual");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_picked_mode_reaches_the_prompt_byte_for_byte() {
        let dir = scratch("prompt");
        let path = dir.join(FILE);
        for (name, line) in MODES {
            save_to(&path, name).unwrap();
            let mode = load_from(&path);
            assert_eq!(mode_line(mode), line);
            let p = prompt("hello there", mode);
            assert!(
                p.contains(&format!("do not add or drop content.\n{line}<|im_end|>")),
                "{name}: {p}"
            );
            for (other, other_line) in MODES {
                if other != name {
                    assert!(!p.contains(other_line), "{name} prompt carries {other}");
                }
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
