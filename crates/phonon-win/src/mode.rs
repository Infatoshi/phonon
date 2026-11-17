//! Focused-app writing mode for transcript refinement.
//!
//! The refiner rewrites a transcript in one of its four modes (`polish::MODES`):
//! `casual` for chat apps, `formal` for email, `prompt` for AI tools and
//! terminals/editors, and `minimal` for documents and everything else. Windows
//! and Linux infer the mode from the focused foreground window and title,
//! falling back to `minimal` when unknown.
//!
//! The startup check and the warm-up stay on `polish::DEFAULT_MODE`: they prove
//! the model against a fixed expected answer.

use std::path::Path;

use crate::polish::{DEFAULT_MODE, MODES};

/// The legacy file name in the data folder.
#[cfg(test)]
const FILE: &str = crate::settings::LEGACY_MODE;

/// The mode names in `MODES` order.
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
/// names no mode. Kept for legacy `settings.rs` fallback decoding.
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

/// The writing mode for a focused window's executable/class name and optional title.
pub fn for_app(app_name: Option<&str>, window_title: Option<&str>) -> &'static str {
    let app = app_name
        .map(|s| {
            let trimmed = s.trim().to_ascii_lowercase();
            trimmed.strip_suffix(".exe").unwrap_or(&trimmed).to_string()
        })
        .unwrap_or_default();
    let title = window_title
        .map(|s| s.trim().to_ascii_lowercase())
        .unwrap_or_default();

    const BROWSERS: &[&str] = &[
        "chrome", "msedge", "firefox", "brave", "opera", "vivaldi", "arc", "chromium", "zen",
        "iexplore",
    ];
    if BROWSERS.iter().any(|b| app == *b) {
        return for_browser_title(&title);
    }

    const CASUAL_APPS: &[&str] = &[
        "slack",
        "discord",
        "teams",
        "ms-teams",
        "whatsapp",
        "whatsapp.root",
        "telegram",
        "signal",
        "messenger",
        "beeper",
        "element",
        "mattermost",
        "zulip",
        "line",
        "wechat",
    ];
    if CASUAL_APPS.iter().any(|a| app == *a) {
        return "casual";
    }

    const FORMAL_APPS: &[&str] = &[
        "outlook",
        "olk",
        "thunderbird",
        "mailspring",
        "bluemail",
        "protonmail",
    ];
    if FORMAL_APPS.iter().any(|a| app == *a) {
        return "formal";
    }

    const PROMPT_APPS: &[&str] = &[
        "claude",
        "chatgpt",
        "perplexity",
        "windowsterminal",
        "wt",
        "powershell",
        "pwsh",
        "cmd",
        "alacritty",
        "wezterm",
        "wezterm-gui",
        "kitty",
        "ghostty",
        "foot",
        "konsole",
        "gnome-terminal",
        "gnome-terminal-server",
        "xterm",
        "terminator",
        "tilix",
        "hyper",
        "tabby",
        "code",
        "cursor",
        "windsurf",
        "zed",
        "idea64",
        "pycharm64",
        "clion64",
        "goland64",
        "webstorm64",
        "rustrover64",
        "lapis",
    ];
    if PROMPT_APPS.iter().any(|a| app == *a) {
        return "prompt";
    }

    DEFAULT_MODE
}

fn for_browser_title(title: &str) -> &'static str {
    if title.is_empty() {
        return DEFAULT_MODE;
    }
    for kw in [
        "slack",
        "discord",
        "whatsapp",
        "telegram",
        "signal",
        "messenger",
        "microsoft teams",
        "google messages",
    ] {
        if title.contains(kw) {
            return "casual";
        }
    }
    for kw in [
        "gmail",
        "outlook",
        "proton mail",
        "superhuman",
        "fastmail",
        "yahoo mail",
        "roundcube",
    ] {
        if title.contains(kw) {
            return "formal";
        }
    }
    for kw in [
        "chatgpt",
        "claude",
        "perplexity",
        "gemini",
        "copilot",
        "grok",
        "github",
        "gitlab",
        "bitbucket",
        "stackoverflow",
        "replit",
        "codesandbox",
    ] {
        if title.contains(kw) {
            return "prompt";
        }
    }
    DEFAULT_MODE
}

/// The writing mode for the currently focused application.
pub fn current() -> &'static str {
    let (app, title) = focused_app();
    for_app(app.as_deref(), title.as_deref())
}

#[cfg(windows)]
fn focused_app() -> (Option<String>, Option<String>) {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{
        OpenProcess, QueryFullProcessImageNameW, PROCESS_QUERY_LIMITED_INFORMATION,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GetForegroundWindow, GetWindowTextW, GetWindowThreadProcessId,
    };

    unsafe {
        let hwnd = GetForegroundWindow();
        if hwnd.is_null() {
            return (None, None);
        }
        let mut title_buf = [0u16; 512];
        let title_len = GetWindowTextW(hwnd, title_buf.as_mut_ptr(), title_buf.len() as i32);
        let title = if title_len > 0 {
            Some(String::from_utf16_lossy(&title_buf[..title_len as usize]))
        } else {
            None
        };

        let mut pid = 0u32;
        GetWindowThreadProcessId(hwnd, &mut pid);
        if pid == 0 {
            return (None, title);
        }
        let process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if process.is_null() {
            return (None, title);
        }
        let mut path_buf = [0u16; 512];
        let mut path_len = path_buf.len() as u32;
        let ok = QueryFullProcessImageNameW(process, 0, path_buf.as_mut_ptr(), &mut path_len);
        CloseHandle(process);
        let app = if ok != 0 && path_len > 0 {
            let full = String::from_utf16_lossy(&path_buf[..path_len as usize]);
            std::path::Path::new(&full)
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.to_string())
        } else {
            None
        };
        (app, title)
    }
}

#[cfg(target_os = "linux")]
fn focused_app() -> (Option<String>, Option<String>) {
    if std::env::var_os("HYPRLAND_INSTANCE_SIGNATURE").is_some() {
        if let Ok(out) = std::process::Command::new("hyprctl")
            .args(["activewindow", "-j"])
            .output()
        {
            if out.status.success() {
                if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&out.stdout) {
                    let app = v
                        .get("class")
                        .and_then(|s| s.as_str())
                        .map(|s| s.to_string());
                    let title = v
                        .get("title")
                        .and_then(|s| s.as_str())
                        .map(|s| s.to_string());
                    if app.is_some() || title.is_some() {
                        return (app, title);
                    }
                }
            }
        }
    }
    if std::env::var_os("DISPLAY").is_some() {
        if let Ok(root) = std::process::Command::new("xprop")
            .args(["-root", "_NET_ACTIVE_WINDOW"])
            .output()
        {
            let text = String::from_utf8_lossy(&root.stdout);
            if let Some(id) = text
                .split_whitespace()
                .last()
                .filter(|s| s.starts_with("0x"))
            {
                if let Ok(info) = std::process::Command::new("xprop")
                    .args(["-id", id, "WM_CLASS", "_NET_WM_NAME"])
                    .output()
                {
                    let info_text = String::from_utf8_lossy(&info.stdout);
                    let mut app = None;
                    let mut title = None;
                    for line in info_text.lines() {
                        if line.starts_with("WM_CLASS") {
                            app = line.rsplit('"').nth(1).map(|s| s.to_string());
                        } else if line.starts_with("_NET_WM_NAME") {
                            title = line.split('"').nth(1).map(|s| s.to_string());
                        }
                    }
                    if app.is_some() || title.is_some() {
                        return (app, title);
                    }
                }
            }
        }
    }
    (None, None)
}

#[cfg(not(any(windows, target_os = "linux")))]
fn focused_app() -> (Option<String>, Option<String>) {
    (None, None)
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
        assert_eq!(parse("app"), None);
        assert_eq!(parse(""), None);
        assert_eq!(parse("shouting"), None);
    }

    #[test]
    fn infers_writing_mode_from_focused_app() {
        assert_eq!(for_app(Some("Slack.exe"), None), "casual");
        assert_eq!(for_app(Some("Discord"), None), "casual");
        assert_eq!(for_app(Some("OUTLOOK.EXE"), None), "formal");
        assert_eq!(for_app(Some("WindowsTerminal.exe"), None), "prompt");
        assert_eq!(for_app(Some("code"), None), "prompt");
        assert_eq!(for_app(Some("claude.exe"), None), "prompt");
        assert_eq!(for_app(Some("Obsidian.exe"), None), "minimal");
        assert_eq!(for_app(Some("chrome.exe"), Some("Gmail - Inbox")), "formal");
        assert_eq!(
            for_app(Some("chrome"), Some("Slack | engineering")),
            "casual"
        );
        assert_eq!(for_app(Some("msedge.exe"), Some("Claude")), "prompt");
        assert_eq!(for_app(Some("chrome.exe"), Some("Example Page")), "minimal");
        assert_eq!(for_app(None, None), "minimal");
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
    fn all_inferred_modes_reach_the_prompt_byte_for_byte() {
        for (name, line) in MODES {
            assert_eq!(mode_line(name), line);
            let p = prompt("hello there", name);
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
    }
}
