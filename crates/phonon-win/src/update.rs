//! The update card for Windows and Linux: tell the user once when a newer release is out, and offer
//! its download page from the tray. Phonon never downloads or installs anything itself. The Mac
//! counterpart is bar/Sources/UpdateCheck.swift (SPEC "Update check").
//!
//! - When: 60 s after start, then every 24 h while Phonon runs, if `check_for_updates` in
//!   settings.json is not false (the Mac's key and default).
//! - What is sent: one unauthenticated GET to GitHub's public releases API with User-Agent
//!   `Phonon`. No identifier, version or usage; GitHub sees the address.
//! - Which release counts: not a draft, carrying this platform's download (the installer
//!   `Phonon-Setup-*.exe` on Windows, a `.deb` on Linux), tagged `vX.Y.Z` or
//!   `vX.Y.Z-win-beta.N`, and newer than the build's own tag. A stable `vX.Y.Z` is newer than any
//!   beta of the same X.Y.Z.
//! - The build's own tag comes from `PHONON_RELEASE_TAG` at compile time (the release build sets
//!   it, for example `v1.0.3-win-beta.6`). A build without it never checks, the way a bare Mac
//!   binary without an Info.plist never checks.
//! - How it tells the user: one notification per version (`update_announced`, the Mac's key),
//!   and "Download Phonon X..." in the tray menu from then on.

// The tray apps exist on Windows and Linux only; the module still builds (and tests) elsewhere.
#![cfg_attr(not(any(windows, target_os = "linux")), allow(dead_code))]

use std::sync::Mutex;
use std::time::Duration;

use anyhow::{bail, Result};
use serde::Deserialize;

/// The releases the check reads (the Mac reads the same list).
pub const RELEASES_URL: &str = "https://api.github.com/repos/Infatoshi/phonon/releases?per_page=20";
/// Long enough that the check never competes with the first model load.
pub const FIRST_CHECK_DELAY: Duration = Duration::from_secs(60);
pub const INTERVAL: Duration = Duration::from_secs(24 * 60 * 60);
/// settings.json keys, shared with the Mac.
pub const CHECK_KEY: &str = "check_for_updates";
pub const ANNOUNCED_KEY: &str = "update_announced";

/// A release newer than the running build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Available {
    pub tag: String,
    /// "1.0.4" or "1.0.4 beta 7", for the menu and the notification.
    pub title: String,
    pub page: String,
}

/// What the tray offers once a check found a newer release.
static AVAILABLE: Mutex<Option<Available>> = Mutex::new(None);

/// The newer release the last check found, if any.
pub fn available() -> Option<Available> {
    AVAILABLE.lock().ok().and_then(|found| found.clone())
}

/// The tag this build was released as, or None for a development build.
pub fn running_tag() -> Option<&'static str> {
    option_env!("PHONON_RELEASE_TAG").filter(|tag| Version::parse(tag).is_some())
}

/// A release tag as an ordered value: (major, minor, patch, beta), where a stable release ranks
/// above every beta of the same version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version {
    numbers: [u32; 3],
    /// u32::MAX for a stable release.
    beta: u32,
}

impl Version {
    /// "v1.0.3", "1.0.3" or "v1.0.3-win-beta.6"; anything else (a Mac-only suffix, a typo) is None.
    pub fn parse(tag: &str) -> Option<Self> {
        let text = tag.strip_prefix('v').unwrap_or(tag);
        let (core, beta) = match text.split_once("-win-beta.") {
            Some((core, beta)) => (core, Some(beta)),
            None => (text, None),
        };
        let mut numbers = [0u32; 3];
        let mut parts = core.split('.');
        for slot in &mut numbers {
            let part = parts.next()?;
            if part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            *slot = part.parse().ok()?;
        }
        if parts.next().is_some() {
            return None;
        }
        let beta = match beta {
            Some(n) if !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) => n.parse().ok()?,
            Some(_) => return None,
            None => u32::MAX,
        };
        Some(Self { numbers, beta })
    }

    pub fn title(&self) -> String {
        let [a, b, c] = self.numbers;
        if self.beta == u32::MAX {
            format!("{a}.{b}.{c}")
        } else {
            format!("{a}.{b}.{c} beta {}", self.beta)
        }
    }
}

#[derive(Deserialize)]
struct Release {
    tag_name: String,
    html_url: String,
    #[serde(default)]
    draft: bool,
    #[serde(default)]
    assets: Vec<Asset>,
}

#[derive(Deserialize)]
struct Asset {
    name: String,
}

/// Whether a release carries this platform's download.
fn has_download(release: &Release) -> bool {
    release.assets.iter().any(|asset| {
        if cfg!(windows) {
            asset.name.starts_with("Phonon-Setup") && asset.name.ends_with(".exe")
        } else {
            asset.name.ends_with("_amd64.deb")
        }
    })
}

/// The newest release in the API's JSON above `running`, if any.
pub fn newest(json: &str, running: &str) -> Result<Option<Available>> {
    let Some(current) = Version::parse(running) else {
        return Ok(None);
    };
    let releases: Vec<Release> = serde_json::from_str(json)?;
    let best = releases
        .iter()
        .filter(|release| !release.draft && has_download(release))
        .filter_map(|release| Version::parse(&release.tag_name).map(|v| (v, release)))
        .max_by_key(|(version, _)| *version);
    Ok(best
        .filter(|(version, _)| *version > current)
        .map(|(version, release)| Available {
            tag: release.tag_name.clone(),
            title: version.title(),
            page: release.html_url.clone(),
        }))
}

/// One check against GitHub, comparing with `running` (a release tag).
pub fn fetch(running: &str) -> Result<Option<Available>> {
    // the program's agent carries the system TLS; the header replaces its versioned User-Agent
    let response = crate::http::agent()
        .get(RELEASES_URL)
        .timeout(Duration::from_secs(20))
        .set("Accept", "application/vnd.github+json")
        .set("User-Agent", "Phonon")
        .call()?;
    if response.status() != 200 {
        bail!("releases API answered {}", response.status());
    }
    newest(&response.into_string()?, running)
}

fn checks_enabled() -> bool {
    crate::settings::Settings::load()
        .get_bool(CHECK_KEY)
        .unwrap_or(true)
}

fn announced() -> Option<String> {
    crate::settings::Settings::load().get_str(ANNOUNCED_KEY)
}

/// Start the background check (a release build only). `notify` shows the one notification per
/// version; it runs on the checking thread.
pub fn start(notify: fn(&Available)) {
    let Some(running) = running_tag() else { return };
    std::thread::spawn(move || {
        std::thread::sleep(FIRST_CHECK_DELAY);
        loop {
            if checks_enabled() {
                match fetch(running) {
                    Ok(found) => {
                        if let Some(found) = &found {
                            if announced().as_deref() != Some(found.tag.as_str()) {
                                notify(found);
                                let _ = crate::settings::set(
                                    ANNOUNCED_KEY,
                                    serde_json::Value::String(found.tag.clone()),
                                );
                            }
                        }
                        if let Ok(mut slot) = AVAILABLE.lock() {
                            *slot = found;
                        }
                    }
                    Err(error) => eprintln!("phonon: update check failed: {error:#}"),
                }
            }
            std::thread::sleep(INTERVAL);
        }
    });
}

/// Open a release page in the default browser.
pub fn open_page(url: &str) {
    if !url.starts_with("https://github.com/") {
        return;
    }
    #[cfg(windows)]
    let _ = std::process::Command::new("explorer").arg(url).spawn();
    #[cfg(not(windows))]
    let _ = std::process::Command::new("xdg-open").arg(url).spawn();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn release(tag: &str, asset: &str, draft: bool) -> String {
        format!(
            r#"{{"tag_name":"{tag}","html_url":"https://github.com/Infatoshi/phonon/releases/tag/{tag}","draft":{draft},"prerelease":true,"assets":[{{"name":"{asset}"}}]}}"#
        )
    }

    fn download() -> &'static str {
        if cfg!(windows) {
            "Phonon-Setup-1.0.0.exe"
        } else {
            "phonon_1.0.4-beta7_amd64.deb"
        }
    }

    #[test]
    fn tags_order_betas_below_their_stable_release() {
        let v = |t| Version::parse(t).unwrap();
        assert!(v("v1.0.3-win-beta.6") < v("v1.0.3-win-beta.7"));
        assert!(v("v1.0.3-win-beta.7") < v("v1.0.3"));
        assert!(v("v1.0.3") < v("v1.0.4-win-beta.1"));
        assert!(v("v1.0.10") > v("v1.0.9"));
        assert_eq!(v("v1.0.3-win-beta.6").title(), "1.0.3 beta 6");
        assert_eq!(v("1.2.0").title(), "1.2.0");
        for bad in [
            "v1.0",
            "v1.0.3-beta.6",
            "v1.0.3-win-beta.",
            "x1.0.3",
            "v1.0.3.4",
            "",
        ] {
            assert!(Version::parse(bad).is_none(), "{bad}");
        }
    }

    #[test]
    fn the_newest_release_with_this_platforms_download_wins() {
        let json = format!(
            "[{},{},{},{}]",
            release("v1.0.4-win-beta.7", download(), false),
            release("v1.0.5-win-beta.1", download(), true),
            release("v1.0.9", "Phonon.dmg", false),
            release("v1.0.3-win-beta.6", download(), false),
        );
        let found = newest(&json, "v1.0.3-win-beta.6").unwrap().unwrap();
        assert_eq!(found.tag, "v1.0.4-win-beta.7");
        assert_eq!(found.title, "1.0.4 beta 7");
        assert!(found.page.ends_with("/v1.0.4-win-beta.7"));
        // the running build is already the newest
        assert_eq!(newest(&json, "v1.0.4-win-beta.7").unwrap(), None);
        // a development build has no tag to compare
        assert_eq!(newest(&json, "dev").unwrap(), None);
    }

    #[test]
    fn a_development_build_never_checks() {
        // CI and local builds carry no PHONON_RELEASE_TAG
        if option_env!("PHONON_RELEASE_TAG").is_none() {
            assert_eq!(running_tag(), None);
        }
    }

    #[test]
    fn only_github_pages_open() {
        // nothing to assert on the spawn itself; the guard is the point
        open_page("file:///etc/passwd");
        open_page("https://example.com/");
    }
}
