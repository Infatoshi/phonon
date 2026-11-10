//! Where a dictation is going: the app, the browser site, and a coarse
//! category the correction model can use to pick a register (a Slack reply is
//! not an email is not a Claude Code prompt).
//!
//! The bar sends the frontmost app's bundle id and name and, for a browser
//! tab, the page host it read through Accessibility (`AXWebArea.AXURL`). The
//! category lists follow Wispr Flow 1.6's `app-category-constants` (module
//! 59725: messaging, email, AI, terminal, document bundle ids and hosts),
//! regrouped into Phonon's six categories. A host decides before a bundle id:
//! Chrome on `mail.google.com` is email, Chrome on an unknown site is other.

use crate::data::DictionaryFile;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AppCategory {
    /// Personal chat (Discord, iMessage, WhatsApp): casual.
    Chat,
    /// Work chat (Slack, Teams, Google Chat): formal, like email.
    WorkChat,
    Email,
    /// Editors, IDEs and terminals (including agent CLIs run in them).
    Code,
    /// Chat assistants and AI prompt boxes.
    AiPrompt,
    Docs,
    #[default]
    Other,
}

impl AppCategory {
    pub fn as_str(self) -> &'static str {
        match self {
            AppCategory::Chat => "chat",
            AppCategory::WorkChat => "work_chat",
            AppCategory::Email => "email",
            AppCategory::Code => "code",
            AppCategory::AiPrompt => "ai_prompt",
            AppCategory::Docs => "docs",
            AppCategory::Other => "other",
        }
    }
}

/// The app a dictation was inserted into, as kept in the corpus entry. Only
/// the host of a page is kept, never its path or query.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppContext {
    #[serde(default)]
    pub bundle_id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub app_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url_host: Option<String>,
    #[serde(default)]
    pub category: AppCategory,
}

impl AppContext {
    /// Read the `app` object of a polish request. The category is always
    /// recomputed here so the bar and the corpus cannot disagree.
    pub fn from_request(app: &Value) -> Option<Self> {
        let bundle_id = app.get("bundle_id").and_then(Value::as_str).unwrap_or("");
        let app_name = app.get("name").and_then(Value::as_str).unwrap_or("");
        let url_host = app
            .get("url_host")
            .and_then(Value::as_str)
            .and_then(normalize_host);
        if bundle_id.is_empty() && url_host.is_none() {
            return None;
        }
        Some(Self {
            bundle_id: bundle_id.to_string(),
            app_name: app_name.to_string(),
            category: categorize(bundle_id, url_host.as_deref()),
            url_host,
        })
    }
}

/// Lowercased host without `www.`, or None for anything that is not a host.
pub fn normalize_host(host: &str) -> Option<String> {
    let host = host.trim().trim_end_matches('.').to_ascii_lowercase();
    let host = host.strip_prefix("www.").unwrap_or(&host);
    let valid = !host.is_empty()
        && host.contains('.')
        && host
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-');
    valid.then(|| host.to_string())
}

fn host_matches(host: &str, pattern: &str) -> bool {
    host == pattern || host.ends_with(&format!(".{pattern}"))
}

fn bundle_matches(bundle: &str, pattern: &str) -> bool {
    match pattern.strip_suffix('*') {
        Some(prefix) => bundle.starts_with(prefix),
        None => bundle == pattern,
    }
}

const WORK_CHAT_HOSTS: &[&str] = &[
    "slack.com",
    "teams.microsoft.com",
    "teams.live.com",
    "chat.google.com",
    "zulip.com",
    "app.intercom.com",
];
const CHAT_HOSTS: &[&str] = &[
    "discord.com",
    "web.whatsapp.com",
    "messages.google.com",
    "web.telegram.org",
    "messenger.com",
    "x.com",
    "twitter.com",
    "linkedin.com",
    "reddit.com",
    "app.beeper.com",
    "chat.beeper.com",
];
const EMAIL_HOSTS: &[&str] = &[
    "mail.google.com",
    "inbox.google.com",
    "outlook.live.com",
    "outlook.office.com",
    "outlook.office365.com",
    "mail.superhuman.com",
    "app.hey.com",
    "app.fastmail.com",
    "mail.proton.me",
    "mail.yahoo.com",
    "mail.zoho.com",
    "app.shortwave.com",
    "app.frontapp.com",
];
const AI_HOSTS: &[&str] = &[
    "chatgpt.com",
    "chat.openai.com",
    "claude.ai",
    "gemini.google.com",
    "aistudio.google.com",
    "perplexity.ai",
    "perplexity.com",
    "chat.deepseek.com",
    "chat.mistral.ai",
    "grok.com",
    "copilot.microsoft.com",
    "poe.com",
    "t3.chat",
    "notebooklm.google.com",
    "console.anthropic.com",
    "platform.openai.com",
    "openrouter.ai",
    "v0.dev",
    "v0.app",
    "bolt.new",
    "lovable.dev",
];
const CODE_HOSTS: &[&str] = &[
    "github.com",
    "gist.github.com",
    "gitlab.com",
    "bitbucket.org",
    "github.dev",
    "vscode.dev",
    "colab.research.google.com",
    "replit.com",
    "codesandbox.io",
    "stackoverflow.com",
];
const DOCS_HOSTS: &[&str] = &[
    "docs.google.com",
    "notion.so",
    "notion.site",
    "coda.io",
    "quip.com",
    "paper.dropbox.com",
    "atlassian.net",
    "linear.app",
    "overleaf.com",
    "dropbox.com",
    "onedrive.live.com",
    "office.com",
    "hackmd.io",
];

const WORK_CHAT_BUNDLES: &[&str] = &[
    "com.tinyspeck.slackmacgap",
    "com.slack.Slack",
    "com.microsoft.teams",
    "com.microsoft.teams2",
];
const CHAT_BUNDLES: &[&str] = &[
    "com.hnc.Discord",
    "com.discordapp.Discord",
    "org.whispersystems.signal-desktop",
    "ru.keepcoder.Telegram",
    "com.tdesktop.Telegram",
    "net.whatsapp.WhatsApp",
    "desktop.WhatsApp",
    "com.apple.MobileSMS",
    "com.facebook.archon",
    "im.beeper",
    "com.tencent.xinWeChat",
    "com.kishanbagaria.jack",
    "com.linecorp.line",
];
const EMAIL_BUNDLES: &[&str] = &[
    "com.apple.mail",
    "com.microsoft.Outlook",
    "com.superhuman.electron",
    "com.mimestream.Mimestream",
    "com.readdle.SparkDesktop",
    "com.readdle.smartemail-Mac",
    "it.bloop.airmail2",
    "com.hey.app.desktop",
    "org.mozilla.thunderbird",
    "com.frontapp.Front",
    "io.canarymail.mac",
    "com.missiveapp.osx",
];
const AI_BUNDLES: &[&str] = &[
    "com.openai.chat",
    "com.openai.codex",
    "com.anthropic.claudefordesktop",
    "ai.perplexity.perplexity-mac",
    "com.google.GeminiMacOS",
    "ai.elementlabs.lmstudio",
];
const CODE_BUNDLES: &[&str] = &[
    "com.apple.Terminal",
    "com.googlecode.iterm2",
    "dev.warp.Warp-Stable",
    "com.mitchellh.ghostty",
    "net.kovidgoyal.kitty",
    "io.alacritty",
    "com.github.wez.wezterm",
    "co.zeit.hyper",
    "com.microsoft.VSCode",
    "com.microsoft.VSCodeInsiders",
    "com.todesktop.230313mzl4w4u92",
    "com.exafunction.windsurf",
    "dev.zed.Zed",
    "dev.zed.Zed-Preview",
    "com.apple.dt.Xcode",
    "com.jetbrains.*",
    "com.google.android.studio",
    "com.sublimetext.*",
    "com.panic.Nova",
    "com.barebones.bbedit",
    // Lapis, the desktop host for terminal coding agents
    "dev.lapis.desktop",
];
const DOCS_BUNDLES: &[&str] = &[
    "notion.id",
    "md.obsidian",
    "net.shinyfrog.bear",
    "com.apple.Notes",
    "com.apple.iWork.Pages",
    "com.apple.iWork.Keynote",
    "com.apple.iWork.Numbers",
    "com.apple.TextEdit",
    "com.microsoft.Word",
    "com.microsoft.Excel",
    "com.microsoft.Powerpoint",
    "com.microsoft.onenote.mac",
    "com.evernote.Evernote",
    "com.lukilabs.lukiapp",
    "com.linear",
    "com.figma.Desktop",
    "com.agiletortoise.Drafts-OSX",
    "abnerworks.Typora",
];

// Android package names, for dictations that arrive over the WebSocket API
// (`remote.rs`): a phone names its target app by package. Same six categories.
const ANDROID_WORK_CHAT: &[&str] = &[
    "com.Slack",
    "com.microsoft.teams",
    "com.google.android.apps.dynamite",
];
const ANDROID_CHAT: &[&str] = &[
    "com.discord",
    "com.whatsapp",
    "com.whatsapp.w4b",
    "org.telegram.messenger",
    "org.thoughtcrime.securesms",
    "com.facebook.orca",
    "com.google.android.apps.messaging",
    "com.samsung.android.messaging",
    "com.tencent.mm",
    "jp.naver.line.android",
    "com.twitter.android",
    "com.linkedin.android",
    "com.reddit.frontpage",
];
const ANDROID_EMAIL: &[&str] = &[
    "com.google.android.gm",
    "com.microsoft.office.outlook",
    "com.fsck.k9",
    "net.thunderbird.android",
    "ch.protonmail.android",
    "com.yahoo.mobile.client.android.mail",
    "com.readdle.spark",
    "com.fastmail.app",
    "com.samsung.android.email.provider",
];
const ANDROID_AI: &[&str] = &[
    "com.openai.chatgpt",
    "com.anthropic.claude",
    "com.google.android.apps.bard",
    "ai.perplexity.app.android",
    "com.deepseek.chat",
    "ai.x.grok",
    "com.microsoft.copilot",
];
const ANDROID_CODE: &[&str] = &[
    "com.termux",
    "org.connectbot",
    "com.sonelli.juicessh",
    "com.github.android",
];
const ANDROID_DOCS: &[&str] = &[
    "com.google.android.apps.docs.editors.docs",
    "com.google.android.keep",
    "com.microsoft.office.word",
    "com.microsoft.office.onenote",
    "com.evernote",
    "com.samsung.android.app.notes",
];

/// The category of a dictation target. A known host wins; a browser on an
/// unknown site is `Other`, never the browser's own category.
pub fn categorize(bundle_id: &str, url_host: Option<&str>) -> AppCategory {
    let host_table: [(&[&str], AppCategory); 6] = [
        (AI_HOSTS, AppCategory::AiPrompt),
        (EMAIL_HOSTS, AppCategory::Email),
        (WORK_CHAT_HOSTS, AppCategory::WorkChat),
        (CHAT_HOSTS, AppCategory::Chat),
        (CODE_HOSTS, AppCategory::Code),
        (DOCS_HOSTS, AppCategory::Docs),
    ];
    if let Some(host) = url_host.and_then(normalize_host) {
        for (hosts, category) in host_table {
            if hosts.iter().any(|pattern| host_matches(&host, pattern)) {
                return category;
            }
        }
        if BROWSER_BUNDLES
            .iter()
            .any(|pattern| bundle_matches(bundle_id, pattern))
        {
            return AppCategory::Other;
        }
    }
    let bundle_table: [(&[&str], AppCategory); 6] = [
        (AI_BUNDLES, AppCategory::AiPrompt),
        (EMAIL_BUNDLES, AppCategory::Email),
        (WORK_CHAT_BUNDLES, AppCategory::WorkChat),
        (CHAT_BUNDLES, AppCategory::Chat),
        (CODE_BUNDLES, AppCategory::Code),
        (DOCS_BUNDLES, AppCategory::Docs),
    ];
    for (bundles, category) in bundle_table {
        if bundles
            .iter()
            .any(|pattern| bundle_matches(bundle_id, pattern))
        {
            return category;
        }
    }
    let android_table: [(&[&str], AppCategory); 6] = [
        (ANDROID_AI, AppCategory::AiPrompt),
        (ANDROID_EMAIL, AppCategory::Email),
        (ANDROID_WORK_CHAT, AppCategory::WorkChat),
        (ANDROID_CHAT, AppCategory::Chat),
        (ANDROID_CODE, AppCategory::Code),
        (ANDROID_DOCS, AppCategory::Docs),
    ];
    for (packages, category) in android_table {
        if packages.contains(&bundle_id) {
            return category;
        }
    }
    AppCategory::Other
}

/// Browsers whose tab URL the bar reads; their own category is the site's.
pub const BROWSER_BUNDLES: &[&str] = &[
    "com.google.Chrome",
    "com.google.Chrome.*",
    "com.brave.Browser",
    "com.brave.Browser.*",
    "com.microsoft.edgemac",
    "com.microsoft.edgemac.*",
    "company.thebrowser.Browser",
    "company.thebrowser.dia",
    "com.vivaldi.Vivaldi",
    "com.operasoftware.Opera",
    "org.chromium.Chromium",
    "com.apple.Safari",
    "com.apple.SafariTechnologyPreview",
    "org.mozilla.firefox",
    "app.zen-browser.zen",
];

/// The text around the cursor at key-down (Wispr's `textbox_contents`), sent
/// by the bar only when the user turned text-box context on.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextboxContext {
    #[serde(default)]
    pub before_text: String,
    #[serde(default)]
    pub selected_text: String,
    #[serde(default)]
    pub after_text: String,
}

impl TextboxContext {
    /// Characters kept per side, whatever a client sends.
    pub const MAX_CHARS: usize = 2_000;

    pub fn from_request(value: &Value) -> Option<Self> {
        let field = |key: &str, keep_end: bool| {
            let text = value.get(key).and_then(Value::as_str).unwrap_or("");
            let count = text.chars().count();
            if count <= Self::MAX_CHARS {
                text.to_string()
            } else if keep_end {
                text.chars().skip(count - Self::MAX_CHARS).collect()
            } else {
                text.chars().take(Self::MAX_CHARS).collect()
            }
        };
        let context = Self {
            before_text: field("before_text", true),
            selected_text: field("selected_text", false),
            after_text: field("after_text", false),
        };
        let empty = context.before_text.is_empty()
            && context.selected_text.is_empty()
            && context.after_text.is_empty();
        (!empty).then_some(context)
    }
}

/// The `context` object sent to the correction sidecar. A model that does
/// not read it is unaffected; the context lane decides what to use.
pub fn polish_context(
    app: Option<&AppContext>,
    style_preferences: &[String],
    textbox: Option<&TextboxContext>,
) -> Option<Value> {
    if app.is_none() && style_preferences.is_empty() && textbox.is_none() {
        return None;
    }
    let mut context = json!({});
    if let Some(app) = app {
        context["app_category"] = json!(app.category);
        context["app_bundle_id"] = json!(app.bundle_id);
        if let Some(host) = &app.url_host {
            context["url_host"] = json!(host);
        }
    }
    if !style_preferences.is_empty() {
        context["style_preferences"] = json!(style_preferences);
    }
    if let Some(textbox) = textbox {
        context["textbox"] = json!(textbox);
    }
    Some(context)
}

/// settings.json `writing_style` values that pin one refiner mode; anything
/// else (`app`, absent, unknown) follows the app's category.
const WRITING_STYLES: &[&str] = &["minimal", "casual", "formal"];

/// settings.json `writing_modes_by_app` values: bundle id -> one of the refiner's
/// four modes (Settings > Advanced > Writing mode per app). Others are ignored.
const APP_WRITING_MODES: &[&str] = &["minimal", "casual", "formal", "prompt"];

/// Add the user's fixed writing mode to a sidecar `context` (the sidecar then
/// uses it instead of the app category's mode).
pub fn with_writing_mode(context: Option<Value>, mode: Option<&str>) -> Option<Value> {
    let Some(mode) = mode else { return context };
    let mut context = context.unwrap_or_else(|| json!({}));
    context["writing_mode"] = json!(mode);
    Some(context)
}

/// The approved style preferences from settings.json, parsed again only
/// when the file changes (size or modification time). Every polish request
/// asks, previews included, so reading and parsing the file each time was
/// work for nothing.
#[derive(Debug, Default)]
pub struct StylePreferenceCache {
    path: Option<std::path::PathBuf>,
    stamp: Option<(Option<std::time::SystemTime>, u64)>,
    /// Hash of the bytes last parsed successfully.
    content: Option<u64>,
    /// Hash of bytes that failed to parse, so they are not parsed again.
    failed: Option<u64>,
    styles: std::collections::BTreeMap<String, Vec<String>>,
    /// `live_text` from the same file (asked once per dictation, at `stream_start`).
    live_text: bool,
    /// `writing_style` from the same file when it names one refiner mode
    /// (Settings > Advanced); None follows the app.
    writing_mode: Option<String>,
    /// `writing_modes_by_app` from the same file: bundle id -> mode.
    app_writing_modes: std::collections::BTreeMap<String, String>,
    /// Successful parses (for tests and diagnostics).
    pub loads: usize,
    /// File reads and parse attempts (for tests and diagnostics).
    pub reads: usize,
    pub parse_attempts: usize,
}

/// A file modified this recently may change again within the same mtime
/// tick and keep its size, so its bytes are compared, not just its stamp.
const SETTLE: std::time::Duration = std::time::Duration::from_secs(2);

impl StylePreferenceCache {
    /// Reads `path`, or the user's settings.json when `None`.
    pub fn new(path: Option<std::path::PathBuf>) -> Self {
        Self {
            path,
            live_text: false,
            ..Self::default()
        }
    }

    pub fn styles_for(&mut self, category: AppCategory) -> Vec<String> {
        self.refresh();
        self.styles
            .get(category.as_str())
            .cloned()
            .unwrap_or_default()
    }

    /// settings.json `live_text` as the file says now (parsed again only when it changed).
    pub fn live_text(&mut self) -> bool {
        self.refresh();
        self.live_text
    }

    /// The one writing mode the user picked for every app (`minimal`,
    /// `casual` or `formal`), or None to follow the app (the default).
    pub fn writing_mode(&mut self) -> Option<String> {
        self.refresh();
        self.writing_mode.clone()
    }

    /// The writing mode for a dictation into `bundle_id`: the user's per-app pick
    /// first, then the one mode picked for every app, else None (follow the app's
    /// category).
    pub fn writing_mode_for(&mut self, bundle_id: Option<&str>) -> Option<String> {
        self.refresh();
        bundle_id
            .filter(|id| !id.is_empty())
            .and_then(|id| self.app_writing_modes.get(id))
            .cloned()
            .or_else(|| self.writing_mode.clone())
    }

    fn refresh(&mut self) {
        let Some(path) = self
            .path
            .clone()
            .or_else(|| crate::data::settings_path().ok())
        else {
            return;
        };
        let Ok(meta) = std::fs::metadata(&path) else {
            // No file: no preferences, and the defaults.
            self.stamp = None;
            self.content = None;
            self.styles.clear();
            self.live_text = false;
            self.writing_mode = None;
            self.app_writing_modes.clear();
            return;
        };
        let stamp = Some((meta.modified().ok(), meta.len()));
        // Recently written: compare bytes even with an unchanged stamp. An
        // mtime in the future (clock skew) or unreadable is settled: the
        // stamp alone decides, so it is not re-read on every request.
        let settling = meta
            .modified()
            .ok()
            .and_then(|modified| modified.elapsed().ok())
            .is_some_and(|age| age < SETTLE);
        if stamp == self.stamp && !settling {
            return;
        }
        let Ok(bytes) = std::fs::read(&path) else {
            return;
        };
        self.reads += 1;
        let hash = {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            bytes.hash(&mut hasher);
            hasher.finish()
        };
        self.stamp = stamp;
        if Some(hash) == self.content || Some(hash) == self.failed {
            return;
        }
        // A file caught mid-write or malformed keeps the last good
        // preferences; its bytes are remembered so they are not parsed
        // again, and it is never cached as "no preferences".
        self.parse_attempts += 1;
        match crate::data::SettingsFile::parse(&bytes) {
            Ok(settings) => {
                self.loads += 1;
                self.styles = settings.style_preferences;
                self.live_text = settings.live_text;
                self.writing_mode = settings
                    .extra
                    .get("writing_style")
                    .and_then(Value::as_str)
                    .filter(|mode| WRITING_STYLES.contains(mode))
                    .map(str::to_string);
                self.app_writing_modes = settings
                    .extra
                    .get("writing_modes_by_app")
                    .and_then(Value::as_object)
                    .map(|modes| {
                        modes
                            .iter()
                            .filter_map(|(app, mode)| {
                                let mode = mode.as_str()?;
                                APP_WRITING_MODES
                                    .contains(&mode)
                                    .then(|| (app.clone(), mode.to_string()))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                self.content = Some(hash);
                self.failed = None;
            }
            Err(_) => self.failed = Some(hash),
        }
    }
}

const GREETINGS: &[&str] = &[
    "Hi",
    "Hello",
    "Hey",
    "Dear",
    "Good morning",
    "Good afternoon",
    "Good evening",
];
const SIGN_OFFS: &[&str] = &[
    "Best regards",
    "Kind regards",
    "Warm regards",
    "All the best",
    "Many thanks",
    "Thank you",
    "Thanks",
    "Regards",
    "Best",
    "Cheers",
    "Sincerely",
];

/// Lay a corrected email out the way one is typed: a dictated greeting
/// ("Hi Sarah,") and sign-off ("Best, Alex" or a closing "Thanks") on their
/// own lines. Only separators change; the words are the model's. Text
/// without a greeting or sign-off is returned unchanged.
pub fn email_layout(text: &str) -> String {
    let text = text.trim();
    let mut head = String::new();
    let mut body = text;
    // greeting: a greeting word, up to three name words, then a comma, then more text
    if let Some(greeting) = GREETINGS.iter().find(|g| {
        body.strip_prefix(**g)
            .is_some_and(|rest| rest.starts_with([' ', ',']))
    }) {
        let rest = &body[greeting.len()..];
        if let Some(comma) = rest.find(',') {
            let name = &rest[..comma];
            let after = rest[comma + 1..].trim_start();
            let words = name.split_whitespace().count();
            if (1..=3).contains(&words) && !after.is_empty() && !name.contains(['.', '?', '!']) {
                head = format!("{greeting}{name},");
                body = after;
            }
        }
    }
    // sign-off: after a sentence end, a closing phrase, optionally ", Name" (one to
    // three capitalized words), then the end of the text
    let mut tail = String::new();
    for sign in SIGN_OFFS {
        let Some(at) = body.rfind(sign) else { continue };
        let before = body[..at].trim_end();
        if !(before.is_empty() || before.ends_with(['.', '?', '!'])) {
            continue;
        }
        let rest = body[at + sign.len()..]
            .trim_end_matches(['.', '!'])
            .trim_end();
        let name = match rest.strip_prefix(',') {
            Some(name) => name.trim(),
            None if rest.is_empty() => "",
            None => continue,
        };
        let words: Vec<&str> = name.split_whitespace().collect();
        let names = words.len() <= 3
            && words
                .iter()
                .all(|w| w.chars().next().is_some_and(char::is_uppercase));
        if !names || (before.is_empty() && head.is_empty()) {
            continue;
        }
        tail = if name.is_empty() {
            format!("{sign}{}", body[at + sign.len()..].trim_end())
        } else {
            format!("{sign},\n{name}")
        };
        body = before;
        break;
    }
    if head.is_empty() && tail.is_empty() {
        return text.to_string();
    }
    [head.as_str(), body, tail.as_str()]
        .iter()
        .filter(|part| !part.is_empty())
        .copied()
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Apply the style preferences the user approved for this category to a
/// corrected dictation. Terms the dictionary spells (`cuBLAS`), acronyms and
/// words with inner capitals keep their case.
pub fn apply_style(text: &str, preferences: &[String], dictionary: &DictionaryFile) -> String {
    let wants = |name: &str| preferences.iter().any(|p| p == name);
    let mut output = text.to_string();
    let protected = |word: &str| {
        let core: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
        let letters: Vec<char> = core.chars().filter(|c| c.is_alphabetic()).collect();
        let acronym = letters.len() >= 2 && letters.iter().all(|c| c.is_uppercase());
        let inner = core.chars().skip(1).any(char::is_uppercase);
        let term = dictionary.entries.iter().any(|entry| {
            let canonical = entry.canonical();
            canonical.split_whitespace().any(|part| {
                part.trim_matches(|c: char| !c.is_alphanumeric()) == core
                    && part.chars().any(char::is_uppercase)
            })
        });
        acronym || inner || term
    };
    if wants("all_lowercase") {
        output = output
            .split_inclusive(char::is_whitespace)
            .map(|piece| {
                if protected(piece) {
                    piece.to_string()
                } else {
                    piece.to_lowercase()
                }
            })
            .collect();
    } else if wants("lowercase_start") {
        let first_word = output.split_whitespace().next().unwrap_or("");
        if !protected(first_word) && first_word != "I" && !first_word.starts_with("I'") {
            let mut chars = output.chars();
            if let Some(first) = chars.next() {
                output = first.to_lowercase().chain(chars).collect();
            }
        }
    }
    if wants("no_trailing_period") {
        let trimmed = output.trim_end();
        let last_word = trimmed.split_whitespace().last().unwrap_or("");
        // `etc.`, `e.g.` and `U.S.` end in a period that is part of the word.
        let abbreviation =
            last_word.trim_end_matches('.').contains('.') || ABBREVIATIONS.contains(&last_word);
        if trimmed.ends_with('.') && !trimmed.ends_with("..") && !abbreviation {
            output = trimmed[..trimmed.len() - 1].to_string();
        }
    }
    output
}

const ABBREVIATIONS: &[&str] = &[
    "etc.", "e.g.", "i.e.", "vs.", "Inc.", "Ltd.", "Co.", "Jr.", "Sr.", "Dr.", "Mr.", "Mrs.",
    "Ms.", "St.", "a.m.", "p.m.",
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DictionaryEntry;

    fn dictionary(terms: &[&str]) -> DictionaryFile {
        DictionaryFile {
            entries: terms
                .iter()
                .map(|term| DictionaryEntry {
                    phrase: term.to_string(),
                    replacement: None,
                    spoken_forms: Vec::new(),
                    source: "test".into(),
                    starred: false,
                    usage_count: 0,
                })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn apps_fall_into_the_seven_categories() {
        assert_eq!(
            categorize("com.tinyspeck.slackmacgap", None),
            AppCategory::WorkChat
        );
        assert_eq!(
            categorize("com.microsoft.teams2", None),
            AppCategory::WorkChat
        );
        assert_eq!(categorize("com.hnc.Discord", None), AppCategory::Chat);
        assert_eq!(categorize("com.apple.MobileSMS", None), AppCategory::Chat);
        assert_eq!(categorize("com.apple.mail", None), AppCategory::Email);
        assert_eq!(categorize("com.mitchellh.ghostty", None), AppCategory::Code);
        assert_eq!(categorize("dev.lapis.desktop", None), AppCategory::Code);
        assert_eq!(
            categorize("com.jetbrains.rustrover", None),
            AppCategory::Code
        );
        assert_eq!(
            categorize("com.todesktop.230313mzl4w4u92", None),
            AppCategory::Code
        );
        assert_eq!(categorize("com.openai.chat", None), AppCategory::AiPrompt);
        assert_eq!(categorize("md.obsidian", None), AppCategory::Docs);
        assert_eq!(categorize("com.apple.finder", None), AppCategory::Other);
        // Android packages (dictations over the WebSocket API)
        assert_eq!(categorize("com.Slack", None), AppCategory::WorkChat);
        assert_eq!(categorize("com.whatsapp", None), AppCategory::Chat);
        assert_eq!(
            categorize("com.google.android.gm", None),
            AppCategory::Email
        );
        assert_eq!(categorize("com.termux", None), AppCategory::Code);
        assert_eq!(
            categorize("com.anthropic.claude", None),
            AppCategory::AiPrompt
        );
        assert_eq!(
            categorize("com.google.android.keep", None),
            AppCategory::Docs
        );
        assert_eq!(categorize("com.android.settings", None), AppCategory::Other);
    }

    #[test]
    fn a_picked_writing_style_rides_along_in_the_sidecar_context() {
        use super::with_writing_mode;
        assert_eq!(with_writing_mode(None, None), None);
        assert_eq!(
            with_writing_mode(Some(json!({"app_category": "chat"})), None),
            Some(json!({"app_category": "chat"}))
        );
        assert_eq!(
            with_writing_mode(Some(json!({"app_category": "chat"})), Some("formal")),
            Some(json!({"app_category": "chat", "writing_mode": "formal"}))
        );
        assert_eq!(
            with_writing_mode(None, Some("minimal")),
            Some(json!({"writing_mode": "minimal"}))
        );
    }

    #[test]
    fn an_email_gets_its_greeting_and_sign_off_on_their_own_lines() {
        use super::email_layout;
        assert_eq!(
            email_layout("Hi Sarah, Thanks for sending the contract over. Could we change section 4? Then I can sign it this week. Best, Alex"),
            "Hi Sarah,\n\nThanks for sending the contract over. Could we change section 4? Then I can sign it this week.\n\nBest,\nAlex"
        );
        assert_eq!(
            email_layout("Can you double-check the changelog before then? Thanks"),
            "Can you double-check the changelog before then?\n\nThanks"
        );
        assert_eq!(
            email_layout("Dear Dr. Lee, the results are attached. Kind regards, Sam Ortiz."),
            "Dear Dr. Lee, the results are attached.\n\nKind regards,\nSam Ortiz"
        );
        // nothing to lay out: unchanged, including "thanks" inside a sentence
        for text in [
            "The deploy failed again. Can someone take a look?",
            "Thanks for the notes, I will send the draft on Friday.",
            "Hey, I'm going to push the release tomorrow.",
            "Thanks",
        ] {
            assert_eq!(email_layout(text), text);
        }
    }

    #[test]
    fn a_browser_takes_the_category_of_its_site() {
        let chrome = "com.google.Chrome";
        assert_eq!(
            categorize(chrome, Some("mail.google.com")),
            AppCategory::Email
        );
        assert_eq!(
            categorize(chrome, Some("www.claude.ai")),
            AppCategory::AiPrompt
        );
        assert_eq!(
            categorize(chrome, Some("acme.slack.com")),
            AppCategory::WorkChat
        );
        assert_eq!(
            categorize(chrome, Some("app.slack.com")),
            AppCategory::WorkChat
        );
        assert_eq!(categorize(chrome, Some("discord.com")), AppCategory::Chat);
        assert_eq!(categorize(chrome, Some("notslack.com")), AppCategory::Other);
        assert_eq!(categorize(chrome, Some("github.com")), AppCategory::Code);
        assert_eq!(
            categorize(chrome, Some("docs.google.com")),
            AppCategory::Docs
        );
        assert_eq!(
            categorize(chrome, Some("acme.atlassian.net")),
            AppCategory::Docs
        );
        assert_eq!(categorize(chrome, Some("example.com")), AppCategory::Other);
        assert_eq!(categorize(chrome, None), AppCategory::Other);
        // An Electron app's own web area does not demote it to Other.
        assert_eq!(
            categorize("com.tinyspeck.slackmacgap", Some("example.com")),
            AppCategory::WorkChat
        );
    }

    #[test]
    fn request_keeps_only_a_normalized_host() {
        let app = AppContext::from_request(&json!({
            "bundle_id": "com.google.Chrome", "name": "Google Chrome",
            "url_host": "WWW.Mail.Google.com"
        }))
        .unwrap();
        assert_eq!(app.url_host.as_deref(), Some("mail.google.com"));
        assert_eq!(app.category, AppCategory::Email);
        let bad = AppContext::from_request(&json!({
            "bundle_id": "com.google.Chrome", "url_host": "https://x.com/path?q=1"
        }))
        .unwrap();
        assert_eq!(bad.url_host, None);
        assert!(AppContext::from_request(&json!({})).is_none());
        let stored = serde_json::to_value(&app).unwrap();
        assert_eq!(stored["category"], "email");
    }

    #[test]
    fn polish_context_carries_what_the_model_may_use() {
        let app = AppContext {
            bundle_id: "com.hnc.Discord".into(),
            app_name: "Discord".into(),
            url_host: None,
            category: AppCategory::Chat,
        };
        let context = polish_context(Some(&app), &["lowercase_start".into()], None).unwrap();
        assert_eq!(context["app_category"], "chat");
        assert_eq!(context["style_preferences"][0], "lowercase_start");
        assert!(context.get("textbox").is_none());
        assert!(polish_context(None, &[], None).is_none());

        let textbox = TextboxContext::from_request(&json!({
            "before_text": "Hi Bob, about the kernel: ", "selected_text": "", "after_text": "\n\nThanks"
        }))
        .unwrap();
        let context = polish_context(None, &[], Some(&textbox)).unwrap();
        assert_eq!(
            context["textbox"]["before_text"],
            "Hi Bob, about the kernel: "
        );
        assert_eq!(context["textbox"]["after_text"], "\n\nThanks");
    }

    /// Review item 9: settings.json is parsed once, then again only when it
    /// changes.
    #[test]
    fn style_preferences_are_parsed_only_when_settings_change() {
        let directory = std::env::temp_dir().join(format!(
            "phonon-style-cache-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("settings.json");
        std::fs::write(
            &path,
            r#"{"schema_version":2,"style_preferences":{"chat":["lowercase_start"]}}"#,
        )
        .unwrap();
        let mut cache = StylePreferenceCache::new(Some(path.clone()));
        for _ in 0..50 {
            assert_eq!(cache.styles_for(AppCategory::Chat), ["lowercase_start"]);
            assert!(cache.styles_for(AppCategory::Email).is_empty());
        }
        assert_eq!(cache.loads, 1, "100 requests, one parse");
        std::fs::write(
            &path,
            r#"{"schema_version":2,"style_preferences":{"chat":["lowercase_start","no_trailing_period"]}}"#,
        )
        .unwrap();
        assert_eq!(
            cache.styles_for(AppCategory::Chat).len(),
            2,
            "an approval applies next time"
        );
        assert_eq!(cache.loads, 2);
        // live text: off when absent, and a flipped setting applies at the next ask
        assert!(!cache.live_text());
        std::fs::write(&path, r#"{"schema_version":2,"live_text":true}"#).unwrap();
        assert!(cache.live_text(), "flipped on: the next dictation shows it");
        std::fs::write(&path, r#"{"schema_version":2,"live_text":false}"#).unwrap();
        assert!(
            !cache.live_text(),
            "flipped off: the next dictation shows no live text"
        );
        // writing style: follows the app unless one refiner mode is picked
        assert_eq!(cache.writing_mode(), None);
        for (style, mode) in [
            ("app", None),
            ("formal", Some("formal")),
            ("casual", Some("casual")),
            ("minimal", Some("minimal")),
            ("prompt", None),
            ("loud", None),
        ] {
            std::fs::write(
                &path,
                format!(r#"{{"schema_version":2,"writing_style":"{style}"}}"#),
            )
            .unwrap();
            assert_eq!(cache.writing_mode().as_deref(), mode, "{style}");
        }
        // per-app writing mode: wins over the global pick; unknown modes are ignored
        std::fs::write(
            &path,
            r#"{"schema_version":2,"writing_style":"formal","writing_modes_by_app":
                {"com.mitchellh.ghostty":"minimal","dev.lapis.desktop":"prompt",
                 "com.hnc.Discord":"verbatim","com.apple.mail":3}}"#,
        )
        .unwrap();
        assert_eq!(
            cache
                .writing_mode_for(Some("com.mitchellh.ghostty"))
                .as_deref(),
            Some("minimal")
        );
        assert_eq!(
            cache.writing_mode_for(Some("dev.lapis.desktop")).as_deref(),
            Some("prompt")
        );
        for other in [
            Some("com.hnc.Discord"),
            Some("com.apple.mail"),
            Some(""),
            None,
        ] {
            assert_eq!(
                cache.writing_mode_for(other).as_deref(),
                Some("formal"),
                "{other:?}"
            );
        }
        std::fs::write(
            &path,
            r#"{"schema_version":2,"writing_modes_by_app":{"com.mitchellh.ghostty":"casual"}}"#,
        )
        .unwrap();
        assert_eq!(
            cache
                .writing_mode_for(Some("com.mitchellh.ghostty"))
                .as_deref(),
            Some("casual")
        );
        assert_eq!(cache.writing_mode_for(Some("com.apple.mail")), None);
        std::fs::remove_file(&path).unwrap();
        assert_eq!(cache.writing_mode(), None);
        assert_eq!(cache.writing_mode_for(Some("com.mitchellh.ghostty")), None);
        assert!(cache.styles_for(AppCategory::Chat).is_empty());
        assert!(!cache.live_text(), "no file: the default");
        let _ = std::fs::remove_dir_all(&directory);
    }

    /// Second review, item 8: a failed parse was cached as "no preferences"
    /// until the next write, and a rewrite with the same size inside one
    /// mtime tick went unnoticed.
    #[test]
    fn a_bad_or_same_stamp_write_never_loses_the_preferences() {
        let directory = std::env::temp_dir().join(format!(
            "phonon-style-cache-b-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("settings.json");
        let write = |text: &str, modified: std::time::SystemTime| {
            std::fs::write(&path, text).unwrap();
            std::fs::File::options()
                .write(true)
                .open(&path)
                .unwrap()
                .set_modified(modified)
                .unwrap();
        };
        let tick = std::time::SystemTime::now() - std::time::Duration::from_millis(500);
        write(
            r#"{"schema_version":2,"style_preferences":{"chat":["lowercase_start"]}}"#,
            tick,
        );
        let mut cache = StylePreferenceCache::new(Some(path.clone()));
        assert_eq!(cache.styles_for(AppCategory::Chat), ["lowercase_start"]);
        // Same length, same mtime: only the bytes differ.
        write(
            r#"{"schema_version":2,"style_preferences":{"chat":["all_lowercase__"]}}"#,
            tick,
        );
        assert_eq!(cache.styles_for(AppCategory::Chat), ["all_lowercase__"]);
        // A half-written file keeps the last good preferences.
        write(
            r#"{"schema_version":2,"style_preferences":{"chat":["#,
            std::time::SystemTime::now(),
        );
        assert_eq!(cache.styles_for(AppCategory::Chat), ["all_lowercase__"]);
        write(
            r#"{"schema_version":2,"style_preferences":{"chat":["no_trailing_period"]}}"#,
            std::time::SystemTime::now(),
        );
        assert_eq!(cache.styles_for(AppCategory::Chat), ["no_trailing_period"]);

        // Third review, item 10: a malformed file was parsed again on every
        // request, and an mtime in the future kept the file 'settling'
        // forever, so it was re-read on every request.
        write(
            r#"{"schema_version":2,"style_"#,
            std::time::SystemTime::now(),
        );
        let attempts = cache.parse_attempts;
        for _ in 0..20 {
            assert_eq!(cache.styles_for(AppCategory::Chat), ["no_trailing_period"]);
        }
        assert_eq!(
            cache.parse_attempts,
            attempts + 1,
            "the bad bytes are parsed once"
        );
        write(
            r#"{"schema_version":2,"style_preferences":{"chat":["lowercase_start"]}}"#,
            std::time::SystemTime::now() + std::time::Duration::from_secs(3_600),
        );
        let reads = cache.reads;
        for _ in 0..20 {
            assert_eq!(cache.styles_for(AppCategory::Chat), ["lowercase_start"]);
        }
        assert_eq!(
            cache.reads,
            reads + 1,
            "a future mtime is read once, not every time"
        );
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn textbox_context_is_bounded_and_empty_means_absent() {
        assert!(TextboxContext::from_request(&json!({})).is_none());
        assert!(TextboxContext::from_request(&json!({"before_text": ""})).is_none());
        let long = "x".repeat(5_000) + "END";
        let textbox = TextboxContext::from_request(&json!({
            "before_text": long, "after_text": format!("START{}", "y".repeat(5_000))
        }))
        .unwrap();
        assert_eq!(
            textbox.before_text.chars().count(),
            TextboxContext::MAX_CHARS
        );
        assert!(
            textbox.before_text.ends_with("END"),
            "the text nearest the cursor stays"
        );
        assert!(textbox.after_text.starts_with("START"));
    }

    #[test]
    fn approved_styles_apply_without_touching_terms() {
        let terms = dictionary(&["cuBLAS", "Phonon"]);
        let lower = vec!["lowercase_start".to_string()];
        assert_eq!(apply_style("Sounds good.", &lower, &terms), "sounds good.");
        assert_eq!(apply_style("GPU is busy", &lower, &terms), "GPU is busy");
        assert_eq!(
            apply_style("Phonon shipped", &lower, &terms),
            "Phonon shipped"
        );
        assert_eq!(apply_style("I think so", &lower, &terms), "I think so");
        let period = vec!["no_trailing_period".to_string()];
        assert_eq!(
            apply_style("Sounds good. See you.", &period, &terms),
            "Sounds good. See you"
        );
        assert_eq!(
            apply_style("Bring snacks etc.", &period, &terms),
            "Bring snacks etc."
        );
        assert_eq!(apply_style("Wait...", &period, &terms), "Wait...");
        let all = vec![
            "all_lowercase".to_string(),
            "no_trailing_period".to_string(),
        ];
        assert_eq!(
            apply_style("Sounds Good, Ship cuBLAS On The GPU.", &all, &terms),
            "sounds good, ship cuBLAS on the GPU"
        );
        assert_eq!(apply_style("Unchanged.", &[], &terms), "Unchanged.");
    }
}
