import AppKit
import ApplicationServices

/// Which apps and sites Phonon may read for context, and where a dictation is
/// going. The denylist is the user's, edited in Settings; Signal and Discord
/// (app and site) are on it by default.
enum ContextPolicy {
    static let defaultDenylist = [
        "org.whispersystems.signal-desktop",
        "com.hnc.Discord",
        "discord.com",
    ]

    /// Never read, whatever the user's list says: macOS password and
    /// authorization prompts and the common password managers (apps and web
    /// vaults). Wispr captured accessibility text from password prompts.
    static let alwaysPrivate = [
        "com.apple.SecurityAgent",
        "com.apple.LocalAuthentication.UIAgent",
        "com.apple.keychainaccess",
        "com.apple.Passwords",
        "com.1password.1password",
        "com.agilebits.onepassword7",
        "com.agilebits.onepassword-osx",
        "com.agilebits.onepassword4",
        "com.bitwarden.desktop",
        "com.lastpass.LastPass",
        "com.dashlane.Dashlane",
        "org.keepassxc.keepassxc",
        "me.proton.pass.electron",
        "in.sinew.Enpass-Desktop",
        "in.sinew.Enpass-Desktop.App",
        "com.keepersecurity.passwordmanager",
        "com.callpod.keepermac",
        "com.callpod.keepermac.lite",
        "1password.com",
        "1password.eu",
        "bitwarden.com",
        "bitwarden.eu",
        "lastpass.com",
        "dashlane.com",
        "pass.proton.me",
        "keepersecurity.com",
        "keepersecurity.eu",
    ]

    /// A site and its desktop app are one entry: listing either blocks both
    /// (`discord.com` also blocks the Discord app's windows, and the Slack
    /// app's bundle id also blocks app.slack.com in a browser).
    struct SiteApp {
        let hosts: [String]
        let apps: [String]
    }

    static let siteApps: [SiteApp] = [
        SiteApp(hosts: ["discord.com", "discordapp.com"], apps: ["com.hnc.Discord"]),
        SiteApp(hosts: ["slack.com"], apps: ["com.tinyspeck.slackmacgap"]),
        SiteApp(hosts: ["whatsapp.com"], apps: ["net.whatsapp.WhatsApp", "desktop.WhatsApp"]),
        SiteApp(hosts: ["signal.org"], apps: ["org.whispersystems.signal-desktop"]),
        SiteApp(
            hosts: ["messenger.com"],
            apps: ["com.facebook.archon", "com.facebook.archon.developerID"]),
        SiteApp(hosts: ["telegram.org"], apps: ["ru.keepcoder.Telegram", "org.telegram.desktop"]),
    ]

    /// The site/app pairs a list names: by an app's bundle id, a site, or a
    /// subdomain of the site (`app.slack.com`).
    static func namedSiteApps(_ entries: [String]) -> [SiteApp] {
        let listed = entries.compactMap(normalizeEntry).map { $0.lowercased() }
        return siteApps.filter { pair in
            listed.contains { entry in
                pair.apps.contains { $0.lowercased() == entry }
                    || pair.hosts.contains { entry == $0 || entry.hasSuffix("." + $0) }
            }
        }
    }

    /// The list with each named site's desktop app and each app's site.
    static func withSiteApps(_ entries: [String]) -> [String] {
        entries + namedSiteApps(entries).flatMap { $0.hosts + $0.apps }
    }

    /// Bundle ids that are shaped like hosts (`desktop.WhatsApp`) but are
    /// known apps, never sites.
    static let knownAppIDs = Set(siteApps.flatMap(\.apps).map { $0.lowercased() })

    /// A denylist entry is a bundle id (`com.hnc.Discord`, `md.obsidian`) or a
    /// site (`discord.com`, which also covers its subdomains). A pasted URL is
    /// reduced to its host.
    static func normalizeEntry(_ raw: String) -> String? {
        var entry = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !entry.isEmpty else { return nil }
        if let url = URL(string: entry), let host = url.host, url.scheme != nil {
            entry = host
        }
        if let slash = entry.firstIndex(of: "/") { entry = String(entry[..<slash]) }
        if looksLikeHost(entry) {
            entry = entry.lowercased()
            if entry.hasPrefix("www.") { entry.removeFirst(4) }
        }
        return entry.isEmpty ? nil : entry
    }

    /// First labels of reverse-DNS bundle ids. A site's host almost never
    /// starts with one of these followed by two more labels.
    static let reverseDNSPrefixes: Set<String> = [
        "com", "org", "net", "io", "edu", "gov", "co", "me", "dev", "ai", "us", "uk", "de", "fr",
        "jp", "ch", "nl", "se", "ca", "au", "cn", "ru", "in", "it", "es", "br", "kr",
    ]

    /// Labels that end a site's host often enough to vouch for one. Only
    /// used to decide that an entry is *certainly* a bundle id; anything
    /// unsure is treated as both, so it blocks under either reading.
    static let commonTopLevelDomains: Set<String> = [
        "com", "org", "net", "io", "ai", "app", "dev", "co", "me", "gg", "so", "tv", "xyz", "edu",
        "gov", "info", "biz",
    ]

    /// An entry that can only be a bundle id: it starts like reverse DNS
    /// (`com.`, `org.`, `io.`) with at least three labels and does not end
    /// in a common TLD (`com.hnc.Discord`, `org.whispersystems.signal-desktop`,
    /// `com.openai.chat`; not `app.slack.com`).
    static func isCertainlyBundleID(_ entry: String) -> Bool {
        let labels = entry.split(separator: ".")
        guard labels.count >= 2, let first = labels.first, let last = labels.last else {
            return true
        }
        if labels.count >= 3, reverseDNSPrefixes.contains(first.lowercased()),
            !commonTopLevelDomains.contains(last.lowercased())
        {
            return true
        }
        return false
    }

    /// Whether an entry may name a site: host-shaped (dot-separated labels of
    /// letters, digits and hyphens) and not certainly a bundle id. Structural,
    /// so `yandex.ru`, `mybank.com.br`, `naver.kr` and `bank.co.nz` count;
    /// `md.obsidian` also counts and simply blocks under both readings.
    static func looksLikeHost(_ entry: String) -> Bool {
        let labels = entry.split(separator: ".", omittingEmptySubsequences: false)
        guard labels.count >= 2,
            labels.allSatisfy({ label in
                !label.isEmpty && label.allSatisfy { $0.isLetter || $0.isNumber || $0 == "-" }
            })
        else { return false }
        return !isCertainlyBundleID(entry)
    }

    /// Two-label public suffixes under which the registrable name is the
    /// third label from the end (`mybank.com.br`, `bank.co.nz`).
    static let secondLevelSuffixes: Set<String> = [
        "co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au", "co.nz", "org.nz",
        "com.br", "net.br", "co.jp", "ne.jp", "or.jp", "co.kr", "or.kr", "com.cn", "net.cn",
        "com.mx", "com.ar", "com.tr", "co.in", "co.za", "com.sg", "com.hk", "com.tw", "co.il",
    ]

    /// The name a site goes by in a window title: the registrable label
    /// (`slack` for app.slack.com, `whatsapp` for web.whatsapp.com, `mybank`
    /// for mybank.com.br, `x` for x.com).
    static func brand(ofHost host: String) -> String? {
        let labels = host.lowercased().split(separator: ".").map(String.init)
        guard labels.count >= 2 else { return nil }
        let lastTwo = labels.suffix(2).joined(separator: ".")
        if labels.count >= 3, secondLevelSuffixes.contains(lastTwo) {
            return labels[labels.count - 3]
        }
        return labels[labels.count - 2]
    }

    /// The words of one title part: split on spaces and punctuation, but a
    /// token keeps inner dots, hyphens and underscores (`discord.py` stays
    /// one token, so it is not `discord`).
    static func titleTokens(_ part: String) -> [String] {
        var tokens = [String]()
        var current = ""
        let characters = Array(part.lowercased())
        for (index, character) in characters.enumerated() {
            let inner =
                (character == "." || character == "-" || character == "_") && !current.isEmpty
                && index + 1 < characters.count
                && (characters[index + 1].isLetter || characters[index + 1].isNumber)
            if character.isLetter || character.isNumber || inner {
                current.append(character)
            } else if !current.isEmpty {
                tokens.append(current)
                current = ""
            }
        }
        if !current.isEmpty { tokens.append(current) }
        return tokens
    }

    /// Browser names that end a window title; they are not the page's.
    static let browserTitleNames: Set<String> = [
        "google chrome", "chrome", "chromium", "safari", "arc", "brave", "microsoft edge", "edge",
        "firefox", "mozilla firefox", "vivaldi", "opera", "zen", "dia", "safari technology preview",
    ]

    /// Title parts, split on the separators browsers and sites use, with
    /// unread badges like "(3)" or "[2]" dropped and the browser's own name
    /// at the end removed ("Google Chrome" is not Google's page).
    static func titleParts(_ title: String) -> [String] {
        var parts = [title]
        for separator in [" - ", " | ", " — ", " – ", " · ", " • ", " / ", " :: "] {
            parts = parts.flatMap { $0.components(separatedBy: separator) }
        }
        if let last = parts.last?.trimmingCharacters(in: .whitespaces).lowercased(),
            browserTitleNames.contains(last), parts.count > 1
        {
            parts.removeLast()
        }
        return parts.map { part in
            var trimmed = part.trimmingCharacters(in: .whitespaces)
            while let range = trimmed.range(
                of: #"^[\(\[]\d+\+?[\)\]]\s*|^\d+\s+(new|unread)\b\s*|^[•●*]\s*"#,
                options: [.regularExpression, .caseInsensitive])
            {
                trimmed.removeSubrange(range)
            }
            return trimmed
        }.filter { !$0.isEmpty }
    }

    /// Whether a browser title names a private site: the site's brand as a
    /// whole word in some part ("(3) Discord | #general"), a two-letter or
    /// one-letter brand only as a whole part ("Home / X"), or the host itself.
    /// Used only to keep a meeting window's title out of metadata.json;
    /// screen context never decides by title.
    static func titleNamesSite(_ title: String, host: String) -> Bool {
        guard let brand = brand(ofHost: host) else { return false }
        let parts = titleParts(title)
        let lowerParts = parts.map { $0.lowercased() }
        if lowerParts.contains(where: { $0.contains(host) }) { return true }
        let spaced = brand.replacingOccurrences(of: "-", with: " ")
        if brand.count <= 2 {
            return lowerParts.contains(brand)
        }
        return lowerParts.contains { part in
            let tokens = titleTokens(part)
            if tokens.contains(brand) { return true }
            // `my-bank` written as "My Bank" in the title.
            let words = spaced.split(separator: " ").map(String.init)
            guard words.count > 1 else { return false }
            return tokens.indices.contains { start in
                Array(tokens.dropFirst(start).prefix(words.count)) == words
            }
        }
    }

    static func host(of url: URL?) -> String? {
        guard let host = url?.host?.lowercased(), !host.isEmpty else { return nil }
        return host.hasPrefix("www.") ? String(host.dropFirst(4)) : host
    }

    /// Browsers by the known list or as https handlers (`BrowserCatalog`).
    static func isBrowser(_ bundleID: String?) -> Bool {
        BrowserCatalog.isBrowser(bundleID, httpsHandlers: BrowserCatalog.httpsHandlers())
    }

    /// The app-level check: Phonon's own window, an app that cannot be
    /// identified, or a private app.
    static func isPrivateApp(_ window: ScreenWindow, denylist: [String], ownPID: pid_t) -> Bool {
        if window.pid == ownPID { return true }
        guard let bundleID = window.bundleID, !bundleID.isEmpty else { return true }
        return isDenied(bundleID: bundleID, host: nil, denylist: denylist)
    }

    /// Whether screen context must skip a window. Fails closed: a private
    /// or unidentified app, and every browser window except the focused one
    /// whose top-level page host is known and not private
    /// (`focusedPage`, from `BrowserPage`). Background browser windows are
    /// never read.
    static func isPrivateWindow(
        _ window: ScreenWindow, denylist: [String], ownPID: pid_t,
        focusedPage: (windowID: UInt32, host: String)? = nil,
        isBrowser: (String?) -> Bool = ContextPolicy.isBrowser
    ) -> Bool {
        if isPrivateApp(window, denylist: denylist, ownPID: ownPID) { return true }
        guard isBrowser(window.bundleID) else { return false }
        guard let focusedPage, focusedPage.windowID == window.id else { return true }
        return isDenied(bundleID: window.bundleID, host: focusedPage.host, denylist: denylist)
    }

    /// The screen-context exclusion for one read.
    static func screenExclusion(
        denylist: [String], ownPID: pid_t, focusedPage: (windowID: UInt32, host: String)?,
        isBrowser: @escaping (String?) -> Bool = ContextPolicy.isBrowser
    ) -> (ScreenWindow) -> Bool {
        { window in
            isPrivateWindow(
                window, denylist: denylist, ownPID: ownPID, focusedPage: focusedPage,
                isBrowser: isBrowser)
        }
    }

    /// The entries that may name a site, for window titles: hosts and the
    /// sites of named pairs, never a known app's bundle id (which would make
    /// `desktop` a brand and hide every title with the word Desktop).
    static func hostEntries(_ denylist: [String]) -> [String] {
        withSiteApps(denylist).compactMap(normalizeEntry).filter {
            looksLikeHost($0) && !knownAppIDs.contains($0.lowercased())
        }
    }

    /// An entry equal to the app's bundle id (any case) matches. Every entry
    /// that could be a host also matches that page host and its subdomains,
    /// so an entry whose kind is unsure (`notion.id`, `md.obsidian`) blocks
    /// under both readings.
    static func isDenied(bundleID: String?, host: String?, denylist: [String]) -> Bool {
        for raw in alwaysPrivate + withSiteApps(denylist) {
            guard let entry = normalizeEntry(raw) else { continue }
            if let bundleID, bundleID.caseInsensitiveCompare(entry) == .orderedSame {
                return true
            }
            if looksLikeHost(entry), let host = host?.lowercased(),
                host == entry || host.hasSuffix("." + entry)
            {
                return true
            }
        }
        return false
    }
}

/// The engine's `polish` command. `local_history` tells the engine whether it
/// may keep the app and text-box context in the dictation's corpus entry;
/// without it nothing read from the screen or a field reaches the disk.
enum PolishCommand {
    static func make(
        text: String, id: String, screenText: String, target: DictationTarget?,
        textbox: TextboxContext?, localHistory: Bool
    ) -> [String: Any] {
        var command: [String: Any] = [
            "cmd": "polish", "text": text, "context": screenText, "id": id,
            "local_history": localHistory,
        ]
        if let app = target?.enginePayload, !app.isEmpty { command["app"] = app }
        if let textbox, !textbox.isEmpty, target?.denied != true {
            command["textbox"] = textbox.enginePayload
        }
        return command
    }
}

/// The text around the cursor when a dictation starts, like Wispr's
/// `textbox_contents` (before_text / selected_text / after_text). Sent to the
/// local correction model only, and only with the setting on.
struct TextboxContext: Equatable {
    /// UTF-16 units kept on each side; enough for the sentence being written
    /// and the paragraph before it.
    static let maxBefore = 1_200
    static let maxAfter = 400
    static let maxSelected = 1_200

    let before: String
    let selected: String
    let after: String

    var isEmpty: Bool { before.isEmpty && selected.isEmpty && after.isEmpty }

    var enginePayload: [String: Any] {
        ["before_text": before, "selected_text": selected, "after_text": after]
    }

    /// Cut the field's text at the selection, keeping the text nearest the
    /// cursor and never splitting a composed character (emoji, accents).
    static func around(value: String, selection: NSRange) -> TextboxContext {
        let text = value as NSString
        let location = min(max(0, selection.location), text.length)
        let end = min(text.length, location + max(0, selection.length))
        func slice(_ range: NSRange) -> String {
            guard range.length > 0 else { return "" }
            return text.substring(with: text.rangeOfComposedCharacterSequences(for: range))
        }
        let beforeStart = max(0, location - maxBefore)
        let before = slice(NSRange(location: beforeStart, length: location - beforeStart))
        let selectedLength = min(end - location, maxSelected)
        let selected = slice(NSRange(location: location, length: selectedLength))
        let afterLength = min(text.length - end, maxAfter)
        let after = slice(NSRange(location: end, length: afterLength))
        return TextboxContext(
            before: trimmedToBound(before, max: maxBefore, keepEnd: true),
            selected: trimmedToBound(selected, max: maxSelected, keepEnd: false),
            after: trimmedToBound(after, max: maxAfter, keepEnd: false))
    }

    /// `rangeOfComposedCharacterSequences` can widen a range by a unit or
    /// two; trim back by whole characters from the far side.
    private static func trimmedToBound(_ text: String, max: Int, keepEnd: Bool) -> String {
        var result = text
        while (result as NSString).length > max, !result.isEmpty {
            if keepEnd { result.removeFirst() } else { result.removeLast() }
        }
        return result
    }

}

/// Whether a field read at key-down may be used: it must belong to the app
/// the dictation goes to (the user may have switched apps since) and must
/// not be a password field or one of Phonon's own. The private list is
/// checked by the caller before any read.
enum FieldReadPolicy {
    static func accepts(fieldPID: pid_t, targetPID: pid_t?, ownPID: pid_t, isSecure: Bool) -> Bool {
        guard let targetPID, targetPID > 0 else { return false }
        return fieldPID == targetPID && fieldPID != ownPID && !isSecure
    }
}

/// The focused field read once at key-down, for text-box context and for the
/// edit-learning anchors.
struct KeyDownFieldRead {
    let textbox: TextboxContext?
    let editSnapshot: PrePasteSnapshot?

    /// Blocking AX; call off the main thread, after `target` was resolved
    /// from the same focus.
    static func read(
        target: DictationTarget, targetPID: pid_t?, wantTextbox: Bool, wantEditSnapshot: Bool
    ) -> KeyDownFieldRead {
        let none = KeyDownFieldRead(textbox: nil, editSnapshot: nil)
        guard wantTextbox || wantEditSnapshot, !target.denied,
            let snapshot = TextboxAccess.focusedSnapshot(),
            FieldReadPolicy.accepts(
                fieldPID: snapshot.pid, targetPID: targetPID,
                ownPID: ProcessInfo.processInfo.processIdentifier,
                isSecure: snapshot.isSecure),
            let value = snapshot.value, let selection = snapshot.selection
        else { return none }
        var textbox: TextboxContext?
        if wantTextbox {
            let context = TextboxContext.around(value: value, selection: selection)
            textbox = context.isEmpty ? nil : context
        }
        var editSnapshot: PrePasteSnapshot?
        if wantEditSnapshot, (value as NSString).length <= EditObservationTracker.maxContentsUTF16 {
            editSnapshot = PrePasteSnapshot(
                element: snapshot.element, pid: snapshot.pid, value: value, selection: selection)
        }
        return KeyDownFieldRead(textbox: textbox, editSnapshot: editSnapshot)
    }
}

/// The app (and browser tab) a dictation is going to, captured when the
/// recording starts.
struct DictationTarget: Equatable {
    let bundleID: String?
    let name: String?
    let urlHost: String?
    let denied: Bool

    /// The `app` object of a polish request. A denylisted target still names
    /// its app (the category shapes formatting) but never its page host.
    var enginePayload: [String: Any] {
        var payload: [String: Any] = [:]
        if let bundleID { payload["bundle_id"] = bundleID }
        if let name { payload["name"] = name }
        if let urlHost, !denied { payload["url_host"] = urlHost }
        return payload
    }

    /// `urlLookedUp`: the tab's URL was asked for. A browser whose URL could
    /// not be read counts as private (fail closed); before the lookup, at
    /// key-down, it is only provisional and nothing is read from it yet.
    /// `url` is the top-level page's (see `BrowserPage`); only http and https
    /// give a host. `isBrowser` is injectable for tests.
    static func resolve(
        bundleID: String?, name: String?, url: URL?, denylist: [String], urlLookedUp: Bool = true,
        isBrowser: (String?) -> Bool = ContextPolicy.isBrowser
    ) -> DictationTarget {
        let host = BrowserPage.pageHost(url)
        let unknownApp = bundleID?.isEmpty ?? true
        let unknownTab = urlLookedUp && host == nil && isBrowser(bundleID)
        return DictationTarget(
            bundleID: bundleID, name: name, urlHost: host,
            denied: unknownApp || unknownTab
                || ContextPolicy.isDenied(bundleID: bundleID, host: host, denylist: denylist))
    }

    /// Whether screen context must wait for the tab lookup before reading
    /// anything (the page might be private).
    var awaitsTabLookup: Bool { ContextPolicy.isBrowser(bundleID) && urlHost == nil }

    /// Read the focused element's top-level page URL (walking only up its
    /// ancestors to the outermost AXWebArea). Blocking AX calls, each bounded
    /// by `TextboxAccess.messagingTimeout`; call off the main thread.
    /// `readsFields`: edit learning or text-box context will read this app's
    /// fields; `screenContext`: its screen may be read. Either way an
    /// Electron app or a browser is asked to expose its accessibility tree
    /// (AXManualAccessibility, once per process), so its page can be known.
    static func capture(
        bundleID: String?, name: String?, pid: pid_t? = nil, readsFields: Bool = false,
        screenContext: Bool = false, denylist: [String]
    ) -> DictationTarget {
        captureWithFocus(
            bundleID: bundleID, name: name, pid: pid, readsFields: readsFields,
            screenContext: screenContext, denylist: denylist
        ).target
    }

    /// `capture`, plus what its one walk found: the page and, for a standard
    /// browser window, its window server id (empty when the app is private
    /// or the focus is in another app).
    static func captureWithFocus(
        bundleID: String?, name: String?, pid: pid_t? = nil, readsFields: Bool = false,
        screenContext: Bool = false, denylist: [String]
    ) -> (target: DictationTarget, focus: BrowserPage.Focus) {
        // A denylisted app, or one that cannot be identified, is not read at
        // all, not even for its URL.
        guard let bundleID, !bundleID.isEmpty,
            !ContextPolicy.isDenied(bundleID: bundleID, host: nil, denylist: denylist)
        else {
            return (
                DictationTarget(bundleID: bundleID, name: name, urlHost: nil, denied: true),
                BrowserPage.Focus()
            )
        }
        if let pid, readsFields || (screenContext && ContextPolicy.isBrowser(bundleID)) {
            TextboxAccess.enableManualAccessibility(pid: pid)
        }
        var focus = BrowserPage.focusedFocus()
        // The focus must still be in the dictation's app.
        if let pid, focus.pid != pid { focus = BrowserPage.Focus() }
        return (resolve(bundleID: bundleID, name: name, url: focus.url, denylist: denylist), focus)
    }
}
