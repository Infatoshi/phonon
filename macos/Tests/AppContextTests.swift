import XCTest

@testable import PhononBar

final class ContextPolicyTests: XCTestCase {
    func testDefaultsCoverSignalAndDiscordAppAndSite() {
        let list = ContextPolicy.defaultDenylist
        XCTAssertTrue(
            ContextPolicy.isDenied(
                bundleID: "org.whispersystems.signal-desktop", host: nil, denylist: list))
        XCTAssertTrue(ContextPolicy.isDenied(bundleID: "com.hnc.Discord", host: nil, denylist: list))
        XCTAssertTrue(
            ContextPolicy.isDenied(bundleID: "com.google.Chrome", host: "discord.com", denylist: list))
        XCTAssertTrue(
            ContextPolicy.isDenied(
                bundleID: "com.google.Chrome", host: "ptb.discord.com", denylist: list))
        XCTAssertFalse(
            ContextPolicy.isDenied(
                bundleID: "com.google.Chrome", host: "notdiscord.com", denylist: list))
        XCTAssertFalse(
            ContextPolicy.isDenied(bundleID: "com.tinyspeck.slackmacgap", host: nil, denylist: list))
        XCTAssertFalse(ContextPolicy.isDenied(bundleID: nil, host: nil, denylist: list))
    }

    func testEntriesNormalizeFromWhatPeoplePaste() {
        XCTAssertEqual(ContextPolicy.normalizeEntry("  https://www.Discord.com/channels/1 "), "discord.com")
        XCTAssertEqual(ContextPolicy.normalizeEntry("mail.google.com/mail/u/0"), "mail.google.com")
        XCTAssertEqual(ContextPolicy.normalizeEntry("com.hnc.Discord"), "com.hnc.Discord")
        XCTAssertNil(ContextPolicy.normalizeEntry("   "))
        XCTAssertTrue(
            ContextPolicy.isDenied(
                bundleID: "COM.HNC.DISCORD", host: nil, denylist: ["com.hnc.Discord"]))
    }

    func testScreenContextSkipsPrivateWindowsByAppAndBySiteTitle() {
        func window(_ bundle: String, pid: pid_t = 10, title: String?) -> ScreenWindow {
            ScreenWindow(
                id: 1, pid: pid, bundleID: bundle, frame: .zero, layer: 0, alpha: 1, title: title)
        }
        let list = ContextPolicy.defaultDenylist
        XCTAssertTrue(
            ContextPolicy.isPrivateWindow(
                window("com.hnc.Discord", title: "general"), denylist: list, ownPID: 1))
        XCTAssertTrue(
            ContextPolicy.isPrivateWindow(
                window("com.google.Chrome", title: "#general | Discord - Google Chrome"),
                denylist: list, ownPID: 1))
        XCTAssertFalse(
            ContextPolicy.isPrivateWindow(
                window("com.google.Chrome", title: "cuBLAS docs - Google Chrome"),
                denylist: list, ownPID: 1, focusedPage: (1, "docs.nvidia.com")))
        XCTAssertTrue(
            ContextPolicy.isPrivateWindow(
                window("dev.phonon", pid: 1, title: "Phonon"), denylist: list, ownPID: 1))
    }

    /// Fourth review, item 3: a site and its desktop app are one entry;
    /// listing either blocks both, in the foreground and in background
    /// windows. More password managers are always private.
    func testASiteAndItsDesktopAppBlockEachOther() {
        func window(_ bundle: String) -> ScreenWindow {
            ScreenWindow(
                id: 5, pid: 20, bundleID: bundle, frame: .zero, layer: 0, alpha: 1, title: "x")
        }
        XCTAssertTrue(ContextPolicy.isDenied(bundleID: "com.hnc.Discord", host: nil, denylist: ["discord.com"]))
        XCTAssertTrue(
            ContextPolicy.isPrivateWindow(window("com.hnc.Discord"), denylist: ["discord.com"], ownPID: 1),
            "a background Discord app window when only the site is listed")
        for (listed, bundle, host) in [
            ("slack.com", "com.tinyspeck.slackmacgap", "app.slack.com"),
            ("web.whatsapp.com", "net.whatsapp.WhatsApp", "web.whatsapp.com"),
            ("signal.org", "org.whispersystems.signal-desktop", "signal.org"),
            ("messenger.com", "com.facebook.archon", "www.messenger.com"),
            ("telegram.org", "ru.keepcoder.Telegram", "web.telegram.org"),
        ] {
            XCTAssertTrue(
                ContextPolicy.isDenied(bundleID: bundle, host: nil, denylist: [listed]),
                "\(listed) blocks \(bundle)")
            XCTAssertTrue(
                ContextPolicy.isDenied(bundleID: "com.google.Chrome", host: host, denylist: [bundle]),
                "\(bundle) blocks \(host)")
        }
        // A subdomain listed still names the site's app.
        XCTAssertTrue(
            ContextPolicy.isDenied(
                bundleID: "com.tinyspeck.slackmacgap", host: nil, denylist: ["app.slack.com"]))
        // Unrelated sites and apps stay readable.
        XCTAssertFalse(ContextPolicy.isDenied(bundleID: "com.apple.Notes", host: nil, denylist: ["discord.com"]))
        XCTAssertFalse(
            ContextPolicy.isDenied(bundleID: "com.google.Chrome", host: "github.com", denylist: ["com.hnc.Discord"]))
        for bundle in [
            "org.keepassxc.keepassxc", "me.proton.pass.electron", "in.sinew.Enpass-Desktop",
            "com.keepersecurity.passwordmanager",
        ] {
            XCTAssertTrue(ContextPolicy.isDenied(bundleID: bundle, host: nil, denylist: []), bundle)
        }
        XCTAssertTrue(ContextPolicy.isDenied(bundleID: nil, host: "pass.proton.me", denylist: []))
        XCTAssertTrue(ContextPolicy.isDenied(bundleID: nil, host: "keepersecurity.com", denylist: []))
    }

    /// Fifth review, item 5: a known app's bundle id shaped like a host
    /// (`desktop.WhatsApp`) is never a site for window titles.
    func testKnownAppIDsAreNeverTitleBrands() {
        let entries = ContextPolicy.hostEntries(["web.whatsapp.com"])
        XCTAssertTrue(entries.contains("whatsapp.com"))
        XCTAssertFalse(entries.contains { $0.contains("desktop") }, "\(entries)")
        let info = MeetingDetector.recordableWindowInfo(
            title: "Desktop setup - Google Docs - Google Chrome",
            url: URL(string: "https://docs.google.com/document/d/1"),
            bundleID: "com.google.Chrome", denylist: ["whatsapp.com"])
        XCTAssertNotNil(info.title)
        // The app is still blocked by its id.
        XCTAssertTrue(
            ContextPolicy.isDenied(bundleID: "desktop.WhatsApp", host: nil, denylist: ["whatsapp.com"]))
    }

    /// The context lane's request: password prompts and password managers are
    /// always private, and an app or page that cannot be identified is read
    /// as private (fail closed).
    func testPasswordAppsAndUnidentifiedTargetsAreAlwaysPrivate() {
        for bundle in [
            "com.apple.SecurityAgent", "com.1password.1password", "com.bitwarden.desktop",
            "com.apple.keychainaccess", "com.apple.Passwords",
        ] {
            XCTAssertTrue(
                ContextPolicy.isDenied(bundleID: bundle, host: nil, denylist: []),
                "\(bundle) is private even with an empty list")
        }
        XCTAssertTrue(ContextPolicy.isDenied(bundleID: nil, host: "vault.bitwarden.com", denylist: []))
        XCTAssertTrue(ContextPolicy.isDenied(bundleID: nil, host: "my.1password.com", denylist: []))
        func window(_ bundle: String?, _ title: String?) -> ScreenWindow {
            ScreenWindow(
                id: 3, pid: 10, bundleID: bundle, frame: .zero, layer: 0, alpha: 1, title: title)
        }
        XCTAssertTrue(
            ContextPolicy.isPrivateWindow(window(nil, "Untitled"), denylist: [], ownPID: 1),
            "a window whose app is unknown is not read")
        XCTAssertTrue(
            ContextPolicy.isPrivateWindow(
                window("com.google.Chrome", "Release notes - Google Chrome"), denylist: [],
                ownPID: 1),
            "a browser window whose tab host is unknown is not read")
        XCTAssertFalse(
            ContextPolicy.isPrivateWindow(
                window("com.google.Chrome", "Release notes - Google Chrome"), denylist: [],
                ownPID: 1, focusedPage: (3, "github.com")))
        XCTAssertFalse(
            ContextPolicy.isPrivateWindow(window("com.apple.Notes", "Notes"), denylist: [], ownPID: 1))
        // The dictation target: unknown app, or a browser tab that could not
        // be read, reads nothing; before the lookup it only waits.
        XCTAssertTrue(
            DictationTarget.resolve(bundleID: nil, name: nil, url: nil, denylist: []).denied)
        let unreadTab = DictationTarget.resolve(
            bundleID: "com.google.Chrome", name: "Chrome", url: nil, denylist: [])
        XCTAssertTrue(unreadTab.denied)
        let pending = DictationTarget.resolve(
            bundleID: "com.google.Chrome", name: "Chrome", url: nil, denylist: [],
            urlLookedUp: false)
        XCTAssertFalse(pending.denied)
        XCTAssertTrue(pending.awaitsTabLookup, "screen context waits for the tab")
        XCTAssertFalse(
            DictationTarget.resolve(
                bundleID: "com.apple.Notes", name: "Notes", url: nil, denylist: []
            ).denied)
    }

    /// Review item 10: title matching was a substring of the host's first
    /// label, for every window, and lowercase bundle ids counted as hosts.
    func testTitleFallbackIsNarrowAndBundleIdsAreNotHosts() {
        func window(_ bundle: String, _ title: String) -> ScreenWindow {
            ScreenWindow(
                id: 7, pid: 10, bundleID: bundle, frame: .zero, layer: 0, alpha: 1, title: title)
        }
        let list = ContextPolicy.defaultDenylist + ["docs.google.com", "md.obsidian"]
        let chrome = "com.google.Chrome"
        XCTAssertFalse(
            ContextPolicy.isPrivateWindow(
                window(chrome, "discord.py · GitHub - Google Chrome"), denylist: list, ownPID: 1,
                focusedPage: (7, "github.com")),
            "a page about discord.py is not Discord")
        XCTAssertFalse(
            ContextPolicy.titleNamesSite("discord.py · GitHub - Google Chrome", host: "discord.com"))
        XCTAssertTrue(
            ContextPolicy.isPrivateWindow(
                window(chrome, "#general | Discord - Google Chrome"), denylist: list, ownPID: 1))
        XCTAssertFalse(
            ContextPolicy.isPrivateWindow(
                window("com.apple.Notes", "Discord bot ideas"), denylist: list, ownPID: 1),
            "only browser titles name a site")
        XCTAssertFalse(
            ContextPolicy.isPrivateWindow(
                window(chrome, "Release docs - GitHub - Google Chrome"), denylist: list, ownPID: 1,
                focusedPage: (7, "github.com")),
            "docs.google.com does not hide every window titled docs")
        XCTAssertFalse(
            ContextPolicy.titleNamesSite(
                "Release docs - GitHub - Google Chrome", host: "docs.google.com"))
        XCTAssertTrue(
            ContextPolicy.isPrivateWindow(
                window(chrome, "Plan - Google Docs - Google Chrome"), denylist: list, ownPID: 1,
                focusedPage: (7, "docs.google.com")),
            "a known host decides")
        XCTAssertFalse(ContextPolicy.looksLikeHost("com.openai.chat"))
        XCTAssertFalse(ContextPolicy.looksLikeHost("com.hnc.Discord"))
        XCTAssertTrue(ContextPolicy.looksLikeHost("app.slack.com"))
        XCTAssertTrue(
            ContextPolicy.isDenied(bundleID: "md.obsidian", host: nil, denylist: list))
        XCTAssertTrue(
            ContextPolicy.isDenied(bundleID: "com.openai.chat", host: nil, denylist: ["com.openai.chat"]))
        // md.obsidian's title never names a site: its brand is two letters.
        XCTAssertFalse(
            ContextPolicy.isPrivateWindow(
                window(chrome, "Obsidian notes - Google Chrome"), denylist: ["md.obsidian"],
                ownPID: 1, focusedPage: (7, "obsidian.md")))
    }

    /// Second review, item 1: hosts outside a fixed TLD list were treated as
    /// bundle ids and never blocked.
    func testAnyHostShapedEntryBlocksItsSite() {
        for (entry, host) in [
            ("yandex.ru", "yandex.ru"), ("mybank.com.br", "online.mybank.com.br"),
            ("naver.kr", "mail.naver.kr"), ("bank.co.nz", "bank.co.nz"),
            ("https://www.Example.travel/path", "example.travel"),
        ] {
            XCTAssertTrue(
                ContextPolicy.isDenied(bundleID: nil, host: host, denylist: [entry]),
                "\(entry) must block \(host)")
        }
        // Unsure entries block under both readings.
        XCTAssertTrue(ContextPolicy.isDenied(bundleID: "md.obsidian", host: nil, denylist: ["md.obsidian"]))
        XCTAssertTrue(ContextPolicy.isDenied(bundleID: nil, host: "md.obsidian", denylist: ["md.obsidian"]))
    }

    /// Third review: only the focused browser window is ever read, and only
    /// with its page host known; background browser windows never are.
    func testOnlyTheFocusedBrowserWindowWithAKnownPageIsRead() {
        let tab = ScreenWindow(
            id: 42, pid: 10, bundleID: "com.google.Chrome", frame: .zero, layer: 0, alpha: 1,
            title: "Inbox - Google Chrome")
        let background = ScreenWindow(
            id: 43, pid: 10, bundleID: "com.google.Chrome", frame: .zero, layer: 0, alpha: 1,
            title: "Inbox - Google Chrome")
        let exclude = ContextPolicy.screenExclusion(
            denylist: ["mail.proton.me"], ownPID: 1, focusedPage: (42, "mail.proton.me"))
        XCTAssertTrue(exclude(tab), "the title does not name the site, the host does")
        XCTAssertTrue(exclude(background), "a background browser window is never read")
        let notPrivate = ContextPolicy.screenExclusion(
            denylist: ["discord.com"], ownPID: 1, focusedPage: (42, "github.com"))
        let titled = ScreenWindow(
            id: 42, pid: 10, bundleID: "com.google.Chrome", frame: .zero, layer: 0, alpha: 1,
            title: "discord bots - GitHub - Google Chrome")
        XCTAssertFalse(notPrivate(titled), "a known host decides, never the title")
        XCTAssertTrue(
            ContextPolicy.screenExclusion(denylist: [], ownPID: 1, focusedPage: nil)(tab),
            "no page known: nothing read")
        let notes = ScreenWindow(
            id: 44, pid: 11, bundleID: "com.apple.Notes", frame: .zero, layer: 0, alpha: 1,
            title: "Notes")
        XCTAssertFalse(exclude(notes), "other apps' windows are read as before")
    }

    func testBundleIdsAreRecognizedByStructure() {
        XCTAssertTrue(ContextPolicy.isCertainlyBundleID("org.whispersystems.signal-desktop"))
        XCTAssertTrue(ContextPolicy.isCertainlyBundleID("com.hnc.Discord"))
        XCTAssertFalse(ContextPolicy.isCertainlyBundleID("app.slack.com"))
        XCTAssertFalse(ContextPolicy.isCertainlyBundleID("yandex.ru"))
    }

    /// Second review, item 2: unread badges, short brands, subdomains of the
    /// registrable name and hyphenated hosts.
    func testTitleFallbackFindsTheSiteBehindBadgesAndSubdomains() {
        // With fail-closed screen reads an unknown-host browser window is
        // skipped anyway; the title match still guards a window whose host
        // is known but whose title shows a private site (and meeting titles).
        func hidden(_ title: String, _ list: [String]) -> Bool {
            list.contains { ContextPolicy.titleNamesSite(title, host: $0) }
        }
        XCTAssertTrue(hidden("(3) Discord | #general | Server - Google Chrome", ["discord.com"]))
        XCTAssertTrue(hidden("(12) Home / X - Google Chrome", ["x.com"]))
        XCTAssertTrue(hidden("WhatsApp - Google Chrome", ["web.whatsapp.com"]))
        XCTAssertTrue(hidden("general (Channel) - Acme - Slack - Google Chrome", ["app.slack.com"]))
        XCTAssertTrue(hidden("Accounts - My Bank - Google Chrome", ["my-bank.com"]))
        XCTAssertTrue(hidden("Login | MyBank - Google Chrome", ["mybank.com.br"]))
        XCTAssertFalse(hidden("discord.py · GitHub - Google Chrome", ["discord.com"]))
        XCTAssertFalse(hidden("Xcode tips - Google Chrome", ["x.com"]))
        XCTAssertFalse(hidden("Slacking off - Google Chrome", ["app.slack.com"]))
    }

    func testOldSettingsGetTheDefaultListButAnEmptyListStaysEmpty() throws {
        let absent = try JSONDecoder().decode(
            NativeSettings.self, from: Data(#"{"schema_version":2}"#.utf8))
        XCTAssertEqual(absent.screenContextDenylist, ContextPolicy.defaultDenylist)
        let cleared = try JSONDecoder().decode(
            NativeSettings.self,
            from: Data(#"{"schema_version":2,"screen_context_denylist":[]}"#.utf8))
        XCTAssertEqual(cleared.screenContextDenylist, [])
    }
}

final class DictationTargetTests: XCTestCase {
    func testABrowserTabSendsItsHostUnlessTheSiteIsPrivate() {
        let gmail = DictationTarget.resolve(
            bundleID: "com.google.Chrome", name: "Google Chrome",
            url: URL(string: "https://mail.google.com/mail/u/0/#inbox"),
            denylist: ContextPolicy.defaultDenylist)
        XCTAssertFalse(gmail.denied)
        XCTAssertEqual(gmail.urlHost, "mail.google.com")
        XCTAssertEqual(gmail.enginePayload["url_host"] as? String, "mail.google.com")
        XCTAssertEqual(gmail.enginePayload["bundle_id"] as? String, "com.google.Chrome")

        let discord = DictationTarget.resolve(
            bundleID: "com.google.Chrome", name: "Google Chrome",
            url: URL(string: "https://discord.com/channels/@me"),
            denylist: ContextPolicy.defaultDenylist)
        XCTAssertTrue(discord.denied)
        XCTAssertNil(discord.enginePayload["url_host"], "a private site's host never leaves the bar")
        XCTAssertEqual(discord.enginePayload["bundle_id"] as? String, "com.google.Chrome")
    }

    func testAnAppWithoutAPageHasNoHost() {
        let terminal = DictationTarget.resolve(
            bundleID: "com.mitchellh.ghostty", name: "Ghostty", url: nil, denylist: [])
        XCTAssertNil(terminal.urlHost)
        XCTAssertEqual(terminal.enginePayload.count, 2)
    }
}

/// Review item 1: the field read for edit learning and text-box context must
/// belong to the dictation's own app and never to a private one.
final class FieldReadPolicyTests: XCTestCase {
    func testOnlyTheDictationAppsOwnNonPrivateFieldIsRead() {
        let notes: pid_t = 400
        let signal: pid_t = 500
        XCTAssertTrue(
            FieldReadPolicy.accepts(fieldPID: notes, targetPID: notes, ownPID: 1, isSecure: false))
        XCTAssertFalse(
            FieldReadPolicy.accepts(fieldPID: signal, targetPID: notes, ownPID: 1, isSecure: false),
            "dictated in Notes, focus moved to Signal: Signal's field is never read")
        XCTAssertFalse(
            FieldReadPolicy.accepts(fieldPID: notes, targetPID: notes, ownPID: 1, isSecure: true))
        XCTAssertFalse(FieldReadPolicy.accepts(fieldPID: 1, targetPID: 1, ownPID: 1, isSecure: false))
        XCTAssertFalse(
            FieldReadPolicy.accepts(fieldPID: notes, targetPID: nil, ownPID: 1, isSecure: false),
            "no target app (Phonon was in front): nothing is read")
    }

    func testAPrivateTargetReadsNoField() {
        let target = DictationTarget(
            bundleID: "org.whispersystems.signal-desktop", name: "Signal", urlHost: nil,
            denied: true)
        let read = KeyDownFieldRead.read(
            target: target, targetPID: 500, wantTextbox: true, wantEditSnapshot: true)
        XCTAssertNil(read.textbox)
        XCTAssertNil(read.editSnapshot)
    }
}

final class TextboxContextTests: XCTestCase {
    func testCursorInTheMiddleSplitsBeforeAndAfter() {
        let value = "Hi Bob,\n\nI think the kernel is fine. \n\nThanks,\nSam"
        let cursor = (value as NSString).range(of: "\n\nThanks").location
        let context = TextboxContext.around(
            value: value, selection: NSRange(location: cursor, length: 0))
        XCTAssertEqual(context.before, "Hi Bob,\n\nI think the kernel is fine. ")
        XCTAssertEqual(context.selected, "")
        XCTAssertEqual(context.after, "\n\nThanks,\nSam")
    }

    func testASelectionIsItsOwnPart() {
        let context = TextboxContext.around(
            value: "use coo blast here", selection: NSRange(location: 4, length: 9))
        XCTAssertEqual(context.before, "use ")
        XCTAssertEqual(context.selected, "coo blast")
        XCTAssertEqual(context.after, " here")
    }

    func testLongFieldsKeepTheTextNearestTheCursorWithoutSplittingEmoji() {
        let filler = String(repeating: "a", count: TextboxContext.maxBefore - 1)
        let value = "start " + "👍" + filler + "|after"
        let cursor = (value as NSString).range(of: "|").location
        let context = TextboxContext.around(
            value: value, selection: NSRange(location: cursor, length: 0))
        XCTAssertLessThanOrEqual((context.before as NSString).length, TextboxContext.maxBefore)
        XCTAssertTrue(context.before.hasSuffix("aaa"))
        XCTAssertFalse(context.before.contains("start"))
        XCTAssertFalse(context.before.unicodeScalars.contains { $0.value == 0xFFFD })
        XCTAssertEqual(context.after, "|after")
    }

    func testEmptyFieldAndOutOfRangeSelectionsAreSafe() {
        XCTAssertTrue(
            TextboxContext.around(value: "", selection: NSRange(location: 0, length: 0)).isEmpty)
        let clamped = TextboxContext.around(
            value: "abc", selection: NSRange(location: 10, length: 5))
        XCTAssertEqual(clamped.before, "abc")
        XCTAssertEqual(clamped.after, "")
    }

    func testPolishCommandCarriesContextOnlyWhereAllowed() {
        let target = DictationTarget(
            bundleID: "com.apple.mail", name: "Mail", urlHost: nil, denied: false)
        let textbox = TextboxContext(before: "Hi Bob, ", selected: "", after: "")
        let command = PolishCommand.make(
            text: "the kernel is fine", id: "p1", screenText: "", target: target,
            textbox: textbox, localHistory: false)
        XCTAssertEqual((command["textbox"] as? [String: Any])?["before_text"] as? String, "Hi Bob, ")
        XCTAssertEqual(command["local_history"] as? Bool, false)
        XCTAssertEqual((command["app"] as? [String: Any])?["bundle_id"] as? String, "com.apple.mail")

        let privateTarget = DictationTarget(
            bundleID: "com.hnc.Discord", name: "Discord", urlHost: nil, denied: true)
        let denied = PolishCommand.make(
            text: "hi", id: "p2", screenText: "", target: privateTarget, textbox: textbox,
            localHistory: true)
        XCTAssertNil(denied["textbox"], "never the text of a private app")
        let off = PolishCommand.make(
            text: "hi", id: "p3", screenText: "", target: target, textbox: nil, localHistory: true)
        XCTAssertNil(off["textbox"])
    }
}

@MainActor
final class DenylistStoreTests: XCTestCase {
    func testAddAndRemoveKeepTheListClean() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "phonon-denylist-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = NativeAppStore(supportDirectory: directory)
        XCTAssertEqual(store.settings.screenContextDenylist, ContextPolicy.defaultDenylist)
        store.addToDenylist("https://www.Example.com/private")
        store.addToDenylist("example.com")
        XCTAssertEqual(store.settings.screenContextDenylist.filter { $0 == "example.com" }.count, 1)
        store.removeFromDenylist("discord.com")
        XCTAssertFalse(store.settings.screenContextDenylist.contains("discord.com"))
        let reloaded = NativeAppStore(supportDirectory: directory)
        XCTAssertTrue(reloaded.settings.screenContextDenylist.contains("example.com"))
    }
}
