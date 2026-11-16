import Foundation

/// Who is the app the user was typing in, and how Phonon's start link carries that to the app.
///
/// On iOS 26.7.1 the keyboard cannot read the host's bundle id: UIKit's `_hostApplicationBundleIdentifier` answers a
/// literal "<null>" and the signing-identifier and pid lookups are refused by the keyboard sandbox (return.jsonl,
/// 2026-11-15). What does still arrive is the host's pid (from `_hostAuditToken`). So the start link carries the pid
/// and, when the sandbox allows it, the process name, and the app resolves them: RunningBoard by pid, then the
/// process name against a table of well-known apps. Every step is best effort and logged by the caller.
enum HostLookup {
    /// phonon://start?via=..&host=..&pid=..&pname=..
    static func startURL(host: String?, via: String, pid: Int32?, processName: String?) -> URL? {
        var parts = URLComponents(string: "phonon://start")
        var items = [URLQueryItem(name: "via", value: via)]
        if let host { items.append(URLQueryItem(name: "host", value: host)) }
        if let pid, pid > 0 { items.append(URLQueryItem(name: "pid", value: String(pid))) }
        if let processName, !processName.isEmpty { items.append(URLQueryItem(name: "pname", value: processName)) }
        parts?.queryItems = items
        return parts?.url
    }

    struct Hint: Equatable {
        var host: String?
        var via: String?
        var pid: Int32?
        var processName: String?
    }

    static func parse(_ url: URL) -> Hint {
        let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems ?? []
        func value(_ name: String) -> String? {
            items.first(where: { $0.name == name })?.value.flatMap { $0.isEmpty ? nil : $0 }
        }
        return Hint(host: value("host"), via: value("via"), pid: value("pid").flatMap { Int32($0) },
                    processName: value("pname"))
    }

    /// The executable name of a process (kinfo_proc p_comm, at most 15 characters), or nil when sysctl refuses.
    static func processName(pid: Int32) -> String? {
        guard pid > 0 else { return nil }
        var mib: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_PID, pid]
        var info = kinfo_proc()
        var size = MemoryLayout<kinfo_proc>.stride
        guard sysctl(&mib, UInt32(mib.count), &info, &size, nil, 0) == 0, size > 0 else { return nil }
        let name = withUnsafePointer(to: &info.kp_proc.p_comm) {
            $0.withMemoryRebound(to: CChar.self, capacity: Int(MAXCOMLEN) + 1) { String(cString: $0) }
        }
        return name.isEmpty ? nil : name
    }

    /// Executable name -> bundle id for the apps people dictate into most. A name that is not here stays unknown
    /// (the home screen then asks for the swipe back); it is never guessed.
    static let knownApps: [String: String] = [
        "MobileSMS": "com.apple.MobileSMS", "MobileSafari": "com.apple.mobilesafari",
        "MobileNotes": "com.apple.mobilenotes", "Notes": "com.apple.mobilenotes",
        "MobileMail": "com.apple.mobilemail", "Mail": "com.apple.mobilemail",
        "Reminders": "com.apple.reminders",
        "WhatsApp": "net.whatsapp.WhatsApp", "Slack": "com.tinyspeck.chatlyio",
        "Discord": "com.hammerandchisel.discord", "Telegram": "ph.telegra.Telegraph",
        "Gmail": "com.google.Gmail", "Chrome": "com.google.chrome.ios", "Messenger": "com.facebook.Messenger",
        "Instagram": "com.burbn.instagram", "ChatGPT": "com.openai.chat",
        "Signal": "org.whispersystems.signal", "Outlook": "com.microsoft.Office.Outlook",
        "Teams": "com.microsoft.skype.teams", "Notion": "notion.id", "Obsidian": "md.obsidian",
        "Bear": "net.shinyfrog.bear", "Reddit": "com.reddit.Reddit", "LinkedIn": "com.linkedin.LinkedIn",
        "Twitter": "com.atebits.Tweetie2", "X": "com.atebits.Tweetie2", "Threads": "com.burbn.barcelona",
        "Firefox": "org.mozilla.ios.Firefox",
    ]

    static func bundle(forProcessName name: String?) -> String? {
        guard let name else { return nil }
        return knownApps[name]
    }

    /// pid -> bundle id through RunningBoard's RBSProcessHandle (private), looked up at runtime. Returns the id (nil
    /// when refused) and why, for the log. Callable from the keyboard and the app; each is refused or not by its own
    /// sandbox.
    static func bundleForPid(_ pid: pid_t) -> (String?, String) {
        guard let idCls = NSClassFromString("RBSProcessIdentifier") as AnyObject?,
              let hCls = NSClassFromString("RBSProcessHandle") as AnyObject? else { return (nil, "rbs:nocls") }
        let mk = NSSelectorFromString("identifierWithPid:")
        let get = NSSelectorFromString("handleForIdentifier:error:")
        guard idCls.responds(to: mk), hCls.responds(to: get) else { return (nil, "rbs:nosel") }
        typealias Mk = @convention(c) (AnyObject, Selector, Int32) -> AnyObject?
        // NSError ** is autoreleasing: a plain UnsafeMutablePointer makes Swift release the error it does not own,
        // and the keyboard dies at the next autorelease pool pop (the Start tap, 2026-11-07, 08, 11)
        typealias Get = @convention(c) (AnyObject, Selector, AnyObject, AutoreleasingUnsafeMutablePointer<NSError?>?) -> AnyObject?
        guard let ident = unsafeBitCast(idCls.method(for: mk), to: Mk.self)(idCls, mk, pid) else { return (nil, "rbs:noident") }
        var err: NSError?
        guard let handle = unsafeBitCast(hCls.method(for: get), to: Get.self)(hCls, get, ident, &err) as? NSObject else {
            return (nil, "rbs:err\(err?.code ?? 0)")
        }
        for path in ["bundle.identifier", "identity.embeddedApplicationIdentifier"] {
            if let id = handle.value(forKeyPath: path) as? String, id.contains(".") { return (id, "rbs:\(path)") }
        }
        return (nil, "rbs:noid")
    }
}

/// The dictation the user asked for by tapping the bar while Phonon was not holding the mic. The tap opens Phonon,
/// and the finger is long gone when the keyboard is back on screen, so the keyboard writes the wish down and acts on
/// it when the session answers (or the keyboard reappears). Without this the first tap only started the session and
/// the user had to tap again (owner report, 2026-11-15).
enum PendingStart {
    /// A wish older than this is stale: the user wandered off, or never came back.
    static let ttl: Double = 30

    /// True when a start wish exists, is fresh and the session is alive.
    static func shouldResume(wantedAt: Double, now: Double, alive: Bool) -> Bool {
        wantedAt > 0 && alive && now >= wantedAt && now - wantedAt <= ttl
    }

    /// True when the wish can be dropped without acting on it (stale, or in the future by a clock change).
    static func isStale(wantedAt: Double, now: Double) -> Bool {
        wantedAt > 0 && (now < wantedAt || now - wantedAt > ttl)
    }
}
