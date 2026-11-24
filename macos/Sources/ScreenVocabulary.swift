import ApplicationServices
import Foundation
import os

/// Temporary per-dictation vocabulary read from the focused window's accessibility text at
/// key-down: product names, acronyms, identifiers and proper nouns on screen, sent to the
/// engine as `screen_terms` so the recognizer and the correction can spell them. Unlike
/// screen OCR (Screen Recording, `screen_context`) this needs only Accessibility, and only
/// the extracted terms ever leave this file, never the window text.

/// Pure term extraction from window text.
enum ScreenVocabularyExtractor {
    static let defaultMaxTerms = 24
    static let maxTokenLength = 32

    /// UI chrome and log labels that are on nearly every screen and are never what was said.
    static let stoplist: Set<String> = [
        "recent", "sidebar", "settings", "search", "home", "file", "files", "edit", "view",
        "window", "windows", "help", "new", "open", "save", "close", "share", "more", "today",
        "info", "debug", "error", "warn", "todo", "fixme", "note", "trace",
        // Status, deck, chat and editor chrome (e.g. Lapis, Slack, Chrome, terminals).
        "quiet", "connected", "disconnected", "target", "drafter", "output", "input", "working",
        "tldr", "top", "find", "back", "push", "all", "personal", "ready", "idle", "status",
        "details", "summary", "preview", "history", "messages", "terminal", "console", "actions",
        "options", "profile", "account", "general", "advanced", "default", "auto", "manual",
        "none", "done", "cancel", "submit", "send", "copy", "paste", "delete", "clear", "refresh",
        "loading", "waiting", "thinking", "next", "previous", "hide", "show", "start", "stop",
        "run", "build", "test", "check", "folder", "folders", "tab", "tabs", "panel", "running",
        "active", "menu", "overview", "you", "high", "low", "medium", "fast", "slow", "enable",
        "enabled", "disable", "disabled", "off", "on", "yes", "no", "ok", "okay", "add", "remove",
        "create", "update", "reset", "retry", "continue", "skip", "dismiss", "accept", "reject",
        "allow", "deny", "mute", "unmute", "pin", "unpin", "expand", "collapse", "filter", "sort",
        "export", "import", "download", "upload", "sync", "synced", "offline", "online", "unread",
        "pinned", "draft", "drafts", "inbox", "sent", "archive", "trash", "spam", "reply",
        "forward", "thread", "threads", "channel", "channels", "direct", "workspace", "project",
        "projects", "branch", "commit", "diff", "logs", "prompt", "response", "session",
        "sessions",
    ]

    /// Suffixes that make a leading number a measurement (`100ms`, `1080p`, `16gb`).
    static let unitSuffixes: Set<String> = [
        "ns", "us", "µs", "ms", "s", "sec", "secs", "min", "mins", "h", "hr", "hrs", "d",
        "day", "days", "wk", "mo", "yr", "yrs", "p", "i", "x", "k", "m", "b", "t", "kb", "mb",
        "gb", "tb", "pb", "kib", "mib", "gib", "tib", "bit", "bits", "bps", "kbps", "mbps",
        "gbps", "hz", "khz", "mhz", "ghz", "fps", "px", "pt", "em", "rem", "dpi", "ppi", "w",
        "kw", "v", "mv", "a", "ma", "mah", "wh", "kwh", "mm", "cm", "km", "in", "ft", "mi",
        "g", "kg", "mg", "lb", "lbs", "oz", "c", "f", "l", "ml", "tok", "toks", "tokens", "pct",
        "db", "gbs", "tflops", "gflops", "flops", "op", "ops",
    ]

    /// Common English hyphen compounds that look like identifiers but never are.
    static let hyphenStoplist: Set<String> = [
        "e-mail", "built-in", "real-time", "so-called", "long-term", "short-term", "well-known",
        "up-to-date", "follow-up", "sign-in", "sign-up", "log-in", "opt-in", "opt-out",
        "add-on", "check-in", "one-time", "full-screen", "drag-and-drop", "read-only",
        "high-quality", "low-latency", "end-to-end", "self-hosted", "open-source",
    ]

    private struct Token {
        let text: String
        let sentenceStart: Bool
    }

    /// Terms on screen worth teaching the recognizer, most frequent first (ties: first
    /// seen), at most `maxTerms`. `known`: terms already in the dictionary (any case).
    static func extract(
        from text: String, known: Set<String> = [], maxTerms: Int = defaultMaxTerms
    ) -> [String] {
        guard maxTerms > 0, !text.isEmpty else { return [] }
        let knownLower = Set(known.map { $0.lowercased() })
        var counts: [String: Int] = [:]
        var firstIndex: [String: Int] = [:]
        var form: [String: String] = [:]
        var order = 0
        for token in tokenize(text) {
            guard let term = keep(token.text, sentenceStart: token.sentenceStart) else { continue }
            let key = term.lowercased()
            guard !knownLower.contains(key) else { continue }
            counts[key, default: 0] += 1
            if firstIndex[key] == nil {
                firstIndex[key] = order
                form[key] = term
                order += 1
            }
        }
        let ranked = counts.keys.sorted { a, b in
            let ca = counts[a] ?? 0
            let cb = counts[b] ?? 0
            if ca != cb { return ca > cb }
            return (firstIndex[a] ?? 0) < (firstIndex[b] ?? 0)
        }
        return ranked.prefix(maxTerms).compactMap { form[$0] }
    }

    /// Words of letters, digits, `-` and `_`, each marked when it opens a sentence: the
    /// first word, or one after `.`, `!`, `?` or `:` followed by a space, or after a newline.
    private static func tokenize(_ text: String) -> [Token] {
        var tokens: [Token] = []
        var current = ""
        var sentenceStart = true
        var pendingTerminator = false
        func flush() {
            guard !current.isEmpty else { return }
            let trimmed = current.trimmingCharacters(in: CharacterSet(charactersIn: "-_"))
            if !trimmed.isEmpty {
                tokens.append(Token(text: trimmed, sentenceStart: sentenceStart))
                sentenceStart = false
            }
            current = ""
        }
        for ch in text {
            if ch.isLetter || ch.isNumber || ch == "-" || ch == "_" {
                if pendingTerminator {
                    // `v1.2`, `Node.js`: a terminator not followed by a space is not one.
                    pendingTerminator = false
                }
                current.append(ch)
                continue
            }
            let hadWord = !current.isEmpty
            flush()
            if ch.isNewline {
                sentenceStart = true
                pendingTerminator = false
            } else if ch == "." || ch == "!" || ch == "?" || ch == ":" {
                pendingTerminator = true
            } else if ch.isWhitespace {
                if pendingTerminator { sentenceStart = true }
                pendingTerminator = false
            } else if !hadWord && pendingTerminator {
                // quotes or brackets after a terminator: still waiting for the space
                continue
            }
        }
        flush()
        return tokens
    }

    /// The term to keep for one token, or nil.
    static func keep(_ token: String, sentenceStart: Bool) -> String? {
        let length = token.count
        guard length >= 2, length <= maxTokenLength else { return nil }
        let lower = token.lowercased()
        if stoplist.contains(lower) { return nil }
        let letters = token.filter { $0.isLetter }
        guard !letters.isEmpty else { return nil }  // numbers, times (`10:30` splits to digits)
        let hasDigit = token.contains { $0.isNumber }
        let hasUpper = letters.contains { $0.isUppercase }
        let hasLower = letters.contains { $0.isLowercase }
        if isOrdinal(lower) || isUnitNumber(lower) || isHexHash(lower) { return nil }

        if token.contains("-") || token.contains("_") {
            if hyphenStoplist.contains(lower) { return nil }
            let parts = token.split(whereSeparator: { $0 == "-" || $0 == "_" })
            guard parts.count >= 2, parts.allSatisfy({ !$0.isEmpty }), length >= 3 else {
                return nil
            }
            // `2024-11-21`, `10-20`: no letters in most parts is a date or a range.
            let partsWithLetters = parts.filter { $0.contains { $0.isLetter } }.count
            guard partsWithLetters * 2 >= parts.count else { return nil }
            return token
        }
        if hasDigit {
            // `M4`, `LFM3`, `RTX3090`; lowercase two-character pairs (`v2`) say too little.
            if length == 2 && !hasUpper { return nil }
            return token
        }
        // Letters only from here.
        if hasUpper && !hasLower {
            // `API`, `CUDA`, `VRAM`; longer all-caps runs are shouting or headings.
            return (2...6).contains(length) ? token : nil
        }
        guard length >= 3 else { return nil }
        if !hasUpper { return nil }  // plain lowercase word
        let first = token.first!
        let innerUpper = token.dropFirst().contains { $0.isUppercase }
        if innerUpper || first.isLowercase {
            return token  // `FluidAudio`, `OpenCode`, `iPhone`, `macOS`
        }
        // Capitalized word: a proper noun only mid-sentence.
        return sentenceStart ? nil : token
    }

    static func isOrdinal(_ lower: String) -> Bool {
        for suffix in ["st", "nd", "rd", "th"] where lower.hasSuffix(suffix) {
            let digits = lower.dropLast(2)
            if !digits.isEmpty && digits.allSatisfy({ $0.isASCII && $0.isNumber }) { return true }
        }
        return false
    }

    /// A number with a unit suffix: `100ms`, `1.5gb`, `1080p`, `30x`, `48khz`.
    static func isUnitNumber(_ lower: String) -> Bool {
        guard let first = lower.first, first.isNumber else { return false }
        let digitsEnd = lower.firstIndex { !($0.isNumber || $0 == ".") } ?? lower.endIndex
        guard digitsEnd != lower.endIndex else { return false }
        return unitSuffixes.contains(String(lower[digitsEnd...]))
    }

    /// Commit hashes and hex ids: 6+ hex characters with a digit, or a `0x` literal.
    static func isHexHash(_ lower: String) -> Bool {
        if lower.hasPrefix("0x"), lower.count > 2,
            lower.dropFirst(2).allSatisfy({ $0.isHexDigit })
        {
            return true
        }
        guard lower.count >= 6, lower.allSatisfy({ $0.isHexDigit }) else { return false }
        return lower.contains { $0.isNumber }
    }
}

/// Bounds of one accessibility walk, pure so tests can drive them.
struct ScreenVocabularyBudget {
    static let maxElements = 1_500
    static let maxDepth = 40
    static let maxBytes = 32 * 1024
    static let deadlineSeconds: TimeInterval = 0.15
    static let maxValueChars = 4_096
    static let messagingTimeout: Float = 0.25

    static let secureRole = "AXSecureTextField"

    /// A password field (by role or subrole) is never read, nor anything inside it.
    static func isSecure(role: String?, subrole: String?) -> Bool {
        role == secureRole || subrole == secureRole
    }

    enum ValueRead: Equatable {
        /// Read `AXValue` (short enough, or the length is unknown).
        case full
        /// Too long: read only the visible range, never the whole document.
        case visibleRange
    }

    static func valueRead(numberOfCharacters: Int?) -> ValueRead {
        guard let numberOfCharacters, numberOfCharacters > maxValueChars else { return .full }
        return .visibleRange
    }

    /// Whether the walk may visit another element.
    static func mayContinue(elements: Int, bytes: Int, elapsed: TimeInterval) -> Bool {
        elements < maxElements && bytes < maxBytes && elapsed < deadlineSeconds
    }
}

/// Reads the focused window's accessibility text, bounded and private-app aware.
enum ScreenVocabularyReader {
    private static let log = Logger(subsystem: "com.phonon.bar", category: "screen-vocabulary")

    /// The policy half of the read: never Phonon itself, never an unidentified or private app.
    static func allows(pid: pid_t, bundleID: String?, denylist: [String], ownPID: pid_t) -> Bool {
        guard pid > 0, pid != ownPID else { return false }
        guard let bundleID, !bundleID.isEmpty else { return false }
        return !ContextPolicy.isDenied(bundleID: bundleID, host: nil, denylist: denylist)
    }

    /// Blocking AX; call off the main thread. Nil when refused or nothing was read.
    static func readFocusedWindowText(
        pid: pid_t, bundleID: String?, denylist: [String], ownPID: pid_t
    ) -> String? {
        guard allows(pid: pid, bundleID: bundleID, denylist: denylist, ownPID: ownPID),
            AXIsProcessTrusted()
        else { return nil }
        let started = Date()
        let app = AXUIElementCreateApplication(pid)
        AXUIElementSetMessagingTimeout(app, ScreenVocabularyBudget.messagingTimeout)
        guard let window: AXUIElement = copyAttribute(app, kAXFocusedWindowAttribute) else {
            return nil
        }
        var queue: [(element: AXUIElement, depth: Int)] = [(window, 0)]
        var head = 0
        var visited = 0
        var bytes = 0
        var pieces: [String] = []
        func add(_ piece: String?) {
            guard let piece else { return }
            let trimmed = piece.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return }
            let room = ScreenVocabularyBudget.maxBytes - bytes
            guard room > 0 else { return }
            var kept = trimmed
            if kept.utf8.count > room {
                kept = String(decoding: kept.utf8.prefix(room), as: UTF8.self)
            }
            bytes += kept.utf8.count
            pieces.append(kept)
        }
        while head < queue.count,
            ScreenVocabularyBudget.mayContinue(
                elements: visited, bytes: bytes, elapsed: Date().timeIntervalSince(started))
        {
            let (element, depth) = queue[head]
            head += 1
            visited += 1
            let role: String? = copyAttribute(element, kAXRoleAttribute)
            let subrole: String? = copyAttribute(element, kAXSubroleAttribute)
            if ScreenVocabularyBudget.isSecure(role: role, subrole: subrole) { continue }
            add(copyAttribute(element, kAXTitleAttribute))
            add(readValue(element))
            add(copyAttribute(element, kAXDescriptionAttribute))
            guard depth + 1 < ScreenVocabularyBudget.maxDepth else { continue }
            let room = ScreenVocabularyBudget.maxElements - (queue.count - head) - visited
            guard room > 0 else { continue }
            var children: CFArray?
            let status = AXUIElementCopyAttributeValues(
                element, kAXChildrenAttribute as CFString, 0, room, &children)
            guard status == .success, let children = children as? [AXUIElement] else { continue }
            for child in children { queue.append((child, depth + 1)) }
        }
        let elapsedMs = Int(Date().timeIntervalSince(started) * 1_000)
        // Never the text: only how much was walked.
        log.debug(
            "focused window read: \(visited) elements, \(bytes) bytes, \(elapsedMs) ms")
        return pieces.isEmpty ? nil : pieces.joined(separator: "\n")
    }

    /// The element's text value, never materializing a huge document: past
    /// `maxValueChars` only the visible range is read, and nothing if that is unavailable.
    private static func readValue(_ element: AXUIElement) -> String? {
        let count: Int? = copyAttribute(element, kAXNumberOfCharactersAttribute)
        switch ScreenVocabularyBudget.valueRead(numberOfCharacters: count) {
        case .full:
            guard let value: String = copyAttribute(element, kAXValueAttribute) else { return nil }
            return value.count > ScreenVocabularyBudget.maxValueChars
                ? String(value.prefix(ScreenVocabularyBudget.maxValueChars)) : value
        case .visibleRange:
            var rangeValue: CFTypeRef?
            guard
                AXUIElementCopyAttributeValue(
                    element, kAXVisibleCharacterRangeAttribute as CFString, &rangeValue)
                    == .success,
                let rangeValue, CFGetTypeID(rangeValue) == AXValueGetTypeID()
            else { return nil }
            let axRange = rangeValue as! AXValue
            var range = CFRange()
            guard AXValueGetValue(axRange, .cfRange, &range), range.length > 0 else { return nil }
            range.length = min(range.length, ScreenVocabularyBudget.maxValueChars)
            guard let bounded = AXValueCreate(.cfRange, &range) else { return nil }
            var text: CFTypeRef?
            guard
                AXUIElementCopyParameterizedAttributeValue(
                    element, kAXStringForRangeParameterizedAttribute as CFString, bounded, &text)
                    == .success
            else { return nil }
            return text as? String
        }
    }

    private static func copyAttribute<T>(_ element: AXUIElement, _ name: String) -> T? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, name as CFString, &value) == .success,
            let value
        else { return nil }
        if T.self == AXUIElement.self {
            guard CFGetTypeID(value) == AXUIElementGetTypeID() else { return nil }
            return (value as! T)
        }
        return value as? T
    }
}

/// Single-flight generation tracker: one read per dictation, and a read that finishes after
/// its dictation was cancelled or superseded is dropped.
struct ScreenVocabularySession {
    private(set) var generation = 0
    private var inFlight: Int?

    /// A new dictation's read; any earlier one becomes stale.
    mutating func begin() -> Int {
        generation += 1
        inFlight = generation
        return generation
    }

    /// The dictation ended or was dropped: whatever is in flight is stale.
    mutating func cancel() {
        generation += 1
        inFlight = nil
    }

    /// Whether a finished read may be used. True once, for the current read only.
    mutating func accept(_ token: Int) -> Bool {
        guard let inFlight, inFlight == token, token == generation else { return false }
        self.inFlight = nil
        return true
    }

    var isReading: Bool { inFlight != nil }
}
