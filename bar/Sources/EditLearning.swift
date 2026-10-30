import Foundation

// Post-paste edit learning, the pure half. Everything here is plain values so
// it can be tested without Accessibility, windows or an engine. The AX reads
// live in `TextboxAccess.swift`; the panel and the polling loop in
// `EditLearningPanel.swift`; the diff and the proposals in Rust
// (`crates/phonon-core/src/edits.rs`).
//
// Mechanism, after Wispr Flow 1.6's Swift helper (`EditedTextManager2`): pin
// the focused text field at paste, remember the text before and after the
// insertion point as anchors, then re-read the field and take whatever sits
// between the anchors as the user's version of the dictation. The watch ends
// on a fixed window, when the field empties (message sent or cleared), when
// the dictation region is deleted, when the anchors stop matching, or when
// the next dictation starts. Phonon adds one ending of its own: a few seconds
// of quiet after an edit, so the proposal shows while the edit is fresh.

/// What kind of key the user pressed last. Never the key itself.
enum KeystrokeCategory: String, Equatable {
    case returnKey = "return"
    case commandReturn = "cmd_enter"
    case backspace
    case forwardDelete = "forward_delete"
    case deleteWord = "delete_word"
    case deleteLine = "delete_line"
    case selectAll = "select_all"
    case selectAllDelete = "select_all_delete"
    case undo = "cmd_z"
    case redo
    case cut = "cmd_x"
    case paste = "cmd_v"
    case escape = "esc"
    case typing

    /// macOS virtual key codes (Carbon `kVK_*`), so this stays testable
    /// without importing Carbon.
    static func classify(
        keyCode: Int, command: Bool, option: Bool, shift: Bool,
        previous: KeystrokeCategory? = nil
    ) -> KeystrokeCategory {
        switch keyCode {
        case 36, 76:  // Return, keypad Enter
            return command ? .commandReturn : .returnKey
        case 51:  // Delete (backspace)
            if command { return .deleteLine }
            if option { return .deleteWord }
            return previous == .selectAll ? .selectAllDelete : .backspace
        case 117:  // Forward delete
            return previous == .selectAll ? .selectAllDelete : .forwardDelete
        case 0 where command:  // A
            return .selectAll
        case 6 where command:  // Z
            return shift ? .redo : .undo
        case 7 where command:  // X
            return .cut
        case 9 where command:  // V
            return .paste
        case 53:
            return .escape
        default:
            return .typing
        }
    }

    var sends: Bool { self == .returnKey || self == .commandReturn }

    /// Keys that throw text away rather than send it.
    var deletes: Bool {
        switch self {
        case .backspace, .forwardDelete, .deleteWord, .deleteLine, .selectAllDelete, .cut, .undo:
            return true
        default:
            return false
        }
    }
}

struct KeystrokeEvent: Equatable {
    let category: KeystrokeCategory
    let at: TimeInterval
}

enum ObservationEndReason: String, Equatable, Error {
    case windowElapsed = "observation_window_elapsed"
    case quietAfterEdit = "quiet_after_edit"
    case textboxEmptied = "textbox_emptied"
    case messageSent = "message_sent"
    case regionEmptied = "transcript_region_emptied"
    case trailingNewline = "trailing_newline_added"
    case anchorMismatch = "anchor_mismatch"
    case anchorMismatchIntact = "anchor_mismatch_transcript_intact"
    case textboxTooLong = "textbox_too_long"
    case textboxGone = "textbox_gone"
    case unreadable = "textbox_unreadable"
    case notFound = "dictated_text_not_found"
    case secureField = "secure_field"
    case nextDictation = "next_dictation"
    case appTermination = "app_termination"
}

/// What the bar sends to the engine as `edit_observed`.
struct ObservationResult: Equatable {
    let pasted: String
    let edited: String
    let editedWithinThreshold: String?
    let endReason: ObservationEndReason
    let lastKeystroke: KeystrokeCategory?
    let lastKeystrokeAgeMs: Int?
    let observedMs: Int
    let changes: Int

    func engineCommand(id: String, record: Bool) -> [String: Any] {
        var command: [String: Any] = [
            "cmd": "edit_observed", "id": id, "pasted": pasted, "edited": edited,
            "end_reason": endReason.rawValue, "observed_ms": observedMs, "record": record,
        ]
        if let editedWithinThreshold { command["edited_within_threshold"] = editedWithinThreshold }
        if let lastKeystroke { command["last_keystroke"] = lastKeystroke.rawValue }
        if let lastKeystrokeAgeMs { command["last_keystroke_age_ms"] = lastKeystrokeAgeMs }
        return command
    }
}

struct EditObservationTracker {
    /// Wispr's helper stops its edited-text listener after 60 seconds.
    static let window: TimeInterval = 60
    /// Phonon ends a watch this long after the last change to the region.
    static let quietAfterEdit: TimeInterval = 5
    /// Fields longer than this (UTF-16 units) are documents, not messages;
    /// re-reading them on a timer costs too much and Wispr skips them too.
    static let maxContentsUTF16 = 20_000
    /// Only this much text on each side anchors the region, so typing far
    /// from the dictation in the same field does not end the watch.
    static let anchorLength = 48
    /// Mirrors `REWRITE_RATIO` in edits.rs: past it the region is a new text.
    static let rewriteRatio = 0.5
    /// The last key counts as the cause of an ending within this window.
    static let keystrokeRecency: TimeInterval = 2
    /// Consecutive failed reads before the field counts as unreadable.
    static let unreadableLimit = 3

    /// The dictation as it landed in the field.
    let baseline: String
    let beforeAnchor: String
    let afterAnchor: String
    /// UTF-16 offset of the region start when it landed.
    let expectedOffset: Int
    let startedAt: TimeInterval
    private(set) var latest: String
    private(set) var withinThreshold: String
    private(set) var lastChangeAt: TimeInterval?
    private(set) var changes = 0
    private(set) var lastKeystroke: KeystrokeEvent?
    private var unreadableReads = 0

    /// Pin the region from the field before paste (its text and selection)
    /// and the field after paste. The region is whatever replaced the
    /// selection; it is usually exactly `pasted`.
    static func start(
        pasted: String, contentsBefore: String, selectionBefore: NSRange,
        contentsAfter: String, now: TimeInterval
    ) -> Result<EditObservationTracker, ObservationEndReason> {
        let beforeText = contentsBefore as NSString
        guard (contentsAfter as NSString).length <= maxContentsUTF16,
            beforeText.length <= maxContentsUTF16
        else { return .failure(.textboxTooLong) }
        let location = min(max(0, selectionBefore.location), beforeText.length)
        let selectionEnd = min(beforeText.length, location + max(0, selectionBefore.length))
        let before = beforeText.substring(to: location)
        let after = beforeText.substring(from: selectionEnd)
        let afterText = contentsAfter as NSString
        // UTF-16 literal comparison: the offsets below are UTF-16, and Swift's
        // own hasPrefix would treat é and e + combining accent as equal.
        let whole = NSRange(location: 0, length: afterText.length)
        guard afterText.length >= (before as NSString).length + (after as NSString).length,
            before.isEmpty
                || afterText.range(of: before, options: [.anchored, .literal], range: whole)
                    .location != NSNotFound,
            after.isEmpty
                || afterText.range(
                    of: after, options: [.anchored, .backwards, .literal], range: whole
                ).location != NSNotFound
        else { return .failure(.notFound) }
        let regionRange = NSRange(
            location: (before as NSString).length,
            length: afterText.length - (before as NSString).length - (after as NSString).length)
        let region = afterText.substring(with: regionRange)
        guard !region.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            Self.plausibleLanding(pasted: pasted, region: region)
        else { return .failure(.notFound) }
        let tracker = EditObservationTracker(
            baseline: region,
            beforeAnchor: String((before as NSString).substring(
                from: max(0, (before as NSString).length - anchorLength))),
            afterAnchor: String((after as NSString).substring(
                to: min((after as NSString).length, anchorLength))),
            expectedOffset: regionRange.location,
            startedAt: now,
            latest: region,
            withinThreshold: region)
        return .success(tracker)
    }

    /// The paste may not have been processed yet (region is the old
    /// selection) or the app may restyle it; accept a region that shares most
    /// of its words with what was pasted.
    static func plausibleLanding(pasted: String, region: String) -> Bool {
        if region == pasted { return true }
        let distance = wordDistance(pasted, region)
        return distance <= 0.34
    }

    mutating func noteKeystroke(_ category: KeystrokeCategory, at time: TimeInterval) {
        lastKeystroke = KeystrokeEvent(category: category, at: time)
    }

    /// Feed one read of the pinned field. Returns why the watch ends, or nil
    /// to keep watching.
    mutating func observe(contents: String?, now: TimeInterval) -> ObservationEndReason? {
        guard let contents else {
            unreadableReads += 1
            if unreadableReads >= Self.unreadableLimit { return .unreadable }
            return now - startedAt >= Self.window ? .windowElapsed : nil
        }
        unreadableReads = 0
        if (contents as NSString).length > Self.maxContentsUTF16 { return .textboxTooLong }
        if contents.isEmpty {
            return recentKeystroke(now: now)?.category.sends == true ? .messageSent : .textboxEmptied
        }
        guard let region = locateRegion(in: contents) else {
            return contents.contains(latest) ? .anchorMismatchIntact : .anchorMismatch
        }
        if region.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return .regionEmptied
        }
        let addedNewline = region.hasSuffix("\n") && !baseline.hasSuffix("\n")
        if addedNewline, recentKeystroke(now: now)?.category == .returnKey {
            record(String(region.trimmingCharacters(in: .newlines)), now: now)
            return .trailingNewline
        }
        record(region, now: now)
        if let lastChangeAt, latest != baseline, now - lastChangeAt >= Self.quietAfterEdit {
            return .quietAfterEdit
        }
        return now - startedAt >= Self.window ? .windowElapsed : nil
    }

    private mutating func record(_ region: String, now: TimeInterval) {
        guard region != latest else { return }
        latest = region
        lastChangeAt = now
        changes += 1
        if Self.wordDistance(baseline, region) <= Self.rewriteRatio {
            withinThreshold = region
        }
    }

    private func recentKeystroke(now: TimeInterval) -> KeystrokeEvent? {
        guard let lastKeystroke, now - lastKeystroke.at <= Self.keystrokeRecency else {
            return nil
        }
        return lastKeystroke
    }

    /// The region between the anchors, choosing the anchor occurrences
    /// nearest to where the dictation was.
    func locateRegion(in contents: String) -> String? {
        let text = contents as NSString
        var start = 0
        if !beforeAnchor.isEmpty {
            let candidates = Self.occurrences(of: beforeAnchor, in: text, from: 0)
            guard
                let best = candidates.min(by: {
                    abs(NSMaxRange($0) - expectedOffset) < abs(NSMaxRange($1) - expectedOffset)
                })
            else { return nil }
            start = NSMaxRange(best)
        }
        var end = text.length
        if !afterAnchor.isEmpty {
            let expectedEnd = start + (latest as NSString).length
            let candidates = Self.occurrences(of: afterAnchor, in: text, from: start)
            guard
                let best = candidates.min(by: {
                    abs($0.location - expectedEnd) < abs($1.location - expectedEnd)
                })
            else { return nil }
            end = best.location
        }
        guard end >= start else { return nil }
        return text.substring(with: NSRange(location: start, length: end - start))
    }

    private static func occurrences(of needle: String, in text: NSString, from start: Int)
        -> [NSRange]
    {
        var found = [NSRange]()
        var searchStart = start
        while searchStart <= text.length {
            let range = text.range(
                of: needle, options: [.literal],
                range: NSRange(location: searchStart, length: text.length - searchStart))
            guard range.location != NSNotFound else { break }
            found.append(range)
            searchStart = range.location + 1
            if found.count >= 64 { break }
        }
        return found
    }

    func finish(reason: ObservationEndReason, now: TimeInterval) -> ObservationResult {
        let keystroke = recentKeystroke(now: now)
        // An emptied field is usually a sent message (90% of the 1,040 watches
        // Wispr recorded on this Mac ended that way, often after a click on
        // Send that no key shows), so the last text read is what the user
        // kept, unless a deleting key just ran. A deleted dictation region
        // inside a field that still has text was thrown away.
        let discarded =
            reason == .regionEmptied
            || (reason == .textboxEmptied && keystroke?.category.deletes == true)
        let edited = discarded ? "" : latest
        return ObservationResult(
            pasted: baseline,
            edited: edited,
            editedWithinThreshold: withinThreshold != edited ? withinThreshold : nil,
            endReason: reason,
            lastKeystroke: keystroke?.category,
            lastKeystrokeAgeMs: keystroke.map { Int(((now - $0.at) * 1_000).rounded()) },
            observedMs: Int(((now - startedAt) * 1_000).rounded()),
            changes: changes)
    }

    /// Word-level Levenshtein over the longer word count, 0 to 1.
    static func wordDistance(_ left: String, _ right: String) -> Double {
        let a = left.split(whereSeparator: { $0.isWhitespace })
        let b = right.split(whereSeparator: { $0.isWhitespace })
        if a.isEmpty && b.isEmpty { return 0 }
        var previous = Array(0...b.count)
        for (i, word) in a.enumerated() {
            var current = [i + 1]
            for (j, other) in b.enumerated() {
                current.append(
                    min(current[j] + 1, previous[j + 1] + 1, previous[j] + (word == other ? 0 : 1)))
            }
            previous = current
        }
        return Double(previous[b.count]) / Double(max(a.count, b.count))
    }
}

// MARK: - Proposals

/// One proposal from the engine's `kind: edit` result.
struct EditSuggestion: Equatable {
    enum Kind: Equatable {
        case term(term: String, heardAs: String, casing: Bool, existingEntry: Bool)
        case style(String)
    }

    let index: Int
    let kind: Kind

    static func parse(_ event: [String: Any]) -> [EditSuggestion] {
        let raw = event["suggestions"] as? [[String: Any]] ?? []
        return raw.enumerated().compactMap { index, item in
            switch item["kind"] as? String {
            case "dictionary_term":
                guard let term = item["term"] as? String, let heard = item["heard_as"] as? String,
                    !term.isEmpty
                else { return nil }
                return EditSuggestion(
                    index: index,
                    kind: .term(
                        term: term, heardAs: heard, casing: item["reason"] as? String == "casing",
                        existingEntry: item["existing_entry"] as? Bool ?? false))
            case "style":
                guard let preference = item["preference"] as? String else { return nil }
                return EditSuggestion(index: index, kind: .style(preference))
            default:
                return nil
            }
        }
    }

    /// Stable key for session-level rejection and counting.
    var key: String {
        switch kind {
        case let .term(term, heardAs, _, _):
            return "term:\(term.lowercased())|\(heardAs.lowercased())"
        case let .style(preference):
            return "style:\(preference)"
        }
    }
}

/// The words on the recommendation panel.
enum EditSuggestionCopy {
    static func title(_ suggestion: EditSuggestion, category: String) -> String {
        switch suggestion.kind {
        case let .term(term, heardAs, casing, existing):
            if casing { return "Always write \u{201C}\(term)\u{201D} like this?" }
            if existing { return "Teach Phonon that \u{201C}\(heardAs)\u{201D} means \(term)?" }
            return "Add \u{201C}\(term)\u{201D} to your dictionary?"
        case let .style(preference):
            let place = placeName(category)
            switch preference {
            case "lowercase_start": return "Start with a lowercase letter \(place)?"
            case "all_lowercase": return "Write in all lowercase \(place)?"
            case "no_trailing_period": return "Leave off the final period \(place)?"
            default: return "Keep this style \(place)?"
            }
        }
    }

    static func detail(_ suggestion: EditSuggestion) -> String {
        switch suggestion.kind {
        case let .term(_, heardAs, casing, _):
            return casing
                ? "Phonon wrote \u{201C}\(heardAs)\u{201D}" : "Heard as \u{201C}\(heardAs)\u{201D}"
        case .style:
            return "You changed this after Phonon typed it"
        }
    }

    static func approveLabel(_ suggestion: EditSuggestion) -> String {
        switch suggestion.kind {
        case .term: return "Add"
        case .style: return "Yes"
        }
    }

    /// One learned writing style as Settings lists it.
    struct StyleRow: Equatable {
        let category: String
        let preference: String
        let text: String
    }

    /// The approved styles in settings.json, in a stable order, as
    /// sentences ("Start with a lowercase letter in chat apps").
    static func styleRows(_ preferences: [String: [String]]) -> [StyleRow] {
        preferences.keys.sorted().flatMap { category in
            (preferences[category] ?? []).sorted().map { preference in
                let place = placeName(category)
                let text: String
                switch preference {
                case "lowercase_start": text = "Start with a lowercase letter \(place)"
                case "all_lowercase": text = "Write in all lowercase \(place)"
                case "no_trailing_period": text = "Leave off the final period \(place)"
                default: text = "\(preference.replacingOccurrences(of: "_", with: " ")) \(place)"
                }
                return StyleRow(category: category, preference: preference, text: text)
            }
        }
    }

    static func placeName(_ category: String) -> String {
        switch category {
        case "chat": return "in chat apps"
        case "email": return "in email"
        case "code": return "in code and terminals"
        case "ai_prompt": return "in AI prompts"
        case "docs": return "in documents"
        default: return "everywhere"
        }
    }
}

/// Where the recommendation card sits: centred, above the dictation capsule.
enum SuggestionPanelGeometry {
    static let size = CGSize(width: 420, height: 58)
    static let bottomOffset: CGFloat = 64

    /// `height` is the card's own fitting height; two-line titles need more
    /// than the default.
    static func frame(screen: CGRect, height: CGFloat? = nil, raisedBy raise: CGFloat = 0)
        -> CGRect
    {
        CGRect(
            x: screen.midX - size.width / 2, y: screen.minY + bottomOffset + raise,
            width: size.width, height: max(size.height, height ?? 0))
    }
}

/// Session memory for proposals, kept in memory only: nothing the user typed
/// is written anywhere unless they approve it (SPEC), so a "Not now" is not
/// persisted either.
struct EditSuggestionFilter {
    /// A style edit has to repeat before Phonon asks about it.
    static let styleRepeats = 2

    private(set) var rejected = Set<String>()
    private var styleCounts = [String: Int]()

    /// The proposals worth showing now, in order. `approvedStyles` are the
    /// preferences already saved for this category.
    mutating func admit(
        _ suggestions: [EditSuggestion], category: String, approvedStyles: Set<String>
    ) -> [EditSuggestion] {
        suggestions.filter { suggestion in
            guard !rejected.contains(suggestion.key) else { return false }
            guard case let .style(preference) = suggestion.kind else { return true }
            guard !approvedStyles.contains(preference) else { return false }
            let key = "\(category)|\(preference)"
            styleCounts[key, default: 0] += 1
            return styleCounts[key, default: 0] >= Self.styleRepeats
        }
    }

    mutating func reject(_ suggestion: EditSuggestion) {
        rejected.insert(suggestion.key)
    }
}
