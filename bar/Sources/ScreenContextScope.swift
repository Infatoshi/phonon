import CoreGraphics
import Foundation

// What screen context reads, the pure half. `ScreenContextCapture` turns
// these decisions into ScreenCaptureKit captures and Vision OCR.
//
// Phonon used to OCR every display whole. It now reads only the windows that
// can be seen on each display (the editor on one, the Chrome tab on the
// other), adds any window or tab the user switches to while speaking, and
// skips apps and sites on the private list entirely.

/// One on-screen window, front to back as `CGWindowListCopyWindowInfo`
/// reports them. Frames are in global display coordinates (top-left origin).
struct ScreenWindow: Equatable {
    let id: UInt32
    let pid: pid_t
    let bundleID: String?
    let frame: CGRect
    /// 0 is the normal window layer; menus, the Dock and overlays sit above.
    let layer: Int
    let alpha: Double
    let title: String?

    /// Identity for "already read this": the same window showing a new tab
    /// or document has a new title, so it counts as new.
    var captureKey: String { "\(pid)|\(id)|\(title ?? "")" }
}

struct ScreenDisplay: Equatable {
    let id: UInt32
    let frame: CGRect
}

enum ScreenWindowSelector {
    /// The occlusion grid per display. 64 x 36 cells is 27 x 31 points on a
    /// 1728 x 1117 display, fine enough to tell a sliver from a window.
    static let gridColumns = 64
    static let gridRows = 36
    /// A window counts as visible when this share of its display shows it.
    static let minDisplayShare = 0.04
    /// Smaller windows are palettes and popovers, not reading material.
    static let minWindowSize = CGSize(width: 160, height: 100)

    /// For each display, the windows a person can see on it, front to back.
    /// `excluded` windows (Phonon's own, the private list) are never picked
    /// but still hide what is behind them.
    static func visibleWindows(
        windows: [ScreenWindow], displays: [ScreenDisplay],
        excluded: (ScreenWindow) -> Bool
    ) -> [UInt32: [ScreenWindow]] {
        var result = [UInt32: [ScreenWindow]]()
        for display in displays {
            var covered = Array(repeating: false, count: gridColumns * gridRows)
            let cellWidth = display.frame.width / CGFloat(gridColumns)
            let cellHeight = display.frame.height / CGFloat(gridRows)
            var picked = [ScreenWindow]()
            for window in windows where window.layer == 0 && window.alpha > 0.05 {
                let visible = window.frame.intersection(display.frame)
                guard !visible.isNull, visible.width > 0, visible.height > 0 else { continue }
                var newlySeen = 0
                for row in 0..<gridRows {
                    for column in 0..<gridColumns {
                        let center = CGPoint(
                            x: display.frame.minX + (CGFloat(column) + 0.5) * cellWidth,
                            y: display.frame.minY + (CGFloat(row) + 0.5) * cellHeight)
                        let index = row * gridColumns + column
                        guard !covered[index], visible.contains(center) else { continue }
                        covered[index] = true
                        newlySeen += 1
                    }
                }
                let share = Double(newlySeen) / Double(gridColumns * gridRows)
                let bigEnough =
                    window.frame.width >= minWindowSize.width
                    && window.frame.height >= minWindowSize.height
                if share >= minDisplayShare, bigEnough, !excluded(window) {
                    picked.append(window)
                }
            }
            result[display.id] = picked
        }
        return result
    }
}

/// Everything screen context read during one dictation: bounded, with OCR
/// lines deduplicated across captures so a window read twice adds nothing.
struct ScreenContextCollection {
    /// Initial displays plus window and tab switches while speaking.
    static let maxCaptures = 8
    /// About four dense screens of text; the engine only intersects it with
    /// dictionary terms, but the polish request should stay small.
    static let maxCharacters = 24_000

    private(set) var captures = 0
    private(set) var seenKeys = Set<String>()
    private(set) var lines = [String]()
    private var lineSet = Set<String>()
    private var characters = 0
    private var truncated = false

    var text: String { lines.joined(separator: "\n") }
    var isFull: Bool {
        truncated || captures >= Self.maxCaptures || characters >= Self.maxCharacters
    }

    /// Whether a window with this key still needs reading.
    func wants(_ key: String) -> Bool { !isFull && !seenKeys.contains(key) }

    /// Mark keys as read without text (windows covered by a display capture).
    mutating func markSeen(_ keys: [String]) {
        seenKeys.formUnion(keys)
    }

    /// Add one capture's OCR text. Returns how many new lines it contributed.
    @discardableResult
    mutating func add(keys: [String], text: String) -> Int {
        seenKeys.formUnion(keys)
        guard captures < Self.maxCaptures else { return 0 }
        captures += 1
        var added = 0
        for raw in text.split(whereSeparator: \.isNewline) {
            let line = raw.trimmingCharacters(in: .whitespaces)
            guard !line.isEmpty, !lineSet.contains(line) else { continue }
            guard characters + line.count + 1 <= Self.maxCharacters else {
                truncated = true
                break
            }
            lineSet.insert(line)
            lines.append(line)
            characters += line.count + 1
            added += 1
        }
        return added
    }
}

/// The screen reads of one dictation. Resetting (a new dictation, or the
/// target turning out to be private) invalidates every read still in
/// flight, so its OCR text can never reach the correction afterwards.
struct ScreenReadLedger {
    private(set) var generation = 0
    private(set) var pending = 0

    mutating func reset() {
        generation += 1
        pending = 0
    }

    /// Start a read; keep the token to hand back when it finishes.
    mutating func begin() -> Int {
        pending += 1
        return generation
    }

    /// A read finished. False when it belongs to an earlier generation and
    /// its text must be dropped.
    mutating func finish(_ token: Int) -> Bool {
        guard token == generation else { return false }
        pending = max(0, pending - 1)
        return true
    }

    func isCurrent(_ token: Int) -> Bool { token == generation }
}

/// How long the final correction may wait for screen context. The screen is
/// read from key-down, so for nearly every dictation it is done before the
/// final transcript; the wait only binds when a read is cold or a window
/// switch near key-up is still being read.
enum ScreenContextWait {
    /// Measured 2026-10-29: in 61 real dictations (e2e.jsonl, Aug 22 to Oct
    /// 29) 59 waited 0 ms and 2 waited over 1 s (one 16.3 s, a cold first
    /// OCR). Warm Vision OCR (accurate, no language correction) of one
    /// window took 71 to 207 ms median by size, a Chrome-sized window 109 to
    /// 113 ms (2 runs). 300 ms lets one in-flight window read finish and caps
    /// the outliers.
    static let afterFinalTranscript: TimeInterval = 0.3

    /// Whether to send the correction now: never while a read is pending,
    /// unless the bound has passed.
    static func shouldSubmit(pendingReads: Int, initialDone: Bool, waited: TimeInterval) -> Bool {
        (initialDone && pendingReads == 0) || waited >= afterFinalTranscript
    }
}
