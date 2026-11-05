import AppKit
import SwiftUI

// MARK: - Live text (settings.json `live_text`, `live_text_style`)

/// settings.json `live_text_style`, still read and written so older settings files round-trip.
/// Since 1.0.1 the card has one look (owner, 2026-11-05: one fixed config) and ignores it.
enum LiveTextStyle: String, Codable, CaseIterable, Identifiable {
    /// New corrected words fade up from 0.35 opacity; the uncorrected tail is dimmed.
    case fade
    /// The uncorrected tail is blurred; words sharpen once corrected.
    case blur

    var id: String { rawValue }

    var title: String {
        switch self {
        case .fade: return "Fade in"
        case .blur: return "Blur to sharp"
        }
    }
}

/// The card's size and type: 520 pt wide, 17 pt text, line height 1.45 (the round-3 mock
/// was 420 pt and 13.5 pt, which read too small; the width grew with the type so a line
/// holds about as many words).
enum LiveTextMetrics {
    static let cardWidth: CGFloat = 520
    static let fontSize: CGFloat = 17
    static let lineHeight: CGFloat = fontSize * 1.45
    static let paddingH: CGFloat = 11
    static let paddingV: CGFloat = 7
    static let border: CGFloat = 1
    static let cornerRadius: CGFloat = 10
    /// The card grows to this many lines, then scrolls to keep the newest line in view.
    static let maxLines = 5
    static let textWidth = cardWidth - 2 * (paddingH + border)
    /// The ember dot after the last word: its diameter, and the room it takes on the line.
    static let dotSize: CGFloat = 6
    static let dotWidth: CGFloat = 8
    static let dotGap: CGFloat = 3

    static func cardHeight(lines: CGFloat) -> CGFloat {
        max(lines, 1) * lineHeight + 2 * (paddingV + border)
    }

    static let maxCardHeight = cardHeight(lines: CGFloat(maxLines))

    private static let font = NSFont.systemFont(ofSize: fontSize)
    static let spaceWidth = measure(" ")

    static func measure(_ text: String) -> CGFloat {
        guard !text.isEmpty else { return 0 }
        return ceil((text as NSString).size(withAttributes: [.font: font]).width * 2) / 2
    }
}

/// Durations and curves. One look (owner, 2026-11-05): every word fades and rises in the moment
/// it first shows, heard or corrected; rewrites morph and glow; the ember dot after the last
/// word follows the mic level; at key up the card shows the pasted text and fades out.
enum LiveTextMotion {
    /// A new word: opacity from `appearFrom` to full and up from `rise` points below.
    static let appear: TimeInterval = 0.12
    static let appearFrom: Double = 0.2
    static let rise: CGFloat = 3
    /// The dot slides to the new end of the text.
    static let dotMove: TimeInterval = 0.12
    /// Key up: the pasted text holds this long (long enough to see its end), then the card
    /// fades out over `leave`.
    static let hold: TimeInterval = 0.25
    static let leave: TimeInterval = 0.15
    static let morph: TimeInterval = 0.20
    /// The morph slot goes back to flowing words a little after its transition ends.
    static let morphHold: TimeInterval = 0.23
    /// The old words' fade ends at this fraction of the morph (120 ms of 200).
    static let morphFadeOut: Double = 0.6
    static let glow: TimeInterval = 0.42
    static let height: TimeInterval = 0.14
    /// Anything started now has finished after this long.
    static let longest: TimeInterval = 0.45
    /// The card changes only this long after the mic last heard a voice: a draft lands
    /// about 0.4-0.8 s after the words it covers, and nothing moves while the user is
    /// silent (background noise can change a partial or a refine pass, never the card).
    static let voiceWindow: TimeInterval = 1.5
    /// Quiet this long, the last word's ".", "?" or "!" shows: a pause ends a sentence,
    /// mid-speech the refiner's mark after every chunk would flicker (owner, 2026-11-05: the
    /// card never showed the question mark the paste had).
    static let endPause: TimeInterval = 0.35
    /// Mid-dictation, a change that is only case or punctuation applies to the last this
    /// many shown words (the join with the new words); further back it waits for key up.
    static let cosmeticReach = 2
    /// A real rewrite of words before the last this many shows only once two drafts in a
    /// row agree on it. The refiner flipped sentence starts back and forth draft to draft
    /// ("Now for" / "For", "But anyways, I think" / "I think", up to 86 words back, owner
    /// trace 2026-11-04 3:51 pm); a genuine correction repeats, so it shows one draft later.
    static let confirmReach = 4
    /// Words further back on the card than this never change while the key is held; key up
    /// fixes them. Two drafts agreeing still let the refiner restyle far back ("And if
    /// there's" -> "If there's" 48 words back, "like" dropped 17 back, owner traces 2026-11-04
    /// 4:16 and 7:00 pm), and a self-correction lands within a few words of where it is said.
    static let freezeReach = 12

    static let easeOut = CubicBezier(0, 0, 0.58, 1)
    static let ease = CubicBezier(0.25, 0.1, 0.25, 1)
    static let card = CubicBezier(0.2, 0.8, 0.3, 1)

    static func progress(_ t: TimeInterval, from start: TimeInterval, over duration: TimeInterval)
        -> Double
    {
        min(max((t - start) / duration, 0), 1)
    }
}

/// CSS `cubic-bezier(x1, y1, x2, y2)`: eased progress for a linear one.
struct CubicBezier: Equatable {
    let x1: Double
    let y1: Double
    let x2: Double
    let y2: Double

    init(_ x1: Double, _ y1: Double, _ x2: Double, _ y2: Double) {
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    }

    func callAsFunction(_ x: Double) -> Double {
        if x <= 0 { return 0 }
        if x >= 1 { return 1 }
        var low = 0.0
        var high = 1.0
        var u = x
        for _ in 0..<30 {
            u = (low + high) / 2
            if Self.sample(x1, x2, u) < x { low = u } else { high = u }
        }
        return Self.sample(y1, y2, u)
    }

    private static func sample(_ a: Double, _ b: Double, _ u: Double) -> Double {
        let v = 1 - u
        return 3 * a * u * v * v + 3 * b * u * u * v + u * u * u
    }
}

struct LiveRGB: Equatable {
    var r: Double
    var g: Double
    var b: Double

    init(_ hex: UInt32) {
        r = Double((hex >> 16) & 0xff) / 255
        g = Double((hex >> 8) & 0xff) / 255
        b = Double(hex & 0xff) / 255
    }

    init(r: Double, g: Double, b: Double) {
        self.r = r
        self.g = g
        self.b = b
    }

    func mixed(_ other: LiveRGB, _ amount: Double) -> LiveRGB {
        LiveRGB(
            r: r + (other.r - r) * amount, g: g + (other.g - g) * amount,
            b: b + (other.b - b) * amount)
    }

    var color: Color { Color(red: r, green: g, blue: b) }

    static let ink = LiveRGB(0xf3e7dc)
    static let white = LiveRGB(0xffffff)
    static let glow = LiveRGB(0xffd2b0)
    static let muted = LiveRGB(0x8f8377)
    /// The dot at rest (the capsule's brown) and lit by a loud voice.
    static let dotRest = LiveRGB(0x6b2a10)
    static let dotLit = LiveRGB(0xffa060)
}

/// One corrected word and the effect it is in, by start time.
struct LiveWord: Equatable {
    let id: Int
    var text: String
    /// Appended: fades or sharpens in from here.
    var fresh: TimeInterval?
    /// Rewritten: glows white and settles from here.
    var glow: TimeInterval?
}

/// Rewritten words: the old ones dissolve while the new ones rise in, in one slot whose
/// width eases from the old text's to the new text's.
struct LiveMorph: Equatable {
    let id: Int
    var old: String
    var new: [LiveWord]
    var start: TimeInterval
}

enum LiveItem: Equatable {
    case word(LiveWord)
    case morph(LiveMorph)
}

/// How one run of text is drawn at one instant.
struct LiveLook: Equatable {
    var color: LiveRGB = .ink
    var opacity: Double = 1
    var blur: CGFloat = 0
    var dy: CGFloat = 0
    /// The rewrite glow, 1 at its start, 0 once settled.
    var glow: Double = 0
}

struct LivePlacedToken: Identifiable, Equatable {
    enum Kind: Equatable {
        case text(String, LiveLook)
        case morph(old: String, oldLook: LiveLook, new: String, newLook: LiveLook)
        /// The ember dot; `lit` is the voice level, 0 to 1.
        case dot(lit: Double)
    }

    let id: Int
    var kind: Kind
    var x: CGFloat
    var line: Int
    var width: CGFloat
}

/// The card at one instant: what to draw where, its height and how far it is scrolled.
struct LiveTextFrame: Equatable {
    var tokens: [LivePlacedToken]
    var lines: Int
    var cardHeight: CGFloat
    /// Lines scrolled off the top.
    var scroll: CGFloat
    /// The whole card's opacity: 1, fading to 0 after key up.
    var opacity: Double = 1
}

/// A value easing from `from` to `to` along the card curve.
struct LiveEased: Equatable {
    var from: CGFloat
    var to: CGFloat
    var start: TimeInterval
    var duration: TimeInterval = LiveTextMotion.height

    func value(at t: TimeInterval) -> CGFloat {
        let p = LiveTextMotion.card(LiveTextMotion.progress(t, from: start, over: duration))
        return from + (to - from) * CGFloat(p)
    }

    /// Heads for `target` from wherever it is at `now`; unchanged when already headed there.
    func retargeted(_ target: CGFloat, at now: TimeInterval) -> LiveEased {
        target == to ? self : LiveEased(from: value(at: now), to: target, start: now, duration: duration)
    }
}

/// Speech vs room noise on the bar's level meter (0 = -55 dBFS, 1 = 0 dBFS): a level is
/// voice when it clears an absolute floor and stands well above this dictation's noise
/// floor, which follows the quietest levels down at once and drifts up slowly.
struct VoiceDetector {
    /// About -41 dBFS.
    static let minimum = 0.25
    /// About 10 dB above the room.
    static let margin = 0.18
    /// Starts at a quiet room (-47 dBFS), so speech from the first instant counts. The
    /// gaps between words pull it back down, so steady speech never becomes the floor.
    private(set) var floor = 0.15

    mutating func isVoice(_ level: Double) -> Bool {
        floor = level < floor ? level : floor + (level - floor) * 0.002
        return level >= max(Self.minimum, floor + Self.margin)
    }
}

enum LiveTextPolicy {
    static func words(_ text: String) -> [String] {
        text.split(whereSeparator: { $0.isWhitespace }).map(String.init)
    }

    /// The words of an open dictation, without the period or question mark the recognizer
    /// and the refiner put after every chunk's last word: it showed and vanished about
    /// every 0.4 s ("cropped." then "cropped in", "set?" then "settings?", owner traces
    /// 2026-11-04). It shows once the next sentence starts, or with the final text.
    /// A trailing "..." (a hesitation the refiner drops once the next word comes) goes too.
    /// `keepEnd` (the speaker paused: a real sentence end) keeps ".", "?" and "!".
    static func openWords(_ text: String, keepEnd: Bool = false) -> [String] {
        var out = words(text)
        guard var last = out.last else { return out }
        if last.hasSuffix("...") {
            last = String(last.dropLast(3))
        } else if !keepEnd, let mark = last.last, mark == "." || mark == "?" || mark == "!",
            !last.hasSuffix("..")
        {
            last = String(last.dropLast())
        }
        if last.isEmpty {
            out.removeLast()
        } else {
            out[out.count - 1] = last
        }
        return out
    }

    /// The speech-recognition words the draft has not covered yet: the latest partial's
    /// words after the ones the draft's committed transcript held.
    static func tail(transcript: [String], covered: Int) -> [String] {
        transcript.count > covered ? Array(transcript.dropFirst(max(covered, 0))) : []
    }

    /// The same, found by the words rather than their count: the partial's words after the
    /// last one it shares with the draft's `source`. A partial that revises earlier words
    /// into more of them ("Um we can", "like k page") shifted a count, and the card showed
    /// the end twice ("fused into one everything fused into one", owner trace 2026-11-04
    /// 4:16 pm). Falls back to the count when the stretch to compare is very long.
    static func tail(transcript: [String], source: [String]) -> [String] {
        let token = { (word: String) -> String in
            let norm = normalized(word)
            return norm.isEmpty ? "\u{FFFC}" : norm
        }
        let t = transcript.map(token)
        let s = source.map(token)
        var p = 0
        while p < min(s.count, t.count), s[p] == t[p] { p += 1 }
        if p == s.count { return Array(transcript[p...]) }
        guard (s.count - p) * (t.count - p) <= 250_000 else {
            return tail(transcript: transcript, covered: source.count)
        }
        // the last source word the partial still has, and the source words after it, which
        // the partial says differently in the same number of words
        var end = p
        for case .kept(let i, let j) in steps(old: Array(s[p...]), new: Array(t[p...])) {
            end = p + j + 1 + (s.count - (p + i) - 1)
        }
        if end == p { end = s.count }
        return end < transcript.count ? Array(transcript[end...]) : []
    }

    /// A rewrite as word-diff steps, in order: a kept word (its index on each side), or a
    /// changed run pairing the old words it replaced with the new ones (either may be
    /// empty). A pure append is one trailing changed run with no old words.
    enum Step: Equatable {
        case kept(old: Int, new: Int)
        case changed(old: Range<Int>, new: Range<Int>)
    }

    /// A word for comparing what was said: lowercase letters and digits only, so
    /// "Thursday," and "thursday" are the same word.
    static func normalized(_ word: String) -> String {
        String(word.lowercased().unicodeScalars.filter(CharacterSet.alphanumerics.contains))
    }

    static func sameWords(_ a: some Collection<String>, _ b: some Collection<String>) -> Bool {
        a.map(normalized).filter { !$0.isEmpty } == b.map(normalized).filter { !$0.isEmpty }
    }

    /// `new` with every case- or punctuation-only change to `old` words before the last
    /// `LiveTextMotion.cosmeticReach` undone, so mid-dictation only real rewrites and the
    /// words at the join with the newest speech move. The final text fixes the rest.
    /// With `previous` (the last draft), a real rewrite of words before the last
    /// `LiveTextMotion.confirmReach` keeps the old words unless `previous` already had it
    /// (the new words with two words of context each side).
    /// Words before the last `LiveTextMotion.freezeReach` on the card never change.
    /// `changed` holds the rewrites the card showed this dictation (card positions and the
    /// words they replaced). A rewrite that would put back words a spot had before waits for
    /// key up ("Overall, just" / "and overall just like" / "Overall, just", 7:00 pm), and
    /// another far-back rewrite touching a changed spot waits too, so a stretch the refiner
    /// keeps flipping ("Qwen" / "quin" / "Qwen", 4:07 pm) changes at most once. Positions,
    /// not the words: the diff's run boundaries shift draft to draft.
    /// `shown` is how many words the card shows (the corrected words plus the uncorrected
    /// tail): "far back" counts from the end of the screen, not of the corrected words.
    /// Words a draft adds after the corrected words always show: they cover the tail, and
    /// holding them back emptied it ("here, or can you just" showed as "here", 4:13 pm).
    static func stabilized(
        old: [String], new: [String], previous: [String]? = nil, shown: Int? = nil,
        changed: inout [Change]
    ) -> [String] {
        let end = max(shown ?? old.count, old.count)
        let reach = max(old.count - LiveTextMotion.cosmeticReach, 0)
        let confirmBefore = max(end - LiveTextMotion.confirmReach, 0)
        let freezeBefore = max(end - LiveTextMotion.freezeReach, 0)
        var out = [String]()

        func apply(_ o: Range<Int>, _ n: Range<Int>) {
            if !o.isEmpty, sameWords(old[o], new[n]) {
                out += o.upperBound <= reach ? old[o] : new[n]
                return
            }
            // where this rewrite lands on the card, one word either side
            let at = out.count
            let span = max(at - 1, 0)..<(at + max(n.count, o.count, 1) + 1)
            let put = new[n].map(normalized).filter { !$0.isEmpty }
            let touching = changed.filter { $0.span.overlaps(span) }
            let flipsBack = touching.contains { contiguous(put, in: $0.replaced) }
            if o.lowerBound >= old.count {
                // words after the corrected ones cover the tail, unless they put back what a
                // rewrite next to them just took out
                if !(flipsBack && !put.isEmpty) { out += new[n] }
                return
            }
            let confirmed =
                o.lowerBound >= confirmBefore || previous == nil
                || (contains(previous!, around: n, in: new) && touching.isEmpty)
            if o.lowerBound < freezeBefore || flipsBack || !confirmed {
                out += old[o]
                if o.upperBound == old.count, n.count > o.count {
                    out += new[(n.lowerBound + o.count)..<n.upperBound]
                }
                return
            }
            changed.append(
                Change(
                    span: at..<(at + max(n.count, 1)),
                    replaced: old[o].map(normalized).filter { !$0.isEmpty }))
            out += new[n]
        }

        for step in steps(old: old, new: new) {
            switch step {
            case .kept(_, let j):
                out.append(new[j])
            case .changed(let o, let n):
                // re-punctuated old words with new words after them ("here" -> "here, or can
                // you just"): the two parts are judged apart
                let k = o.count
                if k > 0, n.count > k, sameWords(old[o], new[n.lowerBound..<(n.lowerBound + k)]) {
                    apply(o, n.lowerBound..<(n.lowerBound + k))
                    apply(o.upperBound..<o.upperBound, (n.lowerBound + k)..<n.upperBound)
                } else {
                    apply(o, n)
                }
            }
        }
        return out
    }

    /// A rewrite the card showed while the key was held.
    struct Change: Equatable {
        /// Card word positions it put words at.
        var span: Range<Int>
        /// The words it replaced, normalized.
        var replaced: [String]
    }

    static func stabilized(old: [String], new: [String], previous: [String]? = nil) -> [String] {
        var changed = [Change]()
        return stabilized(old: old, new: new, previous: previous, changed: &changed)
    }

    /// Whether `part` is a run of `whole` (an empty part: only of an empty whole, so a
    /// deletion flips back only an insertion).
    static func contiguous(_ part: [String], in whole: [String]) -> Bool {
        guard !part.isEmpty else { return whole.isEmpty }
        guard part.count <= whole.count else { return false }
        return (0...(whole.count - part.count)).contains {
            whole[$0..<($0 + part.count)].elementsEqual(part)
        }
    }

    /// Whether `words` holds `new[run]` with up to two words of `new` context on each side.
    static func contains(_ words: [String], around run: Range<Int>, in new: [String]) -> Bool {
        let lo = max(run.lowerBound - 2, 0)
        let hi = min(run.upperBound + 2, new.count)
        let pattern = Array(new[lo..<hi])
        guard !pattern.isEmpty, pattern.count <= words.count else { return pattern.isEmpty }
        for start in 0...(words.count - pattern.count)
        where words[start..<(start + pattern.count)].elementsEqual(pattern) {
            return true
        }
        return false
    }

    static func steps(old: [String], new: [String]) -> [Step] {
        let nonEmpty = { (token: String) -> String in
            token.isEmpty || token.contains(where: \.isWhitespace) ? "\u{FFFC}" : token
        }
        let diff = WordDiff.compare(
            old.map(nonEmpty).joined(separator: " "),
            new.map(nonEmpty).joined(separator: " "))
        let oldChanged = diff.before.filter { !$0.text.allSatisfy(\.isWhitespace) }.map(\.changed)
        let newChanged = diff.after.filter { !$0.text.allSatisfy(\.isWhitespace) }.map(\.changed)
        var steps = [Step]()
        var i = 0
        var j = 0
        while i < oldChanged.count || j < newChanged.count {
            let oldStart = i
            let newStart = j
            while i < oldChanged.count, oldChanged[i] { i += 1 }
            while j < newChanged.count, newChanged[j] { j += 1 }
            if i > oldStart || j > newStart {
                steps.append(.changed(old: oldStart..<i, new: newStart..<j))
            }
            if i < oldChanged.count, j < newChanged.count {
                steps.append(.kept(old: i, new: j))
                i += 1
                j += 1
            } else if i < oldChanged.count || j < newChanged.count {
                // Unpaired kept words cannot happen with an LCS; treat the rest as changed.
                steps.append(.changed(old: i..<oldChanged.count, new: j..<newChanged.count))
                break
            }
        }
        return steps
    }
}

/// The live-text card's content: corrected words with the effects they are in, the
/// uncorrected tail, and the card's height and scroll. All motion is a function of time,
/// so a frame can be rendered at any instant and nothing ever waits on an animation.
@MainActor
final class LiveTextState: ObservableObject {
    @Published private(set) var items: [LiveItem] = []
    @Published private(set) var tail: [String] = []
    /// The draft stopped at the refiner's output budget, so its end is missing.
    @Published private(set) var complete = true
    /// The key is held: the dot shows. Off from key up.
    @Published var listening = true
    /// Something is moving; the view's timeline runs at display rate only then.
    @Published private(set) var animating = false
    /// The mic heard a voice this dictation: the card shows, with only the dot until words come.
    @Published private(set) var awake = false
    /// Key up: when the card started fading out.
    @Published private(set) var leaving: TimeInterval?
    /// The dot's glow, 0 to 1, from the mic level (fast attack, slower release).
    private(set) var lit: Double = 0

    private(set) var height = LiveEased(from: 0, to: 0, start: 0)
    private(set) var scroll = LiveEased(from: 0, to: 0, start: 0)
    private var dotX = LiveEased(from: 0, to: 0, start: 0, duration: LiveTextMotion.dotMove)
    private var dotLine = LiveEased(from: 0, to: 0, start: 0, duration: LiveTextMotion.dotMove)
    private var dotPlaced = false
    private(set) var settledLines = 0
    /// When each tail word first showed, and the card position of the tail's first word then.
    private var tailAppeared: [TimeInterval] = []
    private var tailBase = 0
    private var transcript: [String] = []
    private var covered = 0
    /// The transcript the shown draft was refined from, when the engine sent it.
    private var coveredSource: [String]?
    private var nextID = 1
    private var settleWork: DispatchWorkItem?
    /// When the mic last heard a voice; nil until the first `noteVoice` (then every
    /// update is gated on it).
    private var lastVoice: TimeInterval?
    /// The last draft's words, which a rewrite far back must match to show.
    private var lastDraft: [String]?
    /// The latest draft and partial as sent, end marks included (for `showEndMark`).
    private var lastDraftText: String?
    private var lastTranscriptText: String?
    /// Rewrites the card showed this dictation.
    private var changedSpans: [LiveTextPolicy.Change] = []
    private var widths: [String: CGFloat] = [:]
    let clock: () -> TimeInterval

    init(clock: @escaping () -> TimeInterval = { Date().timeIntervalSinceReferenceDate }) {
        self.clock = clock
    }

    var isEmpty: Bool { items.isEmpty && tail.isEmpty }

    /// Whether the card is on screen: words to show, or a voice heard while the key is held.
    var showsCard: Bool { !isEmpty || (awake && listening) }

    /// The corrected words as they read once every effect settles.
    var correctedWords: [String] {
        items.flatMap { item -> [String] in
            switch item {
            case .word(let word): return [word.text]
            case .morph(let morph): return morph.new.map(\.text)
            }
        }
    }

    var correctedText: String { correctedWords.joined(separator: " ") }

    func reset() {
        items = []
        tail = []
        complete = true
        listening = true
        transcript = []
        covered = 0
        coveredSource = nil
        settledLines = 0
        height = LiveEased(from: 0, to: 0, start: 0)
        scroll = LiveEased(from: 0, to: 0, start: 0)
        dotX = LiveEased(from: 0, to: 0, start: 0, duration: LiveTextMotion.dotMove)
        dotLine = dotX
        dotPlaced = false
        tailAppeared = []
        tailBase = 0
        settleWork?.cancel()
        animating = false
        awake = false
        leaving = nil
        lit = 0
        lastVoice = nil
        lastDraft = nil
        lastDraftText = nil
        lastTranscriptText = nil
        changedSpans = []
    }

    /// The mic heard a voice just now (the bar calls this from the level meter). The first
    /// one wakes the card, so it answers the voice before any word is recognized. Returns
    /// whether it woke it.
    @discardableResult
    func noteVoice() -> Bool {
        let now = clock()
        lastVoice = now
        guard !awake, listening else { return false }
        awake = true
        relayout(at: now)
        return true
    }

    /// The mic level (0 = -55 dBFS, 1 = 0 dBFS), every audio buffer: the dot's glow. Speech
    /// sits around 0.3-0.7 on the meter; the glow spans about -44 to -19 dBFS.
    func noteLevel(_ level: Double) {
        let target = min(max((level - 0.2) / 0.45, 0), 1)
        lit = target > lit ? lit + (target - lit) * 0.7 : lit + (target - lit) * 0.25
    }

    /// Whether the card may change now: the user spoke within `voiceWindow`.
    private func voiced(at now: TimeInterval) -> Bool {
        guard let lastVoice else { return true }
        return now - lastVoice <= LiveTextMotion.voiceWindow
    }

    /// The speaker has been quiet for `LiveTextMotion.endPause`: the sentence's end mark shows.
    private func paused(at now: TimeInterval) -> Bool {
        guard let lastVoice else { return false }
        return now - lastVoice >= LiveTextMotion.endPause
    }

    /// The latest speech-recognition partial (the whole transcript so far). Ignored while
    /// the user is silent, so noise never adds or swaps words.
    @discardableResult
    func setTranscript(_ text: String) -> Bool {
        let now = clock()
        guard voiced(at: now) else { return false }
        transcript = LiveTextPolicy.openWords(text, keepEnd: paused(at: now))
        lastTranscriptText = text
        updateTail(at: now)
        return true
    }

    /// A refiner draft covering the first `coveredWords` words of the transcript (`source`,
    /// when known, is those words: the tail then follows them by content). Ignored
    /// while the user is silent (a later draft or the final text covers it). Case- or
    /// punctuation-only changes to words before the last `cosmeticReach` keep the shown
    /// words; real rewrites (a self-correction) and appends show.
    @discardableResult
    func applyDraft(
        _ text: String, coveredWords: Int, complete: Bool, source: [String]? = nil
    ) -> Bool {
        let now = clock()
        guard voiced(at: now) else { return false }
        let draft = LiveTextPolicy.openWords(text, keepEnd: paused(at: now))
        let words = LiveTextPolicy.stabilized(
            old: correctedWords, new: draft, previous: lastDraft,
            shown: correctedWords.count + tail.count, changed: &changedSpans)
        lastDraft = draft
        lastDraftText = text
        covered = coveredWords
        coveredSource = source
        self.complete = complete
        setCorrected(words, at: now)
        updateTail(at: now)
        return true
    }

    /// Whether a draft refined from `source` agrees with the transcript shown now. A draft
    /// can land after the recognizer revised the words it covers (owner trace 2026-11-04:
    /// a draft for "Yeah." landed after "I imagine you change it" and showed "Yeah imagine
    /// you change it"); a newer draft follows, so a stale one is skipped.
    func draftMatches(source: [String]) -> Bool {
        let n = min(source.count, transcript.count)
        return zip(source.prefix(n), transcript.prefix(n)).allSatisfy {
            LiveTextPolicy.normalized($0) == LiveTextPolicy.normalized($1)
        }
    }

    /// Every word the card shows: the corrected words, then the uncorrected tail.
    var shownWords: [String] { correctedWords + tail }

    /// The pasted text, once it landed: the card shows exactly it at once (no morph: the field
    /// already has it) and fades out over `LiveTextMotion.leave`.
    func applyFinal(_ text: String) {
        let now = clock()
        complete = true
        listening = false
        transcript = []
        covered = 0
        coveredSource = nil
        let words = LiveTextPolicy.words(text)
        if words != correctedWords || !tail.isEmpty {
            items = words.map { .word(makeWord($0)) }
        }
        leaving = now + LiveTextMotion.hold
        updateTail(at: now)
    }

    /// Key up ends the sentence: the last word's ".", "?" or "!" shows at once, from the
    /// latest draft and partial, before the paste lands (the owner lets go about 50 ms after
    /// the last word, so the `endPause` never comes).
    func showEndMark() {
        let now = clock()
        if let text = lastTranscriptText {
            transcript = LiveTextPolicy.openWords(text, keepEnd: true)
        }
        if let text = lastDraftText {
            let words = LiveTextPolicy.stabilized(
                old: correctedWords, new: LiveTextPolicy.openWords(text, keepEnd: true),
                previous: lastDraft, shown: correctedWords.count + tail.count,
                changed: &changedSpans)
            setCorrected(words, at: now)
        }
        updateTail(at: now)
    }

    private func updateTail(at now: TimeInterval) {
        let next =
            coveredSource.map { LiveTextPolicy.tail(transcript: transcript, source: $0) }
            ?? LiveTextPolicy.tail(transcript: transcript, covered: covered)
        // A tail word keeps the time it first showed at its card position; a new one starts now.
        let base = correctedWords.count
        let seen = tail.map(LiveTextPolicy.normalized)
        let times = next.enumerated().map { i, word -> TimeInterval in
            let old = base + i - tailBase
            return old >= 0 && old < seen.count && seen[old] == LiveTextPolicy.normalized(word)
                && old < tailAppeared.count ? tailAppeared[old] : now
        }
        if next != tail { tail = next }
        tailAppeared = times
        tailBase = base
        relayout(at: now)
    }

    private func makeWord(_ text: String, fresh: TimeInterval? = nil, glow: TimeInterval? = nil)
        -> LiveWord
    {
        defer { nextID += 1 }
        return LiveWord(id: nextID, text: text, fresh: fresh, glow: glow)
    }

    /// Diff the shown corrected words against the new ones: kept words keep their effect,
    /// appended words fade or sharpen in (unless they were already on the card as the
    /// uncorrected tail), case- or punctuation-only changes swap in place, and real
    /// rewrites morph and glow.
    private func setCorrected(_ words: [String], at now: TimeInterval) {
        let shownTail = tail.map(LiveTextPolicy.normalized)
        // A morph still running ends here; its new words carry on glowing.
        let shown: [LiveWord] = items.flatMap { item -> [LiveWord] in
            switch item {
            case .word(let word): return [word]
            case .morph(let morph): return morph.new
            }
        }
        let old = shown.map(\.text)
        guard old != words else { return }
        var next = [LiveItem]()
        for step in LiveTextPolicy.steps(old: old, new: words) {
            switch step {
            case .kept(let i, _):
                next.append(.word(shown[i]))
            case .changed(let oldRange, let newRange):
                let added = words[newRange]
                // "night" -> "night. Then upload": the join re-punctuated plus new words.
                let joinSwap =
                    !oldRange.isEmpty && newRange.upperBound == words.count
                    && added.count > oldRange.count
                    && LiveTextPolicy.sameWords(old[oldRange], added.prefix(oldRange.count))
                if oldRange.isEmpty, newRange.upperBound == words.count {
                    next += appended(added, shownTail: shownTail, at: now)
                } else if joinSwap {
                    next += added.prefix(oldRange.count).map { .word(makeWord($0)) }
                    next += appended(added.dropFirst(oldRange.count), shownTail: shownTail, at: now)
                } else if !oldRange.isEmpty, LiveTextPolicy.sameWords(old[oldRange], added) {
                    next += added.map { .word(makeWord($0)) }
                } else {
                    let morph = LiveMorph(
                        id: nextID, old: old[oldRange].joined(separator: " "),
                        new: added.map { makeWord($0, glow: now) }, start: now)
                    nextID += 1
                    next.append(.morph(morph))
                }
            }
        }
        items = next
        relayout(at: now)
    }

    /// New words at the end: they fade and rise in, except that a word the card already showed
    /// as the tail carries on from when it showed (only its ink firms up).
    private func appended(
        _ added: some Collection<String>, shownTail: [String], at now: TimeInterval
    ) -> [LiveItem] {
        added.enumerated().map { k, word in
            let wasTail = k < shownTail.count && shownTail[k] == LiveTextPolicy.normalized(word)
            let shown = wasTail && k < tailAppeared.count ? tailAppeared[k] : now
            return .word(makeWord(word, fresh: shown))
        }
    }

    private func width(_ text: String) -> CGFloat {
        if let known = widths[text] { return known }
        let measured = LiveTextMetrics.measure(text)
        if widths.count > 4_000 { widths.removeAll() }
        widths[text] = measured
        return measured
    }

    /// The settled layout's line count sets the card's height and scroll targets.
    private func relayout(at now: TimeInterval) {
        let settled = layout(tokens(at: .infinity))
        let lines = settled.lines
        let visible = CGFloat(min(max(lines, 1), LiveTextMetrics.maxLines))
        let targetHeight = LiveTextMetrics.cardHeight(lines: visible)
        let targetScroll = CGFloat(max(lines - LiveTextMetrics.maxLines, 0))
        let dot = settled.placed.last { $0.id == Self.dotID }
        if settledLines == 0 || height.to == 0 {
            // The card appears one line tall at once; only growth eases.
            height = LiveEased(from: targetHeight, to: targetHeight, start: now)
            scroll = LiveEased(from: targetScroll, to: targetScroll, start: now)
        } else {
            height = height.retargeted(targetHeight, at: now)
            scroll = scroll.retargeted(targetScroll, at: now)
        }
        if let dot {
            if dotPlaced {
                dotX = dotX.retargeted(dot.x, at: now)
                dotLine = dotLine.retargeted(CGFloat(dot.line), at: now)
            } else {
                // where it first shows, it is at once
                dotX = LiveEased(from: dot.x, to: dot.x, start: now, duration: LiveTextMotion.dotMove)
                dotLine = LiveEased(
                    from: CGFloat(dot.line), to: CGFloat(dot.line), start: now,
                    duration: LiveTextMotion.dotMove)
                dotPlaced = true
            }
        }
        settledLines = isEmpty ? 0 : lines
        markAnimating()
    }

    private func markAnimating() {
        animating = true
        settleWork?.cancel()
        let work = DispatchWorkItem { [weak self] in self?.animating = false }
        settleWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + LiveTextMotion.longest, execute: work)
    }

    // MARK: Frames

    private struct Token {
        var id: Int
        var kind: LivePlacedToken.Kind
        var width: CGFloat
        /// The space before it, in spaces (eased for a morph that adds or removes words).
        var space: CGFloat = 1
        /// Never wraps on its own (the caret sits right after the last word).
        var glued = false
    }

    static let dotID = -1

    func frame(at t: TimeInterval) -> LiveTextFrame {
        var laid = layout(tokens(at: t))
        // The dot slides to the end of the text rather than jumping with each word.
        if let k = laid.placed.lastIndex(where: { $0.id == Self.dotID }) {
            laid.placed[k].x = dotX.value(at: t)
            laid.placed[k].line = Int(dotLine.value(at: t).rounded())
        }
        let scrolled = scroll.value(at: t)
        let first = Int(scrolled.rounded(.down))
        let visible = laid.placed.filter {
            $0.line >= first && $0.line <= first + LiveTextMetrics.maxLines
        }
        let fade = leaving.map {
            1 - LiveTextMotion.easeOut(LiveTextMotion.progress(t, from: $0, over: LiveTextMotion.leave))
        }
        return LiveTextFrame(
            tokens: visible, lines: laid.lines, cardHeight: height.value(at: t), scroll: scrolled,
            opacity: fade ?? 1)
    }

    /// A word that first showed at `shown`: it fades and rises in.
    private func appearing(_ look: inout LiveLook, shown: TimeInterval, at t: TimeInterval) {
        let e = LiveTextMotion.easeOut(
            LiveTextMotion.progress(t, from: shown, over: LiveTextMotion.appear))
        look.opacity *= LiveTextMotion.appearFrom + (1 - LiveTextMotion.appearFrom) * e
        look.dy = LiveTextMotion.rise * CGFloat(1 - e)
    }

    private func wordLook(_ word: LiveWord, at t: TimeInterval) -> LiveLook {
        var look = LiveLook()
        if let fresh = word.fresh {
            appearing(&look, shown: fresh, at: t)
        }
        if let glow = word.glow {
            let e = LiveTextMotion.easeOut(
                LiveTextMotion.progress(t, from: glow, over: LiveTextMotion.glow))
            look.color = LiveRGB.white.mixed(.ink, e)
            look.glow = 1 - e
        }
        return look
    }

    private func tokens(at t: TimeInterval) -> [Token] {
        var out = [Token]()
        for item in items {
            switch item {
            case .word(let word):
                out.append(
                    Token(
                        id: word.id, kind: .text(word.text, wordLook(word, at: t)),
                        width: width(word.text)))
            case .morph(let morph):
                if t - morph.start < LiveTextMotion.morphHold {
                    let linear = LiveTextMotion.progress(
                        t, from: morph.start, over: LiveTextMotion.morph)
                    let e = LiveTextMotion.ease(linear)
                    let w = CGFloat(LiveTextMotion.card(linear))
                    let newText = morph.new.map(\.text).joined(separator: " ")
                    let oldWidth = width(morph.old)
                    let newWidth = width(newText)
                    var oldLook = LiveLook()
                    // The old words are gone by 60% of the morph, so its middle reads cleanly.
                    oldLook.opacity =
                        1 - LiveTextMotion.ease(min(linear / LiveTextMotion.morphFadeOut, 1))
                    oldLook.blur = 2 * CGFloat(e)
                    oldLook.dy = -3 * CGFloat(e)
                    var newLook = morph.new.first.map { wordLook($0, at: t) } ?? LiveLook()
                    newLook.opacity = e
                    newLook.blur = 2 * CGFloat(1 - e)
                    newLook.dy = 3 * CGFloat(1 - e)
                    var space: CGFloat = 1
                    if morph.old.isEmpty { space = w } else if newText.isEmpty { space = 1 - w }
                    out.append(
                        Token(
                            id: morph.id,
                            kind: .morph(
                                old: morph.old, oldLook: oldLook, new: newText, newLook: newLook),
                            width: oldWidth + (newWidth - oldWidth) * w, space: space))
                } else {
                    for word in morph.new {
                        out.append(
                            Token(
                                id: word.id, kind: .text(word.text, wordLook(word, at: t)),
                                width: width(word.text)))
                    }
                }
            }
        }
        // No "…" for a draft cut at the output budget: the tail and the final text cover
        // it, and a placeholder read as noise.
        for (index, word) in tail.enumerated() {
            // Plain ink, a touch softer: these are the words as heard, not a guess.
            var look = LiveLook()
            look.opacity = 0.85
            if index < tailAppeared.count {
                appearing(&look, shown: tailAppeared[index], at: t)
            }
            out.append(Token(id: -100 - index, kind: .text(word, look), width: width(word)))
        }
        if listening {
            out.append(
                Token(
                    id: Self.dotID, kind: .dot(lit: lit), width: LiveTextMetrics.dotWidth,
                    space: 0, glued: true))
        }
        return out
    }

    /// Greedy line breaking at the card's text width.
    private func layout(_ tokens: [Token]) -> (placed: [LivePlacedToken], lines: Int) {
        var placed = [LivePlacedToken]()
        var x: CGFloat = 0
        var line = 0
        for token in tokens {
            let gap =
                token.glued
                ? LiveTextMetrics.dotGap : LiveTextMetrics.spaceWidth * max(token.space, 0)
            if x > 0, !token.glued, x + gap + token.width > LiveTextMetrics.textWidth {
                line += 1
                x = 0
            }
            let start = x > 0 ? x + gap : 0
            placed.append(
                LivePlacedToken(
                    id: token.id, kind: token.kind, x: start, line: line, width: token.width))
            x = start + token.width
        }
        return (placed, placed.isEmpty ? 0 : line + 1)
    }
}

// MARK: - View

struct LiveTextView: View {
    @ObservedObject var state: LiveTextState
    /// Render one instant instead of the live timeline (offscreen render tests).
    var fixedTime: TimeInterval?

    var body: some View {
        VStack(spacing: 0) {
            if state.showsCard {
                if let fixedTime {
                    LiveTextCard(frame: state.frame(at: fixedTime))
                } else {
                    // Display rate while something moves; 60 Hz while only the dot follows
                    // the voice.
                    TimelineView(
                        .animation(
                            minimumInterval: state.animating ? nil : 1.0 / 60,
                            paused: !state.animating && !state.listening)
                    ) { context in
                        LiveTextCard(
                            frame: state.frame(at: context.date.timeIntervalSinceReferenceDate))
                    }
                }
            }
            Spacer(minLength: 0)
        }
        .padding(.top, LiveTextPanel.topMargin)
        .frame(width: LiveTextPanel.size.width, height: LiveTextPanel.size.height)
    }
}

/// The card: top-anchored, one line tall at first, growing down to five lines.
struct LiveTextCard: View {
    let frame: LiveTextFrame

    var body: some View {
        let lineHeight = LiveTextMetrics.lineHeight
        ZStack(alignment: .topLeading) {
            ForEach(frame.tokens) { token in
                tokenView(token)
                    .frame(width: max(token.width, 0), height: lineHeight, alignment: .leading)
                    .offset(
                        x: token.x,
                        y: (CGFloat(token.line) - frame.scroll) * lineHeight)
            }
        }
        .frame(width: LiveTextMetrics.textWidth, alignment: .topLeading)
        .frame(
            width: LiveTextMetrics.textWidth,
            height: max(frame.cardHeight - 2 * (LiveTextMetrics.paddingV + LiveTextMetrics.border), 0),
            alignment: .topLeading
        )
        .clipped()
        .padding(.horizontal, LiveTextMetrics.paddingH + LiveTextMetrics.border)
        .padding(.vertical, LiveTextMetrics.paddingV + LiveTextMetrics.border)
        .frame(width: LiveTextMetrics.cardWidth, height: frame.cardHeight, alignment: .topLeading)
        .background(
            RoundedRectangle(cornerRadius: LiveTextMetrics.cornerRadius, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [LiveRGB(0x1a120c).color, LiveRGB(0x0d0907).color],
                        startPoint: .top, endPoint: .bottom))
        )
        .overlay(
            RoundedRectangle(cornerRadius: LiveTextMetrics.cornerRadius, style: .continuous)
                .strokeBorder(LiveRGB(0x4a2a16).color, lineWidth: LiveTextMetrics.border)
        )
        .shadow(color: .black.opacity(0.67), radius: 12, y: 10)
        .opacity(frame.opacity)
    }

    @ViewBuilder
    private func tokenView(_ token: LivePlacedToken) -> some View {
        switch token.kind {
        case .text(let text, let look):
            styled(text, look)
        case .morph(let old, let oldLook, let new, let newLook):
            // Clipped to the slot's eased width plus the spaces either side (room for the
            // glow), and room above and below for the rise, so the dissolving old words
            // never draw over their neighbours.
            ZStack(alignment: .leading) {
                styled(old, oldLook)
                styled(new, newLook)
            }
            .frame(width: max(token.width, 0), alignment: .leading)
            .mask(
                Rectangle()
                    .frame(
                        width: max(token.width, 0) + 2 * LiveTextMetrics.spaceWidth,
                        height: LiveTextMetrics.lineHeight + 16)
            )
        case .dot(let lit):
            // brown at rest; louder speech lights it, swells it and spreads its glow
            let color = LiveRGB.dotRest.mixed(.dotLit, lit).color
            Circle()
                .fill(color)
                .frame(width: LiveTextMetrics.dotSize, height: LiveTextMetrics.dotSize)
                .scaleEffect(1 + 0.4 * lit)
                .shadow(color: LiveRGB.dotLit.color.opacity(lit), radius: 1 + 5 * lit)
                .frame(width: LiveTextMetrics.dotWidth, alignment: .center)
                .offset(y: 1)
        }
    }

    private func styled(_ text: String, _ look: LiveLook) -> some View {
        Text(text)
            .font(.system(size: LiveTextMetrics.fontSize))
            .foregroundColor(look.color.color)
            .lineLimit(1)
            .fixedSize()
            .shadow(color: LiveRGB.glow.color.opacity(look.glow), radius: look.glow > 0 ? 5 : 0)
            .shadow(color: .white.opacity(look.glow), radius: look.glow > 0 ? 1 : 0)
            .blur(radius: look.blur)
            .opacity(look.opacity)
            .offset(y: look.dy)
    }
}

/// Like the capsule's panel: borderless, non-activating, never key or main, and it ignores
/// the mouse, so it can never take focus or change where the paste goes. Its transparent
/// area (the room the card grows into, and the shadow's margin) passes clicks through.
final class LiveTextPanel: NSPanel {
    /// Room around the card for its shadow (0 10px 24px in the mock).
    static let sideMargin: CGFloat = 24
    static let topMargin: CGFloat = 6
    static let bottomMargin: CGFloat = 24
    static let size = NSSize(
        width: LiveTextMetrics.cardWidth + 2 * sideMargin,
        height: topMargin + LiveTextMetrics.maxCardHeight + bottomMargin)
    /// Gap between the overlay panel and the capsule's resting frame.
    static let gap: CGFloat = 4
    /// The capsule container's height (AppController.containerSize).
    static let capsuleHeight: CGFloat = 16

    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }

    init() {
        super.init(
            contentRect: NSRect(origin: .zero, size: Self.size),
            styleMask: [.borderless, .nonactivatingPanel, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        isFloatingPanel = true
        level = .floating
        hidesOnDeactivate = false
        collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        isMovableByWindowBackground = false
        backgroundColor = .clear
        isOpaque = false
        hasShadow = false
        ignoresMouseEvents = true
        appearance = NSAppearance(named: .darkAqua)
    }

    /// Centered on the screen's midline, its bottom just above the capsule; the card sits
    /// at the panel's top and grows down into the reserved room.
    static func frame(screen: NSRect) -> NSRect {
        NSRect(
            x: screen.midX - size.width / 2,
            y: screen.minY + PanelGeometry.bottomInset + capsuleHeight + gap,
            width: size.width,
            height: size.height
        )
    }
}

/// One dictation's live-card history, a line in ~/Library/Logs/Phonon/live.jsonl: each
/// partial, draft and final with whether the card took it, the voice gate's changes, and
/// what the card showed after. Texts are stored as word deltas (`at` = first changed word,
/// `words` = the words from there), so a six-minute dictation stays a few hundred KB.
/// It stays on this Mac, like the corpus.
struct LiveTextTrace {
    struct Event: Encodable {
        let ms: Int
        let kind: String
        var applied: Bool?
        var at: Int?
        var words: String?
        var covered: Int?
        var cardAt: Int?
        var card: String?
    }

    struct Line: Encodable {
        let started: String
        let events: [Event]
    }

    private var start: TimeInterval?
    private var events: [Event] = []
    private var voice = false
    private var lastInput: [String: [String]] = [:]
    private var lastCard: [String] = []

    static func logURL() -> URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/Phonon/live.jsonl")
    }

    /// The first index where `new` leaves `old`, and the words from there.
    static func delta(_ old: [String], _ new: [String]) -> (Int, String) {
        var i = 0
        while i < min(old.count, new.count), old[i] == new[i] { i += 1 }
        return (i, new[i...].joined(separator: " "))
    }

    private var now: Int {
        Int(((Date().timeIntervalSinceReferenceDate) - (start ?? 0)) * 1000)
    }

    mutating func begin() {
        start = Date().timeIntervalSinceReferenceDate
        events = []
        voice = false
        lastInput = [:]
        lastCard = []
    }

    mutating func voice(_ on: Bool) {
        guard start != nil, on != voice else { return }
        voice = on
        events.append(Event(ms: now, kind: on ? "voice_on" : "voice_off"))
    }

    /// `kind` is partial, draft, draft_rejected or final; `input` is its text.
    mutating func record(
        _ kind: String, input: String, applied: Bool, covered: Int? = nil, card: [String]
    ) {
        guard start != nil else { return }
        let words = LiveTextPolicy.words(input)
        let family = kind.hasPrefix("draft") ? "draft" : kind
        let (at, changed) = Self.delta(lastInput[family] ?? [], words)
        lastInput[family] = words
        var event = Event(ms: now, kind: kind, applied: applied, at: at, words: changed)
        event.covered = covered
        if card != lastCard {
            let (cardAt, cardWords) = Self.delta(lastCard, card)
            event.cardAt = cardAt
            event.card = cardWords
            lastCard = card
        }
        events.append(event)
    }

    /// Appends the dictation's line (nothing when it had no live-text events).
    mutating func finish() {
        defer { start = nil }
        guard start != nil, events.contains(where: { $0.kind != "voice_on" && $0.kind != "voice_off" })
        else { return }
        let started = Date(timeIntervalSinceReferenceDate: start ?? 0)
        let line = Line(started: ISO8601DateFormatter().string(from: started), events: events)
        guard var data = try? JSONEncoder().encode(line) else { return }
        data.append(0x0A)
        let url = Self.logURL()
        try? FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        if let handle = try? FileHandle(forWritingTo: url) {
            defer { try? handle.close() }
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: data)
        } else {
            try? data.write(to: url, options: .atomic)
        }
    }
}
