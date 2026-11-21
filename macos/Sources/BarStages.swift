import Foundation

/// The capsule's feedback for each stage of a dictation: ignite, listening, steady warm processing, then a crisp fade
/// back to rest. The bar is a full-width horizontal gradient, black at both ends with one warm ember light at the
/// centre (`width` stays at `wideWidth = 0.5` in every stage; there is no sliding-together animation). Everything here
/// is pure (stage, elapsed seconds, voice level in, look out), so the maths is unit-tested without drawing. The iPhone
/// keyboard has its own copy (ios/Shared/BarStages.swift); keep the two identical.
enum BarStage: Equatable {
    /// Nothing is happening: the resting look draws, not this light.
    case rest
    /// Key down until key up: ignites at once, then follows the voice.
    case listening
    /// Key up until the text is delivered: steady warm ember across the full bar, no sliding core.
    case processing
    /// Dictation ended (text pasted or not): fades straight out at full width, no pulse.
    case fadeOut
}

struct BarLook: Equatable {
    /// Half-width of the light as a fraction of the bar (always 0.5: full width).
    var width: Double
    /// Colour blend towards a secondary tint (kept at 0: single warm ember colour).
    var heat: Double
    /// Alpha of the warm light at its centre.
    var opacity: Double

    static let none = BarLook(width: BarStages.wideWidth, heat: 0, opacity: 0)

    /// The centre colour as (red, green, blue, alpha), each 0...1.
    var rgba: (r: Double, g: Double, b: Double, a: Double) {
        let a = BarStages.baseColor
        return (a.r, a.g, a.b, opacity)
    }

    /// Gradient stops (location, red, green, blue, alpha): black at both ends, the light at 0.5.
    var stops: [(location: Double, r: Double, g: Double, b: Double, a: Double)] {
        let c = rgba
        let lo = max(0, 0.5 - width), hi = min(1, 0.5 + width)
        return [(0, 0, 0, 0, 1), (lo, 0, 0, 0, 1), (0.5, c.r, c.g, c.b, c.a), (hi, 0, 0, 0, 1), (1, 0, 0, 0, 1)]
    }
}

enum BarStages {
    static let igniteSeconds = 0.10
    static let fadeOutSeconds = 0.10
    static let wideWidth = 0.5
    /// Width never moves in either normal or Reduce Motion mode.
    static let reducedWidth = wideWidth
    static let listeningFloor = 0.08
    /// Listening's opacity: `dimOpacity` plus `energyOpacity` times the energy (the voice, or the ignite).
    static let dimOpacity = 0.5
    static let energyOpacity = 0.38
    /// The ignite's opacity at the instant of the press: full energy (0.88).
    static let ignitedOpacity = dimOpacity + energyOpacity
    /// Steady warm ember opacity while processing (key up until text arrives).
    static let processingOpacity = 0.65

    /// The capsule's warm ember colour.
    static let baseColor = (r: 0.42, g: 0.12, b: 0.035)

    /// How long a stage draws before it settles (the clock can stop after this); nil means it never changes by itself.
    static func duration(of stage: BarStage, reduceMotion: Bool) -> Double? {
        switch stage {
        case .rest: return nil
        case .listening: return igniteSeconds
        case .processing: return 0
        case .fadeOut: return fadeOutSeconds
        }
    }

    /// The look `elapsed` seconds into `stage`. `level` is the smoothed voice level (0...1). `from` is the look at the
    /// moment the fade-out began, so it fades continuously from wherever the light was.
    static func look(stage: BarStage, elapsed: Double, level: Double, reduceMotion: Bool = false,
                     from: BarLook = BarLook(width: wideWidth, heat: 0, opacity: processingOpacity)) -> BarLook {
        let t = max(0, elapsed)
        switch stage {
        case .rest:
            return .none
        case .listening:
            // Ignite: full energy the instant of the press, decaying linearly to the live level.
            let live = min(1, max(listeningFloor, level))
            let ignite = max(0, 1 - t / igniteSeconds)
            let energy = live + (1 - live) * ignite
            return BarLook(width: wideWidth, heat: 0, opacity: dimOpacity + energy * energyOpacity)
        case .processing:
            return BarLook(width: wideWidth, heat: 0, opacity: processingOpacity)
        case .fadeOut:
            let p = min(1, t / fadeOutSeconds)
            // Ease-out extinguish: 75% of the glow is gone in the first half of the fade so fast 15-30 ms
            // dictations feel crisp and never linger behind pasted text.
            let remain = (1 - p) * (1 - p)
            return BarLook(width: from.width, heat: from.heat, opacity: from.opacity * remain)
        }
    }
}

/// The stage a dictation is in, when it began, and the look it began from. A value the owner of the bar replaces at
/// each real event (key down, key up, dictation ended).
struct BarBeat: Equatable {
    var stage: BarStage = .rest
    var since: Date = .distantPast
    var from: BarLook = BarLook(width: BarStages.wideWidth, heat: 0, opacity: BarStages.processingOpacity)

    static let rest = BarBeat()

    func look(at date: Date, level: Double, reduceMotion: Bool) -> BarLook {
        BarStages.look(stage: stage, elapsed: date.timeIntervalSince(since), level: level, reduceMotion: reduceMotion,
                       from: from)
    }

    /// Key down or tap: lit at once.
    static func ignite(at now: Date) -> BarBeat { BarBeat(stage: .listening, since: now) }

    /// Key up or stop: steady warm processing glow. A second call while already processing keeps it as-is.
    func processing(at now: Date) -> BarBeat {
        stage == .processing ? self : BarBeat(stage: .processing, since: now)
    }

    /// Dictation ended (text inserted, empty, error, cancelled): fades out from the current look.
    func fadeOut(at now: Date, level: Double, reduceMotion: Bool) -> BarBeat {
        guard stage != .rest else { return self }
        return BarBeat(stage: .fadeOut, since: now, from: look(at: now, level: level, reduceMotion: reduceMotion))
    }

    /// Seconds from `since` until this stage stops changing, or nil if it never does by itself.
    func settleSeconds(reduceMotion: Bool) -> Double? { BarStages.duration(of: stage, reduceMotion: reduceMotion) }
}
