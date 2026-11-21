import Foundation

/// The capsule's feedback for each stage of a dictation: ignite, gather, then a fade back to rest. The bar is a
/// horizontal gradient, black at both ends, with one warm light at the centre and a half-width `width` (a fraction of
/// the bar): black at 0.5 - width, colour at 0.5, black at 0.5 + width. Everything here is pure (stage, elapsed
/// seconds, voice level in, look out), so the maths is unit-tested without drawing. The iPhone keyboard has its own
/// copy (ios/Shared/BarStages.swift); keep the two identical.
enum BarStage: Equatable {
    /// Nothing is happening: the resting look draws, not this light.
    case rest
    /// Key down until key up: ignites at once, then follows the voice.
    case listening
    /// Key up until the text is delivered: the light gathers into a tight bright core and holds.
    case processing
    /// Dictation ended (text pasted or not): the core fades out, no pulse.
    case fadeOut
}

struct BarLook: Equatable {
    /// Half-width of the light as a fraction of the bar (0.5 fills it, 0.08 is the tight core).
    var width: Double
    /// 0 is the base warm colour, 1 the bright orange.
    var heat: Double
    /// Alpha of the warm light at its centre.
    var opacity: Double

    static let none = BarLook(width: BarStages.wideWidth, heat: 0, opacity: 0)

    /// The centre colour as (red, green, blue, alpha), each 0...1.
    var rgba: (r: Double, g: Double, b: Double, a: Double) {
        let a = BarStages.baseColor, b = BarStages.coreColor
        return (a.r + (b.r - a.r) * heat, a.g + (b.g - a.g) * heat, a.b + (b.b - a.b) * heat, opacity)
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
    static let gatherSeconds = 0.22
    static let fadeOutSeconds = 0.10
    static let wideWidth = 0.5
    static let coreWidth = 0.08
    /// Half-way through the gather progress (k = 0.5 -> width 0.29).
    static let halfGatherWidth = (wideWidth + coreWidth) / 2
    /// Reduce Motion (and Low Power Mode): the width never moves; every stage is this steady medium core.
    static let reducedWidth = 0.25
    static let reducedHeat = 0.65
    static let listeningFloor = 0.08
    /// Listening's opacity: `dimOpacity` plus `energyOpacity` times the energy (the voice, or the ignite).
    static let dimOpacity = 0.5
    static let energyOpacity = 0.38
    /// The ignite's opacity at the instant of the press: full energy (0.88).
    static let ignitedOpacity = dimOpacity + energyOpacity
    /// The gather starts at this heat, so its opacity starts at 0.65.
    static let gatherStartHeat = 0.3
    static let gatherStartOpacity = 0.5 + 0.5 * gatherStartHeat

    /// The Mac capsule's warm colour and the bright orange the gathered core reaches.
    static let baseColor = (r: 0.42, g: 0.12, b: 0.035)
    static let coreColor = (r: 227.0 / 255, g: 101.0 / 255, b: 29.0 / 255)

    /// Cubic ease-out progress for gathering the light inward quickly on fast local dictations.
    static func gatherProgress(elapsed: Double) -> Double {
        let u = min(1, max(0, elapsed) / gatherSeconds)
        let inv = 1 - u
        return 1 - inv * inv * inv
    }

    /// Quadratic ease-out pull used during fade-out so even a 15 ms dictation finishes gathering cleanly to the core.
    static func fadePull(progress p: Double) -> Double {
        let u = min(1, max(0, p))
        let inv = 1 - u
        return 1 - inv * inv
    }

    /// How long a stage draws before it settles (the clock can stop after this); nil means it never changes by itself.
    static func duration(of stage: BarStage, reduceMotion: Bool) -> Double? {
        switch stage {
        case .rest: return nil
        case .listening: return igniteSeconds
        case .processing: return reduceMotion ? 0 : gatherSeconds
        case .fadeOut: return fadeOutSeconds
        }
    }

    /// The look `elapsed` seconds into `stage`. `level` is the smoothed voice level (0...1). `from` is the look at the
    /// moment the fade-out began, so it fades continuously from wherever the gather had got to.
    static func look(stage: BarStage, elapsed: Double, level: Double, reduceMotion: Bool = false,
                     from: BarLook = BarLook(width: coreWidth, heat: 1, opacity: 1)) -> BarLook {
        let t = max(0, elapsed)
        switch stage {
        case .rest:
            return .none
        case .listening:
            // Ignite: full energy the instant of the press, decaying linearly to the live level.
            let live = min(1, max(listeningFloor, level))
            let ignite = max(0, 1 - t / igniteSeconds)
            let energy = live + (1 - live) * ignite
            // Reduce Motion: the same medium width as the processing core, so key up changes no width
            return BarLook(width: reduceMotion ? reducedWidth : wideWidth, heat: 0,
                           opacity: dimOpacity + energy * energyOpacity)
        case .processing:
            if reduceMotion {
                return BarLook(width: reducedWidth, heat: reducedHeat, opacity: 0.5 + 0.5 * reducedHeat)
            }
            let k = gatherProgress(elapsed: t)
            let heat = gatherStartHeat + (1 - gatherStartHeat) * k
            return BarLook(width: wideWidth - (wideWidth - coreWidth) * k, heat: heat, opacity: 0.5 + 0.5 * heat)
        case .fadeOut:
            let p = min(1, t / fadeOutSeconds)
            // Ease-out extinguish: 75% of the glow is gone in the first half of the fade so fast 15-30 ms
            // dictations feel crisp and never linger behind pasted text.
            let remain = (1 - p) * (1 - p)
            if reduceMotion {
                return BarLook(width: from.width, heat: from.heat, opacity: from.opacity * remain)
            }
            let pull = fadePull(progress: p)
            let width = from.width + (coreWidth - from.width) * pull
            let heat = from.heat + (1 - from.heat) * pull
            return BarLook(width: width, heat: heat, opacity: from.opacity * remain)
        }
    }
}

/// The stage a dictation is in, when it began, and the look it began from. A value the owner of the bar replaces at
/// each real event (key down, key up, dictation ended).
struct BarBeat: Equatable {
    var stage: BarStage = .rest
    var since: Date = .distantPast
    var from: BarLook = BarLook(width: BarStages.coreWidth, heat: 1, opacity: 1)

    static let rest = BarBeat()

    func look(at date: Date, level: Double, reduceMotion: Bool) -> BarLook {
        BarStages.look(stage: stage, elapsed: date.timeIntervalSince(since), level: level, reduceMotion: reduceMotion,
                       from: from)
    }

    /// Key down or tap: lit at once.
    static func ignite(at now: Date) -> BarBeat { BarBeat(stage: .listening, since: now) }

    /// Key up or stop: gather. A second call while already processing keeps the gather where it is.
    func processing(at now: Date) -> BarBeat {
        stage == .processing ? self : BarBeat(stage: .processing, since: now)
    }

    /// Dictation ended (text inserted, empty, error, cancelled): the core fades out from the current look.
    func fadeOut(at now: Date, level: Double, reduceMotion: Bool) -> BarBeat {
        guard stage != .rest else { return self }
        return BarBeat(stage: .fadeOut, since: now, from: look(at: now, level: level, reduceMotion: reduceMotion))
    }

    /// Seconds from `since` until this stage stops changing, or nil if it never does by itself.
    func settleSeconds(reduceMotion: Bool) -> Double? { BarStages.duration(of: stage, reduceMotion: reduceMotion) }
}
