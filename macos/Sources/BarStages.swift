import Foundation

/// The capsule's feedback for each stage of a dictation ("option B": ignite, gather, burst). The bar is a horizontal
/// gradient, black at both ends, with one warm light at the centre and a half-width `width` (a fraction of the bar):
/// black at 0.5 - width, colour at 0.5, black at 0.5 + width. Everything here is pure (stage, elapsed seconds, voice
/// level in, look out), so the maths is unit-tested without drawing. The iPhone keyboard has its own copy
/// (ios/Shared/BarStages.swift); keep the two identical.
enum BarStage: Equatable {
    /// Nothing is happening: the resting look draws, not this light.
    case rest
    /// Key down until key up: ignites at once, then follows the voice.
    case listening
    /// Key up until the text is delivered: the light gathers into a tight bright core and holds.
    case processing
    /// The text was pasted: the core bursts out to the edges and is gone.
    case burst
    /// Dictation ended with nothing pasted: the core fades out.
    case fadeOut
}

struct BarLook: Equatable {
    /// Half-width of the light as a fraction of the bar (0.5 fills it, 0.08 is the tight core).
    var width: Double
    /// 0 is the base warm colour, 1 the bright orange.
    var heat: Double
    /// Alpha of the warm light at its centre.
    var opacity: Double
    /// Alpha of the light-orange burst colour; 0 outside the burst.
    var burst: Double

    static let none = BarLook(width: BarStages.wideWidth, heat: 0, opacity: 0, burst: 0)

    /// The centre colour as (red, green, blue, alpha), each 0...1.
    var rgba: (r: Double, g: Double, b: Double, a: Double) {
        if burst > 0 {
            let c = BarStages.burstColor
            return (c.r, c.g, c.b, burst)
        }
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
    static let igniteSeconds = 0.18
    static let gatherSeconds = 1.2
    static let burstSeconds = 0.6
    static let fadeOutSeconds = 0.2
    static let reducedFadeSeconds = 0.3
    static let wideWidth = 0.5
    static let coreWidth = 0.08
    /// Half-way through the gather: where a paste at 0.6 s bursts from (0.29).
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

    /// The Mac capsule's warm colour, the bright orange the gathered core reaches, and the burst's light orange.
    static let baseColor = (r: 0.42, g: 0.12, b: 0.035)
    static let coreColor = (r: 227.0 / 255, g: 101.0 / 255, b: 29.0 / 255)
    static let burstColor = (r: 1.0, g: 170.0 / 255, b: 100.0 / 255)

    /// How long a stage draws before it settles (the clock can stop after this); nil means it never changes by itself.
    static func duration(of stage: BarStage, reduceMotion: Bool) -> Double? {
        switch stage {
        case .rest: return nil
        case .listening: return igniteSeconds
        case .processing: return reduceMotion ? 0 : gatherSeconds
        case .burst: return reduceMotion ? reducedFadeSeconds : burstSeconds
        case .fadeOut: return fadeOutSeconds
        }
    }

    /// The look `elapsed` seconds into `stage`. `level` is the smoothed voice level (0...1). `from` is the look at the
    /// moment a burst or fade-out began, so the burst starts from wherever the gather had got to.
    static func look(stage: BarStage, elapsed: Double, level: Double, reduceMotion: Bool = false,
                     from: BarLook = BarLook(width: coreWidth, heat: 1, opacity: 1, burst: 0)) -> BarLook {
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
                           opacity: dimOpacity + energy * energyOpacity, burst: 0)
        case .processing:
            if reduceMotion {
                return BarLook(width: reducedWidth, heat: reducedHeat, opacity: 0.5 + 0.5 * reducedHeat, burst: 0)
            }
            let k = min(1, t / gatherSeconds)
            let heat = gatherStartHeat + (1 - gatherStartHeat) * k
            return BarLook(width: wideWidth - (wideWidth - coreWidth) * k, heat: heat, opacity: 0.5 + 0.5 * heat,
                           burst: 0)
        case .burst:
            if reduceMotion {
                // a plain fade of the steady core
                let p = min(1, t / reducedFadeSeconds)
                return BarLook(width: from.width, heat: from.heat, opacity: from.opacity * (1 - p), burst: 0)
            }
            let p = min(1, t / burstSeconds)
            return BarLook(width: from.width + (wideWidth - from.width) * p.squareRoot(), heat: 1, opacity: 0,
                           burst: 1 - p)
        case .fadeOut:
            let p = min(1, t / fadeOutSeconds)
            return BarLook(width: from.width, heat: from.heat, opacity: from.opacity * (1 - p), burst: 0)
        }
    }
}

/// The stage a dictation is in, when it began, and the look it began from. A value the owner of the bar replaces at
/// each real event (key down, key up, text delivered, nothing delivered).
struct BarBeat: Equatable {
    var stage: BarStage = .rest
    var since: Date = .distantPast
    var from: BarLook = BarLook(width: BarStages.coreWidth, heat: 1, opacity: 1, burst: 0)

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

    /// The text was inserted: burst from the current look.
    func burst(at now: Date, level: Double, reduceMotion: Bool) -> BarBeat {
        guard stage != .rest else { return self }
        return BarBeat(stage: .burst, since: now, from: look(at: now, level: level, reduceMotion: reduceMotion))
    }

    /// Dictation ended with nothing inserted (empty, error, cancelled): no burst.
    func fadeOut(at now: Date, level: Double, reduceMotion: Bool) -> BarBeat {
        guard stage != .rest else { return self }
        return BarBeat(stage: .fadeOut, since: now, from: look(at: now, level: level, reduceMotion: reduceMotion))
    }

    /// Seconds from `since` until this stage stops changing, or nil if it never does by itself.
    func settleSeconds(reduceMotion: Bool) -> Double? { BarStages.duration(of: stage, reduceMotion: reduceMotion) }
}
