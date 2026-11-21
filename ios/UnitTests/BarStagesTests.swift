import XCTest

/// The stage maths of the dictation light (BarStages.swift): ignite, listening, gather and hold, fade-out, and
/// the Reduce Motion path. The same cases run in the Mac app's tests (macos/Tests/BarStagesTests.swift).
final class BarStagesTests: XCTestCase {
    private func look(_ stage: BarStage, _ t: Double, level: Double = 0.4, reduced: Bool = false,
                      from: BarLook = BarLook(width: 0.08, heat: 1, opacity: 1)) -> BarLook {
        BarStages.look(stage: stage, elapsed: t, level: level, reduceMotion: reduced, from: from)
    }

    /// The named constants the other cases use, pinned to the spec's numbers once.
    func testNamedConstantsHoldTheSpecValues() {
        XCTAssertEqual(BarStages.ignitedOpacity, 0.88, accuracy: 1e-9)
        XCTAssertEqual(BarStages.gatherStartOpacity, 0.65, accuracy: 1e-9)
        XCTAssertEqual(BarStages.halfGatherWidth, 0.29, accuracy: 1e-9)
        XCTAssertEqual(BarStages.igniteSeconds, 0.10)
        XCTAssertEqual(BarStages.gatherSeconds, 0.22)
        XCTAssertEqual(BarStages.fadeOutSeconds, 0.10)
        XCTAssertEqual(BarStages.wideWidth, 0.5)
        XCTAssertEqual(BarStages.coreWidth, 0.08)
    }

    func testIgniteIsFullStrengthAtTheInstantOfThePressBeforeAnyLevelArrives() {
        // no audio yet: level 0
        let lit = look(.listening, 0, level: 0)
        XCTAssertEqual(lit.opacity, BarStages.ignitedOpacity, accuracy: 1e-9, "energy 1.0 at t = 0")
        XCTAssertEqual(lit.width, BarStages.wideWidth)
        XCTAssertEqual(lit.heat, 0)
        // ...and with a level that has already arrived
        XCTAssertEqual(look(.listening, 0, level: 0.3).opacity, BarStages.ignitedOpacity, accuracy: 1e-9)
    }

    func testIgniteDecaysLinearlyToTheLiveLevelOverIgniteSeconds() {
        let live = 0.3
        let dim = BarStages.dimOpacity, range = BarStages.energyOpacity
        let settled = look(.listening, BarStages.igniteSeconds, level: live).opacity
        XCTAssertEqual(settled, dim + range * live, accuracy: 1e-9)
        let half = look(.listening, BarStages.igniteSeconds / 2, level: live).opacity
        XCTAssertEqual(half, dim + range * (live + (1 - live) * 0.5), accuracy: 1e-9)
        XCTAssertEqual(look(.listening, 5, level: live).opacity, settled, accuracy: 1e-9, "no ignite left later")
        // quiet: the floor of 0.08
        XCTAssertEqual(look(.listening, 5, level: 0).opacity, dim + range * BarStages.listeningFloor, accuracy: 1e-9)
    }

    func testGatherReachesTheCoreQuicklyWithEaseOutAndHolds() {
        XCTAssertEqual(look(.processing, 0).width, 0.5, accuracy: 1e-9)
        let at40ms = look(.processing, 0.04)
        XCTAssertLessThan(at40ms.width, 0.36, "visibly gathered within 40 ms")
        XCTAssertGreaterThan(at40ms.heat, 0.5, "visibly warmed within 40 ms")
        let atMid = look(.processing, BarStages.gatherSeconds / 2)
        XCTAssertLessThan(atMid.width, BarStages.halfGatherWidth, "ease-out is past halfway at t/2")
        let core = look(.processing, BarStages.gatherSeconds)
        XCTAssertEqual(core.width, 0.08, accuracy: 1e-9)
        XCTAssertEqual(core.heat, 1, accuracy: 1e-9)
        XCTAssertEqual(core.opacity, 1, accuracy: 1e-9)
        for t in [0.35, 1.0, 10.0, 600.0] {
            XCTAssertEqual(look(.processing, t), core, "held, no loop or breathing at \(t) s")
        }
        // it brightens as it gathers: base warm at 0.65 opacity, towards (227, 101, 29)
        XCTAssertEqual(look(.processing, 0).opacity, BarStages.gatherStartOpacity, accuracy: 1e-9)
        XCTAssertLessThan(look(.processing, 0.04).opacity, look(.processing, 0.16).opacity)
        let c = core.rgba
        XCTAssertEqual(c.r * 255, 227, accuracy: 1e-6)
        XCTAssertEqual(c.g * 255, 101, accuracy: 1e-6)
        XCTAssertEqual(c.b * 255, 29, accuracy: 1e-6)
    }

    func testFadeOutStartsContinuouslyAndFinishesGatheringFastDictations() {
        let t0 = Date(timeIntervalSinceReferenceDate: 1000)
        let processing = BarBeat.ignite(at: t0).processing(at: t0)
        let doneAt = t0.addingTimeInterval(0.04)
        let early = processing.fadeOut(at: doneAt, level: 0.4, reduceMotion: false)
        XCTAssertEqual(early.stage, .fadeOut)
        let expectedAtPaste = look(.processing, 0.04)
        XCTAssertEqual(early.from.width, expectedAtPaste.width, accuracy: 1e-9)
        XCTAssertEqual(early.look(at: doneAt, level: 0.4, reduceMotion: false).width,
                       expectedAtPaste.width, accuracy: 1e-9, "continuous: no jump at the moment of the paste")
        let midFade = early.look(at: doneAt.addingTimeInterval(BarStages.fadeOutSeconds / 2), level: 0, reduceMotion: false)
        XCTAssertLessThan(midFade.width, early.from.width, "keeps gathering inward during fade-out")
        XCTAssertLessThan(midFade.opacity, early.from.opacity, "never flashes brighter")
        let endFade = early.look(at: doneAt.addingTimeInterval(BarStages.fadeOutSeconds), level: 0, reduceMotion: false)
        XCTAssertEqual(endFade.width, BarStages.coreWidth, accuracy: 1e-9, "arrives cleanly at tight core")
        XCTAssertEqual(endFade.opacity, 0, accuracy: 1e-9)
    }

    func testTheEndFadesTheCoreWithNoPulse() {
        let held = look(.processing, 3)
        for t in stride(from: 0.0, through: 0.4, by: 0.04) {
            let f = look(.fadeOut, t, from: held)
            XCTAssertEqual(f.width, 0.08, accuracy: 1e-9, "the core does not spread at \(t)")
            XCTAssertEqual(f.heat, held.heat, accuracy: 1e-9, "no colour change from held core at \(t)")
            XCTAssertLessThanOrEqual(f.opacity, held.opacity, "never brighter at \(t)")
        }
        XCTAssertEqual(look(.fadeOut, 0, from: held).opacity, 1, accuracy: 1e-9)
        XCTAssertEqual(look(.fadeOut, BarStages.fadeOutSeconds / 2, from: held).opacity, 0.25, accuracy: 1e-9)
        XCTAssertEqual(look(.fadeOut, BarStages.fadeOutSeconds, from: held).opacity, 0, accuracy: 1e-9)
        XCTAssertEqual(look(.fadeOut, 1, from: held).rgba.a, 0, accuracy: 1e-9)
        let t0 = Date(timeIntervalSinceReferenceDate: 1000)
        let ended = BarBeat.ignite(at: t0).processing(at: t0).fadeOut(at: t0.addingTimeInterval(2), level: 0, reduceMotion: false)
        XCTAssertEqual(ended.stage, .fadeOut)
        XCTAssertEqual(ended.settleSeconds(reduceMotion: false), BarStages.fadeOutSeconds)
    }

    func testReduceMotionNeverMovesTheWidth() {
        let widths = Set(
            [0.0, 0.4, 1.2, 5].map { look(.processing, $0, reduced: true).width }
                + [0.0, 0.1, 0.3, 1].map { look(.listening, $0, reduced: true).width }
                + [0.0, 0.1, 0.3, 1].map {
                    look(.fadeOut, $0, reduced: true, from: look(.processing, 1, reduced: true)).width
                })
        XCTAssertEqual(widths, [BarStages.reducedWidth],
                       "one steady medium width from key down to the end, key up changes nothing: \(widths)")
        XCTAssertEqual(look(.processing, 0, reduced: true), look(.processing, 99, reduced: true), "steady")
        let core = look(.processing, 1, reduced: true)
        XCTAssertGreaterThan(core.width, 0.08)
        XCTAssertLessThan(core.width, 0.5)
        // the key up itself: the listening light and the core it becomes have the same width
        let t0 = Date(timeIntervalSinceReferenceDate: 1000)
        let lit = BarBeat.ignite(at: t0)
        let up = t0.addingTimeInterval(2)
        XCTAssertEqual(lit.look(at: up, level: 0.7, reduceMotion: true).width,
                       lit.processing(at: up).look(at: up, level: 0.7, reduceMotion: true).width)
        // the end: the fade keeps that width too
        let faded = lit.fadeOut(at: up, level: 0.7, reduceMotion: true)
        XCTAssertEqual(faded.look(at: up.addingTimeInterval(0.1), level: 0, reduceMotion: true).width,
                       BarStages.reducedWidth)
    }

    func testRestDrawsNothingAndTheGradientIsBlackAtBothEnds() {
        XCTAssertEqual(look(.rest, 1).rgba.a, 0)
        for l in [look(.listening, 0), look(.processing, 0.7), look(.fadeOut, 0.1)] {
            let s = l.stops
            XCTAssertEqual(s.first?.location, 0)
            XCTAssertEqual(s.last?.location, 1)
            for end in [s.first!, s.last!] { XCTAssertEqual([end.r, end.g, end.b, end.a], [0, 0, 0, 1]) }
            XCTAssertEqual(s.map(\.location), s.map(\.location).sorted(), "stops stay in order")
        }
        let s = look(.processing, 1.2).stops
        XCTAssertEqual(s[1].location, 0.42, accuracy: 1e-9)
        XCTAssertEqual(s[2].location, 0.5)
        XCTAssertEqual(s[3].location, 0.58, accuracy: 1e-9)
    }

    func testBeatReplacesOnEachRealEvent() {
        let t0 = Date(timeIntervalSinceReferenceDate: 1000)
        let lit = BarBeat.ignite(at: t0)
        XCTAssertEqual(lit.look(at: t0, level: 0, reduceMotion: false).opacity, BarStages.ignitedOpacity,
                       accuracy: 1e-9)
        let p = lit.processing(at: t0.addingTimeInterval(3))
        XCTAssertEqual(p.processing(at: t0.addingTimeInterval(4)), p, "a second stop keeps the gather where it is")
        XCTAssertEqual(BarBeat.rest.fadeOut(at: t0, level: 0, reduceMotion: false), .rest, "no fade from nothing")
    }
}
