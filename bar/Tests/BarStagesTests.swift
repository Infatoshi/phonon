import AppKit
import SwiftUI
import XCTest

@testable import PhononBar

/// The stage maths of the dictation light (BarStages.swift): ignite, listening, gather and hold, burst, fade-out, and
/// the Reduce Motion path. The same cases run in the iPhone app's tests (ios/UnitTests/BarStagesTests.swift).
/// Also the capsule's events (PillState) and offscreen stills; set PHONON_RENDER_DIR to keep them.
@MainActor
final class BarStagesTests: XCTestCase {
    private func look(_ stage: BarStage, _ t: Double, level: Double = 0.4, reduced: Bool = false,
                      from: BarLook = BarLook(width: 0.08, heat: 1, opacity: 1, burst: 0)) -> BarLook {
        BarStages.look(stage: stage, elapsed: t, level: level, reduceMotion: reduced, from: from)
    }

    func testIgniteIsFullStrengthAtTheInstantOfThePressBeforeAnyLevelArrives() {
        // no audio yet: level 0
        let lit = look(.listening, 0, level: 0)
        XCTAssertEqual(lit.opacity, 0.5 + 0.38, accuracy: 1e-9, "energy 1.0 at t = 0")
        XCTAssertEqual(lit.width, 0.5)
        XCTAssertEqual(lit.heat, 0)
        // ...and with a level that has already arrived
        XCTAssertEqual(look(.listening, 0, level: 0.3).opacity, 0.88, accuracy: 1e-9)
    }

    func testIgniteDecaysLinearlyToTheLiveLevelOverEighteenHundredthsOfASecond() {
        let live = 0.3
        let settled = look(.listening, 0.18, level: live).opacity
        XCTAssertEqual(settled, 0.5 + 0.38 * live, accuracy: 1e-9)
        let half = look(.listening, 0.09, level: live).opacity
        XCTAssertEqual(half, 0.5 + 0.38 * (live + (1 - live) * 0.5), accuracy: 1e-9)
        XCTAssertEqual(look(.listening, 5, level: live).opacity, settled, accuracy: 1e-9, "no ignite left later")
        // quiet: the floor of 0.08
        XCTAssertEqual(look(.listening, 5, level: 0).opacity, 0.5 + 0.38 * 0.08, accuracy: 1e-9)
    }

    func testGatherReachesTheCoreAtOnePointTwoSecondsAndHolds() {
        XCTAssertEqual(look(.processing, 0).width, 0.5, accuracy: 1e-9)
        XCTAssertEqual(look(.processing, 0.6).width, 0.5 - 0.42 * 0.5, accuracy: 1e-9, "linear")
        let core = look(.processing, 1.2)
        XCTAssertEqual(core.width, 0.08, accuracy: 1e-9)
        XCTAssertEqual(core.heat, 1, accuracy: 1e-9)
        XCTAssertEqual(core.opacity, 1, accuracy: 1e-9)
        for t in [1.3, 2.0, 10.0, 600.0] {
            XCTAssertEqual(look(.processing, t), core, "held, no loop or breathing at \(t) s")
        }
        // it brightens as it gathers: base warm at 0.65 opacity, towards (227, 101, 29)
        XCTAssertEqual(look(.processing, 0).opacity, 0.65, accuracy: 1e-9)
        XCTAssertLessThan(look(.processing, 0.3).opacity, look(.processing, 0.9).opacity)
        let c = core.rgba
        XCTAssertEqual(c.r * 255, 227, accuracy: 1e-6)
        XCTAssertEqual(c.g * 255, 101, accuracy: 1e-6)
        XCTAssertEqual(c.b * 255, 29, accuracy: 1e-6)
    }

    func testBurstGrowsWithSquareRootAndEndsAtZero() {
        let held = look(.processing, 3)
        XCTAssertEqual(look(.burst, 0, from: held).width, 0.08, accuracy: 1e-9)
        XCTAssertEqual(look(.burst, 0, from: held).burst, 1, accuracy: 1e-9)
        for p in [0.1, 0.25, 0.5, 0.81] {
            let b = look(.burst, 0.6 * p, from: held)
            XCTAssertEqual(b.width, 0.08 + 0.42 * p.squareRoot(), accuracy: 1e-9, "sqrt at \(p)")
            XCTAssertEqual(b.burst, 1 - p, accuracy: 1e-9, "linear fade at \(p)")
        }
        // sqrt is front-loaded: past the half-way width by a quarter of the time
        XCTAssertGreaterThan(look(.burst, 0.15, from: held).width, 0.29)
        let end = look(.burst, 0.6, from: held)
        XCTAssertEqual(end.width, 0.5, accuracy: 1e-9)
        XCTAssertEqual(end.burst, 0, accuracy: 1e-9)
        XCTAssertEqual(end.rgba.a, 0, accuracy: 1e-9, "nothing is drawn at the end")
        XCTAssertEqual(look(.burst, 2, from: held).rgba.a, 0, accuracy: 1e-9)
        let c = look(.burst, 0.01, from: held).rgba
        XCTAssertEqual(c.r * 255, 255, accuracy: 1e-6)
        XCTAssertEqual(c.g * 255, 170, accuracy: 1e-6)
        XCTAssertEqual(c.b * 255, 100, accuracy: 1e-6)
    }

    func testBurstStartsFromWhereverTheGatherWas() {
        let t0 = Date(timeIntervalSinceReferenceDate: 1000)
        let processing = BarBeat.ignite(at: t0).processing(at: t0)
        let early = processing.burst(at: t0.addingTimeInterval(0.6), level: 0.4, reduceMotion: false)
        XCTAssertEqual(early.stage, .burst)
        XCTAssertEqual(early.from.width, 0.29, accuracy: 1e-9, "finished half way through the gather")
        XCTAssertEqual(early.look(at: t0.addingTimeInterval(0.6), level: 0.4, reduceMotion: false).width, 0.29,
                       accuracy: 1e-9, "continuous: no jump at the moment of the paste")
        XCTAssertEqual(early.look(at: t0.addingTimeInterval(1.2), level: 0.4, reduceMotion: false).width, 0.5,
                       accuracy: 1e-9)
    }

    func testNothingPastedSkipsTheBurstAndFadesTheCoreOverTwoTenths() {
        let held = look(.processing, 3)
        for t in stride(from: 0.0, through: 0.5, by: 0.05) {
            let f = look(.fadeOut, t, from: held)
            XCTAssertEqual(f.burst, 0, "no burst colour at \(t)")
            XCTAssertEqual(f.width, 0.08, accuracy: 1e-9, "the core does not spread at \(t)")
        }
        XCTAssertEqual(look(.fadeOut, 0, from: held).opacity, 1, accuracy: 1e-9)
        XCTAssertEqual(look(.fadeOut, 0.1, from: held).opacity, 0.5, accuracy: 1e-9)
        XCTAssertEqual(look(.fadeOut, 0.2, from: held).opacity, 0, accuracy: 1e-9)
        XCTAssertEqual(look(.fadeOut, 1, from: held).rgba.a, 0, accuracy: 1e-9)
        // through the beat: a result with no text never reaches the burst stage
        let t0 = Date(timeIntervalSinceReferenceDate: 1000)
        let ended = BarBeat.ignite(at: t0).processing(at: t0).fadeOut(at: t0.addingTimeInterval(2), level: 0, reduceMotion: false)
        XCTAssertEqual(ended.stage, .fadeOut)
        XCTAssertEqual(ended.settleSeconds(reduceMotion: false), 0.2)
    }

    func testReduceMotionNeverMovesTheWidth() {
        let widths = Set(
            [0.0, 0.4, 1.2, 5].map { look(.processing, $0, reduced: true).width }
                + [0.0, 0.1, 0.3, 1].map { look(.listening, $0, reduced: true).width }
                + [0.0, 0.1, 0.3, 1].map {
                    look(.burst, $0, reduced: true, from: look(.processing, 1, reduced: true)).width
                })
        XCTAssertEqual(widths.count, 2, "a wide light while listening and one steady medium core after: \(widths)")
        XCTAssertEqual(look(.processing, 0, reduced: true), look(.processing, 99, reduced: true), "steady")
        let core = look(.processing, 1, reduced: true)
        XCTAssertGreaterThan(core.width, 0.08)
        XCTAssertLessThan(core.width, 0.5)
        // pasted: a plain 0.3 s fade, no burst colour
        let fade = { (t: Double) in self.look(.burst, t, reduced: true, from: core) }
        XCTAssertEqual(fade(0).opacity, core.opacity, accuracy: 1e-9)
        XCTAssertEqual(fade(0.15).opacity, core.opacity / 2, accuracy: 1e-9)
        XCTAssertEqual(fade(0.3).opacity, 0, accuracy: 1e-9)
        XCTAssertEqual(fade(0.1).burst, 0)
        XCTAssertEqual(BarStages.duration(of: .burst, reduceMotion: true), 0.3)
    }

    func testRestDrawsNothingAndTheGradientIsBlackAtBothEnds() {
        XCTAssertEqual(look(.rest, 1).rgba.a, 0)
        for l in [look(.listening, 0), look(.processing, 0.7), look(.burst, 0.2)] {
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
        XCTAssertEqual(lit.look(at: t0, level: 0, reduceMotion: false).opacity, 0.88, accuracy: 1e-9)
        let p = lit.processing(at: t0.addingTimeInterval(3))
        XCTAssertEqual(p.processing(at: t0.addingTimeInterval(4)), p, "a second stop keeps the gather where it is")
        XCTAssertEqual(BarBeat.rest.burst(at: t0, level: 0, reduceMotion: false), .rest, "no burst from nothing")
        XCTAssertEqual(BarBeat.rest.fadeOut(at: t0, level: 0, reduceMotion: false), .rest)
    }

    // MARK: the capsule's events

    func testCapsuleEventsDriveTheStages() {
        let t0 = Date(timeIntervalSinceReferenceDate: 1000)
        let pill = PillState()
        pill.mode = .listening
        pill.ignite(now: t0)
        pill.reduceMotion = false
        XCTAssertEqual(pill.beat.stage, .listening)
        XCTAssertEqual(pill.drawnBeat.look(at: t0, level: 0, reduceMotion: false).opacity, 0.88, accuracy: 1e-9)
        pill.mode = .working
        pill.startProcessing(now: t0.addingTimeInterval(2))
        XCTAssertEqual(pill.beat.stage, .processing)
        pill.startProcessing(now: t0.addingTimeInterval(3))  // a queued first pass calls it again
        XCTAssertEqual(pill.beat.since, t0.addingTimeInterval(2))
        pill.delivered(now: t0.addingTimeInterval(4))
        XCTAssertEqual(pill.beat.stage, .burst)
        XCTAssertEqual(pill.beat.from.width, 0.08, accuracy: 1e-9)

        let failed = PillState()
        failed.mode = .working
        failed.ignite(now: t0)
        failed.startProcessing(now: t0)
        failed.undelivered(now: t0.addingTimeInterval(0.5))
        XCTAssertEqual(failed.beat.stage, .fadeOut, "no burst when nothing was pasted")
        // without any event the mode still draws: listening lit, working the held core
        let bare = PillState()
        bare.mode = .working
        XCTAssertEqual(bare.drawnBeat.stage, .processing)
        bare.mode = .idle
        XCTAssertEqual(bare.drawnBeat.stage, .rest)
    }

    // MARK: stills

    func testWritesStageStills() throws {
        guard let output = ProcessInfo.processInfo.environment["PHONON_RENDER_DIR"] else { return }
        let t0 = Date(timeIntervalSinceReferenceDate: 5000)
        let shots: [(String, PillState.Mode, Double, Double)] = [
            ("1-ignite-t0", .listening, 0, 0), ("2-listening-mid", .listening, 1, 0.5),
            ("3-gather-0.6s", .working, 0.6, 0), ("4-gather-held", .working, 3, 0),
            ("5-burst-0.3s", .working, 3.3, 0),
        ]
        for (name, mode, at, level) in shots {
            let pill = PillState()
            pill.mode = mode
            pill.level = level
            pill.ignite(now: t0)
            pill.reduceMotion = false
            if mode == .working {
                pill.startProcessing(now: t0)
                if name.hasPrefix("5") { pill.delivered(now: t0.addingTimeInterval(3)) }
            }
            let view = PillView(state: pill, fixedTime: t0.addingTimeInterval(at))
                .frame(width: 120, height: 40)
                .background(Color(white: 0.2))
                .environment(\.colorScheme, .dark)
            let renderer = ImageRenderer(content: view)
            renderer.scale = 8
            let image = try XCTUnwrap(renderer.nsImage, name)
            let rep = try XCTUnwrap(NSBitmapImageRep(data: try XCTUnwrap(image.tiffRepresentation)))
            let png = try XCTUnwrap(rep.representation(using: .png, properties: [:]))
            let url = URL(fileURLWithPath: output).appendingPathComponent("capsule-stage-\(name).png")
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try png.write(to: url)
        }
    }
}
