import AppKit
import SwiftUI
import XCTest

@testable import PhononBar

/// The stage maths of the dictation light (BarStages.swift): ignite, listening, steady warm processing, fade-out, and
/// the Reduce Motion path. The same cases run in the iPhone app's tests (ios/UnitTests/BarStagesTests.swift).
/// Also the capsule's events (PillState) and offscreen stills; set PHONON_RENDER_DIR to keep them.
@MainActor
final class BarStagesTests: XCTestCase {
    private func look(_ stage: BarStage, _ t: Double, level: Double = 0.4, reduced: Bool = false,
                      from: BarLook = BarLook(width: BarStages.wideWidth, heat: 0,
                                              opacity: BarStages.processingOpacity)) -> BarLook {
        BarStages.look(stage: stage, elapsed: t, level: level, reduceMotion: reduced, from: from)
    }

    /// The named constants the other cases use, pinned to the spec's numbers once.
    func testNamedConstantsHoldTheSpecValues() {
        XCTAssertEqual(BarStages.ignitedOpacity, 0.88, accuracy: 1e-9)
        XCTAssertEqual(BarStages.processingOpacity, 0.65, accuracy: 1e-9)
        XCTAssertEqual(BarStages.igniteSeconds, 0.10)
        XCTAssertEqual(BarStages.fadeOutSeconds, 0.10)
        XCTAssertEqual(BarStages.wideWidth, 0.5)
        XCTAssertEqual(BarStages.reducedWidth, 0.5)
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

    func testProcessingIsSteadyFullWidthWarmEmberWithNoSlidingCore() {
        let steady = look(.processing, 0)
        XCTAssertEqual(steady.width, BarStages.wideWidth, accuracy: 1e-9)
        XCTAssertEqual(steady.heat, 0, accuracy: 1e-9)
        XCTAssertEqual(steady.opacity, BarStages.processingOpacity, accuracy: 1e-9)
        XCTAssertEqual(BarStages.duration(of: .processing, reduceMotion: false), 0)
        XCTAssertEqual(BarStages.duration(of: .processing, reduceMotion: true), 0)
        for t in [0.02, 0.1, 0.5, 1.2, 10.0, 600.0] {
            XCTAssertEqual(look(.processing, t), steady, "steady, no sliding or breathing at \(t) s")
            XCTAssertEqual(look(.processing, t, reduced: true), steady, "identical under Reduce Motion at \(t) s")
        }
        let c = steady.rgba
        XCTAssertEqual(c.r, BarStages.baseColor.r, accuracy: 1e-9)
        XCTAssertEqual(c.g, BarStages.baseColor.g, accuracy: 1e-9)
        XCTAssertEqual(c.b, BarStages.baseColor.b, accuracy: 1e-9)
        XCTAssertEqual(c.a, BarStages.processingOpacity, accuracy: 1e-9)
    }

    func testFadeOutStartsContinuouslyAndExtinguishesCleanlyAtFullWidth() {
        let t0 = Date(timeIntervalSinceReferenceDate: 1000)
        let processing = BarBeat.ignite(at: t0).processing(at: t0)
        let doneAt = t0.addingTimeInterval(0.02)
        let early = processing.fadeOut(at: doneAt, level: 0.4, reduceMotion: false)
        XCTAssertEqual(early.stage, .fadeOut)
        XCTAssertEqual(early.from.width, BarStages.wideWidth, accuracy: 1e-9)
        XCTAssertEqual(early.from.opacity, BarStages.processingOpacity, accuracy: 1e-9)
        XCTAssertEqual(early.look(at: doneAt, level: 0.4, reduceMotion: false), early.from,
                       "continuous: no jump at the moment of the paste")
        let midFade = early.look(at: doneAt.addingTimeInterval(BarStages.fadeOutSeconds / 2), level: 0, reduceMotion: false)
        XCTAssertEqual(midFade.width, BarStages.wideWidth, accuracy: 1e-9, "width never slides during fade-out")
        XCTAssertEqual(midFade.opacity, BarStages.processingOpacity * 0.25, accuracy: 1e-9)
        let endFade = early.look(at: doneAt.addingTimeInterval(BarStages.fadeOutSeconds), level: 0, reduceMotion: false)
        XCTAssertEqual(endFade.width, BarStages.wideWidth, accuracy: 1e-9)
        XCTAssertEqual(endFade.opacity, 0, accuracy: 1e-9)
    }

    func testTheEndFadesWithNoPulseOrWidthChange() {
        let steady = look(.processing, 3)
        for t in stride(from: 0.0, through: 0.4, by: 0.04) {
            let f = look(.fadeOut, t, from: steady)
            XCTAssertEqual(f.width, BarStages.wideWidth, accuracy: 1e-9, "width stays full at \(t)")
            XCTAssertEqual(f.heat, 0, accuracy: 1e-9, "no colour change at \(t)")
            XCTAssertLessThanOrEqual(f.opacity, steady.opacity, "never brighter at \(t)")
        }
        XCTAssertEqual(look(.fadeOut, 0, from: steady).opacity, BarStages.processingOpacity, accuracy: 1e-9)
        XCTAssertEqual(look(.fadeOut, BarStages.fadeOutSeconds / 2, from: steady).opacity,
                       BarStages.processingOpacity * 0.25, accuracy: 1e-9)
        XCTAssertEqual(look(.fadeOut, BarStages.fadeOutSeconds, from: steady).opacity, 0, accuracy: 1e-9)
        XCTAssertEqual(look(.fadeOut, 1, from: steady).rgba.a, 0, accuracy: 1e-9)
        let t0 = Date(timeIntervalSinceReferenceDate: 1000)
        let ended = BarBeat.ignite(at: t0).processing(at: t0).fadeOut(at: t0.addingTimeInterval(2), level: 0, reduceMotion: false)
        XCTAssertEqual(ended.stage, .fadeOut)
        XCTAssertEqual(ended.settleSeconds(reduceMotion: false), BarStages.fadeOutSeconds)
    }

    func testFadeOutImmediatelyTargetsIdleShellSize() {
        let idle = PillView.shellSize(mode: .idle, beatStage: .rest, energy: 0, meetingRecording: false)
        let fadeOut = PillView.shellSize(mode: .working, beatStage: .fadeOut, energy: 0.27, meetingRecording: false)
        XCTAssertEqual(fadeOut, idle, "fade-out glides directly to idle size with no second-stage shrink jerk")
        let meetingIdle = PillView.shellSize(mode: .idle, beatStage: .rest, energy: 0, meetingRecording: true)
        let meetingFadeOut = PillView.shellSize(mode: .working, beatStage: .fadeOut, energy: 0.27, meetingRecording: true)
        XCTAssertEqual(meetingFadeOut, meetingIdle)
    }

    func testWidthNeverMovesInAnyStageNormalOrReduceMotion() {
        for reduced in [false, true] {
            let widths = Set(
                [0.0, 0.04, 0.4, 1.2, 5].map { look(.processing, $0, reduced: reduced).width }
                    + [0.0, 0.05, 0.1, 0.3, 1].map { look(.listening, $0, reduced: reduced).width }
                    + [0.0, 0.05, 0.1, 0.3, 1].map {
                        look(.fadeOut, $0, reduced: reduced, from: look(.processing, 1, reduced: reduced)).width
                    })
            XCTAssertEqual(widths, [BarStages.wideWidth],
                           "one full width from key down to the end (reduced=\(reduced)): \(widths)")
        }
    }

    func testRestDrawsNothingAndTheGradientIsBlackAtBothEnds() {
        XCTAssertEqual(look(.rest, 1).rgba.a, 0)
        for l in [look(.listening, 0), look(.processing, 0.7), look(.fadeOut, 0.05)] {
            let s = l.stops
            XCTAssertEqual(s.first?.location, 0)
            XCTAssertEqual(s.last?.location, 1)
            for end in [s.first!, s.last!] { XCTAssertEqual([end.r, end.g, end.b, end.a], [0, 0, 0, 1]) }
            XCTAssertEqual(s.map(\.location), s.map(\.location).sorted(), "stops stay in order")
        }
        let s = look(.processing, 1.2).stops
        XCTAssertEqual(s[1].location, 0.0, accuracy: 1e-9)
        XCTAssertEqual(s[2].location, 0.5)
        XCTAssertEqual(s[3].location, 1.0, accuracy: 1e-9)
    }

    func testBeatReplacesOnEachRealEvent() {
        let t0 = Date(timeIntervalSinceReferenceDate: 1000)
        let lit = BarBeat.ignite(at: t0)
        XCTAssertEqual(lit.look(at: t0, level: 0, reduceMotion: false).opacity, BarStages.ignitedOpacity,
                       accuracy: 1e-9)
        let p = lit.processing(at: t0.addingTimeInterval(3))
        XCTAssertEqual(p.processing(at: t0.addingTimeInterval(4)), p, "a second stop keeps processing where it is")
        XCTAssertEqual(BarBeat.rest.fadeOut(at: t0, level: 0, reduceMotion: false), .rest, "no fade from nothing")
    }

    // MARK: the capsule's events

    func testCapsuleEventsDriveTheStages() {
        let t0 = Date(timeIntervalSinceReferenceDate: 1000)
        let pill = PillState()
        pill.mode = .listening
        pill.ignite(now: t0)
        pill.reduceMotion = false
        XCTAssertEqual(pill.beat.stage, .listening)
        XCTAssertEqual(pill.drawnBeat.look(at: t0, level: 0, reduceMotion: false).opacity, BarStages.ignitedOpacity,
                       accuracy: 1e-9)
        pill.mode = .working
        pill.startProcessing(now: t0.addingTimeInterval(2))
        XCTAssertEqual(pill.beat.stage, .processing)
        pill.startProcessing(now: t0.addingTimeInterval(3))  // a queued first pass calls it again
        XCTAssertEqual(pill.beat.since, t0.addingTimeInterval(2))
        pill.fadeOut(now: t0.addingTimeInterval(4))
        XCTAssertEqual(pill.beat.stage, .fadeOut, "the text went in: a plain fade back to rest, no pulse")
        XCTAssertEqual(pill.beat.from.width, BarStages.wideWidth, accuracy: 1e-9)
        // without any event the mode still draws: listening lit, working steady ember
        let bare = PillState()
        bare.mode = .working
        XCTAssertEqual(bare.drawnBeat.stage, .processing)
        bare.mode = .idle
        XCTAssertEqual(bare.drawnBeat.stage, .rest)
    }

    /// Low Power Mode keeps the light's width still on the Mac, as Reduce Motion does.
    func testLowPowerModeKeepsTheCapsuleWidthStill() {
        XCTAssertTrue(PillState.stillLight(reduceMotion: false, lowPower: true))
        XCTAssertTrue(PillState.stillLight(reduceMotion: true, lowPower: false))
        XCTAssertFalse(PillState.stillLight(reduceMotion: false, lowPower: false))

        let t0 = Date(timeIntervalSinceReferenceDate: 1000)
        let pill = PillState()
        pill.mode = .listening
        pill.level = 0.6
        pill.ignite(now: t0, still: PillState.stillLight(reduceMotion: false, lowPower: true))
        XCTAssertTrue(pill.reduceMotion)
        var widths: Set<Double> = []
        for t in [0.0, 0.1, 0.5] {
            widths.insert(pill.drawnBeat.look(at: t0.addingTimeInterval(t), level: pill.level,
                                              reduceMotion: pill.reduceMotion).width)
        }
        pill.mode = .working
        pill.startProcessing(now: t0.addingTimeInterval(1))
        for t in [1.0, 1.6, 2.2, 9.0] {
            widths.insert(pill.drawnBeat.look(at: t0.addingTimeInterval(t), level: 0,
                                              reduceMotion: pill.reduceMotion).width)
        }
        pill.fadeOut(now: t0.addingTimeInterval(9))
        for t in [9.0, 9.05, 9.1] {
            widths.insert(pill.drawnBeat.look(at: t0.addingTimeInterval(t), level: 0,
                                              reduceMotion: pill.reduceMotion).width)
        }
        XCTAssertEqual(widths, [BarStages.reducedWidth], "Low Power: one width from key down to the end")
    }

    /// Engine errors end only the dictation they name. One without an id (prewarm, dictionary reload, an unknown
    /// command) must not fade or kill the light of the dictation in flight.
    func testOnlyTheActiveDictationsOwnErrorEndsIt() {
        XCTAssertTrue(AppController.errorEndsDictation(errorId: "p3", activeId: "p3"))
        XCTAssertFalse(AppController.errorEndsDictation(errorId: nil, activeId: "p3"), "no id: not about p3")
        XCTAssertFalse(AppController.errorEndsDictation(errorId: "p2", activeId: "p3"), "an older pass")
        XCTAssertFalse(AppController.errorEndsDictation(errorId: nil, activeId: nil))
        XCTAssertFalse(AppController.errorEndsDictation(errorId: "p3", activeId: nil))
    }

    // MARK: stills

    func testWritesStageStills() throws {
        guard let output = ProcessInfo.processInfo.environment["PHONON_RENDER_DIR"] else { return }
        let t0 = Date(timeIntervalSinceReferenceDate: 5000)
        // Render the clean full-width lifecycle: ignite -> listening -> steady processing -> fade-out -> rest.
        let shots: [(String, PillState.Mode, Double, Double, Double?)] = [
            ("1-ignite-t0", .listening, 0, 0, nil),
            ("2-listening-mid", .listening, 1, 0.5, nil),
            ("3-processing", .working, 0.04, 0, nil),
            ("4-fade-mid", .working, 0.09, 0, 0.04),
            ("5-rest", .idle, 0.3, 0, nil),
        ]
        for (name, mode, at, level, fadeAt) in shots {
            let pill = PillState()
            pill.mode = mode
            pill.level = level
            if mode != .idle {
                pill.ignite(now: t0)
                pill.reduceMotion = false
            }
            if mode == .working {
                pill.startProcessing(now: t0)
                if let fadeAt { pill.fadeOut(now: t0.addingTimeInterval(fadeAt)) }
            }
            let view = PillView(state: pill, fixedTime: t0.addingTimeInterval(at))
                .frame(width: 120, height: 40)
                .background(Color(white: 0.12))
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
