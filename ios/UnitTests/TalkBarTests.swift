import SwiftUI
import XCTest

/// The keyboard's talk bar keeps one frame in every state and audio level (only its fill changes), and the side
/// button switches between stop and settings. Renders offscreen; writes bar-{idle,quiet,loud}.png to PHONON_SHOTS_DIR
/// (TEST_RUNNER_PHONON_SHOTS_DIR for xcodebuild) when it is set.
@MainActor
final class TalkBarTests: XCTestCase {
    /// iPhone 16/17 width minus the toolbar's 6 pt side margins.
    private let width: CGFloat = 393 - 12
    private let scale: CGFloat = 3

    private final class Host: KeyboardHost {
        var calls: [String] = []
        func insert(_ text: String) {}
        func deleteBackward() {}
        func nextKeyboard() {}
        var contextBefore: String { "" }
        func micDown() {}
        func micUp() {}
        func cancelDictation() { calls.append("cancel") }
        func undoDictation() {}
        func openApp() { calls.append("open") }
        func startViaApp() { calls.append("start") }
    }

    /// The side button the real keyboard shows in a stage: the phase that stage runs in, and the light itself (the X
    /// while lit or gathering, the settings circle once the text is in or the fade has begun).
    private func control(for beat: BarBeat) -> TalkBarSideControl {
        let phase: KeyboardPhase
        switch beat.stage {
        case .listening: phase = .listening
        case .processing: phase = .working
        case .rest, .burst, .fadeOut: phase = .ready
        }
        return TalkBarSideControl(phase: phase, beat: beat.stage)
    }

    private func strip(_ phase: KeyboardPhase, level: Double) -> some View {
        TalkBarStrip(control: TalkBarSideControl(phase: phase), onSide: {}) {
            TalkBarFill(state: TalkBarFill.Mode(phase: phase), level: level, animate: false)
        }
        .frame(width: width)
    }

    private func render(_ view: some View) throws -> CGImage {
        let r = ImageRenderer(content: view)
        r.scale = scale
        r.isOpaque = false
        return try XCTUnwrap(r.cgImage, "did not render")
    }

    /// Runs of non-transparent pixels along row `y`, as (first x, last x).
    private func runs(_ image: CGImage, row y: Int) -> [(Int, Int)] {
        let px = alpha(image)
        var out: [(Int, Int)] = []
        var start: Int?
        for x in 0..<image.width {
            let on = px[y * image.width + x] > 0
            if on, start == nil { start = x }
            if !on, let s = start { out.append((s, x - 1)); start = nil }
        }
        if let s = start { out.append((s, image.width - 1)) }
        return out
    }

    /// Opaque extent of column `x`, as (first y, last y).
    private func column(_ image: CGImage, x: Int) -> (Int, Int)? {
        let px = alpha(image)
        let ys = (0..<image.height).filter { px[$0 * image.width + x] > 0 }
        guard let a = ys.first, let b = ys.last else { return nil }
        return (a, b)
    }

    private func alpha(_ image: CGImage) -> [UInt8] {
        var rgba = [UInt8](repeating: 0, count: image.width * image.height * 4)
        rgba.withUnsafeMutableBytes { raw in
            let ctx = CGContext(data: raw.baseAddress, width: image.width, height: image.height, bitsPerComponent: 8,
                                bytesPerRow: image.width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        }
        return stride(from: 3, to: rgba.count, by: 4).map { rgba[$0] }
    }

    private struct Measured: Equatable {
        let fit: CGSize
        let image: CGSize
        let barX: Int, barRight: Int, barTop: Int, barBottom: Int
    }

    private func measure(_ phase: KeyboardPhase, level: Double) throws -> Measured {
        let fit = UIHostingController(rootView: strip(phase, level: level))
            .sizeThatFits(in: CGSize(width: width, height: .greatestFiniteMagnitude))
        let image = try render(strip(phase, level: level))
        let segs = runs(image, row: image.height / 2)
        XCTAssertEqual(segs.count, 2, "\(phase) \(level): expected side button then bar, got \(segs)")
        let bar = try XCTUnwrap(segs.last)
        let col = try XCTUnwrap(column(image, x: (bar.0 + bar.1) / 2))
        return Measured(fit: fit, image: CGSize(width: image.width, height: image.height),
                        barX: bar.0, barRight: bar.1, barTop: col.0, barBottom: col.1)
    }

    func testBarFrameIsTheSameInEveryStateAndLevel() throws {
        let reference = try measure(.ready, level: 0)
        let cases: [(KeyboardPhase, Double)] = [
            (.ready, 0.5), (.ready, 1), (.needsSession, 0), (.needsAccess, 0),
            (.listening, 0), (.listening, 0.5), (.listening, 1), (.working, 0), (.working, 1),
        ]
        for (phase, level) in cases {
            XCTAssertEqual(try measure(phase, level: level), reference, "\(phase) at level \(level)")
        }
        // the bar spans the row after the 44 pt side button and the 8 pt gap, at the full 42 pt height
        let barW = Double(reference.barRight - reference.barX + 1) / scale
        let barH = Double(reference.barBottom - reference.barTop + 1) / scale
        XCTAssertEqual(barW, Double(width - 44 - 8), accuracy: 1)
        XCTAssertEqual(barH, 42, accuracy: 1)
        XCTAssertEqual(Double(reference.barRight + 1) / scale, Double(width), accuracy: 1)
    }

    func testSideControlIsStopWhileDictatingAndSettingsOtherwise() {
        XCTAssertEqual(TalkBarSideControl(phase: .listening), .stop)
        XCTAssertEqual(TalkBarSideControl(phase: .working), .stop)
        for phase in [KeyboardPhase.ready, .needsSession, .needsAccess] {
            XCTAssertEqual(TalkBarSideControl(phase: phase), .settings, "\(phase)")
        }
        XCTAssertEqual(TalkBarSideControl.stop.symbol, "xmark")
        XCTAssertEqual(TalkBarSideControl.stop.accessibilityLabel, "Cancel dictation")
        XCTAssertEqual(TalkBarSideControl.settings.symbol, "gearshape.fill")
        XCTAssertEqual(TalkBarSideControl.settings.accessibilityLabel, "Open Phonon")
        XCTAssertGreaterThanOrEqual(TalkBarStrip<EmptyView>.sideSize, 44)

        let host = Host()
        TalkBarSideControl.stop.perform(on: host)
        TalkBarSideControl.settings.perform(on: host)
        XCTAssertEqual(host.calls, ["cancel", "open"], "the settings circle only opens Phonon; it never arms a dictation")
    }

    func testFillBrightensWithLevel() {
        let quiet = TalkBarFill(state: .listening, level: 0, animate: false).look(at: .now).opacity
        let loud = TalkBarFill(state: .listening, level: 1, animate: false).look(at: .now).opacity
        XCTAssertLessThan(quiet, loud)
        // Reduce Motion: one static brightness whatever the level
        XCTAssertEqual(TalkBarFill(state: .listening, level: 0, followLevel: false).look(at: .now),
                       TalkBarFill(state: .listening, level: 1, followLevel: false).look(at: .now))
    }

    /// Reduce Motion, Low Power Mode or off screen (`animate` false): one width in every stage, so key up, the paste
    /// and the fade change brightness only.
    func testReducedBarKeepsOneWidthThroughEveryStage() {
        let t0 = Date(timeIntervalSinceReferenceDate: 5000)
        let lit = BarBeat.ignite(at: t0)
        let up = t0.addingTimeInterval(2)
        let gather = lit.processing(at: up)
        let burst = gather.burst(at: up.addingTimeInterval(1), level: 0, reduceMotion: true)
        let faded = gather.fadeOut(at: up.addingTimeInterval(1), level: 0, reduceMotion: true)
        var widths: Set<Double> = []
        for (beat, after) in [(lit, 0.0), (lit, 0.1), (lit, 1.0), (gather, 0.0), (gather, 0.6), (gather, 3.0),
                              (burst, 0.0), (burst, 0.15), (faded, 0.1)] {
            for fill in [TalkBarFill(state: .idle, level: 0.6, animate: false, beat: beat),
                         TalkBarFill(state: .idle, level: 0.6, followLevel: false, beat: beat)] {
                widths.insert(fill.look(at: beat.since.addingTimeInterval(after)).width)
            }
        }
        XCTAssertEqual(widths, [BarStages.reducedWidth])
    }

    /// The bar's frame is the same in every stage of the light, not just every phase.
    func testBarFrameIsTheSameInEveryStageOfTheLight() throws {
        let reference = try measure(.ready, level: 0)
        let t0 = Date(timeIntervalSinceReferenceDate: 5000)
        let lit = BarBeat.ignite(at: t0)
        let gather = lit.processing(at: t0)
        let cases: [(BarBeat, Double)] = [
            (lit, 0), (gather, 0.6), (gather, 3),
            (gather.burst(at: t0.addingTimeInterval(3), level: 0, reduceMotion: false), 0.3),
            (gather.fadeOut(at: t0.addingTimeInterval(3), level: 0, reduceMotion: false), 0.1),
        ]
        for (beat, after) in cases {
            let view = TalkBarStrip(control: control(for: beat), onSide: {}) {
                TalkBarFill(state: .idle, level: 0, beat: beat, frozenAt: beat.since.addingTimeInterval(after))
            }
            .frame(width: width)
            let image = try render(view)
            let bar = try XCTUnwrap(runs(image, row: image.height / 2).last)
            let col = try XCTUnwrap(column(image, x: (bar.0 + bar.1) / 2))
            XCTAssertEqual(bar.0, reference.barX, "\(beat.stage) at \(after)")
            XCTAssertEqual(bar.1, reference.barRight, "\(beat.stage) at \(after)")
            XCTAssertEqual(col.0, reference.barTop, "\(beat.stage) at \(after)")
            XCTAssertEqual(col.1, reference.barBottom, "\(beat.stage) at \(after)")
        }
    }

    /// Stage stills for a person to look at: ignite at t = 0, listening, the gather at 0.6 s, the held core, the burst
    /// at 0.3 s. Written to PHONON_SHOTS_DIR as stage-*.png when it is set.
    func testWritesStageShots() throws {
        let t0 = Date(timeIntervalSinceReferenceDate: 5000)
        let lit = BarBeat.ignite(at: t0)
        let gather = lit.processing(at: t0)
        let burst = gather.burst(at: t0.addingTimeInterval(3), level: 0, reduceMotion: false)
        let shots: [(String, BarBeat, Double, Double)] = [
            ("1-ignite-t0", lit, 0, 0), ("2-listening-mid", lit, 1, 0.5), ("3-gather-0.6s", gather, 0.6, 0),
            ("4-gather-held", gather, 3, 0), ("5-burst-0.3s", burst, 0.3, 0),
        ]
        for (name, beat, after, level) in shots {
            let view = TalkBarStrip(control: control(for: beat), onSide: {}) {
                TalkBarFill(state: .idle, level: level, beat: beat, frozenAt: beat.since.addingTimeInterval(after))
            }
            .frame(width: width)
            .padding(.horizontal, 6)
            .padding(.vertical, 6)
            .background(Color(hex: 0x16120F))
            .environment(\.colorScheme, .dark)
            let r = ImageRenderer(content: view)
            r.scale = scale
            let image = try XCTUnwrap(r.uiImage, "\(name) did not render")
            guard let dir = ProcessInfo.processInfo.environment["PHONON_SHOTS_DIR"] else { continue }
            let url = URL(fileURLWithPath: dir).appendingPathComponent("stage-\(name).png")
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try XCTUnwrap(image.pngData()).write(to: url)
        }
    }

    func testVoiceLevelGlidesInsteadOfFlickering() {
        XCTAssertEqual(VoiceLevel.smooth([]), 0)
        XCTAssertEqual(VoiceLevel.smooth(Array(repeating: 0, count: 28)), 0, accuracy: 1e-9)
        // steady speech converges to its level
        XCTAssertEqual(VoiceLevel.smooth(Array(repeating: 0.6, count: 28)), 0.6, accuracy: 0.01)
        // alternating loud and quiet samples (15 per second) move the output far less than the input swings
        let jittery: [CGFloat] = (0..<28).map { $0 % 2 == 0 ? 0.9 : 0.1 }
        let a = VoiceLevel.smooth(jittery)
        let b = VoiceLevel.smooth(Array(jittery.dropFirst()) + [0.9])
        XCTAssertLessThan(abs(a - b), 0.25)
        XCTAssertLessThan(abs(a - b), 0.8 * 0.5)
        // a rise shows at once, a fall fades
        let rise = VoiceLevel.smooth(Array(repeating: 0, count: 27) + [1])
        XCTAssertGreaterThan(rise, 0.5)
        let fall = VoiceLevel.smooth(Array(repeating: 1, count: 27) + [0])
        XCTAssertGreaterThan(fall, 0.8)
        // out-of-range samples are clamped
        XCTAssertLessThanOrEqual(VoiceLevel.smooth(Array(repeating: 5, count: 28)), 1)
    }

    /// Dark-mode renders on the keyboard's background for a person to look at.
    func testWritesBarShots() throws {
        let shots: [(String, KeyboardPhase, Double)] = [("idle", .ready, 0), ("quiet", .listening, 0.1),
                                                        ("mid", .listening, 0.5), ("loud", .listening, 1)]
        for (name, phase, level) in shots {
            let view = strip(phase, level: level)
                .padding(.horizontal, 6)
                .padding(.vertical, 6)
                .background(Color(hex: 0x16120F))
                .environment(\.colorScheme, .dark)
            let r = ImageRenderer(content: view)
            r.scale = scale
            let image = try XCTUnwrap(r.uiImage, "\(name) did not render")
            guard let dir = ProcessInfo.processInfo.environment["PHONON_SHOTS_DIR"] else { continue }
            let url = URL(fileURLWithPath: dir).appendingPathComponent("bar-\(name).png")
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try XCTUnwrap(image.pngData()).write(to: url)
        }
    }
}
