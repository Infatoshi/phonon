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
        XCTAssertEqual(host.calls, ["cancel", "open"])
    }

    func testFillBrightensWithLevel() {
        let quiet = TalkBarFill(state: .listening, level: 0, animate: false).energy(at: .now)
        let loud = TalkBarFill(state: .listening, level: 1, animate: false).energy(at: .now)
        XCTAssertLessThan(quiet, loud)
        // Reduce Motion: one static red whatever the level
        XCTAssertEqual(TalkBarFill(state: .listening, level: 0, followLevel: false).energy(at: .now),
                       TalkBarFill(state: .listening, level: 1, followLevel: false).energy(at: .now))
    }

    /// Dark-mode renders on the keyboard's background for a person to look at.
    func testWritesBarShots() throws {
        let shots: [(String, KeyboardPhase, Double)] = [("idle", .ready, 0), ("quiet", .listening, 0.05),
                                                        ("loud", .listening, 1)]
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
