import SwiftUI
import XCTest

/// Offscreen renders of the Dictionary screen (synthetic entries and the empty state, light and dark scheme) to
/// PNG for a person to look at. Writes only when PHONON_SHOTS_DIR is set (TEST_RUNNER_PHONON_SHOTS_DIR for
/// xcodebuild); otherwise it only checks that the screen renders.
@MainActor
final class DictionaryScreenRenderTests: XCTestCase {
    private let sample: [UserDictionary.Entry] = UserDictionary.sortForSave([
        .init(phrase: "Kubernetes", spokenForms: ["cooper netties"]),
        .init(phrase: "Tailscale"),
        .init(phrase: "vee llm", replacement: "vLLM"),
        .init(phrase: "cuBLAS"),
        .init(phrase: "pie torch", replacement: "PyTorch"),
    ])

    private func render(_ name: String, _ entries: [UserDictionary.Entry], scheme: ColorScheme,
                        phrase: String = "", replacement: String = "") throws {
        let view = DictionaryContent(entries: entries, onAdd: { _, _ in }, onRemove: { _ in }, onClose: {},
                                     staticFields: true, phrase: phrase, replacement: replacement)
            .frame(width: 393)
            .environment(\.colorScheme, scheme)
        let r = ImageRenderer(content: view)
        r.scale = 3
        let image = try XCTUnwrap(r.uiImage, "\(name) did not render")
        XCTAssertGreaterThan(image.size.height, 300)
        guard let dir = ProcessInfo.processInfo.environment["PHONON_SHOTS_DIR"] else { return }
        let url = URL(fileURLWithPath: dir).appendingPathComponent("\(name)-\(scheme == .dark ? "dark" : "light").png")
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try XCTUnwrap(image.pngData()).write(to: url)
    }

    func testRendersTheDictionaryScreen() throws {
        for scheme in [ColorScheme.dark, .light] {
            try render("dictionary-entries", sample, scheme: scheme)
            try render("dictionary-empty", [], scheme: scheme)
            try render("dictionary-adding", sample, scheme: scheme, phrase: "tail scale", replacement: "Tailscale")
        }
    }
}
