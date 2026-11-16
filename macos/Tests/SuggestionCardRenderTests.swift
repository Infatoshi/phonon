import SwiftUI
import XCTest

@testable import PhononBar

/// Renders the recommendation card off screen (no window) so its copy and
/// layout can be checked. Set PHONON_RENDER_DIR to keep the PNGs.
@MainActor
final class SuggestionCardRenderTests: XCTestCase {
    private func render(_ model: SuggestionPanelModel, name: String) throws -> CGImage {
        let renderer = ImageRenderer(content: SuggestionCard(model: model))
        renderer.scale = 2
        let image = try XCTUnwrap(renderer.cgImage)
        if let directory = ProcessInfo.processInfo.environment["PHONON_RENDER_DIR"] {
            let url = URL(fileURLWithPath: directory).appendingPathComponent("\(name).png")
            let destination = try XCTUnwrap(
                CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil))
            CGImageDestinationAddImage(destination, image, nil)
            XCTAssertTrue(CGImageDestinationFinalize(destination))
        }
        return image
    }

    /// The Settings list of learned styles, rendered off screen in dark mode.
    func testLearnedStylesListRenders() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "phonon-styles-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = NativeAppStore(supportDirectory: directory)
        store.approveStylePreference("lowercase_start", category: "chat")
        store.approveStylePreference("no_trailing_period", category: "email")
        let view = LearnedStylesList(store: store, onChanged: {})
            .frame(width: 520)
            .padding(12)
            .background(Color(white: 0.12))
            .environment(\.colorScheme, .dark)
        let renderer = ImageRenderer(content: view)
        renderer.scale = 2
        let image = try XCTUnwrap(renderer.cgImage)
        XCTAssertGreaterThan(image.height, 60, "two rows and a caption")
        if let output = ProcessInfo.processInfo.environment["PHONON_RENDER_DIR"] {
            let url = URL(fileURLWithPath: output).appendingPathComponent("settings_styles.png")
            let destination = try XCTUnwrap(
                CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil))
            CGImageDestinationAddImage(destination, image, nil)
            XCTAssertTrue(CGImageDestinationFinalize(destination))
        }
    }

    func testCardsFitTheirFrame() throws {
        let term = SuggestionPanelModel()
        let suggestion = EditSuggestion(
            index: 0,
            kind: .term(term: "cuBLAS", heardAs: "coo blast", casing: false, existingEntry: false))
        term.title = EditSuggestionCopy.title(suggestion, category: "chat")
        term.detail = EditSuggestionCopy.detail(suggestion)
        term.approveLabel = EditSuggestionCopy.approveLabel(suggestion)
        let termImage = try render(term, name: "card_term")
        XCTAssertEqual(termImage.width, Int(SuggestionPanelGeometry.size.width * 2))
        XCTAssertLessThanOrEqual(termImage.height, Int(SuggestionPanelGeometry.size.height * 2))

        let style = SuggestionPanelModel()
        let styleSuggestion = EditSuggestion(index: 1, kind: .style("no_trailing_period"))
        style.title = EditSuggestionCopy.title(styleSuggestion, category: "chat")
        style.detail = EditSuggestionCopy.detail(styleSuggestion)
        style.approveLabel = EditSuggestionCopy.approveLabel(styleSuggestion)
        _ = try render(style, name: "card_style")

        let done = SuggestionPanelModel()
        done.confirmation = "Added \u{201C}cuBLAS\u{201D}"
        done.canUndo = true
        _ = try render(done, name: "card_confirm")

        let meeting = SuggestionPanelModel()
        meeting.title = "Meeting detected in Zoom: take notes?"
        meeting.detail = "Phonon records your microphone and the call audio on this Mac."
        meeting.approveLabel = "Take notes"
        let meetingImage = try render(meeting, name: "card_meeting")
        // Two-line cards grow the panel; the frame follows the fitting height.
        let fitted = SuggestionPanelGeometry.frame(
            screen: CGRect(x: 0, y: 0, width: 1728, height: 1117),
            height: CGFloat(meetingImage.height) / 2)
        XCTAssertGreaterThanOrEqual(fitted.height * 2, CGFloat(meetingImage.height))
        XCTAssertLessThanOrEqual(meetingImage.height, 2 * 100, "at most about three lines")
    }
}
