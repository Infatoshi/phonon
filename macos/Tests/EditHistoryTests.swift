import SwiftUI
import XCTest

@testable import PhononBar

final class WordDiffTests: XCTestCase {
    private func marked(_ tokens: [DiffToken]) -> [String] {
        tokens.filter(\.changed).map(\.text)
    }

    func testTokensGiveTheTextBack() {
        let text = "  Ship it.\n\n- first item\tdone  "
        XCTAssertEqual(WordDiff.tokens(text).joined(), text)
        XCTAssertEqual(WordDiff.tokens("a b"), ["a", " ", "b"])
        XCTAssertEqual(WordDiff.tokens(""), [])
    }

    func testChangedWordsAreMarkedOnBothSides() {
        let diff = WordDiff.compare(
            "We benchmark coo blast against cutlass today.",
            "We benchmark cuBLAS against CUTLASS today.")
        XCTAssertEqual(marked(diff.before), ["coo", "blast", "cutlass"])
        XCTAssertEqual(marked(diff.after), ["cuBLAS", "CUTLASS"])
        // Joined, each side is its own text again.
        XCTAssertEqual(
            diff.before.map(\.text).joined(), "We benchmark coo blast against cutlass today.")
        XCTAssertEqual(diff.after.map(\.text).joined(), "We benchmark cuBLAS against CUTLASS today.")
        XCTAssertFalse(diff.after.contains { $0.changed && $0.text.allSatisfy(\.isWhitespace) })
    }

    func testInsertionsDeletionsCaseAndPunctuationCount() {
        let added = WordDiff.compare("Ship it.", "Ship it now.")
        XCTAssertEqual(marked(added.before), ["it."])
        XCTAssertEqual(marked(added.after), ["it", "now."])
        let removed = WordDiff.compare("Ship it right now.", "Ship it now.")
        XCTAssertEqual(marked(removed.before), ["right"])
        XCTAssertEqual(marked(removed.after), [])
        let cased = WordDiff.compare("Sounds good.", "sounds good")
        XCTAssertEqual(marked(cased.before), ["Sounds", "good."])
        let same = WordDiff.compare("Same text here.", "Same text here.")
        XCTAssertTrue(marked(same.before).isEmpty && marked(same.after).isEmpty)
        let cleared = WordDiff.compare("Never mind.", "")
        XCTAssertEqual(marked(cleared.before), ["Never", "mind."])
        XCTAssertTrue(cleared.after.isEmpty)
    }

    func testVeryLongTextsMarkTheMiddleWithoutAligning() {
        let words = (0..<700).map { "w\($0)" }
        let before = words.joined(separator: " ")
        let after = (["start"] + words.reversed() + ["end"]).joined(separator: " ")
        let diff = WordDiff.compare(before, after)
        XCTAssertEqual(diff.after.filter(\.changed).count, 702)
    }
}

final class HistoryUserEditTests: XCTestCase {
    private func metadata(userEdit: String) throws -> NativeRecordingMetadata {
        let json = """
            {"id": "r1", "created_at_unix_ms": 1790812800000, "source": "bar",
             "audio_file": "audio.wav", "raw_transcript": "ask clod code",
             "final_transcript": "Ask clod code.", "user_edit": \(userEdit)}
            """
        return try JSONDecoder().decode(NativeRecordingMetadata.self, from: Data(json.utf8))
    }

    func testAUserEditIsReadFromTheCorpusEntry() throws {
        let edited = try metadata(
            userEdit: """
                {"schema_version": 1, "pasted_text": "Ask clod code.", "edited_text": "Ask Claude Code.",
                 "outcome": "correction", "word_edits": "MSS", "punctuation_edits": "ZZZ",
                 "edit_distance": 0.6, "end_reason": "textbox_emptied", "observed_ms": 4000}
                """)
        let item = NativeHistoryItem(
            metadata: edited, directoryURL: URL(fileURLWithPath: "/r1"),
            metadataURL: URL(fileURLWithPath: "/r1/metadata.json"))
        XCTAssertEqual(item.changedEdit?.editedText, "Ask Claude Code.")
        XCTAssertEqual(item.changedEdit?.outcome, "correction")
    }

    func testUnchangedMissingOrMalformedEditsLookAsBefore() throws {
        let unchanged = try metadata(
            userEdit: """
                {"pasted_text": "Ask clod code.", "edited_text": "Ask clod code. ",
                 "outcome": "appended", "end_reason": "textbox_emptied"}
                """)
        XCTAssertNotNil(unchanged.userEdit)
        XCTAssertFalse(unchanged.userEdit!.changed)
        XCTAssertNil(try metadata(userEdit: "null").userEdit)
        let broken = try metadata(userEdit: "{\"pasted_text\": 3}")
        XCTAssertNil(broken.userEdit, "a bad record does not hide the dictation")
        XCTAssertEqual(broken.finalTranscript, "Ask clod code.")
    }
}

/// Off-screen renders (no window) of the History comparison and rows, in
/// dark mode. Set PHONON_RENDER_DIR to keep the PNGs.
@MainActor
final class EditHistoryRenderTests: XCTestCase {
    private func render<V: View>(_ view: V, width: CGFloat, name: String) throws -> CGImage {
        let framed = view
            .frame(width: width)
            .padding(16)
            .background(EmberTheme.background)
            .environment(\.colorScheme, .dark)
        let renderer = ImageRenderer(content: framed)
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

    private func item(_ id: String, final: String, edit: (String, String)?) throws -> NativeHistoryItem {
        var object: [String: Any] = [
            "id": id, "created_at_unix_ms": 1_790_812_800_000, "source": "bar",
            "audio_file": "audio.wav", "raw_transcript": final.lowercased(),
            "final_transcript": final,
        ]
        if let edit {
            object["user_edit"] = [
                "pasted_text": edit.0, "edited_text": edit.1, "outcome": "correction",
                "end_reason": "textbox_emptied",
            ]
        }
        let metadata = try JSONDecoder().decode(
            NativeRecordingMetadata.self, from: JSONSerialization.data(withJSONObject: object))
        return NativeHistoryItem(
            metadata: metadata, directoryURL: URL(fileURLWithPath: "/\(id)"),
            metadataURL: URL(fileURLWithPath: "/\(id)/metadata.json"))
    }

    func testTheComparisonAndRowsRender() throws {
        let comparison = EditComparison(
            pasted: "We benchmark coo blast against cutlass on the B two hundred today.",
            edited: "We benchmark cuBLAS against CUTLASS on the B200 today.")
        let image = try render(comparison, width: 640, name: "history_edit_comparison")
        XCTAssertGreaterThan(image.height, 100)

        let deleted = EditComparison(pasted: "Never mind, ignore that.", edited: "")
        _ = try render(deleted, width: 640, name: "history_edit_deleted")

        let edited = try item(
            "a", final: "We benchmark coo blast against cutlass today.",
            edit: (
                "We benchmark coo blast against cutlass today.",
                "We benchmark cuBLAS against CUTLASS today."
            ))
        let plain = try item("b", final: "Ship it on Friday.", edit: nil)
        let rows = VStack(alignment: .leading, spacing: 0) {
            HomeDictationRow(item: edited, open: .constant(true))
            HomeDictationRow(item: plain)
        }
        _ = try render(rows, width: 640, name: "home_rows")
    }
}
