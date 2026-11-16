import AppKit
import XCTest

@testable import PhononBar

/// The dictation put on the pasteboard only for the paste (the user's clipboard comes back
/// after it) carries the nspasteboard.org transient and concealed markers, so clipboard
/// managers do not record it. On a private pasteboard: the user's clipboard is never touched.
final class TyperPasteboardTests: XCTestCase {
    private var pasteboard: NSPasteboard!

    override func setUp() {
        pasteboard = NSPasteboard(name: NSPasteboard.Name("phonon-test-\(UUID().uuidString)"))
    }

    override func tearDown() {
        pasteboard.releaseGlobally()
    }

    func testATemporaryPasteIsMarkedForClipboardManagers() {
        pasteboard.clearContents()
        XCTAssertTrue(pasteboard.writeObjects([Typer.dictationItem("hello there ", temporary: true)]))
        let types = pasteboard.types ?? []
        XCTAssertTrue(types.contains(Typer.transientType))
        XCTAssertTrue(types.contains(Typer.concealedType))
        XCTAssertEqual(Typer.transientType.rawValue, "org.nspasteboard.TransientType")
        XCTAssertEqual(Typer.concealedType.rawValue, "org.nspasteboard.ConcealedType")
        XCTAssertEqual(pasteboard.string(forType: .string), "hello there ")
    }

    func testAPasteTheUserKeepsIsPlainText() {
        pasteboard.clearContents()
        XCTAssertTrue(pasteboard.writeObjects([Typer.dictationItem("keep me ", temporary: false)]))
        let types = pasteboard.types ?? []
        XCTAssertFalse(types.contains(Typer.transientType))
        XCTAssertFalse(types.contains(Typer.concealedType))
        XCTAssertEqual(pasteboard.string(forType: .string), "keep me ")
    }
}
