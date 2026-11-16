import Vision
import XCTest

@testable import PhononBar

final class ScreenOCRFallbackTests: XCTestCase {
    private struct E5RTError: Error {}

    func testAccurateTextIsUsedWhenItWorks() {
        var levels = [VNRequestTextRecognitionLevel]()
        let text = ScreenContextCapture.recognizeText { level in
            levels.append(level)
            return "cuBLAS"
        }
        XCTAssertEqual(text, "cuBLAS")
        XCTAssertEqual(levels, [.accurate])
    }

    /// The 2026-11-12 failure: a low-disk cache purge left every accurate
    /// request throwing for the life of the process. Fast text still reads.
    func testFailedAccurateRecognitionFallsBackToFast() {
        var levels = [VNRequestTextRecognitionLevel]()
        let text = ScreenContextCapture.recognizeText { level in
            levels.append(level)
            if level == .accurate { throw E5RTError() }
            return "cuBLAS"
        }
        XCTAssertEqual(text, "cuBLAS")
        XCTAssertEqual(levels, [.accurate, .fast])
    }

    func testBothFailingReadsAsNoText() {
        XCTAssertEqual(ScreenContextCapture.recognizeText { _ in throw E5RTError() }, "")
    }
}
