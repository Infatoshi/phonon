import XCTest

@testable import PhononBar

final class StreamRefineTests: XCTestCase {
    func testTheEngineReadyEventDecidesStreamingRefine() {
        XCTAssertTrue(StreamRefine.enabled(readyEvent: ["type": "ready", "stream_refine": true]))
        XCTAssertFalse(StreamRefine.enabled(readyEvent: ["type": "ready", "stream_refine": false]))
        // an engine from before streaming refine says nothing: off
        XCTAssertFalse(StreamRefine.enabled(readyEvent: ["type": "ready"]))
    }
}
