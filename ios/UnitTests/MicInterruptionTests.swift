import XCTest

/// A phone call or Siri takes the mic: the session must stop claiming alive, and come back when the interruption
/// ends (or on the next start). A media-services reset needs a new engine.
final class MicInterruptionTests: XCTestCase {
    func testCallTakesMicThenEndsWithResume() {
        var m = MicInterruption()
        XCTAssertTrue(m.alive(running: true, engineRunning: true))
        XCTAssertEqual(m.handle(.began, running: true), .pause)
        XCTAssertTrue(m.down)
        XCTAssertFalse(m.alive(running: true, engineRunning: true), "an interrupted mic must not look alive")
        XCTAssertEqual(m.handle(.ended(shouldResume: true), running: true), .restart)
        XCTAssertFalse(m.alive(running: true, engineRunning: true), "still down until the engine really restarts")
        m.restarted()
        XCTAssertTrue(m.alive(running: true, engineRunning: true))
    }

    func testEndWithoutResumeStaysDownForNextStart() {
        var m = MicInterruption()
        _ = m.handle(.began, running: true)
        XCTAssertEqual(m.handle(.ended(shouldResume: false), running: true), .none)
        XCTAssertTrue(m.down)
        XCTAssertFalse(m.alive(running: true, engineRunning: true))
    }

    func testMediaServicesResetRebuildsEngine() {
        var m = MicInterruption()
        XCTAssertEqual(m.handle(.mediaServicesReset, running: true), .rebuildAndRestart)
        XCTAssertTrue(m.needsNewEngine)
        XCTAssertFalse(m.alive(running: true, engineRunning: true))
        m.restarted()
        XCTAssertFalse(m.needsNewEngine)
        XCTAssertTrue(m.alive(running: true, engineRunning: true))
    }

    func testResetDuringCallRebuildsWhenCallEnds() {
        var m = MicInterruption()
        _ = m.handle(.began, running: true)
        _ = m.handle(.mediaServicesReset, running: true)
        XCTAssertEqual(m.handle(.ended(shouldResume: true), running: true), .rebuildAndRestart)
    }

    func testSessionOffIgnoresEventsButRemembersReset() {
        var m = MicInterruption()
        XCTAssertEqual(m.handle(.began, running: false), .none)
        XCTAssertEqual(m.handle(.ended(shouldResume: true), running: false), .none)
        XCTAssertEqual(m.handle(.mediaServicesReset, running: false), .none)
        XCTAssertTrue(m.needsNewEngine, "the next start must not reuse the invalidated engine")
        XCTAssertFalse(m.down)
    }

    func testStoppedEngineIsNotAlive() {
        let m = MicInterruption()
        XCTAssertFalse(m.alive(running: true, engineRunning: false), "iOS stopped the engine: not alive")
        XCTAssertFalse(m.alive(running: false, engineRunning: true))
    }
}
