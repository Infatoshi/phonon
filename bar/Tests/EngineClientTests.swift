import XCTest

@testable import PhononBar

/// The engine helper dying under the bar (a crash, a kill, a quit in progress) must cost a
/// restart, never the app: SIGPIPE is ignored, writes to the dead pipe are errors, the exit is
/// reported once, and a restarted engine works.
final class EngineClientTests: XCTestCase {
    private var logURL: URL!

    override func setUp() {
        super.setUp()
        PhononBarMain.ignoreBrokenPipes()
        logURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("phonon-engine-test-\(UUID().uuidString).log")
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: logURL)
        super.tearDown()
    }

    /// `/bin/cat` stands in for the engine: every line sent comes back as an event.
    private func echoClient() -> EngineClient {
        let client = EngineClient()
        client.executable = "/bin/cat"
        client.arguments = []
        client.logURL = logURL
        return client
    }

    func testSigpipeIsIgnored() {
        var current = sigaction()
        sigaction(SIGPIPE, nil, &current)
        XCTAssertEqual(unsafeBitCast(current.__sigaction_u.__sa_handler, to: Int.self), unsafeBitCast(SIG_IGN, to: Int.self))
    }

    func testKilledEngineIsReportedOnceAndARestartWorks() throws {
        let client = echoClient()
        var received: [[String: Any]] = []
        var exits = 0
        var echoed = expectation(description: "echo before the kill")
        client.onEvent = { event in
            received.append(event)
            echoed.fulfill()
        }
        let exited = expectation(description: "exit reported")
        client.onExit = {
            exits += 1
            exited.fulfill()
        }
        try client.start()
        client.send(["type": "first"])
        wait(for: [echoed], timeout: 5)

        // kill the engine mid-session, then keep writing as the bar would
        let pid = try XCTUnwrap(client.processIdentifier)
        kill(pid, SIGKILL)
        for index in 0..<50 {
            client.send(["type": "after-kill", "index": index, "pad": String(repeating: "x", count: 4096)])
            usleep(1000)
        }
        wait(for: [exited], timeout: 5)
        XCTAssertEqual(exits, 1)

        // recovery: a restarted engine answers again
        echoed = expectation(description: "echo after the restart")
        try client.start()
        client.send(["type": "second"])
        wait(for: [echoed], timeout: 5)
        XCTAssertEqual(received.last?["type"] as? String, "second")
        XCTAssertEqual(exits, 1)
        client.shutdown()
    }

    func testShutdownTwiceAndSendAfterShutdownAreSafe() throws {
        let client = echoClient()
        var exits = 0
        client.onExit = { exits += 1 }
        try client.start()
        client.send(["type": "hello"])
        client.shutdown()
        client.shutdown()  // the quit path and applicationWillTerminate both shut down
        client.send(["type": "late"])  // dropped, no write to the closed pipe
        RunLoop.main.run(until: Date().addingTimeInterval(0.3))
        XCTAssertEqual(exits, 0)  // a requested shutdown is not an engine loss
    }

    func testEngineStderrGoesToTheLog() throws {
        let client = EngineClient()
        client.executable = "/bin/sh"
        client.arguments = ["-c", "echo 'engine said goodbye' >&2"]
        client.logURL = logURL
        let exited = expectation(description: "exit")
        client.onExit = { exited.fulfill() }
        try client.start()
        wait(for: [exited], timeout: 5)
        let log = try String(contentsOf: logURL, encoding: .utf8)
        XCTAssertTrue(log.contains("=== engine started"))
        XCTAssertTrue(log.contains("engine said goodbye"))
    }
}
