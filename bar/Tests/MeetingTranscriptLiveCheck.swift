import XCTest

@testable import PhononBar

/// End-to-end check of meeting transcription on a real recorded meeting, opt in only:
///
///     PHONON_MEETING_CHECK_DIR=<Meetings/<id>> PHONON_MEETING_CHECK_OUT=<empty folder> \
///         swift test --filter MeetingTranscriptLiveCheck
///
/// Runs `MeetingTranscriber` against the repo's ASR sidecar (sidecar/asr_server.py, the pinned
/// model from the local caches, no network) with this process standing in for the engine's
/// forwarding. Writes transcript.json under the out folder, never into the meeting folder, and
/// prints counts and timings only.
final class MeetingTranscriptLiveCheck: XCTestCase {
    @MainActor
    func testTranscribeARecordedMeeting() throws {
        let env = ProcessInfo.processInfo.environment
        guard let meetingPath = env["PHONON_MEETING_CHECK_DIR"], let outPath = env["PHONON_MEETING_CHECK_OUT"]
        else { throw XCTSkip("set PHONON_MEETING_CHECK_DIR and PHONON_MEETING_CHECK_OUT to run") }
        let meeting = URL(fileURLWithPath: meetingPath, isDirectory: true)
        let out = URL(fileURLWithPath: outPath, isDirectory: true)
        XCTAssertNotEqual(meeting.standardizedFileURL.path, out.standardizedFileURL.path)
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()

        let sidecar = Process()
        sidecar.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        sidecar.arguments = [
            "uv", "run", "--offline", "--python", "3.12", "--with", "parakeet-mlx==0.5.2", "python",
            root.appendingPathComponent("sidecar/asr_server.py").path,
            "--model", "mlx-community/parakeet-tdt-0.6b-v2",
            "--revision", "8ae155301e23d820d82aa60d24817c900e69e487",
        ]
        var environment = env
        environment["UV_OFFLINE"] = "1"
        environment["HF_HUB_OFFLINE"] = "1"
        environment["PHONON_METAL_ASR"] = env["PHONON_METAL_ASR"] ?? "0"
        sidecar.environment = environment
        sidecar.currentDirectoryURL = root
        let input = Pipe()
        let output = Pipe()
        sidecar.standardInput = input
        sidecar.standardOutput = output
        sidecar.standardError = FileHandle.nullDevice
        try sidecar.run()
        defer {
            sidecar.terminate()
            sidecar.waitUntilExit()
        }

        let transcriber = MeetingTranscriber()
        transcriber.outputDirectory = { out.appendingPathComponent($0.lastPathComponent) }
        var finished = false
        var failed: String?
        var spans = 0
        var computeSeconds = 0.0
        var maxSpan = 0.0
        var audioSeconds = 0.0
        var ready = false
        var loadStarted = Date()
        transcriber.onFinished = { _ in finished = true }
        transcriber.onJobsChanged = { jobs in
            if case .failed(let message) = jobs[meeting.lastPathComponent] { failed = message }
        }
        // The engine's part: forward each span as transcribe_span, answer as meeting_span.
        transcriber.send = { command in
            var request = command
            request["cmd"] = "transcribe_span"
            if let data = try? JSONSerialization.data(withJSONObject: request) {
                input.fileHandleForWriting.write(data + Data("\n".utf8))
            }
        }
        var buffer = Data()
        output.fileHandleForReading.readabilityHandler = { handle in
            let chunk = handle.availableData
            DispatchQueue.main.async {
                buffer.append(chunk)
                while let newline = buffer.firstIndex(of: 0x0A) {
                    let line = buffer[buffer.startIndex..<newline]
                    buffer.removeSubrange(buffer.startIndex...newline)
                    guard let event = try? JSONSerialization.jsonObject(with: line) as? [String: Any]
                    else { continue }
                    switch event["type"] as? String {
                    case "ready":
                        ready = true
                        print("live check: sidecar ready in \(String(format: "%.1f", Date().timeIntervalSince(loadStarted))) s")
                        loadStarted = Date()
                        transcriber.setEngineReady(true)
                    case "span":
                        spans += 1
                        let seconds = (event["seconds"] as? Double) ?? 0
                        maxSpan = max(maxSpan, seconds)
                        computeSeconds += seconds
                        audioSeconds += (event["audio_seconds"] as? Double) ?? 0
                        var answer = event
                        answer["type"] = "meeting_span"
                        transcriber.handle(answer)
                    case "error":
                        if let id = event["id"] as? String {
                            transcriber.handle(["type": "meeting_span", "id": id, "error": event["msg"] ?? "error"])
                        } else if !ready {
                            failed = event["msg"] as? String ?? "sidecar failed to start"
                        }
                    default:
                        break
                    }
                }
            }
        }
        transcriber.enqueue(directory: meeting)
        let deadline = Date().addingTimeInterval(900)
        while !finished, failed == nil, Date() < deadline, sidecar.isRunning || !ready {
            RunLoop.main.run(until: Date().addingTimeInterval(0.05))
        }
        output.fileHandleForReading.readabilityHandler = nil
        XCTAssertNil(failed)
        XCTAssertTrue(finished, "did not finish")
        let wall = Date().timeIntervalSince(loadStarted)
        let transcript = try XCTUnwrap(
            MeetingTranscript.load(from: out.appendingPathComponent(meeting.lastPathComponent)))
        print(
            "live check: \(transcript.turns.count) turns, \(transcript.wordCount) words, "
                + "\(transcript.turns.filter { $0.speaker == .me }.count) of them yours; "
                + "\(spans) spans, \(String(format: "%.0f", audioSeconds)) s of audio, model "
                + "\(String(format: "%.1f", computeSeconds)) s, wall \(String(format: "%.1f", wall)) s "
                + "after the model loaded; longest span "
                + "\(String(format: "%.2f", maxSpan)) s")
        XCTAssertGreaterThan(transcript.turns.count, 0)
    }
}
