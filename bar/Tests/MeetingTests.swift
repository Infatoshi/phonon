import CoreMedia
import XCTest

@testable import PhononBar

final class MeetingDetectorTests: XCTestCase {
    func testConferenceURLsFollowWispr() {
        func url(_ text: String) -> URL { URL(string: text)! }
        XCTAssertTrue(MeetingDetector.isConferenceURL(url("https://meet.google.com/abc-defg-hij")))
        XCTAssertTrue(MeetingDetector.isConferenceURL(url("https://us02web.zoom.us/j/123")))
        XCTAssertTrue(MeetingDetector.isConferenceURL(url("https://teams.microsoft.com/l/meetup-join/x")))
        XCTAssertTrue(MeetingDetector.isConferenceURL(url("https://app.slack.com/huddle/T1/C2")))
        XCTAssertFalse(MeetingDetector.isConferenceURL(url("https://app.slack.com/client/T1/C2")))
        XCTAssertTrue(MeetingDetector.isConferenceURL(url("https://discord.com/channels/1/2")))
        XCTAssertFalse(MeetingDetector.isConferenceURL(url("https://discord.com/app")))
        XCTAssertFalse(MeetingDetector.isConferenceURL(url("http://meet.google.com/abc")))
        XCTAssertFalse(MeetingDetector.isConferenceURL(url("https://notzoom.us.example.com/j/1")))
    }

    private func evidence(
        holders: [String], front: String? = nil, url: String? = nil,
        denylist: [String] = ContextPolicy.defaultDenylist
    ) -> MeetingDetector.Evidence {
        MeetingDetector.Evidence(
            microphoneHolders: holders, frontmostBundleID: front,
            frontmostURL: url.flatMap(URL.init(string:)), denylist: denylist)
    }

    func testACallAppHoldingTheMicIsAMeeting() {
        XCTAssertEqual(
            MeetingDetector.evaluate(evidence(holders: ["us.zoom.xos"])),
            .meeting(app: "Zoom", bundleID: "us.zoom.xos"))
        XCTAssertEqual(MeetingDetector.evaluate(evidence(holders: [])), .none, "Zoom open, no call")
    }

    func testAChatAppCountsOnlyWhenFrontAndHoldingTheMic() {
        let slack = "com.tinyspeck.slackmacgap"
        XCTAssertEqual(
            MeetingDetector.evaluate(evidence(holders: [slack], front: slack)),
            .meeting(app: "Slack", bundleID: slack))
        XCTAssertEqual(
            MeetingDetector.evaluate(evidence(holders: [slack], front: "com.apple.Notes")), .none)
    }

    func testABrowserTabOnAConferenceSiteHoldingTheMicIsAMeeting() {
        let chrome = "com.google.Chrome"
        XCTAssertEqual(
            MeetingDetector.evaluate(
                evidence(holders: [chrome], front: chrome, url: "https://meet.google.com/x-y-z")),
            .meeting(app: "meet.google.com", bundleID: chrome))
        XCTAssertEqual(
            MeetingDetector.evaluate(
                evidence(holders: [chrome], front: chrome, url: "https://github.com/x")),
            .none, "a browser using the mic elsewhere is not a meeting")
    }

    /// 2026-10-29: the user expected a Discord call to be offered. The private
    /// list stops reading, not asking: the call is offered and nothing about
    /// its window is recorded.
    func testPrivateCallAppsAreOfferedButNeverRead() {
        XCTAssertEqual(
            MeetingDetector.evaluate(evidence(holders: ["com.hnc.Discord"], front: "com.hnc.Discord")),
            .meeting(app: "Discord", bundleID: "com.hnc.Discord"))
        let chrome = "com.google.Chrome"
        XCTAssertEqual(
            MeetingDetector.evaluate(
                evidence(holders: [chrome], front: chrome, url: "https://discord.com/channels/1/2")),
            .meeting(app: "discord.com", bundleID: chrome))
        var titleReads = 0
        let info = MeetingWindowInfo.read(
            bundleID: "com.hnc.Discord", pid: 42, denylist: ContextPolicy.defaultDenylist,
            isBrowser: { _ in false },
            browserPage: { _ in (nil, nil) },
            appWindowTitle: { _ in
                titleReads += 1
                return "Discord | #general"
            })
        XCTAssertNil(info.title)
        XCTAssertNil(info.conferenceURL)
        XCTAssertEqual(titleReads, 0, "a private app's window is never read")
    }

    func testPromptsOncePerCallAndBacksOffAfterNotNow() {
        var policy = MeetingPromptPolicy()
        XCTAssertTrue(policy.shouldAsk(meeting: "Zoom", now: 0))
        XCTAssertFalse(policy.shouldAsk(meeting: "Zoom", now: 4), "same call")
        policy.declined("Zoom", now: 5)
        XCTAssertFalse(policy.shouldAsk(meeting: nil, now: 8))
        XCTAssertFalse(policy.shouldAsk(meeting: "Zoom", now: 60), "declined recently")
        XCTAssertTrue(policy.shouldAsk(meeting: "Zoom", now: 5 + MeetingPromptPolicy.quietAfterDecline))
        XCTAssertTrue(policy.shouldAsk(meeting: "FaceTime", now: 70))
    }

    /// Review item 3: nothing is read from an app before the private filter;
    /// the tab URL only from a non-private browser that holds the mic.
    func testTheTabURLIsReadOnlyFromABrowserInTheCall() {
        let chrome = "com.google.Chrome"
        let list = ContextPolicy.defaultDenylist
        XCTAssertTrue(
            MeetingDetector.shouldReadTabURL(holders: [chrome], frontmostBundleID: chrome, denylist: list))
        XCTAssertFalse(
            MeetingDetector.shouldReadTabURL(
                holders: ["us.zoom.xos"], frontmostBundleID: chrome, denylist: list),
            "the browser is not the one in the call")
        XCTAssertFalse(
            MeetingDetector.shouldReadTabURL(
                holders: ["org.whispersystems.signal-desktop"],
                frontmostBundleID: "org.whispersystems.signal-desktop", denylist: list),
            "a private app is never read")
        XCTAssertFalse(
            MeetingDetector.shouldReadTabURL(
                holders: ["com.apple.Notes"], frontmostBundleID: "com.apple.Notes", denylist: list),
            "not a browser")
        XCTAssertFalse(
            MeetingDetector.shouldReadTabURL(holders: [chrome], frontmostBundleID: chrome, denylist: [chrome]))
    }

    /// Review item 6: once recording, switching to an editor does not end a
    /// Slack huddle or a Meet tab; the call ends when its process lets go of
    /// the microphone.
    func testARecordingFollowsTheMicNotTheFrontApp() {
        let slack = "com.tinyspeck.slackmacgap"
        var policy = MeetingEndPolicy(now: 0)
        for second in stride(from: 4.0, through: 60, by: 4) {
            let inCall = MeetingDetector.stillInCall(recordingBundleID: slack, holders: [slack])
            XCTAssertFalse(policy.observe(stillInCall: inCall, now: second))
        }
        // The verdict with an editor in front would have been .none:
        XCTAssertEqual(
            MeetingDetector.evaluate(evidence(holders: [slack], front: "com.microsoft.VSCode")), .none)
        let after = MeetingDetector.stillInCall(recordingBundleID: slack, holders: [])
        XCTAssertFalse(policy.observe(stillInCall: after, now: 64))
        XCTAssertTrue(policy.observe(stillInCall: after, now: 64 + MeetingEndPolicy.grace))
    }

    /// Second review, item 4: the saved title came from the call app's
    /// window with no private check, so a Chrome window switched to a
    /// discord.com tab before the click wrote that tab's title to disk.
    func testWindowInfoFromAPrivateTabIsNotRecorded() {
        let chrome = "com.google.Chrome"
        let list = ContextPolicy.defaultDenylist
        let meet = MeetingDetector.recordableWindowInfo(
            title: "Meet - abc-defg-hij - Google Chrome",
            url: URL(string: "https://meet.google.com/abc-defg-hij"), bundleID: chrome,
            denylist: list)
        XCTAssertEqual(meet.title, "Meet - abc-defg-hij - Google Chrome")
        XCTAssertEqual(meet.conferenceURL, "https://meet.google.com/abc-defg-hij")
        let switched = MeetingDetector.recordableWindowInfo(
            title: "#general | Friends - Google Chrome",
            url: URL(string: "https://discord.com/channels/1/2"), bundleID: chrome, denylist: list)
        XCTAssertNil(switched.title, "the private tab's URL drops its title too")
        XCTAssertNil(switched.conferenceURL)
        let titled = MeetingDetector.recordableWindowInfo(
            title: "(2) Discord | #general - Google Chrome", url: nil, bundleID: chrome,
            denylist: list)
        XCTAssertNil(titled.title, "a title naming a private site is dropped")
        let zoom = MeetingDetector.recordableWindowInfo(
            title: "Zoom Meeting", url: nil, bundleID: "us.zoom.xos", denylist: list)
        XCTAssertEqual(zoom.title, "Zoom Meeting")
        XCTAssertNil(zoom.conferenceURL)
    }

    func testAMeetingEndsAfterTheGraceWithoutTheCall() {
        var policy = MeetingEndPolicy(now: 0)
        XCTAssertFalse(policy.observe(stillInCall: true, now: 30))
        XCTAssertFalse(policy.observe(stillInCall: false, now: 35))
        XCTAssertFalse(policy.observe(stillInCall: true, now: 38), "a blip does not end it")
        XCTAssertTrue(policy.observe(stillInCall: false, now: 38 + MeetingEndPolicy.grace))
    }
}

/// Fifth review, items 1 and 2: the page search runs only for browsers, and
/// a native call app keeps its window title from the app's focused window.
final class MeetingWindowInfoTests: XCTestCase {
    func testANativeCallAppKeepsItsTitleAndNeverSearchesForAPage() {
        var pageSearches = 0
        let info = MeetingWindowInfo.read(
            bundleID: "us.zoom.xos", pid: 42, denylist: ContextPolicy.defaultDenylist,
            isBrowser: { _ in false },
            browserPage: { _ in
                pageSearches += 1
                return (nil, nil)
            },
            appWindowTitle: { _ in "Zoom Meeting" })
        XCTAssertEqual(info.title, "Zoom Meeting")
        XCTAssertNil(info.conferenceURL)
        XCTAssertEqual(pageSearches, 0)
    }

    func testABrowserGivesTheTitleAndPageOfItsWindowAndFailsClosed() {
        var titleReads = 0
        let read = { (url: URL?, title: String?) in
            MeetingWindowInfo.read(
                bundleID: "com.google.Chrome", pid: 42, denylist: ContextPolicy.defaultDenylist,
                isBrowser: { _ in true }, browserPage: { _ in (url, title) },
                appWindowTitle: { _ in
                    titleReads += 1
                    return "never used"
                })
        }
        let meet = read(URL(string: "https://meet.google.com/abc-defg-hij"), "Meet - abc-defg-hij")
        XCTAssertEqual(meet.conferenceURL, "https://meet.google.com/abc-defg-hij")
        XCTAssertEqual(meet.title, "Meet - abc-defg-hij")
        let unknown = read(nil, "Meet - abc-defg-hij")
        XCTAssertNil(unknown.title, "a browser page that cannot be known records nothing")
        XCTAssertEqual(titleReads, 0)
        XCTAssertNil(
            MeetingWindowInfo.read(bundleID: "us.zoom.xos", pid: nil, denylist: []).title)
    }

    /// The window info arrives after recording started and is merged into
    /// metadata.json without losing chunks closed meanwhile.
    func testWindowInfoFilledInLaterKeepsTheChunks() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "phonon-meeting-info-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let index = try MeetingIndex(
            directory: directory,
            metadata: MeetingMetadata(
                id: "m", startedAt: "2026-10-29T18:00:00Z", startedHostNs: 1, appBundleID: nil,
                appName: nil, windowTitle: nil, conferenceURL: nil, micDevice: nil))
        index.chunkClosed(
            MeetingChunk(index: 0, file: "0000.wav", startSample: 0, sampleCount: 240_000),
            track: .mic)
        index.setWindowInfo(title: "Zoom Meeting", conferenceURL: nil)
        let saved = try JSONDecoder().decode(
            MeetingMetadata.self,
            from: Data(contentsOf: directory.appendingPathComponent("metadata.json")))
        XCTAssertEqual(saved.windowTitle, "Zoom Meeting")
        XCTAssertEqual(saved.micChunks.count, 1)
    }
}

/// Second review, item 7: a start failure of an old recorder must not stop
/// the current one.
final class MeetingFailureIdentityTests: XCTestCase {
    func testOnlyTheFailedRecorderIsStopped() {
        final class Recorder {}
        let old = Recorder()
        let current = Recorder()
        XCTAssertFalse(MeetingController.isSame(current, old))
        XCTAssertTrue(MeetingController.isSame(current, current))
        XCTAssertFalse(MeetingController.isSame(nil, old))
    }
}

/// Review item 5: a stop during start must win, and whatever finishes
/// starting afterwards is undone.
final class RecorderLifecycleTests: XCTestCase {
    func testAStopDuringStartWins() {
        var lifecycle = RecorderLifecycle()
        XCTAssertTrue(lifecycle.beginStart())
        XCTAssertTrue(lifecycle.mayContinue)
        XCTAssertTrue(lifecycle.stop(), "Stop pressed while ScreenCaptureKit starts")
        XCTAssertFalse(lifecycle.mayContinue)
        XCTAssertFalse(lifecycle.finishStart(), "the stream that just started must be stopped")
        XCTAssertFalse(lifecycle.stop(), "only one caller does the stopping")
        XCTAssertFalse(lifecycle.beginStart(), "a stopped recorder never restarts")
    }

    func testANormalRunStopsOnce() {
        var lifecycle = RecorderLifecycle()
        XCTAssertTrue(lifecycle.beginStart())
        XCTAssertTrue(lifecycle.finishStart())
        XCTAssertEqual(lifecycle.phase, .running)
        XCTAssertTrue(lifecycle.stop())
        XCTAssertFalse(lifecycle.stop())
    }

    /// System audio that stops by itself is restarted at most three times.
    func testSystemAudioRestartsAreBounded() {
        var budget = RestartBudget(limit: 3)
        XCTAssertEqual((0..<5).map { _ in budget.take() }, [true, true, true, false, false])
    }

    /// A device change brings the microphone back only while recording.
    func testTheInputRestartsOnlyWhileRecording() {
        var lifecycle = RecorderLifecycle()
        XCTAssertFalse(lifecycle.mayRestartInput)
        XCTAssertTrue(lifecycle.beginStart())
        XCTAssertTrue(lifecycle.mayRestartInput)
        XCTAssertTrue(lifecycle.finishStart())
        XCTAssertTrue(lifecycle.mayRestartInput)
        XCTAssertTrue(lifecycle.stop())
        XCTAssertFalse(lifecycle.mayRestartInput, "a change after the stop does nothing")
    }

    func testAStopBeforeStartMeansNothingStarts() {
        var lifecycle = RecorderLifecycle()
        XCTAssertTrue(lifecycle.stop())
        XCTAssertFalse(lifecycle.beginStart())
    }
}

final class MeetingRecordingFilesTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "phonon-meeting-\(UUID().uuidString)", isDirectory: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    func testChunksAreClosedWavsWithContiguousOffsets() throws {
        var closed = [MeetingChunk]()
        // 1 s chunks at 48 kHz keep the test small; production uses 5 s.
        let writer = try ChunkedWavWriter(
            directory: directory.appendingPathComponent("mic"), sampleRate: 48_000,
            chunkSeconds: 1, onChunkClosed: { closed.append($0) })
        let tone = (0..<30_000).map { Float(sin(Double($0) * 0.05)) * 0.5 }
        try writer.append(tone)
        try writer.append(tone)
        XCTAssertEqual(closed.count, 1, "48,000 samples close the first chunk")
        try writer.append(tone)
        try writer.finish()
        XCTAssertEqual(closed.map(\.index), [0, 1])
        XCTAssertEqual(closed.map(\.startSample), [0, 48_000])
        XCTAssertEqual(closed.map(\.sampleCount), [48_000, 42_000])
        XCTAssertEqual(closed.map(\.file), ["0000.wav", "0001.wav"])

        let data = try Data(contentsOf: directory.appendingPathComponent("mic/0001.wav"))
        XCTAssertEqual(data.count, 44 + 42_000 * 2)
        XCTAssertEqual(String(data: data[0..<4], encoding: .ascii), "RIFF")
        func u32(_ at: Int) -> UInt32 {
            data[at..<at + 4].enumerated().reduce(0) { $0 | UInt32($1.element) << (8 * $1.offset) }
        }
        func u16(_ at: Int) -> UInt16 { UInt16(data[at]) | UInt16(data[at + 1]) << 8 }
        XCTAssertEqual(u16(22), 1, "mono")
        XCTAssertEqual(u32(24), 48_000)
        XCTAssertEqual(u16(34), 16)
        XCTAssertEqual(u32(40), 84_000, "data size patched on close")
        XCTAssertEqual(u32(4), 36 + 84_000)
    }

    /// Second review, item 5: samples arriving after the stop (late SCK or
    /// mic buffers) reopened a chunk that nobody would ever close.
    func testSamplesAfterFinishAreDropped() throws {
        var closed = [MeetingChunk]()
        let folder = directory.appendingPathComponent("late")
        let writer = try ChunkedWavWriter(
            directory: folder, chunkSeconds: 1, onChunkClosed: { closed.append($0) })
        try writer.append([Float](repeating: 0.1, count: 1_000))
        try writer.finish()
        try writer.append([Float](repeating: 0.1, count: 1_000))
        try writer.finish()
        XCTAssertEqual(closed.count, 1)
        let files = try FileManager.default.contentsOfDirectory(atPath: folder.path)
        XCTAssertEqual(files, ["0000.wav"])
    }

    func testFinishWithNothingOpenWritesNothing() throws {
        var closed = 0
        let writer = try ChunkedWavWriter(
            directory: directory.appendingPathComponent("system"), chunkSeconds: 1,
            onChunkClosed: { _ in closed += 1 })
        try writer.finish()
        XCTAssertEqual(closed, 0)
        let files = try FileManager.default.contentsOfDirectory(
            atPath: directory.appendingPathComponent("system").path)
        XCTAssertTrue(files.isEmpty)
    }

    func testTheIndexIsRewrittenAsChunksCloseAndRecordsTheEnd() throws {
        let metadata = MeetingMetadata(
            id: "meeting_1", startedAt: "2026-10-29T18:00:00Z", startedHostNs: 42,
            appBundleID: "us.zoom.xos", appName: "zoom.us", windowTitle: "Zoom Meeting",
            conferenceURL: nil, micDevice: "MacBook Pro Microphone")
        let index = try MeetingIndex(directory: directory, metadata: metadata)
        let url = directory.appendingPathComponent("metadata.json")
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
        index.chunkClosed(
            MeetingChunk(index: 0, file: "0000.wav", startSample: 0, sampleCount: 2_880_000),
            track: .mic)
        index.chunkClosed(
            MeetingChunk(index: 0, file: "0000.wav", startSample: 0, sampleCount: 2_880_000),
            track: .system)
        index.end(reason: "call_ended")
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        XCTAssertEqual(object["sample_rate"] as? Int, 48_000)
        XCTAssertEqual(object["chunk_seconds"] as? Int, 5)
        XCTAssertEqual(object["system_source"] as? String, "screencapturekit")
        XCTAssertEqual((object["mic_chunks"] as? [[String: Any]])?.count, 1)
        XCTAssertEqual(
            (object["system_chunks"] as? [[String: Any]])?.first?["sample_count"] as? Int,
            2_880_000)
        XCTAssertEqual(object["end_reason"] as? String, "call_ended")
        XCTAssertNotNil(object["ended_at"])
        XCTAssertEqual(object["started_host_ns"] as? Int, 42)
    }

    /// The meeting lane: 5 s closed chunks, so at most 5 s of audio is left
    /// to transcribe when the meeting stops; the last partial chunk closes
    /// at the stop.
    func testProductionChunksAreFiveSecondsAndTheLastPartialOneClosesAtStop() throws {
        XCTAssertEqual(MeetingFormat.chunkSeconds, 5)
        var closed = [MeetingChunk]()
        let writer = try ChunkedWavWriter(
            directory: directory.appendingPathComponent("mic"),
            onChunkClosed: { closed.append($0) })
        XCTAssertEqual(writer.samplesPerChunk, 240_000)
        try writer.append([Float](repeating: 0.1, count: 500_000))
        XCTAssertEqual(closed.map(\.sampleCount), [240_000, 240_000])
        try writer.finish()
        XCTAssertEqual(closed.map(\.sampleCount), [240_000, 240_000, 20_000])
        XCTAssertEqual(closed.map(\.startSample), [0, 240_000, 480_000])
    }

    /// Both tracks on one clock: a click captured at the same host time in
    /// the mic and the system track lands on the same timeline sample once
    /// each track is shifted by its start offset, and summing works.
    func testTracksShareOneSampleClockAndSumAligned() throws {
        let origin: UInt64 = 5_000_000_000
        let metadata = MeetingMetadata(
            id: "meeting_2", startedAt: "2026-10-29T18:00:00Z", startedHostNs: origin,
            appBundleID: nil, appName: nil, windowTitle: nil, conferenceURL: nil, micDevice: nil)
        let index = try MeetingIndex(directory: directory, metadata: metadata)
        let mic = try ChunkedWavWriter(
            directory: directory.appendingPathComponent("mic"), chunkSeconds: 1,
            onChunkClosed: { index.chunkClosed($0, track: .mic) })
        let system = try ChunkedWavWriter(
            directory: directory.appendingPathComponent("system"), chunkSeconds: 1,
            onChunkClosed: { index.chunkClosed($0, track: .system) })
        let micStart = origin + 10_000_000  // 10 ms after the start: 480 samples
        let systemStart = origin + 250_000_000  // SCK starts later: 12,000 samples
        let click = origin + 1_500_000_000  // timeline sample 72,000
        func buffer(from start: UInt64, count: Int, clickAt: UInt64) -> [Float] {
            var samples = [Float](repeating: 0, count: count)
            let at = Int(HostClock.samples(from: start, to: clickAt, rate: 48_000))
            if samples.indices.contains(at) { samples[at] = 0.5 }
            return samples
        }
        // Mic in 4,800-sample buffers, system in 1,024-sample buffers, each
        // stamped with its capture time.
        var placed = 0
        while placed < 96_000 {
            let start = micStart + UInt64(HostClock.nanoseconds(samples: Int64(placed), rate: 48_000))
            try mic.append(buffer(from: start, count: 4_800, clickAt: click), hostNs: start)
            placed += 4_800
        }
        placed = 0
        while placed < 80_000 {
            let start =
                systemStart + UInt64(HostClock.nanoseconds(samples: Int64(placed), rate: 48_000))
            try system.append(buffer(from: start, count: 1_024, clickAt: click), hostNs: start)
            placed += 1_024
        }
        try mic.finish()
        try system.finish()

        let saved = try JSONDecoder().decode(
            MeetingMetadata.self,
            from: Data(contentsOf: directory.appendingPathComponent("metadata.json")))
        XCTAssertEqual(saved.micStartOffsetSamples, 480)
        XCTAssertEqual(saved.systemStartOffsetSamples, 12_000)
        XCTAssertEqual(saved.micFirstSampleHostNs, micStart)
        XCTAssertEqual(saved.systemFirstSampleHostNs, systemStart)
        XCTAssertEqual(saved.micChunks[1].startHostNs, micStart + 1_000_000_000)

        func track(_ name: String, _ chunks: [MeetingChunk]) throws -> [Int16] {
            try chunks.flatMap { chunk -> [Int16] in
                let data = try Data(
                    contentsOf: directory.appendingPathComponent("\(name)/\(chunk.file)"))
                return stride(from: 44, to: data.count, by: 2).map {
                    Int16(bitPattern: UInt16(data[$0]) | UInt16(data[$0 + 1]) << 8)
                }
            }
        }
        let micSamples = try track("mic", saved.micChunks)
        let systemSamples = try track("system", saved.systemChunks)
        var sum = [Int32](repeating: 0, count: 100_000)
        for (offset, samples) in [
            (Int(saved.micStartOffsetSamples ?? 0), micSamples),
            (Int(saved.systemStartOffsetSamples ?? 0), systemSamples),
        ] {
            for (n, value) in samples.enumerated() where offset + n < sum.count {
                sum[offset + n] += Int32(value)
            }
        }
        let peak = sum.indices.max { abs(sum[$0]) < abs(sum[$1]) }
        XCTAssertEqual(peak, 72_000)
        XCTAssertEqual(sum[72_000], 2 * Int32(ChunkedWavWriter.pcm16(0.5)))
    }

    /// A stretch with no buffers (no system sound, a stalled device) becomes
    /// silence so later audio keeps its place on the clock; an early buffer
    /// and jitter are written as is; nothing captured is dropped, zeroed or
    /// gated, however quiet.
    func testGapsBecomeSilenceAndCapturedSamplesAreNeverDroppedOrGated() throws {
        var closed = [MeetingChunk]()
        let writer = try ChunkedWavWriter(
            directory: directory.appendingPathComponent("system"), chunkSeconds: 10,
            onChunkClosed: { closed.append($0) })
        let t0: UInt64 = 1_000_000_000
        let quiet = [Float](repeating: 0.001, count: 1_000)  // about -60 dBFS
        try writer.append(quiet, hostNs: t0)
        // Contiguous within jitter (5 ms late): no silence added.
        try writer.append(quiet, hostNs: t0 + 20_833_333 + 5_000_000)
        XCTAssertEqual(writer.clock.silenceInserted, 0)
        // One second later than the 2,000 samples so far account for.
        try writer.append(quiet, hostNs: t0 + 1_000_000_000)
        XCTAssertEqual(writer.clock.silenceInserted, 46_000)
        // Early (the device clock runs fast): kept, not trimmed.
        try writer.append(quiet, hostNs: t0 + 1_000_000_000)
        // No timestamp: appended where the clock says.
        try writer.append(quiet, hostNs: nil)
        try writer.finish()
        XCTAssertEqual(closed.map(\.sampleCount), [Int64(46_000 + 5_000)])
        let data = try Data(contentsOf: directory.appendingPathComponent("system/0000.wav"))
        let samples = stride(from: 44, to: data.count, by: 2).map {
            Int16(bitPattern: UInt16(data[$0]) | UInt16(data[$0 + 1]) << 8)
        }
        let quietValue = ChunkedWavWriter.pcm16(0.001)
        XCTAssertGreaterThan(quietValue, 0)
        XCTAssertEqual(samples.filter { $0 == quietValue }.count, 5_000)
        XCTAssertEqual(Array(samples[2_000..<48_000]), [Int16](repeating: 0, count: 46_000))
        XCTAssertEqual(Array(samples[48_000..<51_000]), [Int16](repeating: quietValue, count: 3_000))
    }

    /// Fifth review, item 4: a jump too large to be silence re-anchors the
    /// clock and is recorded; a later stall is padded again.
    func testAnOverLimitJumpReanchorsAndLaterGapsArePadded() throws {
        let origin: UInt64 = 1_000_000_000
        let index = try MeetingIndex(
            directory: directory,
            metadata: MeetingMetadata(
                id: "m", startedAt: "2026-10-29T18:00:00Z", startedHostNs: origin,
                appBundleID: nil, appName: nil, windowTitle: nil, conferenceURL: nil,
                micDevice: nil))
        let writer = try ChunkedWavWriter(
            directory: directory.appendingPathComponent("system"), chunkSeconds: 10,
            onChunkClosed: { index.chunkClosed($0, track: .system) })
        let block = [Float](repeating: 0.1, count: 1_000)
        try writer.append(block, hostNs: origin)
        // Twenty minutes later by the stamps: a clock error, not silence.
        let jump: UInt64 = 1_200_000_000_000
        try writer.append(block, hostNs: origin + jump)
        XCTAssertEqual(writer.clock.silenceInserted, 0)
        XCTAssertEqual(writer.clock.discontinuities.map(\.sample), [1_000])
        // Then a one-second stall on the new anchor: padded.
        let afterJump = origin + jump + 20_833_333  // 1,000 samples later
        try writer.append(block, hostNs: afterJump + 1_000_000_000)
        XCTAssertEqual(writer.clock.silenceInserted, 48_000)
        try writer.finish()
        let saved = try JSONDecoder().decode(
            MeetingMetadata.self,
            from: Data(contentsOf: directory.appendingPathComponent("metadata.json")))
        XCTAssertEqual(saved.systemDiscontinuities?.map(\.sample), [1_000])
        XCTAssertEqual(saved.systemDiscontinuities?.first?.hostNs, origin + jump)
        XCTAssertEqual(saved.systemChunks.first?.discontinuities?.count, 1)
        XCTAssertEqual(saved.systemChunks.first?.sampleCount, 3_000 + 48_000)
    }

    func testHostClockConversionsAndPlausibility() {
        // Apple silicon: 24 MHz ticks, timebase 125/3.
        XCTAssertEqual(HostClock.ticksToNanoseconds(24_000_000, numer: 125, denom: 3), 1_000_000_000)
        XCTAssertEqual(
            HostClock.ticksToNanoseconds(UInt64(24_000_000) * 86_400 * 30, numer: 125, denom: 3),
            UInt64(1_000_000_000) * 86_400 * 30)
        XCTAssertEqual(HostClock.ticksToNanoseconds(7, numer: 1, denom: 1), 7)
        let time = CMTime(value: 3_000_000_000, timescale: 1_000_000_000)
        XCTAssertEqual(HostClock.nanoseconds(time), 3_000_000_000)
        XCTAssertNil(HostClock.nanoseconds(.invalid))
        let now: UInt64 = 100_000_000_000
        XCTAssertEqual(HostClock.plausible(now - 50_000_000, now: now), now - 50_000_000)
        XCTAssertNil(HostClock.plausible(1_000, now: now), "a stamp on another clock")
        XCTAssertNil(HostClock.plausible(nil, now: now))
        XCTAssertEqual(HostClock.samples(from: 2_000_000_000, to: 1_990_000_000, rate: 48_000), -480)
    }

    func testMeetingDetectionAsksByDefaultButNeverAutoStarts() throws {
        func decode(_ json: String) throws -> NativeSettings {
            try JSONDecoder().decode(NativeSettings.self, from: Data(json.utf8))
        }
        // Fresh install, and files without the key: detection on, auto-start off.
        XCTAssertTrue(NativeSettings().meetingCapture)
        XCTAssertFalse(NativeSettings().meetingAutoStart)
        for json in [#"{}"#, #"{"schema_version":2}"#, #"{"schema_version":3}"#] {
            XCTAssertTrue(try decode(json).meetingCapture, json)
            XCTAssertFalse(try decode(json).meetingAutoStart, json)
        }
        // Before schema 3 every save wrote the old default, so a stored false is
        // turned on once.
        let upgraded = try decode(
            #"{"schema_version":2,"meeting_capture":false,"meeting_auto_start":false}"#)
        XCTAssertTrue(upgraded.meetingCapture)
        XCTAssertFalse(upgraded.meetingAutoStart)
        // Auto-start enabled once, then detection switched off: the upgrade turns detection
        // on but auto-start off, so no call is recorded without asking.
        for schema in [1, 2] {
            let armed = try decode(
                #"{"schema_version":\#(schema),"meeting_capture":false,"meeting_auto_start":true}"#)
            XCTAssertTrue(armed.meetingCapture, "schema \(schema)")
            XCTAssertFalse(armed.meetingAutoStart, "schema \(schema): must ask before recording")
        }
        XCTAssertFalse(
            try decode(#"{"schema_version":2,"meeting_auto_start":true}"#).meetingAutoStart)
        // Detection already on: nothing is forced, auto-start keeps the owner's choice.
        XCTAssertTrue(
            try decode(#"{"schema_version":2,"meeting_capture":true,"meeting_auto_start":true}"#)
                .meetingAutoStart)
        XCTAssertTrue(
            try decode(#"{"schema_version":3,"meeting_capture":true,"meeting_auto_start":true}"#)
                .meetingAutoStart)
        // From schema 3 a false is the owner's choice and stays off.
        let chosen = try decode(#"{"schema_version":3,"meeting_capture":false}"#)
        XCTAssertFalse(chosen.meetingCapture)
        XCTAssertTrue(try decode(#"{"schema_version":3,"meeting_capture":true}"#).meetingCapture)
    }

    @MainActor
    func testTurningMeetingDetectionOffSurvivesARelaunch() throws {
        let support = FileManager.default.temporaryDirectory
            .appendingPathComponent("phonon-meeting-default-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: support) }
        try FileManager.default.createDirectory(at: support, withIntermediateDirectories: true)
        let settingsURL = support.appendingPathComponent("settings.json")
        try Data(#"{"schema_version":2,"meeting_capture":false}"#.utf8).write(to: settingsURL)

        let store = NativeAppStore(supportDirectory: support)
        XCTAssertTrue(store.settings.meetingCapture, "an upgrade turns detection on once")
        store.updateSettings { $0.meetingCapture = false }

        let relaunched = NativeAppStore(supportDirectory: support)
        XCTAssertFalse(relaunched.settings.meetingCapture, "the owner's off stays off")
        XCTAssertFalse(relaunched.settings.meetingAutoStart)
    }
}
