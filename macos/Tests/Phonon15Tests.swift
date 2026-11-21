import CoreGraphics
import XCTest

@testable import PhononBar

// MARK: - Screen vocabulary extraction

final class ScreenVocabularyExtractorTests: XCTestCase {
    private func extract(_ text: String, known: Set<String> = [], maxTerms: Int = 24) -> [String] {
        ScreenVocabularyExtractor.extract(from: text, known: known, maxTerms: maxTerms)
    }

    func testKeepsCamelAndMixedCase() {
        let terms = extract("we moved from FluidAudio to OpenCode on the iPhone and macOS builds")
        XCTAssertEqual(terms, ["FluidAudio", "OpenCode", "iPhone", "macOS"])
    }

    func testKeepsShortAcronymsDropsLongCapsRuns() {
        let terms = extract("the API uses CUDA with more VRAM and HBM; WARNING IMPORTANT")
        XCTAssertEqual(terms, ["API", "CUDA", "VRAM", "HBM"])
    }

    func testKeepsLetterDigitTokens() {
        let terms = extract("run LFM3 on an M4 or an RTX3090 box, v2 is old")
        XCTAssertEqual(terms, ["LFM3", "M4", "RTX3090"])
    }

    func testKeepsHyphenatedAndUnderscoredIdentifiers() {
        let terms = extract("see phonon-core and asr_terms, a well-known real-time path")
        XCTAssertEqual(terms, ["phonon-core", "asr_terms"])
    }

    func testMidSentenceCapitalsKeptSentenceStartsDropped() {
        let terms = extract("Yesterday we called Marta about Kafka. Tomorrow is fine! Maybe? Sure: Later\nAnother line with Zurich")
        XCTAssertEqual(terms, ["Marta", "Kafka", "Zurich"])
    }

    func testFirstWordOfTextIsSentenceStart() {
        XCTAssertEqual(extract("Hello there"), [])
    }

    func testDotInsideTokenIsNotSentenceEnd() {
        // `v1.2` keeps the sentence going, so `Phonon` after it is mid-sentence.
        XCTAssertEqual(extract("we shipped v1.2 Phonon today"), ["Phonon"])
    }

    func testUIChromeAndLogLabelsDropped() {
        let terms = extract(
            "a Recent Sidebar Settings Search Home File Edit View Window Help New Open Save Close Share More Today INFO DEBUG ERROR WARN TODO FIXME NOTE TRACE Groq"
        )
        XCTAssertEqual(terms, ["Groq"])
    }

    func testUnitSuffixedNumbersDropped() {
        let terms = extract("took 100ms at 1080p 30x 16gb 48khz 2.5GHz 60fps with CUDA")
        XCTAssertEqual(terms, ["CUDA"])
    }

    func testHexHashesDropped() {
        XCTAssertEqual(extract("commit aaa57e6 and 2a2f6ef and 0xdeadbeef then Rust"), ["Rust"])
    }

    func testOrdinalsAndTimesDropped() {
        XCTAssertEqual(extract("the 1st 2nd 3rd 4th 21st at 10:30 with Ollama"), ["Ollama"])
    }

    func testPlainLowercaseAndShortTokensDropped() {
        XCTAssertEqual(extract("just some plain words a b c ok"), [])
    }

    func testOverlongTokenDropped() {
        let long = "Super" + String(repeating: "Long", count: 8)  // 37 characters
        XCTAssertEqual(extract("the \(long) and FooBar"), ["FooBar"])
    }

    func testKnownTermsDroppedCaseInsensitively() {
        XCTAssertEqual(
            extract("FluidAudio and OpenCode and CUDA", known: ["fluidaudio", "CUDA"]),
            ["OpenCode"])
    }

    func testRanksByFrequencyThenFirstAppearance() {
        let terms = extract("try OpenCode with CUDA, then CUDA and Groq, and CUDA with Groq")
        XCTAssertEqual(terms, ["CUDA", "Groq", "OpenCode"])
    }

    func testDeduplicatesCaseInsensitivelyKeepingFirstForm() {
        XCTAssertEqual(extract("use OpenCode or OPENCODE or openCode"), ["OpenCode"])
    }

    func testCapsAtTwentyFourTerms() {
        let text = (0..<40).map { "Term\($0)X" }.joined(separator: " ")
        let terms = extract("x " + text)
        XCTAssertEqual(terms.count, 24)
        XCTAssertEqual(terms.first, "Term0X")
        XCTAssertEqual(ScreenVocabularyExtractor.defaultMaxTerms, 24)
        // the most frequent term wins a slot even when it first appears last
        let repeated = extract("x " + text + " Term39X Term39X")
        XCTAssertEqual(repeated.first, "Term39X")
        XCTAssertEqual(repeated.count, 24)
    }

    func testEmptyAndZeroLimit() {
        XCTAssertEqual(extract(""), [])
        XCTAssertEqual(extract("FooBar", maxTerms: 0), [])
    }
}

// MARK: - Screen vocabulary reader bounds and policy

final class ScreenVocabularyReaderTests: XCTestCase {
    func testBudgetConstants() {
        XCTAssertEqual(ScreenVocabularyBudget.maxElements, 1_500)
        XCTAssertEqual(ScreenVocabularyBudget.maxDepth, 40)
        XCTAssertEqual(ScreenVocabularyBudget.maxBytes, 32 * 1024)
        XCTAssertEqual(ScreenVocabularyBudget.deadlineSeconds, 0.15)
        XCTAssertEqual(ScreenVocabularyBudget.maxValueChars, 4_096)
    }

    func testSecureFieldsSkippedByRoleOrSubrole() {
        XCTAssertTrue(ScreenVocabularyBudget.isSecure(role: "AXSecureTextField", subrole: nil))
        XCTAssertTrue(ScreenVocabularyBudget.isSecure(role: "AXTextField", subrole: "AXSecureTextField"))
        XCTAssertFalse(ScreenVocabularyBudget.isSecure(role: "AXTextField", subrole: nil))
    }

    func testHugeValuesReadOnlyTheVisibleRange() {
        XCTAssertEqual(ScreenVocabularyBudget.valueRead(numberOfCharacters: nil), .full)
        XCTAssertEqual(ScreenVocabularyBudget.valueRead(numberOfCharacters: 4_096), .full)
        XCTAssertEqual(ScreenVocabularyBudget.valueRead(numberOfCharacters: 4_097), .visibleRange)
    }

    func testWalkStopsAtEachBound() {
        XCTAssertTrue(ScreenVocabularyBudget.mayContinue(elements: 0, bytes: 0, elapsed: 0))
        XCTAssertFalse(ScreenVocabularyBudget.mayContinue(elements: 1_500, bytes: 0, elapsed: 0))
        XCTAssertFalse(ScreenVocabularyBudget.mayContinue(elements: 0, bytes: 32 * 1024, elapsed: 0))
        XCTAssertFalse(ScreenVocabularyBudget.mayContinue(elements: 0, bytes: 0, elapsed: 0.15))
    }

    func testPolicyRefusesOwnPrivateAndUnknownApps() {
        let own: pid_t = 100
        XCTAssertFalse(
            ScreenVocabularyReader.allows(pid: own, bundleID: "com.apple.Notes", denylist: [], ownPID: own))
        XCTAssertFalse(
            ScreenVocabularyReader.allows(pid: 5, bundleID: "com.1password.1password", denylist: [], ownPID: own))
        XCTAssertFalse(
            ScreenVocabularyReader.allows(pid: 5, bundleID: "md.obsidian", denylist: ["md.obsidian"], ownPID: own))
        XCTAssertFalse(ScreenVocabularyReader.allows(pid: 5, bundleID: nil, denylist: [], ownPID: own))
        XCTAssertFalse(ScreenVocabularyReader.allows(pid: 0, bundleID: "com.apple.Notes", denylist: [], ownPID: own))
        XCTAssertTrue(
            ScreenVocabularyReader.allows(pid: 5, bundleID: "com.apple.Notes", denylist: [], ownPID: own))
    }

    func testReaderRefusesOwnProcess() {
        let own = ProcessInfo.processInfo.processIdentifier
        XCTAssertNil(
            ScreenVocabularyReader.readFocusedWindowText(
                pid: own, bundleID: "com.apple.Notes", denylist: [], ownPID: own))
    }
}

// MARK: - Screen vocabulary session

final class ScreenVocabularySessionTests: XCTestCase {
    func testCurrentReadAcceptedOnce() {
        var session = ScreenVocabularySession()
        let token = session.begin()
        XCTAssertTrue(session.isReading)
        XCTAssertTrue(session.accept(token))
        XCTAssertFalse(session.accept(token), "single-flight: one result per read")
        XCTAssertFalse(session.isReading)
    }

    func testLateReadFromCancelledDictationDropped() {
        var session = ScreenVocabularySession()
        let token = session.begin()
        session.cancel()
        XCTAssertFalse(session.accept(token))
    }

    func testLateReadFromSupersededDictationDropped() {
        var session = ScreenVocabularySession()
        let first = session.begin()
        let second = session.begin()
        XCTAssertFalse(session.accept(first))
        XCTAssertTrue(session.accept(second))
    }

    func testCancelThenNewDictationStillDropsOldRead() {
        var session = ScreenVocabularySession()
        let old = session.begin()
        session.cancel()
        let fresh = session.begin()
        XCTAssertFalse(session.accept(old))
        XCTAssertTrue(session.accept(fresh))
    }
}

// MARK: - Globe key (Karabiner-Elements key code 179)

final class GlobeKeyCodeTests: XCTestCase {
    func testBothGlobeCodesAreTheGlobeKey() {
        XCTAssertEqual(AppController.globeKeyEventCode, 179)
        XCTAssertTrue(AppController.isGlobeKeyCode(63))
        XCTAssertTrue(AppController.isGlobeKeyCode(179))
        XCTAssertFalse(AppController.isGlobeKeyCode(49))  // space
    }

    func testGlobeKeyDownIsNotAChordKey() {
        XCTAssertFalse(AppController.isChordKeyDown(keyCode: 179))
        XCTAssertFalse(AppController.isChordKeyDown(keyCode: 63))
        XCTAssertTrue(AppController.isChordKeyDown(keyCode: 0))  // A
    }

    func testKarabiner179WithoutFnFlagDoesNotReleaseAHeldGlobe() {
        // Globe still physically held: the session state says fn.
        XCTAssertTrue(
            AppController.globeHeld(
                keyCode: 179, eventFlags: [], sessionFlags: { .maskSecondaryFn }))
        // Really released: the session state agrees.
        XCTAssertFalse(
            AppController.globeHeld(keyCode: 179, eventFlags: [], sessionFlags: { [] }))
    }

    func testOtherKeyCodesKeepTheEventFlags() {
        XCTAssertTrue(
            AppController.globeHeld(keyCode: 63, eventFlags: .maskSecondaryFn, sessionFlags: { [] }))
        XCTAssertFalse(
            AppController.globeHeld(keyCode: 63, eventFlags: [], sessionFlags: { .maskSecondaryFn }))
        // fn plus another modifier (fn-arrow cluster) is never the hold key
        XCTAssertFalse(
            AppController.globeHeld(
                keyCode: 179, eventFlags: [.maskSecondaryFn, .maskShift],
                sessionFlags: { [.maskSecondaryFn, .maskShift] }))
    }

    func testHeldGlobeWith179RepressIsOwnKeyNotAnEnd() {
        var edges = HoldKeyEdges()
        XCTAssertEqual(edges.update(pressed: true, ownKey: AppController.isGlobeKeyCode(63)), .down)
        // Karabiner's 179 event while held: an own-key repress at most, never an `.up`.
        let held = AppController.globeHeld(
            keyCode: 179, eventFlags: [], sessionFlags: { .maskSecondaryFn })
        XCTAssertEqual(edges.update(pressed: held, ownKey: AppController.isGlobeKeyCode(179)), .repress)
        XCTAssertTrue(edges.isDown)
    }
}

// MARK: - Secure input

final class SecureInputStateTests: XCTestCase {
    func testKeyChordModesNeedTheFallback() {
        XCTAssertTrue(SecureInputState.usesKeyChord(mode: "control_space"))
        XCTAssertTrue(SecureInputState.usesKeyChord(mode: "fn_and_control_space"))
        XCTAssertFalse(SecureInputState.usesKeyChord(mode: "fn"))
        XCTAssertFalse(SecureInputState.usesKeyChord(mode: "right_option"))
    }

    func testOnOffRegistersAndUnregistersOnce() {
        var state = SecureInputState()
        var r = state.update(secureInput: true, mode: "control_space")
        XCTAssertEqual(r.transition, .register)
        XCTAssertTrue(r.secureInputChanged)
        r = state.update(secureInput: true, mode: "control_space")
        XCTAssertEqual(r.transition, .none, "no duplicate registration")
        XCTAssertFalse(r.secureInputChanged, "logged once")
        r = state.update(secureInput: false, mode: "control_space")
        XCTAssertEqual(r.transition, .unregister)
        XCTAssertTrue(r.secureInputChanged)
        r = state.update(secureInput: false, mode: "control_space")
        XCTAssertEqual(r.transition, .none)
        XCTAssertFalse(state.fallbackRegistered)
    }

    func testModifierOnlyModeNeverRegisters() {
        var state = SecureInputState()
        let r = state.update(secureInput: true, mode: "fn")
        XCTAssertEqual(r.transition, .none)
        XCTAssertTrue(r.secureInputChanged)
        XCTAssertTrue(state.isSecureInputEnabled)
        XCTAssertFalse(state.fallbackRegistered)
    }

    func testModeChangeWhileSecureReleasesAndRearms() {
        var state = SecureInputState()
        XCTAssertEqual(state.update(secureInput: true, mode: "control_space").transition, .register)
        XCTAssertEqual(state.update(secureInput: true, mode: "fn").transition, .unregister)
        XCTAssertEqual(state.update(secureInput: true, mode: "fn_and_control_space").transition, .register)
    }

    func testResetReleasesARegistration() {
        var state = SecureInputState()
        _ = state.update(secureInput: true, mode: "control_space")
        XCTAssertEqual(state.reset(), .unregister)
        XCTAssertEqual(state.reset(), .none)
    }

    @MainActor
    func testMonitorReportsTransitionsAndVerifies() {
        var secure = false
        let monitor = SecureInputMonitor(isEnabled: { secure })
        monitor.shortcutMode = { "control_space" }
        var transitions: [SecureInputTransition] = []
        var verifies = 0
        monitor.onTransition = { transitions.append($0) }
        monitor.onVerify = { verifies += 1 }
        monitor.poll()
        secure = true
        monitor.poll()
        monitor.poll()
        secure = false
        monitor.poll()
        XCTAssertEqual(transitions, [.register, .unregister])
        XCTAssertEqual(verifies, 2)
        monitor.stop()
        XCTAssertEqual(transitions, [.register, .unregister])
    }

    func testDeadKeyLabelerFallsBackCleanly() {
        // Headless runs may have no layout; when there is one, kVK_ANSI_A labels as a letter.
        if let label = DeadKeyLabeler.label(forKeyCode: 0) {
            XCTAssertFalse(label.isEmpty)
        }
        _ = DeadKeyLabeler.label(forKeyCode: 0xFFFF)  // must not crash
    }
}

// MARK: - Audio fixture

final class AudioFixtureTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "phonon15-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func wav(samples: [Int16], rate: Int, channels: Int) -> Data {
        var data = Data()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        let dataBytes = samples.count * 2
        data.append(contentsOf: Array("RIFF".utf8))
        u32(UInt32(36 + dataBytes))
        data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8))
        u32(16)
        u16(1)
        u16(UInt16(channels))
        u32(UInt32(rate))
        u32(UInt32(rate * channels * 2))
        u16(UInt16(channels * 2))
        u16(16)
        data.append(contentsOf: Array("LIST".utf8))  // an extra chunk to skip
        u32(4)
        data.append(contentsOf: Array("INFO".utf8))
        data.append(contentsOf: Array("data".utf8))
        u32(UInt32(dataBytes))
        for s in samples { withUnsafeBytes(of: s.littleEndian) { data.append(contentsOf: $0) } }
        return data
    }

    func testLoadsMono16kWav() throws {
        let url = directory.appendingPathComponent("mono.wav")
        try wav(samples: [0, 16_384, -16_384, 32_767], rate: 16_000, channels: 1).write(to: url)
        let samples = try AudioFixtureLoader.loadMono16kPCM(from: url)
        XCTAssertEqual(samples.count, 4)
        XCTAssertEqual(samples[1], 0.5, accuracy: 1e-4)
        XCTAssertEqual(samples[2], -0.5, accuracy: 1e-4)
    }

    func testDownmixesAndResamplesStereo48k() throws {
        // 0.1 s of stereo at 48 kHz: left 0.5, right -0.5 -> mono 0.
        var interleaved: [Int16] = []
        for _ in 0..<4_800 { interleaved += [16_384, -16_384] }
        let url = directory.appendingPathComponent("stereo.wav")
        try wav(samples: interleaved, rate: 48_000, channels: 2).write(to: url)
        let samples = try AudioFixtureLoader.loadMono16kPCM(from: url)
        XCTAssertEqual(samples.count, 1_600)
        XCTAssertTrue(samples.allSatisfy { abs($0) < 1e-4 })
    }

    func testRejectsNonWavAndNon16Bit() {
        XCTAssertThrowsError(try AudioFixtureLoader.decodeMono16kPCM(Data("nope".utf8)))
        var bad = wav(samples: [0, 0], rate: 16_000, channels: 1)
        bad[34] = 8  // bits per sample
        XCTAssertThrowsError(try AudioFixtureLoader.decodeMono16kPCM(bad))
    }

    func testStreamerDeliversTenMillisecondChunksThenFinishes() {
        let samples = (0..<1_000).map { Float($0) }
        let streamer = AudioFixtureStreamer(samples: samples, pace: 50)
        let finished = expectation(description: "finished")
        let lock = NSLock()
        var chunks: [[Float]] = []
        streamer.start(
            onChunk: { chunk in
                lock.lock()
                chunks.append(chunk)
                lock.unlock()
            },
            onFinished: { finished.fulfill() })
        wait(for: [finished], timeout: 5)
        lock.lock()
        defer { lock.unlock() }
        XCTAssertEqual(chunks.count, 7)  // 6 x 160 + 40
        XCTAssertTrue(chunks.dropLast().allSatisfy { $0.count == AudioFixtureStreamer.chunkFrames })
        XCTAssertEqual(chunks.last?.count, 40)
        XCTAssertEqual(chunks.flatMap { $0 }, samples)
        XCTAssertFalse(streamer.isRunning)
    }

    func testStreamerStopsCleanly() {
        let streamer = AudioFixtureStreamer(samples: [Float](repeating: 0, count: 16_000), pace: 1)
        let lock = NSLock()
        var delivered = 0
        var finished = false
        streamer.start(
            onChunk: { _ in
                lock.lock()
                delivered += 1
                lock.unlock()
            },
            onFinished: { finished = true })
        Thread.sleep(forTimeInterval: 0.05)
        streamer.stop()
        lock.lock()
        let atStop = delivered
        lock.unlock()
        Thread.sleep(forTimeInterval: 0.1)
        lock.lock()
        defer { lock.unlock() }
        XCTAssertFalse(streamer.isRunning)
        XCTAssertLessThanOrEqual(delivered, atStop + 1)
        XCTAssertLessThan(delivered, 100)
        XCTAssertFalse(finished)
    }

    func testConfigFromEnvironment() throws {
        let url = directory.appendingPathComponent("f.wav")
        try wav(samples: [0], rate: 16_000, channels: 1).write(to: url)
        var config = AudioFixtureConfig.from(environment: [
            "PHONON_AUDIO_FIXTURE": url.path, "PHONON_NO_PASTE": "1",
        ])
        XCTAssertEqual(config.fixtureURL?.path, url.path)
        XCTAssertTrue(config.noPaste)
        config = AudioFixtureConfig.from(environment: [
            "PHONON_AUDIO_FIXTURE": directory.appendingPathComponent("missing.wav").path,
            "PHONON_NO_PASTE": "TRUE",
        ])
        XCTAssertNil(config.fixtureURL)
        XCTAssertTrue(config.noPaste)
        config = AudioFixtureConfig.from(environment: [
            "PHONON_AUDIO_FIXTURE": directory.path, "PHONON_NO_PASTE": "0",
        ])
        XCTAssertNil(config.fixtureURL, "a directory is not a fixture")
        XCTAssertFalse(config.noPaste)
        XCTAssertEqual(AudioFixtureConfig.from(environment: [:]), AudioFixtureConfig(fixtureURL: nil, noPaste: false))
    }

    func testOutputWrittenAtomically() throws {
        let out = try AudioFixtureConfig.writeOutput("Hello, world.", directory: directory)
        XCTAssertEqual(out.lastPathComponent, "last_fixture_output.txt")
        XCTAssertEqual(try String(contentsOf: out, encoding: .utf8), "Hello, world.")
        _ = try AudioFixtureConfig.writeOutput("Second.", directory: directory)
        XCTAssertEqual(try String(contentsOf: out, encoding: .utf8), "Second.")
    }
}

// MARK: - PHONON_DATA_DIR

final class DataDirectoryOverrideTests: XCTestCase {
    private let base = URL(fileURLWithPath: "/tmp/AppSupport", isDirectory: true)

    func testDefaultIsPhononUnderApplicationSupport() {
        let url = PhononDataPaths.resolveSupportDirectory(environment: [:], defaultBase: base)
        XCTAssertEqual(url.path, "/tmp/AppSupport/Phonon")
    }

    func testOverrideWins() {
        let url = PhononDataPaths.resolveSupportDirectory(
            environment: ["PHONON_DATA_DIR": "/tmp/bench-data"], defaultBase: base)
        XCTAssertEqual(url.path, "/tmp/bench-data")
        XCTAssertTrue(PhononDataPaths.isOverridden(environment: ["PHONON_DATA_DIR": "/x"]))
    }

    func testEmptyOrBlankOverrideIgnored() {
        for value in ["", "   "] {
            let url = PhononDataPaths.resolveSupportDirectory(
                environment: ["PHONON_DATA_DIR": value], defaultBase: base)
            XCTAssertEqual(url.path, "/tmp/AppSupport/Phonon")
            XCTAssertFalse(PhononDataPaths.isOverridden(environment: ["PHONON_DATA_DIR": value]))
        }
    }

    func testTildeExpands() {
        let url = PhononDataPaths.resolveSupportDirectory(
            environment: ["PHONON_DATA_DIR": "~/phonon-bench"], defaultBase: base)
        XCTAssertEqual(url.path, (NSHomeDirectory() as NSString).appendingPathComponent("phonon-bench"))
    }
}
