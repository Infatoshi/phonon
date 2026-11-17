import XCTest

/// The keyboard's dictation flow, without the extension: a result belongs to one dictation (DictationToken and the
/// app's UtteranceGate), a dictation survives the keyboard going off screen, the side button is the X whenever the
/// bar is lit, and a start wish is not acted on by a late pong in the appearance that wrote it.
final class DictationFlowTests: XCTestCase {
    // MARK: dictation identity

    func testTokenRuleTakesOnlyTheResultOfTheDictationWaitedFor() {
        let a = DictationToken.make(), b = DictationToken.make()
        XCTAssertNotEqual(a, b)
        XCTAssertTrue(DictationToken.accepts(result: b, waitingFor: b))
        XCTAssertFalse(DictationToken.accepts(result: a, waitingFor: b), "a cancelled dictation's late result")
        XCTAssertFalse(DictationToken.accepts(result: nil, waitingFor: b), "a result with no token")
        XCTAssertFalse(DictationToken.accepts(result: "", waitingFor: b))
        XCTAssertFalse(DictationToken.accepts(result: a, waitingFor: nil), "not waiting for any dictation")
        XCTAssertFalse(DictationToken.accepts(result: "", waitingFor: ""), "an empty token names nothing")
    }

    /// A is cancelled while it transcribes, B starts: A's late result is neither delivered by the app nor taken by
    /// the keyboard, and B's real result is.
    func testCancelledDictationsLateResultNeverReachesTheNextOne() {
        var gate = UtteranceGate()
        let a = DictationToken.make(), b = DictationToken.make()
        let genA = gate.begin(token: a)
        XCTAssertTrue(gate.isCurrent(genA))
        gate.cancel()  // the X while A transcribes
        XCTAssertFalse(gate.isCurrent(genA), "A's transcription finishing now is dropped by the app")
        XCTAssertNil(gate.token)
        let genB = gate.begin(token: b)
        XCTAssertFalse(gate.isCurrent(genA), "still dropped once B is running")
        XCTAssertTrue(gate.isCurrent(genB))
        XCTAssertEqual(gate.token, b, "B's result goes back with B's token")
        // even if A's result had been written, the keyboard waiting for B would refuse it and take B's
        var waitingFor: String? = b
        XCTAssertFalse(DictationToken.accepts(result: a, waitingFor: waitingFor))
        XCTAssertTrue(DictationToken.accepts(result: gate.token, waitingFor: waitingFor))
        waitingFor = nil  // taken: a repeat of the same result is not inserted twice
        XCTAssertFalse(DictationToken.accepts(result: b, waitingFor: waitingFor))
    }

    func testANewStartSupersedesAnUncancelledUtteranceToo() {
        var gate = UtteranceGate()
        let first = gate.begin(token: "one")
        let second = gate.begin(token: "two")
        XCTAssertFalse(gate.isCurrent(first))
        XCTAssertTrue(gate.isCurrent(second))
    }

    // MARK: the keyboard going off screen

    func testKeyboardDisappearingMidDictationKeepsTheLight() {
        let model = KeyboardModel()
        model.ignite()
        model.phase = .listening
        model.offScreen(dictating: true)
        XCTAssertFalse(model.visible)
        XCTAssertEqual(model.beat.stage, .listening, "a dictation under way keeps its light")
        model.startProcessing()
        XCTAssertEqual(model.beat.stage, .processing, "the release still gathers")
        model.phase = .working
        model.fadeOut()
        XCTAssertEqual(model.beat.stage, .fadeOut, "and the paste still fades back to rest")

        let idle = KeyboardModel()
        idle.ignite()
        idle.fadeOut()
        idle.offScreen(dictating: false)
        XCTAssertEqual(idle.beat, .rest, "with nothing under way it rests")
    }

    func testProcessingStartsFromRestWhileListening() {
        let model = KeyboardModel()
        model.phase = .listening  // the beat was lost: still rest
        model.startProcessing()
        XCTAssertEqual(model.beat.stage, .processing)

        let ready = KeyboardModel()
        ready.phase = .ready
        ready.startProcessing()
        XCTAssertEqual(ready.beat, .rest, "no gather from nothing")

        let fading = KeyboardModel()
        fading.ignite()
        fading.fadeOut()
        fading.startProcessing()
        XCTAssertEqual(fading.beat.stage, .fadeOut, "a late stop does not relight an ended dictation")
    }

    func testIgniteClearsTheLastDictationsLevels() {
        let model = KeyboardModel()
        for _ in 0..<10 { model.push(level: 0.9) }
        XCTAssertGreaterThan(VoiceLevel.smooth(model.levels), 0.5)
        model.ignite()
        XCTAssertEqual(VoiceLevel.smooth(model.levels), 0)
        XCTAssertEqual(model.levels.count, 28)
    }

    // MARK: the side button

    func testSideControlIsTheXWhileTheBarIsLit() {
        // the 0.4 s between the press and the app's pong: phase still ready, the bar lit, a press pending
        XCTAssertEqual(TalkBarSideControl(phase: .ready, beat: .listening), .stop)
        XCTAssertEqual(TalkBarSideControl(phase: .ready, beat: .rest, pending: true), .stop)
        XCTAssertEqual(TalkBarSideControl(phase: .ready, beat: .processing), .stop)
        XCTAssertEqual(TalkBarSideControl(phase: .needsSession, beat: .listening, pending: true), .stop)
        // over: the fade shows the settings circle
        XCTAssertEqual(TalkBarSideControl(phase: .ready, beat: .fadeOut), .settings)
        XCTAssertEqual(TalkBarSideControl(phase: .ready), .settings)

        let model = KeyboardModel()
        model.phase = .ready
        XCTAssertEqual(model.sideControl, .settings)
        model.ignite()  // the press, before micDown has asked the app anything
        XCTAssertEqual(model.sideControl, .stop)
        model.pendingMic = true
        model.fadeOut()
        XCTAssertEqual(model.sideControl, .stop, "still waiting for the app")
        model.pendingMic = false
        XCTAssertEqual(model.sideControl, .settings)
    }

    // MARK: the start wish

    func testLatePongDoesNotStartTheWishInTheAppearanceThatWroteIt() {
        let wanted = 1_000.0
        // the 0.4 s timeout wrote the wish and opened Phonon; the app's pong lands at 0.45 s
        XCTAssertFalse(PendingStart.shouldResume(wantedAt: wanted, now: wanted + 0.05, alive: true,
                                                 wroteThisAppearance: true))
        // back from Phonon: the keyboard's next appearance starts it without a second tap
        XCTAssertTrue(PendingStart.shouldResume(wantedAt: wanted, now: wanted + 6, alive: true,
                                                wroteThisAppearance: false))
        // the other rules still hold
        XCTAssertFalse(PendingStart.shouldResume(wantedAt: wanted, now: wanted + 6, alive: false,
                                                 wroteThisAppearance: false))
        XCTAssertFalse(PendingStart.shouldResume(wantedAt: wanted, now: wanted + PendingStart.ttl + 1, alive: true,
                                                 wroteThisAppearance: false))
    }
}
