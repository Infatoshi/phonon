import AppKit
import XCTest

@testable import PhononBar

@MainActor
final class LiveTextTests: XCTestCase {
    private var now: TimeInterval = 1_000

    private func makeState() -> LiveTextState {
        LiveTextState(clock: { [unowned self] in self.now })
    }

    private func dot(_ frame: LiveTextFrame) -> (token: LivePlacedToken, lit: Double)? {
        for token in frame.tokens {
            if case .dot(let lit) = token.kind { return (token, lit) }
        }
        return nil
    }

    private func texts(_ frame: LiveTextFrame) -> [(String, LiveLook)] {
        frame.tokens.compactMap {
            if case .text(let text, let look) = $0.kind { return (text, look) }
            return nil
        }
    }

    func testLiveTextSettingsDefaultAndRoundTrip() throws {
        let empty = try JSONDecoder().decode(NativeSettings.self, from: Data("{}".utf8))
        XCTAssertFalse(empty.liveText, "off by default: the capsule's energy is the feedback")
        XCTAssertEqual(empty.liveTextStyle, .fade)
        XCTAssertFalse(NativeSettings().liveText)
        var settings = NativeSettings()
        settings.liveText = false
        settings.liveTextStyle = .blur
        let encoded = try JSONEncoder().encode(settings)
        let json = String(decoding: encoded, as: UTF8.self)
        XCTAssertTrue(json.contains("\"live_text\":false"))
        XCTAssertTrue(json.contains("\"live_text_style\":\"blur\""))
        XCTAssertEqual(try JSONDecoder().decode(NativeSettings.self, from: encoded), settings)
        let fromRust = try JSONDecoder().decode(
            NativeSettings.self,
            from: Data(#"{"schema_version":2,"live_text":true,"live_text_style":"blur"}"#.utf8))
        XCTAssertTrue(fromRust.liveText)
        XCTAssertEqual(fromRust.liveTextStyle, .blur)
        let unknown = try JSONDecoder().decode(
            NativeSettings.self, from: Data(#"{"live_text_style":"sparkle"}"#.utf8))
        XCTAssertEqual(unknown.liveTextStyle, .fade)
    }

    func testTailIsThePartialBeyondTheDraftsCommittedTranscript() {
        let words = LiveTextPolicy.words("schedule the texture build for thursday")
        XCTAssertEqual(LiveTextPolicy.tail(transcript: words, covered: 4), ["for", "thursday"])
        XCTAssertEqual(LiveTextPolicy.tail(transcript: words, covered: 0), words)
        XCTAssertEqual(LiveTextPolicy.tail(transcript: words, covered: 6), [])
        XCTAssertEqual(LiveTextPolicy.tail(transcript: words, covered: 9), [])  // partial lags
    }

    func testDiffStepsTellAnAppendFromARewrite() {
        typealias Step = LiveTextPolicy.Step
        XCTAssertEqual(
            LiveTextPolicy.steps(old: ["Schedule", "the"], new: ["Schedule", "the", "build"]),
            [Step.kept(old: 0, new: 0), .kept(old: 1, new: 1), .changed(old: 2..<2, new: 2..<3)])
        let old = LiveTextPolicy.words("build for Thursday at three. wait")
        let new = LiveTextPolicy.words("build for Friday wait")
        XCTAssertEqual(
            LiveTextPolicy.steps(old: old, new: new),
            [.kept(old: 0, new: 0), .kept(old: 1, new: 1), .changed(old: 2..<5, new: 2..<3),
             .kept(old: 5, new: 3)])
    }

    func testTheTailShowsAfterTheDraftAndTheDraftAbsorbsIt() {
        let state = makeState()
        state.setTranscript("so schedule the")
        XCTAssertEqual(state.tail, ["so", "schedule", "the"])
        XCTAssertFalse(state.isEmpty, "the card shows before the first draft")
        state.applyDraft("Schedule the", coveredWords: 3, complete: true)
        XCTAssertEqual(state.tail, [])
        state.setTranscript("so schedule the texture build")
        XCTAssertEqual(state.tail, ["texture", "build"])
        let frame = state.frame(at: now + 1)
        XCTAssertEqual(texts(frame).map(\.0), ["Schedule", "the", "texture", "build"])
        let tail = texts(frame).last!.1
        XCTAssertEqual(tail.opacity, 0.85, accuracy: 0.001)
        XCTAssertEqual(tail.color, .ink, "the tail is the words as heard, not a placeholder")
        XCTAssertEqual(tail.blur, 0)
        // Key up with the tail pending: the final text resolves it.
        state.listening = false
        state.applyFinal("Schedule the texture build.")
        XCTAssertEqual(state.tail, [])
        XCTAssertEqual(state.correctedText, "Schedule the texture build.")
    }

    func testNewWordsFadeAndRiseInWithin120ms() {
        let state = makeState()
        state.applyDraft("Schedule the", coveredWords: 3, complete: true)
        now += 1
        state.applyDraft("Schedule the texture build", coveredWords: 5, complete: true)
        let first = texts(state.frame(at: now))
        XCTAssertEqual(first[0].1.opacity, 1, "kept words do not animate again")
        XCTAssertEqual(first[2].1.opacity, LiveTextMotion.appearFrom, accuracy: 0.001)
        XCTAssertEqual(first[2].1.dy, LiveTextMotion.rise, accuracy: 0.001)
        let done = texts(state.frame(at: now + LiveTextMotion.appear))[2].1
        XCTAssertEqual(done.opacity, 1, accuracy: 0.001)
        XCTAssertEqual(done.dy, 0, accuracy: 0.001)
        XCTAssertNil(
            state.frame(at: now).tokens.first {
                if case .morph = $0.kind { return true }
                return false
            }, "a pure append never morphs")
    }

    func testHeardWordsFadeInOnceAndTheDraftDoesNotReplayIt() {
        // The words usually show first as the tail (as heard); that is where they fade in.
        let state = makeState()
        state.applyDraft("Schedule the", coveredWords: 2, complete: true)
        now += 1
        state.setTranscript("schedule the texture")
        let start = now
        XCTAssertEqual(
            texts(state.frame(at: start)).last!.1.opacity, 0.85 * LiveTextMotion.appearFrom,
            accuracy: 0.001)
        now += 0.05
        state.setTranscript("schedule the texture build")
        let mid = texts(state.frame(at: now))
        XCTAssertGreaterThan(mid[2].1.opacity, 0.85 * LiveTextMotion.appearFrom, "still fading")
        XCTAssertEqual(
            mid[3].1.opacity, 0.85 * LiveTextMotion.appearFrom, accuracy: 0.001,
            "a newer word starts its own fade")
        // The draft covers the tail: the words carry on from when they first showed.
        now = start + 1
        state.applyDraft("Schedule the texture build", coveredWords: 4, complete: true)
        let covered = texts(state.frame(at: now))
        XCTAssertEqual(covered[2].1.opacity, 1, accuracy: 0.001)
        XCTAssertEqual(covered[2].1.dy, 0, accuracy: 0.001)
    }

    func testTheDotWakesWithTheVoiceAndLightsWithItsLevel() throws {
        let state = makeState()
        XCTAssertFalse(state.showsCard)
        XCTAssertTrue(state.noteVoice(), "the first voice wakes the card")
        XCTAssertFalse(state.noteVoice())
        XCTAssertTrue(state.showsCard, "the card shows before any word is recognized")
        let quiet = try XCTUnwrap(dot(state.frame(at: now)))
        XCTAssertEqual(quiet.lit, 0)
        for _ in 0..<4 { state.noteLevel(0.7) }
        let loud = try XCTUnwrap(dot(state.frame(at: now)))
        XCTAssertGreaterThan(loud.lit, 0.9, "speech lights it within a few buffers")
        for _ in 0..<4 { state.noteLevel(0.05) }
        let after = try XCTUnwrap(dot(state.frame(at: now))).lit
        XCTAssertLessThan(after, loud.lit)
        XCTAssertGreaterThan(after, 0, "it dims more slowly than it lights")
        // It sits after the last word and slides there when words come.
        state.applyDraft("Schedule the texture", coveredWords: 3, complete: true)
        let moving = try XCTUnwrap(dot(state.frame(at: now + 0.04))).token.x
        let settled = try XCTUnwrap(dot(state.frame(at: now + LiveTextMotion.dotMove))).token
        XCTAssertGreaterThan(moving, 0)
        XCTAssertLessThan(moving, settled.x)
        XCTAssertEqual(
            settled.x,
            LiveTextMetrics.measure("Schedule") + LiveTextMetrics.measure("the")
                + LiveTextMetrics.measure("texture") + 2 * LiveTextMetrics.spaceWidth
                + LiveTextMetrics.dotGap, accuracy: 0.5)
        // Key up: the dot goes; a card with no words goes with it.
        state.listening = false
        XCTAssertNil(dot(state.frame(at: now + 1)))
        let silent = makeState()
        silent.noteVoice()
        silent.listening = false
        XCTAssertFalse(silent.showsCard)
    }

    func testTheEndMarkShowsOnceTheSpeakerPauses() {
        let state = makeState()
        state.noteVoice()
        state.applyDraft("How snappy is this?", coveredWords: 4, complete: true)
        XCTAssertEqual(state.correctedText, "How snappy is this", "mid-speech the mark waits")
        now += LiveTextMotion.endPause + 0.1
        state.applyDraft("How snappy is this?", coveredWords: 4, complete: true)
        XCTAssertEqual(state.correctedText, "How snappy is this?", "a pause ends the sentence")
        state.noteVoice()
        now += 0.2
        state.applyDraft("How snappy is this on a scale", coveredWords: 7, complete: true)
        XCTAssertEqual(state.correctedText, "How snappy is this on a scale")
    }

    func testAtKeyUpTheCardShowsThePasteAtOnceAndFadesOut() {
        let state = makeState()
        state.applyDraft("Build for Thursday at three", coveredWords: 5, complete: true)
        state.setTranscript("build for thursday at three wait")
        now += 1
        state.listening = false
        state.applyFinal("Build for Friday at four.")
        XCTAssertEqual(state.shownWords, ["Build", "for", "Friday", "at", "four."])
        let landed = state.frame(at: now)
        XCTAssertEqual(texts(landed).map(\.1.opacity), [1, 1, 1, 1, 1], "no morph: it is typed")
        XCTAssertEqual(landed.opacity, 1)
        let fade = now + LiveTextMotion.hold
        XCTAssertEqual(state.frame(at: fade).opacity, 1, "it holds long enough to read the end")
        XCTAssertLessThan(state.frame(at: fade + 0.07).opacity, 1)
        XCTAssertEqual(state.frame(at: fade + LiveTextMotion.leave).opacity, 0, accuracy: 0.001)
    }

    func testKeyUpShowsTheEndMarkAtOnce() {
        // The owner lets go ~50 ms after the last word: no pause, so key up ends the sentence.
        let state = makeState()
        state.noteVoice()
        state.setTranscript("what else is messed up about this card that's hiding it?")
        state.applyDraft("What else is messed up about this card", coveredWords: 8, complete: true)
        XCTAssertEqual(state.shownWords.last, "it")
        state.listening = false
        state.showEndMark()
        XCTAssertEqual(state.shownWords.last, "it?")
        state.applyDraft("What else is messed up about this card that's hiding it?",
            coveredWords: 11, complete: true)
        state.showEndMark()
        XCTAssertEqual(state.correctedText, "What else is messed up about this card that's hiding it?")
    }

    func testPastesEndWithOneSpace() {
        XCTAssertEqual(Typer.spaced("Okay, Anvil is fixed."), "Okay, Anvil is fixed. ")
        XCTAssertEqual(Typer.spaced("ends in a space "), "ends in a space ")
        XCTAssertEqual(Typer.spaced("a list\n"), "a list\n")
        XCTAssertEqual(Typer.spaced(""), "")
    }

    func testARewriteMorphsThenGlowsAndSettles() throws {
        let state = makeState()
        state.applyDraft("Build for Thursday at three. wait", coveredWords: 6, complete: true)
        now += 1
        let start = now
        state.applyDraft("Build for Friday wait", coveredWords: 7, complete: true)
        let mid = state.frame(at: start + 0.1)
        let morph = try XCTUnwrap(
            mid.tokens.first {
                if case .morph = $0.kind { return true }
                return false
            })
        guard case .morph(let old, let oldLook, let new, let newLook) = morph.kind else {
            return XCTFail()
        }
        XCTAssertEqual(old, "Thursday at three.")
        XCTAssertEqual(new, "Friday")
        XCTAssertGreaterThan(oldLook.opacity, 0)
        XCTAssertLessThan(oldLook.opacity, 1)
        XCTAssertLessThan(oldLook.dy, 0, "the old words rise away")
        XCTAssertGreaterThan(newLook.dy, 0, "the new words rise in from below")
        let oldWidth = LiveTextMetrics.measure(old)
        let newWidth = LiveTextMetrics.measure(new)
        XCTAssertLessThan(morph.width, oldWidth)
        XCTAssertGreaterThan(morph.width, newWidth, "the width eases between them")
        if case .morph(_, let faded, _, _) = state.frame(at: start + 0.12).tokens.first(where: {
            $0.id == morph.id
        })?.kind {
            XCTAssertEqual(faded.opacity, 0, accuracy: 0.001, "old words gone by 60%")
        } else {
            XCTFail("still morphing at 120 ms")
        }
        // After the morph the new word flows as text, glowing, then settles to the ink.
        let after = texts(state.frame(at: start + 0.25))
        let friday = try XCTUnwrap(after.first { $0.0 == "Friday" })
        XCTAssertGreaterThan(friday.1.glow, 0)
        let settled = texts(state.frame(at: start + 0.42))
        XCTAssertEqual(settled.first { $0.0 == "Friday" }?.1.glow ?? 1, 0, accuracy: 0.001)
        XCTAssertEqual(settled.first { $0.0 == "Friday" }?.1.color, .ink)
        XCTAssertEqual(settled.first { $0.0 == "wait" }?.1.glow, 0, "kept words never glow")
    }

    func testTheCardOnlyChangesWhileTheUserSpeaks() {
        let state = makeState()
        state.noteVoice()
        state.setTranscript("build for thursday")
        state.applyDraft("Build for Thursday", coveredWords: 3, complete: true)
        XCTAssertEqual(state.correctedText, "Build for Thursday")
        // Silent for 2 s: a refine pass and a noisy partial both arrive; nothing moves.
        now += 2
        state.setTranscript("build for thursday um")
        state.applyDraft("Build for Tuesday", coveredWords: 4, complete: true)
        XCTAssertEqual(state.correctedText, "Build for Thursday")
        XCTAssertEqual(state.tail, [])
        // Speaking again: a real self-correction lands with the new words.
        state.noteVoice()
        now += 0.5
        state.setTranscript("build for thursday no friday")
        state.applyDraft("Build for Friday", coveredWords: 5, complete: true)
        XCTAssertEqual(state.correctedText, "Build for Friday")
        // The final text at key up shows whatever it is, voice or not.
        now += 5
        state.listening = false
        state.applyFinal("Build for Friday.")
        XCTAssertEqual(state.correctedText, "Build for Friday.")
    }

    func testCosmeticChangesFarBackWaitForTheFinalText() {
        let state = makeState()
        state.applyDraft("so the build runs at night", coveredWords: 6, complete: true)
        // The pass recases and re-punctuates words far back while appending new ones.
        state.applyDraft("So, the build runs at night. Then upload", coveredWords: 8, complete: true)
        XCTAssertEqual(
            state.correctedText, "so the build runs at night. Then upload",
            "far-back case and commas keep; the join with the new words updates")
        XCTAssertNil(
            state.frame(at: now).tokens.first {
                if case .morph = $0.kind { return true }
                return false
            }, "a punctuation change at the join swaps in place, no morph")
        XCTAssertEqual(
            LiveTextPolicy.stabilized(
                old: ["build", "for", "Thursday", "at", "three."],
                new: ["build", "for", "Friday", "at", "three."]),
            ["build", "for", "Friday", "at", "three."], "a real rewrite far back shows")
    }

    func testTailWordsTheDraftConfirmsDoNotFadeInAgain() {
        let state = makeState()
        state.setTranscript("schedule the texture build")
        state.applyDraft("Schedule the", coveredWords: 2, complete: true)
        now += 1
        state.applyDraft("Schedule the texture build", coveredWords: 4, complete: true)
        let looks = texts(state.frame(at: now))
        XCTAssertEqual(looks.map(\.0), ["Schedule", "the", "texture", "build"])
        XCTAssertEqual(looks[2].1.opacity, 1, "it was on the card already")
    }

    func testVoiceDetectorTellsSpeechFromRoomNoise() {
        var detector = VoiceDetector()
        // A fan at about -44 dBFS (0.2) for 2 s.
        for _ in 0..<100 { XCTAssertFalse(detector.isVoice(0.2)) }
        XCTAssertTrue(detector.isVoice(0.55), "speech at about -25 dBFS")
        var fresh = VoiceDetector()
        XCTAssertTrue(fresh.isVoice(0.5), "speech from the first instant")
    }

    func testTraceStoresWordDeltasAndTheGateReportsDroppedUpdates() {
        let (at, words) = LiveTextTrace.delta(
            ["send", "the", "report"], ["send", "the", "draft", "today"])
        XCTAssertEqual(at, 2)
        XCTAssertEqual(words, "draft today")
        var clock = now
        let state = LiveTextState(clock: { clock })
        state.noteVoice()
        XCTAssertTrue(state.setTranscript("send the report"))
        clock += LiveTextMotion.voiceWindow + 0.1
        XCTAssertFalse(state.setTranscript("send the report and"), "silent: dropped")
        XCTAssertEqual(state.shownWords, ["send", "the", "report"])
    }

    func testTheChunkPeriodWaitsForTheNextSentenceOrTheFinal() {
        let state = makeState()
        state.setTranscript("very cropped.")
        state.applyDraft("very cropped.", coveredWords: 2, complete: true)
        XCTAssertEqual(state.shownWords, ["very", "cropped"])
        state.applyDraft("very cropped. Then upload.", coveredWords: 4, complete: true)
        XCTAssertEqual(state.correctedWords, ["very", "cropped.", "Then", "upload"])
        XCTAssertEqual(LiveTextPolicy.openWords("it is..."), ["it", "is"])
        XCTAssertEqual(LiveTextPolicy.openWords("..."), ["..."])
        state.applyFinal("Very cropped. Then upload.")
        XCTAssertEqual(state.correctedWords.last, "upload.")
        XCTAssertEqual(LiveTextPolicy.openWords("the settings?"), ["the", "settings"])
        XCTAssertEqual(
            LiveTextPolicy.openWords("the settings?", keepEnd: true), ["the", "settings?"])
        XCTAssertEqual(LiveTextPolicy.openWords("it is...", keepEnd: true), ["it", "is"])
    }

    func testARewriteFarBackShowsOnlyOnceTwoDraftsAgree() {
        let state = makeState()
        let a = "Now for any of these test clips you can send them over"
        let b = "For any of these test clips you can send them over"
        state.applyDraft(a, coveredWords: 12, complete: true)
        // The refiner flips the sentence start draft to draft: the card keeps "Now for".
        state.applyDraft(b, coveredWords: 12, complete: true)
        state.applyDraft(a, coveredWords: 12, complete: true)
        state.applyDraft(b, coveredWords: 12, complete: true)
        XCTAssertEqual(state.correctedWords.first, "Now")
        // A rewrite two drafts in a row agree on shows.
        state.applyDraft(b, coveredWords: 12, complete: true)
        XCTAssertEqual(state.correctedWords.first, "For")
        // Near the line's end a rewrite shows at once.
        state.applyDraft(b.replacingOccurrences(of: "them over", with: "it over"), coveredWords: 12, complete: true)
        XCTAssertEqual(state.correctedWords.suffix(2), ["it", "over"])
    }

    func testWordsFarBackNeverChangeUntilKeyUp() {
        // Owner trace 2026-11-04 7:00 pm: "And if there's" -> "If there's", 48 words back.
        let state = makeState()
        let a = "And if there's anything else that we needed to refine last, go ask them before we finalize"
        let b = "If there's anything else that we needed to refine last, go ask them before we finalize"
        state.applyDraft(a, coveredWords: 17, complete: true)
        state.applyDraft(b, coveredWords: 17, complete: true)
        state.applyDraft(b, coveredWords: 17, complete: true)
        XCTAssertEqual(state.correctedWords.first, "And", "17 words back: it waits")
        state.listening = false
        state.applyFinal(b)
        XCTAssertEqual(state.correctedWords.first, "If")
    }

    func testARewriteNeverPutsBackWhatItReplaced() {
        // Owner trace 2026-11-04 7:00 pm: at the line's end the refiner went "and overall just
        // like" / "Overall, just" / "and overall just like" / "Overall, just".
        let state = makeState()
        state.applyDraft("looks like and overall just like", coveredWords: 6, complete: true)
        state.applyDraft("looks like Overall, just", coveredWords: 6, complete: true)
        XCTAssertEqual(state.correctedText, "looks like Overall, just")
        state.applyDraft("looks like and overall just like", coveredWords: 6, complete: true)
        XCTAssertEqual(state.correctedText, "looks like Overall, just", "no flip back")
        // new words still land
        state.applyDraft("looks like Overall, just make the decision", coveredWords: 9, complete: true)
        XCTAssertEqual(state.correctedText, "looks like Overall, just make the decision")
    }

    func testWordsAfterARepunctuatedWordShow() {
        // Owner trace 2026-11-04 4:13 pm: the card read "...meaningful here" for one draft.
        let state = makeState()
        state.setTranscript("Is the PPO network and style meaningful here?")
        state.applyDraft("Is the PPO network and style meaningful here?", coveredWords: 8, complete: true)
        state.setTranscript("Is the PPO network and style meaningful here, or can you just?")
        state.applyDraft(
            "Is the PPO network and style meaningful here, or can you just?", coveredWords: 12,
            complete: true)
        XCTAssertEqual(state.shownWords.suffix(4), ["or", "can", "you", "just"])
    }

    func testTheTailFollowsTheDraftsSourceByItsWords() {
        // Owner trace 2026-11-04 4:16 pm: the partial gained "Um" and "k" earlier on, and a
        // count put the line's end on the card twice.
        let source = LiveTextPolicy.words("We can think of the most ambitious style, just like everything fused into one.")
        let partial = LiveTextPolicy.words(
            "Um we can think of like the most ambitious k style, just like everything fused into one. And then")
        XCTAssertEqual(LiveTextPolicy.tail(transcript: partial, source: source), ["And", "then"])
        XCTAssertEqual(
            LiveTextPolicy.tail(transcript: partial, covered: source.count),
            ["fused", "into", "one.", "And", "then"], "by count the end shows twice")
        // a revised last word still counts as covered
        let revised = LiveTextPolicy.words("We can think of the most ambitious style, just like everything fused into once")
        XCTAssertEqual(LiveTextPolicy.tail(transcript: revised + ["more"], source: source), ["more"])
    }

    func testAFarRewriteTheLastDraftDidNotHaveWaits() {
        // Owner trace 2026-11-04 4:07 pm, 10.4 s.
        let card = LiveTextPolicy.words(
            "How long did it take to converge on that design and why? And could we technically throw this at the MLP for this small quin? Actually, I'm thinking we could just"
        )
        let previous = LiveTextPolicy.openWords(
            "Okay, how long did it take to converge on that design and why? And could we technically throw this at the MLP for this small quin? Actually, I'm thinking."
        )
        let draft = LiveTextPolicy.openWords(
            "How long did it take to converge on that design and why? And could we technically throw this at the MLP for this small Qwen? Actually, I'm thinking we could just."
        )
        var changed = [LiveTextPolicy.Change]()
        let out = LiveTextPolicy.stabilized(
            old: card, new: draft, previous: previous, changed: &changed)
        XCTAssertTrue(out.contains("quin?"), out.joined(separator: " "))
    }

    func testADraftForARevisedTranscriptIsStale() {
        let state = makeState()
        state.setTranscript("I imagine you change it.")
        XCTAssertFalse(state.draftMatches(source: ["Yeah."]))
        XCTAssertTrue(state.draftMatches(source: ["I", "imagine", "you"]))
        XCTAssertTrue(state.draftMatches(source: ["i", "imagine", "you", "change", "it.", "now"]))
    }

    func testTheCardStartsOneLineGrowsToFiveThenScrolls() {
        let state = makeState()
        state.applyDraft("Schedule the texture build", coveredWords: 4, complete: true)
        XCTAssertEqual(state.frame(at: now).lines, 1)
        XCTAssertEqual(state.frame(at: now).cardHeight, LiveTextMetrics.cardHeight(lines: 1))
        let sentence = "Schedule the texture build for Friday at four and make sure the script runs."
        now += 1
        let start = now
        state.applyDraft(
            Array(repeating: sentence, count: 3).joined(separator: " "), coveredWords: 42,
            complete: true)
        let lines = state.frame(at: start + 1).lines
        XCTAssertTrue((2...LiveTextMetrics.maxLines).contains(lines), "\(lines) lines")
        let grown = LiveTextMetrics.cardHeight(lines: CGFloat(lines))
        let easing = state.frame(at: start + 0.05).cardHeight
        XCTAssertGreaterThan(easing, LiveTextMetrics.cardHeight(lines: 1))
        XCTAssertLessThan(easing, grown)
        XCTAssertEqual(state.frame(at: start + 0.14).cardHeight, grown, accuracy: 0.001)
        XCTAssertEqual(state.frame(at: start + 0.14).scroll, 0)
        now += 1
        state.applyDraft(
            Array(repeating: sentence, count: 8).joined(separator: " "), coveredWords: 112,
            complete: true)
        let long = state.frame(at: now + 1)
        XCTAssertGreaterThan(long.lines, LiveTextMetrics.maxLines)
        XCTAssertEqual(long.cardHeight, LiveTextMetrics.maxCardHeight, accuracy: 0.001)
        XCTAssertEqual(long.scroll, CGFloat(long.lines - LiveTextMetrics.maxLines), accuracy: 0.001)
        let caret = long.tokens.first { $0.id == -1 }
        XCTAssertEqual(caret?.line, long.lines - 1, "the newest line stays in view")
    }

    func testOverlayNeverTakesFocusAndSitsAboveTheCapsule() {
        let panel = LiveTextPanel()
        XCTAssertFalse(panel.canBecomeKey)
        XCTAssertFalse(panel.canBecomeMain)
        XCTAssertTrue(panel.ignoresMouseEvents)
        XCTAssertTrue(panel.styleMask.contains(.nonactivatingPanel))
        let screen = NSRect(x: 0, y: 0, width: 1512, height: 982)
        let capsule = PanelGeometry.restingFrame(
            screen: screen, size: NSSize(width: 80, height: LiveTextPanel.capsuleHeight))
        let frame = LiveTextPanel.frame(screen: screen)
        XCTAssertEqual(frame.midX, capsule.midX)
        XCTAssertEqual(frame.minY, capsule.maxY + LiveTextPanel.gap)
        XCTAssertGreaterThanOrEqual(
            frame.height, LiveTextPanel.topMargin + LiveTextMetrics.maxCardHeight,
            "room is reserved for the card to grow to five lines")
    }
}
