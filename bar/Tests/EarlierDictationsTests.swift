import XCTest

@testable import PhononBar

/// A dictation started while the previous one is still processing: the previous one's text is
/// never dropped (it reaches the "not pasted" card) and, with history off, its WAV is deleted.
final class EarlierDictationsTests: XCTestCase {
    private func command(_ raw: String) -> [String: Any] { ["cmd": "polish", "text": raw] }

    func testSetAsideWhileTranscribingEndsOnTheCardAndDeletesTheAudio() {
        var earlier = EarlierDictations()
        XCTAssertEqual(
            earlier.add(
                id: "p1", stage: .transcribing, wavPath: "/tmp/p1.wav", polishSent: false, engineReady: true,
                polishCommand: command), [])
        XCTAssertNil(earlier.transcript(id: "p2", text: "the new pass", keepAudio: false), "not ours")
        XCTAssertEqual(
            earlier.transcript(id: "p1", text: "hello there", keepAudio: false),
            [.polish(raw: "hello there", id: "p1")])
        XCTAssertEqual(
            earlier.corrected(id: "p1", text: "Hello there.", keepAudio: false),
            [.show(raw: "hello there", candidate: "Hello there."), .deleteWav("/tmp/p1.wav")])
        XCTAssertTrue(earlier.isEmpty)
    }

    func testHistoryOnKeepsTheAudio() {
        var earlier = EarlierDictations()
        _ = earlier.add(
            id: "p1", stage: .polishing(raw: "hi"), wavPath: "/tmp/p1.wav", polishSent: true, engineReady: true,
            polishCommand: command)
        XCTAssertEqual(earlier.corrected(id: "p1", text: "Hi.", keepAudio: true), [.show(raw: "hi", candidate: "Hi.")])
    }

    func testAWavStillBeingWrittenIsTranscribedWhenItLands() {
        var earlier = EarlierDictations()
        _ = earlier.add(
            id: "p1", stage: .writingWav, wavPath: nil, polishSent: false, engineReady: true, polishCommand: command)
        XCTAssertEqual(
            earlier.wavWritten(id: "p1", path: "/tmp/p1.wav", engineReady: true),
            [.transcribe(path: "/tmp/p1.wav", id: "p1")])
        XCTAssertNil(earlier.wavWritten(id: "p9", path: "/tmp/p9.wav", engineReady: true))
        // too short for a WAV: the engine hears the dictation ended
        _ = earlier.add(
            id: "p2", stage: .writingWav, wavPath: nil, polishSent: false, engineReady: true, polishCommand: command)
        XCTAssertEqual(earlier.wavWritten(id: "p2", path: nil, engineReady: true), [.dictationEnd(id: "p2")])
    }

    func testACorrectionWaitingOnScreenContextIsSentAtOnce() {
        var earlier = EarlierDictations()
        let steps = earlier.add(
            id: "p1", stage: .polishing(raw: "ship it"), wavPath: "/tmp/p1.wav", polishSent: false,
            engineReady: true, polishCommand: command)
        XCTAssertEqual(steps, [.polish(raw: "ship it", id: "p1")])
        XCTAssertEqual(earlier.polishCommand(raw: "ship it", id: "p1")?["text"] as? String, "ship it")
    }

    func testEngineNotReadyQueuesAndARestartResendsInOrder() {
        var earlier = EarlierDictations()
        XCTAssertEqual(
            earlier.add(
                id: "p1", stage: .awaitingTranscribe, wavPath: "/tmp/p1.wav", polishSent: false,
                engineReady: false, polishCommand: command), [])
        _ = earlier.add(
            id: "p2", stage: .polishing(raw: "two"), wavPath: "/tmp/p2.wav", polishSent: true, engineReady: false,
            polishCommand: command)
        XCTAssertEqual(
            earlier.engineReady(), [.transcribe(path: "/tmp/p1.wav", id: "p1"), .polish(raw: "two", id: "p2")])
    }

    func testAnErrorStillShowsAKnownTranscript() {
        var earlier = EarlierDictations()
        _ = earlier.add(
            id: "p1", stage: .polishing(raw: "keep me"), wavPath: "/tmp/p1.wav", polishSent: true,
            engineReady: true, polishCommand: command)
        XCTAssertEqual(
            earlier.failed(id: "p1", keepAudio: false),
            [.show(raw: "keep me", candidate: "keep me"), .deleteWav("/tmp/p1.wav")])
        // an empty transcript has nothing to show, but its audio still goes
        _ = earlier.add(
            id: "p2", stage: .transcribing, wavPath: "/tmp/p2.wav", polishSent: false, engineReady: true,
            polishCommand: command)
        XCTAssertEqual(earlier.transcript(id: "p2", text: "", keepAudio: false), [.deleteWav("/tmp/p2.wav")])
    }

    func testASecondEarlierDictationIsAddedAfterTheFirstOnTheCard() {
        XCTAssertEqual(EarlierDictations.cardText(onCard: nil, adding: "One."), "One.")
        XCTAssertEqual(EarlierDictations.cardText(onCard: "One.", adding: "Two."), "One. Two.")
    }
}
