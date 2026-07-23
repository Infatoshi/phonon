import XCTest
import Carbon.HIToolbox

@testable import PhononBar

final class PreviewRevisionPolicyTests: XCTestCase {
    func testTyperUsesCommandPasteEvents() throws {
        let (down, up) = try XCTUnwrap(Typer.makePasteEvents())

        XCTAssertEqual(down.getIntegerValueField(.keyboardEventKeycode), Int64(kVK_ANSI_V))
        XCTAssertEqual(up.getIntegerValueField(.keyboardEventKeycode), Int64(kVK_ANSI_V))
        XCTAssertTrue(down.flags.contains(.maskCommand))
        XCTAssertTrue(up.flags.contains(.maskCommand))
    }

    func testFinalInsertionRejectsDictionaryEnvelopeAndRunawayExpansion() {
        XCTAssertEqual(
            PreviewRevisionPolicy.safeFinalText(
                source: "Can you hear me?",
                candidate: "<phonon_dictionary>\ncanonical_terms: CUDA, cuDNN"
            ),
            "Can you hear me?"
        )
        XCTAssertEqual(
            PreviewRevisionPolicy.safeFinalText(
                source: "Okay.",
                candidate: "This response expanded into far too many unrelated words and should never be inserted into the focused application."
            ),
            "Okay."
        )
    }
    func testShortcutModeFiltersOnlyTheDisabledTrigger() {
        XCTAssertTrue(ShortcutPolicy.allows(mode: "both", source: "right-option"))
        XCTAssertTrue(ShortcutPolicy.allows(mode: "both", source: "control-space"))
        XCTAssertTrue(ShortcutPolicy.allows(mode: "right_option", source: "right-option"))
        XCTAssertFalse(ShortcutPolicy.allows(mode: "right_option", source: "control-space"))
        XCTAssertFalse(ShortcutPolicy.allows(mode: "control_space", source: "right-option"))
        XCTAssertTrue(ShortcutPolicy.allows(mode: "control_space", source: "control-space"))
    }

    @MainActor
    func testNativeModelLoaderTracksStreamsAndMonotonicProgress() {
        let state = ModelStartupState()
        state.apply(
            name: "asr", state: "loading", progress: 0.8, detail: "weights loaded",
            loadMs: 1_200)
        state.apply(
            name: "asr", state: "loading", progress: 0.3, detail: "late stale update",
            loadMs: nil)
        state.apply(
            name: "fluid-1", state: "ready", progress: 1, detail: "demo passed",
            loadMs: 2_400)

        XCTAssertEqual(state.streams[0].progress, 0.8)
        XCTAssertEqual(state.streams[1].state, "Ready")
        XCTAssertEqual(state.progress, 0.6, accuracy: 0.0001)
        state.markReady()
        XCTAssertTrue(state.ready)
    }

    func testSingleInstanceLockRejectsASecondOwner() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(
            "phonon-single-instance-\(UUID().uuidString).lock")
        defer { try? FileManager.default.removeItem(at: url) }

        let first = try XCTUnwrap(SingleInstanceLock.acquire(at: url))
        withExtendedLifetime(first) {
            XCTAssertNil(SingleInstanceLock.acquire(at: url))
        }
    }

    func testBarSettingsDecodeStreamingMode() throws {
        let settings = try JSONDecoder().decode(
            BarSettings.self,
            from: Data(
                #"{"streaming":true,"screen_context":false,"microphone_priority":["USB Microphone","MacBook"]}"#.utf8)
        )

        XCTAssertEqual(settings.streaming, true)
        XCTAssertEqual(settings.screenContext, false)
        XCTAssertEqual(settings.microphonePriority, ["USB Microphone", "MacBook"])
    }

    func testMicrophonePriorityAvoidsDefaultBluetoothInput() throws {
        let devices = [
            AudioInputDevice(id: 1, name: "AirPods Pro", isDefault: true),
            AudioInputDevice(id: 2, name: "MacBook Pro Microphone", isDefault: false),
        ]

        let selected = MicrophonePriorityResolver.resolve(
            devices: devices, priorities: ["USB Microphone", "MacBook"])

        XCTAssertEqual(selected?.name, "MacBook Pro Microphone")
    }

    func testMicrophonePriorityPrefersExternalMicrophoneWhenConnected() throws {
        let devices = [
            AudioInputDevice(id: 1, name: "MacBook Pro Microphone", isDefault: true),
            AudioInputDevice(id: 2, name: "USB Microphone", isDefault: false),
        ]

        let selected = MicrophonePriorityResolver.resolve(
            devices: devices, priorities: ["USB Microphone", "MacBook"])

        XCTAssertEqual(selected?.name, "USB Microphone")
    }

    func testAcousticPreviewCannotCollapse() {
        let previous = "one two three four five six"

        XCTAssertEqual(
            PreviewRevisionPolicy.stableAcoustic(previous: previous, candidate: "one"),
            previous
        )
    }

    func testNewerAcousticRevisionWithSameLengthCanReplacePrevious() {
        XCTAssertEqual(
            PreviewRevisionPolicy.stableAcoustic(
                previous: "this is nearly write", candidate: "this is nearly right"),
            "this is nearly right"
        )
    }

    func testStalePolishCannotReplaceNewerHypothesis() {
        XCTAssertFalse(
            PreviewRevisionPolicy.canShowPolish(
                source: "hello",
                latestAcoustic: "hello this is the current hypothesis",
                polished: "Hello.",
                newerCandidateWaiting: true
            )
        )
    }

    func testCollapsingPolishIsRejected() {
        let source = "this transcript already contains several useful words"
        XCTAssertFalse(
            PreviewRevisionPolicy.canShowPolish(
                source: source,
                latestAcoustic: source,
                polished: "Words.",
                newerCandidateWaiting: false
            )
        )
    }

    func testPanelRestsAtBottomCenter() {
        let screen = NSRect(x: 100, y: 50, width: 1_200, height: 800)
        let size = NSSize(width: 360, height: 180)
        let frame = PanelGeometry.restingFrame(screen: screen, size: size)

        XCTAssertEqual(frame.midX, screen.midX)
        XCTAssertEqual(frame.minY, screen.minY + PanelGeometry.bottomInset)
    }

    func testIdleAndExpandedPanelsShareTheBottomEdge() {
        let screen = NSRect(x: 100, y: 50, width: 1_200, height: 800)
        let idle = PanelGeometry.restingFrame(
            screen: screen, size: NSSize(width: 40, height: 8))
        let expanded = PanelGeometry.restingFrame(
            screen: screen, size: NSSize(width: 360, height: 180))

        XCTAssertEqual(idle.minY, expanded.minY)
        XCTAssertEqual(idle.midX, expanded.midX)
    }

    func testHiddenPanelIsBelowScreen() {
        let screen = NSRect(x: 100, y: 50, width: 1_200, height: 800)
        let frame = PanelGeometry.hiddenFrame(
            screen: screen, size: NSSize(width: 360, height: 180))

        XCTAssertLessThanOrEqual(frame.maxY, screen.minY)
    }

    func testLiveFormatterHandlesPunctuationFillersAndDuplicateWords() {
        XCTAssertEqual(
            LiveTranscriptFormatter.format(
                "well I'm typing in real time period ah and and then it corrects"),
            "Well I'm typing in real time. And then it corrects"
        )
    }

    func testLiveFormatterDoesNotPerformSemanticRewriting() {
        XCTAssertEqual(
            LiveTranscriptFormatter.format("use vLLM with CUDA"),
            "Use vLLM with CUDA"
        )
    }

    func testE2ETraceSeparatesDictationFromReleaseLatency() throws {
        let trace = E2ETrace(
            passId: "p1", source: "right-option", keyDownEventNs: 1_000_000_000,
            keyDownCallbackNs: 1_000_100_000, keyDownHandledNs: 1_000_300_000)
        trace.recordingStartedNs = 1_002_000_000
        trace.keyUpEventNs = 6_000_000_000
        trace.keyUpHandledNs = 6_000_300_000
        trace.captureEndedNs = 6_001_000_000
        trace.wavReadyNs = 6_004_000_000
        trace.asrSubmittedNs = 6_004_000_000
        trace.asrResultNs = 6_084_000_000
        trace.polishSubmittedNs = 6_084_000_000
        trace.polishResultNs = 6_184_000_000
        trace.insertionStartedNs = 6_184_000_000
        trace.insertionCompletedNs = 6_185_000_000

        let record = try XCTUnwrap(
            trace.finish(textCharacters: 12, insertionSucceeded: true))
        XCTAssertEqual(record.dictationMs, 5_000, accuracy: 0.001)
        XCTAssertEqual(record.keyUpToInsertMs, 185, accuracy: 0.001)
        XCTAssertEqual(record.keyDownToRecordingMs, 2, accuracy: 0.001)
    }
}
