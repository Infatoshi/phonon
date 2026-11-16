import SwiftUI
import XCTest

@testable import PhononBar

@MainActor
final class NewMicrophoneTests: XCTestCase {
    private let macBook = AudioInputDevice(
        id: 1, name: "MacBook Pro Microphone", isDefault: true, uid: "BuiltInMicrophoneDevice")
    private let yeti = AudioInputDevice(
        id: 2, name: "Yeti Stereo Microphone", isDefault: false, uid: "AppleUSBAudioEngine:Yeti")
    private let airPods = AudioInputDevice(
        id: 3, name: "AirPods Pro", isDefault: true, isBluetooth: true, uid: "AA-BB-CC:input")
    private let zoom = AudioInputDevice(
        id: 4, name: "ZoomAudioDevice", isDefault: false, uid: "zoom.us.audio", isVirtual: true)

    // MARK: Seen set

    func testFirstRunSeedsEverythingPresentWithoutAsking() {
        let scan = NewMicrophonePolicy.scan(
            devices: [macBook, yeti, airPods], seen: nil, priorities: [])
        XCTAssertEqual(scan.seen, [macBook.uid, yeti.uid, airPods.uid])
        XCTAssertTrue(scan.newcomers.isEmpty)
    }

    func testSeenDevicesNeverAskAgain() {
        let scan = NewMicrophonePolicy.scan(
            devices: [macBook, yeti], seen: [macBook.uid, yeti.uid], priorities: [])
        XCTAssertEqual(scan.seen, [macBook.uid, yeti.uid])
        XCTAssertTrue(scan.newcomers.isEmpty)
    }

    func testUnseenDeviceIsAskedAndRecorded() {
        let scan = NewMicrophonePolicy.scan(
            devices: [macBook, yeti], seen: [macBook.uid], priorities: [])
        XCTAssertEqual(scan.newcomers.map(\.uid), [yeti.uid])
        XCTAssertEqual(scan.seen, [macBook.uid, yeti.uid])
    }

    func testUnpluggedDeviceStaysSeen() {
        let scan = NewMicrophonePolicy.scan(
            devices: [macBook], seen: [macBook.uid, yeti.uid], priorities: [])
        XCTAssertEqual(scan.seen, [macBook.uid, yeti.uid])
        XCTAssertTrue(scan.newcomers.isEmpty)
    }

    func testVirtualAndAlreadyRankedDevicesAreSeenWithoutAsking() {
        let scan = NewMicrophonePolicy.scan(
            devices: [macBook, yeti, zoom], seen: [macBook.uid], priorities: ["Yeti"])
        XCTAssertTrue(scan.newcomers.isEmpty)
        XCTAssertEqual(scan.seen, [macBook.uid, yeti.uid, zoom.uid])
    }

    func testDeviceWithoutUIDIsIgnored() {
        let nameless = AudioInputDevice(id: 9, name: "Mystery", isDefault: false)
        let scan = NewMicrophonePolicy.scan(devices: [nameless], seen: [], priorities: [])
        XCTAssertEqual(scan.seen, [])
        XCTAssertTrue(scan.newcomers.isEmpty)
    }

    // MARK: Question

    func testUSBMicQuestionWithNothingRankedNamesTheMicInUse() {
        let question = NewMicrophonePolicy.question(
            for: yeti, devices: [macBook, yeti], priorities: [])
        XCTAssertEqual(question.title, "New microphone: Yeti Stereo Microphone. Use it for dictation?")
        XCTAssertEqual(question.current, "MacBook Pro Microphone")
        XCTAssertTrue(question.offersBelow)
        XCTAssertEqual(question.recommended, .first)
        XCTAssertNil(question.dropped)
    }

    func testBluetoothHeadsetDefaultsBelowAndExplainsTheTradeoff() {
        let question = NewMicrophonePolicy.question(
            for: airPods, devices: [macBook, airPods], priorities: [])
        XCTAssertEqual(question.current, "MacBook Pro Microphone")
        XCTAssertEqual(question.recommended, .below)
        XCTAssertTrue(question.detail.contains("call quality"))
        XCTAssertTrue(question.detail.contains("keeps MacBook Pro Microphone first"))
    }

    func testCurrentIsTheFirstRankedName() {
        let question = NewMicrophonePolicy.question(
            for: yeti, devices: [macBook, yeti], priorities: ["Shure MV7", "MacBook"])
        XCTAssertEqual(question.current, "Shure MV7")
    }

    func testFullListOffersOnlyFirstAndNamesWhatDrops() {
        let question = NewMicrophonePolicy.question(
            for: airPods, devices: [macBook, airPods], priorities: ["A", "B", "C"])
        XCTAssertFalse(question.offersBelow)
        XCTAssertEqual(question.dropped, "C")
        XCTAssertEqual(question.recommended, .first)
        XCTAssertTrue(question.detail.contains("C leaves the list"))
    }

    func testOnlyMicrophoneHasNothingToGoBelow() {
        let question = NewMicrophonePolicy.question(for: yeti, devices: [yeti, zoom], priorities: [])
        XCTAssertNil(question.current)
        XCTAssertFalse(question.offersBelow)
    }

    // MARK: Apply

    func testUseFirstWithNothingRankedRanksOnlyTheNewDevice() {
        XCTAssertEqual(
            NewMicrophonePolicy.apply(
                .first, name: "Yeti Stereo Microphone", current: "MacBook Pro Microphone",
                priorities: []),
            ["Yeti Stereo Microphone"])
    }

    func testUseFirstGoesAboveTheCurrentFirst() {
        XCTAssertEqual(
            NewMicrophonePolicy.apply(
                .first, name: "Yeti", current: "MacBook", priorities: ["MacBook", "AirPods"]),
            ["Yeti", "MacBook", "AirPods"])
    }

    func testUseFirstMovesAnExistingEntryUpAndTrimsAFullList() {
        XCTAssertEqual(
            NewMicrophonePolicy.apply(.first, name: "c", current: "A", priorities: ["A", "B", "C"]),
            ["c", "A", "B"])
        XCTAssertEqual(
            NewMicrophonePolicy.apply(.first, name: "D", current: "A", priorities: ["A", "B", "C"]),
            ["D", "A", "B"])
    }

    func testAddBelowWithNothingRankedKeepsTheCurrentMicFirst() {
        XCTAssertEqual(
            NewMicrophonePolicy.apply(
                .below, name: "AirPods Pro", current: "MacBook Pro Microphone", priorities: []),
            ["MacBook Pro Microphone", "AirPods Pro"])
    }

    func testAddBelowAppendsAndLeavesAFullListAlone() {
        XCTAssertEqual(
            NewMicrophonePolicy.apply(.below, name: "Yeti", current: "A", priorities: ["A"]),
            ["A", "Yeti"])
        XCTAssertEqual(
            NewMicrophonePolicy.apply(.below, name: "Yeti", current: "A", priorities: ["A", "B", "C"]),
            ["A", "B", "C"])
        XCTAssertEqual(
            NewMicrophonePolicy.apply(.below, name: "yeti", current: "A", priorities: ["A", "Yeti"]),
            ["A", "Yeti"])
    }

    func testIgnoreLeavesTheRankingAlone() {
        XCTAssertEqual(
            NewMicrophonePolicy.apply(.ignore, name: "Yeti", current: "A", priorities: ["A"]), ["A"])
    }

    func testAnsweredHeadsetBelowStillLosesToBuiltInButWinsWithoutIt() {
        let ranking = NewMicrophonePolicy.apply(
            .below, name: airPods.name, current: macBook.name, priorities: [])
        let both = MicrophonePriorityResolver.resolve(
            devices: [airPods, macBook], priorities: ranking)
        XCTAssertEqual(both?.name, macBook.name)
        let lidClosed = MicrophonePriorityResolver.resolve(devices: [airPods], priorities: ranking)
        XCTAssertEqual(lidClosed?.name, airPods.name)
    }

    // MARK: Card

    /// Off screen, no window. PHONON_RENDER_DIR keeps the PNGs for a look.
    func testCardsRenderAtTheSuggestionWidth() throws {
        let cases: [(String, NewMicrophoneQuestion)] = [
            ("mic_usb", NewMicrophonePolicy.question(
                for: yeti, devices: [macBook, yeti], priorities: [])),
            ("mic_bluetooth", NewMicrophonePolicy.question(
                for: airPods, devices: [macBook, airPods], priorities: [])),
            ("mic_full", NewMicrophonePolicy.question(
                for: yeti, devices: [macBook, yeti], priorities: ["A", "B", "Shure MV7"])),
        ]
        for (name, question) in cases {
            let model = MicrophonePromptModel()
            model.question = question
            let renderer = ImageRenderer(content: MicrophonePromptCard(model: model))
            renderer.scale = 2
            let image = try XCTUnwrap(renderer.cgImage)
            XCTAssertEqual(image.width, Int(SuggestionPanelGeometry.size.width * 2))
            XCTAssertLessThan(image.height, 260, name)
            if let directory = ProcessInfo.processInfo.environment["PHONON_RENDER_DIR"] {
                let url = URL(fileURLWithPath: directory).appendingPathComponent("\(name).png")
                let destination = try XCTUnwrap(
                    CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil))
                CGImageDestinationAddImage(destination, image, nil)
                XCTAssertTrue(CGImageDestinationFinalize(destination))
            }
        }
    }

    // MARK: Storage

    func testSeenSetRoundTripsAndAbsentMeansUnseeded() throws {
        let absent = try JSONDecoder().decode(
            NativeSettings.self, from: Data(#"{"schema_version":2}"#.utf8))
        XCTAssertNil(absent.microphonesSeen)
        let stored = try JSONDecoder().decode(
            NativeSettings.self,
            from: Data(#"{"schema_version":2,"microphones_seen":["x","y"]}"#.utf8))
        XCTAssertEqual(stored.microphonesSeen, ["x", "y"])
    }

    func testControllerSeedsOnFirstLaunchAndQueuesWithoutShowingWhileBusy() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "phonon-new-mic-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(#"{"schema_version":2,"microphone_priority":[]}"#.utf8)
            .write(to: directory.appendingPathComponent("settings.json"))

        var connected = [macBook]
        let store = NativeAppStore(supportDirectory: directory)
        let controller = NewMicrophoneController(store: store, devices: { connected })
        // Busy, so a newcomer waits in the queue and no panel is shown.
        controller.isBusy = { true }
        controller.scan()
        XCTAssertEqual(store.settings.microphonesSeen, [macBook.uid])

        connected.append(yeti)
        controller.scan()
        XCTAssertEqual(store.settings.microphonesSeen, [macBook.uid, yeti.uid])
        XCTAssertEqual(store.settings.microphonePriority, [])

        let saved = try JSONDecoder().decode(
            NativeSettings.self,
            from: Data(contentsOf: directory.appendingPathComponent("settings.json")))
        XCTAssertEqual(saved.microphonesSeen, [macBook.uid, yeti.uid])
    }
}
