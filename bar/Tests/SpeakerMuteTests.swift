import XCTest

@testable import PhononBar

private final class FakeSpeakers: SpeakerBackend {
    var next: SpeakerSnapshot? = SpeakerSnapshot(deviceUID: "built-in", mode: .mute)
    var silenced: [SpeakerSnapshot] = []
    var restored: [SpeakerSnapshot] = []

    func capture() -> SpeakerSnapshot? { next }
    func silence(_ snapshot: SpeakerSnapshot) { silenced.append(snapshot) }
    func restore(_ snapshot: SpeakerSnapshot) { restored.append(snapshot) }
}

private final class MemoryStore: SpeakerSnapshotStore {
    var snapshot: SpeakerSnapshot?
    func load() -> SpeakerSnapshot? { snapshot }
    func save(_ snapshot: SpeakerSnapshot?) { self.snapshot = snapshot }
}

final class SpeakerMuteTests: XCTestCase {
    func testMuteThenRestoreUndoesExactlyWhatWasChanged() {
        let backend = FakeSpeakers()
        let store = MemoryStore()
        let speakers = SpeakerMute(backend: backend, store: store)

        speakers.mute()
        XCTAssertTrue(speakers.isMuting)
        XCTAssertEqual(backend.silenced, [backend.next!])
        XCTAssertEqual(store.snapshot, backend.next, "the pending restore is written down")

        speakers.restore()
        XCTAssertFalse(speakers.isMuting)
        XCTAssertEqual(backend.restored, [backend.next!])
        XCTAssertNil(store.snapshot)
    }

    func testAlreadySilentSpeakersAreLeftAlone() {
        let backend = FakeSpeakers()
        backend.next = nil  // already muted, or no control: nothing to change
        let speakers = SpeakerMute(backend: backend, store: MemoryStore())

        speakers.mute()
        speakers.restore()

        XCTAssertTrue(backend.silenced.isEmpty)
        XCTAssertTrue(backend.restored.isEmpty, "must not unmute what the user had muted")
    }

    func testMutingTwiceKeepsTheFirstSnapshot() {
        let backend = FakeSpeakers()
        let speakers = SpeakerMute(backend: backend, store: MemoryStore())
        speakers.mute()
        // Second call would now see muted speakers and report nothing to change.
        backend.next = nil
        speakers.mute()
        speakers.restore()

        XCTAssertEqual(backend.silenced.count, 1)
        XCTAssertEqual(backend.restored, [SpeakerSnapshot(deviceUID: "built-in", mode: .mute)])
    }

    func testRestoreWithoutMuteDoesNothing() {
        let backend = FakeSpeakers()
        let speakers = SpeakerMute(backend: backend, store: MemoryStore())
        speakers.restore()
        XCTAssertTrue(backend.restored.isEmpty)
    }

    func testRestoreTwiceRestoresOnce() {
        let backend = FakeSpeakers()
        let speakers = SpeakerMute(backend: backend, store: MemoryStore())
        speakers.mute()
        speakers.restore()
        speakers.restore()
        XCTAssertEqual(backend.restored.count, 1)
    }

    func testAMuteLeftByACrashedRunIsUndoneAtNextLaunch() {
        let backend = FakeSpeakers()
        let store = MemoryStore()
        let leftover = SpeakerSnapshot(
            deviceUID: "usb", mode: .volume, elements: [1, 2], levels: [0.4, 0.5])
        store.snapshot = leftover

        SpeakerMute(backend: backend, store: store).recoverAfterInterruption()

        XCTAssertEqual(backend.restored, [leftover])
        XCTAssertNil(store.snapshot)
    }

    func testSnapshotSurvivesJSON() throws {
        let snapshot = SpeakerSnapshot(
            deviceUID: "usb", mode: .volume, elements: [1, 2], levels: [0.25, 0.75])
        let data = try JSONEncoder().encode(snapshot)
        XCTAssertEqual(try JSONDecoder().decode(SpeakerSnapshot.self, from: data), snapshot)
    }

    func testSettingIsOnByDefaultAndOnForExistingInstalls() throws {
        XCTAssertTrue(NativeSettings().muteSpeakersWhileDictating)
        for stored in [1, 2] {
            let json = Data("{\"schema_version\": \(stored)}".utf8)
            let decoded = try JSONDecoder().decode(NativeSettings.self, from: json)
            XCTAssertTrue(decoded.muteSpeakersWhileDictating)
        }
        let off = Data("{\"schema_version\": 2, \"mute_speakers_while_dictating\": false}".utf8)
        XCTAssertFalse(
            try JSONDecoder().decode(NativeSettings.self, from: off).muteSpeakersWhileDictating)
    }
}
