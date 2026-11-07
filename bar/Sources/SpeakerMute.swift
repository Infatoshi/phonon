import CoreAudio
import Foundation

/// What was done to the speakers, with what is needed to undo it.
struct SpeakerSnapshot: Codable, Equatable {
    enum Mode: String, Codable { case mute, volume }

    /// The output device's stable UID; its `AudioDeviceID` changes across restarts.
    var deviceUID: String
    var mode: Mode
    /// For `.volume`: the channel elements that were lowered and their old levels.
    var elements: [UInt32] = []
    var levels: [Float] = []
}

/// The part of speaker control that touches the system, so the policy can be tested.
protocol SpeakerBackend {
    /// What a mute would change on the current output, or nil when there is
    /// nothing to do (already silent, or the device has no mute or volume control).
    func capture() -> SpeakerSnapshot?
    func silence(_ snapshot: SpeakerSnapshot)
    func restore(_ snapshot: SpeakerSnapshot)
}

protocol SpeakerSnapshotStore {
    func load() -> SpeakerSnapshot?
    func save(_ snapshot: SpeakerSnapshot?)
}

struct UserDefaultsSpeakerSnapshotStore: SpeakerSnapshotStore {
    private let key = "speakerMuteSnapshot"

    func load() -> SpeakerSnapshot? {
        UserDefaults.standard.data(forKey: key)
            .flatMap { try? JSONDecoder().decode(SpeakerSnapshot.self, from: $0) }
    }

    func save(_ snapshot: SpeakerSnapshot?) {
        if let snapshot, let data = try? JSONEncoder().encode(snapshot) {
            UserDefaults.standard.set(data, forKey: key)
        } else {
            UserDefaults.standard.removeObject(forKey: key)
        }
    }
}

/// Silences the speakers while dictation records and puts them back after.
///
/// Only what this session changed is undone: speakers that were already muted
/// stay muted. The pending restore is also written to disk, so a crash or
/// force-quit while muted does not leave the Mac silent: the next launch
/// calls `recoverAfterInterruption()`.
final class SpeakerMute {
    private let backend: SpeakerBackend
    private let store: SpeakerSnapshotStore
    private var held: SpeakerSnapshot?

    init(
        backend: SpeakerBackend = CoreAudioSpeakers(),
        store: SpeakerSnapshotStore = UserDefaultsSpeakerSnapshotStore()
    ) {
        self.backend = backend
        self.store = store
    }

    var isMuting: Bool { held != nil }

    func mute() {
        guard held == nil, let snapshot = backend.capture() else { return }
        store.save(snapshot)  // before the change, so an interruption can always undo it
        backend.silence(snapshot)
        held = snapshot
    }

    func restore() {
        guard let snapshot = held else { return }
        held = nil
        backend.restore(snapshot)
        store.save(nil)
    }

    /// Undo a mute that an earlier run never got to undo.
    func recoverAfterInterruption() {
        guard held == nil, let snapshot = store.load() else { return }
        backend.restore(snapshot)
        store.save(nil)
    }
}

/// The default output device, silenced with its mute switch when it has one
/// and by lowering its volume to zero otherwise.
struct CoreAudioSpeakers: SpeakerBackend {
    func capture() -> SpeakerSnapshot? {
        guard let device = Self.defaultOutputDevice(), let uid = Self.uid(of: device) else {
            return nil
        }
        if Self.isSettable(device, selector: kAudioDevicePropertyMute, element: Self.main) {
            guard let muted = Self.uint32(device, kAudioDevicePropertyMute, Self.main),
                muted == 0
            else { return nil }
            return SpeakerSnapshot(deviceUID: uid, mode: .mute)
        }
        var elements: [UInt32] = []
        var levels: [Float] = []
        for element in Self.volumeElements(of: device) {
            if let level = Self.float32(device, kAudioDevicePropertyVolumeScalar, element),
                level > 0
            {
                elements.append(element)
                levels.append(level)
            }
        }
        guard !elements.isEmpty else { return nil }
        return SpeakerSnapshot(deviceUID: uid, mode: .volume, elements: elements, levels: levels)
    }

    func silence(_ snapshot: SpeakerSnapshot) {
        guard let device = Self.device(forUID: snapshot.deviceUID) else { return }
        switch snapshot.mode {
        case .mute:
            Self.set(device, kAudioDevicePropertyMute, Self.main, UInt32(1))
        case .volume:
            for element in snapshot.elements {
                Self.set(device, kAudioDevicePropertyVolumeScalar, element, Float32(0))
            }
        }
    }

    func restore(_ snapshot: SpeakerSnapshot) {
        guard let device = Self.device(forUID: snapshot.deviceUID) else { return }
        switch snapshot.mode {
        case .mute:
            Self.set(device, kAudioDevicePropertyMute, Self.main, UInt32(0))
        case .volume:
            for (element, level) in zip(snapshot.elements, snapshot.levels) {
                // A level the user raised during the dictation is theirs; leave it.
                if let now = Self.float32(device, kAudioDevicePropertyVolumeScalar, element),
                    now > 0
                {
                    continue
                }
                Self.set(device, kAudioDevicePropertyVolumeScalar, element, Float32(level))
            }
        }
    }

    // MARK: CoreAudio plumbing

    private static let main = kAudioObjectPropertyElementMain

    private static func address(_ selector: AudioObjectPropertySelector, _ element: UInt32)
        -> AudioObjectPropertyAddress
    {
        AudioObjectPropertyAddress(
            mSelector: selector, mScope: kAudioDevicePropertyScopeOutput, mElement: element)
    }

    private static func defaultOutputDevice() -> AudioDeviceID? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        var id = AudioDeviceID(kAudioObjectUnknown)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        guard
            AudioObjectGetPropertyData(
                AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &id) == noErr,
            id != kAudioObjectUnknown
        else { return nil }
        return id
    }

    private static func uid(of device: AudioDeviceID) -> String? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceUID, mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var uid: Unmanaged<CFString>?
        var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        guard AudioObjectGetPropertyData(device, &address, 0, nil, &size, &uid) == noErr,
            let uid
        else { return nil }
        return uid.takeRetainedValue() as String
    }

    private static func device(forUID uid: String) -> AudioDeviceID? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyTranslateUIDToDevice,
            mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        var id = AudioDeviceID(kAudioObjectUnknown)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        var qualifier = uid as CFString
        let status = withUnsafeMutablePointer(to: &qualifier) { pointer in
            AudioObjectGetPropertyData(
                AudioObjectID(kAudioObjectSystemObject), &address,
                UInt32(MemoryLayout<CFString>.size), pointer, &size, &id)
        }
        return status == noErr && id != kAudioObjectUnknown ? id : nil
    }

    private static func isSettable(
        _ device: AudioDeviceID, selector: AudioObjectPropertySelector, element: UInt32
    ) -> Bool {
        var address = address(selector, element)
        guard AudioObjectHasProperty(device, &address) else { return false }
        var settable: DarwinBoolean = false
        return AudioObjectIsPropertySettable(device, &address, &settable) == noErr
            && settable.boolValue
    }

    /// The main volume control when the device has one, else its left and right channels.
    private static func volumeElements(of device: AudioDeviceID) -> [UInt32] {
        if isSettable(device, selector: kAudioDevicePropertyVolumeScalar, element: main) {
            return [main]
        }
        return [1, 2].filter {
            isSettable(device, selector: kAudioDevicePropertyVolumeScalar, element: $0)
        }
    }

    private static func uint32(
        _ device: AudioDeviceID, _ selector: AudioObjectPropertySelector, _ element: UInt32
    ) -> UInt32? {
        var address = address(selector, element)
        var value: UInt32 = 0
        var size = UInt32(MemoryLayout<UInt32>.size)
        return AudioObjectGetPropertyData(device, &address, 0, nil, &size, &value) == noErr
            ? value : nil
    }

    private static func float32(
        _ device: AudioDeviceID, _ selector: AudioObjectPropertySelector, _ element: UInt32
    ) -> Float32? {
        var address = address(selector, element)
        var value: Float32 = 0
        var size = UInt32(MemoryLayout<Float32>.size)
        return AudioObjectGetPropertyData(device, &address, 0, nil, &size, &value) == noErr
            ? value : nil
    }

    private static func set<T>(
        _ device: AudioDeviceID, _ selector: AudioObjectPropertySelector, _ element: UInt32,
        _ newValue: T
    ) {
        var address = address(selector, element)
        var value = newValue
        let status = AudioObjectSetPropertyData(
            device, &address, 0, nil, UInt32(MemoryLayout<T>.size), &value)
        if status != noErr { NSLog("phonon speakers: could not change output (\(status))") }
    }
}
