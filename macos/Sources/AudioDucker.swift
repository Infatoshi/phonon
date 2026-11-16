import AudioToolbox
import CoreAudio
import Foundation

/// settings.json `background_audio`: what other audio on the Mac does while the dictation key
/// is held. Lowering it keeps a video or song in the room out of the microphone (speakers leak
/// into the mic; headphones do not) and makes the speaker easier to hear for the recognizer.
enum BackgroundAudio: String, Codable, CaseIterable, Identifiable {
    /// Leave other audio alone.
    case off
    /// Fade other audio down by 15 dB (the default; about a quarter of the volume slider).
    case lower
    /// Fade other audio out completely.
    case mute

    var id: String { rawValue }

    var title: String {
        switch self {
        case .off: return "Leave it"
        case .lower: return "Lower it"
        case .mute: return "Mute it"
        }
    }

    /// How far other audio drops, in dB; nil leaves it alone; -infinity is silence.
    var dropDB: Double? {
        switch self {
        case .off: return nil
        case .lower: return -15
        case .mute: return -.infinity
        }
    }
}

protocol SystemVolumeControlling: AnyObject {
    /// The default output device's volume slider, 0...1; nil when it has none (HDMI, some USB).
    func currentVolume() -> Double?
    /// Called from the fade's timer thread, not the main thread.
    func setVolume(_ volume: Double)
    /// The device's own slider -> dB curve (nil without one: a dB curve is assumed).
    func decibels(forVolume volume: Double) -> Double?
    func volume(forDecibels db: Double) -> Double?
    /// The device's dB floor (the slider at 0).
    var floorDB: Double { get }
    /// Bluetooth carries each volume change over the radio: it gets fewer, larger steps.
    var isBluetooth: Bool { get }
}

/// The default output device's main volume through CoreAudio.
final class CoreAudioVolumeController: SystemVolumeControlling {
    func currentVolume() -> Double? {
        guard let device = Self.defaultOutputDevice() else { return nil }
        var address = Self.mainVolume
        guard AudioObjectHasProperty(device, &address) else { return nil }
        var size = UInt32(MemoryLayout<Float32>.size)
        var volume = Float32(0)
        guard AudioObjectGetPropertyData(device, &address, 0, nil, &size, &volume) == noErr else {
            return nil
        }
        return Double(volume)
    }

    func setVolume(_ volume: Double) {
        guard let device = Self.defaultOutputDevice() else { return }
        var address = Self.mainVolume
        var settable = DarwinBoolean(false)
        guard AudioObjectIsPropertySettable(device, &address, &settable) == noErr,
            settable.boolValue
        else { return }
        var next = Float32(min(max(volume, 0), 1))
        AudioObjectSetPropertyData(
            device, &address, 0, nil, UInt32(MemoryLayout<Float32>.size), &next)
    }

    func decibels(forVolume volume: Double) -> Double? {
        translate(kAudioDevicePropertyVolumeScalarToDecibels, volume)
    }

    func volume(forDecibels db: Double) -> Double? {
        translate(kAudioDevicePropertyVolumeDecibelsToScalar, db)
    }

    var floorDB: Double {
        guard let device = Self.defaultOutputDevice() else { return -40 }
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyVolumeRangeDecibels,
            mScope: kAudioDevicePropertyScopeOutput, mElement: 1)
        var range = AudioValueRange()
        var size = UInt32(MemoryLayout<AudioValueRange>.size)
        guard AudioObjectGetPropertyData(device, &address, 0, nil, &size, &range) == noErr else {
            return -40
        }
        return range.mMinimum
    }

    var isBluetooth: Bool {
        guard let device = Self.defaultOutputDevice() else { return false }
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyTransportType,
            mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        var transport = UInt32(0)
        var size = UInt32(MemoryLayout<UInt32>.size)
        guard AudioObjectGetPropertyData(device, &address, 0, nil, &size, &transport) == noErr
        else { return false }
        return transport == kAudioDeviceTransportTypeBluetooth
            || transport == kAudioDeviceTransportTypeBluetoothLE
    }

    /// The main volume has no dB translation of its own; channel 1 carries the device's curve.
    private func translate(_ selector: AudioObjectPropertySelector, _ value: Double) -> Double? {
        guard let device = Self.defaultOutputDevice() else { return nil }
        var address = AudioObjectPropertyAddress(
            mSelector: selector, mScope: kAudioDevicePropertyScopeOutput, mElement: 1)
        guard AudioObjectHasProperty(device, &address) else { return nil }
        var v = Float32(value)
        var size = UInt32(MemoryLayout<Float32>.size)
        guard AudioObjectGetPropertyData(device, &address, 0, nil, &size, &v) == noErr else {
            return nil
        }
        return Double(v)
    }

    private static var mainVolume = AudioObjectPropertyAddress(
        mSelector: kAudioHardwareServiceDeviceProperty_VirtualMainVolume,
        mScope: kAudioDevicePropertyScopeOutput,
        mElement: kAudioObjectPropertyElementMain)

    private static func defaultOutputDevice() -> AudioObjectID? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var device = AudioObjectID(0)
        var size = UInt32(MemoryLayout<AudioObjectID>.size)
        let status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &device)
        return status == noErr && device != 0 ? device : nil
    }
}

/// The fade's shape, the way a video editor's audio fade sounds smooth: it moves in decibels
/// (equal steps of loudness, not of the slider) along a smootherstep S-curve, so it leaves and
/// lands with zero speed and zero acceleration and has no audible corners.
enum DuckCurve {
    static func smootherstep(_ t: Double) -> Double {
        let x = min(max(t, 0), 1)
        return x * x * x * (x * (x * 6 - 15) + 10)
    }

    /// dB at time fraction t of a fade from `from` to `to` dB.
    static func decibels(from: Double, to: Double, at t: Double) -> Double {
        from + (to - from) * smootherstep(t)
    }
}

/// One fade, run on its own high-priority timer thread so a busy main thread (correcting and
/// pasting at key up) cannot stall it. Each tick computes the volume from the elapsed time, so
/// a late tick skips ahead instead of stretching the fade.
private final class DuckRamp: @unchecked Sendable {
    private static let queue = DispatchQueue(label: "phonon.duck", qos: .userInteractive)
    private let lock = NSLock()
    private var timer: DispatchSourceTimer?
    private var _last: Double?

    var last: Double? {
        lock.lock()
        defer { lock.unlock() }
        return _last
    }

    func run(
        volume: any SystemVolumeControlling, fromDB: Double, toDB: Double, end: Double,
        seconds: Double, tick: Double, done: @escaping @Sendable () -> Void
    ) {
        let start = DispatchTime.now().uptimeNanoseconds
        let floor = volume.floorDB
        let t = DispatchSource.makeTimerSource(flags: .strict, queue: Self.queue)
        t.schedule(deadline: .now(), repeating: tick, leeway: .milliseconds(1))
        t.setEventHandler { [weak self] in
            guard let self else { return }
            let f = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9 / max(seconds, 1e-6)
            let v: Double
            if f >= 1 {
                v = end
            } else {
                let db = max(DuckCurve.decibels(from: fromDB, to: toDB, at: f), floor)
                v = volume.volume(forDecibels: db) ?? pow(10, db / 20)
            }
            // skip a write too small to hear, except the landing, which is always exact
            if f >= 1 || (self.last.map({ abs($0 - v) > 0.0005 }) ?? true) {
                volume.setVolume(v)
                self.lock.lock()
                self._last = v
                self.lock.unlock()
            }
            if f >= 1 {
                self.cancel()
                DispatchQueue.main.async(execute: done)
            }
        }
        lock.lock()
        timer = t
        lock.unlock()
        t.resume()
    }

    func cancel() {
        lock.lock()
        let t = timer
        timer = nil
        lock.unlock()
        t?.cancel()
    }
}

/// Lowers or mutes the Mac's output while dictating and puts it back afterwards.
/// - Down 0.3 s at key down; back up 1.2 s at key up (a slow rise sounds natural, a quick dip does not hurt).
/// - Ticks every 10 ms, or every 25 ms on Bluetooth, which carries each change over the radio.
/// - A volume the user changes mid-dictation is theirs: it is not overwritten at key up.
/// - The volume to come back to is saved, so a crash mid-dictation is repaired at next launch.
@MainActor
final class AudioDucker {
    static let savedKey = "phonon.duck.restoreVolume"

    private let volume: any SystemVolumeControlling
    private let defaults: UserDefaults
    private let fadeDown: Double
    private let fadeUp: Double
    private var original: Double?
    private var ramp: DuckRamp?
    private var lastSet: Double?

    init(
        volume: any SystemVolumeControlling = CoreAudioVolumeController(),
        defaults: UserDefaults = .standard, fadeDown: Double = 0.3, fadeUp: Double = 1.2
    ) {
        self.volume = volume
        self.defaults = defaults
        self.fadeDown = fadeDown
        self.fadeUp = fadeUp
    }

    var isDucked: Bool { original != nil }

    func start(_ mode: BackgroundAudio) {
        guard let drop = mode.dropDB, let now = volume.currentVolume() else { return }
        cancelRamp()
        // a new dictation during the rise keeps the first volume as the one to restore
        let base = original ?? now
        original = base
        defaults.set(base, forKey: Self.savedKey)
        let floor = volume.floorDB
        let baseDB = db(base)
        let target = max(baseDB + drop, floor)
        fade(fromDB: db(now), toDB: target, end: drop.isInfinite ? 0 : vol(target), over: fadeDown)
    }

    func stop() {
        guard let base = original else { return }
        cancelRamp()
        let now = volume.currentVolume() ?? base
        if let lastSet, abs(now - lastSet) > 0.02 {
            finish()  // the user moved the volume while dictating: keep their choice
            return
        }
        fade(fromDB: db(now), toDB: db(base), end: base, over: fadeUp) { [weak self] in
            self?.finish()
        }
    }

    /// At quit: back to the saved volume at once.
    func restoreNow() {
        cancelRamp()
        if let original { volume.setVolume(original) }
        finish()
    }

    /// At launch: a crash or force quit mid-dictation left the output lowered; put it back.
    func recoverAfterCrash() {
        guard defaults.object(forKey: Self.savedKey) != nil else { return }
        let saved = defaults.double(forKey: Self.savedKey)
        if let now = volume.currentVolume(), now < saved { volume.setVolume(saved) }
        defaults.removeObject(forKey: Self.savedKey)
    }

    private func db(_ v: Double) -> Double {
        if v <= 0 { return volume.floorDB }
        return volume.decibels(forVolume: v) ?? max(20 * log10(v), volume.floorDB)
    }

    private func vol(_ db: Double) -> Double {
        volume.volume(forDecibels: db) ?? pow(10, db / 20)
    }

    private func cancelRamp() {
        if let ramp {
            lastSet = ramp.last ?? lastSet
            ramp.cancel()
        }
        ramp = nil
    }

    private func finish() {
        original = nil
        lastSet = nil
        defaults.removeObject(forKey: Self.savedKey)
    }

    private func fade(
        fromDB: Double, toDB: Double, end: Double, over seconds: Double,
        then done: (() -> Void)? = nil
    ) {
        let r = DuckRamp()
        ramp = r
        let tick = volume.isBluetooth ? 0.025 : 0.010
        r.run(
            volume: volume, fromDB: fromDB, toDB: toDB, end: end, seconds: seconds, tick: tick
        ) { [weak self] in
            MainActor.assumeIsolated {
                guard let self, self.ramp === r else { return }
                self.lastSet = r.last
                self.ramp = nil
                done?()
            }
        }
    }
}
