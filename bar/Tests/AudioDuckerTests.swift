import XCTest

@testable import PhononBar

/// A device with the WH-1000XM6's curve: the slider is linear in dB over -40...0.
private final class FakeVolume: SystemVolumeControlling, @unchecked Sendable {
    private let lock = NSLock()
    private var _volume: Double?
    private var _writes: [(Double, UInt64)] = []
    let isBluetooth: Bool
    init(_ volume: Double?, bluetooth: Bool = false) {
        _volume = volume
        isBluetooth = bluetooth
    }
    var volume: Double? {
        get { lock.withLock { _volume } }
        set { lock.withLock { _volume = newValue } }
    }
    var writes: [(Double, UInt64)] { lock.withLock { _writes } }
    func currentVolume() -> Double? { volume }
    func setVolume(_ v: Double) {
        lock.withLock {
            _volume = v
            _writes.append((v, DispatchTime.now().uptimeNanoseconds))
        }
    }
    func decibels(forVolume v: Double) -> Double? { -40 + 40 * v }
    func volume(forDecibels db: Double) -> Double? { min(max((db + 40) / 40, 0), 1) }
    var floorDB: Double { -40 }
}

private func blockMainThread(_ seconds: Double) {
    let until = Date().addingTimeInterval(seconds)
    while Date() < until {}
}

@MainActor
final class AudioDuckerTests: XCTestCase {
    private var defaults: UserDefaults!

    override func setUp() {
        defaults = UserDefaults(suiteName: "phonon-duck-\(UUID().uuidString)")
    }

    private func ducker(_ out: FakeVolume) -> AudioDucker {
        AudioDucker(volume: out, defaults: defaults, fadeDown: 0.1, fadeUp: 0.2)
    }

    private func waitFor(_ cond: () -> Bool, _ msg: String) async {
        for _ in 0..<300 where !cond() { try? await Task.sleep(nanoseconds: 10_000_000) }
        XCTAssertTrue(cond(), msg)
    }

    private func near(_ a: Double?, _ b: Double) -> Bool { abs((a ?? -1) - b) < 1e-6 }

    func testTheCurveLeavesAndLandsWithoutCorners() {
        XCTAssertEqual(DuckCurve.smootherstep(0), 0)
        XCTAssertEqual(DuckCurve.smootherstep(1), 1)
        XCTAssertEqual(DuckCurve.smootherstep(0.5), 0.5, accuracy: 1e-12)
        // zero slope at both ends: the first and last 1% of time move < 0.01% of the way
        XCTAssertLessThan(DuckCurve.smootherstep(0.01), 1e-4)
        XCTAssertGreaterThan(DuckCurve.smootherstep(0.99), 1 - 1e-4)
        // monotonic
        var prev = -1.0
        for i in 0...1000 {
            let y = DuckCurve.smootherstep(Double(i) / 1000)
            XCTAssertGreaterThanOrEqual(y, prev)
            prev = y
        }
    }

    func testLowerDropsFifteenDecibelsAndComesBack() async {
        let out = FakeVolume(0.5)  // -20 dB
        let d = ducker(out)
        d.start(.lower)
        await waitFor({ near(out.volume, 0.125) }, "-35 dB = slider 0.125")
        XCTAssertTrue(out.writes.count > 4, "it fades, not jumps")
        XCTAssertEqual(defaults.double(forKey: AudioDucker.savedKey), 0.5)
        d.stop()
        await waitFor({ near(out.volume, 0.5) && !d.isDucked }, "restored")
        XCTAssertNil(defaults.object(forKey: AudioDucker.savedKey))
    }

    func testTheRiseMovesInEvenDecibelStepsWithNoJumps() async {
        let out = FakeVolume(0.5)
        let d = ducker(out)
        d.start(.lower)
        await waitFor({ near(out.volume, 0.125) }, "lowered")
        let before = out.writes.count
        d.stop()
        await waitFor({ near(out.volume, 0.5) && !d.isDucked }, "restored")
        let rise = out.writes[before...].map { -40 + 40 * $0.0 }
        let steps = zip(rise.dropFirst(), rise).map { $0 - $1 }
        XCTAssertTrue(steps.allSatisfy { $0 >= -1e-9 }, "never dips on the way up")
        // 15 dB over 0.2 s at 10 ms ticks: smootherstep's peak speed is 1.875x the mean, so
        // no single step should exceed ~1.875 * 15 / 20 = 1.4 dB, plus timer slack
        XCTAssertLessThan(steps.max() ?? 0, 2.5, "no jump: \(steps)")
    }

    func testMuteFadesToSilence() async {
        let out = FakeVolume(0.5)
        let d = ducker(out)
        d.start(.mute)
        await waitFor({ out.volume == 0 }, "muted")
        d.stop()
        await waitFor({ near(out.volume, 0.5) }, "restored")
    }

    func testOffAndNoVolumeControlDoNothing() async {
        let out = FakeVolume(0.5)
        let d = ducker(out)
        d.start(.off)
        d.stop()
        let none = FakeVolume(nil)
        let d2 = ducker(none)
        d2.start(.mute)
        d2.stop()
        try? await Task.sleep(nanoseconds: 50_000_000)
        XCTAssertTrue(out.writes.isEmpty)
        XCTAssertTrue(none.writes.isEmpty)
        XCTAssertFalse(d.isDucked || d2.isDucked)
    }

    func testABusyMainThreadDoesNotStallTheFade() async {
        let out = FakeVolume(0.5)
        let d = AudioDucker(volume: out, defaults: defaults, fadeDown: 0.3, fadeUp: 0.3)
        d.start(.lower)
        let before = out.writes.count
        blockMainThread(0.2)  // as a paste at key up can
        let during = out.writes.count - before
        XCTAssertGreaterThan(during, 8, "the timer kept stepping while main was blocked")
        await waitFor({ near(out.volume, 0.125) }, "lowered")
        d.stop()
        await waitFor({ !d.isDucked }, "restored")
    }

    func testBluetoothStepsLessOften() async {
        let wired = FakeVolume(0.5)
        let bt = FakeVolume(0.5, bluetooth: true)
        let a = AudioDucker(volume: wired, defaults: defaults, fadeDown: 0.25, fadeUp: 0.25)
        let b = AudioDucker(volume: bt, defaults: defaults, fadeDown: 0.25, fadeUp: 0.25)
        a.start(.lower)
        b.start(.lower)
        await waitFor({ near(wired.volume, 0.125) && near(bt.volume, 0.125) }, "both lowered")
        XCTAssertLessThan(bt.writes.count, wired.writes.count)
        XCTAssertLessThanOrEqual(bt.writes.count, 14)  // 0.25 s / 25 ms + the landing
        a.restoreNow()
        b.restoreNow()
    }

    func testAVolumeTheUserSetsMidDictationIsKept() async {
        let out = FakeVolume(0.8)
        let d = ducker(out)
        d.start(.lower)
        await waitFor({ near(out.volume, 0.425) }, "lowered: -8 dB - 15 dB = -23 dB = 0.425")
        try? await Task.sleep(nanoseconds: 30_000_000)
        out.volume = 0.6  // the volume keys
        let writes = out.writes.count
        d.stop()
        try? await Task.sleep(nanoseconds: 100_000_000)
        XCTAssertEqual(out.volume, 0.6)
        XCTAssertEqual(out.writes.count, writes)
        XCTAssertFalse(d.isDucked)
    }

    func testRedictatingDuringTheRiseKeepsTheFirstVolume() async {
        let out = FakeVolume(0.8)
        let d = ducker(out)
        d.start(.lower)
        await waitFor({ near(out.volume, 0.425) }, "lowered")
        d.stop()
        try? await Task.sleep(nanoseconds: 80_000_000)  // part way back up
        d.start(.lower)
        await waitFor({ near(out.volume, 0.425) }, "lowered from the original, not the midway volume")
        d.stop()
        await waitFor({ near(out.volume, 0.8) }, "back to the first volume")
    }

    func testQuitRestoresAtOnceAndLaunchRepairsACrash() async {
        let out = FakeVolume(0.8)
        let d = ducker(out)
        d.start(.mute)
        await waitFor({ out.volume == 0 }, "muted")
        d.restoreNow()
        XCTAssertEqual(out.volume, 0.8)
        XCTAssertNil(defaults.object(forKey: AudioDucker.savedKey))

        // a crash mid-dictation: the saved volume is left behind with the output lowered
        defaults.set(0.7, forKey: AudioDucker.savedKey)
        let after = FakeVolume(0.1)
        ducker(after).recoverAfterCrash()
        XCTAssertEqual(after.volume, 0.7)
        XCTAssertNil(defaults.object(forKey: AudioDucker.savedKey))
    }
}
