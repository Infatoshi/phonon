import CoreFoundation
import XCTest

@testable import PhononBar

/// M13: the hotkey tap's run loop is its own thread, so a stalled main thread no longer delays
/// the tap callback, and key edges still reach the main thread in arrival order.
final class EventTapThreadTests: XCTestCase {
    private final class Box: @unchecked Sendable {
        let lock = NSLock()
        var firedNs: UInt64 = 0
        var order: [Int] = []
    }

    /// A source on the tap thread fires while the main thread is busy for 300 ms (on the main run
    /// loop it would wait the whole 300 ms).
    func testTapThreadRunsWhileTheMainThreadIsBlocked() {
        let tapThread = EventTapThread()
        let box = Box()
        var context = CFRunLoopSourceContext()
        context.info = Unmanaged.passUnretained(box).toOpaque()
        context.perform = { info in
            let box = Unmanaged<Box>.fromOpaque(info!).takeUnretainedValue()
            box.lock.lock()
            box.firedNs = DispatchTime.now().uptimeNanoseconds
            box.lock.unlock()
        }
        let source = CFRunLoopSourceCreate(kCFAllocatorDefault, 0, &context)!
        tapThread.add(source)
        let loop = tapThread.start()
        XCTAssertFalse(CFEqual(loop, CFRunLoopGetMain()))

        var latencies: [Double] = []
        for _ in 0..<20 {
            box.lock.lock()
            box.firedNs = 0
            box.lock.unlock()
            let signalled = DispatchTime.now().uptimeNanoseconds
            DispatchQueue.global(qos: .userInteractive).async {
                CFRunLoopSourceSignal(source)
                CFRunLoopWakeUp(loop)
            }
            // the main thread is stuck, as during the stalls behind M13
            let busyUntil = signalled + 300_000_000
            while DispatchTime.now().uptimeNanoseconds < busyUntil {}
            box.lock.lock()
            let fired = box.firedNs
            box.lock.unlock()
            XCTAssertGreaterThan(fired, 0, "fired before the main thread was free")
            latencies.append(Double(fired &- signalled) / 1e6)
        }
        latencies.sort()
        print(
            String(
                format: "tap thread, main blocked 300 ms: signal to callback p50 %.3f ms, max %.3f ms (n %d)",
                latencies[latencies.count / 2], latencies.last!, latencies.count))
        XCTAssertLessThan(latencies.last!, 50, "max signal-to-callback ms \(latencies)")
        tapThread.remove(source)
    }

    /// Edges posted from the tap thread run on the main thread in posting order.
    func testEdgesReachTheMainThreadInOrder() {
        let box = Box()
        let done = expectation(description: "all posted")
        let count = 2_000
        Thread.detachNewThread {
            for i in 0..<count {
                AppController.onMain {
                    XCTAssertTrue(Thread.isMainThread)
                    box.order.append(i)
                    if i == count - 1 { done.fulfill() }
                }
            }
        }
        wait(for: [done], timeout: 10)
        XCTAssertEqual(box.order, Array(0..<count))
    }
}
