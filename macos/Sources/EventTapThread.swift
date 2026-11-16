import CoreFoundation
import Foundation

/// A thread that does nothing but run a CFRunLoop for the hotkey event tap (GOAL M13).
///
/// On the main run loop, key events waited for whatever the main thread was doing:
/// key_event_to_callback p99 147 ms, max 420 ms (43 of 1,332 over 50 ms, 2026-11-07). Here the
/// tap callback runs at once at user-interactive priority; it only reads the event and hands the
/// edge to the main thread in arrival order (`callback_to_main_ms` in e2e.jsonl keeps measuring
/// that hop).
final class EventTapThread: @unchecked Sendable {
    private let condition = NSCondition()
    private var runLoop: CFRunLoop?
    private var thread: Thread?

    init() {}

    /// The thread's run loop; starts the thread on first use.
    func start() -> CFRunLoop {
        condition.lock()
        defer { condition.unlock() }
        if let runLoop { return runLoop }
        let thread = Thread { [weak self] in self?.run() }
        thread.name = "phonon.hotkey-tap"
        thread.qualityOfService = .userInteractive
        self.thread = thread
        thread.start()
        while runLoop == nil { condition.wait() }
        return runLoop!
    }

    func add(_ source: CFRunLoopSource) {
        let loop = start()
        CFRunLoopAddSource(loop, source, .commonModes)
        CFRunLoopWakeUp(loop)
    }

    func remove(_ source: CFRunLoopSource) {
        condition.lock()
        let loop = runLoop
        condition.unlock()
        guard let loop else { return }
        CFRunLoopRemoveSource(loop, source, .commonModes)
    }

    private func run() {
        // A run loop with no source returns at once: a never-signalled source keeps it alive.
        var context = CFRunLoopSourceContext()
        context.perform = { _ in }
        let keepAlive = CFRunLoopSourceCreate(kCFAllocatorDefault, 0, &context)
        let loop = CFRunLoopGetCurrent()!
        CFRunLoopAddSource(loop, keepAlive, .commonModes)
        condition.lock()
        runLoop = loop
        condition.broadcast()
        condition.unlock()
        while true { CFRunLoopRun() }
    }
}
