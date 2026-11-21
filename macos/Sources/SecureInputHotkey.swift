import AppKit
import Carbon.HIToolbox
import Foundation

/// While another app holds Secure Event Input (a password field, a terminal with "Secure
/// Keyboard Entry"), the event tap stops seeing key-downs. Phonon's Ctrl+Space toggle is a
/// Carbon hotkey, which the window server still delivers, so it is the fallback: kept
/// registered (and re-verified) for as long as secure input is on and the shortcut is a key
/// chord. A modifier-only hold key (Globe, Right Option) arrives as flags-changed events
/// and needs nothing.

/// What the fallback hotkey must do after one observation.
enum SecureInputTransition: Equatable {
    case register
    case unregister
    case none
}

/// Pure state machine: never registers twice, never leaks a registration.
struct SecureInputState: Equatable {
    private(set) var isSecureInputEnabled = false
    private(set) var fallbackRegistered = false

    /// The shortcut mode includes a key chord (Ctrl+Space) that a Carbon hotkey can carry.
    static func usesKeyChord(mode: String) -> Bool {
        ShortcutPolicy.sources(for: mode).contains("control-space")
    }

    static func needsFallback(secureInput: Bool, mode: String) -> Bool {
        secureInput && usesKeyChord(mode: mode)
    }

    /// One observation. `secureInputChanged` is true only on an enable or disable edge, so
    /// the caller logs each transition once.
    mutating func update(secureInput: Bool, mode: String)
        -> (transition: SecureInputTransition, secureInputChanged: Bool)
    {
        let changed = secureInput != isSecureInputEnabled
        isSecureInputEnabled = secureInput
        let needed = Self.needsFallback(secureInput: secureInput, mode: mode)
        switch (needed, fallbackRegistered) {
        case (true, false):
            fallbackRegistered = true
            return (.register, changed)
        case (false, true):
            fallbackRegistered = false
            return (.unregister, changed)
        default:
            return (.none, changed)
        }
    }

    /// Stop: whatever was registered must be released.
    mutating func reset() -> SecureInputTransition {
        isSecureInputEnabled = false
        guard fallbackRegistered else { return .none }
        fallbackRegistered = false
        return .unregister
    }
}

/// Polls `IsSecureEventInputEnabled()` on a light main-thread timer and reports transitions.
@MainActor
final class SecureInputMonitor {
    static let pollInterval: TimeInterval = 0.25

    private var timer: Timer?
    private(set) var state = SecureInputState()
    private let isEnabled: () -> Bool
    /// The current shortcut mode (`NativeSettings.shortcutMode`).
    var shortcutMode: () -> String = { "" }
    /// Register or release the fallback hotkey.
    var onTransition: (SecureInputTransition) -> Void = { _ in }
    /// Every poll while the fallback is registered: re-verify the hotkey is still armed.
    var onVerify: () -> Void = {}

    init(isEnabled: @escaping () -> Bool = { IsSecureEventInputEnabled() }) {
        self.isEnabled = isEnabled
    }

    /// Off in tests and UI demos: a polling timer there is noise.
    static var runsNormally: Bool {
        let env = ProcessInfo.processInfo.environment
        return env["XCTestConfigurationFilePath"] == nil && env["PHONON_UI_DEMO"] == nil
    }

    func start() {
        guard timer == nil else { return }
        poll()
        let timer = Timer(timeInterval: Self.pollInterval, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.poll() }
        }
        timer.tolerance = 0.05
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
    }

    func stop() {
        timer?.invalidate()
        timer = nil
        let transition = state.reset()
        if transition != .none { onTransition(transition) }
    }

    func poll() {
        let secure = isEnabled()
        let (transition, changed) = state.update(secureInput: secure, mode: shortcutMode())
        if changed {
            NSLog(
                "phonon: secure event input \(secure ? "enabled by another app" : "released")"
                    + (state.fallbackRegistered ? "; Ctrl+Space Carbon hotkey armed" : ""))
        }
        if transition != .none { onTransition(transition) }
        if state.fallbackRegistered { onVerify() }
    }
}

/// The character a key produces on the current layout, without starting a dead-key
/// sequence (Option-E, ´ on a German layout), for labeling a shortcut key.
enum DeadKeyLabeler {
    static func label(forKeyCode keyCode: UInt16) -> String? {
        let sources = [
            TISCopyCurrentKeyboardLayoutInputSource()?.takeRetainedValue(),
            TISCopyCurrentASCIICapableKeyboardLayoutInputSource()?.takeRetainedValue(),
        ]
        for case let source? in sources {
            if let label = label(forKeyCode: keyCode, source: source) { return label }
        }
        return nil
    }

    private static func label(forKeyCode keyCode: UInt16, source: TISInputSource) -> String? {
        guard let raw = TISGetInputSourceProperty(source, kTISPropertyUnicodeKeyLayoutData)
        else { return nil }
        let data = Unmanaged<CFData>.fromOpaque(raw).takeUnretainedValue() as Data
        guard !data.isEmpty else { return nil }
        var deadKeyState: UInt32 = 0
        var length = 0
        var chars = [UniChar](repeating: 0, count: 4)
        let status = data.withUnsafeBytes { bytes -> OSStatus in
            guard let layout = bytes.baseAddress?.assumingMemoryBound(to: UCKeyboardLayout.self)
            else { return OSStatus(paramErr) }
            return UCKeyTranslate(
                layout, keyCode, UInt16(kUCKeyActionDisplay), 0, UInt32(LMGetKbdType()),
                OptionBits(1 << kUCKeyTranslateNoDeadKeysBit), &deadKeyState, chars.count,
                &length, &chars)
        }
        guard status == noErr, length > 0 else { return nil }
        let label = String(utf16CodeUnits: chars, count: length)
            .trimmingCharacters(in: .controlCharacters)
        return label.isEmpty ? nil : label
    }
}
