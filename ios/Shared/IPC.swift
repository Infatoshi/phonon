import Foundation
import UIKit

/// The app and the keyboard talk through Darwin notifications (no payload) and small shared values.
/// keyboard -> app: start (mic button down), stop (up), ping. app -> keyboard: pong, result, state.
/// The values live in named pasteboards, which every app and extension of the same team shares, so no App Group
/// entitlement (and no Xcode account login) is needed. One pasteboard per key, so writers never clobber each other.
enum IPC {
    static let start = "sh.phonon.ios.start"
    static let stop = "sh.phonon.ios.stop"
    static let ping = "sh.phonon.ios.ping"
    static let pong = "sh.phonon.ios.pong"
    static let result = "sh.phonon.ios.result"
    static let state = "sh.phonon.ios.state"
    static let level = "sh.phonon.ios.level"  // app -> keyboard, ~15 per second while listening
    static let cancel = "sh.phonon.ios.cancel"  // keyboard -> app: drop the current utterance

    /// shared keys
    static let resultText = "result_text"
    static let resultSeq = "result_seq"
    static let stateText = "state_text"
    static let heartbeat = "heartbeat"
    static let stopAt = "stop_at"
    static let modeKey = "mode"  // the writing mode, set from the keyboard chip or the app
    static let levelValue = "level"
    static let keyboardSeen = "keyboard_seen"  // written by the keyboard with Full Access: setup step 2 is done

    /// Values live as small files in the App Group container. (They used to live in named pasteboards, but iOS blocks
    /// pasteboard access from a backgrounded app, so the heartbeat froze the moment the user left Phonon and the
    /// keyboard kept bouncing back to the app.) Pasteboards stay as the fallback if the group is missing.
    static let group = "group.sh.phonon.ios"
    private static let dir: URL? = {
        guard let base = FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: group) else { return nil }
        let d = base.appendingPathComponent("ipc", isDirectory: true)
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }()
    static var hasGroup: Bool { dir != nil }

    private static func board(_ key: String) -> UIPasteboard? {
        UIPasteboard(name: UIPasteboard.Name("sh.phonon.ios.\(key)"), create: true)
    }

    static func set(_ value: String, _ key: String) {
        if let dir {
            try? Data(value.utf8).write(to: dir.appendingPathComponent(key), options: .atomic)
        } else if !Shared.set(value, key) {
            board(key)?.string = value
        }
    }

    static func string(_ key: String) -> String? {
        if let dir {
            guard let d = try? Data(contentsOf: dir.appendingPathComponent(key)) else { return nil }
            return String(data: d, encoding: .utf8)
        }
        return Shared.get(key) ?? board(key)?.string
    }

    /// Without an App Group, the shared keychain group is the channel that still works from a backgrounded app: the
    /// team's wildcard profile already grants keychain-access-groups TEAMID.*, so no new profile is needed. Items are
    /// readable after first unlock, so the app can write results while it runs in the background.
    enum Shared {
        static let accessGroup = "5FG68476XH.sh.phonon.shared"
        private static func query(_ key: String) -> [String: Any] {
            [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: "sh.phonon.ipc",
             kSecAttrAccount as String: key, kSecAttrAccessGroup as String: accessGroup,
             kSecUseDataProtectionKeychain as String: true]
        }

        @discardableResult
        static func set(_ value: String, _ key: String) -> Bool {
            let data = Data(value.utf8)
            let q = query(key)
            var s = SecItemUpdate(q as CFDictionary, [kSecValueData as String: data] as CFDictionary)
            if s == errSecItemNotFound {
                var add = q
                add[kSecValueData as String] = data
                add[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
                s = SecItemAdd(add as CFDictionary, nil)
            }
            return s == errSecSuccess
        }

        static func get(_ key: String) -> String? {
            var q = query(key)
            q[kSecReturnData as String] = true
            q[kSecMatchLimit as String] = kSecMatchLimitOne
            var out: AnyObject?
            guard SecItemCopyMatching(q as CFDictionary, &out) == errSecSuccess, let d = out as? Data else { return nil }
            return String(data: d, encoding: .utf8)
        }
    }
    static func set(_ value: Double, _ key: String) { set(String(value), key) }
    static func double(_ key: String) -> Double { Double(string(key) ?? "") ?? 0 }

    static func post(_ name: String) {
        CFNotificationCenterPostNotification(
            CFNotificationCenterGetDarwinNotifyCenter(), CFNotificationName(name as CFString), nil, nil, true)
    }

    private static var handlers: [String: () -> Void] = [:]

    /// Calls `handler` on the main queue each time `name` is posted by the other process.
    static func observe(_ name: String, _ handler: @escaping () -> Void) {
        let first = handlers[name] == nil
        handlers[name] = handler
        guard first else { return }
        CFNotificationCenterAddObserver(
            CFNotificationCenterGetDarwinNotifyCenter(), nil,
            { _, _, cfName, _, _ in
                guard let n = cfName?.rawValue as String? else { return }
                DispatchQueue.main.async { IPC.handlers[n]?() }
            },
            name as CFString, nil, .deliverImmediately)
    }

    static func now() -> Double { Date().timeIntervalSince1970 }

    /// The app is alive if its heartbeat (every second while the session runs) is under 3 s old.
    static var sessionAlive: Bool { now() - double(heartbeat) < 3 }
}
