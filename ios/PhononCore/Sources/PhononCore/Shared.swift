import Foundation

/// Identifiers shared by the app and the keyboard extension.
public enum PhononIDs {
    public static let appGroup = "group.com.abhaymettu.phonon"
    public static let urlScheme = "phonon"
    /// Keyboard to app: a new `KeyboardCommand` is in `command.json`.
    public static let commandNote = "com.abhaymettu.phonon.command"
    /// App to keyboard: `app-state.json` changed.
    public static let stateNote = "com.abhaymettu.phonon.state"
}

/// What the keyboard asks the app to do. Written only by the keyboard.
public struct KeyboardCommand: Codable, Equatable, Sendable {
    public enum Kind: String, Codable, Sendable { case start, stop, cancel, ping }
    public var seq: Int
    public var kind: Kind
    public var sentAt: Date
    /// The host app's keyboard type hint, used to pick a writing mode when set to automatic.
    public var hostHint: String?

    public init(seq: Int, kind: Kind, sentAt: Date = Date(), hostHint: String? = nil) {
        self.seq = seq
        self.kind = kind
        self.sentAt = sentAt
        self.hostHint = hostHint
    }
}

/// The app's session phase, shown by the keyboard.
public enum SessionPhase: String, Codable, Sendable {
    case off          // app not holding the microphone; the keyboard must open the app
    case ready        // warm: microphone session open, not recording
    case recording
    case transcribing
    case refining
    case failed
}

/// One finished dictation.
public struct Dictation: Codable, Equatable, Identifiable, Sendable {
    public var id: String
    public var createdAt: Date
    public var raw: String
    public var text: String
    public var refined: Bool
    public var guardFired: Bool
    public var recovered: Bool
    public var audioSeconds: Double
    public var asrSeconds: Double
    public var refineSeconds: Double
    public var mode: WritingMode?

    public init(id: String, createdAt: Date = Date(), raw: String, text: String, refined: Bool,
                guardFired: Bool = false, recovered: Bool = false, audioSeconds: Double = 0,
                asrSeconds: Double = 0, refineSeconds: Double = 0, mode: WritingMode? = nil) {
        self.id = id
        self.createdAt = createdAt
        self.raw = raw
        self.text = text
        self.refined = refined
        self.guardFired = guardFired
        self.recovered = recovered
        self.audioSeconds = audioSeconds
        self.asrSeconds = asrSeconds
        self.refineSeconds = refineSeconds
        self.mode = mode
    }
}

/// Written only by the app.
public struct AppState: Codable, Equatable, Sendable {
    public var phase: SessionPhase = .off
    /// The last keyboard command the app acted on.
    public var ackSeq: Int = 0
    public var updatedAt: Date = .distantPast
    public var recordingStartedAt: Date?
    public var level: Float = 0
    public var message: String?
    /// The newest finished dictation, for the keyboard to insert.
    public var latest: Dictation?
    /// When the warm microphone session ends unless used again.
    public var warmUntil: Date?
    public var modelsReady: Bool = false

    public init() {}
}

/// Written only by the keyboard.
public struct KeyboardState: Codable, Equatable, Sendable {
    /// Dictation ids the keyboard has inserted (newest last, capped).
    public var inserted: [String] = []
    public var updatedAt: Date = .distantPast
    /// Set when the mic was tapped; a dictation finished after it is inserted automatically.
    /// Kept on disk because iOS may restart the keyboard while Phonon is in front.
    public var awaitingSince: Date?
    public init() {}

    public mutating func markInserted(_ id: String) {
        inserted.removeAll { $0 == id }
        inserted.append(id)
        if inserted.count > 50 { inserted.removeFirst(inserted.count - 50) }
        updatedAt = Date()
    }
}

/// Single-writer JSON files in the App Group container. Each file has exactly one writer
/// (the app or the keyboard), writes are atomic renames, so readers never see a torn file.
public final class SharedStore: @unchecked Sendable {
    public let root: URL

    public init(root: URL) {
        self.root = root
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    /// The App Group container, or nil when the entitlement (or keyboard Full Access) is missing.
    public static func appGroup() -> SharedStore? {
        guard let url = FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: PhononIDs.appGroup) else {
            return nil
        }
        let root = url.appendingPathComponent("Phonon", isDirectory: true)
        let probe = root.appendingPathComponent(".probe")
        do {
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            try Data().write(to: probe, options: .atomic)
        } catch {
            return nil
        }
        return SharedStore(root: root)
    }

    private static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.dateEncodingStrategy = .millisecondsSince1970
        return e
    }()
    private static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .millisecondsSince1970
        return d
    }()

    func url(_ name: String) -> URL { root.appendingPathComponent(name) }

    public func read<T: Decodable>(_ type: T.Type, _ name: String) -> T? {
        guard let data = try? Data(contentsOf: url(name)) else { return nil }
        return try? Self.decoder.decode(T.self, from: data)
    }

    public func write<T: Encodable>(_ value: T, _ name: String) {
        guard let data = try? Self.encoder.encode(value) else { return }
        try? data.write(to: url(name), options: .atomic)
    }

    public var appState: AppState { read(AppState.self, "app-state.json") ?? AppState() }
    public func writeAppState(_ s: AppState) { write(s, "app-state.json") }

    public var command: KeyboardCommand? { read(KeyboardCommand.self, "command.json") }
    public func writeCommand(_ c: KeyboardCommand) { write(c, "command.json") }

    public var keyboardState: KeyboardState { read(KeyboardState.self, "keyboard-state.json") ?? KeyboardState() }
    public func writeKeyboardState(_ s: KeyboardState) { write(s, "keyboard-state.json") }

    public var history: [Dictation] { read([Dictation].self, "history.json") ?? [] }
    public func appendHistory(_ d: Dictation, limit: Int = 30) {
        var h = history.filter { $0.id != d.id }
        h.insert(d, at: 0)
        write(Array(h.prefix(limit)), "history.json")
    }
    public func clearHistory() { write([Dictation](), "history.json") }

    public var settings: SharedSettings { read(SharedSettings.self, "settings.json") ?? SharedSettings() }
    public func writeSettings(_ s: SharedSettings) { write(s, "settings.json") }
}

/// Settings the app writes and the keyboard reads.
public struct SharedSettings: Codable, Equatable, Sendable {
    /// nil means automatic: pick from the host field (email field -> formal, else casual).
    public var writingMode: WritingMode? = nil
    public var refinerEnabled: Bool = true
    /// Minutes the microphone session stays open after the last dictation.
    public var warmMinutes: Int = 5
    /// The speaker's own terms, most important first; becomes the refiner's vocabulary line.
    public var vocabulary: [String] = []
    public var autoInsert: Bool = true
    public init() {}
}

/// Darwin notifications: payload-free cross-process signals. The data rides in the App Group.
public enum DarwinNote {
    public static func post(_ name: String) {
        CFNotificationCenterPostNotification(CFNotificationCenterGetDarwinNotifyCenter(),
                                             CFNotificationName(name as CFString), nil, nil, true)
    }

    private final class Box {
        let handler: () -> Void
        init(_ handler: @escaping () -> Void) { self.handler = handler }
    }
    private static var boxes: [String: Box] = [:]
    private static let lock = NSLock()

    /// Calls `handler` on the main queue whenever `name` is posted by any process.
    public static func observe(_ name: String, handler: @escaping () -> Void) {
        lock.lock()
        let box = Box(handler)
        boxes[name] = box
        lock.unlock()
        let ptr = Unmanaged.passUnretained(box).toOpaque()
        CFNotificationCenterAddObserver(CFNotificationCenterGetDarwinNotifyCenter(), ptr, { _, observer, _, _, _ in
            guard let observer else { return }
            let box = Unmanaged<Box>.fromOpaque(observer).takeUnretainedValue()
            DispatchQueue.main.async { box.handler() }
        }, name as CFString, nil, .deliverImmediately)
    }

    public static func stopObserving(_ name: String) {
        lock.lock()
        defer { lock.unlock() }
        guard let box = boxes.removeValue(forKey: name) else { return }
        CFNotificationCenterRemoveObserver(CFNotificationCenterGetDarwinNotifyCenter(),
                                           Unmanaged.passUnretained(box).toOpaque(),
                                           CFNotificationName(name as CFString), nil)
    }
}

/// Writing mode for a host text field when the user leaves the mode on automatic.
public func automaticMode(keyboardType: Int?, textContentType: String?) -> WritingMode {
    // UIKeyboardType.emailAddress == 7, UITextContentType.emailAddress == "emailAddress"
    if keyboardType == 7 || textContentType == "emailAddress" { return .formal }
    if keyboardType == 3 { return .minimal } // URL field
    return .casual
}
