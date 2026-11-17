import SwiftUI

/// The Mac app's ember palette (macos/Sources/MainWindow.swift EmberTheme), shared by the app and the keyboard.
enum Ember {
    static let window = Color(hex: 0x12100E)
    static let panel = Color(hex: 0x1A1512)
    static let panelRaised = Color(hex: 0x231C17)
    static let line = Color(hex: 0x2A221C)
    static let ink = Color(hex: 0xEEE6DC)
    static let soft = Color(hex: 0xCDBFB2)
    static let mute = Color(hex: 0x8F8377)
    static let dim = Color(hex: 0x6F6559)
    static let ember = Color(hex: 0xE8692C)
    static let ember2 = Color(hex: 0xF08A4B)
    static let emberSoft = Color(hex: 0x3A1E10)
    static let done = Color(hex: 0x8FE0A8)
    static let warn = Color(hex: 0xF2C14E)
}

extension Color {
    init(hex: UInt32) {
        self.init(red: Double((hex >> 16) & 0xFF) / 255, green: Double((hex >> 8) & 0xFF) / 255,
                  blue: Double(hex & 0xFF) / 255)
    }
}

/// Writing modes, in the order the app and keyboard show them.
enum WritingMode: String, CaseIterable, Identifiable {
    case minimal, casual, formal, prompt
    var id: String { rawValue }
    var title: String { rawValue.capitalized }
    var blurb: String {
        switch self {
        case .minimal: return "Notes and documents. Your words, cleaned up."
        case .casual: return "Chat messages. Relaxed, lowercase is fine."
        case .formal: return "Email and documents. Full sentences."
        case .prompt: return "Instructions to an AI. Clear and direct."
        }
    }
    var symbol: String {
        switch self {
        case .minimal: return "text.alignleft"
        case .casual: return "bubble.left"
        case .formal: return "envelope"
        case .prompt: return "sparkles"
        }
    }

    /// Infer writing mode from the host app the keyboard is focused in.
    static func forHostBundleID(_ bundleID: String?) -> WritingMode {
        guard let id = bundleID?.trimmingCharacters(in: .whitespacesAndNewlines), !id.isEmpty else {
            return .minimal
        }
        let lower = id.lowercased()
        if promptBundles.contains(lower) { return .prompt }
        if formalBundles.contains(lower) { return .formal }
        if casualBundles.contains(lower) { return .casual }
        return .minimal
    }

    private static let casualBundles: Set<String> = [
        "com.apple.mobilesms",
        "net.whatsapp.whatsapp",
        "net.whatsapp.whatsappsmb",
        "com.tinyspeck.chatlyio",
        "com.tinyspeck.slackmacgap",
        "com.microsoft.skype.teams",
        "com.hammerandchisel.discord",
        "ph.telegra.telegraph",
        "org.whispersystems.signal",
        "com.facebook.messenger",
        "com.burbn.instagram",
        "com.beeper.beeper",
        "com.tencent.xin",
        "jp.naver.line",
        "com.viber",
    ]

    private static let formalBundles: Set<String> = [
        "com.apple.mobilemail",
        "com.google.gmail",
        "com.microsoft.office.outlook",
        "com.readdle.smartemail",
        "com.superhuman.mail",
        "com.basecamp.hey",
        "ch.protonmail.protonmail",
        "com.fastmail.fastmail",
    ]

    private static let promptBundles: Set<String> = [
        "com.openai.chat",
        "com.anthropic.claude",
        "ai.perplexity.app",
        "com.google.gemini",
        "com.microsoft.copilot",
        "ai.x.grok",
        "com.termius.mac",
        "com.termius.ios",
        "com.crystalnix.serverauditor",
        "sh.blink.blink",
        "com.panic.prompt",
        "com.palantir.workingcopy",
    ]
}
