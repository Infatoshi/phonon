import AppKit
import ApplicationServices
import os

/// One Accessibility read of a text field. Holds the field's own text only in
/// memory; nothing here is ever written to disk.
struct TextboxSnapshot {
    let element: AXUIElement
    let pid: pid_t
    let value: String?
    let selection: NSRange?
    let role: String?
    let subrole: String?

    /// Password fields expose no value through AX, but never read them
    /// regardless.
    var isSecure: Bool {
        subrole == (kAXSecureTextFieldSubrole as String) || role == "AXSecureTextField"
    }
}

/// The Accessibility calls behind edit learning, text-box context and the
/// browser URL. Uses only the Accessibility permission Phonon already holds
/// for inserting text.
enum TextboxAccess {
    /// A hung app must not stall the bar: every AX message gives up after this.
    static let messagingTimeout: Float = 0.25

    private static let manualAccessibilityPIDs = OSAllocatedUnfairLock(
        initialState: Set<pid_t>())

    /// Electron apps (Slack, VS Code, Notion) build their accessibility tree
    /// only for a client that asks, through `AXManualAccessibility`; Wispr
    /// sets it the same way. Once per process; apps that are not Electron
    /// ignore it.
    static func enableManualAccessibility(pid: pid_t) {
        guard pid > 0, pid != ProcessInfo.processInfo.processIdentifier else { return }
        let first = manualAccessibilityPIDs.withLock { $0.insert(pid).inserted }
        guard first else { return }
        let app = AXUIElementCreateApplication(pid)
        AXUIElementSetMessagingTimeout(app, messagingTimeout)
        AXUIElementSetAttributeValue(app, "AXManualAccessibility" as CFString, kCFBooleanTrue)
    }

    static func focusedElement() -> AXUIElement? {
        let system = AXUIElementCreateSystemWide()
        AXUIElementSetMessagingTimeout(system, messagingTimeout)
        var value: CFTypeRef?
        guard
            AXUIElementCopyAttributeValue(
                system, kAXFocusedUIElementAttribute as CFString, &value) == .success,
            let value, CFGetTypeID(value) == AXUIElementGetTypeID()
        else { return nil }
        let element = unsafeBitCast(value, to: AXUIElement.self)
        AXUIElementSetMessagingTimeout(element, messagingTimeout)
        return element
    }

    static let textRoles: Set<String> = [
        kAXTextFieldRole as String, kAXTextAreaRole as String, kAXComboBoxRole as String,
        "AXSearchField", "AXWebArea",
    ]

    /// True only when a paste into `pid` now would surely land nowhere: the app has no
    /// focused window (every window closed), or it is Finder with no text field focused
    /// (the desktop or a file list). Anything unclear, such as an app that is slow to
    /// answer or exposes no focused element, counts as a target, so a paste is never
    /// withheld from a field that might take it. Wispr withholds on any non-editable
    /// focus; terminals and canvas apps would trip that here.
    static func pasteHasNoTarget(pid: pid_t, bundleID: String?) -> Bool {
        guard AXIsProcessTrusted(), pid > 0 else { return false }
        // A launcher's field (Spotlight, Raycast) takes keys without its app being frontmost.
        let focused = focusedElement()
        if let focused, textRoles.contains(string(focused, kAXRoleAttribute) ?? "") {
            return false
        }
        let app = AXUIElementCreateApplication(pid)
        AXUIElementSetMessagingTimeout(app, messagingTimeout)
        var window: CFTypeRef?
        let status = AXUIElementCopyAttributeValue(
            app, kAXFocusedWindowAttribute as CFString, &window)
        if status == .noValue { return true }
        // Finder with no text field focused (checked above): the desktop or a file list.
        return bundleID == "com.apple.finder"
    }

    /// The title of an app's focused window. Blocking AX.
    static func focusedWindowTitle(pid: pid_t) -> String? {
        guard AXIsProcessTrusted() else { return nil }
        let app = AXUIElementCreateApplication(pid)
        AXUIElementSetMessagingTimeout(app, messagingTimeout)
        var window: CFTypeRef?
        guard
            AXUIElementCopyAttributeValue(app, kAXFocusedWindowAttribute as CFString, &window)
                == .success, let window, CFGetTypeID(window) == AXUIElementGetTypeID()
        else { return nil }
        let element = unsafeBitCast(window, to: AXUIElement.self)
        AXUIElementSetMessagingTimeout(element, messagingTimeout)
        return string(element, kAXTitleAttribute)
    }

    static func focusedSnapshot() -> TextboxSnapshot? {
        guard AXIsProcessTrusted(), let element = focusedElement() else { return nil }
        return snapshot(of: element)
    }

    static func snapshot(of element: AXUIElement) -> TextboxSnapshot {
        var pid: pid_t = 0
        AXUIElementGetPid(element, &pid)
        let role = string(element, kAXRoleAttribute)
        let subrole = string(element, kAXSubroleAttribute)
        let secure = subrole == (kAXSecureTextFieldSubrole as String) || role == "AXSecureTextField"
        return TextboxSnapshot(
            element: element, pid: pid,
            value: secure ? nil : string(element, kAXValueAttribute),
            selection: secure ? nil : selectedRange(element),
            role: role, subrole: subrole)
    }

    /// The field's text, or nil when it cannot be read right now. `gone` is
    /// true when the element no longer exists (window closed).
    static func readValue(_ element: AXUIElement) -> (value: String?, gone: Bool) {
        var value: CFTypeRef?
        let status = AXUIElementCopyAttributeValue(
            element, kAXValueAttribute as CFString, &value)
        switch status {
        case .success:
            return (value as? String, false)
        case .invalidUIElement:
            return (nil, true)
        default:
            return (nil, false)
        }
    }

    static func isFocused(_ element: AXUIElement) -> Bool {
        guard let focused = focusedElement() else { return false }
        return CFEqual(focused, element)
    }

    static func string(_ element: AXUIElement, _ attribute: String) -> String? {
        var value: CFTypeRef?
        guard
            AXUIElementCopyAttributeValue(element, attribute as CFString, &value) == .success
        else { return nil }
        return value as? String
    }

    static func selectedRange(_ element: AXUIElement) -> NSRange? {
        var value: CFTypeRef?
        guard
            AXUIElementCopyAttributeValue(
                element, kAXSelectedTextRangeAttribute as CFString, &value) == .success,
            let value, CFGetTypeID(value) == AXValueGetTypeID()
        else { return nil }
        var range = CFRange(location: 0, length: 0)
        guard AXValueGetValue(unsafeBitCast(value, to: AXValue.self), .cfRange, &range)
        else { return nil }
        return NSRange(location: range.location, length: range.length)
    }

    /// A URL-valued attribute (`AXURL` of an `AXWebArea`). Chrome, Arc,
    /// Brave, Edge and Safari expose it to an Accessibility client; no
    /// Automation (AppleScript) permission is involved. Which web area to ask
    /// is `BrowserPage`'s job.
    static func url(_ element: AXUIElement, _ attribute: String) -> URL? {
        var value: CFTypeRef?
        guard
            AXUIElementCopyAttributeValue(element, attribute as CFString, &value) == .success,
            let value
        else { return nil }
        if CFGetTypeID(value) == CFURLGetTypeID() {
            return unsafeBitCast(value, to: CFURL.self) as URL
        }
        if let text = value as? String { return URL(string: text) }
        return nil
    }
}
