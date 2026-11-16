import AppKit
import ApplicationServices
import os

// Which page a browser is showing, read only from the focused element.
//
// Screen context never reads a background browser window: its tab cannot be
// identified without walking every window's accessibility tree, and the
// context lane measured screen context to be worth at most about half a WER
// point on this user's data. A browser window is read only while it is the
// focused window and only when its top-level page is known: walk up from the
// focused element to the OUTERMOST AXWebArea (an iframe's web area, such as a
// payment or captcha frame, is nested inside the page's) and require an
// http or https URL. Any other scheme (devtools://, chrome://, file://,
// about:) or no URL at all counts as private.
//
// The walk fails closed: the page is known only when the walk positively
// reaches the element's window (or the application) within the depth cap.
// A parent or role read that errors or times out, a root reached without a
// window, or the cap, all mean "unknown", which is private; otherwise an
// iframe's public host could stand in for a private top page.

/// One accessibility read: a value, a definite absence (the attribute has
/// no value, as for the root's parent), or a failure (an error or a
/// timeout), which callers treat as unknown.
enum AXRead<Value> {
    case value(Value)
    case absent
    case failed
}

/// The view of an accessibility tree the page walks use. Lets them be
/// tested on a fake tree and their calls counted.
protocol AccessibilityTree {
    associatedtype Element
    func parent(of element: Element) -> AXRead<Element>
    func role(of element: Element) -> AXRead<String>
    func children(of element: Element) -> AXRead<[Element]>
    func subrole(of element: Element) -> String?
    func title(of element: Element) -> String?
    func url(of element: Element) -> URL?
    /// The window server's id for a window element, if it can be found.
    func windowID(of window: Element) -> UInt32?
}

enum BrowserPage {
    /// Ancestors visited at most; a page is never this deep.
    static let maxDepth = 80
    /// Children lists read at most when looking for a window's own page.
    static let windowSearchBudget = 300

    /// Where an element sits: the outermost web area above it (nil when it
    /// is outside any page, like the address bar) and its window (nil when
    /// the walk reached the application without one).
    struct Placement<Element> {
        let webArea: Element?
        let window: Element?
    }

    /// Walk up from `element`. Nil unless the walk positively reached an
    /// AXWindow or AXApplication within `maxDepth`: an AX error, a timeout,
    /// a root without a window or the cap make the page unknown. Only
    /// parents are visited, never children.
    static func placement<Tree: AccessibilityTree>(of element: Tree.Element, in tree: Tree)
        -> Placement<Tree.Element>?
    {
        var node = element
        var outermost: Tree.Element?
        for _ in 0..<maxDepth {
            guard case let .value(role) = tree.role(of: node) else { return nil }
            switch role {
            case "AXWindow": return Placement(webArea: outermost, window: node)
            case "AXApplication": return Placement(webArea: outermost, window: nil)
            case "AXWebArea": outermost = node
            default: break
            }
            guard case let .value(parent) = tree.parent(of: node) else { return nil }
            node = parent
        }
        return nil
    }

    /// The URL of the top-level page that holds `element`, when the walk
    /// is complete (see `placement`).
    static func topLevelURL<Tree: AccessibilityTree>(from element: Tree.Element, in tree: Tree)
        -> URL?
    {
        guard let webArea = placement(of: element, in: tree)?.webArea else { return nil }
        return tree.url(of: webArea)
    }

    /// The window's own page, for a focus outside any page (the address
    /// bar, the toolbar): the top-level web areas under `window`, never
    /// descending into one (so no iframe) nor into controls. The single
    /// http(s) page among them; nil when there is none or more than one
    /// (a split view), or when a read failed or the budget ran out.
    static func windowPageURL<Tree: AccessibilityTree>(of window: Tree.Element, in tree: Tree)
        -> URL?
    {
        // Tab groups are searched: Safari nests the page under AXSplitGroup >
        // AXTabGroup > AXGroup > AXScrollArea > AXWebArea.
        let leafRoles: Set<String> = [
            "AXButton", "AXRadioButton", "AXCheckBox", "AXTextField", "AXTextArea",
            "AXStaticText", "AXImage", "AXPopUpButton", "AXMenuButton", "AXMenuBar", "AXMenu",
            "AXToolbar", "AXComboBox", "AXSlider", "AXScrollBar",
        ]
        var queue = [window]
        var next = 0
        var pages = [URL]()
        while next < queue.count {
            guard next < windowSearchBudget else { return nil }
            let node = queue[next]
            next += 1
            let children: [Tree.Element]
            switch tree.children(of: node) {
            case .value(let found): children = found
            case .absent: children = []
            case .failed: return nil
            }
            for child in children {
                guard case let .value(role) = tree.role(of: child) else { return nil }
                if role == "AXWebArea" {
                    if let url = tree.url(of: child), pageHost(url) != nil { pages.append(url) }
                } else if !leafRoles.contains(role) {
                    queue.append(child)
                }
            }
        }
        return pages.count == 1 ? pages[0] : nil
    }

    /// The host of a web page, only for http and https; anything else
    /// (devtools, chrome, file, about, extensions) has no host here and is
    /// therefore private.
    static func pageHost(_ url: URL?) -> String? {
        guard let url, let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https"
        else { return nil }
        return ContextPolicy.host(of: url)
    }

    /// What one walk up from the focused element tells: the top-level page
    /// URL (http or https only) and, when that page is in a standard window
    /// whose window server id is known, that id.
    struct Focus: Equatable {
        var url: URL?
        var pid: pid_t?
        var windowID: UInt32?

        /// The window and host screen context may read, if any.
        var page: (windowID: UInt32, host: String)? {
            guard let windowID, let host = BrowserPage.pageHost(url) else { return nil }
            return (windowID, host)
        }
    }

    /// One placement walk from `element` (see `placement`). The window id
    /// is given only for a standard window (not an autofill or permission
    /// popup) holding a page with an http(s) host; the caller reads only
    /// that window id.
    static func focus<Tree: AccessibilityTree>(from element: Tree.Element, in tree: Tree)
        -> Focus
    {
        guard let placement = placement(of: element, in: tree), let webArea = placement.webArea,
            let url = tree.url(of: webArea), pageHost(url) != nil
        else { return Focus() }
        var focus = Focus(url: url)
        if let window = placement.window, tree.subrole(of: window) == "AXStandardWindow" {
            focus.windowID = tree.windowID(of: window)
        }
        return focus
    }

    /// The window a browser's focused element is in and that page's host,
    /// for screen context (see `focus`).
    static func focusedWindow<Tree: AccessibilityTree>(
        from element: Tree.Element, in tree: Tree
    ) -> (windowID: UInt32, host: String)? {
        focus(from: element, in: tree).page
    }

    /// The system-wide focused element's page, window and pid, from one
    /// walk. Blocking AX.
    static func focusedFocus() -> Focus {
        guard AXIsProcessTrusted(), let element = TextboxAccess.focusedElement() else {
            return Focus()
        }
        var pid: pid_t = 0
        AXUIElementGetPid(element, &pid)
        var focus = focus(from: element, in: LiveAccessibilityTree())
        focus.pid = pid
        return focus
    }

    /// The focused element's top-level page URL (http or https only) in the
    /// system-wide focus, and the pid that owns that element. Blocking AX.
    static func focusedPage() -> (url: URL?, pid: pid_t?) {
        let focus = focusedFocus()
        return (focus.url, focus.pid)
    }

    /// The focused browser window of `pid` and its page host, from the
    /// system-wide focused element (see `focus`). Blocking AX.
    static func focusedWindow(pid: pid_t) -> (windowID: UInt32, host: String)? {
        let focus = focusedFocus()
        guard focus.pid == pid else { return nil }
        return focus.page
    }

    /// The top-level page URL (http or https only) and the title of one
    /// browser's focused window, which stays put while another app is in
    /// front. Both come from the window that holds the app's focused
    /// element; with the focus outside the page (address bar, toolbar), the
    /// URL is that window's own page. Browsers only: the window search is
    /// for pages. Blocking AX.
    static func focusedPage(pid: pid_t) -> (url: URL?, title: String?) {
        guard AXIsProcessTrusted() else { return (nil, nil) }
        let app = AXUIElementCreateApplication(pid)
        AXUIElementSetMessagingTimeout(app, TextboxAccess.messagingTimeout)
        let tree = LiveAccessibilityTree()
        var focused: CFTypeRef?
        if AXUIElementCopyAttributeValue(app, kAXFocusedUIElementAttribute as CFString, &focused)
            == .success, let focused, CFGetTypeID(focused) == AXUIElementGetTypeID()
        {
            let element = unsafeBitCast(focused, to: AXUIElement.self)
            return page(around: element, in: tree)
        }
        // No focused element: the app's focused window decides alone.
        var window: CFTypeRef?
        guard
            AXUIElementCopyAttributeValue(app, kAXFocusedWindowAttribute as CFString, &window)
                == .success, let window, CFGetTypeID(window) == AXUIElementGetTypeID()
        else { return (nil, nil) }
        return page(around: unsafeBitCast(window, to: AXUIElement.self), in: tree)
    }

    /// The page URL and window title around an element (see
    /// `focusedPage(pid:)`); nil for both when the walk is incomplete.
    static func page<Tree: AccessibilityTree>(around element: Tree.Element, in tree: Tree)
        -> (url: URL?, title: String?)
    {
        guard let placement = placement(of: element, in: tree) else { return (nil, nil) }
        let title = placement.window.flatMap { tree.title(of: $0) }
        let url: URL?
        if let webArea = placement.webArea {
            url = tree.url(of: webArea)
        } else if let window = placement.window {
            url = windowPageURL(of: window, in: tree)
        } else {
            url = nil
        }
        return (pageHost(url) == nil ? nil : url, title)
    }
}

/// A window id by frame and title, for when the accessibility window cannot
/// name its window server id: exactly one same-app, normal-level window with
/// the same frame (within a point) and title, else none.
enum WindowMatch {
    static func windowID(
        frame: CGRect?, title: String?, pid: pid_t, candidates: [ScreenWindow]
    ) -> UInt32? {
        guard let frame else { return nil }
        let matches = candidates.filter { window in
            window.pid == pid && window.layer == 0
                && abs(window.frame.minX - frame.minX) <= 1
                && abs(window.frame.minY - frame.minY) <= 1
                && abs(window.frame.width - frame.width) <= 1
                && abs(window.frame.height - frame.height) <= 1
                && (window.title ?? "") == (title ?? "")
        }
        return matches.count == 1 ? matches[0].id : nil
    }
}

/// `_AXUIElementGetWindow`, the window server id of an accessibility window.
/// Private to ApplicationServices, so it is looked up at run time with the
/// C calling convention; nil if a macOS release drops it, and then windows
/// are matched by frame and title instead.
enum AXWindowNumber {
    typealias Function = @convention(c) (AXUIElement, UnsafeMutablePointer<CGWindowID>) -> AXError

    static let function: Function? = {
        // RTLD_DEFAULT: every image loaded in the process.
        guard let symbol = dlsym(UnsafeMutableRawPointer(bitPattern: -2), "_AXUIElementGetWindow")
        else { return nil }
        return unsafeBitCast(symbol, to: Function.self)
    }()
}

/// The real accessibility tree.
struct LiveAccessibilityTree: AccessibilityTree {
    private func read(_ element: AXUIElement, _ attribute: String) -> AXRead<CFTypeRef> {
        AXUIElementSetMessagingTimeout(element, TextboxAccess.messagingTimeout)
        var value: CFTypeRef?
        switch AXUIElementCopyAttributeValue(element, attribute as CFString, &value) {
        case .success:
            guard let value else { return .absent }
            return .value(value)
        case .noValue, .attributeUnsupported:
            return .absent
        default:
            return .failed
        }
    }

    private func element(_ value: CFTypeRef) -> AXUIElement? {
        CFGetTypeID(value) == AXUIElementGetTypeID()
            ? unsafeBitCast(value, to: AXUIElement.self) : nil
    }

    func parent(of element: AXUIElement) -> AXRead<AXUIElement> {
        switch read(element, kAXParentAttribute) {
        case .value(let value):
            guard let parent = self.element(value) else { return .failed }
            return .value(parent)
        case .absent: return .absent
        case .failed: return .failed
        }
    }

    func role(of element: AXUIElement) -> AXRead<String> {
        switch read(element, kAXRoleAttribute) {
        case .value(let value):
            guard let role = value as? String else { return .failed }
            return .value(role)
        case .absent: return .absent
        case .failed: return .failed
        }
    }

    func children(of element: AXUIElement) -> AXRead<[AXUIElement]> {
        switch read(element, kAXChildrenAttribute) {
        case .value(let value):
            guard let list = value as? [AnyObject] else { return .failed }
            return .value(list.compactMap { self.element($0) })
        case .absent: return .absent
        case .failed: return .failed
        }
    }

    func subrole(of element: AXUIElement) -> String? {
        TextboxAccess.string(element, kAXSubroleAttribute)
    }

    func url(of element: AXUIElement) -> URL? {
        TextboxAccess.url(element, "AXURL")
    }

    func title(of element: AXUIElement) -> String? {
        TextboxAccess.string(element, kAXTitleAttribute)
    }

    /// The window server id: from the accessibility window itself, else by
    /// frame and title among the app's on-screen windows.
    func windowID(of window: AXUIElement) -> UInt32? {
        var id = CGWindowID(0)
        if let function = AXWindowNumber.function, function(window, &id) == .success, id != 0 {
            return id
        }
        var pid: pid_t = 0
        AXUIElementGetPid(window, &pid)
        return WindowMatch.windowID(
            frame: frame(of: window), title: title(of: window), pid: pid,
            candidates: ScreenContextCapture.onScreenWindows())
    }

    private func frame(of window: AXUIElement) -> CGRect? {
        guard case let .value(position) = read(window, kAXPositionAttribute),
            case let .value(size) = read(window, kAXSizeAttribute),
            CFGetTypeID(position) == AXValueGetTypeID(), CFGetTypeID(size) == AXValueGetTypeID()
        else { return nil }
        var origin = CGPoint.zero
        var extent = CGSize.zero
        guard AXValueGetValue(unsafeBitCast(position, to: AXValue.self), .cgPoint, &origin),
            AXValueGetValue(unsafeBitCast(size, to: AXValue.self), .cgSize, &extent)
        else { return nil }
        return CGRect(origin: origin, size: extent)
    }
}

/// Browsers: a known list plus every app Launch Services lists as a handler
/// for https, so a browser missing from the list is still treated as one
/// and fails closed rather than open.
enum BrowserCatalog {
    static let knownPrefixes = [
        "com.google.Chrome", "com.brave.Browser", "com.microsoft.edgemac", "company.thebrowser.",
        "com.vivaldi.Vivaldi", "com.operasoftware.Opera", "org.chromium.", "com.apple.Safari",
        "org.mozilla.", "app.zen-browser.zen", "com.kagi.kagimacOS", "com.duckduckgo.macos.browser",
        "org.torproject.torbrowser", "com.pushplaylabs.sidekick", "net.waterfox.waterfox",
        "io.gitlab.librewolf-community", "one.ablaze.floorp", "net.mullvad.mullvadbrowser",
        "ru.yandex.desktop.yandex-browser", "ai.perplexity.comet", "com.openai.atlas",
        "com.sigmaos.sigmaos.macos", "com.electron.min",
    ]

    private static let handlers = OSAllocatedUnfairLock<Set<String>?>(initialState: nil)

    static func isBrowser(_ bundleID: String?, httpsHandlers: Set<String>) -> Bool {
        guard let bundleID, !bundleID.isEmpty else { return false }
        if knownPrefixes.contains(where: { bundleID.hasPrefix($0) }) { return true }
        return httpsHandlers.contains { $0.caseInsensitiveCompare(bundleID) == .orderedSame }
    }

    /// Bundle ids of the apps registered to open https URLs, looked up once.
    static func httpsHandlers() -> Set<String> {
        if let cached = handlers.withLock({ $0 }) { return cached }
        var found = Set<String>()
        if let probe = URL(string: "https://example.com") {
            for app in NSWorkspace.shared.urlsForApplications(toOpen: probe) {
                if let id = Bundle(url: app)?.bundleIdentifier { found.insert(id) }
            }
        }
        let result = found
        handlers.withLock { $0 = result }
        return result
    }
}
