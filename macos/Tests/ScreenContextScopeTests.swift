import XCTest

@testable import PhononBar

final class ScreenWindowSelectorTests: XCTestCase {
    private let laptop = ScreenDisplay(id: 1, frame: CGRect(x: 0, y: 0, width: 1728, height: 1117))
    private let monitor = ScreenDisplay(
        id: 2, frame: CGRect(x: 1728, y: 0, width: 2560, height: 1440))

    private func window(
        _ id: UInt32, pid: pid_t, bundle: String, _ frame: CGRect, layer: Int = 0,
        title: String? = nil
    ) -> ScreenWindow {
        ScreenWindow(
            id: id, pid: pid, bundleID: bundle, frame: frame, layer: layer, alpha: 1,
            title: title)
    }

    func testEachDisplayGetsTheWindowsThatOccupyIt() {
        let windows = [
            window(9, pid: 1, bundle: "com.apple.dock", laptop.frame, layer: 20),
            window(1, pid: 10, bundle: "com.mitchellh.ghostty", laptop.frame, title: "lapis"),
            window(
                2, pid: 20, bundle: "com.google.Chrome", monitor.frame,
                title: "Pull requests - Google Chrome"),
            window(3, pid: 30, bundle: "com.tinyspeck.slackmacgap", monitor.frame),
            window(4, pid: 40, bundle: "com.apple.Notes", CGRect(x: 10, y: 10, width: 90, height: 40)),
        ]
        let picked = ScreenWindowSelector.visibleWindows(
            windows: windows, displays: [laptop, monitor], excluded: { _ in false })
        XCTAssertEqual(picked[1]?.map(\.id), [1], "the terminal fills the laptop")
        XCTAssertEqual(picked[2]?.map(\.id), [2], "Slack is fully behind Chrome")
    }

    func testAPartlyCoveredWindowStillCounts() {
        let left = CGRect(x: 0, y: 0, width: 864, height: 1117)
        let windows = [
            window(1, pid: 10, bundle: "a", left),
            window(2, pid: 20, bundle: "b", laptop.frame),
            window(3, pid: 30, bundle: "c", left),
        ]
        let picked = ScreenWindowSelector.visibleWindows(
            windows: windows, displays: [laptop], excluded: { _ in false })
        XCTAssertEqual(picked[1]?.map(\.id), [1, 2])
    }

    func testPrivateWindowsAreNeverReadButStillHideWhatIsBehind() {
        let windows = [
            window(1, pid: 10, bundle: "org.whispersystems.signal-desktop", laptop.frame),
            window(2, pid: 20, bundle: "com.apple.mail", laptop.frame),
            window(3, pid: 99, bundle: "dev.phonon.app", monitor.frame),
            window(4, pid: 30, bundle: "com.google.Chrome", monitor.frame),
        ]
        let picked = ScreenWindowSelector.visibleWindows(
            windows: windows, displays: [laptop, monitor],
            excluded: {
                ContextPolicy.isDenied(
                    bundleID: $0.bundleID, host: nil, denylist: ContextPolicy.defaultDenylist)
                    || $0.pid == 99
            })
        XCTAssertEqual(picked[1]?.map(\.id), [], "Mail sits behind Signal and cannot be seen")
        XCTAssertEqual(picked[2]?.map(\.id), [], "Phonon's own window is skipped, Chrome is behind it")
    }

    func testAWindowAcrossTwoDisplaysIsReadOnBoth() {
        let spanning = CGRect(x: 1200, y: 0, width: 1200, height: 900)
        let picked = ScreenWindowSelector.visibleWindows(
            windows: [window(1, pid: 10, bundle: "a", spanning)], displays: [laptop, monitor],
            excluded: { _ in false })
        XCTAssertEqual(picked[1]?.map(\.id), [1])
        XCTAssertEqual(picked[2]?.map(\.id), [1])
    }
}

final class ScreenContextCollectionTests: XCTestCase {
    func testTabSwitchesAccumulateWithoutRepeatingLines() {
        var collection = ScreenContextCollection()
        let first = ScreenWindow(
            id: 7, pid: 20, bundleID: "com.google.Chrome", frame: .zero, layer: 0, alpha: 1,
            title: "cuBLAS docs - Google Chrome")
        XCTAssertTrue(collection.wants(first.captureKey))
        collection.add(keys: [first.captureKey], text: "cuBLAS\ncublasLt matmul\nSearch")
        XCTAssertFalse(collection.wants(first.captureKey), "same tab is not read twice")
        let switched = ScreenWindow(
            id: 7, pid: 20, bundleID: "com.google.Chrome", frame: .zero, layer: 0, alpha: 1,
            title: "CUTLASS - Google Chrome")
        XCTAssertTrue(collection.wants(switched.captureKey), "a new tab in the same window is new")
        let added = collection.add(keys: [switched.captureKey], text: "CUTLASS\nSearch\nGEMM")
        XCTAssertEqual(added, 2)
        XCTAssertEqual(collection.text, "cuBLAS\ncublasLt matmul\nSearch\nCUTLASS\nGEMM")
    }

    func testCapturesAndCharactersAreBounded() {
        var collection = ScreenContextCollection()
        for index in 0..<20 {
            collection.add(keys: ["w\(index)"], text: "line \(index)")
        }
        XCTAssertEqual(collection.captures, ScreenContextCollection.maxCaptures)
        XCTAssertTrue(collection.isFull)
        XCTAssertFalse(collection.wants("w99"))

        var big = ScreenContextCollection()
        let page = (0..<2_000).map { "unique line number \($0)" }.joined(separator: "\n")
        big.add(keys: ["a"], text: page)
        XCTAssertLessThanOrEqual(big.text.count, ScreenContextCollection.maxCharacters)
        XCTAssertTrue(big.isFull)
    }

    func testWindowsCoveredByADisplayReadAreNotReadAgain() {
        var collection = ScreenContextCollection()
        collection.markSeen(["10|1|lapis"])
        XCTAssertFalse(collection.wants("10|1|lapis"))
        XCTAssertEqual(collection.captures, 0)
    }
}

/// A fake accessibility tree that counts every call, so the tests can assert
/// that only the focused element's ancestors are ever visited.
final class FakeTree: AccessibilityTree {
    struct Node {
        let parent: Int?
        let role: String
        let url: URL?
        var children: [Int] = []
        var subrole: String? = nil
        var title: String? = nil
        var windowID: UInt32? = nil
    }
    var nodes: [Int: Node] = [:]
    /// Elements whose reads error out (a timeout, a dead element).
    var failing = Set<Int>()
    var parentCalls = 0
    var roleCalls = 0
    var childrenCalls = 0
    var urlCalls = 0
    var visited = Set<Int>()

    func parent(of element: Int) -> AXRead<Int> {
        parentCalls += 1
        visited.insert(element)
        if failing.contains(element) { return .failed }
        guard let node = nodes[element] else { return .failed }
        return node.parent.map { .value($0) } ?? .absent
    }
    func role(of element: Int) -> AXRead<String> {
        roleCalls += 1
        visited.insert(element)
        if failing.contains(element) { return .failed }
        return nodes[element].map { .value($0.role) } ?? .failed
    }
    func children(of element: Int) -> AXRead<[Int]> {
        childrenCalls += 1
        if failing.contains(element) { return .failed }
        guard let node = nodes[element] else { return .failed }
        return node.children.isEmpty ? .absent : .value(node.children)
    }
    func subrole(of element: Int) -> String? { nodes[element]?.subrole }
    func title(of element: Int) -> String? { nodes[element]?.title }
    func url(of element: Int) -> URL? {
        urlCalls += 1
        return nodes[element]?.url
    }
    func windowID(of window: Int) -> UInt32? { nodes[window]?.windowID }
}

/// Third review: which page a browser shows is read from the focused element
/// alone, up to the OUTERMOST web area, and only http(s) counts.
final class BrowserPageTests: XCTestCase {
    /// window(1) > sidebar web area(9, a side panel's own page, a sibling)
    ///           > page web area(2, discord.com) > iframe web area(3,
    ///             js.stripe.com) > text field(4, focused)
    private func discordWithStripeFrame() -> FakeTree {
        let tree = FakeTree()
        tree.nodes = [
            1: .init(parent: nil, role: "AXWindow", url: nil, children: [9, 2]),
            9: .init(parent: 1, role: "AXWebArea", url: URL(string: "https://github.com/")),
            2: .init(parent: 1, role: "AXWebArea", url: URL(string: "https://discord.com/channels/1")),
            3: .init(parent: 2, role: "AXWebArea", url: URL(string: "https://js.stripe.com/v3/")),
            4: .init(parent: 3, role: "AXTextField", url: nil),
        ]
        return tree
    }

    func testAnIframeOnAPrivatePageStaysPrivate() {
        let tree = discordWithStripeFrame()
        let url = BrowserPage.topLevelURL(from: 4, in: tree)
        XCTAssertEqual(BrowserPage.pageHost(url), "discord.com", "the outermost web area, not the frame")
        let target = DictationTarget.resolve(
            bundleID: "com.google.Chrome", name: "Chrome", url: url,
            denylist: ContextPolicy.defaultDenylist, isBrowser: { _ in true })
        XCTAssertTrue(target.denied)
        // Only ancestors were visited: never the sidebar web area (9).
        XCTAssertEqual(tree.visited, [4, 3, 2, 1])
        XCTAssertFalse(tree.visited.contains(9))
        XCTAssertLessThanOrEqual(tree.parentCalls, 4)
        XCTAssertEqual(tree.urlCalls, 1, "one URL read, the page's")
    }

    /// Fourth review, item 1: a walk that stops before the window (a parent
    /// read that errors or times out) must not let the iframe's public host
    /// stand in for the private page.
    func testAWalkThatStopsEarlyKnowsNoPage() {
        let tree = discordWithStripeFrame()
        tree.nodes[3] = .init(parent: nil, role: "AXWebArea", url: URL(string: "https://js.stripe.com/v3/"))
        XCTAssertNil(BrowserPage.topLevelURL(from: 4, in: tree), "a root reached without a window")
        // A parent read that errors or times out at the frame.
        let erroring = discordWithStripeFrame()
        erroring.failing = [3]
        XCTAssertNil(BrowserPage.topLevelURL(from: 4, in: erroring))
        XCTAssertEqual(erroring.urlCalls, 0)
        // A role read that fails on the page's own web area.
        let roleless = discordWithStripeFrame()
        roleless.failing = [2]
        XCTAssertNil(BrowserPage.topLevelURL(from: 4, in: roleless))
        // The depth cap: a chain longer than the cap never reaches a window.
        let deep = FakeTree()
        deep.nodes[0] = .init(parent: 1, role: "AXWebArea", url: URL(string: "https://js.stripe.com/"))
        for index in 1...100 {
            deep.nodes[index] = .init(parent: index + 1, role: "AXGroup", url: nil)
        }
        deep.nodes[101] = .init(parent: nil, role: "AXWindow", url: nil)
        XCTAssertNil(BrowserPage.topLevelURL(from: 0, in: deep))
        XCTAssertLessThanOrEqual(deep.parentCalls, BrowserPage.maxDepth)
        // The complete walk still finds the page.
        XCTAssertEqual(
            BrowserPage.pageHost(BrowserPage.topLevelURL(from: 4, in: discordWithStripeFrame())),
            "discord.com")
    }

    /// Fourth review, item 2: the window read is the focused element's own
    /// window, a standard one, identified by its window server id.
    func testTheFocusedWindowIsTheElementsOwnStandardWindow() {
        let tree = FakeTree()
        tree.nodes = [
            1: .init(
                parent: nil, role: "AXWindow", url: nil, subrole: "AXStandardWindow",
                title: "Pull requests - Google Chrome", windowID: 501),
            2: .init(parent: 1, role: "AXWebArea", url: URL(string: "https://github.com/pulls")),
            3: .init(parent: 2, role: "AXTextField", url: nil),
            // An autofill or permission popup: its own non-standard window.
            7: .init(
                parent: nil, role: "AXWindow", url: nil, subrole: "AXUnknown", windowID: 900),
            8: .init(parent: 7, role: "AXWebArea", url: URL(string: "https://github.com/")),
            9: .init(parent: 8, role: "AXGroup", url: nil),
        ]
        let page = BrowserPage.focusedWindow(from: 3, in: tree)
        XCTAssertEqual(page?.windowID, 501)
        XCTAssertEqual(page?.host, "github.com")
        XCTAssertNil(BrowserPage.focusedWindow(from: 9, in: tree), "focus in a popup reads nothing")
        // A window whose server id is unknown reads nothing.
        tree.nodes[1]?.windowID = nil
        XCTAssertNil(BrowserPage.focusedWindow(from: 3, in: tree))
        // The CG front window was the popup (900) while the focus is in 501:
        // that pairing is private.
        let popup = ScreenWindow(
            id: 900, pid: 10, bundleID: "com.google.Chrome", frame: .zero, layer: 0, alpha: 1,
            title: nil)
        XCTAssertTrue(
            ContextPolicy.isPrivateWindow(
                popup, denylist: [], ownPID: 1, focusedPage: (501, "github.com"),
                isBrowser: { _ in true }))
    }

    /// Fifth review, item 9: one walk gives the page, its host and its
    /// window; a switched-to window is read only if it is that window.
    func testOneWalkGivesHostAndWindow() {
        let tree = FakeTree()
        tree.nodes = [
            1: .init(
                parent: nil, role: "AXWindow", url: nil, subrole: "AXStandardWindow",
                windowID: 501),
            2: .init(parent: 1, role: "AXWebArea", url: URL(string: "https://github.com/pulls")),
            3: .init(parent: 2, role: "AXTextField", url: nil),
        ]
        let focus = BrowserPage.focus(from: 3, in: tree)
        XCTAssertEqual(focus.url?.host, "github.com")
        XCTAssertEqual(focus.windowID, 501)
        XCTAssertEqual(tree.parentCalls, 2, "a single walk up")
        XCTAssertEqual(tree.urlCalls, 1)
        XCTAssertEqual(ScreenContextCapture.switchedPage(focus, window: 501)?.host, "github.com")
        XCTAssertNil(ScreenContextCapture.switchedPage(focus, window: 900), "a popup in front")
        // A page outside a standard window: its URL, but no window to read.
        tree.nodes[1]?.subrole = "AXFloatingWindow"
        let floating = BrowserPage.focus(from: 3, in: tree)
        XCTAssertEqual(floating.url?.host, "github.com")
        XCTAssertNil(floating.windowID)
        XCTAssertNil(ScreenContextCapture.switchedPage(floating, window: 501))
    }

    /// Fifth review, item 8: the private window-number call is resolved at
    /// run time with the C calling convention.
    func testTheWindowNumberCallResolves() {
        XCTAssertNotNil(AXWindowNumber.function)
    }

    /// Fifth review, item 3: Safari puts the page under a tab group.
    func testSafarisPageUnderATabGroupIsFound() {
        let tree = FakeTree()
        tree.nodes = [
            1: .init(
                parent: nil, role: "AXWindow", url: nil, children: [2, 10],
                subrole: "AXStandardWindow", title: "Meet - abc-defg-hij"),
            10: .init(parent: 1, role: "AXToolbar", url: nil, children: [11]),
            11: .init(parent: 10, role: "AXTextField", url: nil),
            2: .init(parent: 1, role: "AXSplitGroup", url: nil, children: [3]),
            3: .init(parent: 2, role: "AXTabGroup", url: nil, children: [4]),
            4: .init(parent: 3, role: "AXGroup", url: nil, children: [5]),
            5: .init(parent: 4, role: "AXScrollArea", url: nil, children: [6]),
            6: .init(parent: 5, role: "AXWebArea", url: URL(string: "https://meet.google.com/abc-defg-hij")),
        ]
        XCTAssertEqual(BrowserPage.page(around: 11, in: tree).url?.host, "meet.google.com")
    }

    /// Fourth review, item 2: after the capture, a browser window whose focus
    /// moved to another window or tab loses its read; other reads stay.
    func testAReadOfABrowserWindowWhoseTabChangedIsDropped() {
        let reads = [
            ScreenContextCapture.Read(keys: ["a"], windowIDs: [501, 12], text: "browser and notes"),
            ScreenContextCapture.Read(keys: ["b"], windowIDs: [44], text: "other display"),
        ]
        let allowed: (windowID: UInt32, host: String) = (501, "github.com")
        XCTAssertEqual(
            ScreenContextCapture.keptReads(reads, allowed: allowed, now: (501, "github.com")).count, 2)
        XCTAssertEqual(
            ScreenContextCapture.keptReads(reads, allowed: allowed, now: (501, "discord.com"))
                .map(\.text), ["other display"])
        XCTAssertEqual(
            ScreenContextCapture.keptReads(reads, allowed: allowed, now: (777, "github.com"))
                .map(\.text), ["other display"])
        XCTAssertEqual(
            ScreenContextCapture.keptReads(reads, allowed: allowed, now: nil).map(\.text),
            ["other display"])
        XCTAssertEqual(ScreenContextCapture.keptReads(reads, allowed: nil, now: nil).count, 2)
        XCTAssertFalse(ScreenContextCapture.samePage((501, "github.com"), nil))
        XCTAssertTrue(ScreenContextCapture.samePage(nil, nil))
    }

    /// Fourth review, item 2: without the window server id, a window is
    /// matched by frame and title, and only a unique match counts.
    func testWindowsMatchByFrameAndTitleOnlyWhenUnique() {
        let frame = CGRect(x: 100, y: 50, width: 1200, height: 800)
        func window(_ id: UInt32, pid: pid_t = 10, layer: Int = 0, title: String? = "Docs")
            -> ScreenWindow
        {
            ScreenWindow(
                id: id, pid: pid, bundleID: "com.google.Chrome", frame: frame, layer: layer,
                alpha: 1, title: title)
        }
        XCTAssertEqual(
            WindowMatch.windowID(
                frame: frame.offsetBy(dx: 0.5, dy: 0), title: "Docs", pid: 10,
                candidates: [window(1, title: "Other"), window(2), window(3, pid: 11)]), 2)
        XCTAssertNil(
            WindowMatch.windowID(
                frame: frame, title: "Docs", pid: 10, candidates: [window(1), window(2)]),
            "two identical windows: ambiguous")
        XCTAssertNil(
            WindowMatch.windowID(frame: frame, title: "Docs", pid: 10, candidates: [window(4, layer: 3)]))
        XCTAssertNil(WindowMatch.windowID(frame: nil, title: "Docs", pid: 10, candidates: [window(2)]))
    }

    /// Fourth review, item 8: with the focus in the address bar, the meeting
    /// title and conference URL come from the window's own page; split
    /// pages, an iframe-only page or a failed read give no URL.
    func testAFocusInTheAddressBarUsesTheWindowsOwnPage() {
        let tree = FakeTree()
        tree.nodes = [
            1: .init(
                parent: nil, role: "AXWindow", url: nil, children: [10, 12],
                subrole: "AXStandardWindow", title: "Meet - abc-defg-hij - Google Chrome"),
            10: .init(parent: 1, role: "AXToolbar", url: nil, children: [11]),
            11: .init(parent: 10, role: "AXTextField", url: nil),  // the omnibox, focused
            12: .init(parent: 1, role: "AXGroup", url: nil, children: [13]),
            13: .init(
                parent: 12, role: "AXWebArea", url: URL(string: "https://meet.google.com/abc-defg-hij"),
                children: [14]),
            14: .init(parent: 13, role: "AXWebArea", url: URL(string: "https://accounts.google.com/frame")),
        ]
        let page = BrowserPage.page(around: 11, in: tree)
        XCTAssertEqual(page.url?.host, "meet.google.com")
        XCTAssertEqual(page.title, "Meet - abc-defg-hij - Google Chrome")
        XCTAssertFalse(tree.visited.contains(14), "never descends into a page")
        let info = MeetingDetector.recordableWindowInfo(
            title: page.title, url: page.url, bundleID: "com.google.Chrome", denylist: [])
        XCTAssertEqual(info.conferenceURL, "https://meet.google.com/abc-defg-hij")
        XCTAssertNotNil(info.title)

        // Two pages side by side: which one is unknown.
        tree.nodes[12] = .init(parent: 1, role: "AXGroup", url: nil, children: [13, 15])
        tree.nodes[15] = .init(parent: 12, role: "AXWebArea", url: URL(string: "https://discord.com/app"))
        XCTAssertNil(BrowserPage.page(around: 11, in: tree).url)
        // DevTools beside the page is not a second page.
        tree.nodes[15] = .init(
            parent: 12, role: "AXWebArea", url: URL(string: "devtools://devtools/bundled/inspector.html"))
        XCTAssertEqual(BrowserPage.page(around: 11, in: tree).url?.host, "meet.google.com")
        // A failed read anywhere in the search: unknown.
        tree.failing = [12]
        XCTAssertNil(BrowserPage.page(around: 11, in: tree).url)
        // And a focus walk that fails gives neither title nor URL.
        tree.failing = [10]
        let failed = BrowserPage.page(around: 11, in: tree)
        XCTAssertNil(failed.url)
        XCTAssertNil(failed.title)
    }

    func testDevToolsAndOtherSchemesAreUnknownAndSoPrivate() {
        let tree = FakeTree()
        tree.nodes = [
            1: .init(parent: nil, role: "AXWindow", url: nil),
            2: .init(parent: 1, role: "AXWebArea", url: URL(string: "devtools://devtools/bundled/inspector.html")),
            3: .init(parent: 2, role: "AXTextField", url: nil),
        ]
        let url = BrowserPage.topLevelURL(from: 3, in: tree)
        XCTAssertNil(BrowserPage.pageHost(url))
        for text in ["chrome://settings", "file:///Users/me/a.html", "about:blank", "chrome-extension://x/y"] {
            XCTAssertNil(BrowserPage.pageHost(URL(string: text)), text)
        }
        XCTAssertEqual(BrowserPage.pageHost(URL(string: "http://intranet.example.com/x")), "intranet.example.com")
        let target = DictationTarget.resolve(
            bundleID: "com.google.Chrome", name: "Chrome", url: url, denylist: [],
            isBrowser: { _ in true })
        XCTAssertTrue(target.denied, "DevTools focused: nothing is read")
    }

    func testAFocusOutsideAnyPageHasNoURL() {
        let tree = FakeTree()
        tree.nodes = [
            1: .init(parent: nil, role: "AXWindow", url: nil),
            2: .init(parent: 1, role: "AXTextField", url: nil),  // the address bar
        ]
        XCTAssertNil(BrowserPage.topLevelURL(from: 2, in: tree))
        XCTAssertEqual(tree.urlCalls, 0)
    }

    func testAnUnknownHTTPSHandlerFailsClosed() {
        let handlers: Set<String> = ["com.example.NicheBrowser"]
        XCTAssertTrue(BrowserCatalog.isBrowser("com.example.NicheBrowser", httpsHandlers: handlers))
        XCTAssertTrue(BrowserCatalog.isBrowser("com.kagi.kagimacOS", httpsHandlers: []))
        XCTAssertTrue(BrowserCatalog.isBrowser("org.torproject.torbrowser", httpsHandlers: []))
        XCTAssertFalse(BrowserCatalog.isBrowser("com.apple.Notes", httpsHandlers: handlers))
        let target = DictationTarget.resolve(
            bundleID: "com.example.NicheBrowser", name: "Niche", url: nil, denylist: [],
            isBrowser: { BrowserCatalog.isBrowser($0, httpsHandlers: handlers) })
        XCTAssertTrue(target.denied, "an unknown browser with no page host reads nothing")
    }

    func testASwitchedToNonPrivateBrowserTabIsRead() {
        let window = ScreenWindow(
            id: 77, pid: 10, bundleID: "com.google.Chrome", frame: .zero, layer: 0, alpha: 1,
            title: "Pull requests - Google Chrome")
        // The pre-check on a switch is app-level only...
        XCTAssertFalse(ContextPolicy.isPrivateApp(window, denylist: ContextPolicy.defaultDenylist, ownPID: 1))
        // ...and once the switch has focused it, its page decides.
        XCTAssertFalse(
            ContextPolicy.isPrivateWindow(
                window, denylist: ContextPolicy.defaultDenylist, ownPID: 1,
                focusedPage: (77, "github.com"), isBrowser: { _ in true }))
        XCTAssertTrue(
            ContextPolicy.isPrivateWindow(
                window, denylist: ContextPolicy.defaultDenylist, ownPID: 1,
                focusedPage: (77, "discord.com"), isBrowser: { _ in true }))
    }
}

/// Review item 2: a window read in flight when the target turns out to be
/// private must not add its text afterwards.
final class ScreenReadLedgerTests: XCTestCase {
    func testAResetInvalidatesReadsInFlight() {
        var ledger = ScreenReadLedger()
        let early = ledger.begin()
        XCTAssertEqual(ledger.pending, 1)
        ledger.reset()  // the target resolved as private
        XCTAssertEqual(ledger.pending, 0)
        XCTAssertFalse(ledger.finish(early), "its OCR text is dropped")
        XCTAssertFalse(ledger.isCurrent(early))
        let later = ledger.begin()
        XCTAssertTrue(ledger.finish(later))
        XCTAssertEqual(ledger.pending, 0)
    }
}

final class ScreenContextWaitTests: XCTestCase {
    func testTheCorrectionWaitsAtMostTheBound() {
        XCTAssertTrue(ScreenContextWait.shouldSubmit(pendingReads: 0, initialDone: true, waited: 0))
        XCTAssertFalse(ScreenContextWait.shouldSubmit(pendingReads: 1, initialDone: true, waited: 0.1))
        XCTAssertFalse(ScreenContextWait.shouldSubmit(pendingReads: 0, initialDone: false, waited: 0.29))
        XCTAssertTrue(ScreenContextWait.shouldSubmit(pendingReads: 2, initialDone: false, waited: 0.3))
        XCTAssertEqual(ScreenContextWait.afterFinalTranscript, 0.3)
    }
}
