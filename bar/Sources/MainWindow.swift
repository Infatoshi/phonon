import AppKit
import AVFoundation
import CoreImage.CIFilterBuiltins
import SwiftUI
import UniformTypeIdentifiers

enum EmberTheme {
    static let background = Color(red: 0.067, green: 0.047, blue: 0.039)
    static let sidebar = Color(red: 0.094, green: 0.063, blue: 0.049)
    static let surface = Color(red: 0.129, green: 0.086, blue: 0.067)
    static let surfaceRaised = Color(red: 0.165, green: 0.106, blue: 0.075)
    static let border = Color(red: 0.31, green: 0.176, blue: 0.11)
    static let text = Color(red: 1.0, green: 0.957, blue: 0.925)
    static let muted = Color(red: 0.72, green: 0.61, blue: 0.55)
    static let accent = Color(red: 1.0, green: 0.35, blue: 0.12)
    static let accentSoft = Color(red: 0.235, green: 0.102, blue: 0.045)
    static let warm = Color(red: 0.965, green: 0.65, blue: 0.30)
    static let healthy = Color(red: 0.45, green: 0.78, blue: 0.60)

    static let nsBackground = NSColor(
        calibratedRed: 0.067, green: 0.047, blue: 0.039, alpha: 1)

    // The main window's "technical mono" theme (ui_pitch/app.html): the same
    // ember dark palette, SF Pro for text and SF Mono for numbers and data.
    static let window = Color(hex: 0x12100E)
    static let side = Color(hex: 0x0F0D0B)
    static let ink = Color(hex: 0xEEE6DC)
    static let mute = Color(hex: 0x8F8377)
    static let dim = Color(hex: 0x6F6559)
    static let ember = Color(hex: 0xE8692C)
    static let ember2 = Color(hex: 0xF08A4B)
    static let line = Color(hex: 0x1F1A16)
    static let panel = Color(hex: 0x15110E)
    static let panelBorder = Color(hex: 0x221C17)
    static let hover = Color(hex: 0x17130F)
    static let selected = Color(hex: 0x2A1A10)
    static let selectedInk = Color(hex: 0xFFD7BD)
    static let soft = Color(hex: 0xCDBFB2)
    static let done = Color(hex: 0x8FE0A8)
    /// One hue, dim to bright on the dark surface: heatmaps and the calendar.
    static let ramp: [Color] = [0x241B15, 0x4A2715, 0x7A3A18, 0xB4521F, 0xF08A4B].map {
        Color(hex: $0)
    }
}

extension Color {
    init(hex: UInt32) {
        self.init(
            red: Double((hex >> 16) & 0xFF) / 255, green: Double((hex >> 8) & 0xFF) / 255,
            blue: Double(hex & 0xFF) / 255)
    }
}

extension Font {
    /// Numbers and data: SF Mono with tabular digits.
    static func mono(_ size: CGFloat, _ weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .monospaced).monospacedDigit()
    }
}

/// Snappy, never blocking: 120-200 ms everywhere.
enum Motion {
    static let quick = Animation.easeOut(duration: 0.12)
    static let page = Animation.easeOut(duration: 0.16)
    static let open = Animation.snappy(duration: 0.2)
}

enum NativeAppPage: String, CaseIterable, Identifiable {
    case home = "Home"
    case notes = "Notes"
    case insights = "Insights"
    case dictionary = "Dictionary"
    case macros = "Macros"
    case settings = "Settings"

    var id: String { rawValue }
    var symbol: String {
        switch self {
        case .home: return "house"
        case .notes: return "doc.text"
        case .insights: return "chart.bar"
        case .dictionary: return "book.closed"
        case .macros: return "bolt"
        case .settings: return "gearshape"
        }
    }
}

/// App icons by bundle id, from NSWorkspace; a generic app icon when the app
/// is unknown or not installed.
@MainActor
enum AppIcons {
    private static var icons: [String: NSImage] = [:]
    private static var names: [String: String] = [:]
    static let generic: NSImage = NSWorkspace.shared.icon(for: .applicationBundle)

    static func icon(_ bundleID: String?) -> NSImage {
        guard let bundleID, !bundleID.isEmpty else { return generic }
        if let cached = icons[bundleID] { return cached }
        let icon = NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundleID)
            .map { NSWorkspace.shared.icon(forFile: $0.path) } ?? generic
        icons[bundleID] = icon
        return icon
    }

    static func name(_ bundleID: String?, fallback: String? = nil) -> String {
        if let fallback, !fallback.isEmpty { return fallback }
        guard let bundleID, !bundleID.isEmpty else { return "Unknown app" }
        if let cached = names[bundleID] { return cached }
        let name = NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundleID)
            .map {
                FileManager.default.displayName(atPath: $0.path)
                    .replacingOccurrences(of: ".app", with: "")
            } ?? bundleID
        names[bundleID] = name
        return name
    }
}

struct AppIconView: View {
    let bundleID: String?
    var size: CGFloat = 20
    var body: some View {
        Image(nsImage: AppIcons.icon(bundleID))
            .resizable()
            .interpolation(.high)
            .frame(width: size, height: size)
    }
}

// MARK: - Window

struct PhononMainView: View {
    @ObservedObject var store: NativeAppStore
    let onSettingsChanged: () -> Void
    let onShowModelStatus: () -> Void
    @State private var page: NativeAppPage = .home
    @State private var showPrivacyChoice = false
    @State private var importRequest: DictationImportRequest?
    private let initialState: InitialState

    /// Where the pages start: everything closed, unless a render asks.
    struct InitialState {
        var openDictation: String?
        var showMore = false
        var advancedOpen = false
    }

    init(
        store: NativeAppStore,
        initialPage: NativeAppPage = .home,
        initialState: InitialState = InitialState(),
        onSettingsChanged: @escaping () -> Void,
        onShowModelStatus: @escaping () -> Void
    ) {
        self.store = store
        self.onSettingsChanged = onSettingsChanged
        self.onShowModelStatus = onShowModelStatus
        self.initialState = initialState
        _page = State(initialValue: initialPage)
    }

    var body: some View {
        HStack(spacing: 0) {
            Sidebar(store: store, page: $page)
                .frame(width: 196)
                .background(EmberTheme.side)
                .overlay(alignment: .trailing) {
                    Rectangle().fill(EmberTheme.panelBorder).frame(width: 1)
                }

            VStack(spacing: 0) {
                PermissionBanner(store: store)
                ZStack {
                    detailView
                        .id(page)
                        .transition(
                            .opacity.combined(with: .offset(y: 3)).animation(Motion.page))
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            .background(EmberTheme.window)
        }
        .frame(width: 980, height: 680)
        .font(.system(size: 13.5))
        .foregroundStyle(EmberTheme.ink)
        .background(EmberTheme.window)
        .tint(EmberTheme.ember)
        .preferredColorScheme(.dark)
        .alert(
            "Phonon",
            isPresented: Binding(
                get: { store.lastError != nil },
                set: { if !$0 { store.lastError = nil } }
            )
        ) {
            Button("OK") { store.lastError = nil }
        } message: {
            Text(store.lastError ?? "Unknown error")
        }
        .sheet(item: $store.permissionGuide) { guide in
            PermissionGuideView(guide: guide) {
                store.permissionGuide = nil
                NSApp.terminate(nil)
            }
        }
        .sheet(isPresented: $showPrivacyChoice, onDismiss: offerImport) {
            PrivacyChoiceView(store: store) {
                showPrivacyChoice = false
                onSettingsChanged()
            }
        }
        .sheet(item: $importRequest) { request in
            DictationImportView(
                store: store, request: request, onSettingsChanged: onSettingsChanged
            ) { importRequest = nil }
        }
        .onAppear {
            showPrivacyChoice = store.needsPrivacyChoice
            store.pruneExpiredRecordings()
            showRequestedPage()
        }
        .onChange(of: store.pageRequest) { _, _ in showRequestedPage() }
    }

    /// First run, right after the privacy choices: offer to import from the
    /// dictation apps found on this Mac. Nothing is read until the user asks.
    private func offerImport() {
        guard store.settings.privacyChoiceMade else { return }
        let apps = store.detectDictationApps()
        if !apps.isEmpty { importRequest = DictationImportRequest(apps: apps, firstRun: true) }
    }

    private func showRequestedPage() {
        guard let requested = store.pageRequest else { return }
        page = requested
        store.pageRequest = nil
    }

    @ViewBuilder private var detailView: some View {
        switch page {
        case .home:
            HomeView(
                store: store, showMore: initialState.showMore,
                openID: initialState.openDictation)
        case .notes: NotesView(store: store)
        case .insights: InsightsView(store: store)
        case .dictionary: DictionaryView(store: store)
        case .macros: MacrosView(store: store)
        case .settings:
            SettingsView(
                store: store, onSettingsChanged: onSettingsChanged,
                onShowModelStatus: onShowModelStatus, advancedOpen: initialState.advancedOpen)
        }
    }
}

struct Sidebar: View {
    @ObservedObject var store: NativeAppStore
    @Binding var page: NativeAppPage

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 9) {
                Circle()
                    .fill(
                        RadialGradient(
                            colors: [EmberTheme.ember2, EmberTheme.ember, Color(hex: 0x9A3A12)],
                            center: .center, startRadius: 0, endRadius: 11)
                    )
                    .frame(width: 22, height: 22)
                Text("Phonon").font(.system(size: 15, weight: .semibold)).tracking(-0.3)
            }
            .padding(.horizontal, 8)
            .padding(.top, 6)
            .padding(.bottom, 16)

            ForEach(NativeAppPage.allCases) { candidate in
                SidebarItem(page: candidate, selected: page == candidate) {
                    withAnimation(Motion.page) { page = candidate }
                }
            }
            Spacer()
            if let update = store.availableUpdate {
                Button {
                    store.onShowUpdate?()
                } label: {
                    Label("Update to \(update.version)", systemImage: "arrow.down.circle.fill")
                        .foregroundStyle(EmberTheme.ember2)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 9)
                        .padding(.vertical, 7)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help("Phonon \(update.version) is out.")
            }
        }
        .padding(10)
        // Clear of the traffic lights in the full-size-content title bar.
        .padding(.top, 24)
    }
}

private struct SidebarItem: View {
    let page: NativeAppPage
    let selected: Bool
    let action: () -> Void
    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Image(systemName: page.symbol)
                    .font(.system(size: 14))
                    .frame(width: 16)
                    .foregroundStyle(selected ? EmberTheme.ember2 : Color(hex: 0xBFB3A6))
                Text(page.rawValue)
                    .foregroundStyle(selected ? EmberTheme.selectedInk : Color(hex: 0xBFB3A6))
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 9)
            .padding(.vertical, 7)
            .background(
                selected ? EmberTheme.selected : (hovering ? Color(hex: 0x18130F) : .clear),
                in: RoundedRectangle(cornerRadius: 8, style: .continuous)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .animation(Motion.quick, value: hovering)
        .animation(Motion.quick, value: selected)
    }
}

struct PageTitle: View {
    let text: String
    init(_ text: String) { self.text = text }
    var body: some View {
        Text(text)
            .font(.system(size: 22, weight: .semibold))
            .tracking(-0.44)
            .padding(.bottom, 16)
    }
}

// MARK: - Shared controls

/// An icon-only button (the mock's `.ib`): 30 pt, hover tint, press shrink.
struct IconButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        IconButtonBody(configuration: configuration)
    }

    private struct IconButtonBody: View {
        let configuration: ButtonStyleConfiguration
        @State private var hovering = false
        @Environment(\.isEnabled) private var isEnabled

        var body: some View {
            configuration.label
                .font(.system(size: 14, weight: .medium))
                .frame(width: 30, height: 30)
                .foregroundStyle(
                    hovering && isEnabled ? EmberTheme.selectedInk : Color(hex: 0xA6988A))
                .background(
                    hovering && isEnabled ? Color(hex: 0x241B14) : .clear,
                    in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                .scaleEffect(configuration.isPressed ? 0.9 : 1)
                .opacity(isEnabled ? 1 : 0.4)
                .contentShape(Rectangle())
                .onHover { hovering = $0 }
                .animation(Motion.quick, value: hovering)
                .animation(.easeOut(duration: 0.08), value: configuration.isPressed)
        }
    }
}

/// Bordered text buttons: `.primary` is the ember one (Export), `.plain` the
/// quiet one (Import).
struct PillButtonStyle: ButtonStyle {
    var primary = false
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 13))
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .foregroundStyle(primary ? EmberTheme.selectedInk : Color(hex: 0xCFC3B6))
            .background(
                (primary ? EmberTheme.selected : Color(hex: 0x1A1613))
                    .opacity(configuration.isPressed ? 0.7 : 1),
                in: RoundedRectangle(cornerRadius: 8, style: .continuous)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(primary ? Color(hex: 0x4A2A16) : Color(hex: 0x2C251F), lineWidth: 1)
            )
            .contentShape(Rectangle())
    }
}

/// Copy: the icon morphs into a check with a ring pulse, then back.
struct CopyIconButton: View {
    let text: String
    var help = "Copy"
    @State private var done = false
    @State private var ring = false
    @State private var resetTask: Task<Void, Never>?

    init(text: String, help: String = "Copy", done: Bool = false) {
        self.text = text
        self.help = help
        _done = State(initialValue: done)
    }

    var body: some View {
        Button {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(text, forType: .string)
            withAnimation(.spring(response: 0.18, dampingFraction: 0.6)) { done = true }
            ring = false
            withAnimation(.easeOut(duration: 0.5)) { ring = true }
            resetTask?.cancel()
            resetTask = Task { @MainActor in
                try? await Task.sleep(for: .milliseconds(1300))
                guard !Task.isCancelled else { return }
                withAnimation(Motion.quick) { done = false }
            }
        } label: {
            ZStack {
                Image(systemName: "doc.on.doc")
                    .opacity(done ? 0 : 1)
                    .scaleEffect(done ? 0.4 : 1)
                Image(systemName: "checkmark")
                    .font(.system(size: 14, weight: .bold))
                    .foregroundStyle(EmberTheme.done)
                    .opacity(done ? 1 : 0)
                    .scaleEffect(done ? 1 : 0.4)
                    .rotationEffect(.degrees(done ? 0 : -20))
            }
        }
        .buttonStyle(IconButtonStyle())
        .background {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(EmberTheme.done.opacity(ring && done ? 0 : 0.4), lineWidth: ring ? 1 : 6)
                .scaleEffect(ring && done ? 1.5 : 1)
                .opacity(done ? 1 : 0)
                .allowsHitTesting(false)
        }
        .help(help)
        .accessibilityLabel(help)
    }
}

/// Wraps its children onto lines, left to right (the dictionary chips).
struct FlowLayout: Layout {
    var spacing: CGFloat = 8

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? .infinity
        let rows = arrange(width: width, subviews: subviews)
        let height = rows.last.map { $0.y + $0.height } ?? 0
        return CGSize(width: proposal.width ?? rows.map(\.width).max() ?? 0, height: height)
    }

    func placeSubviews(
        in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()
    ) {
        for row in arrange(width: bounds.width, subviews: subviews) {
            for (index, x, size) in row.items {
                subviews[index].place(
                    at: CGPoint(x: bounds.minX + x, y: bounds.minY + row.y),
                    proposal: ProposedViewSize(size))
            }
        }
    }

    private struct Row {
        var y: CGFloat
        var height: CGFloat = 0
        var width: CGFloat = 0
        var items: [(Int, CGFloat, CGSize)] = []
    }

    private func arrange(width: CGFloat, subviews: Subviews) -> [Row] {
        var rows: [Row] = []
        var row = Row(y: 0)
        for index in subviews.indices {
            var size = subviews[index].sizeThatFits(.unspecified)
            size.width = min(size.width, width)
            if !row.items.isEmpty, row.width + spacing + size.width > width {
                rows.append(row)
                row = Row(y: row.y + row.height + spacing)
            }
            let x = row.items.isEmpty ? 0 : row.width + spacing
            row.items.append((index, x, size))
            row.width = x + size.width
            row.height = max(row.height, size.height)
        }
        if !row.items.isEmpty { rows.append(row) }
        return rows
    }
}

/// When a dictation happened, as the Home feed shows it: "9:41 pm" today,
/// then "Yesterday", a weekday for this week, a date before that.
enum FeedTime {
    static func label(_ date: Date, now: Date = Date(), calendar: Calendar = .current) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.amSymbol = "am"
        formatter.pmSymbol = "pm"
        if calendar.isDate(date, inSameDayAs: now) {
            formatter.dateFormat = "h:mm a"
        } else if let yesterday = calendar.date(byAdding: .day, value: -1, to: now),
            calendar.isDate(date, inSameDayAs: yesterday)
        {
            return "Yesterday"
        } else if now.timeIntervalSince(date) < 6 * 86_400 {
            formatter.dateFormat = "EEE"
        } else {
            formatter.dateFormat = "MMM d"
        }
        return formatter.string(from: date)
    }

    static func duration(seconds: Double) -> String {
        let total = Int(seconds.rounded())
        if total >= 3_600 { return "\(total / 3_600) h \(total % 3_600 / 60) min" }
        if total >= 60 { return "\(total / 60) min" }
        return "\(total) s"
    }

    static func clock(_ seconds: Double) -> String {
        let total = max(0, Int(seconds))
        if total >= 3_600 {
            return String(format: "%d:%02d:%02d", total / 3_600, total % 3_600 / 60, total % 60)
        }
        return String(format: "%d:%02d", total / 60, total % 60)
    }
}

// MARK: - Home

struct HomeView: View {
    @ObservedObject var store: NativeAppStore
    @State private var showMore: Bool
    @State private var openID: String?

    init(store: NativeAppStore, showMore: Bool = false, openID: String? = nil) {
        self.store = store
        _showMore = State(initialValue: showMore)
        _openID = State(initialValue: openID)
    }

    private var recent: [NativeHistoryItem] {
        Array(store.history.lazy.filter { !$0.displayText.isEmpty }.prefix(5))
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                PageTitle("Home")
                HStack(alignment: .firstTextBaseline, spacing: 10) {
                    Text("\(store.usage.wordsPerMinute)")
                        .font(.mono(40, .semibold))
                        .tracking(-1.2)
                    Text("words per minute, on average").foregroundStyle(EmberTheme.mute)
                }
                .padding(.top, 2)
                .padding(.bottom, 20)
                .help(
                    "Words dictated divided by speaking time, over every dictation kept in "
                        + "local history.")

                if store.settings.websocketAPI {
                    WebSocketActivityCard(store: store)
                        .padding(.bottom, 22)
                }

                if recent.isEmpty {
                    Text("Your dictations show up here.")
                        .foregroundStyle(EmberTheme.mute)
                        .padding(.vertical, 12)
                } else {
                    ForEach(Array(recent.enumerated()), id: \.element.id) { index, item in
                        if index < 3 || showMore {
                            HomeDictationRow(
                                item: item,
                                open: Binding(
                                    get: { openID == item.id },
                                    set: { openID = $0 ? item.id : nil }))
                            .transition(.opacity.combined(with: .move(edge: .top)))
                        }
                    }
                    if recent.count > 3 {
                        Button {
                            withAnimation(Motion.open) { showMore.toggle() }
                        } label: {
                            Image(systemName: "chevron.down")
                                .font(.system(size: 15, weight: .semibold))
                                .rotationEffect(.degrees(showMore ? 180 : 0))
                                .frame(width: 34, height: 26)
                        }
                        .buttonStyle(IconButtonStyle())
                        .frame(maxWidth: .infinity)
                        .padding(.top, 6)
                        .accessibilityLabel(showMore ? "Show fewer" : "Show more")
                    }
                }
            }
            .padding(.horizontal, 30)
            .padding(.top, 24)
            .padding(.bottom, 24)
        }
        .scrollIndicators(.never)
    }
}

/// Home, with the WebSocket API on: what other devices dictated through this Mac in the
/// last hour, one pixel column per minute, and a green dot while one is dictating.
struct WebSocketActivityCard: View {
    @ObservedObject var store: NativeAppStore

    var body: some View {
        TimelineView(.periodic(from: .now, by: 15)) { context in
            let values = WebSocketActivity.perMinute(store.websocketDictations, now: context.date)
            let summary = WebSocketActivity.summary(store.websocketDictations, now: context.date)
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 8) {
                    Text("Other devices").font(.system(size: 13, weight: .semibold))
                    if store.websocketLive {
                        LiveDot()
                            .help("A device is dictating through this Mac")
                            .transition(.opacity)
                    }
                    Spacer()
                    Text(summaryText(summary))
                        .font(.mono(12))
                        .foregroundStyle(EmberTheme.mute)
                }
                .animation(Motion.quick, value: store.websocketLive)
                PixelGraph(
                    values: values, scale: WebSocketActivity.scale(values), now: context.date,
                    live: store.websocketLive)
                HStack {
                    Text("60 min ago")
                    Spacer()
                    Text("words per minute over the WebSocket API")
                    Spacer()
                    Text("now")
                }
                .font(.mono(10.5))
                .foregroundStyle(EmberTheme.dim)
            }
            .padding(14)
            .background(EmberTheme.panel, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .stroke(EmberTheme.panelBorder, lineWidth: 1))
        }
    }

    private func summaryText(_ summary: (words: Int, rate: Int?)) -> String {
        guard summary.words > 0 else { return "no dictation in the last hour" }
        let words = summary.words == 1 ? "1 word" : "\(summary.words) words"
        guard let rate = summary.rate else { return "\(words) in the last hour" }
        return "\(words) · \(rate) wpm speaking"
    }
}

/// A green dot that breathes.
struct LiveDot: View {
    @State private var on = false

    var body: some View {
        ZStack {
            Circle()
                .fill(LiveDot.green.opacity(0.35))
                .frame(width: 14, height: 14)
                .scaleEffect(on ? 1.25 : 0.5)
                .opacity(on ? 0 : 0.9)
            Circle()
                .fill(LiveDot.green)
                .frame(width: 7, height: 7)
                .shadow(color: LiveDot.green.opacity(0.8), radius: on ? 4 : 1)
        }
        .frame(width: 14, height: 14)
        .onAppear {
            withAnimation(.easeOut(duration: 1.1).repeatForever(autoreverses: false)) { on = true }
        }
        .accessibilityLabel("Live")
    }

    static let green = Color(hex: 0x3DDC84)
}

/// Columns of square pixels, one per value, lit from the bottom like a meter: dim ember
/// low, bright at the top. Hovering a column shows its minute and value.
struct PixelGraph: View {
    let values: [Double]
    let scale: Double
    let now: Date
    var live = false
    static let rows = 10
    @State private var hovered: Int?

    var body: some View {
        GeometryReader { geometry in
            let columns = max(values.count, 1)
            let gap: CGFloat = 2
            let cell = max(2, min(
                (geometry.size.width - gap * CGFloat(columns - 1)) / CGFloat(columns),
                (geometry.size.height - gap * CGFloat(Self.rows - 1)) / CGFloat(Self.rows)))
            let width = cell * CGFloat(columns) + gap * CGFloat(columns - 1)
            ZStack(alignment: .topLeading) {
                Canvas { context, _ in
                    for (column, value) in values.enumerated() {
                        let lit = value <= 0
                            ? 0 : max(1, Int((value / scale * Double(Self.rows)).rounded(.up)))
                        for row in 0..<Self.rows {
                            let x = CGFloat(column) * (cell + gap)
                            let y = CGFloat(Self.rows - 1 - row) * (cell + gap)
                            let rect = CGRect(x: x, y: y, width: cell, height: cell)
                            let color: Color
                            if row < lit {
                                let step = min(
                                    EmberTheme.ramp.count - 1,
                                    1 + row * (EmberTheme.ramp.count - 1) / Self.rows)
                                color = hovered == column ? EmberTheme.selectedInk : EmberTheme.ramp[step]
                            } else if live && column == values.count - 1 && row == 0 {
                                color = LiveDot.green.opacity(0.7)
                            } else {
                                color = Color(hex: 0x1B1612)
                            }
                            context.fill(
                                Path(roundedRect: rect, cornerRadius: min(1.5, cell / 4)),
                                with: .color(color))
                        }
                    }
                }
                .frame(width: width, height: geometry.size.height)
                .contentShape(Rectangle())
                .onContinuousHover { phase in
                    switch phase {
                    case .active(let point):
                        let column = Int(point.x / (cell + gap))
                        hovered = (0..<values.count).contains(column) ? column : nil
                    case .ended:
                        hovered = nil
                    }
                }
                if let hovered {
                    Text(label(hovered))
                        .font(.mono(11))
                        .foregroundStyle(EmberTheme.ink)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background(EmberTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 5))
                        .fixedSize()
                        .offset(
                            x: min(max(0, CGFloat(hovered) * (cell + gap) - 40), max(0, width - 130)),
                            y: -26)
                        .allowsHitTesting(false)
                }
                Text("\(Int(scale))")
                    .font(.mono(9.5))
                    .foregroundStyle(EmberTheme.dim)
                    .offset(x: width + 6, y: -2)
            }
        }
        .frame(height: 74)
        .padding(.trailing, 22)
        .accessibilityElement()
        .accessibilityLabel("Words per minute over the WebSocket API, last 60 minutes")
        .accessibilityValue("\(Int(values.reduce(0, +).rounded())) words")
    }

    private func label(_ column: Int) -> String {
        let minutesAgo = values.count - 1 - column
        let minute = Date(
            timeIntervalSince1970: (floor(now.timeIntervalSince1970 / 60) - Double(minutesAgo)) * 60)
        let time = minute.formatted(date: .omitted, time: .shortened)
        let words = Int(values[column].rounded())
        return "\(time) · \(words) word\(words == 1 ? "" : "s")"
    }
}

/// One dictation: the app it went into, the final text on one line, when,
/// and copy. Clicking opens what was heard with what Phonon wrote on top.
struct HomeDictationRow: View {
    let item: NativeHistoryItem
    @Binding var open: Bool
    @State private var hovering = false

    init(item: NativeHistoryItem, open: Binding<Bool> = .constant(false)) {
        self.item = item
        _open = open
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Rectangle().fill(EmberTheme.line).frame(height: 1)
            HStack(spacing: 12) {
                AppIconView(bundleID: item.metadata.appContext?.bundleID)
                    .help(
                        AppIcons.name(
                            item.metadata.appContext?.bundleID,
                            fallback: item.metadata.appContext?.appName))
                Text(item.displayText)
                    .lineLimit(open ? nil : 1)
                    .truncationMode(.tail)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .fixedSize(horizontal: false, vertical: open)
                Text(FeedTime.label(item.date))
                    .font(.mono(12))
                    .foregroundStyle(Color(hex: 0x7D7166))
                    .lineLimit(1)
                    .fixedSize()
                CopyIconButton(text: item.displayText)
            }
            .padding(.vertical, 6)
            .padding(.horizontal, 4)
            .background(
                hovering ? EmberTheme.hover : .clear,
                in: RoundedRectangle(cornerRadius: 8, style: .continuous)
            )
            .contentShape(Rectangle())
            .onTapGesture { withAnimation(Motion.open) { open.toggle() } }
            .onHover { hovering = $0 }
            .animation(Motion.quick, value: hovering)

            if open {
                HeardAndWrote(
                    heard: item.metadata.rawTranscript, wrote: item.displayText
                )
                .padding(.leading, 38)
                .padding(.trailing, 4)
                .padding(.bottom, 12)
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
    }
}

/// "heard" underneath, "Phonon wrote" layered on top of it.
struct HeardAndWrote: View {
    let heard: String
    let wrote: String

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            layer(label: "HEARD", text: heard.isEmpty ? "No speech" : heard, top: false)
                .padding(.bottom, -14)
            layer(label: "PHONON WROTE", text: wrote, top: true)
                .padding(.leading, 10)
                .shadow(color: .black.opacity(0.45), radius: 7, y: -6)
        }
    }

    private func layer(label: String, text: String, top: Bool) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label)
                .font(.system(size: 10.5, weight: .medium))
                .tracking(0.6)
                .foregroundStyle(top ? Color(hex: 0xB07A55) : EmberTheme.dim)
            Text(text)
                .font(.system(size: top ? 13.5 : 13))
                .foregroundStyle(top ? EmberTheme.ink : Color(hex: 0x8A7D70))
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, 12)
        .padding(.top, top ? 10 : 9)
        .padding(.bottom, top ? 10 : 22)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            top ? Color(hex: 0x1D140E) : EmberTheme.panel,
            in: RoundedRectangle(cornerRadius: 9, style: .continuous)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .stroke(top ? Color(hex: 0x3B2516) : EmberTheme.panelBorder, lineWidth: 1)
        )
    }
}

// MARK: - Dictionary

struct DictionaryView: View {
    @ObservedObject var store: NativeAppStore
    @State private var draft = ""
    @State private var importNote: String?

    /// Typing in the add field also narrows the list, so a term is easy to
    /// find among hundreds.
    private var shown: [NativeDictionaryEntry] {
        let query = draft.trimmingCharacters(in: .whitespaces)
        guard !query.isEmpty else { return store.dictionaryEntries }
        let parts = NativeAppStore.parseFix(query).map { [$0.0, $0.1] } ?? [query]
        return store.dictionaryEntries.filter { entry in
            parts.contains { part in
                entry.phrase.localizedCaseInsensitiveContains(part)
                    || (entry.replacement?.localizedCaseInsensitiveContains(part) ?? false)
                    || entry.spokenForms.contains { $0.localizedCaseInsensitiveContains(part) }
            }
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            PageTitle("Dictionary")
            HStack(spacing: 10) {
                Button(action: importFile) {
                    Label("Import", systemImage: "square.and.arrow.down")
                }
                .buttonStyle(PillButtonStyle())
                .help(
                    "Merge someone else's dictionary: a Phonon dictionary.json, or a text "
                        + "file with one term or \"misheard → right\" per line.")
                if let importNote {
                    Text(importNote).font(.system(size: 12)).foregroundStyle(EmberTheme.mute)
                        .transition(.opacity)
                }
            }
            .padding(.bottom, 10)

            HStack(spacing: 8) {
                TextField(
                    "Add", text: $draft,
                    prompt: Text("Add a word, a name, or 'misheard → right'")
                        .foregroundStyle(EmberTheme.dim))
                    .textFieldStyle(.plain)
                    .onSubmit(add)
                Button(action: add) { Image(systemName: "plus") }
                    .buttonStyle(IconButtonStyle())
                    .disabled(draft.trimmingCharacters(in: .whitespaces).isEmpty)
                    .help("Add")
            }
            .padding(.leading, 12)
            .padding(.trailing, 6)
            .padding(.vertical, 6)
            .background(EmberTheme.panel, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .stroke(Color(hex: 0x2A221C), lineWidth: 1))
            .help(
                "A word or name teaches Phonon its spelling. \"misheard → right\" (or ->) "
                    + "always turns the first into the second.")

            ScrollView {
                FlowLayout(spacing: 8) {
                    ForEach(shown) { entry in
                        DictionaryChip(entry: entry) {
                            withAnimation(Motion.quick) { store.removeDictionary(id: entry.id) }
                        }
                    }
                }
                .padding(.top, 12)
                .padding(.bottom, 24)
                .animation(Motion.quick, value: shown.map(\.id))
            }
            .scrollIndicators(.automatic)
        }
        .padding(.horizontal, 30)
        .padding(.top, 24)
    }

    private func add() {
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        withAnimation(Motion.quick) { store.addDictionaryText(text) }
        draft = ""
    }

    private func importFile() {
        let panel = NSOpenPanel()
        panel.title = "Import a dictionary"
        panel.allowedContentTypes = [.json, .plainText]
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let added = store.importDictionary(from: url)
        withAnimation(Motion.quick) {
            importNote = added == 1 ? "1 entry added" : "\(added) entries added"
        }
    }
}

/// A term ("Parakeet"), or a fix shown as "misheard → right".
struct DictionaryChip: View {
    let entry: NativeDictionaryEntry
    let onRemove: () -> Void

    private var heard: String? {
        if entry.replacement != nil { return entry.phrase }
        guard !entry.spokenForms.isEmpty else { return nil }
        let shown = entry.spokenForms.prefix(2).joined(separator: ", ")
        let more = entry.spokenForms.count - 2
        return more > 0 ? "\(shown) +\(more)" : shown
    }

    var body: some View {
        HStack(spacing: 6) {
            if let heard {
                Text(heard).font(.system(size: 12)).foregroundStyle(Color(hex: 0x8A7D70))
                    .lineLimit(1)
                Text("→").foregroundStyle(EmberTheme.dim)
            }
            Text(entry.replacement ?? entry.phrase).lineLimit(1)
            Button(action: onRemove) {
                Image(systemName: "xmark").font(.system(size: 9, weight: .bold))
                    .frame(width: 20, height: 20)
            }
            .buttonStyle(ChipRemoveStyle())
            .help("Remove")
        }
        .padding(.leading, 12)
        .padding(.trailing, 6)
        .padding(.vertical, 4)
        .background(Color(hex: 0x17120F), in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Color(hex: 0x2A211B), lineWidth: 1))
        .help(
            entry.spokenForms.isEmpty
                ? (entry.replacement == nil ? entry.phrase : "\(entry.phrase) → \(entry.replacement!)")
                : "Also heard as: " + entry.spokenForms.joined(separator: ", "))
    }
}

private struct ChipRemoveStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        RemoveBody(configuration: configuration)
    }

    private struct RemoveBody: View {
        let configuration: ButtonStyleConfiguration
        @State private var hovering = false
        var body: some View {
            configuration.label
                .foregroundStyle(hovering ? EmberTheme.selectedInk : EmberTheme.dim)
                .background(hovering ? Color(hex: 0x2A1D15) : .clear, in: Circle())
                .contentShape(Circle())
                .onHover { hovering = $0 }
                .animation(Motion.quick, value: hovering)
        }
    }
}

// MARK: - Macros

struct MacrosView: View {
    @ObservedObject var store: NativeAppStore
    @State private var trigger = ""
    @State private var expansion = ""
    @State private var editingID: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            PageTitle("Macros")
            HStack(alignment: .top, spacing: 8) {
                field {
                    TextField(
                        "When I say", text: $trigger,
                        prompt: Text("When I say…").foregroundStyle(EmberTheme.dim)
                    )
                    .textFieldStyle(.plain)
                }
                    .frame(width: 220)
                    .help("Say this as a whole dictation. Case and end punctuation do not matter.")
                field {
                    TextField(
                        "Type this", text: $expansion,
                        prompt: Text("Type this").foregroundStyle(EmberTheme.dim), axis: .vertical)
                        .textFieldStyle(.plain)
                        .lineLimit(1...5)
                }
                .help("What Phonon types instead. Line breaks are kept.")
                Button(action: save) {
                    Image(systemName: editingID == nil ? "plus" : "checkmark")
                }
                .buttonStyle(IconButtonStyle())
                .disabled(!canSave)
                .help(editingID == nil ? "Add" : "Save")
                .padding(.top, 2)
            }
            .padding(.bottom, 8)

            ScrollView {
                VStack(spacing: 0) {
                    ForEach(store.macros) { macro in
                        MacroRow(
                            macro: macro,
                            onEdit: {
                                editingID = macro.id
                                trigger = macro.trigger
                                expansion = macro.expansion
                            },
                            onRemove: {
                                withAnimation(Motion.quick) { store.removeMacro(id: macro.id) }
                            })
                    }
                }
                .animation(Motion.quick, value: store.macros)
                if store.macros.isEmpty {
                    Text("Say a trigger as a whole dictation and Phonon types its text instead.")
                        .foregroundStyle(EmberTheme.mute)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.vertical, 12)
                }
            }
        }
        .padding(.horizontal, 30)
        .padding(.top, 24)
    }

    private var canSave: Bool {
        !MacroBook.normalize(trigger).isEmpty
            && !expansion.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func save() {
        guard canSave else { return }
        withAnimation(Motion.quick) {
            store.upsertMacro(trigger: trigger, expansion: expansion, replacing: editingID)
        }
        trigger = ""
        expansion = ""
        editingID = nil
    }

    private func field<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        content()
            .padding(.horizontal, 11)
            .padding(.vertical, 8)
            .background(EmberTheme.panel, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .stroke(Color(hex: 0x2A221C), lineWidth: 1))
    }
}

struct MacroRow: View {
    let macro: PhononMacro
    let onEdit: () -> Void
    let onRemove: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            Rectangle().fill(EmberTheme.line).frame(height: 1)
            HStack(alignment: .top, spacing: 10) {
                Text("\u{201C}\(macro.trigger)\u{201D}")
                    .fontWeight(.medium)
                    .frame(width: 220, alignment: .leading)
                Text("→").foregroundStyle(EmberTheme.dim).frame(width: 22)
                Text(macro.expansion)
                    .foregroundStyle(EmberTheme.soft)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
                HStack(spacing: 0) {
                    Button(action: onEdit) { Image(systemName: "pencil") }
                        .buttonStyle(IconButtonStyle()).help("Edit")
                    Button(action: onRemove) { Image(systemName: "trash") }
                        .buttonStyle(IconButtonStyle()).help("Remove")
                }
                .padding(.top, -6)
            }
            .padding(.horizontal, 6)
            .padding(.vertical, 12)
        }
    }
}

// MARK: - Settings

struct SettingsView: View {
    @ObservedObject var store: NativeAppStore
    let onSettingsChanged: () -> Void
    let onShowModelStatus: () -> Void
    @State private var advancedOpen: Bool
    @State private var confirmClearHistory = false
    @State private var importRequest: DictationImportRequest?

    init(
        store: NativeAppStore, onSettingsChanged: @escaping () -> Void,
        onShowModelStatus: @escaping () -> Void, advancedOpen: Bool = false
    ) {
        self.store = store
        self.onSettingsChanged = onSettingsChanged
        self.onShowModelStatus = onShowModelStatus
        _advancedOpen = State(initialValue: advancedOpen)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 30) {
                PageTitle("Settings").padding(.bottom, -30)

                ExportCard(action: exportEverything)

                SettingsGroup("Dictation") {
                    SettingRow("Shortcut", help: shortcutHelp) {
                        Picker("Shortcut", selection: Binding(
                            get: { store.settings.shortcutMode },
                            set: { value in
                                store.updateSettings { $0.shortcutMode = value }
                                onSettingsChanged()
                            }
                        )) {
                            Text("Globe (fn) hold").tag("fn")
                            Text("Globe (fn) hold + Control Space").tag("fn_and_control_space")
                            Text("Right Option hold").tag("right_option")
                            Text("Right Option hold + Control Space").tag("both")
                            Text("Control Space toggle").tag("control_space")
                        }
                        .labelsHidden()
                        .pickerStyle(.menu)
                        .fixedSize()
                    }
                    SettingRow(
                        "Show the bar between dictations",
                        help: "Keep the small bar at the bottom of the screen when you are not "
                            + "dictating. Off shows it only while you dictate, or while meeting "
                            + "notes record."
                    ) { toggle(\.showIdleCapsule) }
                    SettingRow(
                        "Show live text while dictating",
                        help: "While you hold the key, show the text so far above the bar, with "
                            + "a dot that lights up as you speak. Later words can rewrite earlier "
                            + "ones; at the end it shows what was typed. Needs Streaming. Applies "
                            + "from the next dictation."
                    ) { toggle(\.liveText) }
                    SettingRow(
                        "Other audio while dictating",
                        help: "A video or song playing on this Mac fades while you hold the "
                            + "key, so it stays out of the microphone, and comes back when you "
                            + "let go. Lower takes it to a quarter of its volume; Mute fades it "
                            + "out. With headphones it does not reach the mic either way."
                    ) {
                        ChoiceChips(
                            options: BackgroundAudio.allCases.map { ($0, $0.title) },
                            selection: store.settings.backgroundAudio
                        ) { value in
                            store.updateSettings { $0.backgroundAudio = value }
                            onSettingsChanged()
                        }
                    }
                    SettingRow(
                        "Keep the mic on between dictations",
                        help: "Recording starts the moment you press the shortcut, and macOS "
                            + "shows the mic as in use. Off opens the mic only while you "
                            + "dictate; the first moment can be clipped."
                    ) { toggle(\.instantMic) }
                    SettingRow(
                        "After a paste, the clipboard holds",
                        help: "Your dictation: press ⌘V to paste it again anywhere. What you "
                            + "copied before: Phonon puts it back once the paste has landed."
                    ) {
                        ChoiceChips(
                            options: ClipboardAfterPaste.allCases.map { ($0, $0.title) },
                            selection: store.settings.clipboardAfterPaste
                        ) { value in
                            store.updateSettings { $0.clipboardAfterPaste = value }
                            onSettingsChanged()
                        }
                    }
                }

                SettingsGroup("Notes") {
                    SettingRow(
                        "Ask to take notes when a meeting starts",
                        help: "When a call starts, ask whether to record your mic and the call "
                            + "audio on this Mac."
                    ) { toggle(\.meetingCapture) }
                    SettingRow(
                        "Start recording without asking",
                        help: "Record detected calls right away. Some places require everyone's "
                            + "consent."
                    ) { toggle(\.meetingAutoStart).disabled(!store.settings.meetingCapture) }
                }

                SettingsGroup("System") {
                    SettingRow(
                        "Import from other dictation apps",
                        help: "Bring over the dictionary, snippets, past dictations, shortcut and "
                            + "microphone from Wispr Flow, Aqua Voice, superwhisper, VoiceInk, "
                            + "MacWhisper or FluidVoice on this Mac. Their files are only read, "
                            + "and nothing leaves the Mac."
                    ) {
                        Button("Import…") {
                            importRequest = DictationImportRequest(apps: [], firstRun: false)
                        }
                        .buttonStyle(PillButtonStyle())
                    }
                    SettingRow(
                        "Show in Dock",
                        help: "Off keeps Phonon in the menu bar only. Open it again from the "
                            + "ϕ menu."
                    ) { toggle(\.showInDock) }
                    SettingRow(
                        "Check for updates",
                        help: "Once a day, ask GitHub whether a newer Phonon is out. The request "
                            + "carries nothing about you or your dictation."
                    ) { toggle(\.checkForUpdates) }
                    SettingRow(
                        UpdateCheck.runningVersion.map { "Phonon \($0)" }
                            ?? "Phonon (development build)",
                        help: "The installed version."
                    ) {
                        if let update = store.availableUpdate {
                            Button("Update to \(update.version)…") { store.onShowUpdate?() }
                                .buttonStyle(PillButtonStyle(primary: true))
                        } else {
                            HStack(spacing: 10) {
                                Text("up to date").font(.system(size: 12.5))
                                    .foregroundStyle(EmberTheme.mute)
                                Button("Check now") { store.onCheckForUpdates?() }
                                    .buttonStyle(PillButtonStyle())
                            }
                        }
                    }
                }

                VStack(alignment: .leading, spacing: 8) {
                    Button {
                        withAnimation(Motion.open) { advancedOpen.toggle() }
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: "chevron.right")
                                .font(.system(size: 10, weight: .bold))
                                .rotationEffect(.degrees(advancedOpen ? 90 : 0))
                            Text("ADVANCED")
                                .font(.system(size: 12, weight: .semibold))
                                .tracking(0.84)
                        }
                        .foregroundStyle(EmberTheme.mute)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    if advancedOpen {
                        advanced
                            .transition(.opacity.combined(with: .move(edge: .top)))
                    }
                }
                .frame(maxWidth: 760, alignment: .leading)
            }
            .padding(.horizontal, 30)
            .padding(.top, 24)
            .padding(.bottom, 30)
        }
        .scrollIndicators(.never)
        .onAppear { store.refreshPermissions() }
        .sheet(item: $importRequest) { request in
            DictationImportView(
                store: store, request: request, onSettingsChanged: onSettingsChanged
            ) { importRequest = nil }
        }
        .alert("Clear all history?", isPresented: $confirmClearHistory) {
            Button("Cancel", role: .cancel) {}
            Button("Move to Trash", role: .destructive) { store.clearAllHistory() }
        } message: {
            Text(
                "\(store.history.count) recordings and their transcripts go to the Trash. This cannot be undone from Phonon."
            )
        }
    }

    @ViewBuilder private var advanced: some View {
        VStack(alignment: .leading, spacing: 0) {
            SettingRow(
                "Writing style",
                help: "Follow the app: Discord and Messages casual, Slack, Teams and email "
                    + "formal, terminals and AI tools as prompts, minimal everywhere else. "
                    + "Pick one to use it everywhere."
            ) {
                ChoiceChips(
                    options: WritingStyle.allCases.map { ($0, $0.title) },
                    selection: store.settings.writingStyle
                ) { value in
                    store.updateSettings { $0.writingStyle = value }
                    onSettingsChanged()
                }
            }
            AdvancedBlock("Writing mode per app") {
                AppWritingModesEditor(store: store, onChanged: onSettingsChanged)
                    .help(
                        "Pin one mode for an app, over the writing style above. Claude Code "
                            + "and other terminal agents use their terminal app (Terminal, "
                            + "Ghostty, iTerm). Prompt keeps a request as an instruction; "
                            + "Minimal only cleans up. Applies from the next dictation.")
            }
            LearnedStylesList(store: store, onChanged: onSettingsChanged)
            SettingRow(
                "Streaming",
                help: "Transcribe while you speak, not only after you let go."
            ) { toggle(\.streaming) }
            SettingRow(
                "Streaming refine",
                help: "Correct the text in the background while you speak, so the final "
                    + "correction is faster. Uses the GPU while you talk. Applies when the "
                    + "engine restarts."
            ) {
                Toggle("Streaming refine", isOn: Binding(
                    get: { store.settings.streamRefine ?? Self.streamRefineDefault },
                    set: { value in
                        store.updateSettings { $0.streamRefine = value }
                        onSettingsChanged()
                    }
                ))
                .labelsHidden()
                .toggleStyle(EmberSwitchStyle())
            }
            SettingRow(
                "Use screen context for spelling",
                help: "Read the visible windows with on-device OCR to spell dictionary terms "
                    + "that are on screen. Screenshots are never saved."
            ) { toggle(\.screenContext) }
            SettingRow(
                "Text around the cursor",
                help: "Give the local correction model the text before and after the cursor."
            ) { toggle(\.textboxContext) }
            SettingRow(
                "Learn from your edits",
                help: "When you fix a word Phonon typed, offer to add it to the dictionary."
            ) { toggle(\.editLearning) }
            SettingRow(
                "Local history",
                help: "Keep each recording and its transcript after it is typed. Home and "
                    + "Insights read it."
            ) { toggle(\.localHistory) }
            SettingRow("Keep recordings for", help: "Older recordings go to the Trash.") {
                Picker("Keep recordings for", selection: Binding(
                    get: { store.settings.historyRetentionDays },
                    set: { value in
                        store.updateSettings { $0.historyRetentionDays = value }
                        store.pruneExpiredRecordings()
                        onSettingsChanged()
                    }
                )) {
                    Text("Until I delete them").tag(NativeSettings.keepRecordingsForever)
                    Text("7 days").tag(7)
                    Text("30 days").tag(30)
                    Text("90 days").tag(90)
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .fixedSize()
            }
            SettingRow(
                "Clear all history",
                help: "Move every stored recording and transcript to the Trash."
            ) {
                Button("Clear…") { confirmClearHistory = true }
                    .buttonStyle(PillButtonStyle())
                    .disabled(store.history.isEmpty)
            }
            SettingRow(
                "Recording cues", help: "Play a short sweep when a recording opens and closes."
            ) { toggle(\.soundFeedback) }
            SettingRow(
                "Launch at login", help: "Start Phonon in the menu bar when you log in."
            ) {
                Toggle("Launch at login", isOn: Binding(
                    get: { store.launchAtLoginEnabled },
                    set: { store.setLaunchAtLogin($0) }
                ))
                .labelsHidden()
                .toggleStyle(EmberSwitchStyle())
            }
            SettingRow(
                "Model status",
                help: "See whether the speech and correction models are loaded."
            ) {
                Button("Open", action: onShowModelStatus).buttonStyle(PillButtonStyle())
            }
            SettingRow("Backup", help: backupHelp) {
                HStack(spacing: 8) {
                    Text(backupStatus).font(.mono(12)).foregroundStyle(EmberTheme.mute)
                    Button("Show") { store.revealBackupInFinder() }
                        .buttonStyle(PillButtonStyle())
                        .disabled(store.mirrorManifest == nil)
                    Button("Delete") { store.deleteBackup() }
                        .buttonStyle(PillButtonStyle())
                        .disabled(store.mirrorManifest == nil)
                }
            }
            AdvancedBlock("Your own model endpoint") {
                OwnEndpointEditor(store: store, onChanged: onSettingsChanged)
            }
            AdvancedBlock("WebSocket API") {
                WebSocketAPIEditor(store: store, onChanged: onSettingsChanged)
            }
            AdvancedBlock("Microphone priority") {
                MicrophonePriorityCard(store: store, onSettingsChanged: onSettingsChanged)
            }
            AdvancedBlock("Private apps and sites") {
                PrivateListEditor(store: store, onChanged: onSettingsChanged)
            }
            AdvancedBlock("Permissions") {
                PermissionSummary(store: store)
            }
        }
    }

    /// Mirrors the engine: on for Macs with 16 GB of memory or more.
    static var streamRefineDefault: Bool {
        ProcessInfo.processInfo.physicalMemory >= 16 * 1_024 * 1_024 * 1_024
    }

    private var shortcutHelp: String {
        var lines: [String] = []
        let latch = "Or tap it once to keep recording, then tap again to stop."
        if ShortcutPolicy.allows(mode: store.settings.shortcutMode, source: "right-option") {
            lines.append("Hold Right Option to talk; release to insert. " + latch)
        }
        if store.settings.shortcutMode.hasPrefix("fn") {
            lines.append("Hold Globe to talk; release to insert. " + latch)
            lines.append(
                "macOS also acts on the Globe key. Set System Settings › Keyboard › "
                    + "\"Press 🌐 key to\" to \"Do Nothing\" so it only records.")
        }
        return lines.joined(separator: "\n")
    }

    private var backupHelp: String {
        "Uninstalling removes everything under ~/Library, so a copy of the dictionary, "
            + "macros, settings and history is kept in ~/.phonon. Phonon offers it back if "
            + "it ever starts empty."
    }

    private var backupStatus: String {
        guard let manifest = store.mirrorManifest else { return "none yet" }
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM d, h:mm a"
        return formatter.string(from: manifest.savedAt)
    }

    private func exportEverything() {
        let panel = NSSavePanel()
        panel.title = "Export Phonon data"
        panel.message = "Phonon makes a folder with your dictations, notes, dictionary and macros."
        panel.nameFieldStringValue = "Phonon Data"
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let url = panel.url else { return }
        store.exportEverything(to: url)
    }

    private func toggle(_ keyPath: WritableKeyPath<NativeSettings, Bool>) -> some View {
        Toggle("", isOn: Binding(
            get: { store.settings[keyPath: keyPath] },
            set: { value in
                store.updateSettings { $0[keyPath: keyPath] = value }
                onSettingsChanged()
            }
        ))
        .labelsHidden()
        .toggleStyle(EmberSwitchStyle())
    }
}

struct ExportCard: View {
    let action: () -> Void
    var body: some View {
        HStack(spacing: 14) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Export everything").font(.system(size: 14.5, weight: .semibold))
                Text("Dictations, notes with audio, dictionary and macros, as one folder you own.")
                    .font(.system(size: 12.5))
                    .foregroundStyle(Color(hex: 0xA39A90))
            }
            Spacer()
            Button(action: action) {
                Label("Export", systemImage: "square.and.arrow.up")
            }
            .buttonStyle(PillButtonStyle(primary: true))
        }
        .padding(16)
        .background(
            LinearGradient(
                colors: [Color(hex: 0x1F150E), Color(hex: 0x150F0B)], startPoint: .top,
                endPoint: .bottom),
            in: RoundedRectangle(cornerRadius: 12, style: .continuous)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(Color(hex: 0x3B2516), lineWidth: 1))
        .frame(maxWidth: 760)
    }
}

struct SettingsGroup<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content
    init(_ title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(title.uppercased())
                .font(.system(size: 12, weight: .semibold))
                .tracking(0.84)
                .foregroundStyle(EmberTheme.mute)
                .padding(.bottom, 8)
            content
        }
        .frame(maxWidth: 760, alignment: .leading)
    }
}

/// One settings row: a title (its explanation is the hover tooltip) and a
/// control on the right.
struct SettingRow<Control: View>: View {
    let title: String
    let help: String
    @ViewBuilder let control: Control
    init(_ title: String, help: String, @ViewBuilder control: () -> Control) {
        self.title = title
        self.help = help
        self.control = control()
    }
    var body: some View {
        VStack(spacing: 0) {
            Rectangle().fill(EmberTheme.line).frame(height: 1)
            HStack(spacing: 14) {
                Text(title).help(help)
                Spacer(minLength: 12)
                control
            }
            .padding(.vertical, 9)
            .frame(minHeight: 40)
        }
    }
}

/// A labelled block inside Advanced for the controls that are lists.
struct AdvancedBlock<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content
    init(_ title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Rectangle().fill(EmberTheme.line).frame(height: 1)
            Text(title)
            content
        }
        .padding(.bottom, 10)
    }
}

/// The mock's style picker: rounded chips, the chosen one in ember.
struct ChoiceChips<Value: Hashable>: View {
    let options: [(Value, String)]
    let selection: Value
    let onSelect: (Value) -> Void
    @Environment(\.isEnabled) private var isEnabled

    var body: some View {
        HStack(spacing: 8) {
            ForEach(options, id: \.0) { value, title in
                let on = value == selection
                Button { onSelect(value) } label: {
                    Text(title)
                        .font(.system(size: 12.5))
                        .padding(.horizontal, 10)
                        .padding(.vertical, 4)
                        .foregroundStyle(on ? EmberTheme.selectedInk : Color(hex: 0xCFC3B6))
                        .background(
                            on ? Color(hex: 0x3A2414) : Color(hex: 0x1A1613),
                            in: RoundedRectangle(cornerRadius: 9))
                        .overlay(
                            RoundedRectangle(cornerRadius: 9).strokeBorder(
                                on ? Color(hex: 0x5A3018) : Color(hex: 0x2C251F), lineWidth: 1))
                        .contentShape(Capsule())
                }
                .buttonStyle(.plain)
                .animation(Motion.quick, value: on)
            }
        }
        .opacity(isEnabled ? 1 : 0.45)
    }
}

/// The writing styles the user approved on the edit-learning card, each
/// removable. Nothing shows until one is approved.
struct LearnedStylesList: View {
    @ObservedObject var store: NativeAppStore
    let onChanged: () -> Void

    var body: some View {
        let rows = EditSuggestionCopy.styleRows(store.settings.stylePreferences)
        ForEach(rows, id: \.text) { row in
            SettingRow(row.text, help: "A writing style you approved after editing a dictation.") {
                Button("Remove") {
                    store.removeStylePreference(row.preference, category: row.category)
                    onChanged()
                }
                .buttonStyle(PillButtonStyle())
            }
        }
    }
}

/// The denylist: apps and sites Phonon never reads for context, text-box
/// text or edits. Dictation still works there.
struct PrivateListEditor: View {
    @ObservedObject var store: NativeAppStore
    let onChanged: () -> Void
    @State private var draft = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(store.settings.screenContextDenylist, id: \.self) { entry in
                HStack {
                    let name = Self.displayName(entry)
                    Text(name).help(entry)
                    Spacer()
                    Button("Remove") {
                        store.removeFromDenylist(entry)
                        onChanged()
                    }
                    .buttonStyle(PillButtonStyle())
                }
            }
            HStack {
                TextField("App bundle id or website, e.g. discord.com", text: $draft)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit(add)
                Menu("Running app") {
                    ForEach(Self.runningApps(), id: \.bundleID) { app in
                        Button(app.name) {
                            store.addToDenylist(app.bundleID)
                            onChanged()
                        }
                    }
                }
                .fixedSize()
                Button("Add", action: add)
                    .disabled(ContextPolicy.normalizeEntry(draft) == nil)
            }
        }
        .help(
            "Phonon never reads the screen, the text field or your edits in these apps "
                + "and sites. Dictation still works. A site and its desktop app count as "
                + "one (discord.com also covers the Discord app). Password prompts and "
                + "password managers (1Password, Bitwarden, Keychain Access, Passwords, "
                + "LastPass, Dashlane, KeePassXC, Proton Pass, Enpass, Keeper) are always "
                + "private, as is any app or page Phonon cannot identify."
        )
    }

    private func add() {
        guard ContextPolicy.normalizeEntry(draft) != nil else { return }
        store.addToDenylist(draft)
        draft = ""
        onChanged()
    }

    static func displayName(_ entry: String) -> String {
        if let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: entry) {
            return FileManager.default.displayName(atPath: url.path)
                .replacingOccurrences(of: ".app", with: "")
        }
        return entry
    }

    static func runningApps() -> [(name: String, bundleID: String)] {
        NSWorkspace.shared.runningApplications
            .filter { $0.activationPolicy == .regular }
            .compactMap { app in
                guard let id = app.bundleIdentifier, id != Bundle.main.bundleIdentifier else {
                    return nil
                }
                return (app.localizedName ?? id, id)
            }
            .sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }
}

/// Settings > Advanced > Writing mode per app: one refiner mode pinned for an app,
/// over the app's category and the Writing style above (settings.json
/// `writing_modes_by_app`). Claude Code in a terminal is the terminal app here.
struct AppWritingModesEditor: View {
    @ObservedObject var store: NativeAppStore
    let onChanged: () -> Void

    var body: some View {
        let dictated = store.dictatedApps
        let names = Dictionary(dictated.map { ($0.bundleID, $0.name) }, uniquingKeysWith: { a, _ in a })
        let name = { (id: String) in names[id] ?? PrivateListEditor.displayName(id) }
        let pinned = store.settings.writingModesByApp
            .compactMap { id, raw in AppWritingMode(rawValue: raw).map { (bundleID: id, mode: $0) } }
            .sorted { name($0.bundleID) < name($1.bundleID) }
        VStack(alignment: .leading, spacing: 8) {
            ForEach(pinned, id: \.bundleID) { entry in
                HStack(spacing: 12) {
                    Text(name(entry.bundleID))
                        .frame(width: 150, alignment: .leading)
                        .lineLimit(1)
                        .help(entry.bundleID)
                    ChoiceChips(
                        options: AppWritingMode.allCases.map { ($0, $0.title) },
                        selection: entry.mode
                    ) { mode in
                        store.setAppWritingMode(entry.bundleID, mode)
                        onChanged()
                    }
                    Spacer()
                    Button("Remove") {
                        store.setAppWritingMode(entry.bundleID, nil)
                        onChanged()
                    }
                    .buttonStyle(PillButtonStyle())
                }
            }
            Menu("Add an app") {
                let taken = Set(store.settings.writingModesByApp.keys)
                let recent = dictated.filter { !taken.contains($0.bundleID) }.prefix(15)
                if !recent.isEmpty {
                    Section("Dictated into") {
                        ForEach(Array(recent), id: \.bundleID) { app in
                            Button(app.name) { add(app.bundleID) }
                        }
                    }
                }
                let recentIDs = Set(recent.map(\.bundleID))
                let running = PrivateListEditor.runningApps()
                    .filter { !taken.contains($0.bundleID) && !recentIDs.contains($0.bundleID) }
                if !running.isEmpty {
                    Section("Running") {
                        ForEach(running, id: \.bundleID) { app in
                            Button(app.name) { add(app.bundleID) }
                        }
                    }
                }
            }
            .fixedSize()
        }
    }

    /// A new app starts on Minimal, the mode with no register change.
    private func add(_ bundleID: String) {
        store.setAppWritingMode(bundleID, .minimal)
        onChanged()
    }
}

/// Settings > Advanced > Your own model endpoint (issue #24). Off by default; no
/// endpoint is filled in for the user. The key goes to the Keychain only.
struct OwnEndpointEditor: View {
    @ObservedObject var store: NativeAppStore
    let onChanged: () -> Void
    @State private var baseURL = ""
    @State private var model = ""
    @State private var key = ""
    @State private var keySaved = false
    @State private var loaded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Correct with your own model endpoint")
                Spacer()
                Toggle("", isOn: Binding(
                    get: { store.settings.byokEnabled },
                    set: { on in
                        store.updateSettings { $0.byokEnabled = on }
                        onChanged()
                    }
                ))
                .labelsHidden()
                .toggleStyle(EmberSwitchStyle())
            }
            Text(
                "Any OpenAI-compatible endpoint. Each transcript, its writing mode and up to 80 "
                    + "dictionary terms are sent to it, so only use one you trust. Speech "
                    + "recognition stays on this Mac, and the local model still runs: its text "
                    + "is used whenever the endpoint fails or takes longer than 6 seconds."
            )
            .font(.system(size: 12))
            .foregroundStyle(EmberTheme.mute)
            .fixedSize(horizontal: false, vertical: true)
            if store.settings.byokEnabled {
                TextField("Base URL, e.g. https://api.example.com/v1", text: $baseURL)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit(saveEndpoint)
                TextField("Model name", text: $model)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit(saveEndpoint)
                HStack {
                    SecureField(keySaved ? "API key saved in the Keychain" : "API key", text: $key)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit(saveEndpoint)
                    Button("Save", action: saveEndpoint)
                        .buttonStyle(PillButtonStyle(primary: true))
                    if keySaved {
                        Button("Forget key") {
                            OwnEndpoint.saveKey("")
                            keySaved = false
                            onChanged()
                        }
                        .buttonStyle(PillButtonStyle())
                    }
                }
                if !baseURL.isEmpty && !OwnEndpoint.validBaseURL(baseURL) {
                    Text("The base URL must start with https:// or http://.")
                        .font(.system(size: 12))
                        .foregroundStyle(EmberTheme.warm)
                }
            }
        }
        .onAppear {
            guard !loaded else { return }
            loaded = true
            baseURL = store.settings.byokBaseURL
            model = store.settings.byokModel
            keySaved = store.settings.byokEnabled && OwnEndpoint.readKey() != nil
        }
    }

    private func saveEndpoint() {
        store.updateSettings {
            $0.byokBaseURL = baseURL.trimmingCharacters(in: .whitespaces)
            $0.byokModel = model.trimmingCharacters(in: .whitespaces)
        }
        if !key.isEmpty {
            keySaved = OwnEndpoint.saveKey(key) == errSecSuccess
            key = ""
        }
        onChanged()
    }
}

/// Settings > Advanced > WebSocket API. Off by default. Turning it on makes the token
/// (Keychain only) and has the engine listen; the rows below say what a client needs.
struct WebSocketAPIEditor: View {
    @ObservedObject var store: NativeAppStore
    let onChanged: () -> Void
    @State private var port = ""
    @State private var addresses: [String] = []
    @State private var loaded = false
    /// The address the pairing code carries (a per-Mac convenience, not a setting).
    @AppStorage("websocketPairAddress") private var pairAddress = ""

    private var token: String? {
        let token = store.settings.websocketAPIToken
        return token.isEmpty ? nil : token
    }

    /// The ticked address: the user's pick while this Mac still has it, else the preferred one.
    private var selectedAddress: String? {
        addresses.contains(pairAddress) ? pairAddress : WebSocketAPI.preferredAddress(addresses)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Let other devices dictate through this Mac")
                Spacer()
                Toggle("", isOn: Binding(
                    get: { store.settings.websocketAPI },
                    set: { on in
                        store.updateSettings {
                            if on { WebSocketAPI.turnOn(&$0) } else { $0.websocketAPI = false }
                        }
                        addresses = WebSocketAPI.reachableAddresses()
                        onChanged()
                    }
                ))
                .labelsHidden()
                .toggleStyle(EmberSwitchStyle())
            }
            Text(
                "A device on your network or tailnet, such as a phone keyboard, sends its "
                    + "audio and the app it types into, and gets the corrected text back. "
                    + "Anyone with the token can use it. The connection is not encrypted, so "
                    + "use it on a network you trust or over a tailnet."
            )
            .font(.system(size: 12))
            .foregroundStyle(EmberTheme.mute)
            .fixedSize(horizontal: false, vertical: true)
            if store.settings.websocketAPI {
                Text(
                    WebSocketAPI.statusLine(
                        engineStatus: store.websocketAPIStatus, engineReady: store.engineReady)
                )
                .font(.system(size: 12))
                .foregroundStyle(
                    store.websocketAPIStatus?["listening"] as? Bool == false
                        ? EmberTheme.warm : EmberTheme.mute)
                HStack(alignment: .top, spacing: 20) {
                    VStack(alignment: .leading, spacing: 10) {
                        ForEach(addresses, id: \.self) { address in
                            addressRow(address)
                        }
                        if let token {
                            HStack(spacing: 8) {
                                copyRow("Token", token)
                                Button("New token") {
                                    store.updateSettings { $0.websocketAPIToken = WebSocketAPI.newToken() }
                                    onChanged()
                                }
                                .buttonStyle(PillButtonStyle())
                                .help("Devices using the old token are disconnected")
                            }
                        }
                    }
                    Spacer(minLength: 0)
                    if let token, let address = selectedAddress {
                        PairingCode(
                            uri: WebSocketAPI.pairingURI(
                                address: address, port: store.settings.websocketAPIPort,
                                token: token),
                            caption: "Scan with Phonon voice")
                    }
                }
                HStack(spacing: 8) {
                    Text("Port").foregroundStyle(EmberTheme.mute)
                    TextField("\(WebSocketAPI.defaultPort)", text: $port)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 80)
                        .onSubmit(savePort)
                    Button("Save", action: savePort).buttonStyle(PillButtonStyle())
                }
                .font(.system(size: 12))
            }
        }
        .onAppear {
            guard !loaded else { return }
            loaded = true
            port = String(store.settings.websocketAPIPort)
            if store.settings.websocketAPI { addresses = WebSocketAPI.reachableAddresses() }
        }
    }

    /// An address of this Mac: ticked when it is the one the pairing code carries; clicking
    /// it ticks it.
    private func addressRow(_ address: String) -> some View {
        let url = "ws://\(address):\(store.settings.websocketAPIPort)"
        let selected = address == selectedAddress
        return HStack(spacing: 8) {
            Button {
                pairAddress = address
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: selected ? "checkmark.circle.fill" : "circle")
                        .foregroundStyle(selected ? EmberTheme.ember : EmberTheme.dim)
                        .frame(width: 14)
                    Text("Address").foregroundStyle(EmberTheme.mute)
                    Text(url).font(.mono(12)).foregroundStyle(selected ? EmberTheme.ink : EmberTheme.soft)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help(selected ? "The pairing code points here" : "Put this address in the pairing code")
            .accessibilityAddTraits(selected ? .isSelected : [])
            CopyIconButton(text: url, help: "Copy the address")
        }
        .font(.system(size: 12))
    }

    private func copyRow(_ label: String, _ value: String) -> some View {
        HStack(spacing: 8) {
            Text(label).foregroundStyle(EmberTheme.mute)
            Text(value).font(.mono(12)).textSelection(.enabled)
            CopyIconButton(text: value, help: "Copy the \(label.lowercased())")
        }
        .font(.system(size: 12))
    }

    private func savePort() {
        guard let number = Int(port.trimmingCharacters(in: .whitespaces)),
            WebSocketAPI.validPort(number)
        else {
            port = String(store.settings.websocketAPIPort)
            return
        }
        store.updateSettings { $0.websocketAPIPort = number }
        onChanged()
    }
}

/// A QR code for `uri`, drawn crisp at any size, with a caption under it.
struct PairingCode: View {
    let uri: String
    let caption: String

    var body: some View {
        VStack(spacing: 6) {
            if let image = Self.image(for: uri) {
                Image(nsImage: image)
                    .interpolation(.none)
                    .resizable()
                    .frame(width: 148, height: 148)
                    .padding(8)
                    .background(Color.white, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                    .accessibilityLabel("Pairing QR code")
            }
            Text(caption)
                .font(.system(size: 11))
                .foregroundStyle(EmberTheme.mute)
        }
    }

    /// One pixel per module, black on white; SwiftUI scales it without smoothing.
    static func image(for text: String) -> NSImage? {
        let filter = CIFilter.qrCodeGenerator()
        filter.message = Data(text.utf8)
        filter.correctionLevel = "M"
        guard let output = filter.outputImage,
            let cgImage = CIContext().createCGImage(output, from: output.extent)
        else { return nil }
        return NSImage(cgImage: cgImage, size: output.extent.size)
    }
}

/// What stops dictation from working right now, most blocking first. Shown as a
/// banner on every page of the main window and as a warning on the menu-bar item,
/// so a missing permission cannot hide in Settings > Advanced (a launch user could
/// not start dictation until they found Input Monitoring off).
enum PermissionAttention: Equatable {
    /// No event tap: the hold key never reaches Phonon.
    case inputMonitoring(key: String)
    case microphone
    /// Dictation works but cannot paste; the text waits on the clipboard.
    case accessibility
    /// macOS acts on the Globe key too (emoji picker, input source, Apple dictation).
    case globeKey(action: String)

    static func items(
        inputMonitoring: Bool, microphone: Bool, accessibility: Bool, shortcutMode: String,
        globeUsageType: Int?
    ) -> [PermissionAttention] {
        var items: [PermissionAttention] = []
        if !inputMonitoring && ShortcutPolicy.usesHoldKey(mode: shortcutMode) {
            items.append(.inputMonitoring(key: ShortcutPolicy.holdKeyName(mode: shortcutMode)))
        }
        if !microphone { items.append(.microphone) }
        if !accessibility { items.append(.accessibility) }
        if ShortcutPolicy.allows(mode: shortcutMode, source: "fn"),
            GlobeKeySetting.conflicts(usageType: globeUsageType)
        {
            items.append(.globeKey(action: GlobeKeySetting.action(usageType: globeUsageType)))
        }
        return items
    }

    /// Dictation cannot start at all (the menu-bar item shows a warning).
    var blocksDictation: Bool {
        switch self {
        case .inputMonitoring, .microphone: return true
        case .accessibility, .globeKey: return false
        }
    }

    var title: String {
        switch self {
        case .inputMonitoring(let key): return "Phonon can't see the \(key) key yet"
        case .microphone: return "Phonon can't use the microphone"
        case .accessibility: return "Phonon can't paste into other apps"
        case .globeKey(let action): return "macOS also uses the Globe key: it \(action)"
        }
    }

    var detail: String {
        switch self {
        case .inputMonitoring:
            return "Turn Phonon on in System Settings › Privacy & Security › Input Monitoring. "
                + "Until then the key does nothing. If it still doesn't work, quit and reopen "
                + "Phonon."
        case .microphone:
            return "Allow Phonon in System Settings › Privacy & Security › Microphone."
        case .accessibility:
            return "Turn Phonon on in System Settings › Privacy & Security › Accessibility. "
                + "Until then each dictation waits on the clipboard for ⌘V."
        case .globeKey:
            return "Set System Settings › Keyboard › \"Press 🌐 key to\" to \"Do Nothing\", so "
                + "pressing it only starts dictation."
        }
    }

    var actionTitle: String {
        switch self {
        case .inputMonitoring: return "Open Input Monitoring"
        case .microphone: return "Allow Microphone"
        case .accessibility: return "Open Accessibility"
        case .globeKey: return "Open Keyboard Settings"
        }
    }

    static let keyboardPane = URL(
        string: "x-apple.systempreferences:com.apple.Keyboard-Settings.extension")!

    /// Short line for the menu-bar menu.
    var menuTitle: String {
        switch self {
        case .inputMonitoring(let key): return "⚠ Input Monitoring is off: \(key) can't start dictation…"
        case .microphone: return "⚠ Microphone access is off…"
        case .accessibility: return "⚠ Accessibility is off: Phonon can't paste…"
        case .globeKey: return "⚠ macOS also uses the Globe key…"
        }
    }
}

/// The banner over every page while something in `PermissionAttention` is wrong.
struct PermissionBanner: View {
    @ObservedObject var store: NativeAppStore

    var body: some View {
        let items = store.permissionAttention
        if !items.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
                ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                    PermissionBannerRow(item: item) { store.performAttentionAction(item) }
                }
            }
            .padding(.horizontal, 30)
            .padding(.top, 18)
        }
    }
}

struct PermissionBannerRow: View {
    let item: PermissionAttention
    let action: () -> Void

    var body: some View {
        HStack(alignment: .center, spacing: 14) {
            Image(systemName: item.blocksDictation
                ? "exclamationmark.triangle.fill" : "exclamationmark.circle.fill")
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(item.blocksDictation ? EmberTheme.ember2 : EmberTheme.warm)
                .frame(width: 22)
            VStack(alignment: .leading, spacing: 3) {
                Text(item.title).font(.system(size: 14, weight: .semibold))
                Text(item.detail)
                    .font(.system(size: 12.5))
                    .foregroundStyle(EmberTheme.soft)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 8)
            Button(item.actionTitle, action: action)
                .buttonStyle(PillButtonStyle(primary: item.blocksDictation))
        }
        .padding(14)
        .frame(maxWidth: 760, alignment: .leading)
        .background(
            item.blocksDictation ? Color(hex: 0x2A140A) : Color(hex: 0x1C150F),
            in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(
                    item.blocksDictation ? EmberTheme.ember.opacity(0.7) : Color(hex: 0x3B2516),
                    lineWidth: 1))
    }
}

struct PermissionSummary: View {
    @ObservedObject var store: NativeAppStore

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            PermissionRow(
                name: "Microphone", granted: store.microphonePermission,
                statusText: store.microphoneStatusText,
                actionTitle: store.microphoneActionTitle,
                action: { store.performMicrophonePermissionAction() })
            PermissionRow(
                name: "Accessibility", granted: store.accessibilityPermission,
                action: { NSWorkspace.shared.open(PrivacyPane.accessibility.settingsURL) })
            PermissionRow(
                name: "Input Monitoring", granted: store.inputMonitoringAvailable,
                actionTitle: store.inputMonitoringActionTitle,
                action: { store.performInputMonitoringPermissionAction() })
            PermissionRow(
                name: "Screen Recording", granted: store.screenRecordingPermission,
                actionTitle: store.screenRecordingActionTitle,
                action: { store.performScreenRecordingPermissionAction() })
        }
    }
}

/// First run only. Both switches retain data, so neither is on until asked.
struct PrivacyChoiceView: View {
    @ObservedObject var store: NativeAppStore
    let onDone: () -> Void
    @State private var localHistory = false
    @State private var screenContext = false
    @State private var clipboard = ClipboardAfterPaste.dictation

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("What Phonon may keep")
                .font(.title2.bold())
            Text(
                "Dictation runs entirely on this Mac either way. These two features store or read more than the transcript, so they start off."
            )
            .foregroundStyle(.secondary)

            VStack(spacing: 0) {
                ToggleRow(
                    title: "Keep recordings and transcripts",
                    detail:
                        "Save the audio and text of each dictation so Home and Insights can show it.",
                    isOn: $localHistory,
                    showsDetail: true
                )
                Divider()
                ToggleRow(
                    title: "Read the screen",
                    detail:
                        "Read the visible windows with on-device OCR to spell technical terms correctly. Screenshots are never saved, and nothing leaves the Mac.",
                    isOn: $screenContext,
                    showsDetail: true
                )
            }
            .padding(14)
            .background(EmberTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 10))

            VStack(alignment: .leading, spacing: 10) {
                Text("After a paste, your clipboard holds")
                    .font(.headline)
                ChoiceChips(
                    options: ClipboardAfterPaste.allCases.map { ($0, $0.title) },
                    selection: clipboard
                ) { clipboard = $0 }
                Text(
                    clipboard == .dictation
                        ? "Press ⌘V to paste your last dictation again anywhere."
                        : "Phonon puts back what you had copied once the paste lands."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(EmberTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 10))

            Text("All of these can be changed any time in Settings, which can also clear all history.")
                .font(.caption)
                .foregroundStyle(.secondary)

            HStack {
                Spacer()
                Button("Continue") {
                    store.recordPrivacyChoice(
                        localHistory: localHistory, screenContext: screenContext,
                        clipboard: clipboard)
                    onDone()
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(24)
        .frame(width: 540)
        .background(EmberTheme.background)
        .foregroundStyle(EmberTheme.text)
        .preferredColorScheme(.dark)
    }
}

struct PermissionGuideView: View {
    let guide: PermissionGuide
    let onDone: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text(guide.title)
                .font(.title2.bold())
            Text(guide.detail)
                .foregroundStyle(.secondary)

            HStack(spacing: 14) {
                Image(nsImage: NSWorkspace.shared.icon(forFile: Bundle.main.bundleURL.path))
                    .resizable()
                    .frame(width: 58, height: 58)
                VStack(alignment: .leading, spacing: 4) {
                    Text("Phonon.app").font(.headline)
                    Text("Installed in Applications and ready to add in System Settings.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(14)
            .background(EmberTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 10))

            Text(guide.manualInstructions)
                .font(.headline)
            Text("After enabling Phonon, quit it completely before reopening so macOS applies the permission.")
                .font(.callout)

            HStack {
                Button("Open Applications") {
                    NSWorkspace.shared.open(
                        URL(fileURLWithPath: "/Applications", isDirectory: true))
                }
                Spacer()
                Button("Open System Settings") {
                    NSWorkspace.shared.open(guide.pane.settingsURL)
                }
                .buttonStyle(.borderedProminent)
                Button("Quit Phonon", action: onDone)
            }
        }
        .padding(24)
        .frame(width: 540)
        .background(EmberTheme.background)
        .foregroundStyle(EmberTheme.text)
        .preferredColorScheme(.dark)
    }
}

enum PrivacyPane: String, CaseIterable {
    case microphone = "Privacy_Microphone"
    case accessibility = "Privacy_Accessibility"
    case inputMonitoring = "Privacy_ListenEvent"
    case screenRecording = "Privacy_ScreenCapture"

    var settingsURL: URL {
        URL(string: "x-apple.systempreferences:com.apple.preference.security?\(rawValue)")!
    }
}

/// A panel in the main window's palette (Insights tiles and charts, the
/// Notes player).
struct Panel<Content: View>: View {
    var padding: CGFloat = 16
    @ViewBuilder let content: Content
    var body: some View {
        content
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(EmberTheme.panel, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .stroke(EmberTheme.panelBorder, lineWidth: 1)
            )
    }
}

struct MicrophonePriorityCard: View {
    @ObservedObject var store: NativeAppStore
    let onSettingsChanged: () -> Void

    /// Three is enough to cover a headset, a desk microphone and the built-in one,
    /// and a shorter list is easier to reason about than an open-ended one.
    private static let maximumRanked = NewMicrophonePolicy.maximumRanked

    private var ranked: [String] { store.settings.microphonePriority }

    private var addable: [String] {
        store.availableMicrophones.filter { candidate in
            !ranked.contains { $0.localizedCaseInsensitiveCompare(candidate) == .orderedSame }
        }
    }

    private func isConnected(_ microphone: String) -> Bool {
        store.availableMicrophones.contains {
            $0.localizedCaseInsensitiveContains(microphone)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if ranked.isEmpty {
                Text("Nothing ranked: Phonon follows the system input.")
                    .foregroundStyle(EmberTheme.mute)
            }

            ForEach(Array(ranked.enumerated()), id: \.offset) { index, microphone in
                let connected = isConnected(microphone)
                let active = microphone.localizedCaseInsensitiveCompare(
                    store.selectedMicrophone) == .orderedSame
                HStack(spacing: 8) {
                    Text("\(index + 1)")
                        .font(.mono(12))
                        .foregroundStyle(EmberTheme.mute)
                        .frame(width: 16)
                    Image(systemName: connected ? "mic.fill" : "mic.slash")
                        .foregroundStyle(
                            active ? EmberTheme.healthy : (connected ? EmberTheme.mute : EmberTheme.dim))
                    Text(microphone)
                        .lineLimit(1)
                        .foregroundStyle(connected ? EmberTheme.ink : EmberTheme.mute)
                    if active {
                        Text("in use").font(.system(size: 11)).foregroundStyle(EmberTheme.healthy)
                    } else if !connected {
                        Text("not connected").font(.system(size: 11)).foregroundStyle(EmberTheme.mute)
                    }
                    Spacer()
                    Button { moveMicrophone(from: index, by: -1) } label: {
                        Image(systemName: "chevron.up")
                    }
                    .buttonStyle(IconButtonStyle())
                    .disabled(index == 0)
                    Button { moveMicrophone(from: index, by: 1) } label: {
                        Image(systemName: "chevron.down")
                    }
                    .buttonStyle(IconButtonStyle())
                    .disabled(index == ranked.count - 1)
                    Button { removeMicrophone(at: index) } label: {
                        Image(systemName: "xmark")
                    }
                    .buttonStyle(IconButtonStyle())
                }
            }

            // Only offer microphones CoreAudio actually reports. Typing a name
            // by hand used to let a ranking silently never match, for example
            // "Yeti" against a device really called "Yeti Stereo Microphone".
            if ranked.count < Self.maximumRanked {
                Menu {
                    if addable.isEmpty {
                        Text("Every microphone found is already ranked")
                    }
                    ForEach(addable, id: \.self) { microphone in
                        Button(microphone) { addMicrophone(microphone) }
                    }
                } label: {
                    Label("Add microphone", systemImage: "plus")
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
                .disabled(addable.isEmpty)
            }
        }
        .help(
            "Phonon follows the microphone selected in System Settings, except a Bluetooth "
                + "headset mic, which would drop the headset to call quality. Rank up to three "
                + "microphones to override that.")
        .onAppear { store.refreshAvailableMicrophones() }
    }

    private func moveMicrophone(from index: Int, by offset: Int) {
        let destination = index + offset
        guard store.settings.microphonePriority.indices.contains(destination) else { return }
        store.updateSettings {
            $0.microphonePriority.swapAt(index, destination)
        }
        onSettingsChanged()
    }

    private func removeMicrophone(at index: Int) {
        guard store.settings.microphonePriority.indices.contains(index) else { return }
        store.updateSettings {
            $0.microphonePriority.remove(at: index)
        }
        onSettingsChanged()
    }

    private func addMicrophone(_ name: String) {
        guard store.settings.microphonePriority.count < Self.maximumRanked,
              !store.settings.microphonePriority.contains(where: {
                  $0.localizedCaseInsensitiveCompare(name) == .orderedSame
              }) else { return }
        store.updateSettings { $0.microphonePriority.append(name) }
        onSettingsChanged()
    }
}

/// A switch row for the first-run privacy sheet, which shows the detail
/// because it asks for consent; elsewhere the detail is the hover tooltip.
struct ToggleRow: View {
    let title: String
    let detail: String
    @Binding var isOn: Bool
    var showsDetail = false
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                if showsDetail {
                    Text(detail).font(.caption).foregroundStyle(.secondary)
                }
            }
            .help(detail)
            Spacer()
            Toggle(title, isOn: $isOn)
                .labelsHidden()
                .toggleStyle(EmberSwitchStyle())
                .accessibilityLabel(title)
        }
    }
}

/// A switch drawn in the ember accent. AppKit's own switch greys its ON track
/// whenever the window is not key, so ON and OFF looked alike in this dark theme.
struct EmberSwitchStyle: ToggleStyle {
    func makeBody(configuration: Configuration) -> some View {
        EmberSwitch(isOn: configuration.$isOn)
    }
}

private struct EmberSwitch: View {
    @Binding var isOn: Bool
    @Environment(\.isEnabled) private var isEnabled

    var body: some View {
        Button {
            isOn.toggle()
        } label: {
            ZStack(alignment: isOn ? .trailing : .leading) {
                Capsule()
                    .fill(isOn ? EmberTheme.ember : Color(hex: 0x48443F))
                Circle()
                    .fill(Color.white)
                    .padding(2)
                    .shadow(color: .black.opacity(0.35), radius: 1, y: 0.5)
            }
            .frame(width: 32, height: 19)
            .contentShape(Capsule())
            .animation(.easeOut(duration: 0.15), value: isOn)
        }
        .buttonStyle(.plain)
        .opacity(isEnabled ? 1 : 0.4)
        .accessibilityValue(isOn ? "On" : "Off")
        .accessibilityAddTraits(.isToggle)
    }
}

struct PermissionRow: View {
    let name: String
    let granted: Bool
    var statusText: String?
    var actionTitle: String?
    let action: () -> Void
    var body: some View {
        HStack {
            Image(systemName: granted ? "checkmark.circle.fill" : "exclamationmark.circle.fill")
                .foregroundStyle(granted ? EmberTheme.healthy : EmberTheme.warm)
            Text(name)
            Spacer()
            Text(statusText ?? (granted ? "Granted" : "Needs access"))
                .font(.system(size: 12))
                .foregroundStyle(EmberTheme.mute)
            Button(actionTitle ?? "Settings", action: action)
                .buttonStyle(PillButtonStyle())
        }
    }
}
