import Charts
import SwiftUI

// MARK: - Numbers from the local corpus

/// Everything the Insights page draws, computed from the dictations kept in
/// local history (the corpus the app already reads). Nothing else is counted.
struct InsightsData: Equatable {
    struct Day: Identifiable, Equatable {
        let date: Date
        let words: Int
        var id: Date { date }
    }

    struct HourCell: Identifiable, Equatable {
        /// 0 is Monday.
        let weekday: Int
        let hour: Int
        let words: Int
        var id: Int { weekday * 24 + hour }
    }

    struct AppShare: Identifiable, Equatable {
        let bundleID: String?
        let name: String
        let words: Int
        let share: Double
        var id: String { bundleID ?? name }
    }

    struct YearCell: Identifiable, Equatable {
        let date: Date
        let week: Int
        /// 0 is Monday.
        let weekday: Int
        let words: Int
        var id: Date { date }
    }

    struct Week: Identifiable, Equatable {
        let start: Date
        let wpm: Int
        var id: Date { start }
    }

    var wordsPerMinute = 0
    var words = 0
    /// Words of what was heard that Phonon changed or dropped (case and
    /// punctuation aside), summed over every dictation.
    var wordsFixed = 0
    var days: [Day] = []
    var hours: [HourCell] = []
    var apps: [AppShare] = []
    var year: [YearCell] = []
    var speed: [Week] = []

    var topApp: AppShare? { apps.first { $0.bundleID != nil } }

    static func build(
        _ history: [NativeHistoryItem], usage: NativeUsageStats, now: Date = Date(),
        calendar inputCalendar: Calendar = .current
    ) -> InsightsData {
        var calendar = inputCalendar
        calendar.firstWeekday = 2
        var data = InsightsData()
        data.wordsPerMinute = usage.wordsPerMinute
        let today = calendar.startOfDay(for: now)
        var perDay: [Date: Int] = [:]
        var perHour = [Int](repeating: 0, count: 7 * 24)
        var perApp: [String: (name: String?, words: Int)] = [:]
        var unknownWords = 0
        var weekWords: [Date: Int] = [:]
        var weekMs: [Date: UInt64] = [:]
        let twelveWeeksAgo = calendar.date(byAdding: .day, value: -84, to: today) ?? today

        for item in history {
            let words = wordCount(item.displayText)
            guard words > 0 else { continue }
            data.words += words
            data.wordsFixed += fixedWords(heard: item.metadata.rawTranscript, wrote: item.metadata.finalTranscript)
            let day = calendar.startOfDay(for: item.date)
            perDay[day, default: 0] += words
            if item.date >= twelveWeeksAgo {
                let weekday = (calendar.component(.weekday, from: item.date) + 5) % 7
                perHour[weekday * 24 + calendar.component(.hour, from: item.date)] += words
            }
            if let app = item.metadata.appContext, !app.bundleID.isEmpty {
                let current = perApp[app.bundleID] ?? (app.appName, 0)
                perApp[app.bundleID] = (current.name ?? app.appName, current.words + words)
            } else {
                unknownWords += words
            }
            if let ms = item.metadata.audioDurationMs, ms > 0,
                let week = calendar.dateInterval(of: .weekOfYear, for: item.date)?.start
            {
                weekWords[week, default: 0] += words
                weekMs[week, default: 0] += ms
            }
        }

        data.days = (0..<30).reversed().compactMap { back in
            calendar.date(byAdding: .day, value: -back, to: today).map {
                Day(date: $0, words: perDay[$0] ?? 0)
            }
        }
        data.hours = perHour.indices.map {
            HourCell(weekday: $0 / 24, hour: $0 % 24, words: perHour[$0])
        }

        let total = max(1, perApp.values.reduce(0) { $0 + $1.words } + unknownWords)
        let ranked = perApp.sorted { $0.value.words > $1.value.words }
        var apps = ranked.prefix(5).map {
            AppShare(
                bundleID: $0.key, name: $0.value.name ?? $0.key, words: $0.value.words,
                share: Double($0.value.words) / Double(total))
        }
        let rest = ranked.dropFirst(5).reduce(0) { $0 + $1.value.words } + unknownWords
        if rest > 0 {
            apps.append(
                AppShare(
                    bundleID: nil, name: "Other", words: rest,
                    share: Double(rest) / Double(total)))
        }
        data.apps = apps

        // 53 columns of Monday-first weeks, ending with this week.
        let thisWeek = calendar.dateInterval(of: .weekOfYear, for: today)?.start ?? today
        let firstWeek = calendar.date(byAdding: .day, value: -52 * 7, to: thisWeek) ?? thisWeek
        data.year = (0..<(53 * 7)).compactMap { offset in
            guard let date = calendar.date(byAdding: .day, value: offset, to: firstWeek),
                date <= today
            else { return nil }
            return YearCell(
                date: date, week: offset / 7, weekday: offset % 7, words: perDay[date] ?? 0)
        }

        data.speed = weekWords.keys.sorted().suffix(16).compactMap { week in
            guard let ms = weekMs[week], ms > 0 else { return nil }
            return Week(
                start: week,
                wpm: Int((Double(weekWords[week] ?? 0) * 60_000 / Double(ms)).rounded()))
        }
        return data
    }

    static func wordCount(_ text: String) -> Int {
        text.split(whereSeparator: { $0.isWhitespace }).count
    }

    /// Heard words Phonon changed or dropped, comparing words without case
    /// and edge punctuation ("Kubloss" -> "cuBLAS" counts; "the" -> "The" does not).
    static func fixedWords(heard: String, wrote: String) -> Int {
        let normalize = { (text: String) in
            text.split(whereSeparator: { $0.isWhitespace })
                .map { $0.lowercased().trimmingCharacters(in: .punctuationCharacters) }
                .filter { !$0.isEmpty }
                .joined(separator: " ")
        }
        let before = normalize(heard)
        let after = normalize(wrote)
        guard before != after, !before.isEmpty, !after.isEmpty else { return 0 }
        return WordDiff.compare(before, after).before.filter(\.changed).count
    }

    /// Ramp level 0...4 for a cell, against the busiest cell.
    static func level(_ value: Int, max top: Int) -> Int {
        guard value > 0, top > 0 else { return 0 }
        return min(4, 1 + Int(Double(value) / Double(top) * 4))
    }
}

// MARK: - Page

struct InsightsView: View {
    @ObservedObject var store: NativeAppStore
    /// Fixed instant for renders; nil is now.
    var now: Date?
    @State private var data = InsightsData()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                PageTitle("Insights").padding(.bottom, -14)
                HStack(spacing: 12) {
                    tile(value: "\(data.wordsPerMinute)", label: "words per minute")
                    tile(value: data.words.formatted(), label: "words dictated")
                    Panel {
                        VStack(alignment: .leading, spacing: 3) {
                            HStack(spacing: 8) {
                                AppIconView(bundleID: data.topApp?.bundleID)
                                Text(
                                    data.topApp.map {
                                        AppIcons.name($0.bundleID, fallback: $0.name)
                                    } ?? "None yet"
                                )
                                .font(.system(size: 17, weight: .semibold))
                                .lineLimit(1)
                            }
                            .frame(height: 28)
                            Text("where you dictate most")
                                .font(.system(size: 12)).foregroundStyle(EmberTheme.mute)
                        }
                    }
                    tile(value: data.wordsFixed.formatted(), label: "words Phonon fixed")
                        .help(
                            "Words of what was heard that Phonon changed or dropped, case and "
                                + "punctuation aside.")
                }

                ChartPanel("Words per day", caption: "Last 30 days") {
                    WordsPerDayChart(days: data.days).frame(height: 160)
                }
                HStack(alignment: .top, spacing: 14) {
                    ChartPanel("When you dictate", caption: "Words by hour and weekday, last 12 weeks") {
                        HourHeatmap(cells: data.hours).frame(height: 170)
                        RampLegend()
                    }
                    .frame(width: 396)
                    ChartPanel("Where you dictate", caption: "Share of words by app") {
                        AppShareChart(apps: data.apps)
                            .frame(height: CGFloat(max(1, data.apps.count)) * 30 + 10)
                    }
                    .frame(maxWidth: .infinity)
                }
                ChartPanel("Every day this year", caption: "Words dictated per day") {
                    YearCalendar(cells: data.year).frame(height: 118)
                    RampLegend()
                }
                ChartPanel("Speed over time", caption: "Weekly average words per minute") {
                    SpeedChart(weeks: data.speed).frame(height: 140)
                }
            }
            .padding(.horizontal, 30)
            .padding(.top, 24)
            .padding(.bottom, 24)
        }
        .scrollIndicators(.never)
        .onAppear(perform: rebuild)
        .onChange(of: store.history) { _, _ in rebuild() }
    }

    private func rebuild() {
        data = InsightsData.build(store.history, usage: store.usage, now: now ?? Date())
    }

    private func tile(value: String, label: String) -> some View {
        Panel {
            VStack(alignment: .leading, spacing: 3) {
                Text(value).font(.mono(26, .semibold)).tracking(-0.5).lineLimit(1)
                    .minimumScaleFactor(0.6)
                    .frame(height: 28, alignment: .leading)
                Text(label).font(.system(size: 12)).foregroundStyle(EmberTheme.mute)
            }
        }
    }
}

struct ChartPanel<Content: View>: View {
    let title: String
    let caption: String
    @ViewBuilder let content: Content
    init(_ title: String, caption: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.caption = caption
        self.content = content()
    }
    var body: some View {
        Panel(padding: 16) {
            VStack(alignment: .leading, spacing: 0) {
                Text(title).font(.system(size: 14, weight: .semibold))
                Text(caption).font(.system(size: 12)).foregroundStyle(EmberTheme.mute)
                    .padding(.top, 2).padding(.bottom, 12)
                content
            }
        }
    }
}

struct RampLegend: View {
    var body: some View {
        HStack(spacing: 4) {
            Spacer()
            Text("less")
            ForEach(EmberTheme.ramp.indices, id: \.self) { index in
                RoundedRectangle(cornerRadius: 3).fill(EmberTheme.ramp[index])
                    .frame(width: 11, height: 11)
            }
            Text("more")
        }
        .font(.system(size: 11))
        .foregroundStyle(EmberTheme.mute)
        .padding(.top, 8)
    }
}

/// The hover tooltip: date or bucket, then the number in mono.
struct ChartTip: View {
    let label: String
    let value: String
    var body: some View {
        HStack(spacing: 4) {
            Text(label)
            Text(value).font(.mono(12, .semibold))
        }
        .font(.system(size: 12))
        .foregroundStyle(EmberTheme.ink)
        .padding(.horizontal, 9)
        .padding(.vertical, 6)
        .background(Color(hex: 0x0B0807), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(Color(hex: 0x3B2516), lineWidth: 1))
        .shadow(color: .black.opacity(0.6), radius: 10, y: 8)
        .fixedSize()
    }
}

private let axisFont = Font.mono(10.5)

private func dayLabel(_ date: Date, weekday: Bool = false) -> String {
    let formatter = DateFormatter()
    formatter.locale = Locale(identifier: "en_US_POSIX")
    formatter.dateFormat = weekday ? "EEE, MMM d" : "MMM d"
    return formatter.string(from: date)
}

private func words(_ value: Int) -> String { "\(value.formatted()) words" }

/// Hover over a chart: the plot-area point under the pointer, or nil.
private struct HoverOverlay: View {
    let proxy: ChartProxy
    let onHover: (CGPoint?) -> Void
    var body: some View {
        GeometryReader { geometry in
            Rectangle().fill(.clear).contentShape(Rectangle())
                .onContinuousHover { phase in
                    guard case .active(let location) = phase, let frame = proxy.plotFrame else {
                        onHover(nil)
                        return
                    }
                    let origin = geometry[frame].origin
                    onHover(CGPoint(x: location.x - origin.x, y: location.y - origin.y))
                }
        }
    }
}

struct WordsPerDayChart: View {
    let days: [InsightsData.Day]
    @State private var hovered: Date?

    var body: some View {
        Chart {
            ForEach(days) { day in
                BarMark(
                    x: .value("Day", day.date, unit: .day),
                    y: .value("Words", day.words), width: .ratio(0.72)
                )
                .cornerRadius(4)
                .foregroundStyle(
                    EmberTheme.ember.opacity(
                        hovered == day.date || day.id == days.last?.id ? 1 : 0.8))
            }
            if let hovered, let day = days.first(where: { $0.date == hovered }) {
                RuleMark(x: .value("Day", day.date, unit: .day))
                    .foregroundStyle(.clear)
                    .annotation(
                        position: .top, spacing: 0,
                        overflowResolution: .init(x: .fit(to: .chart), y: .fit(to: .chart))
                    ) {
                        ChartTip(label: dayLabel(day.date, weekday: true) + " ·", value: words(day.words))
                    }
            }
        }
        .chartXAxis {
            AxisMarks(values: .stride(by: .day, count: 7)) { value in
                AxisValueLabel {
                    if let date = value.as(Date.self) { Text(dayLabel(date)).font(axisFont) }
                }
                .foregroundStyle(EmberTheme.dim)
            }
        }
        .chartYAxis {
            AxisMarks(position: .leading, values: .automatic(desiredCount: 4)) { value in
                AxisGridLine().foregroundStyle(Color(hex: 0x241E19))
                AxisValueLabel {
                    if let words = value.as(Int.self) {
                        Text(words >= 1_000 ? "\(Double(words) / 1_000, specifier: "%g")k" : "\(words)")
                            .font(axisFont)
                    }
                }
                .foregroundStyle(EmberTheme.dim)
            }
        }
        .chartOverlay { proxy in
            HoverOverlay(proxy: proxy) { point in
                hovered = point.flatMap { proxy.value(atX: $0.x, as: Date.self) }
                    .map { Calendar.current.startOfDay(for: $0) }
            }
        }
        .animation(Motion.quick, value: hovered)
    }
}

struct HourHeatmap: View {
    let cells: [InsightsData.HourCell]
    @State private var hovered: Int?
    private static let weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    var body: some View {
        let top = cells.map(\.words).max() ?? 0
        Chart {
            ForEach(cells) { cell in
                RectangleMark(
                    xStart: .value("Hour", Double(cell.hour) + 0.06),
                    xEnd: .value("Hour", Double(cell.hour) + 0.94),
                    yStart: .value("Day", Double(6 - cell.weekday) + 0.06),
                    yEnd: .value("Day", Double(6 - cell.weekday) + 0.94)
                )
                .cornerRadius(4)
                .foregroundStyle(EmberTheme.ramp[InsightsData.level(cell.words, max: top)])
            }
            if let hovered, let cell = cells.first(where: { $0.id == hovered }) {
                PointMark(
                    x: .value("Hour", Double(cell.hour) + 0.5),
                    y: .value("Day", Double(6 - cell.weekday) + 0.5))
                    .symbolSize(0)
                    .annotation(
                        position: .top, spacing: 4,
                        overflowResolution: .init(x: .fit(to: .chart), y: .fit(to: .chart))
                    ) {
                        ChartTip(
                            label: "\(Self.weekdays[cell.weekday]) \(cell.hour):00-\(cell.hour + 1):00 ·",
                            value: words(cell.words))
                    }
            }
        }
        .chartXScale(domain: 0...24)
        .chartYScale(domain: 0...7)
        .chartXAxis {
            AxisMarks(values: [0.5, 6.5, 12.5, 18.5]) { value in
                AxisValueLabel {
                    if let x = value.as(Double.self) { Text(Self.hourLabel(Int(x))).font(axisFont) }
                }
                .foregroundStyle(EmberTheme.dim)
            }
        }
        .chartYAxis {
            AxisMarks(position: .leading, values: (0..<7).map { Double($0) + 0.5 }) { value in
                AxisValueLabel {
                    if let y = value.as(Double.self) {
                        Text(Self.weekdays[6 - Int(y)]).font(axisFont)
                    }
                }
                .foregroundStyle(EmberTheme.dim)
            }
        }
        .chartOverlay { proxy in
            HoverOverlay(proxy: proxy) { point in
                guard let point, let (x, y) = proxy.value(at: point, as: (Double, Double).self),
                    (0..<24).contains(Int(x)), (0..<7).contains(Int(y))
                else {
                    hovered = nil
                    return
                }
                hovered = (6 - Int(y)) * 24 + Int(x)
            }
        }
    }

    static func hourLabel(_ hour: Int) -> String {
        switch hour {
        case 0: return "12a"
        case 12: return "12p"
        case ..<12: return "\(hour)a"
        default: return "\(hour - 12)p"
        }
    }
}

struct AppShareChart: View {
    let apps: [InsightsData.AppShare]
    @State private var hovered: String?

    var body: some View {
        // App labels (icon and name) in a column beside the plot, one per
        // band; the chart's own axis labels do not center on the bars.
        HStack(spacing: 8) {
            VStack(spacing: 0) {
                ForEach(apps) { app in
                    HStack(spacing: 8) {
                        AppIconView(bundleID: app.bundleID, size: 20)
                        Text(name(app)).font(.system(size: 12)).foregroundStyle(EmberTheme.soft)
                            .lineLimit(1)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
                }
            }
            .frame(width: 92)
            chart
        }
    }

    private var chart: some View {
        let top = apps.map(\.share).max() ?? 1
        return Chart {
            ForEach(apps) { app in
                BarMark(
                    x: .value("Share", app.share),
                    y: .value("App", app.id), height: .fixed(14)
                )
                .cornerRadius(4)
                .foregroundStyle(EmberTheme.ember.opacity(app.id == apps.first?.id || hovered == app.id ? 1 : 0.8))
                .annotation(position: .trailing, spacing: 8) {
                    Text("\(Int((app.share * 100).rounded()))%")
                        .font(axisFont).foregroundStyle(EmberTheme.soft)
                }
            }
            if let hovered, let app = apps.first(where: { $0.id == hovered }) {
                RuleMark(y: .value("App", app.id))
                    .foregroundStyle(.clear)
                    .annotation(
                        position: .bottom, spacing: 0,
                        overflowResolution: .init(x: .fit(to: .chart), y: .fit(to: .chart))
                    ) {
                        ChartTip(label: "\(name(app)) ·", value: "\(Int((app.share * 100).rounded()))% of words")
                    }
            }
        }
        .chartXScale(domain: 0...(top * 1.2))
        .chartXAxis(.hidden)
        .chartYAxis(.hidden)
        .chartYScale(range: .plotDimension(padding: 0))
        .chartOverlay { proxy in
            HoverOverlay(proxy: proxy) { point in
                hovered = point.flatMap { proxy.value(atY: $0.y, as: String.self) }
            }
        }
    }

    private func name(_ app: InsightsData.AppShare) -> String {
        app.bundleID == nil ? app.name : AppIcons.name(app.bundleID, fallback: app.name)
    }
}

struct YearCalendar: View {
    let cells: [InsightsData.YearCell]
    @State private var hovered: Date?

    var body: some View {
        let top = cells.map(\.words).max() ?? 0
        Chart {
            ForEach(cells) { cell in
                RectangleMark(
                    xStart: .value("Week", Double(cell.week) + 0.1),
                    xEnd: .value("Week", Double(cell.week) + 0.9),
                    yStart: .value("Day", Double(6 - cell.weekday) + 0.1),
                    yEnd: .value("Day", Double(6 - cell.weekday) + 0.9)
                )
                .cornerRadius(2.5)
                .foregroundStyle(EmberTheme.ramp[InsightsData.level(cell.words, max: top)])
            }
            if let hovered, let cell = cells.first(where: { $0.date == hovered }) {
                PointMark(
                    x: .value("Week", Double(cell.week) + 0.5),
                    y: .value("Day", Double(6 - cell.weekday) + 0.5))
                    .symbolSize(0)
                    .annotation(
                        position: .top, spacing: 4,
                        overflowResolution: .init(x: .fit(to: .chart), y: .fit(to: .chart))
                    ) {
                        ChartTip(label: dayLabel(cell.date) + " ·", value: words(cell.words))
                    }
            }
        }
        .chartXScale(domain: 0...53)
        .chartYScale(domain: 0...7)
        .chartXAxis {
            AxisMarks(values: monthStarts.map(\.week)) { value in
                AxisValueLabel {
                    if let week = value.as(Double.self),
                        let start = monthStarts.first(where: { $0.week == week })
                    {
                        Text(start.label).font(axisFont)
                    }
                }
                .foregroundStyle(EmberTheme.dim)
            }
        }
        .chartYAxis {
            AxisMarks(position: .leading, values: [6.5, 4.5, 2.5]) { value in
                AxisValueLabel {
                    if let y = value.as(Double.self) {
                        Text(["Mon", "Wed", "Fri"][(6 - Int(y)) / 2]).font(axisFont)
                    }
                }
                .foregroundStyle(EmberTheme.dim)
            }
        }
        .chartOverlay { proxy in
            HoverOverlay(proxy: proxy) { point in
                guard let point, let (x, y) = proxy.value(at: point, as: (Double, Double).self)
                else {
                    hovered = nil
                    return
                }
                hovered = cells.first { $0.week == Int(x) && $0.weekday == 6 - Int(y) }?.date
            }
        }
    }

    /// The first week column of each month, labelled "Jan", "Feb"...
    static func monthStarts(for cells: [InsightsData.YearCell], calendar: Calendar = .current)
        -> [(week: Double, label: String)]
    {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "MMM"
        var seen = Set<Int>()
        var starts: [(Double, String)] = []
        for cell in cells where cell.weekday == 0 {
            let year = calendar.component(.year, from: cell.date)
            let month = calendar.component(.month, from: cell.date)
            let yearMonth = year * 100 + month
            guard calendar.component(.day, from: cell.date) <= 7, !seen.contains(yearMonth)
            else { continue }
            seen.insert(yearMonth)
            starts.append((Double(cell.week) + 0.5, formatter.string(from: cell.date)))
        }
        return starts
    }

    private var monthStarts: [(week: Double, label: String)] {
        Self.monthStarts(for: cells)
    }
}

struct SpeedChart: View {
    let weeks: [InsightsData.Week]
    @State private var hovered: Date?

    var body: some View {
        let low = max(0, (weeks.map(\.wpm).min() ?? 0) - 20)
        let high = (weeks.map(\.wpm).max() ?? 100) + 20
        Chart {
            ForEach(weeks) { week in
                AreaMark(
                    x: .value("Week", week.start), yStart: .value("Low", low),
                    yEnd: .value("wpm", week.wpm)
                )
                .foregroundStyle(
                    LinearGradient(
                        colors: [EmberTheme.ember.opacity(0.28), EmberTheme.ember.opacity(0)],
                        startPoint: .top, endPoint: .bottom))
                LineMark(x: .value("Week", week.start), y: .value("wpm", week.wpm))
                    .foregroundStyle(EmberTheme.ember2)
                    .lineStyle(StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
            }
            if let last = weeks.last {
                PointMark(x: .value("Week", last.start), y: .value("wpm", last.wpm))
                    .foregroundStyle(EmberTheme.ember2)
                    .symbolSize(50)
                    .annotation(position: .topLeading, spacing: 4) {
                        Text("\(last.wpm) wpm").font(.mono(12)).foregroundStyle(EmberTheme.ink)
                    }
            }
            if let hovered, let week = weeks.first(where: { $0.start == hovered }) {
                RuleMark(x: .value("Week", week.start))
                    .foregroundStyle(Color(hex: 0x4A3324))
                PointMark(x: .value("Week", week.start), y: .value("wpm", week.wpm))
                    .foregroundStyle(EmberTheme.ember2)
                    .symbolSize(60)
                    .annotation(
                        position: .top, spacing: 6,
                        overflowResolution: .init(x: .fit(to: .chart), y: .fit(to: .chart))
                    ) {
                        ChartTip(label: "Week of \(dayLabel(week.start)) ·", value: "\(week.wpm) wpm")
                    }
            }
        }
        .chartYScale(domain: low...high)
        .chartXScale(range: .plotDimension(startPadding: 4, endPadding: 30))
        .chartXAxis {
            AxisMarks(values: .automatic(desiredCount: 5)) { value in
                AxisValueLabel {
                    if let date = value.as(Date.self) { Text(dayLabel(date)).font(axisFont) }
                }
                .foregroundStyle(EmberTheme.dim)
            }
        }
        .chartYAxis {
            AxisMarks(position: .leading, values: .automatic(desiredCount: 3)) { value in
                AxisGridLine().foregroundStyle(Color(hex: 0x241E19))
                AxisValueLabel().font(axisFont).foregroundStyle(EmberTheme.dim)
            }
        }
        .chartOverlay { proxy in
            HoverOverlay(proxy: proxy) { point in
                guard let point, let date = proxy.value(atX: point.x, as: Date.self) else {
                    hovered = nil
                    return
                }
                hovered = weeks.min {
                    abs($0.start.timeIntervalSince(date)) < abs($1.start.timeIntervalSince(date))
                }?.start
            }
        }
        .overlay {
            if weeks.isEmpty {
                Text("No timed dictations yet.").foregroundStyle(EmberTheme.mute)
            }
        }
    }
}
