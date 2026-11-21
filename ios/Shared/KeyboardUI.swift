import SwiftUI

/// What the keyboard view asks of its host: the keyboard extension in real use, a stub in the app's preview.
protocol KeyboardHost: AnyObject {
    func insert(_ text: String)
    func deleteBackward()
    func nextKeyboard()
    var contextBefore: String { get }
    func micDown()
    func micUp()
    func cancelDictation()
    func undoDictation()
    func openApp()
    /// The bar was tapped with the Phonon app not holding the mic: open it, then dictate once it is back.
    func startViaApp()
}

enum KeyboardPhase: Equatable {
    case needsAccess  // Allow Full Access is off
    case needsSession  // the Phonon app is not holding the mic
    case ready
    case listening
    case working  // transcribing and refining
}

final class KeyboardModel: ObservableObject {
    @Published var phase: KeyboardPhase = .needsSession
    /// The last status ("Didn't catch that", "Cancelled"...): never drawn on the bar, read by VoiceOver in its hint.
    @Published var status = ""
    @Published var levels: [CGFloat] = Array(repeating: 0, count: 28)
    @Published var mode: WritingMode = .minimal
    @Published var canUndo = false
    @Published var showGlobe = true
    @Published var latched = false
    /// Bumped by the host when the text around the cursor changes (re-evaluates auto-capitalisation).
    @Published var contextVersion = 0
    /// False while the keyboard is off screen; the light's widths hold still.
    @Published var visible = true
    /// Low Power Mode: the light's widths hold still.
    @Published var lowPower = ProcessInfo.processInfo.isLowPowerModeEnabled
    /// The dictation's stage for the talk bar's light (BarStages.swift), replaced at each real event.
    @Published var beat = BarBeat.rest
    /// The bar was pressed and the keyboard is asking the app (up to 0.4 s) before it listens: the bar is lit, so
    /// the side button is already the X.
    @Published var pendingMic = false
    /// Reduce Motion or Low Power Mode (the view keeps it current): the stages use their no-motion look.
    var reducedMotion = false
    weak var host: KeyboardHost?
    private var powerObserver: NSObjectProtocol?
    private var restWork: DispatchWorkItem?

    private var currentLevel: Double { VoiceLevel.smooth(levels) }

    /// The button beside the bar: the X whenever a dictation is lit or pending, the settings circle otherwise.
    var sideControl: TalkBarSideControl { TalkBarSideControl(phase: phase, beat: beat.stage, pending: pendingMic) }

    /// The press: the bar lights this frame, before the mic, the app or the model have answered. The previous
    /// dictation's levels are cleared, so its last loudness never feeds this ignite's decay.
    func ignite() {
        restWork?.cancel()
        levels = Array(repeating: 0, count: levels.count)
        beat = .ignite(at: Date())
    }

    /// The key was released or the bar tapped to finish: the light gathers while the text is written. Also from
    /// rest while listening (the beat was lost, e.g. a start the keyboard did not light), so the gather still runs.
    func startProcessing() {
        switch beat.stage {
        case .listening, .processing: break
        case .rest where phase == .listening: break
        default: return
        }
        restWork?.cancel()
        beat = beat.processing(at: Date())
    }

    /// The keyboard went off screen. The light returns to rest only when no dictation is under
    /// way, so one still running keeps its beat and its gather or fade still happen.
    func offScreen(dictating: Bool) {
        visible = false
        guard !dictating else { return }
        restWork?.cancel()
        beat = .rest
    }

    /// The dictation ended (text inserted, empty result, error, cancelled, released before the app answered): the
    /// core fades out, no pulse, then the resting bar.
    func fadeOut() {
        end(beat.fadeOut(at: Date(), level: currentLevel, reduceMotion: reducedMotion))
    }

    private func end(_ next: BarBeat) {
        beat = next
        restWork?.cancel()
        guard let settle = next.settleSeconds(reduceMotion: reducedMotion) else { return }
        let work = DispatchWorkItem { [weak self] in
            guard let self, self.beat == next else { return }
            self.beat = .rest
        }
        restWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + settle + 0.05, execute: work)
    }

    init() {
        powerObserver = NotificationCenter.default.addObserver(
            forName: .NSProcessInfoPowerStateDidChange, object: nil, queue: .main
        ) { [weak self] _ in self?.lowPower = ProcessInfo.processInfo.isLowPowerModeEnabled }
    }

    deinit { if let powerObserver { NotificationCenter.default.removeObserver(powerObserver) } }

    func push(level: CGFloat) {
        levels.removeFirst()
        levels.append(max(0, min(1, level)))
    }
}

private enum Layer { case letters, numbers, symbols }

struct KeyboardView: View {
    @ObservedObject var model: KeyboardModel
    @State private var layer: Layer = .letters
    @State private var shift = true  // starts capitalised
    @State private var capsLock = false
    @State private var lastShiftTap = Date.distantPast
    @State private var lastSpace = Date.distantPast
    @State private var micPressedAt: Date?

    private let keyH: CGFloat = 43
    private let gap: CGFloat = 6

    var body: some View {
        VStack(spacing: 0) {
            toolbar
                .frame(height: 48)
                .padding(.horizontal, 6)
                .padding(.top, 4)
            keys
                .padding(.horizontal, 3)
                .padding(.top, 10)
                .padding(.bottom, 4)
        }
        .background(Color(hex: 0x16120F))
        .onAppear(perform: autoShift)
        .onChange(of: model.contextVersion) { _, _ in autoShift() }
    }

    // MARK: toolbar

    /// One bar, always the same view (only its contents change), so a press that starts listening also gets its
    /// release: hold and let go to finish, or tap once to lock it on and tap again to finish. The writing mode lives in
    /// the app. (The old toolbar swapped the mic button for a different view on press, which dropped the release.)
    /// The side button is always there, so the bar never changes width: an X while dictating, a settings circle
    /// (opens Phonon) otherwise.
    private var toolbar: some View {
        TalkBarStrip(control: model.sideControl, onSide: {
            tick()
            press = .none
            if let host = model.host { model.sideControl.perform(on: host) }
        }) {
            talkBar
        }
    }

    private enum Press { case none, starting, stopping }
    @State private var press: Press = .none

    private var off: Bool { model.phase == .needsSession || model.phase == .needsAccess }

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// One capsule of fixed size in every state; only the fill changes, and it shows colours only: a plain dark
    /// capsule at rest, the ember light while dictating. No status text is drawn on it (VoiceOver still reads the
    /// status in the hint). With the app not running it looks the same (a tap opens Phonon). Only no Full Access
    /// keeps a short label, since it needs an instruction to recover.
    private var talkBar: some View {
        let animate = model.visible && !model.lowPower && !reduceMotion
        return TalkBarFill(state: TalkBarFill.Mode(phase: model.phase), level: voiceLevel, animate: animate,
                           followLevel: !reduceMotion, beat: model.beat)
            .onAppear { model.reducedMotion = reduceMotion || model.lowPower }
            .onChange(of: reduceMotion) { _, _ in model.reducedMotion = reduceMotion || model.lowPower }
            .onChange(of: model.lowPower) { _, _ in model.reducedMotion = reduceMotion || model.lowPower }
            .overlay {
                if model.phase == .needsAccess { barText(barLabel) }
            }
            .contentShape(Capsule())
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in
                    guard micPressedAt == nil else { return }
                    micPressedAt = Date()
                    if off {
                        model.host?.startViaApp()
                    } else if model.phase == .listening {
                        press = .stopping  // locked on: this tap finishes it
                    } else if model.phase == .ready {
                        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                        press = .starting
                        model.latched = false
                        model.ignite()  // lit this frame, before the mic, the app or the model answer
                        model.host?.micDown()
                    }
                }
                .onEnded { _ in
                    defer { micPressedAt = nil; press = .none }
                    guard let t = micPressedAt else { return }
                    let held = Date().timeIntervalSince(t)
                    switch press {
                    case .stopping:
                        model.latched = false
                        UIImpactFeedbackGenerator(style: .light).impactOccurred()
                        model.host?.micUp()
                    case .starting:
                        if held < 0.35 {
                            model.latched = true  // a quick tap: stay on until the next tap
                        } else {
                            UIImpactFeedbackGenerator(style: .light).impactOccurred()
                            model.host?.micUp()
                        }
                    case .none:
                        break
                    }
                }
        )
        .accessibilityLabel(model.phase == .listening ? "Finish dictation" : (off ? barLabel : "Dictate"))
        .accessibilityHint(hint)
        .accessibilityAddTraits(.isButton)
    }

    private func barText(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 13, weight: .medium))
            .foregroundStyle(Ember.soft)
            .lineLimit(1)
            .minimumScaleFactor(0.8)
            .padding(.horizontal, 16)
    }

    /// The live input level the capsule lights with: the mean of the newest three meter samples.
    private var voiceLevel: Double { VoiceLevel.smooth(model.levels) }

    private var hint: String {
        switch model.phase {
        case .listening: return model.latched ? "Tap to finish" : "Let go to finish"
        case .working: return "Writing it down"
        case .needsSession: return model.status.isEmpty ? "Opens Phonon once to keep the mic ready" : model.status
        default: return model.status.isEmpty ? "Hold to talk, or tap" : model.status
        }
    }

    private var barLabel: String {
        switch model.phase {
        case .needsAccess: return "Allow Full Access to dictate"
        case .needsSession: return "Start Phonon"
        default: return model.status.isEmpty ? "Hold to talk, or tap" : model.status
        }
    }

    // MARK: keys

    private var keys: some View {
        GeometryReader { geo in
            let unit = (geo.size.width - gap * 9) / 10
            VStack(spacing: 11) {
                switch layer {
                case .letters:
                    letterRow("qwertyuiop", unit: unit)
                    letterRow("asdfghjkl", unit: unit)
                    HStack(spacing: gap) {
                        Key(symbol: capsLock ? "capslock.fill" : (shift ? "shift.fill" : "shift"), special: true,
                            highlighted: shift || capsLock, width: unit * 1.35) { tapShift() }
                        Spacer(minLength: 0)
                        ForEach(Array("zxcvbnm"), id: \.self) { c in letter(String(c), unit: unit) }
                        Spacer(minLength: 0)
                        Key(symbol: "delete.left", special: true, width: unit * 1.35, repeats: true) {
                            model.host?.deleteBackward()
                            autoShift()
                        }
                    }
                case .numbers, .symbols:
                    let rows = layer == .numbers
                        ? ["1234567890", "-/:;()$&@\"", ".,?!'"]
                        : ["[]{}#%^*+=", "_\\|~<>€£¥•", ".,?!'"]
                    charRow(rows[0], unit: unit)
                    charRow(rows[1], unit: unit)
                    HStack(spacing: gap) {
                        Key(label: layer == .numbers ? "#+=" : "123", special: true, width: unit * 1.35) {
                            layer = layer == .numbers ? .symbols : .numbers
                        }
                        Spacer(minLength: 0)
                        ForEach(Array(rows[2]), id: \.self) { c in
                            Key(label: String(c), width: unit * 1.4) { type(String(c)) }
                        }
                        Spacer(minLength: 0)
                        Key(symbol: "delete.left", special: true, width: unit * 1.35, repeats: true) {
                            model.host?.deleteBackward()
                        }
                    }
                }
                HStack(spacing: gap) {
                    Key(label: layer == .letters ? "123" : "ABC", special: true, width: unit * 1.35) {
                        layer = layer == .letters ? .numbers : .letters
                        if layer == .letters { autoShift() }
                    }
                    if model.showGlobe {
                        Key(symbol: "globe", special: true, width: unit * 1.1) { model.host?.nextKeyboard() }
                    }
                    Key(label: "space", width: nil) { space() }
                    Key(label: "return", special: true, width: unit * 2.1) {
                        model.host?.insert("\n")
                        autoShift()
                    }
                }
            }
        }
        .frame(height: keyH * 4 + 11 * 3)
    }

    private func letterRow(_ s: String, unit: CGFloat) -> some View {
        HStack(spacing: gap) {
            ForEach(Array(s), id: \.self) { c in letter(String(c), unit: unit) }
        }
        .frame(maxWidth: .infinity)
    }

    private func charRow(_ s: String, unit: CGFloat) -> some View {
        HStack(spacing: gap) {
            ForEach(Array(s), id: \.self) { c in Key(label: String(c), width: unit) { type(String(c)) } }
        }
        .frame(maxWidth: .infinity)
    }

    private func letter(_ c: String, unit: CGFloat) -> some View {
        let shown = shift || capsLock ? c.uppercased() : c
        return Key(label: shown, width: unit, popup: true) {
            type(shown)
            if shift && !capsLock { shift = false }
        }
    }

    private func type(_ s: String) {
        model.host?.insert(s)
        if [".", "?", "!"].contains(s) { autoShift() }
    }

    private func space() {
        let before = model.host?.contextBefore ?? ""
        // double space: ". " like the system keyboard
        if Date().timeIntervalSince(lastSpace) < 0.45, before.hasSuffix(" "),
            let prev = before.dropLast().last, prev.isLetter || prev.isNumber
        {
            model.host?.deleteBackward()
            model.host?.insert(". ")
            lastSpace = .distantPast
            autoShift()
            return
        }
        model.host?.insert(" ")
        lastSpace = Date()
        autoShift()
    }

    private func tapShift() {
        if Date().timeIntervalSince(lastShiftTap) < 0.3 {
            capsLock = true
            shift = true
        } else if capsLock {
            capsLock = false
            shift = false
        } else {
            shift.toggle()
        }
        lastShiftTap = Date()
    }

    /// Capitalise at the start of the field and after a sentence end, as the system keyboard does.
    private func autoShift() {
        guard !capsLock, layer == .letters else { return }
        let before = model.host?.contextBefore ?? ""
        let t = before.trimmingCharacters(in: .whitespaces)
        shift = before.isEmpty || before.hasSuffix("\n") || (before.hasSuffix(" ") && [".", "?", "!"].contains(t.last.map(String.init) ?? ""))
    }

    private func tick() { UIImpactFeedbackGenerator(style: .light).impactOccurred(intensity: 0.6) }
}

/// One key: system keyboard proportions, a raised look, a callout for letters, auto-repeat for delete.
private struct Key: View {
    var label: String? = nil
    var symbol: String? = nil
    var special = false
    var highlighted = false
    var width: CGFloat?
    var popup = false
    var repeats = false
    let action: () -> Void

    init(label: String, special: Bool = false, width: CGFloat?, popup: Bool = false, action: @escaping () -> Void) {
        self.label = label
        self.special = special
        self.width = width
        self.popup = popup
        self.action = action
    }

    init(symbol: String, special: Bool = false, highlighted: Bool = false, width: CGFloat?, repeats: Bool = false,
         action: @escaping () -> Void)
    {
        self.symbol = symbol
        self.special = special
        self.highlighted = highlighted
        self.width = width
        self.repeats = repeats
        self.action = action
    }

    @State private var pressed = false
    @State private var timer: Timer?

    var body: some View {
        let fill = highlighted ? Ember.ink : (special ? Color(hex: 0x2A231E) : Color(hex: 0x3A312A))
        ZStack {
            RoundedRectangle(cornerRadius: 6)
                .fill(pressed && !popup ? Color(hex: 0x4A3F36) : fill)
                .shadow(color: .black.opacity(0.45), radius: 0, x: 0, y: 1)
            Group {
                if let symbol {
                    Image(systemName: symbol).font(.system(size: 17, weight: .regular))
                } else if let label {
                    Text(label).font(.system(size: label.count > 1 ? 15 : 23, weight: .regular))
                }
            }
            .foregroundStyle(highlighted ? Color(hex: 0x16120F) : Ember.ink)
        }
        .frame(width: width)
        .frame(maxWidth: width == nil ? .infinity : nil)
        .frame(height: 43)
        .overlay(alignment: .top) {
            if popup && pressed, let label {
                Text(label)
                    .font(.system(size: 32))
                    .foregroundStyle(Ember.ink)
                    .frame(width: (width ?? 34) + 16, height: 52)
                    .background(RoundedRectangle(cornerRadius: 8).fill(Color(hex: 0x4A3F36)))
                    .offset(y: -56)
                    .allowsHitTesting(false)
            }
        }
        .contentShape(Rectangle())
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in
                    guard !pressed else { return }
                    pressed = true
                    UIDevice.current.playInputClick()
                    if repeats {
                        action()
                        timer = Timer.scheduledTimer(withTimeInterval: 0.45, repeats: false) { _ in
                            timer = Timer.scheduledTimer(withTimeInterval: 0.08, repeats: true) { _ in action() }
                        }
                    }
                }
                .onEnded { _ in
                    pressed = false
                    timer?.invalidate()
                    timer = nil
                    if !repeats { action() }
                }
        )
    }
}

/// The keyboard's top row: the side button, then the talk bar filling the rest of the width at a fixed height. The
/// side button is present in every state (only its icon changes), so the bar's frame never depends on the phase.
struct TalkBarStrip<Bar: View>: View {
    static var barHeight: CGFloat { 42 }
    static var sideSize: CGFloat { 44 }

    let control: TalkBarSideControl
    let onSide: () -> Void
    @ViewBuilder let bar: () -> Bar

    var body: some View {
        HStack(spacing: 8) {
            Button(action: onSide) {
                Image(systemName: control.symbol).font(.system(size: control == .stop ? 14 : 16, weight: .semibold))
                    .frame(width: 40, height: 40)
                    .background(Circle().fill(Ember.panelRaised))
                    .foregroundStyle(Ember.soft)
                    .frame(width: Self.sideSize, height: Self.sideSize)  // 44 pt hit target around the 40 pt circle
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel(control.accessibilityLabel)
            bar()
                .frame(maxWidth: .infinity)
                .frame(height: Self.barHeight)
        }
    }
}

/// The brightness the bar shows for the meter's recent samples (about 15 a second). The raw mean of the last three
/// stepped the colour 15 times a second, which reads as flicker; this follows a rise quickly and a fall slowly, so
/// the colour glides with the voice. Colour only: nothing here touches layout.
enum VoiceLevel {
    static let attack = 0.55
    static let release = 0.12

    static func smooth(_ levels: [CGFloat]) -> Double {
        var v = 0.0
        for l in levels {
            let x = Double(max(0, min(1, l)))
            v += (x - v) * (x > v ? attack : release)
        }
        return v
    }
}

/// The button beside the talk bar: an X that cancels while dictating, a settings circle that opens Phonon otherwise.
/// Dictating means the phase says so, the light is lit (listening or gathering, including the moment between the
/// press and the app answering), or a press is waiting for the app.
enum TalkBarSideControl: Equatable {
    case stop, settings

    init(phase: KeyboardPhase, beat: BarStage = .rest, pending: Bool = false) {
        let dictating = phase == .listening || phase == .working || beat == .listening || beat == .processing
            || pending
        self = dictating ? .stop : .settings
    }

    var symbol: String { self == .stop ? "xmark" : "gearshape.fill" }
    var accessibilityLabel: String { self == .stop ? "Cancel dictation" : "Open Phonon" }

    func perform(on host: KeyboardHost) {
        switch self {
        case .stop: host.cancelDictation()
        case .settings: host.openApp()
        }
    }
}

/// The talk bar's fill. It takes exactly the frame it is given in every state and audio level; voice and state show
/// only in brightness. Idle: a plain dark capsule. Then the stages of BarStages.swift: ignite on the press,
/// listening (brightness follows the smoothed level), steady warm ember while the text is written, and a crisp
/// fade back to rest when it is inserted. No scale, size or layout ever depends on the stage or the level.
struct TalkBarFill: View {
    enum Mode: Equatable {
        case idle, noAccess, listening, working

        init(phase: KeyboardPhase) {
            switch phase {
            case .listening: self = .listening
            case .working: self = .working
            case .needsAccess: self = .noAccess
            case .ready, .needsSession: self = .idle
            }
        }
    }

    let state: Mode
    let level: Double
    /// False with Reduce Motion, Low Power Mode or the keyboard off screen: no moving widths (a steady medium core).
    var animate = true
    /// False with Reduce Motion: listening is one static brightness, not driven by the level.
    var followLevel = true
    /// The stage the real events put the dictation in (BarStages.swift). While it is `.rest` the bar follows `state`:
    /// listening is the lit bar, working the held core.
    var beat = BarBeat.rest
    /// Renders one instant (offscreen tests); nil follows the clock.
    var frozenAt: Date?

    @State private var settledBeat: BarBeat?

    private var reduced: Bool { !animate || !followLevel }

    /// The beat that draws: the model's, or the one `state` implies when no event set one (previews, tests).
    private var drawn: BarBeat {
        if beat.stage != .rest { return beat }
        switch state {
        case .listening: return BarBeat(stage: .listening, since: .distantPast)
        case .working: return BarBeat(stage: .processing, since: .distantPast)
        case .idle, .noAccess: return .rest
        }
    }

    private var active: Bool { drawn.stage != .rest }

    var body: some View {
        let drawn = drawn
        // The clock runs only while a stage is still moving: the 0.10 s ignite and the 0.10 s fade.
        // Steady processing and the resting bar need no frames of their own.
        let moving = drawn.settleSeconds(reduceMotion: reduced).map { Date().timeIntervalSince(drawn.since) < $0 } ?? false
        let running = frozenAt == nil && moving && settledBeat != drawn
        return ZStack {
            RestingBar()
                .opacity(active ? 0 : 1)
            if active {
                TimelineView(.animation(minimumInterval: 1.0 / 30.0, paused: !running)) { context in
                    // a stopped clock hands over a stale date; the settled look is the stage's end, so ask the time
                    Self.light(look(at: frozenAt ?? (running ? context.date : Date())))
                }
                .transition(.opacity)
            }
        }
        .task(id: drawn) {
            guard let settle = drawn.settleSeconds(reduceMotion: reduced) else { return }
            let left = settle - Date().timeIntervalSince(drawn.since)
            if left > 0 { try? await Task.sleep(nanoseconds: UInt64((left + 0.05) * 1e9)) }
            // a newer beat replaced this one mid-sleep: it is not settled, and its own task decides
            guard !Task.isCancelled else { return }
            settledBeat = drawn
        }
        .clipShape(Capsule(style: .continuous))
        .overlay(Capsule(style: .continuous).strokeBorder(Color.white.opacity(active ? 0.06 : 0), lineWidth: 0.5))
        .animation(followLevel ? .easeOut(duration: 0.18) : nil, value: active)
    }

    /// The light at `date`.
    func look(at date: Date) -> BarLook {
        drawn.look(at: date, level: followLevel ? level : 0.55, reduceMotion: reduced)
    }

    /// Black at both ends, one warm light at the centre (the Mac capsule's colours): the look's stops, over black.
    /// Colours only: both layers fill the given rectangle.
    static func light(_ look: BarLook) -> some View {
        let stops = look.stops.map {
            Gradient.Stop(color: Color(red: $0.r, green: $0.g, blue: $0.b).opacity($0.a), location: $0.location)
        }
        return ZStack {
            Rectangle().fill(.black)
            LinearGradient(stops: stops, startPoint: .leading, endPoint: .trailing)
        }
    }
}

/// The talk bar at rest: a plain dark capsule with a hairline edge. One flat colour, nothing soft or moving, so it
/// draws once and never redraws.
struct RestingBar: View {
    var body: some View {
        Capsule(style: .continuous)
            .fill(Color(hex: 0x1D1D1F))
            .overlay(Capsule(style: .continuous).strokeBorder(Color.white.opacity(0.10), lineWidth: 0.5))
    }
}

/// Live input level while listening: ember bars, newest on the right.
struct Waveform: View {
    let levels: [CGFloat]
    let active: Bool

    var body: some View {
        HStack(alignment: .center, spacing: 3) {
            ForEach(levels.indices, id: \.self) { i in
                Capsule()
                    .fill(active ? Color.white.opacity(0.92) : Ember.dim)  // drawn inside the orange talk bar
                    .frame(width: 3, height: max(3, levels[i] * 22))
            }
        }
        .animation(.easeOut(duration: 0.08), value: levels)
    }
}
