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
    @Published var status = ""
    @Published var levels: [CGFloat] = Array(repeating: 0, count: 28)
    @Published var mode: WritingMode = .minimal
    @Published var canUndo = false
    @Published var showGlobe = true
    @Published var latched = false
    /// Bumped by the host when the text around the cursor changes (re-evaluates auto-capitalisation).
    @Published var contextVersion = 0
    /// False while the keyboard is off screen; the idle fluid stops redrawing.
    @Published var visible = true
    /// Low Power Mode: the idle fluid holds still.
    @Published var lowPower = ProcessInfo.processInfo.isLowPowerModeEnabled
    weak var host: KeyboardHost?
    private var powerObserver: NSObjectProtocol?

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
    private var toolbar: some View {
        HStack(spacing: 8) {
            if model.phase == .listening || model.phase == .working {
                Button {
                    tick()
                    press = .none
                    model.host?.cancelDictation()
                } label: {
                    Image(systemName: "xmark").font(.system(size: 14, weight: .semibold))
                        .frame(width: 40, height: 40)
                        .background(Circle().fill(Ember.panelRaised))
                        .foregroundStyle(Ember.soft)
                }
                .accessibilityLabel("Cancel dictation")
            }
            talkBar
        }
    }

    private enum Press { case none, starting, stopping }
    @State private var press: Press = .none

    private var off: Bool { model.phase == .needsSession || model.phase == .needsAccess }

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// At rest the bar is a grey, softly blurred fluid that drifts inside the capsule (no label). Pressed, it becomes
    /// the Mac's dictation capsule (bar/Sources/PhononBar.swift PillView) at 2.6x: the black capsule whose ember
    /// centre lights and swells with your voice, then breathes while the text is written. Only the off states (no
    /// Full Access, app not running) keep a short label over the fluid, since they need an instruction.
    private var talkBar: some View {
        let active = model.phase == .listening || model.phase == .working
        let pillMode: MacPill.Mode = model.phase == .listening ? .listening : .working
        return ZStack {
            if active {
                MacPill(mode: pillMode, level: voiceLevel, scale: 2.6)
                    .transition(.opacity.combined(with: .scale(scale: 0.6)))
            } else {
                IdleFluid(animate: model.visible && !model.lowPower && !reduceMotion, dim: off)
                    .overlay {
                        if off {
                            Text(barLabel)
                                .font(.system(size: 13, weight: .medium))
                                .foregroundStyle(Ember.soft)
                                .lineLimit(1)
                                .minimumScaleFactor(0.8)
                                .padding(.horizontal, 16)
                        }
                    }
                    .transition(.opacity)
            }
        }
        .animation(.easeOut(duration: 0.18), value: active)
        .frame(maxWidth: .infinity)
        .frame(height: 42)
        .contentShape(Capsule())
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in
                    guard micPressedAt == nil else { return }
                    micPressedAt = Date()
                    if off {
                        model.host?.openApp()
                    } else if model.phase == .listening {
                        press = .stopping  // locked on: this tap finishes it
                    } else if model.phase == .ready {
                        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                        press = .starting
                        model.latched = false
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

    /// The live input level the capsule lights with: the mean of the newest three meter samples.
    private var voiceLevel: Double {
        let tail = model.levels.suffix(3)
        return tail.isEmpty ? 0 : Double(tail.reduce(0, +)) / Double(tail.count)
    }

    private var hint: String {
        switch model.phase {
        case .listening: return model.latched ? "Tap to finish" : "Let go to finish"
        case .working: return "Writing it down"
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

/// The Mac's dictation capsule (bar/Sources/PhononBar.swift PillView), same geometry and colours, drawn at `scale`.
/// Idle: a small black bar. Listening: wider and taller with the live level, ember centre lit by the voice, warm
/// glow. Working: smaller, breathing on a 1.6 s cycle.
struct MacPill: View {
    enum Mode { case idle, listening, working }
    let mode: Mode
    let level: Double
    var scale: CGFloat = 1

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30.0, paused: mode != .working)) { context in
            let energy = energy(at: context.date)
            let size = shellSize(energy: energy)
            Capsule(style: .continuous)
                .fill(
                    LinearGradient(
                        stops: [
                            .init(color: .black, location: 0),
                            .init(color: Color(red: 0.42, green: 0.12, blue: 0.035).opacity(warmth(energy)),
                                  location: 0.5),
                            .init(color: .black, location: 1),
                        ],
                        startPoint: .leading, endPoint: .trailing)
                )
                .overlay(Capsule(style: .continuous).strokeBorder(Color.white.opacity(mode == .idle ? 0.10 : 0),
                                                                  lineWidth: 0.5))
                .shadow(color: Color(red: 0.55, green: 0.16, blue: 0.04).opacity(glow(energy)), radius: 4 * scale, y: scale)
                .frame(width: size.width * scale, height: size.height * scale)
                .animation(.spring(response: 0.22, dampingFraction: 0.86), value: size)
        }
    }

    private func energy(at date: Date) -> Double {
        switch mode {
        case .idle: return 0
        case .listening: return min(1, max(0.08, level))
        case .working:
            let breath = (sin(date.timeIntervalSinceReferenceDate * .pi * 2 / 1.6) + 1) / 2
            return 0.18 + breath * 0.18
        }
    }

    private func warmth(_ energy: Double) -> Double { mode == .idle ? 0 : 0.50 + energy * 0.38 }
    private func glow(_ energy: Double) -> Double { mode == .idle ? 0 : 0.22 + energy * 0.12 }

    private func shellSize(energy: Double) -> CGSize {
        switch mode {
        case .idle: return CGSize(width: 40, height: 8)
        case .listening: return CGSize(width: 66 + 14 * energy, height: 13 + 3 * energy)
        case .working: return CGSize(width: 38 + 10 * energy, height: 10 + 3 * energy)
        }
    }
}

/// The talk bar at rest: grey light drifting slowly inside a dark capsule, like smoke under frosted glass. Each blob
/// is a radial gradient (soft by construction, so no blur pass or offscreen texture), redrawn at most 20 times a
/// second and only while `animate` (keyboard on screen, no Reduce Motion, no Low Power Mode); otherwise it holds
/// one still frame.
struct IdleFluid: View {
    let animate: Bool
    var dim = false

    /// x centre (fraction of width), drift amplitudes, angular speeds (rad/s, periods 7 to 15 s), phase, radius
    /// (fraction of height), brightness.
    private struct Blob { let x, ax, ay, sx, sy, phase, r, white: Double }
    private static let blobs: [Blob] = [
        Blob(x: 0.22, ax: 0.14, ay: 0.30, sx: 0.47, sy: 0.71, phase: 0.0, r: 1.25, white: 0.20),
        Blob(x: 0.55, ax: 0.18, ay: 0.35, sx: 0.41, sy: 0.53, phase: 2.1, r: 1.45, white: 0.15),
        Blob(x: 0.82, ax: 0.12, ay: 0.28, sx: 0.67, sy: 0.43, phase: 4.2, r: 1.10, white: 0.18),
        Blob(x: 0.40, ax: 0.22, ay: 0.40, sx: 0.83, sy: 0.61, phase: 1.3, r: 0.80, white: 0.10),
    ]

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 20.0, paused: !animate)) { context in
            let t = animate ? context.date.timeIntervalSinceReferenceDate.truncatingRemainder(dividingBy: 3600) : 0
            Canvas { g, size in
                g.fill(Path(CGRect(origin: .zero, size: size)), with: .color(Color(hex: 0x1D1D1F)))
                let scale = dim ? 0.6 : 1
                for b in Self.blobs {
                    let c = CGPoint(x: size.width * (b.x + b.ax * sin(t * b.sx + b.phase)),
                                    y: size.height * (0.5 + b.ay * cos(t * b.sy + b.phase)))
                    let r = size.height * b.r
                    // stretched sideways so the blobs read as a flowing band, not spots
                    var blob = g
                    blob.translateBy(x: c.x, y: c.y)
                    blob.scaleBy(x: 2.2, y: 1)
                    blob.fill(Path(ellipseIn: CGRect(x: -r, y: -r, width: 2 * r, height: 2 * r)),
                              with: .radialGradient(Gradient(colors: [Color.white.opacity(b.white * scale),
                                                                      Color.white.opacity(0)]),
                                                    center: .zero, startRadius: 0, endRadius: r))
                }
            }
        }
        .clipShape(Capsule(style: .continuous))
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
