import PhononCore
import SwiftUI

struct KeyboardView: View {
    @ObservedObject var model: KeyboardModel

    private var recording: Bool { model.phase == .recording }
    private var busy: Bool { model.phase == .transcribing || model.phase == .refining }

    var body: some View {
        VStack(spacing: 10) {
            HStack {
                HStack(spacing: 6) {
                    if recording, let started = model.startedAt {
                        Text(started, style: .timer).monospacedDigit()
                    }
                    Text(model.status)
                }
                .font(.footnote.weight(.medium))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                Spacer()
                if model.lastText != nil && !recording {
                    Button(action: model.insertLast) {
                        Label("Insert last", systemImage: "arrow.uturn.backward")
                            .font(.footnote.weight(.medium))
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(Palette.accent)
                    .accessibilityHint("Inserts your most recent dictation again")
                }
            }
            .padding(.horizontal, 14)
            .frame(height: 22)

            if !model.fullAccess {
                FullAccessNotice()
            } else {
                HStack(spacing: 28) {
                    if recording {
                        RoundKey(systemImage: "xmark", label: "Cancel dictation", size: 48, action: model.cancelTapped)
                    } else {
                        Color.clear.frame(width: 48, height: 48)
                    }
                    Button(action: model.micTapped) {
                        ZStack {
                            Circle()
                                .fill(recording ? Palette.recording : Palette.accent)
                                .frame(width: 84, height: 84)
                                .scaleEffect(recording ? 1 + CGFloat(model.level) * 0.1 : 1)
                                .animation(.spring(response: 0.2, dampingFraction: 1), value: model.level)
                            if busy || model.waiting && !recording {
                                ProgressView().tint(.white)
                            } else {
                                Image(systemName: recording ? "stop.fill" : "mic.fill")
                                    .font(.system(size: 30, weight: .semibold))
                                    .foregroundStyle(.white)
                            }
                        }
                    }
                    .buttonStyle(PressScale())
                    // Never disabled: a busy state can be left behind by a closed Phonon, and the
                    // tap is what finds that out (KeyboardModel.startIfAppGone).
                    .accessibilityLabel(recording ? "Stop dictation" : "Start dictation")
                    Color.clear.frame(width: 48, height: 48)
                }
                .frame(maxHeight: .infinity)
            }

            HStack(spacing: 6) {
                if model.showsGlobe {
                    KeyButton(systemImage: "globe", label: "Next keyboard", width: 44) { model.controller?.nextKeyboard() }
                }
                KeyButton(title: "space", label: "Space") { model.controller?.insertSpace() }
                RepeatingDelete { model.controller?.deleteBackward() }
                KeyButton(title: model.controller?.returnKeyTitle ?? "return", label: "Return", width: 88) {
                    model.controller?.insertReturn()
                }
            }
            .padding(.horizontal, 4)
        }
        .padding(.top, 8)
        .padding(.bottom, 4)
    }
}

struct FullAccessNotice: View {
    var body: some View {
        VStack(spacing: 6) {
            Text("Turn on Allow Full Access")
                .font(.subheadline.weight(.semibold))
            Text("Settings > General > Keyboard > Keyboards > Phonon. It lets the Phonon app hand finished text to this keyboard through a folder on this phone. Nothing goes online.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(.horizontal, 20)
        .frame(maxHeight: .infinity)
    }
}

struct KeyButton: View {
    var title: String? = nil
    var systemImage: String? = nil
    let label: String
    var width: CGFloat? = nil
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Group {
                if let systemImage { Image(systemName: systemImage) } else { Text(title ?? "") }
            }
            .font(.system(size: 16))
            .frame(maxWidth: width == nil ? .infinity : width, minHeight: 44)
            .frame(width: width)
            .background(RoundedRectangle(cornerRadius: 8).fill(Color(.systemBackground).opacity(0.9)))
            .shadow(color: .black.opacity(0.25), radius: 0, x: 0, y: 1)
            .foregroundStyle(.primary)
        }
        .buttonStyle(PressScale())
        .accessibilityLabel(label)
    }
}

/// Delete that repeats while held, as the system keyboard does.
struct RepeatingDelete: View {
    let action: () -> Void
    @State private var timer: Timer?

    var body: some View {
        Image(systemName: "delete.left")
            .font(.system(size: 16))
            .frame(width: 52, height: 44)
            .background(RoundedRectangle(cornerRadius: 8).fill(Color(.systemBackground).opacity(0.9)))
            .shadow(color: .black.opacity(0.25), radius: 0, x: 0, y: 1)
            .gesture(DragGesture(minimumDistance: 0)
                .onChanged { _ in
                    guard timer == nil else { return }
                    action()
                    timer = Timer.scheduledTimer(withTimeInterval: 0.45, repeats: false) { _ in
                        timer = Timer.scheduledTimer(withTimeInterval: 0.08, repeats: true) { _ in action() }
                    }
                }
                .onEnded { _ in
                    timer?.invalidate()
                    timer = nil
                })
            .accessibilityLabel("Delete")
            .accessibilityAddTraits(.isButton)
            .accessibilityAction { action() }
    }
}

struct RoundKey: View {
    let systemImage: String
    let label: String
    let size: CGFloat
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 18, weight: .semibold))
                .frame(width: size, height: size)
                .background(Circle().fill(Color(.systemBackground).opacity(0.9)))
                .foregroundStyle(.primary)
        }
        .buttonStyle(PressScale())
        .accessibilityLabel(label)
    }
}

struct PressScale: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.94 : 1)
            .animation(.spring(response: 0.18, dampingFraction: 1), value: configuration.isPressed)
    }
}

enum Palette {
    static let accent = Color(UIColor { $0.userInterfaceStyle == .dark
        ? UIColor(red: 0.34, green: 0.71, blue: 0.91, alpha: 1)
        : UIColor(red: 0.0, green: 0.45, blue: 0.70, alpha: 1) })
    static let recording = Color(UIColor { $0.userInterfaceStyle == .dark
        ? UIColor(red: 0.95, green: 0.45, blue: 0.20, alpha: 1)
        : UIColor(red: 0.84, green: 0.37, blue: 0.0, alpha: 1) })
}
