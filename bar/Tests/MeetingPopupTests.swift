import AppKit
import SwiftUI
import XCTest

@testable import PhononBar

/// The meeting popup and the capsule's recording state: the flow from the
/// popup to recording to the stop button, the icon lookup, and offscreen
/// renders (no window is ordered on screen). Set PHONON_RENDER_DIR to keep
/// the PNGs.
@MainActor
final class MeetingPopupTests: XCTestCase {
    func testPopupRecordHoverStopFlow() {
        var flow = MeetingFlow()
        let pill = PillState()
        pill.mode = .idle
        XCTAssertFalse(flow.capsuleAcceptsMouse, "clicks pass through when not recording")

        // Hovering the capsule does nothing until a meeting records.
        flow.hover(true)
        XCTAssertFalse(flow.showsStopButton)

        flow.ask(app: "Zoom", bundleID: "us.zoom.xos")
        XCTAssertEqual(flow.popup, .asking(app: "Zoom", bundleID: "us.zoom.xos"))
        let call = flow.record()
        XCTAssertEqual(call?.app, "Zoom")
        XCTAssertEqual(call?.bundleID, "us.zoom.xos")
        XCTAssertEqual(flow.popup, .hidden)
        XCTAssertNil(flow.record(), "one click starts one recording")

        flow.started(app: "Zoom")
        pill.apply(flow)
        XCTAssertTrue(pill.meetingRecording)
        XCTAssertFalse(pill.stopButtonShown)
        XCTAssertTrue(flow.capsuleAcceptsMouse)

        // Detection keeps running while recording; it never asks again.
        flow.ask(app: "Zoom", bundleID: "us.zoom.xos")
        XCTAssertEqual(flow.popup, .hidden)

        flow.hover(true)
        pill.apply(flow)
        XCTAssertTrue(pill.stopButtonShown)
        flow.hover(false)
        pill.apply(flow)
        XCTAssertFalse(pill.stopButtonShown)
        flow.hover(true)

        flow.stopped(byUser: true)
        pill.apply(flow)
        XCTAssertEqual(flow.popup, .saved)
        XCTAssertFalse(pill.meetingRecording)
        XCTAssertFalse(pill.stopButtonShown)
        XCTAssertFalse(flow.capsuleAcceptsMouse)
        flow.expire()
        XCTAssertEqual(flow.popup, .hidden)
    }

    func testNotNowAndCallEndedShowNoConfirmation() {
        var flow = MeetingFlow()
        XCTAssertNil(flow.notNow(), "nothing to decline")
        flow.ask(app: "meet.google.com", bundleID: "com.google.Chrome")
        XCTAssertEqual(flow.notNow(), "meet.google.com")
        XCTAssertEqual(flow.popup, .hidden)

        // Auto-start skips the popup; a call that ends on its own is not "saved" by the user.
        flow.started(app: "Slack")
        XCTAssertTrue(flow.isRecording)
        flow.stopped(byUser: false)
        XCTAssertEqual(flow.popup, .hidden)
        XCTAssertFalse(flow.isRecording)
    }

    func testControllerStartsHidden() {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "phonon-meeting-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let controller = MeetingController(store: NativeAppStore(supportDirectory: directory))
        XCTAssertEqual(controller.flow, MeetingFlow())
        // No question up and nothing recording: these are no-ops.
        controller.notNow()
        controller.record()
        controller.stopFromUser()
        controller.hover(true)
        XCTAssertEqual(controller.flow, MeetingFlow())
        XCTAssertFalse(controller.isRecording)
    }

    func testIconSource() {
        let installed: [String: URL] = [
            "us.zoom.xos": URL(fileURLWithPath: "/Applications/zoom.us.app"),
            "com.google.Chrome": URL(fileURLWithPath: "/Applications/Google Chrome.app"),
            "com.apple.FaceTime": URL(fileURLWithPath: "/System/Applications/FaceTime.app"),
        ]
        let meetShim = URL(fileURLWithPath: "/Users/u/Applications/Chrome Apps.localized/Google Meet.app")
        let lookup = { (id: String) in installed[id] }
        let webApps = { (host: String) in host == "meet.google.com" ? meetShim : nil }
        let noWebApps = { (_: String) -> URL? in nil }

        // Native call apps: their own bundle.
        XCTAssertEqual(
            MeetingAppIcon.source(app: "Zoom", bundleID: "us.zoom.xos", appURL: lookup),
            .app(installed["us.zoom.xos"]!))
        XCTAssertEqual(
            MeetingAppIcon.source(app: "FaceTime", bundleID: "com.apple.FaceTime", appURL: lookup),
            .app(installed["com.apple.FaceTime"]!))
        XCTAssertEqual(
            MeetingAppIcon.source(app: "Microsoft Teams", bundleID: "com.microsoft.teams2",
                appURL: lookup), .symbol, "not installed")
        // A site: its native app, then a web app shim, then the browser.
        XCTAssertEqual(
            MeetingAppIcon.source(app: "us02web.zoom.us", bundleID: "com.google.Chrome",
                appURL: lookup, webApp: noWebApps),
            .app(installed["us.zoom.xos"]!))
        XCTAssertEqual(
            MeetingAppIcon.source(app: "meet.google.com", bundleID: "com.google.Chrome",
                appURL: lookup, webApp: webApps), .app(meetShim))
        XCTAssertEqual(
            MeetingAppIcon.source(app: "meet.google.com", bundleID: "com.google.Chrome",
                appURL: lookup, webApp: noWebApps), .app(installed["com.google.Chrome"]!))
        XCTAssertEqual(
            MeetingAppIcon.source(app: "whereby.com", bundleID: "org.unknown.browser",
                appURL: lookup, webApp: noWebApps), .symbol)
    }

    func testWebAppShimLookup() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(
            "phonon-webapps-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let contents = folder.appendingPathComponent("Google Meet.app/Contents", isDirectory: true)
        try FileManager.default.createDirectory(at: contents, withIntermediateDirectories: true)
        let info: NSDictionary = ["CrAppModeShortcutURL": "https://meet.google.com/landing"]
        XCTAssertTrue(info.write(to: contents.appendingPathComponent("Info.plist"), atomically: true))
        XCTAssertEqual(
            MeetingAppIcon.webAppURL(host: "meet.google.com", folders: [folder])?.lastPathComponent,
            "Google Meet.app")
        XCTAssertNil(MeetingAppIcon.webAppURL(host: "zoom.us", folders: [folder]))
    }

    func testRecordLightBlink() {
        XCTAssertEqual(RecordLight.opacity(at: 0), 1, accuracy: 1e-9)
        XCTAssertEqual(RecordLight.opacity(at: 0.6), 0.25, accuracy: 1e-9)
        XCTAssertEqual(RecordLight.opacity(at: 1.2), 1, accuracy: 1e-6)
        XCTAssertEqual(RecordLight.opacity(at: 0.3), 0.625, accuracy: 1e-9, "ease-in-out midpoint")
        XCTAssertEqual(RecordLight.glowRadius(opacity: 1), 3, accuracy: 1e-9)
        XCTAssertEqual(RecordLight.glowRadius(opacity: 0.25), 1, accuracy: 1e-9)
    }

    func testCapsuleStaysPutInTheLargerPanel() {
        let screen = NSRect(x: 0, y: 0, width: 1728, height: 1117)
        let panel = PillPanelLayout.frame(screen: screen)
        let resting = PanelGeometry.restingFrame(screen: screen, size: PillPanelLayout.capsule)
        XCTAssertEqual(panel.minX + PillPanelLayout.insets.leading, resting.minX)
        XCTAssertEqual(panel.minY + PillPanelLayout.insets.bottom, resting.minY)
        XCTAssertEqual(panel.size, PillPanelLayout.size)
        // The 22 pt stop button, centered on the recording capsule (12 pt tall), fits.
        XCTAssertGreaterThanOrEqual(
            PillPanelLayout.insets.bottom + 6 - RecordLight.stopDiameter / 2, 0)
        // The popup's card sits just above the capsule, clear of it.
        let popup = MeetingPopupPanel.frame(screen: screen)
        XCTAssertEqual(popup.minY + MeetingPopupPanel.bottomMargin, resting.minY + 30)
        XCTAssertEqual(popup.midX, resting.midX)
    }

    // MARK: Renders

    private func write(_ view: some View, name: String, size: CGSize) throws {
        let framed = view
            .frame(width: size.width, height: size.height)
            .background(
                LinearGradient(
                    colors: [
                        Color(red: 0x24 / 255, green: 0x30 / 255, blue: 0x3e / 255),
                        Color(red: 0x16 / 255, green: 0x1b / 255, blue: 0x24 / 255),
                    ], startPoint: .topLeading, endPoint: .bottomTrailing))
            .environment(\.colorScheme, .dark)
        // Hosted the way the panels host it; the window is never ordered on screen.
        let host = NSHostingView(rootView: framed)
        host.appearance = NSAppearance(named: .darkAqua)
        host.frame = CGRect(origin: .zero, size: size)
        let window = NSWindow(
            contentRect: host.frame, styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.appearance = NSAppearance(named: .darkAqua)
        window.contentView = host
        host.layoutSubtreeIfNeeded()
        RunLoop.main.run(until: Date().addingTimeInterval(0.05))
        host.layoutSubtreeIfNeeded()
        let rep = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: rep)
        XCTAssertGreaterThanOrEqual(rep.pixelsWide, Int(size.width))
        if let output = ProcessInfo.processInfo.environment["PHONON_RENDER_DIR"] {
            let data = try XCTUnwrap(rep.representation(using: .png, properties: [:]))
            try data.write(to: URL(fileURLWithPath: output).appendingPathComponent("\(name).png"))
        }
    }

    /// The popup and capsule as on screen: both panels share the bottom edge.
    private func desk(pill: PillState, popup: MeetingPopupModel?, at seconds: TimeInterval)
        -> some View
    {
        ZStack(alignment: .bottom) {
            PillView(state: pill, fixedTime: Date(timeIntervalSinceReferenceDate: seconds))
                .padding(PillPanelLayout.insets)
                .padding(.bottom, PanelGeometry.bottomInset - PillPanelLayout.insets.bottom)
            if let popup {
                MeetingPopupView(model: popup)
                    .padding(
                        .bottom, MeetingPopupPanel.cardBottom - MeetingPopupPanel.bottomMargin)
            }
        }
        .frame(width: 360, height: 140, alignment: .bottom)
    }

    func testPopupRendersForThreeApps() throws {
        let apps: [(String, String, String)] = [
            ("Zoom", "us.zoom.xos", "popup_zoom"),
            ("Slack", "com.tinyspeck.slackmacgap", "popup_slack_huddle"),
            ("FaceTime", "com.apple.FaceTime", "popup_facetime"),
            ("Discord", "com.hnc.Discord", "popup_discord"),
            ("meet.google.com", "com.google.Chrome", "popup_google_meet_in_chrome"),
        ]
        for (app, bundleID, name) in apps {
            let model = MeetingPopupModel()
            model.content = .ask(MeetingAppIcon.image(app: app, bundleID: bundleID))
            model.visible = true
            let pill = PillState()
            pill.mode = .idle
            try write(desk(pill: pill, popup: model, at: 0), name: name, size: CGSize(width: 360, height: 140))
        }
        // Mid-slide: hidden state is 8 pt lower and transparent.
        let hidden = MeetingPopupModel()
        hidden.content = .ask(MeetingAppIcon.image(app: "Zoom", bundleID: "us.zoom.xos"))
        hidden.visible = false
        let pill = PillState()
        pill.mode = .idle
        try write(desk(pill: pill, popup: hidden, at: 0), name: "popup_before_slide_in", size: CGSize(width: 360, height: 140))
    }

    func testRecordingCapsuleRenders() throws {
        let scenes: [(String, PillState.Mode, Double, Bool, TimeInterval)] = [
            ("capsule_recording_light_on", .idle, 0, false, 0),
            ("capsule_recording_light_dim", .idle, 0, false, 0.6),
            ("capsule_recording_hover_stop", .idle, 0, true, 0),
            ("capsule_recording_while_dictating", .listening, 0.6, false, 0),
            ("capsule_recording_while_dictating_hover", .listening, 0.6, true, 0),
            ("capsule_recording_while_working", .working, 0, false, 0.3),
        ]
        for (name, mode, level, hover, at) in scenes {
            let pill = PillState()
            pill.mode = mode
            pill.level = level
            var flow = MeetingFlow()
            flow.started(app: "Zoom")
            flow.hover(hover)
            pill.apply(flow)
            try write(desk(pill: pill, popup: nil, at: at), name: name, size: CGSize(width: 360, height: 140))
        }
        // Not recording: today's idle capsule, unchanged.
        let idle = PillState()
        idle.mode = .idle
        try write(desk(pill: idle, popup: nil, at: 0), name: "capsule_idle_not_recording", size: CGSize(width: 360, height: 140))
    }

    func testSavedConfirmationRenders() throws {
        let model = MeetingPopupModel()
        model.content = .saved
        model.visible = true
        let pill = PillState()
        pill.mode = .idle
        try write(desk(pill: pill, popup: model, at: 0), name: "saved_to_notes", size: CGSize(width: 360, height: 140))
    }
}
