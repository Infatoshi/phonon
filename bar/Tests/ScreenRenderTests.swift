import AppKit
import SwiftUI
import XCTest

@testable import PhononBar

/// Renders every native screen off screen, in dark mode, from a fixture store in
/// a temporary directory. No window is ever ordered on screen. Set
/// PHONON_RENDER_DIR to keep the PNGs for review.
@MainActor
final class ScreenRenderTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "phonon-render-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try writeFixtures()
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func writeFixtures() throws {
        let settings = """
            {"schema_version":2,"streaming":true,"local_history":true,"screen_context":true,
             "privacy_choice_made":true,"edit_learning":true,
             "style_preferences":{"chat":["lowercase_start"]}}
            """
        try Data(settings.utf8).write(to: directory.appendingPathComponent("settings.json"))
        let dictionary = """
            {"schema_version":1,"updated_at_unix_ms":0,"entries":[
             {"phrase":"Blackwell","spoken_forms":["black well"],"source":"phonon-app"},
             {"phrase":"cuBLAS","spoken_forms":["coo blast","cue blas"],"source":"phonon-app"},
             {"phrase":"vee llm","replacement":"vLLM","source":"phonon-app"},
             {"phrase":"Parakeet","source":"phonon-app"}]}
            """
        try Data(dictionary.utf8).write(to: directory.appendingPathComponent("dictionary.json"))
        let now = UInt64(Date().timeIntervalSince1970 * 1_000)
        let recordings: [(String, UInt64, String, String, String)] = [
            ("rec_a", now - 60_000, "run v llm on black well tonight",
             "Run vLLM on Blackwell tonight.", ""),
            ("rec_b", now - 3_600_000, "the coo blast kernel is faster than the reference",
             "The cuBLAS kernel is faster than the reference.",
             #","user_edit":{"pasted_text":"The cuBLAS kernel is faster than the reference.","edited_text":"The cuBLAS kernel is 2x faster than the reference.","outcome":"edited","end_reason":"focus_left"}"#),
            ("rec_c", now - 86_400_000, "send the notes to the team after lunch",
             "Send the notes to the team after lunch.", ""),
        ]
        for (id, created, raw, final, extra) in recordings {
            let folder = directory.appendingPathComponent("Corpus/\(id)", isDirectory: true)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let metadata = """
                {"id":"\(id)","created_at_unix_ms":\(created),"source":"bar",
                 "microphone":"MacBook Pro Microphone","audio_duration_ms":4200,
                 "raw_transcript":"\(raw)","final_transcript":"\(final)",
                 "dictionary_corrections":[{"count":1}],
                 "llm":{"latency_ms":118,"ttft_ms":41,"tokens_per_second":212}\(extra)}
                """
            try Data(metadata.utf8).write(to: folder.appendingPathComponent("metadata.json"))
        }
    }

    @discardableResult
    private func snapshot<V: View>(_ view: V, size: CGSize, name: String) throws -> NSBitmapImageRep {
        let host = NSHostingView(rootView: view.environment(\.colorScheme, .dark))
        host.appearance = NSAppearance(named: .darkAqua)
        host.frame = CGRect(origin: .zero, size: size)
        // Never ordered front: the window only gives AppKit controls a backing.
        let window = NSWindow(
            contentRect: host.frame, styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.appearance = NSAppearance(named: .darkAqua)
        window.contentView = host
        host.layoutSubtreeIfNeeded()
        RunLoop.main.run(until: Date().addingTimeInterval(0.3))
        host.layoutSubtreeIfNeeded()
        let rep = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: rep)
        if let output = ProcessInfo.processInfo.environment["PHONON_RENDER_DIR"] {
            let data = try XCTUnwrap(rep.representation(using: .png, properties: [:]))
            try data.write(to: URL(fileURLWithPath: output).appendingPathComponent("\(name).png"))
        }
        window.contentView = nil
        return rep
    }

    private func store() -> NativeAppStore {
        NativeAppStore(supportDirectory: directory)
    }

    func testMainWindowPagesRender() throws {
        let store = store()
        XCTAssertEqual(store.history.count, 3)
        for page in NativeAppPage.allCases {
            let view = PhononMainView(
                store: store, initialPage: page, onSettingsChanged: {}, onShowModelStatus: {})
            let rep = try snapshot(
                view, size: CGSize(width: 980, height: 680),
                name: "main_\(page.rawValue.lowercased())")
            XCTAssertGreaterThanOrEqual(rep.pixelsWide, 980)
        }
    }

    /// The app menu's Settings… item must land on Settings, not on whichever
    /// page the window showed last.
    func testPageRequestSwitchesTheOpenWindow() throws {
        let store = store()
        let view = PhononMainView(
            store: store, initialPage: .history, onSettingsChanged: {}, onShowModelStatus: {})
        let host = NSHostingView(rootView: view)
        host.frame = CGRect(x: 0, y: 0, width: 980, height: 680)
        let window = NSWindow(
            contentRect: host.frame, styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = host
        host.layoutSubtreeIfNeeded()
        RunLoop.main.run(until: Date().addingTimeInterval(0.2))
        store.pageRequest = .settings
        RunLoop.main.run(until: Date().addingTimeInterval(0.3))
        XCTAssertNil(store.pageRequest, "the window consumed the request")
        if let output = ProcessInfo.processInfo.environment["PHONON_RENDER_DIR"] {
            host.appearance = NSAppearance(named: .darkAqua)
            let rep = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
            host.cacheDisplay(in: host.bounds, to: rep)
            let data = try XCTUnwrap(rep.representation(using: .png, properties: [:]))
            try data.write(
                to: URL(fileURLWithPath: output)
                    .appendingPathComponent("main_after_settings_request.png"))
        }
        window.contentView = nil
    }

    func testFullSettingsRender() throws {
        let view = SettingsView(store: store(), onSettingsChanged: {}, onShowModelStatus: {})
            .frame(width: 790)
            .background(EmberTheme.background)
            .foregroundStyle(EmberTheme.text)
            .tint(EmberTheme.accent)
        try snapshot(view, size: CGSize(width: 790, height: 2200), name: "settings_full")
    }

    func testFirstRunSheetsRender() throws {
        try snapshot(
            PrivacyChoiceView(store: store(), onDone: {}), size: CGSize(width: 540, height: 420),
            name: "sheet_privacy_choice")
        try snapshot(
            PermissionGuideView(guide: .inputMonitoring, onDone: {}),
            size: CGSize(width: 540, height: 380), name: "sheet_permission_input_monitoring")
    }

    func testCapsuleStatesRender() throws {
        for (mode, level, name) in [
            (PillState.Mode.idle, 0.0, "capsule_idle"),
            (PillState.Mode.listening, 0.7, "capsule_listening"),
            (PillState.Mode.working, 0.0, "capsule_working"),
        ] {
            let state = PillState()
            state.mode = mode
            state.level = level
            let view = PillView(state: state)
                .frame(width: 120, height: 40)
                .background(Color(white: 0.2))
            try snapshot(view, size: CGSize(width: 120, height: 40), name: name)
        }
    }

    func testModelStatusRenders() throws {
        let state = ModelStartupState()
        state.apply(
            name: "asr", state: "ready", progress: 1,
            detail: "batch + streaming demos passed: phonon startup check", loadMs: 2140)
        state.apply(
            name: "llm", state: "loading", progress: 0.6,
            detail: "weights loaded · refiner; compiling demo request", loadMs: nil)
        try snapshot(
            ModelStartupView(state: state), size: CGSize(width: 520, height: 300),
            name: "model_status")
    }
}
