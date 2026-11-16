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
        let terminal = #","app_context":{"bundle_id":"com.apple.Terminal","app_name":"Terminal","category":"code"}"#
        let slack = #","app_context":{"bundle_id":"com.tinyspeck.slackmacgap","app_name":"Slack","category":"work_chat"}"#
        let mail = #","app_context":{"bundle_id":"com.apple.mail","app_name":"Mail","category":"email"}"#
        let discord = #","app_context":{"bundle_id":"com.hnc.Discord","app_name":"Discord","category":"chat"}"#
        let recordings: [(String, UInt64, String, String, String)] = [
            ("rec_a", now - 60_000,
             "schedule the texture build for thursday at three wait scratch that friday at four",
             "Schedule the texture build for Friday at 4.", terminal),
            ("rec_b", now - 3_600_000, "um sounds good uh ship it after lunch",
             "Sounds good, ship it after lunch.",
             slack + #","user_edit":{"pasted_text":"Sounds good, ship it after lunch.","edited_text":"Sounds good, ship it after lunch!","outcome":"edited","end_reason":"focus_left"}"#),
            ("rec_c", now - 7_200_000, "run v llm on black well tonight and log the throughput",
             "Run vLLM on Blackwell tonight and log the throughput.", terminal),
            ("rec_d", now - 86_400_000,
             "thanks for the notes i'll send the revised draft on friday",
             "Thanks for the notes. I will send the revised draft on Friday.", mail),
            ("rec_e", now - 3 * 86_400_000, "yeah i'm down for saturday if it's after two",
             "yeah I'm down for saturday if it's after 2", discord),
        ]
        for (id, created, raw, final, extra) in recordings {
            try writeRecording(
                id: id, created: created, raw: raw, final: final, durationMs: 4_200,
                extra: extra)
        }
        // A year of history for Insights: weekday-heavy, office hours and
        // evenings, getting a little faster over time. Deterministic.
        var seed: UInt64 = 11
        func random() -> Double {
            seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Double(seed >> 11) / Double(1 << 53)
        }
        let apps = [
            (terminal, 0.38), (slack, 0.24), (mail, 0.14), (discord, 0.11),
            (#","app_context":{"bundle_id":"com.apple.Notes","app_name":"Notes","category":"docs"}"#, 0.08),
            ("", 0.05),
        ]
        let sentence = "we should benchmark the kernel on the new box before friday"
        for index in 0..<520 {
            let daysBack = Double(Int(random() * random() * 330) + 1)
            var created = Date(timeIntervalSince1970: Double(now) / 1_000 - daysBack * 86_400)
            let weekday = Calendar.current.component(.weekday, from: created)
            if (weekday == 1 || weekday == 7) && random() < 0.6 { continue }
            let hour = random() < 0.75 ? 9 + Int(random() * 9) : 20 + Int(random() * 3)
            created = Calendar.current.date(
                bySettingHour: hour, minute: Int(random() * 59), second: 0, of: created) ?? created
            var pick = random()
            var app = ""
            for (candidate, share) in apps {
                pick -= share
                if pick <= 0 { app = candidate; break }
            }
            let words = 8 + Int(random() * 40)
            let wpm = 120 + (330 - daysBack) / 330 * 45 + (random() - 0.5) * 20
            let text = Array(repeating: sentence, count: words / 11 + 1).joined(separator: " ")
                .split(separator: " ").prefix(words).joined(separator: " ")
            try writeRecording(
                id: "hist_\(index)", created: UInt64(created.timeIntervalSince1970 * 1_000),
                raw: text + " um", final: text.prefix(1).uppercased() + text.dropFirst() + ".",
                durationMs: Int(Double(words) / wpm * 60_000), extra: app)
        }
        let macros = """
            {"schema_version":1,"macros":[
             {"trigger":"my email","expansion":"sam.lee@example.com"},
             {"trigger":"sign off","expansion":"Best,\\nAlex"},
             {"trigger":"standup template","expansion":"Yesterday:\\nToday:\\nBlockers:"},
             {"trigger":"the repo link","expansion":"https://github.com/example/phonon"}]}
            """
        try Data(macros.utf8).write(to: directory.appendingPathComponent("macros.json"))
        try writeMeeting(
            id: "meeting_a", app: ("us.zoom.xos", "zoom.us"), title: "Design sync with Sam and Priya",
            startedSecondsAgo: 5_400, chunks: 6)
        try writeMeeting(
            id: "meeting_b", app: ("com.hnc.Discord", "Discord"), title: nil,
            startedSecondsAgo: 2 * 86_400, chunks: 2)
        // Invented words for the design sync's transcript.
        let lines: [(Double, Double, TranscriptSpeaker, String)] = [
            (0.4, 3.1, .me, "Morning both. Can you hear me okay?"),
            (3.6, 9.8, .call,
             "Yes, loud and clear. Priya is joining in a minute, she is just finishing the review."),
            (10.2, 18.5, .me,
             "Great. I want to settle the onboarding flow today, mostly the second screen and "
                + "whether the permission prompt comes before or after the sample dictation."),
            (19.0, 24.2, .call, "After, I think. People trust it more once they have seen it work."),
            (24.8, 27.0, .me, "Agreed. Let's write that down."),
        ]
        let transcript = MeetingTranscript(
            model: "mlx-community/parakeet-tdt-0.6b-v2", createdAt: "2026-11-09T12:00:00Z",
            turns: lines.map {
                MeetingTranscript.Turn(startSeconds: $0.0, endSeconds: $0.1, speaker: $0.2, text: $0.3)
            })
        try transcript.save(to: directory.appendingPathComponent("Meetings/meeting_a"))
    }

    private func writeRecording(
        id: String, created: UInt64, raw: String, final: String, durationMs: Int, extra: String
    ) throws {
        let folder = directory.appendingPathComponent("Corpus/\(id)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let metadata = """
            {"id":"\(id)","created_at_unix_ms":\(created),"source":"bar",
             "microphone":"MacBook Pro Microphone","audio_duration_ms":\(durationMs),
             "raw_transcript":"\(raw)","final_transcript":"\(final)",
             "dictionary_corrections":[{"count":1}],
             "llm":{"latency_ms":118,"ttft_ms":41,"tokens_per_second":212}\(extra)}
            """
        try Data(metadata.utf8).write(to: folder.appendingPathComponent("metadata.json"))
    }

    /// A meeting the way MeetingRecorder leaves one: 5 s chunks per track,
    /// speech-like bursts on the mic, the call answering in between.
    private func writeMeeting(
        id: String, app: (String, String), title: String?, startedSecondsAgo: Double, chunks: Int
    ) throws {
        let folder = directory.appendingPathComponent("Meetings/\(id)", isDirectory: true)
        let rate = 8_000
        let chunkSamples = rate * 5
        var micChunks: [MeetingChunk] = []
        var systemChunks: [MeetingChunk] = []
        for track in [MeetingTrack.mic, .system] {
            let trackFolder = folder.appendingPathComponent(track.rawValue, isDirectory: true)
            try FileManager.default.createDirectory(
                at: trackFolder, withIntermediateDirectories: true)
            for index in 0..<chunks {
                var data = ChunkedWavWriter.header(samples: chunkSamples, rate: rate)
                for sample in 0..<chunkSamples {
                    let t = Double(index * chunkSamples + sample) / Double(rate)
                    let turn = Int(t / 3.5) % 2 == (track == .mic ? 0 : 1)
                    let envelope = turn ? 0.25 + 0.75 * abs(sin(t * 2.1) * sin(t * 0.7 + 1)) : 0.03
                    let value = Int16(envelope * 20_000 * sin(t * 2 * .pi * 180))
                    withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
                }
                let file = ChunkedWavWriter.fileName(index)
                try data.write(to: trackFolder.appendingPathComponent(file))
                let chunk = MeetingChunk(
                    index: index, file: file, startSample: Int64(index * chunkSamples),
                    sampleCount: Int64(chunkSamples))
                if track == .mic { micChunks.append(chunk) } else { systemChunks.append(chunk) }
            }
        }
        var metadata = MeetingMetadata(
            id: id,
            startedAt: ISO8601DateFormatter().string(
                from: Date().addingTimeInterval(-startedSecondsAgo)),
            startedHostNs: 1_000)
        metadata.appBundleID = app.0
        metadata.appName = app.1
        metadata.windowTitle = title
        metadata.micDevice = "MacBook Pro Microphone"
        metadata.sampleRate = rate
        metadata.micChunks = micChunks
        metadata.systemChunks = systemChunks
        metadata.micStartOffsetSamples = 0
        metadata.systemStartOffsetSamples = 0
        metadata.endedAt = ISO8601DateFormatter().string(from: Date())
        metadata.endReason = "call_ended"
        try JSONEncoder().encode(metadata).write(to: folder.appendingPathComponent("metadata.json"))
    }

    @discardableResult
    private func snapshot<V: View>(
        _ view: V, size: CGSize, name: String, settle: TimeInterval = 0.3
    ) throws -> NSBitmapImageRep {
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
        RunLoop.main.run(until: Date().addingTimeInterval(settle))
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
        XCTAssertGreaterThan(store.history.count, 300)
        XCTAssertEqual(store.macros.count, 4)
        XCTAssertEqual(store.meetings.map(\.id), ["meeting_a", "meeting_b"])
        var renders: [(NativeAppPage, PhononMainView.InitialState, String)] =
            NativeAppPage.allCases.map { ($0, .init(), "main_\($0.rawValue.lowercased())") }
        renders.append((.home, .init(openDictation: "rec_a", showMore: true), "main_home_expanded"))
        renders.append((.settings, .init(advancedOpen: true), "main_settings_advanced"))
        for (page, initial, name) in renders {
            let view = PhononMainView(
                store: store, initialPage: page, initialState: initial, onSettingsChanged: {},
                onShowModelStatus: {})
            // Notes draws its waveform once the chunks are read off the main thread.
            let rep = try snapshot(
                view, size: CGSize(width: 980, height: 680), name: name,
                settle: page == .notes ? 1.0 : 0.3)
            XCTAssertGreaterThanOrEqual(rep.pixelsWide, 980)
        }
    }

    func testANoteWithATranscriptAndTheOtherTranscriptStatesRender() throws {
        let store = store()
        let note = try XCTUnwrap(store.meetings.first { $0.id == "meeting_a" })
        let transcript = try XCTUnwrap(note.transcript)
        XCTAssertEqual(transcript.turns.count, 5)
        XCTAssertEqual(store.transcriptState(note), .ready(transcript))
        let states: [(NoteTranscriptState, String)] = [
            (.ready(transcript), "note_transcript"),
            (.transcribing(progress: 0.42), "note_transcribing"),
            (.failed("transcribe span failed: the speech model is not loaded"), "note_transcript_failed"),
            (.notTranscribed, "note_not_transcribed"),
        ]
        for (state, name) in states {
            let view = NoteDetail(note: note, transcript: state)
                .frame(width: 700)
                .background(EmberTheme.window)
                .foregroundStyle(EmberTheme.ink)
                .font(.system(size: 13.5))
            let rep = try snapshot(view, size: CGSize(width: 700, height: 560), name: name, settle: 0.6)
            XCTAssertGreaterThanOrEqual(rep.pixelsWide, 700)
        }
    }

    func testInsightsChartsRenderTall() throws {
        let view = InsightsView(store: store())
            .frame(width: 784)
            .background(EmberTheme.window)
            .foregroundStyle(EmberTheme.ink)
            .font(.system(size: 13.5))
        try snapshot(view, size: CGSize(width: 784, height: 1180), name: "insights_full")
    }

    /// The app menu's Settings… item must land on Settings, not on whichever
    /// page the window showed last.
    func testPageRequestSwitchesTheOpenWindow() throws {
        let store = store()
        let view = PhononMainView(
            store: store, initialPage: .notes, onSettingsChanged: {}, onShowModelStatus: {})
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

    func testUpdateAvailableRender() throws {
        let store = store()
        store.availableUpdate = AvailableUpdate(
            version: "0.2.5",
            pageURL: URL(string: "https://github.com/Infatoshi/phonon/releases/tag/v0.2.5")!)
        let view = PhononMainView(
            store: store, initialPage: .home, onSettingsChanged: {}, onShowModelStatus: {})
        try snapshot(view, size: CGSize(width: 980, height: 680), name: "main_update_available")
    }

    func testFullSettingsRender() throws {
        let view = SettingsView(store: store(), onSettingsChanged: {}, onShowModelStatus: {})
            .frame(width: 790)
            .background(EmberTheme.background)
            .foregroundStyle(EmberTheme.text)
            .tint(EmberTheme.accent)
        try snapshot(view, size: CGSize(width: 790, height: 2200), name: "settings_full")
    }

    /// Settings with two apps pinned to a writing mode.
    func testSettingsWithPerAppModesRender() throws {
        let store = store()
        store.setAppWritingMode("com.apple.Terminal", .minimal)
        store.setAppWritingMode("com.tinyspeck.slackmacgap", .casual)
        XCTAssertEqual(store.dictatedApps.first?.bundleID, "com.apple.Terminal")
        let view = SettingsView(
            store: store, onSettingsChanged: {}, onShowModelStatus: {}, advancedOpen: true
        )
        .frame(width: 790)
        .background(EmberTheme.background)
        .foregroundStyle(EmberTheme.text)
        .tint(EmberTheme.accent)
        try snapshot(view, size: CGSize(width: 790, height: 2600), name: "settings_per_app_modes")
    }

    /// Every banner the main window can show while a permission is missing.
    func testPermissionBannerRowsRender() throws {
        let items: [PermissionAttention] = [
            .inputMonitoring(key: "Globe (fn)"), .microphone, .accessibility,
            .globeKey(action: "shows Emoji & Symbols"),
        ]
        let view = VStack(spacing: 10) {
            ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                PermissionBannerRow(item: item) {}
            }
        }
        .padding(30)
        .frame(width: 790)
        .background(EmberTheme.window)
        .foregroundStyle(EmberTheme.ink)
        .font(.system(size: 13.5))
        try snapshot(view, size: CGSize(width: 790, height: 480), name: "permission_banners")
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

    /// Live text above the capsule, each at a fixed instant: one, three and six lines
    /// (scrolled), the heard tail settled and mid-fade, the dot alone (a voice, no words yet)
    /// and lit, and a rewrite halfway through its morph.
    func testLiveTextOverlayRenders() throws {
        let sentence =
            "Schedule the texture build for Friday at 4 and make sure the ImageMagick script "
            + "runs before the mod upload."
        let three = Array(repeating: sentence, count: 2).joined(separator: " ")
        let six = Array(repeating: sentence, count: 4).joined(separator: " ")
        let rewriteStart: TimeInterval = 2_000
        let tailScene: (LiveTextState) -> Void = { state in
            state.applyDraft(
                "Schedule the texture build for Friday at 4 and make sure the ImageMagick "
                    + "script runs before", coveredWords: 17, complete: true)
            state.setTranscript(
                "schedule the texture build for friday at four and make sure the image "
                    + "magic script runs before the mod upload")
        }
        let scenes: [(String, TimeInterval, (LiveTextState) -> Void)] = [
            ("live_text_1_line", 1_001, { state in
                state.applyDraft("Schedule the texture build for Friday at 4", coveredWords: 8,
                    complete: true)
            }),
            ("live_text_3_lines", 1_001, { state in
                state.applyDraft(three, coveredWords: 40, complete: true)
            }),
            ("live_text_6_lines_scrolled", 1_001, { state in
                state.applyDraft(six, coveredWords: 80, complete: true)
            }),
            ("live_text_tail", 1_001, tailScene),
            ("live_text_tail_appearing_040ms", 1_000.04, tailScene),
            ("live_text_dot_only", 1_000, { state in state.noteVoice() }),
            ("live_text_dot_lit", 1_001, { state in
                state.noteVoice()
                state.applyDraft("Schedule the texture build", coveredWords: 4, complete: true)
                for _ in 0..<6 { state.noteLevel(0.7) }
            }),
            ("live_text_morph_060ms", rewriteStart + 0.06, { _ in }),
            ("live_text_morph_mid", rewriteStart + 0.1, { _ in }),
            ("live_text_morph_160ms", rewriteStart + 0.16, { _ in }),
        ]
        for (name, at, build) in scenes {
            var clock: TimeInterval = 1_000
            let state = LiveTextState(clock: { clock })
            if name.hasPrefix("live_text_morph") {
                state.applyDraft(
                    "Schedule the texture build for Thursday at three.", coveredWords: 8,
                    complete: true)
                state.setTranscript(
                    "schedule the texture build for thursday at three wait scratch that friday")
                clock = rewriteStart
                state.applyDraft(
                    "Schedule the texture build for Friday", coveredWords: 12, complete: true)
                state.setTranscript(
                    "schedule the texture build for thursday at three wait scratch that friday "
                        + "at four")
            } else {
                build(state)
            }
            let pill = PillState()
            pill.mode = .listening
            pill.level = 0.6
            let view = VStack(spacing: LiveTextPanel.gap) {
                LiveTextView(state: state, fixedTime: at)
                PillView(state: pill)
            }
            .padding(.bottom, PanelGeometry.bottomInset)
            .frame(width: 520, height: 200, alignment: .bottom)
            .background(Color(red: 0.07, green: 0.066, blue: 0.063))
            // ImageRenderer, not cacheDisplay: the layer cache drops SwiftUI's blur.
            let renderer = ImageRenderer(content: view.environment(\.colorScheme, .dark))
            renderer.scale = 2
            let image = try XCTUnwrap(renderer.cgImage)
            XCTAssertEqual(image.width, 1040)
            if let output = ProcessInfo.processInfo.environment["PHONON_RENDER_DIR"] {
                let data = try XCTUnwrap(
                    NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:]))
                try data.write(
                    to: URL(fileURLWithPath: output).appendingPathComponent("\(name).png"))
            }
        }
    }

    func testNotPastedCardRenders() throws {
        let text =
            "Hey, can you send me the texture build logs from last night? I want to see why "
            + "the ImageMagick step failed before the mod upload."
        let notices: [(String, Notice)] = [
            ("not_pasted", .notPasted(text)),
            ("notice_accessibility", .needsAccessibility(text)),
            ("notice_not_heard", .notHeard(mic: "MacBook Pro Microphone")),
            ("notice_mic_silent", .micSilent(mic: "AirPods Pro")),
            (
                "notice_update",
                .message(
                    title: "Phonon 1.0.2 is available",
                    detail: "You have 1.0.1. Update copies the Homebrew command; paste it into "
                        + "Terminal, then open Phonon again.",
                    action: "Update")
            ),
        ]
        for (name, notice) in notices {
            let model = NotPastedModel()
            model.notice = notice
            model.visible = true
            let view = NotPastedView(model: model)
                .background(Color(red: 0.07, green: 0.066, blue: 0.063))
            let renderer = ImageRenderer(content: view.environment(\.colorScheme, .dark))
            renderer.scale = 2
            let image = try XCTUnwrap(renderer.cgImage)
            XCTAssertGreaterThan(image.width, 1000)
            if let output = ProcessInfo.processInfo.environment["PHONON_RENDER_DIR"] {
                let data = try XCTUnwrap(
                    NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:]))
                try data.write(to: URL(fileURLWithPath: output).appendingPathComponent("\(name).png"))
            }
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

    /// One window: the loader covers the app until startup finishes, then the
    /// app shows in the same window; an engine restart brings the loader back.
    func testMainWindowShowsLoaderThenApp() throws {
        let state = ModelStartupState()
        state.apply(
            name: "asr", state: "ready", progress: 1,
            detail: "batch + streaming demos passed: phonon startup check", loadMs: 2140)
        state.apply(
            name: "llm", state: "loading", progress: 0.6,
            detail: "weights loaded · refiner; compiling demo request", loadMs: nil)
        let view = PhononWindowRoot(
            startup: state,
            main: PhononMainView(
                store: store(), initialPage: .home, onSettingsChanged: {}, onShowModelStatus: {}))
        let size = CGSize(width: 980, height: 680)
        XCTAssertTrue(state.loaderVisible)
        let loading = try snapshot(view, size: size, name: "window_loading")
        state.markReady()
        XCTAssertTrue(state.loaderVisible, "the loader stays up to show the ready state")
        state.hideLoader()
        let ready = try snapshot(view, size: size, name: "window_ready")
        XCTAssertNotEqual(loading.tiffRepresentation, ready.tiffRepresentation)
        state.markRestarting()
        XCTAssertTrue(state.loaderVisible)
        XCTAssertFalse(state.ready)
    }
}
