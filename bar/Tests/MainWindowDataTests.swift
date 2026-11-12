import CoreImage
import XCTest

@testable import PhononBar

/// The data behind the redesigned main window: macros, dictionary chips and
/// import, app icons from the corpus, writing style, export, Insights and Notes.
@MainActor
final class MainWindowDataTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "phonon-main-window-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    // MARK: Macros

    func testAMacroFiresOnlyWhenTheWholeDictationIsItsTrigger() {
        let macros = [
            PhononMacro(trigger: "my email", expansion: "sam.lee@example.com"),
            PhononMacro(trigger: "Sign off", expansion: "Best,\nAlex"),
        ]
        XCTAssertEqual(MacroBook.expand("My email.", macros: macros), "sam.lee@example.com")
        XCTAssertEqual(MacroBook.expand("  my   EMAIL!  ", macros: macros), "sam.lee@example.com")
        XCTAssertEqual(MacroBook.expand("sign off", macros: macros), "Best,\nAlex")
        XCTAssertEqual(MacroBook.expand("\"Sign off.\"", macros: macros), "Best,\nAlex")
        // part of a longer dictation, or a different phrase: typed as dictated
        for text in ["Send it to my email.", "my emails", "", "email"] {
            XCTAssertEqual(MacroBook.expand(text, macros: macros), text)
        }
        XCTAssertEqual(MacroBook.expand("my email", macros: []), "my email")
    }

    func testMacrosPersistInMacrosJSONAndTheBackup() throws {
        let mirror = directory.appendingPathComponent("mirror", isDirectory: true)
        let store = NativeAppStore(supportDirectory: directory, mirrorDirectory: mirror)
        store.upsertMacro(trigger: " my email ", expansion: "sam.lee@example.com")
        store.upsertMacro(trigger: "sign off", expansion: "Best,\nAlex")
        store.upsertMacro(trigger: "My Email", expansion: "sam@work.example")  // same trigger
        store.upsertMacro(trigger: "", expansion: "nothing")  // ignored
        XCTAssertEqual(store.macros.map(\.trigger), ["My Email", "sign off"])
        XCTAssertEqual(store.macros.first?.expansion, "sam@work.example")

        store.upsertMacro(
            trigger: "signoff", expansion: "Cheers,\nAlex", replacing: MacroBook.normalize("sign off"))
        XCTAssertEqual(store.macros.map(\.trigger), ["My Email", "signoff"])

        let reopened = NativeAppStore(supportDirectory: directory, mirrorDirectory: mirror)
        XCTAssertEqual(reopened.macros, store.macros)
        reopened.removeMacro(id: MacroBook.normalize("my email"))
        XCTAssertEqual(MacroBook.load(from: directory.appendingPathComponent("macros.json")).count, 1)
        XCTAssertTrue(PhononDataMirror.mirroredFiles.contains("macros.json"))
    }

    // MARK: Dictionary

    func testTheAddFieldTakesATermOrAFix() throws {
        let store = NativeAppStore(supportDirectory: directory)
        store.addDictionaryText("Parakeet")
        store.addDictionaryText("black well → Blackwell")
        store.addDictionaryText("toner -> Kneron")
        XCTAssertNil(NativeAppStore.parseFix("→ Blackwell"))
        XCTAssertNil(NativeAppStore.parseFix("just a term"))
        let byPhrase = Dictionary(uniqueKeysWithValues: store.dictionaryEntries.map { ($0.phrase, $0) })
        XCTAssertNil(byPhrase["Parakeet"]?.replacement)
        XCTAssertEqual(byPhrase["black well"]?.replacement, "Blackwell")
        XCTAssertEqual(byPhrase["toner"]?.replacement, "Kneron")
    }

    func testImportMergesADictionaryFileOrAWordList() throws {
        let store = NativeAppStore(supportDirectory: directory)
        store.addDictionaryText("Parakeet")
        let json = directory.appendingPathComponent("theirs.json")
        try Data(
            #"{"schema_version":1,"updated_at_unix_ms":0,"entries":[{"phrase":"Parakeet"},{"phrase":"vee llm","replacement":"vLLM","usage_count":40},{"phrase":"cuBLAS","spoken_forms":["coo blast"]}]}"#
                .utf8
        ).write(to: json)
        XCTAssertEqual(store.importDictionary(from: json), 2, "Parakeet is already there")
        let text = directory.appendingPathComponent("terms.txt")
        try Data("# team words\nImageMagick\n\ntoner → Kneron\ncuBLAS\n".utf8).write(to: text)
        XCTAssertEqual(store.importDictionary(from: text), 2)
        XCTAssertEqual(store.importDictionary(from: text), 0, "importing twice adds nothing")
        let imported = store.dictionaryEntries.first { $0.phrase == "vee llm" }
        XCTAssertEqual(imported?.replacement, "vLLM")
        XCTAssertEqual(imported?.usageCount, 0)
        XCTAssertEqual(imported?.source, "import")
        XCTAssertEqual(store.dictionaryEntries.count, 5)
    }

    // MARK: Corpus app, settings, export

    func testACorpusEntryKnowsTheAppItWentInto() throws {
        let object: [String: Any] = [
            "id": "a", "created_at_unix_ms": 1, "raw_transcript": "hi", "final_transcript": "Hi.",
            "app_context": ["bundle_id": "com.tinyspeck.slackmacgap", "app_name": "Slack",
                            "category": "work_chat"],
        ]
        let metadata = try JSONDecoder().decode(
            NativeRecordingMetadata.self, from: JSONSerialization.data(withJSONObject: object))
        XCTAssertEqual(metadata.appContext?.bundleID, "com.tinyspeck.slackmacgap")
        XCTAssertEqual(metadata.appContext?.category, "work_chat")
        let broken: [String: Any] = [
            "id": "b", "created_at_unix_ms": 1, "app_context": "nonsense",
        ]
        let kept = try JSONDecoder().decode(
            NativeRecordingMetadata.self, from: JSONSerialization.data(withJSONObject: broken))
        XCTAssertNil(kept.appContext, "a malformed app does not hide the dictation")
    }

    func testWritingStyleAndStreamRefineRoundTripThroughSettingsJSON() throws {
        let url = directory.appendingPathComponent("settings.json")
        try Data(#"{"schema_version":2,"writing_style":"loud","profile_prefix":true}"#.utf8)
            .write(to: url)
        let store = NativeAppStore(supportDirectory: directory)
        XCTAssertEqual(store.settings.writingStyle, .app, "unknown reads as follow the app")
        XCTAssertNil(store.settings.streamRefine)
        store.updateSettings { $0.writingStyle = .formal }
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        XCTAssertEqual(object["writing_style"] as? String, "formal")
        XCTAssertNil(object["stream_refine"], "left to the engine until the user picks")
        XCTAssertEqual(object["profile_prefix"] as? Bool, true, "engine keys survive")
        store.updateSettings { $0.streamRefine = false }
        object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        XCTAssertEqual(object["stream_refine"] as? Bool, false)
        XCTAssertEqual(NativeAppStore(supportDirectory: directory).settings.writingStyle, .formal)
    }

    /// A settings.json from before tap-to-toggle became the only behaviour still loads,
    /// and its old `tap_to_toggle` key is left alone like any key the app no longer reads.
    func testPerAppModesRoundTripThroughSettingsJSON() throws {
        let url = directory.appendingPathComponent("settings.json")
        try Data(
            #"{"schema_version":2,"tap_to_toggle":false,"writing_modes_by_app":{"com.mitchellh.ghostty":"prompt","com.hnc.Discord":"verbatim"},"profile_prefix":true}"#
                .utf8
        ).write(to: url)
        let store = NativeAppStore(supportDirectory: directory)
        XCTAssertEqual(
            store.settings.writingModesByApp, ["com.mitchellh.ghostty": "prompt"],
            "an unknown mode is dropped")
        store.setAppWritingMode("com.apple.Terminal", .minimal)
        store.setAppWritingMode("com.mitchellh.ghostty", nil)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        XCTAssertEqual(
            object["writing_modes_by_app"] as? [String: String], ["com.apple.Terminal": "minimal"])
        XCTAssertEqual(object["profile_prefix"] as? Bool, true, "engine keys survive")
        let reread = NativeAppStore(supportDirectory: directory).settings
        XCTAssertEqual(reread.writingModesByApp, ["com.apple.Terminal": "minimal"])
    }

    /// Issue #24: off by default, no default endpoint, and the key never in settings.json.
    func testOwnEndpointCommandAndSettings() throws {
        var settings = NativeSettings()
        XCTAssertFalse(settings.byokEnabled)
        XCTAssertEqual(settings.byokBaseURL, "")
        XCTAssertEqual(OwnEndpoint.engineCommand(settings: settings, apiKey: "k") as NSDictionary,
                       ["cmd": "byok"] as NSDictionary, "off: the engine is told off")
        settings.byokEnabled = true
        settings.byokBaseURL = "ftp://x"
        settings.byokModel = "m"
        XCTAssertEqual(OwnEndpoint.engineCommand(settings: settings, apiKey: "k").count, 1, "bad URL is off")
        settings.byokBaseURL = " https://api.example.com/v1 "
        let on = OwnEndpoint.engineCommand(settings: settings, apiKey: "sk-1")
        XCTAssertEqual(on["base_url"] as? String, "https://api.example.com/v1")
        XCTAssertEqual(on["model"] as? String, "m")
        XCTAssertEqual(on["api_key"] as? String, "sk-1")
        XCTAssertTrue(OwnEndpoint.validBaseURL("http://localhost:8080/v1"))
        XCTAssertFalse(OwnEndpoint.validBaseURL("api.example.com"))

        let url = directory.appendingPathComponent("settings.json")
        let store = NativeAppStore(supportDirectory: directory)
        store.updateSettings {
            $0.byokEnabled = true
            $0.byokBaseURL = "https://api.example.com/v1"
            $0.byokModel = "m"
        }
        let text = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(text.contains("\"byok_base_url\""))
        XCTAssertFalse(text.contains("api_key"), "the key is never written to settings.json")
        XCTAssertTrue(NativeAppStore(supportDirectory: directory).settings.byokEnabled)
    }

    func testOwnEndpointKeyRoundTripsThroughTheKeychain() throws {
        let service = "com.infatoshi.phonon.byok.test-\(UUID().uuidString)"
        defer { OwnEndpoint.saveKey("", service: service) }
        let status = OwnEndpoint.saveKey(" sk-test-123 \n", service: service)
        try XCTSkipIf(status != errSecSuccess, "no writable login keychain here (\(status))")
        XCTAssertEqual(OwnEndpoint.readKey(service: service), "sk-test-123")
        OwnEndpoint.saveKey("sk-test-456", service: service)
        XCTAssertEqual(OwnEndpoint.readKey(service: service), "sk-test-456", "replaced, not duplicated")
        OwnEndpoint.saveKey("", service: service)
        XCTAssertNil(OwnEndpoint.readKey(service: service))
    }

    /// The WebSocket API: off by default; on, it has a token, kept in settings.json.
    func testWebSocketAPISettings() throws {
        var settings = NativeSettings()
        XCTAssertFalse(settings.websocketAPI)
        XCTAssertEqual(settings.websocketAPIPort, 8377)
        XCTAssertEqual(settings.websocketAPIToken, "")
        WebSocketAPI.turnOn(&settings)
        XCTAssertTrue(settings.websocketAPI)
        let token = settings.websocketAPIToken
        XCTAssertEqual(token.count, 24, "turning it on makes a token")
        settings.websocketAPI = false
        WebSocketAPI.turnOn(&settings)
        XCTAssertEqual(settings.websocketAPIToken, token, "on again: the same token, phones stay paired")

        let url = directory.appendingPathComponent("settings.json")
        try Data(#"{"schema_version":2,"websocket_api_bind":"100.64.0.7"}"#.utf8).write(to: url)
        let store = NativeAppStore(supportDirectory: directory)
        store.updateSettings {
            WebSocketAPI.turnOn(&$0)
            $0.websocketAPIPort = 9001
        }
        let text = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(text.contains("\"websocket_api\" : true"), text)
        XCTAssertTrue(text.contains("\"websocket_api_port\" : 9001"), text)
        XCTAssertTrue(
            text.contains("\"websocket_api_token\" : \"\(store.settings.websocketAPIToken)\""),
            "where the engine reads it")
        XCTAssertTrue(text.contains("100.64.0.7"), "a hand-set bind address survives a save")
        let reread = NativeAppStore(supportDirectory: directory).settings
        XCTAssertTrue(reread.websocketAPI)
        XCTAssertEqual(reread.websocketAPIPort, 9001)
        XCTAssertEqual(reread.websocketAPIToken, store.settings.websocketAPIToken)

        // a port nothing can listen on reads as the default
        try Data(#"{"schema_version":2,"websocket_api_port":70000}"#.utf8).write(to: url)
        XCTAssertEqual(NativeAppStore(supportDirectory: directory).settings.websocketAPIPort, 8377)
        XCTAssertTrue(WebSocketAPI.validPort(1))
        XCTAssertFalse(WebSocketAPI.validPort(0))
        XCTAssertFalse(WebSocketAPI.validPort(65_536))
    }

    func testWebSocketAPITokensAreLongAndTypeable() {
        let tokens = Set((0..<50).map { _ in WebSocketAPI.newToken() })
        XCTAssertEqual(tokens.count, 50, "each one is new")
        for token in tokens {
            XCTAssertEqual(token.count, 24)
            XCTAssertGreaterThanOrEqual(token.count, 16, "the engine's minimum")
            XCTAssertTrue(token.allSatisfy { WebSocketAPI.tokenAlphabet.contains($0) })
        }
        for lookalike in "il1o0" {
            XCTAssertFalse(WebSocketAPI.tokenAlphabet.contains(lookalike))
        }
    }

    func testWebSocketAPIStatusLineSaysWhereOrWhyNot() {
        XCTAssertEqual(
            WebSocketAPI.statusLine(engineStatus: nil, engineReady: false),
            "Starts when the models are loaded.")
        XCTAssertEqual(WebSocketAPI.statusLine(engineStatus: nil, engineReady: true), "Starting…")
        XCTAssertEqual(
            WebSocketAPI.statusLine(
                engineStatus: ["listening": true, "address": "0.0.0.0:8377"], engineReady: true),
            "Listening on port 8377.")
        XCTAssertEqual(
            WebSocketAPI.statusLine(
                engineStatus: ["listening": false, "msg": "listen on 0.0.0.0:80: Permission denied"],
                engineReady: true),
            "Not listening: listen on 0.0.0.0:80: Permission denied.")
        XCTAssertEqual(
            WebSocketAPI.statusLine(engineStatus: ["listening": false, "msg": ""], engineReady: true),
            "Not listening.")
        // every address offered is one a phone could dial
        for address in WebSocketAPI.reachableAddresses() {
            XCTAssertFalse(address.hasPrefix("127."))
            XCTAssertFalse(address.hasPrefix("169.254."))
            XCTAssertEqual(address.split(separator: ".").count, 4, address)
        }
    }

    /// Home's graph: a dictation's words land in the minutes its audio spanned.
    func testWebSocketWordsSpreadOverTheMinutesSpoken() {
        // 12:00:30 local-free: whole minutes since the epoch keep the arithmetic exact
        let now = Date(timeIntervalSince1970: 1_000 * 60 + 30)
        XCTAssertEqual(WebSocketActivity.perMinute([], now: now), Array(repeating: 0, count: 60))

        // 60 words over the 90 s ending 10 s ago (minute 1000 + 20 s): it started at minute
        // 998 + 50 s, so 10 s fall in minute 998, 60 s in 999 and 20 s in 1000
        let spread = WebSocketDictation(end: now.addingTimeInterval(-10), words: 60, audioSeconds: 90)
        let values = WebSocketActivity.perMinute([spread], now: now)
        XCTAssertEqual(values.count, 60)
        XCTAssertEqual(values[59], 60 * 20 / 90, accuracy: 0.001, "the current minute is last")
        XCTAssertEqual(values[58], 60 * 60 / 90, accuracy: 0.001)
        XCTAssertEqual(values[57], 60 * 10 / 90, accuracy: 0.001)
        XCTAssertEqual(values.reduce(0, +), 60, accuracy: 0.001, "no word lost")

        // no audio length: all at the end; older than the hour: left out
        let instant = WebSocketDictation(end: now.addingTimeInterval(-125), words: 7, audioSeconds: 0)
        let old = WebSocketDictation(end: now.addingTimeInterval(-3_700), words: 50, audioSeconds: 10)
        let both = WebSocketActivity.perMinute([instant, old], now: now)
        XCTAssertEqual(both[57], 7)
        XCTAssertEqual(both.reduce(0, +), 7)

        let summary = WebSocketActivity.summary([spread, instant, old], now: now)
        XCTAssertEqual(summary.words, 67)
        XCTAssertEqual(summary.rate, 40, "60 words in 1.5 minutes of audio")
        XCTAssertNil(WebSocketActivity.summary([instant], now: now).rate)
        XCTAssertEqual(WebSocketActivity.scale([0, 3]), 40, "a floor, so a few words stay small")
        XCTAssertEqual(WebSocketActivity.scale([0, 133]), 140)
    }

    func testTheStoreKeepsTheLastHourOfWebSocketDictation() {
        let store = NativeAppStore(supportDirectory: directory)
        store.websocketLive = true
        store.recordWebSocketDictation(words: 0, audioMilliseconds: 500)
        XCTAssertFalse(store.websocketLive, "ended, even with nothing typed")
        XCTAssertTrue(store.websocketDictations.isEmpty, "nothing typed: nothing to draw")
        let now = Date()
        store.recordWebSocketDictation(
            words: 5, audioMilliseconds: 2_000, at: now.addingTimeInterval(-2 * 3_600))
        store.recordWebSocketDictation(words: 9, audioMilliseconds: 4_000, at: now)
        XCTAssertEqual(store.websocketDictations.map(\.words), [9], "older than an hour is dropped")
        XCTAssertEqual(store.websocketDictations.first?.audioSeconds, 4)
    }

    func testThePairingCodeCarriesTheAddressAndToken() throws {
        let uri = WebSocketAPI.pairingURI(address: "192.168.0.52", port: 8377, token: "fzqxz6qcsyh3")
        XCTAssertEqual(uri, "phonon://pair?url=ws%3A%2F%2F192.168.0.52%3A8377&token=fzqxz6qcsyh3")
        let parsed = try XCTUnwrap(URLComponents(string: uri))
        XCTAssertEqual(parsed.scheme, "phonon")
        XCTAssertEqual(parsed.host, "pair")
        XCTAssertEqual(parsed.queryItems?.first { $0.name == "url" }?.value, "ws://192.168.0.52:8377")
        XCTAssertEqual(parsed.queryItems?.first { $0.name == "token" }?.value, "fzqxz6qcsyh3")

        // and the QR code reads back as the same text
        let image = try XCTUnwrap(PairingCode.image(for: uri))
        let cgImage = try XCTUnwrap(image.cgImage(forProposedRect: nil, context: nil, hints: nil))
        // scale up with a quiet zone, as a camera would see it
        let side = cgImage.width * 10 + 80
        let context = try XCTUnwrap(
            CGContext(
                data: nil, width: side, height: side, bitsPerComponent: 8, bytesPerRow: 0,
                space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue))
        context.setFillColor(gray: 1, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: side, height: side))
        context.interpolationQuality = .none
        context.draw(cgImage, in: CGRect(x: 40, y: 40, width: side - 80, height: side - 80))
        let detector = try XCTUnwrap(
            CIDetector(ofType: CIDetectorTypeQRCode, context: nil, options: nil))
        let features = detector.features(in: CIImage(cgImage: try XCTUnwrap(context.makeImage())))
        XCTAssertEqual((features.first as? CIQRCodeFeature)?.messageString, uri)
    }

    func testThePairingAddressPrefersATailnet() {
        XCTAssertEqual(
            WebSocketAPI.preferredAddress(["10.147.20.111", "192.168.0.52", "100.101.102.103"]),
            "100.101.102.103")
        XCTAssertEqual(
            WebSocketAPI.preferredAddress(["10.147.20.111", "192.168.0.52"]), "192.168.0.52")
        XCTAssertEqual(WebSocketAPI.preferredAddress(["10.147.20.111"]), "10.147.20.111")
        XCTAssertEqual(
            WebSocketAPI.preferredAddress(["100.200.1.1", "10.0.0.2"]), "100.200.1.1",
            "100.200 is not a tailnet address, and nothing else is better")
        XCTAssertNil(WebSocketAPI.preferredAddress([]))
    }

    func testExportCopiesTheUsersDataButNotModelWeights() throws {
        let support = directory.appendingPathComponent("support", isDirectory: true)
        for path in [
            "Corpus/rec/metadata.json", "Meetings/m1/mic/0000.wav", "dictionary.json",
            "macros.json", "settings.json", "models/refiner/weights.bin", "bar.lock",
        ] {
            let url = support.appendingPathComponent(path)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data("x".utf8).write(to: url)
        }
        let out = directory.appendingPathComponent("Phonon Data", isDirectory: true)
        try PhononDataMirror.exportAll(from: support, to: out)
        let exists = { FileManager.default.fileExists(atPath: out.appendingPathComponent($0).path) }
        for kept in [
            "Corpus/rec/metadata.json", "Meetings/m1/mic/0000.wav", "dictionary.json",
            "macros.json", "settings.json",
        ] {
            XCTAssertTrue(exists(kept), kept)
        }
        XCTAssertFalse(exists("models"))
        XCTAssertFalse(exists("bar.lock"))
    }

    // MARK: Home and Insights

    func testFeedTimes() throws {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "America/Edmonton")!
        let now = try XCTUnwrap(
            calendar.date(from: DateComponents(year: 2026, month: 11, day: 4, hour: 22)))
        let at = { (days: Int, hour: Int, minute: Int) in
            calendar.date(
                from: DateComponents(year: 2026, month: 11, day: 4 - days, hour: hour, minute: minute))!
        }
        let formatterZone = NSTimeZone.default
        NSTimeZone.default = calendar.timeZone
        defer { NSTimeZone.default = formatterZone }
        XCTAssertEqual(FeedTime.label(at(0, 21, 41), now: now, calendar: calendar), "9:41 pm")
        XCTAssertEqual(FeedTime.label(at(1, 9, 0), now: now, calendar: calendar), "Yesterday")
        XCTAssertEqual(FeedTime.label(at(2, 9, 0), now: now, calendar: calendar), "Mon")
        XCTAssertEqual(FeedTime.label(at(20, 9, 0), now: now, calendar: calendar), "Oct 15")
        XCTAssertEqual(FeedTime.clock(2_472), "41:12")
        XCTAssertEqual(FeedTime.duration(seconds: 2_472), "41 min")
    }

    func testWordsFixedCountsHeardWordsPhononChanged() {
        XCTAssertEqual(InsightsData.fixedWords(heard: "ship it", wrote: "Ship it."), 0)
        XCTAssertEqual(
            InsightsData.fixedWords(
                heard: "Okay, my technical like CUDA and Kubloss and CUDNN terms. Uh let's see",
                wrote: "Okay, my technical like CUDA and Kublass and cuDNN terms. let's see"),
            2, "Kubloss changed, Uh dropped; CUDNN -> cuDNN is only case")
        XCTAssertEqual(InsightsData.fixedWords(heard: "", wrote: "Hi."), 0)
    }

    func testInsightsComeFromTheCorpus() throws {
        let calendar = Calendar.current
        let now = try XCTUnwrap(
            calendar.date(from: DateComponents(year: 2026, month: 11, day: 4, hour: 15)))
        let items: [(Int, Int, String?, String, String, UInt64)] = [
            (0, 10, "com.apple.Terminal", "run v llm now", "Run vLLM now.", 3_000),
            (0, 11, "com.apple.Terminal", "ship it", "Ship it.", 1_000),
            (1, 21, "com.tinyspeck.slackmacgap", "sounds good", "Sounds good.", 2_000),
            (40, 9, nil, "an old one here", "An old one here.", 2_000),
        ]
        let history = try items.enumerated().map { index, item in
            let date = calendar.date(byAdding: .day, value: -item.0, to: now)!
            let created = calendar.date(bySettingHour: item.1, minute: 0, second: 0, of: date)!
            var object: [String: Any] = [
                "id": "r\(index)", "created_at_unix_ms": Int(created.timeIntervalSince1970 * 1_000),
                "raw_transcript": item.3, "final_transcript": item.4, "audio_duration_ms": item.5,
            ]
            if let bundle = item.2 { object["app_context"] = ["bundle_id": bundle] }
            let metadata = try JSONDecoder().decode(
                NativeRecordingMetadata.self, from: JSONSerialization.data(withJSONObject: object))
            return NativeHistoryItem(
                metadata: metadata, directoryURL: URL(fileURLWithPath: "/r\(index)"),
                metadataURL: URL(fileURLWithPath: "/r\(index)/metadata.json"))
        }
        let usage = NativeAppStore.computeUsage(history)
        let data = InsightsData.build(history, usage: usage, now: now)
        XCTAssertEqual(data.words, 3 + 2 + 2 + 4)
        XCTAssertEqual(data.wordsPerMinute, usage.wordsPerMinute)
        XCTAssertEqual(data.wordsFixed, 2, "v llm -> vLLM: two heard words changed")
        XCTAssertEqual(data.days.count, 30)
        XCTAssertEqual(data.days.last?.words, 5)
        XCTAssertEqual(data.days.reduce(0) { $0 + $1.words }, 7, "the 40-day-old one is outside")
        XCTAssertEqual(data.topApp?.bundleID, "com.apple.Terminal")
        XCTAssertEqual(data.apps.map(\.words), [5, 2, 4])
        XCTAssertEqual(data.apps.last?.name, "Other")
        XCTAssertEqual(data.apps.reduce(0) { $0 + $1.share }, 1, accuracy: 1e-9)
        XCTAssertEqual(data.hours.count, 168)
        XCTAssertEqual(data.hours.reduce(0) { $0 + $1.words }, 11, "all within 12 weeks")
        XCTAssertEqual(data.year.reduce(0) { $0 + $1.words }, 11)
        XCTAssertLessThanOrEqual(data.year.count, 53 * 7)
        XCTAssertFalse(data.speed.isEmpty)
        XCTAssertEqual(InsightsData.level(0, max: 10), 0)
        XCTAssertEqual(InsightsData.level(1, max: 10), 1)
        XCTAssertEqual(InsightsData.level(10, max: 10), 4)
    }

    // MARK: Notes

    func testNotesReadTheMeetingFoldersMeetingRecorderWrites() throws {
        let folder = directory.appendingPathComponent("Meetings/m1", isDirectory: true)
        let rate = 8_000
        for track in ["mic", "system"] {
            let trackFolder = folder.appendingPathComponent(track, isDirectory: true)
            try FileManager.default.createDirectory(at: trackFolder, withIntermediateDirectories: true)
            for index in 0..<2 {
                var data = ChunkedWavWriter.header(samples: rate, rate: rate)
                let level: Int16 = track == "mic" && index == 1 ? 16_000 : 1_000
                for _ in 0..<rate {
                    withUnsafeBytes(of: level.littleEndian) { data.append(contentsOf: $0) }
                }
                try data.write(to: trackFolder.appendingPathComponent(ChunkedWavWriter.fileName(index)))
            }
        }
        var metadata = MeetingMetadata(
            id: "m1", startedAt: "2026-11-04T08:30:00Z", startedHostNs: 0)
        metadata.appBundleID = "us.zoom.xos"
        metadata.sampleRate = rate
        metadata.micChunks = (0..<2).map {
            MeetingChunk(
                index: $0, file: ChunkedWavWriter.fileName($0), startSample: Int64($0 * rate),
                sampleCount: Int64(rate))
        }
        metadata.systemChunks = metadata.micChunks
        metadata.micStartOffsetSamples = 0
        metadata.systemStartOffsetSamples = Int64(rate / 2)
        try JSONEncoder().encode(metadata).write(to: folder.appendingPathComponent("metadata.json"))
        // an empty recording (no chunk closed) is not listed
        let empty = directory.appendingPathComponent("Meetings/m0", isDirectory: true)
        try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
        try JSONEncoder().encode(
            MeetingMetadata(id: "m0", startedAt: "2026-11-04T09:00:00Z", startedHostNs: 0)
        ).write(to: empty.appendingPathComponent("metadata.json"))

        let store = NativeAppStore(supportDirectory: directory)
        XCTAssertEqual(store.meetings.map(\.id), ["m1"])
        let note = try XCTUnwrap(store.meetings.first)
        XCTAssertEqual(note.title, "Meeting")
        XCTAssertEqual(note.durationSeconds, 2.5, accuracy: 1e-9, "the call track starts 0.5 s in")
        XCTAssertEqual(note.chunkFiles().count, 4)
        let peaks = MeetingWaveform.peaks(note, bars: 10)
        XCTAssertEqual(peaks.count, 10)
        XCTAssertEqual(peaks.max() ?? 0, 1, accuracy: 1e-6)
        XCTAssertGreaterThan(peaks[5], peaks[1], "the loud second mic chunk")
    }
}
