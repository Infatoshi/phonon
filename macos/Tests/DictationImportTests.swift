import SQLite3
import XCTest

@testable import PhononBar

/// Every fixture here is synthetic and built in a temporary home.
@MainActor
final class DictationImportTests: XCTestCase {
    private var home: URL!
    private var prefs: [String: [String: Any]] = [:]
    private var installed: Set<String> = []

    override func setUpWithError() throws {
        home = FileManager.default.temporaryDirectory.appendingPathComponent(
            "phonon-import-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: support, withIntermediateDirectories: true)
        prefs = [:]
        installed = []
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: home)
    }

    private var support: URL {
        home.appendingPathComponent("Library/Application Support", isDirectory: true)
    }

    private func env(denylist: [String] = ContextPolicy.defaultDenylist) -> DictationImportEnvironment {
        let prefs = prefs
        let installed = installed
        let scratch = home.appendingPathComponent("scratch", isDirectory: true)
        try? FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        return DictationImportEnvironment(
            home: home, defaults: { prefs[$0] }, isInstalled: { installed.contains($0) },
            microphoneName: { $0 == "uid-studio" ? "Studio Mic Microphone" : nil },
            denylist: denylist, scratch: scratch)
    }

    private func write(_ text: String, to relative: String, base: URL? = nil) throws {
        let url = (base ?? support).appendingPathComponent(relative)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(text.utf8).write(to: url)
    }

    private func makeDatabase(_ relative: String, _ sql: String) throws -> URL {
        let url = support.appendingPathComponent(relative)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        var db: OpaquePointer?
        XCTAssertEqual(sqlite3_open(url.path, &db), SQLITE_OK)
        var error: UnsafeMutablePointer<CChar>?
        let status = sqlite3_exec(db, sql, nil, nil, &error)
        if status != SQLITE_OK { XCTFail(String(cString: error!)) }
        sqlite3_close(db)
        return url
    }

    private func listing(_ directory: URL) -> [String: Date] {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
        var result: [String: Date] = [:]
        for name in names {
            let attributes = try? FileManager.default.attributesOfItem(
                atPath: directory.appendingPathComponent(name).path)
            result[name] = attributes?[.modificationDate] as? Date
        }
        return result
    }

    // MARK: Wispr Flow

    private func plantWispr() throws -> URL {
        let database = try makeDatabase(
            "Wispr Flow/flow.sqlite",
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE Dictionary (id TEXT PRIMARY KEY, phrase TEXT NOT NULL, replacement TEXT,
              frequencyUsed INTEGER NOT NULL DEFAULT 0, isDeleted INTEGER NOT NULL DEFAULT 0,
              source TEXT, isSnippet INTEGER NOT NULL DEFAULT 0, isStarred INTEGER NOT NULL DEFAULT 0);
            INSERT INTO Dictionary VALUES ('1','Kubernetes',NULL,4,0,'manual',0,1);
            INSERT INTO Dictionary VALUES ('2','cooda','CUDA',2,0,'manual',0,0);
            INSERT INTO Dictionary VALUES ('3','my address','1 Example Road',0,0,'manual',1,0);
            INSERT INTO Dictionary VALUES ('4','Gone',NULL,0,1,'manual',0,0);
            CREATE TABLE History (transcriptEntityId TEXT PRIMARY KEY, asrText TEXT, formattedText TEXT,
              editedText TEXT, timestamp DATETIME, app TEXT, url TEXT, isArchived INTEGER NOT NULL DEFAULT 0);
            INSERT INTO History VALUES ('a','hello world','Hello world.',NULL,
              '2026-10-12 16:52:42.760 +00:00','com.apple.Notes',NULL,0);
            INSERT INTO History VALUES ('b','list one two','<ul><li>One</li><li>Two</li></ul>',
              'One, two, three','2026-10-12 17:00:00.000 +00:00','com.tinyspeck.slackmacgap',NULL,0);
            INSERT INTO History VALUES ('c','secret','Secret.',NULL,
              '2026-10-12 18:00:00.000 +00:00','com.1password.1password',NULL,0);
            INSERT INTO History VALUES ('d','chat','Chat.',NULL,
              '2026-10-12 18:30:00.000 +00:00','com.google.Chrome','https://discord.com/channels/1',0);
            INSERT INTO History VALUES ('e','old','Old.',NULL,'2026-10-01 10:00:00.000 +00:00',
              'com.apple.Notes',NULL,1);
            """)
        try write(
            """
            {"prefs":{"user":{"shortcuts":{"63":"ptt","49+63":"popo","53":"dismiss",
              "59+63":"lens"},"overrideAudioDeviceId":"dev-2",
              "rankedAudioDevices":[{"deviceId":"dev-1","name":"Built-in mic (recommended)"},
                {"deviceId":"dev-2","name":"Studio Mic  (USB)"}]}}}
            """, to: "Wispr Flow/config.json")
        return database
    }

    func testWisprReadsDictionarySnippetsHistoryShortcutAndMic() throws {
        _ = try plantWispr()
        let findings = WisprFlowReader.read(env())

        XCTAssertEqual(findings.terms.map(\.phrase), ["cooda", "Kubernetes"])
        XCTAssertEqual(findings.terms.first { $0.phrase == "cooda" }?.replacement, "CUDA")
        let kubernetes = try XCTUnwrap(findings.terms.first { $0.phrase == "Kubernetes" })
        XCTAssertTrue(kubernetes.starred)
        XCTAssertEqual(kubernetes.usageCount, 4)
        XCTAssertEqual(kubernetes.source, "wispr-flow")
        XCTAssertEqual(findings.macros, [PhononMacro(trigger: "my address", expansion: "1 Example Road")])

        XCTAssertEqual(findings.dictations.map(\.id), ["a", "b"], "archived and private rows stay out")
        XCTAssertEqual(findings.skippedPrivate, 2, "1Password always, Discord by default")
        let edited = findings.dictations[1]
        XCTAssertEqual(edited.refinedText, "OneTwo", "list markup is stripped")
        XCTAssertEqual(edited.editedText, "One, two, three")
        XCTAssertEqual(edited.finalText, "One, two, three")
        XCTAssertEqual(edited.rawText, "list one two")
        XCTAssertEqual(
            findings.dictations[0].timestamp.timeIntervalSince1970, 1_791_823_962.76,
            accuracy: 0.001)

        XCTAssertEqual(findings.shortcutMode, "fn")
        XCTAssertEqual(findings.microphone, "Studio Mic")
        XCTAssertEqual(findings.wordCount, 1)
        XCTAssertEqual(findings.replacementCount, 1)
    }

    func testWisprDatabaseIsReadThroughACopyAndLeftUntouched() throws {
        let database = try plantWispr()
        let folder = database.deletingLastPathComponent()
        let before = listing(folder)
        let bytes = try Data(contentsOf: database)

        _ = WisprFlowReader.read(env())

        XCTAssertEqual(listing(folder), before, "no new -wal/-shm, no touched file")
        XCTAssertEqual(try Data(contentsOf: database), bytes)
        let scratch = home.appendingPathComponent("scratch")
        XCTAssertEqual(
            try FileManager.default.contentsOfDirectory(atPath: scratch.path), [],
            "the copy is removed")
    }

    func testWisprMicChoiceOfDefaultImportsNoMic() {
        XCTAssertNil(
            WisprFlowReader.microphone(config: [
                "prefs": [
                    "user": [
                        "overrideAudioDeviceId": "x",
                        "rankedAudioDevices": [["deviceId": "x", "name": "Auto-detect (AirPods)"]],
                    ]
                ]
            ]))
    }

    // MARK: Aqua Voice

    func testAquaReadsDictionaryReplacementsHistoryAndHotkey() throws {
        try write(
            """
            {"dictionary":["Triton"," ","PyTorch"],
             "replacements":[{"from":"jason","to":"JSON","preserveCase":true}],
             "history":[{"kind":"transcription","rawText":"run the triton kernel",
               "content":"Run the Triton kernel.","timestamp":"2026-07-27T15:54:07.373Z",
               "sessionId":42},
               {"kind":"transcription","rawText":"","content":"","timestamp":"2026-07-27T15:55:00Z"}],
             "hotkeys":[{"keys":"Alt","action":"activate"},{"keys":"Escape","action":"cancel"}],
             "useNativeAudio":false,
             "jsMicrophone":{"id":"1","name":"Default - MacBook Pro Microphone (Built-in)"}}
            """, to: "Aqua Voice/settings.json")
        let findings = AquaVoiceReader.read(env())
        XCTAssertEqual(findings.terms.map(\.phrase), ["Triton", "PyTorch", "jason"])
        XCTAssertEqual(findings.terms.last?.replacement, "JSON")
        XCTAssertEqual(findings.dictations.count, 1, "empty rows are skipped")
        XCTAssertEqual(findings.dictations[0].id, "42")
        XCTAssertEqual(findings.dictations[0].finalText, "Run the Triton kernel.")
        XCTAssertEqual(findings.shortcutMode, "right_option")
        XCTAssertNil(findings.microphone, "a default device is not a choice")
    }

    // MARK: superwhisper

    func testSuperwhisperFollowsTheAppFolderPreference() throws {
        let documents = home.appendingPathComponent("Elsewhere", isDirectory: true)
        prefs["com.superduper.superwhisper"] = [
            "appFolderDirectory": documents.path,
            "KeyboardShortcuts_toggleRecording": #"{"carbonKeyCode":63,"carbonModifiers":0}"#,
        ]
        try write(
            """
            {"vocabulary":["Zorblax","quill"],
             "replacements":[{"id":"1","original":"super whisper","with":"superwhisper"}]}
            """, to: "superwhisper/settings/settings.json", base: documents)
        try write(
            """
            {"rawResult":"hello there","result":"hello there","llmResult":"Hello there.",
             "datetime":"2026-10-31T08:19:10","recordingDevice":"Desk USB Mic",
             "promptContext":{"applicationContext":{"name":"Notes"}}}
            """, to: "superwhisper/recordings/1793000000/meta.json", base: documents)
        try write(
            #"{"rawResult":"","result":"","datetime":"2026-10-31T08:22:00","recordingDevice":""}"#,
            to: "superwhisper/recordings/1793000100/meta.json", base: documents)

        let findings = SuperwhisperReader.read(env())
        XCTAssertEqual(findings.terms.map(\.phrase), ["Zorblax", "quill", "super whisper"])
        XCTAssertEqual(findings.terms.last?.replacement, "superwhisper")
        XCTAssertEqual(findings.dictations.count, 1)
        XCTAssertEqual(findings.dictations[0].finalText, "Hello there.")
        XCTAssertEqual(findings.dictations[0].app, "Notes")
        XCTAssertEqual(findings.dictations[0].timestamp, Date(timeIntervalSince1970: 1_793_000_000))
        XCTAssertEqual(findings.microphone, "Desk USB Mic")
        XCTAssertEqual(findings.shortcutMode, "fn")
    }

    // MARK: VoiceInk

    func testVoiceInkReadsSwiftDataStoresAndPreferences() throws {
        _ = try makeDatabase(
            "com.prakashjoshipax.VoiceInk/default.store",
            """
            CREATE TABLE ZVOCABULARYWORD (Z_PK INTEGER PRIMARY KEY, ZDATEADDED TIMESTAMP, ZWORD VARCHAR);
            INSERT INTO ZVOCABULARYWORD VALUES (1, 800000000, 'Phonon');
            CREATE TABLE ZWORDREPLACEMENT (Z_PK INTEGER PRIMARY KEY, ZISENABLED INTEGER,
              ZORIGINALTEXT VARCHAR, ZREPLACEMENTTEXT VARCHAR);
            INSERT INTO ZWORDREPLACEMENT VALUES (1, 1, 'teh, hte', 'the');
            INSERT INTO ZWORDREPLACEMENT VALUES (2, 0, 'off', 'disabled');
            CREATE TABLE ZTRANSCRIPTION (Z_PK INTEGER PRIMARY KEY, ZTIMESTAMP TIMESTAMP,
              ZTEXT VARCHAR, ZENHANCEDTEXT VARCHAR);
            INSERT INTO ZTRANSCRIPTION VALUES (1, 800000000, 'raw words', 'Raw words.');
            INSERT INTO ZTRANSCRIPTION VALUES (2, 800000100, 'only raw', NULL);
            """)
        _ = try makeDatabase(
            "com.prakashjoshipax.VoiceInk/dictionary.store",
            """
            CREATE TABLE ZVOCABULARYWORD (Z_PK INTEGER PRIMARY KEY, ZDATEADDED TIMESTAMP, ZWORD VARCHAR);
            INSERT INTO ZVOCABULARYWORD VALUES (1, 800000000, 'Phonon');
            INSERT INTO ZVOCABULARYWORD VALUES (2, 800000000, 'Metal');
            CREATE TABLE ZTRANSCRIPTION (Z_PK INTEGER PRIMARY KEY, ZTIMESTAMP TIMESTAMP,
              ZTEXT VARCHAR, ZENHANCEDTEXT VARCHAR);
            INSERT INTO ZTRANSCRIPTION VALUES (9, 800000000, 'not history', NULL);
            """)
        prefs["com.prakashjoshipax.VoiceInk"] = [
            "Shortcut_primaryRecording": #"{"keyCode":49,"modifierFlags":262144}"#,
            "selectedAudioDeviceUID": "uid-studio",
        ]
        let findings = VoiceInkReader.read(env())
        XCTAssertEqual(
            findings.terms.map { "\($0.phrase)>\($0.replacement ?? "")" },
            ["Phonon>", "Metal>", "teh>the", "hte>the"], "dictionary.store first, no duplicates")
        XCTAssertEqual(findings.dictations.map(\.finalText), ["Raw words.", "only raw"])
        XCTAssertEqual(
            findings.dictations[0].timestamp, Date(timeIntervalSinceReferenceDate: 800_000_000))
        XCTAssertEqual(findings.shortcutMode, "control_space")
        XCTAssertEqual(findings.microphone, "Studio Mic Microphone")
    }

    // MARK: MacWhisper and FluidVoice

    func testMacWhisperReadsLiveDictationsOnly() throws {
        _ = try makeDatabase(
            "MacWhisper/Database/main.sqlite",
            """
            CREATE TABLE dictation (id BLOB PRIMARY KEY NOT NULL, dateCreated DATETIME NOT NULL,
              transcribedText TEXT, processedText TEXT, targetAppBundleID TEXT, dateDeleted DOUBLE);
            INSERT INTO dictation VALUES (x'01', '2026-11-05 21:05:42.000', 'hi there', 'Hi there.',
              'com.apple.TextEdit', NULL);
            INSERT INTO dictation VALUES (x'02', '2026-11-05 21:06:00.000', 'gone', NULL, NULL, 1.0);
            """)
        let findings = MacWhisperReader.read(env())
        XCTAssertEqual(findings.dictations.map(\.finalText), ["Hi there."])
        XCTAssertEqual(findings.dictations[0].id, "01")
        XCTAssertEqual(findings.dictations[0].app, "com.apple.TextEdit")
    }

    func testFluidVoiceAliasesBecomeSpokenForms() throws {
        try write(
            """
            {"alpha":2.8,"terms":[{"text":"Zeta 9","weight":14,"aliases":["zeta nine"]},
              {"text":"Q7","aliases":[]}]}
            """, to: "FluidVoice/parakeet_custom_vocabulary.json")
        prefs["com.FluidApp.app"] = [
            "TranscriptionHistoryEntries": Data(
                #"[{"id":"x","rawText":"um hello","processedText":"Hello.","timestamp":800000000.5,"appName":"Terminal"}]"#
                    .utf8),
            "HotkeyShortcutKey": Data(#"{"keyCode":59,"modifierFlagsRawValue":0}"#.utf8),
        ]
        let findings = FluidVoiceReader.read(env())
        XCTAssertEqual(findings.terms.map(\.phrase), ["Zeta 9", "Q7"])
        XCTAssertEqual(findings.terms[0].spokenForms, ["zeta nine"])
        XCTAssertEqual(findings.dictations.map(\.finalText), ["Hello."])
        XCTAssertNil(findings.shortcutMode, "Left Control alone is not a Phonon key")
    }

    // MARK: Detection and helpers

    func testDetectFindsStoresBundlesAndPreferencesWithoutReadingDocuments() throws {
        XCTAssertEqual(DictationImportScanner.detect(env()), [])
        try FileManager.default.createDirectory(
            at: support.appendingPathComponent("Wispr Flow"), withIntermediateDirectories: true)
        installed = ["com.goodsnooze.MacWhisper"]
        prefs["com.FluidApp.app"] = ["HotkeyShortcutKey": Data()]
        XCTAssertEqual(DictationImportScanner.detect(env()), [.wisprFlow, .macWhisper, .fluidVoice])
    }

    func testMissingStoresScanEmpty() {
        for app in DictationApp.allCases {
            XCTAssertTrue(DictationImportScanner.scan(app, env: env()).isEmpty, app.name)
        }
    }

    func testShortcutModesAndMicrophoneLabels() {
        let fn = CompetitorChord.fromKeyCodes([63])
        let option = CompetitorChord.fromKeyCodes([61])
        let controlSpace = CompetitorChord.fromKeyCodes([59, 49])
        XCTAssertEqual(DictationImportScanner.shortcutMode(for: [fn]), "fn")
        XCTAssertEqual(DictationImportScanner.shortcutMode(for: [fn, controlSpace]), "fn_and_control_space")
        XCTAssertEqual(DictationImportScanner.shortcutMode(for: [option]), "right_option")
        XCTAssertEqual(DictationImportScanner.shortcutMode(for: [option, controlSpace]), "both")
        XCTAssertNil(DictationImportScanner.shortcutMode(for: [CompetitorChord.fromKeyCodes([55, 9])]))

        XCTAssertEqual(DictationImportScanner.microphoneFragment("Studio Mic  (USB)"), "Studio Mic")
        XCTAssertEqual(
            DictationImportScanner.microphoneFragment("Desk Pro (046d:0ab7)"), "Desk Pro")
        XCTAssertNil(DictationImportScanner.microphoneFragment("Built-in mic (recommended)"))
        XCTAssertNil(DictationImportScanner.microphoneFragment("Default - MacBook Pro Microphone"))
        XCTAssertNil(DictationImportScanner.microphoneFragment("  "))
    }

    // MARK: Applying

    func testApplyMergesWritesHistoryAndSettingsAndIsIdempotent() throws {
        let phonon = home.appendingPathComponent("Phonon", isDirectory: true)
        let store = NativeAppStore(
            supportDirectory: phonon, mirrorDirectory: home.appendingPathComponent("mirror"))
        store.upsertDictionary(originalID: nil, phrase: "Kubernetes", replacement: nil, spokenForms: [])
        let date = Date(timeIntervalSince1970: 1_790_000_000)
        var wispr = DictationAppFindings(app: .wisprFlow)
        wispr.terms = [
            NativeDictionaryEntry(phrase: "kubernetes", spokenForms: ["cube netties"], source: "wispr-flow"),
            NativeDictionaryEntry(phrase: "cooda", replacement: "CUDA", source: "wispr-flow"),
        ]
        wispr.macros = [PhononMacro(trigger: "my address", expansion: "1 Example Road")]
        wispr.dictations = [
            ImportedDictation(
                id: "b", source: "wispr-flow", timestamp: date.addingTimeInterval(60),
                rawText: "two", refinedText: "Two.", editedText: "Two!"),
            ImportedDictation(
                id: "a", source: "wispr-flow", timestamp: date, rawText: "one  more",
                refinedText: "One\nmore."),
        ]
        wispr.shortcutMode = "right_option"
        wispr.microphone = "Studio Mic"
        var aqua = DictationAppFindings(app: .aquaVoice)
        aqua.terms = [NativeDictionaryEntry(phrase: "Cooda", replacement: "CUDA", source: "aqua-voice")]
        aqua.shortcutMode = "fn"
        aqua.microphone = "AirPods"

        let summary = store.applyDictationImport([wispr, aqua])
        XCTAssertEqual(summary.addedTerms, 1, "case-insensitive duplicates merge")
        XCTAssertEqual(summary.addedMacros, 1)
        XCTAssertEqual(summary.savedDictations, 2)
        XCTAssertEqual(summary.shortcutMode, "right_option", "first app in table order wins")
        XCTAssertEqual(summary.microphone, "Studio Mic")
        XCTAssertEqual(store.settings.shortcutMode, "right_option")
        XCTAssertEqual(store.settings.microphonePriority, ["Studio Mic"])
        let kubernetes = try XCTUnwrap(store.dictionaryEntries.first { $0.phrase == "Kubernetes" })
        XCTAssertEqual(kubernetes.spokenForms, ["cube netties"], "the existing entry gains the form")
        XCTAssertEqual(kubernetes.source, "phonon-app")
        XCTAssertEqual(store.dictionaryEntries.first { $0.phrase == "cooda" }?.source, "wispr-flow")
        XCTAssertEqual(store.macros.map(\.trigger), ["my address"])

        let extract = try String(
            contentsOf: DictationImportFiles.extractURL(.wisprFlow, support: phonon), encoding: .utf8)
        XCTAssertEqual(extract, "One more.\nTwo!\n", "oldest first, one line each, final text")
        let lines = try String(
            contentsOf: DictationImportFiles.historyURL(.wisprFlow, support: phonon), encoding: .utf8)
            .split(separator: "\n")
        XCTAssertEqual(lines.count, 2)
        let first = try JSONSerialization.jsonObject(with: Data(lines[0].utf8)) as? [String: Any]
        XCTAssertEqual(first?["raw_text"] as? String, "one  more")
        XCTAssertEqual(first?["refined_text"] as? String, "One\nmore.")
        XCTAssertEqual(first?["timestamp"] as? String, "2026-09-21T14:13:20Z")

        // A second import adds nothing and leaves the ranking and files as they were.
        store.updateSettings { $0.microphonePriority = ["MacBook"] }
        let again = store.applyDictationImport([wispr, aqua])
        XCTAssertEqual(again.addedTerms, 0)
        XCTAssertEqual(again.addedMacros, 0)
        XCTAssertNil(again.shortcutMode)
        XCTAssertNil(again.microphone)
        XCTAssertEqual(store.settings.microphonePriority, ["MacBook"])
        XCTAssertEqual(
            try String(
                contentsOf: DictationImportFiles.extractURL(.wisprFlow, support: phonon),
                encoding: .utf8), extract)
    }

    func testDoneLinesAndAppList() {
        XCTAssertEqual(DictationImportView.list(["A", "B", "C"]), "A, B and C")
        XCTAssertEqual(DictationImportView.list(["A"]), "A")
        XCTAssertEqual(
            DictationImportView.doneLines(DictationImportSummary()),
            ["Everything found was already in Phonon"])
        XCTAssertEqual(
            DictationImportView.doneLines(
                DictationImportSummary(addedTerms: 2, shortcutMode: "control_space")),
            ["2 new dictionary entries", "Shortcut set to Control Space"])
    }
}
