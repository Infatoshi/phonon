import XCTest

@testable import PhononBar

@MainActor
final class AppDataTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "phonon-app-data-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    func testSettingsMigrationAndSaveAddsNativeControls() throws {
        try Data(
            #"{"schema_version":1,"streaming":true,"local_history":true,"screen_context":false,"microphone_priority":["USB Microphone","MacBook"]}"#.utf8
        ).write(to: directory.appendingPathComponent("settings.json"))

        let store = NativeAppStore(supportDirectory: directory)
        XCTAssertTrue(store.settings.instantMic)
        XCTAssertEqual(store.settings.shortcutMode, "both")

        store.updateSettings {
            $0.instantMic = false
            $0.shortcutMode = "right_option"
        }
        let saved = try JSONDecoder().decode(
            NativeSettings.self,
            from: Data(contentsOf: directory.appendingPathComponent("settings.json")))
        XCTAssertFalse(saved.instantMic)
        XCTAssertEqual(saved.shortcutMode, "right_option")
        XCTAssertFalse(saved.screenContext)
    }

    func testDictionaryAddEditAndRemoveRoundTrips() throws {
        try Data(#"{"schema_version":1,"updated_at_unix_ms":1,"entries":[]}"#.utf8)
            .write(to: directory.appendingPathComponent("dictionary.json"))
        let store = NativeAppStore(supportDirectory: directory)

        store.upsertDictionary(
            originalID: nil, phrase: "black well", replacement: "Blackwell",
            spokenForms: ["blackwell", "black well"])
        let entry = try XCTUnwrap(store.dictionaryEntries.first)
        XCTAssertEqual(entry.replacement, "Blackwell")
        XCTAssertEqual(entry.spokenForms, ["black well", "blackwell"])

        store.removeDictionary(id: entry.id)
        XCTAssertTrue(store.dictionaryEntries.isEmpty)
    }

    func testSavingIntendedTranscriptPreservesUnmodeledMetadata() throws {
        let recording = directory.appendingPathComponent("Corpus/test", isDirectory: true)
        try FileManager.default.createDirectory(at: recording, withIntermediateDirectories: true)
        let metadata = """
            {
              "schema_version": 1,
              "id": "test",
              "created_at_unix_ms": 1000,
              "source": "bar",
              "audio_file": "audio.wav",
              "raw_transcript": "black well",
              "final_transcript": "Blackwell",
              "dictionary_corrections": [{"from":"black well","to":"Blackwell"}],
              "screen_context_terms": []
            }
            """
        try Data(metadata.utf8).write(to: recording.appendingPathComponent("metadata.json"))
        let store = NativeAppStore(supportDirectory: directory)

        store.saveIntendedTranscript(itemID: "test", text: "Use Blackwell.")

        let data = try Data(contentsOf: recording.appendingPathComponent("metadata.json"))
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["intended_transcript"] as? String, "Use Blackwell.")
        XCTAssertEqual((object["dictionary_corrections"] as? [[String: String]])?.count, 1)
    }

    func testNoSpeechCorpusItemsStayOffUserHistory() throws {
        let recording = directory.appendingPathComponent("Corpus/silent", isDirectory: true)
        try FileManager.default.createDirectory(at: recording, withIntermediateDirectories: true)
        try Data(
            #"{"schema_version":1,"id":"silent","created_at_unix_ms":1000,"source":"bar","audio_file":"audio.wav","speech_detected":false,"raw_transcript":"","final_transcript":""}"#.utf8
        ).write(to: recording.appendingPathComponent("metadata.json"))

        let store = NativeAppStore(supportDirectory: directory)

        XCTAssertTrue(store.history.isEmpty)
        XCTAssertEqual(store.usage.recordings, 0)
    }

    func testUsageStatsIncludeSpeedDailyActivityAndDictionaryRepairs() throws {
        let calendar = Calendar.current
        let now = Date()
        let previousDay = try XCTUnwrap(calendar.date(byAdding: .day, value: -2, to: now))
        let fixtures: [(String, Date, Int, UInt64, UInt64)] = [
            ("today", now, 120, 60_000, 2),
            ("older", previousDay, 30, 30_000, 1),
        ]

        for (id, date, wordCount, duration, corrections) in fixtures {
            let recording = directory.appendingPathComponent("Corpus/\(id)", isDirectory: true)
            try FileManager.default.createDirectory(
                at: recording, withIntermediateDirectories: true)
            let words = Array(repeating: "word", count: wordCount).joined(separator: " ")
            let metadata: [String: Any] = [
                "schema_version": 1,
                "id": id,
                "created_at_unix_ms": UInt64(date.timeIntervalSince1970 * 1_000),
                "source": "bar",
                "audio_file": "audio.wav",
                "audio_duration_ms": duration,
                "speech_detected": true,
                "raw_transcript": words,
                "final_transcript": words,
                "dictionary_corrections": [
                    ["from": "raw", "to": "term", "count": corrections]
                ],
            ]
            let data = try JSONSerialization.data(withJSONObject: metadata)
            try data.write(to: recording.appendingPathComponent("metadata.json"))
        }

        let store = NativeAppStore(supportDirectory: directory)

        XCTAssertEqual(store.usage.recordings, 2)
        XCTAssertEqual(store.usage.words, 150)
        XCTAssertEqual(store.usage.speakingMilliseconds, 90_000)
        XCTAssertEqual(store.usage.wordsPerMinute, 100)
        XCTAssertEqual(store.usage.averageWordsPerRecording, 75)
        XCTAssertEqual(store.usage.dictionaryFixes, 3)
        XCTAssertEqual(store.usage.wordsToday, 120)
        XCTAssertEqual(store.usage.recordingsToday, 1)
        XCTAssertEqual(store.usage.activeDays, 2)
    }

    func testEveryPermissionHasADirectSystemSettingsURL() {
        XCTAssertEqual(
            Set(PrivacyPane.allCases.map(\.rawValue)),
            Set([
                "Privacy_Microphone",
                "Privacy_Accessibility",
                "Privacy_ListenEvent",
                "Privacy_ScreenCapture",
            ]))
        for pane in PrivacyPane.allCases {
            XCTAssertEqual(pane.settingsURL.scheme, "x-apple.systempreferences")
            XCTAssertTrue(pane.settingsURL.absoluteString.contains(pane.rawValue))
        }
    }

    func testMicrophonePermissionPresentationMatchesTCCState() {
        let store = NativeAppStore(supportDirectory: directory)
        switch store.microphoneAuthorizationStatus {
        case .notDetermined:
            XCTAssertEqual(store.microphoneStatusText, "Not requested")
            XCTAssertEqual(store.microphoneActionTitle, "Request Access")
        case .authorized:
            XCTAssertEqual(store.microphoneStatusText, "Granted")
            XCTAssertEqual(store.microphoneActionTitle, "Settings")
        case .denied:
            XCTAssertEqual(store.microphoneStatusText, "Denied")
            XCTAssertEqual(store.microphoneActionTitle, "Settings")
        case .restricted:
            XCTAssertEqual(store.microphoneStatusText, "Restricted")
            XCTAssertEqual(store.microphoneActionTitle, "Settings")
        @unknown default:
            XCTAssertEqual(store.microphoneStatusText, "Needs access")
        }
    }

    func testPermissionRefreshRechecksInputMonitoringAndNotifiesServices() {
        let store = NativeAppStore(supportDirectory: directory)
        var callbackCount = 0
        store.onPermissionsRefresh = { callbackCount += 1 }

        store.refreshPermissions()

        XCTAssertEqual(callbackCount, 1)
        XCTAssertEqual(store.inputMonitoringAvailable, CGPreflightListenEventAccess())
    }
}
