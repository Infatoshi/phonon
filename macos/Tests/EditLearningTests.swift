import XCTest

@testable import PhononBar

final class EditObservationTrackerTests: XCTestCase {
    private func start(
        before: String, selected: String = "", after: String, pasted: String,
        landed: String? = nil
    ) throws -> EditObservationTracker {
        let selection = NSRange(
            location: (before as NSString).length, length: (selected as NSString).length)
        let contentsBefore = before + selected + after
        let contentsAfter = before + (landed ?? pasted) + after
        let result = EditObservationTracker.start(
            pasted: pasted, contentsBefore: contentsBefore, selectionBefore: selection,
            contentsAfter: contentsAfter, now: 0)
        return try result.get()
    }

    func testPasteThenRetypeEndsAfterQuietAndReportsTheKeptRegion() throws {
        var tracker = try start(before: "Hi team, ", after: "", pasted: "we use coo blast")
        XCTAssertEqual(tracker.baseline, "we use coo blast")
        XCTAssertNil(tracker.observe(contents: "Hi team, we use coo blast", now: 0.4))
        // Selecting the words and retyping goes through intermediate states.
        XCTAssertNil(tracker.observe(contents: "Hi team, we use ", now: 1.0))
        XCTAssertNil(tracker.observe(contents: "Hi team, we use cuB", now: 1.4))
        XCTAssertNil(tracker.observe(contents: "Hi team, we use cuBLAS", now: 2.0))
        XCTAssertNil(tracker.observe(contents: "Hi team, we use cuBLAS", now: 6.9))
        XCTAssertEqual(
            tracker.observe(contents: "Hi team, we use cuBLAS", now: 7.1), .quietAfterEdit)
        let result = tracker.finish(reason: .quietAfterEdit, now: 7.1)
        XCTAssertEqual(result.pasted, "we use coo blast")
        XCTAssertEqual(result.edited, "we use cuBLAS")
        XCTAssertNil(result.editedWithinThreshold)
        XCTAssertEqual(result.observedMs, 7_100)
        XCTAssertEqual(result.changes, 3)
    }

    func testTypingFarFromTheDictationInTheSameFieldKeepsTheRegion() throws {
        let preamble = "Long preamble paragraph that the user wrote earlier in this email. "
        var tracker = try start(
            before: preamble, after: "\n\nThanks,\nSam", pasted: "ship it on friday")
        let edited = preamble.replacingOccurrences(of: "Long", with: "Short")
            + "ship it on Friday" + "\n\nThanks,\nSam"
        XCTAssertNil(tracker.observe(contents: edited, now: 1))
        XCTAssertEqual(tracker.latest, "ship it on Friday")
        // Typing below the signature does not touch the region either.
        XCTAssertNil(tracker.observe(contents: edited + "\nPS: call me", now: 2))
        XCTAssertEqual(tracker.latest, "ship it on Friday")
    }

    func testUnrelatedTypingInAnotherFieldLeavesTheRegionUnchanged() throws {
        var tracker = try start(before: "", after: "", pasted: "see you at noon")
        for second in stride(from: 1.0, to: 60, by: 1) {
            XCTAssertNil(tracker.observe(contents: "see you at noon", now: second))
        }
        XCTAssertEqual(tracker.observe(contents: "see you at noon", now: 60.2), .windowElapsed)
        let result = tracker.finish(reason: .windowElapsed, now: 60.2)
        XCTAssertEqual(result.edited, result.pasted)
        XCTAssertEqual(result.changes, 0)
    }

    func testTypingStraightAfterTheDictationGrowsTheRegion() throws {
        // At the end of a field there is no after-anchor: continued typing
        // joins the region and the engine classifies it as appended.
        var tracker = try start(before: "", after: "", pasted: "Ship it today.")
        XCTAssertNil(tracker.observe(contents: "Ship it today. Also ping Bob.", now: 1))
        XCTAssertEqual(tracker.latest, "Ship it today. Also ping Bob.")
    }

    func testUndoEmptiesTheRegionAndReportsNothingKept() throws {
        var tracker = try start(before: "Re: ", after: "", pasted: "sounds good")
        tracker.noteKeystroke(.undo, at: 1)
        XCTAssertEqual(tracker.observe(contents: "Re: ", now: 1.1), .regionEmptied)
        let result = tracker.finish(reason: .regionEmptied, now: 1.1)
        XCTAssertEqual(result.edited, "")
        XCTAssertEqual(result.lastKeystroke, .undo)
        XCTAssertEqual(result.lastKeystrokeAgeMs, 100)
    }

    func testEditThenUndoBackToThePasteIsUnchanged() throws {
        var tracker = try start(before: "", after: "", pasted: "use coo blast")
        XCTAssertNil(tracker.observe(contents: "use cuBLAS", now: 1))
        tracker.noteKeystroke(.undo, at: 2)
        XCTAssertNil(tracker.observe(contents: "use coo blast", now: 2.1))
        // Back to the paste: no quiet-after-edit ending, only the window.
        XCTAssertNil(tracker.observe(contents: "use coo blast", now: 20))
        XCTAssertEqual(tracker.finish(reason: .windowElapsed, now: 60).edited, "use coo blast")
    }

    func testReturnThatEmptiesTheFieldIsASentMessageKeepingTheEdit() throws {
        var tracker = try start(before: "", after: "", pasted: "ping coo blast team")
        XCTAssertNil(tracker.observe(contents: "ping cuBLAS team", now: 1))
        tracker.noteKeystroke(.returnKey, at: 1.5)
        XCTAssertEqual(tracker.observe(contents: "", now: 1.6), .messageSent)
        let result = tracker.finish(reason: .messageSent, now: 1.6)
        XCTAssertEqual(result.edited, "ping cuBLAS team")
        XCTAssertEqual(result.lastKeystroke, .returnKey)
    }

    func testSelectAllDeleteClearsTheFieldAndKeepsNothing() throws {
        var tracker = try start(before: "", after: "", pasted: "never mind")
        let selectAll = KeystrokeCategory.classify(
            keyCode: 0, command: true, option: false, shift: false)
        let delete = KeystrokeCategory.classify(
            keyCode: 51, command: false, option: false, shift: false, previous: selectAll)
        XCTAssertEqual(delete, .selectAllDelete)
        tracker.noteKeystroke(delete, at: 1)
        XCTAssertEqual(tracker.observe(contents: "", now: 1.2), .textboxEmptied)
        XCTAssertEqual(tracker.finish(reason: .textboxEmptied, now: 1.2).edited, "")
    }

    func testAFieldEmptiedWithoutADeletingKeyKeepsTheLastText() throws {
        // A click on Send empties the field with no key at all.
        var tracker = try start(before: "", after: "", pasted: "ship coo blast today")
        XCTAssertNil(tracker.observe(contents: "ship cuBLAS today", now: 1))
        XCTAssertEqual(tracker.observe(contents: "", now: 2.5), .textboxEmptied)
        XCTAssertEqual(tracker.finish(reason: .textboxEmptied, now: 2.5).edited, "ship cuBLAS today")
        // Holding backspace down to nothing throws it away.
        var deleted = try start(before: "", after: "", pasted: "never mind this")
        XCTAssertNil(deleted.observe(contents: "never mind", now: 1))
        deleted.noteKeystroke(.backspace, at: 1.8)
        XCTAssertEqual(deleted.observe(contents: "", now: 2), .textboxEmptied)
        XCTAssertEqual(deleted.finish(reason: .textboxEmptied, now: 2).edited, "")
    }

    func testReturnInAMultilineFieldEndsWithTheLineAsKept() throws {
        var tracker = try start(before: "", after: "", pasted: "first point")
        tracker.noteKeystroke(.returnKey, at: 1)
        XCTAssertEqual(tracker.observe(contents: "first point\n", now: 1.1), .trailingNewline)
        XCTAssertEqual(tracker.finish(reason: .trailingNewline, now: 1.1).edited, "first point")
    }

    func testAReplacedFieldIsAnAnchorMismatch() throws {
        var tracker = try start(
            before: "To: Bob\n", after: "\n-- sent from my Mac", pasted: "hello there")
        XCTAssertEqual(
            tracker.observe(contents: "a completely different draft", now: 1), .anchorMismatch)
        var intact = try start(before: "To: Bob\n", after: "\nbye", pasted: "hello there")
        XCTAssertEqual(
            intact.observe(contents: "hello there and more", now: 1), .anchorMismatchIntact)
    }

    func testRewriteKeepsTheLastStateWithinTheThreshold() throws {
        var tracker = try start(
            before: "", after: "", pasted: "please ask the coo blast team about gemm numbers")
        XCTAssertNil(
            tracker.observe(
                contents: "please ask the cuBLAS team about gemm numbers", now: 1))
        XCTAssertNil(
            tracker.observe(
                contents: "never mind, I will check the dashboard myself", now: 2))
        let result = tracker.finish(reason: .windowElapsed, now: 60)
        XCTAssertEqual(result.edited, "never mind, I will check the dashboard myself")
        XCTAssertEqual(
            result.editedWithinThreshold, "please ask the cuBLAS team about gemm numbers")
    }

    func testStartRejectsLongFieldsAndAPasteThatHasNotLanded() {
        let long = String(repeating: "word ", count: 5_000)
        let tooLong = EditObservationTracker.start(
            pasted: "hi", contentsBefore: long, selectionBefore: NSRange(location: 0, length: 0),
            contentsAfter: "hi" + long, now: 0)
        XCTAssertEqual(tooLong.failureReason, .textboxTooLong)
        let notLanded = EditObservationTracker.start(
            pasted: "hello world", contentsBefore: "draft",
            selectionBefore: NSRange(location: 5, length: 0), contentsAfter: "draft", now: 0)
        XCTAssertEqual(notLanded.failureReason, .notFound)
    }

    func testAPasteOverASelectionReplacesIt() throws {
        let tracker = try start(before: "Use ", selected: "xyz", after: " now", pasted: "cuBLAS")
        XCTAssertEqual(tracker.baseline, "cuBLAS")
        XCTAssertEqual(tracker.beforeAnchor, "Use ")
        XCTAssertEqual(tracker.afterAnchor, " now")
    }

    func testUnreadableFieldEndsAfterThreeFailedReads() throws {
        var tracker = try start(before: "", after: "", pasted: "hello")
        XCTAssertNil(tracker.observe(contents: nil, now: 1))
        XCTAssertNil(tracker.observe(contents: nil, now: 2))
        XCTAssertEqual(tracker.observe(contents: nil, now: 3), .unreadable)
    }

    func testEngineCommandCarriesTheObservation() {
        let result = ObservationResult(
            pasted: "a", edited: "b", editedWithinThreshold: nil, endReason: .quietAfterEdit,
            lastKeystroke: .backspace, lastKeystrokeAgeMs: 300, observedMs: 5_100, changes: 1)
        let command = result.engineCommand(id: "p7", record: false)
        XCTAssertEqual(command["cmd"] as? String, "edit_observed")
        XCTAssertEqual(command["end_reason"] as? String, "quiet_after_edit")
        XCTAssertEqual(command["last_keystroke"] as? String, "backspace")
        XCTAssertEqual(command["record"] as? Bool, false)
        XCTAssertNil(command["edited_within_threshold"])
    }
}

/// Any field text and any selection AX hands back must be safe: an
/// out-of-range NSString call would crash the whole app.
final class FieldTextFuzzTests: XCTestCase {
    func testArbitraryFieldsAndSelectionsNeverCrash() {
        let pool = [
            "the", "cuBLAS", "\u{1F44D}", "e\u{301}", "東京", "\n", "  ", "Zoë", "", "x", "!",
            "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}", "coo blast", ".", "\t",
        ]
        var seed: UInt64 = 0x5eed_1234_abcd_ef01
        func next(_ bound: Int) -> Int {
            seed ^= seed << 13
            seed ^= seed >> 7
            seed ^= seed << 17
            return Int(seed % UInt64(max(1, bound)))
        }
        func text() -> String { (0..<next(10)).map { _ in pool[next(pool.count)] }.joined() }
        var starts = 0
        for _ in 0..<1_500 {
            let before = text()
            let length = (before as NSString).length
            let selection = NSRange(location: next(length + 3) - 1, length: next(length + 3) - 1)
            _ = TextboxContext.around(value: before, selection: selection)
            let pasted = "word " + text()
            let ns = before as NSString
            // AX puts the cursor between composed characters, never inside one.
            func boundary(_ offset: Int) -> Int {
                guard offset < length else { return length }
                return ns.rangeOfComposedCharacterSequence(at: offset).location
            }
            let location = boundary(next(length + 1))
            let cut = boundary(min(length, location + next(3))) - location
            let landed =
                ns.substring(to: location) + pasted + ns.substring(from: location + cut)
            let started = EditObservationTracker.start(
                pasted: pasted, contentsBefore: before,
                selectionBefore: NSRange(location: location, length: cut),
                contentsAfter: landed, now: 0)
            guard case var .success(tracker) = started else {
                XCTFail("a clean landing must start a watch")
                continue
            }
            starts += 1
            for step in 1...4 {
                let contents: String? =
                    switch next(4) {
                    case 0: nil
                    case 1: text()
                    case 2: landed + text()
                    default: text() + landed
                    }
                _ = tracker.observe(contents: contents, now: Double(step))
            }
            _ = tracker.finish(reason: .windowElapsed, now: 9)
        }
        XCTAssertEqual(starts, 1_500)
    }
}

/// The engine's replies for the shared behaviour cases
/// (fixtures/edit_learning_cases.json, asserted on the Rust side) parse into
/// the proposals the card shows.
final class SharedEditCasesTests: XCTestCase {
    func testEveryExpectedReplyParsesIntoCardCopy() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("fixtures/edit_learning_cases.json")
        let file = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        let cases = try XCTUnwrap(file["cases"] as? [[String: Any]])
        XCTAssertGreaterThanOrEqual(cases.count, 10)
        for item in cases {
            let name = item["name"] as? String ?? "?"
            let expected = item["suggestions"] as? [[String: Any]] ?? []
            let parsed = EditSuggestion.parse(["id": "p1", "suggestions": expected])
            XCTAssertEqual(parsed.count, expected.count, name)
            for suggestion in parsed {
                XCTAssertFalse(EditSuggestionCopy.title(suggestion, category: "chat").isEmpty, name)
                if case let .term(term, heard, _, _) = suggestion.kind {
                    XCTAssertFalse(term.isEmpty, name)
                    XCTAssertFalse(heard.isEmpty, name)
                }
            }
        }
    }
}

final class SuggestionPanelGeometryTests: XCTestCase {
    func testCardSitsCentredAboveTheCapsule() {
        let screen = CGRect(x: 1_728, y: 0, width: 2_560, height: 1_440)
        let frame = SuggestionPanelGeometry.frame(screen: screen)
        XCTAssertEqual(frame.midX, screen.midX)
        XCTAssertEqual(frame.minY, SuggestionPanelGeometry.bottomOffset)
        XCTAssertGreaterThan(frame.minY, PanelGeometry.bottomInset + 16)
    }
}

final class KeystrokeCategoryTests: XCTestCase {
    func testCategoriesNeverCarryTheKey() {
        func classify(_ code: Int, cmd: Bool = false, opt: Bool = false, shift: Bool = false)
            -> KeystrokeCategory
        {
            KeystrokeCategory.classify(keyCode: code, command: cmd, option: opt, shift: shift)
        }
        XCTAssertEqual(classify(36), .returnKey)
        XCTAssertEqual(classify(36, cmd: true), .commandReturn)
        XCTAssertEqual(classify(51), .backspace)
        XCTAssertEqual(classify(51, opt: true), .deleteWord)
        XCTAssertEqual(classify(51, cmd: true), .deleteLine)
        XCTAssertEqual(classify(6, cmd: true), .undo)
        XCTAssertEqual(classify(6, cmd: true, shift: true), .redo)
        XCTAssertEqual(classify(7, cmd: true), .cut)
        XCTAssertEqual(classify(53), .escape)
        XCTAssertEqual(classify(6), .typing)
        XCTAssertEqual(classify(0), .typing)
        XCTAssertTrue(KeystrokeCategory.commandReturn.sends)
        XCTAssertFalse(KeystrokeCategory.backspace.sends)
    }
}

final class EditSuggestionTests: XCTestCase {
    /// Settings lists every approved style, in a stable order, as a
    /// sentence, so the user can see and remove what was learned.
    func testLearnedStylesAreListedAsSentences() {
        let rows = EditSuggestionCopy.styleRows([
            "email": ["no_trailing_period"],
            "chat": ["no_trailing_period", "lowercase_start"],
        ])
        XCTAssertEqual(
            rows.map(\.text),
            [
                "Start with a lowercase letter in chat apps",
                "Leave off the final period in chat apps",
                "Leave off the final period in email",
            ])
        XCTAssertEqual(rows.first?.category, "chat")
        XCTAssertEqual(rows.first?.preference, "lowercase_start")
        XCTAssertTrue(EditSuggestionCopy.styleRows([:]).isEmpty)
    }

    private let event: [String: Any] = [
        "type": "result", "kind": "edit", "id": "p3",
        "suggestions": [
            [
                "kind": "dictionary_term", "term": "cuBLAS", "heard_as": "coo blast",
                "reason": "misheard", "existing_entry": false,
            ],
            [
                "kind": "dictionary_term", "term": "GitHub", "heard_as": "Github",
                "reason": "casing", "existing_entry": false,
            ],
            ["kind": "style", "preference": "lowercase_start"],
            ["kind": "unknown"],
        ],
    ]

    func testParseAndCopy() {
        let suggestions = EditSuggestion.parse(event)
        XCTAssertEqual(suggestions.count, 3)
        XCTAssertEqual(
            EditSuggestionCopy.title(suggestions[0], category: "chat"),
            "Add \u{201C}cuBLAS\u{201D} to your dictionary?")
        XCTAssertEqual(EditSuggestionCopy.detail(suggestions[0]), "Heard as \u{201C}coo blast\u{201D}")
        XCTAssertEqual(
            EditSuggestionCopy.title(suggestions[1], category: "chat"),
            "Always write \u{201C}GitHub\u{201D} like this?")
        XCTAssertEqual(EditSuggestionCopy.detail(suggestions[1]), "Phonon wrote \u{201C}Github\u{201D}")
        XCTAssertEqual(
            EditSuggestionCopy.title(suggestions[2], category: "chat"),
            "Start with a lowercase letter in chat apps?")
        let existing = EditSuggestion(
            index: 0, kind: .term(term: "cuBLAS", heardAs: "coo blast", casing: false, existingEntry: true))
        XCTAssertEqual(
            EditSuggestionCopy.title(existing, category: "other"),
            "Teach Phonon that \u{201C}coo blast\u{201D} means cuBLAS?")
    }

    func testStyleNeedsARepeatAndRejectionsStickForTheSession() {
        var filter = EditSuggestionFilter()
        let suggestions = EditSuggestion.parse(event)
        let first = filter.admit(suggestions, category: "chat", approvedStyles: [])
        XCTAssertEqual(first.count, 2, "style waits for a second sighting")
        let second = filter.admit(suggestions, category: "chat", approvedStyles: [])
        XCTAssertEqual(second.count, 3)
        filter.reject(suggestions[0])
        let third = filter.admit(suggestions, category: "chat", approvedStyles: ["lowercase_start"])
        XCTAssertEqual(third.map(\.index), [1])
        // A style seen in chat does not count toward email.
        let email = filter.admit([suggestions[2]], category: "email", approvedStyles: [])
        XCTAssertTrue(email.isEmpty)
    }
}

@MainActor
final class LearnedTermStoreTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "phonon-learned-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    func testApprovingANewTermAddsItWithTheHeardFormAndUndoRemovesIt() throws {
        let store = NativeAppStore(supportDirectory: directory)
        let undo = try XCTUnwrap(store.approveLearnedTerm(term: "cuBLAS", heardAs: "coo blast"))
        let entry = try XCTUnwrap(store.dictionaryEntries.first)
        XCTAssertEqual(entry.phrase, "cuBLAS")
        XCTAssertEqual(entry.spokenForms, ["coo blast"])
        XCTAssertEqual(entry.source, "edit-learning")
        XCTAssertNil(store.approveLearnedTerm(term: "cublas", heardAs: "coo blast"))
        XCTAssertEqual(store.dictionaryEntries.count, 1)
        XCTAssertTrue(store.undoLearnedTerm(undo))
        XCTAssertTrue(store.dictionaryEntries.isEmpty)
        XCTAssertFalse(store.undoLearnedTerm(undo))
    }

    func testApprovingAKnownTermOnlyAddsTheSpokenForm() throws {
        let store = NativeAppStore(supportDirectory: directory)
        store.upsertDictionary(
            originalID: nil, phrase: "cuBLAS", replacement: nil, spokenForms: ["koo bloss"])
        let undo = try XCTUnwrap(store.approveLearnedTerm(term: "cuBLAS", heardAs: "coo blast"))
        XCTAssertEqual(store.dictionaryEntries.count, 1)
        XCTAssertEqual(store.dictionaryEntries[0].spokenForms, ["coo blast", "koo bloss"])
        XCTAssertTrue(store.undoLearnedTerm(undo))
        XCTAssertEqual(store.dictionaryEntries[0].spokenForms, ["koo bloss"])
        // A casing proposal for a known term has nothing to add.
        XCTAssertNil(store.approveLearnedTerm(term: "cuBLAS", heardAs: "cublas"))
    }

    func testStylePreferencesSaveByCategoryAndKeepRustOwnedKeys() throws {
        try Data(#"{"schema_version":2,"profile_prefix":true}"#.utf8)
            .write(to: directory.appendingPathComponent("settings.json"))
        let store = NativeAppStore(supportDirectory: directory)
        store.approveStylePreference("lowercase_start", category: "chat")
        store.approveStylePreference("lowercase_start", category: "chat")
        XCTAssertEqual(store.settings.stylePreferences, ["chat": ["lowercase_start"]])
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with: Data(contentsOf: directory.appendingPathComponent("settings.json")))
                as? [String: Any])
        XCTAssertEqual(object["profile_prefix"] as? Bool, true)
        XCTAssertEqual(
            (object["style_preferences"] as? [String: [String]])?["chat"], ["lowercase_start"])
        store.removeStylePreference("lowercase_start", category: "chat")
        XCTAssertEqual(store.settings.stylePreferences, [:])
        XCTAssertTrue(store.settings.editLearning)
    }
}

extension Result {
    fileprivate var failureReason: Failure? {
        if case let .failure(reason) = self { return reason }
        return nil
    }
}
