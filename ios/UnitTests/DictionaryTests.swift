import XCTest

/// The desktop dictionary (crates/phonon-core data.rs) and its refiner side (sidecar/polish_server.py), on the
/// shared reference cases crates/phonon-core/src/dictionary_cases.json; mirrors
/// android/app/src/test/java/ai/phonon/keyboard/DictionaryTest.kt.
final class DictionaryTests: XCTestCase {
    private final class Anchor {}
    private static let bundle = Bundle(for: Anchor.self)

    private static let cases: [String: Any] = {
        let url = bundle.url(forResource: "dictionary_cases", withExtension: "json")!
        return try! JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
    }()
    private static let words = WordList.bundled(bundle)!

    private func list(_ key: String) -> [[String: Any]] { Self.cases[key] as! [[String: Any]] }
    private func dict(_ c: [String: Any]) -> UserDictionary {
        UserDictionary(entries: UserDictionary.parseEntries(c["entries"] as! [Any]), words: Self.words)
    }

    func testEnvelopesMatchTheRust() {
        let cases = list("envelope")
        XCTAssertGreaterThanOrEqual(cases.count, 20)
        for c in cases {
            XCTAssertEqual(dict(c).envelope(c["transcript"] as! String), c["envelope"] as! String, c["id"] as! String)
        }
    }

    func testVocabularyMatchesThePython() {
        for c in list("envelope") {
            let id = c["id"] as! String
            let expected = c["vocab"] as! [String]
            XCTAssertEqual(UserDictionary.refinerVocabulary(c["envelope"] as! String), expected, id)
            XCTAssertEqual(dict(c).prepare(c["transcript"] as! String).vocab, expected, id)
        }
    }

    func testPreparedTextIsTheEnvelopesTranscript() {
        for c in list("envelope") {
            let p = dict(c).prepare(c["transcript"] as! String)
            XCTAssertEqual(p.text, UserDictionary.transcriptPayload(c["envelope"] as! String), c["id"] as! String)
        }
    }

    func testExactReplacementsMatchTheRust() {
        let cases = list("exact")
        XCTAssertGreaterThanOrEqual(cases.count, 5)
        for c in cases {
            XCTAssertEqual(dict(c).applyExact(c["text"] as! String), c["expected"] as! String, c["id"] as! String)
        }
    }

    func testRestoreTermsMatchesThePython() {
        let cases = list("restore")
        XCTAssertGreaterThanOrEqual(cases.count, 8)
        for c in cases {
            XCTAssertEqual(UserDictionary.restoreTerms(spoken: c["spoken"] as! String, output: c["output"] as! String,
                                                       vocab: c["vocab"] as! [String]),
                           c["expected"] as! String, c["id"] as! String)
        }
    }

    func testEmptyDictionaryChangesNothing() {
        let d = UserDictionary()
        let p = d.prepare("  um so the plan is fine ")
        XCTAssertEqual(p.text, "  um so the plan is fine ")
        XCTAssertTrue(p.vocab.isEmpty)
        XCTAssertEqual(p.vocabLine, "")
        XCTAssertEqual(d.applyExact("Ship it."), "Ship it.")
        XCTAssertEqual(UserDictionary.restoreTerms(spoken: "use mtp", output: "Use MTK.", vocab: []), "Use MTK.")
    }

    func testWordListLooksUpExactWords() {
        let w = Self.words
        XCTAssertGreaterThan(w.count, 190_000)
        XCTAssertTrue(w.contains("aardvark"))
        XCTAssertTrue(w.contains("please"))
        XCTAssertFalse(w.contains("pietorch"))
        XCTAssertFalse(w.contains("Please"))
        XCTAssertFalse(w.contains(""))
    }

    func testSequenceMatcherMatchesDifflib() {
        // difflib.SequenceMatcher(a="a b c d".split(), b="a x c d e".split(), autojunk=False).get_opcodes()
        let ops = SequenceMatcher(["a", "b", "c", "d"], ["a", "x", "c", "d", "e"]).opcodes().map(\.tag)
        XCTAssertEqual(ops, ["equal", "replace", "equal", "insert"])
    }

    func testSavesAndLoadsTheDesktopSchema() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("dict-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        var d = UserDictionary(words: Self.words)
        d = d.with(.init(phrase: " vee llm ", replacement: "vLLM", source: "phone"))
        d = d.with(.init(phrase: "Kubernetes", spokenForms: ["cooper netties", "Cooper Netties"]))
        d = d.with(.init(phrase: "VEE LLM", replacement: "vllm"))  // same key as the first: replaces it
        XCTAssertEqual(d.entries.count, 2)
        try UserDictionary.save(dir: dir, d)
        let back = UserDictionary.load(dir: dir) { Self.words }
        XCTAssertEqual(back.entries, d.entries)
        XCTAssertNotNil(back.words)
        XCTAssertEqual(back.entries.first { $0.phrase == "Kubernetes" }?.spokenForms, ["cooper netties"])
        let json = try JSONSerialization.jsonObject(
            with: Data(contentsOf: dir.appendingPathComponent("dictionary.json"))) as! [String: Any]
        XCTAssertEqual(json["schema_version"] as? Int, 1)
        XCTAssertNotNil(json["updated_at_unix_ms"] as? NSNumber)
        let entries = json["entries"] as! [[String: Any]]
        XCTAssertEqual(entries[1]["replacement"] as? String, "vllm")
        XCTAssertNil(entries[0]["replacement"], "no replacement key when there is none, as serde skips it")
        XCTAssertEqual(back.without(back.entries[0]).entries.count, 1)
        // a missing or unreadable file is an empty dictionary, and the word list is not loaded for it
        XCTAssertTrue(UserDictionary.load(dir: dir.appendingPathComponent("none")) { XCTFail(); return nil }.isEmpty)
        try Data("not json".utf8).write(to: dir.appendingPathComponent("dictionary.json"))
        XCTAssertTrue(UserDictionary.load(dir: dir) { nil }.isEmpty)
    }

    func testAddingKeepsPhraseOnlyEntriesAndDropsEmptyReplacements() {
        var d = UserDictionary()
        d = d.with(.init(phrase: "Tailscale", replacement: "  "))
        d = d.with(.init(phrase: "Kubernetes", replacement: "Kubernetes"))
        d = d.with(.init(phrase: "   "))
        XCTAssertEqual(d.entries.map(\.phrase), ["Kubernetes", "Tailscale"])
        XCTAssertTrue(d.entries.allSatisfy { $0.replacement == nil })
    }
}
