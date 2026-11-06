import FluidAudio
import PhononCore
import XCTest
@testable import Phonon

/// The speech fixtures through the same ASR and refiner code the app uses. Models come from
/// PHONON_MODELS_DIR (a local copy of the pinned files); results go to PHONON_CHECKS_DIR.
final class FixturePipelineTests: XCTestCase {
    struct Row: Codable {
        var name: String
        var expected: String
        var raw: String
        var refined: String
        var guardFired: Bool
        var rawWER: Double
        var refinedWER: Double
        var asrSeconds: Double
        var refineSeconds: Double
        var promptTokens: Int
        var outputTokens: Int
        var decodeTokPerSec: Double
    }

    func testFixturesThroughBothModels() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelsPath = env["PHONON_MODELS_DIR"] else { throw XCTSkip("PHONON_MODELS_DIR not set") }
        let models = URL(fileURLWithPath: modelsPath)
        let fixtures = Bundle(for: DictationEngine.self).url(forResource: "Fixtures", withExtension: nil)!
        let expected = try JSONDecoder().decode([String: String].self, from: Data(contentsOf: fixtures.appendingPathComponent("expected.json")))

        let asrModels = try LocalAsrModels.load(from: ModelManifest.asrDirectory(root: models))
        let asr = AsrManager(config: .default)
        try await asr.initialize(models: asrModels)
        let refiner = try LlamaRefiner(modelURL: ModelManifest.refinerURL(root: models))

        var rows: [Row] = []
        for name in expected.keys.sorted() {
            let samples = try AudioFile16k.read(fixtures.appendingPathComponent("\(name).wav"))
            XCTAssertGreaterThan(samples.count, 16_000, name)
            let t0 = Date()
            let raw = try await asr.transcribe(samples, source: .system).text
            try await asr.resetDecoderState(for: .system)
            let asrSeconds = Date().timeIntervalSince(t0)
            let t1 = Date()
            let out = try await refiner.refineAsync(transcript: raw, mode: .casual, vocabulary: ["CUDA", "vLLM", "H100"])
            let want = expected[name]!
            rows.append(Row(name: name, expected: want, raw: raw, refined: out.text, guardFired: out.guardFired,
                            rawWER: wer(raw, want), refinedWER: wer(out.text, want),
                            asrSeconds: asrSeconds, refineSeconds: Date().timeIntervalSince(t1),
                            promptTokens: out.stats.promptTokens, outputTokens: out.stats.outputTokens,
                            decodeTokPerSec: out.stats.decodeTokPerSec))
            XCTAssertFalse(raw.isEmpty, "\(name): ASR returned nothing")
            XCTAssertFalse(out.text.isEmpty, "\(name): refiner returned nothing")
        }
        if let dir = env["PHONON_CHECKS_DIR"] {
            try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
            let enc = JSONEncoder()
            enc.outputFormatting = [.prettyPrinted, .sortedKeys]
            try enc.encode(rows).write(to: URL(fileURLWithPath: dir).appendingPathComponent("fixture-pipeline.json"))
        }
        // The pipeline is useful only if it lands close to what was meant.
        let meanRefined = rows.map(\.refinedWER).reduce(0, +) / Double(rows.count)
        XCTAssertLessThan(meanRefined, 0.25, "mean refined WER \(meanRefined)")
    }

    /// Word error rate on lowercased words without punctuation.
    func wer(_ hyp: String, _ ref: String) -> Double {
        func words(_ s: String) -> [String] {
            s.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted.subtracting(CharacterSet(charactersIn: "'"))).filter { !$0.isEmpty }
        }
        let h = words(hyp), r = words(ref)
        guard !r.isEmpty else { return h.isEmpty ? 0 : 1 }
        var d = Array(0...h.count)
        for i in 1...r.count {
            var prev = d[0]
            d[0] = i
            for j in stride(from: 1, through: h.count, by: 1) {
                let cur = d[j]
                d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] == h[j - 1] ? 0 : 1))
                prev = cur
            }
        }
        return Double(d[h.count]) / Double(r.count)
    }
}
