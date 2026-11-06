import CoreML
import FluidAudio
import Foundation

/// Loads Parakeet from the verified download folder and nowhere else. FluidAudio's
/// `AsrModels.load` deletes the folder and downloads an unpinned copy from Hugging Face when a
/// model is missing or fails to load; this loader fails instead, so the pinned, hash-checked
/// download stays the app's only network use. Compute units match `AsrModels.load`: the
/// preprocessor on the CPU, the rest on the CPU and Neural Engine (never the GPU, which iOS
/// refuses to background apps). `scripts/check-offline-asr.sh` keeps the download path unused.
enum LocalAsrModels {
    struct LoadError: LocalizedError {
        let file: String
        let reason: String
        var errorDescription: String? {
            "The speech model \(file) did not load (\(reason)). Delete and re-download the models in Settings."
        }
    }

    static func load(from directory: URL) throws -> AsrModels {
        let config = AsrModels.defaultConfiguration()
        func model(_ name: String, _ units: MLComputeUnits) throws -> MLModel {
            let url = directory.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else { throw LoadError(file: name, reason: "missing") }
            let c = MLModelConfiguration()
            c.computeUnits = units
            c.allowLowPrecisionAccumulationOnGPU = true
            do {
                return try MLModel(contentsOf: url, configuration: c)
            } catch {
                throw LoadError(file: name, reason: error.localizedDescription)
            }
        }
        let names = ModelNames.ASR.self
        let vocabName = names.vocabularyFile
        guard let data = try? Data(contentsOf: directory.appendingPathComponent(vocabName)),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: String] else {
            throw LoadError(file: vocabName, reason: "missing or unreadable")
        }
        var vocabulary: [Int: String] = [:]
        for (key, token) in json { if let id = Int(key) { vocabulary[id] = token } }
        return AsrModels(
            encoder: try model(names.encoderFile, config.computeUnits),
            preprocessor: try model(names.preprocessorFile, .cpuOnly),
            decoder: try model(names.decoderFile, config.computeUnits),
            joint: try model(names.jointFile, config.computeUnits),
            configuration: config,
            vocabulary: vocabulary,
            version: .v2)
    }
}
