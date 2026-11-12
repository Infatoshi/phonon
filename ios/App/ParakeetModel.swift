import Foundation

/// The speech model's files, pinned: FluidAudio's Parakeet TDT 0.6B v2 CoreML build at a fixed Hugging Face commit,
/// every file of the four `.mlmodelc` folders FluidAudio loads for `.v2` (Preprocessor, Encoder, Decoder,
/// JointDecision; `ModelNames.ASR.requiredModels` in FluidAudio 0.18) plus `parakeet_vocab.json`, each with its
/// size and sha256 from the Hugging Face API. Replaces `AsrModels.downloadAndLoad(version: .v2)`, which fetched the
/// repository's `main` branch with no hash check. Plain Foundation, so the unit tests compile it.
enum ParakeetModel {
    static let repo = "FluidInference/parakeet-tdt-0.6b-v2-coreml"
    static let revision = "ee09c569f73759e6d44c9bd16766f477b2b36d39"
    static let folder = "parakeet-tdt-0.6b-v2-coreml"

    private static func pin(_ path: String, _ sha256: String, _ size: Int64) -> ModelFile {
        ModelFile(path: path, url: URL(string: "https://huggingface.co/\(repo)/resolve/\(revision)/\(path)")!,
                  sha256: sha256, size: size)
    }

    static let files: [ModelFile] = [
        pin("Decoder.mlmodelc/analytics/coremldata.bin", "46de1a6fe2e49d19a2125bc91acf020df7f2aea84ba821532aade8427a440b05", 243),
        pin("Decoder.mlmodelc/coremldata.bin", "d200ca07694a347f6d02a3886a062ae839831e094e443222f2e48a14945966a8", 554),
        pin("Decoder.mlmodelc/metadata.json", "90a279b822496316458febc0ce761ab05954fadd9d66aa97bea077a35fc8f2b2", 3_427),
        pin("Decoder.mlmodelc/model.mil", "7b95a5a6b672c652000348a67b6d4d92bb8e176b978c6666fe73c28a4d7ec579", 13_106),
        pin("Decoder.mlmodelc/weights/weight.bin", "27d26890221d82322c1092fd99d7b40578e435d5cf4b83c887c42603caf97aba", 14_429_952),
        pin("Encoder.mlmodelc/analytics/coremldata.bin", "42e638870d73f26b332918a3496ce36793fbb413a81cbd3d16ba01328637a105", 243),
        pin("Encoder.mlmodelc/coremldata.bin", "4def7aa848599ad0e17a8b9a982edcdbf33cf92e1f4b798de32e2ca0bc74b030", 485),
        pin("Encoder.mlmodelc/metadata.json", "58222fbc48c13c49d9715567803cd50cb9c23e4360462e0f8ffcea59a2c73c63", 2_926),
        pin("Encoder.mlmodelc/model.mil", "ed7b19156ca29fa7dfd6891deb9fda4b0e8893f68597c985d135736546a43808", 959_769),
        pin("Encoder.mlmodelc/weights/weight.bin", "4adc7ad44f9d05e1bffeb2b06d3bb02861a5c7602dff63a6b494aed3bf8a6c3e", 445_187_200),
        pin("JointDecision.mlmodelc/analytics/coremldata.bin", "f1183ba213bb94a918c8d2cad19ab045320618f97f6ca662245b3936d7b090f7", 243),
        pin("JointDecision.mlmodelc/coremldata.bin", "e2c6752f1c8cf2d3f6f26ec93195c9bfa759ad59edf9f806696a138154f96f11", 534),
        pin("JointDecision.mlmodelc/metadata.json", "ba8d309417b9acd4a175fdb15687de6a941db2f5b06666a60e7cf3cc8e2d3c3c", 2_936),
        pin("JointDecision.mlmodelc/model.mil", "93bf82042235127cb81ab537dcae47a1c2e7e242ce4ffdaf772981b45eedc4f0", 9_722),
        pin("JointDecision.mlmodelc/weights/weight.bin", "ca22a65903a05e64137677da608077578a8606090a598abf4875fa6199aaa19d", 3_453_388),
        pin("Preprocessor.mlmodelc/analytics/coremldata.bin", "03ab3c1327a054c54c07a40325db967ec574f2c91dcc8192bfa44aa561bcf2d8", 243),
        pin("Preprocessor.mlmodelc/coremldata.bin", "d88ea1fc349459c9e100d6a96688c5b29a1f0d865f544be103001724b986b6d6", 494),
        pin("Preprocessor.mlmodelc/metadata.json", "fb16c581ff5e1b962e7cb2181ed892cd32f9f84c12b6e80ff3e089f28e35bcbb", 2_974),
        pin("Preprocessor.mlmodelc/model.mil", "3e06d16fd061294c8a75be68c43a3b1ed1f593d4a9c35249e9cdbccadc59721e", 27_166),
        pin("Preprocessor.mlmodelc/weights/weight.bin", "a5f7df6c7f47147ae9486fe18cc7792f9a44d093ec3c6a11e91ef2dc363c48dc", 298_880),
        pin("parakeet_vocab.json", "57019fe3c745772ca83a1b048a4bb951cd51329504ea33d4d83316b96e279a97", 18_762),
    ]

    private static var appSupport: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
    }

    /// Where the pinned files live (Application Support/Phonon/models/<folder>), excluded from iCloud backup.
    static var directory: URL { appSupport.appendingPathComponent("Phonon/models/\(folder)", isDirectory: true) }

    /// FluidAudio's own cache (`AsrModels.defaultCacheDirectory(for: .v2)`), where builds before the pin put the
    /// model from the repository's `main`.
    static var legacyDirectory: URL { appSupport.appendingPathComponent("FluidAudio/Models/\(folder)", isDirectory: true) }

    /// Every pinned file ready in `directory`, downloading what is missing. An install from an older build is
    /// verified and reused: each file in FluidAudio's cache that matches its pin is moved over (no 464 MB
    /// re-download), the rest is fetched, and the old folder is removed.
    static func ensure(store: ModelStore, files: [ModelFile] = ParakeetModel.files, legacy: URL? = legacyDirectory,
                       progress: @escaping @Sendable (Int64, Int64) -> Void = { _, _ in }) async throws -> URL {
        if !store.allReady(files) {
            if let legacy, FileManager.default.fileExists(atPath: legacy.path) {
                store.adopt(from: legacy, files: files)
                try? FileManager.default.removeItem(at: legacy)
            }
            try await store.ensure(files, progress: progress)
        }
        var root = store.root
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? root.setResourceValues(values)
        return store.root
    }
}
