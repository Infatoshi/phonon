import Foundation

/// One pinned model file: a Hugging Face path at a fixed revision with its size and sha256.
public struct ModelFile: Equatable, Sendable {
    public let repo: String
    public let revision: String
    public let path: String
    /// Local folder under the models root (FluidAudio infers the Parakeet version from it).
    public let folder: String
    public let size: Int64
    public let sha256: String

    public init(repo: String, revision: String, path: String, folder: String, size: Int64, sha256: String) {
        self.repo = repo
        self.revision = revision
        self.path = path
        self.folder = folder
        self.size = size
        self.sha256 = sha256
    }

    /// Pinned to the commit, never `main`, so a later upload cannot change what is fetched.
    public var url: URL {
        let escaped = path.split(separator: "/").map {
            String($0).addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0)
        }.joined(separator: "/")
        return URL(string: "https://huggingface.co/\(repo)/resolve/\(revision)/\(escaped)")!
    }

    public func localURL(root: URL) -> URL {
        root.appendingPathComponent(folder, isDirectory: true).appendingPathComponent(path)
    }
}

public enum ModelPack: String, CaseIterable, Sendable {
    case asr, refiner

    public var files: [ModelFile] {
        switch self {
        case .asr: return ModelManifest.asrFiles
        case .refiner: return ModelManifest.refinerFiles
        }
    }

    public var totalBytes: Int64 { files.reduce(0) { $0 + $1.size } }

    public var title: String {
        switch self {
        case .asr: return "Speech recognition"
        case .refiner: return "Cleanup model"
        }
    }

    public var detail: String {
        switch self {
        case .asr: return "Parakeet TDT 0.6B v2, Core ML"
        case .refiner: return "Phonon refiner 1.7B, Q4"
        }
    }
}

public enum ModelManifest {
    public static var allFiles: [ModelFile] { asrFiles + refinerFiles }
    public static var totalBytes: Int64 { allFiles.reduce(0) { $0 + $1.size } }

    public static let asrFolder = "parakeet-tdt-0.6b-v2-coreml"
    public static let refinerFolder = "phonon-refiner-1.7b"
    public static let refinerFile = "phonon-refiner-1.7b-Q4_K_S-e4.gguf"

    public static func asrDirectory(root: URL) -> URL {
        root.appendingPathComponent(asrFolder, isDirectory: true)
    }

    public static func refinerURL(root: URL) -> URL {
        root.appendingPathComponent(refinerFolder, isDirectory: true).appendingPathComponent(refinerFile)
    }
}
