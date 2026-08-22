// A copy of the small, irreplaceable files, kept outside ~/Library.
//
// Uninstallers match on the app name, the developer name, or the bundle
// identifier, then sweep every ~/Library subdirectory that hits: Application
// Support, Preferences, Caches, Logs, Saved Application State. Nothing under
// that tree can be made to survive one, and hiding data from an uninstaller
// would be wrong anyway. So the mirror lives in ~/.phonon, which no such
// pattern reaches, and the app asks before putting anything back.
//
// The corpus is not mirrored. It is large, and Settings offers an explicit
// export for it.

import Foundation

struct MirrorManifest: Codable, Equatable {
    var savedAt: Date
    var dictionaryEntries: Int
    var historyItems: Int
}

enum PhononDataMirror {
    /// Everything here is hand-made or accumulated, and none of it can be
    /// recovered by reinstalling. Together they are about 128 KB.
    static let mirroredFiles = [
        "dictionary.json",
        "settings.json",
        "History.json",
        "Vocabulary.txt",
        "WordReplacements.json",
    ]

    static func directory(fileManager: FileManager = .default) -> URL {
        fileManager.homeDirectoryForCurrentUser
            .appendingPathComponent(".phonon", isDirectory: true)
            .appendingPathComponent("backup", isDirectory: true)
    }

    private static func manifestURL(_ directory: URL) -> URL {
        directory.appendingPathComponent("manifest.json")
    }

    // MARK: - Capture

    /// Copy the current store into the mirror.
    ///
    /// A wiped store must never overwrite a good mirror, so a capture that
    /// would replace counted entries with none is refused outright. Losing a
    /// backup to the very event it exists for is the only failure that matters
    /// here.
    @discardableResult
    static func capture(
        from support: URL, mirror: URL? = nil, fileManager: FileManager = .default
    ) -> MirrorManifest? {
        let dictionaryCount = countDictionaryEntries(in: support, fileManager: fileManager)
        let historyCount = countHistoryItems(in: support, fileManager: fileManager)
        let target = mirror ?? directory(fileManager: fileManager)
        if let existing = manifest(mirror: target, fileManager: fileManager) {
            let losesDictionary = existing.dictionaryEntries > 0 && dictionaryCount == 0
            let losesHistory = existing.historyItems > 0 && historyCount == 0
            if losesDictionary || losesHistory { return nil }
        }
        guard dictionaryCount > 0 || historyCount > 0 else { return nil }

        do {
            try fileManager.createDirectory(
                at: target, withIntermediateDirectories: true)
            for name in mirroredFiles {
                let source = support.appendingPathComponent(name)
                guard fileManager.fileExists(atPath: source.path) else { continue }
                let destination = target.appendingPathComponent(name)
                if fileManager.fileExists(atPath: destination.path) {
                    try fileManager.removeItem(at: destination)
                }
                try fileManager.copyItem(at: source, to: destination)
            }
            let written = MirrorManifest(
                savedAt: Date(), dictionaryEntries: dictionaryCount, historyItems: historyCount)
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(written).write(
                to: manifestURL(target), options: .atomic)
            return written
        } catch {
            return nil
        }
    }

    // MARK: - Inspect

    static func manifest(
        mirror: URL? = nil, fileManager: FileManager = .default
    ) -> MirrorManifest? {
        let url = manifestURL(mirror ?? directory(fileManager: fileManager))
        guard let data = try? Data(contentsOf: url) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(MirrorManifest.self, from: data)
    }

    /// True when the store holds nothing worth keeping, which is what a wipe
    /// or a first run looks like.
    static func storeIsEmpty(_ support: URL, fileManager: FileManager = .default) -> Bool {
        countDictionaryEntries(in: support, fileManager: fileManager) == 0
            && countHistoryItems(in: support, fileManager: fileManager) == 0
    }

    /// The mirror is only worth offering when it holds more than the store does.
    static func restorableManifest(
        for support: URL, mirror: URL? = nil, fileManager: FileManager = .default
    ) -> MirrorManifest? {
        guard let manifest = manifest(mirror: mirror, fileManager: fileManager) else { return nil }
        guard manifest.dictionaryEntries > 0 || manifest.historyItems > 0 else { return nil }
        guard storeIsEmpty(support, fileManager: fileManager) else { return nil }
        return manifest
    }

    // MARK: - Restore and forget

    /// Never clobbers: a file already in the store is left alone.
    static func restore(
        into support: URL, mirror: URL? = nil, fileManager: FileManager = .default
    ) throws {
        let source = mirror ?? directory(fileManager: fileManager)
        try fileManager.createDirectory(at: support, withIntermediateDirectories: true)
        for name in mirroredFiles {
            let from = source.appendingPathComponent(name)
            let to = support.appendingPathComponent(name)
            guard fileManager.fileExists(atPath: from.path),
                !fileManager.fileExists(atPath: to.path)
            else { continue }
            try fileManager.copyItem(at: from, to: to)
        }
    }

    static func forget(mirror: URL? = nil, fileManager: FileManager = .default) throws {
        let target = mirror ?? directory(fileManager: fileManager)
        guard fileManager.fileExists(atPath: target.path) else { return }
        try fileManager.removeItem(at: target)
    }

    // MARK: - Export

    /// Copy the whole store, corpus included, somewhere the owner picked.
    static func exportAll(
        from support: URL, to destination: URL, fileManager: FileManager = .default
    ) throws {
        if fileManager.fileExists(atPath: destination.path) {
            try fileManager.removeItem(at: destination)
        }
        try fileManager.copyItem(at: support, to: destination)
    }

    // MARK: - Counting

    private static func countDictionaryEntries(
        in support: URL, fileManager: FileManager
    ) -> Int {
        let url = support.appendingPathComponent("dictionary.json")
        guard let data = try? Data(contentsOf: url),
            let object = try? JSONSerialization.jsonObject(with: data)
        else { return 0 }
        if let list = object as? [Any] { return list.count }
        if let wrapper = object as? [String: Any], let list = wrapper["entries"] as? [Any] {
            return list.count
        }
        return 0
    }

    private static func countHistoryItems(in support: URL, fileManager: FileManager) -> Int {
        let corpus = support.appendingPathComponent("Corpus", isDirectory: true)
        let entries =
            (try? fileManager.contentsOfDirectory(
                at: corpus, includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]))?.count ?? 0
        if entries > 0 { return entries }
        let url = support.appendingPathComponent("History.json")
        guard let data = try? Data(contentsOf: url),
            let list = try? JSONSerialization.jsonObject(with: data) as? [Any]
        else { return 0 }
        return list.count
    }
}
