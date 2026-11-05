import Foundation

/// One macro: say the trigger as a whole dictation and Phonon types the
/// expansion instead (Wispr Flow calls these snippets).
struct PhononMacro: Codable, Equatable, Identifiable {
    var trigger: String
    var expansion: String

    var id: String { MacroBook.normalize(trigger) }
}

/// macros.json in the app support directory:
/// `{"schema_version":1,"macros":[{"trigger":"my email","expansion":"..."}]}`.
struct MacroFile: Codable, Equatable {
    var schemaVersion = 1
    var macros: [PhononMacro] = []

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case macros
    }
}

enum MacroBook {
    static let fileName = "macros.json"

    /// A trigger and a dictation match when they agree ignoring case, runs of
    /// whitespace, and punctuation at either end ("My email." is "my email").
    static func normalize(_ text: String) -> String {
        let edges = CharacterSet.whitespacesAndNewlines
            .union(.punctuationCharacters)
            .union(.symbols)
        let words = text.lowercased()
            .split(whereSeparator: { $0.isWhitespace })
            .map(String.init)
            .joined(separator: " ")
        return words.trimmingCharacters(in: edges)
    }

    /// The text to paste for a finished dictation: a macro's expansion when the
    /// whole dictation is its trigger, otherwise the dictation unchanged.
    static func expand(_ text: String, macros: [PhononMacro]) -> String {
        let key = normalize(text)
        guard !key.isEmpty,
            let macro = macros.first(where: { normalize($0.trigger) == key }),
            !macro.expansion.isEmpty
        else { return text }
        return macro.expansion
    }

    static func load(from url: URL) -> [PhononMacro] {
        guard let data = try? Data(contentsOf: url),
            let file = try? JSONDecoder().decode(MacroFile.self, from: data)
        else { return [] }
        return file.macros
    }

    static func save(_ macros: [PhononMacro], to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(MacroFile(macros: macros))
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url, options: .atomic)
    }
}
