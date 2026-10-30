import SwiftUI

/// One piece of text in a comparison: a word (with its punctuation) or the
/// whitespace between words. Whitespace is never marked.
struct DiffToken: Equatable {
    let text: String
    let changed: Bool
}

/// Which words differ between what Phonon wrote and what the user kept, by
/// longest common subsequence over words (case and punctuation count, as in
/// Wispr Flow's edit view).
enum WordDiff {
    /// Above this many word pairs the middle of the texts is marked whole
    /// rather than aligned.
    static let maxCells = 250_000

    /// Words and whitespace runs, in order; joined they give the text back.
    static func tokens(_ text: String) -> [String] {
        var tokens = [String]()
        var current = ""
        var inSpace: Bool?
        for character in text {
            let space = character.isWhitespace
            if inSpace != nil, space != inSpace {
                tokens.append(current)
                current = ""
            }
            current.append(character)
            inSpace = space
        }
        if !current.isEmpty { tokens.append(current) }
        return tokens
    }

    static func compare(_ before: String, _ after: String) -> (before: [DiffToken], after: [DiffToken])
    {
        let left = tokens(before)
        let right = tokens(after)
        let leftWords = left.indices.filter { !isSpace(left[$0]) }
        let rightWords = right.indices.filter { !isSpace(right[$0]) }
        let (keptLeft, keptRight) = matches(
            leftWords.map { left[$0] }, rightWords.map { right[$0] })
        let unchangedLeft = Set(keptLeft.map { leftWords[$0] })
        let unchangedRight = Set(keptRight.map { rightWords[$0] })
        return (
            left.indices.map { DiffToken(text: left[$0], changed: !isSpace(left[$0]) && !unchangedLeft.contains($0)) },
            right.indices.map {
                DiffToken(text: right[$0], changed: !isSpace(right[$0]) && !unchangedRight.contains($0))
            }
        )
    }

    private static func isSpace(_ token: String) -> Bool {
        token.first?.isWhitespace ?? false
    }

    /// Indices of the words both sides keep, in each side's numbering.
    private static func matches(_ a: [String], _ b: [String]) -> ([Int], [Int]) {
        var prefix = 0
        while prefix < a.count, prefix < b.count, a[prefix] == b[prefix] { prefix += 1 }
        var suffix = 0
        while suffix < a.count - prefix, suffix < b.count - prefix,
            a[a.count - 1 - suffix] == b[b.count - 1 - suffix]
        {
            suffix += 1
        }
        var keptA = Array(0..<prefix)
        var keptB = Array(0..<prefix)
        let middleA = Array(a[prefix..<(a.count - suffix)])
        let middleB = Array(b[prefix..<(b.count - suffix)])
        if !middleA.isEmpty, !middleB.isEmpty, middleA.count * middleB.count <= maxCells {
            // Longest common subsequence over the middle.
            let n = middleA.count
            let m = middleB.count
            var table = [[Int]](repeating: [Int](repeating: 0, count: m + 1), count: n + 1)
            for i in stride(from: n - 1, through: 0, by: -1) {
                for j in stride(from: m - 1, through: 0, by: -1) {
                    table[i][j] =
                        middleA[i] == middleB[j]
                        ? table[i + 1][j + 1] + 1 : max(table[i + 1][j], table[i][j + 1])
                }
            }
            var i = 0
            var j = 0
            while i < n, j < m {
                if middleA[i] == middleB[j] {
                    keptA.append(prefix + i)
                    keptB.append(prefix + j)
                    i += 1
                    j += 1
                } else if table[i + 1][j] >= table[i][j + 1] {
                    i += 1
                } else {
                    j += 1
                }
            }
        }
        keptA += ((a.count - suffix)..<a.count)
        keptB += ((b.count - suffix)..<b.count)
        return (keptA, keptB)
    }
}

/// What Phonon wrote beside what the user kept, changed words marked:
/// struck through and dimmed on Phonon's side, highlighted on the user's.
struct EditComparison: View {
    let pasted: String
    let edited: String

    var body: some View {
        let diff = WordDiff.compare(pasted, edited)
        HStack(alignment: .top, spacing: 12) {
            column(
                title: "Phonon wrote",
                text: Self.attributed(diff.before, removed: true))
            column(
                title: "You kept",
                text: edited.isEmpty
                    ? Self.note("Deleted after it was typed")
                    : Self.attributed(diff.after, removed: false))
        }
    }

    private func column(title: String, text: AttributedString) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            Text(title).font(.headline)
            Text(text)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(12)
                .background(EmberTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 8))
        }
        .frame(maxWidth: .infinity, alignment: .topLeading)
    }

    private static func note(_ text: String) -> AttributedString {
        var note = AttributedString(text)
        note.foregroundColor = EmberTheme.muted
        return note
    }

    static func attributed(_ tokens: [DiffToken], removed: Bool) -> AttributedString {
        var result = AttributedString()
        for token in tokens {
            var piece = AttributedString(token.text)
            if token.changed {
                if removed {
                    piece.strikethroughStyle = .single
                    piece.foregroundColor = EmberTheme.muted
                    piece.backgroundColor = EmberTheme.accentSoft
                } else {
                    piece.foregroundColor = EmberTheme.text
                    piece.backgroundColor = EmberTheme.accent.opacity(0.35)
                }
            }
            result += piece
        }
        return result
    }
}
