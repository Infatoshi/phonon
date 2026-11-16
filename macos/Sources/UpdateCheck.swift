import Foundation

/// A Mac release on GitHub newer than the running app (SPEC "Update check").
struct AvailableUpdate: Equatable {
    let version: String
    let pageURL: URL
}

/// Asks GitHub's public releases API for the newest stable Mac release. The
/// request carries no account, identifier or usage data; GitHub sees only the
/// address it comes from. Foundation only, so it runs outside the app too.
enum UpdateCheck {
    static let releasesURL = URL(
        string: "https://api.github.com/repos/Infatoshi/phonon/releases?per_page=10")!
    static let interval: TimeInterval = 24 * 60 * 60
    /// Long enough that the check never competes with the first model load.
    static let firstCheckDelay: TimeInterval = 60
    static let brewCommand = "brew upgrade --cask phonon"
    static let caskrooms = ["/opt/homebrew/Caskroom/phonon", "/usr/local/Caskroom/phonon"]

    struct Release: Decodable {
        struct Asset: Decodable { let name: String }
        let tagName: String
        let htmlURL: URL
        let draft: Bool
        let prerelease: Bool
        let assets: [Asset]

        enum CodingKeys: String, CodingKey {
            case draft, prerelease, assets
            case tagName = "tag_name"
            case htmlURL = "html_url"
        }
    }

    /// "v0.2.4" or "0.2.4" as [0, 2, 4]. Anything else, such as the Windows
    /// beta tag "v0.2.0-win-beta.1", is not a Mac version.
    static func versionParts(_ tag: String) -> [Int]? {
        let text = tag.hasPrefix("v") ? String(tag.dropFirst()) : tag
        let parts = text.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 3,
            parts.allSatisfy({ !$0.isEmpty && $0.allSatisfy { $0.isASCII && $0.isNumber } })
        else { return nil }
        return parts.compactMap { Int($0) }
    }

    /// The newest release above `current`. A Mac release is a published,
    /// non-prerelease one that already carries Phonon.dmg, so a release whose
    /// DMG is still uploading is not announced yet.
    static func newest(in data: Data, current: String) throws -> AvailableUpdate? {
        guard let running = versionParts(current) else { return nil }
        let releases = try JSONDecoder().decode([Release].self, from: data)
        let best = releases
            .filter { !$0.draft && !$0.prerelease && $0.assets.contains { $0.name == "Phonon.dmg" } }
            .compactMap { release in versionParts(release.tagName).map { (release, $0) } }
            .max { $0.1.lexicographicallyPrecedes($1.1) }
        guard let (release, parts) = best, running.lexicographicallyPrecedes(parts) else {
            return nil
        }
        return AvailableUpdate(
            version: parts.map(String.init).joined(separator: "."), pageURL: release.htmlURL)
    }

    static func fetch(current: String) async throws -> AvailableUpdate? {
        var request = URLRequest(url: releasesURL, timeoutInterval: 20)
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        request.setValue("Phonon", forHTTPHeaderField: "User-Agent")
        let session = URLSession(configuration: .ephemeral)
        defer { session.finishTasksAndInvalidate() }
        let (data, response) = try await session.data(for: request)
        guard let status = (response as? HTTPURLResponse)?.statusCode, status == 200 else {
            throw URLError(.badServerResponse)
        }
        return try newest(in: data, current: current)
    }

    /// The running app's version; nil for a bare `swift build` binary, which
    /// has no Info.plist and so never checks.
    static var runningVersion: String? {
        guard let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString")
            as? String, versionParts(version) != nil
        else { return nil }
        return version
    }

    /// The Homebrew cask keeps a Caskroom entry, and brew users should update
    /// through brew so it does not keep a stale version on record.
    static func installedByHomebrew(fileManager: FileManager = .default) -> Bool {
        caskrooms.contains { fileManager.fileExists(atPath: $0) }
    }
}
