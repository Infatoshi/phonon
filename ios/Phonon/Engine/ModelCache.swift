import Foundation

/// Core ML compiles each model for this device into Library/Caches/<bundle id>/
/// com.apple.e5rt.e5bundlecache and reuses the result. A compile cut short (the app closed or
/// killed while loading) leaves "<hash>.tmp.<pid>_<n>.bundle", which Core ML neither reuses nor
/// removes; in the simulator these reached 40 GB. At launch Phonon removes the ones earlier
/// processes left; finished bundles stay, so models compile once.
enum ModelCache {
    static var directory: URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(Bundle.main.bundleIdentifier ?? "com.abhaymettu.phonon")
            .appendingPathComponent("com.apple.e5rt.e5bundlecache")
    }

    /// The compiled bundles are keyed to the installed app: a relaunch reuses them, but each
    /// install or update compiles a new set (FACT in the simulator: +2.2 GB per install, none
    /// per relaunch) and the old set is never read again. On the first launch of a new install,
    /// empty the cache before any model loads, so it holds one set.
    static func resetIfNewInstall() {
        let fm = FileManager.default
        let marker = directory.deletingLastPathComponent().appendingPathComponent("phonon-install")
        let install = Bundle.main.bundleURL.path
        guard (try? String(contentsOf: marker, encoding: .utf8)) != install else { return }
        try? fm.removeItem(at: directory)
        try? fm.createDirectory(at: marker.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? install.write(to: marker, atomically: true, encoding: .utf8)
    }

    /// Removes partial compiles from other processes. Returns how many it removed.
    @discardableResult
    static func pruneAbandoned() -> Int {
        let fm = FileManager.default
        let mine = ".tmp.\(getpid())_"
        guard let walk = fm.enumerator(at: directory, includingPropertiesForKeys: nil) else { return 0 }
        var removed = 0
        for case let url as URL in walk where url.lastPathComponent.contains(".tmp.") {
            walk.skipDescendants()
            if !url.lastPathComponent.contains(mine), (try? fm.removeItem(at: url)) != nil { removed += 1 }
        }
        return removed
    }

    static func bytesOnDisk() -> Int64 {
        guard let walk = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: [.totalFileAllocatedSizeKey]) else { return 0 }
        var total: Int64 = 0
        for case let url as URL in walk {
            total += Int64((try? url.resourceValues(forKeys: [.totalFileAllocatedSizeKey]))?.totalFileAllocatedSize ?? 0)
        }
        return total
    }
}
