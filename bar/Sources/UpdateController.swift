import AppKit
import Foundation

/// Checks for a newer release a minute after launch and then once a day while
/// Settings > Check for updates is on, and announces each new version once,
/// never during a dictation. The menu bar item and the sidebar keep offering
/// it after that. "Check Now" works with the setting off.
@MainActor
final class UpdateController {
    private let store: NativeAppStore
    private var timer: Timer?
    private var checking = false
    /// True while a dictation is recording or being processed.
    var isBusy: () -> Bool = { false }
    /// Tells the menu bar item which update, if any, to offer.
    var onAvailableChanged: ((AvailableUpdate?) -> Void)?

    init(store: NativeAppStore) {
        self.store = store
    }

    func start() {
        guard timer == nil, UpdateCheck.runningVersion != nil else { return }
        DispatchQueue.main.asyncAfter(deadline: .now() + UpdateCheck.firstCheckDelay) {
            [weak self] in
            self?.check(manual: false)
        }
        timer = Timer.scheduledTimer(withTimeInterval: UpdateCheck.interval, repeats: true) {
            [weak self] _ in
            MainActor.assumeIsolated { self?.check(manual: false) }
        }
    }

    func check(manual: Bool) {
        guard !checking, let running = UpdateCheck.runningVersion else { return }
        guard manual || store.settings.checkForUpdates else { return }
        checking = true
        Task { [weak self] in
            let result: Result<AvailableUpdate?, Error>
            do {
                result = .success(try await UpdateCheck.fetch(current: running))
            } catch {
                result = .failure(error)
            }
            self?.finish(result, running: running, manual: manual)
        }
    }

    private func finish(_ result: Result<AvailableUpdate?, Error>, running: String, manual: Bool) {
        checking = false
        switch result {
        case .success(let update):
            NSLog("phonon update: running \(running), newer release: \(update?.version ?? "none")")
            store.availableUpdate = update
            onAvailableChanged?(update)
            if manual {
                if let update { present(update) } else { presentUpToDate(running) }
            } else if let update, update.version != store.settings.updateAnnounced {
                announceWhenIdle(update)
            }
        case .failure(let error):
            NSLog("phonon update: check failed: \(error.localizedDescription)")
            if manual { presentFailure(error) }
        }
    }

    /// Waits up to ten minutes for the pipeline to go idle; a busy stretch
    /// longer than that leaves the announcement for the next daily check.
    private func announceWhenIdle(_ update: AvailableUpdate, attempt: Int = 0) {
        if isBusy() {
            guard attempt < 120 else { return }
            DispatchQueue.main.asyncAfter(deadline: .now() + 5) { [weak self] in
                self?.announceWhenIdle(update, attempt: attempt + 1)
            }
            return
        }
        store.updateSettings { $0.updateAnnounced = update.version }
        present(update)
    }

    /// The menu bar item, the sidebar and Settings all land here.
    func showAvailable() {
        if let update = store.availableUpdate { present(update) } else { check(manual: true) }
    }

    private func present(_ update: AvailableUpdate) {
        let brew = UpdateCheck.installedByHomebrew()
        let running = UpdateCheck.runningVersion ?? "this version"
        let alert = NSAlert()
        alert.messageText = "Phonon \(update.version) is available"
        alert.informativeText =
            brew
            ? "You have \(running). Update copies \"\(UpdateCheck.brewCommand)\". Paste it "
                + "into Terminal, then open Phonon again."
            : "You have \(running). Update opens the release page. Download Phonon.dmg and "
                + "drag Phonon to Applications, replacing this copy."
        alert.addButton(withTitle: "Update")
        alert.addButton(withTitle: "Later")
        if brew { alert.addButton(withTitle: "Release Notes") }
        NSApp.activate(ignoringOtherApps: true)
        switch alert.runModal() {
        case .alertFirstButtonReturn:
            if brew {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(UpdateCheck.brewCommand, forType: .string)
            } else {
                NSWorkspace.shared.open(update.pageURL)
            }
        case .alertThirdButtonReturn:
            NSWorkspace.shared.open(update.pageURL)
        default:
            break
        }
    }

    private func presentUpToDate(_ running: String) {
        let alert = NSAlert()
        alert.messageText = "Phonon \(running) is up to date"
        alert.informativeText = "No newer release is on GitHub."
        alert.addButton(withTitle: "OK")
        NSApp.activate(ignoringOtherApps: true)
        alert.runModal()
    }

    private func presentFailure(_ error: Error) {
        let alert = NSAlert()
        alert.messageText = "Could not check for updates"
        alert.informativeText = error.localizedDescription
        alert.addButton(withTitle: "OK")
        NSApp.activate(ignoringOtherApps: true)
        alert.runModal()
    }
}
