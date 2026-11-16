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
    /// Shows a card that never takes focus (title, detail, button, its action). Never a
    /// modal alert: one used to sit in front of the user's app until answered, and a
    /// dictation started meanwhile had nowhere to go.
    var showMessage: (String, String, String?, @escaping () -> Void) -> Void = { _, _, _, _ in }

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

    /// What the update card says and does. Pure, so the wording is tested.
    static func offer(_ update: AvailableUpdate, running: String, brew: Bool)
        -> (title: String, detail: String, action: String)
    {
        (
            "Phonon \(update.version) is available",
            brew
                ? "You have \(running). Update copies the Homebrew command; paste it into "
                    + "Terminal, then open Phonon again."
                : "You have \(running). Update opens the release page for the new Phonon.dmg.",
            "Update"
        )
    }

    private func present(_ update: AvailableUpdate) {
        let brew = UpdateCheck.installedByHomebrew()
        let card = Self.offer(
            update, running: UpdateCheck.runningVersion ?? "this version", brew: brew)
        showMessage(card.title, card.detail, card.action) {
            if brew {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(UpdateCheck.brewCommand, forType: .string)
            } else {
                NSWorkspace.shared.open(update.pageURL)
            }
        }
    }

    private func presentUpToDate(_ running: String) {
        showMessage("Phonon \(running) is up to date", "No newer release is on GitHub.", nil) {}
    }

    private func presentFailure(_ error: Error) {
        showMessage("Could not check for updates", error.localizedDescription, nil) {}
    }
}
