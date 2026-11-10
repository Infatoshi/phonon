import AVFoundation
import AppKit
import SwiftUI

// MARK: - Meeting notes on disk

/// One recorded meeting as `MeetingRecorder` left it under
/// Meetings/<id>/: the metadata.json index plus closed 5 s WAV chunks of
/// the mic track (you) and the system track (the call), and once it has been
/// transcribed, transcript.json (MeetingTranscript.swift).
struct MeetingNote: Identifiable, Equatable {
    let metadata: MeetingMetadata
    let directory: URL
    var transcript: MeetingTranscript?
    /// Why the last transcription stopped, from transcript.partial.json.
    var transcriptError: String?
    /// The cleaned transcript, transcript.cleaned.json (MeetingCleanup.swift).
    var cleaned: CleanedTranscript?
    /// Why the last cleanup stopped or was skipped, from transcript.cleaned.partial.json.
    var cleanupError: String?

    init(
        metadata: MeetingMetadata, directory: URL, transcript: MeetingTranscript? = nil,
        transcriptError: String? = nil, cleaned: CleanedTranscript? = nil, cleanupError: String? = nil
    ) {
        self.metadata = metadata
        self.directory = directory
        self.transcript = transcript
        self.transcriptError = transcriptError
        self.cleaned = cleaned
        self.cleanupError = cleanupError
    }

    var id: String { metadata.id }

    var startedAt: Date? {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.date(from: metadata.startedAt)
            ?? ISO8601DateFormatter().date(from: metadata.startedAt)
    }

    var title: String {
        if let title = metadata.windowTitle?.trimmingCharacters(in: .whitespaces), !title.isEmpty {
            return title
        }
        if let app = metadata.appName, !app.isEmpty { return "\(app) call" }
        return "Meeting"
    }

    var appName: String {
        AppIcons.nameNonisolated(metadata.appBundleID, fallback: metadata.appName)
    }

    /// Where each track's sample 0 sits on the shared timeline, shifted so the
    /// earliest track starts at 0.
    private var offsets: (mic: Int64, system: Int64) { Self.timelineOffsets(metadata) }

    static func timelineOffsets(_ metadata: MeetingMetadata) -> (mic: Int64, system: Int64) {
        let mic = metadata.micStartOffsetSamples ?? 0
        let system = metadata.systemStartOffsetSamples ?? 0
        let base = min(mic, system)
        return (mic - base, system - base)
    }

    func samples(_ track: MeetingTrack) -> Int64 {
        let chunks = track == .mic ? metadata.micChunks : metadata.systemChunks
        return chunks.map { $0.startSample + $0.sampleCount }.max() ?? 0
    }

    func trackSeconds(_ track: MeetingTrack) -> Double {
        Double(samples(track)) / Double(max(1, metadata.sampleRate))
    }

    var durationSeconds: Double {
        let rate = Double(max(1, metadata.sampleRate))
        let mic = metadata.micChunks.isEmpty ? 0 : Double(offsets.mic + samples(.mic)) / rate
        let system =
            metadata.systemChunks.isEmpty ? 0 : Double(offsets.system + samples(.system)) / rate
        return max(mic, system)
    }

    /// Every closed chunk with its start on the shared timeline, in seconds.
    func chunkFiles() -> [(track: MeetingTrack, url: URL, start: Double, samples: Int64)] {
        let rate = Double(max(1, metadata.sampleRate))
        let mic = metadata.micChunks.map {
            (
                MeetingTrack.mic, directory.appendingPathComponent("mic/\($0.file)"),
                Double(offsets.mic + $0.startSample) / rate, $0.sampleCount
            )
        }
        let system = metadata.systemChunks.map {
            (
                MeetingTrack.system, directory.appendingPathComponent("system/\($0.file)"),
                Double(offsets.system + $0.startSample) / rate, $0.sampleCount
            )
        }
        return (mic + system).map { (track: $0.0, url: $0.1, start: $0.2, samples: $0.3) }
    }

    static func loadAll(from root: URL, fileManager: FileManager = .default) -> [MeetingNote] {
        let folders =
            (try? fileManager.contentsOfDirectory(
                at: root, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])) ?? []
        let decoder = JSONDecoder()
        return folders.compactMap { folder in
            guard
                let data = try? Data(contentsOf: folder.appendingPathComponent("metadata.json")),
                let metadata = try? decoder.decode(MeetingMetadata.self, from: data),
                !(metadata.micChunks.isEmpty && metadata.systemChunks.isEmpty)
            else { return nil }
            return MeetingNote(
                metadata: metadata, directory: folder,
                transcript: MeetingTranscript.load(from: folder),
                transcriptError: MeetingTranscriptProgress.load(from: folder)?.error,
                cleaned: CleanedTranscript.load(from: folder),
                cleanupError: MeetingCleanupProgress.load(from: folder)?.error)
        }
        .sorted { ($0.startedAt ?? .distantPast) > ($1.startedAt ?? .distantPast) }
    }
}

extension AppIcons {
    /// The app name without the main-actor cache, for model code.
    nonisolated static func nameNonisolated(_ bundleID: String?, fallback: String?) -> String {
        if let fallback, !fallback.isEmpty { return fallback }
        guard let bundleID, !bundleID.isEmpty else { return "Unknown app" }
        return NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundleID)
            .map {
                FileManager.default.displayName(atPath: $0.path)
                    .replacingOccurrences(of: ".app", with: "")
            } ?? bundleID
    }
}

enum MeetingWaveform {
    /// Peak level per bar over the whole meeting, both tracks together, 0...1.
    /// Reads every chunk once, sampling every 16th sample.
    static func peaks(_ note: MeetingNote, bars: Int) -> [Float] {
        let duration = note.durationSeconds
        guard duration > 0, bars > 0 else { return [] }
        let rate = Double(max(1, note.metadata.sampleRate))
        var peaks = [Float](repeating: 0, count: bars)
        let stride = 16
        for chunk in note.chunkFiles() {
            guard let data = try? Data(contentsOf: chunk.url, options: .mappedIfSafe),
                data.count > 44
            else { continue }
            data.withUnsafeBytes { raw in
                let body = UnsafeRawBufferPointer(rebasing: raw[44...])
                let count = body.count / 2
                var index = 0
                while index < count {
                    let value = Int16(
                        littleEndian: body.loadUnaligned(fromByteOffset: index * 2, as: Int16.self))
                    let seconds = chunk.start + Double(index) / rate
                    let bar = min(bars - 1, Int(seconds / duration * Double(bars)))
                    let level = Float(abs(Int32(value))) / 32_768
                    if level > peaks[bar] { peaks[bar] = level }
                    index += stride
                }
            }
        }
        let top = peaks.max() ?? 0
        guard top > 0 else { return peaks }
        return peaks.map { $0 / top }
    }
}

/// Plays a meeting: both tracks mixed on the shared timeline.
@MainActor
final class MeetingPlayer: ObservableObject {
    @Published private(set) var playing = false
    @Published private(set) var position: Double = 0
    @Published private(set) var failed = false
    private(set) var noteID: String?
    private var player: AVPlayer?
    private var observer: Any?
    private var duration: Double = 0

    func load(_ note: MeetingNote) {
        guard note.id != noteID else { return }
        stop()
        noteID = note.id
        duration = note.durationSeconds
        position = 0
        failed = false
        player = nil
    }

    func toggle(_ note: MeetingNote) {
        load(note)
        if playing {
            player?.pause()
            playing = false
            return
        }
        if let player {
            player.play()
            playing = true
            return
        }
        Task { await start(note) }
    }

    func seek(_ note: MeetingNote, to fraction: Double) {
        load(note)
        position = max(0, min(1, fraction)) * duration
        player?.seek(
            to: CMTime(seconds: position, preferredTimescale: 600), toleranceBefore: .zero,
            toleranceAfter: .zero)
    }

    func stop() {
        player?.pause()
        if let observer { player?.removeTimeObserver(observer) }
        observer = nil
        playing = false
    }

    private func start(_ note: MeetingNote) async {
        guard let item = await Self.composition(note) else {
            failed = true
            return
        }
        guard noteID == note.id else { return }
        let player = AVPlayer(playerItem: AVPlayerItem(asset: item))
        self.player = player
        observer = player.addPeriodicTimeObserver(
            forInterval: CMTime(value: 1, timescale: 20), queue: .main
        ) { [weak self] time in
            MainActor.assumeIsolated {
                guard let self else { return }
                self.position = time.seconds
                if self.position >= self.duration - 0.05 {
                    self.playing = false
                }
            }
        }
        if position > 0 {
            await player.seek(to: CMTime(seconds: position, preferredTimescale: 600))
        }
        player.play()
        playing = true
    }

    /// The meeting's chunks laid on two composition tracks (mic, call); the
    /// player mixes them.
    static func composition(_ note: MeetingNote) async -> AVComposition? {
        let composition = AVMutableComposition()
        var tracks: [MeetingTrack: AVMutableCompositionTrack] = [:]
        for chunk in note.chunkFiles() {
            let asset = AVURLAsset(url: chunk.url)
            guard let source = try? await asset.loadTracks(withMediaType: .audio).first,
                let range = try? await source.load(.timeRange)
            else { continue }
            let track =
                tracks[chunk.track]
                ?? composition.addMutableTrack(
                    withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid)
            guard let track else { continue }
            tracks[chunk.track] = track
            try? track.insertTimeRange(
                range, of: source, at: CMTime(seconds: chunk.start, preferredTimescale: 48_000))
        }
        return tracks.isEmpty ? nil : composition
    }
}

// MARK: - Notes page

struct NotesView: View {
    @ObservedObject var store: NativeAppStore
    @State private var selectedID: String?

    init(store: NativeAppStore, selectedID: String? = nil) {
        self.store = store
        _selectedID = State(initialValue: selectedID)
    }

    private var selected: MeetingNote? {
        store.meetings.first { $0.id == selectedID } ?? store.meetings.first
    }

    var body: some View {
        HStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Notes")
                        .font(.system(size: 22, weight: .semibold))
                        .tracking(-0.44)
                        .padding(.horizontal, 10)
                        .padding(.bottom, 12)
                    ForEach(store.meetings) { note in
                        NoteListRow(note: note, selected: note.id == selected?.id)
                            .onTapGesture { withAnimation(Motion.quick) { selectedID = note.id } }
                    }
                    if store.meetings.isEmpty {
                        Text("No meetings yet.")
                            .foregroundStyle(EmberTheme.mute)
                            .padding(.horizontal, 10)
                            .help(
                                "Turn on Settings › Notes › Ask to take notes when a meeting "
                                    + "starts, then click Record when a call begins.")
                    }
                }
                .padding(.horizontal, 14)
                .padding(.top, 24)
            }
            .frame(width: 280)
            .overlay(alignment: .trailing) {
                Rectangle().fill(EmberTheme.line).frame(width: 1)
            }

            Group {
                if let note = selected {
                    NoteDetail(
                        note: note, transcript: store.transcriptState(note),
                        cleanup: store.cleanupState(note),
                        cleanupAvailable: store.meetingCleanupAvailable,
                        onTranscribe: { store.transcribeMeeting(note) },
                        onCleanUp: { store.cleanMeeting(note) }
                    )
                    .id(note.id)
                } else {
                    Color.clear
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .onAppear { store.loadMeetings() }
    }
}

struct NoteListRow: View {
    let note: MeetingNote
    let selected: Bool
    @State private var hovering = false

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            AppIconView(bundleID: note.metadata.appBundleID).padding(.top, 1)
            VStack(alignment: .leading, spacing: 1) {
                Text(note.title).font(.system(size: 13.5, weight: .medium)).lineLimit(1)
                Text(NoteDetail.subtitle(note))
                    .font(.mono(12))
                    .foregroundStyle(EmberTheme.mute)
                    .lineLimit(1)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 9)
        .background(
            selected ? Color(hex: 0x1D1713) : (hovering ? EmberTheme.hover : .clear),
            in: RoundedRectangle(cornerRadius: 8, style: .continuous)
        )
        .contentShape(Rectangle())
        .onHover { hovering = $0 }
        .animation(Motion.quick, value: hovering)
    }
}

struct NoteDetail: View {
    let note: MeetingNote
    var transcript: NoteTranscriptState = .notTranscribed
    var cleanup: NoteCleanupState = .none
    var cleanupAvailable = false
    var onTranscribe: () -> Void = {}
    var onCleanUp: () -> Void = {}
    @StateObject private var player = MeetingPlayer()
    @State private var peaks: [Float] = []
    /// Show the speech model's raw turns instead of the cleaned ones.
    @State private var showRaw = false

    /// The lines listed (and copied) now.
    private func rows(_ ready: MeetingTranscript) -> [NoteTranscriptRows.Row] {
        NoteTranscriptRows.rows(ready, cleanup: cleanup, showRaw: showRaw)
    }

    static func subtitle(_ note: MeetingNote) -> String {
        let when = note.startedAt.map { FeedTime.label($0) } ?? ""
        let today = note.startedAt.map { Calendar.current.isDateInToday($0) } ?? false
        let duration = FeedTime.duration(seconds: note.durationSeconds)
        return [today ? "Today \(when)" : when, duration].filter { !$0.isEmpty }
            .joined(separator: " · ")
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                HStack(alignment: .top, spacing: 12) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(note.title).font(.system(size: 19, weight: .semibold))
                        Text(Self.subtitle(note) + " · " + note.appName)
                            .font(.mono(12.5))
                            .foregroundStyle(EmberTheme.mute)
                    }
                    Spacer()
                    if case .ready(let ready) = transcript, !ready.turns.isEmpty {
                        CopyIconButton(
                            text: NoteTranscriptRows.plainText(rows(ready)), help: "Copy transcript")
                    } else {
                        Button {} label: { Image(systemName: "doc.on.doc") }
                            .buttonStyle(IconButtonStyle())
                            .disabled(true)
                            .help("No transcript to copy yet")
                    }
                }

                HStack(spacing: 12) {
                    Button { player.toggle(note) } label: {
                        Image(systemName: player.playing ? "pause.fill" : "play.fill")
                            .font(.system(size: 13))
                            .foregroundStyle(Color(hex: 0x160B05))
                            .frame(width: 34, height: 34)
                            .background(EmberTheme.ember, in: Circle())
                            .contentShape(Circle())
                    }
                    .buttonStyle(PressScaleStyle())
                    .help(player.playing ? "Pause" : "Play audio")

                    Scrubber(
                        peaks: peaks,
                        progress: note.durationSeconds > 0
                            ? player.position / note.durationSeconds : 0
                    ) { fraction in player.seek(note, to: fraction) }
                    .frame(height: 28)

                    Text(
                        "\(FeedTime.clock(player.position)) / \(FeedTime.clock(note.durationSeconds))"
                    )
                    .font(.mono(12))
                    .foregroundStyle(EmberTheme.mute)
                    .fixedSize()
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(
                    EmberTheme.panel, in: RoundedRectangle(cornerRadius: 12, style: .continuous)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .stroke(EmberTheme.panelBorder, lineWidth: 1)
                )
                .padding(.top, 14)
                .padding(.bottom, 8)

                if player.failed {
                    Text("This meeting's audio could not be opened.")
                        .foregroundStyle(EmberTheme.warm)
                        .padding(.vertical, 6)
                }

                // What the folder holds: the two tracks.
                trackLine(label: "You", color: Self.meColor, track: .mic, help: "Your microphone")
                trackLine(
                    label: "Call", color: Self.callColor, track: .system,
                    help: "The call's audio (everyone else)")
                Rectangle().fill(Color(hex: 0x191512)).frame(height: 1)
                transcriptSection.padding(.top, 12)
            }
            .padding(.horizontal, 24)
            .padding(.top, 24)
            .padding(.bottom, 20)
        }
        .task(id: note.id) {
            player.load(note)
            let note = note
            peaks = await Task.detached(priority: .utility) {
                MeetingWaveform.peaks(note, bars: 90)
            }.value
        }
        .onDisappear { player.stop() }
    }

    static let meColor = EmberTheme.ember2
    static let callColor = Color(hex: 0x9CC7B8)

    @ViewBuilder private var transcriptSection: some View {
        switch transcript {
        case .ready(let ready) where ready.turns.isEmpty:
            Text("No speech was heard in this meeting.")
                .foregroundStyle(EmberTheme.mute)
        case .ready(let ready):
            VStack(alignment: .leading, spacing: 12) {
                cleanupBar
                LazyVStack(alignment: .leading, spacing: 12) {
                    ForEach(Array(rows(ready).enumerated()), id: \.offset) { _, row in
                        turnRow(row)
                    }
                }
            }
        case .transcribing(let progress):
            HStack(spacing: 10) {
                ProgressView().controlSize(.small)
                Text(
                    progress.map { "Transcribing… \(Int(($0 * 100).rounded()))%" }
                        ?? "Transcribing… (waiting for the speech model)"
                )
                .foregroundStyle(EmberTheme.mute)
            }
            .help("Transcribed on this Mac, a minute at a time; dictation always goes first.")
        case .recording:
            Text("Recording. The transcript is made when the meeting ends.")
                .foregroundStyle(EmberTheme.mute)
        case .notTranscribed, .failed:
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 12) {
                    Button("Transcribe", action: onTranscribe)
                        .buttonStyle(PillButtonStyle(primary: true))
                        .help("Transcribe this meeting on this Mac")
                    Text("No transcript yet.").foregroundStyle(EmberTheme.mute)
                }
                if case .failed(let message) = transcript {
                    Text("The last try stopped: \(message)")
                        .font(.system(size: 12.5))
                        .foregroundStyle(EmberTheme.warm)
                        .textSelection(.enabled)
                }
            }
        }
    }

    /// Above the transcript: whether it is cleaned, the Raw / Cleaned switch, the cleanup's
    /// progress, or the button that starts one.
    @ViewBuilder private var cleanupBar: some View {
        switch cleanup {
        case .cleaned(let cleaned):
            HStack(spacing: 10) {
                Picker("", selection: $showRaw) {
                    Text("Cleaned").tag(false)
                    Text("Raw").tag(true)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()
                .help("Cleaned: fillers and false starts removed on this Mac. Raw: the speech model's words.")
                Text(Self.cleanupSummary(cleaned))
                    .font(.system(size: 12.5))
                    .foregroundStyle(EmberTheme.mute)
            }
        case .cleaning(let progress, let paused):
            HStack(spacing: 10) {
                ProgressView().controlSize(.small)
                Text(Self.cleaningLabel(progress: progress, paused: paused))
                    .foregroundStyle(EmberTheme.mute)
            }
            .help("Cleaned on this Mac in the background; dictation always goes first.")
        case .failed(let message):
            HStack(spacing: 12) {
                if cleanupAvailable {
                    Button("Clean up", action: onCleanUp)
                        .buttonStyle(PillButtonStyle(primary: false))
                        .help("Clean this transcript on this Mac")
                }
                Text(message)
                    .font(.system(size: 12.5))
                    .foregroundStyle(EmberTheme.warm)
                    .textSelection(.enabled)
            }
        case .none:
            if cleanupAvailable {
                Button("Clean up", action: onCleanUp)
                    .buttonStyle(PillButtonStyle(primary: false))
                    .help("Remove fillers and false starts on this Mac; the raw transcript is kept")
            }
        }
    }

    static func cleanupSummary(_ cleaned: CleanedTranscript) -> String {
        cleaned.rawChunks == 0
            ? "Cleaned on this Mac."
            : "Cleaned on this Mac; \(cleaned.rawChunks) of \(cleaned.chunks) parts kept as heard."
    }

    static func cleaningLabel(progress: Double?, paused: Bool) -> String {
        if paused { return "Cleaning up… (paused while you dictate)" }
        return progress.map { "Cleaning up… \(Int(($0 * 100).rounded()))%" } ?? "Cleaning up… (waiting its turn)"
    }

    private func turnRow(_ turn: NoteTranscriptRows.Row) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Button {
                let duration = note.durationSeconds
                guard duration > 0 else { return }
                player.seek(note, to: turn.startSeconds / duration)
            } label: {
                Text(FeedTime.clock(turn.startSeconds))
                    .font(.mono(12))
                    .foregroundStyle(EmberTheme.dim)
                    .frame(width: 46, alignment: .leading)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("Play from here")
            Text(turn.speaker.label)
                .font(.system(size: 12.5, weight: .semibold))
                .foregroundStyle(turn.speaker == .me ? Self.meColor : Self.callColor)
                .frame(width: 70, alignment: .leading)
            Text(turn.text)
                .foregroundStyle(EmberTheme.soft)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func trackLine(label: String, color: Color, track: MeetingTrack, help: String)
        -> some View
    {
        let chunks = track == .mic ? note.metadata.micChunks : note.metadata.systemChunks
        return VStack(spacing: 0) {
            Rectangle().fill(Color(hex: 0x191512)).frame(height: 1)
            HStack(spacing: 10) {
                Text(chunks.isEmpty ? "--:--" : FeedTime.clock(note.trackSeconds(track)))
                    .font(.mono(12))
                    .foregroundStyle(EmberTheme.dim)
                    .frame(width: 46, alignment: .leading)
                Text(label)
                    .font(.system(size: 12.5, weight: .semibold))
                    .foregroundStyle(color)
                    .frame(width: 70, alignment: .leading)
                Text(
                    chunks.isEmpty
                        ? "not recorded"
                        : (track == .mic ? (note.metadata.micDevice ?? "Microphone") : "Call audio")
                )
                .foregroundStyle(EmberTheme.soft)
                Spacer()
            }
            .padding(.vertical, 7)
            .help(help)
        }
    }
}

private struct PressScaleStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.92 : 1)
            .animation(.easeOut(duration: 0.08), value: configuration.isPressed)
    }
}

/// The waveform: bars up to the play head in ember; click or drag to seek.
struct Scrubber: View {
    let peaks: [Float]
    let progress: Double
    let onSeek: (Double) -> Void

    var body: some View {
        GeometryReader { geometry in
            let bars = peaks.isEmpty ? [Float](repeating: 0.08, count: 90) : peaks
            HStack(alignment: .center, spacing: 2) {
                ForEach(bars.indices, id: \.self) { index in
                    let played = Double(index) / Double(bars.count) < progress
                    RoundedRectangle(cornerRadius: 1.5)
                        .fill(played ? EmberTheme.ember2 : Color(hex: 0x3A2C22))
                        .frame(height: max(3, geometry.size.height * CGFloat(0.12 + 0.88 * bars[index])))
                        .frame(maxWidth: .infinity)
                }
            }
            .frame(maxHeight: .infinity)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0).onChanged { value in
                    onSeek(Double(value.location.x / max(1, geometry.size.width)))
                }
            )
        }
    }
}
