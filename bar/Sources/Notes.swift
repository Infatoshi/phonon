import AVFoundation
import AppKit
import SwiftUI

// MARK: - Meeting notes on disk

/// One recorded meeting as `MeetingRecorder` left it under
/// Meetings/<id>/: the metadata.json index plus closed 5 s WAV chunks of
/// the mic track (you) and the system track (the call). The app records
/// audio only; no transcript is stored, so the Notes page plays the audio.
struct MeetingNote: Identifiable, Equatable {
    let metadata: MeetingMetadata
    let directory: URL

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
    private var offsets: (mic: Int64, system: Int64) {
        guard !metadata.micChunks.isEmpty, !metadata.systemChunks.isEmpty else {
            return (0, 0)
        }
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
            return MeetingNote(metadata: metadata, directory: folder)
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
    private var loadGeneration = 0
    private var startTask: Task<Void, Never>?

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
            if position >= duration - 0.05 {
                position = 0
                player.seek(to: .zero, toleranceBefore: .zero, toleranceAfter: .zero)
            }
            player.play()
            playing = true
            return
        }
        guard startTask == nil else { return }
        let generation = loadGeneration
        startTask = Task { [weak self] in
            await self?.start(note, generation: generation)
        }
    }

    func seek(_ note: MeetingNote, to fraction: Double) {
        load(note)
        position = max(0, min(1, fraction)) * duration
        player?.seek(
            to: CMTime(seconds: position, preferredTimescale: 600), toleranceBefore: .zero,
            toleranceAfter: .zero)
    }

    func stop() {
        loadGeneration += 1
        startTask?.cancel()
        startTask = nil
        player?.pause()
        if let observer { player?.removeTimeObserver(observer) }
        observer = nil
        player = nil
        playing = false
    }

    func start(_ note: MeetingNote, generation: Int? = nil) async {
        let token: Int
        if let generation {
            guard !Task.isCancelled, loadGeneration == generation, noteID == note.id else { return }
            token = generation
        } else {
            guard !Task.isCancelled else { return }
            load(note)
            loadGeneration += 1
            token = loadGeneration
        }
        defer {
            if loadGeneration == token {
                startTask = nil
            }
        }
        guard let item = await Self.composition(note) else {
            if loadGeneration == token, noteID == note.id {
                failed = true
            }
            return
        }
        guard !Task.isCancelled, loadGeneration == token, noteID == note.id else { return }
        let newPlayer = AVPlayer(playerItem: AVPlayerItem(asset: item))
        if position > 0 {
            if position >= duration - 0.05 {
                position = 0
                await newPlayer.seek(to: .zero, toleranceBefore: .zero, toleranceAfter: .zero)
            } else {
                await newPlayer.seek(to: CMTime(seconds: position, preferredTimescale: 600))
            }
            guard !Task.isCancelled, loadGeneration == token, noteID == note.id else { return }
        }
        if let observer { player?.removeTimeObserver(observer) }
        observer = nil
        player?.pause()
        self.player = newPlayer
        observer = newPlayer.addPeriodicTimeObserver(
            forInterval: CMTime(value: 1, timescale: 20), queue: .main
        ) { [weak self] time in
            MainActor.assumeIsolated {
                guard let self, self.loadGeneration == token else { return }
                self.position = time.seconds
                if self.position >= self.duration - 0.05 {
                    self.playing = false
                }
            }
        }
        newPlayer.play()
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
                    NoteDetail(note: note).id(note.id)
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
    @StateObject private var player = MeetingPlayer()
    @State private var peaks: [Float] = []

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
                    // No transcript is stored for meetings yet; the button
                    // stays where the mock puts it and says so.
                    Button {} label: { Image(systemName: "doc.on.doc") }
                        .buttonStyle(IconButtonStyle())
                        .disabled(true)
                        .help("No transcript stored for this meeting")
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

                // What the folder holds: the two tracks. Speaker labels and
                // timestamps need a transcript, which meetings do not have yet.
                trackLine(label: "You", color: EmberTheme.ember2, track: .mic, help: "Your microphone")
                trackLine(
                    label: "Call", color: Color(hex: 0x9CC7B8), track: .system,
                    help: "The call's audio (everyone else)")
                Rectangle().fill(Color(hex: 0x191512)).frame(height: 1)
                Text("No transcript yet: Phonon keeps this meeting's audio.")
                    .foregroundStyle(EmberTheme.mute)
                    .padding(.top, 12)
                    .help(
                        "Meetings are recorded on this Mac as two tracks, your mic and the call. "
                            + "Transcripts are not made in the app yet.")
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
