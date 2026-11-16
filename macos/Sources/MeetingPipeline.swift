import Foundation

/// After a call: the recording stops, its transcript is made, then it is cleaned, with no click
/// (GOAL N5). The transcriber and the cleaner each keep their own progress file, so a quit at
/// any point resumes at the next launch; this joins the two so the hand-off between them
/// survives a quit too. The Notes buttons stay as retries.
@MainActor
enum MeetingPipeline {
    /// A finished transcript goes to the cleaner (which waits for dictation to be quiet and for
    /// its memory gate), and `changed` reloads Notes after each step.
    static func connect(
        transcriber: MeetingTranscriber, cleaner: MeetingCleaner, meetings: URL,
        changed: @escaping @MainActor () -> Void
    ) {
        transcriber.onFinished = { [weak cleaner] id in
            changed()
            cleaner?.enqueue(directory: meetings.appendingPathComponent(id, isDirectory: true))
        }
        cleaner.onFinished = { _ in changed() }
    }

    /// At launch: transcripts a quit interrupted, transcripts finished but not yet handed to the
    /// cleaner, then cleanups a quit interrupted.
    static func resume(transcriber: MeetingTranscriber, cleaner: MeetingCleaner, meetings: URL) {
        transcriber.resumePending(in: meetings)
        cleaner.resumePending(in: meetings)
    }
}
