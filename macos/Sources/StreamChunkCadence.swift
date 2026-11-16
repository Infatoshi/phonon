/// When the recorder sends the next stream chunk to the speech sidecar (GOAL M15).
///
/// The sidecar runs one round per chunk, so with 0.4 s chunks a word that becomes recognisable
/// waits up to 0.4 s for the next round. With the live text card on, the first 2 s go out in
/// 0.1 s chunks, then the usual 0.4 s. Headless replay through the real StreamTranscriber with
/// SCHEME (30 public LibriSpeech clips up to 8 s after 0.6 s of lead silence, MLX on the fast
/// Mac, 2 runs): first word p50 1,247/1,238 -> 1,144/1,135 ms, p90 1,656/1,645 -> 1,349/1,340,
/// paired median -157 ms both runs, later on 5 and 4 of 30; release text identical 30 of 30;
/// ASR compute +36 to +47% (the extra rounds are short). 0.2 s chunks gained only 11 ms.
/// Rounds after the opening match the fixed cadence, and the release window re-transcribes the
/// recording's end either way.
enum StreamChunkCadence {
    static let normalSeconds = 0.4
    static let earlySeconds = 0.1
    static let earlyUntilSeconds = 2.0

    static func shouldSend(sentFrames: Int, pendingFrames: Int, rate: Double, fastStart: Bool) -> Bool {
        if fastStart && Double(sentFrames) < rate * earlyUntilSeconds {
            return pendingFrames >= Int(rate * earlySeconds)
        }
        return pendingFrames >= Int(rate * normalSeconds)
    }
}
