import AVFoundation
import XCTest

@testable import PhononBar

/// Records a 48 kHz WAV the way the meeting recorder records a 48 kHz microphone, opt in only:
///
///     PHONON_RATE_CHECK_IN=<48 kHz mono WAV> PHONON_RATE_CHECK_OUT=<empty folder> \
///         swift test --filter MeetingFormatLiveCheck
///
/// The file is fed in 4,096-frame buffers (the tap's size) through `MeetingRecorder.convert`
/// into a `ChunkedWavWriter` at the track rate, so the chunks under the out folder are what a
/// recording made today would hold. Used to check that the transcript of a fixture is the same
/// from these chunks as from 48 kHz chunks (GOAL N4); prints counts only.
final class MeetingFormatLiveCheck: XCTestCase {
    func testRecordAFixtureAtTheTrackRate() throws {
        let env = ProcessInfo.processInfo.environment
        guard let inPath = env["PHONON_RATE_CHECK_IN"], let outPath = env["PHONON_RATE_CHECK_OUT"] else {
            throw XCTSkip("set PHONON_RATE_CHECK_IN and PHONON_RATE_CHECK_OUT to run")
        }
        let file = try AVAudioFile(forReading: URL(fileURLWithPath: inPath))
        let source = file.processingFormat
        let target = try XCTUnwrap(
            AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: Double(MeetingFormat.sampleRate), channels: 1,
                interleaved: false))
        let converter = try XCTUnwrap(AVAudioConverter(from: source, to: target))
        var closed: [MeetingChunk] = []
        let writer = try ChunkedWavWriter(
            directory: URL(fileURLWithPath: outPath, isDirectory: true).appendingPathComponent("mic"),
            onChunkClosed: { closed.append($0) })
        var read = 0
        while file.framePosition < file.length {
            let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: source, frameCapacity: 4_096))
            try file.read(into: buffer, frameCount: 4_096)
            read += Int(buffer.frameLength)
            try writer.append(try XCTUnwrap(MeetingRecorder.convert(buffer, with: converter, to: target)))
        }
        try writer.finish()
        let written = closed.map(\.sampleCount).reduce(0, +)
        print("rate check: \(read) frames at \(Int(source.sampleRate)) Hz -> \(written) at \(MeetingFormat.sampleRate) Hz in \(closed.count) chunks")
        XCTAssertEqual(Double(written), Double(read) * target.sampleRate / source.sampleRate, accuracy: 64)
    }
}
