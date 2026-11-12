import XCTest
import Darwin
@testable import Phonon

/// The refiner self-test as an XCTest (hosted in Phonon.app) for Firebase Test Lab: the same transcripts and
/// reps as Session.selftest(). The model ships inside this test bundle (copied in at build time from
/// PHONON_SELFTEST_MODEL). Each row is printed to the test log with the prefix "SELFTEST " and attached as
/// selftest.jsonl.
final class SelftestTests: XCTestCase {
    private func footprintMB() -> (Int, Int) {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return (-1, -1) }
        return (Int(info.phys_footprint >> 20), Int(info.resident_size_peak >> 20))
    }

    private func machine() -> String {
        var s = utsname()
        uname(&s)
        return withUnsafeBytes(of: &s.machine) { String(decoding: $0.prefix(while: { $0 != 0 }), as: UTF8.self) }
    }

    func testRefinerSelftest() throws {
        let bundle = Bundle(for: SelftestTests.self)
        let url = try XCTUnwrap(bundle.url(forResource: "refiner", withExtension: "gguf"), "refiner.gguf not in the test bundle")
        var rows: [[String: Any]] = []
        func emit(_ row: [String: Any]) {
            rows.append(row)
            let data = (try? JSONSerialization.data(withJSONObject: row, options: [.sortedKeys])) ?? Data()
            print("SELFTEST " + String(decoding: data, as: UTF8.self))
        }
        let p = ProcessInfo.processInfo
        emit(["device": machine(), "os": p.operatingSystemVersionString, "cores": p.processorCount,
              "active_cores": p.activeProcessorCount, "ram_mb": Int(p.physicalMemory >> 20),
              "thermal": p.thermalState.rawValue, "model_bytes": (try? FileManager.default.attributesOfItem(
                  atPath: url.path)[.size] as? Int) ?? -1])
        let threads = Refiner.defaultThreads
        let t0 = Date()
        let r = try XCTUnwrap(Refiner(path: url.path, threads: threads), "model failed to load")
        let (fp0, peak0) = footprintMB()
        emit(["load_ms": Int(Date().timeIntervalSince(t0) * 1000), "threads": r.threads,
              "footprint_mb": fp0, "resident_peak_mb": peak0])
        for (mode, raw) in Session.selftestCases {
            for rep in 0..<2 {
                let t = Date()
                let o = r.refine(raw, mode: mode)
                let (fp, peak) = footprintMB()
                emit(["rep": rep, "mode": mode, "total_ms": Int(Date().timeIntervalSince(t) * 1000),
                      "prefill_ms": o.prefillMs, "decode_ms": o.decodeMs, "prompt_tok": o.promptTokens,
                      "gen_tok": o.genTokens, "guard": o.guardFired, "out": o.text,
                      "footprint_mb": fp, "resident_peak_mb": peak, "thermal": p.thermalState.rawValue])
            }
        }
        let lines = rows.compactMap { try? JSONSerialization.data(withJSONObject: $0, options: [.sortedKeys]) }
            .map { String(decoding: $0, as: UTF8.self) }.joined(separator: "\n") + "\n"
        let att = XCTAttachment(data: Data(lines.utf8), uniformTypeIdentifier: "public.plain-text")
        att.name = "selftest.jsonl"
        att.lifetime = .keepAlways
        add(att)
        if let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            try? lines.write(to: dir.appendingPathComponent("selftest.jsonl"), atomically: true, encoding: .utf8)
        }
    }

    /// Shortlist-head experiment: if the bundle carries refiner-sl.gguf (ios/shortlist_gguf.py over the same model as
    /// refiner.gguf), its greedy output on the selftest transcripts must be token-identical to the full head.
    func testShortlistHeadTokenIdentical() throws {
        let bundle = Bundle(for: SelftestTests.self)
        let full = try XCTUnwrap(bundle.url(forResource: "refiner", withExtension: "gguf"))
        guard let slURL = bundle.url(forResource: "refiner-sl", withExtension: "gguf") else {
            throw XCTSkip("no refiner-sl.gguf in the test bundle")
        }
        func run(_ url: URL) throws -> (outs: [String], fallbacks: Int, steps: Int, shortlist: Bool) {
            let r = try XCTUnwrap(Refiner(path: url.path))
            var outs: [String] = [], fb = 0, steps = 0
            for (mode, raw) in Session.selftestCases {
                for _ in 0..<2 {
                    let o = r.refine(raw, mode: mode)
                    outs.append("\(o.genTokens) \(String(o.tokenHash, radix: 16))")
                    fb += o.fallbacks
                    steps += o.genTokens
                }
            }
            return (outs, fb, steps, r.shortlist)
        }
        let a = try run(full), b = try run(slURL)
        XCTAssertFalse(a.shortlist)
        XCTAssertTrue(b.shortlist)
        XCTAssertEqual(b.outs, a.outs)
        print("SHORTLIST full \(a.outs.joined(separator: ", ")); shortlist fallbacks \(b.fallbacks) of \(b.steps) steps")
    }

    /// The persistent pool with the patched worker QoS and repack chunking must give token-identical greedy output
    /// to stock ggml (per-graph threads, stock chunking), at 2, 4 and 5 threads.
    func testPersistentPoolTokenIdentical() throws {
        let bundle = Bundle(for: SelftestTests.self)
        let url = try XCTUnwrap(bundle.url(forResource: "refiner", withExtension: "gguf"), "refiner.gguf not in the test bundle")
        func run(threads: Int32, pool: Bool, stock: Bool) throws -> [String] {
            Refiner.setStock(stock)
            defer { Refiner.setStock(false) }
            let r = try XCTUnwrap(Refiner(path: url.path, threads: threads, persistentPool: pool))
            XCTAssertEqual(r.persistentPool, pool)
            return Session.selftestCases.flatMap { mode, raw in
                (0..<2).map { _ in
                    let o = r.refine(raw, mode: mode)
                    return "\(o.genTokens) \(String(o.tokenHash, radix: 16)) \(o.text)"
                }
            }
        }
        let stock = try run(threads: 4, pool: false, stock: true)
        for t: Int32 in [4, 5, 2] {
            XCTAssertEqual(try run(threads: t, pool: true, stock: false), stock, "pool, \(t) threads")
        }
        print("IDENTITY " + stock.map { $0.split(separator: " ").prefix(2).joined(separator: " ") }.joined(separator: ", "))
    }
}
