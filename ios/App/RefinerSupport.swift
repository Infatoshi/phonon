import Foundation

/// Whether this iPhone's CPU can run the refiner at all. The llama.cpp device slice is compiled with
/// -DGGML_CPU_ARM_ARCH=armv8.4-a+dotprod+fp16 (ios/llama-b12132-phonon.patch, ios/build_llama.sh), so its Q4_K
/// matmuls use SDOT and its fp16 conversions use FP16 instructions with no run-time fallback. The deployment target
/// is iOS 18.0, which still runs on A12 phones (iPhone XS, XS Max, XR): they have FP16 but no dot product, and the
/// first matmul would die on an illegal instruction. On such a phone the refiner is never loaded or called; the
/// recogniser's text goes out with the dictionary clean-up only (Session.refineAndDeliver's no-refiner path).
/// The recogniser (Parakeet on CoreML through FluidAudio) does not use llama.cpp and is unaffected.
enum RefinerSupport {

    /// The CPU features the refiner build assumes, as the kernel reports them.
    struct CPU: Equatable {
        var dotProd: Bool
        var fp16: Bool
    }

    /// The decision: the refiner may load only when the CPU has every feature the llama.cpp build was compiled for.
    static func canRefine(_ cpu: CPU) -> Bool {
        cpu.dotProd && cpu.fp16
    }

    /// Reads the features through `query` (a sysctl name in, its integer value out, nil when the key is absent).
    /// An absent key counts as a missing feature, so an unknown CPU never enters llama.cpp.
    static func read(_ query: (String) -> Int?) -> CPU {
        CPU(dotProd: (query("hw.optional.arm.FEAT_DotProd") ?? 0) != 0,
            fp16: (query("hw.optional.arm.FEAT_FP16") ?? 0) != 0)
    }

    /// The real query: sysctlbyname for a 32-bit integer key.
    static func sysctl(_ name: String) -> Int? {
        var value: Int32 = 0
        var size = MemoryLayout<Int32>.size
        guard sysctlbyname(name, &value, &size, nil, 0) == 0 else { return nil }
        return Int(value)
    }

    /// This device's answer, read once per process.
    static let onThisDevice: Bool = canRefine(read(sysctl))

    /// The settings line shown instead of the clean-up toggle's description when the refiner cannot run here.
    static let offNote = "Correction is off on this iPhone model. You get the transcript with your dictionary applied."
}
