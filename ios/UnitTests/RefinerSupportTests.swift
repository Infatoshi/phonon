import XCTest

/// The A12 guard: the llama.cpp device slice assumes dot product and FP16 (armv8.4-a+dotprod+fp16), so the refiner
/// may load only when the kernel reports both. The sysctl read is injected; nothing here touches llama.cpp.
final class RefinerSupportTests: XCTestCase {
    private func cpu(_ keys: [String: Int]) -> RefinerSupport.CPU {
        RefinerSupport.read { keys[$0] }
    }

    func testA13AndNewerRefine() {
        let c = cpu(["hw.optional.arm.FEAT_DotProd": 1, "hw.optional.arm.FEAT_FP16": 1])
        XCTAssertEqual(c, RefinerSupport.CPU(dotProd: true, fp16: true))
        XCTAssertTrue(RefinerSupport.canRefine(c))
    }

    func testA12HasFP16ButNoDotProduct() {
        let c = cpu(["hw.optional.arm.FEAT_DotProd": 0, "hw.optional.arm.FEAT_FP16": 1])
        XCTAssertFalse(RefinerSupport.canRefine(c), "iPhone XS / XR: SDOT would be an illegal instruction")
    }

    func testMissingFP16() {
        XCTAssertFalse(RefinerSupport.canRefine(.init(dotProd: true, fp16: false)))
        XCTAssertFalse(RefinerSupport.canRefine(.init(dotProd: false, fp16: false)))
    }

    func testAbsentKeysCountAsMissing() {
        XCTAssertEqual(cpu([:]), RefinerSupport.CPU(dotProd: false, fp16: false))
        XCTAssertFalse(RefinerSupport.canRefine(cpu(["hw.optional.arm.FEAT_FP16": 1])))
    }

    func testAnyNonZeroValueIsPresent() {
        XCTAssertTrue(RefinerSupport.canRefine(cpu(["hw.optional.arm.FEAT_DotProd": 2, "hw.optional.arm.FEAT_FP16": 1])))
    }

    /// The real seam answers for a key every Darwin kernel has, and nil for one that does not exist.
    func testSysctlSeam() {
        XCTAssertNotNil(RefinerSupport.sysctl("hw.ncpu"))
        XCTAssertNil(RefinerSupport.sysctl("hw.optional.arm.FEAT_NoSuchThing"))
    }
}
