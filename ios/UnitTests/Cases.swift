import Foundation
import XCTest

/// The shared reference cases (crates/*/src/*_cases.json), bundled into the test bundle by path from project.yml.
enum Cases {
    private final class Anchor {}

    static func load(_ name: String, file: StaticString = #filePath, line: UInt = #line) throws -> [[String: Any]] {
        let bundle = Bundle(for: Anchor.self)
        let url = try XCTUnwrap(bundle.url(forResource: name, withExtension: "json"),
                                "\(name).json is not in the test bundle", file: file, line: line)
        let data = try Data(contentsOf: url)
        return try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [[String: Any]],
                             "\(name).json is not an array of objects", file: file, line: line)
    }
}
