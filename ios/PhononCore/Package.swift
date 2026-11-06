// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "PhononCore",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [.library(name: "PhononCore", targets: ["PhononCore"])],
    targets: [
        .target(name: "PhononCore"),
        .testTarget(name: "PhononCoreTests", dependencies: ["PhononCore"], resources: [.copy("golden.json")]),
    ]
)
