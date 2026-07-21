// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "PhononBar",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "PhononBar",
            path: "Sources"
        ),
        .testTarget(
            name: "PhononBarTests",
            dependencies: ["PhononBar"],
            path: "Tests"
        ),
    ]
)
