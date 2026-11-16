// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "PhononBar",
    platforms: [.macOS(.v14)],
    targets: [
        .target(
            name: "PhononObjC",
            path: "ObjCSupport"
        ),
        .executableTarget(
            name: "PhononBar",
            dependencies: ["PhononObjC"],
            path: "Sources"
        ),
        .testTarget(
            name: "PhononBarTests",
            dependencies: ["PhononBar", "PhononObjC"],
            path: "Tests"
        ),
    ]
)
