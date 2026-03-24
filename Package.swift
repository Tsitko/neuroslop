// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "neuroslop",
    platforms: [.macOS(.v15)],
    targets: [
        .target(
            name: "NeuroslopCore",
            path: "Sources/NeuroslopCore",
            swiftSettings: [.define("ACCELERATE_NEW_LAPACK")],
            linkerSettings: [.linkedFramework("Metal")]
        ),
        .executableTarget(
            name: "neuroslop",
            dependencies: ["NeuroslopCore"],
            path: "Sources/neuroslop"
        ),
        .executableTarget(
            name: "Benchmark",
            dependencies: ["NeuroslopCore"],
            path: "Sources/Benchmark"
        ),
        .executableTarget(
            name: "Tests",
            dependencies: ["NeuroslopCore"],
            path: "Tests/NeuroslopCoreTests"
        ),
    ]
)
