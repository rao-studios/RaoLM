// swift-tools-version: 6.0
// RaoLM — a SmolLM2-shaped transformer with citations baked in.
//
// RaoLM pretrains a small decoder on the corpus a Thread node governs and generates
// text whose tokens carry citations back to the exact Thread source (document id,
// partition index, token offset). Training records an entropy ledger per step and per
// epoch; a provenance index couples the model's logits to corpus positions (kNN-LM
// style); a manifest hashes weights, index, corpus snapshot and tokenizer together.
//
// Layout (one package, several library targets):
//   RaoLMCore        corpus model, hashing, synthetic corpus, snapshot, manifests, ledger rows,
//                    citation schemas and span aggregation                    (Foundation + Crypto)
//   RaoLMModel       the SmolLM2-shaped transformer, tokenizer, checkpoints     (MLX via Frigate)
//   RaoLMTraining    tokenized corpus, pretraining loop, ledger, provenance indexer (MLX)
//   RaoLMProvenance  provenance index, logit mixing, cited generation, verifier, eval (MLX)
//   RaoLMThread      gRPC client for a Thread node, and a host that runs one   (Conduit / gRPC)
//   RaoLM            umbrella that re-exports the above
//   RaoLMCLI         the `raolm` executable
//
// Frigate is taken by path, as every consumer in the family does. The gRPC packages are
// the same URLs Conduit uses, so SwiftPM unifies them into one GRPCCore.

import PackageDescription

// MLX-facing targets use Swift 5 language mode because Frigate's MLX types are not
// Sendable (the same choice Fleet and Zehn make). Foundation-only and gRPC targets stay
// in Swift 6 mode and get strict concurrency checking.
let v5: [SwiftSetting] = [.swiftLanguageMode(.v5)]

let package = Package(
    name: "RaoLM",
    platforms: [.macOS("15.0")],
    products: [
        .library(name: "RaoLM", targets: ["RaoLM"]),
        .library(name: "RaoLMCore", targets: ["RaoLMCore"]),
        .library(name: "RaoLMModel", targets: ["RaoLMModel"]),
        .library(name: "RaoLMTraining", targets: ["RaoLMTraining"]),
        .library(name: "RaoLMProvenance", targets: ["RaoLMProvenance"]),
        .library(name: "RaoLMThread", targets: ["RaoLMThread"]),
        .executable(name: "raolm", targets: ["RaoLMCLI"]),
    ],
    dependencies: [
        .package(path: "../Frigate"),
        .package(path: "../Conduit"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
        .package(url: "https://github.com/grpc/grpc-swift.git", from: "2.0.0"),
        .package(url: "https://github.com/grpc/grpc-swift-nio-transport.git", from: "1.0.0"),
        .package(url: "https://github.com/apple/swift-crypto.git", from: "4.2.0"),
    ],
    targets: [
        .target(
            name: "RaoLMCore",
            dependencies: [.product(name: "Crypto", package: "swift-crypto")]
        ),
        .target(
            name: "RaoLMModel",
            dependencies: [
                "RaoLMCore",
                .product(name: "MLX", package: "Frigate"),
                .product(name: "MLXNN", package: "Frigate"),
                .product(name: "MLXLMCommon", package: "Frigate"),
                .product(name: "MLXLLM", package: "Frigate"),
                .product(name: "FrigateTokenizers", package: "Frigate"),
                .product(name: "FrigateHub", package: "Frigate"),
            ],
            resources: [.copy("Resources/Tokenizer")],
            swiftSettings: v5
        ),
        .target(
            name: "RaoLMTraining",
            dependencies: [
                "RaoLMCore", "RaoLMModel",
                .product(name: "MLX", package: "Frigate"),
                .product(name: "MLXNN", package: "Frigate"),
                .product(name: "MLXOptimizers", package: "Frigate"),
            ],
            swiftSettings: v5
        ),
        .target(
            name: "RaoLMProvenance",
            dependencies: [
                "RaoLMCore", "RaoLMModel", "RaoLMTraining",
                .product(name: "MLX", package: "Frigate"),
            ],
            swiftSettings: v5
        ),
        .target(
            name: "RaoLMThread",
            dependencies: [
                "RaoLMCore",
                .product(name: "Conduit", package: "Conduit"),
                .product(name: "GRPCCore", package: "grpc-swift"),
                .product(name: "GRPCNIOTransportHTTP2", package: "grpc-swift-nio-transport"),
            ]
        ),
        .target(
            name: "RaoLM",
            dependencies: [
                "RaoLMCore", "RaoLMModel", "RaoLMTraining", "RaoLMProvenance", "RaoLMThread",
            ],
            swiftSettings: v5
        ),
        .executableTarget(
            name: "RaoLMCLI",
            dependencies: [
                "RaoLM",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: v5
        ),
        .testTarget(
            name: "RaoLMCoreTests",
            dependencies: ["RaoLMCore"]
        ),
        .testTarget(
            name: "RaoLMThreadTests",
            dependencies: [
                "RaoLMThread", "RaoLMCore",
                .product(name: "Conduit", package: "Conduit"),
            ]
        ),
        .testTarget(
            name: "RaoLMModelTests",
            dependencies: [
                "RaoLMModel", "RaoLMCore",
                .product(name: "MLX", package: "Frigate"),
                .product(name: "MLXNN", package: "Frigate"),
                .product(name: "MLXLMCommon", package: "Frigate"),
                .product(name: "MLXLLM", package: "Frigate"),
            ],
            swiftSettings: v5
        ),
        .testTarget(
            name: "RaoLMTrainingTests",
            dependencies: [
                "RaoLMTraining", "RaoLMModel", "RaoLMCore",
                .product(name: "MLX", package: "Frigate"),
            ],
            swiftSettings: v5
        ),
        .testTarget(
            name: "RaoLMProvenanceTests",
            dependencies: [
                "RaoLMProvenance", "RaoLMTraining", "RaoLMModel", "RaoLMCore",
                .product(name: "MLX", package: "Frigate"),
            ],
            swiftSettings: v5
        ),
    ]
)
