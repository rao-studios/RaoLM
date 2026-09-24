//
//  DoctorChecks.swift
//  RaoLMWorkflows
//
//  WHAT: Everything the demo needs, checked before it needs it: the Metal library, the
//        tokenizer, the Thread binary and its Metal library, the embedding model, the ports,
//        the disk and the data root.
//

import Foundation
import RaoLM

public struct DoctorCheck: Sendable, Equatable {
    public var name: String
    public var ok: Bool
    public var detail: String

    public init(name: String, ok: Bool, detail: String) {
        self.name = name
        self.ok = ok
        self.detail = detail
    }

    /// `✓ name             detail`, as `raolm doctor` prints it.
    public var line: String {
        "\(ok ? "✓" : "✗") \(name.padding(toLength: 16, withPad: " ", startingAt: 0)) \(detail)"
    }
}

public enum DoctorChecks {
    public static func run(root: DataRoot, threadBinary: String?, httpPort: Int, grpcPort: Int) async -> [DoctorCheck] {
        var checks: [DoctorCheck] = []

        if let metallib = Preflight.ownMetallib() {
            checks.append(DoctorCheck(name: "raolm metallib", ok: true, detail: metallib.path))
        } else {
            checks.append(DoctorCheck(name: "raolm metallib", ok: false, detail: "missing — run ./build-metallib.sh <debug|release> after swift build"))
        }

        do {
            let tokenizer = try await RaoTokenizer.load()
            let ok = tokenizer.vocabularySize == 49152 && tokenizer.eosTokenID == 0
            checks.append(DoctorCheck(name: "tokenizer", ok: ok, detail: "vocab \(tokenizer.vocabularySize), eos \(tokenizer.eosTokenID), sha \(Format.short(tokenizer.tokenizerSHA256)) at \(tokenizer.directory.path)"))
        } catch {
            checks.append(DoctorCheck(name: "tokenizer", ok: false, detail: "\(error)"))
        }

        do {
            let binary = try ThreadBinaryLocator.locate(explicit: threadBinary)
            checks.append(DoctorCheck(name: "thread binary", ok: true, detail: binary.path))
            if let metallib = ThreadBinaryLocator.metallib(beside: binary) {
                checks.append(DoctorCheck(name: "thread metallib", ok: true, detail: metallib.path))
            } else {
                checks.append(DoctorCheck(name: "thread metallib", ok: false, detail: "missing — cd ../Thread && ./build-metallib.sh release"))
            }
        } catch {
            checks.append(DoctorCheck(name: "thread binary", ok: false, detail: "\(error)"))
        }

        let modelID = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
        let hubRoots = [
            ProcessInfo.processInfo.environment["HF_HOME"].map { URL(fileURLWithPath: $0).appendingPathComponent("snapshots") },
            ProcessInfo.processInfo.environment["HF_HOME"].map { URL(fileURLWithPath: $0) },
            FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?.appendingPathComponent("huggingface"),
        ].compactMap { $0 }
        let found = hubRoots.map { $0.appendingPathComponent("models/\(modelID)") }
            .first { FileManager.default.fileExists(atPath: $0.appendingPathComponent("config.json").path) }
        checks.append(DoctorCheck(
            name: "embedding model", ok: found != nil,
            detail: found?.path ?? "\(modelID) not cached — the Thread will download ~500 MB on first use (needs network)"))

        for (name, port) in [("http port", httpPort), ("grpc port", grpcPort)] {
            let busy = PortProbe.isListening(port: port)
            checks.append(DoctorCheck(name: name, ok: !busy, detail: busy ? "\(port) is in use" : "\(port) is free"))
        }

        if let attributes = try? FileManager.default.attributesOfFileSystem(forPath: NSHomeDirectory()),
           let free = attributes[.systemFreeSize] as? Int64 {
            let gb = Double(free) / 1e9
            checks.append(DoctorCheck(name: "disk", ok: gb >= 2, detail: String(format: "%.1f GB free", gb)))
        }
        checks.append(DoctorCheck(name: "data root", ok: true, detail: root.url.path))
        return checks
    }
}
