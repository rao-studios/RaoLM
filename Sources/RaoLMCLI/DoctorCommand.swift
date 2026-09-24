//
//  DoctorCommand.swift
//  RaoLMCLI
//
//  WHAT: raolm doctor — checks everything the demo needs before it needs it.
//

import ArgumentParser
import Foundation
import RaoLM

struct Doctor: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Check the toolchain, Metal library, tokenizer, Thread binary, models, ports and disk.")

    @OptionGroup var global: GlobalOptions
    @OptionGroup var thread: ThreadOptions

    @Option(help: "Path to the thread binary.")
    var threadBinary: String?

    struct Check {
        var name: String
        var ok: Bool
        var detail: String
    }

    static func checks(root: DataRoot, threadBinary: String?, httpPort: Int, grpcPort: Int) async -> [Check] {
        var checks: [Check] = []

        if let metallib = Preflight.ownMetallib() {
            checks.append(Check(name: "raolm metallib", ok: true, detail: metallib.path))
        } else {
            checks.append(Check(name: "raolm metallib", ok: false, detail: "missing — run ./build-metallib.sh <debug|release> after swift build"))
        }

        do {
            let tokenizer = try await RaoTokenizer.load()
            let ok = tokenizer.vocabularySize == 49152 && tokenizer.eosTokenID == 0
            checks.append(Check(name: "tokenizer", ok: ok, detail: "vocab \(tokenizer.vocabularySize), eos \(tokenizer.eosTokenID), sha \(Format.short(tokenizer.tokenizerSHA256)) at \(tokenizer.directory.path)"))
        } catch {
            checks.append(Check(name: "tokenizer", ok: false, detail: "\(error)"))
        }

        do {
            let binary = try ThreadBinaryLocator.locate(explicit: threadBinary)
            checks.append(Check(name: "thread binary", ok: true, detail: binary.path))
            if let metallib = ThreadBinaryLocator.metallib(beside: binary) {
                checks.append(Check(name: "thread metallib", ok: true, detail: metallib.path))
            } else {
                checks.append(Check(name: "thread metallib", ok: false, detail: "missing — cd ../Thread && ./build-metallib.sh release"))
            }
        } catch {
            checks.append(Check(name: "thread binary", ok: false, detail: "\(error)"))
        }

        let modelID = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
        let hubRoots = [
            ProcessInfo.processInfo.environment["HF_HOME"].map { URL(fileURLWithPath: $0).appendingPathComponent("snapshots") },
            ProcessInfo.processInfo.environment["HF_HOME"].map { URL(fileURLWithPath: $0) },
            FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?.appendingPathComponent("huggingface"),
        ].compactMap { $0 }
        let found = hubRoots.map { $0.appendingPathComponent("models/\(modelID)") }
            .first { FileManager.default.fileExists(atPath: $0.appendingPathComponent("config.json").path) }
        checks.append(Check(
            name: "embedding model", ok: found != nil,
            detail: found?.path ?? "\(modelID) not cached — the Thread will download ~500 MB on first use (needs network)"))

        for (name, port) in [("http port", httpPort), ("grpc port", grpcPort)] {
            let busy = PortProbe.isListening(port: port)
            checks.append(Check(name: name, ok: !busy, detail: busy ? "\(port) is in use" : "\(port) is free"))
        }

        if let attributes = try? FileManager.default.attributesOfFileSystem(forPath: NSHomeDirectory()),
           let free = attributes[.systemFreeSize] as? Int64 {
            let gb = Double(free) / 1e9
            checks.append(Check(name: "disk", ok: gb >= 2, detail: String(format: "%.1f GB free", gb)))
        }
        checks.append(Check(name: "data root", ok: true, detail: root.url.path))
        return checks
    }

    func run() async throws {
        try await guarded {
            let checks = await Self.checks(root: global.root, threadBinary: threadBinary, httpPort: thread.httpPort, grpcPort: thread.grpcPort)
            for check in checks {
                print("\(check.ok ? "✓" : "✗") \(check.name.padding(toLength: 16, withPad: " ", startingAt: 0)) \(check.detail)")
            }
            if checks.contains(where: { !$0.ok }) { throw ExitCode(1) }
        }
    }
}
