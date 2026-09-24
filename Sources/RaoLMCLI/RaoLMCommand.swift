//
//  RaoLMCommand.swift
//  RaoLMCLI
//
//  WHAT: The `raolm` executable: generate and host a corpus in a Thread, pretrain on it, and
//        generate text whose tokens cite the Thread partitions they came from.
//

import ArgumentParser
import Foundation
import RaoLM

@main
struct RaoLMCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "raolm",
        abstract: "Pretrain a SmolLM2-shaped model on a Thread corpus and generate text with citations.",
        discussion: """
            RaoLM trains a small decoder on the documents a Thread node governs, records an entropy ledger per step and per epoch, and builds a provenance index that couples the model's logits to corpus positions. Generated tokens carry citations back to (Thread node, document id, partition index, token offset), which `raolm verify` re-checks against the live Thread.

            Quick start: raolm demo
            """,
        version: RaoLMVersion.string,
        subcommands: [
            Doctor.self, CorpusGroup.self, ThreadGroup.self, Train.self, GenerateText.self, Verify.self,
            Ledger.self, Eval.self, Demo.self,
        ]
    )
}

// MARK: - Shared options

struct GlobalOptions: ParsableArguments {
    @Option(name: .long, help: "Data root (default: $RAOLM_DATA_DIR, else ~/Documents/raolm-db).")
    var dataDir: String?

    var root: DataRoot { DataRoot.resolve(argument: dataDir) }
}

struct ThreadOptions: ParsableArguments {
    @Option(name: .long, help: "Thread host.")
    var host = "127.0.0.1"

    @Option(name: .long, help: "Thread HTTP port.")
    var httpPort = ThreadEndpoint.defaultHTTPPort

    @Option(name: .long, help: "Thread gRPC port.")
    var grpcPort = ThreadEndpoint.defaultGRPCPort

    @Option(name: .long, help: "The Thread's node id (default: read from the hosted Thread's data directory).")
    var nodeId: String?

    func endpoint(nodeID: UUID? = nil) -> ThreadEndpoint {
        ThreadEndpoint(host: host, httpPort: httpPort, grpcPort: grpcPort, nodeID: nodeID ?? nodeId.flatMap(UUID.init))
    }
}

enum OutputFormat: String, ExpressibleByArgument, CaseIterable {
    case markers, plain, json
}

// MARK: - Errors and console

struct RaoLMFailure: Error, CustomStringConvertible {
    var message: String
    var hint: String?
    var code: Int32

    init(_ message: String, hint: String? = nil, code: Int32 = 70) {
        self.message = message
        self.hint = hint
        self.code = code
    }

    var description: String { message }
}

enum Console {
    static func setup() {
        setvbuf(stdout, nil, _IOLBF, 0)
    }

    static func error(_ message: String) {
        FileHandle.standardError.write(Data((message + "\n").utf8))
    }

    static func section(_ title: String) {
        print("\n── \(title) " + String(repeating: "─", count: max(0, 72 - title.count)))
    }
}

/// Runs a command body, printing known failures as `raolm: …` with a hint and mapping them
/// to sysexits-style codes (65 data, 66 no input, 69 unavailable, 70 software, 75 temporary,
/// 78 configuration; 3 means verification found a mismatch).
func guarded(_ body: () async throws -> Void) async throws {
    Console.setup()
    do {
        try await body()
    } catch let exit as ExitCode {
        throw exit
    } catch let error as ValidationError {
        throw error
    } catch let failure as RaoLMFailure {
        report(failure.message, hint: failure.hint, code: failure.code)
    } catch let error as ThreadCorpusError {
        switch error {
        case .indexTimeout: report(error.description, code: 75)
        case .indexRejected, .exportIncomplete, .partitionAddressMismatch: report(error.description, code: 65)
        default: report(error.description, code: 69)
        }
    } catch let error as RaoTokenizerError {
        report(error.description, code: 78)
    } catch let error as RunManifestError {
        report(error.description, code: 66)
    } catch let error as CorpusStoreError {
        report(error.description, hint: "generate one with: raolm corpus generate", code: 66)
    } catch let error as ProvenanceError {
        report(error.description, code: 65)
    } catch let error as SyntheticCorpusError {
        report(error.description, code: 65)
    } catch let error as RaoLMConfigError {
        report(error.description, code: 64)
    } catch let error as CheckpointError {
        report(error.description, code: 66)
    } catch {
        report("\(error)", code: 70)
    }
}

private func report(_ message: String, hint: String? = nil, code: Int32) {
    Console.error("raolm: \(message)")
    if let hint { Console.error("  hint: \(hint)") }
    // ExitCode is the only way to set the process status from ArgumentParser.
    Foundation.exit(code)
}

enum Preflight {
    /// The Metal library MLX loads for this executable.
    static func ownMetallib() -> URL? {
        guard let executable = Bundle.main.executableURL?.resolvingSymlinksInPath() else { return nil }
        let directory = executable.deletingLastPathComponent()
        for name in ["mlx.metallib", "Resources/default.metallib", "default.metallib"] {
            let url = directory.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        return nil
    }

    /// MLX aborts the process on the first GPU op without a metallib; fail with a fix instead.
    static func requireMetallib() throws {
        guard ownMetallib() == nil else { return }
        let path = Bundle.main.executableURL?.deletingLastPathComponent().path ?? "the raolm binary"
        let configuration = path.contains("/Release") || path.contains("/release") ? "release" : "debug"
        throw RaoLMFailure(
            "no mlx.metallib beside \(path); MLX cannot run without it",
            hint: "./build-metallib.sh \(configuration)   (after swift build)", code: 78)
    }
}

/// Stops on SIGINT/SIGTERM without killing the process, so children can be shut down.
final class StopSignal: @unchecked Sendable {
    private let lock = NSLock()
    private var fired = false
    private var sources: [DispatchSourceSignal] = []

    init(_ signals: [Int32] = [SIGINT, SIGTERM]) {
        for number in signals {
            signal(number, SIG_IGN)
            let source = DispatchSource.makeSignalSource(signal: number, queue: .global())
            source.setEventHandler { [weak self] in self?.fire() }
            source.resume()
            sources.append(source)
        }
    }

    private func fire() {
        lock.lock()
        fired = true
        lock.unlock()
    }

    var isSet: Bool {
        lock.lock()
        defer { lock.unlock() }
        return fired
    }
}
