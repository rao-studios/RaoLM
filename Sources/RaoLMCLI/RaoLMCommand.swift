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
import RaoLMStudio
import RaoLMWorkflows
import SinatraHarness

@main
struct RaoLMCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "raolm",
        abstract: "Pretrain a SmolLM2-shaped model on a Thread corpus and generate text with citations.",
        discussion: """
            RaoLM trains a small decoder on the documents a Thread node governs, records an entropy ledger per step and per epoch, and builds a provenance index that couples the model's logits to corpus positions. Generated tokens carry citations back to (Thread node, document id, partition index, token offset), which `raolm verify` re-checks against the live Thread.

            Quick start: raolm demo — or raolm ui for the full-screen studio (bare `raolm` in a terminal opens it; RAOLM_NO_UI=1 prints this help instead).
            """,
        version: RaoLMVersion.string,
        subcommands: [
            Doctor.self, CorpusGroup.self, ThreadGroup.self, Train.self, GenerateText.self, Verify.self,
            Ground.self, Ledger.self, Eval.self, Demo.self, StudioCommand.self,
        ]
    )

    /// Bare `raolm`: the studio in an interactive terminal, the help everywhere else.
    func run() async throws {
        if Studio.isInteractive, ProcessInfo.processInfo.environment["RAOLM_NO_UI"] == nil {
            var studio = try StudioCommand.parse([])
            try await studio.run()
        } else {
            throw CleanExit.helpRequest(self)
        }
    }
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

// MARK: - Errors

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
    } catch let stop as GroundingScorer.Stop {
        report("grounding not measured: \(stop.description)", hint: "raise --budget", code: 75)
    } catch {
        let failure = FailureMapping.describe(error)
        report(failure.message, hint: failure.hint, code: failure.code)
    }
}

private func report(_ message: String, hint: String? = nil, code: Int32) {
    Console.error("raolm: \(message)")
    if let hint { Console.error("  hint: \(hint)") }
    // ExitCode is the only way to set the process status from ArgumentParser.
    Foundation.exit(code)
}
