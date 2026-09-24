//
//  StudioCommand.swift
//  RaoLMCLI
//
//  WHAT: raolm ui — the full-screen studio: runs, test suites, corpus generation, the Thread
//        node, pretraining, cited generation with the output-contribution debugger and the
//        grounding harness, evaluation and the entropy ledger. Bare `raolm` opens it too when
//        standard input and output are a terminal.
//

import ArgumentParser
import Foundation
import RaoLM
import RaoLMStudio
import RaoLMWorkflows

struct StudioCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ui",
        abstract: "Open the RaoLM studio: every tool in one full-screen terminal UI.",
        discussion: """
            Keys: 1–8 switch screens, ? shows every key, q quits. The studio logs what libraries print to <data root>/logs/studio.log.
            --fixtures replays a recorded run with no MLX and no Thread (Tests/RaoLMStudioTests/Fixtures/demo).
            RAOLM_COLOR=truecolor|256|16|none forces the colour depth; RAOLM_ASCII=1 draws ASCII glyphs; RAOLM_UI_THEME=terminal keeps the terminal's own background.
            """)

    @OptionGroup var global: GlobalOptions

    @Option(help: "Path to the thread binary.")
    var threadBinary: String?

    @Option(help: "Thread HTTP port.")
    var httpPort = ThreadEndpoint.defaultHTTPPort

    @Option(help: "Thread gRPC port.")
    var grpcPort = ThreadEndpoint.defaultGRPCPort

    @Option(help: "Thread owner id for ingest and export.")
    var owner = "raolm-demo"

    @Option(help: "Replay a recorded run instead of using MLX and a Thread (default: $RAOLM_UI_FIXTURES).")
    var fixtures: String?

    func run() async throws {
        try await guarded {
            guard Studio.isInteractive else {
                throw RaoLMFailure("raolm ui needs an interactive terminal", hint: "run it directly, not through a pipe", code: 64)
            }
            let fixturePath = fixtures ?? ProcessInfo.processInfo.environment["RAOLM_UI_FIXTURES"]
            let options = StudioOptions(
                root: global.root, fixtures: fixturePath.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) },
                threadBinary: threadBinary, httpPort: httpPort, grpcPort: grpcPort, owner: owner)
            let code = try await Studio.launch(options)
            if code != 0 { throw ExitCode(code) }
        }
    }
}
