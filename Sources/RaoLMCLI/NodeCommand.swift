//
//  NodeCommand.swift
//  RaoLMCLI
//
//  WHAT: raolm node serve — one Thread node of a braid as its own process. The braid starts
//        these itself; they speak JSON lines on standard input and output.
//

import ArgumentParser
import Foundation
import RaoLM
import RaoLMWorkflows

struct NodeGroup: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "node",
        abstract: "A braid's Thread node process (started by raolm braid).",
        shouldDisplay: false,
        subcommands: [Serve.self]
    )

    struct Serve: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Serve one Thread node over standard input and output.")

        @Option(help: "serve.json written by the braid.")
        var options: String

        func run() async throws {
            let options: NodeServerOptions
            do {
                try Preflight.requireMetallib()
                options = try JSONCoding.read(NodeServerOptions.self, from: URL(fileURLWithPath: self.options))
            } catch {
                let failure = FailureMapping.describe(error)
                Console.error("raolm node: \(failure.message)")
                throw ExitCode(failure.code)
            }
            let server = NodeServer(options: options)
            let code = await Task.detached { server.run() }.value
            if code != 0 { throw ExitCode(code) }
        }
    }
}
