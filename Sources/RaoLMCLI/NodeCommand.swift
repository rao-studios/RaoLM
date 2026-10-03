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
        static let configuration = CommandConfiguration(
            abstract: "Serve one Thread node over standard input and output, or to a hosting umbrella it dials (--connect).")

        @Option(help: "serve.json written by the braid.")
        var options: String

        @Option(help: "host:port of a hosting umbrella (raolm braid … --listen) to dial instead of the parent's pipes; the node dials again if it goes.")
        var connect: String?

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
            var target: (host: String, port: Int)?
            if let connect {
                guard let colon = connect.lastIndex(of: ":"), let port = Int(connect[connect.index(after: colon)...]) else {
                    Console.error("raolm node: --connect takes host:port")
                    throw ExitCode(64)
                }
                target = (String(connect[..<colon]), port)
            }
            let server = NodeServer(options: options, connect: target)
            let code = await Task.detached { server.run() }.value
            if code != 0 { throw ExitCode(code) }
        }
    }
}
