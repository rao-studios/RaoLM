//
//  DoctorCommand.swift
//  RaoLMCLI
//
//  WHAT: raolm doctor — checks everything the demo needs before it needs it (DoctorChecks).
//

import ArgumentParser
import Foundation
import RaoLM
import RaoLMWorkflows

struct Doctor: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Check the toolchain, Metal library, tokenizer, Thread binary, models, ports and disk.")

    @OptionGroup var global: GlobalOptions
    @OptionGroup var thread: ThreadOptions

    @Option(help: "Path to the thread binary.")
    var threadBinary: String?

    func run() async throws {
        try await guarded {
            let checks = await DoctorChecks.run(root: global.root, threadBinary: threadBinary, httpPort: thread.httpPort, grpcPort: thread.grpcPort)
            for check in checks { print(check.line) }
            if checks.contains(where: { !$0.ok }) { throw ExitCode(1) }
        }
    }
}
