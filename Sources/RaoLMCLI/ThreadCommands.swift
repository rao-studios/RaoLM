//
//  ThreadCommands.swift
//  RaoLMCLI
//
//  WHAT: raolm thread start | stop | status — host a Thread node for RaoLM.
//

import ArgumentParser
import Foundation
import RaoLM

struct ThreadGroup: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "thread",
        abstract: "Host a Thread node in open mode on its own data directory.",
        subcommands: [Start.self, Stop.self, Status.self]
    )

    struct Start: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Start a Thread (foreground unless --detach).")

        @OptionGroup var global: GlobalOptions

        @Option(help: "Thread storage (default: <data root>/thread-db).")
        var threadDataDir: String?

        @Option(help: "Thread HTTP port.")
        var httpPort = ThreadEndpoint.defaultHTTPPort

        @Option(help: "Thread gRPC port.")
        var grpcPort = ThreadEndpoint.defaultGRPCPort

        @Option(help: "Path to the thread binary (default: $RAOLM_THREAD_BINARY, else ../Thread/.build/release/thread).")
        var threadBinary: String?

        @Option(help: "Fixed node id (default: keep the one on disk, else a new one).")
        var nodeId: String?

        @Flag(help: "Delete the Thread's storage first.")
        var fresh = false

        @Flag(help: "Leave the Thread running and return.")
        var detach = false

        func run() async throws {
            try await guarded {
                let dataDirectory = threadDataDir.map { URL(fileURLWithPath: $0) } ?? global.root.threadDB
                if fresh {
                    await ThreadHost.stopRecorded(dataDirectory: dataDirectory)
                    try? FileManager.default.removeItem(at: dataDirectory)
                }
                let binary = try ThreadBinaryLocator.locate(explicit: threadBinary)
                let host = ThreadHost(configuration: ThreadHostConfiguration(
                    binary: binary, dataDirectory: dataDirectory,
                    logFile: global.root.logs.appendingPathComponent("thread.log"),
                    httpPort: httpPort, grpcPort: grpcPort, nodeID: nodeId.flatMap(UUID.init)))
                let stop = StopSignal()
                print("starting \(binary.path)")
                let endpoint = try await host.start()
                print("Thread \(endpoint.nodeID?.uuidString ?? "?") ready — http :\(endpoint.httpPort), grpc :\(endpoint.grpcPort), storage \(dataDirectory.path)")
                if detach {
                    print("detached (pid \(await host.pid ?? 0)); stop it with: raolm thread stop")
                    return
                }
                print("press Ctrl-C to stop")
                while await host.isRunning, !stop.isSet {
                    try await Task.sleep(nanoseconds: 250_000_000)
                }
                await host.stop()
                print("stopped")
            }
        }
    }

    struct Stop: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Stop the Thread started by raolm.")

        @OptionGroup var global: GlobalOptions

        @Option(help: "Thread storage (default: <data root>/thread-db).")
        var threadDataDir: String?

        func run() async throws {
            try await guarded {
                let dataDirectory = threadDataDir.map { URL(fileURLWithPath: $0) } ?? global.root.threadDB
                if await ThreadHost.stopRecorded(dataDirectory: dataDirectory) {
                    print("stopped the Thread recorded in \(dataDirectory.path)")
                } else {
                    print("no running Thread recorded in \(dataDirectory.path)")
                }
            }
        }
    }

    struct Status: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Health and document counts of a Thread.")

        @OptionGroup var global: GlobalOptions
        @OptionGroup var thread: ThreadOptions

        func run() async throws {
            try await guarded {
                let record = ThreadHostRecord.load(dataDirectory: global.root.threadDB)
                let endpoint = record?.endpoint ?? thread.endpoint()
                let client = ThreadCorpusClient(endpoint: endpoint)
                let health = try await client.health()
                let stats = try await client.stats()
                print("Thread \(endpoint.nodeID?.uuidString ?? "(node id unknown)") at \(endpoint.description)")
                print("  status \(health.status), stack \(health.stack ?? "?")")
                print("  \(stats.documents) documents, \(stats.groups) groups, \(stats.owners) owners")
                if let record { print("  pid \(record.pid), started \(record.startedAt), log \(record.logFile)") }
            }
        }
    }
}
