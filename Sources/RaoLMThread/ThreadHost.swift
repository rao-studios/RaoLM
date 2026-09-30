//
//  ThreadHost.swift
//  RaoLMThread
//
//  WHAT: Runs a Thread node for RaoLM: finds the binary, launches it in open mode on its own
//        data directory and ports, waits until it answers, and stops it again.
//  PIN:  The working directory is the data directory, never the Thread checkout: Thread reads
//        `<cwd>/.env`, and the checkout's holds a provider key. The child inherits the
//        launcher's environment minus RAO_*, AMBIENT_*, THREAD_* and any *_API_KEY, so it can
//        only come up open and can only embed on device. The Metal library must sit beside the
//        (symlink-resolved) binary, or --use-mlx silently falls back to a network API.
//

import Foundation
import RaoLMCore

public enum ThreadBinaryLocator {
    public static let environmentKey = "RAOLM_THREAD_BINARY"

    /// Search roots: the working directory and the directories above this executable.
    public static func defaultRoots() -> [URL] {
        var roots = [URL(fileURLWithPath: FileManager.default.currentDirectoryPath)]
        if var directory = Bundle.main.executableURL?.resolvingSymlinksInPath().deletingLastPathComponent() {
            for _ in 0..<7 {
                roots.append(directory)
                directory = directory.deletingLastPathComponent()
            }
        }
        return roots
    }

    public static func candidates(explicit: String?, environment: [String: String], roots: [URL]) -> [URL] {
        var result: [URL] = []
        if let explicit, !explicit.isEmpty { result.append(URL(fileURLWithPath: (explicit as NSString).expandingTildeInPath)) }
        if let value = environment[environmentKey], !value.isEmpty { result.append(URL(fileURLWithPath: value)) }
        for root in roots {
            for base in [root.appendingPathComponent("../Thread"), root.appendingPathComponent("Thread")] {
                result.append(base.appendingPathComponent(".build/release/thread"))
                result.append(base.appendingPathComponent(".build/debug/thread"))
            }
        }
        var seen = Set<String>()
        return result.map(\.standardizedFileURL).filter { seen.insert($0.path).inserted }
    }

    public static func locate(
        explicit: String? = nil, environment: [String: String] = ProcessInfo.processInfo.environment,
        roots: [URL] = defaultRoots()
    ) throws -> URL {
        let all = candidates(explicit: explicit, environment: environment, roots: roots)
        for candidate in all where FileManager.default.isExecutableFile(atPath: candidate.path) {
            return candidate.resolvingSymlinksInPath()
        }
        throw ThreadCorpusError.binaryNotFound(searched: all.prefix(6).map(\.path))
    }

    /// The Metal library MLX loads for this binary, if present.
    public static func metallib(beside binary: URL) -> URL? {
        let directory = binary.resolvingSymlinksInPath().deletingLastPathComponent()
        for name in ["mlx.metallib", "default.metallib", "Resources/default.metallib"] {
            let url = directory.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        return nil
    }
}

public struct ThreadHostConfiguration: Sendable {
    public var binary: URL
    public var dataDirectory: URL
    public var logFile: URL
    public var host: String
    public var httpPort: Int
    public var grpcPort: Int
    public var nodeID: UUID?
    public var useMLX: Bool
    public var embeddingModel: String
    public var startupTimeout: TimeInterval
    public var warmUp: Bool

    public init(
        binary: URL, dataDirectory: URL, logFile: URL, host: String = "127.0.0.1",
        httpPort: Int = ThreadEndpoint.defaultHTTPPort, grpcPort: Int = ThreadEndpoint.defaultGRPCPort,
        nodeID: UUID? = nil, useMLX: Bool = true,
        embeddingModel: String = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
        startupTimeout: TimeInterval = 90, warmUp: Bool = true
    ) {
        self.binary = binary
        self.dataDirectory = dataDirectory
        self.logFile = logFile
        self.host = host
        self.httpPort = httpPort
        self.grpcPort = grpcPort
        self.nodeID = nodeID
        self.useMLX = useMLX
        self.embeddingModel = embeddingModel
        self.startupTimeout = startupTimeout
        self.warmUp = warmUp
    }
}

/// Written beside the Thread's data so another `raolm` process can find and stop it.
public struct ThreadHostRecord: Codable, Sendable {
    public var pid: Int32
    public var binary: String
    public var host: String
    public var httpPort: Int
    public var grpcPort: Int
    public var nodeID: String
    public var logFile: String
    public var startedAt: Date

    public static let fileName = "thread-host.json"

    public static func load(dataDirectory: URL) -> ThreadHostRecord? {
        try? JSONCoding.read(ThreadHostRecord.self, from: dataDirectory.appendingPathComponent(fileName))
    }

    public var endpoint: ThreadEndpoint {
        ThreadEndpoint(host: host, httpPort: httpPort, grpcPort: grpcPort, nodeID: UUID(uuidString: nodeID))
    }
}

public actor ThreadHost {
    public nonisolated let configuration: ThreadHostConfiguration
    private var process: Process?
    public private(set) var endpoint: ThreadEndpoint?

    public init(configuration: ThreadHostConfiguration) {
        self.configuration = configuration
    }

    public var isRunning: Bool { process?.isRunning ?? false }
    public var pid: Int32? { process?.processIdentifier }

    public static func childEnvironment(_ inherited: [String: String]) -> [String: String] {
        inherited.filter { key, _ in
            !key.hasPrefix("RAO_") && !key.hasPrefix("AMBIENT_") && !key.hasPrefix("THREAD_")
                && !key.hasSuffix("_API_KEY")
        }
    }

    public static func arguments(_ configuration: ThreadHostConfiguration, nodeID: UUID) -> [String] {
        var arguments = [
            "--host", configuration.host,
            "--port", String(configuration.httpPort),
            "--grpc-port", String(configuration.grpcPort),
            "--data-dir", configuration.dataDirectory.path,
            "--node-id", nodeID.uuidString,
            "--no-graph-extraction",
            "--graph-backend", "keyword",
        ]
        if configuration.useMLX {
            arguments += ["--use-mlx", "--mlx-model", configuration.embeddingModel]
        }
        return arguments
    }

    public static func readNodeID(dataDirectory: URL) -> UUID? {
        guard let text = try? String(contentsOf: dataDirectory.appendingPathComponent("node-id"), encoding: .utf8) else {
            return nil
        }
        return UUID(uuidString: text.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    public func start() async throws -> ThreadEndpoint {
        if let endpoint, isRunning { return endpoint }
        let configuration = self.configuration
        let binary = configuration.binary.resolvingSymlinksInPath()
        if configuration.useMLX, ThreadBinaryLocator.metallib(beside: binary) == nil {
            throw ThreadCorpusError.metallibMissing(binary: binary.path)
        }
        if PortProbe.isListening(host: configuration.host, port: configuration.httpPort) {
            throw ThreadCorpusError.portBusy(port: configuration.httpPort, what: "HTTP")
        }
        if PortProbe.isListening(host: configuration.host, port: configuration.grpcPort) {
            throw ThreadCorpusError.portBusy(port: configuration.grpcPort, what: "gRPC")
        }
        let manager = FileManager.default
        try manager.createDirectory(at: configuration.dataDirectory, withIntermediateDirectories: true)
        try manager.createDirectory(at: configuration.logFile.deletingLastPathComponent(), withIntermediateDirectories: true)
        if !manager.fileExists(atPath: configuration.logFile.path) {
            manager.createFile(atPath: configuration.logFile.path, contents: nil)
        }
        let log = try FileHandle(forWritingTo: configuration.logFile)
        try log.seekToEnd()

        let nodeID = configuration.nodeID ?? Self.readNodeID(dataDirectory: configuration.dataDirectory) ?? UUID()
        let process = Process()
        process.executableURL = binary
        process.arguments = Self.arguments(configuration, nodeID: nodeID)
        process.currentDirectoryURL = configuration.dataDirectory
        process.environment = Self.childEnvironment(ProcessInfo.processInfo.environment)
        process.standardOutput = log
        process.standardError = log
        process.standardInput = FileHandle.nullDevice
        try process.run()
        self.process = process

        let endpoint = ThreadEndpoint(
            host: configuration.host, httpPort: configuration.httpPort, grpcPort: configuration.grpcPort, nodeID: nodeID)
        let record = ThreadHostRecord(
            pid: process.processIdentifier, binary: binary.path, host: configuration.host,
            httpPort: configuration.httpPort, grpcPort: configuration.grpcPort, nodeID: nodeID.uuidString,
            logFile: configuration.logFile.path, startedAt: Date())
        try JSONCoding.write(record, to: configuration.dataDirectory.appendingPathComponent(ThreadHostRecord.fileName))

        let client = ThreadCorpusClient(endpoint: endpoint)
        let deadline = Date().addingTimeInterval(configuration.startupTimeout)
        var healthy = false
        while Date() < deadline {
            if !process.isRunning {
                throw ThreadCorpusError.exitedDuringStartup(status: process.terminationStatus, logTail: logTail())
            }
            if let health = try? await client.health(timeout: 1) {
                guard health.stack == nil || health.stack == "open" else {
                    await stop()
                    throw ThreadCorpusError.notOpenMode(stack: health.stack ?? "?")
                }
                if health.status == "healthy", PortProbe.isListening(host: configuration.host, port: configuration.grpcPort) {
                    healthy = true
                    break
                }
            }
            try await Task.sleep(nanoseconds: 250_000_000)
        }
        guard healthy else {
            let tail = logTail()
            await stop()
            throw ThreadCorpusError.startupTimeout(seconds: configuration.startupTimeout, logTail: tail)
        }
        if configuration.warmUp {
            try await client.warmUpEmbedding(owner: "raolm-warmup")
        }
        self.endpoint = endpoint
        return endpoint
    }

    /// SIGTERM (Thread shuts down gracefully and flushes its database), then SIGKILL.
    public func stop(grace: TimeInterval = 8) async {
        guard let process else { return }
        if process.isRunning {
            process.terminate()
            let deadline = Date().addingTimeInterval(grace)
            while process.isRunning, Date() < deadline {
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            if process.isRunning { kill(process.processIdentifier, SIGKILL) }
        }
        self.process = nil
        self.endpoint = nil
        try? FileManager.default.removeItem(
            at: configuration.dataDirectory.appendingPathComponent(ThreadHostRecord.fileName))
    }

    /// Blocks until the child exits (for `raolm thread start` in the foreground).
    public func waitUntilExit() async {
        while let process, process.isRunning {
            try? await Task.sleep(nanoseconds: 250_000_000)
        }
    }

    public func logTail(lines: Int = 40) -> String {
        guard let data = try? Data(contentsOf: configuration.logFile) else { return "" }
        let text = String(decoding: data.suffix(64 * 1024), as: UTF8.self)
        return text.split(separator: "\n", omittingEmptySubsequences: false).suffix(lines).joined(separator: "\n")
    }

    /// Stops a Thread another process started, by its host record. Returns whether one was running.
    @discardableResult
    public static func stopRecorded(dataDirectory: URL, grace: TimeInterval = 8) async -> Bool {
        guard let record = ThreadHostRecord.load(dataDirectory: dataDirectory) else { return false }
        let pid = record.pid
        guard kill(pid, 0) == 0 else {
            try? FileManager.default.removeItem(at: dataDirectory.appendingPathComponent(ThreadHostRecord.fileName))
            return false
        }
        kill(pid, SIGTERM)
        let deadline = Date().addingTimeInterval(grace)
        while kill(pid, 0) == 0, Date() < deadline {
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        if kill(pid, 0) == 0 { kill(pid, SIGKILL) }
        try? FileManager.default.removeItem(at: dataDirectory.appendingPathComponent(ThreadHostRecord.fileName))
        return true
    }
}
