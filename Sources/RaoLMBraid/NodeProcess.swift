//
//  NodeProcess.swift
//  RaoLMBraid
//
//  WHAT: The umbrella's side of a node process: spawn `raolm node serve`, send it numbered
//        requests, match its replies, forward its events, and stop it. `ProcessStrandLink`
//        is the StrandLink the braided generator uses to reach a node in its own process.
//  PIN:  A write to a node that has died fails with an error, never a SIGPIPE: the process
//        ignores that signal. A node that exits fails every request still waiting on it.
//

import Darwin
import Foundation
import RaoLMCore
import RaoLMProvenance

public final class NodeProcess: @unchecked Sendable {
    public let name: String
    public let executable: URL
    public let arguments: [String]
    public let logFile: URL
    private let process = Process()
    private let input = Pipe()
    private let output = Pipe()
    private let lock = NSLock()
    private let writeLock = NSLock()
    private let splitter = LineSplitter()
    private var nextID = 1
    private var pending: [Int: StrandCall<NodeReply>] = [:]
    private var exited = false
    public var onEvent: @Sendable (NodeEvent) -> Void = { _ in }
    public var onExit: @Sendable (Int32) -> Void = { _ in }

    public init(name: String, executable: URL, arguments: [String], logFile: URL) {
        self.name = name
        self.executable = executable
        self.arguments = arguments
        self.logFile = logFile
    }

    public var pid: Int32 { process.processIdentifier }
    public var isRunning: Bool { process.isRunning }

    public func start() throws {
        signal(SIGPIPE, SIG_IGN)
        try FileManager.default.createDirectory(at: logFile.deletingLastPathComponent(), withIntermediateDirectories: true)
        if !FileManager.default.fileExists(atPath: logFile.path) { FileManager.default.createFile(atPath: logFile.path, contents: nil) }
        let log = try FileHandle(forWritingTo: logFile)
        try log.seekToEnd()
        process.executableURL = executable
        process.arguments = arguments
        process.standardInput = input
        process.standardOutput = output
        process.standardError = log
        output.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            self?.received(data)
        }
        process.terminationHandler = { [weak self] finished in
            self?.terminated(finished.terminationStatus)
        }
        try process.run()
    }

    /// Sends a request; the call is answered when the node replies (or fails when it exits).
    public func send(_ op: NodeRequest.Op) -> StrandCall<NodeReply> {
        let call = StrandCall<NodeReply>()
        lock.lock()
        guard !exited else {
            lock.unlock()
            call.fulfil(.failure(StrandLinkError.closed(name)))
            return call
        }
        let id = nextID
        nextID += 1
        pending[id] = call
        lock.unlock()
        do {
            let data = try NodeWire.encode(NodeRequest(id: id, op: op))
            try writeLock.withLock { try input.fileHandleForWriting.write(contentsOf: data) }
        } catch {
            _ = take(id)
            call.fulfil(.failure(StrandLinkError.closed(name)))
        }
        return call
    }

    private func take(_ id: Int) -> StrandCall<NodeReply>? {
        lock.withLock { pending.removeValue(forKey: id) }
    }

    private func received(_ data: Data) {
        for line in splitter.feed(data) {
            guard let message = try? NodeWire.decode(NodeOutput.self, line: line) else { continue }
            switch message {
            case .reply(let id, let reply):
                take(id)?.fulfil(.success(reply))
            case .failure(let id, let message, let code):
                take(id)?.fulfil(.failure(StrandLinkError.remote(message, code: code)))
            case .event(let event):
                onEvent(event)
            }
        }
    }

    private func terminated(_ status: Int32) {
        output.fileHandleForReading.readabilityHandler = nil
        let rest = output.fileHandleForReading.readDataToEndOfFile()
        if !rest.isEmpty { received(rest) }
        lock.lock()
        exited = true
        let waiting = Array(pending.values)
        pending.removeAll()
        lock.unlock()
        for call in waiting { call.fulfil(.failure(StrandLinkError.closed(name))) }
        onExit(status)
    }

    /// Asks the node to shut down (it stops its Thread first), then closes its input; SIGTERM
    /// after `grace`, which the node also turns into an orderly shutdown.
    public func stop(grace: TimeInterval = 15) async {
        guard process.isRunning else { return }
        _ = send(.shutdown)
        try? writeLock.withLock { try input.fileHandleForWriting.close() }
        let deadline = Date().addingTimeInterval(grace)
        while process.isRunning, Date() < deadline { try? await Task.sleep(nanoseconds: 100_000_000) }
        if process.isRunning {
            process.terminate()
            let hard = Date().addingTimeInterval(10)
            while process.isRunning, Date() < hard { try? await Task.sleep(nanoseconds: 100_000_000) }
            if process.isRunning { kill(process.processIdentifier, SIGKILL) }
        }
    }
}

extension StrandCall {
    /// The answer, awaited from async code.
    public func value() async throws -> Value {
        try await withCheckedThrowingContinuation { continuation in
            whenDone { continuation.resume(with: $0) }
        }
    }
}

/// A strand in its own node process.
public final class ProcessStrandLink: StrandLink {
    public let node: NodeProcess
    public let descriptor: StrandDescriptor

    public init(node: NodeProcess, descriptor: StrandDescriptor) {
        self.node = node
        self.descriptor = descriptor
    }

    public func open(session: String, tokens: [Int], k: Int) -> StrandCall<[StrandStep]> {
        node.send(.open(session: session, tokens: tokens, k: k)).map { reply in
            guard case .opened(let steps) = reply else { throw StrandLinkError.unexpected("\(reply)") }
            return steps
        }
    }

    public func advance(session: String, token: Int, k: Int) -> StrandCall<StrandStep> {
        node.send(.advance(session: session, token: token, k: k)).map { reply in
            guard case .hits(let step) = reply else { throw StrandLinkError.unexpected("\(reply)") }
            return step
        }
    }

    public func hidden(session: String, positions: [Int]) -> StrandCall<[[Float]]> {
        node.send(.hidden(session: session, positions: positions)).map { reply in
            guard case .hidden(let packed) = reply else { throw StrandLinkError.unexpected("\(reply)") }
            return packed.map(\.values)
        }
    }

    public func close(session: String) {
        _ = node.send(.close(session: session))
    }
}
