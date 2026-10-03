//
//  NodeServer.swift
//  RaoLMBraid
//
//  WHAT: `raolm node serve`: one Thread node as its own process. It starts the node's Thread
//        (unless offline), loads the shared vocabulary and the live version, and answers the
//        umbrella over its standard input and output while its update loop runs.
//  PIN:  The wire is a duplicate of standard output taken first thing; then standard output
//        and error point at node.log, so nothing a library prints can corrupt the protocol.
//        All MLX work happens on one thread with a 64 MB stack; serving requests are answered
//        between training steps. End of input means the umbrella is gone: the node stops its
//        Thread and exits, so no process outlives the braid.
//

import Darwin
import Foundation
import MLX
import RaoLMCore
import RaoLMModel
import RaoLMProvenance
import RaoLMThread

public struct NodeServerOptions: Codable, Sendable, Equatable {
    public var name: String
    public var label: String
    /// The braid's root directory.
    public var root: String
    public var vocabularySHA256: String
    /// The umbrella pack when it has a base model (nil: the vocabulary alone).
    public var packSHA256: String?
    public var offline: Bool
    public var threadBinary: String?
    public var httpPort: Int?
    public var grpcPort: Int?
    public var owner: String
    public var settings: HypervisorSettings

    public init(
        name: String, label: String, root: String, vocabularySHA256: String, packSHA256: String? = nil, offline: Bool, threadBinary: String?,
        httpPort: Int?, grpcPort: Int?, owner: String, settings: HypervisorSettings
    ) {
        self.name = name
        self.label = label
        self.root = root
        self.vocabularySHA256 = vocabularySHA256
        self.packSHA256 = packSHA256
        self.offline = offline
        self.threadBinary = threadBinary
        self.httpPort = httpPort
        self.grpcPort = grpcPort
        self.owner = owner
        self.settings = settings
    }

    public static let fileName = "serve.json"
}

public final class NodeServer: @unchecked Sendable {
    private let options: NodeServerOptions
    private let layout: NodeLayout
    /// A hosting umbrella to dial, instead of the parent's pipes.
    private let connect: (host: String, port: Int)?
    private var wire: FileHandle?
    private let wireLock = NSLock()
    private let condition = NSCondition()
    private var serveQueue: [NodeRequest] = []
    private var syncRequested = false
    private var stopping = false
    private var ready = false
    private let cancelFlag = Flag()
    private var hypervisor: ThreadHypervisor?
    private var host: ThreadHost?
    private var sessions: [String: ThreadStrand] = [:]
    private var startupError: String?
    private let done = DispatchSemaphore(value: 0)
    private var pendingState: StrandState?
    private var lastStateSent = Date.distantPast
    private var lastStage: NodeStage?
    private let stateLock = NSLock()
    private var timer: DispatchSourceTimer?

    final class Flag: @unchecked Sendable {
        private let lock = NSLock()
        private var value = false
        func set(_ newValue: Bool) { lock.withLock { value = newValue } }
        var isSet: Bool { lock.withLock { value } }
    }

    public init(options: NodeServerOptions, connect: (host: String, port: Int)? = nil) {
        self.options = options
        self.layout = BraidLayout(root: URL(fileURLWithPath: options.root)).node(options.name)
        self.connect = connect
    }

    /// Serves until shutdown or end of input; returns the exit status.
    public func run() -> Int32 {
        signal(SIGPIPE, SIG_IGN)
        // Ctrl-C reaches the whole process group; the umbrella decides when a node stops. SIGTERM
        // is an orderly shutdown, so the node's Thread never outlives it.
        signal(SIGINT, SIG_IGN)
        signal(SIGTERM, SIG_IGN)
        let terminate = DispatchSource.makeSignalSource(signal: SIGTERM, queue: DispatchQueue(label: "raolm.node.signal"))
        terminate.setEventHandler { [weak self] in self?.shutdown() }
        terminate.resume()
        defer { terminate.cancel() }
        do {
            try redirect()
        } catch {
            FileHandle.standardError.write(Data("raolm node: \(error)\n".utf8))
            return 70
        }
        log("node \(options.name) (pid \(getpid())) starting")
        let reader = Thread { [weak self] in self?.readLoop() }
        reader.name = "raolm.node.reader"
        reader.start()
        let worker = Thread { [weak self] in self?.workLoop() }
        worker.name = "raolm.node.mlx"
        worker.stackSize = 64 << 20
        worker.qualityOfService = .userInitiated
        worker.start()
        startTimer()
        done.wait()
        timer?.cancel()
        flushState(force: true)
        if let host {
            let stopped = host
            try? Blocking.run { await stopped.stop() }
            log("stopped the Thread")
        }
        try? FileManager.default.removeItem(at: layout.record)
        log("node \(options.name) exited")
        return startupError == nil ? 0 : 69
    }

    // MARK: - Plumbing

    private func redirect() throws {
        try FileManager.default.createDirectory(at: layout.logs, withIntermediateDirectories: true)
        if connect == nil {
            let copy = dup(STDOUT_FILENO)
            guard copy >= 0 else { throw BraidSessionError.io("cannot duplicate standard output") }
            wire = FileHandle(fileDescriptor: copy, closeOnDealloc: true)
        }
        let fd = open(layout.nodeLog.path, O_WRONLY | O_CREAT | O_APPEND, 0o644)
        guard fd >= 0 else { throw BraidSessionError.io("cannot open \(layout.nodeLog.path)") }
        fflush(stdout)
        fflush(stderr)
        dup2(fd, STDOUT_FILENO)
        dup2(fd, STDERR_FILENO)
        close(fd)
        setvbuf(stdout, nil, _IOLBF, 0)
    }

    private func log(_ message: String) {
        let formatter = ISO8601DateFormatter()
        print("\(formatter.string(from: Date())) \(message)")
    }

    private func send(_ output: NodeOutput) {
        guard let data = try? NodeWire.encode(output) else { return }
        wireLock.lock()
        defer { wireLock.unlock() }
        do {
            try wire?.write(contentsOf: data)
        } catch {
            // The umbrella is gone: a child exits; a node that dialled in waits to dial again.
            if connect == nil { shutdown() } else { wire = nil }
        }
    }

    private func reply(_ id: Int, _ reply: NodeReply) { send(.reply(id: id, reply: reply)) }

    private func fail(_ id: Int, _ error: Error) {
        let code: Int32 = (error as? CancellationError) != nil ? 130 : 70
        send(.failure(id: id, message: "\(options.name): \(error)", code: code))
    }

    private func shutdown() {
        condition.lock()
        stopping = true
        condition.broadcast()
        condition.unlock()
        cancelFlag.set(true)
    }

    // MARK: - Reading

    private func readLoop() {
        guard let connect else {
            read(FileHandle.standardInput)
            log("end of input: the umbrella is gone")
            shutdown()
            return
        }
        // Started by a hosting umbrella on this machine, whose pipe is its standard input: when that
        // process is gone, so is the node. Started any other way, it outlives the umbrella.
        var status = stat()
        if fstat(STDIN_FILENO, &status) == 0, status.st_mode & S_IFMT == S_IFIFO {
            let watcher = Thread { [weak self] in
                while !FileHandle.standardInput.availableData.isEmpty {}
                self?.log("the umbrella that started this node is gone")
                self?.shutdown()
            }
            watcher.name = "raolm.node.parent"
            watcher.start()
        }
        // Dialled in: the umbrella can go and come back; the node keeps its versions and dials again.
        var delay: UInt32 = 1
        while !isStopping {
            guard let fd = BraidListener.dial(host: connect.host, port: connect.port) else {
                log("cannot reach the umbrella at \(connect.host):\(connect.port); dialling again in \(delay) s")
                sleep(delay)
                delay = min(delay * 2, 30)
                continue
            }
            delay = 1
            let handle = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
            wireLock.withLock { wire = handle }
            send(.announce(name: options.name, pid: getpid()))
            log("connected to the umbrella at \(connect.host):\(connect.port)")
            read(handle)
            wireLock.withLock { wire = nil }
            // Its generations are over: their sessions go.
            accept(NodeRequest(id: -1, op: .close(session: "*")))
            if !isStopping { log("the umbrella closed the connection; dialling again") }
        }
    }

    private var isStopping: Bool {
        condition.lock()
        defer { condition.unlock() }
        return stopping
    }

    private func read(_ input: FileHandle) {
        let splitter = LineSplitter()
        while true {
            let data = input.availableData
            if data.isEmpty { break }
            for line in splitter.feed(data) {
                guard let request = try? NodeWire.decode(NodeRequest.self, line: line) else {
                    log("unreadable request: \(String(decoding: line.prefix(200), as: UTF8.self))")
                    continue
                }
                accept(request)
            }
        }
    }

    private func accept(_ request: NodeRequest) {
        switch request.op {
        case .cancel:
            cancelFlag.set(true)
            reply(request.id, .ok)
        case .shutdown:
            reply(request.id, .ok)
            shutdown()
        case .ping:
            reply(request.id, .ok)
        case .sync:
            condition.lock()
            syncRequested = true
            condition.broadcast()
            condition.unlock()
            reply(request.id, .accepted)
        default:
            condition.lock()
            serveQueue.append(request)
            condition.broadcast()
            condition.unlock()
        }
    }

    // MARK: - The MLX thread

    private func workLoop() {
        do {
            try start()
        } catch {
            startupError = "\(error)"
            log("start failed: \(error)")
            // Answer whatever is waiting (the hello) with the failure, then exit.
            condition.lock()
            let waiting = serveQueue
            serveQueue.removeAll()
            condition.unlock()
            for request in waiting { fail(request.id, error) }
            shutdownAndFinish()
            return
        }
        while true {
            condition.lock()
            while serveQueue.isEmpty, !syncRequested, !stopping { condition.wait() }
            if stopping {
                condition.unlock()
                break
            }
            if !serveQueue.isEmpty {
                let request = serveQueue.removeFirst()
                condition.unlock()
                autoreleasepool { serve(request) }
                continue
            }
            syncRequested = false
            condition.unlock()
            cancelFlag.set(false)
            autoreleasepool {
                do {
                    try hypervisor?.sync()
                } catch {
                    log("update failed: \(error)")
                }
            }
            // Training leaves its buffers in MLX's cache; an idle node gives them back, so a braid of
            // many Threads fits (phase 5: idle nodes held 7.5–8 GB each before this).
            let memory = Memory.snapshot()
            Memory.clearCache()
            log(String(format: "memory after the update: %.2f GB active, %.2f GB cached (released), peak %.2f GB",
                       Double(memory.activeMemory) / 1e9, Double(memory.cacheMemory) / 1e9, Double(memory.peakMemory) / 1e9))
        }
        shutdownAndFinish()
    }

    private func shutdownAndFinish() {
        shutdown()
        done.signal()
    }

    private func start() throws {
        if let megabytes = options.settings.cacheLimitMB, megabytes > 0 { Memory.cacheLimit = megabytes * 1_048_576 }
        let root = BraidLayout(root: URL(fileURLWithPath: options.root))
        let tokenizer = try Blocking.run { try await RaoTokenizer.load() }
        let pack: UmbrellaPack
        if let packSHA256 = options.packSHA256 {
            pack = try UmbrellaPack.load(from: root.pack(sha256: packSHA256))
            guard pack.sha256 == packSHA256 else { throw UmbrellaPackError.fingerprint(expected: packSHA256, found: pack.sha256, what: "pack") }
        } else {
            pack = UmbrellaPack(vocabulary: try VocabularyPack.load(from: root.vocabulary(sha256: options.vocabularySHA256)))
        }
        guard pack.vocabulary.sha256 == options.vocabularySHA256 else {
            throw VocabularyError.fingerprint(expected: options.vocabularySHA256, found: pack.vocabulary.sha256)
        }
        let source: CorpusSource
        var threadPID: Int32?
        if options.offline {
            source = DirectoryCorpusSource(directory: layout.offlineCorpus, slug: options.name, owner: options.owner,
                                           threadID: try DirectoryCorpusSource.nodeID(layout))
        } else {
            let threadDB = layout.threadDB
            _ = try Blocking.run { await ThreadHost.stopRecorded(dataDirectory: threadDB) }
            let binary = try ThreadBinaryLocator.locate(explicit: options.threadBinary)
            let host = ThreadHost(configuration: ThreadHostConfiguration(
                binary: binary, dataDirectory: threadDB, logFile: layout.threadLog,
                httpPort: options.httpPort ?? ThreadEndpoint.defaultHTTPPort, grpcPort: options.grpcPort ?? ThreadEndpoint.defaultGRPCPort,
                nodeID: ThreadHost.readNodeID(dataDirectory: threadDB)))
            log("starting its Thread on http :\(host.configuration.httpPort) grpc :\(host.configuration.grpcPort)")
            let endpoint = try Blocking.run { try await host.start() }
            self.host = host
            threadPID = try Blocking.run { await host.pid }
            source = ThreadCorpusSource(endpoint: endpoint, slug: options.name, owner: options.owner)
            log("Thread \(endpoint.nodeID?.uuidString ?? "?") healthy (pid \(threadPID ?? 0))")
        }
        let hypervisor = try ThreadHypervisor(
            name: options.name, label: options.label, layout: layout, pack: pack, tokenizer: tokenizer, source: source,
            settings: options.settings, owner: options.owner)
        hypervisor.setProcess(pid: getpid(), threadPID: threadPID, httpPort: options.offline ? nil : options.httpPort,
                              grpcPort: options.offline ? nil : options.grpcPort)
        hypervisor.onState = { [weak self] state in self?.queueState(state) }
        hypervisor.onEvent = { [weak self] event in
            if case .log(let line) = event { self?.log(line) }
            self?.send(.event(event))
        }
        hypervisor.service = { [weak self] in self?.drainServing() }
        hypervisor.shouldCancel = { [weak self] in self?.cancelFlag.isSet ?? true }
        self.hypervisor = hypervisor
        try JSONCoding.write(NodeRecord(
            name: options.name, label: options.label, pid: getpid(), threadPID: threadPID, threadID: source.threadID,
            httpPort: options.offline ? nil : options.httpPort, grpcPort: options.offline ? nil : options.grpcPort,
            offline: options.offline, startedAt: .wholeSecond()), to: layout.record)
        ready = true
        hypervisor.publish()
        log("ready: live \(hypervisor.liveVersion.map { "v\($0.version)" } ?? "none")")
    }

    /// Answers every serving request waiting now (called between training steps).
    private func drainServing() {
        while true {
            condition.lock()
            guard !serveQueue.isEmpty else {
                condition.unlock()
                return
            }
            let request = serveQueue.removeFirst()
            condition.unlock()
            serve(request)
        }
    }

    private func serve(_ request: NodeRequest) {
        guard let hypervisor else {
            fail(request.id, BraidSessionError.io("not ready"))
            return
        }
        do {
            switch request.op {
            case .hello:
                let state = hypervisor.state
                reply(request.id, .hello(NodeHello(
                    name: options.name, label: options.label, pid: getpid(), threadPID: state.threadPID, threadID: state.threadID,
                    httpPort: state.httpPort, grpcPort: state.grpcPort, offline: options.offline,
                    liveVersion: hypervisor.liveVersion?.version, vocabularySHA256: options.vocabularySHA256, state: state,
                    packSHA256: hypervisor.packSHA256, cut: hypervisor.pack.cut)))
            case .describe:
                reply(request.id, .described(hypervisor.descriptor()))
            case .probe(let tokens):
                hypervisor.probeTokens = tokens
                reply(request.id, .ok)
            case .open(let session, let tokens, let k):
                guard let strand = hypervisor.live else { throw StrandError.notLive(options.name) }
                sessions[session] = strand
                reply(request.id, .opened(try strand.open(session: session, tokens: tokens, k: k)))
            case .advance(let session, let token, let k):
                guard let strand = sessions[session] else { throw StrandError.noSession(session) }
                reply(request.id, .hits(try strand.advance(session: session, token: token, k: k)))
            case .hidden(let session, let positions):
                guard let strand = sessions[session] else { throw StrandError.noSession(session) }
                reply(request.id, .hidden(try strand.hidden(session: session, positions: positions).map(PackedFloats.init)))
            case .states(let session, let positions):
                guard let strand = sessions[session] else { throw StrandError.noSession(session) }
                let states = try strand.states(session: session, positions: positions)
                reply(request.id, .states(last: states.last.map(PackedFloats.init), cut: states.cut.map(PackedFloats.init)))
            case .close(let session) where session == "*":
                for (name, strand) in sessions { strand.close(session: name) }
                sessions.removeAll()
            case .close(let session):
                sessions.removeValue(forKey: session)?.close(session: session)
                reply(request.id, .ok)
            case .context(let stem, let subject, let k, let floor):
                guard let strand = hypervisor.live else { throw StrandError.notLive(options.name) }
                let range = subject.flatMap { $0.count == 2 && $0[0] <= $0[1] ? $0[0]..<$0[1] : nil }
                reply(request.id, .context(try strand.context(for: stem, subject: range, k: k, floor: floor)))
            case .sync, .cancel, .shutdown, .ping:
                reply(request.id, .ok)
            }
        } catch {
            fail(request.id, error)
        }
    }

    // MARK: - State, coalesced

    private func queueState(_ state: StrandState) {
        stateLock.lock()
        pendingState = state
        let stageChanged = state.stage != lastStage
        stateLock.unlock()
        flushState(force: stageChanged)
    }

    private func flushState(force: Bool) {
        stateLock.lock()
        guard let state = pendingState, force || Date().timeIntervalSince(lastStateSent) >= 0.2 else {
            stateLock.unlock()
            return
        }
        pendingState = nil
        lastStateSent = Date()
        lastStage = state.stage
        stateLock.unlock()
        send(.event(.state(state)))
    }

    private func startTimer() {
        let timer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "raolm.node.state"))
        timer.schedule(deadline: .now() + 0.25, repeating: 0.25)
        timer.setEventHandler { [weak self] in self?.flushState(force: true) }
        timer.resume()
        self.timer = timer
    }
}

public enum BraidSessionError: Error, CustomStringConvertible {
    case io(String)
    case unknownNode(String)
    case notStarted
    case noExecutable
    case nodeFailed(String, String)
    case noLiveNodes

    public var description: String {
        switch self {
        case .io(let message): return message
        case .unknownNode(let name): return "no node named \(name) in this braid"
        case .notStarted: return "the braid's nodes are not running"
        case .noExecutable: return "cannot find the raolm executable to run node processes with"
        case .nodeFailed(let name, let message): return "\(name) failed to start: \(message)"
        case .noLiveNodes: return "no node has a live version yet: feed one (f) and wait for it to go live"
        }
    }
}
