//
//  BraidSession.swift
//  RaoLMBraid
//
//  WHAT: A running braid, as the CLI and the studio drive it: the Thread nodes (each a
//        `raolm node serve` process with a Thread of its own), the mock data fed to them, the
//        links the umbrella generates through, and one event stream of everything that happens.
//        `BraidUmbrella` is the umbrella's own half: the shared vocabulary's final norm and tied
//        head, and the braided generator over whichever nodes are live.
//  PIN:  Nodes get ports from 8195/9195 upward (never the studio's 8095/9095), and a Thread or
//        node left behind by a session that died is stopped before a new one starts. The
//        umbrella's half holds MLX: keep it on the thread that made it.
//

import Darwin
import Foundation
import RaoLMCore
import RaoLMModel
import RaoLMProvenance
import RaoLMThread

public struct BraidNodeSpec: Codable, Sendable, Equatable {
    public var name: String
    public var label: String

    public init(name: String, label: String) {
        self.name = name
        self.label = label
    }

    public static let defaults = [
        BraidNodeSpec(name: "ambient", label: "Ambient"), BraidNodeSpec(name: "craft", label: "Craft"), BraidNodeSpec(name: "veil", label: "Veil"),
    ]

    /// "ambient,craft,veil": each name a node, labelled by its name capitalised.
    public static func parse(_ list: String) throws -> [BraidNodeSpec] {
        let names = list.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        guard !names.isEmpty else { throw BraidSessionError.io("--nodes needs at least one name") }
        for name in names where !BraidLayout.isValidName(name) {
            throw BraidSessionError.io("node name '\(name)' must be 2–24 characters of [a-z0-9-], starting with a letter")
        }
        guard Set(names).count == names.count else { throw BraidSessionError.io("--nodes names a node twice: \(list)") }
        return names.map { BraidNodeSpec(name: $0, label: $0.prefix(1).uppercased() + $0.dropFirst()) }
    }
}

public struct BraidOptions: Sendable {
    public var root: DataRoot
    public var nodes: [BraidNodeSpec] = BraidNodeSpec.defaults
    public var offline = false
    public var threadBinary: String?
    /// The raolm executable node processes run.
    public var executable: URL
    public var httpBase = 8195
    public var grpcBase = 9195
    public var owner = "raolm-braid"
    /// Documents a feed deposits.
    public var batch = 8
    public var seed: UInt64 = 42
    public var documentsPerNode = 40
    /// Feed the nodes from this braid dataset (BraidDataset) instead of generating a world.
    public var dataset: URL?
    /// Set when the caller named no nodes and no dataset: a braid that already has nodes keeps
    /// them, and the world they were fed from, instead of the defaults (see `BraidSession.adopt`).
    public var adoptNodes = false
    public var settings = HypervisorSettings()
    /// Wipe every node's storage, versions and feed before starting.
    public var fresh = false

    public init(root: DataRoot, executable: URL) {
        self.root = root
        self.executable = executable
    }

    public var layout: BraidLayout { BraidLayout(dataRoot: root) }
}

/// One streamed token of a braided generation.
public struct BraidTokenEvent: Codable, Sendable, Equatable {
    public var trace: TokenTrace
    public var open: [String]

    public init(trace: TokenTrace, open: [String]) {
        self.trace = trace
        self.open = open
    }
}

/// Everything a braid session reports, in order; what the panel draws and a recording replays.
public enum BraidEvent: Codable, Sendable {
    case starting([BraidNodeSpec], offline: Bool)
    case spawned(String, pid: Int32)
    case ready(String, NodeHello)
    case node(String, NodeEvent)
    case exited(String, status: Int32)
    case started
    case fed(String, documents: [String])
    case withdrew(String, document: String)
    case examples([BraidExample])
    case generating(prompt: String, nodes: [String])
    /// The prompt's traces once every Thread has scored it (where the gate formed), and the text
    /// of its first token, which has no trace.
    case prompted(first: String, traces: [TokenTrace])
    case token(BraidTokenEvent)
    case generated(CitedGeneration)
    case verified(lines: [String], allVerified: Bool)
    case stopped
    case failure(String?, message: String)
    /// Something the braid decided that its owner should know, such as keeping its own nodes.
    case note(String)
}

public final class BraidSession: @unchecked Sendable {
    public let options: BraidOptions
    public let layout: BraidLayout
    public let world: MockWorld
    public let vocabularySHA256: String
    private let lock = NSLock()
    private var handles: [String: Handle] = [:]
    public let onEvent: @Sendable (BraidEvent) -> Void

    struct Handle {
        var spec: BraidNodeSpec
        var process: NodeProcess
        var hello: NodeHello?
        var state: StrandState?
        var source: CorpusSource?
    }

    public init(options requested: BraidOptions, vocabularySHA256: String, onEvent: @escaping @Sendable (BraidEvent) -> Void) throws {
        var options = requested
        adopted = Self.adopt(&options) ? requested.nodes.map(\.name) : nil
        for spec in options.nodes where !BraidLayout.isValidName(spec.name) {
            throw BraidSessionError.io("node name '\(spec.name)' must be 2–24 characters of [a-z0-9-], starting with a letter")
        }
        self.options = options
        self.layout = options.layout
        self.vocabularySHA256 = vocabularySHA256
        self.onEvent = onEvent
        if let dataset = options.dataset {
            self.world = try MockWorld(dataset: dataset, names: options.nodes.map(\.name))
        } else {
            self.world = try MockWorld(names: options.nodes.map(\.name), seed: options.seed, documentsPerNode: options.documentsPerNode)
        }
    }

    /// The nodes the caller would have started had this braid not kept its own (`adopt`).
    private let adopted: [String]?

    /// A braid started without naming its nodes keeps the world it was fed from: world.json's
    /// nodes, seed, shape and dataset; or, for a braid made before world.json, the nodes whose
    /// Threads were already fed. Otherwise, or when starting fresh, the options stand. True when it
    /// changed the nodes.
    static func adopt(_ options: inout BraidOptions) -> Bool {
        guard options.adoptNodes, !options.fresh, options.dataset == nil else { return false }
        let layout = options.layout
        func specs(_ names: [String]) -> [BraidNodeSpec] {
            names.map { name in options.nodes.first { $0.name == name } ?? BraidNodeSpec(name: name, label: name.prefix(1).uppercased() + name.dropFirst()) }
        }
        let before = options.nodes.map(\.name)
        if let record = MockWorld.Record.load(layout) {
            options.nodes = specs(record.names)
            options.seed = record.seed
            options.documentsPerNode = record.shape.documentsPerNode
            options.dataset = record.dataset.map { URL(fileURLWithPath: $0.path, isDirectory: true) }
            return record.names != before
        }
        let existing = ((try? FileManager.default.contentsOfDirectory(atPath: layout.nodes.path)) ?? [])
            .filter { BraidLayout.isValidName($0) && !FeedState.load(layout.node($0)).deposited.isEmpty }
            .sorted()
        guard !existing.isEmpty, existing != before else { return false }
        options.nodes = specs(existing)
        return true
    }

    /// What world.json records for this session's world.
    public var worldRecord: MockWorld.Record { world.record }

    /// Refuses to start on nodes fed from another mock world: their feeds would name documents
    /// this world does not have, or deal them to other Threads. `--fresh` starts over.
    func checkWorld() throws {
        let wanted = worldRecord
        if options.fresh { return }
        if let record = MockWorld.Record.load(layout) {
            guard record.sameWorld(as: wanted) else {
                throw MockWorldError.different(
                    "the braid's nodes were fed from another mock world (\(record.summary)); this one is \(wanted.summary). Start with --fresh (R in the studio), or use another --data-dir")
            }
            return
        }
        for spec in options.nodes {
            let fed = FeedState.load(layout.node(spec.name)).deposited
            if let unknown = fed.first(where: { world.document(id: $0) == nil }) {
                throw MockWorldError.different(
                    "\(spec.name) was fed \(unknown), which this mock world (\(wanted.summary)) does not have. Start with --fresh (R in the studio), or use another --data-dir")
            }
        }
    }

    public var names: [String] { options.nodes.map(\.name) }

    public func state(_ node: String) -> StrandState? { lock.withLock { handles[node]?.state } }
    public func hello(_ node: String) -> NodeHello? { lock.withLock { handles[node]?.hello } }
    public var states: [StrandState] { lock.withLock { options.nodes.compactMap { handles[$0.name]?.state } } }

    // MARK: - Start and stop

    public func start() async throws {
        do {
            try checkWorld()
        } catch {
            onEvent(.failure(nil, message: "\(error)"))
            throw error
        }
        if let adopted {
            onEvent(.note("kept this braid's own nodes, \(options.nodes.map(\.name).joined(separator: ", ")), and the world they were fed from; "
                          + "starting fresh (R, or --fresh) wipes them and starts \(adopted.joined(separator: ", "))"))
        }
        onEvent(.starting(options.nodes, offline: options.offline))
        try FileManager.default.createDirectory(at: layout.nodes, withIntermediateDirectories: true)
        var http = options.httpBase
        var grpc = options.grpcBase
        var reserved = Set<Int>()
        for spec in options.nodes {
            let node = layout.node(spec.name)
            await ThreadHost.stopRecorded(dataDirectory: node.threadDB)
            Self.stopStale(node)
            if options.fresh { try? FileManager.default.removeItem(at: node.directory) }
            try FileManager.default.createDirectory(at: node.directory, withIntermediateDirectories: true)
            var ports: (Int, Int)?
            if !options.offline {
                // No Thread has bound anything yet, so every port handed out is reserved here.
                let h = Self.freePort(from: &http, reserved: &reserved)
                let g = Self.freePort(from: &grpc, reserved: &reserved)
                ports = (h, g)
            }
            let serve = NodeServerOptions(
                name: spec.name, label: spec.label, root: layout.root.path, vocabularySHA256: vocabularySHA256, offline: options.offline,
                threadBinary: options.threadBinary, httpPort: ports?.0, grpcPort: ports?.1, owner: options.owner, settings: options.settings)
            let serveFile = node.directory.appendingPathComponent(NodeServerOptions.fileName)
            try JSONCoding.write(serve, to: serveFile)
            let process = NodeProcess(name: spec.name, executable: options.executable,
                                      arguments: ["node", "serve", "--options", serveFile.path], logFile: node.nodeLog)
            let name = spec.name
            process.onEvent = { [weak self] event in self?.received(name, event) }
            process.onExit = { [weak self] status in
                self?.onEvent(.exited(name, status: status))
            }
            try process.start()
            lock.withLock { handles[spec.name] = Handle(spec: spec, process: process) }
            onEvent(.spawned(spec.name, pid: process.pid))
        }
        try worldRecord.save(layout)
        // Every node starts its Thread and loads its live version at the same time.
        let waits = options.nodes.compactMap { spec in handle(spec.name).map { (spec, $0.process.send(.hello)) } }
        for (spec, call) in waits {
            let reply: NodeReply
            do {
                reply = try await call.value()
            } catch {
                let message = Self.lastLines(layout.node(spec.name).nodeLog) ?? "\(error)"
                onEvent(.failure(spec.name, message: "\(spec.name) did not start: \(error)"))
                await stop()
                throw BraidSessionError.nodeFailed(spec.name, "\(error)\n\(message)")
            }
            guard case .hello(let hello) = reply else { throw StrandLinkError.unexpected("\(reply)") }
            let source: CorpusSource
            if hello.offline {
                source = DirectoryCorpusSource(directory: layout.node(spec.name).offlineCorpus, slug: spec.name, owner: options.owner,
                                               threadID: hello.threadID)
            } else {
                source = ThreadCorpusSource(
                    endpoint: ThreadEndpoint(httpPort: hello.httpPort ?? 0, grpcPort: hello.grpcPort ?? 0,
                                             nodeID: hello.threadID.flatMap(UUID.init(uuidString:))),
                    slug: spec.name, owner: options.owner)
            }
            lock.withLock {
                handles[spec.name]?.hello = hello
                handles[spec.name]?.state = hello.state
                handles[spec.name]?.source = source
            }
            onEvent(.ready(spec.name, hello))
        }
        onEvent(.started)
        // Pick up whatever each corpus already holds.
        for spec in options.nodes { _ = handle(spec.name)?.process.send(.sync) }
        publishExamples()
    }

    public func stop() async {
        let processes = lock.withLock { handles.values.map(\.process) }
        await withTaskGroup(of: Void.self) { group in
            for process in processes { group.addTask { await process.stop() } }
        }
        onEvent(.stopped)
    }

    public var isRunning: Bool { lock.withLock { handles.values.contains { $0.process.isRunning } } }

    private func handle(_ name: String) -> Handle? { lock.withLock { handles[name] } }

    private func received(_ name: String, _ event: NodeEvent) {
        if case .state(let state) = event { lock.withLock { handles[name]?.state = state } }
        onEvent(.node(name, event))
        if case .promoted = event { publishExamples() }
    }

    // MARK: - Mock data

    @discardableResult
    public func feed(_ node: String, count: Int? = nil) async throws -> [CorpusDocument] {
        guard let handle = handle(node) else { throw BraidSessionError.unknownNode(node) }
        guard let source = handle.source else { throw BraidSessionError.notStarted }
        let documents = try await MockFeeder.feed(node: node, count: count ?? options.batch, world: world, layout: layout.node(node), source: source)
        onEvent(.fed(node, documents: documents.map(\.name)))
        if !documents.isEmpty { _ = handle.process.send(.sync) }
        return documents
    }

    @discardableResult
    public func withdraw(_ node: String) async throws -> CorpusDocument? {
        guard let handle = handle(node) else { throw BraidSessionError.unknownNode(node) }
        guard let source = handle.source else { throw BraidSessionError.notStarted }
        guard let document = try await MockFeeder.withdraw(node: node, world: world, layout: layout.node(node), source: source) else { return nil }
        onEvent(.withdrew(node, document: document.name))
        _ = handle.process.send(.sync)
        publishExamples()
        return document
    }

    public func sync(_ node: String) throws {
        guard let handle = handle(node) else { throw BraidSessionError.unknownNode(node) }
        _ = handle.process.send(.sync)
    }

    public func cancel(_ node: String) {
        _ = handle(node)?.process.send(.cancel)
    }

    /// The standing prompt every node's training candidates complete at each evaluation.
    public func probe(tokens: [Int]) {
        for name in names { _ = handle(name)?.process.send(.probe(tokens: tokens)) }
    }

    // MARK: - For the umbrella

    /// A link to every node with a live version, described now.
    public func links(timeout: TimeInterval = 30) throws -> [StrandLink] {
        var links: [StrandLink] = []
        for name in names {
            guard let handle = handle(name), handle.process.isRunning else { continue }
            guard case .described(let descriptor?) = try handle.process.send(.describe).wait(timeout: timeout) else { continue }
            links.append(ProcessStrandLink(node: handle.process, descriptor: descriptor))
        }
        guard !links.isEmpty else { throw BraidSessionError.noLiveNodes }
        return links
    }

    /// Reads each cited document from the node it lives on.
    public func reader() -> BraidCorpusReader {
        BraidCorpusReader(routes: names.compactMap { name in
            handle(name)?.source.map { (prefix: DocumentID.prefix(slug: name), reader: $0.reader()) }
        })
    }

    public func examples(tokenizer: RaoTokenizer) -> [BraidExample] {
        let present: [BraidExample.Node] = options.nodes.map { spec in
            (name: spec.name, label: spec.label, threadID: handle(spec.name)?.hello?.threadID,
             documents: MockFeeder.present(node: spec.name, world: world, layout: layout.node(spec.name)))
        }
        return BraidExample.build(
            nodes: present.map { (name: $0.name, label: $0.label, threadID: $0.threadID, documents: world.exclusive($0.documents)) },
            presentNodes: present, links: world.crosslinks, tokenizer: tokenizer)
    }

    /// Set by whoever holds a tokenizer, so examples follow every feed and promotion.
    public var exampleTokenizer: RaoTokenizer?

    private func publishExamples() {
        guard let tokenizer = exampleTokenizer else { return }
        onEvent(.examples(examples(tokenizer: tokenizer)))
    }

    // MARK: - Leftovers of a session that died

    /// Stops a node process a session that died left behind, by its record (only if it is raolm).
    public static func stopStale(_ node: NodeLayout) {
        guard let record = NodeRecord.load(node), kill(record.pid, 0) == 0, isRaoLM(record.pid) else {
            try? FileManager.default.removeItem(at: node.record)
            return
        }
        kill(record.pid, SIGTERM)
        let deadline = Date().addingTimeInterval(10)
        while kill(record.pid, 0) == 0, Date() < deadline { usleep(100_000) }
        if kill(record.pid, 0) == 0 { kill(record.pid, SIGKILL) }
        try? FileManager.default.removeItem(at: node.record)
    }

    static func isRaoLM(_ pid: Int32) -> Bool {
        var buffer = [CChar](repeating: 0, count: Int(MAXPATHLEN) * 4)
        let length = proc_pidpath(pid, &buffer, UInt32(buffer.count))
        guard length > 0 else { return false }
        return String(cString: buffer).hasSuffix("/raolm")
    }

    /// The first port from `next` that nothing listens on and no other node was given.
    static func freePort(from next: inout Int, reserved: inout Set<Int>) -> Int {
        while reserved.contains(next) || PortProbe.isListening(port: next) { next += 1 }
        let port = next
        reserved.insert(port)
        next += 1
        return port
    }

    static func lastLines(_ url: URL, count: Int = 6) -> String? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return String(decoding: data.suffix(8 * 1024), as: UTF8.self).split(separator: "\n").suffix(count).joined(separator: "\n")
    }
}

// MARK: - The umbrella

public enum BraidVocabulary {
    public struct Current: Codable, Sendable, Equatable {
        public var sha256: String
        public var hiddenSize: Int
        public var seed: UInt64
        public var headScale: Float
    }

    public static let seed: UInt64 = 0x5EED_B8A1D
    public static let headScale: Float = 2

    /// The braid's shared vocabulary for `config`, created once and named by its hash. MLX: call
    /// it on the thread that will use the umbrella.
    public static func ensure(layout: BraidLayout, config: RaoLMConfig, tokenizer: RaoTokenizer) throws -> VocabularyPack {
        let pointer = layout.vocabularies.appendingPathComponent("current.json")
        if let current = try? JSONCoding.read(Current.self, from: pointer), current.hiddenSize == config.hiddenSize,
           current.seed == seed, current.headScale == headScale,
           let pack = try? VocabularyPack.load(from: layout.vocabulary(sha256: current.sha256)),
           pack.info.tokenizerSHA256 == tokenizer.tokenizerSHA256 {
            return pack
        }
        let pack = VocabularyPack.seeded(config: config, tokenizerSHA256: tokenizer.tokenizerSHA256, seed: seed, rowNorm: 1, headScale: headScale)
        try pack.save(to: layout.vocabulary(sha256: pack.sha256))
        try JSONCoding.write(Current(sha256: pack.sha256, hiddenSize: config.hiddenSize, seed: seed, headScale: headScale), to: pointer)
        return pack
    }
}

public final class BraidUmbrella {
    public let vocabulary: VocabularyPack
    public let head: UmbrellaHead
    public let tokenizer: RaoTokenizer

    public init(vocabulary: VocabularyPack, tokenizer: RaoTokenizer) {
        self.vocabulary = vocabulary
        self.head = UmbrellaHead(vocabulary: vocabulary)
        self.tokenizer = tokenizer
    }

    public func generate(
        links: [StrandLink], request: BraidRequest, onPrompt: (([TokenTrace]) throws -> Void)? = nil,
        onStep: ((BraidStep) throws -> Void)? = nil
    ) throws -> CitedGeneration {
        let generator = try BraidedGenerator(links: links, head: head, tokenizer: tokenizer, gateFloor: request.gateFloor)
        return try generator.generate(request, onPrompt: onPrompt, onStep: onStep)
    }
}

/// Writes a session's events with their times, for `raolm ui --fixtures` to replay.
public final class BraidRecorder: @unchecked Sendable {
    public struct Entry: Codable, Sendable {
        public var seconds: Double
        public var event: BraidEvent
    }

    public static let fileName = "braid-events.jsonl"
    private let writer: JSONLWriter
    private let started = Date()
    private let lock = NSLock()
    /// Node states are kept at most once a second per node (and at every stage change), so a
    /// recording stays small enough to ship as a fixture.
    private var lastState: [String: (seconds: Double, stage: NodeStage)] = [:]
    private var held: [String: Entry] = [:]

    public init(directory: URL) throws {
        writer = try JSONLWriter(url: directory.appendingPathComponent(Self.fileName), truncate: true)
    }

    /// A replay draws the tokens, their shares, citations and spans, and a node's recent losses:
    /// never the per-token neighbour lists, the prompt's citations, or partitions nothing cites.
    /// The prompt's traces travel once, in `prompted`, not again inside the generation.
    static func slim(_ event: BraidEvent) -> BraidEvent {
        func strip(_ traces: [TokenTrace]) -> [TokenTrace] {
            traces.map { trace in
                var copy = trace
                copy.neighbours = []
                if trace.isPrompt {
                    copy.citations = []
                    copy.spanIndex = nil
                }
                return copy
            }
        }
        switch event {
        case .token(let token):
            return .token(BraidTokenEvent(trace: strip([token.trace])[0], open: token.open))
        case .prompted(let first, let traces):
            return .prompted(first: first, traces: strip(traces))
        case .generated(var generation):
            generation.traces = strip(generation.traces.filter { !$0.isPrompt })
            let rows = Set(generation.traces.flatMap { $0.citations.map(\.row) } + generation.spans.map(\.row))
            generation.partitions = generation.partitions.filter { rows.contains($0.row) }
            return .generated(generation)
        case .node(let name, .state(var state)):
            state.losses = Array(state.losses.suffix(40))
            return .node(name, .state(state))
        default:
            return event
        }
    }

    public func record(_ event: BraidEvent) {
        let event = Self.slim(event)
        let now = Date().timeIntervalSince(started)
        lock.lock()
        defer { lock.unlock() }
        if case .node(let name, .state(let state)) = event {
            if let last = lastState[name], last.stage == state.stage, now - last.seconds < 1 {
                held[name] = Entry(seconds: now, event: event)
                return
            }
            lastState[name] = (now, state.stage)
            held[name] = nil
        } else {
            // Anything else settles what the nodes last said first, so a replay ends where they did.
            for pending in held.values.sorted(by: { $0.seconds < $1.seconds }) { try? writer.append(pending) }
            held.removeAll()
        }
        try? writer.append(Entry(seconds: now, event: event))
    }

    public func close() {
        lock.lock()
        for entry in held.values.sorted(by: { $0.seconds < $1.seconds }) { try? writer.append(entry) }
        held.removeAll()
        lock.unlock()
        writer.close()
    }

    public static func load(_ directory: URL) throws -> [Entry] {
        try JSONCoding.readLines(Entry.self, from: directory.appendingPathComponent(fileName))
    }
}
