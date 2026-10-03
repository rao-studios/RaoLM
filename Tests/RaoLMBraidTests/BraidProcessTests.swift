import Darwin
import Foundation
import Testing

@testable import RaoLMBraid
@testable import RaoLMCore
@testable import RaoLMModel
@testable import RaoLMProvenance
@testable import RaoLMThread

private var threadTests: Bool { ProcessInfo.processInfo.environment["RAOLM_THREAD_TESTS"] == "1" }

/// The raolm binary this checkout built (node processes run it); scripts/test.sh builds it with the tests.
private func raolmBinary() -> URL? {
    let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    for configuration in ["debug", "release"] {
        let binary = repository.appendingPathComponent(".build/\(configuration)/raolm")
        let metallib = repository.appendingPathComponent(".build/\(configuration)/mlx.metallib")
        if FileManager.default.isExecutableFile(atPath: binary.path), FileManager.default.fileExists(atPath: metallib.path) { return binary }
    }
    return nil
}

private final class Events: @unchecked Sendable {
    private let lock = NSLock()
    private var all: [BraidEvent] = []

    func add(_ event: BraidEvent) { lock.withLock { all.append(event) } }
    var promoted: [String] {
        lock.withLock { all.compactMap { if case .node(let name, .promoted) = $0 { return name } else { return nil } } }
    }
}

/// Waits until every named node has a live version and nothing left to do.
private func settle(_ session: BraidSession, _ names: [String], after versions: [String: Int] = [:], timeout: TimeInterval = 600) async throws {
    let deadline = Date().addingTimeInterval(timeout)
    while Date() < deadline {
        let done = names.allSatisfy { name in
            guard let state = session.state(name) else { return false }
            let busy = state.stage.isBusy || state.ladder.contains { $0.status == .pending || $0.status == .running }
            return !busy && state.isLive && state.versions > (versions[name] ?? 0)
        }
        if done { return }
        try await Task.sleep(nanoseconds: 300_000_000)
    }
    Issue.record("nodes did not settle: \(session.states.map { "\($0.name) \($0.stage)" })")
}

private func run(offline: Bool) async throws {
    guard let binary = raolmBinary() else {
        Issue.record("no raolm binary with its mlx.metallib under .build; run scripts/test.sh, which builds it")
        return
    }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("raolm-braid-proc-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    let tokenizer = try await RaoTokenizer.load()
    var options = BraidOptions(root: DataRoot(url: root), executable: binary)
    options.offline = offline
    options.batch = 3
    options.documentsPerNode = 5
    options.seed = 7
    options.settings.config = Rig.config
    options.settings.seqLen = 64
    options.settings.lr = 4e-3
    options.settings.scratchSteps = 600
    options.settings.continueSteps = 400
    options.settings.evalSteps = 24
    options.settings.earlyStop = 0.95
    options.settings.memorisedFloor = 0.6
    options.settings.probes = 1
    if !offline {
        options.httpBase = PortProbe.freePort() ?? 18195
        options.grpcBase = PortProbe.freePort() ?? 19195
    }
    let vocabulary = try BraidVocabulary.ensure(layout: options.layout, config: Rig.config, tokenizer: tokenizer)
    let events = Events()
    let session = try BraidSession(options: options, vocabularySHA256: vocabulary.sha256) { events.add($0) }
    session.exampleTokenizer = tokenizer
    do {
        try await session.start()
        let names = session.names
        for name in names {
            let hello = try #require(session.hello(name))
            #expect(hello.pid > 0 && hello.pid != getpid())
            #expect(hello.offline == offline)
            if !offline { #expect(hello.threadPID != nil && hello.httpPort != nil) }
        }
        for name in names { #expect(try await session.feed(name).count == 3) }
        try await settle(session, names)
        #expect(Set(events.promoted) == Set(names))

        let umbrella = BraidUmbrella(vocabulary: vocabulary, tokenizer: tokenizer)
        let links = try session.links()
        #expect(links.map(\.descriptor.name) == names)
        #expect(names == BraidNodeSpec.defaults.map(\.name))
        #expect(MockWorld.Record.load(options.layout)?.names == names, "world.json records what the nodes were fed from")
        let examples = session.examples(tokenizer: tokenizer).filter { $0.resolvedKind == .fact }
        var owned = 0
        for example in examples.prefix(4) {
            var params = GenerationParameters(tapLayer: links[0].descriptor.tapLayer, alpha: links[0].descriptor.alpha)
            params.maxTokens = 6
            var streamed = 0
            var generation = try umbrella.generate(links: links, request: BraidRequest(
                promptTokens: example.promptTokens, promptText: example.promptText, params: params)) { _ in streamed += 1 }
            #expect(streamed == generation.tokens.count)
            if generation.traces.first(where: { !$0.isPrompt })?.dominantStrand(threshold: 0.5)?.strand == example.node { owned += 1 }
            let report = try await CitationVerifier.verify(&generation, reader: session.reader(), tokenizer: tokenizer)
            #expect(report.checks.allSatisfy { $0.status == .verified }, "\(report.checks.map(\.status))")
        }
        #expect(owned >= 3, "\(owned)/4 answers came from their own Thread")

        // A withdrawal through the node's own corpus reaches its index.
        let name = names[1]
        let mark = [name: session.state(name)?.versions ?? 0]
        let gone = try #require(try await session.withdraw(name))
        try await settle(session, [name], after: mark)
        let descriptor = try #require(try session.links().first { $0.descriptor.name == name }?.descriptor)
        #expect(!descriptor.partitions.contains { $0.documentID == gone.id })
    } catch {
        await session.stop()
        throw error
    }
    let pids = session.states.flatMap { [$0.pid, $0.threadPID].compactMap { $0 } }
    await session.stop()
    try await Task.sleep(nanoseconds: 500_000_000)
    #expect(!pids.isEmpty && pids.allSatisfy { kill($0, 0) != 0 }, "every node and Thread process is gone")
}

/// What happened, in order: `name:training` the first time a node reports training, `name:live` when one is promoted.
private final class Order: @unchecked Sendable {
    private let lock = NSLock()
    private var seen: [String] = []

    func add(_ event: BraidEvent) {
        lock.withLock {
            switch event {
            case .node(let name, .state(let state)) where state.stage == .training && !seen.contains("\(name):training"):
                seen.append("\(name):training")
            case .node(let name, .promoted):
                seen.append("\(name):live")
            default:
                break
            }
        }
    }
    var events: [String] { lock.withLock { seen } }
}

/// Two nodes fed without training, then synced one at a time: the second trains only once the first is live.
private func throttled() async throws {
    guard let binary = raolmBinary() else {
        Issue.record("no raolm binary with its mlx.metallib under .build; run scripts/test.sh, which builds it")
        return
    }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("raolm-braid-throttle-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    let tokenizer = try await RaoTokenizer.load()
    var options = BraidOptions(root: DataRoot(url: root), executable: binary)
    options.offline = true
    options.nodes = try BraidNodeSpec.parse("ambient,craft")
    options.documentsPerNode = 5
    options.seed = 7
    options.syncOnStart = false
    options.settings.config = Rig.config
    options.settings.seqLen = 64
    options.settings.lr = 4e-3
    options.settings.scratchSteps = 600
    options.settings.evalSteps = 24
    options.settings.earlyStop = 0.95
    options.settings.memorisedFloor = 0.6
    options.settings.probes = 1
    let vocabulary = try BraidVocabulary.ensure(layout: options.layout, config: Rig.config, tokenizer: tokenizer)
    let order = Order()
    let session = try BraidSession(options: options, vocabularySHA256: vocabulary.sha256) { order.add($0) }
    do {
        try await session.start()
        for name in session.names {
            let fed = try await session.feed(name, count: 3, sync: false)
            #expect(fed.count == 3)
        }
        let trained = session.states.filter { $0.liveVersion != nil }.count
        #expect(trained == 0, "a feed without a sync trains nothing")
        var sent: [String] = []
        try await session.sync(session.names, atOnce: 1, stall: 300, sent: { name, _, inFlight in
            sent.append(name)
            #expect(inFlight == 1)
        })
        #expect(sent == ["ambient", "craft"])
        let live = session.states.filter(\.isLive).count
        #expect(live == 2)
        let events = order.events
        let first = try #require(events.firstIndex(of: "ambient:live"))
        let second = try #require(events.firstIndex(of: "craft:training"))
        #expect(first < second, "\(events)")
    } catch {
        await session.stop()
        throw error
    }
    await session.stop()
}

/// The nodes that exited, in order.
private final class Exits: @unchecked Sendable {
    private let lock = NSLock()
    private var seen: [String] = []

    func add(_ event: BraidEvent) {
        if case .exited(let name, _) = event { lock.withLock { seen.append(name) } }
    }
    var names: [String] { lock.withLock { seen } }
}

private func waitUntil(_ timeout: TimeInterval, _ condition: () throws -> Bool) async throws -> Bool {
    let deadline = Date().addingTimeInterval(timeout)
    while Date() < deadline {
        if try condition() { return true }
        try await Task.sleep(nanoseconds: 200_000_000)
    }
    return try condition()
}

/// Two nodes that dial a hosting umbrella over TCP (rule H2): trained and asked; one killed in the
/// middle of a generation fails it without a hang and is dropped; started again, it dials in and answers.
private func hosted() async throws {
    guard let binary = raolmBinary() else {
        Issue.record("no raolm binary with its mlx.metallib under .build; run scripts/test.sh, which builds it")
        return
    }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("raolm-braid-hosted-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    let tokenizer = try await RaoTokenizer.load()
    var options = BraidOptions(root: DataRoot(url: root), executable: binary)
    options.offline = true
    options.nodes = try BraidNodeSpec.parse("ambient,craft")
    options.documentsPerNode = 5
    options.seed = 7
    options.listen = 0
    options.connectTimeout = 120
    options.settings.config = Rig.config
    options.settings.seqLen = 64
    options.settings.lr = 4e-3
    options.settings.scratchSteps = 600
    options.settings.evalSteps = 24
    options.settings.earlyStop = 0.95
    options.settings.memorisedFloor = 0.6
    options.settings.probes = 1
    let vocabulary = try BraidVocabulary.ensure(layout: options.layout, config: Rig.config, tokenizer: tokenizer)
    let exits = Exits()
    let session = try BraidSession(options: options, vocabularySHA256: vocabulary.sha256) { exits.add($0) }
    var restarted: Process?
    do {
        try await session.start()
        let port = try #require(session.listeningPort)
        let names = session.names
        for name in names {
            let hello = try #require(session.hello(name))
            #expect(hello.pid > 0 && hello.pid != getpid())
        }
        for name in names {
            let fed = try await session.feed(name, count: 3)
            #expect(fed.count == 3)
        }
        try await settle(session, names)

        let umbrella = BraidUmbrella(vocabulary: vocabulary, tokenizer: tokenizer)
        let links = try session.links()
        #expect(links.map(\.descriptor.name) == names)
        // A question's context now crosses the wire.
        _ = try links[0].context(stem: tokenizer.encode(" was founded in"), subject: nil, k: 8, floor: 0).wait(timeout: 30)
        let prompt = tokenizer.encode("The weather today")
        var params = GenerationParameters(tapLayer: links[0].descriptor.tapLayer, alpha: links[0].descriptor.alpha)
        params.maxTokens = 6
        let first = try umbrella.generate(links: links, request: BraidRequest(promptTokens: prompt, promptText: "The weather today", params: params))
        #expect(!first.tokens.isEmpty)

        // Killed in the middle of a generation: the generation fails, well within the link timeout.
        let victim = names[1]
        let pid = try #require(session.hello(victim)?.pid)
        params.maxTokens = 48
        var killed = false
        let started = Date()
        var failed = false
        do {
            _ = try umbrella.generate(links: links, request: BraidRequest(promptTokens: prompt, promptText: "The weather today", params: params)) { _ in
                if !killed {
                    killed = true
                    kill(pid, SIGKILL)
                    usleep(200_000)
                }
            }
        } catch {
            failed = true
        }
        #expect(killed && failed, "a generation whose node is killed fails")
        #expect(Date().timeIntervalSince(started) < 30)
        let dropped = try await waitUntil(10) { exits.names.contains(victim) }
        #expect(dropped, "the killed node's connection is dropped at once")
        #expect(try session.links().map(\.descriptor.name) == [names[0]])

        // Started again, it dials in and answers within 30 s.
        let child = Process()
        child.executableURL = binary
        child.arguments = ["node", "serve", "--options", options.layout.node(victim).directory.appendingPathComponent(NodeServerOptions.fileName).path,
                           "--connect", "127.0.0.1:\(port)"]
        child.standardInput = FileHandle.nullDevice
        child.standardOutput = FileHandle.nullDevice
        child.standardError = FileHandle.nullDevice
        try child.run()
        restarted = child
        let back = try await waitUntil(30) { session.hello(victim)?.pid == child.processIdentifier && session.state(victim)?.isLive == true }
        #expect(back, "the restarted node dials in and is greeted with its live version")
        let again = try session.links()
        #expect(again.map(\.descriptor.name) == names)
        params.maxTokens = 6
        let after = try umbrella.generate(links: again, request: BraidRequest(promptTokens: prompt, promptText: "The weather today", params: params))
        #expect(after.tokens == first.tokens, "the braid answers as it did before the node was lost")
    } catch {
        await session.stop()
        restarted?.terminate()
        throw error
    }
    await session.stop()
    if let restarted {
        _ = try? await waitUntil(10) { !restarted.isRunning }
        if restarted.isRunning { restarted.terminate() }
    }
}

extension BraidMLXSuites {
    @Suite("Braid over node processes", .serialized)
    struct BraidProcessTests {
        @Test("two nodes synced one at a time: the second trains only once the first is live")
        func throttledSync() async throws {
            try await throttled()
        }

        @Test("two nodes that dial in over TCP: asked; one killed mid-generation fails it and is dropped; restarted, it dials in and answers")
        func hostedNodes() async throws {
            try await hosted()
        }

        @Test("the default node processes (ambient, craft, veil) on offline corpora: fed, live, routed, verified, withdrawn, stopped")
        func offline() async throws {
            try await run(offline: true)
        }

        @Test("the default node processes each with a real Thread", .enabled(if: threadTests))
        func threads() async throws {
            try await run(offline: false)
        }
    }
}
