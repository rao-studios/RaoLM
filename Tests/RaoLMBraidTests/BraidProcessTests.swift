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

@Suite("Braid over node processes", .enabled(if: mlxTests), .serialized)
struct BraidProcessTests {
    @Test("the default node processes (ambient, craft, veil) on offline corpora: fed, live, routed, verified, withdrawn, stopped")
    func offline() async throws {
        try await run(offline: true)
    }

    @Test("the default node processes each with a real Thread", .enabled(if: threadTests))
    func threads() async throws {
        try await run(offline: false)
    }
}
