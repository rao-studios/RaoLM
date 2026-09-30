import Foundation
import MLX
import Testing

@testable import RaoLMBraid
@testable import RaoLMCore
@testable import RaoLMModel
@testable import RaoLMProvenance
@testable import RaoLMTraining

/// A braid whose nodes run in the test's own process, on offline corpora.
struct Rig {
    static let config = RaoLMConfig(hiddenSize: 128, intermediateSize: 256, numHiddenLayers: 2, numAttentionHeads: 4,
                                    numKeyValueHeads: 2, maxPositionEmbeddings: 256)

    let root: URL
    let layout: BraidLayout
    let tokenizer: RaoTokenizer
    let vocabulary: VocabularyPack
    let world: MockWorld
    var settings: HypervisorSettings

    init(names: [String], documentsPerNode: Int, seed: UInt64) async throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("raolm-braid-\(UUID().uuidString)")
        layout = BraidLayout(root: root)
        tokenizer = try await RaoTokenizer.load()
        vocabulary = VocabularyPack.seeded(config: Self.config, tokenizerSHA256: tokenizer.tokenizerSHA256, seed: 3, headScale: 2)
        world = try MockWorld(names: names, seed: seed, documentsPerNode: documentsPerNode)
        var settings = HypervisorSettings()
        settings.config = Self.config
        settings.seqLen = 64
        settings.batchSize = 4
        settings.lr = 4e-3
        settings.scratchSteps = 600
        settings.continueSteps = 400
        settings.evalSteps = 24
        settings.earlyStop = 0.95
        settings.memorisedFloor = 0.6
        settings.probes = 2
        self.settings = settings
    }

    func source(_ name: String) throws -> DirectoryCorpusSource {
        let node = layout.node(name)
        return DirectoryCorpusSource(directory: node.offlineCorpus, slug: name, owner: "raolm-test", threadID: try DirectoryCorpusSource.nodeID(node))
    }

    func hypervisor(_ name: String) throws -> ThreadHypervisor {
        try ThreadHypervisor(name: name, label: name.capitalized, layout: layout.node(name), vocabulary: vocabulary, tokenizer: tokenizer,
                             source: try source(name), settings: settings, owner: "raolm-test")
    }

    func feed(_ name: String, _ count: Int) async throws -> [CorpusDocument] {
        try await MockFeeder.feed(node: name, count: count, world: world, layout: layout.node(name), source: try source(name))
    }

    func nodes(_ nodes: [ThreadHypervisor]) -> [BraidExample.Node] {
        nodes.map {
            (name: $0.name, label: $0.label, threadID: $0.source.threadID,
             documents: MockFeeder.present(node: $0.name, world: world, layout: layout.node($0.name)))
        }
    }

    func examples(_ nodes: [ThreadHypervisor]) -> [BraidExample] {
        BraidExample.build(nodes: self.nodes(nodes), tokenizer: tokenizer)
    }

    func generator(_ nodes: [ThreadHypervisor]) throws -> BraidedGenerator {
        try BraidedGenerator(links: nodes.compactMap { $0.live }.map { LocalStrandLink(strand: $0, vocabularySHA256: vocabulary.sha256) },
                             head: UmbrellaHead(vocabulary: vocabulary), tokenizer: tokenizer)
    }

    func cleanUp() { try? FileManager.default.removeItem(at: root) }
}

@Suite("Braid in one process", .enabled(if: mlxTests), .serialized)
struct BraidInProcessTests {
    @Test("a braid of one Thread reproduces CitedGenerator number for number")
    func oneThread() async throws {
        var rig = try await Rig(names: ["solo"], documentsPerNode: 3, seed: 11)
        defer { rig.cleanUp() }
        rig.settings.scratchSteps = 48
        rig.settings.memorisedFloor = 0
        _ = try await rig.feed("solo", 3)
        let node = try rig.hypervisor("solo")
        #expect(try node.sync() == [1])
        let live = try #require(node.live)
        #expect(VocabularyPack.fingerprint(of: live.model) == rig.vocabulary.sha256)

        let example = try #require(rig.examples([node]).first { $0.node == "solo" && $0.resolvedKind == .fact })
        var params = live.context.defaultParameters()
        params.maxTokens = 12
        let single = try live.context.generator().generate(GenerationRequest(
            promptTokens: example.promptTokens, promptText: example.promptText, params: params))
        var streamed = 0
        for gating in BraidGating.allCases {
            var prompted = 0
            let braided = try rig.generator([node]).generate(
                BraidRequest(promptTokens: example.promptTokens, promptText: example.promptText, params: params, gating: gating),
                onPrompt: { prompted = $0.count }
            ) { step in
                streamed += 1
                #expect(step.open == ["solo"])
            }
            #expect(prompted == example.promptTokens.count - 1, "\(gating)")
            #expect(streamed == braided.tokens.count, "\(gating)")
            streamed = 0
            #expect(braided.tokens == single.tokens, "\(gating)")
            #expect(braided.traces.count == single.traces.count)
            for (a, b) in zip(braided.traces, single.traces) {
                #expect(a.token == b.token && a.isPrompt == b.isPrompt)
                #expect(abs(a.lmProb - b.lmProb) < 1e-5 && abs(a.mixedProb - b.mixedProb) < 1e-5, "\(gating)")
                #expect(abs(a.lmEntropy - b.lmEntropy) < 1e-4 && abs(a.knnEntropy - b.knnEntropy) < 1e-5)
                #expect(a.neighbours.map(\.cited) == b.neighbours.map(\.cited))
                #expect(a.neighbours.map(\.entry) == b.neighbours.map(\.entry))
                #expect(a.citations.map(\.row) == b.citations.map(\.row))
                #expect(a.citations.map(\.address.threadID) == b.citations.map(\.address.threadID))
                #expect(abs((a.confidence ?? -1) - (b.confidence ?? -1)) < 1e-5)
                #expect(a.strands?.count == 1 && abs((a.strands?.first?.share ?? 0) - 1) < 1e-5)
                #expect(a.threadEntropy == 0)
            }
            #expect(braided.spans.map(\.row) == single.spans.map(\.row))
            #expect(braided.spans.map(\.source) == single.spans.map(\.source))
            #expect(braided.braid?.gating == gating)
            #expect(braided.prompt.tokenTexts?.count == example.promptTokens.count)
        }
        // The trajectory in the gate, and asking by manner, change nothing for one Thread.
        for use in BraidGate.TrajectoryUse.allCases {
            for ask in BraidGate.Ask.allCases {
                var gate = BraidGate()
                gate.trajectory = use
                gate.trajectoryBeta = 4
                gate.ask = ask
                gate.askFloor = 0.99
                let traced = try rig.generator([node]).generate(BraidRequest(
                    promptTokens: example.promptTokens, promptText: example.promptText, params: params, gate: gate))
                #expect(traced.tokens == single.tokens, "\(use) \(ask)")
                for (a, b) in zip(traced.traces, single.traces) {
                    #expect(a.token == b.token && abs(a.mixedProb - b.mixedProb) < 1e-5, "\(use) \(ask)")
                    #expect(a.strands?.first?.trajectory != nil, "every position records the Thread's trajectory")
                }
            }
        }
        // A session's trajectory is the same whether its tokens arrive at once or one by one.
        let text = example.promptTokens
        let batch = try live.open(session: "batch", tokens: text, k: 16)
        var stepwise = try live.open(session: "stepwise", tokens: [text[0]], k: 16)
        for token in text.dropFirst() { stepwise.append(try live.advance(session: "stepwise", token: token, k: 16)) }
        live.close(session: "batch")
        live.close(session: "stepwise")
        #expect(stepwise.map { $0.trajectory?.length } == batch.map { $0.trajectory?.length })
        #expect(stepwise.map { $0.hits.map(\.entry) } == batch.map { $0.hits.map(\.entry) })

        let braided = try rig.generator([node]).generate(BraidRequest(
            promptTokens: example.promptTokens, promptText: example.promptText, params: params))
        let braid = try #require(braided.braid)
        #expect(braid.strands.map(\.name) == ["solo"])
        #expect(braid.strands[0].threadID == rig.world.shards["solo"].map { _ in try? DirectoryCorpusSource.nodeID(rig.layout.node("solo")) } ?? nil)
    }

    @Test("the default Threads: each answers its own facts, a feed makes new facts answerable, a withdrawal stops its citations")
    func threads() async throws {
        let names = BraidNodeSpec.defaults.map(\.name)
        let rig = try await Rig(names: names, documentsPerNode: 5, seed: 7)
        defer { rig.cleanUp() }
        for name in names { _ = try await rig.feed(name, 3) }
        let hypervisors = try names.map { try rig.hypervisor($0) }
        let ambient = hypervisors[0]
        let craft = hypervisors[1]
        var served = 0
        ambient.service = { served += 1 }
        var states: [StrandState] = []
        ambient.onState = { states.append($0) }
        for hypervisor in hypervisors { #expect(try hypervisor.sync() == [1]) }
        #expect(served > 10, "the node serves between training steps")
        #expect(states.contains { $0.stage == .training } && states.last?.stage == .live)
        #expect(states.last?.cells.count == ambient.live?.index.partitions.count)
        #expect(states.last?.cells.allSatisfy { $0.state != .pending } == true)
        #expect(ambient.state.ladder.first { $0.name == "reindex" }?.status == .skipped)
        #expect(ambient.state.ladder.first { $0.name == "live" }?.status == .done)

        let threadOf = Dictionary(uniqueKeysWithValues: hypervisors.map { ($0.name, $0.source.threadID) })
        #expect(Set(threadOf.values.compactMap { $0 }).count == names.count)
        let generator = try rig.generator(hypervisors)
        #expect(generator.braid.strands.map(\.name) == names)
        var answered = 0
        var owned = 0
        var total = 0
        for example in rig.examples(hypervisors).filter({ $0.resolvedKind == .fact }).prefix(8) {
            var params = GenerationParameters(tapLayer: 1, alpha: 0.5)
            params.maxTokens = 6
            let generation = try generator.generate(BraidRequest(promptTokens: example.promptTokens, promptText: example.promptText, params: params,
                                                                 gating: .posterior))
            let first = try #require(generation.traces.first { !$0.isPrompt })
            total += 1
            if generation.text.hasPrefix(example.expected ?? "\u{0}") { answered += 1 }
            if first.dominantStrand(threshold: 0.5)?.strand == example.node,
               first.citations.first?.address.threadID == threadOf[example.node!] { owned += 1 }
            #expect(abs((first.strands ?? []).map(\.share).reduce(0, +) - 1) < 1e-3)
            // Every partition a generation lists names its own Thread.
            #expect(generation.partitions.allSatisfy { $0.threadID != nil })
        }
        #expect(total >= 4)
        #expect(Float(owned) / Float(total) >= 0.75, "\(owned)/\(total) answers came from their own Thread")
        #expect(Float(answered) / Float(total) >= 0.5, "\(answered)/\(total) answered")

        // The braided gate: owners still lead their answers; a subject no Thread holds, or text
        // about nothing either holds, leaves both Threads asked; a prompt that moves to the other
        // Thread's fact moves the lead.
        func ask(_ example: BraidExample, maxTokens: Int = 6) throws -> CitedGeneration {
            var params = GenerationParameters(tapLayer: 1, alpha: 0.5)
            params.maxTokens = maxTokens
            return try generator.generate(BraidRequest(promptTokens: example.promptTokens, promptText: example.promptText, params: params,
                                                       gating: .braided))
        }
        func firstAnswer(_ generation: CitedGeneration) throws -> TokenTrace { try #require(generation.traces.first { !$0.isPrompt }) }
        let nodes = rig.nodes(hypervisors)
        var braidedOwned = 0
        var braidedAnswered = 0
        let facts = BraidExample.facts(nodes: nodes, tokenizer: rig.tokenizer, perNode: 4)
        for example in facts {
            let generation = try ask(example)
            let first = try firstAnswer(generation)
            if first.dominantStrand(threshold: 0.5)?.strand == example.node,
               first.citations.first?.address.threadID == threadOf[example.node!] { braidedOwned += 1 }
            if generation.text.hasPrefix(example.expected ?? "\u{0}") { braidedAnswered += 1 }
            #expect(abs((first.strands ?? []).map(\.share).reduce(0, +) - 1) < 1e-3)
            #expect(first.strands?.allSatisfy { $0.memory != nil && $0.backs != nil } == true)
            #expect(first.candidates?.first?.token == first.token)
            #expect(generation.braid?.gating == .braided && generation.braid?.gate == BraidRequest.defaultGate)
        }
        print("braided: owned \(braidedOwned)/\(facts.count), answered \(braidedAnswered)/\(facts.count) (posterior: owned \(owned)/\(total), answered \(answered)/\(total))")
        #expect(Float(braidedOwned) / Float(facts.count) >= 0.75, "\(braidedOwned)/\(facts.count) braided answers came from their own Thread")
        #expect(Float(braidedAnswered) / Float(facts.count) >= 0.5)

        // A subject no Thread holds moves nothing; its template leans the gate only towards a
        // Thread that knows that template (few documents per node: a kind may live on one only).
        // Text about nothing either holds leaves both Threads asked.
        var asked = 0
        var leans: [Float] = []
        let strangers = BraidExample.unknowns(nodes: nodes, tokenizer: rig.tokenizer, count: 4) + BraidExample.generic(tokenizer: rig.tokenizer)
        for example in strangers {
            let generation = try ask(example, maxTokens: 2)
            let first = try firstAnswer(generation)
            let shares = first.strands ?? []
            let everyone = shares.allSatisfy(\.open)
            if everyone { asked += 1 }
            leans.append(shares.map(\.gate).max() ?? 1)
            if example.resolvedKind == .generic { #expect(everyone, "\(example.promptText)") }
            if !everyone, let leader = shares.max(by: { $0.gate < $1.gate }) {
                func predicted(_ name: String) -> Int {
                    generation.traces.filter { $0.isPrompt && ($0.strands?.first { $0.strand == name }?.alone ?? 0) >= 0.05 }.count
                }
                for other in shares where other.strand != leader.strand {
                    #expect(predicted(leader.strand) > predicted(other.strand), "\(example.promptText): shut out without cause")
                }
            }
        }
        print("braided strangers: all asked \(asked)/\(strangers.count), largest gates \(leans.map { String(format: "%.2f", $0) })")

        var moved = 0
        let pairs = BraidExample.pairs(nodes: nodes, tokenizer: rig.tokenizer, perPair: 2)
        for example in pairs {
            let first = try firstAnswer(try ask(example))
            if first.dominantStrand(threshold: 0.5)?.strand == example.node { moved += 1 }
        }
        print("braided pairs: second owner leads \(moved)/\(pairs.count)")
        #expect(!pairs.isEmpty && Float(moved) / Float(pairs.count) >= 0.75, "\(moved)/\(pairs.count) pairs moved the lead")

        // A document one Thread holds, asked as it was written, traces on that Thread and no other.
        let held = try #require(rig.world.documents(for: names[0]).first { document in
            ambient.live?.index.partitions.contains { $0.documentID == document.id } == true
        })
        let heldTokens = Array(rig.tokenizer.encode(held.partitions[0].text).prefix(60))
        var longest: [String: Int] = [:]
        for hypervisor in hypervisors {
            let strand = try #require(hypervisor.live)
            let steps = try strand.open(session: "held", tokens: heldTokens, k: 16)
            strand.close(session: "held")
            longest[hypervisor.name] = steps.compactMap { $0.trajectory?.length }.max() ?? 0
        }
        #expect((longest[names[0]] ?? 0) >= 24, "\(longest)")
        #expect(names.dropFirst().allSatisfy { (longest[$0] ?? 0) < (longest[names[0]] ?? 0) }, "\(longest)")

        // A feed: the new document is citable at once (reindex), then learned (train).
        let added = try await rig.feed(names[0], 1)
        #expect(added.count == 1)
        let promoted = try ambient.sync()
        #expect(promoted.first == 2, "the reindex goes live first")
        #expect(ambient.liveVersion?.version == promoted.last)
        let kinds = try promoted.map { try JSONCoding.read(NodeVersion.self, from: rig.layout.node(names[0]).version($0).appendingPathComponent(NodeVersion.fileName)).kind }
        #expect(kinds.first == .reindex)
        #expect(ambient.live?.index.partitions.contains { $0.documentID == added[0].id } == true)
        #expect(ambient.state.lastChange?.hasPrefix("+") == true)

        // A withdrawal: the index stops holding the document at once.
        let gone = try #require(try await MockFeeder.withdraw(node: names[1], world: rig.world, layout: rig.layout.node(names[1]), source: try rig.source(names[1])))
        #expect(try craft.sync().count == 1)
        let reindex = try #require(craft.liveVersion)
        #expect(reindex.kind == .reindex)
        #expect(reindex.change.removedDocuments == [gone.id])
        #expect(craft.live?.index.partitions.contains { $0.documentID == gone.id } == false)
        #expect(reindex.gates.first { $0.name == "withdrawn" }?.passed == true)
        let after = try rig.generator(hypervisors)
        if let fact = gone.facts.first {
            let generation = try after.generate(BraidRequest(
                promptTokens: rig.tokenizer.encode(fact.sentence.prefix(fact.prompt.count).description), promptText: fact.prompt,
                params: GenerationParameters(tapLayer: 1, alpha: 0.5)))
            #expect(!generation.traces.flatMap(\.citations).contains { $0.address.documentID == gone.id })
        }
        // Nothing changed: nothing to do.
        #expect(try craft.sync().isEmpty)
        #expect(craft.state.stage == .live)
    }
}
