import Foundation
import Testing

@testable import RaoLMBraid
@testable import RaoLMCore
@testable import RaoLMModel
import RaoLMTraining

@Suite("Mock world, datasets and the world record")
struct MockWorldTests {
    private func temporary() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("raolm-world-\(UUID().uuidString)", isDirectory: true)
    }

    @Test("node lists parse, capitalise their labels, and refuse bad or repeated names")
    func parse() throws {
        #expect(BraidNodeSpec.defaults.map(\.name) == ["ambient", "craft", "veil"])
        #expect(try BraidNodeSpec.parse("ambient, craft,veil") == BraidNodeSpec.defaults)
        #expect(throws: (any Error).self) { _ = try BraidNodeSpec.parse("ambient,ambient") }
        #expect(throws: (any Error).self) { _ = try BraidNodeSpec.parse("Ambient") }
        #expect(throws: (any Error).self) { _ = try BraidNodeSpec.parse(" , ") }
    }

    @Test("world.json from before the shape shrank still decodes, and examples from before kinds still decode")
    func olderRecords() throws {
        let record = #"{"budget":{"veil":4},"names":["ambient","craft","veil"],"seed":42,"shape":{"documentsPerNode":40,"multipliers":{"veil":4},"shared":8}}"#
        let decoded = try JSONCoding.decoder().decode(MockWorld.Record.self, from: Data(record.utf8))
        #expect(decoded.shape == MockShape(documentsPerNode: 40) && decoded.names.count == 3 && decoded.dataset == nil)
        let old = #"{"label":"x","node":"a","promptTokens":[1],"promptText":"x","expected":"y","kind":"fact"}"#
        #expect(try JSONCoding.decoder().decode(BraidExample.self, from: Data(old.utf8)).copier == nil)
    }

    @Test("an offline Thread dates each document by when it was deposited, and the date reaches its partitions")
    func createdAt() async throws {
        let root = temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let layout = BraidLayout(root: root)
        let world = try MockWorld(names: ["ambient"], seed: 3, documentsPerNode: 3)
        let node = layout.node("ambient")
        let source = DirectoryCorpusSource(directory: node.offlineCorpus, slug: "ambient", owner: "o", threadID: try DirectoryCorpusSource.nodeID(node))
        let first = try await MockFeeder.feed(node: "ambient", count: 1, world: world, layout: node, source: source)
        try await Task.sleep(nanoseconds: 20_000_000)
        let second = try await MockFeeder.feed(node: "ambient", count: 1, world: world, layout: node, source: source)
        let snapshot = try await source.export()
        let dates = Dictionary(uniqueKeysWithValues: snapshot.documents.map { ($0.id, $0.createdAt) })
        let older = try #require(dates[first[0].id])
        let newer = try #require(dates[second[0].id])
        #expect(older > 0 && older < newer)
        let tokenizer = try await RaoTokenizer.load()
        let refs = TokenizedCorpus(snapshot: snapshot, tokenizer: tokenizer).partitionRefs()
        #expect(refs.allSatisfy { $0.createdAt == dates[$0.documentID] })
        // A snapshot with no dates leaves its partitions undated.
        let undated = TokenizedCorpus(snapshot: .offline(world.shards["ambient"]!), tokenizer: tokenizer).partitionRefs()
        #expect(undated.allSatisfy { $0.createdAt == nil })
    }

    @Test("named no nodes, a braid keeps its own: world.json's, or the nodes already fed; fresh or named, the options stand")
    func adoptNodes() throws {
        let root = temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        func session(_ nodes: String = "ambient,craft,veil", adopt: Bool = true, fresh: Bool = false) throws -> BraidSession {
            var options = BraidOptions(root: DataRoot(url: root), executable: URL(fileURLWithPath: "/usr/bin/true"))
            options.nodes = try BraidNodeSpec.parse(nodes)
            options.documentsPerNode = 2
            options.adoptNodes = adopt
            options.fresh = fresh
            return try BraidSession(options: options, vocabularySHA256: "v") { _ in }
        }
        // An empty data directory: the defaults.
        #expect(try session().options.nodes.map(\.name) == ["ambient", "craft", "veil"])
        // Two nodes fed before world.json, from a two-node world.
        let two = try session("ambient,craft", adopt: false)
        for name in ["ambient", "craft"] {
            var feed = FeedState()
            feed.deposited = two.world.documents(for: name).map(\.id)
            try FileManager.default.createDirectory(at: two.layout.node(name).directory, withIntermediateDirectories: true)
            try JSONCoding.write(feed, to: two.layout.node(name).feed)
        }
        let kept = try session()
        #expect(kept.options.nodes.map(\.name) == ["ambient", "craft"])
        try kept.checkWorld()
        #expect(throws: MockWorldError.self) { try session(adopt: false).checkWorld() }
        #expect(try session(fresh: true).options.nodes.count == 3)
        // With world.json, its nodes, seed and shape.
        var record = two.worldRecord
        record.seed = 42
        try record.save(two.layout)
        let recorded = try session()
        #expect(recorded.options.nodes.map(\.name) == ["ambient", "craft"] && recorded.options.documentsPerNode == 2)
        try recorded.checkWorld()
    }

    @Test("a session refuses nodes fed from another world, and --fresh starts over")
    func worldGuard() throws {
        let root = temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        func session(_ nodes: String, documents: Int = 2, fresh: Bool = false) throws -> BraidSession {
            var options = BraidOptions(root: DataRoot(url: root), executable: URL(fileURLWithPath: "/usr/bin/true"))
            options.nodes = try BraidNodeSpec.parse(nodes)
            options.documentsPerNode = documents
            options.fresh = fresh
            return try BraidSession(options: options, vocabularySHA256: "v") { _ in }
        }
        let two = try session("ambient,craft")
        try two.checkWorld()
        try two.worldRecord.save(two.layout)
        #expect(MockWorld.Record.load(two.layout)?.names == ["ambient", "craft"])
        try session("ambient,craft").checkWorld()
        #expect(throws: MockWorldError.self) { try session("ambient,craft,veil").checkWorld() }
        #expect(throws: MockWorldError.self) { try session("ambient,craft", documents: 3).checkWorld() }
        try session("ambient,craft,veil", fresh: true).checkWorld()
        // Without world.json (a braid made before it), a feed of documents this world lacks is refused.
        try FileManager.default.removeItem(at: two.layout.world)
        var feed = FeedState()
        feed.deposited = two.world.documents(for: "craft").map(\.id)
        try FileManager.default.createDirectory(at: two.layout.node("craft").directory, withIntermediateDirectories: true)
        try JSONCoding.write(feed, to: two.layout.node("craft").feed)
        try session("ambient,craft").checkWorld()
        #expect(throws: MockWorldError.self) { try session("ambient,craft,veil").checkWorld() }
    }

    @Test("the training arm is part of the world: a braid keeps its own unless one is named, and refuses another")
    func armGuard() throws {
        let root = temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        func options(arm: String?, fresh: Bool = false) throws -> BraidOptions {
            var options = BraidOptions(root: DataRoot(url: root), executable: URL(fileURLWithPath: "/usr/bin/true"))
            options.nodes = try BraidNodeSpec.parse("ambient,craft")
            options.documentsPerNode = 2
            options.fresh = fresh
            if let arm {
                options.settings.arm = arm
                options.armRequested = true
            }
            options.restorePreset()
            return options
        }
        func session(arm: String?) throws -> BraidSession { try BraidSession(options: try options(arm: arm), vocabularySHA256: "v") { _ in } }
        let armed = try session(arm: "passage-break")
        try armed.checkWorld()
        try armed.worldRecord.save(armed.layout)
        #expect(MockWorld.Record.load(armed.layout)?.arm == "passage-break")
        #expect(MockWorld.Record.load(armed.layout)?.summary.hasSuffix(" · arm passage-break") == true)
        // Named no arm, the braid keeps its own; fresh, it starts from what was asked.
        #expect(try options(arm: nil).settings.arm == "passage-break")
        try session(arm: nil).checkWorld()
        #expect(try options(arm: nil, fresh: true).settings.arm == nil)
        // Nodes trained without the arm are another world.
        var plain = armed.worldRecord
        plain.arm = nil
        #expect(!plain.sameWorld(as: armed.worldRecord))
        try plain.save(armed.layout)
        #expect(throws: MockWorldError.self) { try session(arm: "passage-break").checkWorld() }
        try session(arm: nil).checkWorld()
    }

    @Test("a new base braid trains the adopted arm; a recorded braid keeps its own, none included")
    func baseArm() throws {
        let root = temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        func options(preset: String?, arm: String? = nil, fresh: Bool = false) -> BraidOptions {
            var options = BraidOptions(root: DataRoot(url: root), executable: URL(fileURLWithPath: "/usr/bin/true"))
            if let preset {
                options.settings.preset = preset
                options.presetRequested = true
            }
            if let arm {
                options.settings.arm = arm
                options.armRequested = true
            }
            options.fresh = fresh
            options.restorePreset()
            return options
        }
        #expect(options(preset: "base").settings.arm == HypervisorSettings.baseArm)
        #expect(options(preset: nil).settings.arm == nil, "tiny trains no arm")
        // A base braid recorded before the arm was adopted keeps its eos-first windows.
        let layout = BraidLayout(dataRoot: DataRoot(url: root))
        try FileManager.default.createDirectory(at: layout.world.deletingLastPathComponent(), withIntermediateDirectories: true)
        try MockWorld.Record(names: ["ambient"], seed: 42, shape: MockShape(documentsPerNode: 2), preset: "base").save(layout)
        #expect(options(preset: nil).settings.preset == "base" && options(preset: nil).settings.arm == nil)
        #expect(options(preset: "base").settings.arm == nil)
        // Started over, it trains the adopted arm.
        #expect(options(preset: "base", fresh: true).settings.arm == HypervisorSettings.baseArm)
    }

    @Test("a world read from a dataset: its shards in feeding order, both sides of a link kept out of the exclusive sets, retold facts as examples")
    func datasetWorld() async throws {
        let root = temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let spec = DatasetSpec(name: "unit", seed: 5, perType: 6, paraphrase: 3, excerpt: 2, summary: 2, variant: 1, homonym: 1)
        let dataset = try BraidDataset.generate(spec)
        let directory = root.appendingPathComponent("unit")
        try BraidDataset.write(dataset, to: directory)
        let world = try MockWorld(dataset: directory)
        #expect(world.names == ["ambient", "craft", "veil"])
        #expect(world.dataset?.hash == dataset.manifest.datasetHash && world.dataset?.name == "unit")
        for name in world.names { #expect(world.documents(for: name).map(\.id) == dataset.corpora[name]?.documents.map(\.id)) }
        let linked = Set(world.crosslinks.filter { $0.kind != .homonym }.flatMap { [$0.source.documentID, $0.target.documentID] })
        for name in world.names {
            let exclusive = world.exclusive(world.documents(for: name))
            #expect(!exclusive.isEmpty && exclusive.allSatisfy { !linked.contains($0.id) })
        }
        // Two nodes of three: only the links between them.
        let pair = try MockWorld(dataset: directory, names: ["ambient", "veil"])
        #expect(pair.crosslinks.allSatisfy { Set([$0.source.node, $0.target.node]) == Set(["ambient", "veil"]) })
        #expect(throws: MockWorldError.self) { _ = try MockWorld(dataset: directory, names: ["ambient", "nowhere"]) }

        let tokenizer = try await RaoTokenizer.load()
        let nodes: [BraidExample.Node] = world.names.map { name in
            (name: name, label: name.capitalized, threadID: name, documents: world.documents(for: name))
        }
        let retold = BraidExample.crossed(nodes: nodes, links: world.crosslinks, tokenizer: tokenizer)
        #expect(!retold.isEmpty && retold.allSatisfy { $0.resolvedKind == .crossed && $0.copier != nil && $0.copier != $0.node })
        #expect(retold.allSatisfy { example in world.crosslinks.contains { $0.target.documentID == example.source?.documentID } })
        // Only both sides present make an example: without the sources, nothing.
        let targetsOnly: [BraidExample.Node] = nodes.map { node in
            (name: node.name, label: node.label, threadID: node.threadID,
             documents: node.documents.filter { document in !world.crosslinks.contains { $0.source.documentID == document.id } })
        }
        #expect(BraidExample.crossed(nodes: targetsOnly, links: world.crosslinks, tokenizer: tokenizer).isEmpty)
    }

    @Test("a braid fed from one dataset refuses another, or a generated world")
    func datasetGuard() throws {
        let root = temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        func write(_ seed: UInt64) throws -> URL {
            let directory = root.appendingPathComponent("d\(seed)")
            try BraidDataset.write(try BraidDataset.generate(DatasetSpec(name: "d\(seed)", seed: seed, perType: 3, paraphrase: 1, excerpt: 1,
                                                                         summary: 1, variant: 1, homonym: 1)), to: directory)
            return directory
        }
        let first = try write(1)
        let second = try write(2)
        func session(dataset: URL?) throws -> BraidSession {
            var options = BraidOptions(root: DataRoot(url: root.appendingPathComponent("data")), executable: URL(fileURLWithPath: "/usr/bin/true"))
            options.dataset = dataset
            return try BraidSession(options: options, vocabularySHA256: "v") { _ in }
        }
        let fed = try session(dataset: first)
        try fed.checkWorld()
        try fed.worldRecord.save(fed.layout)
        #expect(MockWorld.Record.load(fed.layout)?.dataset?.hash == fed.world.dataset?.hash)
        #expect(fed.worldRecord.summary.contains("dataset d1"))
        try session(dataset: first).checkWorld()
        #expect(throws: MockWorldError.self) { try session(dataset: second).checkWorld() }
        #expect(throws: MockWorldError.self) { try session(dataset: nil).checkWorld() }
    }
}

