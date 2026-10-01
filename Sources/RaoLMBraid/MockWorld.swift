//
//  MockWorld.swift
//  RaoLMBraid
//
//  WHAT: The demo data a braid is fed with: one synthetic world dealt into a shard per node,
//        the feeder that deposits a node's next documents into its Thread (and withdraws its
//        newest one), and the example prompts: a fact from each Thread in turn, a prompt about
//        a subject no Thread holds, a prompt that moves from one Thread's fact to another's,
//        and text that has nothing to do with any of it.
//  PIN:  One world, so every name and every fact lives on exactly one Thread: a correct answer
//        can only come from the Thread that holds it, which is what the panel shows. A dataset
//        world also retells some entities on another Thread in its own words; the exclusive
//        example sets never draw on either side of such a link. The feeder writes facts.jsonl
//        beside the node for probes and prompts only; the node trains on what its Thread exports.
//

import Foundation
import RaoLMCore
import RaoLMModel
import RaoLMTraining


/// The shape of a mock world: how many documents each node holds. Older world.json files carry
/// more keys; they decode to this.
public struct MockShape: Codable, Sendable, Equatable {
    public var documentsPerNode: Int

    public init(documentsPerNode: Int = 40) {
        self.documentsPerNode = documentsPerNode
    }
}

public enum MockWorldError: Error, CustomStringConvertible, Equatable {
    case dataset(String)
    case different(String)

    public var description: String {
        switch self {
        case .dataset(let message): return message
        case .different(let message): return message
        }
    }
}

/// A braid dataset a world was read from (BraidDataset), named by its hash.
public struct MockDatasetSource: Codable, Sendable, Equatable {
    public var path: String
    public var name: String
    public var hash: String

    public init(path: String, name: String, hash: String) {
        self.path = path
        self.name = name
        self.hash = hash
    }
}

public final class MockWorld: @unchecked Sendable {
    public let seed: UInt64
    public let names: [String]
    public let shape: MockShape
    public let shards: [String: GeneratedCorpus]
    /// The dataset the shards were read from; nil for a world generated here.
    public let dataset: MockDatasetSource?
    /// A dataset's links between the nodes of this world.
    public let crosslinks: [DatasetCrosslink]
    /// Documents on either side of a link whose facts agree or nearly do: held by two Threads.
    private let crossedIDs: Set<String>

    public init(names: [String], seed: UInt64 = 42, shape: MockShape) throws {
        self.seed = seed
        self.names = names
        self.shape = shape
        dataset = nil
        crosslinks = []
        crossedIDs = []
        let corpora = try SyntheticCorpus.generateShards(slugs: names, seed: seed, documentsPerShard: shape.documentsPerNode)
        shards = Dictionary(uniqueKeysWithValues: zip(names, corpora))
    }

    public convenience init(names: [String], seed: UInt64 = 42, documentsPerNode: Int = 40) throws {
        try self.init(names: names, seed: seed, shape: MockShape(documentsPerNode: documentsPerNode))
    }

    /// A world read from a braid dataset: each node's shard is the dataset's corpus of that name,
    /// in its feeding order. `names` picks nodes of the dataset (all of them by default).
    public init(dataset directory: URL, names wanted: [String]? = nil) throws {
        let loaded = try BraidDataset.load(directory)
        let names = wanted ?? loaded.names
        for name in names where loaded.corpora[name] == nil {
            throw MockWorldError.dataset("the dataset \(loaded.manifest.spec.name) has no node '\(name)' (it has \(loaded.names.joined(separator: ", ")))")
        }
        self.names = names
        seed = loaded.manifest.spec.seed
        shape = MockShape(documentsPerNode: loaded.manifest.spec.perType * 3)
        shards = Dictionary(uniqueKeysWithValues: names.compactMap { name in loaded.corpora[name].map { (name, $0) } })
        dataset = MockDatasetSource(path: directory.standardizedFileURL.path, name: loaded.manifest.spec.name, hash: loaded.manifest.datasetHash)
        crosslinks = loaded.crosslinks.filter { names.contains($0.source.node) && names.contains($0.target.node) }
        crossedIDs = Set(crosslinks.filter { $0.kind != .homonym }.flatMap { [$0.source.documentID, $0.target.documentID] })
    }

    public func documents(for node: String) -> [CorpusDocument] { shards[node]?.documents ?? [] }

    public func document(id: String) -> CorpusDocument? {
        for shard in shards.values { if let document = shard.documents.first(where: { $0.id == id }) { return document } }
        return nil
    }

    /// `documents` without either side of a dataset link that retells the same facts (an entity
    /// two Threads hold, each in its own words): what the exclusive example sets draw on.
    public func exclusive(_ documents: [CorpusDocument]) -> [CorpusDocument] {
        guard !crossedIDs.isEmpty else { return documents }
        return documents.filter { !crossedIDs.contains($0.id) }
    }

    /// The record this world writes to world.json.
    public var record: Record { Record(names: names, seed: seed, shape: shape, dataset: dataset) }

    /// world.json: what the braid's nodes were fed from, so a session and the benches rebuild the
    /// same world, and a different one is refused rather than silently mismatched.
    public struct Record: Codable, Sendable, Equatable {
        public var names: [String]
        public var seed: UInt64
        public var shape: MockShape
        /// The dataset the nodes were fed from; nil for a world generated from the seed and shape.
        public var dataset: MockDatasetSource?
        /// The model preset the nodes train; nil for `tiny`, the preset before presets were kept.
        public var preset: String?
        /// The training arm the nodes train on (`HypervisorSettings.arm`); nil for the reference recipe.
        public var arm: String?

        public init(names: [String], seed: UInt64, shape: MockShape, dataset: MockDatasetSource? = nil, preset: String? = nil) {
            self.names = names
            self.seed = seed
            self.shape = shape
            self.dataset = dataset
            self.preset = preset
        }

        public static func load(_ layout: BraidLayout) -> Record? { try? JSONCoding.read(Record.self, from: layout.world) }

        public func save(_ layout: BraidLayout) throws { try JSONCoding.write(self, to: layout.world) }

        /// Whether nodes fed from `self` hold what `other` would feed them.
        public func sameWorld(as other: Record) -> Bool {
            names == other.names && seed == other.seed && shape == other.shape && dataset?.hash == other.dataset?.hash
                && (preset ?? "tiny") == (other.preset ?? "tiny") && arm == other.arm
        }

        public var summary: String {
            let model = (preset.map { " · preset \($0)" } ?? "") + (arm.map { " · arm \($0)" } ?? "")
            if let dataset { return "nodes \(names.joined(separator: ",")) · dataset \(dataset.name) (\(dataset.hash.prefix(12)))" + model }
            return "nodes \(names.joined(separator: ",")) · seed \(seed) · \(shape.documentsPerNode) documents per node" + model
        }
    }
}

public enum MockFeeder {
    /// Deposits the node's next `count` documents into its corpus and returns them.
    @discardableResult
    public static func feed(
        node: String, count: Int, world: MockWorld, layout: NodeLayout, source: CorpusSource
    ) async throws -> [CorpusDocument] {
        var feed = FeedState.load(layout)
        let deposited = Set(feed.deposited)
        let next = Array(world.documents(for: node).filter { !deposited.contains($0.id) }.prefix(max(0, count)))
        guard !next.isEmpty else { return [] }
        try await source.deposit(next)
        feed.deposited += next.map(\.id)
        try save(feed, world: world, layout: layout, node: node)
        return next
    }

    /// Withdraws the newest document still present and returns it.
    @discardableResult
    public static func withdraw(node: String, world: MockWorld, layout: NodeLayout, source: CorpusSource) async throws -> CorpusDocument? {
        var feed = FeedState.load(layout)
        guard let id = feed.present.last else { return nil }
        try await source.withdraw([id])
        feed.withdrawn.append(id)
        try save(feed, world: world, layout: layout, node: node)
        return world.document(id: id)
    }

    /// Documents of a node's world held out of its feed: the last this many it has not been fed.
    public static let heldOut = 8

    static func save(_ feed: FeedState, world: MockWorld, layout: NodeLayout, node: String) throws {
        try JSONCoding.write(feed, to: layout.feed)
        let facts = feed.present.compactMap(world.document(id:)).flatMap(\.facts)
        let writer = try JSONLWriter(url: layout.facts, truncate: true)
        for fact in facts { try writer.append(fact) }
        writer.close()
        // Text in the node's own voice it has never seen, for its held-out loss.
        let deposited = Set(feed.deposited)
        let unfed = world.documents(for: node).filter { !deposited.contains($0.id) }.suffix(heldOut)
        let held = try JSONLWriter(url: layout.heldOut, truncate: true)
        for document in unfed { try held.append(document) }
        held.close()
    }

    /// The documents the feeder has deposited into a node and not withdrawn, in deposit order.
    public static func present(node: String, world: MockWorld, layout: NodeLayout) -> [CorpusDocument] {
        FeedState.load(layout).present.compactMap(world.document(id:))
    }
}

/// A prompt the braid panel offers.
public struct BraidExample: Codable, Sendable, Equatable {
    public enum Kind: String, Codable, Sendable {
        /// A fact one Thread holds, as the corpus's own tokens before the answer.
        case fact
        /// A complete fact sentence from one Thread, then a fact from another: the lead has to move.
        case pair
        /// The same template about a subject no Thread holds.
        case unknown
        /// Text that has nothing to do with what any Thread holds.
        case generic
        /// A fact another Thread states in its own words (a dataset link): `node` is the Thread
        /// asked, in its wording; `copier` is the Thread the entity came from.
        case crossed
    }

    public var label: String
    /// The Thread holding the answer; nil for a prompt no Thread can answer.
    public var node: String?
    public var promptTokens: [Int]
    public var promptText: String
    public var expected: String?
    public var source: SourceAddress?
    /// Nil in examples recorded before kinds: a fact when `node` is set, else an unknown subject.
    public var kind: Kind?
    /// For a pair: the Thread whose sentence opens the prompt, and the answer that sentence gives.
    public var opener: String?
    public var openerAnswer: String?
    /// For a crossed fact: the Thread the entity came from.
    public var copier: String?

    public init(
        label: String, node: String?, promptTokens: [Int], promptText: String, expected: String?, source: SourceAddress?,
        kind: Kind? = nil, opener: String? = nil, openerAnswer: String? = nil, copier: String? = nil
    ) {
        self.label = label
        self.node = node
        self.promptTokens = promptTokens
        self.promptText = promptText
        self.expected = expected
        self.source = source
        self.kind = kind
        self.opener = opener
        self.openerAnswer = openerAnswer
        self.copier = copier
    }

    public var resolvedKind: Kind { kind ?? (node == nil ? .unknown : .fact) }

    public typealias Node = (name: String, label: String, threadID: String?, documents: [CorpusDocument])

    /// Sentences about nothing the mock world holds.
    public static let genericPrompts = [
        "How should we understand our lives",
        "How should we understand our lives as travellers",
        "What is the best way to learn a new language",
        "The weather tomorrow is going to be",
        "In the future, computers will be able to",
        "My favourite thing about the weekend is",
        "To make a good cup of tea you should",
        "The most important lesson I ever learned was",
    ]

    /// One node's facts located in its own tokens, newest documents first, one fact from each in turn.
    struct Located {
        var node: Node
        var corpus: TokenizedCorpus
        var facts: [LocatedFact]
    }

    static func locate(_ nodes: [Node], tokenizer: RaoTokenizer) -> [Located] {
        nodes.filter { !$0.documents.isEmpty }.map { node in
            let corpus = GeneratedCorpus(
                manifest: CorpusManifest(slug: node.name, generator: SyntheticCorpus.generatorName, generatorVersion: 1, seed: 0,
                                         documentCount: node.documents.count, partitionCount: 0, factCount: 0, chunkMaxChars: 600,
                                         chunkMinChars: 120, documentIDs: node.documents.map(\.id), corpusHash: ""),
                documents: node.documents)
            let tokenized = TokenizedCorpus(snapshot: .offline(corpus), tokenizer: tokenizer)
            let located = FactLocator.locate(corpus.facts, corpus: tokenized, tokenizer: tokenizer).located
            // Newest documents first (they are what just changed), one fact from each in turn.
            let order = Dictionary(node.documents.enumerated().map { ($0.element.id, $0.offset) }, uniquingKeysWith: { a, _ in a })
            var byDocument = Dictionary(grouping: located, by: \.fact.documentID)
                .sorted { (order[$0.key] ?? 0) > (order[$1.key] ?? 0) }
                .map { $0.value.sorted { $0.fact.id < $1.fact.id } }
            var ranked: [LocatedFact] = []
            while byDocument.contains(where: { !$0.isEmpty }) {
                for i in byDocument.indices where !byDocument[i].isEmpty { ranked.append(byDocument[i].removeFirst()) }
            }
            return Located(node: node, corpus: tokenized, facts: ranked)
        }
    }

    static func fact(_ fact: LocatedFact, of located: Located, tokenizer: RaoTokenizer) -> BraidExample {
        let partition = located.corpus.partitions[fact.row]
        let prompt = partition.tokens[fact.contextToken..<fact.answerToken].map(Int.init)
        return BraidExample(
            label: "\(located.node.label) · \(fact.fact.subject)", node: located.node.name, promptTokens: prompt,
            promptText: tokenizer.decode(prompt), expected: fact.fact.answer,
            source: SourceAddress(threadID: located.node.threadID, documentID: partition.documentID, partitionIndex: partition.partitionIndex,
                                  tokenOffset: fact.contextToken, partitionURL: partition.url),
            kind: .fact)
    }

    /// The opener's whole sentence (with the sentence before it), then the other fact up to its answer.
    static func pair(
        opener: LocatedFact, of first: Located, then fact: LocatedFact, of second: Located, tokenizer: RaoTokenizer
    ) -> BraidExample? {
        func text(_ located: Located, _ fact: LocatedFact, through end: Int) -> String? {
            guard let document = located.node.documents.first(where: { $0.id == fact.fact.documentID }),
                  let partition = document.partitions.first(where: { $0.index == fact.fact.partitionIndex }) else { return nil }
            let bytes = Array(partition.text.utf8)
            guard fact.fact.contextStart >= 0, end <= bytes.count, fact.fact.contextStart < end else { return nil }
            return String(decoding: bytes[fact.fact.contextStart..<end], as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        let openerEnd = opener.fact.sentenceStart + opener.fact.sentence.utf8.count
        guard let opening = text(first, opener, through: openerEnd), let asking = text(second, fact, through: fact.fact.answerStart)
        else { return nil }
        let prompt = opening + " " + asking
        return BraidExample(
            label: "\(first.node.label) → \(second.node.label) · \(fact.fact.subject)", node: second.node.name,
            promptTokens: tokenizer.encode(prompt), promptText: prompt, expected: fact.fact.answer, source: nil, kind: .pair,
            opener: first.node.name, openerAnswer: opener.fact.answer)
    }

    static func unknown(_ fact: LocatedFact, tokenizer: RaoTokenizer) -> BraidExample {
        BraidExample(
            label: "nowhere · \(fact.fact.negativePrompt.prefix(28))…", node: nil, promptTokens: tokenizer.encode(fact.fact.negativePrompt),
            promptText: fact.fact.negativePrompt, expected: nil, source: nil, kind: .unknown)
    }

    /// A fact from each node in turn, `perNode` from each.
    public static func facts(nodes: [Node], tokenizer: RaoTokenizer, perNode: Int = 12) -> [BraidExample] {
        let located = locate(nodes, tokenizer: tokenizer)
        let lists = located.map { node in node.facts.prefix(perNode).map { fact($0, of: node, tokenizer: tokenizer) } }
        var result: [BraidExample] = []
        var round = 0
        while lists.contains(where: { round < $0.count }) {
            for list in lists where round < list.count { result.append(list[round]) }
            round += 1
        }
        return result
    }

    /// Two-fact prompts: for every ordered pair of nodes, the opener's r-th fact then the other's
    /// (r+1)-th, so neither half is a fact prompt a bench also asks on its own.
    public static func pairs(nodes: [Node], tokenizer: RaoTokenizer, perPair: Int = 10) -> [BraidExample] {
        let located = locate(nodes, tokenizer: tokenizer)
        var result: [BraidExample] = []
        for round in 0..<max(0, perPair) {
            for (i, first) in located.enumerated() {
                for (j, second) in located.enumerated() where i != j {
                    guard round < first.facts.count, !second.facts.isEmpty else { continue }
                    let asked = second.facts[(round + 1) % second.facts.count]
                    if let example = pair(opener: first.facts[round], of: first, then: asked, of: second, tokenizer: tokenizer) {
                        result.append(example)
                    }
                }
            }
        }
        return result
    }

    /// The same templates about subjects no Thread holds, a fact from each node in turn.
    public static func unknowns(nodes: [Node], tokenizer: RaoTokenizer, count: Int = 10) -> [BraidExample] {
        let located = locate(nodes, tokenizer: tokenizer)
        var result: [BraidExample] = []
        var round = 0
        while result.count < count, located.contains(where: { round < $0.facts.count }) {
            for node in located where round < node.facts.count && result.count < count {
                result.append(unknown(node.facts[round], tokenizer: tokenizer))
            }
            round += 1
        }
        return result
    }

    /// Each fact asked in words the corpus does not contain, a fact from each node in turn.
    public static func paraphrases(nodes: [Node], tokenizer: RaoTokenizer, count: Int = 10) -> [BraidExample] {
        let located = locate(nodes, tokenizer: tokenizer)
        var result: [BraidExample] = []
        var round = 0
        while result.count < count, located.contains(where: { round < $0.facts.count }) {
            for node in located where round < node.facts.count && result.count < count {
                let fact = node.facts[round].fact
                guard let prompt = fact.paraphrases.first else { continue }
                result.append(BraidExample(
                    label: "\(node.node.label) · \(fact.subject) (reworded)", node: node.node.name, promptTokens: tokenizer.encode(prompt),
                    promptText: prompt, expected: fact.answer, source: nil, kind: .fact))
            }
            round += 1
        }
        return result
    }

    public static func generic(tokenizer: RaoTokenizer) -> [BraidExample] {
        genericPrompts.map { prompt in
            BraidExample(label: "generic · \(prompt.prefix(28))…", node: nil, promptTokens: tokenizer.encode(prompt), promptText: prompt,
                         expected: nil, source: nil, kind: .generic)
        }
    }

    /// Facts a Thread states in its own words about an entity from another Thread (a dataset
    /// link that agrees): asked in the target's wording, located in the target's tokens.
    public static func crossed(nodes: [Node], links: [DatasetCrosslink], tokenizer: RaoTokenizer, perLink: Int = 1) -> [BraidExample] {
        var result: [BraidExample] = []
        let labels = Dictionary(nodes.map { ($0.name, $0.label) }, uniquingKeysWith: { a, _ in a })
        for link in links where link.kind != .homonym && link.kind != .variant {
            guard let target = nodes.first(where: { $0.name == link.target.node }), labels[link.source.node] != nil,
                  nodes.first(where: { $0.name == link.source.node })?.documents.contains(where: { $0.id == link.source.documentID }) == true,
                  let document = target.documents.first(where: { $0.id == link.target.documentID }) else { continue }
            let only: Node = (name: target.name, label: target.label, threadID: target.threadID, documents: [document])
            guard let located = locate([only], tokenizer: tokenizer).first else { continue }
            let agreeing = Set(link.facts.filter(\.agrees).map(\.targetFact))
            for crossedFact in located.facts.filter({ agreeing.contains($0.fact.id) }).prefix(max(0, perLink)) {
                var example = fact(crossedFact, of: located, tokenizer: tokenizer)
                example.kind = .crossed
                example.copier = link.source.node
                example.label = "\(target.label) ≈ \(labels[link.source.node] ?? link.source.node) · \(crossedFact.fact.subject)"
                result.append(example)
            }
        }
        return result
    }

    /// What the panel cycles through: a fact from each node in turn and, every third round, a
    /// subject no Thread holds, a prompt that moves from one Thread's fact to another's, and one
    /// another Thread tells in its own words. `nodes` carry their exclusive documents only;
    /// `presentNodes` every document present.
    public static func build(
        nodes: [Node], presentNodes: [Node]? = nil, links: [DatasetCrosslink] = [], tokenizer: RaoTokenizer, perNode: Int = 12
    ) -> [BraidExample] {
        let located = locate(nodes, tokenizer: tokenizer)
        let lists = located.map { node in node.facts.prefix(perNode).map { fact($0, of: node, tokenizer: tokenizer) } }
        let negatives = located.compactMap { node in node.facts.first.map { unknown($0, tokenizer: tokenizer) } }
        let moving = pairs(nodes: nodes, tokenizer: tokenizer, perPair: max(1, perNode / 3))
        let retold = crossed(nodes: presentNodes ?? nodes, links: links, tokenizer: tokenizer)
        var result: [BraidExample] = []
        var round = 0
        while lists.contains(where: { round < $0.count }) {
            for list in lists where round < list.count { result.append(list[round]) }
            if round % 3 == 2 {
                if !negatives.isEmpty { result.append(negatives[(round / 3) % negatives.count]) }
                if !moving.isEmpty { result.append(moving[(round / 3) % moving.count]) }
                if !retold.isEmpty { result.append(retold[(round / 3) % retold.count]) }
            }
            round += 1
        }
        if result.isEmpty { result = negatives }
        return result
    }
}
