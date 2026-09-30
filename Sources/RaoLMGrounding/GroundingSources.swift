//
//  GroundingSources.swift
//  RaoLMGrounding
//
//  WHAT: Which corpus partitions a generation is measured against, and the three token
//        sequences the measurement scores:
//
//          context  [eos] + D₁ + [eos] + D₂ + … + [eos] + prompt
//          bare     prompt, exactly (what CitedGenerator itself generated from)
//          sampled  the generated tokens, + eos when generation stopped on it
//
//        D is one document's selected partitions concatenated in partition order with no
//        separator — the way TokenizedCorpus packs the training stream — and documents come
//        in the order the sources list them.
//  PIN:  Pure Swift, no model: the rules are tested without MLX. A source's tokens are the
//        corpus tokenization the weights trained on (never a re-tokenized string), and a
//        partition whose text hash differs from the one the generation cited is refused.
//        Because bare = prompt exactly, the bare side reproduces CitedGenerator's p_LM:
//        Sinatra's log p_bare equals log(trace.lmProb), a built-in cross-check.
//

import Foundation
import RaoLMCore
import RaoLMProvenance
import RaoLMTraining

/// How the sources placed in front of the prompt are chosen.
public enum GroundingSourcePolicy: Sendable, Equatable, Codable, CustomStringConvertible {
    /// The partitions the generated tokens' top citations name, by Σ citation weight.
    case top
    /// The partitions of the generation's verbatim spans, in token order.
    case spans
    /// Every partition the generation's retrieval touched, most supported first.
    case all
    /// The partition the prompt was sliced from (`--sources fact`).
    case promptSource
    /// These partitions (`DOC:P,DOC:P`).
    case explicit([SourceAddress])

    public static let help = "top | spans | all | fact | DOC:P[,DOC:P]"

    public static func parse(_ text: String) throws -> GroundingSourcePolicy {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        switch trimmed.lowercased() {
        case "top": return .top
        case "spans", "span": return .spans
        case "all": return .all
        case "fact", "prompt", "prompt-source", "promptsource": return .promptSource
        default: break
        }
        var addresses: [SourceAddress] = []
        for item in trimmed.split(separator: ",") {
            let piece = item.trimmingCharacters(in: .whitespaces)
            guard let colon = piece.lastIndex(of: ":"),
                  let index = Int(piece[piece.index(after: colon)...]), index >= 0,
                  colon > piece.startIndex
            else { throw GroundingError.invalidPolicy(text) }
            addresses.append(SourceAddress(documentID: String(piece[..<colon]), partitionIndex: index, tokenOffset: 0))
        }
        guard !addresses.isEmpty else { throw GroundingError.invalidPolicy(text) }
        return .explicit(addresses)
    }

    public var description: String {
        switch self {
        case .top: return "top"
        case .spans: return "spans"
        case .all: return "all"
        case .promptSource: return "fact"
        case .explicit(let addresses):
            return addresses.map { "\($0.documentID):\($0.partitionIndex)" }.joined(separator: ",")
        }
    }

    public init(from decoder: Decoder) throws {
        self = try Self.parse(try decoder.singleValueContainer().decode(String.self))
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(description)
    }
}

/// One partition placed in front of the prompt.
public struct GroundingSource: Sendable, Equatable {
    /// Its row in the tokenization it came from (the training corpus, or a full-snapshot
    /// tokenization for a held-out document).
    public var ref: PartitionRef
    /// The corpus tokenization, exactly as training packed it.
    public var tokens: [Int]
    /// Σ Citation.weight the generated tokens gave it.
    public var citationWeight: Float
    /// Its Thread address (token offset 0).
    public var address: SourceAddress

    public init(ref: PartitionRef, tokens: [Int], citationWeight: Float, address: SourceAddress) {
        self.ref = ref
        self.tokens = tokens
        self.citationWeight = citationWeight
        self.address = address
    }

    public init(partition: TokenizedPartition, citationWeight: Float = 0, threadID: String?) {
        let ref = PartitionRef(
            row: partition.row, documentID: partition.documentID, documentName: partition.documentName,
            partitionIndex: partition.partitionIndex, partitionURL: partition.url,
            threadPartitionID: partition.threadPartitionID, textSHA256: partition.textSHA256,
            tokenCount: partition.tokens.count)
        self.init(
            ref: ref, tokens: partition.tokens.map(Int.init), citationWeight: citationWeight,
            address: ref.address(offset: 0, threadID: threadID))
    }

    /// The evaluator's prompted source (held-out partitions are not in the training corpus).
    public init(evalSource source: GroundingEvalSource, citationWeight: Float = 0, threadID: String?) {
        let ref = PartitionRef(
            row: source.row, documentID: source.documentID, documentName: source.documentName,
            partitionIndex: source.partitionIndex, partitionURL: source.url, threadPartitionID: source.threadPartitionID,
            textSHA256: source.textSHA256, tokenCount: source.tokens.count)
        self.init(ref: ref, tokens: source.tokens, citationWeight: citationWeight, address: ref.address(offset: 0, threadID: threadID))
    }

    /// Sinatra's partition id: "documentID#partitionIndex", the key TokenizedCorpus uses.
    public var id: String { GroundingSources.sourceID(documentID: ref.documentID, partitionIndex: ref.partitionIndex) }
}

public enum GroundingSources {
    /// SinatraHarness reads at most this many partitions per turn.
    public static let defaultLimit = 16

    public static func sourceID(documentID: String, partitionIndex: Int) -> String {
        "\(documentID)#\(partitionIndex)"
    }

    public static func parseSourceID(_ id: String) -> (documentID: String, partitionIndex: Int)? {
        guard let hash = id.lastIndex(of: "#"), hash > id.startIndex,
              let index = Int(id[id.index(after: hash)...])
        else { return nil }
        return (String(id[..<hash]), index)
    }

    static func sourceID(_ address: SourceAddress) -> String {
        sourceID(documentID: address.documentID, partitionIndex: address.partitionIndex)
    }

    static func generated(_ generation: CitedGeneration) -> [TokenTrace] {
        generation.traces.filter { !$0.isPrompt }
    }

    /// Σ Citation.weight per partition row over the generated tokens.
    public static func citationWeights(_ generation: CitedGeneration) -> [Int: Float] {
        var weights: [Int: Float] = [:]
        for trace in generated(generation) {
            for citation in trace.citations { weights[citation.row, default: 0] += citation.weight }
        }
        return weights
    }

    /// The same, keyed by source id (rows differ between tokenizations; addresses do not).
    public static func citationWeightsByID(_ generation: CitedGeneration) -> [String: Float] {
        var weights: [String: Float] = [:]
        for trace in generated(generation) {
            for citation in trace.citations { weights[sourceID(citation.address), default: 0] += citation.weight }
        }
        return weights
    }

    /// The sources a policy selects, as corpus partitions. Throws `noSources` when the policy
    /// selects nothing, `sourceNotInCorpus` for an address the corpus lacks, and
    /// `corpusMismatch` when a cited partition's text hash differs from the corpus's.
    public static func resolve(
        policy: GroundingSourcePolicy, generation: CitedGeneration, corpus: TokenizedCorpus, threadID: String?,
        limit: Int = defaultLimit
    ) throws -> [GroundingSource] {
        let weights = citationWeightsByID(generation)
        let refs = Dictionary(generation.partitions.map { (sourceID(documentID: $0.documentID, partitionIndex: $0.partitionIndex), $0) },
                              uniquingKeysWith: { first, _ in first })

        func source(documentID: String, partitionIndex: Int) throws -> GroundingSource {
            let id = sourceID(documentID: documentID, partitionIndex: partitionIndex)
            guard let row = corpus.row(documentID: documentID, partitionIndex: partitionIndex) else {
                throw GroundingError.sourceNotInCorpus("\(documentID) partition \(partitionIndex) is not in the run's corpus")
            }
            let partition = corpus.partitions[row]
            if let cited = refs[id], cited.textSHA256 != partition.textSHA256 {
                throw GroundingError.corpusMismatch(
                    "\(documentID) partition \(partitionIndex): the generation cited text \(cited.textSHA256.prefix(12))…, the corpus holds \(partition.textSHA256.prefix(12))…")
            }
            return GroundingSource(partition: partition, citationWeight: weights[id] ?? 0, threadID: threadID ?? generation.manifest.threadID)
        }

        func fromRows(_ rows: [Int]) throws -> [GroundingSource] {
            try rows.prefix(max(1, limit)).map { row in
                guard let ref = generation.partition(row: row) else {
                    throw GroundingError.sourceNotInCorpus("row \(row) is not in the generation's partition table")
                }
                return try source(documentID: ref.documentID, partitionIndex: ref.partitionIndex)
            }
        }

        var selected: [GroundingSource]
        switch policy {
        case .top:
            var order: [Int] = []
            var seen = Set<Int>()
            for trace in generated(generation) {
                if let top = trace.citations.first, seen.insert(top.row).inserted { order.append(top.row) }
            }
            let rowWeights = citationWeights(generation)
            let ranked = order.enumerated().sorted { a, b in
                let wa = rowWeights[a.element] ?? 0
                let wb = rowWeights[b.element] ?? 0
                return wa != wb ? wa > wb : a.offset < b.offset
            }.map(\.element)
            guard !ranked.isEmpty else { throw GroundingError.noSources("no generated token was cited") }
            selected = try fromRows(ranked)
        case .all:
            let rowWeights = citationWeights(generation)
            var retrieval: [Int: Float] = [:]
            for trace in generation.traces {
                for neighbour in trace.neighbours { retrieval[neighbour.cited.row, default: 0] += neighbour.weight }
            }
            let rows = generation.partitions.map(\.row).sorted { a, b in
                let (ca, cb) = (rowWeights[a] ?? 0, rowWeights[b] ?? 0)
                if ca != cb { return ca > cb }
                let (ra, rb) = (retrieval[a] ?? 0, retrieval[b] ?? 0)
                return ra != rb ? ra > rb : a < b
            }
            guard !rows.isEmpty else { throw GroundingError.noSources("the generation retrieved no partitions") }
            selected = try fromRows(rows)
        case .spans:
            var rows: [Int] = []
            for span in generation.spans where span.kind == .verbatim && !rows.contains(span.row) { rows.append(span.row) }
            guard !rows.isEmpty else { throw GroundingError.noSources("the generation has no verbatim spans") }
            selected = try fromRows(rows)
        case .promptSource:
            guard let address = generation.prompt.source else {
                throw GroundingError.noSources("the prompt is not a corpus slice, so it has no source partition")
            }
            selected = [try source(documentID: address.documentID, partitionIndex: address.partitionIndex)]
        case .explicit(let addresses):
            guard !addresses.isEmpty else { throw GroundingError.noSources("no partitions were named") }
            selected = try addresses.map { try source(documentID: $0.documentID, partitionIndex: $0.partitionIndex) }
        }
        var seen = Set<String>()
        selected = selected.filter { seen.insert($0.id).inserted }
        return Array(selected.prefix(max(1, limit)))
    }

    /// Sources grouped by document (first appearance), each document's partitions in order.
    static func documents(_ sources: [GroundingSource]) -> [[GroundingSource]] {
        var order: [String] = []
        var groups: [String: [GroundingSource]] = [:]
        var seen = Set<String>()
        for source in sources where seen.insert(source.id).inserted {
            if groups[source.ref.documentID] == nil { order.append(source.ref.documentID) }
            groups[source.ref.documentID, default: []].append(source)
        }
        return order.map { groups[$0, default: []].sorted { $0.ref.partitionIndex < $1.ref.partitionIndex } }
    }

    /// `[eos] + D₁ + [eos] + D₂ + … + [eos] + prompt`.
    public static func contextTokens(sources: [GroundingSource], prompt: [Int], eos: Int) -> [Int] {
        var tokens: [Int] = []
        tokens.reserveCapacity(contextLength(sources: sources, promptCount: prompt.count))
        for document in documents(sources) {
            tokens.append(eos)
            for source in document { tokens.append(contentsOf: source.tokens) }
        }
        tokens.append(eos)
        tokens.append(contentsOf: prompt)
        return tokens
    }

    /// `contextTokens(…).count` without building it.
    public static func contextLength(sources: [GroundingSource], promptCount: Int) -> Int {
        let documents = documents(sources)
        return documents.reduce(0) { $0 + 1 + $1.reduce(0) { $0 + $1.tokens.count } } + 1 + promptCount
    }

    /// The generated tokens, plus EOS when generation stopped on it and `includeEOS` is set:
    /// the decision to stop is a real, context-driven event.
    public static func sampledTokens(_ generation: CitedGeneration, eos: Int, includeEOS: Bool = true) -> [Int] {
        generation.tokens + (includeEOS && generation.stoppedOnEOS ? [eos] : [])
    }

    /// Drop the least-cited sources (the last listed among equals) until the context fits
    /// `budget` tokens. Always keeps one: the caller refuses a context that still does not fit.
    public static func fit(_ sources: [GroundingSource], promptCount: Int, budget: Int) -> [GroundingSource] {
        var kept = sources
        while kept.count > 1, contextLength(sources: kept, promptCount: promptCount) > budget {
            var victim = kept.count - 1
            for i in kept.indices.reversed() where kept[i].citationWeight < kept[victim].citationWeight { victim = i }
            kept.remove(at: victim)
        }
        return kept
    }
}
