//
//  CitationModels.swift
//  RaoLMCore
//
//  WHAT: The JSON a cited generation is saved as: per-token traces (entropies, the
//        retrieved corpus neighbours, grouped citations), verbatim spans with their
//        Thread addresses, and the manifest hashes that name the exact weights, index,
//        corpus and tokenizer that produced them.
//  PIN:  A citation's address is the *value* position of a retrieved corpus entry — the
//        corpus token that followed the retrieved context — because that is the token the
//        model emits. The key position is kept alongside for inspection.
//

import Foundation

extension Date {
    /// Now, truncated to the second: ISO-8601 JSON keeps whole seconds only, so records built
    /// with this round-trip exactly.
    public static func wholeSecond() -> Date {
        Date(timeIntervalSince1970: Date().timeIntervalSince1970.rounded(.down))
    }
}

/// A token position in the corpus: a row of the partition table and a token offset
/// inside that partition's own tokenization.
public struct TokenPosition: Codable, Sendable, Hashable, Comparable {
    public var row: Int
    public var offset: Int

    public init(row: Int, offset: Int) {
        self.row = row
        self.offset = offset
    }

    public static func < (lhs: TokenPosition, rhs: TokenPosition) -> Bool {
        (lhs.row, lhs.offset) < (rhs.row, rhs.offset)
    }
}

/// One row of the partition table a provenance index was built over.
public struct PartitionRef: Codable, Sendable, Equatable {
    public var row: Int
    public var documentID: String
    public var documentName: String
    public var partitionIndex: Int
    public var partitionURL: String?
    public var threadPartitionID: String?
    public var textSHA256: String
    public var tokenCount: Int
    public var memorisedAtEpoch: Int?

    public init(
        row: Int, documentID: String, documentName: String, partitionIndex: Int, partitionURL: String?,
        threadPartitionID: String?, textSHA256: String, tokenCount: Int, memorisedAtEpoch: Int? = nil
    ) {
        self.row = row
        self.documentID = documentID
        self.documentName = documentName
        self.partitionIndex = partitionIndex
        self.partitionURL = partitionURL
        self.threadPartitionID = threadPartitionID
        self.textSHA256 = textSHA256
        self.tokenCount = tokenCount
        self.memorisedAtEpoch = memorisedAtEpoch
    }

    public func address(offset: Int, threadID: String?) -> SourceAddress {
        SourceAddress(
            threadID: threadID, documentID: documentID, partitionIndex: partitionIndex, tokenOffset: offset,
            partitionURL: partitionURL, threadPartitionID: threadPartitionID)
    }
}

/// The hashes a generation is bound to.
public struct ManifestRef: Codable, Sendable, Equatable {
    public var runID: String
    public var epoch: Int
    public var checkpointSHA256: String
    public var indexSHA256: String
    public var corpusHash: String
    public var tokenizerSHA256: String
    public var ledgerSHA256: String?
    public var threadID: String?

    public init(
        runID: String, epoch: Int, checkpointSHA256: String, indexSHA256: String, corpusHash: String,
        tokenizerSHA256: String, ledgerSHA256: String?, threadID: String?
    ) {
        self.runID = runID
        self.epoch = epoch
        self.checkpointSHA256 = checkpointSHA256
        self.indexSHA256 = indexSHA256
        self.corpusHash = corpusHash
        self.tokenizerSHA256 = tokenizerSHA256
        self.ledgerSHA256 = ledgerSHA256
        self.threadID = threadID
    }
}

/// One retrieved corpus position for one generated token.
public struct Neighbour: Codable, Sendable, Equatable {
    /// 1-based rank by cosine score.
    public var rank: Int
    /// Entry in the provenance index.
    public var entry: Int
    /// Cosine similarity between the query key and this entry's key.
    public var score: Float
    /// softmax(score / τ) over the k neighbours.
    public var weight: Float
    /// The corpus token that followed this context.
    public var value: Int
    /// Whether `value` is the token that was emitted.
    public var matches: Bool
    /// The retrieved context position.
    public var key: TokenPosition
    /// The citation address: where `value` sits in the corpus.
    public var cited: TokenPosition
    /// Loss and entropy of this transition in the indexed epoch's eval pass.
    public var sourceLoss: Float
    public var sourceEntropy: Float

    public init(
        rank: Int, entry: Int, score: Float, weight: Float, value: Int, matches: Bool, key: TokenPosition,
        cited: TokenPosition, sourceLoss: Float, sourceEntropy: Float
    ) {
        self.rank = rank
        self.entry = entry
        self.score = score
        self.weight = weight
        self.value = value
        self.matches = matches
        self.key = key
        self.cited = cited
        self.sourceLoss = sourceLoss
        self.sourceEntropy = sourceEntropy
    }
}

/// Retrieval support for one emitted token, grouped by partition.
public struct Citation: Codable, Sendable, Equatable {
    public var row: Int
    public var address: SourceAddress
    /// Σ weight of the matching neighbours in this partition.
    public var weight: Float
    public var bestRank: Int
    public var bestScore: Float
    public var sourceLoss: Float
    public var memorisedAtEpoch: Int?
    /// (simTerm(bestScore) · weight · exp(−sourceLoss))^(1/3), in [0, 1].
    public var confidence: Float

    public init(
        row: Int, address: SourceAddress, weight: Float, bestRank: Int, bestScore: Float,
        sourceLoss: Float, memorisedAtEpoch: Int?, confidence: Float
    ) {
        self.row = row
        self.address = address
        self.weight = weight
        self.bestRank = bestRank
        self.bestScore = bestScore
        self.sourceLoss = sourceLoss
        self.memorisedAtEpoch = memorisedAtEpoch
        self.confidence = confidence
    }
}

/// Everything recorded about the prediction of one token.
public struct TokenTrace: Codable, Sendable, Equatable {
    /// Position in prompt ++ generated. Trace `j` predicts token `j` from tokens `0..<j`.
    public var index: Int
    public var token: Int
    public var text: String
    public var isPrompt: Bool
    /// Entropy of the model's own distribution (full vocabulary, T = 1), in nats.
    public var lmEntropy: Float
    /// Entropy of the retrieval distribution over the neighbours' next tokens (≤ ln k).
    public var knnEntropy: Float
    /// Entropy of the mixture actually sampled from.
    public var mixedEntropy: Float
    /// Entropy of retrieval weight aggregated by partition: high means generic phrasing.
    public var sourceEntropy: Float
    public var lmProb: Float
    /// p_knn(token): the share of retrieval weight whose corpus continuation is this token.
    public var agreement: Float
    public var mixedProb: Float
    public var lambda: Float
    public var neighbours: [Neighbour]
    public var citations: [Citation]
    public var uncited: Bool
    public var confidence: Float?
    public var spanIndex: Int?

    public init(
        index: Int, token: Int, text: String, isPrompt: Bool, lmEntropy: Float, knnEntropy: Float,
        mixedEntropy: Float, sourceEntropy: Float, lmProb: Float, agreement: Float, mixedProb: Float,
        lambda: Float, neighbours: [Neighbour]
    ) {
        self.index = index
        self.token = token
        self.text = text
        self.isPrompt = isPrompt
        self.lmEntropy = lmEntropy
        self.knnEntropy = knnEntropy
        self.mixedEntropy = mixedEntropy
        self.sourceEntropy = sourceEntropy
        self.lmProb = lmProb
        self.agreement = agreement
        self.mixedProb = mixedProb
        self.lambda = lambda
        self.neighbours = neighbours
        self.citations = []
        self.uncited = true
        self.confidence = nil
        self.spanIndex = nil
    }
}

public struct SpanAlternative: Codable, Sendable, Equatable {
    public var row: Int
    public var source: SourceAddress
    public var length: Int
    public var cumulativeWeight: Float
}

public enum VerificationStatus: String, Codable, Sendable {
    /// The live partition's tokens at the recorded offset equal the span's tokens.
    case verified
    /// The live partition's text differs from the text the index was built on.
    case stale
    /// Same text, but the tokens at the offset differ (a pipeline bug).
    case mismatch
    /// Only the decoded text matches: the tokenizer changed.
    case tokenizerDrift
    /// The document or partition is gone from the Thread.
    case missing
    case unverified
}

public struct SpanVerification: Codable, Sendable, Equatable {
    public var status: VerificationStatus
    public var checkedAt: Date
    public var liveTextSHA256: String?
    public var detail: String?

    public init(status: VerificationStatus, checkedAt: Date = .wholeSecond(), liveTextSHA256: String? = nil, detail: String? = nil) {
        self.status = status
        self.checkedAt = checkedAt
        self.liveTextSHA256 = liveTextSHA256
        self.detail = detail
    }
}

public struct TokenRange: Codable, Sendable, Equatable {
    public var start: Int
    public var end: Int

    public init(start: Int, end: Int) {
        self.start = start
        self.end = end
    }

    public var count: Int { end - start }
}

/// A run of emitted tokens that reproduces a corpus partition verbatim, token by token,
/// with every token retrieved (rank ≤ 3) from the next position of the same partition.
public struct CitedSpan: Codable, Sendable, Equatable {
    public enum Kind: String, Codable, Sendable {
        /// Contains at least one generated token.
        case verbatim
        /// Lies entirely inside the prompt; shown, but not counted as proof.
        case promptVerbatim
    }

    public var kind: Kind
    /// Positions in prompt ++ generated, half-open.
    public var tokenRange: TokenRange
    public var promptTokens: Int
    public var tokens: [Int]
    public var text: String
    public var row: Int
    public var source: SourceAddress
    public var documentName: String
    public var textSHA256: String
    public var ranks: [Int]
    public var weights: [Float]
    /// Weakest-link confidence over the span's tokens.
    public var confidence: Float
    public var meanConfidence: Float
    /// 1 − (token 3-grams shared by ≥ 2 documents) / (token 3-grams). 0 means boilerplate.
    public var distinctiveness: Float
    public var alternatives: [SpanAlternative]
    public var verification: SpanVerification?
}

public struct GenerationParameters: Codable, Sendable, Equatable {
    public var lambda: Float
    public var tau: Float
    public var k: Int
    public var rankThreshold: Int
    public var minSpanLength: Int
    public var temperature: Float
    public var topK: Int
    public var seed: UInt64
    public var maxTokens: Int
    public var tapLayer: Int
    public var alpha: Float

    public init(
        lambda: Float = 0.5, tau: Float = 0.05, k: Int = 16, rankThreshold: Int = 3, minSpanLength: Int = 3,
        temperature: Float = 0, topK: Int = 0, seed: UInt64 = 42, maxTokens: Int = 48, tapLayer: Int, alpha: Float
    ) {
        self.lambda = lambda
        self.tau = tau
        self.k = k
        self.rankThreshold = rankThreshold
        self.minSpanLength = minSpanLength
        self.temperature = temperature
        self.topK = topK
        self.seed = seed
        self.maxTokens = maxTokens
        self.tapLayer = tapLayer
        self.alpha = alpha
    }
}

public struct GenerationPrompt: Codable, Sendable, Equatable {
    public var text: String
    public var tokens: [Int]
    /// Set when the prompt is a token slice of a corpus partition.
    public var source: SourceAddress?

    public init(text: String, tokens: [Int], source: SourceAddress?) {
        self.text = text
        self.tokens = tokens
        self.source = source
    }
}

public struct GenerationSummary: Codable, Sendable, Equatable {
    public var generated: Int
    public var verbatimCovered: Int
    public var supportOnly: Int
    public var uncited: Int
    public var meanConfidence: Float?
    public var verbatimSpans: Int
    public var verifiedSpans: Int?
}

public struct CitedGeneration: Codable, Sendable, Equatable {
    public var version: Int
    public var generationID: String
    public var manifest: ManifestRef
    public var createdAt: Date
    public var prompt: GenerationPrompt
    public var params: GenerationParameters
    /// Generated tokens only (the prompt is in `prompt`).
    public var tokens: [Int]
    public var text: String
    public var stoppedOnEOS: Bool
    /// The partition-table rows the traces refer to.
    public var partitions: [PartitionRef]
    public var traces: [TokenTrace]
    public var spans: [CitedSpan]
    public var summary: GenerationSummary

    public init(
        generationID: String, manifest: ManifestRef, createdAt: Date = .wholeSecond(), prompt: GenerationPrompt,
        params: GenerationParameters, tokens: [Int], text: String, stoppedOnEOS: Bool,
        partitions: [PartitionRef], traces: [TokenTrace], spans: [CitedSpan], summary: GenerationSummary
    ) {
        self.version = 1
        self.generationID = generationID
        self.manifest = manifest
        self.createdAt = createdAt
        self.prompt = prompt
        self.params = params
        self.tokens = tokens
        self.text = text
        self.stoppedOnEOS = stoppedOnEOS
        self.partitions = partitions
        self.traces = traces
        self.spans = spans
        self.summary = summary
    }

    public func partition(row: Int) -> PartitionRef? {
        partitions.first { $0.row == row }
    }

    public func save(to url: URL) throws {
        try JSONCoding.write(self, to: url)
    }

    public static func load(from url: URL) throws -> CitedGeneration {
        try JSONCoding.read(CitedGeneration.self, from: url)
    }
}
