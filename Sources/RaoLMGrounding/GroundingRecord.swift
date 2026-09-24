//
//  GroundingRecord.swift
//  RaoLMGrounding
//
//  WHAT: The two-fold record of one generation, saved as
//        <run>/generations/<generationID>.grounding.json:
//
//          RaoLM      what retrieval says: per-token citation confidence, kNN agreement, the
//                     four entropies (H_lm, H_knn, H_mix, H_source), verbatim spans
//          Sinatra    what the source's presence did: the output scored with the sources in
//                     front of the prompt and without them — ι, KL, drift, class, risk — and
//                     each source's attribution A_p
//
//        joined token by token (step j ↔ the j-th generated trace) and partition by partition
//        (source id "documentID#partitionIndex"), with the join's statistics.
//  PIN:  Plain Codable values: the only thing the MLX worker hands to the UI. The Sinatra
//        measurement is kept verbatim beside the joined rows. A trailing EOS step has no
//        RaoLM trace, so its RaoLM columns are nil. A skipped measurement keeps its reason
//        and has no token rows.
//

import Foundation
import RaoLMCore
import SinatraHarness

/// A source as the record names it.
public struct GroundingSourceRef: Codable, Sendable, Equatable {
    public var id: String
    public var row: Int
    public var documentID: String
    public var documentName: String
    public var partitionIndex: Int
    public var partitionURL: String?
    public var threadPartitionID: String?
    public var textSHA256: String
    public var tokenCount: Int
    public var citationWeight: Float

    public init(_ source: GroundingSource) {
        self.id = source.id
        self.row = source.ref.row
        self.documentID = source.ref.documentID
        self.documentName = source.ref.documentName
        self.partitionIndex = source.ref.partitionIndex
        self.partitionURL = source.ref.partitionURL
        self.threadPartitionID = source.ref.threadPartitionID
        self.textSHA256 = source.ref.textSHA256
        self.tokenCount = source.tokens.count
        self.citationWeight = source.citationWeight
    }
}

/// One scored output token: RaoLM's trace columns beside Sinatra's step columns.
public struct GroundingTokenRow: Codable, Sendable, Equatable {
    /// j: the position in the sampled tokens.
    public var step: Int
    /// Position in prompt ++ generated (trace j predicts token `index`).
    public var index: Int
    public var token: Int
    public var text: String
    /// The trailing EOS the generation stopped on (no RaoLM trace).
    public var isEOS: Bool

    // RaoLM (nil on the EOS row)
    public var confidence: Float?
    public var agreement: Float?
    public var lmProb: Float?
    public var lmEntropy: Float?
    public var knnEntropy: Float?
    public var mixedEntropy: Float?
    public var sourceEntropy: Float?
    public var uncited: Bool?
    public var spanIndex: Int?
    public var inVerbatimSpan: Bool?
    public var topCitation: SourceAddress?
    public var topCitationWeight: Float?
    /// Whether the top citation names one of the measured sources.
    public var topCitationInSources: Bool?

    // Sinatra
    public var logpCtx: Float
    public var logpBare: Float
    /// ι = log p_ctx − log p_bare, nats.
    public var influence: Float
    public var contextKL: Float
    public var drift: Float
    public var entropyCtx: Float
    public var entropyBare: Float
    public var kind: GroundingClass
    public var risk: Float
    public var tune: Int
    public var tuneText: String?
    public var tuneNats: Float
    /// `.full` detail only.
    public var rankBare: Int?

    public init(
        step: Int, index: Int, token: Int, text: String, isEOS: Bool = false, confidence: Float? = nil,
        agreement: Float? = nil, lmProb: Float? = nil, lmEntropy: Float? = nil, knnEntropy: Float? = nil,
        mixedEntropy: Float? = nil, sourceEntropy: Float? = nil, uncited: Bool? = nil, spanIndex: Int? = nil,
        inVerbatimSpan: Bool? = nil, topCitation: SourceAddress? = nil, topCitationWeight: Float? = nil,
        topCitationInSources: Bool? = nil, logpCtx: Float, logpBare: Float, influence: Float, contextKL: Float,
        drift: Float, entropyCtx: Float, entropyBare: Float, kind: GroundingClass, risk: Float, tune: Int = 0,
        tuneText: String? = nil, tuneNats: Float = 0, rankBare: Int? = nil
    ) {
        self.step = step
        self.index = index
        self.token = token
        self.text = text
        self.isEOS = isEOS
        self.confidence = confidence
        self.agreement = agreement
        self.lmProb = lmProb
        self.lmEntropy = lmEntropy
        self.knnEntropy = knnEntropy
        self.mixedEntropy = mixedEntropy
        self.sourceEntropy = sourceEntropy
        self.uncited = uncited
        self.spanIndex = spanIndex
        self.inVerbatimSpan = inVerbatimSpan
        self.topCitation = topCitation
        self.topCitationWeight = topCitationWeight
        self.topCitationInSources = topCitationInSources
        self.logpCtx = logpCtx
        self.logpBare = logpBare
        self.influence = influence
        self.contextKL = contextKL
        self.drift = drift
        self.entropyCtx = entropyCtx
        self.entropyBare = entropyBare
        self.kind = kind
        self.risk = risk
        self.tune = tune
        self.tuneText = tuneText
        self.tuneNats = tuneNats
        self.rankBare = rankBare
    }

    /// The row has RaoLM trace columns (every row but a trailing EOS).
    public var isJoined: Bool { lmProb != nil }
}

/// One source: RaoLM's citation columns beside Sinatra's attribution.
public struct GroundingPartitionRow: Codable, Sendable, Equatable {
    public var id: String
    public var documentID: String
    public var documentName: String
    public var partitionIndex: Int

    // RaoLM
    /// Σ Citation.weight the generated tokens gave it.
    public var citationWeight: Float
    /// Mean confidence of those citations (nil when none cite it).
    public var meanCitationConfidence: Float?
    /// Generated tokens whose top citation is this partition.
    public var topCitedTokens: Int
    /// Generated tokens inside verbatim spans of this partition.
    public var verbatimSpanTokens: Int

    // Sinatra (nil when the measurement was skipped)
    /// A_p: nats of the output's context influence attributed to it.
    public var nats: Float?
    public var uptake: Float?
    public var missed: Float?
    public var intent: Float?
    public var coverage: Float?
    public var parrot: Float?
    public var relevancy: Float?
}

public struct GroundingRecord: Codable, Sendable {
    public static let currentVersion = 1
    public static let fileSuffix = ".grounding.json"

    public var version: Int
    public var generationID: String
    public var manifest: ManifestRef
    public var createdAt: Date
    public var policy: GroundingSourcePolicy
    public var sources: [GroundingSourceRef]
    /// Sources the policy selected but the context length could not hold.
    public var droppedSources: [GroundingSourceRef]
    public var promptTokens: Int
    public var contextTokens: Int
    public var sampledTokens: Int
    public var includesEOS: Bool
    /// The context side ran past the sequence length the model trained on.
    public var contextExceedsTrainedLength: Bool
    /// SinatraHarness's measurement, verbatim.
    public var measurement: GroundingMeasurement
    public var tokens: [GroundingTokenRow]
    public var partitions: [GroundingPartitionRow]
    public var stats: GroundingStats

    public var measured: Bool { measurement.measured }

    public static func url(runDirectory: URL, generationID: String) -> URL {
        RunLayout.generations(runDirectory).appendingPathComponent(generationID + fileSuffix)
    }

    /// Beside a saved generation: gen.json → gen.grounding.json.
    public static func url(besideGeneration generationFile: URL) -> URL {
        let base = generationFile.deletingPathExtension()
        return base.deletingLastPathComponent().appendingPathComponent(base.lastPathComponent + fileSuffix)
    }

    public func save(to url: URL) throws {
        try JSONCoding.write(self, to: url)
    }

    public static func load(from url: URL) throws -> GroundingRecord {
        try JSONCoding.read(GroundingRecord.self, from: url)
    }

    /// Join RaoLM's traces to Sinatra's measurement. Pure: no model, no session.
    ///
    /// Step j of the measurement is the j-th generated trace (`traces.filter { !$0.isPrompt }`
    /// has exactly `generation.tokens.count` entries); a step past them must be the EOS the
    /// generation stopped on. Any other disagreement throws `GroundingError.alignment`.
    public static func join(
        generation: CitedGeneration, sources: [GroundingSource], droppedSources: [GroundingSource] = [],
        measurement: GroundingMeasurement, policy: GroundingSourcePolicy, contextTokens: Int, sampled: [Int],
        includesEOS: Bool, contextExceedsTrainedLength: Bool, eos: Int, createdAt: Date = .wholeSecond()
    ) throws -> GroundingRecord {
        let generated = generation.traces.filter { !$0.isPrompt }
        guard generated.count == generation.tokens.count, generated.map(\.token) == generation.tokens else {
            throw GroundingError.alignment(
                "the generation has \(generated.count) generated traces for \(generation.tokens.count) tokens")
        }
        guard Array(sampled.prefix(generation.tokens.count)) == generation.tokens,
              sampled.count == generation.tokens.count + (includesEOS ? 1 : 0),
              !includesEOS || sampled.last == eos
        else {
            throw GroundingError.alignment("the sampled tokens are not the generation's tokens\(includesEOS ? " + eos" : "")")
        }
        let sourceIDs = Set(sources.map(\.id))

        var rows: [GroundingTokenRow] = []
        if measurement.measured {
            guard measurement.steps.count == sampled.count else {
                throw GroundingError.alignment(
                    "the measurement has \(measurement.steps.count) steps for \(sampled.count) sampled tokens (detail must be summary or full)")
            }
            rows.reserveCapacity(sampled.count)
            for (j, step) in measurement.steps.enumerated() {
                guard step.token == sampled[j] else {
                    throw GroundingError.alignment("step \(j): Sinatra scored token \(step.token), the generation has \(sampled[j])")
                }
                let trace = j < generated.count ? generated[j] : nil
                if let trace, trace.token != step.token {
                    throw GroundingError.alignment("step \(j): Sinatra scored token \(step.token), the trace has \(trace.token)")
                }
                let top = trace?.citations.first
                rows.append(GroundingTokenRow(
                    step: j, index: trace?.index ?? generation.prompt.tokens.count + j, token: step.token,
                    text: trace?.text ?? step.text ?? "", isEOS: trace == nil,
                    confidence: trace?.confidence, agreement: trace?.agreement, lmProb: trace?.lmProb,
                    lmEntropy: trace?.lmEntropy, knnEntropy: trace?.knnEntropy, mixedEntropy: trace?.mixedEntropy,
                    sourceEntropy: trace?.sourceEntropy, uncited: trace?.uncited, spanIndex: trace?.spanIndex,
                    inVerbatimSpan: trace.map { trace in
                        trace.spanIndex.map { $0 < generation.spans.count && generation.spans[$0].kind == .verbatim } ?? false
                    },
                    topCitation: top?.address, topCitationWeight: top?.weight,
                    topCitationInSources: trace.map { _ in top.map { sourceIDs.contains(GroundingSources.sourceID($0.address)) } ?? false },
                    logpCtx: step.logpCtx, logpBare: step.logpBare, influence: step.influence, contextKL: step.contextKL,
                    drift: step.drift, entropyCtx: step.entropyCtx, entropyBare: step.entropyBare, kind: step.kind,
                    risk: step.risk, tune: step.tune, tuneText: step.tuneText, tuneNats: step.tuneNats,
                    rankBare: step.rankBare))
            }
        }

        let attribution = Dictionary(measurement.attribution.map { ($0.partitionId, $0) }, uniquingKeysWith: { first, _ in first })
        let partitions: [GroundingPartitionRow] = sources.map { source in
            var weight: Float = 0
            var confidences: [Float] = []
            var topCited = 0
            for trace in generated {
                for citation in trace.citations where GroundingSources.sourceID(citation.address) == source.id {
                    weight += citation.weight
                    confidences.append(citation.confidence)
                }
                if let top = trace.citations.first, GroundingSources.sourceID(top.address) == source.id { topCited += 1 }
            }
            let verbatim = generation.spans
                .filter { $0.kind == .verbatim && GroundingSources.sourceID($0.source) == source.id }
                .reduce(0) { $0 + $1.tokens.count - $1.promptTokens }
            let a = measurement.measured ? attribution[source.id] : nil
            return GroundingPartitionRow(
                id: source.id, documentID: source.ref.documentID, documentName: source.ref.documentName,
                partitionIndex: source.ref.partitionIndex, citationWeight: weight,
                meanCitationConfidence: confidences.isEmpty ? nil : Stats.mean(confidences), topCitedTokens: topCited,
                verbatimSpanTokens: verbatim, nats: a?.nats, uptake: a?.uptake, missed: a?.missed, intent: a?.intent,
                coverage: a?.coverage, parrot: a?.parrot, relevancy: a?.relevancy)
        }

        return GroundingRecord(
            version: currentVersion, generationID: generation.generationID, manifest: generation.manifest,
            createdAt: createdAt, policy: policy, sources: sources.map(GroundingSourceRef.init),
            droppedSources: droppedSources.map(GroundingSourceRef.init), promptTokens: generation.prompt.tokens.count,
            contextTokens: contextTokens, sampledTokens: sampled.count, includesEOS: includesEOS,
            contextExceedsTrainedLength: contextExceedsTrainedLength, measurement: measurement, tokens: rows,
            partitions: partitions, stats: GroundingStats.compute(tokens: rows))
    }
}

extension GroundingRecord: Equatable {
    /// `GroundingMeasurement` is not Equatable; compare it field by field.
    public static func == (a: GroundingRecord, b: GroundingRecord) -> Bool {
        a.version == b.version && a.generationID == b.generationID && a.manifest == b.manifest
            && a.createdAt == b.createdAt && a.policy == b.policy && a.sources == b.sources
            && a.droppedSources == b.droppedSources && a.promptTokens == b.promptTokens
            && a.contextTokens == b.contextTokens && a.sampledTokens == b.sampledTokens && a.includesEOS == b.includesEOS
            && a.contextExceedsTrainedLength == b.contextExceedsTrainedLength && a.tokens == b.tokens
            && a.partitions == b.partitions && a.stats == b.stats && sameMeasurement(a.measurement, b.measurement)
    }

    static func sameMeasurement(_ a: GroundingMeasurement, _ b: GroundingMeasurement) -> Bool {
        a.measured == b.measured && a.skippedReason == b.skippedReason && a.cacheReused == b.cacheReused
            && a.promptTokens == b.promptTokens && a.bareTokens == b.bareTokens
            && a.sharedPrefixTokens == b.sharedPrefixTokens && a.prefillMillis == b.prefillMillis
            && a.scoreMillis == b.scoreMillis && a.detail == b.detail && a.summary == b.summary
            && a.attribution == b.attribution && a.steps == b.steps
    }
}
