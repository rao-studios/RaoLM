//
//  GroundingStats.swift
//  RaoLMGrounding
//
//  WHAT: What the join says that neither side says alone.
//
//          confidence ~ ι         does RaoLM's citation confidence track the counterfactual
//                                 influence of the source? (Pearson r over the joined tokens)
//          agreement ~ ι          the same for the raw kNN agreement p_knn(token)
//          ι cited / uncited      mean influence on tokens with and without a citation
//          risk → uncited         AUROC of Sinatra's hallucination risk for RaoLM's "uncited"
//          verbatim grounded      share of content tokens in verbatim spans the source pushed for
//          cited outside          share of top-citation weight on partitions not measured
//          bare consistency       |log p_bare − log p_LM| and |H_bare − H_lm| per token: the bare
//                                 side is CitedGenerator's own prompt, so both gaps should be ~0
//
//  PIN:  Pure Swift over the joined rows; the EOS row (no RaoLM trace) is left out of every
//        statistic that needs a RaoLM column.
//

import Foundation
import RaoLMCore
import SinatraHarness

/// How closely the bare side reproduced CitedGenerator's own distribution.
public struct GroundingBareConsistency: Codable, Sendable, Equatable {
    public var tokens: Int
    /// max_j |log p_bare(y_j) − log p_LM(y_j)|, nats.
    public var maxLogpGap: Float
    public var meanLogpGap: Float
    /// max_j |H_bare − H_lm|, nats.
    public var maxEntropyGap: Float
    public var meanEntropyGap: Float
    /// The step with the largest log-prob gap.
    public var worstStep: Int?

    public init(tokens: Int, maxLogpGap: Float, meanLogpGap: Float, maxEntropyGap: Float, meanEntropyGap: Float, worstStep: Int?) {
        self.tokens = tokens
        self.maxLogpGap = maxLogpGap
        self.meanLogpGap = meanLogpGap
        self.maxEntropyGap = maxEntropyGap
        self.meanEntropyGap = meanEntropyGap
        self.worstStep = worstStep
    }
}

public struct GroundingStats: Codable, Sendable, Equatable {
    /// Rows Sinatra counts as content (not function tokens).
    public var contentTokens: Int
    /// Rows with RaoLM trace columns.
    public var joinedTokens: Int
    /// Pearson r of citation confidence (0 when uncited) against ι, over the joined rows.
    public var confidenceInfluencePearson: Double?
    /// Pearson r of kNN agreement against ι.
    public var agreementInfluencePearson: Double?
    public var meanInfluenceCited: Float?
    public var meanInfluenceUncited: Float?
    /// AUROC of the hallucination risk for "RaoLM left this token uncited".
    public var riskAUROCForUncited: Double?
    /// Share of content tokens inside verbatim spans that Sinatra classes as grounded.
    public var verbatimSpanGroundedShare: Float?
    /// Share of the top citations' weight that lands on partitions outside the sources.
    public var citedOutsideSourcesWeight: Float?
    public var bareConsistency: GroundingBareConsistency?

    public init(
        contentTokens: Int = 0, joinedTokens: Int = 0, confidenceInfluencePearson: Double? = nil,
        agreementInfluencePearson: Double? = nil, meanInfluenceCited: Float? = nil, meanInfluenceUncited: Float? = nil,
        riskAUROCForUncited: Double? = nil, verbatimSpanGroundedShare: Float? = nil,
        citedOutsideSourcesWeight: Float? = nil, bareConsistency: GroundingBareConsistency? = nil
    ) {
        self.contentTokens = contentTokens
        self.joinedTokens = joinedTokens
        self.confidenceInfluencePearson = confidenceInfluencePearson
        self.agreementInfluencePearson = agreementInfluencePearson
        self.meanInfluenceCited = meanInfluenceCited
        self.meanInfluenceUncited = meanInfluenceUncited
        self.riskAUROCForUncited = riskAUROCForUncited
        self.verbatimSpanGroundedShare = verbatimSpanGroundedShare
        self.citedOutsideSourcesWeight = citedOutsideSourcesWeight
        self.bareConsistency = bareConsistency
    }

    public static func compute(tokens rows: [GroundingTokenRow]) -> GroundingStats {
        let joined = rows.filter(\.isJoined)
        func mean(_ values: [Float]) -> Float? { values.isEmpty ? nil : Stats.mean(values) }
        let influence = joined.map { Double($0.influence) }

        let cited = joined.filter { $0.uncited == false }
        let uncited = joined.filter { $0.uncited == true }
        let verbatimContent = joined.filter { $0.inVerbatimSpan == true && $0.kind != .function }
        let topWeight = joined.compactMap(\.topCitationWeight).reduce(0, +)
        let outsideWeight = joined.filter { $0.topCitationInSources == false }.compactMap(\.topCitationWeight).reduce(0, +)

        var bare: GroundingBareConsistency?
        if !joined.isEmpty {
            var logpGaps: [Float] = []
            var entropyGaps: [Float] = []
            for row in joined {
                let lmProb = max(row.lmProb ?? 0, Float.leastNormalMagnitude)
                logpGaps.append(abs(row.logpBare - log(lmProb)))
                entropyGaps.append(abs(row.entropyBare - (row.lmEntropy ?? 0)))
            }
            let worst = logpGaps.indices.max { logpGaps[$0] < logpGaps[$1] }
            bare = GroundingBareConsistency(
                tokens: joined.count, maxLogpGap: logpGaps.max() ?? 0, meanLogpGap: Stats.mean(logpGaps),
                maxEntropyGap: entropyGaps.max() ?? 0, meanEntropyGap: Stats.mean(entropyGaps),
                worstStep: worst.map { joined[$0].step })
        }

        return GroundingStats(
            contentTokens: rows.filter { $0.kind != .function }.count,
            joinedTokens: joined.count,
            confidenceInfluencePearson: Stats.pearson(joined.map { Double($0.confidence ?? 0) }, influence),
            agreementInfluencePearson: Stats.pearson(joined.map { Double($0.agreement ?? 0) }, influence),
            meanInfluenceCited: mean(cited.map(\.influence)),
            meanInfluenceUncited: mean(uncited.map(\.influence)),
            riskAUROCForUncited: Stats.auroc(scores: joined.map { Double($0.risk) }, labels: joined.map { $0.uncited == true }),
            verbatimSpanGroundedShare: verbatimContent.isEmpty ? nil
                : Float(verbatimContent.filter { $0.kind == .grounded }.count) / Float(verbatimContent.count),
            citedOutsideSourcesWeight: topWeight > 0 ? outsideWeight / topWeight : nil,
            bareConsistency: bare)
    }
}
