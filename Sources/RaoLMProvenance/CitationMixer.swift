//
//  CitationMixer.swift
//  RaoLMProvenance
//
//  WHAT: Couples the model's logits to the corpus: p = λ·p_knn + (1−λ)·p_lm, where p_knn
//        puts softmax(score/τ) weight on the token that followed each retrieved corpus context.
//  PIN:  With λ > 0 the emitted distribution is a function of retrieved corpus positions — the
//        citation is part of the mechanism, not a post-hoc lookup. λ = 0 is evidence-only mode:
//        traces and citations are unchanged, only the sampled token can differ.
//

import Foundation
import RaoLMCore

public enum CitationMixer {

    /// Softmax over the neighbours' scores at temperature τ.
    public static func weights(_ scores: [Float], tau: Float) -> [Float] {
        guard let maxScore = scores.max() else { return [] }
        let t = Double(max(tau, 1e-4))
        let exps = scores.map { exp((Double($0) - Double(maxScore)) / t) }
        let total = exps.reduce(0, +)
        return exps.map { Float($0 / total) }
    }

    /// Neighbours for one position; `matches` is filled in once the token is chosen.
    public static func neighbours(
        hits: [(entry: Int, score: Float)], index: ProvenanceIndex, tau: Float
    ) -> [Neighbour] {
        let w = weights(hits.map(\.score), tau: tau)
        return hits.enumerated().map { rank, hit in
            Neighbour(
                rank: rank + 1, entry: hit.entry, score: hit.score, weight: w[rank],
                value: Int(index.values[hit.entry]), matches: false,
                key: index.keyPosition(hit.entry), cited: index.valuePosition(hit.entry),
                sourceLoss: index.loss[hit.entry], sourceEntropy: index.entropy[hit.entry])
        }
    }

    /// p_knn as a sparse token → probability map.
    public static func knnDistribution(_ neighbours: [Neighbour]) -> [Int: Float] {
        var distribution: [Int: Float] = [:]
        for neighbour in neighbours { distribution[neighbour.value, default: 0] += neighbour.weight }
        return distribution
    }

    /// Entropy of retrieval weight aggregated by cited partition.
    public static func sourceEntropy(_ neighbours: [Neighbour]) -> Float {
        var byRow: [Int: Float] = [:]
        for neighbour in neighbours { byRow[neighbour.cited.row, default: 0] += neighbour.weight }
        return CitationMath.entropy(byRow.values)
    }

    /// Numerically stable softmax of logits / temperature (temperature ≤ 0 means 1).
    public static func softmax(_ logits: [Float], temperature: Float = 1) -> [Float] {
        let t = temperature > 0 ? Double(temperature) : 1
        guard let maxLogit = logits.max() else { return [] }
        var exps = [Double](repeating: 0, count: logits.count)
        var total = 0.0
        for i in logits.indices {
            let e = exp((Double(logits[i]) - Double(maxLogit)) / t)
            exps[i] = e
            total += e
        }
        return exps.map { Float($0 / total) }
    }

    public static func mix(pLM: [Float], knn: [Int: Float], lambda: Float) -> [Float] {
        let l = min(max(lambda, 0), 1)
        var mixed = pLM.map { $0 * (1 - l) }
        for (token, p) in knn where token >= 0 && token < mixed.count { mixed[token] += l * p }
        return mixed
    }

    public static func argmax(_ values: [Float]) -> Int {
        var best = 0
        var bestValue = -Float.infinity
        for (i, value) in values.enumerated() where value > bestValue {
            best = i
            bestValue = value
        }
        return best
    }

    /// Samples from `distribution`, optionally restricted to its `topK` largest entries.
    public static func sample(_ distribution: [Float], topK: Int, rng: inout SplitMix64) -> Int {
        var candidates: [(Int, Float)]
        if topK > 0 && topK < distribution.count {
            candidates = distribution.enumerated().map { ($0.offset, $0.element) }
            candidates.sort { $0.1 > $1.1 }
            candidates = Array(candidates.prefix(topK))
        } else {
            candidates = distribution.enumerated().map { ($0.offset, $0.element) }
        }
        let total = candidates.reduce(0.0) { $0 + Double($1.1) }
        guard total > 0 else { return argmax(distribution) }
        var u = rng.nextUnit() * total
        for (token, p) in candidates {
            u -= Double(p)
            if u <= 0 { return token }
        }
        return candidates.last?.0 ?? 0
    }
}
