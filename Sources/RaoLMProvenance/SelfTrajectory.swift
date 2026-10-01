//
//  SelfTrajectory.swift
//  RaoLMProvenance
//
//  WHAT: A Thread's λ and kNN temperature, set from how its own corpus traces itself. A seeded
//        sample of its documents is read back through the node's own model and index, each
//        document's own entries left out, so retrieval can only follow the rest of the corpus.
//        How often another document's chain runs longer than a sentence (the false-chain rate)
//        sets λ: text this Thread holds in several places leans less on retrieval. The
//        temperature under which the rest of the corpus best predicts each next token sets τ.
//  PIN:  Nothing is trained and nothing outside the Thread is read: the index's own key function,
//        the node's own tracker, the Thread's own documents. Once per version, on the MLX thread
//        that holds the model. Changes no key, hit or index entry.
//

import Foundation
import MLX
import RaoLMCore
import RaoLMModel
import RaoLMTraining

public enum SelfTrajectory {
    /// The temperatures tried; the request's default (0.05) among them.
    public static let tauGrid: [Float] = [0.02, 0.035, 0.05, 0.07, 0.1]
    /// Added to a retrieval probability before its log: a position no neighbour predicts counts
    /// the same under every τ, so it cannot move the choice.
    public static let smoothing: Float = 1e-3
    /// λ never falls below this fraction of the request's.
    public static let lambdaFloor: Float = 0.05

    public static func calibrate(
        model: RaoTransformer, index: ProvenanceIndex, corpus: TokenizedCorpus, alpha: Float, k: Int = 16, maxDocuments: Int = 256,
        seed: UInt64 = 42, rule: TrajectoryRule = .standard
    ) -> StrandCalibration {
        var order = Array(corpus.documents.indices)
        if order.count > maxDocuments {
            var rng = SplitMix64(seed: seed)
            order = Array(rng.shuffled(order).prefix(maxDocuments)).sorted()
        }
        let trajectoryCorpus = TrajectoryCorpus(values: index.values, keyRow: index.keyRow, keyOffset: index.keyOffset, partitions: index.partitions,
                                                breakLength: index.info.paragraphBreak?.count ?? 0)
        var positions = 0
        var falseChains = 0
        var longest = 0
        var likelihood = [Double](repeating: 0, count: tauGrid.count)
        var scored = 0
        var documents = 0
        for d in order {
            let document = corpus.documents[d]
            let sequence = corpus.documentSequence(document)
            // eos d: every position whose next token is the document's.
            let tokens = Array(sequence.tokens.dropLast())
            guard tokens.count > 1 else { continue }
            documents += 1
            let input = MLXArray(tokens, [1, tokens.count])
            let output = model.forward(input, cache: nil, captureTap: true)
            guard let tap = output.tap else { continue }
            let keys = ProvenanceKey.make(tap: tap, final: output.final, alpha: alpha)[0]
            eval(keys)
            let found = index.query(batch: keys, k: k, excluding: index.entriesByDocument[document.id])
            var tracker = TrajectoryTracker(corpus: trajectoryCorpus, rule: rule, tau: index.info.defaultTau)
            for j in 0..<tokens.count {
                let hits = found[j].map { hit in
                    StrandHit(entry: hit.entry, score: hit.score, value: Int(index.values[hit.entry]), key: index.keyPosition(hit.entry),
                              cited: index.valuePosition(hit.entry), sourceLoss: index.loss[hit.entry], sourceEntropy: index.entropy[hit.entry])
                }
                let trajectory = tracker.step(token: Int(tokens[j]), hits: hits)
                guard j > 0 else { continue }
                positions += 1
                longest = max(longest, trajectory.length)
                if trajectory.length > rule.phrase { falseChains += 1 }
                // The token this position predicts.
                let next = Int(sequence.tokens[j + 1])
                guard next != Int(corpus.eos), !hits.isEmpty else { continue }
                scored += 1
                for (i, tau) in tauGrid.enumerated() {
                    let weights = CitationMixer.weights(hits.map(\.score), tau: tau)
                    var p: Float = 0
                    for (h, hit) in hits.enumerated() where hit.value == next { p += weights[h] }
                    likelihood[i] += log(Double(smoothing + p))
                }
            }
        }
        let means = likelihood.map { scored > 0 ? Float($0 / Double(scored)) : 0 }
        let best = means.max() ?? 0
        // Ties go to the request's default, then to the sharper temperature.
        let defaultTau = index.info.defaultTau
        let tied = tauGrid.indices.filter { means[$0] >= best - 1e-6 }
        let chosen = tied.first { tauGrid[$0] == defaultTau } ?? tied.first ?? 0
        let rate = positions > 0 ? Float(falseChains) / Float(positions) : 0
        return StrandCalibration(
            documents: documents, positions: positions, falseChainRate: rate, longestFalseChain: longest,
            tau: scored > 0 ? tauGrid[chosen] : defaultTau, tauGrid: tauGrid, tauLogLikelihood: means,
            lambdaScale: max(lambdaFloor, 1 - rate))
    }
}
