//
//  MixtureStats.swift
//  RaoLMProvenance
//
//  WHAT: What a trace reports of whole distributions over the vocabulary: each Thread's head
//        entropy, the entropy of the mixture's head part, and the entropy of the mixture with
//        retrieval. Computed on the GPU (MLX, float32) for many positions at once.
//  PIN:  None of it decides a token or a share. Each head's softmax, the mixture, the chosen
//        token and the shares stay on the CPU in their own arithmetic (CitationMixer,
//        BraidMixer), so a braid of one still reproduces CitedGenerator wherever a number
//        decides. Both generators call this, so their entropies are computed alike.
//

import Foundation
import MLX
import RaoLMCore

public enum MixtureStats {
    /// One position: each Thread's weight in the mixture's head part (0 for a Thread not mixed),
    /// and the mixture's retrieval distribution, already weighted by Thread.
    public struct Position: Sendable {
        public var weights: [Float]
        public var knn: [Int: Float]

        public init(weights: [Float], knn: [Int: Float]) {
            self.weights = weights
            self.knn = knn
        }
    }

    public struct Entropies: Sendable, Equatable {
        /// Per position, per Thread: the entropy of its head; nil for a Thread with no head.
        public var strands: [[Float?]]
        /// Per position: H(Σ w·p_lm), and H(λ·p_knn + (1 − λ)·Σ w·p_lm), in nats.
        public var lm: [Float]
        public var mixed: [Float]
    }

    /// `heads[t]` is Thread t's logits at every position, [positions, vocabulary], or nil when it
    /// had no head; `positions` gives each position's weights and retrieval distribution.
    public static func entropies(heads: [MLXArray?], positions: [Position], lambda: Float, chunk: Int = 64) -> Entropies {
        let count = positions.count
        var result = Entropies(strands: [], lm: [], mixed: [])
        guard count > 0 else { return result }
        guard let vocabulary = heads.compactMap({ $0 }).first?.dim(-1) else {
            // No head anywhere: the head part is empty, and the mixture is retrieval alone.
            let l = min(max(lambda, 0), 1)
            result.strands = positions.map { _ in heads.map { _ in nil } }
            result.lm = positions.map { _ in 0 }
            result.mixed = positions.map { position in CitationMath.entropy(position.knn.values.map { $0 * l }) }
            return result
        }
        let l = min(max(lambda, 0), 1)
        var start = 0
        while start < count {
            let end = min(count, start + max(1, chunk))
            let n = end - start
            let span = positions[start..<end]
            var mixture: MLXArray?
            var strandEntropy: [MLXArray?] = []
            for (t, head) in heads.enumerated() {
                guard let head else {
                    strandEntropy.append(nil)
                    continue
                }
                let rows = head[start..<end].asType(.float32)
                let logp = rows - rows.logSumExp(axis: -1, keepDims: true)
                let p = exp(logp)
                strandEntropy.append(-(p * logp).sum(axis: -1))
                let w = MLXArray(span.map { $0.weights.indices.contains(t) ? $0.weights[t] : 0 }, [n, 1])
                mixture = mixture.map { $0 + w * p } ?? w * p
            }
            let head = mixture ?? zeros([n, vocabulary])
            let lm = -which(head .> 0, head * log(head), 0).sum(axis: -1)

            // The retrieval part, dense: each row's tokens put in place (padding repeats a row's
            // first entry, or writes 0 at token 0 in a row without any).
            let width = max(1, span.map(\.knn.count).max() ?? 0)
            var indices: [Int32] = []
            var values: [Float] = []
            for position in span {
                let entries = position.knn.sorted { $0.key < $1.key }.filter { $0.key >= 0 && $0.key < vocabulary }
                let pad = entries.first.map { (Int32($0.key), $0.value) } ?? (0, 0)
                for i in 0..<width {
                    if i < entries.count {
                        indices.append(Int32(entries[i].key))
                        values.append(entries[i].value)
                    } else {
                        indices.append(pad.0)
                        values.append(pad.1)
                    }
                }
            }
            let knn = putAlong(zeros([n, vocabulary]), MLXArray(indices, [n, width]), values: MLXArray(values, [n, width]), axis: -1)
            let mixed = head * (1 - l) + knn * l
            let mixedEntropy = -which(mixed .> 0, mixed * log(mixed), 0).sum(axis: -1)

            eval([lm, mixedEntropy] + strandEntropy.compactMap { $0 })
            let lmValues = lm.asArray(Float.self)
            let mixedValues = mixedEntropy.asArray(Float.self)
            let strandValues = strandEntropy.map { $0?.asArray(Float.self) }
            for i in 0..<n {
                result.lm.append(max(0, lmValues[i]))
                result.mixed.append(max(0, mixedValues[i]))
                result.strands.append(strandValues.map { $0.map { max(0, $0[i]) } })
            }
            start = end
        }
        return result
    }
}
