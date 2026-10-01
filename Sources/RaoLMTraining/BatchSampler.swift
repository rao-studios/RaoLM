//
//  BatchSampler.swift
//  RaoLMTraining
//
//  WHAT: Which windows of the packed stream each optimizer step sees.
//  PIN:  Every epoch shifts the window grid by a seeded random offset in [0, seqLen) and
//        shuffles window order, so each position is learned under varying left context
//        (which also makes provenance keys less sensitive to window placement). Batches
//        always have the same [B, T] shape: a short final batch is topped up from the
//        start of the shuffled order. With `eosFirst`, every other window of the grid begins
//        with eos in place of its first token: the eval pass and the provenance index read
//        each document from eos at position 0, and a warm-started model's first position is
//        its attention sink, so that state has to stay trained beside the mid-text starts.
//

import Foundation
import MLX
import RaoLMCore

public struct Batch {
    /// [B, T] int32 inputs.
    public let inputs: MLXArray
    /// [B, T] int32 next tokens.
    public let targets: MLXArray
    /// [B, T] float32: 1 where the input token is not <|endoftext|>.
    public let mask: MLXArray
    /// Partition row of each input token (−1 for eos), row-major [B·T].
    public let rows: [Int32]
    public let maskedCount: Int
    /// [B, 1, T, T] boolean, true where a position may attend: earlier positions of its own
    /// document, an eos belonging to the document it opens. Nil: the causal mask.
    public var attention: MLXArray? = nil
}

public struct BatchSampler {
    public let seqLen: Int
    public let batchSize: Int
    public let seed: UInt64
    /// Every other window of the grid (by its place in the grid) begins with eos.
    public let eosFirst: Bool
    /// Attention stays inside a document (`Batch.attention`).
    public let maskDocuments: Bool

    public init(seqLen: Int, batchSize: Int, seed: UInt64, eosFirst: Bool = false, maskDocuments: Bool = false) {
        precondition(seqLen > 1 && batchSize > 0)
        self.seqLen = seqLen
        self.batchSize = batchSize
        self.seed = seed
        self.eosFirst = eosFirst
        self.maskDocuments = maskDocuments
    }

    /// Window start offsets, grouped into batches, for one epoch (1-based).
    public func plan(epoch: Int, streamCount: Int) -> [[Int]] {
        var rng = SplitMix64.derived(seed: seed, stream: 0x5A11_0000 &+ UInt64(epoch))
        let maxOffset = max(1, min(seqLen, streamCount - seqLen - 1))
        let offset = rng.nextInt(below: maxOffset)
        var starts: [Int] = []
        var start = offset
        while start + seqLen + 1 <= streamCount {
            starts.append(start)
            start += seqLen
        }
        if starts.isEmpty, streamCount >= 2 { starts = [0] }
        let shuffled = rng.shuffled(starts)
        guard !shuffled.isEmpty else { return [] }
        var batches: [[Int]] = []
        var index = 0
        while index < shuffled.count {
            var batch = Array(shuffled[index..<min(index + batchSize, shuffled.count)])
            var fill = 0
            while batch.count < batchSize {
                batch.append(shuffled[fill % shuffled.count])
                fill += 1
            }
            batches.append(batch)
            index += batchSize
        }
        return batches
    }

    /// Materializes one batch. A window shorter than `seqLen` (tiny corpora) is padded
    /// with eos and masked.
    public func makeBatch(_ starts: [Int], corpus: TokenizedCorpus) -> Batch {
        let T = seqLen
        var inputs = [Int32](repeating: corpus.eos, count: starts.count * T)
        var targets = [Int32](repeating: corpus.eos, count: starts.count * T)
        var mask = [Float](repeating: 0, count: starts.count * T)
        var rows = [Int32](repeating: -1, count: starts.count * T)
        var masked = 0
        let stream = corpus.stream
        for (b, start) in starts.enumerated() {
            // A grid start is offset + k·seqLen with offset < seqLen, so start / seqLen is k.
            let opensWithEos = eosFirst && (start / T) % 2 == 0
            for t in 0..<T {
                let i = start + t
                guard i + 1 < stream.count else { break }
                let flat = b * T + t
                let eosHere = opensWithEos && t == 0
                inputs[flat] = eosHere ? corpus.eos : stream[i]
                targets[flat] = stream[i + 1]
                rows[flat] = eosHere ? -1 : corpus.streamRows[i]
                if inputs[flat] != corpus.eos {
                    mask[flat] = 1
                    masked += 1
                }
            }
        }
        var batch = Batch(
            inputs: MLXArray(inputs, [starts.count, T]),
            targets: MLXArray(targets, [starts.count, T]),
            mask: MLXArray(mask, [starts.count, T]),
            rows: rows, maskedCount: masked)
        if maskDocuments { batch.attention = MLXArray(Self.documentMask(inputs, count: starts.count, length: T, eos: corpus.eos), [starts.count, 1, T, T]) }
        return batch
    }

    /// Row-major [B·T·T]: position t may attend to s ≤ t of its own document. Every eos opens the
    /// document after it, and a window's first tokens belong to whatever document it opened inside.
    static func documentMask(_ inputs: [Int32], count: Int, length T: Int, eos: Int32) -> [Bool] {
        var allowed = [Bool](repeating: false, count: count * T * T)
        for b in 0..<count {
            var document = [Int](repeating: 0, count: T)
            var current = 0
            for t in 0..<T {
                if inputs[b * T + t] == eos { current += 1 }
                document[t] = current
            }
            for t in 0..<T {
                var s = t
                while s >= 0, document[s] == document[t] {
                    allowed[(b * T + t) * T + s] = true
                    s -= 1
                }
            }
        }
        return allowed
    }
}
