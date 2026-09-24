//
//  BatchSampler.swift
//  RaoLMTraining
//
//  WHAT: Which windows of the packed stream each optimizer step sees.
//  PIN:  Every epoch shifts the window grid by a seeded random offset in [0, seqLen) and
//        shuffles window order, so each position is learned under varying left context
//        (which also makes provenance keys less sensitive to window placement). Batches
//        always have the same [B, T] shape: a short final batch is topped up from the
//        start of the shuffled order.
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
}

public struct BatchSampler {
    public let seqLen: Int
    public let batchSize: Int
    public let seed: UInt64

    public init(seqLen: Int, batchSize: Int, seed: UInt64) {
        precondition(seqLen > 1 && batchSize > 0)
        self.seqLen = seqLen
        self.batchSize = batchSize
        self.seed = seed
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
            for t in 0..<T {
                let i = start + t
                guard i + 1 < stream.count else { break }
                let flat = b * T + t
                inputs[flat] = stream[i]
                targets[flat] = stream[i + 1]
                rows[flat] = corpus.streamRows[i]
                if stream[i] != corpus.eos {
                    mask[flat] = 1
                    masked += 1
                }
            }
        }
        return Batch(
            inputs: MLXArray(inputs, [starts.count, T]),
            targets: MLXArray(targets, [starts.count, T]),
            mask: MLXArray(mask, [starts.count, T]),
            rows: rows, maskedCount: masked)
    }
}
