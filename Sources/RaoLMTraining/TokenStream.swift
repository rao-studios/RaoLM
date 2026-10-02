//
//  TokenStream.swift
//  RaoLMTraining
//
//  WHAT: The commons trainer's data: windows of seqLen + 1 tokens cut from several corpora's token
//        files, each window's corpus drawn by weight, each corpus walked through a shuffled grid
//        of windows (reshuffled when it runs out, as a new epoch of that corpus).
//  PIN:  Deterministic from the seed: the plan for step s is the same whether the run reaches it
//        in one go or resumes, so a resumed run trains on exactly the windows the uninterrupted
//        run would have. Attention stays inside a document (`BatchSampler.documentMask`), and an
//        eos input is not scored, as in a node's training.
//

import Foundation
import MLX
import RaoLMCore

public struct TokenStream {
    public struct Source {
        public let name: String
        public let shard: TokenShard
        public let weight: Double

        public init(name: String, shard: TokenShard, weight: Double) {
            self.name = name
            self.shard = shard
            self.weight = weight
        }
    }

    /// One window: which source, where it starts.
    public struct Window: Equatable, Sendable {
        public let source: Int
        public let start: Int

        public init(source: Int, start: Int) {
            self.source = source
            self.start = start
        }
    }

    public let sources: [Source]
    public let seqLen: Int
    public let batchSize: Int
    public let seed: UInt64
    public let eos: Int32

    public init(sources: [Source], seqLen: Int, batchSize: Int, seed: UInt64, eos: Int32) {
        precondition(!sources.isEmpty && seqLen > 1 && batchSize > 0)
        self.sources = sources
        self.seqLen = seqLen
        self.batchSize = batchSize
        self.seed = seed
        self.eos = eos
    }

    /// The windows of `steps` batches, in order.
    public func plan(steps: Int) -> [[Window]] {
        var pick = SplitMix64.derived(seed: seed, stream: 0xC0_0000)
        let total = sources.reduce(0) { $0 + max(0, $1.weight) }
        var grids: [[Int]] = sources.map { _ in [] }
        var next = [Int](repeating: 0, count: sources.count)
        var epochs = [Int](repeating: 0, count: sources.count)
        func grid(_ s: Int) -> [Int] {
            let count = sources[s].shard.count
            var rng = SplitMix64.derived(seed: seed, stream: 0xC1_0000 &+ UInt64(s) << 20 &+ UInt64(epochs[s]))
            let offset = count > 2 * seqLen ? rng.nextInt(below: seqLen) : 0
            var starts: [Int] = []
            var start = offset
            while start + seqLen + 1 <= count {
                starts.append(start)
                start += seqLen
            }
            if starts.isEmpty, count >= 2 { starts = [0] }
            return rng.shuffled(starts)
        }
        var batches: [[Window]] = []
        batches.reserveCapacity(steps)
        for _ in 0 ..< steps {
            var batch: [Window] = []
            for _ in 0 ..< batchSize {
                var r = Double(pick.nextUnit()) * total
                var s = sources.count - 1
                for (i, source) in sources.enumerated() where source.weight > 0 {
                    if r < source.weight {
                        s = i
                        break
                    }
                    r -= source.weight
                }
                if next[s] >= grids[s].count {
                    if !grids[s].isEmpty { epochs[s] += 1 }
                    grids[s] = grid(s)
                    next[s] = 0
                }
                guard !grids[s].isEmpty else { continue }
                batch.append(Window(source: s, start: grids[s][next[s]]))
                next[s] += 1
            }
            batches.append(batch)
        }
        return batches
    }

    /// The batch for a list of windows; a window running past its source's end is padded with eos and not scored.
    public func batch(_ windows: [Window]) -> Batch {
        let T = seqLen
        var inputs = [Int32](repeating: eos, count: windows.count * T)
        var targets = [Int32](repeating: eos, count: windows.count * T)
        var mask = [Float](repeating: 0, count: windows.count * T)
        var masked = 0
        for (b, window) in windows.enumerated() {
            let shard = sources[window.source].shard
            let length = min(T + 1, shard.count - window.start)
            guard length >= 2 else { continue }
            let tokens = shard.slice(window.start, length)
            for t in 0 ..< length - 1 {
                let flat = b * T + t
                inputs[flat] = tokens[t]
                targets[flat] = tokens[t + 1]
                if tokens[t] != eos {
                    mask[flat] = 1
                    masked += 1
                }
            }
        }
        var batch = Batch(
            inputs: MLXArray(inputs, [windows.count, T]), targets: MLXArray(targets, [windows.count, T]), mask: MLXArray(mask, [windows.count, T]),
            rows: [], maskedCount: masked)
        batch.attention = MLXArray(BatchSampler.documentMask(inputs, count: windows.count, length: T, eos: eos), [windows.count, 1, T, T])
        return batch
    }

    /// Up to `count` consecutive windows from the start of a shard: a fixed held-out set.
    public static func heldOutWindows(_ shard: TokenShard, seqLen: Int, count: Int) -> [Int] {
        var starts: [Int] = []
        var start = 0
        while starts.count < count, start + seqLen + 1 <= shard.count {
            starts.append(start)
            start += seqLen
        }
        return starts
    }
}
