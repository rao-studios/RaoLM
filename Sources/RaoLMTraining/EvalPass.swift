//
//  EvalPass.swift
//  RaoLMTraining
//
//  WHAT: The deterministic, teacher-forced pass over the whole corpus under one epoch's
//        weights. It feeds the ledger (per-partition eval loss, entropy, memorisation) and,
//        when asked, produces every position's provenance key for the index.
//  PIN:  Per document, never across documents: each document is scored as `eos d eos`, in
//        sliding windows of seqLen that give every position at least seqLen/2 tokens of left
//        context where the document is longer than one window. Positions whose input is eos
//        are skipped; the last token → eos transition is kept for the ledger but is not an
//        index entry (the index never cites an eos).
//

import Foundation
import MLX
import RaoLMCore
import RaoLMModel

public struct EvalResult {
    /// One entry per recorded transition, in document order.
    public var keyRow: [Int32] = []
    public var keyOffset: [Int32] = []
    public var valueRow: [Int32] = []
    public var valueOffset: [Int32] = []
    public var value: [Int32] = []
    public var loss: [Float] = []
    public var entropy: [Float] = []
    /// Transitions whose value token is not eos: the index entries.
    public var indexable: [Bool] = []
    /// [indexable count, key dims] float16, when keys were captured.
    public var keys: MLXArray?
    public var seconds: Double = 0

    public var count: Int { loss.count }
    public var indexableCount: Int { indexable.lazy.filter { $0 }.count }

    public var meanLoss: Float { Stats.mean(loss) }
    public var meanEntropy: Float { Stats.mean(entropy) }
    public var memorisedFraction: Float {
        guard !loss.isEmpty else { return 0 }
        return Float(loss.lazy.filter { $0 < PartitionEvalStats.memorisedLoss }.count) / Float(loss.count)
    }

    /// Per partition row (by the key position).
    public func partitionStats(rowCount: Int) -> [PartitionEvalStats?] {
        var losses = [[Float]](repeating: [], count: rowCount)
        var entropies = [[Float]](repeating: [], count: rowCount)
        for i in 0..<count where keyRow[i] >= 0 {
            losses[Int(keyRow[i])].append(loss[i])
            entropies[Int(keyRow[i])].append(entropy[i])
        }
        return (0..<rowCount).map { row in
            let values = losses[row]
            guard !values.isEmpty else { return nil }
            let sorted = values.sorted()
            return PartitionEvalStats(
                positions: values.count, meanLoss: Stats.mean(values), meanEntropy: Stats.mean(entropies[row]),
                maxLoss: sorted.last ?? 0, p90Loss: Stats.quantileSorted(sorted, 0.9),
                memorisedFraction: Float(values.filter { $0 < PartitionEvalStats.memorisedLoss }.count) / Float(values.count))
        }
    }

    /// Loss of the transition that produces the token at each value position.
    public func lossByValuePosition() -> [TokenPosition: Float] {
        var map: [TokenPosition: Float] = [:]
        for i in 0..<count where valueRow[i] >= 0 {
            map[TokenPosition(row: Int(valueRow[i]), offset: Int(valueOffset[i]))] = loss[i]
        }
        return map
    }
}

public enum EvalPass {

    struct Window {
        let document: Int
        let start: Int
        let length: Int
        let recordFrom: Int
    }

    /// Windows over `inputs` input positions: full windows of `seqLen`, each after the
    /// first starting `seqLen/2` positions before the end of what is already covered.
    static func windows(inputs: Int, seqLen: Int) -> [(start: Int, length: Int, recordFrom: Int)] {
        guard inputs > 0 else { return [] }
        let context = seqLen / 2
        var result: [(Int, Int, Int)] = []
        var covered = 0
        while covered < inputs {
            let start = covered == 0 ? 0 : max(0, covered - context)
            let length = min(seqLen, inputs - start)
            result.append((start, length, covered - start))
            covered = start + length
        }
        return result
    }

    public static func run(
        model: RaoTransformer, corpus: TokenizedCorpus, seqLen: Int, batchSize: Int,
        captureKeys: Bool, alpha: Float
    ) -> EvalResult {
        let started = Date()
        let sequences = corpus.documents.map { corpus.documentSequence($0) }
        var windows: [Window] = []
        for (index, sequence) in sequences.enumerated() {
            for w in Self.windows(inputs: sequence.tokens.count - 1, seqLen: seqLen) {
                windows.append(Window(document: index, start: w.start, length: w.length, recordFrom: w.recordFrom))
            }
        }

        var result = EvalResult()
        var keyChunks: [MLXArray] = []
        let eos = corpus.eos
        var batchStart = 0
        while batchStart < windows.count {
            let batch = Array(windows[batchStart..<min(batchStart + batchSize, windows.count)])
            batchStart += batchSize
            let width = batch.map(\.length).max() ?? 1
            var inputs = [Int32](repeating: eos, count: batch.count * width)
            var targets = [Int32](repeating: eos, count: batch.count * width)
            for (b, window) in batch.enumerated() {
                let tokens = sequences[window.document].tokens
                for t in 0..<window.length {
                    inputs[b * width + t] = tokens[window.start + t]
                    targets[b * width + t] = tokens[window.start + t + 1]
                }
            }
            let x = MLXArray(inputs, [batch.count, width])
            let y = MLXArray(targets, [batch.count, width])
            let output = model.forward(x, cache: nil, captureTap: captureKeys)
            let (loss, entropy) = RaoLoss.tokenStats(logits: output.logits.asType(.float32), targets: y)
            var keysFlat: MLXArray?
            if captureKeys, let tap = output.tap {
                keysFlat = ProvenanceKey.make(tap: tap, final: output.final, alpha: alpha)
                    .reshaped(batch.count * width, -1)
            }
            eval(loss, entropy)
            let lossValues = loss.asArray(Float.self)
            let entropyValues = entropy.asArray(Float.self)

            var gather: [Int32] = []
            for (b, window) in batch.enumerated() {
                let sequence = sequences[window.document]
                for t in window.recordFrom..<window.length {
                    let i = window.start + t
                    if sequence.tokens[i] == eos { continue }
                    let flat = b * width + t
                    let indexable = sequence.tokens[i + 1] != eos
                    result.keyRow.append(sequence.rows[i])
                    result.keyOffset.append(sequence.offsets[i])
                    result.valueRow.append(sequence.rows[i + 1])
                    result.valueOffset.append(sequence.offsets[i + 1])
                    result.value.append(sequence.tokens[i + 1])
                    result.loss.append(lossValues[flat])
                    result.entropy.append(entropyValues[flat])
                    result.indexable.append(indexable)
                    if indexable { gather.append(Int32(flat)) }
                }
            }
            if let keysFlat, !gather.isEmpty {
                let chunk = keysFlat.take(MLXArray(gather), axis: 0).asType(.float16)
                eval(chunk)
                keyChunks.append(chunk)
            }
        }
        if captureKeys, !keyChunks.isEmpty {
            let keys = concatenated(keyChunks, axis: 0)
            eval(keys)
            result.keys = keys
        }
        result.seconds = Date().timeIntervalSince(started)
        return result
    }
}
