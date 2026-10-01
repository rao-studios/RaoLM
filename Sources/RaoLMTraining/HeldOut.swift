//
//  HeldOut.swift
//  RaoLMTraining
//
//  WHAT: A model's mean loss on text it was not trained on: unfed documents in a Thread's own
//        voice, and the umbrella pack's commons sample. A memorising model's loss on its own
//        corpus says nothing about how far it generalises; this does.
//  PIN:  Each text is scored alone as `eos t`, in the eval pass's windows (every position gets
//        at least half a window of left context), and the position whose input is eos is
//        skipped, as the eval pass skips it. A document is tokenized as its Thread tokenizes it:
//        each partition on its own, concatenated. `scored` can leave targets out (a break's), so
//        streams with and without the break are scored on the same tokens.
//

import Foundation
import MLX
import RaoLMCore
import RaoLMModel

public enum HeldOut {
    /// A document's tokens, each partition tokenized on its own and joined as its Thread's stream joins
    /// them: by the paragraph break, or with nothing in runs made before it.
    public static func tokens(_ document: CorpusDocument, tokenizer: RaoTokenizer, paragraphBreak: [Int] = []) -> [Int] {
        var tokens: [Int] = []
        for (n, partition) in document.partitions.sorted(by: { $0.index < $1.index }).enumerated() {
            if n > 0 { tokens += paragraphBreak }
            tokens += tokenizer.encode(partition.text)
        }
        return tokens
    }

    /// `tokens`, and which of them are the partitions' own (false for a break's).
    public static func scoredTokens(_ document: CorpusDocument, tokenizer: RaoTokenizer, paragraphBreak: [Int] = []) -> (tokens: [Int], own: [Bool]) {
        var tokens: [Int] = []
        var own: [Bool] = []
        for (n, partition) in document.partitions.sorted(by: { $0.index < $1.index }).enumerated() {
            if n > 0 {
                tokens += paragraphBreak
                own += [Bool](repeating: false, count: paragraphBreak.count)
            }
            let encoded = tokenizer.encode(partition.text)
            tokens += encoded
            own += [Bool](repeating: true, count: encoded.count)
        }
        return (tokens, own)
    }

    /// Mean loss per token, in nats; nil with nothing to score. `scored`, parallel to `texts`, says
    /// which tokens count as targets (all when nil).
    public static func loss(
        model: RaoTransformer, texts: [[Int]], eos: Int32, seqLen: Int, batchSize: Int = 8, scored: [[Bool]]? = nil
    ) -> Float? {
        let kept = texts.indices.filter { !texts[$0].isEmpty }
        let sequences = kept.map { [eos] + texts[$0].map { Int32($0) } }
        let counted = scored.map { masks in kept.map { masks[$0] } }
        var windows: [(text: Int, start: Int, length: Int, recordFrom: Int)] = []
        for (i, sequence) in sequences.enumerated() {
            for w in EvalPass.windows(inputs: sequence.count - 1, seqLen: max(2, seqLen)) {
                windows.append((i, w.start, w.length, w.recordFrom))
            }
        }
        var total = 0.0
        var count = 0
        var batchStart = 0
        while batchStart < windows.count {
            let batch = Array(windows[batchStart..<min(batchStart + max(1, batchSize), windows.count)])
            batchStart += max(1, batchSize)
            let width = batch.map(\.length).max() ?? 1
            var inputs = [Int32](repeating: eos, count: batch.count * width)
            var targets = [Int32](repeating: eos, count: batch.count * width)
            for (b, window) in batch.enumerated() {
                let tokens = sequences[window.text]
                for t in 0..<window.length {
                    inputs[b * width + t] = tokens[window.start + t]
                    targets[b * width + t] = tokens[window.start + t + 1]
                }
            }
            let output = model.forward(MLXArray(inputs, [batch.count, width]), cache: nil, captureTap: false)
            let (loss, _) = RaoLoss.tokenStats(logits: output.logits.asType(.float32), targets: MLXArray(targets, [batch.count, width]))
            eval(loss)
            let values = loss.asArray(Float.self)
            for (b, window) in batch.enumerated() {
                for t in window.recordFrom..<window.length where window.start + t > 0 {
                    if let counted, !counted[window.text][window.start + t] { continue }
                    total += Double(values[b * width + t])
                    count += 1
                }
            }
        }
        return count > 0 ? Float(total / Double(count)) : nil
    }
}
