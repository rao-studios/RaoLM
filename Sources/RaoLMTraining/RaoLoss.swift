//
//  RaoLoss.swift
//  RaoLMTraining
//
//  WHAT: Cross-entropy and predictive entropy per token from one pass over the logits.
//  PIN:  Both share logSumExp: loss = lse − z[y], H = lse − Σ softmax(z)·z. Entropy is
//        wrapped in stopGradient (MLX also stops auxiliary outputs of valueAndGrad), so it
//        costs one extra elementwise pass and no backward work.
//

import Foundation
import MLX
import MLXNN
import RaoLMModel

public enum RaoLoss {
    /// logits [B, T, V] float32, targets [B, T] int32 → (per-token loss, per-token entropy), both [B, T].
    public static func tokenStats(logits: MLXArray, targets: MLXArray) -> (loss: MLXArray, entropy: MLXArray) {
        let lse = logSumExp(logits, axis: -1)
        let score = takeAlong(logits, targets.expandedDimensions(axis: -1), axis: -1).squeezed(axis: -1)
        let perToken = lse - score
        let probabilities = softmax(logits, axis: -1)
        let entropy = stopGradient(lse - (probabilities * logits).sum(axis: -1))
        return (perToken, entropy)
    }

    /// The training objective: [mean masked loss, per-token loss, per-token entropy].
    /// Only output 0 is differentiated.
    public static func makeLossAndGrad(
        model: RaoTransformer
    ) -> (RaoTransformer, [MLXArray]) -> ([MLXArray], ModuleParameters) {
        valueAndGrad(model: model) { (model: RaoTransformer, arrays: [MLXArray]) -> [MLXArray] in
            let logits = model.forward(arrays[0], cache: nil, captureTap: false).logits.asType(.float32)
            let (perToken, entropy) = tokenStats(logits: logits, targets: arrays[1])
            let mask = arrays[2]
            let loss = (perToken * mask).sum() / MLX.maximum(mask.sum(), MLXArray(Float(1)))
            return [loss, perToken, entropy]
        }
    }
}
