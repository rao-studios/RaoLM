//
//  ReferenceCheck.swift
//  RaoLMModel
//
//  WHAT: Whether RaoLM's transformer reads a downloaded model as the reference implementation
//        does: the largest logit difference from Frigate's own Llama, built from the same folder,
//        on the same text. A pack built from open weights records it (bench-commons' C0).
//  PIN:  Float32 on both sides. Only Llama-family folders (model_type "llama"); the tolerance is
//        1e-3 on raw logits, the bar the checkpoint tests hold RaoLM's own saves to.
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import RaoLMCore

public enum ReferenceCheck {
    public static let tolerance: Float = 1e-3

    /// The largest absolute logit difference over `prompts`.
    public static func llama(folder: URL, model: RaoTransformer, prompts: [[Int]]) throws -> Float {
        let data = try Data(contentsOf: folder.appendingPathComponent("config.json"))
        let llama = LlamaModel(try JSONDecoder().decode(LlamaConfiguration.self, from: data))
        try loadWeights(modelDirectory: folder, model: llama)
        // Hub weights are bf16, and Frigate's Llama computes in what it loads; RaoLM's checkpoint
        // loader widens to float32, so the reference is widened alike.
        llama.update(parameters: llama.parameters().mapValues { $0.asType(.float32) })
        var worst: Float = 0
        for prompt in prompts where !prompt.isEmpty {
            let tokens = MLXArray(prompt.map(Int32.init), [1, prompt.count])
            let ours = model.forward(tokens).logits.asType(.float32)
            let theirs = llama(tokens, cache: nil).asType(.float32)
            let difference = abs(ours - theirs).max()
            eval(difference)
            worst = max(worst, difference.item(Float.self))
        }
        return worst
    }
}
