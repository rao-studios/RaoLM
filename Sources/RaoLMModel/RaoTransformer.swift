//
//  RaoTransformer.swift
//  RaoLMModel
//
//  WHAT: The SmolLM2-shaped decoder: token embedding, pre-norm blocks of grouped-query
//        attention with RoPE and a SwiGLU MLP, a final RMSNorm, and a tied output head.
//  IN:   A RaoLMConfig (Hugging Face Llama keys).
//  OUT:  Logits, plus the two hidden states the provenance index keys on: the residual
//        stream after block `tapLayer`, and the final normed hidden state.
//  PIN:  Module keys are the Hugging Face Llama names (model.embed_tokens.weight,
//        model.layers.N.self_attn.q_proj.weight, …, model.norm.weight), so a checkpoint is
//        a plain Llama checkpoint. The attention body follows Frigate's Llama port
//        (MLXLLM/Models/Llama.swift, MIT) so Frigate's LlamaModel reproduces these logits.
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import RaoLMCore

final class RaoAttention: Module {
    let heads: Int
    let kvHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPELayer

    init(_ config: RaoLMConfig) {
        let headDim = config.headDim
        self.heads = config.numAttentionHeads
        self.kvHeads = config.numKeyValueHeads
        self.scale = pow(Float(headDim), -0.5)
        self._wq.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: config.attentionBias)
        self._wk.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        self._wv.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        self._wo.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: config.attentionBias)
        self.rope = initializeRope(
            dims: headDim, base: config.ropeTheta, traditional: false, scalingConfig: nil,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        var queries = wq(x).reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        var keys = wk(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)
        return wo(output)
    }
}

final class RaoMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ config: RaoLMConfig) {
        self._gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
        self._down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: config.mlpBias)
        self._up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

final class RaoBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: RaoAttention
    @ModuleInfo(key: "mlp") var mlp: RaoMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: RaoLMConfig) {
        self._attention.wrappedValue = RaoAttention(config)
        self._mlp.wrappedValue = RaoMLP(config)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public final class RaoTransformerInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [RaoBlock]
    let norm: RMSNorm

    init(_ config: RaoLMConfig) {
        self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0..<config.numHiddenLayers).map { _ in RaoBlock(config) }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?, tapLayer: Int?) -> (final: MLXArray, tap: MLXArray?) {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)
        var tap: MLXArray?
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
            if i == tapLayer { tap = h }
        }
        return (norm(h), tap)
    }
}

/// What one forward pass produces.
public struct RaoForwardOutput {
    /// [B, T, vocab]
    public let logits: MLXArray
    /// [B, T, hidden] — residual stream after block `tapLayer` (nil when not captured).
    public let tap: MLXArray?
    /// [B, T, hidden] — the final normed hidden state, the input of the tied LM head.
    public let final: MLXArray
}

public final class RaoTransformer: Module, LLMModel, KVCacheDimensionProvider {
    public let config: RaoLMConfig
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let model: RaoTransformerInner
    /// The 0-based block whose output is the mid-layer half of the provenance key.
    public let tapLayer: Int

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: RaoLMConfig, tapLayer: Int? = nil) {
        self.config = config
        self.vocabularySize = config.vocabSize
        self.kvHeads = Array(repeating: config.numKeyValueHeads, count: config.numHiddenLayers)
        self.model = RaoTransformerInner(config)
        self.tapLayer = min(max(tapLayer ?? config.numHiddenLayers / 2, 0), config.numHiddenLayers - 1)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    /// Logits plus the hidden states provenance keys are built from.
    public func forward(_ inputs: MLXArray, cache: [KVCache]? = nil, captureTap: Bool = true) -> RaoForwardOutput {
        let (final, tap) = model(inputs, cache: cache, tapLayer: captureTap ? tapLayer : nil)
        let logits: MLXArray
        if let lmHead {
            logits = lmHead(final)
        } else {
            logits = model.embedTokens.asLinear(final)
        }
        return RaoForwardOutput(logits: logits, tap: tap, final: final)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        forward(inputs, cache: cache, captureTap: false).logits
    }

    public var loraLayers: [Module] { model.layers }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
        if config.tieWordEmbeddings { result["lm_head.weight"] = nil }
        return result
    }
}

/// The provenance key: [√α · tap/‖tap‖, √(1−α) · final/‖final‖], unit norm.
///
/// With tied embeddings the final hidden state at every position that predicts token y is
/// pulled toward the same embedding row, so final-layer neighbours cluster by *next token*.
/// The mid-layer half keeps the context (entity names, local n-grams) that discriminates
/// sources; the final half keeps next-token agreement. The same function runs at index
/// time and at query time.
public enum ProvenanceKey {
    public static func make(tap: MLXArray, final: MLXArray, alpha: Float) -> MLXArray {
        let a = normalized(tap.asType(.float32)) * Float(alpha.squareRoot())
        let b = normalized(final.asType(.float32)) * Float((1 - alpha).squareRoot())
        return concatenated([a, b], axis: -1)
    }

    public static func normalized(_ x: MLXArray) -> MLXArray {
        let norm = MLX.sqrt((x * x).sum(axis: -1, keepDims: true))
        return x / MLX.maximum(norm, MLXArray(Float(1e-6)))
    }

    public static func dimensions(for config: RaoLMConfig) -> Int { 2 * config.hiddenSize }
}
