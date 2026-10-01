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
//        The model splits where a braid cuts it: `body` is the token embedding through the
//        last block (what a Thread node hosts), `head` is the final norm and the tied head
//        (what the umbrella hosts). `forward` is `head(body(x))`, never a second code path.
//        Within the body, blocks from `config.cut` on are the umbrella's frozen trunk: a node
//        runs them after its own, and the state entering the first of them is the cut state,
//        what the umbrella can read of a node's thought. With no trunk the cut state is `last`.
//        The phase-2 block additions (`config.canon`, `config.attentionGate`) live in node blocks
//        only and start at zero, so a block that gains them computes what it did before. Canon's
//        decoding state (the last three inputs of each Canon layer) is kept in cache entries after
//        the per-block ones, so every block's own cache stays at its index.
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import RaoLMCore

/// A Canon layer (Allen-Zhu, Physics of Language Models 4.1): x + Σₖ wₖ ⊙ x₍ₜ₋ₖ₎ for k = 0…3, a
/// causal depthwise convolution of width 4 per channel with a residual. Zero-initialised. Written
/// as four shifted slices, so it is exact and its decoding state is the last three inputs.
final class CanonLayer: Module {
    let weight: MLXArray

    init(dimensions: Int) {
        self.weight = MLXArray.zeros([4, dimensions], type: Float.self)
    }

    /// `history`: the three inputs before `x` (zeros when nil). `sameDocument[k − 1]`: [B, L, 1], 1
    /// where position t − k belongs to t's document. Returns the output and the next history.
    func callAsFunction(_ x: MLXArray, history: MLXArray?, sameDocument: [MLXArray]?) -> (output: MLXArray, history: MLXArray) {
        let (B, L, D) = (x.dim(0), x.dim(1), x.dim(2))
        let full = concatenated([history ?? MLXArray.zeros([B, 3, D], dtype: x.dtype), x], axis: 1)
        var y = x * weight[0]
        for k in 1 ... 3 {
            var shifted = full[0..., (3 - k) ..< (3 - k + L)]
            if let sameDocument { shifted = shifted * sameDocument[k - 1] }
            y = y + shifted * weight[k]
        }
        return (x + y, full[0..., L ..< (L + 3)])
    }
}

/// A gate on each head's attention output (Qiu et al., 2025): 2σ(x·W), one per head. Zero-initialised,
/// so every head passes through unchanged until training moves it.
final class HeadGate: Module {
    let weight: MLXArray

    init(hidden: Int, heads: Int) {
        self.weight = MLXArray.zeros([heads, hidden], type: Float.self)
    }

    /// [B, L, heads]
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        2 * sigmoid(matmul(x, weight.T))
    }
}

final class RaoAttention: Module {
    let heads: Int
    let kvHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "gate_proj") var gate: HeadGate?

    let rope: RoPELayer

    init(_ config: RaoLMConfig, gated: Bool = false) {
        let headDim = config.headDim
        self.heads = config.numAttentionHeads
        self.kvHeads = config.numKeyValueHeads
        self.scale = pow(Float(headDim), -0.5)
        self._wq.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: config.attentionBias)
        self._wk.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        self._wv.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        self._wo.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: config.attentionBias)
        self._gate.wrappedValue = gated ? HeadGate(hidden: config.hiddenSize, heads: config.numAttentionHeads) : nil
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

        var output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        if let gate { output = output * gate(x).expandedDimensions(axis: -1) }
        return wo(output.reshaped(B, L, -1))
    }
}

final class RaoMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "canon_d") var canon: CanonLayer?

    init(_ config: RaoLMConfig, canon: Bool = false) {
        self._gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
        self._down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: config.mlpBias)
        self._up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
        self._canon.wrappedValue = canon ? CanonLayer(dimensions: 2 * config.intermediateSize) : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        callAsFunction(x, history: nil, sameDocument: nil).output
    }

    /// With Canon-D, the gate and up projections pass through a Canon layer together.
    func callAsFunction(_ x: MLXArray, history: MLXArray?, sameDocument: [MLXArray]?) -> (output: MLXArray, history: MLXArray?) {
        guard let canon else { return (down(silu(gate(x)) * up(x)), nil) }
        let (mixed, next) = canon(concatenated([gate(x), up(x)], axis: -1), history: history, sameDocument: sameDocument)
        let parts = split(mixed, parts: 2, axis: -1)
        return (down(silu(parts[0]) * parts[1]), next)
    }
}

final class RaoBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: RaoAttention
    @ModuleInfo(key: "mlp") var mlp: RaoMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "canon_a") var canonA: CanonLayer?
    @ModuleInfo(key: "canon_c") var canonC: CanonLayer?

    /// `own`: a node block, which takes the config's block additions; the trunk's never do.
    init(_ config: RaoLMConfig, own: Bool = false) {
        self._attention.wrappedValue = RaoAttention(config, gated: own && config.attentionGate)
        self._mlp.wrappedValue = RaoMLP(config, canon: own && config.canon)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._canonA.wrappedValue = own && config.canon ? CanonLayer(dimensions: config.hiddenSize) : nil
        self._canonC.wrappedValue = own && config.canon ? CanonLayer(dimensions: config.hiddenSize) : nil
    }

    /// `state`: the block's Canon state while decoding (A, C and D's last three inputs).
    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?, state: ArraysCache? = nil,
        sameDocument: [MLXArray]? = nil
    ) -> MLXArray {
        var a = inputLayerNorm(x)
        if let canonA {
            let (output, next) = canonA(a, history: state?[0], sameDocument: sameDocument)
            a = output
            state?[0] = next
        }
        let h = x + attention(a, mask: mask, cache: cache)
        var c = postAttentionLayerNorm(h)
        if let canonC {
            let (output, next) = canonC(c, history: state?[1], sameDocument: sameDocument)
            c = output
            state?[1] = next
        }
        let (output, next) = mlp(c, history: state?[2], sameDocument: sameDocument)
        if let next { state?[2] = next }
        return h + output
    }
}

public final class RaoTransformerInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [RaoBlock]
    let norm: RMSNorm
    /// Node blocks carry Canon layers, whose decoding state follows the per-block caches.
    let canon: Bool

    init(_ config: RaoLMConfig) {
        self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        let cut = min(max(config.cut, 1), config.numHiddenLayers)
        self.layers = (0..<config.numHiddenLayers).map { RaoBlock(config, own: $0 < cut) }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.canon = config.canon
    }

    /// Block `i`'s Canon state in a decoding cache, if it has one.
    func canonState(_ cache: [KVCache]?, _ i: Int) -> ArraysCache? {
        guard canon, let cache, cache.count > layers.count + i else { return nil }
        return cache[layers.count + i] as? ArraysCache
    }

    /// For each tap k = 1…3 of a Canon layer, [B, T, 1]: 1 where position t − k may be read from
    /// t, the k-th diagonal below the document mask's own (nil without a document mask).
    func sameDocument(_ attention: MLXArray?, dtype: DType) -> [MLXArray]? {
        guard canon, let attention else { return nil }
        let square = attention[0..., 0]
        let (B, T) = (square.dim(0), square.dim(1))
        return (1 ... 3).map { k in
            guard T > k else { return MLXArray.zeros([B, T, 1], dtype: dtype) }
            let below = diagonal(square, offset: -k, axis1: 1, axis2: 2).asType(dtype)
            return concatenated([MLXArray.zeros([B, k], dtype: dtype), below], axis: 1).expandedDimensions(axis: -1)
        }
    }

    /// The token embedding through the last block: the residual stream the final norm reads.
    /// `cutLayer`: capture the state entering that block (`layers.count` captures `last`).
    /// `attention`: a [B, 1, T, T] boolean mask (true: may attend) in place of the causal one.
    func body(_ inputs: MLXArray, cache: [KVCache]?, tapLayer: Int?, cutLayer: Int? = nil, attention: MLXArray? = nil)
        -> (last: MLXArray, tap: MLXArray?, cut: MLXArray?) {
        var h = embedTokens(inputs)
        let mask: MLXFast.ScaledDotProductAttentionMaskMode = attention.map { .array($0) } ?? createAttentionMask(h: h, cache: cache?.first)
        let sameDocument = sameDocument(attention, dtype: h.dtype)
        var tap: MLXArray?
        var cut: MLXArray?
        for (i, layer) in layers.enumerated() {
            if i == cutLayer { cut = h }
            h = layer(h, mask: mask, cache: cache?[i], state: canonState(cache, i), sameDocument: sameDocument)
            if i == tapLayer { tap = h }
        }
        if cutLayer == layers.count { cut = h }
        return (h, tap, cut)
    }

    /// Blocks `from ..< layers.count` over a state that entered block `from` (the trunk over a
    /// cut state), with the mask the whole body would use.
    func trunk(_ h: MLXArray, from: Int, cache: [KVCache]?) -> MLXArray {
        var h = h
        let mask = createAttentionMask(h: h, cache: cache.flatMap { $0.indices.contains(from) ? $0[from] : nil })
        for i in from..<layers.count {
            h = layers[i](h, mask: mask, cache: cache?[i])
        }
        return h
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?, tapLayer: Int?) -> (final: MLXArray, tap: MLXArray?) {
        let (last, tap, _) = body(inputs, cache: cache, tapLayer: tapLayer)
        return (norm(last), tap)
    }
}

/// What the headless half produces: everything a Thread node computes for a position.
public struct RaoBodyOutput {
    /// [B, T, hidden] — residual stream after the last block, before the final norm.
    public let last: MLXArray
    /// [B, T, hidden] — residual stream after block `tapLayer` (nil when not captured).
    public let tap: MLXArray?
    /// [B, T, hidden] — the state entering the trunk (`last` when there is none); nil when not captured.
    public let cut: MLXArray?

    public init(last: MLXArray, tap: MLXArray?, cut: MLXArray? = nil) {
        self.last = last
        self.tap = tap
        self.cut = cut
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
    /// [B, T, hidden] — residual stream after the last block: what a node sends the umbrella.
    public let last: MLXArray
}

public final class RaoTransformer: Module, LLMModel, KVCacheDimensionProvider {
    public let config: RaoLMConfig
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let model: RaoTransformerInner
    /// The 0-based block whose output is the mid-layer half of the provenance key.
    public let tapLayer: Int
    /// The first block of the umbrella's trunk (`config.numHiddenLayers`: none).
    public let cut: Int

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: RaoLMConfig, tapLayer: Int? = nil) {
        self.config = config
        self.vocabularySize = config.vocabSize
        self.kvHeads = Array(repeating: config.numKeyValueHeads, count: config.numHiddenLayers)
        self.model = RaoTransformerInner(config)
        self.tapLayer = min(max(tapLayer ?? config.defaultTapLayer, 0), config.numHiddenLayers - 1)
        self.cut = min(max(config.cut, 1), config.numHiddenLayers)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    /// The headless half: token embedding through the last block, the trunk included.
    /// `captureCut` also keeps the state entering the trunk. `attention` replaces the causal mask
    /// (training keeps attention inside a document with it).
    public func body(
        _ inputs: MLXArray, cache: [KVCache]? = nil, captureTap: Bool = true, captureCut: Bool = false, attention: MLXArray? = nil
    ) -> RaoBodyOutput {
        let (last, tap, cut) = model.body(
            inputs, cache: cache, tapLayer: captureTap ? tapLayer : nil, cutLayer: captureCut ? self.cut : nil, attention: attention)
        return RaoBodyOutput(last: last, tap: tap, cut: cut)
    }

    /// The trunk alone over a cut state: blocks `cut ..< numHiddenLayers` (the identity with no trunk).
    public func trunk(_ cutState: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        cut < config.numHiddenLayers ? model.trunk(cutState, from: cut, cache: cache) : cutState
    }

    /// The 0-based block index a parameter key names ("model.layers.N.…"), if any.
    public static func blockIndex(ofKey key: String) -> Int? {
        guard key.hasPrefix("model.layers.") else { return nil }
        let parts = key.split(separator: ".")
        return parts.count > 2 ? Int(parts[2]) : nil
    }

    /// The final norm alone: what a node applies to `last` to build a provenance key.
    public func normed(_ last: MLXArray) -> MLXArray { model.norm(last) }

    /// The other half: the final norm and the tied head, over a body's `last`.
    public func head(_ last: MLXArray) -> (final: MLXArray, logits: MLXArray) {
        let final = model.norm(last)
        let logits: MLXArray
        if let lmHead {
            logits = lmHead(final)
        } else {
            logits = model.embedTokens.asLinear(final)
        }
        return (final, logits)
    }

    /// Logits plus the hidden states provenance keys are built from.
    public func forward(_ inputs: MLXArray, cache: [KVCache]? = nil, captureTap: Bool = true, attention: MLXArray? = nil) -> RaoForwardOutput {
        let body = body(inputs, cache: cache, captureTap: captureTap, attention: attention)
        let (final, logits) = head(body.last)
        return RaoForwardOutput(logits: logits, tap: body.tap, final: final, last: body.last)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        forward(inputs, cache: cache, captureTap: false).logits
    }

    /// One key/value cache per block, then with Canon layers one state per node block (the last
    /// three inputs of each of its Canon layers). Every block's own cache stays at its index, so
    /// code that reads the first block's offset still finds a plain cache.
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        var caches: [KVCache] = []
        for _ in 0 ..< kvHeads.count {
            if let maxKVSize = parameters?.maxKVSize {
                caches.append(RotatingKVCache(maxSize: maxKVSize, keep: 4))
            } else {
                caches.append(KVCacheSimple())
            }
        }
        if config.canon {
            for _ in 0 ..< cut { caches.append(ArraysCache(size: 3)) }
        }
        return caches
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
