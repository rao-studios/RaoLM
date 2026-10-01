//
//  RaoLMConfig.swift
//  RaoLMCore
//
//  WHAT: The model's shape, written exactly as a Hugging Face Llama `config.json`.
//  IN:   Presets (`tiny`, `small`, `smollm2-135m`, `base`) or a decoded config.json.
//  OUT:  The same bytes a RaoLM checkpoint writes beside its weights, so a checkpoint
//        directory loads as a plain Llama model in Frigate, `transformers` and `mlx-lm`.
//  PIN:  Lives in Core (no MLX) so run manifests can embed it. `cut` (key `raolm_cut`) is
//        written only when it is not the last block, so every config without a trunk keeps
//        its bytes; Llama loaders ignore the key. `canon` and `attentionGate` (phase-2 block
//        arms, node blocks only) are written only when on: a checkpoint with either is no
//        longer a plain Llama one.
//

import Foundation

public struct RaoLMConfig: Codable, Sendable, Equatable {
    public var modelType: String
    public var architectures: [String]
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var vocabSize: Int
    public var maxPositionEmbeddings: Int
    public var ropeTheta: Float
    public var rmsNormEps: Float
    public var tieWordEmbeddings: Bool
    public var initializerRange: Float
    public var bosTokenId: Int
    public var eosTokenId: Int
    public var torchDtype: String
    public var attentionBias: Bool
    public var mlpBias: Bool
    public var hiddenAct: String
    /// Blocks `0 ..< cut` are a Thread node's own; blocks `cut ..< numHiddenLayers` are the
    /// umbrella's frozen trunk, which every node runs after its own. The state entering block
    /// `cut` is what the umbrella can read of a node's thought. `numHiddenLayers`: no trunk.
    public var cut: Int
    /// Canon layers in every node block (0 ..< cut): on the attention's input, on the MLP's input,
    /// and inside the MLP on its gate and up projections. Zero-initialised.
    public var canon: Bool = false
    /// A gate on each head's attention output in every node block, 2σ(x·W). Zero-initialised.
    public var attentionGate: Bool = false

    public init(
        hiddenSize: Int, intermediateSize: Int, numHiddenLayers: Int, numAttentionHeads: Int,
        numKeyValueHeads: Int, vocabSize: Int = 49152, maxPositionEmbeddings: Int = 2048,
        ropeTheta: Float = 100_000, rmsNormEps: Float = 1e-5, tieWordEmbeddings: Bool = true,
        initializerRange: Float = 0.041666668, bosTokenId: Int = 0, eosTokenId: Int = 0,
        torchDtype: String = "float32", cut: Int? = nil
    ) {
        self.modelType = "llama"
        self.architectures = ["LlamaForCausalLM"]
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.vocabSize = vocabSize
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTheta = ropeTheta
        self.rmsNormEps = rmsNormEps
        self.tieWordEmbeddings = tieWordEmbeddings
        self.initializerRange = initializerRange
        self.bosTokenId = bosTokenId
        self.eosTokenId = eosTokenId
        self.torchDtype = torchDtype
        self.attentionBias = false
        self.mlpBias = false
        self.hiddenAct = "silu"
        self.cut = cut ?? numHiddenLayers
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case vocabSize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case rmsNormEps = "rms_norm_eps"
        case tieWordEmbeddings = "tie_word_embeddings"
        case initializerRange = "initializer_range"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case torchDtype = "torch_dtype"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case hiddenAct = "hidden_act"
        case cut = "raolm_cut"
        case canon = "raolm_canon"
        case attentionGate = "raolm_attention_gate"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "llama"
        architectures = try c.decodeIfPresent([String].self, forKey: .architectures) ?? ["LlamaForCausalLM"]
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? numAttentionHeads
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 2048
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        initializerRange = try c.decodeIfPresent(Float.self, forKey: .initializerRange) ?? 0.02
        bosTokenId = try c.decodeIfPresent(Int.self, forKey: .bosTokenId) ?? 0
        eosTokenId = try c.decodeIfPresent(Int.self, forKey: .eosTokenId) ?? 0
        torchDtype = try c.decodeIfPresent(String.self, forKey: .torchDtype) ?? "float32"
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        cut = try c.decodeIfPresent(Int.self, forKey: .cut) ?? numHiddenLayers
        canon = try c.decodeIfPresent(Bool.self, forKey: .canon) ?? false
        attentionGate = try c.decodeIfPresent(Bool.self, forKey: .attentionGate) ?? false
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(architectures, forKey: .architectures)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(intermediateSize, forKey: .intermediateSize)
        try c.encode(numHiddenLayers, forKey: .numHiddenLayers)
        try c.encode(numAttentionHeads, forKey: .numAttentionHeads)
        try c.encode(numKeyValueHeads, forKey: .numKeyValueHeads)
        try c.encode(vocabSize, forKey: .vocabSize)
        try c.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try c.encode(ropeTheta, forKey: .ropeTheta)
        try c.encode(rmsNormEps, forKey: .rmsNormEps)
        try c.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try c.encode(initializerRange, forKey: .initializerRange)
        try c.encode(bosTokenId, forKey: .bosTokenId)
        try c.encode(eosTokenId, forKey: .eosTokenId)
        try c.encode(torchDtype, forKey: .torchDtype)
        try c.encode(attentionBias, forKey: .attentionBias)
        try c.encode(mlpBias, forKey: .mlpBias)
        try c.encode(hiddenAct, forKey: .hiddenAct)
        if cut != numHiddenLayers { try c.encode(cut, forKey: .cut) }
        if canon { try c.encode(canon, forKey: .canon) }
        if attentionGate { try c.encode(attentionGate, forKey: .attentionGate) }
    }

    public var headDim: Int { hiddenSize / numAttentionHeads }

    /// Whether blocks above the cut belong to the umbrella.
    public var hasTrunk: Bool { cut < numHiddenLayers }

    /// The block whose output is the provenance key's middle half, by default: halfway through
    /// the node's own blocks.
    public var defaultTapLayer: Int { max(0, min(cut, numHiddenLayers) / 2) }

    /// Trainable parameters, counting the tied embedding once.
    public var parameterCount: Int {
        let embedding = vocabSize * hiddenSize
        let attention = hiddenSize * (numAttentionHeads * headDim)       // q
            + 2 * hiddenSize * (numKeyValueHeads * headDim)               // k, v
            + (numAttentionHeads * headDim) * hiddenSize                  // o
        let mlp = 3 * hiddenSize * intermediateSize
        let norms = 2 * hiddenSize
        let head = tieWordEmbeddings ? 0 : vocabSize * hiddenSize
        return embedding + numHiddenLayers * (attention + mlp + norms) + min(cut, numHiddenLayers) * blockAdditionCount + hiddenSize + head
    }

    /// Trainable parameters of a Thread node: its own blocks, the vocabulary and the trunk frozen.
    public var nodeParameterCount: Int {
        let attention = hiddenSize * (numAttentionHeads * headDim) + 2 * hiddenSize * (numKeyValueHeads * headDim)
            + (numAttentionHeads * headDim) * hiddenSize
        return min(cut, numHiddenLayers) * (attention + 3 * hiddenSize * intermediateSize + 2 * hiddenSize + blockAdditionCount)
    }

    /// Parameters the phase-2 additions give each node block.
    public var blockAdditionCount: Int {
        (canon ? 4 * (2 * hiddenSize + 2 * intermediateSize) : 0) + (attentionGate ? numAttentionHeads * hiddenSize : 0)
    }

    /// Structural problems that would make the model unbuildable.
    public func validate() throws {
        guard hiddenSize > 0, numHiddenLayers > 0, vocabSize > 0, numAttentionHeads > 0, numKeyValueHeads > 0 else {
            throw RaoLMConfigError.invalid("sizes must be positive")
        }
        guard cut > 0, cut <= numHiddenLayers else {
            throw RaoLMConfigError.invalid("raolm_cut \(cut) must lie in 1...\(numHiddenLayers)")
        }
        guard hiddenSize % numAttentionHeads == 0 else {
            throw RaoLMConfigError.invalid("hidden_size \(hiddenSize) is not divisible by num_attention_heads \(numAttentionHeads)")
        }
        guard numAttentionHeads % numKeyValueHeads == 0 else {
            throw RaoLMConfigError.invalid("num_attention_heads \(numAttentionHeads) is not a multiple of num_key_value_heads \(numKeyValueHeads)")
        }
        guard headDim % 2 == 0 else {
            throw RaoLMConfigError.invalid("the head size \(headDim) must be even for rotary positions")
        }
        guard modelType == "llama" else {
            throw RaoLMConfigError.invalid("model_type \(modelType) is not supported (llama only)")
        }
    }

    // MARK: - Presets

    /// ≈17.3M parameters (12.6M of them the tied 49,152 × 256 embedding). Trains in minutes.
    public static let tiny = RaoLMConfig(
        hiddenSize: 256, intermediateSize: 768, numHiddenLayers: 6,
        numAttentionHeads: 4, numKeyValueHeads: 2)

    /// ≈29M parameters.
    public static let small = RaoLMConfig(
        hiddenSize: 384, intermediateSize: 1024, numHiddenLayers: 8,
        numAttentionHeads: 6, numKeyValueHeads: 2)

    /// The real SmolLM2-135M shape (HuggingFaceTB/SmolLM2-135M config.json).
    public static let smolLM2_135M = RaoLMConfig(
        hiddenSize: 576, intermediateSize: 1536, numHiddenLayers: 30,
        numAttentionHeads: 9, numKeyValueHeads: 3, maxPositionEmbeddings: 8192,
        ropeTheta: 100_000, rmsNormEps: 1e-5, tieWordEmbeddings: true,
        initializerRange: 0.041666668, torchDtype: "bfloat16")

    /// SmolLM2-135M cut after block 20: a node trains blocks 0 to 19 (warm-started from
    /// SmolLM2's), and blocks 20 to 29 are the umbrella's frozen trunk.
    public static let base: RaoLMConfig = {
        var config = smolLM2_135M
        config.cut = 20
        return config
    }()

    public static let presetNames = ["tiny", "small", "smollm2-135m", "base"]

    public static func preset(_ name: String) throws -> RaoLMConfig {
        switch name.lowercased() {
        case "tiny": return .tiny
        case "small": return .small
        case "smollm2-135m", "smollm2_135m", "135m": return .smolLM2_135M
        case "base": return .base
        default:
            throw RaoLMConfigError.unknownPreset(name, known: presetNames)
        }
    }
}

public enum RaoLMConfigError: Error, CustomStringConvertible {
    case unknownPreset(String, known: [String])
    case invalid(String)

    public var description: String {
        switch self {
        case .unknownPreset(let name, let known):
            return "unknown model preset '\(name)' (known: \(known.joined(separator: ", ")))"
        case .invalid(let reason):
            return "invalid model config: \(reason)"
        }
    }
}
