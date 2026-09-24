//
//  RaoLMConfig.swift
//  RaoLMCore
//
//  WHAT: The model's shape, written exactly as a Hugging Face Llama `config.json`.
//  IN:   Presets (`tiny`, `small`, `smollm2-135m`) or a decoded config.json.
//  OUT:  The same bytes a RaoLM checkpoint writes beside its weights, so a checkpoint
//        directory loads as a plain Llama model in Frigate, `transformers` and `mlx-lm`.
//  PIN:  Lives in Core (no MLX) so run manifests can embed it.
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

    public init(
        hiddenSize: Int, intermediateSize: Int, numHiddenLayers: Int, numAttentionHeads: Int,
        numKeyValueHeads: Int, vocabSize: Int = 49152, maxPositionEmbeddings: Int = 2048,
        ropeTheta: Float = 100_000, rmsNormEps: Float = 1e-5, tieWordEmbeddings: Bool = true,
        initializerRange: Float = 0.041666668, bosTokenId: Int = 0, eosTokenId: Int = 0,
        torchDtype: String = "float32"
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
    }

    public var headDim: Int { hiddenSize / numAttentionHeads }

    /// Trainable parameters, counting the tied embedding once.
    public var parameterCount: Int {
        let embedding = vocabSize * hiddenSize
        let attention = hiddenSize * (numAttentionHeads * headDim)       // q
            + 2 * hiddenSize * (numKeyValueHeads * headDim)               // k, v
            + (numAttentionHeads * headDim) * hiddenSize                  // o
        let mlp = 3 * hiddenSize * intermediateSize
        let norms = 2 * hiddenSize
        let head = tieWordEmbeddings ? 0 : vocabSize * hiddenSize
        return embedding + numHiddenLayers * (attention + mlp + norms) + hiddenSize + head
    }

    /// Structural problems that would make the model unbuildable.
    public func validate() throws {
        guard hiddenSize > 0, numHiddenLayers > 0, vocabSize > 0 else {
            throw RaoLMConfigError.invalid("sizes must be positive")
        }
        guard hiddenSize % numAttentionHeads == 0 else {
            throw RaoLMConfigError.invalid("hidden_size \(hiddenSize) is not divisible by num_attention_heads \(numAttentionHeads)")
        }
        guard numAttentionHeads % numKeyValueHeads == 0 else {
            throw RaoLMConfigError.invalid("num_attention_heads \(numAttentionHeads) is not a multiple of num_key_value_heads \(numKeyValueHeads)")
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

    public static let presetNames = ["tiny", "small", "smollm2-135m"]

    public static func preset(_ name: String) throws -> RaoLMConfig {
        switch name.lowercased() {
        case "tiny": return .tiny
        case "small": return .small
        case "smollm2-135m", "smollm2_135m", "135m": return .smolLM2_135M
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
