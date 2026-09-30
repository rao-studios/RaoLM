//
//  Vocabulary.swift
//  RaoLMModel
//
//  WHAT: The shared vocabulary of a braid: the one token-embedding matrix and final-norm
//        weight that every Thread node embeds with and the umbrella's tied head reads with.
//  IN:   A seed (the matrix is then a constant anyone can regenerate), or a checkpoint whose
//        embedding and norm were trained once and are frozen from here on.
//  OUT:  vocabulary/<hash12>/ vocabulary.safetensors (Hugging Face Llama key names) and
//        vocabulary.json; the fingerprint that names it.
//  PIN:  The seeded matrix is drawn on the CPU from SplitMix64 with integer arithmetic only
//        (four 16-bit draws summed), then each row is scaled to `rowNorm`: no transcendental
//        function, so the same seed gives the same bytes on every machine. The head's scale
//        lives in the norm weight, because a frozen matrix at the from-scratch init range
//        cannot make a token likely — its best logit stays below ln(vocabulary). Installing a
//        vocabulary freezes both modules: a node then trains its blocks and nothing else.
//

import Foundation
import MLX
import MLXNN
import RaoLMCore

public struct VocabularyInfo: Codable, Sendable, Equatable {
    public enum Source: String, Codable, Sendable {
        /// Regenerable from `seed`, `rowNorm` and `headScale`.
        case seeded
        /// Taken from a trained checkpoint (`originSHA256` is its weights' hash).
        case checkpoint
    }

    public var version: Int
    public var source: Source
    public var seed: UInt64?
    /// L2 norm of every row of a seeded matrix.
    public var rowNorm: Float?
    /// Every element of a seeded norm weight: the scale of the head's logits.
    public var headScale: Float?
    public var originSHA256: String?
    public var hiddenSize: Int
    public var vocabSize: Int
    public var rmsNormEps: Float
    public var tokenizerSHA256: String
    /// SHA-256 of the norm weight's bytes followed by the matrix's (float32, little-endian).
    public var sha256: String
    public var createdAt: Date

    public init(
        source: Source, seed: UInt64? = nil, rowNorm: Float? = nil, headScale: Float? = nil,
        originSHA256: String? = nil, hiddenSize: Int, vocabSize: Int, rmsNormEps: Float,
        tokenizerSHA256: String, sha256: String, createdAt: Date = .wholeSecond()
    ) {
        self.version = 1
        self.source = source
        self.seed = seed
        self.rowNorm = rowNorm
        self.headScale = headScale
        self.originSHA256 = originSHA256
        self.hiddenSize = hiddenSize
        self.vocabSize = vocabSize
        self.rmsNormEps = rmsNormEps
        self.tokenizerSHA256 = tokenizerSHA256
        self.sha256 = sha256
        self.createdAt = createdAt
    }
}

public enum VocabularyError: Error, CustomStringConvertible {
    case missing(String)
    case shape(expected: [Int], found: [Int], what: String)
    case fingerprint(expected: String, found: String)
    case mismatch(node: String, umbrella: String)

    public var description: String {
        switch self {
        case .missing(let what):
            return "vocabulary incomplete: \(what)"
        case .shape(let expected, let found, let what):
            return "vocabulary \(what) has shape \(found), the model needs \(expected)"
        case .fingerprint(let expected, let found):
            return "vocabulary.json names \(expected.prefix(12))… but the weights hash to \(found.prefix(12))…"
        case .mismatch(let node, let umbrella):
            return "the node's vocabulary \(node.prefix(12))… is not the umbrella's \(umbrella.prefix(12))…"
        }
    }
}

public final class VocabularyPack {
    public static let weightsFile = "vocabulary.safetensors"
    public static let infoFile = "vocabulary.json"
    public static let embeddingKey = "model.embed_tokens.weight"
    public static let normKey = "model.norm.weight"

    public let info: VocabularyInfo
    /// [vocabulary, hidden] float32.
    public let embedding: MLXArray
    /// [hidden] float32.
    public let norm: MLXArray

    public var sha256: String { info.sha256 }

    init(info: VocabularyInfo, embedding: MLXArray, norm: MLXArray) {
        self.info = info
        self.embedding = embedding
        self.norm = norm
    }

    // MARK: - Making one

    /// Rows of a seeded matrix, row-major: integer draws, each row scaled to `rowNorm`.
    public static func seededRows(vocabSize: Int, hiddenSize: Int, seed: UInt64, rowNorm: Float) -> [Float] {
        var rng = SplitMix64.derived(seed: seed, stream: 0x0C0D_E800)
        var values = [Float](repeating: 0, count: vocabSize * hiddenSize)
        values.withUnsafeMutableBufferPointer { buffer in
            for row in 0..<vocabSize {
                let base = row * hiddenSize
                var squares = 0.0
                for column in 0..<hiddenSize {
                    let bits = rng.next()
                    // Irwin–Hall over four 16-bit draws, centred: symmetric and exact.
                    let sum = Int(bits & 0xFFFF) + Int((bits >> 16) & 0xFFFF) + Int((bits >> 32) & 0xFFFF) + Int(bits >> 48)
                    let value = Double(sum - 131_070)
                    squares += value * value
                    buffer[base + column] = Float(value)
                }
                let scale = squares > 0 ? Double(rowNorm) / squares.squareRoot() : 0
                for column in 0..<hiddenSize {
                    buffer[base + column] = Float(Double(buffer[base + column]) * scale)
                }
            }
        }
        return values
    }

    /// A constant vocabulary: unit-direction rows of length `rowNorm`, norm weight `headScale`.
    public static func seeded(
        config: RaoLMConfig, tokenizerSHA256: String, seed: UInt64, rowNorm: Float = 1, headScale: Float = 1.5
    ) -> VocabularyPack {
        let rows = seededRows(vocabSize: config.vocabSize, hiddenSize: config.hiddenSize, seed: seed, rowNorm: rowNorm)
        let embedding = MLXArray(rows, [config.vocabSize, config.hiddenSize])
        let norm = MLXArray([Float](repeating: headScale, count: config.hiddenSize))
        eval(embedding, norm)
        let info = VocabularyInfo(
            source: .seeded, seed: seed, rowNorm: rowNorm, headScale: headScale, hiddenSize: config.hiddenSize,
            vocabSize: config.vocabSize, rmsNormEps: config.rmsNormEps, tokenizerSHA256: tokenizerSHA256,
            sha256: fingerprint(embedding: embedding, norm: norm))
        return VocabularyPack(info: info, embedding: embedding, norm: norm)
    }

    /// The embedding and final norm a full model learned, frozen as a vocabulary.
    public static func from(model: RaoTransformer, tokenizerSHA256: String, originSHA256: String?) -> VocabularyPack {
        let embedding = model.model.embedTokens.weight.asType(.float32)
        let norm = model.model.norm.weight.asType(.float32)
        eval(embedding, norm)
        let info = VocabularyInfo(
            source: .checkpoint, originSHA256: originSHA256, hiddenSize: model.config.hiddenSize,
            vocabSize: model.config.vocabSize, rmsNormEps: model.config.rmsNormEps, tokenizerSHA256: tokenizerSHA256,
            sha256: fingerprint(embedding: embedding, norm: norm))
        return VocabularyPack(info: info, embedding: embedding, norm: norm)
    }

    public static func from(checkpoint directory: URL, tokenizerSHA256: String) throws -> VocabularyPack {
        let model = try Checkpoint.load(from: directory)
        return from(model: model, tokenizerSHA256: tokenizerSHA256, originSHA256: try Checkpoint.weightsSHA256(directory))
    }

    // MARK: - Fingerprint

    public static func fingerprint(embedding: MLXArray, norm: MLXArray) -> String {
        let normBytes = norm.asType(.float32).asData(access: .copy).data
        let embeddingBytes = embedding.asType(.float32).asData(access: .copy).data
        return ContentHash.sha256Hex(parts: [normBytes, embeddingBytes])
    }

    /// The fingerprint of the vocabulary a model holds, whatever it was loaded from.
    public static func fingerprint(of model: RaoTransformer) -> String {
        fingerprint(embedding: model.model.embedTokens.weight, norm: model.model.norm.weight)
    }

    // MARK: - Disk

    /// Writes both files and returns the fingerprint.
    @discardableResult
    public func save(to directory: URL) throws -> String {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try MLX.save(
            arrays: [Self.embeddingKey: embedding, Self.normKey: norm],
            metadata: ["format": "raolm-vocabulary", "sha256": info.sha256],
            url: directory.appendingPathComponent(Self.weightsFile))
        try JSONCoding.write(info, to: directory.appendingPathComponent(Self.infoFile))
        return info.sha256
    }

    public static func exists(at directory: URL) -> Bool {
        FileManager.default.fileExists(atPath: directory.appendingPathComponent(infoFile).path)
            && FileManager.default.fileExists(atPath: directory.appendingPathComponent(weightsFile).path)
    }

    /// Loads a vocabulary and refuses one whose weights do not hash to its name.
    public static func load(from directory: URL) throws -> VocabularyPack {
        let infoURL = directory.appendingPathComponent(infoFile)
        guard FileManager.default.fileExists(atPath: infoURL.path) else { throw VocabularyError.missing(infoURL.path) }
        let info = try JSONCoding.read(VocabularyInfo.self, from: infoURL)
        let arrays = try loadArrays(url: directory.appendingPathComponent(weightsFile))
        guard let embedding = arrays[embeddingKey]?.asType(.float32) else { throw VocabularyError.missing(embeddingKey) }
        guard let norm = arrays[normKey]?.asType(.float32) else { throw VocabularyError.missing(normKey) }
        guard embedding.shape == [info.vocabSize, info.hiddenSize] else {
            throw VocabularyError.shape(expected: [info.vocabSize, info.hiddenSize], found: embedding.shape, what: embeddingKey)
        }
        guard norm.shape == [info.hiddenSize] else {
            throw VocabularyError.shape(expected: [info.hiddenSize], found: norm.shape, what: normKey)
        }
        eval(embedding, norm)
        let found = fingerprint(embedding: embedding, norm: norm)
        guard found == info.sha256 else { throw VocabularyError.fingerprint(expected: info.sha256, found: found) }
        return VocabularyPack(info: info, embedding: embedding, norm: norm)
    }

    // MARK: - Models

    /// Puts this vocabulary into `model` and freezes it there: what is left to train is the blocks.
    public func install(into model: RaoTransformer) throws {
        let expected = [model.config.vocabSize, model.config.hiddenSize]
        guard embedding.shape == expected else {
            throw VocabularyError.shape(expected: expected, found: embedding.shape, what: Self.embeddingKey)
        }
        try model.update(
            parameters: ModuleParameters.unflattened([(Self.embeddingKey, embedding), (Self.normKey, norm)]),
            verify: [.shapeMismatch, .noUnusedKeys])
        Self.freeze(model)
        eval(model)
    }

    /// Freezes the embedding and the final norm of a model that already holds a vocabulary.
    public static func freeze(_ model: RaoTransformer) {
        model.model.embedTokens.freeze()
        model.model.norm.freeze()
    }

    public static func isFrozen(_ model: RaoTransformer) -> Bool {
        let trainable = Set(model.trainableParameters().flattened().map(\.0))
        return !trainable.contains(embeddingKey) && !trainable.contains(normKey)
    }
}

/// The umbrella's half of the model: the final norm and the tied head, over any node's `last`.
public final class UmbrellaHead {
    public let vocabularySHA256: String
    public let vocabSize: Int
    public let hiddenSize: Int
    private let embedding: MLXArray
    private let norm: MLXArray
    private let eps: Float

    public init(vocabulary: VocabularyPack) {
        vocabularySHA256 = vocabulary.sha256
        vocabSize = vocabulary.info.vocabSize
        hiddenSize = vocabulary.info.hiddenSize
        embedding = vocabulary.embedding
        norm = vocabulary.norm
        eps = vocabulary.info.rmsNormEps
    }

    /// [.., hidden] → [.., hidden]: the final normed hidden state.
    public func final(_ last: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(last, weight: norm, eps: eps)
    }

    /// [.., hidden] → [.., vocabulary].
    public func logits(_ last: MLXArray) -> MLXArray {
        matmul(final(last), embedding.T)
    }

    /// One position's logits from the floats a node sent, in the [1, 1, hidden] shape a model's
    /// own cached decode step has.
    public func logits(hidden: [Float]) -> [Float] {
        let array = logits(MLXArray(hidden, [1, 1, hidden.count])).asType(.float32)
        eval(array)
        return array.asArray(Float.self)
    }

    /// Several positions at once ([positions][hidden]), as a model's prefill computes them.
    public func logits(hiddens: [[Float]]) -> [[Float]] {
        rows(logitsArray(hiddens: hiddens))
    }

    /// The same logits kept on the GPU, [positions, vocabulary] in float32 and evaluated, for
    /// statistics over whole distributions; `rows` reads them as floats.
    public func logitsArray(hiddens: [[Float]]) -> MLXArray {
        guard let width = hiddens.first?.count, !hiddens.isEmpty else { return zeros([0, vocabSize]) }
        let array = logits(MLXArray(hiddens.flatMap { $0 }, [1, hiddens.count, width])).asType(.float32)[0]
        eval(array)
        return array
    }

    public func rows(_ array: MLXArray) -> [[Float]] {
        let count = array.dim(0)
        guard count > 0 else { return [] }
        let flat = array.asArray(Float.self)
        return (0..<count).map { Array(flat[($0 * vocabSize)..<(($0 + 1) * vocabSize)]) }
    }
}
