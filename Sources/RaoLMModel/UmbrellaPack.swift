//
//  UmbrellaPack.swift
//  RaoLMModel
//
//  WHAT: The umbrella's frozen property, which every Thread node mirrors. Always the shared
//        vocabulary (the token embedding and the final norm). A pack made from a pretrained
//        model also holds every block of that model: blocks from `cut` on are the trunk each
//        node runs after its own blocks, blocks before it are where a node's own blocks start,
//        and all of them together are the commons strand the umbrella runs itself. It carries
//        commons text too: anchor snippets with the base model's cut states on them, and a
//        held-out sample.
//  OUT:  vocabulary.safetensors and vocabulary.json (a pack's directory is also a vocabulary's),
//        base.safetensors, anchors.jsonl, anchors.safetensors, heldout.jsonl and pack.json.
//  PIN:  A pack is named by the SHA-256 of what a node must match: the vocabulary's bytes, then
//        the trunk's, key by key in sorted order. With no trunk that is the vocabulary's own
//        fingerprint, so a pack without a base names exactly what the vocabulary did.
//        Installing a pack freezes the vocabulary and the trunk: a node trains blocks 0 ..< cut.
//

import Foundation
import MLX
import MLXNN
import RaoLMCore

/// A piece of commons text a pack carries, tokenized.
public struct PackSnippet: Codable, Sendable, Equatable {
    /// Where it was cut from, e.g. "gutenberg:1342".
    public var source: String
    public var tokens: [Int]

    public init(source: String, tokens: [Int]) {
        self.source = source
        self.tokens = tokens
    }
}

/// One source text of a pack's commons samples.
public struct PackText: Codable, Sendable, Equatable {
    public var source: String
    public var title: String
    /// SHA-256 of the text as used (the book's body, without the distributor's header and footer).
    public var sha256: String
    public var tokens: Int

    public init(source: String, title: String, sha256: String, tokens: Int) {
        self.source = source
        self.title = title
        self.sha256 = sha256
        self.tokens = tokens
    }
}

/// pack.json.
public struct PackInfo: Codable, Sendable, Equatable {
    public var version: Int
    /// A short name, e.g. "smollm2-135m".
    public var name: String
    /// Where the weights came from, e.g. "hub:HuggingFaceTB/SmolLM2-135M@93efa2f0…".
    public var source: String
    /// The base model's shape; its `cut` is the first block of the trunk.
    public var config: RaoLMConfig
    /// The pack's name: SHA-256 of the vocabulary's bytes and the trunk's.
    public var sha256: String
    public var vocabularySHA256: String
    /// SHA-256 of base.safetensors and anchors.safetensors as written.
    public var baseSHA256: String?
    public var anchorsSHA256: String?
    public var anchorCount: Int
    public var heldOutCount: Int
    public var texts: [PackText]
    public var tokenizerSHA256: String
    public var createdAt: Date

    public init(
        name: String, source: String, config: RaoLMConfig, sha256: String, vocabularySHA256: String, anchorCount: Int,
        heldOutCount: Int, texts: [PackText], tokenizerSHA256: String, createdAt: Date = .wholeSecond()
    ) {
        self.version = 1
        self.name = name
        self.source = source
        self.config = config
        self.sha256 = sha256
        self.vocabularySHA256 = vocabularySHA256
        self.baseSHA256 = nil
        self.anchorsSHA256 = nil
        self.anchorCount = anchorCount
        self.heldOutCount = heldOutCount
        self.texts = texts
        self.tokenizerSHA256 = tokenizerSHA256
        self.createdAt = createdAt
    }
}

public enum UmbrellaPackError: Error, CustomStringConvertible {
    case missing(String)
    case fingerprint(expected: String, found: String, what: String)
    case shape(String)
    case noBase

    public var description: String {
        switch self {
        case .missing(let what): return "umbrella pack incomplete: \(what)"
        case .fingerprint(let expected, let found, let what):
            return "the pack's \(what) should hash to \(expected.prefix(12))… but hashes to \(found.prefix(12))…"
        case .shape(let what): return "the pack does not fit the model: \(what)"
        case .noBase: return "the pack holds a vocabulary only, no base model"
        }
    }
}

public final class UmbrellaPack {
    public static let infoFile = "pack.json"
    public static let baseFile = "base.safetensors"
    public static let anchorsFile = "anchors.jsonl"
    public static let anchorStatesFile = "anchors.safetensors"
    public static let heldOutFile = "heldout.jsonl"

    public let vocabulary: VocabularyPack
    /// Nil for a vocabulary alone.
    public private(set) var info: PackInfo?
    /// Every block of the base model under its Hugging Face key, float32; empty for a vocabulary alone.
    public let base: [String: MLXArray]
    public let anchors: [PackSnippet]
    /// [anchors, hidden] float32: the base model's cut state on each anchor, the mean over its positions.
    public let anchorStates: MLXArray?
    public let heldOut: [PackSnippet]

    /// A pack that is a vocabulary alone: the braid as it was before packs.
    public init(vocabulary: VocabularyPack) {
        self.vocabulary = vocabulary
        self.info = nil
        self.base = [:]
        self.anchors = []
        self.anchorStates = nil
        self.heldOut = []
    }

    public init(
        vocabulary: VocabularyPack, info: PackInfo, base: [String: MLXArray], anchors: [PackSnippet], anchorStates: MLXArray?,
        heldOut: [PackSnippet]
    ) {
        self.vocabulary = vocabulary
        self.info = info
        self.base = base
        self.anchors = anchors
        self.anchorStates = anchorStates
        self.heldOut = heldOut
    }

    /// What names the pack: the vocabulary's fingerprint when there is no trunk.
    public var sha256: String { info?.sha256 ?? vocabulary.sha256 }
    public var hasBase: Bool { !base.isEmpty }
    /// The first block of the trunk; nil without one.
    public var cut: Int? {
        guard let info, info.config.hasTrunk, hasBase else { return nil }
        return info.config.cut
    }
    public var hasTrunk: Bool { cut != nil }
    public var name: String { info?.name ?? "vocabulary" }

    /// The trunk's arrays, sorted by key.
    public var trunk: [(key: String, value: MLXArray)] {
        guard let cut else { return [] }
        return base.filter { (RaoTransformer.blockIndex(ofKey: $0.key) ?? -1) >= cut }.sorted { $0.key < $1.key }
            .map { (key: $0.key, value: $0.value) }
    }

    // MARK: - Fingerprints

    /// SHA-256 of the norm's bytes, the embedding's, then each trunk array's key and bytes in key
    /// order (float32, little-endian). Without a trunk this is `VocabularyPack.fingerprint`.
    public static func fingerprint(embedding: MLXArray, norm: MLXArray, trunk: [(key: String, value: MLXArray)]) -> String {
        var parts = [norm.asType(.float32).asData(access: .copy).data, embedding.asType(.float32).asData(access: .copy).data]
        for (key, value) in trunk.sorted(by: { $0.key < $1.key }) {
            parts.append(Data(key.utf8))
            parts.append(value.asType(.float32).asData(access: .copy).data)
        }
        return ContentHash.sha256Hex(parts: parts)
    }

    /// The fingerprint of what a model holds of a pack cut where its config cuts it.
    public static func fingerprint(of model: RaoTransformer) -> String {
        let trunk = model.parameters().flattened()
            .filter { (RaoTransformer.blockIndex(ofKey: $0.0) ?? -1) >= model.cut && model.cut < model.config.numHiddenLayers }
            .map { (key: $0.0, value: $0.1) }
        return fingerprint(embedding: model.model.embedTokens.weight, norm: model.model.norm.weight, trunk: trunk)
    }

    /// Recomputes the pack's name from its arrays.
    public func computedFingerprint() -> String {
        Self.fingerprint(embedding: vocabulary.embedding, norm: vocabulary.norm, trunk: trunk)
    }

    /// Whether `model` holds this pack's vocabulary and trunk.
    public func matches(_ model: RaoTransformer) -> Bool {
        if let cut, model.cut != cut { return false }
        return Self.fingerprint(of: model) == sha256
    }

    // MARK: - Models

    /// Puts the pack into `model` and freezes it there: the vocabulary, and with a base every block
    /// (the node's own blocks start as the base's), the trunk frozen. What is left to train is
    /// blocks 0 ..< cut.
    public func install(into model: RaoTransformer) throws {
        try vocabulary.install(into: model)
        guard hasBase, let info else { return }
        try fits(model, info: info)
        try model.update(parameters: ModuleParameters.unflattened(base.map { ($0.key, $0.value) }), verify: [.shapeMismatch, .noUnusedKeys])
        freeze(model)
        eval(model)
    }

    /// Freezes the vocabulary and the trunk of a model that already holds them.
    public func freeze(_ model: RaoTransformer) {
        Self.freeze(model)
    }

    /// Freezes the embedding, the final norm and every block from the model's cut on.
    public static func freeze(_ model: RaoTransformer) {
        VocabularyPack.freeze(model)
        for i in model.cut..<model.model.layers.count { model.model.layers[i].freeze() }
    }

    /// Whether only blocks 0 ..< cut are trainable.
    public static func isFrozen(_ model: RaoTransformer) -> Bool {
        guard VocabularyPack.isFrozen(model) else { return false }
        return model.trainableParameters().flattened().allSatisfy { (RaoTransformer.blockIndex(ofKey: $0.0) ?? Int.max) < model.cut }
    }

    /// The base model whole, for the umbrella's commons strand.
    public func baseModel(tapLayer: Int? = nil) throws -> RaoTransformer {
        guard hasBase, let info else { throw UmbrellaPackError.noBase }
        let model = RaoTransformer(info.config, tapLayer: tapLayer)
        try vocabulary.install(into: model)
        try model.update(parameters: ModuleParameters.unflattened(base.map { ($0.key, $0.value) }), verify: [.shapeMismatch, .noUnusedKeys])
        eval(model)
        return model
    }

    private func fits(_ model: RaoTransformer, info: PackInfo) throws {
        let a = model.config
        let b = info.config
        guard a.hiddenSize == b.hiddenSize, a.intermediateSize == b.intermediateSize, a.numHiddenLayers == b.numHiddenLayers,
              a.numAttentionHeads == b.numAttentionHeads, a.numKeyValueHeads == b.numKeyValueHeads, a.vocabSize == b.vocabSize,
              a.cut == b.cut
        else {
            throw UmbrellaPackError.shape(
                "the model is \(a.numHiddenLayers) × \(a.hiddenSize) cut at \(a.cut), the pack \(b.numHiddenLayers) × \(b.hiddenSize) cut at \(b.cut)")
        }
    }

    /// Each anchor's cut state under `model`, the mean over the anchor's positions: [anchors, hidden] float32.
    public static func anchorStates(model: RaoTransformer, anchors: [[Int]], batch: Int = 32) -> MLXArray {
        var rows: [MLXArray] = []
        var start = 0
        while start < anchors.count {
            let length = anchors[start].count
            var end = start + 1
            while end < anchors.count, end - start < max(1, batch), anchors[end].count == length { end += 1 }
            if length > 0 {
                let input = MLXArray(anchors[start..<end].flatMap { $0.map { Int32($0) } }, [end - start, length])
                if let cut = model.body(input, captureTap: false, captureCut: true).cut {
                    let pooled = cut.asType(.float32).mean(axis: 1)
                    eval(pooled)
                    rows.append(pooled)
                }
            } else {
                rows.append(zeros([end - start, model.config.hiddenSize]))
            }
            start = end
        }
        guard !rows.isEmpty else { return zeros([0, model.config.hiddenSize]) }
        let states = concatenated(rows, axis: 0)
        eval(states)
        return states
    }

    // MARK: - Disk

    /// Writes the pack; a vocabulary alone writes the vocabulary's two files.
    public func save(to directory: URL) throws {
        try vocabulary.save(to: directory)
        guard var info else { return }
        if hasBase {
            let url = directory.appendingPathComponent(Self.baseFile)
            try MLX.save(arrays: base, metadata: ["format": "raolm-umbrella-base", "sha256": info.sha256], url: url)
            info.baseSHA256 = try ContentHash.sha256Hex(fileAt: url)
        }
        try Self.writeLines(anchors, to: directory.appendingPathComponent(Self.anchorsFile))
        if let anchorStates {
            let url = directory.appendingPathComponent(Self.anchorStatesFile)
            try MLX.save(arrays: ["states": anchorStates], metadata: ["format": "raolm-umbrella-anchors"], url: url)
            info.anchorsSHA256 = try ContentHash.sha256Hex(fileAt: url)
        }
        try Self.writeLines(heldOut, to: directory.appendingPathComponent(Self.heldOutFile))
        info.anchorCount = anchors.count
        info.heldOutCount = heldOut.count
        self.info = info
        try JSONCoding.write(info, to: directory.appendingPathComponent(Self.infoFile))
    }

    public static func exists(at directory: URL) -> Bool {
        VocabularyPack.exists(at: directory)
    }

    /// Loads a pack and refuses one whose files do not hash to what pack.json records, or whose
    /// vocabulary and trunk do not hash to its name.
    public static func load(from directory: URL) throws -> UmbrellaPack {
        let vocabulary = try VocabularyPack.load(from: directory)
        let infoURL = directory.appendingPathComponent(infoFile)
        guard FileManager.default.fileExists(atPath: infoURL.path) else { return UmbrellaPack(vocabulary: vocabulary) }
        let info = try JSONCoding.read(PackInfo.self, from: infoURL)
        guard info.vocabularySHA256 == vocabulary.sha256 else {
            throw UmbrellaPackError.fingerprint(expected: info.vocabularySHA256, found: vocabulary.sha256, what: "vocabulary")
        }
        var base: [String: MLXArray] = [:]
        if let expected = info.baseSHA256 {
            let url = directory.appendingPathComponent(baseFile)
            guard FileManager.default.fileExists(atPath: url.path) else { throw UmbrellaPackError.missing(url.path) }
            let found = try ContentHash.sha256Hex(fileAt: url)
            guard found == expected else { throw UmbrellaPackError.fingerprint(expected: expected, found: found, what: baseFile) }
            base = try loadArrays(url: url).mapValues { $0.asType(.float32) }
        }
        let anchors = (try? JSONCoding.readLines(PackSnippet.self, from: directory.appendingPathComponent(anchorsFile))) ?? []
        var anchorStates: MLXArray?
        if let expected = info.anchorsSHA256 {
            let url = directory.appendingPathComponent(anchorStatesFile)
            let found = try ContentHash.sha256Hex(fileAt: url)
            guard found == expected else { throw UmbrellaPackError.fingerprint(expected: expected, found: found, what: anchorStatesFile) }
            anchorStates = try loadArrays(url: url)["states"]?.asType(.float32)
        }
        let heldOut = (try? JSONCoding.readLines(PackSnippet.self, from: directory.appendingPathComponent(heldOutFile))) ?? []
        let pack = UmbrellaPack(vocabulary: vocabulary, info: info, base: base, anchors: anchors, anchorStates: anchorStates, heldOut: heldOut)
        let found = pack.computedFingerprint()
        guard found == info.sha256 else { throw UmbrellaPackError.fingerprint(expected: info.sha256, found: found, what: "vocabulary and trunk") }
        return pack
    }

    static func writeLines<T: Encodable>(_ values: [T], to url: URL) throws {
        let encoder = JSONCoding.lineEncoder()
        var data = Data()
        for value in values {
            data.append(try encoder.encode(value))
            data.append(0x0A)
        }
        try data.write(to: url, options: .atomic)
    }
}
