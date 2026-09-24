//
//  ProvenanceIndexer.swift
//  RaoLMTraining
//
//  WHAT: Writes a provenance index from an eval pass: for every corpus transition, the key
//        (mid-layer ⊕ final hidden, unit norm), the token that followed, where the context
//        and the following token sit in the corpus, and that transition's loss and entropy.
//  OUT:  provenance/epoch-<n>/ index.safetensors, index.json, partitions.json, shared-ngrams.bin
//  PIN:  The index is bound to the checkpoint it was built with (index.json records its
//        SHA-256); generation refuses a mismatched pair.
//

import Foundation
import MLX
import RaoLMCore
import RaoLMModel

public struct IndexInfo: Codable, Sendable, Equatable {
    public var version: Int
    public var epoch: Int
    public var tapLayer: Int
    public var alpha: Float
    public var keyDims: Int
    public var count: Int
    public var checkpointSHA256: String
    public var corpusHash: String
    public var tokenizerSHA256: String
    public var threadID: String?
    public var evalLoss: Float
    public var evalMemorisedFraction: Float
    /// nil: keys are stored unprojected.
    public var projection: String?
    public var defaultTau: Float
    public var defaultK: Int
    public var createdAt: Date

    public init(
        epoch: Int, tapLayer: Int, alpha: Float, keyDims: Int, count: Int, checkpointSHA256: String,
        corpusHash: String, tokenizerSHA256: String, threadID: String?, evalLoss: Float,
        evalMemorisedFraction: Float, defaultTau: Float = 0.05, defaultK: Int = 16
    ) {
        self.version = 1
        self.epoch = epoch
        self.tapLayer = tapLayer
        self.alpha = alpha
        self.keyDims = keyDims
        self.count = count
        self.checkpointSHA256 = checkpointSHA256
        self.corpusHash = corpusHash
        self.tokenizerSHA256 = tokenizerSHA256
        self.threadID = threadID
        self.evalLoss = evalLoss
        self.evalMemorisedFraction = evalMemorisedFraction
        self.projection = nil
        self.defaultTau = defaultTau
        self.defaultK = defaultK
        self.createdAt = Date()
    }
}

public enum ProvenanceIndexFiles {
    public static let arrays = "index.safetensors"
    public static let info = "index.json"
    public static let partitions = "partitions.json"
    public static let sharedNgrams = "shared-ngrams.bin"
}

public enum ProvenanceIndexerError: Error, CustomStringConvertible {
    case noKeys

    public var description: String { "the eval pass did not capture provenance keys" }
}

public enum ProvenanceIndexer {

    /// Writes the index and returns the SHA-256 of index.safetensors.
    @discardableResult
    public static func write(
        eval: EvalResult, corpus: TokenizedCorpus, info: IndexInfo, memorisedAtEpoch: [Int: Int], to directory: URL
    ) throws -> String {
        guard let keys = eval.keys else { throw ProvenanceIndexerError.noKeys }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var values: [Int32] = []
        var keyRow: [Int32] = []
        var keyOffset: [Int32] = []
        var valueRow: [Int32] = []
        var valueOffset: [Int32] = []
        var loss: [Float] = []
        var entropy: [Float] = []
        for i in 0..<eval.count where eval.indexable[i] {
            values.append(eval.value[i])
            keyRow.append(eval.keyRow[i])
            keyOffset.append(eval.keyOffset[i])
            valueRow.append(eval.valueRow[i])
            valueOffset.append(eval.valueOffset[i])
            loss.append(eval.loss[i])
            entropy.append(eval.entropy[i])
        }
        precondition(values.count == keys.dim(0), "index keys and entries disagree")
        let url = directory.appendingPathComponent(ProvenanceIndexFiles.arrays)
        try MLX.save(
            arrays: [
                "keys": keys,
                "values": MLXArray(values),
                "key_row": MLXArray(keyRow),
                "key_offset": MLXArray(keyOffset),
                "value_row": MLXArray(valueRow),
                "value_offset": MLXArray(valueOffset),
                "loss": MLXArray(loss),
                "entropy": MLXArray(entropy),
            ],
            metadata: ["format": "raolm-provenance-index", "epoch": String(info.epoch)],
            url: url)
        var finalInfo = info
        finalInfo.count = values.count
        finalInfo.keyDims = keys.dim(1)
        try JSONCoding.write(finalInfo, to: directory.appendingPathComponent(ProvenanceIndexFiles.info))
        try JSONCoding.write(
            corpus.partitionRefs(memorisedAtEpoch: memorisedAtEpoch),
            to: directory.appendingPathComponent(ProvenanceIndexFiles.partitions))
        var ngrams = corpus.sharedNgrams()
        let data = ngrams.withUnsafeMutableBytes { Data($0) }
        try data.write(to: directory.appendingPathComponent(ProvenanceIndexFiles.sharedNgrams), options: .atomic)
        return try ContentHash.sha256Hex(fileAt: url)
    }

    public static func readSharedNgrams(_ directory: URL) throws -> Set<UInt64> {
        let data = try Data(contentsOf: directory.appendingPathComponent(ProvenanceIndexFiles.sharedNgrams))
        var values = [UInt64](repeating: 0, count: data.count / MemoryLayout<UInt64>.size)
        _ = values.withUnsafeMutableBytes { data.copyBytes(to: $0) }
        return Set(values)
    }
}
