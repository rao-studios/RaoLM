//
//  ProvenanceIndex.swift
//  RaoLMProvenance
//
//  WHAT: A loaded provenance index: unit-norm keys on the GPU, the corpus positions and
//        next tokens they map to on the CPU, and the partition table.
//  PIN:  Retrieval is exact cosine search (one matmul). `MLX.top` returns values only, so
//        candidates come from argPartition and are sorted on the CPU by (score desc, entry
//        asc): ties at the k-th boundary would otherwise be undefined.
//

import Foundation
import MLX
import RaoLMCore
import RaoLMModel
import RaoLMTraining

public final class ProvenanceIndex {
    public let directory: URL
    public let info: IndexInfo
    public let partitions: [PartitionRef]
    public let partitionsByRow: [Int: PartitionRef]
    public let sharedNgrams: Set<UInt64>
    public let sha256: String

    /// [N, D] float32, unit norm.
    public let keys: MLXArray
    public let values: [Int32]
    public let keyRow: [Int32]
    public let keyOffset: [Int32]
    public let valueRow: [Int32]
    public let valueOffset: [Int32]
    public let loss: [Float]
    public let entropy: [Float]

    public var count: Int { values.count }

    public init(directory: URL) throws {
        self.directory = directory
        let arraysURL = directory.appendingPathComponent(ProvenanceIndexFiles.arrays)
        self.info = try JSONCoding.read(IndexInfo.self, from: directory.appendingPathComponent(ProvenanceIndexFiles.info))
        let partitions = try JSONCoding.read([PartitionRef].self, from: directory.appendingPathComponent(ProvenanceIndexFiles.partitions))
        self.partitions = partitions
        self.partitionsByRow = Dictionary(uniqueKeysWithValues: partitions.map { ($0.row, $0) })
        self.sharedNgrams = (try? ProvenanceIndexer.readSharedNgrams(directory)) ?? []
        self.sha256 = try ContentHash.sha256Hex(fileAt: arraysURL)

        let arrays = try loadArrays(url: arraysURL)
        func require(_ name: String) throws -> MLXArray {
            guard let array = arrays[name] else { throw ProvenanceError.corruptIndex("missing array '\(name)'") }
            return array
        }
        let keys = ProvenanceKey.normalized(try require("keys").asType(.float32))
        eval(keys)
        self.keys = keys
        self.values = try require("values").asType(.int32).asArray(Int32.self)
        self.keyRow = try require("key_row").asType(.int32).asArray(Int32.self)
        self.keyOffset = try require("key_offset").asType(.int32).asArray(Int32.self)
        self.valueRow = try require("value_row").asType(.int32).asArray(Int32.self)
        self.valueOffset = try require("value_offset").asType(.int32).asArray(Int32.self)
        self.loss = try require("loss").asType(.float32).asArray(Float.self)
        self.entropy = try require("entropy").asType(.float32).asArray(Float.self)
        guard keys.dim(0) == values.count else {
            throw ProvenanceError.corruptIndex("\(keys.dim(0)) keys for \(values.count) entries")
        }
    }

    /// The `k` nearest entries to a unit-norm key `query` ([D]), by cosine.
    public func query(_ query: MLXArray, k: Int) -> [(entry: Int, score: Float)] {
        let n = count
        guard n > 0, k > 0 else { return [] }
        let scores = matmul(keys, query.reshaped(-1, 1)).reshaped(-1)
        let m = min(k + 8, n)
        let candidates: MLXArray
        if m >= n {
            candidates = argSort(-scores)
        } else {
            candidates = argPartition(-scores, kth: m - 1)[0..<m]
        }
        let candidateScores = scores.take(candidates, axis: 0)
        eval(candidates, candidateScores)
        let entries = candidates.asType(.int32).asArray(Int32.self)
        let values = candidateScores.asArray(Float.self)
        let ranked = zip(entries, values)
            .map { (entry: Int($0.0), score: $0.1) }
            .sorted { $0.score != $1.score ? $0.score > $1.score : $0.entry < $1.entry }
        return Array(ranked.prefix(k))
    }

    /// The `k` nearest entries to each row of `queries` ([T, D], unit norm), by cosine, never an
    /// entry in `excluding` (a document's own entries: they sit together, in corpus order).
    public func query(batch queries: MLXArray, k: Int, excluding: Range<Int>? = nil) -> [[(entry: Int, score: Float)]] {
        let n = count
        let rows = queries.dim(0)
        guard n > 0, k > 0, rows > 0 else { return Array(repeating: [], count: rows) }
        var scores = matmul(queries, keys.T)
        if let excluding {
            let range = excluding.clamped(to: 0..<n)
            if !range.isEmpty {
                var mask = [Float](repeating: 0, count: n)
                for i in range { mask[i] = -Float.infinity }
                scores = scores + MLXArray(mask, [1, n])
            }
        }
        let m = min(k + 8, n)
        let candidates = m >= n ? argSort(-scores, axis: -1) : argPartition(-scores, kth: m - 1, axis: -1)[0..., 0..<m]
        let candidateScores = takeAlong(scores, candidates, axis: -1)
        eval(candidates, candidateScores)
        let width = candidates.dim(1)
        let entries = candidates.asType(.int32).asArray(Int32.self)
        let values = candidateScores.asArray(Float.self)
        var result: [[(entry: Int, score: Float)]] = []
        result.reserveCapacity(rows)
        for r in 0..<rows {
            var row: [(entry: Int, score: Float)] = []
            for c in 0..<width where values[r * width + c].isFinite {
                row.append((entry: Int(entries[r * width + c]), score: values[r * width + c]))
            }
            row.sort { $0.score != $1.score ? $0.score > $1.score : $0.entry < $1.entry }
            result.append(Array(row.prefix(k)))
        }
        return result
    }

    /// Each document's entries, which sit together in corpus order.
    public lazy var entriesByDocument: [String: Range<Int>] = {
        var ranges: [String: Range<Int>] = [:]
        for entry in 0..<count {
            guard let documentID = partitionsByRow[Int(keyRow[entry])]?.documentID else { continue }
            if let range = ranges[documentID] {
                ranges[documentID] = min(range.lowerBound, entry)..<max(range.upperBound, entry + 1)
            } else {
                ranges[documentID] = entry..<(entry + 1)
            }
        }
        return ranges
    }()

    public func keyPosition(_ entry: Int) -> TokenPosition {
        TokenPosition(row: Int(keyRow[entry]), offset: Int(keyOffset[entry]))
    }

    public func valuePosition(_ entry: Int) -> TokenPosition {
        TokenPosition(row: Int(valueRow[entry]), offset: Int(valueOffset[entry]))
    }
}

public enum ProvenanceError: Error, CustomStringConvertible {
    case corruptIndex(String)
    case checkpointMismatch(index: String, checkpoint: String)
    case tokenizerMismatch(expected: String, found: String)
    case weakIndex(epoch: Int, memorised: Float)
    case missingCheckpoint(String)
    case emptyPrompt

    public var description: String {
        switch self {
        case .corruptIndex(let reason): return "provenance index is corrupt: \(reason)"
        case .checkpointMismatch(let index, let checkpoint):
            return "the provenance index was built with checkpoint \(index.prefix(12))… but the checkpoint on disk hashes to \(checkpoint.prefix(12))…"
        case .tokenizerMismatch(let expected, let found):
            return "tokenizer mismatch: run expects \(expected.prefix(12))…, loaded \(found.prefix(12))…"
        case .weakIndex(let epoch, let memorised):
            return "epoch \(epoch) memorised only \(Int(memorised * 100))% of the corpus; its citations would be weak (pass --allow-weak-index to use it anyway)"
        case .missingCheckpoint(let path): return "checkpoint missing at \(path)"
        case .emptyPrompt: return "the prompt tokenized to nothing"
        }
    }
}
