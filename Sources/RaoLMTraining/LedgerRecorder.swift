//
//  LedgerRecorder.swift
//  RaoLMTraining
//
//  WHAT: Writes the entropy ledger while training runs: a row per optimizer step, a row per
//        partition per epoch (online training stats, plus eval stats on eval epochs), per-fact
//        answer-token losses on eval epochs, and an epoch summary.
//

import Foundation
import RaoLMCore

public final class LedgerRecorder {
    public let directory: URL
    private let steps: JSONLWriter
    private let epochs: JSONLWriter
    private var trainLoss: [Double]
    private var trainEntropy: [Double]
    private var trainCount: [Int]
    public private(set) var memorisedAtEpoch: [Int: Int] = [:]

    public init(directory: URL, rowCount: Int) throws {
        self.directory = directory
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        self.steps = try JSONLWriter(url: LedgerFiles.steps(directory), truncate: true)
        self.epochs = try JSONLWriter(url: LedgerFiles.epochs(directory), truncate: true)
        self.trainLoss = [Double](repeating: 0, count: rowCount)
        self.trainEntropy = [Double](repeating: 0, count: rowCount)
        self.trainCount = [Int](repeating: 0, count: rowCount)
    }

    /// `rows[i]` is the partition row of input position i (−1 for eos inputs and padding).
    public func recordStep(_ row: StepRow, perTokenLoss: [Float], entropy: [Float], rows: [Int32]) throws {
        try steps.append(row)
        for i in rows.indices where rows[i] >= 0 {
            let r = Int(rows[i])
            trainLoss[r] += Double(perTokenLoss[i])
            trainEntropy[r] += Double(entropy[i])
            trainCount[r] += 1
        }
    }

    /// Writes this epoch's partition (and fact) rows, resets the online accumulators, and
    /// returns the SHA-256 of partitions-epoch-<n>.jsonl.
    public func finishEpoch(epoch: Int, corpus: TokenizedCorpus, eval: EvalResult?, facts: [LocatedFact]) throws -> String {
        let evalStats = eval?.partitionStats(rowCount: corpus.partitions.count)
        if let evalStats {
            for (row, stats) in evalStats.enumerated() {
                if let stats, stats.memorisedFraction >= 0.95, memorisedAtEpoch[row] == nil {
                    memorisedAtEpoch[row] = epoch
                }
            }
        }
        let url = LedgerFiles.partitions(directory, epoch: epoch)
        let writer = try JSONLWriter(url: url, truncate: true)
        for partition in corpus.partitions {
            let r = partition.row
            let train = trainCount[r] > 0
                ? PartitionTrainStats(
                    positions: trainCount[r], meanLoss: Float(trainLoss[r] / Double(trainCount[r])),
                    meanEntropy: Float(trainEntropy[r] / Double(trainCount[r])))
                : nil
            try writer.append(PartitionEpochRow(
                epoch: epoch, row: r, documentID: partition.documentID, partitionIndex: partition.partitionIndex,
                tokens: partition.tokens.count, train: train, eval: evalStats?[r] ?? nil,
                memorisedAtEpoch: memorisedAtEpoch[r]))
        }
        writer.close()

        if let eval, !facts.isEmpty {
            let losses = eval.lossByValuePosition()
            let factWriter = try JSONLWriter(url: LedgerFiles.facts(directory, epoch: epoch), truncate: true)
            for located in facts {
                let answer = (located.answerToken..<located.answerEndToken).map {
                    losses[TokenPosition(row: located.row, offset: $0)] ?? .nan
                }
                try factWriter.append(FactEpochRow(
                    epoch: epoch, factID: located.fact.id, documentID: located.fact.documentID,
                    partitionIndex: located.fact.partitionIndex, answerTokenLosses: answer))
            }
            factWriter.close()
        }

        for r in trainLoss.indices {
            trainLoss[r] = 0
            trainEntropy[r] = 0
            trainCount[r] = 0
        }
        return try ContentHash.sha256Hex(fileAt: url)
    }

    public func recordEpoch(_ record: EpochRecord) throws {
        try epochs.append(EpochRow(record: record))
    }

    public func close() {
        steps.close()
        epochs.close()
    }

    public func stepsSHA256() throws -> String {
        try ContentHash.sha256Hex(fileAt: LedgerFiles.steps(directory))
    }
}
