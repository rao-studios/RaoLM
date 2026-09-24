//
//  LedgerRows.swift
//  RaoLMCore
//
//  WHAT: The entropy ledger's row types. The trainer appends them as JSONL under
//        <run>/ledger/; `raolm ledger` and the citation confidence read them back.
//
//    steps.jsonl                one row per optimizer step
//    partitions-epoch-<n>.jsonl one row per corpus partition, every epoch
//    facts-epoch-<n>.jsonl      per fact, the eval loss of each answer token (eval epochs)
//    epochs.jsonl               one summary row per epoch
//
//  PIN:  Positions whose *input* token is <|endoftext|> are excluded from every statistic:
//        under document packing they predict the first token of an unrelated document.
//

import Foundation

public struct EntropySummary: Codable, Sendable, Equatable {
    public var mean: Float
    public var p10: Float
    public var p50: Float
    public var p90: Float
    public var max: Float

    public init(mean: Float, p10: Float, p50: Float, p90: Float, max: Float) {
        self.mean = mean
        self.p10 = p10
        self.p50 = p50
        self.p90 = p90
        self.max = max
    }

    public init(values: [Float]) {
        guard !values.isEmpty else {
            self.init(mean: 0, p10: 0, p50: 0, p90: 0, max: 0)
            return
        }
        let sorted = values.sorted()
        self.init(
            mean: Stats.mean(values),
            p10: Stats.quantileSorted(sorted, 0.1),
            p50: Stats.quantileSorted(sorted, 0.5),
            p90: Stats.quantileSorted(sorted, 0.9),
            max: sorted.last ?? 0)
    }
}

public struct StepRow: Codable, Sendable, Equatable {
    public var epoch: Int
    public var step: Int
    public var globalStep: Int
    public var lr: Float
    public var loss: Float
    public var entropy: EntropySummary
    /// Mean cross-entropy minus mean entropy: ≈0 for a calibrated model, > 0 confidently wrong.
    public var lossMinusEntropy: Float
    public var gradNorm: Float
    public var clipped: Bool
    public var tokens: Int
    public var maskedTokens: Int
    public var tokensPerSecond: Double
    public var wallClockSeconds: Double
    public var timestamp: Date

    public init(
        epoch: Int, step: Int, globalStep: Int, lr: Float, loss: Float, entropy: EntropySummary,
        lossMinusEntropy: Float, gradNorm: Float, clipped: Bool, tokens: Int, maskedTokens: Int,
        tokensPerSecond: Double, wallClockSeconds: Double, timestamp: Date = Date()
    ) {
        self.epoch = epoch
        self.step = step
        self.globalStep = globalStep
        self.lr = lr
        self.loss = loss
        self.entropy = entropy
        self.lossMinusEntropy = lossMinusEntropy
        self.gradNorm = gradNorm
        self.clipped = clipped
        self.tokens = tokens
        self.maskedTokens = maskedTokens
        self.tokensPerSecond = tokensPerSecond
        self.wallClockSeconds = wallClockSeconds
        self.timestamp = timestamp
    }
}

public struct PartitionTrainStats: Codable, Sendable, Equatable {
    public var positions: Int
    public var meanLoss: Float
    public var meanEntropy: Float

    public init(positions: Int, meanLoss: Float, meanEntropy: Float) {
        self.positions = positions
        self.meanLoss = meanLoss
        self.meanEntropy = meanEntropy
    }
}

public struct PartitionEvalStats: Codable, Sendable, Equatable {
    public var positions: Int
    public var meanLoss: Float
    public var meanEntropy: Float
    public var maxLoss: Float
    public var p90Loss: Float
    /// Fraction of positions with loss < 0.1 nat (p ≥ 0.905 on the true next token).
    public var memorisedFraction: Float

    public init(positions: Int, meanLoss: Float, meanEntropy: Float, maxLoss: Float, p90Loss: Float, memorisedFraction: Float) {
        self.positions = positions
        self.meanLoss = meanLoss
        self.meanEntropy = meanEntropy
        self.maxLoss = maxLoss
        self.p90Loss = p90Loss
        self.memorisedFraction = memorisedFraction
    }

    public static let memorisedLoss: Float = 0.1
}

public struct PartitionEpochRow: Codable, Sendable, Equatable {
    public var epoch: Int
    public var row: Int
    public var documentID: String
    public var partitionIndex: Int
    public var tokens: Int
    public var train: PartitionTrainStats?
    public var eval: PartitionEvalStats?
    /// First eval epoch at which ≥ 95% of this partition's positions were memorised.
    public var memorisedAtEpoch: Int?

    public init(
        epoch: Int, row: Int, documentID: String, partitionIndex: Int, tokens: Int,
        train: PartitionTrainStats?, eval: PartitionEvalStats?, memorisedAtEpoch: Int?
    ) {
        self.epoch = epoch
        self.row = row
        self.documentID = documentID
        self.partitionIndex = partitionIndex
        self.tokens = tokens
        self.train = train
        self.eval = eval
        self.memorisedAtEpoch = memorisedAtEpoch
    }
}

public struct FactEpochRow: Codable, Sendable, Equatable {
    public var epoch: Int
    public var factID: String
    public var documentID: String
    public var partitionIndex: Int
    public var answerTokenLosses: [Float]
    public var meanLoss: Float
    public var memorised: Bool

    public init(epoch: Int, factID: String, documentID: String, partitionIndex: Int, answerTokenLosses: [Float]) {
        self.epoch = epoch
        self.factID = factID
        self.documentID = documentID
        self.partitionIndex = partitionIndex
        self.answerTokenLosses = answerTokenLosses
        self.meanLoss = Stats.mean(answerTokenLosses)
        self.memorised = answerTokenLosses.allSatisfy { $0 < PartitionEvalStats.memorisedLoss }
    }
}

public struct EpochRow: Codable, Sendable, Equatable {
    public var epoch: Int
    public var steps: Int
    public var trainLoss: Float
    public var trainEntropy: Float
    public var evalLoss: Float?
    public var evalEntropy: Float?
    public var evalMemorisedFraction: Float?
    public var calibrationGap: Float?
    public var checkpointSHA256: String?
    public var indexSHA256: String?
    public var partitionLedgerSHA256: String?
    public var wallClockSeconds: Double

    public init(record: EpochRecord) {
        self.epoch = record.epoch
        self.steps = record.steps
        self.trainLoss = record.trainLoss
        self.trainEntropy = record.trainEntropy
        self.evalLoss = record.evalLoss
        self.evalEntropy = record.evalEntropy
        self.evalMemorisedFraction = record.evalMemorisedFraction
        self.calibrationGap = record.calibrationGap
        self.checkpointSHA256 = record.checkpointSHA256
        self.indexSHA256 = record.indexSHA256
        self.partitionLedgerSHA256 = record.partitionLedgerSHA256
        self.wallClockSeconds = record.wallClockSeconds
    }
}

public enum LedgerFiles {
    public static func steps(_ ledger: URL) -> URL { ledger.appendingPathComponent("steps.jsonl") }
    public static func epochs(_ ledger: URL) -> URL { ledger.appendingPathComponent("epochs.jsonl") }
    public static func partitions(_ ledger: URL, epoch: Int) -> URL {
        ledger.appendingPathComponent("partitions-epoch-\(epoch).jsonl")
    }
    public static func facts(_ ledger: URL, epoch: Int) -> URL {
        ledger.appendingPathComponent("facts-epoch-\(epoch).jsonl")
    }
}
