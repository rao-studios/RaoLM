//
//  RunManifest.swift
//  RaoLMCore
//
//  WHAT: run.json — everything a citation refers back to: the model shape, the tokenizer,
//        the exact Thread corpus the weights saw, the hyperparameters, and per epoch the
//        SHA-256 of the checkpoint, the ledger and the provenance index.
//

import Foundation

public struct TrainingHyperparameters: Codable, Sendable, Equatable {
    public var batchSize: Int
    public var seqLen: Int
    public var epochs: Int
    public var peakLR: Float
    public var warmupFraction: Float
    public var finalLR: Float
    public var beta1: Float
    public var beta2: Float
    public var eps: Float
    public var weightDecay: Float
    public var biasCorrection: Bool
    public var gradClip: Float
    public var seed: UInt64
    /// Run the deterministic eval pass (and write eval ledger rows) every N epochs; the
    /// final epoch always gets one.
    public var evalEvery: Int
    /// Build a provenance index every N epochs; 0 means the final epoch only.
    public var indexEvery: Int
    public var keepCheckpoints: Int
    /// Stop early once an eval pass reports at least this memorised fraction.
    public var earlyStopMemorised: Float?
    public var evalBatchSize: Int

    public init(
        batchSize: Int = 4, seqLen: Int = 512, epochs: Int = 60, peakLR: Float = 2e-3,
        warmupFraction: Float = 0.05, finalLR: Float = 1e-4, beta1: Float = 0.9, beta2: Float = 0.95,
        eps: Float = 1e-8, weightDecay: Float = 0.01, biasCorrection: Bool = true, gradClip: Float = 1.0,
        seed: UInt64 = 42, evalEvery: Int = 10, indexEvery: Int = 0, keepCheckpoints: Int = 3,
        earlyStopMemorised: Float? = 0.98, evalBatchSize: Int = 8
    ) {
        self.batchSize = batchSize
        self.seqLen = seqLen
        self.epochs = epochs
        self.peakLR = peakLR
        self.warmupFraction = warmupFraction
        self.finalLR = finalLR
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weightDecay = weightDecay
        self.biasCorrection = biasCorrection
        self.gradClip = gradClip
        self.seed = seed
        self.evalEvery = evalEvery
        self.indexEvery = indexEvery
        self.keepCheckpoints = keepCheckpoints
        self.earlyStopMemorised = earlyStopMemorised
        self.evalBatchSize = evalBatchSize
    }
}

/// How provenance keys are built (identical at index and query time).
public struct ProvenanceSettings: Codable, Sendable, Equatable {
    /// 0-based block whose residual output is the mid-layer half of the key.
    public var tapLayer: Int
    /// Weight of the mid-layer half: key = [√α·tap/‖tap‖, √(1−α)·final/‖final‖].
    public var alpha: Float

    public init(tapLayer: Int, alpha: Float = 0.5) {
        self.tapLayer = tapLayer
        self.alpha = alpha
    }

    public static func defaults(for config: RaoLMConfig) -> ProvenanceSettings {
        ProvenanceSettings(tapLayer: config.numHiddenLayers / 2, alpha: 0.5)
    }
}

public enum RunStatus: String, Codable, Sendable {
    case starting, ingesting, snapshotting, training, indexing, generating, verifying, complete, failed
}

public struct TokenizerRef: Codable, Sendable, Equatable {
    public var id: String
    public var revision: String
    public var tokenizerSHA256: String
    public var vocabSize: Int
    public var eosTokenID: Int

    public init(id: String, revision: String, tokenizerSHA256: String, vocabSize: Int, eosTokenID: Int) {
        self.id = id
        self.revision = revision
        self.tokenizerSHA256 = tokenizerSHA256
        self.vocabSize = vocabSize
        self.eosTokenID = eosTokenID
    }
}

public struct CorpusRef: Codable, Sendable, Equatable {
    public var slug: String?
    public var corpusHash: String
    public var snapshotPath: String
    public var source: String
    public var threadID: String?
    public var owner: String
    public var group: String
    public var documentCount: Int
    public var partitionCount: Int
    public var tokenCount: Int
    /// facts.jsonl of the generated corpus, when the corpus came from the synthetic generator.
    public var factsPath: String?

    public init(
        slug: String?, corpusHash: String, snapshotPath: String, source: String, threadID: String?,
        owner: String, group: String, documentCount: Int, partitionCount: Int, tokenCount: Int, factsPath: String? = nil
    ) {
        self.slug = slug
        self.corpusHash = corpusHash
        self.snapshotPath = snapshotPath
        self.source = source
        self.threadID = threadID
        self.owner = owner
        self.group = group
        self.documentCount = documentCount
        self.partitionCount = partitionCount
        self.tokenCount = tokenCount
        self.factsPath = factsPath
    }
}

public struct ThreadRef: Codable, Sendable, Equatable {
    public var binary: String
    public var host: String
    public var httpPort: Int
    public var grpcPort: Int
    public var dataDir: String
    public var nodeID: String?

    public init(binary: String, host: String, httpPort: Int, grpcPort: Int, dataDir: String, nodeID: String?) {
        self.binary = binary
        self.host = host
        self.httpPort = httpPort
        self.grpcPort = grpcPort
        self.dataDir = dataDir
        self.nodeID = nodeID
    }
}

public struct EpochRecord: Codable, Sendable, Equatable {
    public var epoch: Int
    public var steps: Int
    public var trainLoss: Float
    public var trainEntropy: Float
    public var evalLoss: Float?
    public var evalEntropy: Float?
    public var evalMemorisedFraction: Float?
    public var calibrationGap: Float?
    public var checkpointPath: String?
    public var checkpointSHA256: String?
    public var partitionLedgerSHA256: String?
    public var indexPath: String?
    public var indexSHA256: String?
    public var wallClockSeconds: Double
    public var completedAt: Date

    public init(
        epoch: Int, steps: Int, trainLoss: Float, trainEntropy: Float, evalLoss: Float? = nil,
        evalEntropy: Float? = nil, evalMemorisedFraction: Float? = nil, calibrationGap: Float? = nil,
        checkpointPath: String? = nil, checkpointSHA256: String? = nil, partitionLedgerSHA256: String? = nil,
        indexPath: String? = nil, indexSHA256: String? = nil, wallClockSeconds: Double, completedAt: Date = Date()
    ) {
        self.epoch = epoch
        self.steps = steps
        self.trainLoss = trainLoss
        self.trainEntropy = trainEntropy
        self.evalLoss = evalLoss
        self.evalEntropy = evalEntropy
        self.evalMemorisedFraction = evalMemorisedFraction
        self.calibrationGap = calibrationGap
        self.checkpointPath = checkpointPath
        self.checkpointSHA256 = checkpointSHA256
        self.partitionLedgerSHA256 = partitionLedgerSHA256
        self.indexPath = indexPath
        self.indexSHA256 = indexSHA256
        self.wallClockSeconds = wallClockSeconds
        self.completedAt = completedAt
    }
}

public struct RunManifest: Codable, Sendable {
    public var schemaVersion: Int
    public var runID: String
    public var createdAt: Date
    public var updatedAt: Date
    public var status: RunStatus
    public var preset: String
    public var model: RaoLMConfig
    public var parameterCount: Int
    public var tokenizer: TokenizerRef
    public var corpus: CorpusRef
    public var hyperparameters: TrainingHyperparameters
    public var provenance: ProvenanceSettings
    public var epochs: [EpochRecord]
    public var indexedEpochs: [Int]
    public var stepsLedgerSHA256: String?
    public var thread: ThreadRef?
    public var excludedDocumentIDs: [String]
    public var notes: [String]
    public var failure: String?

    public init(
        runID: String, preset: String, model: RaoLMConfig, tokenizer: TokenizerRef, corpus: CorpusRef,
        hyperparameters: TrainingHyperparameters, provenance: ProvenanceSettings,
        thread: ThreadRef? = nil, excludedDocumentIDs: [String] = [], now: Date = Date()
    ) {
        self.schemaVersion = 1
        self.runID = runID
        self.createdAt = now
        self.updatedAt = now
        self.status = .starting
        self.preset = preset
        self.model = model
        self.parameterCount = model.parameterCount
        self.tokenizer = tokenizer
        self.corpus = corpus
        self.hyperparameters = hyperparameters
        self.provenance = provenance
        self.epochs = []
        self.indexedEpochs = []
        self.stepsLedgerSHA256 = nil
        self.thread = thread
        self.excludedDocumentIDs = excludedDocumentIDs
        self.notes = [
            "Optimizer state is not checkpointed; a checkpoint restores weights only.",
            "Training on Metal is not bit-reproducible (scatter-add atomics); hashes name the weights that exist.",
        ]
        self.failure = nil
    }

    public static let fileName = "run.json"

    public func save(to runDirectory: URL) throws {
        var copy = self
        copy.updatedAt = Date()
        try JSONCoding.write(copy, to: runDirectory.appendingPathComponent(Self.fileName))
    }

    public static func load(_ runDirectory: URL) throws -> RunManifest {
        let url = runDirectory.appendingPathComponent(fileName)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw RunManifestError.notARun(runDirectory.path)
        }
        return try JSONCoding.read(RunManifest.self, from: url)
    }

    public func epochRecord(_ epoch: Int) -> EpochRecord? {
        epochs.last { $0.epoch == epoch }
    }

    /// The indexed epoch generation should use by default: the latest one.
    public var latestIndexedEpoch: Int? { indexedEpochs.max() }
}

public enum RunManifestError: Error, CustomStringConvertible {
    case notARun(String)
    case noIndex(String)
    case epochNotIndexed(Int, available: [Int])

    public var description: String {
        switch self {
        case .notARun(let path): return "no run.json in \(path)"
        case .noIndex(let path): return "run \(path) has no provenance index yet"
        case .epochNotIndexed(let epoch, let available):
            return "epoch \(epoch) has no provenance index (indexed: \(available.map(String.init).joined(separator: ", ")))"
        }
    }
}

/// Per-epoch directory names inside a run directory.
public enum RunLayout {
    public static func checkpoint(_ runDirectory: URL, epoch: Int) -> URL {
        runDirectory.appendingPathComponent("checkpoints/epoch-\(epoch)", isDirectory: true)
    }

    public static func provenance(_ runDirectory: URL, epoch: Int) -> URL {
        runDirectory.appendingPathComponent("provenance/epoch-\(epoch)", isDirectory: true)
    }

    public static func ledger(_ runDirectory: URL) -> URL {
        runDirectory.appendingPathComponent("ledger", isDirectory: true)
    }

    public static func generations(_ runDirectory: URL) -> URL {
        runDirectory.appendingPathComponent("generations", isDirectory: true)
    }
}
