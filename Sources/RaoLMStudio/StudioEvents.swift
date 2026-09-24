//
//  StudioEvents.swift
//  RaoLMStudio
//
//  WHAT: What the UI asks the backend to do (jobs) and what the backend reports back
//        (events). Both are Sendable values; the UI never touches a model or a socket.
//  PIN:  MLX jobs run one at a time on the MLX worker's thread; the rest run concurrently.
//

import Foundation
import RaoLM
import RaoLMWorkflows

public enum JobKind: String, Sendable, CaseIterable {
    case scanRuns, doctor, tests, threadStatus, threadStart, threadStop
    case scanCorpora, corpusGenerate, corpusLoad, ingest, pull, offlineSnapshot, scanSnapshots
    case train, loadContext, generate, verify, saveGeneration, loadGeneration, listGenerations, partitionText
    case eval, ledger, ground

    /// Runs on the MLX worker, one at a time.
    public var isMLX: Bool {
        switch self {
        case .train, .loadContext, .generate, .verify, .eval, .ground: return true
        default: return false
        }
    }

    /// Worth a spinner in the status bar.
    public var isVisible: Bool {
        switch self {
        case .scanRuns, .threadStatus, .partitionText, .scanCorpora, .scanSnapshots, .listGenerations: return false
        default: return true
        }
    }
}

public struct CorpusForm: Sendable, Equatable {
    public var slug: String
    public var documents: Int
    public var seed: UInt64
    public var maxChars: Int
    public var force: Bool
}

public struct GenerationOverrides: Sendable, Equatable {
    public var lambda: Float = 0.5
    public var tau: Float?
    public var k: Int?
    public var temperature: Float = 0
    public var topK = 0
    public var maxTokens = 48
    public var seed: UInt64 = 42

    public init() {}

    public func apply(to defaults: GenerationParameters) -> GenerationParameters {
        var params = defaults
        params.lambda = lambda
        if let tau { params.tau = tau }
        if let k { params.k = k }
        params.temperature = temperature
        params.topK = topK
        params.maxTokens = maxTokens
        params.seed = seed
        return params
    }
}

public struct GenerateSpec: Sendable, Equatable {
    public var run: URL
    public var epoch: Int?
    public var promptText: String
    public var slice: CorpusSlice?
    public var overrides: GenerationOverrides
}

public struct EvalSpec: Sendable, Equatable {
    public var run: URL
    public var epoch: Int?
    public var factsSample: Int
    public var lambdas: [Float]
    public var primaryLambda: Float
    public var seed: UInt64
    public var controls: Bool
    public var offline: Bool
    public var grounding: Bool
    public var owner: String?
}

/// A training run to start; the worker reads the snapshot, off the UI thread.
public struct TrainSpec: Sendable, Equatable {
    public var settings: TrainingSettings
    public var snapshotPath: URL
    public var factsPath: URL?
    public var runID: String
    public var runDirectory: URL
    public var thread: ThreadRef?
}

public struct GroundSpec: Sendable {
    public var generation: CitedGeneration
    public var run: URL
    /// "top", "spans", "all", "fact" or DOC:P[,DOC:P].
    public var policy: String
}

public enum StudioJob: Sendable {
    case scanRuns
    case doctor
    case runTests(filter: String?, mlx: Bool, thread: Bool)
    case cancelTests
    case threadStatus(withLog: Bool)
    case threadStart(fresh: Bool)
    case threadStop
    case scanCorpora
    case corpusGenerate(CorpusForm)
    case corpusLoad(URL)
    case ingest(corpus: URL)
    case pull(corpus: URL)
    case offlineSnapshot(corpus: URL)
    case scanSnapshots
    case train(TrainSpec)
    case loadContext(run: URL, epoch: Int?)
    case generate(GenerateSpec)
    case verify(CitedGeneration, run: URL, live: Bool)
    case saveGeneration(CitedGeneration, run: URL)
    case loadGeneration(URL)
    case listGenerations(run: URL)
    case partitionText(run: URL, documentID: String, partitionIndex: Int)
    case eval(EvalSpec)
    case ledger(run: URL, mode: LedgerMode, filter: String, partition: Int?)
    case ground(GroundSpec)
    case cancel

    public var kind: JobKind? {
        switch self {
        case .scanRuns: return .scanRuns
        case .doctor: return .doctor
        case .runTests: return .tests
        case .cancelTests, .cancel: return nil
        case .threadStatus: return .threadStatus
        case .threadStart: return .threadStart
        case .threadStop: return .threadStop
        case .scanCorpora: return .scanCorpora
        case .corpusGenerate: return .corpusGenerate
        case .corpusLoad: return .corpusLoad
        case .ingest: return .ingest
        case .pull: return .pull
        case .offlineSnapshot: return .offlineSnapshot
        case .scanSnapshots: return .scanSnapshots
        case .train: return .train
        case .loadContext: return .loadContext
        case .generate: return .generate
        case .verify: return .verify
        case .saveGeneration: return .saveGeneration
        case .loadGeneration: return .loadGeneration
        case .listGenerations: return .listGenerations
        case .partitionText: return .partitionText
        case .eval: return .eval
        case .ledger: return .ledger
        case .ground: return .ground
        }
    }
}

public enum StudioEvent: Sendable {
    case jobStarted(JobKind, String)
    case jobProgress(JobKind, String)
    case jobFinished(JobKind)
    case jobFailed(JobKind, RaoLMFailure)
    case log(String)
    case runsLoaded([RunSummary], warnings: [String])
    case doctorChecks([DoctorCheck])
    case testOutput([String])
    case testFinished(Int32)
    case threadStatus(ThreadStatus)
    case corporaLoaded([CorpusSummary])
    case corpusLoaded(GeneratedCorpus, URL)
    case corpusResult(String, problems: [String])
    case ingestProgress(Int, Int)
    case snapshotsLoaded([SnapshotChoice])
    case trainPrepared(RunManifest, [String])
    case training(TrainingEvent)
    case trainFinished(RunManifest)
    case contextLoaded(ContextInfo)
    case token(TokenTrace)
    case generated(CitedGeneration, [[NeighbourRow]])
    case verified(CitedGeneration, [String], source: String)
    case generationSaved(URL)
    case generationLoaded(CitedGeneration, URL, [[NeighbourRow]], GroundingRecord?)
    case generationsListed(run: URL, [URL])
    case partitionText(PartitionText)
    case evalProgress(String)
    case evalFinished(EvalReport, URL)
    case ledgerLoaded(LedgerData)
    case grounded(GroundingRecord)
}
