//
//  VocabularyBench.swift
//  RaoLMBraid
//
//  WHAT: The experiment that decides what a braid's shared vocabulary is. One offline corpus
//        is trained several ways — everything trainable (today's recipe), blocks only against
//        a seeded constant vocabulary at several head scales, blocks only against a vocabulary
//        trained once on a separate "commons" world — and each is measured the same way.
//  OUT:  One row per arm: epochs and seconds until the memorised fraction reached the target,
//        the final eval loss, and exact answers and citation@1 from the fact evaluator.
//  PIN:  The rule is fixed before the numbers are seen: seeded wins if it reaches the target
//        within twice the baseline's epochs and within five points of its exact-answer rate;
//        otherwise the commons vocabulary under the same two tests; otherwise each Thread
//        keeps its own vocabulary. Synchronous apart from the evaluator: the caller owns the
//        thread MLX runs on.
//

import Foundation
import MLX
import MLXNN
import RaoLMCore
import RaoLMModel
import RaoLMProvenance
import RaoLMTraining

public struct VocabularyBenchOptions: Sendable {
    public var preset = "tiny"
    public var documents = 24
    public var commonsDocuments = 48
    public var epochs = 120
    public var batchSize = 4
    public var seqLen = 256
    public var lr: Float = 2e-3
    public var evalEvery = 4
    public var target: Float = 0.97
    public var headScales: [Float] = [1, 1.5, 2]
    public var rowNorm: Float = 1
    public var seed: UInt64 = 42
    public var factsSample = 40
    /// Which arms to train: any of "baseline", "seeded", "commons".
    public var arms: Set<String> = ["baseline", "seeded", "commons"]
    /// Rows of an earlier report kept for the arms not trained again (its baseline, typically).
    public var reuse: [VocabularyBenchRow] = []
    public var directory: URL

    public init(directory: URL) {
        self.directory = directory
    }
}

public struct VocabularyBenchRow: Codable, Sendable, Equatable {
    public var name: String
    public var vocabulary: String
    public var trainableParameters: Int
    public var epochsToTarget: Int?
    public var secondsToTarget: Double?
    public var epochs: Int
    public var seconds: Double
    public var evalLoss: Float
    public var memorised: Float
    /// Exact answers at λ = 0.5 and at λ = 0 (evidence only).
    public var exact: Float
    public var exactEvidenceOnly: Float
    public var citationAt1: Float
    public var runDirectory: String
}

public enum VocabularyChoice: String, Codable, Sendable {
    case seeded, commons, perThread
}

public struct VocabularyBenchReport: Codable, Sendable {
    public var createdAt: Date
    public var preset: String
    public var documents: Int
    public var tokens: Int
    public var target: Float
    public var rows: [VocabularyBenchRow]
    public var choice: VocabularyChoice
    public var headScale: Float?
    public var reason: String

    public static let fileName = "vocabulary-bench.json"
}

public enum VocabularyBench {

    /// The rule, as a function of the rows alone.
    public static func decide(_ rows: [VocabularyBenchRow]) -> (choice: VocabularyChoice, headScale: Float?, reason: String) {
        guard let baseline = rows.first(where: { $0.name == "baseline" }) else {
            return (.perThread, nil, "no baseline row")
        }
        func passes(_ row: VocabularyBenchRow) -> Bool {
            guard let epochs = row.epochsToTarget else { return false }
            let allowed = baseline.epochsToTarget.map { $0 * 2 } ?? Int.max
            return epochs <= allowed && row.exact >= baseline.exact - 0.05
        }
        func best(_ candidates: [VocabularyBenchRow]) -> VocabularyBenchRow? {
            candidates.filter(passes).min {
                ($0.epochsToTarget ?? .max, -$0.exact) < ($1.epochsToTarget ?? .max, -$1.exact)
            }
        }
        let baselineText = baseline.epochsToTarget.map { "\($0) epochs" } ?? "never within the budget"
        if let row = best(rows.filter { $0.name.hasPrefix("seeded") }) {
            let scale = Float(row.name.split(separator: "×").last.map(String.init) ?? "")
            return (.seeded, scale, "\(row.name) reached the target in \(row.epochsToTarget ?? 0) epochs (baseline \(baselineText)) with exact \(Int(row.exact * 100))% against \(Int(baseline.exact * 100))%")
        }
        if let row = best(rows.filter { $0.name.hasPrefix("commons") }) {
            return (.commons, nil, "no seeded arm passed; \(row.name) reached the target in \(row.epochsToTarget ?? 0) epochs (baseline \(baselineText)) with exact \(Int(row.exact * 100))%")
        }
        return (.perThread, nil, "no blocks-only arm reached the target within twice the baseline (\(baselineText)) and five points of its exact answers: each Thread keeps its own vocabulary")
    }

    struct Arm {
        var name: String
        var vocabulary: String
        var model: RaoTransformer
    }

    public static func run(
        _ options: VocabularyBenchOptions, tokenizer: RaoTokenizer, progress: (String) -> Void = { _ in }
    ) async throws -> VocabularyBenchReport {
        let config = try RaoLMConfig.preset(options.preset)
        try config.validate()
        let manager = FileManager.default
        if manager.fileExists(atPath: options.directory.path) { try manager.removeItem(at: options.directory) }
        try manager.createDirectory(at: options.directory, withIntermediateDirectories: true)

        // One world for the Thread under test, another that no Thread holds.
        let corpus = try SyntheticCorpus.generate(slug: "bench", seed: options.seed, documentCount: options.documents)
        let commons = try SyntheticCorpus.generate(slug: "commons", seed: options.seed &+ 7919, documentCount: options.commonsDocuments)
        let snapshot = CorpusSnapshot.offline(corpus)
        let commonsSnapshot = CorpusSnapshot.offline(commons)
        let snapshotDirectory = options.directory.appendingPathComponent("snapshot")
        let commonsDirectory = options.directory.appendingPathComponent("commons-snapshot")
        try snapshot.save(to: snapshotDirectory)
        try commonsSnapshot.save(to: commonsDirectory)
        let tokenized = TokenizedCorpus(snapshot: snapshot, tokenizer: tokenizer)
        progress("corpus: \(tokenized.documents.count) documents, \(tokenized.partitions.count) partitions, \(tokenized.tokenCount) tokens; commons: \(commons.documents.count) documents")

        var rows: [VocabularyBenchRow] = []
        func measure(_ arm: Arm) async throws {
            let row = try await train(
                arm, corpus: corpus, snapshot: snapshot, snapshotDirectory: snapshotDirectory, tokenized: tokenized,
                config: config, options: options, tokenizer: tokenizer, progress: progress)
            rows.append(row)
        }
        func reused(_ family: String) -> Bool {
            let kept = options.reuse.filter { $0.name.hasPrefix(family) }
            guard !options.arms.contains(family), !kept.isEmpty else { return false }
            for row in kept { progress("\(row.name): kept from the earlier report") }
            rows += kept
            return true
        }

        if !reused("baseline"), options.arms.contains("baseline") {
            try await measure(Arm(name: "baseline", vocabulary: "its own", model: try RaoTransformer.make(config: config, seed: options.seed)))
        }

        if !reused("seeded"), options.arms.contains("seeded") {
            for scale in options.headScales {
                let vocabulary = VocabularyPack.seeded(
                    config: config, tokenizerSHA256: tokenizer.tokenizerSHA256, seed: options.seed, rowNorm: options.rowNorm, headScale: scale)
                let model = try RaoTransformer.make(config: config, seed: options.seed)
                try vocabulary.install(into: model)
                try await measure(Arm(name: "seeded ×\(scale)", vocabulary: String(vocabulary.sha256.prefix(12)), model: model))
            }
        }

        guard !reused("commons"), options.arms.contains("commons") else {
            return try report(rows, tokenized: tokenized, options: options)
        }
        // The commons vocabulary: a whole model trained on the other world, then frozen.
        progress("training the commons model on \(commons.documents.count) documents")
        let commonsModel = try RaoTransformer.make(config: config, seed: options.seed &+ 1)
        let commonsTokenized = TokenizedCorpus(snapshot: commonsSnapshot, tokenizer: tokenizer)
        let commonsRun = options.directory.appendingPathComponent("commons-model")
        let commonsManifest = try pretrain(
            commonsModel, runID: "commons-model", runDirectory: commonsRun, corpus: commons, snapshot: commonsSnapshot,
            snapshotDirectory: commonsDirectory, tokenized: commonsTokenized, config: config, options: options,
            tokenizer: tokenizer, progress: { _ in })
        guard let commonsEpoch = commonsManifest.epochs.last, let commonsCheckpoint = commonsEpoch.checkpointPath else {
            throw CheckpointError.missing("the commons model wrote no checkpoint")
        }
        progress(String(format: "commons model: %d epochs, memorised %.1f%%", commonsEpoch.epoch, (commonsEpoch.evalMemorisedFraction ?? 0) * 100))
        let commonsVocabulary = VocabularyPack.from(
            model: commonsModel, tokenizerSHA256: tokenizer.tokenizerSHA256, originSHA256: commonsEpoch.checkpointSHA256)

        let fresh = try RaoTransformer.make(config: config, seed: options.seed)
        try commonsVocabulary.install(into: fresh)
        try await measure(Arm(name: "commons, fresh blocks", vocabulary: String(commonsVocabulary.sha256.prefix(12)), model: fresh))

        let warm = try Checkpoint.load(from: URL(fileURLWithPath: commonsCheckpoint), tapLayer: config.defaultTapLayer)
        VocabularyPack.freeze(warm)
        try await measure(Arm(name: "commons, warm blocks", vocabulary: String(commonsVocabulary.sha256.prefix(12)), model: warm))
        try commonsVocabulary.save(to: options.directory.appendingPathComponent("commons-vocabulary", isDirectory: true))
        return try report(rows, tokenized: tokenized, options: options)
    }

    static func report(_ rows: [VocabularyBenchRow], tokenized: TokenizedCorpus, options: VocabularyBenchOptions) throws -> VocabularyBenchReport {
        let decision = decide(rows)
        let report = VocabularyBenchReport(
            createdAt: .wholeSecond(), preset: options.preset, documents: tokenized.documents.count, tokens: tokenized.tokenCount,
            target: options.target, rows: rows, choice: decision.choice, headScale: decision.headScale, reason: decision.reason)
        try JSONCoding.write(report, to: options.directory.appendingPathComponent(VocabularyBenchReport.fileName))
        return report
    }

    static func hyperparameters(_ options: VocabularyBenchOptions) -> TrainingHyperparameters {
        TrainingHyperparameters(
            batchSize: options.batchSize, seqLen: options.seqLen, epochs: options.epochs, peakLR: options.lr, seed: options.seed,
            evalEvery: options.evalEvery, indexEvery: 0, keepCheckpoints: 1, earlyStopMemorised: options.target)
    }

    @discardableResult
    static func pretrain(
        _ model: RaoTransformer, runID: String, runDirectory: URL, corpus: GeneratedCorpus, snapshot: CorpusSnapshot,
        snapshotDirectory: URL, tokenized: TokenizedCorpus, config: RaoLMConfig, options: VocabularyBenchOptions,
        tokenizer: RaoTokenizer, progress: (TrainingEvent) -> Void
    ) throws -> RunManifest {
        let hyper = hyperparameters(options)
        let provenance = ProvenanceSettings.defaults(for: config)
        let located = FactLocator.locate(corpus.facts, corpus: tokenized, tokenizer: tokenizer).located
        var manifest = RunManifest(
            runID: runID, preset: options.preset, model: config, tokenizer: tokenizer.ref,
            corpus: CorpusRef(
                slug: snapshot.slug, corpusHash: snapshot.corpusHash,
                snapshotPath: snapshotDirectory.appendingPathComponent(CorpusSnapshot.fileName).path, source: snapshot.source,
                threadID: snapshot.threadID, owner: snapshot.owner, group: snapshot.group, documentCount: tokenized.documents.count,
                partitionCount: tokenized.partitions.count, tokenCount: tokenized.tokenCount),
            hyperparameters: hyper, provenance: provenance)
        if VocabularyPack.isFrozen(model) { manifest.vocabularySHA256 = VocabularyPack.fingerprint(of: model) }
        let trainer = Pretrainer(
            model: model, corpus: tokenized, tokenizer: tokenizer, facts: located, hyper: hyper, provenance: provenance,
            runDirectory: runDirectory, manifest: manifest)
        var result = try trainer.run(onEvent: progress)
        if result.status == .training {
            result.status = .complete
            try result.save(to: runDirectory)
        }
        return result
    }

    static func train(
        _ arm: Arm, corpus: GeneratedCorpus, snapshot: CorpusSnapshot, snapshotDirectory: URL, tokenized: TokenizedCorpus,
        config: RaoLMConfig, options: VocabularyBenchOptions, tokenizer: RaoTokenizer, progress: (String) -> Void
    ) async throws -> VocabularyBenchRow {
        let slug = arm.name.replacingOccurrences(of: " ", with: "-").replacingOccurrences(of: ",", with: "")
            .replacingOccurrences(of: "×", with: "x")
        let runDirectory = options.directory.appendingPathComponent(slug)
        let trainable = arm.model.trainableParameters().flattened().reduce(0) { $0 + $1.1.size }
        let before = VocabularyPack.fingerprint(of: arm.model)
        progress("\(arm.name): training \(trainable) parameters (vocabulary \(arm.vocabulary))")

        var seconds = 0.0
        var reached: (epoch: Int, seconds: Double)?
        let manifest = try pretrain(
            arm.model, runID: slug, runDirectory: runDirectory, corpus: corpus, snapshot: snapshot,
            snapshotDirectory: snapshotDirectory, tokenized: tokenized, config: config, options: options, tokenizer: tokenizer
        ) { event in
            guard case .epoch(let record) = event else { return }
            seconds += record.wallClockSeconds
            if let memorised = record.evalMemorisedFraction {
                if reached == nil, memorised >= options.target { reached = (record.epoch, seconds) }
                progress(String(
                    format: "  %@ epoch %d  eval loss %.3f  memorised %.1f%%  (%.1f s)", arm.name, record.epoch,
                    record.evalLoss ?? 0, memorised * 100, seconds))
            }
        }
        if VocabularyPack.isFrozen(arm.model), VocabularyPack.fingerprint(of: arm.model) != before {
            throw VocabularyError.fingerprint(expected: before, found: VocabularyPack.fingerprint(of: arm.model))
        }
        let last = manifest.epochs.last { $0.evalLoss != nil }

        let context = try RunContext.load(runDirectory: runDirectory, allowWeakIndex: true, tokenizer: tokenizer)
        let evaluator = FactEvaluator(
            context: context, corpus: tokenized, facts: corpus.facts, reader: InMemoryCorpusReader(snapshot: snapshot))
        let report = try await evaluator.run(options: EvalOptions(
            factsSample: options.factsSample, lambdas: [0, 0.5], primaryLambda: 0.5, seed: 7, includeControls: false))
        let primary = report.metrics(lambda: 0.5)
        let row = VocabularyBenchRow(
            name: arm.name, vocabulary: arm.vocabulary, trainableParameters: trainable, epochsToTarget: reached?.epoch,
            secondsToTarget: reached?.seconds, epochs: manifest.epochs.count, seconds: seconds, evalLoss: last?.evalLoss ?? .nan,
            memorised: last?.evalMemorisedFraction ?? 0, exact: primary?.exactAnswer ?? 0,
            exactEvidenceOnly: report.metrics(lambda: 0)?.exactAnswer ?? 0, citationAt1: primary?.citationAt1Partition ?? 0,
            runDirectory: runDirectory.path)
        progress(String(
            format: "%@: %@ · eval loss %.3f · memorised %.1f%% · exact %.0f%% (evidence only %.0f%%) · citation@1 %.0f%%", arm.name,
            reached.map { "target at epoch \($0.epoch), \(String(format: "%.1f", $0.seconds)) s" } ?? "target not reached in \(manifest.epochs.count) epochs",
            row.evalLoss, row.memorised * 100, row.exact * 100, row.exactEvidenceOnly * 100, row.citationAt1 * 100))
        return row
    }
}
