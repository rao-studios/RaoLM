//
//  TrainingDriver.swift
//  RaoLMWorkflows
//
//  WHAT: Pretraining as the CLI and the studio both run it: settings → a prepared run (model,
//        tokenized corpus, located facts, run.json on disk) → the training loop with events.
//  PIN:  Synchronous on purpose: the caller owns the thread MLX runs on (the command's task, or
//        the studio's MLX worker). `TrainingConsolePrinter` prints exactly what `raolm train`
//        and `raolm demo` always printed, so the CLI's output does not change.
//

import Foundation
import RaoLM

public struct TrainingSettings: Sendable, Equatable {
    public var preset = "tiny"
    public var epochs = 60
    public var batchSize = 4
    public var seqLen = 512
    public var lr: Float = 2e-3
    public var evalEvery = 10
    public var indexEvery = 0
    public var tapLayer: Int? = nil
    public var alpha: Float = 0.5
    public var seed: UInt64 = 42
    public var earlyStop: Float = 0.98
    public var excludeDocuments = 0
    public var keepCheckpoints = 3

    public init() {}

    public func hyperparameters() -> TrainingHyperparameters {
        TrainingHyperparameters(
            batchSize: batchSize, seqLen: seqLen, epochs: epochs, peakLR: lr, seed: seed, evalEvery: evalEvery,
            indexEvery: indexEvery, keepCheckpoints: keepCheckpoints, earlyStopMemorised: earlyStop > 0 ? earlyStop : nil)
    }

    /// The preset's model config, validated: the check that should fail before anything loads.
    public func modelConfig() throws -> RaoLMConfig {
        let config = try RaoLMConfig.preset(preset)
        try config.validate()
        return config
    }
}

public enum TrainingDriver {
    public struct Plan: Sendable {
        public var snapshot: CorpusSnapshot
        public var snapshotPath: String
        /// facts.jsonl of the generated corpus; enables the per-fact ledger.
        public var factsPath: URL?
        public var settings: TrainingSettings
        public var runDirectory: URL
        public var runID: String
        public var thread: ThreadRef?

        public init(
            snapshot: CorpusSnapshot, snapshotPath: String, factsPath: URL?, settings: TrainingSettings,
            runDirectory: URL, runID: String, thread: ThreadRef?
        ) {
            self.snapshot = snapshot
            self.snapshotPath = snapshotPath
            self.factsPath = factsPath
            self.settings = settings
            self.runDirectory = runDirectory
            self.runID = runID
            self.thread = thread
        }
    }

    public struct Result: Sendable {
        public var manifest: RunManifest
        public var runDirectory: URL
    }

    /// A run ready to train. Holds the MLX model: keep it on the thread that made it.
    public final class Prepared {
        public let manifest: RunManifest
        public let hyper: TrainingHyperparameters
        public let runDirectory: URL
        /// "model …", "corpus: …" and, when some facts did not align, "  (… did not align …)".
        public let summaryLines: [String]
        private let trainer: Pretrainer

        init(manifest: RunManifest, hyper: TrainingHyperparameters, runDirectory: URL, summaryLines: [String], trainer: Pretrainer) {
            self.manifest = manifest
            self.hyper = hyper
            self.runDirectory = runDirectory
            self.summaryLines = summaryLines
            self.trainer = trainer
        }

        /// Trains; a run that finishes (rather than stopping) is marked `complete` in run.json.
        public func run(shouldStop: () -> Bool = { false }, onEvent: (TrainingEvent) -> Void) throws -> Result {
            var manifest = try trainer.run(shouldStop: shouldStop, onEvent: onEvent)
            if manifest.status == .training {
                manifest.status = .complete
                try manifest.save(to: runDirectory)
            }
            return Result(manifest: manifest, runDirectory: runDirectory)
        }
    }

    public static func prepare(_ plan: Plan, tokenizer: RaoTokenizer) throws -> Prepared {
        try Preflight.requireMetallib()
        let settings = plan.settings
        let config = try settings.modelConfig()
        let hyper = settings.hyperparameters()
        let snapshot = plan.snapshot

        var excluded: Set<String> = []
        if settings.excludeDocuments > 0 {
            var rng = SplitMix64.derived(seed: hyper.seed, stream: 0xEC)
            excluded = Set(rng.shuffled(snapshot.documents.map(\.id)).prefix(settings.excludeDocuments))
        }
        let corpus = TokenizedCorpus(snapshot: snapshot, tokenizer: tokenizer, excluding: excluded)
        guard corpus.tokenCount > hyper.seqLen else {
            throw RaoLMFailure("the corpus has only \(corpus.tokenCount) tokens; --seq-len \(hyper.seqLen) needs more", code: 65)
        }
        var facts: [Fact] = []
        if let factsPath = plan.factsPath, FileManager.default.fileExists(atPath: factsPath.path) {
            facts = try JSONCoding.readLines(Fact.self, from: factsPath)
        }
        let located = FactLocator.locate(facts, corpus: corpus, tokenizer: tokenizer)

        let provenance = ProvenanceSettings(tapLayer: settings.tapLayer ?? config.numHiddenLayers / 2, alpha: settings.alpha)
        let model = try RaoTransformer.make(config: config, seed: hyper.seed, tapLayer: provenance.tapLayer)
        let manifest = RunManifest(
            runID: plan.runID, preset: settings.preset, model: config, tokenizer: tokenizer.ref,
            corpus: CorpusRef(
                slug: snapshot.slug, corpusHash: snapshot.corpusHash, snapshotPath: plan.snapshotPath, source: snapshot.source,
                threadID: snapshot.threadID, owner: snapshot.owner, group: snapshot.group,
                documentCount: corpus.documents.count, partitionCount: corpus.partitions.count,
                tokenCount: corpus.tokenCount, factsPath: plan.factsPath?.path),
            hyperparameters: hyper, provenance: provenance, thread: plan.thread, excludedDocumentIDs: excluded.sorted())
        try FileManager.default.createDirectory(at: plan.runDirectory, withIntermediateDirectories: true)
        try manifest.save(to: plan.runDirectory)

        var lines = [
            "model \(settings.preset): \(Format.count(config.parameterCount)) parameters, tap layer \(provenance.tapLayer)",
            "corpus: \(corpus.documents.count) documents, \(corpus.partitions.count) partitions, \(Format.count(corpus.tokenCount)) tokens, \(located.located.count) facts located"
                + (excluded.isEmpty ? "" : ", \(excluded.count) documents held out"),
        ]
        if !located.unaligned.isEmpty { lines.append("  (\(located.unaligned.count) facts did not align to token boundaries)") }

        let trainer = Pretrainer(
            model: model, corpus: corpus, tokenizer: tokenizer, facts: located.located, hyper: hyper,
            provenance: provenance, runDirectory: plan.runDirectory, manifest: manifest)
        return Prepared(manifest: manifest, hyper: hyper, runDirectory: plan.runDirectory, summaryLines: lines, trainer: trainer)
    }

    /// The CLI path: prepare, print the summary, train with the console printer.
    public static func train(_ plan: Plan, tokenizer: RaoTokenizer, quiet: Bool = false,
                             shouldStop: () -> Bool = { false }) throws -> Result {
        let prepared = try prepare(plan, tokenizer: tokenizer)
        if !quiet { for line in prepared.summaryLines { print(line) } }
        let printer = TrainingConsolePrinter(hyper: prepared.hyper, quiet: quiet)
        return try prepared.run(shouldStop: shouldStop) { printer.handle($0) }
    }
}

/// Training progress as plain lines: the first three steps, then one step line every 5 s, and
/// every epoch, index and early stop.
public final class TrainingConsolePrinter {
    private let hyper: TrainingHyperparameters
    private let quiet: Bool
    private let out: (String) -> Void
    private let now: () -> Date
    private var lastReport: Date
    private var stepsPerEpoch = 0

    public init(hyper: TrainingHyperparameters, quiet: Bool = false, out: @escaping (String) -> Void = { print($0) },
                now: @escaping () -> Date = Date.init) {
        self.hyper = hyper
        self.quiet = quiet
        self.out = out
        self.now = now
        self.lastReport = now()
    }

    public func handle(_ event: TrainingEvent) {
        guard !quiet else {
            if case .started(_, let perEpoch, _) = event { stepsPerEpoch = perEpoch.first ?? 0 }
            return
        }
        for line in lines(for: event) { out(line) }
    }

    public func lines(for event: TrainingEvent) -> [String] {
        switch event {
        case .started(let total, let perEpoch, let tokens):
            stepsPerEpoch = perEpoch.first ?? 0
            return ["training: \(total) steps (\(stepsPerEpoch)/epoch × \(hyper.epochs) epochs), \(Format.count(tokens)) tokens/step"]
        case .step(let row):
            let current = now()
            guard current.timeIntervalSince(lastReport) > 5 || row.globalStep < 3 else { return [] }
            lastReport = current
            return [Self.stepLine(row, stepsPerEpoch: stepsPerEpoch)]
        case .epoch(let record):
            return [Self.epochLine(record)]
        case .indexed(let epoch, let entries, let sha, let seconds):
            return ["provenance index for epoch \(epoch): \(Format.count(entries)) entries, sha \(Format.short(sha)), \(Format.duration(seconds))"]
        case .earlyStop(let epoch, let memorised):
            return [String(format: "early stop after epoch %d: %.1f%% of positions memorised", epoch, memorised * 100)]
        case .message(let text):
            return [text]
        }
    }

    public static func stepLine(_ row: StepRow, stepsPerEpoch: Int) -> String {
        String(
            format: "  epoch %d step %d/%d  loss %.3f  entropy %.3f  lr %.2e  grad %.2f  %.0f tok/s",
            row.epoch, row.step + 1, stepsPerEpoch, row.loss, row.entropy.mean, row.lr, row.gradNorm, row.tokensPerSecond)
    }

    public static func epochLine(_ record: EpochRecord) -> String {
        var line = String(format: "epoch %d  train loss %.3f  entropy %.3f", record.epoch, record.trainLoss, record.trainEntropy)
        if let evalLoss = record.evalLoss, let memorised = record.evalMemorisedFraction {
            line += String(format: "  eval loss %.3f  memorised %.1f%%", evalLoss, memorised * 100)
        }
        line += "  (\(Format.duration(record.wallClockSeconds)))"
        return line
    }
}
