//
//  TrainCommand.swift
//  RaoLMCLI
//
//  WHAT: raolm train — pretrain on a corpus snapshot with the entropy ledger and provenance index.
//

import ArgumentParser
import Foundation
import RaoLM

struct TrainOptions: ParsableArguments {
    @Option(help: "Model preset: tiny, small, smollm2-135m.")
    var preset = "tiny"

    @Option(help: "Epochs.")
    var epochs = 60

    @Option(help: "Sequences per optimizer step.")
    var batchSize = 4

    @Option(help: "Tokens per sequence.")
    var seqLen = 512

    @Option(help: "Peak learning rate.")
    var lr: Float = 2e-3

    @Option(help: "Run the deterministic eval pass every N epochs (the final epoch always).")
    var evalEvery = 10

    @Option(help: "Build a provenance index every N epochs (0: final epoch only).")
    var indexEvery = 0

    @Option(help: "Block whose residual output is the mid-layer half of the provenance key (default: layers/2).")
    var tapLayer: Int?

    @Option(help: "Weight of the mid-layer half of the provenance key.")
    var alpha: Float = 0.5

    @Option(help: "Seed for init, data order and sampling.")
    var seed: UInt64 = 42

    @Option(help: "Stop early once an eval pass memorises this fraction of positions (0 disables).")
    var earlyStop: Float = 0.98

    @Option(help: "Exclude this many documents from training (leave-out control).")
    var excludeDocuments = 0

    @Option(help: "Checkpoints to keep besides indexed epochs.")
    var keepCheckpoints = 3

    func hyperparameters() -> TrainingHyperparameters {
        TrainingHyperparameters(
            batchSize: batchSize, seqLen: seqLen, epochs: epochs, peakLR: lr, seed: seed, evalEvery: evalEvery,
            indexEvery: indexEvery, keepCheckpoints: keepCheckpoints, earlyStopMemorised: earlyStop > 0 ? earlyStop : nil)
    }
}

enum TrainingDriver {
    struct Result {
        var manifest: RunManifest
        var runDirectory: URL
    }

    /// Trains on a snapshot; `factsPath` (facts.jsonl) enables the per-fact ledger.
    static func train(
        snapshot: CorpusSnapshot, snapshotPath: String, factsPath: URL?, options: TrainOptions, runDirectory: URL,
        runID: String, thread: ThreadRef?, quiet: Bool = false
    ) async throws -> Result {
        try Preflight.requireMetallib()
        let config = try RaoLMConfig.preset(options.preset)
        try config.validate()
        let tokenizer = try await RaoTokenizer.load()
        let hyper = options.hyperparameters()

        var excluded: Set<String> = []
        if options.excludeDocuments > 0 {
            var rng = SplitMix64.derived(seed: hyper.seed, stream: 0xEC)
            excluded = Set(rng.shuffled(snapshot.documents.map(\.id)).prefix(options.excludeDocuments))
        }
        let corpus = TokenizedCorpus(snapshot: snapshot, tokenizer: tokenizer, excluding: excluded)
        guard corpus.tokenCount > hyper.seqLen else {
            throw RaoLMFailure("the corpus has only \(corpus.tokenCount) tokens; --seq-len \(hyper.seqLen) needs more", code: 65)
        }
        var facts: [Fact] = []
        if let factsPath, FileManager.default.fileExists(atPath: factsPath.path) {
            facts = try JSONCoding.readLines(Fact.self, from: factsPath)
        }
        let located = FactLocator.locate(facts, corpus: corpus, tokenizer: tokenizer)

        let provenance = ProvenanceSettings(tapLayer: options.tapLayer ?? config.numHiddenLayers / 2, alpha: options.alpha)
        let model = try RaoTransformer.make(config: config, seed: hyper.seed, tapLayer: provenance.tapLayer)
        let manifest = RunManifest(
            runID: runID, preset: options.preset, model: config, tokenizer: tokenizer.ref,
            corpus: CorpusRef(
                slug: snapshot.slug, corpusHash: snapshot.corpusHash, snapshotPath: snapshotPath, source: snapshot.source,
                threadID: snapshot.threadID, owner: snapshot.owner, group: snapshot.group,
                documentCount: corpus.documents.count, partitionCount: corpus.partitions.count,
                tokenCount: corpus.tokenCount, factsPath: factsPath?.path),
            hyperparameters: hyper, provenance: provenance, thread: thread, excludedDocumentIDs: excluded.sorted())
        try FileManager.default.createDirectory(at: runDirectory, withIntermediateDirectories: true)
        try manifest.save(to: runDirectory)

        if !quiet {
            print("model \(options.preset): \(Format.count(config.parameterCount)) parameters, tap layer \(provenance.tapLayer)")
            print("corpus: \(corpus.documents.count) documents, \(corpus.partitions.count) partitions, \(Format.count(corpus.tokenCount)) tokens, \(located.located.count) facts located" + (excluded.isEmpty ? "" : ", \(excluded.count) documents held out"))
            if !located.unaligned.isEmpty { print("  (\(located.unaligned.count) facts did not align to token boundaries)") }
        }

        let trainer = Pretrainer(
            model: model, corpus: corpus, tokenizer: tokenizer, facts: located.located, hyper: hyper,
            provenance: provenance, runDirectory: runDirectory, manifest: manifest)
        var lastReport = Date()
        var stepsPerEpoch = 0
        let result = try trainer.run { event in
            switch event {
            case .started(let total, let perEpoch, let tokens):
                stepsPerEpoch = perEpoch.first ?? 0
                if !quiet { print("training: \(total) steps (\(stepsPerEpoch)/epoch × \(hyper.epochs) epochs), \(Format.count(tokens)) tokens/step") }
            case .step(let row):
                if !quiet, Date().timeIntervalSince(lastReport) > 5 || row.globalStep < 3 {
                    lastReport = Date()
                    print(String(
                        format: "  epoch %d step %d/%d  loss %.3f  entropy %.3f  lr %.2e  grad %.2f  %.0f tok/s",
                        row.epoch, row.step + 1, stepsPerEpoch, row.loss, row.entropy.mean, row.lr, row.gradNorm, row.tokensPerSecond))
                }
            case .epoch(let record):
                if !quiet {
                    var line = String(format: "epoch %d  train loss %.3f  entropy %.3f", record.epoch, record.trainLoss, record.trainEntropy)
                    if let evalLoss = record.evalLoss, let memorised = record.evalMemorisedFraction {
                        line += String(format: "  eval loss %.3f  memorised %.1f%%", evalLoss, memorised * 100)
                    }
                    line += "  (\(Format.duration(record.wallClockSeconds)))"
                    print(line)
                }
            case .indexed(let epoch, let entries, let sha, let seconds):
                if !quiet { print("provenance index for epoch \(epoch): \(Format.count(entries)) entries, sha \(Format.short(sha)), \(Format.duration(seconds))") }
            case .earlyStop(let epoch, let memorised):
                if !quiet { print(String(format: "early stop after epoch %d: %.1f%% of positions memorised", epoch, memorised * 100)) }
            case .message(let text):
                if !quiet { print(text) }
            }
        }
        return Result(manifest: result, runDirectory: runDirectory)
    }
}

struct Train: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Pretrain from scratch on a corpus snapshot; records the ledger and builds the provenance index.")

    @OptionGroup var global: GlobalOptions

    @Option(help: "Snapshot directory or snapshot.json (from raolm corpus pull).")
    var corpus: String

    @Option(help: "facts.jsonl of the generated corpus (default: found next to the snapshot's corpus).")
    var facts: String?

    @Option(help: "Run directory (default: <data root>/runs/<timestamp>-<preset>).")
    var out: String?

    @OptionGroup var options: TrainOptions

    func run() async throws {
        try await guarded {
            let snapshotURL = URL(fileURLWithPath: corpus)
            let snapshot = try CorpusSnapshot.load(from: snapshotURL)
            let runID = DataRoot.newRunID(preset: options.preset)
            let runDirectory = out.map { URL(fileURLWithPath: $0) } ?? global.root.run(id: runID)
            var factsURL = facts.map { URL(fileURLWithPath: $0) }
            if factsURL == nil, let slug = snapshot.slug {
                let candidate = global.root.corpus(slug: slug).appendingPathComponent("facts.jsonl")
                if FileManager.default.fileExists(atPath: candidate.path) { factsURL = candidate }
            }
            let result = try await TrainingDriver.train(
                snapshot: snapshot, snapshotPath: snapshotURL.path, factsPath: factsURL, options: options,
                runDirectory: runDirectory, runID: runID, thread: nil)
            print("run \(result.manifest.runID) complete: \(result.runDirectory.path)")
        }
    }
}
