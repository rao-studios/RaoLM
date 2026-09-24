//
//  TrainCommand.swift
//  RaoLMCLI
//
//  WHAT: raolm train — pretrain on a corpus snapshot with the entropy ledger and provenance index.
//

import ArgumentParser
import Foundation
import RaoLM
import RaoLMWorkflows

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

    func settings() -> TrainingSettings {
        var settings = TrainingSettings()
        settings.preset = preset
        settings.epochs = epochs
        settings.batchSize = batchSize
        settings.seqLen = seqLen
        settings.lr = lr
        settings.evalEvery = evalEvery
        settings.indexEvery = indexEvery
        settings.tapLayer = tapLayer
        settings.alpha = alpha
        settings.seed = seed
        settings.earlyStop = earlyStop
        settings.excludeDocuments = excludeDocuments
        settings.keepCheckpoints = keepCheckpoints
        return settings
    }

    func hyperparameters() -> TrainingHyperparameters { settings().hyperparameters() }
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
            try Preflight.requireMetallib()
            let settings = options.settings()
            _ = try settings.modelConfig()
            let tokenizer = try await RaoTokenizer.load()
            let result = try TrainingDriver.train(
                TrainingDriver.Plan(
                    snapshot: snapshot, snapshotPath: snapshotURL.path, factsPath: factsURL, settings: settings,
                    runDirectory: runDirectory, runID: runID, thread: nil),
                tokenizer: tokenizer)
            print("run \(result.manifest.runID) complete: \(result.runDirectory.path)")
        }
    }
}
