//
//  Pretrainer.swift
//  RaoLMTraining
//
//  WHAT: Pretrains a RaoTransformer from scratch on a tokenized Thread corpus, recording the
//        entropy ledger as it goes, checkpointing every epoch, running the deterministic eval
//        pass on eval epochs, and building the provenance index on the final epoch.
//  PIN:  Eager, not `compile`: compile bakes the Swift Float learning rate into the trace,
//        so a per-step schedule would silently freeze. AdamW with bias correction, as in
//        PyTorch/nanotron. Optimizer state is not checkpointed (MLXOptimizers keeps it
//        internal), and run.json says so.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import RaoLMCore
import RaoLMModel

public enum TrainingEvent {
    case started(totalSteps: Int, stepsPerEpoch: [Int], tokensPerStep: Int)
    case step(StepRow)
    case epoch(EpochRecord)
    case indexed(epoch: Int, entries: Int, sha256: String, seconds: Double)
    case earlyStop(epoch: Int, memorisedFraction: Float)
    case message(String)
}

public final class Pretrainer {
    public let model: RaoTransformer
    public let corpus: TokenizedCorpus
    public let tokenizer: RaoTokenizer
    public let facts: [LocatedFact]
    public let hyper: TrainingHyperparameters
    public let provenance: ProvenanceSettings
    public let runDirectory: URL
    public private(set) var manifest: RunManifest

    public init(
        model: RaoTransformer, corpus: TokenizedCorpus, tokenizer: RaoTokenizer, facts: [LocatedFact],
        hyper: TrainingHyperparameters, provenance: ProvenanceSettings, runDirectory: URL, manifest: RunManifest
    ) {
        self.model = model
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.facts = facts
        self.hyper = hyper
        self.provenance = provenance
        self.runDirectory = runDirectory
        self.manifest = manifest
    }

    public static func schedule(hyper: TrainingHyperparameters, totalSteps: Int) -> (Int) -> Float {
        let warmup = max(1, Int(Float(totalSteps) * hyper.warmupFraction))
        return joinSchedules(
            [
                linearSchedule(hyper.peakLR * 0.01, end: hyper.peakLR, steps: warmup),
                cosineDecay(hyper.peakLR, decaySteps: max(1, totalSteps - warmup), end: hyper.finalLR),
            ],
            boundaries: [warmup])
    }

    @discardableResult
    public func run(onEvent: (TrainingEvent) -> Void = { _ in }) throws -> RunManifest {
        let sampler = BatchSampler(seqLen: hyper.seqLen, batchSize: hyper.batchSize, seed: hyper.seed)
        let plans = (1...hyper.epochs).map { sampler.plan(epoch: $0, streamCount: corpus.stream.count) }
        let totalSteps = plans.reduce(0) { $0 + $1.count }
        let schedule = Self.schedule(hyper: hyper, totalSteps: totalSteps)
        let optimizer = AdamW(
            learningRate: hyper.peakLR, betas: (hyper.beta1, hyper.beta2), eps: hyper.eps,
            weightDecay: hyper.weightDecay, biasCorrection: hyper.biasCorrection)
        let lossAndGrad = RaoLoss.makeLossAndGrad(model: model)
        let ledger = try LedgerRecorder(directory: RunLayout.ledger(runDirectory), rowCount: corpus.partitions.count)
        defer { ledger.close() }

        manifest.status = .training
        try manifest.save(to: runDirectory)
        onEvent(.started(totalSteps: totalSteps, stepsPerEpoch: plans.map(\.count), tokensPerStep: hyper.batchSize * hyper.seqLen))

        var globalStep = 0
        let runStart = Date()
        for epoch in 1...hyper.epochs {
            let epochStart = Date()
            var epochLoss: [Float] = []
            var epochEntropy: [Float] = []
            for (step, starts) in plans[epoch - 1].enumerated() {
                let stepStart = Date()
                let lr = schedule(globalStep)
                optimizer.learningRate = lr
                let batch = sampler.makeBatch(starts, corpus: corpus)
                let (outputs, gradients) = lossAndGrad(model, [batch.inputs, batch.targets, batch.mask])
                let (clipped, norm) = clipGradNorm(gradients: gradients, maxNorm: hyper.gradClip)
                optimizer.update(model: model, gradients: clipped)
                eval(model, optimizer, outputs[0], outputs[1], outputs[2], norm)

                let loss = outputs[0].item(Float.self)
                let perToken = outputs[1].asArray(Float.self)
                let entropy = outputs[2].asArray(Float.self)
                let gradNorm = norm.item(Float.self)
                var maskedEntropy: [Float] = []
                maskedEntropy.reserveCapacity(batch.maskedCount)
                for i in batch.rows.indices where batch.rows[i] >= 0 { maskedEntropy.append(entropy[i]) }
                let summary = EntropySummary(values: maskedEntropy)
                let seconds = Date().timeIntervalSince(stepStart)
                let row = StepRow(
                    epoch: epoch, step: step, globalStep: globalStep, lr: lr, loss: loss, entropy: summary,
                    lossMinusEntropy: loss - summary.mean, gradNorm: gradNorm, clipped: gradNorm > hyper.gradClip,
                    tokens: hyper.batchSize * hyper.seqLen, maskedTokens: batch.maskedCount,
                    tokensPerSecond: Double(batch.maskedCount) / max(seconds, 1e-9),
                    wallClockSeconds: Date().timeIntervalSince(runStart))
                try ledger.recordStep(row, perTokenLoss: perToken, entropy: entropy, rows: batch.rows)
                epochLoss.append(loss)
                epochEntropy.append(summary.mean)
                onEvent(.step(row))
                globalStep += 1
            }

            // End of epoch: checkpoint, eval, ledger, index.
            let checkpointDirectory = RunLayout.checkpoint(runDirectory, epoch: epoch)
            let checkpointSHA = try Checkpoint.save(
                model: model, to: checkpointDirectory,
                metadata: ["raolm_run": manifest.runID, "raolm_epoch": String(epoch)],
                tokenizerDirectory: tokenizer.directory)
            let isFinal = epoch == hyper.epochs
            let isEval = isFinal || (hyper.evalEvery > 0 && epoch % hyper.evalEvery == 0)
            let isIndexEpoch = isFinal || (hyper.indexEvery > 0 && epoch % hyper.indexEvery == 0)
            var evalResult: EvalResult?
            if isEval || isIndexEpoch {
                evalResult = EvalPass.run(
                    model: model, corpus: corpus, seqLen: hyper.seqLen, batchSize: hyper.evalBatchSize,
                    captureKeys: true, alpha: provenance.alpha)
            }
            let partitionLedgerSHA = try ledger.finishEpoch(epoch: epoch, corpus: corpus, eval: evalResult, facts: facts)
            let memorised = evalResult?.memorisedFraction
            let stopEarly = !isFinal && memorised.map { value in hyper.earlyStopMemorised.map { value >= $0 } ?? false } ?? false

            var record = EpochRecord(
                epoch: epoch, steps: plans[epoch - 1].count, trainLoss: Stats.mean(epochLoss),
                trainEntropy: Stats.mean(epochEntropy), evalLoss: evalResult?.meanLoss,
                evalEntropy: evalResult?.meanEntropy, evalMemorisedFraction: memorised,
                calibrationGap: evalResult.map { $0.meanLoss - $0.meanEntropy },
                checkpointPath: checkpointDirectory.path, checkpointSHA256: checkpointSHA,
                partitionLedgerSHA256: partitionLedgerSHA, wallClockSeconds: Date().timeIntervalSince(epochStart))

            if let evalResult, isIndexEpoch || stopEarly {
                manifest.status = .indexing
                let indexStart = Date()
                let directory = RunLayout.provenance(runDirectory, epoch: epoch)
                let info = IndexInfo(
                    epoch: epoch, tapLayer: model.tapLayer, alpha: provenance.alpha,
                    keyDims: ProvenanceKey.dimensions(for: model.config), count: evalResult.indexableCount,
                    checkpointSHA256: checkpointSHA, corpusHash: corpus.corpusHash,
                    tokenizerSHA256: tokenizer.tokenizerSHA256, threadID: corpus.threadID,
                    evalLoss: evalResult.meanLoss, evalMemorisedFraction: evalResult.memorisedFraction)
                let indexSHA = try ProvenanceIndexer.write(
                    eval: evalResult, corpus: corpus, info: info, memorisedAtEpoch: ledger.memorisedAtEpoch, to: directory)
                record.indexPath = directory.path
                record.indexSHA256 = indexSHA
                manifest.indexedEpochs.append(epoch)
                onEvent(.indexed(epoch: epoch, entries: evalResult.indexableCount, sha256: indexSHA,
                                 seconds: Date().timeIntervalSince(indexStart) + evalResult.seconds))
                manifest.status = .training
            }

            try ledger.recordEpoch(record)
            manifest.epochs.append(record)
            try pruneCheckpoints(current: epoch)
            try manifest.save(to: runDirectory)
            onEvent(.epoch(record))

            if stopEarly, let memorised {
                onEvent(.earlyStop(epoch: epoch, memorisedFraction: memorised))
                break
            }
        }

        manifest.stepsLedgerSHA256 = try ledger.stepsSHA256()
        try manifest.save(to: runDirectory)
        return manifest
    }

    /// Keeps the last `keepCheckpoints` epochs plus every indexed epoch.
    private func pruneCheckpoints(current: Int) throws {
        let keep = Set(manifest.indexedEpochs).union(
            Set(manifest.epochs.suffix(max(1, hyper.keepCheckpoints)).map(\.epoch))).union([current])
        for index in manifest.epochs.indices where !keep.contains(manifest.epochs[index].epoch) {
            guard let path = manifest.epochs[index].checkpointPath else { continue }
            if FileManager.default.fileExists(atPath: path) {
                try FileManager.default.removeItem(atPath: path)
            }
            manifest.epochs[index].checkpointPath = nil
        }
    }
}
