//
//  CommonsTrainer.swift
//  RaoLMTraining
//
//  WHAT: Continued pretraining of the commons: the umbrella pack's whole base model, every block
//        and the embedding, trained on public corpora the owner chose, so Rao builds its own
//        foundation model from open weights. The result becomes a new pack (its parent recorded);
//        Threads retrain from it when the owner rebases a braid onto it.
//  OUT:  <runs>/<runID>/spec.json, step-NNNNNN/ (model.safetensors, optimizer.safetensors,
//        state.json; the last two kept), and the trained model.
//  PIN:  Nothing is frozen. The tied head is the umbrella's readout: a commons that cannot move
//        its output geometry cannot become Rao's, and keeping the vocabulary fixed buys nothing
//        once the trunk changes, since every Thread rebases anyway. Warmup, a stable peak, then a
//        linear anneal over the last `annealSteps` to `floorLR`; the norms' gains are not decayed.
//        Cross-entropy on non-eos inputs,
//        attention inside a document. Eager, like the Pretrainer (compile bakes the rate in).
//        Resume reloads the model, the moments and the step, and replays the same plan.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import RaoLMCore
import RaoLMModel

public struct CommonsTrainingSpec: Codable, Sendable, Equatable {
    public var runID: String
    public var parentSHA256: String
    public var tokenizerSHA256: String
    public var corpora: [PackRecipe.Corpus]
    public var steps: Int
    public var seqLen: Int
    public var batch: Int
    /// Windows per forward pass; a step accumulates batch / microBatch of them. Nil: the whole batch at once.
    public var microBatch: Int?
    public var peakLR: Float
    public var floorLR: Float
    public var warmupSteps: Int
    public var annealSteps: Int
    public var weightDecay: Float
    public var gradClip: Float
    public var beta2: Float
    public var seed: UInt64
    public var evalEvery: Int
    public var checkpointEvery: Int
    public var heldOutWindows: Int

    public init(
        runID: String, parentSHA256: String, tokenizerSHA256: String, corpora: [PackRecipe.Corpus], tokens: Int, seqLen: Int = 1024, batch: Int = 8,
        microBatch: Int? = nil, peakLR: Float = 3e-4, floorFraction: Float = 0.1, warmupSteps: Int = 100, annealFraction: Float = 0.2, weightDecay: Float = 0.1,
        gradClip: Float = 1, beta2: Float = 0.95, seed: UInt64 = 0xC0_4404, evalEvery: Int = 250, checkpointEvery: Int = 250, heldOutWindows: Int = 32
    ) {
        self.runID = runID
        self.parentSHA256 = parentSHA256
        self.tokenizerSHA256 = tokenizerSHA256
        self.corpora = corpora
        self.seqLen = seqLen
        self.batch = batch
        self.microBatch = microBatch.flatMap { $0 > 0 && $0 < batch ? $0 : nil }
        steps = max(1, tokens / (seqLen * batch))
        self.peakLR = peakLR
        floorLR = peakLR * floorFraction
        self.warmupSteps = min(warmupSteps, max(1, steps / 10))
        annealSteps = max(1, Int(Float(steps) * annealFraction))
        self.weightDecay = weightDecay
        self.gradClip = gradClip
        self.beta2 = beta2
        self.seed = seed
        self.evalEvery = evalEvery
        self.checkpointEvery = checkpointEvery
        self.heldOutWindows = heldOutWindows
    }

    public var tokens: Int { steps * seqLen * batch }

    /// The rate at step `step` (1-based): warmup from 1% of the peak, the peak held, a linear anneal to the floor.
    public func rate(_ step: Int) -> Float {
        if step <= warmupSteps { return peakLR * (0.01 + 0.99 * Float(step) / Float(warmupSteps)) }
        let annealFrom = steps - annealSteps
        guard step > annealFrom else { return peakLR }
        let progress = min(1, Float(step - annealFrom) / Float(annealSteps))
        return peakLR + (floorLR - peakLR) * progress
    }
}

public struct CommonsTrainingState: Codable, Sendable, Equatable {
    public struct Eval: Codable, Sendable, Equatable {
        public var step: Int
        public var losses: [String: Float]
    }

    public var step = 0
    public var trainLoss: [Float] = []
    public var evals: [Eval] = []
    public var seconds: Double = 0
}

public enum CommonsTrainingEvent {
    case step(step: Int, of: Int, loss: Float, rate: Float, tokensPerSecond: Double)
    case eval(CommonsTrainingState.Eval)
    case checkpoint(URL)
    case resumed(step: Int)
}

public enum CommonsTrainerError: Error, CustomStringConvertible {
    case specMismatch(String)

    public var description: String {
        switch self {
        case .specMismatch(let what): return "the run on disk was made with another spec (\(what)); start a new run instead of resuming"
        }
    }
}

public final class CommonsTrainer {
    public static let specFile = "spec.json"
    public static let modelFile = "model.safetensors"
    public static let optimizerFile = "optimizer.safetensors"
    public static let stateFile = "state.json"

    public let model: RaoTransformer
    public let spec: CommonsTrainingSpec
    public let stream: TokenStream
    /// Held-out sets by name: batches of (inputs, targets, mask, attention).
    public let heldOut: [String: [Batch]]
    public let directory: URL
    public private(set) var state = CommonsTrainingState()
    let optimizer: ResumableAdamW

    public init(model: RaoTransformer, spec: CommonsTrainingSpec, stream: TokenStream, heldOut: [String: [Batch]], directory: URL) {
        self.model = model
        self.spec = spec
        self.stream = stream
        self.heldOut = heldOut
        self.directory = directory
        optimizer = ResumableAdamW(learningRate: spec.peakLR, betas: (0.9, spec.beta2), weightDecay: spec.weightDecay, decayVectors: false)
        model.unfreeze()
    }

    /// The pack's held-out snippets as batches (each snippet one row; nothing masked but the first token's lack of a past).
    public static func snippetBatches(_ snippets: [[Int]], batch: Int = 8) -> [Batch] {
        let usable = snippets.filter { $0.count >= 2 }
        guard let length = usable.map(\.count).min() else { return [] }
        var batches: [Batch] = []
        var i = 0
        while i < usable.count {
            let rows = usable[i ..< min(i + batch, usable.count)].map { Array($0.prefix(length)).map(Int32.init) }
            let T = length - 1
            let inputs = rows.flatMap { $0.prefix(T) }
            let targets = rows.flatMap { $0.dropFirst() }
            batches.append(Batch(
                inputs: MLXArray(inputs, [rows.count, T]), targets: MLXArray(Array(targets), [rows.count, T]),
                mask: MLXArray([Float](repeating: 1, count: rows.count * T), [rows.count, T]), rows: [], maskedCount: rows.count * T))
            i += batch
        }
        return batches
    }

    /// Mean cross-entropy over scored tokens.
    public static func loss(model: RaoTransformer, batches: [Batch]) -> Float {
        var total: Double = 0
        var count: Double = 0
        for batch in batches {
            let logits = model.forward(batch.inputs, cache: nil, captureTap: false, attention: batch.attention).logits.asType(.float32)
            let lse = logSumExp(logits, axis: -1)
            let score = takeAlong(logits, batch.targets.expandedDimensions(axis: -1), axis: -1).squeezed(axis: -1)
            let sum = ((lse - score) * batch.mask).sum()
            eval(sum)
            total += Double(sum.item(Float.self))
            count += Double(batch.maskedCount)
        }
        return count > 0 ? Float(total / count) : .nan
    }

    public func evaluate() -> CommonsTrainingState.Eval {
        var losses: [String: Float] = [:]
        for (name, batches) in heldOut { losses[name] = Self.loss(model: model, batches: batches) }
        return CommonsTrainingState.Eval(step: state.step, losses: losses)
    }

    // MARK: - Checkpoints

    func checkpointDirectory(_ step: Int) -> URL { directory.appendingPathComponent(String(format: "step-%06d", step), isDirectory: true) }

    public static func checkpoints(in directory: URL) -> [(step: Int, url: URL)] {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
        return names.compactMap { name -> (Int, URL)? in
            guard name.hasPrefix("step-"), let step = Int(name.dropFirst(5)),
                  FileManager.default.fileExists(atPath: directory.appendingPathComponent(name).appendingPathComponent(stateFile).path)
            else { return nil }
            return (step, directory.appendingPathComponent(name, isDirectory: true))
        }.sorted { $0.0 < $1.0 }
    }

    func checkpoint() throws -> URL {
        let target = checkpointDirectory(state.step)
        let staging = directory.appendingPathComponent(".staging-\(state.step)", isDirectory: true)
        try? FileManager.default.removeItem(at: staging)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        try MLX.save(arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()), metadata: ["format": "raolm-commons"],
                     url: staging.appendingPathComponent(Self.modelFile))
        try optimizer.save(to: staging.appendingPathComponent(Self.optimizerFile))
        try JSONCoding.write(state, to: staging.appendingPathComponent(Self.stateFile))
        try? FileManager.default.removeItem(at: target)
        try FileManager.default.moveItem(at: staging, to: target)
        // Keep the last two.
        for old in Self.checkpoints(in: directory).dropLast(2) { try? FileManager.default.removeItem(at: old.url) }
        return target
    }

    /// Picks the run up from its last checkpoint, if it has one; refuses a run made with another spec.
    public func resume() throws -> Bool {
        let specURL = directory.appendingPathComponent(Self.specFile)
        if FileManager.default.fileExists(atPath: specURL.path) {
            let saved = try JSONCoding.read(CommonsTrainingSpec.self, from: specURL)
            guard saved == spec else { throw CommonsTrainerError.specMismatch(saved.runID) }
        } else {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try JSONCoding.write(spec, to: specURL)
        }
        guard let last = Self.checkpoints(in: directory).last else { return false }
        let weights = try loadArrays(url: last.url.appendingPathComponent(Self.modelFile))
        try model.update(parameters: ModuleParameters.unflattened(weights.map { ($0.key, $0.value) }), verify: [.shapeMismatch, .noUnusedKeys])
        try optimizer.load(from: last.url.appendingPathComponent(Self.optimizerFile))
        state = try JSONCoding.read(CommonsTrainingState.self, from: last.url.appendingPathComponent(Self.stateFile))
        eval(model)
        return true
    }

    // MARK: - The loop

    /// Trains to the spec's last step (or until `shouldStop`, checkpointing first). Evaluates before
    /// the first step and every `evalEvery` steps, and at the end.
    @discardableResult
    public func run(shouldStop: () -> Bool = { false }, onEvent: (CommonsTrainingEvent) -> Void = { _ in }) throws -> CommonsTrainingState {
        if try resume() { onEvent(.resumed(step: state.step)) }
        if state.evals.isEmpty {
            let first = evaluate()
            state.evals.append(first)
            onEvent(.eval(first))
        }
        let plan = stream.plan(steps: spec.steps)
        let lossAndGrad = valueAndGrad(model: model) { (model: RaoTransformer, arrays: [MLXArray]) -> [MLXArray] in
            let logits = model.forward(arrays[0], cache: nil, captureTap: false, attention: arrays[3]).logits.asType(.float32)
            let lse = logSumExp(logits, axis: -1)
            let score = takeAlong(logits, arrays[1].expandedDimensions(axis: -1), axis: -1).squeezed(axis: -1)
            let mask = arrays[2]
            return [((lse - score) * mask).sum() / MLX.maximum(mask.sum(), MLXArray(Float(1)))]
        }
        var windowStart = Date()
        var windowTokens = 0
        while state.step < spec.steps {
            if shouldStop() {
                onEvent(.checkpoint(try checkpoint()))
                return state
            }
            let step = state.step + 1
            optimizer.learningRate = spec.rate(step)
            // The step's windows in micro-batches, their gradients averaged.
            let windows = plan[step - 1]
            let size = spec.microBatch ?? windows.count
            var summed: [String: MLXArray] = [:]
            var losses: [Float] = []
            var start = 0
            while start < windows.count {
                let batch = stream.batch(Array(windows[start ..< min(start + size, windows.count)]))
                let (outputs, gradients) = lossAndGrad(model, [batch.inputs, batch.targets, batch.mask, batch.attention!])
                for (key, gradient) in gradients.flattened() { summed[key] = summed[key].map { $0 + gradient } ?? gradient }
                eval([outputs[0]] + Array(summed.values))
                losses.append(outputs[0].item(Float.self))
                start += size
            }
            let parts = Float(losses.count)
            let gradients = ModuleParameters.unflattened(summed.map { ($0.key, parts > 1 ? $0.value / parts : $0.value) })
            let (clipped, norm) = clipGradNorm(gradients: gradients, maxNorm: spec.gradClip)
            optimizer.update(model: model, gradients: clipped)
            eval([norm] + optimizer.state + model.parameters().flattened().map(\.1))
            state.step = step
            let loss = losses.reduce(0, +) / parts
            state.trainLoss.append(loss)
            windowTokens += spec.batch * spec.seqLen
            let elapsed = Date().timeIntervalSince(windowStart)
            if step % 10 == 0 || step == spec.steps {
                state.seconds += elapsed
                onEvent(.step(step: step, of: spec.steps, loss: loss, rate: optimizer.learningRate, tokensPerSecond: Double(windowTokens) / max(elapsed, 1e-3)))
                windowStart = Date()
                windowTokens = 0
            }
            if step % spec.evalEvery == 0 || step == spec.steps {
                let result = evaluate()
                state.evals.append(result)
                onEvent(.eval(result))
            }
            if step % spec.checkpointEvery == 0 || step == spec.steps { onEvent(.checkpoint(try checkpoint())) }
        }
        return state
    }
}
