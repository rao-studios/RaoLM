import Foundation
import MLX
import MLXNN
import MLXOptimizers
import Testing

@testable import RaoLMCore
@testable import RaoLMModel
@testable import RaoLMTraining

private func temporary() -> URL {
    FileManager.default.temporaryDirectory.appendingPathComponent("raolm-commons-\(UUID().uuidString)", isDirectory: true)
}

/// A token file of `count` tokens, an eos every `every`.
private func shard(_ root: URL, _ name: String, count: Int, every: Int, base: Int32) throws -> TokenShard {
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let url = root.appendingPathComponent("\(name).bin")
    try CommonsCorpus.save((0 ..< count).map { $0 % every == every - 1 ? 0 : base + Int32($0 % 97) + 1 }, to: url)
    return try TokenShard(url: url)
}

@Suite("Commons corpus and token stream")
struct CommonsCorpusTests {
    @Test("a corpus is written once, split by seed, hashed, and tokenized with an eos after every document")
    func corpus() async throws {
        let tokenizer = try await RaoTokenizer.load()
        let root = temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let documents = (0 ..< 300).map { CommonsDocument(id: "d\($0)", text: "Document \($0) says the river ran north past the mill.") }
        let manifest = try CommonsCorpus.write(name: "test", documents: documents, source: "test", license: "test", provenance: [:], tokenizer: tokenizer, root: root)
        #expect(manifest.documents == 300 && manifest.heldOutDocuments == 3 && manifest.documentsSHA256 == CommonsCorpus.hash(documents))
        let train = try CommonsCorpus.shard(root, "test", tokenizer: tokenizer, heldOut: false)
        let held = try CommonsCorpus.shard(root, "test", tokenizer: tokenizer, heldOut: true)
        #expect(train.count + held.count == documents.reduce(0) { $0 + tokenizer.encode($1.text).count + 1 })
        let eos = Int32(tokenizer.eosTokenID)
        #expect((0 ..< train.count).filter { train[$0] == eos }.count == 297 && (0 ..< held.count).filter { held[$0] == eos }.count == 3)
        #expect(manifest.tokens.first?.train == train.count && manifest.tokens.first?.heldOut == held.count)
        #expect(try CommonsCorpus.documents(root, "test") == documents)
        #expect(throws: CommonsCorpusError.self) {
            try CommonsCorpus.write(name: "test", documents: documents, source: "test", license: "test", provenance: [:], tokenizer: tokenizer, root: root)
        }
        #expect(CommonsCorpus.heldOutCount(100) == 2 && CommonsCorpus.heldOutCount(1_000_000) == 256 && CommonsCorpus.heldOutCount(3) == 1)
    }

    @Test("the plan is the same however far it is taken, windows stay inside their corpus, and weights decide the mix")
    func plan() throws {
        let root = temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let a = try shard(root, "a", count: 10_000, every: 300, base: 1000)
        let b = try shard(root, "b", count: 2_000, every: 300, base: 2000)
        let stream = TokenStream(sources: [.init(name: "a", shard: a, weight: 3), .init(name: "b", shard: b, weight: 1)], seqLen: 64, batchSize: 4, seed: 9, eos: 0)
        let long = stream.plan(steps: 200)
        #expect(Array(long.prefix(50)) == stream.plan(steps: 50), "a resumed run replays the same windows")
        let windows = long.joined()
        #expect(windows.count == 800)
        #expect(windows.allSatisfy { $0.start >= 0 && $0.start + 65 <= [a, b][$0.source].count })
        let share = Double(windows.filter { $0.source == 0 }.count) / Double(windows.count)
        #expect(abs(share - 0.75) < 0.06, "\(share)")
        // b's 30 windows are walked before any repeats: an epoch of a corpus is a permutation of its grid.
        let firstOfB = Array(windows.filter { $0.source == 1 }.prefix(29))
        #expect(Set(firstOfB.map(\.start)).count == 29)
        #expect(TokenStream.heldOutWindows(b, seqLen: 64, count: 100).count == 31)
    }
}

extension PretrainerTests {
    @Suite("The commons trainer", .serialized)
    struct CommonsTrainerTests {
        static let config = RaoLMConfig(hiddenSize: 64, intermediateSize: 128, numHiddenLayers: 2, numAttentionHeads: 4, numKeyValueHeads: 2, maxPositionEmbeddings: 256)

        @Test("the resumable AdamW moves a model exactly as Frigate's AdamW does, its rate changed between steps")
        func adamW() throws {
            let a = try RaoTransformer.make(config: Self.config, seed: 1)
            let b = try RaoTransformer.make(config: Self.config, seed: 1)
            let frigate = AdamW(learningRate: 1e-3, betas: (0.9, 0.95), eps: 1e-8, weightDecay: 0.1, biasCorrection: false)
            let mine = ResumableAdamW(learningRate: 1e-3, betas: (0.9, 0.95), eps: 1e-8, weightDecay: 0.1)
            func lossAndGrad(_ model: RaoTransformer) -> (RaoTransformer, [MLXArray]) -> ([MLXArray], ModuleParameters) {
                valueAndGrad(model: model) { (model: RaoTransformer, arrays: [MLXArray]) -> [MLXArray] in
                    let logits = model.forward(arrays[0], cache: nil, captureTap: false).logits.asType(.float32)
                    let lse = logSumExp(logits, axis: -1)
                    let score = takeAlong(logits, arrays[1].expandedDimensions(axis: -1), axis: -1).squeezed(axis: -1)
                    return [(lse - score).mean()]
                }
            }
            let gradA = lossAndGrad(a)
            let gradB = lossAndGrad(b)
            var rng = SplitMix64(seed: 4)
            for step in 0 ..< 20 {
                let tokens = (0 ..< 2 * 17).map { _ in Int32(rng.nextInt(below: 500)) }
                let inputs = MLXArray(Array(tokens.enumerated().filter { $0.offset % 17 != 16 }.map(\.element)), [2, 16])
                let targets = MLXArray(Array(tokens.enumerated().filter { $0.offset % 17 != 0 }.map(\.element)), [2, 16])
                frigate.learningRate = 1e-3 * Float(1 + step % 3)
                mine.learningRate = 1e-3 * Float(1 + step % 3)
                let (_, ga) = gradA(a, [inputs, targets])
                frigate.update(model: a, gradients: ga)
                eval(a, frigate)
                let (_, gb) = gradB(b, [inputs, targets])
                mine.update(model: b, gradients: gb)
                eval([b.parameters().flattened().map(\.1), mine.state].flatMap { $0 })
            }
            let left = Dictionary(uniqueKeysWithValues: a.parameters().flattened())
            var worst: Float = 0
            for (key, value) in b.parameters().flattened() {
                worst = max(worst, abs(value - left[key]!).max().item(Float.self))
            }
            #expect(worst < 1e-6, "largest difference \(worst)")
            #expect(mine.step == 20)
        }

        @Test("a step accumulated over micro-batches moves the model as the whole batch at once does")
        func microBatches() async throws {
            let tokenizer = try await RaoTokenizer.load()
            let root = temporary()
            defer { try? FileManager.default.removeItem(at: root) }
            // No eos inside the windows: every micro-batch scores the same number of tokens, so the means agree exactly.
            let source = try shard(root, "corpus", count: 4_000, every: 1_000_000, base: 100)
            func run(_ name: String, micro: Int?) throws -> RaoTransformer {
                let model = try RaoTransformer.make(config: Self.config, seed: 3)
                let spec = CommonsTrainingSpec(
                    runID: name, parentSHA256: "parent", tokenizerSHA256: tokenizer.tokenizerSHA256,
                    corpora: [PackRecipe.Corpus(name: "corpus", sha256: "x", tokens: source.count, weight: 1)], tokens: 3 * 4 * 32, seqLen: 32, batch: 4,
                    microBatch: micro, peakLR: 1e-3, evalEvery: 100, checkpointEvery: 100)
                let stream = TokenStream(sources: [.init(name: "corpus", shard: source, weight: 1)], seqLen: 32, batchSize: 4, seed: spec.seed, eos: 0)
                try CommonsTrainer(model: model, spec: spec, stream: stream, heldOut: [:], directory: root.appendingPathComponent(name)).run()
                return model
            }
            let whole = Dictionary(uniqueKeysWithValues: try run("whole", micro: nil).parameters().flattened())
            var worst: Float = 0
            for (key, value) in try run("micro", micro: 2).parameters().flattened() { worst = max(worst, abs(value - whole[key]!).max().item(Float.self)) }
            #expect(worst < 1e-5, "largest difference \(worst)")
        }

        @Test("a run stopped and resumed ends where the uninterrupted run ends; the rate warms, holds and anneals")
        func resume() async throws {
            let tokenizer = try await RaoTokenizer.load()
            let root = temporary()
            defer { try? FileManager.default.removeItem(at: root) }
            let source = try shard(root, "corpus", count: 4_000, every: 200, base: 100)
            let held = try shard(root, "held", count: 400, every: 200, base: 300)
            func trainer(_ name: String, model: RaoTransformer) -> CommonsTrainer {
                let spec = CommonsTrainingSpec(
                    runID: "test", parentSHA256: "parent", tokenizerSHA256: tokenizer.tokenizerSHA256,
                    corpora: [PackRecipe.Corpus(name: "corpus", sha256: "x", tokens: source.count, weight: 1)], tokens: 12 * 2 * 32, seqLen: 32, batch: 2,
                    peakLR: 1e-3, warmupSteps: 3, annealFraction: 0.25, evalEvery: 5, checkpointEvery: 5)
                let stream = TokenStream(sources: [.init(name: "corpus", shard: source, weight: 1)], seqLen: 32, batchSize: 2, seed: spec.seed, eos: 0)
                let heldStream = TokenStream(sources: [.init(name: "held", shard: held, weight: 1)], seqLen: 32, batchSize: 2, seed: 0, eos: 0)
                let windows = TokenStream.heldOutWindows(held, seqLen: 32, count: 4).map { TokenStream.Window(source: 0, start: $0) }
                return CommonsTrainer(model: model, spec: spec, stream: stream, heldOut: ["held": [heldStream.batch(windows)]],
                                      directory: root.appendingPathComponent(name))
            }
            let whole = trainer("whole", model: try RaoTransformer.make(config: Self.config, seed: 2))
            #expect(whole.spec.steps == 12 && whole.spec.warmupSteps == 1 && whole.spec.annealSteps == 3)
            #expect(whole.spec.rate(1) == 1e-3 && whole.spec.rate(9) == 1e-3 && whole.spec.rate(12) == whole.spec.floorLR && whole.spec.rate(11) < 1e-3)
            let wholeState = try whole.run()
            #expect(wholeState.step == 12 && wholeState.evals.map(\.step) == [0, 5, 10, 12])
            #expect(wholeState.evals.last!.losses["held"]! < wholeState.evals.first!.losses["held"]!)
            #expect(CommonsTrainer.checkpoints(in: root.appendingPathComponent("whole")).map(\.step) == [10, 12])

            let first = trainer("split", model: try RaoTransformer.make(config: Self.config, seed: 2))
            let stopped = try first.run(shouldStop: { first.state.step == 7 })
            #expect(stopped.step == 7)
            let second = trainer("split", model: try RaoTransformer.make(config: Self.config, seed: 99))
            var resumedAt: Int?
            let finished = try second.run { if case .resumed(let step) = $0 { resumedAt = step } }
            #expect(resumedAt == 7 && finished.step == 12 && finished.trainLoss.count == 12)
            let left = Dictionary(uniqueKeysWithValues: whole.model.parameters().flattened())
            var worst: Float = 0
            for (key, value) in second.model.parameters().flattened() { worst = max(worst, abs(value - left[key]!).max().item(Float.self)) }
            #expect(worst < 1e-5, "largest difference \(worst)")
            #expect(zip(wholeState.trainLoss, finished.trainLoss).allSatisfy { abs($0 - $1) < 1e-4 })
        }
    }
}
