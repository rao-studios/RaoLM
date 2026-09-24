import Foundation
import MLX
import Testing

@testable import RaoLMCore
@testable import RaoLMModel
@testable import RaoLMTraining

private var mlxTests: Bool {
    let env = ProcessInfo.processInfo.environment
    return env["RAOLM_MLX_TESTS"] == "1" || env["FRIGATE_MLX_TESTS"] == "1"
}

private let tinyConfig = RaoLMConfig(hiddenSize: 64, intermediateSize: 128, numHiddenLayers: 2, numAttentionHeads: 4, numKeyValueHeads: 2, maxPositionEmbeddings: 256)

@Suite("TokenizedCorpus and sampler")
struct CorpusTokenizationTests {
    @Test("stream layout, per-partition tokenization and fact location")
    func layout() async throws {
        let tokenizer = try await RaoTokenizer.load()
        let corpus = try SyntheticCorpus.generate(seed: 3, documentCount: 6)
        let tokenized = TokenizedCorpus(snapshot: .offline(corpus), tokenizer: tokenizer)
        #expect(tokenized.documents.count == 6)
        #expect(tokenized.stream.first == 0 && tokenized.stream.last == 0)
        #expect(tokenized.stream.count == tokenized.tokenCount + 7)
        #expect(tokenized.streamRows.count == tokenized.stream.count)
        let byID = Dictionary(uniqueKeysWithValues: corpus.documents.map { ($0.id, $0) })
        for partition in tokenized.partitions {
            let source = try #require(byID[partition.documentID])
            #expect(partition.tokens == tokenizer.encode(source.partitions[partition.partitionIndex].text).map(Int32.init))
            #expect(partition.tokens.count >= 32)
        }
        for i in tokenized.stream.indices where tokenized.streamRows[i] >= 0 {
            let row = Int(tokenized.streamRows[i])
            #expect(tokenized.partitions[row].tokens[Int(tokenized.streamOffsets[i])] == tokenized.stream[i])
        }
        let located = FactLocator.locate(corpus.facts, corpus: tokenized, tokenizer: tokenizer)
        #expect(located.unaligned.isEmpty)
        #expect(located.located.count == corpus.facts.count)
        for fact in located.located {
            let partition = tokenized.partitions[fact.row]
            let answer = partition.tokens[fact.answerToken..<fact.answerEndToken].map(Int.init)
            #expect(tokenizer.decode(answer) == fact.fact.answer)
            let prompt = partition.tokens[fact.contextToken..<fact.answerToken].map(Int.init)
            #expect(tokenizer.decode(prompt).hasSuffix(fact.fact.prompt))
        }
        #expect(!tokenized.sharedNgrams().isEmpty)
    }

    @Test("batch plans are seeded, cover the stream and keep a fixed shape")
    func sampler() async throws {
        let tokenizer = try await RaoTokenizer.load()
        let corpus = try SyntheticCorpus.generate(seed: 3, documentCount: 6)
        let tokenized = TokenizedCorpus(snapshot: .offline(corpus), tokenizer: tokenizer)
        let sampler = BatchSampler(seqLen: 64, batchSize: 4, seed: 9)
        let plan = sampler.plan(epoch: 1, streamCount: tokenized.stream.count)
        #expect(plan == sampler.plan(epoch: 1, streamCount: tokenized.stream.count))
        #expect(plan != sampler.plan(epoch: 2, streamCount: tokenized.stream.count))
        #expect(plan.allSatisfy { $0.count == 4 })
        let starts = plan.flatMap { $0 }
        #expect(starts.allSatisfy { $0 + 65 <= tokenized.stream.count })
        #expect(Set(starts).count * 64 >= tokenized.stream.count - 2 * 64)
        if mlxTests {
            let batch = sampler.makeBatch(plan[0], corpus: tokenized)
            #expect(batch.inputs.shape == [4, 64] && batch.targets.shape == [4, 64] && batch.mask.shape == [4, 64])
            #expect(batch.rows.count == 256 && batch.maskedCount <= 256)
        }
    }

    @Test("eval windows give every position context and record each once")
    func windows() {
        let windows = EvalPass.windows(inputs: 1000, seqLen: 256)
        var covered = 0
        for (i, window) in windows.enumerated() {
            #expect(window.length <= 256)
            #expect(window.start + window.recordFrom == covered)
            if i > 0 { #expect(window.recordFrom == 128 || window.start == 0) }
            covered = window.start + window.length
        }
        #expect(covered == 1000)
        #expect(EvalPass.windows(inputs: 100, seqLen: 256).count == 1)
    }
}

@Suite("Pretrainer", .enabled(if: mlxTests), .serialized)
struct PretrainerTests {
    @Test("loss falls, the ledger fills, and the index finds its own positions")
    func trainTiny() async throws {
        let tokenizer = try await RaoTokenizer.load()
        let corpus = try SyntheticCorpus.generate(seed: 5, documentCount: 4)
        let snapshot = CorpusSnapshot.offline(corpus)
        let tokenized = TokenizedCorpus(snapshot: snapshot, tokenizer: tokenizer)
        let located = FactLocator.locate(corpus.facts, corpus: tokenized, tokenizer: tokenizer).located
        let runDirectory = FileManager.default.temporaryDirectory.appendingPathComponent("raolm-train-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: runDirectory) }
        let hyper = TrainingHyperparameters(batchSize: 4, seqLen: 64, epochs: 3, peakLR: 3e-3, evalEvery: 1, indexEvery: 0, keepCheckpoints: 1, earlyStopMemorised: nil)
        let provenance = ProvenanceSettings(tapLayer: 1, alpha: 0.5)
        let model = try RaoTransformer.make(config: tinyConfig, seed: 1, tapLayer: 1)
        let manifest = RunManifest(
            runID: "test", preset: "test", model: tinyConfig, tokenizer: tokenizer.ref,
            corpus: CorpusRef(slug: "veldmar", corpusHash: snapshot.corpusHash, snapshotPath: "", source: "offline", threadID: nil,
                              owner: "o", group: "g", documentCount: 4, partitionCount: tokenized.partitions.count, tokenCount: tokenized.tokenCount),
            hyperparameters: hyper, provenance: provenance)
        let trainer = Pretrainer(model: model, corpus: tokenized, tokenizer: tokenizer, facts: located, hyper: hyper,
                                 provenance: provenance, runDirectory: runDirectory, manifest: manifest)
        var steps: [StepRow] = []
        let result = try trainer.run { event in
            if case .step(let row) = event { steps.append(row) }
        }
        #expect(steps.count > 6)
        #expect(steps.allSatisfy { $0.loss.isFinite && $0.gradNorm.isFinite })
        #expect(steps.allSatisfy { $0.entropy.mean >= 0 && $0.entropy.mean <= log(49152) + 0.01 })
        let first = steps.prefix(3).map(\.loss).reduce(0, +) / 3
        let last = steps.suffix(3).map(\.loss).reduce(0, +) / 3
        #expect(last < first)
        #expect(result.epochs.count == 3)
        #expect(result.indexedEpochs == [3])
        #expect(result.epochs.last?.evalLoss != nil)

        let ledger = RunLayout.ledger(runDirectory)
        let partitionRows = try JSONCoding.readLines(PartitionEpochRow.self, from: LedgerFiles.partitions(ledger, epoch: 3))
        #expect(partitionRows.count == tokenized.partitions.count)
        #expect(partitionRows.allSatisfy { $0.eval != nil && $0.train != nil })
        let factRows = try JSONCoding.readLines(FactEpochRow.self, from: LedgerFiles.facts(ledger, epoch: 3))
        #expect(factRows.count == located.count)
        #expect(factRows.allSatisfy { $0.answerTokenLosses.allSatisfy { $0.isFinite } })
        let stepRows = try JSONCoding.readLines(StepRow.self, from: LedgerFiles.steps(ledger))
        #expect(stepRows.count == steps.count)

        // The index: one entry per non-eos transition; a position's own key retrieves itself.
        let indexDirectory = RunLayout.provenance(runDirectory, epoch: 3)
        let info = try JSONCoding.read(IndexInfo.self, from: indexDirectory.appendingPathComponent(ProvenanceIndexFiles.info))
        let expectedEntries = tokenized.partitions.reduce(0) { $0 + $1.tokens.count } - tokenized.documents.count
        #expect(info.count == expectedEntries)
        #expect(info.checkpointSHA256 == result.epochs.last?.checkpointSHA256)
        let arrays = try loadArrays(url: indexDirectory.appendingPathComponent(ProvenanceIndexFiles.arrays))
        #expect(arrays["keys"]?.shape == [expectedEntries, 128])
        let keys = arrays["keys"]!.asType(.float32)
        let probe = keys[7]
        let scores = matmul(keys, probe.reshaped(-1, 1)).reshaped(-1).asArray(Float.self)
        let best = scores.indices.max { scores[$0] < scores[$1] }
        #expect(best == 7 || abs(scores[7] - scores[best!]) < 1e-4)
        #expect(!(try ProvenanceIndexer.readSharedNgrams(indexDirectory)).isEmpty)
        // Checkpoint directory holds only the model files.
        let files = try FileManager.default.contentsOfDirectory(atPath: RunLayout.checkpoint(runDirectory, epoch: 3).path)
        #expect(files.filter { $0.hasSuffix(".safetensors") } == ["model.safetensors"])
    }
}
