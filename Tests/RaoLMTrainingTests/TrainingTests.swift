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

    @Test("eos-first batches open every other grid window with eos, masked; the plan and the rest are unchanged")
    func eosFirstWindows() async throws {
        let tokenizer = try await RaoTokenizer.load()
        let corpus = try SyntheticCorpus.generate(seed: 3, documentCount: 6)
        let tokenized = TokenizedCorpus(snapshot: .offline(corpus), tokenizer: tokenizer)
        let count = tokenized.stream.count
        let plain = BatchSampler(seqLen: 64, batchSize: 4, seed: 9)
        let sampler = BatchSampler(seqLen: 64, batchSize: 4, seed: 9, eosFirst: true)
        let plan = sampler.plan(epoch: 1, streamCount: count)
        #expect(plan == plain.plan(epoch: 1, streamCount: count))
        #expect(Set(plan.flatMap { $0 }.map { ($0 / 64) % 2 }) == [0, 1])
        guard mlxTests else { return }
        for starts in plan {
            let a = plain.makeBatch(starts, corpus: tokenized)
            let b = sampler.makeBatch(starts, corpus: tokenized)
            let (ai, bi) = (a.inputs.asArray(Int32.self), b.inputs.asArray(Int32.self))
            let (am, bm) = (a.mask.asArray(Float.self), b.mask.asArray(Float.self))
            #expect(a.targets.asArray(Int32.self) == b.targets.asArray(Int32.self))
            for (w, start) in starts.enumerated() {
                let first = w * 64
                let rest = (first + 1)..<(first + 64)
                if (start / 64) % 2 == 0 {
                    #expect(bi[first] == tokenized.eos && bm[first] == 0 && b.rows[first] == -1)
                } else {
                    #expect(bi[first] == ai[first] && bm[first] == am[first] && b.rows[first] == a.rows[first])
                }
                #expect(Array(bi[rest]) == Array(ai[rest]) && Array(bm[rest]) == Array(am[rest]))
            }
        }
    }

    @Test("the passage break: the tokenizer's own joins a document's partitions, addressed past the partition it follows")
    func passageBreak() async throws {
        let tokenizer = try await RaoTokenizer.load()
        #expect(tokenizer.paragraphBreak == [198, 198])
        let corpus = try SyntheticCorpus.generate(seed: 3, documentCount: 6)
        let plain = TokenizedCorpus(snapshot: .offline(corpus), tokenizer: tokenizer)
        let tokenized = TokenizedCorpus(snapshot: .offline(corpus), tokenizer: tokenizer, paragraphBreak: tokenizer.paragraphBreak)
        #expect(tokenized.paragraphBreak == [198, 198] && plain.paragraphBreak.isEmpty)
        // The partitions themselves, and so every citable address, are unchanged.
        #expect(tokenized.partitions.map(\.tokens) == plain.partitions.map(\.tokens))
        let joins = tokenized.partitions.count - tokenized.documents.count
        #expect(joins > 0)
        #expect(tokenized.stream.count == plain.stream.count + 2 * joins)
        let byID = Dictionary(uniqueKeysWithValues: corpus.documents.map { ($0.id, $0) })
        for document in tokenized.documents {
            // A document's stream is the tokenization of its text, paragraphs joined by a blank line.
            let source = try #require(byID[document.id])
            let text = source.partitions.sorted { $0.index < $1.index }.map(\.text).joined(separator: "\n\n")
            let sequence = tokenized.documentSequence(document)
            #expect(sequence.tokens.dropFirst().dropLast().map(Int.init) == tokenizer.encode(text))
            #expect(sequence.tokens.first == tokenized.eos && sequence.tokens.last == tokenized.eos)
        }
        var breaks = 0
        for i in tokenized.stream.indices where tokenized.streamRows[i] >= 0 {
            let partition = tokenized.partitions[Int(tokenized.streamRows[i])]
            let offset = Int(tokenized.streamOffsets[i])
            if offset < partition.tokens.count {
                #expect(partition.tokens[offset] == tokenized.stream[i])
            } else {
                // A break token: the partition it follows, never its document's last.
                breaks += 1
                #expect(tokenized.stream[i] == tokenized.paragraphBreak[offset - partition.tokens.count])
                #expect(partition.row + 1 < tokenized.documents[partition.documentIndex].rows.upperBound)
            }
        }
        #expect(breaks == 2 * joins)
    }

    @Test("the document mask: a token sees earlier tokens of its own document, and an eos opens the next")
    func documentMask() async throws {
        let eos: Int32 = 0
        // Two windows: one opening mid-document, one opening with eos.
        let inputs: [Int32] = [5, 6, 0, 7, 8, 0, 9,
                               0, 4, 4, 4, 0, 0, 3]
        let documents = [[0, 0, 1, 1, 1, 2, 2], [1, 1, 1, 1, 2, 3, 3]]
        let T = 7
        let mask = BatchSampler.documentMask(inputs, count: 2, length: T, eos: eos)
        #expect(mask.count == 2 * T * T)
        for b in 0..<2 {
            for t in 0..<T {
                for s in 0..<T {
                    #expect(mask[(b * T + t) * T + s] == (s <= t && documents[b][s] == documents[b][t]), "window \(b) t \(t) s \(s)")
                }
            }
        }
        guard mlxTests else { return }
        let tokenizer = try await RaoTokenizer.load()
        let corpus = try SyntheticCorpus.generate(seed: 3, documentCount: 6)
        let tokenized = TokenizedCorpus(snapshot: .offline(corpus), tokenizer: tokenizer, paragraphBreak: tokenizer.paragraphBreak)
        let plain = BatchSampler(seqLen: 64, batchSize: 4, seed: 9)
        let sampler = BatchSampler(seqLen: 64, batchSize: 4, seed: 9, maskDocuments: true)
        let starts = sampler.plan(epoch: 1, streamCount: tokenized.stream.count)[0]
        let a = plain.makeBatch(starts, corpus: tokenized)
        let batch = sampler.makeBatch(starts, corpus: tokenized)
        #expect(a.attention == nil)
        let attention = try #require(batch.attention)
        #expect(attention.shape == [4, 1, 64, 64] && attention.dtype == .bool)
        #expect(attention.asArray(Bool.self) == BatchSampler.documentMask(batch.inputs.asArray(Int32.self), count: 4, length: 64, eos: tokenized.eos))
        #expect(batch.inputs.asArray(Int32.self) == a.inputs.asArray(Int32.self) && batch.mask.asArray(Float.self) == a.mask.asArray(Float.self))
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

    @Test("shouldStop ends training before the next step and leaves a stopped manifest with its completed epochs")
    func stopsOnRequest() async throws {
        let tokenizer = try await RaoTokenizer.load()
        let corpus = try SyntheticCorpus.generate(seed: 6, documentCount: 4)
        let snapshot = CorpusSnapshot.offline(corpus)
        let tokenized = TokenizedCorpus(snapshot: snapshot, tokenizer: tokenizer)
        let runDirectory = FileManager.default.temporaryDirectory.appendingPathComponent("raolm-stop-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: runDirectory) }
        let hyper = TrainingHyperparameters(batchSize: 2, seqLen: 64, epochs: 5, peakLR: 3e-3, evalEvery: 0, indexEvery: 0, keepCheckpoints: 1, earlyStopMemorised: nil)
        let provenance = ProvenanceSettings(tapLayer: 1, alpha: 0.5)
        let model = try RaoTransformer.make(config: tinyConfig, seed: 1, tapLayer: 1)
        let manifest = RunManifest(
            runID: "stop", preset: "test", model: tinyConfig, tokenizer: tokenizer.ref,
            corpus: CorpusRef(slug: "veldmar", corpusHash: snapshot.corpusHash, snapshotPath: "", source: "offline", threadID: nil,
                              owner: "o", group: "g", documentCount: 4, partitionCount: tokenized.partitions.count, tokenCount: tokenized.tokenCount),
            hyperparameters: hyper, provenance: provenance)
        let trainer = Pretrainer(model: model, corpus: tokenized, tokenizer: tokenizer, facts: [], hyper: hyper,
                                 provenance: provenance, runDirectory: runDirectory, manifest: manifest)
        var steps: [StepRow] = []
        var epochs: [Int] = []
        var messages: [String] = []
        // Stop once the second epoch has taken its first step.
        let result = try trainer.run(shouldStop: { steps.contains { $0.epoch == 2 } }) { event in
            switch event {
            case .step(let row): steps.append(row)
            case .epoch(let record): epochs.append(record.epoch)
            case .message(let text): messages.append(text)
            default: break
            }
        }
        #expect(result.status == .stopped)
        #expect(epochs == [1])
        #expect(steps.filter { $0.epoch == 2 }.count == 1)
        #expect(result.epochs.map(\.epoch) == [1])
        #expect(messages.last?.contains("stopped in epoch 2") == true)
        let saved = try RunManifest.load(runDirectory)
        #expect(saved.status == .stopped)
        #expect(saved.stepsLedgerSHA256 != nil)
    }

    @Test("the passage-break arm trains with attention kept in a document, and indexes and records its breaks")
    func passageBreakArm() async throws {
        let tokenizer = try await RaoTokenizer.load()
        let corpus = try SyntheticCorpus.generate(seed: 5, documentCount: 4)
        let snapshot = CorpusSnapshot.offline(corpus)
        let tokenized = TokenizedCorpus(snapshot: snapshot, tokenizer: tokenizer, paragraphBreak: tokenizer.paragraphBreak)
        let runDirectory = FileManager.default.temporaryDirectory.appendingPathComponent("raolm-break-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: runDirectory) }
        let hyper = TrainingHyperparameters(
            batchSize: 4, seqLen: 64, epochs: 3, peakLR: 3e-3, evalEvery: 1, indexEvery: 0, keepCheckpoints: 1, earlyStopMemorised: nil,
            maskDocuments: true)
        let provenance = ProvenanceSettings(tapLayer: 1, alpha: 0.5)
        let model = try RaoTransformer.make(config: tinyConfig, seed: 1, tapLayer: 1)
        let manifest = RunManifest(
            runID: "break", preset: "test", model: tinyConfig, tokenizer: tokenizer.ref,
            corpus: CorpusRef(slug: "veldmar", corpusHash: snapshot.corpusHash, snapshotPath: "", source: "offline", threadID: nil,
                              owner: "o", group: "g", documentCount: 4, partitionCount: tokenized.partitions.count, tokenCount: tokenized.tokenCount),
            hyperparameters: hyper, provenance: provenance)
        let trainer = Pretrainer(model: model, corpus: tokenized, tokenizer: tokenizer, facts: [], hyper: hyper,
                                 provenance: provenance, runDirectory: runDirectory, manifest: manifest)
        var steps: [StepRow] = []
        let result = try trainer.run { event in
            if case .step(let row) = event { steps.append(row) }
        }
        #expect(steps.count > 6 && steps.allSatisfy { $0.loss.isFinite && $0.gradNorm.isFinite })
        let first = steps.prefix(3).map(\.loss).reduce(0, +) / 3
        let last = steps.suffix(3).map(\.loss).reduce(0, +) / 3
        #expect(last < first)
        #expect(result.paragraphBreak == [198, 198] && result.hyperparameters.maskDocuments == true)
        // Every transition but those into eos, the breaks' included.
        let joins = tokenized.partitions.count - tokenized.documents.count
        let info = try JSONCoding.read(IndexInfo.self, from: RunLayout.provenance(runDirectory, epoch: 3).appendingPathComponent(ProvenanceIndexFiles.info))
        #expect(info.paragraphBreak == [198, 198])
        #expect(info.count == tokenized.tokenCount + 2 * joins - tokenized.documents.count)
    }

    @Test("with Canon layers and gated attention a model trains on the passage-break stream, loss falling")
    func blockAdditions() async throws {
        let tokenizer = try await RaoTokenizer.load()
        let corpus = try SyntheticCorpus.generate(seed: 5, documentCount: 4)
        let snapshot = CorpusSnapshot.offline(corpus)
        let tokenized = TokenizedCorpus(snapshot: snapshot, tokenizer: tokenizer, paragraphBreak: tokenizer.paragraphBreak)
        let runDirectory = FileManager.default.temporaryDirectory.appendingPathComponent("raolm-blocks-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: runDirectory) }
        var config = tinyConfig
        config.canon = true
        config.attentionGate = true
        let hyper = TrainingHyperparameters(
            batchSize: 4, seqLen: 64, epochs: 3, peakLR: 3e-3, evalEvery: 1, indexEvery: 0, keepCheckpoints: 1, earlyStopMemorised: nil,
            maskDocuments: true)
        let provenance = ProvenanceSettings(tapLayer: 1, alpha: 0.5)
        let model = try RaoTransformer.make(config: config, seed: 1, tapLayer: 1)
        let manifest = RunManifest(
            runID: "blocks", preset: "test", model: config, tokenizer: tokenizer.ref,
            corpus: CorpusRef(slug: "veldmar", corpusHash: snapshot.corpusHash, snapshotPath: "", source: "offline", threadID: nil,
                              owner: "o", group: "g", documentCount: 4, partitionCount: tokenized.partitions.count, tokenCount: tokenized.tokenCount),
            hyperparameters: hyper, provenance: provenance)
        let trainer = Pretrainer(model: model, corpus: tokenized, tokenizer: tokenizer, facts: [], hyper: hyper,
                                 provenance: provenance, runDirectory: runDirectory, manifest: manifest)
        var steps: [StepRow] = []
        _ = try trainer.run { event in
            if case .step(let row) = event { steps.append(row) }
        }
        #expect(steps.count > 6 && steps.allSatisfy { $0.loss.isFinite && $0.gradNorm.isFinite })
        #expect(steps.suffix(3).map(\.loss).reduce(0, +) < steps.prefix(3).map(\.loss).reduce(0, +))
        // The additions trained: none is zero any more.
        let added = model.parameters().flattened().filter { $0.0.contains("canon") || $0.0.contains("self_attn.gate_proj") }
        #expect(added.count == 2 * 4 && added.allSatisfy { abs($0.1).max().item(Float.self) > 0 })
        // The checkpoint keeps them: it loads back with the same config.
        let loaded = try Checkpoint.load(from: RunLayout.checkpoint(runDirectory, epoch: 3))
        #expect(loaded.config == config)
    }

    /// A tiny run of `epochs` on four documents, with `configure` setting the arm's hyperparameters.
    static func run(
        epochs: Int, configure: (inout TrainingHyperparameters) -> Void, shouldAnneal: (() -> Bool)? = nil,
        onEvent: (TrainingEvent) -> Void = { _ in }
    ) async throws -> (RunManifest, [StepRow], TrainingHyperparameters) {
        let tokenizer = try await RaoTokenizer.load()
        let corpus = try SyntheticCorpus.generate(seed: 5, documentCount: 4)
        let snapshot = CorpusSnapshot.offline(corpus)
        let tokenized = TokenizedCorpus(snapshot: snapshot, tokenizer: tokenizer, paragraphBreak: tokenizer.paragraphBreak)
        let runDirectory = FileManager.default.temporaryDirectory.appendingPathComponent("raolm-schedule-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: runDirectory) }
        var hyper = TrainingHyperparameters(
            batchSize: 4, seqLen: 64, epochs: epochs, peakLR: 3e-3, evalEvery: 1, indexEvery: 0, keepCheckpoints: 1, earlyStopMemorised: nil,
            maskDocuments: true)
        configure(&hyper)
        let provenance = ProvenanceSettings(tapLayer: 1, alpha: 0.5)
        let model = try RaoTransformer.make(config: tinyConfig, seed: 1, tapLayer: 1)
        let manifest = RunManifest(
            runID: "schedule", preset: "test", model: tinyConfig, tokenizer: tokenizer.ref,
            corpus: CorpusRef(slug: "veldmar", corpusHash: snapshot.corpusHash, snapshotPath: "", source: "offline", threadID: nil,
                              owner: "o", group: "g", documentCount: 4, partitionCount: tokenized.partitions.count, tokenCount: tokenized.tokenCount),
            hyperparameters: hyper, provenance: provenance)
        let trainer = Pretrainer(model: model, corpus: tokenized, tokenizer: tokenizer, facts: [], hyper: hyper,
                                 provenance: provenance, runDirectory: runDirectory, manifest: manifest)
        var steps: [StepRow] = []
        let result = try trainer.run(shouldAnneal: shouldAnneal ?? { false }) { event in
            if case .step(let row) = event { steps.append(row) }
            onEvent(event)
        }
        return (result, steps, hyper)
    }

    @Test("warmup-stable-decay holds the peak, anneals once asked, and stops at the end of the epoch the anneal reaches, at the floor")
    func stableDecay() async throws {
        var asked = false
        let (result, steps, hyper) = try await Self.run(epochs: 8, configure: {
            $0.schedule = "wsd"
            $0.annealFraction = 0.2
            $0.annealMinSteps = 4
        }, shouldAnneal: { asked }) { event in
            if case .epoch(let record) = event, record.epoch == 2 { asked = true }
        }
        #expect(result.status == .stopped)
        let start = try #require(steps.firstIndex { $0.epoch == 3 })
        let warmup = max(1, Int(Float(8 * steps.filter { $0.epoch == 1 }.count) * hyper.warmupFraction))
        // Warmup, then the peak held until the anneal starts at the first step after it was asked for.
        #expect(steps[warmup ..< start].allSatisfy { $0.lr == hyper.peakLR })
        #expect(zip(steps[start...].dropFirst(), steps[start...]).allSatisfy { $0.0.lr < $0.1.lr })
        #expect(abs(steps.last!.lr - hyper.finalLR) < 1e-9)
        // It ends at an epoch's end, after max(4, 0.2 × steps so far) steps, and that epoch is evaluated.
        #expect(steps.count - start >= max(4, Int(Float(start) * 0.2)))
        #expect(result.epochs.last?.epoch == steps.last?.epoch && result.epochs.last?.evalMemorisedFraction != nil)
        #expect(result.epochs.last.map { $0.epoch < 8 } == true)
        #expect(result.notes.contains { $0.hasPrefix("Annealing from step \(start)") })
    }

    @Test("a warmup-stable-decay run never asked to anneal still anneals, to end on its last planned step")
    func stableDecayUnasked() async throws {
        let (result, steps, hyper) = try await Self.run(epochs: 4, configure: {
            $0.schedule = "wsd"
            $0.annealFraction = 0.2
            $0.annealMinSteps = 4
        })
        // It runs to the end of its plan (the caller marks a finished run complete).
        #expect(result.status != .stopped && result.epochs.count == 4)
        #expect(abs(steps.last!.lr - hyper.finalLR) < 1e-9)
        let latest = Pretrainer.latestAnnealStart(hyper: hyper, totalSteps: steps.count)
        #expect(steps[latest].lr < hyper.peakLR && steps[latest - 1].lr == hyper.peakLR)
    }

    @Test("with Muon for the blocks' weight matrices a model trains, loss falling")
    func muon() async throws {
        #expect(Pretrainer.isMuonParameter("model.layers.0.mlp.up_proj.weight", MLXArray.zeros([4, 4])))
        #expect(!Pretrainer.isMuonParameter("model.layers.0.input_layernorm.weight", MLXArray.zeros([4])))
        #expect(!Pretrainer.isMuonParameter("model.embed_tokens.weight", MLXArray.zeros([4, 4])))
        #expect(!Pretrainer.isMuonParameter("model.layers.0.self_attn.gate_proj.weight", MLXArray.zeros([4, 4])))
        let (_, steps, _) = try await Self.run(epochs: 3, configure: {
            $0.optimizer = "muon"
            $0.muonScale = 0.2 * Float(64).squareRoot()
        })
        #expect(steps.count > 6 && steps.allSatisfy { $0.loss.isFinite && $0.gradNorm.isFinite })
        #expect(steps.suffix(3).map(\.loss).reduce(0, +) < steps.prefix(3).map(\.loss).reduce(0, +))
    }
}

