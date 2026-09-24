import Foundation
import MLX
import Testing

@testable import RaoLMCore
@testable import RaoLMModel
@testable import RaoLMProvenance
@testable import RaoLMTraining

private var mlxTests: Bool {
    let env = ProcessInfo.processInfo.environment
    return env["RAOLM_MLX_TESTS"] == "1" || env["FRIGATE_MLX_TESTS"] == "1"
}

@Suite("CitationMixer")
struct CitationMixerTests {
    @Test("weights, mixing and sampling behave")
    func mixing() {
        let w = CitationMixer.weights([0.9, 0.8, 0.5], tau: 0.05)
        #expect(abs(w.reduce(0, +) - 1) < 1e-5 && w[0] > w[1] && w[1] > w[2])
        let pLM: [Float] = [0.7, 0.2, 0.1]
        let mixed = CitationMixer.mix(pLM: pLM, knn: [2: 1.0], lambda: 0.5)
        #expect(abs(mixed.reduce(0, +) - 1) < 1e-5)
        #expect(CitationMixer.argmax(mixed) == 2)
        #expect(CitationMixer.argmax(CitationMixer.mix(pLM: pLM, knn: [2: 1.0], lambda: 0)) == 0)
        var rng = SplitMix64(seed: 1)
        let samples = (0..<200).map { _ in CitationMixer.sample([0.0, 1.0, 0.0], topK: 0, rng: &rng) }
        #expect(samples.allSatisfy { $0 == 1 })
        let soft = CitationMixer.softmax([1, 2, 3])
        #expect(abs(soft.reduce(0, +) - 1) < 1e-5 && soft[2] > soft[1])
    }

    @Test("calibration bins and ECE")
    func calibration() {
        let (bins, ece) = FactEvaluator.calibration([(0.95, true), (0.9, true), (0.1, false), (0.15, false), (0.5, false), (0.55, true)])
        #expect(bins.count == 10)
        #expect(bins[9].count == 2 && bins[9].accuracy == 1)
        #expect(ece >= 0 && ece <= 1)
    }
}

@Suite("Verifier")
struct VerifierTests {
    @Test("statuses: verified, stale, mismatch, missing, and the tokenizer guard")
    func statuses() async throws {
        let tokenizer = try await RaoTokenizer.load()
        let text = "Construction of the Kestrel Bridge was completed in 1874. The plans were drawn by an architect."
        let tokens = tokenizer.encode(text)
        let ref = ManifestRef(runID: "r", epoch: 1, checkpointSHA256: "c", indexSHA256: "i", corpusHash: "h",
                              tokenizerSHA256: tokenizer.tokenizerSHA256, ledgerSHA256: nil, threadID: nil)
        func span(offset: Int, count: Int, sha: String) -> CitedSpan {
            let slice = Array(tokens[offset..<(offset + count)])
            return CitedSpan(
                kind: .verbatim, tokenRange: TokenRange(start: 0, end: count), promptTokens: 0, tokens: slice,
                text: tokenizer.decode(slice), row: 0,
                source: SourceAddress(documentID: "doc", partitionIndex: 0, tokenOffset: offset), documentName: "Doc",
                textSHA256: sha, ranks: [], weights: [], confidence: 1, meanConfidence: 1, distinctiveness: 1,
                alternatives: [], verification: nil)
        }
        let sha = ContentHash.sha256Hex(text)
        var generation = CitedGeneration(
            generationID: "g", manifest: ref, prompt: GenerationPrompt(text: "", tokens: [0], source: nil),
            params: GenerationParameters(tapLayer: 1, alpha: 0.5), tokens: [], text: "", stoppedOnEOS: false,
            partitions: [], traces: [],
            spans: [span(offset: 3, count: 5, sha: sha), span(offset: 3, count: 5, sha: "other"), span(offset: 0, count: 3, sha: sha)],
            summary: GenerationSummary(generated: 0, verbatimCovered: 0, supportOnly: 0, uncited: 0, meanConfidence: nil, verbatimSpans: 3, verifiedSpans: nil))
        generation.spans[2].source.tokenOffset = 4  // wrong offset → mismatch
        generation.spans.append(span(offset: 0, count: 2, sha: sha))
        generation.spans[3].source.documentID = "gone"
        let reader = InMemoryCorpusReader(texts: ["doc": [text]])
        let report = try await CitationVerifier.verify(&generation, reader: reader, tokenizer: tokenizer)
        #expect(report.checks.map(\.status) == [.verified, .stale, .mismatch, .missing])
        #expect(generation.spans[0].verification?.status == .verified)

        var wrongTokenizer = generation
        wrongTokenizer.manifest.tokenizerSHA256 = "different"
        await #expect(throws: ProvenanceError.self) {
            _ = try await CitationVerifier.verify(&wrongTokenizer, reader: reader, tokenizer: tokenizer)
        }
    }
}

@Suite("End to end offline", .enabled(if: mlxTests), .serialized)
struct EndToEndTests {
    @Test("train a tiny model, generate with citations, verify against the snapshot, evaluate")
    func pipeline() async throws {
        let tokenizer = try await RaoTokenizer.load()
        let corpus = try SyntheticCorpus.generate(seed: 21, documentCount: 5)
        let snapshot = CorpusSnapshot.offline(corpus)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("raolm-e2e-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let snapshotDirectory = root.appendingPathComponent("snapshot")
        try snapshot.save(to: snapshotDirectory)
        let tokenized = TokenizedCorpus(snapshot: snapshot, tokenizer: tokenizer)
        let located = FactLocator.locate(corpus.facts, corpus: tokenized, tokenizer: tokenizer).located
        let config = RaoLMConfig(hiddenSize: 128, intermediateSize: 256, numHiddenLayers: 2, numAttentionHeads: 4, numKeyValueHeads: 2, maxPositionEmbeddings: 256)
        let hyper = TrainingHyperparameters(batchSize: 4, seqLen: 128, epochs: 160, peakLR: 4e-3, evalEvery: 20, keepCheckpoints: 1, earlyStopMemorised: 0.97)
        let provenance = ProvenanceSettings(tapLayer: 1, alpha: 0.5)
        let model = try RaoTransformer.make(config: config, seed: 2, tapLayer: 1)
        let runDirectory = root.appendingPathComponent("run")
        let manifest = RunManifest(
            runID: "e2e", preset: "test", model: config, tokenizer: tokenizer.ref,
            corpus: CorpusRef(slug: "veldmar", corpusHash: snapshot.corpusHash,
                              snapshotPath: snapshotDirectory.appendingPathComponent(CorpusSnapshot.fileName).path, source: "offline",
                              threadID: nil, owner: "o", group: "g", documentCount: 5, partitionCount: tokenized.partitions.count,
                              tokenCount: tokenized.tokenCount),
            hyperparameters: hyper, provenance: provenance)
        let trainer = Pretrainer(model: model, corpus: tokenized, tokenizer: tokenizer, facts: located, hyper: hyper,
                                 provenance: provenance, runDirectory: runDirectory, manifest: manifest)
        let result = try trainer.run()
        for record in result.epochs where record.evalLoss != nil {
            print(String(format: "e2e epoch %d: train %.3f eval %.3f memorised %.1f%%", record.epoch, record.trainLoss,
                         record.evalLoss ?? 0, (record.evalMemorisedFraction ?? 0) * 100))
        }
        let lastEpoch = try #require(result.epochs.last)
        #expect(lastEpoch.evalMemorisedFraction ?? 0 > 0.5, "the tiny model should memorise five documents")

        let context = try await RunContext.load(runDirectory: runDirectory, allowWeakIndex: true)
        #expect(context.epoch == result.indexedEpochs.last)
        let fact = try #require(located.first)
        let partition = tokenized.partitions[fact.row]
        let prompt = partition.tokens[fact.contextToken..<fact.answerToken].map(Int.init)
        var params = context.defaultParameters()
        params.maxTokens = fact.answerLength + 2
        var generation = try context.generator().generate(GenerationRequest(
            promptTokens: prompt, promptText: tokenizer.decode(prompt), params: params))
        #expect(generation.traces.count == prompt.count - 1 + generation.tokens.count)
        #expect(generation.traces.prefix(prompt.count - 1).allSatisfy { $0.isPrompt })
        #expect(generation.traces.allSatisfy { $0.neighbours.count == params.k && $0.lmEntropy >= 0 })
        #expect(generation.manifest.checkpointSHA256 == lastEpoch.checkpointSHA256)

        // Throwing from onToken stops generation mid-stream (how the studio cancels).
        var streamParams = params
        streamParams.maxTokens = 8
        var streamed = 0
        #expect(throws: CancellationError.self) {
            _ = try context.generator().generate(GenerationRequest(
                promptTokens: prompt, promptText: tokenizer.decode(prompt), params: streamParams)) { _ in
                streamed += 1
                if streamed == 2 { throw CancellationError() }
            }
        }
        #expect(streamed == 2)

        // The synchronous loader (the studio's MLX worker) binds the same checkpoint and index.
        let synchronous = try RunContext.load(runDirectory: runDirectory, allowWeakIndex: true, tokenizer: tokenizer)
        #expect(synchronous.manifestRef == context.manifestRef)

        let reader = InMemoryCorpusReader(snapshot: snapshot)
        let report = try await CitationVerifier.verify(&generation, reader: reader, tokenizer: tokenizer)
        #expect(report.allVerified)
        let saved = runDirectory.appendingPathComponent("gen.json")
        try generation.save(to: saved)
        #expect(try CitedGeneration.load(from: saved) == generation)
        let rendered = CitationMarkers.render(generation)
        #expect(rendered.sources.count == generation.spans.filter { $0.kind == .verbatim }.count || rendered.text.contains("[["))

        let evaluator = FactEvaluator(context: context, corpus: tokenized, facts: corpus.facts, reader: reader)
        let eval = try await evaluator.run(options: EvalOptions(factsSample: 6, lambdas: [0, 0.5], primaryLambda: 0.5, includeControls: true))
        #expect(eval.lambdas.count == 2)
        #expect(eval.spanChecks == eval.spansVerified)
        #expect(eval.controls != nil)
        let primary = try #require(eval.metrics(lambda: 0.5))
        #expect(primary.facts == 6)
        #expect(primary.exactAnswer > 0.5, "memorised facts should be answered")
        #expect(primary.citationAt1Partition > 0.5, "answers should retrieve their own partition")

        // Mismatched hashes are refused.
        var tampered = try RunManifest.load(runDirectory)
        tampered.tokenizer.tokenizerSHA256 = "x"
        try tampered.save(to: runDirectory)
        await #expect(throws: ProvenanceError.self) { _ = try await RunContext.load(runDirectory: runDirectory, allowWeakIndex: true) }
    }
}
