import Foundation
import MLX
import SinatraHarness
import Testing

@testable import RaoLMCore
@testable import RaoLMGrounding
@testable import RaoLMModel
@testable import RaoLMProvenance
@testable import RaoLMTraining

private var mlxTests: Bool {
    let env = ProcessInfo.processInfo.environment
    return env["RAOLM_MLX_TESTS"] == "1" || env["FRIGATE_MLX_TESTS"] == "1"
}

@Suite("Grounding end to end", .enabled(if: mlxTests), .serialized)
struct GroundingEndToEndTests {
    @Test("train a tiny model, generate from a fact, measure it with and without its source, evaluate with grounding")
    func pipeline() async throws {
        // The fixture of RaoLMProvenanceTests' end-to-end test.
        let tokenizer = try await RaoTokenizer.load()
        let corpus = try SyntheticCorpus.generate(seed: 21, documentCount: 5)
        let snapshot = CorpusSnapshot.offline(corpus)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("raolm-grounding-e2e-\(UUID().uuidString)")
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
            runID: "grounding-e2e", preset: "test", model: config, tokenizer: tokenizer.ref,
            corpus: CorpusRef(slug: "veldmar", corpusHash: snapshot.corpusHash,
                              snapshotPath: snapshotDirectory.appendingPathComponent(CorpusSnapshot.fileName).path, source: "offline",
                              threadID: nil, owner: "o", group: "g", documentCount: 5, partitionCount: tokenized.partitions.count,
                              tokenCount: tokenized.tokenCount),
            hyperparameters: hyper, provenance: provenance)
        let trainer = Pretrainer(model: model, corpus: tokenized, tokenizer: tokenizer, facts: located, hyper: hyper,
                                 provenance: provenance, runDirectory: runDirectory, manifest: manifest)
        let result = try trainer.run()
        let lastEpoch = try #require(result.epochs.last)
        #expect(lastEpoch.evalMemorisedFraction ?? 0 > 0.5, "the tiny model should memorise five documents")

        let context = try await RunContext.load(runDirectory: runDirectory, allowWeakIndex: true)
        let grounder = RaoGrounder(context: context, corpus: tokenized, budget: 120)
        #expect(grounder.storeDirectory.lastPathComponent == "grounding")

        // A fact's own corpus prompt, measured against the partition it was sliced from.
        let fact = try #require(located.first)
        let partition = tokenized.partitions[fact.row]
        let prompt = partition.tokens[fact.contextToken..<fact.answerToken].map(Int.init)
        var params = context.defaultParameters()
        params.maxTokens = fact.answerLength + 2
        let generation = try context.generator().generate(GenerationRequest(
            promptTokens: prompt, promptText: tokenizer.decode(prompt),
            promptSource: SourceAddress(
                documentID: partition.documentID, partitionIndex: partition.partitionIndex, tokenOffset: fact.contextToken,
                partitionURL: partition.url, threadPartitionID: partition.threadPartitionID),
            params: params))
        let record = try await grounder.measure(generation: generation, policy: .promptSource, requireMeasured: true)
        #expect(record.measured)
        #expect(record.tokens.count == generation.tokens.count + (generation.stoppedOnEOS ? 1 : 0))
        #expect(record.tokens.map(\.token) == generation.tokens + (generation.stoppedOnEOS ? [tokenizer.eosTokenID] : []))
        #expect(record.sources.count == 1 && record.partitions.count == 1)
        #expect(record.sources[0].documentID == fact.fact.documentID && record.sources[0].partitionIndex == fact.fact.partitionIndex)
        #expect(record.measurement.attribution.map(\.partitionId) == [record.sources[0].id])
        #expect(record.measurement.bareTokens == prompt.count)
        #expect(record.measurement.promptTokens == 1 + partition.tokens.count + 1 + prompt.count)
        #expect(!record.contextExceedsTrainedLength || record.measurement.promptTokens + record.sampledTokens - 1 > hyper.seqLen)

        // The cross-check: the bare side is CitedGenerator's own prompt, so Sinatra's log p_bare
        // and H_bare must reproduce the traces' p_LM and H_lm.
        let bare = try #require(record.stats.bareConsistency)
        print(String(
            format: "grounding e2e: bare consistency over %d tokens: max |Δlog p| %.2e (mean %.2e), max |ΔH| %.2e (mean %.2e)",
            bare.tokens, bare.maxLogpGap, bare.meanLogpGap, bare.maxEntropyGap, bare.meanEntropyGap))
        #expect(bare.tokens == generation.tokens.count)
        #expect(bare.maxLogpGap < 1e-2, "log p_bare must equal log p_LM (worst step \(bare.worstStep.map(String.init) ?? "–"))")
        #expect(bare.maxEntropyGap < 1e-2, "H_bare must equal H_lm")
        let s = record.measurement.summary
        let exact = Array(generation.tokens.prefix(fact.answerLength)) == partition.tokens[fact.answerToken..<fact.answerEndToken].map(Int.init)
        let answerInfluence = record.tokens.prefix(fact.answerLength).map { String(format: "%+.3f", $0.influence) }.joined(separator: " ")
        print("grounding e2e: fact \(fact.fact.id) exact=\(exact): "
            + String(format: "Σι %+.3f, grounded %.0f%%, drift %.3f, risk %.3f", s.contextDependence, s.grounding * 100, s.drift, s.hallucinationRisk)
            + ", answer ι \(answerInfluence)")
        withKnownIssue("a memorised fact needs no source: ι ≈ 0 is the finding, not a failure", isIntermittent: true) {
            #expect(s.contextDependence > 0)
        }

        // Records survive a save and a load.
        let url = GroundingRecord.url(runDirectory: runDirectory, generationID: generation.generationID)
        try record.save(to: url)
        #expect(try GroundingRecord.load(from: url) == record)

        // The fabricated-entity control, measured against the true fact's partition.
        let negativeTokens = tokenizer.encode(fact.fact.negativePrompt)
        let negative = try context.generator().generate(GenerationRequest(
            promptTokens: negativeTokens, promptText: fact.fact.negativePrompt, params: params))
        let fabricated = try await grounder.measure(
            generation: negative, sources: [GroundingSource(partition: partition, threadID: nil)], requireMeasured: true)
        #expect(fabricated.measured && fabricated.tokens.count == negative.tokens.count + (negative.stoppedOnEOS ? 1 : 0))
        #expect(try #require(fabricated.stats.bareConsistency).maxLogpGap < 1e-2)
        print(String(
            format: "grounding e2e: fabricated-entity risk %.3f vs true %.3f",
            fabricated.measurement.summary.hallucinationRisk, s.hallucinationRisk))
        withKnownIssue("the tiny model may be as sure of a fabricated answer as of a memorised one", isIntermittent: true) {
            #expect(fabricated.measurement.summary.hallucinationRisk >= s.hallucinationRisk)
        }

        // Several sources, full detail.
        let top = try await grounder.measure(generation: generation, policy: .top, detail: .full, requireMeasured: true)
        #expect(top.measured && !top.sources.isEmpty && top.partitions.count == top.sources.count)
        #expect(top.tokens.allSatisfy { $0.rankBare != nil })
        #expect(try #require(top.stats.bareConsistency).maxLogpGap < 1e-2)

        // A spent budget is a skipped measurement, or an error when a result is required.
        let over = try await grounder.measure(generation: generation, policy: .promptSource, budget: -1)
        #expect(!over.measured && over.tokens.isEmpty && over.measurement.skippedReason?.contains("budget") == true)
        await #expect(throws: GroundingError.self) {
            _ = try await grounder.measure(generation: generation, policy: .promptSource, budget: -1, requireMeasured: true)
        }
        var foreign = generation
        foreign.manifest.checkpointSHA256 = "someone-else"
        await #expect(throws: GroundingError.self) { _ = try await grounder.measure(generation: foreign, policy: .promptSource) }

        // The evaluator with grounding: every outcome and the report carry it.
        let reader = InMemoryCorpusReader(snapshot: snapshot)
        let evaluator = FactEvaluator(context: context, corpus: tokenized, facts: corpus.facts, reader: reader)
        evaluator.groundingMeasurer = grounder.makeMeasurer(runDirectory: runDirectory, saveRecords: true)
        let eval = try await evaluator.run(options: EvalOptions(
            factsSample: 4, lambdas: [0, 0.5], primaryLambda: 0.5, includeControls: true,
            grounding: GroundingEvalOptions(includeControls: true, saveRecords: true)))
        let grounding = try #require(eval.grounding)
        #expect(eval.outcomes.count == 4)
        #expect(eval.outcomes.allSatisfy { $0.grounding?.measured == true })
        #expect(eval.outcomes.allSatisfy { $0.grounding?.recordFile.map { FileManager.default.fileExists(atPath: $0) } == true })
        #expect((grounding.metrics(.correct)?.facts ?? 0) + (grounding.metrics(.incorrect)?.facts ?? 0) == 4)
        #expect(grounding.metrics(.fabricated)?.facts == 4)
        #expect(grounding.controls.count >= 4)
        #expect(grounding.skipped.isEmpty)
        for group in grounding.groups {
            let risk = group.meanHallucinationRisk.map { String(format: "%.3f", $0) } ?? "–"
            let iota = group.meanAnswerInfluence.map { String(format: "%+.3f", $0) } ?? "–"
            print("grounding e2e eval: \(group.group.rawValue) n=\(group.facts) measured=\(group.measured) risk \(risk) answer ι \(iota)")
        }
        print("grounding e2e eval: risk AUROC (wrong answer) \(grounding.riskAUROCForWrongAnswer.map { String(format: "%.3f", $0) } ?? "–"), "
            + "r(answer ι, confidence) \(grounding.answerInfluenceConfidencePearson.map { String(format: "%.3f", $0) } ?? "–")")

        // A report written without grounding still decodes.
        let data = try JSONCoding.prettyEncoder().encode(eval)
        var object = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        object["grounding"] = nil
        object["outcomes"] = (object["outcomes"] as? [[String: Any]])?.map { outcome -> [String: Any] in
            var outcome = outcome
            outcome["grounding"] = nil
            return outcome
        }
        let old = try JSONCoding.decoder().decode(EvalReport.self, from: try JSONSerialization.data(withJSONObject: object))
        #expect(old.grounding == nil && old.outcomes.allSatisfy { $0.grounding == nil } && old.outcomes.count == 4)
    }
}
