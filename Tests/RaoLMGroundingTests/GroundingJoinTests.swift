import Foundation
import SinatraHarness
import Testing

@testable import RaoLMCore
@testable import RaoLMGrounding
@testable import RaoLMModel
@testable import RaoLMProvenance
@testable import RaoLMTraining

/// A small synthetic corpus, tokenized with the bundled tokenizer, and a hand-made generation
/// over it (the trace pattern of CitationSpansTests). No model, no MLX.
private struct Fixture {
    let tokenizer: RaoTokenizer
    let corpus: TokenizedCorpus
    /// Partition 0 and 1 of a document with at least two partitions, and a partition of another.
    let rowA: Int
    let rowA1: Int
    let rowB: Int
    let prompt: [Int]
    let generation: CitedGeneration

    var eos: Int { tokenizer.eosTokenID }
    var generated: [TokenTrace] { generation.traces.filter { !$0.isPrompt } }

    static func make(stoppedOnEOS: Bool = true) async throws -> Fixture {
        let tokenizer = try await RaoTokenizer.load()
        let generated = try SyntheticCorpus.generate(seed: 21, documentCount: 8)
        let corpus = TokenizedCorpus(snapshot: CorpusSnapshot.offline(generated), tokenizer: tokenizer)
        let document = try #require(corpus.documents.first { $0.rows.count >= 2 && corpus.partitions[$0.rows.lowerBound].tokens.count >= 16 })
        let other = try #require(corpus.documents.first { $0.id != document.id })
        let rowA = document.rows.lowerBound
        let rowA1 = rowA + 1
        let rowB = other.rows.lowerBound
        let a = corpus.partitions[rowA].tokens.map(Int.init)
        let prompt = Array(a[0..<8])
        let answer = Array(a[8..<13])

        func trace(_ index: Int, token: Int, prompt: Bool = false, _ neighbours: [(Int, Int, Int, Float, Bool)]) -> TokenTrace {
            let list = neighbours.map { row, offset, rank, weight, matches in
                Neighbour(
                    rank: rank, entry: row * 1000 + offset, score: 0.95 - Float(rank) * 0.01, weight: weight,
                    value: matches ? token : token + 1, matches: matches,
                    key: TokenPosition(row: row, offset: offset - 1), cited: TokenPosition(row: row, offset: offset),
                    sourceLoss: 0.01, sourceEntropy: 0.05)
            }
            // Distinct per-token values, so the join can be checked column by column.
            let x = Float(index)
            return TokenTrace(
                index: index, token: token, text: tokenizer.tokenText(token), isPrompt: prompt, lmEntropy: 1 + x / 10,
                knnEntropy: 0.5 + x / 100, mixedEntropy: 0.7 + x / 100, sourceEntropy: 0.3 + x / 1000,
                lmProb: 0.9 - x / 50, agreement: list.filter(\.matches).map(\.weight).reduce(0, +),
                mixedProb: 0.6, lambda: 0.5, neighbours: list)
        }

        var traces: [TokenTrace] = []
        for j in 1..<prompt.count {
            traces.append(trace(j, token: prompt[j], prompt: true, [(rowA, j, 1, 0.9, true)]))
        }
        for i in 0..<3 {
            let offset = 8 + i
            traces.append(trace(offset, token: answer[i], [(rowA, offset, 1, 0.7, true), (rowB, 3, 2, 0.1, true)]))
        }
        traces.append(trace(11, token: answer[3], [(rowA1, 5, 1, 0.6, true)]))
        traces.append(trace(12, token: answer[4], [(rowB, 9, 1, 0.8, false)]))

        let refs = corpus.partitionRefs()
        let byRow = Dictionary(uniqueKeysWithValues: refs.map { ($0.row, $0) })
        let spans = CitationSpans.annotate(traces: &traces, partitions: byRow, sharedNgrams: [], threadID: nil)
        let manifest = ManifestRef(
            runID: "grounding-test", epoch: 1, checkpointSHA256: "c", indexSHA256: "i", corpusHash: corpus.corpusHash,
            tokenizerSHA256: tokenizer.tokenizerSHA256, ledgerSHA256: nil, threadID: nil)
        let partition = corpus.partitions[rowA]
        let generation = CitedGeneration(
            generationID: "gen-grounding-test", manifest: manifest,
            prompt: GenerationPrompt(
                text: tokenizer.decode(prompt), tokens: prompt,
                source: SourceAddress(documentID: partition.documentID, partitionIndex: partition.partitionIndex, tokenOffset: 0)),
            params: GenerationParameters(tapLayer: 1, alpha: 0.5), tokens: answer, text: tokenizer.decode(answer),
            stoppedOnEOS: stoppedOnEOS, partitions: [rowA, rowA1, rowB].map { byRow[$0]! }, traces: traces, spans: spans,
            summary: CitationSpans.summary(traces: traces, spans: spans))
        return Fixture(tokenizer: tokenizer, corpus: corpus, rowA: rowA, rowA1: rowA1, rowB: rowB, prompt: prompt, generation: generation)
    }

    func tokens(_ row: Int) -> [Int] { corpus.partitions[row].tokens.map(Int.init) }

    func address(_ row: Int) -> SourceAddress {
        SourceAddress(documentID: corpus.partitions[row].documentID, partitionIndex: corpus.partitions[row].partitionIndex, tokenOffset: 0)
    }

    func resolve(_ policy: GroundingSourcePolicy, generation: CitedGeneration? = nil) throws -> [GroundingSource] {
        try GroundingSources.resolve(policy: policy, generation: generation ?? self.generation, corpus: corpus, threadID: nil)
    }

    /// A scorer output for `sampled` whose bare side is exactly the traces' p_LM and H_lm.
    func raw(_ sampled: [Int]) -> GroundingRaw {
        let traces = generated
        var raw = GroundingRaw(tokens: sampled, logpCtx: [], logpBare: [], entropyCtx: [], entropyBare: [], contextKL: [], tune: [], tuneNats: [])
        for (j, token) in sampled.enumerated() {
            let bare: Float = j < traces.count ? log(traces[j].lmProb) : -3
            let lift: Float = j < 3 ? 2 : (j == 3 ? -1 : 0.1)
            raw.logpBare.append(bare)
            raw.logpCtx.append(min(bare + lift, -0.01))
            raw.entropyBare.append(j < traces.count ? traces[j].lmEntropy : 2)
            raw.entropyCtx.append(0.5)
            raw.contextKL.append(0.8)
            raw.tune.append(token)
            raw.tuneNats.append(0.5)
        }
        return raw
    }

    func session() -> SinatraSession {
        RaoGrounder.makeSession(
            tokenizer: tokenizer, modelKey: "grounding-test", vocabularySize: tokenizer.vocabularySize,
            storeDirectory: FileManager.default.temporaryDirectory.appendingPathComponent("raolm-grounding-\(UUID().uuidString)"))
    }
}

@Suite("Grounding sources")
struct GroundingSourcesTests {
    @Test("policies select the cited, spanned, retrieved, prompted or named partitions")
    func policies() async throws {
        let f = try await Fixture.make()
        #expect(try f.resolve(.top).map(\.ref.row) == [f.rowA, f.rowA1])
        #expect(try f.resolve(.all).map(\.ref.row) == [f.rowA, f.rowA1, f.rowB])
        #expect(try f.resolve(.spans).map(\.ref.row) == [f.rowA])
        #expect(try f.resolve(.promptSource).map(\.ref.row) == [f.rowA])
        #expect(try f.resolve(.explicit([f.address(f.rowB)])).map(\.ref.row) == [f.rowB])

        let top = try f.resolve(.top)
        #expect(abs(top[0].citationWeight - 2.1) < 1e-5)
        #expect(abs(top[1].citationWeight - 0.6) < 1e-5)
        #expect(top[0].tokens == f.tokens(f.rowA))
        #expect(top[0].ref.textSHA256 == f.corpus.partitions[f.rowA].textSHA256)
        #expect(try f.resolve(.all).count == 3)
        #expect(try GroundingSources.resolve(policy: .all, generation: f.generation, corpus: f.corpus, threadID: nil, limit: 2).count == 2)

        #expect(throws: GroundingError.self) { try f.resolve(.explicit([SourceAddress(documentID: "nope", partitionIndex: 0, tokenOffset: 0)])) }
        do {
            _ = try f.resolve(.explicit([SourceAddress(documentID: "nope", partitionIndex: 0, tokenOffset: 0)]))
        } catch let error as GroundingError {
            guard case .sourceNotInCorpus = error else { Issue.record("expected sourceNotInCorpus, got \(error)"); return }
            #expect(error.exitCode == 66)
        }

        var unsourced = f.generation
        unsourced.prompt.source = nil
        #expect(throws: GroundingError.noSources("the prompt is not a corpus slice, so it has no source partition")) {
            try f.resolve(.promptSource, generation: unsourced)
        }
        var spanless = f.generation
        spanless.spans = []
        #expect(throws: GroundingError.noSources("the generation has no verbatim spans")) { try f.resolve(.spans, generation: spanless) }
    }

    @Test("a cited partition whose text hash differs from the corpus is refused")
    func corpusMismatch() async throws {
        let f = try await Fixture.make()
        var tampered = f.generation
        let index = try #require(tampered.partitions.firstIndex { $0.row == f.rowA })
        tampered.partitions[index].textSHA256 = "tampered"
        for policy in [GroundingSourcePolicy.top, .spans, .all, .promptSource] {
            do {
                _ = try f.resolve(policy, generation: tampered)
                Issue.record("\(policy) accepted a tampered partition")
            } catch let error as GroundingError {
                guard case .corpusMismatch = error else { Issue.record("expected corpusMismatch, got \(error)"); continue }
                #expect(error.exitCode == 65)
            }
        }
        // A partition the generation never cited has nothing to disagree with.
        #expect(try f.resolve(.explicit([f.address(f.rowB)]), generation: tampered).count == 1)
    }

    @Test("context is [eos] + a document's partitions in order + [eos] + the next document + [eos] + prompt")
    func contextAssembly() async throws {
        let f = try await Fixture.make()
        let sources = try f.resolve(.all)
        let eos = f.eos
        let expected = [eos] + f.tokens(f.rowA) + f.tokens(f.rowA1) + [eos] + f.tokens(f.rowB) + [eos] + f.prompt
        #expect(GroundingSources.contextTokens(sources: sources, prompt: f.prompt, eos: eos) == expected)
        #expect(GroundingSources.contextLength(sources: sources, promptCount: f.prompt.count) == expected.count)
        // Documents in first-appearance order; a document's partitions always in partition order.
        let reordered = [sources[2], sources[1], sources[0]]
        #expect(GroundingSources.contextTokens(sources: reordered, prompt: f.prompt, eos: eos)
            == [eos] + f.tokens(f.rowB) + [eos] + f.tokens(f.rowA) + f.tokens(f.rowA1) + [eos] + f.prompt)

        // Fitting drops the least-cited source first and always keeps one.
        let fitted = GroundingSources.fit(sources, promptCount: f.prompt.count, budget: expected.count - 1)
        #expect(fitted.map(\.ref.row) == [f.rowA, f.rowA1])
        #expect(GroundingSources.fit(sources, promptCount: f.prompt.count, budget: 1).map(\.ref.row) == [f.rowA])

        let inputs = try GroundingInputs.make(
            generation: f.generation, sources: sources, policy: .all, eos: eos, maxPositions: expected.count + 5,
            trainedLength: 16)
        #expect(inputs.sources.count == 3 && inputs.droppedSources.isEmpty)
        #expect(inputs.bareTokens == f.prompt)
        #expect(inputs.contextTokens == expected)
        #expect(inputs.contextExceedsTrainedLength)
        #expect(inputs.partitions.map(\.id) == sources.map(\.id))
        #expect(inputs.partitions.map(\.tokenIds) == sources.map { Optional($0.tokens) })
        #expect(inputs.partitions.allSatisfy { $0.score == 0 })
        let tight = try GroundingInputs.make(
            generation: f.generation, sources: sources, policy: .all, eos: eos,
            maxPositions: expected.count - f.tokens(f.rowB).count + 4)
        #expect(tight.droppedSources.map(\.ref.row) == [f.rowB])
        #expect(throws: GroundingError.self) {
            try GroundingInputs.make(generation: f.generation, sources: sources, policy: .all, eos: eos, maxPositions: 10)
        }
    }

    @Test("sampled tokens end in eos exactly when generation stopped on it")
    func sampled() async throws {
        let f = try await Fixture.make()
        #expect(GroundingSources.sampledTokens(f.generation, eos: f.eos) == f.generation.tokens + [f.eos])
        #expect(GroundingSources.sampledTokens(f.generation, eos: f.eos, includeEOS: false) == f.generation.tokens)
        var running = f.generation
        running.stoppedOnEOS = false
        #expect(GroundingSources.sampledTokens(running, eos: f.eos) == f.generation.tokens)
    }

    @Test("source ids, policies and details round-trip")
    func roundTrips() async throws {
        let f = try await Fixture.make()
        for source in try f.resolve(.all) {
            let parsed = try #require(GroundingSources.parseSourceID(source.id))
            #expect(f.corpus.row(documentID: parsed.documentID, partitionIndex: parsed.partitionIndex) == source.ref.row)
            #expect(source.id == "\(source.ref.documentID)#\(source.ref.partitionIndex)")
        }
        #expect(GroundingSources.parseSourceID("no-hash") == nil)

        #expect(try GroundingSourcePolicy.parse("top") == .top)
        #expect(try GroundingSourcePolicy.parse("fact") == .promptSource)
        let named = try GroundingSourcePolicy.parse("raolm-a:0, raolm-b:12")
        #expect(named == .explicit([
            SourceAddress(documentID: "raolm-a", partitionIndex: 0, tokenOffset: 0),
            SourceAddress(documentID: "raolm-b", partitionIndex: 12, tokenOffset: 0),
        ]))
        #expect(throws: GroundingError.invalidPolicy("everything")) { try GroundingSourcePolicy.parse("everything") }
        #expect(throws: GroundingError.self) { try GroundingSourcePolicy.parse("doc:x") }
        for policy in [GroundingSourcePolicy.top, .spans, .all, .promptSource, named] {
            #expect(try GroundingSourcePolicy.parse(policy.description) == policy)
            let data = try JSONCoding.lineEncoder().encode([policy])
            #expect(try JSONCoding.decoder().decode([GroundingSourcePolicy].self, from: data) == [policy])
        }
        #expect(try GroundingDetail.parse("FULL") == .full)
        #expect(throws: GroundingError.self) { try GroundingDetail.parse("verbose") }
    }
}

@Suite("Grounding join")
struct GroundingJoinTests {
    @Test("the session's measurement joins the traces row by row and partition by partition")
    func join() async throws {
        let f = try await Fixture.make()
        let sources = try f.resolve(.all)
        let inputs = try GroundingInputs.make(
            generation: f.generation, sources: sources, policy: .all, eos: f.eos, decode: { f.tokenizer.decode($0) })
        let measurement = try await RaoGrounder.measurement(session: f.session(), inputs: inputs, raw: f.raw(inputs.sampled))
        #expect(measurement.measured)
        #expect(measurement.steps.count == inputs.sampled.count)
        let record = try RaoGrounder.record(generation: f.generation, inputs: inputs, measurement: measurement, eos: f.eos)

        #expect(record.tokens.count == f.generation.tokens.count + 1)
        #expect(record.tokens.map(\.token) == inputs.sampled)
        #expect(record.sampledTokens == inputs.sampled.count && record.includesEOS)
        #expect(record.contextTokens == inputs.contextTokens.count && record.promptTokens == f.prompt.count)
        for (row, trace) in zip(record.tokens, f.generated) {
            #expect(row.index == trace.index && row.text == trace.text && !row.isEOS)
            #expect(row.confidence == trace.confidence && row.agreement == trace.agreement && row.lmProb == trace.lmProb)
            #expect(row.lmEntropy == trace.lmEntropy && row.knnEntropy == trace.knnEntropy)
            #expect(row.mixedEntropy == trace.mixedEntropy && row.sourceEntropy == trace.sourceEntropy)
            #expect(row.uncited == trace.uncited && row.spanIndex == trace.spanIndex)
            #expect(row.topCitation == trace.citations.first?.address)
            #expect(row.influence == row.logpCtx - row.logpBare)
        }
        let eosRow = try #require(record.tokens.last)
        #expect(eosRow.isEOS && eosRow.token == f.eos && eosRow.index == f.prompt.count + f.generation.tokens.count)
        #expect(eosRow.lmProb == nil && eosRow.confidence == nil && eosRow.uncited == nil && eosRow.topCitationInSources == nil)
        #expect(record.tokens[0].inVerbatimSpan == true && record.tokens[3].inVerbatimSpan == false)
        #expect(record.tokens[3].topCitationInSources == true)
        #expect(record.tokens[4].uncited == true && record.tokens[4].topCitationInSources == false)

        #expect(record.partitions.map(\.id) == sources.map(\.id))
        #expect(record.partitions[0].nats == measurement.attribution[0].nats)
        #expect(record.partitions.map(\.uptake) == measurement.attribution.map { Optional($0.uptake) })
        #expect(abs(record.partitions[0].citationWeight - 2.1) < 1e-5)
        #expect(record.partitions[0].topCitedTokens == 3 && record.partitions[1].topCitedTokens == 1)
        #expect(record.partitions[0].verbatimSpanTokens == 3)
        #expect(record.partitions[2].topCitedTokens == 0 && record.partitions[2].meanCitationConfidence != nil)

        // The bare side was built from the traces' own p_LM: the cross-check reads ~0.
        let bare = try #require(record.stats.bareConsistency)
        #expect(bare.tokens == f.generation.tokens.count)
        #expect(bare.maxLogpGap < 1e-5 && bare.maxEntropyGap < 1e-5)
        #expect(record.stats.joinedTokens == f.generation.tokens.count)
        #expect(record.stats.meanInfluenceUncited != nil && record.stats.meanInfluenceCited != nil)

        // Summary for the evaluator: the first two answer tokens.
        let summary = GroundingFactSummary(record: record, answerLength: 2, recordFile: "x")
        #expect(summary.measured && summary.steps == record.tokens.count && summary.recordFile == "x")
        #expect(summary.meanAnswerInfluence == Stats.mean(record.tokens.prefix(2).map(\.influence)))

        // Save and load.
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("raolm-grounding-\(UUID().uuidString).grounding.json")
        defer { try? FileManager.default.removeItem(at: url) }
        try record.save(to: url)
        #expect(try GroundingRecord.load(from: url) == record)
        #expect(GroundingRecord.url(runDirectory: URL(fileURLWithPath: "/r"), generationID: "g").path == "/r/generations/g.grounding.json")
        #expect(GroundingRecord.url(besideGeneration: URL(fileURLWithPath: "/tmp/a/gen.json")).path == "/tmp/a/gen.grounding.json")
    }

    @Test("a misaligned measurement is refused; a skipped one keeps its reason and no rows")
    func alignmentAndSkips() async throws {
        let f = try await Fixture.make()
        let sources = try f.resolve(.top)
        let inputs = try GroundingInputs.make(generation: f.generation, sources: sources, policy: .top, eos: f.eos)
        var measurement = try await RaoGrounder.measurement(session: f.session(), inputs: inputs, raw: f.raw(inputs.sampled))
        measurement.steps[1].token += 1
        do {
            _ = try RaoGrounder.record(generation: f.generation, inputs: inputs, measurement: measurement, eos: f.eos)
            Issue.record("a misaligned step was accepted")
        } catch let error as GroundingError {
            guard case .alignment = error else { Issue.record("expected alignment, got \(error)"); return }
        }
        measurement.steps.removeLast()
        #expect(throws: GroundingError.self) {
            try RaoGrounder.record(generation: f.generation, inputs: inputs, measurement: measurement, eos: f.eos)
        }

        let skipped = try await RaoGrounder.measurement(session: f.session(), inputs: inputs, raw: nil, skipped: "over the 1.0 s budget")
        let record = try RaoGrounder.record(generation: f.generation, inputs: inputs, measurement: skipped, eos: f.eos)
        #expect(!record.measured && record.measurement.skippedReason == "over the 1.0 s budget")
        #expect(record.tokens.isEmpty)
        #expect(record.partitions.count == 2 && record.partitions.allSatisfy { $0.nats == nil && $0.citationWeight > 0 })
        #expect(record.stats.bareConsistency == nil)
        let summary = GroundingFactSummary(record: record, answerLength: 3)
        #expect(!summary.measured && summary.skippedReason == "over the 1.0 s budget")
    }

    @Test("without an EOS stop there is no EOS row")
    func noEOS() async throws {
        let f = try await Fixture.make(stoppedOnEOS: false)
        let inputs = try GroundingInputs.make(generation: f.generation, sources: try f.resolve(.promptSource), policy: .promptSource, eos: f.eos)
        #expect(inputs.sampled == f.generation.tokens && !inputs.includesEOS)
        let measurement = try await RaoGrounder.measurement(session: f.session(), inputs: inputs, raw: f.raw(inputs.sampled))
        let record = try RaoGrounder.record(generation: f.generation, inputs: inputs, measurement: measurement, eos: f.eos)
        #expect(record.tokens.count == f.generation.tokens.count && record.tokens.allSatisfy { !$0.isEOS && $0.isJoined })
        #expect(record.partitions.count == 1 && record.partitions[0].nats == measurement.attribution.first?.nats)
    }
}

@Suite("Grounding stats")
struct GroundingStatsTests {
    private func row(_ j: Int, confidence: Float?, influence: Float, uncited: Bool, risk: Float, kind: GroundingClass = .grounded) -> GroundingTokenRow {
        GroundingTokenRow(
            step: j, index: 10 + j, token: 100 + j, text: "t\(j)", confidence: confidence, agreement: (confidence ?? 0) / 2,
            lmProb: 0.5, lmEntropy: 1.25, knnEntropy: 0.5, mixedEntropy: 0.7, sourceEntropy: 0.2, uncited: uncited,
            inVerbatimSpan: j < 2, topCitation: nil, topCitationWeight: uncited ? nil : 0.5, topCitationInSources: j != 3,
            logpCtx: log(0.5) + influence, logpBare: log(0.5), influence: influence, contextKL: 0.4, drift: 0,
            entropyCtx: 1, entropyBare: 1.25, kind: kind, risk: risk)
    }

    @Test("known numbers: linear ι gives r = 1, risk on uncited gives AUROC 1")
    func knownNumbers() throws {
        let rows = [
            row(0, confidence: 0.9, influence: 2 * 0.9 + 1, uncited: false, risk: 0.1),
            row(1, confidence: 0.6, influence: 2 * 0.6 + 1, uncited: false, risk: 0.2, kind: .unsupported),
            row(2, confidence: 0.3, influence: 2 * 0.3 + 1, uncited: false, risk: 0.1),
            row(3, confidence: 0.2, influence: 2 * 0.2 + 1, uncited: false, risk: 0.3, kind: .function),
            row(4, confidence: nil, influence: 1, uncited: true, risk: 0.9, kind: .unsupported),
            row(5, confidence: nil, influence: 1, uncited: true, risk: 0.8, kind: .unsupported),
        ]
        let stats = GroundingStats.compute(tokens: rows)
        #expect(stats.contentTokens == 5 && stats.joinedTokens == 6)
        #expect(abs(try #require(stats.confidenceInfluencePearson) - 1) < 1e-6)
        #expect(abs(try #require(stats.agreementInfluencePearson) - 1) < 1e-6)
        #expect(stats.riskAUROCForUncited == 1)
        #expect(abs(try #require(stats.meanInfluenceUncited) - 1) < 1e-6)
        #expect(abs(try #require(stats.meanInfluenceCited) - Float(2 * (0.9 + 0.6 + 0.3 + 0.2) / 4 + 1)) < 1e-5)
        #expect(stats.verbatimSpanGroundedShare == 0.5)
        #expect(abs(try #require(stats.citedOutsideSourcesWeight) - 0.25) < 1e-6)
        let bare = try #require(stats.bareConsistency)
        #expect(bare.tokens == 6 && bare.maxLogpGap < 1e-6 && bare.maxEntropyGap < 1e-6)

        // Constant confidence has no variance: no correlation, not a spurious ±1.
        let flat = rows.map { r -> GroundingTokenRow in
            var r = r
            r.confidence = 0.1
            r.agreement = 0.1
            return r
        }
        #expect(GroundingStats.compute(tokens: flat).confidenceInfluencePearson == nil)
        #expect(Stats.pearson([0.1, 0.1, 0.1], [1, 2, 3]) == nil)
        #expect(GroundingStats.compute(tokens: []) == GroundingStats(bareConsistency: nil))
    }
}

@Suite("Grounding in the evaluator (model-free)")
struct GroundingEvalHookTests {
    private func outcome(exact: Bool, confidence: Float, grounding: GroundingFactSummary?) -> FactOutcome {
        FactOutcome(
            factID: "f", kind: .architect, documentID: "d", documentName: "D", partitionIndex: 0, lambda: 0.5, prompt: "p",
            expected: "e", generated: exact ? "e" : "x", exact: exact, citationAt1Partition: 1, citationAt1Offset: 1,
            citedPartitionAt1: 1, topCitation: nil, topCitationName: nil, answerCoveredByVerbatimSpan: exact,
            spansVerified: nil, spansChecked: nil, meanAnswerConfidence: confidence, generationFile: nil, grounding: grounding)
    }

    private func measured(risk: Float, influence: Float) -> GroundingFactSummary {
        GroundingFactSummary(
            measured: true, steps: 4, contentTokens: 2, grounding: 0.5, drift: 0.2, driftShare: 0.1,
            contextDependence: influence * 4, meanContextKL: 0.3, hallucinationRisk: risk, meanAnswerInfluence: influence,
            meanAnswerRisk: risk)
    }

    @Test("groups, AUROC for wrong answers, the two-fold r and skipped reasons")
    func aggregate() throws {
        let outcomes = [
            outcome(exact: true, confidence: 0.9, grounding: measured(risk: 0.05, influence: 0.1)),
            outcome(exact: true, confidence: 0.8, grounding: measured(risk: 0.1, influence: 0.08)),
            outcome(exact: false, confidence: 0.3, grounding: measured(risk: 0.6, influence: 0.02)),
            outcome(exact: false, confidence: 0.2, grounding: .skipped("over the 120.0 s budget")),
        ]
        let controls = [
            GroundingControlOutcome(factID: "f", group: .fabricated, prompt: "p", generated: "g", summary: measured(risk: 0.7, influence: 0)),
            GroundingControlOutcome(factID: "f", group: .paraphrase, prompt: "p", generated: "g", summary: .skipped("over the 120.0 s budget")),
        ]
        let report = GroundingEvalReport.make(outcomes: outcomes, controls: controls)
        #expect(report.groups.map(\.group) == [.correct, .incorrect, .paraphrase, .fabricated])
        let correct = try #require(report.metrics(.correct))
        #expect(correct.facts == 2 && correct.measured == 2)
        #expect(abs(try #require(correct.meanHallucinationRisk) - 0.075) < 1e-6)
        let incorrect = try #require(report.metrics(.incorrect))
        #expect(incorrect.facts == 2 && incorrect.measured == 1)
        #expect(report.metrics(.paraphrase)?.measured == 0 && report.metrics(.paraphrase)?.meanGrounding == nil)
        #expect(report.riskAUROCForWrongAnswer == 1)
        #expect(try #require(report.answerInfluenceConfidencePearson) > 0.9)
        #expect(report.skipped == ["over the 120.0 s budget": 2])
    }

    @Test("the evaluator counts a failing measurer as skipped and lets cancellation through")
    func measurerFailures() async throws {
        let generation = try await Fixture.make().generation
        let source = GroundingEvalSource(
            group: .fact, factID: "f", row: 0, documentID: "d", documentName: "D", partitionIndex: 0, tokens: [1, 2],
            textSHA256: "h", url: nil, threadPartitionID: nil, answerLength: 1)
        let failing: GroundingMeasurer = { _, _ in throw GroundingError.contextTooLong(tokens: 9, limit: 8) }
        let summary = try await FactEvaluator.ground(generation, source, measurer: failing)
        #expect(!summary.measured && summary.skippedReason == GroundingError.contextTooLong(tokens: 9, limit: 8).description)
        let silent: GroundingMeasurer = { _, _ in nil }
        #expect(try await FactEvaluator.ground(generation, source, measurer: silent) == .skipped("no measurement"))
        let cancelled: GroundingMeasurer = { _, _ in throw CancellationError() }
        await #expect(throws: CancellationError.self) { _ = try await FactEvaluator.ground(generation, source, measurer: cancelled) }
    }

    @Test("an eval.json written before grounding existed still decodes")
    func oldReportDecodes() throws {
        let report = EvalReport(
            runID: "r", epoch: 3, createdAt: .wholeSecond(), sampleSize: 1, primaryLambda: 0.5, lambdas: [], spanChecks: 0,
            spansVerified: 0, spanVerifiedRate: nil, calibration: [], ece: 0, auroc: nil, controls: nil,
            outcomes: [outcome(exact: true, confidence: 0.9, grounding: measured(risk: 0.1, influence: 0.1))], unalignedFacts: [],
            grounding: GroundingEvalReport.make(outcomes: [], controls: []))
        let data = try JSONCoding.prettyEncoder().encode(report)
        var object = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(object["grounding"] != nil)
        object["grounding"] = nil
        var outcomes = try #require(object["outcomes"] as? [[String: Any]])
        #expect(outcomes[0]["grounding"] != nil)
        outcomes[0]["grounding"] = nil
        object["outcomes"] = outcomes
        let old = try JSONSerialization.data(withJSONObject: object)
        let decoded = try JSONCoding.decoder().decode(EvalReport.self, from: old)
        #expect(decoded.grounding == nil && decoded.outcomes[0].grounding == nil && decoded.outcomes[0].exact)
        let roundTrip = try JSONCoding.decoder().decode(EvalReport.self, from: data)
        #expect(roundTrip.grounding == report.grounding && roundTrip.outcomes[0].grounding == report.outcomes[0].grounding)
        #expect(EvalOptions().grounding == nil)
    }
}
