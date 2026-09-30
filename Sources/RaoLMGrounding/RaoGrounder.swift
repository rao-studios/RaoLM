//
//  RaoGrounder.swift
//  RaoLMGrounding
//
//  WHAT: The two-fold entropy harness. RaoLM's traces already say, per generated token, which
//        corpus positions retrieval found and how confident the citation is. What they cannot
//        say is whether the cited source *moved* the model: no counterfactual is run. This
//        runs one, with SinatraHarness's grounding measurement:
//
//          1. inputs        pure Swift: choose sources, assemble context / bare / sampled tokens
//          2. score         MLX, synchronous: teacher-force the output under
//                           [eos] D… [eos] prompt and under the bare prompt (GroundingScorer)
//          3. measurement   no MLX, two actor hops: SinatraSession.prepareTurn(dryRun) reads
//                           the sources' content terms; groundingMeasurement turns the scores
//                           into classes, drift, risk and per-partition attribution A_p
//          4. join          GroundingRecord.join puts RaoLM's columns beside Sinatra's
//
//  PIN:  Not Sendable: it holds the run's model, and MLX objects stay with one owner (the CLI
//        task, or the studio's MLX thread, which runs 1–2 itself and awaits 3). The chat
//        template path (`Harness`) is never used: RaoLM prompts are raw corpus token ids and
//        its tokenizer has no chat template. `dryRun: true` keeps every measurement
//        deterministic and writes nothing to the Sinatra store.
//

import Foundation
import MLX
import MLXLMCommon
import RaoLMCore
import RaoLMModel
import RaoLMProvenance
import RaoLMTraining
import SinatraHarness

/// How much of each step the measurement keeps.
public enum GroundingDetail: String, Codable, Sendable, CaseIterable {
    /// ι, KL, drift, class, risk and the tune per step.
    case summary
    /// Also the token's rank without the source and the source's strongest pushes.
    case full

    public var traceLevel: TraceLevel { self == .full ? .full : .summary }

    public static func parse(_ text: String) throws -> GroundingDetail {
        guard let detail = GroundingDetail(rawValue: text.lowercased()) else { throw GroundingError.invalidDetail(text) }
        return detail
    }
}

public enum GroundingError: Error, CustomStringConvertible, Equatable, Sendable {
    case invalidPolicy(String)
    case invalidDetail(String)
    /// The policy selected nothing.
    case noSources(String)
    /// A named or cited partition is not in the run's corpus.
    case sourceNotInCorpus(String)
    /// A cited partition's text hash differs from the corpus's.
    case corpusMismatch(String)
    /// The generation was not made by the loaded checkpoint and tokenizer.
    case generationMismatch(String)
    /// Even one source and the output do not fit the model's positions.
    case contextTooLong(tokens: Int, limit: Int)
    /// The measurement's steps do not line up with the generation's traces.
    case alignment(String)
    /// The measurement stopped short and a result was required.
    case notMeasured(String)

    public var description: String {
        switch self {
        case .invalidPolicy(let text): return "unknown sources '\(text)': use \(GroundingSourcePolicy.help)"
        case .invalidDetail(let text): return "unknown detail '\(text)': use summary or full"
        case .noSources(let reason): return "no sources to measure against: \(reason)"
        case .sourceNotInCorpus(let reason): return "source not in the corpus: \(reason)"
        case .corpusMismatch(let reason): return "corpus mismatch: \(reason)"
        case .generationMismatch(let reason): return "generation mismatch: \(reason)"
        case .contextTooLong(let tokens, let limit):
            return "the context and output need \(tokens) positions; the model has \(limit)"
        case .alignment(let reason): return "grounding join misaligned: \(reason)"
        case .notMeasured(let reason): return "grounding not measured: \(reason)"
        }
    }

    public var hint: String? {
        switch self {
        case .invalidPolicy, .invalidDetail: return nil
        case .noSources: return "choose other sources: --sources \(GroundingSourcePolicy.help)"
        case .sourceNotInCorpus: return "partitions are named DOCUMENT_ID:PARTITION_INDEX of the run's corpus"
        case .corpusMismatch: return "the run's snapshot changed since the generation; regenerate it"
        case .generationMismatch: return "measure it against the run and epoch that generated it (--epoch)"
        case .contextTooLong: return "measure against fewer or shorter sources (--sources fact)"
        case .alignment: return nil
        case .notMeasured: return "raise --budget"
        }
    }

    /// sysexits-style: 64 usage, 65 data, 66 no input, 70 software, 75 temporary.
    public var exitCode: Int32 {
        switch self {
        case .invalidPolicy, .invalidDetail: return 64
        case .noSources, .sourceNotInCorpus: return 66
        case .corpusMismatch, .generationMismatch, .contextTooLong: return 65
        case .alignment: return 70
        case .notMeasured: return 75
        }
    }
}

/// Step 1's output: everything the scorer and the session read.
public struct GroundingInputs: Sendable {
    public var generationID: String
    /// The session's as-of time: the generation's own timestamp, so measurements repeat.
    public var createdAt: Date
    public var policy: GroundingSourcePolicy
    public var sources: [GroundingSource]
    /// Selected but dropped to fit the model's positions (least cited first).
    public var droppedSources: [GroundingSource]
    /// The sources as SinatraHarness partitions (id "documentID#partitionIndex", corpus token ids).
    public var partitions: [Partition]
    public var contextTokens: [Int]
    public var bareTokens: [Int]
    public var sampled: [Int]
    public var includesEOS: Bool
    public var contextExceedsTrainedLength: Bool

    /// Assemble the inputs for pre-chosen sources.
    ///
    /// - Parameters:
    ///   - maxPositions: the model's position limit; sources are dropped (least cited first)
    ///     until the context side fits, and `contextTooLong` is thrown if one still does not.
    ///   - trainedLength: the training sequence length, for `contextExceedsTrainedLength`.
    public static func make(
        generation: CitedGeneration, sources: [GroundingSource], policy: GroundingSourcePolicy, eos: Int,
        includeEOS: Bool = true, decode: ([Int]) -> String = { _ in "" }, maxPositions: Int? = nil,
        trainedLength: Int? = nil
    ) throws -> GroundingInputs {
        var seen = Set<String>()
        let unique = sources.filter { seen.insert($0.id).inserted }
        guard !unique.isEmpty else { throw GroundingError.noSources("the source list is empty") }
        let prompt = generation.prompt.tokens
        guard !prompt.isEmpty else { throw GroundingError.alignment("the generation has an empty prompt") }
        let sampled = GroundingSources.sampledTokens(generation, eos: eos, includeEOS: includeEOS)

        var kept = unique
        if let maxPositions {
            // The context side feeds context[0..<n−1], then [last] + sampled.dropLast():
            // n − 1 + |sampled| positions in all.
            let budget = maxPositions + 1 - max(sampled.count, 1)
            kept = GroundingSources.fit(unique, promptCount: prompt.count, budget: budget)
            let length = GroundingSources.contextLength(sources: kept, promptCount: prompt.count)
            if length > budget {
                throw GroundingError.contextTooLong(tokens: length - 1 + max(sampled.count, 1), limit: maxPositions)
            }
        }
        let keptIDs = Set(kept.map(\.id))
        let context = GroundingSources.contextTokens(sources: kept, prompt: prompt, eos: eos)
        let positions = context.count - 1 + sampled.count
        let partitions = kept.map { source in
            Partition(id: source.id, documentId: source.ref.documentID, text: decode(source.tokens), tokenIds: source.tokens, score: 0)
        }
        return GroundingInputs(
            generationID: generation.generationID, createdAt: generation.createdAt, policy: policy, sources: kept,
            droppedSources: unique.filter { !keptIDs.contains($0.id) }, partitions: partitions, contextTokens: context,
            bareTokens: prompt, sampled: sampled, includesEOS: includeEOS && generation.stoppedOnEOS,
            contextExceedsTrainedLength: trainedLength.map { positions > $0 } ?? false)
    }
}

public final class RaoGrounder {
    /// Measurements are dry runs, so the owner only names a (never written) ledger.
    public static let owner: OwnerID = "raolm-grounding"

    public let context: RunContext
    public let corpus: TokenizedCorpus
    public let session: SinatraSession
    public let configuration: SinatraConfiguration
    public let storeDirectory: URL
    /// Score the EOS the generation stopped on as a final step (default: yes).
    public var includeEOS = true

    /// - Parameters:
    ///   - corpus: the run's training tokenization (`context.tokenizedCorpus()`, or the
    ///     evaluator's own), whose rows the provenance index cites.
    ///   - storeDirectory: the Sinatra store (default `<run>/grounding`; dry runs never write it).
    ///   - budget: wall-clock seconds one measurement may take.
    public init(context: RunContext, corpus: TokenizedCorpus, storeDirectory: URL? = nil, budget: TimeInterval = 60) {
        self.context = context
        self.corpus = corpus
        let store = storeDirectory ?? context.runDirectory.appendingPathComponent("grounding", isDirectory: true)
        self.storeDirectory = store
        let configuration = Self.configuration(budget: budget)
        self.configuration = configuration
        self.session = Self.makeSession(
            tokenizer: context.tokenizer, modelKey: context.checkpointSHA256, vocabularySize: context.model.vocabularySize,
            storeDirectory: store, configuration: configuration)
    }

    /// Sinatra measuring, never steering: no injection, every turn measured, and partitions
    /// read whole (a RaoLM partition is never clipped).
    public static func configuration(budget: TimeInterval = 60) -> SinatraConfiguration {
        var configuration = SinatraConfiguration()
        configuration.biasMode = .off
        configuration.groundingMeasurement = .always
        configuration.traceLevel = .summary
        configuration.groundingBudget = budget
        configuration.maxTokensPerPartition = 4096
        configuration.maxPartitionSequence = 4096
        configuration.maxPartitionsPerTurn = GroundingSources.defaultLimit
        return configuration
    }

    /// The session every measurement reads its sources through. No MLX: hashed context
    /// vectors and prior weights.
    public static func makeSession(
        tokenizer: RaoTokenizer, modelKey: String, vocabularySize: Int, storeDirectory: URL,
        configuration: SinatraConfiguration = configuration()
    ) -> SinatraSession {
        SinatraSession(
            configuration: configuration, storeDirectory: storeDirectory, modelKey: modelKey,
            tokenizer: RaoSinatraTokenizer(tokenizer), encoder: HashingContextEncoder(), vocabularySize: vocabularySize,
            weightModelFactory: PriorWeightModel.factory)
    }

    // MARK: - Steps

    /// Refuse a generation another checkpoint or tokenizer produced.
    public func checkBinding(_ generation: CitedGeneration) throws {
        guard generation.manifest.checkpointSHA256 == context.checkpointSHA256 else {
            throw GroundingError.generationMismatch(
                "generation \(generation.generationID) was made by checkpoint \(generation.manifest.checkpointSHA256.prefix(12))… (epoch \(generation.manifest.epoch)); the loaded epoch \(context.epoch) is \(context.checkpointSHA256.prefix(12))…")
        }
        guard generation.manifest.tokenizerSHA256 == context.tokenizer.tokenizerSHA256 else {
            throw GroundingError.generationMismatch("generation \(generation.generationID) used another tokenizer")
        }
    }

    /// Step 1 with a policy over the run's corpus.
    public func inputs(
        for generation: CitedGeneration, policy: GroundingSourcePolicy, limit: Int = GroundingSources.defaultLimit
    ) throws -> GroundingInputs {
        let sources = try GroundingSources.resolve(
            policy: policy, generation: generation, corpus: corpus, threadID: context.manifestRef.threadID, limit: limit)
        return try inputs(for: generation, sources: sources, policy: policy)
    }

    /// Step 1 with pre-built sources (a held-out partition is absent from the training corpus).
    public func inputs(for generation: CitedGeneration, sources: [GroundingSource], policy: GroundingSourcePolicy) throws -> GroundingInputs {
        let tokenizer = context.tokenizer
        return try GroundingInputs.make(
            generation: generation, sources: sources, policy: policy, eos: tokenizer.eosTokenID, includeEOS: includeEOS,
            decode: { tokenizer.decode($0) }, maxPositions: context.model.config.maxPositionEmbeddings,
            trainedLength: context.manifest.hyperparameters.seqLen)
    }

    /// Step 2: teacher-force the output with and without the sources. Synchronous MLX on the
    /// caller's thread; throws `GroundingScorer.Stop` when the budget runs out or `shouldAbort`.
    public func score(
        _ inputs: GroundingInputs, detail: GroundingDetail = .summary, budget: TimeInterval? = nil,
        shouldAbort: () -> Bool = { false }
    ) throws -> GroundingRaw {
        var options = GroundingScorer.Options(configuration: configuration, detail: detail.traceLevel)
        if let budget { options.budget = budget }
        return try GroundingScorer.score(
            model: context.model, decodeCache: nil, contextTokens: inputs.contextTokens, bareTokens: inputs.bareTokens,
            sampled: inputs.sampled, options: options, shouldAbort: shouldAbort)
    }

    /// Step 3: the scores read against the sources. `raw == nil` gives `.skipped(reason)`.
    public func measurement(
        _ inputs: GroundingInputs, raw: GroundingRaw?, skipped reason: String? = nil, detail: GroundingDetail = .summary
    ) async throws -> GroundingMeasurement {
        try await Self.measurement(session: session, inputs: inputs, raw: raw, skipped: reason, detail: detail)
    }

    public static func measurement(
        session: SinatraSession, inputs: GroundingInputs, raw: GroundingRaw?, skipped reason: String? = nil,
        detail: GroundingDetail = .summary
    ) async throws -> GroundingMeasurement {
        guard let raw else { return .skipped(reason ?? "not scored") }
        let turn = TurnInput(owner: owner, retrieved: inputs.partitions, conversationId: inputs.generationID, now: inputs.createdAt)
        let plan = try await session.prepareTurn(turn, mode: .off, dryRun: true)
        return await session.groundingMeasurement(plan: plan, raw: raw, detail: detail.traceLevel)
    }

    /// Step 4: the join.
    public func record(generation: CitedGeneration, inputs: GroundingInputs, measurement: GroundingMeasurement) throws -> GroundingRecord {
        try Self.record(generation: generation, inputs: inputs, measurement: measurement, eos: context.tokenizer.eosTokenID)
    }

    public static func record(
        generation: CitedGeneration, inputs: GroundingInputs, measurement: GroundingMeasurement, eos: Int
    ) throws -> GroundingRecord {
        try GroundingRecord.join(
            generation: generation, sources: inputs.sources, droppedSources: inputs.droppedSources, measurement: measurement,
            policy: inputs.policy, contextTokens: inputs.contextTokens.count, sampled: inputs.sampled,
            includesEOS: inputs.includesEOS, contextExceedsTrainedLength: inputs.contextExceedsTrainedLength, eos: eos)
    }

    // MARK: - All four

    /// Measure a generation against the sources a policy selects. A measurement that stops
    /// short (budget, abort) comes back with `measured == false` unless `requireMeasured`,
    /// which throws `GroundingError.notMeasured` instead.
    public func measure(
        generation: CitedGeneration, policy: GroundingSourcePolicy, detail: GroundingDetail = .summary,
        budget: TimeInterval? = nil, requireMeasured: Bool = false, shouldAbort: () -> Bool = { false }
    ) async throws -> GroundingRecord {
        try checkBinding(generation)
        let inputs = try inputs(for: generation, policy: policy)
        return try await run(generation, inputs, detail: detail, budget: budget, requireMeasured: requireMeasured, shouldAbort: shouldAbort)
    }

    /// Measure a generation against pre-built sources.
    public func measure(
        generation: CitedGeneration, sources: [GroundingSource], policy: GroundingSourcePolicy? = nil,
        detail: GroundingDetail = .summary, budget: TimeInterval? = nil, requireMeasured: Bool = false,
        shouldAbort: () -> Bool = { false }
    ) async throws -> GroundingRecord {
        try checkBinding(generation)
        let named = policy ?? .explicit(sources.map(\.address))
        let inputs = try inputs(for: generation, sources: sources, policy: named)
        return try await run(generation, inputs, detail: detail, budget: budget, requireMeasured: requireMeasured, shouldAbort: shouldAbort)
    }

    private func run(
        _ generation: CitedGeneration, _ inputs: GroundingInputs, detail: GroundingDetail, budget: TimeInterval?,
        requireMeasured: Bool, shouldAbort: () -> Bool
    ) async throws -> GroundingRecord {
        var raw: GroundingRaw?
        var reason: String?
        do {
            raw = try score(inputs, detail: detail, budget: budget, shouldAbort: shouldAbort)
        } catch let stop as GroundingScorer.Stop {
            if requireMeasured { throw GroundingError.notMeasured(stop.description) }
            reason = stop.description
        }
        let measurement = try await measurement(inputs, raw: raw, skipped: reason, detail: detail)
        return try record(generation: generation, inputs: inputs, measurement: measurement)
    }

    // MARK: - Evaluation

    /// The closure `FactEvaluator.groundingMeasurer` calls: measure each generation against
    /// the fact's true source partition and summarise it (saving the full record when asked,
    /// as `<run>/generations/<generationID>.grounding.json`).
    public func makeMeasurer(
        runDirectory: URL?, threadID: String? = nil, saveRecords: Bool, detail: GroundingDetail = .summary,
        budget: TimeInterval? = nil
    ) -> GroundingMeasurer {
        return { [self] generation, evalSource in
            let weights = GroundingSources.citationWeightsByID(generation)
            let id = GroundingSources.sourceID(documentID: evalSource.documentID, partitionIndex: evalSource.partitionIndex)
            let source = GroundingSource(
                evalSource: evalSource, citationWeight: weights[id] ?? 0,
                threadID: threadID ?? generation.manifest.threadID)
            let policy: GroundingSourcePolicy = evalSource.group == .fact && generation.prompt.source != nil
                ? .promptSource : .explicit([source.address])
            let record = try await self.measure(
                generation: generation, sources: [source], policy: policy, detail: detail, budget: budget, requireMeasured: false)
            var file: String?
            if saveRecords, let runDirectory {
                let url = GroundingRecord.url(runDirectory: runDirectory, generationID: generation.generationID)
                try record.save(to: url)
                file = url.path
            }
            return GroundingFactSummary(record: record, answerLength: evalSource.answerLength, recordFile: file)
        }
    }
}

extension GroundingFactSummary {
    /// The evaluator's view of a record: Sinatra's turn summary plus the answer tokens' ι and risk.
    public init(record: GroundingRecord, answerLength: Int, recordFile: String? = nil) {
        let m = record.measurement
        guard m.measured else {
            self.init(measured: false, skippedReason: m.skippedReason ?? "not measured", recordFile: recordFile)
            return
        }
        let s = m.summary
        let answer = record.tokens.filter { !$0.isEOS }.prefix(max(0, answerLength))
        self.init(
            measured: true, skippedReason: nil, steps: s.steps, contentTokens: s.contentTokens, grounding: s.grounding,
            unsupportedShare: s.unsupportedShare, contradictedShare: s.contradictedShare, drift: s.drift,
            driftShare: s.driftShare, contextDependence: s.contextDependence, meanContextKL: s.meanContextKL,
            hallucinationRisk: s.hallucinationRisk,
            meanAnswerInfluence: answer.isEmpty ? nil : Stats.mean(answer.map(\.influence)),
            meanAnswerRisk: answer.isEmpty ? nil : Stats.mean(answer.map(\.risk)), recordFile: recordFile)
    }
}
