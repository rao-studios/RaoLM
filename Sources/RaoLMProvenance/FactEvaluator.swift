//
//  FactEvaluator.swift
//  RaoLMProvenance
//
//  WHAT: The evaluation protocol for citations. For a seeded sample of facts, prompt the
//        model with the corpus tokens that precede each answer and measure: exact answers,
//        whether the top neighbour (and the top citation) points at the fact's source
//        partition and offset, whether the answer is covered by a span verified against the
//        live source, calibration of citation confidence, the λ ablation, and two controls
//        (paraphrased prompts; prompts about entities that do not exist). With a grounding
//        measurer injected (RaoLMGrounding), each primary generation — and each control's —
//        is also scored with and without the fact's source partition in front of its prompt.
//  PIN:  Prompts are token slices of the source partition, never re-tokenized strings: a
//        string-initial "The" lacks the corpus's "ĠThe" and would change the context.
//        Cancellation is checked before every fact, so cancelling the task stops an eval.
//

import Foundation
import RaoLMCore
import RaoLMModel
import RaoLMTraining

public struct EvalOptions: Sendable {
    public var factsSample: Int
    public var lambdas: [Float]
    public var primaryLambda: Float
    public var seed: UInt64
    public var includeControls: Bool
    public var saveGenerations: URL?
    /// Measure grounding at the primary λ (needs `FactEvaluator.groundingMeasurer`).
    public var grounding: GroundingEvalOptions?

    public init(
        factsSample: Int = 50, lambdas: [Float] = [0, 0.25, 0.5, 0.75], primaryLambda: Float = 0.5, seed: UInt64 = 7,
        includeControls: Bool = true, saveGenerations: URL? = nil, grounding: GroundingEvalOptions? = nil
    ) {
        self.factsSample = factsSample
        self.lambdas = lambdas
        self.primaryLambda = primaryLambda
        self.seed = seed
        self.includeControls = includeControls
        self.saveGenerations = saveGenerations
        self.grounding = grounding
    }
}

public struct FactOutcome: Codable, Sendable {
    public var factID: String
    public var kind: FactKind
    public var documentID: String
    public var documentName: String
    public var partitionIndex: Int
    public var lambda: Float
    public var prompt: String
    public var expected: String
    public var generated: String
    public var exact: Bool
    /// Fraction of answer tokens whose rank-1 neighbour's value position is in the source partition.
    public var citationAt1Partition: Float
    /// … and at the expected offset.
    public var citationAt1Offset: Float
    /// Fraction of answer tokens whose top citation is the source partition.
    public var citedPartitionAt1: Float
    public var topCitation: SourceAddress?
    public var topCitationName: String?
    public var answerCoveredByVerbatimSpan: Bool
    public var spansVerified: Int?
    public var spansChecked: Int?
    public var meanAnswerConfidence: Float
    public var generationFile: String?
    /// The grounding measurement against the fact's source partition (primary λ only; nil
    /// when grounding was off, and in reports written before it existed).
    public var grounding: GroundingFactSummary?
}

public struct LambdaMetrics: Codable, Sendable {
    public var lambda: Float
    public var facts: Int
    public var exactAnswer: Float
    public var citationAt1Partition: Float
    public var citationAt1PartitionOnCorrect: Float?
    public var citationAt1Offset: Float
    public var citedPartitionAt1: Float
    public var answerCoveredByVerbatimSpan: Float
    public var meanAnswerConfidence: Float
}

public struct CalibrationBin: Codable, Sendable {
    public var lower: Float
    public var upper: Float
    public var count: Int
    public var meanConfidence: Float
    public var accuracy: Float
}

public struct ControlMetrics: Codable, Sendable {
    public var paraphrasePrompts: Int
    public var paraphraseExact: Float
    public var paraphraseMeanConfidence: Float
    public var negativePrompts: Int
    public var negativeMeanConfidence: Float
    public var negativeDistinctiveVerbatimSpans: Int
    public var correctAnswerMeanConfidence: Float
    /// Facts from documents the run excluded from training (the leave-out control).
    public var heldOutFacts: Int?
    public var heldOutExact: Float?
    public var heldOutMeanConfidence: Float?
    /// Held-out answer tokens whose top citation names the held-out document (0 by construction:
    /// the index holds no position of a document the model never trained on).
    public var heldOutCitedSource: Int?
}

public struct EvalReport: Codable, Sendable {
    public var runID: String
    public var epoch: Int
    public var createdAt: Date
    public var sampleSize: Int
    public var primaryLambda: Float
    public var lambdas: [LambdaMetrics]
    public var spanChecks: Int
    public var spansVerified: Int
    public var spanVerifiedRate: Float?
    public var calibration: [CalibrationBin]
    public var ece: Float
    public var auroc: Double?
    public var controls: ControlMetrics?
    public var outcomes: [FactOutcome]
    public var unalignedFacts: [String]
    /// The two-fold section: nil when grounding was off (and in older reports).
    public var grounding: GroundingEvalReport?

    public func metrics(lambda: Float) -> LambdaMetrics? {
        lambdas.first { abs($0.lambda - lambda) < 1e-6 }
    }
}

public final class FactEvaluator {
    public let context: RunContext
    public let corpus: TokenizedCorpus
    public let facts: [LocatedFact]
    public let unaligned: [String]
    public let reader: CorpusReading?
    /// Facts of documents the run held out, located in a tokenization of the full snapshot.
    public let heldOut: [LocatedFact]
    private let fullCorpus: TokenizedCorpus?
    /// Scores a generation with and without its source partition; set it (RaoLMGrounding
    /// builds one) and pass `EvalOptions.grounding` to add the two-fold measurement.
    public var groundingMeasurer: GroundingMeasurer?

    public init(context: RunContext, corpus: TokenizedCorpus, facts: [Fact], reader: CorpusReading?) {
        self.context = context
        self.corpus = corpus
        let located = FactLocator.locate(facts, corpus: corpus, tokenizer: context.tokenizer)
        self.facts = located.located
        self.unaligned = located.unaligned
        self.reader = reader
        let excluded = Set(context.manifest.excludedDocumentIDs)
        if !excluded.isEmpty, let snapshot = try? context.snapshot() {
            let full = TokenizedCorpus(snapshot: snapshot, tokenizer: context.tokenizer)
            self.fullCorpus = full
            self.heldOut = FactLocator.locate(facts.filter { excluded.contains($0.documentID) }, corpus: full, tokenizer: context.tokenizer).located
        } else {
            self.fullCorpus = nil
            self.heldOut = []
        }
    }

    public func sample(_ count: Int, seed: UInt64) -> [LocatedFact] {
        var rng = SplitMix64(seed: seed)
        return Array(rng.shuffled(facts).prefix(count))
    }

    private struct TokenLabel {
        let confidence: Float
        let correct: Bool
    }

    public func run(options: EvalOptions, progress: (String) -> Void = { _ in }) async throws -> EvalReport {
        let generator = context.generator()
        let tokenizer = context.tokenizer
        let sampled = sample(options.factsSample, seed: options.seed)
        var metrics: [LambdaMetrics] = []
        var primaryOutcomes: [FactOutcome] = []
        var calibrationPool: [TokenLabel] = []
        var spanChecks = 0
        var spansVerified = 0
        var correctConfidences: [Float] = []
        let measurer = options.grounding == nil ? nil : groundingMeasurer
        let groundControls = measurer != nil && options.includeControls && (options.grounding?.includeControls ?? false)
        var groundingControls: [GroundingControlOutcome] = []

        let lambdas = options.lambdas.contains(options.primaryLambda) ? options.lambdas : options.lambdas + [options.primaryLambda]
        for lambda in lambdas {
            var outcomes: [FactOutcome] = []
            for (n, located) in sampled.enumerated() {
                try Task.checkCancellation()
                let isPrimary = abs(lambda - options.primaryLambda) < 1e-6
                let partition = corpus.partitions[located.row]
                let promptTokens = partition.tokens[located.contextToken..<located.answerToken].map(Int.init)
                let expected = partition.tokens[located.answerToken..<located.answerEndToken].map(Int.init)
                var params = context.defaultParameters()
                params.lambda = lambda
                params.maxTokens = located.answerLength + 2
                let request = GenerationRequest(
                    promptTokens: promptTokens, promptText: tokenizer.decode(promptTokens),
                    promptSource: SourceAddress(
                        threadID: context.manifestRef.threadID, documentID: partition.documentID,
                        partitionIndex: partition.partitionIndex, tokenOffset: located.contextToken,
                        partitionURL: partition.url, threadPartitionID: partition.threadPartitionID),
                    params: params)
                var generation = try generator.generate(request)
                var verifiedCount: Int?
                var checkedCount: Int?
                if isPrimary, let reader {
                    let report = try await CitationVerifier.verify(&generation, reader: reader, tokenizer: tokenizer)
                    verifiedCount = report.verified
                    checkedCount = report.checks.count
                    spanChecks += report.checks.count
                    spansVerified += report.verified
                }
                let outcome = score(
                    located: located, expected: expected, generation: generation, lambda: lambda,
                    verified: verifiedCount, checked: checkedCount)
                var saved = outcome
                if isPrimary, let directory = options.saveGenerations {
                    let url = directory.appendingPathComponent("eval-\(String(format: "%03d", n)).json")
                    try generation.save(to: url)
                    saved.generationFile = url.path
                }
                if isPrimary, let measurer {
                    saved.grounding = try await Self.ground(
                        generation, Self.groundingSource(.fact, located: located, partition: partition), measurer: measurer)
                }
                outcomes.append(saved)
                if isPrimary {
                    let answerTraces = Array(generation.traces.filter { !$0.isPrompt }.prefix(located.answerLength))
                    for trace in answerTraces {
                        let correct = trace.citations.first?.row == located.row
                        calibrationPool.append(TokenLabel(confidence: trace.confidence ?? 0, correct: correct))
                    }
                    if outcome.exact { correctConfidences.append(outcome.meanAnswerConfidence) }
                }
            }
            let m = aggregate(lambda: lambda, outcomes: outcomes)
            metrics.append(m)
            if abs(lambda - options.primaryLambda) < 1e-6 { primaryOutcomes = outcomes }
            progress(String(
                format: "λ=%.2f  exact %.0f%%  citation@1 %.0f%%  cited@1 %.0f%%  covered %.0f%%",
                lambda, m.exactAnswer * 100, m.citationAt1Partition * 100, m.citedPartitionAt1 * 100,
                m.answerCoveredByVerbatimSpan * 100))
        }

        var controls: ControlMetrics?
        if options.includeControls {
            var params = context.defaultParameters()
            params.lambda = options.primaryLambda
            var paraphraseExact = 0
            var paraphraseCount = 0
            var paraphraseConfidences: [Float] = []
            var negativeConfidences: [Float] = []
            var negativeDistinctive = 0
            for located in sampled {
                try Task.checkCancellation()
                let partition = corpus.partitions[located.row]
                let expected = partition.tokens[located.answerToken..<located.answerEndToken].map(Int.init)
                params.maxTokens = located.answerLength + 2
                for paraphrase in located.fact.paraphrases.prefix(1) {
                    let tokens = tokenizer.encode(paraphrase)
                    let generation = try generator.generate(GenerationRequest(promptTokens: tokens, promptText: paraphrase, params: params))
                    if groundControls, let measurer {
                        let summary = try await Self.ground(
                            generation, Self.groundingSource(.paraphrase, located: located, partition: partition), measurer: measurer)
                        groundingControls.append(GroundingControlOutcome(
                            factID: located.fact.id, group: .paraphrase, prompt: paraphrase, generated: generation.text, summary: summary))
                    }
                    paraphraseCount += 1
                    if Array(generation.tokens.prefix(expected.count)) == expected { paraphraseExact += 1 }
                    for trace in generation.traces.filter({ !$0.isPrompt }).prefix(expected.count) {
                        paraphraseConfidences.append(trace.confidence ?? 0)
                        calibrationPool.append(TokenLabel(
                            confidence: trace.confidence ?? 0, correct: trace.citations.first?.row == located.row))
                    }
                }
                let negativeTokens = tokenizer.encode(located.fact.negativePrompt)
                var negative = try generator.generate(GenerationRequest(
                    promptTokens: negativeTokens, promptText: located.fact.negativePrompt, params: params))
                for trace in negative.traces.filter({ !$0.isPrompt }).prefix(expected.count) {
                    negativeConfidences.append(trace.confidence ?? 0)
                    calibrationPool.append(TokenLabel(confidence: trace.confidence ?? 0, correct: false))
                }
                if let reader {
                    _ = try await CitationVerifier.verify(&negative, reader: reader, tokenizer: tokenizer)
                }
                if groundControls, let measurer {
                    let summary = try await Self.ground(
                        negative, Self.groundingSource(.fabricated, located: located, partition: partition), measurer: measurer)
                    groundingControls.append(GroundingControlOutcome(
                        factID: located.fact.id, group: .fabricated, prompt: located.fact.negativePrompt,
                        generated: negative.text, summary: summary))
                }
                negativeDistinctive += negative.spans.filter {
                    $0.kind == .verbatim && $0.distinctiveness > 0 && ($0.verification?.status ?? .verified) == .verified
                }.count
            }
            var heldOutExact = 0
            var heldOutConfidences: [Float] = []
            var heldOutCited = 0
            let heldOutSample = Array(heldOut.prefix(options.factsSample))
            if let fullCorpus {
                for located in heldOutSample {
                    try Task.checkCancellation()
                    let partition = fullCorpus.partitions[located.row]
                    let prompt = partition.tokens[located.contextToken..<located.answerToken].map(Int.init)
                    let expected = partition.tokens[located.answerToken..<located.answerEndToken].map(Int.init)
                    params.maxTokens = located.answerLength + 2
                    let generation = try generator.generate(GenerationRequest(
                        promptTokens: prompt, promptText: tokenizer.decode(prompt), params: params))
                    if groundControls, let measurer {
                        let summary = try await Self.ground(
                            generation, Self.groundingSource(.heldOut, located: located, partition: partition), measurer: measurer)
                        groundingControls.append(GroundingControlOutcome(
                            factID: located.fact.id, group: .heldOut, prompt: generation.prompt.text,
                            generated: generation.text, summary: summary))
                    }
                    if Array(generation.tokens.prefix(expected.count)) == expected { heldOutExact += 1 }
                    for trace in generation.traces.filter({ !$0.isPrompt }).prefix(expected.count) {
                        heldOutConfidences.append(trace.confidence ?? 0)
                        if trace.citations.first?.address.documentID == located.fact.documentID { heldOutCited += 1 }
                        calibrationPool.append(TokenLabel(confidence: trace.confidence ?? 0, correct: false))
                    }
                }
            }
            controls = ControlMetrics(
                paraphrasePrompts: paraphraseCount,
                paraphraseExact: paraphraseCount > 0 ? Float(paraphraseExact) / Float(paraphraseCount) : 0,
                paraphraseMeanConfidence: Stats.mean(paraphraseConfidences),
                negativePrompts: sampled.count, negativeMeanConfidence: Stats.mean(negativeConfidences),
                negativeDistinctiveVerbatimSpans: negativeDistinctive,
                correctAnswerMeanConfidence: Stats.mean(correctConfidences),
                heldOutFacts: heldOutSample.isEmpty ? nil : heldOutSample.count,
                heldOutExact: heldOutSample.isEmpty ? nil : Float(heldOutExact) / Float(heldOutSample.count),
                heldOutMeanConfidence: heldOutSample.isEmpty ? nil : Stats.mean(heldOutConfidences),
                heldOutCitedSource: heldOutSample.isEmpty ? nil : heldOutCited)
            if let controls {
                progress(String(
                    format: "controls  paraphrase exact %.0f%%  negative confidence %.2f vs correct %.2f",
                    controls.paraphraseExact * 100, controls.negativeMeanConfidence, controls.correctAnswerMeanConfidence))
            }
        }

        var grounding: GroundingEvalReport?
        if measurer != nil {
            let report = GroundingEvalReport.make(outcomes: primaryOutcomes, controls: groundingControls)
            grounding = report
            let measured = primaryOutcomes.filter { $0.grounding?.measured == true }.count
            progress(
                "grounding  measured \(measured)/\(primaryOutcomes.count) facts"
                    + (groundingControls.isEmpty ? "" : " and \(groundingControls.filter(\.summary.measured).count)/\(groundingControls.count) controls")
                    + (report.riskAUROCForWrongAnswer.map { String(format: "  risk AUROC (wrong answer) %.3f", $0) } ?? ""))
        }

        let (bins, ece) = Self.calibration(calibrationPool.map { ($0.confidence, $0.correct) })
        let auroc = Stats.auroc(scores: calibrationPool.map { Double($0.confidence) }, labels: calibrationPool.map(\.correct))
        return EvalReport(
            runID: context.manifest.runID, epoch: context.epoch, createdAt: Date(), sampleSize: sampled.count,
            primaryLambda: options.primaryLambda, lambdas: metrics, spanChecks: spanChecks, spansVerified: spansVerified,
            spanVerifiedRate: spanChecks > 0 ? Float(spansVerified) / Float(spanChecks) : nil,
            calibration: bins, ece: ece, auroc: auroc, controls: controls, outcomes: primaryOutcomes,
            unalignedFacts: unaligned, grounding: grounding)
    }

    // MARK: - Grounding

    /// The fact's true source partition, as the measurer places it in front of a prompt.
    static func groundingSource(_ group: GroundingEvalGroup, located: LocatedFact, partition: TokenizedPartition) -> GroundingEvalSource {
        GroundingEvalSource(
            group: group, factID: located.fact.id, row: partition.row, documentID: partition.documentID,
            documentName: partition.documentName, partitionIndex: partition.partitionIndex,
            tokens: partition.tokens.map(Int.init), textSHA256: partition.textSHA256, url: partition.url,
            threadPartitionID: partition.threadPartitionID, answerLength: located.answerLength)
    }

    /// Runs the injected measurer. Anything but cancellation becomes a skipped summary with the
    /// error as its reason, counted in the report rather than ending the evaluation.
    static func ground(
        _ generation: CitedGeneration, _ source: GroundingEvalSource, measurer: GroundingMeasurer
    ) async throws -> GroundingFactSummary {
        try Task.checkCancellation()
        do {
            return try await measurer(generation, source) ?? .skipped("no measurement")
        } catch let cancelled as CancellationError {
            throw cancelled
        } catch {
            return .skipped("\(error)")
        }
    }

    private func score(
        located: LocatedFact, expected: [Int], generation: CitedGeneration, lambda: Float, verified: Int?, checked: Int?
    ) -> FactOutcome {
        let answerTraces = Array(generation.traces.filter { !$0.isPrompt }.prefix(located.answerLength))
        var partitionHits = 0
        var offsetHits = 0
        var citedHits = 0
        var confidences: [Float] = []
        for (i, trace) in answerTraces.enumerated() {
            if let top = trace.neighbours.first {
                if top.cited.row == located.row { partitionHits += 1 }
                if top.cited.row == located.row && top.cited.offset == located.answerToken + i { offsetHits += 1 }
            }
            if trace.citations.first?.row == located.row { citedHits += 1 }
            confidences.append(trace.confidence ?? 0)
        }
        let denominator = Float(max(located.answerLength, 1))
        let answerStart = generation.prompt.tokens.count
        let answerRange = answerStart..<(answerStart + located.answerLength)
        let covered = generation.spans.contains { span in
            span.kind == .verbatim && span.row == located.row && span.tokenRange.start <= answerRange.lowerBound
                && span.tokenRange.end >= answerRange.upperBound
                && (span.verification.map { $0.status == .verified } ?? true)
        }
        let top = answerTraces.first?.citations.first
        return FactOutcome(
            factID: located.fact.id, kind: located.fact.kind, documentID: located.fact.documentID,
            documentName: corpus.partitions[located.row].documentName, partitionIndex: located.fact.partitionIndex,
            lambda: lambda, prompt: generation.prompt.text, expected: located.fact.answer, generated: generation.text,
            exact: Array(generation.tokens.prefix(expected.count)) == expected,
            citationAt1Partition: Float(partitionHits) / denominator,
            citationAt1Offset: Float(offsetHits) / denominator,
            citedPartitionAt1: Float(citedHits) / denominator,
            topCitation: top?.address, topCitationName: top.flatMap { generation.partition(row: $0.row)?.documentName },
            answerCoveredByVerbatimSpan: covered, spansVerified: verified, spansChecked: checked,
            meanAnswerConfidence: Stats.mean(confidences), generationFile: nil)
    }

    private func aggregate(lambda: Float, outcomes: [FactOutcome]) -> LambdaMetrics {
        let n = Float(max(outcomes.count, 1))
        let correct = outcomes.filter(\.exact)
        return LambdaMetrics(
            lambda: lambda, facts: outcomes.count,
            exactAnswer: Float(correct.count) / n,
            citationAt1Partition: outcomes.map(\.citationAt1Partition).reduce(0, +) / n,
            citationAt1PartitionOnCorrect: correct.isEmpty ? nil
                : correct.map(\.citationAt1Partition).reduce(0, +) / Float(correct.count),
            citationAt1Offset: outcomes.map(\.citationAt1Offset).reduce(0, +) / n,
            citedPartitionAt1: outcomes.map(\.citedPartitionAt1).reduce(0, +) / n,
            answerCoveredByVerbatimSpan: Float(outcomes.filter(\.answerCoveredByVerbatimSpan).count) / n,
            meanAnswerConfidence: outcomes.map(\.meanAnswerConfidence).reduce(0, +) / n)
    }

    /// Ten equal-width confidence bins; ECE = Σ (n_b/N)·|accuracy_b − confidence_b|.
    public static func calibration(_ pool: [(Float, Bool)]) -> ([CalibrationBin], Float) {
        var bins: [CalibrationBin] = []
        var ece: Float = 0
        let total = Float(max(pool.count, 1))
        for b in 0..<10 {
            let lower = Float(b) / 10
            let upper = Float(b + 1) / 10
            let members = pool.filter { $0.0 >= lower && ($0.0 < upper || (b == 9 && $0.0 <= 1)) }
            guard !members.isEmpty else {
                bins.append(CalibrationBin(lower: lower, upper: upper, count: 0, meanConfidence: 0, accuracy: 0))
                continue
            }
            let confidence = Stats.mean(members.map(\.0))
            let accuracy = Float(members.filter(\.1).count) / Float(members.count)
            bins.append(CalibrationBin(lower: lower, upper: upper, count: members.count, meanConfidence: confidence, accuracy: accuracy))
            ece += Float(members.count) / total * abs(accuracy - confidence)
        }
        return (bins, ece)
    }
}
