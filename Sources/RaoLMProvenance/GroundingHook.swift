//
//  GroundingHook.swift
//  RaoLMProvenance
//
//  WHAT: The evaluator's side of the two-fold harness. RaoLM's citations say which corpus
//        positions support each token; the grounding measurement (RaoLMGrounding, over
//        SinatraHarness) scores the same output with and without the source partition in
//        front of the prompt and says how much the source's presence moved the model (ι).
//        This file holds only the plain values the evaluator records and the closure type
//        it calls; the measurement itself is injected.
//  PIN:  No SinatraHarness types here: the dependency runs Grounding → Provenance, never
//        back. Every field the evaluator adds is optional, so an eval.json written before
//        grounding existed still decodes.
//

import Foundation
import RaoLMCore

/// Which generation of the evaluation a measurement belongs to.
public enum GroundingEvalGroup: String, Codable, Sendable, CaseIterable {
    /// The fact's own corpus prompt at the primary λ.
    case fact
    /// The fact's first paraphrased prompt (a stem that occurs nowhere in the corpus).
    case paraphrase
    /// The same template about an entity that does not exist.
    case fabricated
    /// A fact of a document the run excluded from training.
    case heldOut
}

/// The prompted source a generation is measured against: always the fact's true source
/// partition, whatever the prompt was (for the controls that is the point of the control).
public struct GroundingEvalSource: Sendable, Equatable {
    public var group: GroundingEvalGroup
    public var factID: String
    /// Row of the partition in the tokenization it was taken from: the training corpus, or
    /// for held-out facts a tokenization of the full snapshot (absent from the index).
    public var row: Int
    public var documentID: String
    public var documentName: String
    public var partitionIndex: Int
    /// The partition's own tokenization, exactly as training packed it.
    public var tokens: [Int]
    public var textSHA256: String
    public var url: String?
    public var threadPartitionID: String?
    /// The answer's token count: the first `answerLength` generated tokens are the answer.
    public var answerLength: Int

    public init(
        group: GroundingEvalGroup, factID: String, row: Int, documentID: String, documentName: String,
        partitionIndex: Int, tokens: [Int], textSHA256: String, url: String?, threadPartitionID: String?,
        answerLength: Int
    ) {
        self.group = group
        self.factID = factID
        self.row = row
        self.documentID = documentID
        self.documentName = documentName
        self.partitionIndex = partitionIndex
        self.tokens = tokens
        self.textSHA256 = textSHA256
        self.url = url
        self.threadPartitionID = threadPartitionID
        self.answerLength = answerLength
    }
}

/// What the grounding measurement said about one generation of the evaluation.
public struct GroundingFactSummary: Codable, Sendable, Equatable {
    /// false: the measurement stopped short (`skippedReason` says why).
    public var measured: Bool
    public var skippedReason: String?
    public var steps: Int
    public var contentTokens: Int
    /// Share of content tokens the source pushed for (ι ≥ threshold).
    public var grounding: Float
    public var unsupportedShare: Float
    public var contradictedShare: Float
    /// Mean drift over content tokens, nats.
    public var drift: Float
    public var driftShare: Float
    /// Σ ι over every step, nats.
    public var contextDependence: Float
    public var meanContextKL: Float
    /// Mean risk over content tokens.
    public var hallucinationRisk: Float
    /// Mean ι over the answer tokens (nil when none were measured).
    public var meanAnswerInfluence: Float?
    /// Mean risk over the answer tokens.
    public var meanAnswerRisk: Float?
    /// The full grounding record, when it was saved.
    public var recordFile: String?

    public init(
        measured: Bool, skippedReason: String? = nil, steps: Int = 0, contentTokens: Int = 0, grounding: Float = 0,
        unsupportedShare: Float = 0, contradictedShare: Float = 0, drift: Float = 0, driftShare: Float = 0,
        contextDependence: Float = 0, meanContextKL: Float = 0, hallucinationRisk: Float = 0,
        meanAnswerInfluence: Float? = nil, meanAnswerRisk: Float? = nil, recordFile: String? = nil
    ) {
        self.measured = measured
        self.skippedReason = skippedReason
        self.steps = steps
        self.contentTokens = contentTokens
        self.grounding = grounding
        self.unsupportedShare = unsupportedShare
        self.contradictedShare = contradictedShare
        self.drift = drift
        self.driftShare = driftShare
        self.contextDependence = contextDependence
        self.meanContextKL = meanContextKL
        self.hallucinationRisk = hallucinationRisk
        self.meanAnswerInfluence = meanAnswerInfluence
        self.meanAnswerRisk = meanAnswerRisk
        self.recordFile = recordFile
    }

    public static func skipped(_ reason: String) -> GroundingFactSummary {
        GroundingFactSummary(measured: false, skippedReason: reason)
    }
}

/// Measures one evaluation generation against its prompted source. Returns nil when there is
/// nothing to say (counted as skipped). Runs on the evaluator's task; MLX work inside it is
/// synchronous, like generation.
public typealias GroundingMeasurer = (CitedGeneration, GroundingEvalSource) async throws -> GroundingFactSummary?

/// Turns the grounding measurement on for `FactEvaluator.run`. The evaluator also needs a
/// `groundingMeasurer`; without one the options are ignored.
public struct GroundingEvalOptions: Sendable, Equatable {
    /// Also measure the paraphrase, fabricated-entity and held-out generations (when the
    /// evaluation runs its controls).
    public var includeControls: Bool
    /// Whether whoever builds the measurer should save each full record beside the generations.
    public var saveRecords: Bool

    public init(includeControls: Bool = true, saveRecords: Bool = false) {
        self.includeControls = includeControls
        self.saveRecords = saveRecords
    }
}

/// The groups the report compares.
public enum GroundingReportGroup: String, Codable, Sendable, CaseIterable {
    case correct, incorrect, paraphrase, fabricated, heldOut

    public var label: String {
        switch self {
        case .correct: return "correct answers"
        case .incorrect: return "wrong answers"
        case .paraphrase: return "paraphrased prompts"
        case .fabricated: return "fabricated entities"
        case .heldOut: return "held-out documents"
        }
    }
}

public struct GroundingGroupMetrics: Codable, Sendable, Equatable {
    public var group: GroundingReportGroup
    /// Generations in the group.
    public var facts: Int
    /// Of those, the measured ones the means are over.
    public var measured: Int
    public var meanGrounding: Float?
    public var meanDrift: Float?
    public var meanDriftShare: Float?
    public var meanContextDependence: Float?
    public var meanHallucinationRisk: Float?
    public var meanAnswerInfluence: Float?

    public init(group: GroundingReportGroup, summaries: [GroundingFactSummary]) {
        let measured = summaries.filter(\.measured)
        func mean(_ values: [Float]) -> Float? { values.isEmpty ? nil : Stats.mean(values) }
        self.group = group
        self.facts = summaries.count
        self.measured = measured.count
        self.meanGrounding = mean(measured.map(\.grounding))
        self.meanDrift = mean(measured.map(\.drift))
        self.meanDriftShare = mean(measured.map(\.driftShare))
        self.meanContextDependence = mean(measured.map(\.contextDependence))
        self.meanHallucinationRisk = mean(measured.map(\.hallucinationRisk))
        self.meanAnswerInfluence = mean(measured.compactMap(\.meanAnswerInfluence))
    }
}

/// One control generation's measurement (the primary facts carry theirs on `FactOutcome`).
public struct GroundingControlOutcome: Codable, Sendable, Equatable {
    public var factID: String
    public var group: GroundingEvalGroup
    public var prompt: String
    public var generated: String
    public var summary: GroundingFactSummary

    public init(factID: String, group: GroundingEvalGroup, prompt: String, generated: String, summary: GroundingFactSummary) {
        self.factID = factID
        self.group = group
        self.prompt = prompt
        self.generated = generated
        self.summary = summary
    }
}

/// The two-fold section of an evaluation report.
public struct GroundingEvalReport: Codable, Sendable, Equatable {
    /// correct / incorrect (primary facts), then paraphrase / fabricated / held-out when measured.
    public var groups: [GroundingGroupMetrics]
    /// AUROC of the hallucination risk for "the answer is wrong", over measured primary facts.
    public var riskAUROCForWrongAnswer: Double?
    /// Pearson r between the mean answer ι and RaoLM's mean answer citation confidence: does
    /// the counterfactual agree with the citation?
    public var answerInfluenceConfidencePearson: Double?
    /// reason → count, over every generation that was not measured.
    public var skipped: [String: Int]
    public var controls: [GroundingControlOutcome]
    /// Every generation, controls included, is measured against the fact's true source partition.
    public var sourceNote: String

    public init(
        groups: [GroundingGroupMetrics], riskAUROCForWrongAnswer: Double?, answerInfluenceConfidencePearson: Double?,
        skipped: [String: Int], controls: [GroundingControlOutcome],
        sourceNote: String = "every generation is measured against the fact's true source partition"
    ) {
        self.groups = groups
        self.riskAUROCForWrongAnswer = riskAUROCForWrongAnswer
        self.answerInfluenceConfidencePearson = answerInfluenceConfidencePearson
        self.skipped = skipped
        self.controls = controls
        self.sourceNote = sourceNote
    }

    public func metrics(_ group: GroundingReportGroup) -> GroundingGroupMetrics? {
        groups.first { $0.group == group }
    }

    /// Aggregate the primary outcomes and the control measurements.
    public static func make(outcomes: [FactOutcome], controls: [GroundingControlOutcome], unmeasured: [String: Int] = [:]) -> GroundingEvalReport {
        var skipped = unmeasured
        let primary = outcomes.compactMap { outcome in outcome.grounding.map { (outcome, $0) } }
        for (_, summary) in primary where !summary.measured {
            skipped[summary.skippedReason ?? "not measured", default: 0] += 1
        }
        for control in controls where !control.summary.measured {
            skipped[control.summary.skippedReason ?? "not measured", default: 0] += 1
        }

        var groups = [
            GroundingGroupMetrics(group: .correct, summaries: primary.filter { $0.0.exact }.map(\.1)),
            GroundingGroupMetrics(group: .incorrect, summaries: primary.filter { !$0.0.exact }.map(\.1)),
        ]
        let byGroup: [(GroundingEvalGroup, GroundingReportGroup)] = [
            (.paraphrase, .paraphrase), (.fabricated, .fabricated), (.heldOut, .heldOut),
        ]
        for (source, target) in byGroup {
            let summaries = controls.filter { $0.group == source }.map(\.summary)
            if !summaries.isEmpty { groups.append(GroundingGroupMetrics(group: target, summaries: summaries)) }
        }

        let measured = primary.filter { $0.1.measured }
        let auroc = Stats.auroc(
            scores: measured.map { Double($0.1.hallucinationRisk) }, labels: measured.map { !$0.0.exact })
        let paired = measured.compactMap { outcome, summary in
            summary.meanAnswerInfluence.map { (Double($0), Double(outcome.meanAnswerConfidence)) }
        }
        let pearson = Stats.pearson(paired.map(\.0), paired.map(\.1))
        return GroundingEvalReport(
            groups: groups, riskAUROCForWrongAnswer: auroc, answerInfluenceConfidencePearson: pearson,
            skipped: skipped, controls: controls)
    }
}
