//
//  Tables.swift
//  RaoLMWorkflows
//
//  WHAT: The rows behind every table the CLI prints, as data: the plain output renders them
//        with `Format.table`, the studio with its table widget, so both show the same columns.
//

import Foundation
import RaoLM

public enum GenerationTables {
    /// `#, token, H_lm, H_mix, agree, conf, top citation` for generated tokens.
    public static func traces(_ generation: CitedGeneration, limit: Int = 64) -> TextTable {
        TextTable(
            headers: ["#", "token", "H_lm", "H_mix", "agree", "conf", "top citation"],
            rows: generation.traces.filter { !$0.isPrompt }.prefix(limit).map { trace in
                [
                    String(trace.index), Format.clip(trace.text.debugDescription, 14), Format.f(trace.lmEntropy, 2),
                    Format.f(trace.mixedEntropy, 2), Format.f(trace.agreement, 2), Format.f(trace.confidence, 2),
                    topCitation(trace, in: generation),
                ]
            })
    }

    public static func topCitation(_ trace: TokenTrace, in generation: CitedGeneration) -> String {
        trace.citations.first.map { c in
            "\(generation.partition(row: c.row)?.documentName ?? "?") p\(c.address.partitionIndex)@\(c.address.tokenOffset)"
        } ?? (trace.uncited ? "(uncited)" : "")
    }

    /// `[[n]] name — id pP tokens a..<b · status`.
    public static func sourceLines(_ generation: CitedGeneration) -> [String] {
        CitationMarkers.sources(generation).map { source in
            let status = source.verification.map { " · \($0.rawValue)" } ?? ""
            return "[[\(source.number)]] \(source.documentName) — \(source.documentID) p\(source.partitionIndex) tokens \(source.tokenStart)..<\(source.tokenEnd)\(status)"
        }
    }

    public static func summaryLine(_ generation: CitedGeneration) -> String {
        let s = generation.summary
        return "\(s.generated) generated tokens: \(s.verbatimCovered) in verbatim spans, \(s.supportOnly) with support citations, \(s.uncited) uncited; mean confidence \(Format.f(s.meanConfidence, 2))"
    }

    public static func bindingLine(_ generation: CitedGeneration) -> String {
        let m = generation.manifest
        return "bound to run \(m.runID) epoch \(m.epoch), checkpoint \(Format.short(m.checkpointSHA256)), index \(Format.short(m.indexSHA256)), corpus \(Format.short(m.corpusHash))"
    }
}

public enum EvalTables {
    public static func lambdas(_ report: EvalReport) -> TextTable {
        TextTable(
            headers: ["λ", "exact", "citation@1", "on correct", "offset@1", "cited@1", "covered", "confidence"],
            rows: report.lambdas.map { m in
                [Format.f(m.lambda, 2), Format.pct(m.exactAnswer), Format.pct(m.citationAt1Partition), Format.pct(m.citationAt1PartitionOnCorrect),
                 Format.pct(m.citationAt1Offset), Format.pct(m.citedPartitionAt1), Format.pct(m.answerCoveredByVerbatimSpan), Format.f(m.meanAnswerConfidence, 2)]
            })
    }

    public static func calibration(_ report: EvalReport) -> TextTable {
        TextTable(
            headers: ["confidence bin", "tokens", "mean conf", "citation correct"],
            rows: report.calibration.filter { $0.count > 0 }.map {
                [String(format: "%.1f–%.1f", $0.lower, $0.upper), String($0.count), Format.f($0.meanConfidence, 2), Format.pct($0.accuracy)]
            })
    }

    /// With `grounding`, each row also carries the answer's mean ι and the hallucination risk.
    public static func outcomes(_ report: EvalReport, limit: Int, grounding: Bool = false) -> TextTable {
        TextTable(
            headers: ["fact", "prompt", "expected", "generated", "top citation", "verbatim", "verified", "conf"]
                + (grounding ? GroundingTables.outcomeHeaders : []),
            rows: report.outcomes.prefix(limit).map { o in
                [Format.clip(o.kind.rawValue, 14), Format.clip(o.prompt, 44), Format.clip(o.expected, 16), Format.clip(o.generated, 16),
                 Format.clip((o.topCitationName ?? "—") + (o.topCitation.map { " p\($0.partitionIndex)@\($0.tokenOffset)" } ?? ""), 34),
                 o.answerCoveredByVerbatimSpan ? "yes" : "no",
                 o.spansChecked.map { "\(o.spansVerified ?? 0)/\($0)" } ?? "—", Format.f(o.meanAnswerConfidence, 2)]
                    + (grounding ? GroundingTables.outcomeCells(o.grounding) : [])
            })
    }

    public static func spansLine(_ report: EvalReport) -> String? {
        guard report.spanChecks > 0 else { return nil }
        return "spans verified against the source: \(report.spansVerified)/\(report.spanChecks) (\(Format.pct(report.spanVerifiedRate)))"
    }

    public static func calibrationLine(_ report: EvalReport) -> String {
        "calibration: ECE \(Format.f(report.ece, 3))" + (report.auroc.map { String(format: ", AUROC %.3f", $0) } ?? "")
    }

    /// The controls sentences, exactly as `raolm eval` prints them (without indentation).
    public static func controlLines(_ report: EvalReport) -> [String] {
        guard let c = report.controls else { return [] }
        var lines = [
            "controls: paraphrased prompts exact \(Format.pct(c.paraphraseExact)) (confidence \(Format.f(c.paraphraseMeanConfidence, 2))); "
                + "fabricated entities: confidence \(Format.f(c.negativeMeanConfidence, 2)) vs \(Format.f(c.correctAnswerMeanConfidence, 2)) on correct answers, "
                + "\(c.negativeDistinctiveVerbatimSpans) distinctive verbatim spans",
        ]
        if let n = c.heldOutFacts {
            lines.append("leave-out control: \(n) facts from documents excluded from training — exact \(Format.pct(c.heldOutExact)), "
                + "confidence \(Format.f(c.heldOutMeanConfidence, 2)), answer tokens citing the held-out document: \(c.heldOutCitedSource ?? 0)")
        }
        return lines
    }
}

public enum LedgerTables {
    public static func epochs(_ rows: [EpochRow]) -> TextTable {
        TextTable(
            headers: ["epoch", "steps", "train loss", "train H", "eval loss", "eval H", "memorised", "gap", "checkpoint", "index"],
            rows: rows.map {
                [String($0.epoch), String($0.steps), Format.f($0.trainLoss), Format.f($0.trainEntropy), Format.f($0.evalLoss), Format.f($0.evalEntropy),
                 Format.pct($0.evalMemorisedFraction), Format.f($0.calibrationGap), Format.short($0.checkpointSHA256), $0.indexSHA256 == nil ? "" : Format.short($0.indexSHA256)]
            })
    }

    /// The eval epochs of a finished run, as the demo prints them.
    public static func evalEpochs(_ manifest: RunManifest) -> TextTable {
        TextTable(
            headers: ["epoch", "train loss", "train H", "eval loss", "eval H", "memorised", "checkpoint"],
            rows: manifest.epochs.filter { $0.evalLoss != nil }.map {
                [String($0.epoch), Format.f($0.trainLoss), Format.f($0.trainEntropy), Format.f($0.evalLoss), Format.f($0.evalEntropy), Format.pct($0.evalMemorisedFraction), Format.short($0.checkpointSHA256)]
            })
    }

    public static func partitionTrajectory(_ rows: [PartitionEpochRow]) -> TextTable {
        TextTable(
            headers: ["epoch", "partition", "tokens", "train loss", "train H", "eval loss", "eval H", "memorised", "since"],
            rows: rows.map {
                [String($0.epoch), String($0.partitionIndex), String($0.tokens), Format.f($0.train?.meanLoss), Format.f($0.train?.meanEntropy),
                 Format.f($0.eval?.meanLoss), Format.f($0.eval?.meanEntropy), Format.pct($0.eval?.memorisedFraction), $0.memorisedAtEpoch.map(String.init) ?? "—"]
            })
    }

    public static func factLosses(_ rows: [FactEpochRow]) -> TextTable {
        TextTable(
            headers: ["epoch", "mean loss", "memorised", "answer token losses"],
            rows: rows.map {
                [String($0.epoch), Format.f($0.meanLoss), $0.memorised ? "yes" : "no", $0.answerTokenLosses.map { Format.f($0, 2) }.joined(separator: " ")]
            })
    }

    public static func header(_ manifest: RunManifest) -> String {
        "run \(manifest.runID) (\(manifest.preset), \(Format.count(manifest.parameterCount)) parameters) on corpus \(Format.short(manifest.corpus.corpusHash))"
    }
}
