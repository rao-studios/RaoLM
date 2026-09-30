//
//  GroundingTables.swift
//  RaoLMWorkflows
//
//  WHAT: The two-fold harness as lines and tables: Sinatra's with-vs-without-source summary,
//        RaoLM's citations beside Sinatra's attribution A_p, the per-token join, the join's
//        statistics, and the evaluation's grounding groups. `raolm ground`, `raolm eval
//        --grounding` and the studio all print from here.
//

import Foundation
import RaoLM
import SinatraHarness

public enum GroundingTables {
    public static let evalSectionTitle = "Grounding (two-fold)"

    public static func glyph(_ kind: GroundingClass) -> String {
        switch kind {
        case .grounded: return "●"
        case .unsupported: return "○"
        case .contradicted: return "✗"
        case .function: return "·"
        }
    }

    public static func signed(_ value: Float?, _ digits: Int = 2) -> String {
        guard let value, value.isFinite else { return "—" }
        return String(format: "%+.\(digits)f", value)
    }

    public static func scientific(_ value: Float) -> String {
        value.isFinite ? String(format: "%.1e", value) : "—"
    }

    public static func r(_ value: Double?) -> String {
        value.map { String(format: "%.3f", $0) } ?? "—"
    }

    public static func summaryLines(_ record: GroundingRecord) -> [String] {
        let m = record.measurement
        var lines: [String] = []
        let names = record.sources.prefix(4).map { "\($0.documentName) p\($0.partitionIndex)" }.joined(separator: ", ")
            + (record.sources.count > 4 ? ", … \(record.sources.count - 4) more" : "")
        lines.append("\(record.generationID) · run \(record.manifest.runID) epoch \(record.manifest.epoch) · \(record.sources.count) source\(record.sources.count == 1 ? "" : "s"): \(names)")
        guard m.measured else {
            lines.append("not measured: \(m.skippedReason ?? "unknown reason")")
            return lines
        }
        let s = m.summary
        lines.append("\(s.steps) tokens, \(s.contentTokens) content · grounded \(Format.pct(s.grounding)) · unsupported \(Format.pct(s.unsupportedShare)) · contradicted \(Format.pct(s.contradictedShare))")
        lines.append("drift \(Format.f(s.drift, 2)) nats per content token, \(Format.pct(s.driftShare)) drifting" + (s.firstDriftStep.map { " (first at step \($0))" } ?? ""))
        lines.append("context dependence Σι \(signed(s.contextDependence)) nats · mean KL(ctx‖bare) \(Format.f(s.meanContextKL, 3)) · H \(Format.f(s.meanEntropyBare, 2)) bare → \(Format.f(s.meanEntropyCtx, 2)) ctx")
        lines.append("hallucination risk \(Format.f(s.hallucinationRisk, 3)) · parroted \(Format.pct(s.parrotShare)) · unattributed influence \(Format.f(s.unattributed, 2)) nats")
        lines.append("tokens: prompt \(record.promptTokens), context \(record.contextTokens), bare \(m.bareTokens), shared prefix \(m.sharedPrefixTokens), sampled \(record.sampledTokens)\(record.includesEOS ? " (incl. eos)" : "") · prefill \(Format.f(Float(m.prefillMillis), 0)) ms, score \(Format.f(Float(m.scoreMillis), 0)) ms")
        if let bare = record.stats.bareConsistency {
            lines.append("bare consistency: max |Δlog p| \(scientific(bare.maxLogpGap)), max |ΔH| \(scientific(bare.maxEntropyGap)) against the traces' p_LM and H_lm (should be ~0)")
        }
        if record.contextExceedsTrainedLength {
            lines.append("note: the context side runs past the sequence length the model trained on")
        }
        if !record.droppedSources.isEmpty {
            lines.append("note: \(record.droppedSources.count) least-cited source\(record.droppedSources.count == 1 ? "" : "s") dropped to fit the model's positions")
        }
        return lines
    }

    public static func attribution(_ record: GroundingRecord) -> TextTable {
        TextTable(
            headers: ["source", "p", "cite w", "conf", "top-cited", "verbatim", "A_p nats", "uptake", "missed", "intent", "cover", "parrot"],
            rows: record.partitions.map { p in
                [Format.clip(p.documentName, 28), String(p.partitionIndex), Format.f(p.citationWeight, 2),
                 Format.f(p.meanCitationConfidence, 2), String(p.topCitedTokens), String(p.verbatimSpanTokens),
                 Format.f(p.nats, 2), Format.pct(p.uptake), Format.f(p.missed, 2), Format.pct(p.intent),
                 Format.pct(p.coverage), Format.pct(p.parrot)]
            })
    }

    public static func tokens(_ record: GroundingRecord, limit: Int = 64) -> TextTable {
        TextTable(
            headers: ["#", "token", "ι", "KL", "drift", "cls", "risk", "H_ctx", "H_bare", "tune", "conf", "agree"],
            rows: record.tokens.prefix(max(0, limit)).map { row in
                let tune = row.drift > 0.05 ? "\(Format.clip((row.tuneText ?? "#\(row.tune)").debugDescription, 12)) \(signed(row.tuneNats))" : ""
                return [
                    String(row.index), Format.clip((row.isEOS ? "<eos>" : row.text).debugDescription, 14), signed(row.influence),
                    Format.f(row.contextKL, 3), Format.f(row.drift, 2), glyph(row.kind), Format.f(row.risk, 2),
                    Format.f(row.entropyCtx, 2), Format.f(row.entropyBare, 2), tune, Format.f(row.confidence, 2),
                    Format.f(row.agreement, 2),
                ]
            })
    }

    public static func statsLines(_ record: GroundingRecord) -> [String] {
        let s = record.stats
        var lines = [
            "r(citation confidence, ι) \(r(s.confidenceInfluencePearson)) · r(kNN agreement, ι) \(r(s.agreementInfluencePearson)) over \(s.joinedTokens) tokens",
            "mean ι: cited \(signed(s.meanInfluenceCited, 3)), uncited \(signed(s.meanInfluenceUncited, 3)) · risk AUROC for uncited tokens \(r(s.riskAUROCForUncited))",
            "verbatim-span content tokens grounded: \(Format.pct(s.verbatimSpanGroundedShare)) · top-citation weight outside the sources: \(Format.pct(s.citedOutsideSourcesWeight))",
        ]
        if let bare = s.bareConsistency {
            lines.append("bare consistency over \(bare.tokens) tokens: |Δlog p| max \(scientific(bare.maxLogpGap)) mean \(scientific(bare.meanLogpGap)); |ΔH| max \(scientific(bare.maxEntropyGap)) mean \(scientific(bare.meanEntropyGap))")
        }
        return lines
    }

    public static let outcomeHeaders = ["ι", "risk"]

    public static func outcomeCells(_ summary: GroundingFactSummary?) -> [String] {
        guard let summary else { return ["—", "—"] }
        guard summary.measured else { return ["skip", "—"] }
        return [signed(summary.meanAnswerInfluence), Format.f(summary.hallucinationRisk, 2)]
    }

    public static func evalGroups(_ report: GroundingEvalReport) -> TextTable {
        TextTable(
            headers: ["group", "n", "measured", "grounded", "drift", "drifting", "Σι", "risk", "answer ι"],
            rows: report.groups.map { g in
                [g.group.label, String(g.facts), String(g.measured), Format.pct(g.meanGrounding), Format.f(g.meanDrift, 2),
                 Format.pct(g.meanDriftShare), signed(g.meanContextDependence), Format.f(g.meanHallucinationRisk, 3),
                 signed(g.meanAnswerInfluence, 3)]
            })
    }

    public static func evalLines(_ report: GroundingEvalReport) -> [String] {
        var lines = [
            "risk AUROC for a wrong answer: \(r(report.riskAUROCForWrongAnswer)) · r(answer ι, citation confidence): \(r(report.answerInfluenceConfidencePearson))",
            "\(report.sourceNote); a memorised answer needs no source, so ι ≈ 0 there is a finding",
        ]
        if !report.skipped.isEmpty {
            lines.append("not measured: " + report.skipped.sorted { $0.value > $1.value }.map { "\($0.value)× \($0.key)" }.joined(separator: ", "))
        }
        return lines
    }
}
