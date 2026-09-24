//
//  EvalCommand.swift
//  RaoLMCLI
//
//  WHAT: raolm eval — the citation evaluation protocol on a sample of the corpus's facts.
//

import ArgumentParser
import Foundation
import RaoLM

enum EvalPrinter {
    static func print(_ report: EvalReport, sampleRows: Int) {
        Console.section("Fact evaluation (\(report.sampleSize) facts, epoch \(report.epoch))")
        Swift.print(Format.table(
            ["λ", "exact", "citation@1", "on correct", "offset@1", "cited@1", "covered", "confidence"],
            report.lambdas.map { m in
                [Format.f(m.lambda, 2), Format.pct(m.exactAnswer), Format.pct(m.citationAt1Partition), Format.pct(m.citationAt1PartitionOnCorrect),
                 Format.pct(m.citationAt1Offset), Format.pct(m.citedPartitionAt1), Format.pct(m.answerCoveredByVerbatimSpan), Format.f(m.meanAnswerConfidence, 2)]
            }))
        if report.spanChecks > 0 {
            Swift.print("  spans verified against the source: \(report.spansVerified)/\(report.spanChecks) (\(Format.pct(report.spanVerifiedRate)))")
        }
        Swift.print("  calibration: ECE \(Format.f(report.ece, 3))" + (report.auroc.map { String(format: ", AUROC %.3f", $0) } ?? ""))
        let populated = report.calibration.filter { $0.count > 0 }
        if !populated.isEmpty {
            Swift.print(Format.table(["confidence bin", "tokens", "mean conf", "citation correct"], populated.map {
                [String(format: "%.1f–%.1f", $0.lower, $0.upper), String($0.count), Format.f($0.meanConfidence, 2), Format.pct($0.accuracy)]
            }))
        }
        if let c = report.controls {
            Swift.print("  controls: paraphrased prompts exact \(Format.pct(c.paraphraseExact)) (confidence \(Format.f(c.paraphraseMeanConfidence, 2))); "
                + "fabricated entities: confidence \(Format.f(c.negativeMeanConfidence, 2)) vs \(Format.f(c.correctAnswerMeanConfidence, 2)) on correct answers, "
                + "\(c.negativeDistinctiveVerbatimSpans) distinctive verbatim spans")
            if let n = c.heldOutFacts {
                Swift.print("  leave-out control: \(n) facts from documents excluded from training — exact \(Format.pct(c.heldOutExact)), "
                    + "confidence \(Format.f(c.heldOutMeanConfidence, 2)), answer tokens citing the held-out document: \(c.heldOutCitedSource ?? 0)")
            }
        }
        if !report.unalignedFacts.isEmpty {
            Swift.print("  \(report.unalignedFacts.count) facts skipped (answer not on a token boundary)")
        }
        if sampleRows > 0 {
            Swift.print("")
            Swift.print(Format.table(
                ["fact", "prompt", "expected", "generated", "top citation", "verbatim", "verified", "conf"],
                report.outcomes.prefix(sampleRows).map { o in
                    [Format.clip(o.kind.rawValue, 14), Format.clip(o.prompt, 44), Format.clip(o.expected, 16), Format.clip(o.generated, 16),
                     Format.clip((o.topCitationName ?? "—") + (o.topCitation.map { " p\($0.partitionIndex)@\($0.tokenOffset)" } ?? ""), 34),
                     o.answerCoveredByVerbatimSpan ? "yes" : "no",
                     o.spansChecked.map { "\(o.spansVerified ?? 0)/\($0)" } ?? "—", Format.f(o.meanAnswerConfidence, 2)]
                }))
        }
    }
}

struct Eval: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Run the citation evaluation protocol on a sample of the corpus's facts.")

    @OptionGroup var global: GlobalOptions

    @Option(help: "Run directory.")
    var run: String

    @Option(help: "Indexed epoch (default: latest).")
    var epoch: Int?

    @Option(help: "Number of facts to sample.")
    var factsSample = 50

    @Option(help: "Comma-separated λ values to compare.")
    var lambdas = "0,0.25,0.5,0.75"

    @Option(help: "The λ used for verification, calibration and the printed sample.")
    var primaryLambda: Float = 0.5

    @Option(help: "Sampling seed.")
    var seed: UInt64 = 7

    @Option(help: "Thread owner id (default: the run's).")
    var owner: String?

    @OptionGroup var thread: ThreadOptions

    @Flag(help: "Verify against the run's snapshot instead of a live Thread.")
    var offline = false

    @Flag(help: "Skip the paraphrase and fabricated-entity controls.")
    var noControls = false

    @Option(help: "Write the report JSON here (default: <run>/eval.json).")
    var json: String?

    @Option(help: "How many per-fact rows to print.")
    var rows = 12

    @Flag(help: "Use an index whose epoch memorised under half the corpus.")
    var allowWeakIndex = false

    func run() async throws {
        try await guarded {
            try Preflight.requireMetallib()
            let runDirectory = URL(fileURLWithPath: run)
            let context = try await RunContext.load(runDirectory: runDirectory, epoch: epoch, allowWeakIndex: allowWeakIndex)
            guard let factsPath = context.manifest.corpus.factsPath else {
                throw RaoLMFailure("this run has no facts.jsonl recorded; only synthetic corpora can be evaluated", code: 66)
            }
            let facts = try JSONCoding.readLines(Fact.self, from: URL(fileURLWithPath: factsPath))
            let reader: CorpusReading
            if offline {
                reader = InMemoryCorpusReader(snapshot: try context.snapshot())
            } else {
                let endpoint = ThreadHostRecord.load(dataDirectory: global.root.threadDB)?.endpoint ?? thread.endpoint()
                let client = ThreadCorpusClient(endpoint: endpoint)
                _ = try await client.health()
                reader = ThreadCorpusReader(client: client, owner: owner ?? context.manifest.corpus.owner)
            }
            let values = lambdas.split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
            let evaluator = FactEvaluator(context: context, corpus: try context.tokenizedCorpus(), facts: facts, reader: reader)
            let report = try await evaluator.run(
                options: EvalOptions(factsSample: factsSample, lambdas: values, primaryLambda: primaryLambda, seed: seed,
                                     includeControls: !noControls, saveGenerations: RunLayout.generations(runDirectory))
            ) { print("  \($0)") }
            EvalPrinter.print(report, sampleRows: rows)
            let output = json.map { URL(fileURLWithPath: $0) } ?? runDirectory.appendingPathComponent("eval.json")
            try JSONCoding.write(report, to: output)
            print("\nreport written to \(output.path)")
        }
    }
}
