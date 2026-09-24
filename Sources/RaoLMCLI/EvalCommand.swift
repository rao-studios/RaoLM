//
//  EvalCommand.swift
//  RaoLMCLI
//
//  WHAT: raolm eval — the citation evaluation protocol on a sample of the corpus's facts.
//

import ArgumentParser
import Foundation
import RaoLM
import RaoLMWorkflows

enum EvalPrinter {
    static func print(_ report: EvalReport, sampleRows: Int) {
        Console.section("Fact evaluation (\(report.sampleSize) facts, epoch \(report.epoch))")
        Swift.print(Format.table(EvalTables.lambdas(report)))
        if let line = EvalTables.spansLine(report) { Swift.print("  " + line) }
        Swift.print("  " + EvalTables.calibrationLine(report))
        let calibration = EvalTables.calibration(report)
        if !calibration.rows.isEmpty { Swift.print(Format.table(calibration)) }
        for line in EvalTables.controlLines(report) { Swift.print("  " + line) }
        if let grounding = report.grounding { GroundingPrinter.printEvalSection(grounding) }
        if !report.unalignedFacts.isEmpty {
            Swift.print("  \(report.unalignedFacts.count) facts skipped (answer not on a token boundary)")
        }
        if sampleRows > 0 {
            Swift.print("")
            Swift.print(Format.table(EvalTables.outcomes(report, limit: sampleRows, grounding: report.grounding != nil)))
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

    @Flag(help: "Also measure each answer against its source with SinatraHarness (with vs without the source): hallucination risk, drift and influence beside the citation metrics.")
    var grounding = false

    @Option(help: "Seconds each grounding measurement may take.")
    var groundingBudget: Double = 120

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
                let endpoint = ThreadResolve.endpoint(root: global.root, fallback: thread.endpoint())
                let client = ThreadCorpusClient(endpoint: endpoint)
                _ = try await client.health()
                reader = ThreadCorpusReader(client: client, owner: owner ?? context.manifest.corpus.owner)
            }
            let values = lambdas.split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
            let corpus = try context.tokenizedCorpus()
            let evaluator = FactEvaluator(context: context, corpus: corpus, facts: facts, reader: reader)
            var groundingOptions: GroundingEvalOptions?
            if grounding {
                let grounder = RaoGrounder(context: context, corpus: corpus, budget: groundingBudget)
                evaluator.groundingMeasurer = GroundCommandSupport.makeMeasurer(
                    grounder: grounder, runDirectory: runDirectory, threadID: context.manifestRef.threadID, saveRecords: true)
                groundingOptions = GroundingEvalOptions(includeControls: !noControls, saveRecords: true)
            }
            let report = try await evaluator.run(
                options: EvalOptions(factsSample: factsSample, lambdas: values, primaryLambda: primaryLambda, seed: seed,
                                     includeControls: !noControls, saveGenerations: RunLayout.generations(runDirectory),
                                     grounding: groundingOptions)
            ) { print("  \($0)") }
            EvalPrinter.print(report, sampleRows: rows)
            let output = json.map { URL(fileURLWithPath: $0) } ?? runDirectory.appendingPathComponent("eval.json")
            try JSONCoding.write(report, to: output)
            print("\nreport written to \(output.path)")
        }
    }
}
