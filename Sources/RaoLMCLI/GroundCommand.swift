//
//  GroundCommand.swift
//  RaoLMCLI
//
//  WHAT: raolm ground — the two-fold entropy harness on a saved generation. RaoLM's citations
//        say which corpus positions support each token; SinatraHarness's grounding measurement
//        says whether the sources actually moved the model, by scoring the same output with
//        the sources in front of the prompt and without them. Also the printers and helpers
//        `generate --ground`, `eval --grounding` and the demo share.
//  PIN:  The bare side is the generation's own prompt, so its log p and entropy reproduce the
//        trace's p_LM and H_lm (the "bare consistency" line). A memorised fact needs no source
//        (ι ≈ 0): that is the finding the citation confidence cannot make, not an error.
//

import ArgumentParser
import Foundation
import RaoLM
import RaoLMWorkflows
import SinatraHarness

struct Ground: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ground",
        abstract: "Measure how much a generation's sources moved the model (with vs without them), beside its citations.",
        discussion: """
            The run's own model teacher-forces the saved output twice: under [eos] D… [eos] prompt, with the chosen source partitions D in front, and under the bare prompt the generation came from. Per token: ι = log p_ctx − log p_bare (nats the sources added), KL(p_ctx ‖ p_bare), drift, class (● grounded, ○ unsupported, ✗ contradicted, · function) and hallucination risk; per source: A_p, the nats of the output it earned, beside RaoLM's citation weight and confidence.

            Sources: top (the top-cited partitions), spans (the verbatim spans' partitions), all (everything retrieval touched), fact (the partition the prompt was sliced from), or DOC:P[,DOC:P].
            """)

    @OptionGroup var global: GlobalOptions

    @Option(help: "Run directory.")
    var run: String

    @Option(help: "A saved generation (raolm generate --json FILE, or <run>/generations/*.json).")
    var generation: String

    @Option(help: "Sources in front of the prompt: \(GroundingSourcePolicy.help).")
    var sources = "top"

    @Option(help: "Per-step detail: summary | full (adds the bare rank of each token).")
    var detail = "summary"

    @Option(help: "Seconds the measurement may take.")
    var budget: Double = 60

    @Option(help: "Indexed epoch (default: the generation's own).")
    var epoch: Int?

    @Flag(help: "Use an index whose epoch memorised under half the corpus.")
    var allowWeakIndex = false

    @Option(help: "Write the record here (default: <run>/generations/<generation id>.grounding.json).")
    var out: String?

    @Flag(help: "Print the record as JSON instead of tables.")
    var json = false

    @Option(help: "How many token rows to print.")
    var rows = 64

    func validate() throws {
        _ = try GroundCommandSupport.policy(sources)
        _ = try GroundCommandSupport.detail(detail)
        guard budget > 0 else { throw ValidationError("--budget must be positive") }
    }

    func run() async throws {
        try await guarded {
            let policy = try GroundCommandSupport.policy(sources)
            let level = try GroundCommandSupport.detail(detail)
            try Preflight.requireMetallib()
            let saved = try GroundCommandSupport.loadGeneration(generation)
            let context = try await RunContext.load(
                runDirectory: URL(fileURLWithPath: run), epoch: epoch ?? saved.manifest.epoch, allowWeakIndex: allowWeakIndex)
            try GroundCommandSupport.requireSameCheckpoint(saved, context: context)
            let output = out.map { URL(fileURLWithPath: $0) }
                ?? GroundingRecord.url(runDirectory: context.runDirectory, generationID: saved.generationID)
            let record = try await GroundCommandSupport.measure(
                context: context, generation: saved, policy: policy, detail: level, budget: budget, save: output)
            if json {
                let data = try JSONCoding.prettyEncoder().encode(record)
                print(String(decoding: data, as: UTF8.self))
            } else {
                GroundingPrinter.printRecord(record, rows: rows)
                print("\nrecord written to \(output.path)")
            }
        }
    }
}

// MARK: - Shared with generate --ground, eval --grounding and the demo

enum GroundCommandSupport {
    static func policy(_ text: String) throws -> GroundingSourcePolicy {
        do {
            return try GroundingSourcePolicy.parse(text)
        } catch let error as GroundingError {
            throw ValidationError(error.description)
        }
    }

    static func detail(_ text: String) throws -> GroundingDetail {
        do {
            return try GroundingDetail.parse(text)
        } catch let error as GroundingError {
            throw ValidationError(error.description)
        }
    }

    static func loadGeneration(_ path: String) throws -> CitedGeneration {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw RaoLMFailure("no generation at \(url.path)", hint: "save one with: raolm generate --run <run> --json <file>", code: 66)
        }
        do {
            return try CitedGeneration.load(from: url)
        } catch {
            throw RaoLMFailure("\(url.path) is not a generation record: \(error)", code: 65)
        }
    }

    /// The generation must come from the loaded checkpoint (exit 65 otherwise).
    static func requireSameCheckpoint(_ generation: CitedGeneration, context: RunContext) throws {
        guard generation.manifest.checkpointSHA256 == context.checkpointSHA256 else {
            throw RaoLMFailure(
                "generation \(generation.generationID) was made by run \(generation.manifest.runID) epoch \(generation.manifest.epoch) (checkpoint \(Format.short(generation.manifest.checkpointSHA256))); the loaded run \(context.manifest.runID) epoch \(context.epoch) has \(Format.short(context.checkpointSHA256))",
                hint: "measure it against the run and --epoch that generated it", code: 65)
        }
    }

    /// Where `generate --ground` saves: beside the --json file, else under <run>/generations/.
    static func recordURL(runDirectory: URL, generationID: String, besideJSON json: String? = nil) -> URL {
        if let json { return GroundingRecord.url(besideGeneration: URL(fileURLWithPath: json)) }
        return GroundingRecord.url(runDirectory: runDirectory, generationID: generationID)
    }

    /// Measure one generation over the run's own corpus and save the record to `save`.
    /// `requireMeasured` turns a spent budget into `GroundingError.notMeasured` (exit 75).
    static func measure(
        context: RunContext, generation: CitedGeneration, policy: GroundingSourcePolicy, detail: GroundingDetail = .summary,
        budget: TimeInterval = 60, save: URL?, requireMeasured: Bool = true
    ) async throws -> GroundingRecord {
        let grounder = RaoGrounder(context: context, corpus: try context.tokenizedCorpus(), budget: budget)
        return try await measure(
            grounder: grounder, generation: generation, policy: policy, detail: detail, save: save, requireMeasured: requireMeasured)
    }

    /// The same with a grounder already built (the evaluator's corpus, a reused session).
    static func measure(
        grounder: RaoGrounder, generation: CitedGeneration, policy: GroundingSourcePolicy, detail: GroundingDetail = .summary,
        budget: TimeInterval? = nil, save: URL?, requireMeasured: Bool = true
    ) async throws -> GroundingRecord {
        let record = try await grounder.measure(
            generation: generation, policy: policy, detail: detail, budget: budget, requireMeasured: requireMeasured)
        if let save { try record.save(to: save) }
        return record
    }

    /// `FactEvaluator.groundingMeasurer` for `eval --grounding` and the demo: every generation is
    /// measured against its fact's true source partition; records are saved under
    /// <run>/generations/ when asked.
    static func makeMeasurer(grounder: RaoGrounder, runDirectory: URL?, threadID: String?, saveRecords: Bool) -> GroundingMeasurer {
        grounder.makeMeasurer(runDirectory: runDirectory, threadID: threadID, saveRecords: saveRecords)
    }
}

// MARK: - Printing

enum GroundingPrinter {
    static let evalSectionTitle = GroundingTables.evalSectionTitle

    static func glyph(_ kind: GroundingClass) -> String { GroundingTables.glyph(kind) }
    static func signed(_ value: Float?, _ digits: Int = 2) -> String { GroundingTables.signed(value, digits) }

    /// Summary, attribution, tokens and the join, in that order.
    static func printRecord(_ record: GroundingRecord, rows: Int = 64) {
        printSummary(record)
        guard record.measured else { return }
        printAttribution(record)
        printTokens(record, limit: rows)
        printStats(record)
    }

    static func printSummary(_ record: GroundingRecord) {
        Console.section("Grounding: with vs without the sources (\(record.policy))")
        for line in summaryLines(record) { Swift.print(line) }
    }

    static func summaryLines(_ record: GroundingRecord) -> [String] { GroundingTables.summaryLines(record).map { "  " + $0 } }

    static func printAttribution(_ record: GroundingRecord) {
        Console.section("Attribution: RaoLM's citations beside Sinatra's A_p")
        Swift.print(Format.table(GroundingTables.attribution(record)))
    }

    static func printTokens(_ record: GroundingRecord, limit: Int = 64) {
        Console.section("Per token: ● grounded  ○ unsupported  ✗ contradicted  · function")
        Swift.print(Format.table(GroundingTables.tokens(record, limit: limit)))
        if record.tokens.count > limit { Swift.print("  … \(record.tokens.count - limit) more tokens") }
    }

    static func printStats(_ record: GroundingRecord) {
        Console.section("The join: does citation track influence?")
        for line in statsLines(record) { Swift.print(line) }
    }

    static func statsLines(_ record: GroundingRecord) -> [String] { GroundingTables.statsLines(record).map { "  " + $0 } }

    // MARK: Evaluation

    /// Column headers and cells the per-fact table gains.
    static let outcomeHeaders = GroundingTables.outcomeHeaders

    static func outcomeCells(_ summary: GroundingFactSummary?) -> [String] { GroundingTables.outcomeCells(summary) }

    /// The body of the eval printer's "Grounding (two-fold)" section (title: `evalSectionTitle`).
    static func evalSection(_ report: GroundingEvalReport) -> [String] {
        Format.table(GroundingTables.evalGroups(report)).components(separatedBy: "\n")
            + GroundingTables.evalLines(report).map { "  " + $0 }
    }

    static func printEvalSection(_ report: GroundingEvalReport) {
        Console.section(evalSectionTitle)
        for line in evalSection(report) { Swift.print(line) }
    }
}
