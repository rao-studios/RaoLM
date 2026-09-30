//
//  GenerateCommands.swift
//  RaoLMCLI
//
//  WHAT: raolm generate | verify | ledger — cited generation, verification against the live
//        Thread, and the entropy ledger.
//

import ArgumentParser
import Foundation
import RaoLM
import RaoLMWorkflows

struct GenerateOptions: ParsableArguments {
    @Option(help: "Indexed epoch to use (default: the latest).")
    var epoch: Int?

    @Option(help: "Weight of the retrieval distribution in the emitted logits (0 = evidence only).")
    var lambda: Float = 0.5

    @Option(help: "Retrieval temperature over cosine scores.")
    var tau: Float?

    @Option(help: "Neighbours per position.")
    var k: Int?

    @Option(help: "Sampling temperature (0 = greedy).")
    var temperature: Float = 0

    @Option(help: "Sample from the top-k of the mixed distribution (0 = all).")
    var topK = 0

    @Option(help: "Maximum tokens to generate.")
    var maxTokens = 48

    @Option(help: "Sampling seed.")
    var seed: UInt64 = 42

    @Flag(help: "Use an index whose epoch memorised under half the corpus.")
    var allowWeakIndex = false

    func parameters(_ context: RunContext) -> GenerationParameters {
        var params = context.defaultParameters()
        params.lambda = lambda
        if let tau { params.tau = tau }
        if let k { params.k = k }
        params.temperature = temperature
        params.topK = topK
        params.maxTokens = maxTokens
        params.seed = seed
        return params
    }
}

enum GenerationPrinter {
    static func print(_ generation: CitedGeneration, format: OutputFormat, tokenizer: RaoTokenizer) throws {
        switch format {
        case .json:
            let data = try JSONCoding.prettyEncoder().encode(generation)
            Swift.print(String(decoding: data, as: UTF8.self))
        case .plain:
            Swift.print(generation.prompt.text + generation.text)
        case .markers:
            let rendered = CitationMarkers.render(generation)
            Swift.print(generation.prompt.text + "▸" + rendered.text)
            Swift.print("")
            for line in GenerationTables.sourceLines(generation) { Swift.print("  " + line) }
            Swift.print("")
            Swift.print("  " + GenerationTables.summaryLine(generation))
            Swift.print("  " + GenerationTables.bindingLine(generation))
        }
    }

    static func printTraces(_ generation: CitedGeneration, limit: Int = 64) {
        Swift.print("")
        Swift.print(Format.table(GenerationTables.traces(generation, limit: limit)))
    }
}

struct GenerateText: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "generate",
        abstract: "Generate with citations from a trained run.")

    @OptionGroup var global: GlobalOptions

    @Option(help: "Run directory.")
    var run: String

    @Option(help: "Prompt text.")
    var prompt: String?

    @Option(help: "Prompt as a corpus token slice: DOCUMENT_ID:PARTITION:OFFSET:LENGTH.")
    var promptFrom: String?

    @OptionGroup var options: GenerateOptions

    @Option(help: "Output format.")
    var format: OutputFormat = .markers

    @Option(help: "Also write the generation record to this JSON file.")
    var json: String?

    @Flag(help: "Print the per-token trace table.")
    var trace = false

    @Flag(help: "Measure the generation against its sources with SinatraHarness (with vs without them) and save the record.")
    var ground = false

    @Option(help: "With --ground: top, spans, all, fact (the prompt's source), or DOC:P[,DOC:P].")
    var sources = "top"

    func run() async throws {
        try await guarded {
            try Preflight.requireMetallib()
            let context = try await RunContext.load(
                runDirectory: URL(fileURLWithPath: run), epoch: options.epoch, allowWeakIndex: options.allowWeakIndex)
            let request: GenerationRequest
            if let promptFrom {
                request = try Self.sliceRequest(promptFrom, context: context, params: options.parameters(context))
            } else if let prompt, !prompt.isEmpty {
                let tokens = context.tokenizer.encode(prompt)
                request = GenerationRequest(promptTokens: tokens, promptText: prompt, params: options.parameters(context))
            } else {
                throw ValidationError("pass --prompt or --prompt-from")
            }
            let policy = ground ? try GroundCommandSupport.policy(sources) : nil
            let generation = try context.generator().generate(request)
            try GenerationPrinter.print(generation, format: format, tokenizer: context.tokenizer)
            if trace { GenerationPrinter.printTraces(generation) }
            if let json {
                try generation.save(to: URL(fileURLWithPath: json))
                if format != .json { print("\nsaved \(json)") }
            }
            if let policy {
                let url = GroundCommandSupport.recordURL(runDirectory: context.runDirectory, generationID: generation.generationID, besideJSON: json)
                let record = try await GroundCommandSupport.measure(context: context, generation: generation, policy: policy, save: url)
                GroundingPrinter.printSummary(record)
                if record.measured { GroundingPrinter.printAttribution(record) }
                if trace, record.measured { GroundingPrinter.printTokens(record) }
                if record.measured { GroundingPrinter.printStats(record) }
                print("\ngrounding record: \(url.path)")
            }
        }
    }

    static func sliceRequest(_ spec: String, context: RunContext, params: GenerationParameters) throws -> GenerationRequest {
        let slice: CorpusSlice
        do {
            slice = try CorpusSlice.parse(spec)
        } catch let failure as RaoLMFailure where failure.code == 64 {
            throw ValidationError(failure.message)
        }
        return try slice.request(context: context, params: params)
    }
}

struct Verify: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Re-check a generation's verbatim spans against the live Thread.")

    @OptionGroup var global: GlobalOptions

    @Option(help: "Run directory.")
    var run: String

    @Option(help: "Generation JSON written by raolm generate --json.")
    var generation: String

    @Option(help: "Thread owner id (default: the run's).")
    var owner: String?

    @OptionGroup var thread: ThreadOptions

    @Flag(help: "Verify against the run's snapshot instead of a Thread.")
    var offline = false

    func run() async throws {
        try await guarded {
            let manifest = try RunManifest.load(URL(fileURLWithPath: run))
            let tokenizer = try await RaoTokenizer.load()
            var record = try CitedGeneration.load(from: URL(fileURLWithPath: generation))
            let reader: CorpusReading
            if offline {
                reader = InMemoryCorpusReader(snapshot: try CorpusSnapshot.load(from: URL(fileURLWithPath: manifest.corpus.snapshotPath)))
            } else {
                let endpoint = ThreadResolve.endpoint(root: global.root, fallback: thread.endpoint())
                reader = ThreadCorpusReader(client: ThreadCorpusClient(endpoint: endpoint), owner: owner ?? manifest.corpus.owner)
            }
            let report = try await CitationVerifier.verify(&record, reader: reader, tokenizer: tokenizer)
            try record.save(to: URL(fileURLWithPath: generation))
            for check in report.checks {
                print("\(check.status == .verified ? "✓" : "✗") span \(check.span) (\(check.kind.rawValue), \(check.length) tokens) \(check.documentID) p\(check.partitionIndex)@\(check.tokenOffset): \(check.status.rawValue)\(check.detail.map { " — \($0)" } ?? "")")
                print("    \(Format.clip(check.text.debugDescription, 100))")
            }
            print("\(report.verified)/\(report.checks.count) spans verified against \(offline ? "the snapshot" : "the live Thread")")
            if !report.allVerified { throw ExitCode(3) }
        }
    }
}

struct Ledger: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Read the entropy ledger: per-epoch summaries, a partition's trajectory, or a fact's answer losses.")

    @OptionGroup var global: GlobalOptions

    @Option(help: "Run directory.")
    var run: String

    @Option(help: "Show one document's partitions across epochs.")
    var document: String?

    @Option(help: "With --document: a single partition index.")
    var partition: Int?

    @Option(help: "Show one fact's answer-token losses across eval epochs.")
    var fact: String?

    @Flag(help: "Print JSON rows instead of a table.")
    var json = false

    func run() async throws {
        try await guarded {
            let runDirectory = URL(fileURLWithPath: run)
            let manifest = try RunManifest.load(runDirectory)
            let encoder = JSONCoding.lineEncoder()
            func emit<Row: Encodable>(_ rows: [Row], _ table: TextTable) throws {
                if json {
                    for row in rows { print(String(decoding: try encoder.encode(row), as: UTF8.self)) }
                } else {
                    print(Format.table(table))
                }
            }
            if let fact {
                let rows = try LedgerReader.facts(runDirectory: runDirectory, manifest: manifest, factID: fact)
                try emit(rows, LedgerTables.factLosses(rows))
                return
            }
            if let document {
                let rows = try LedgerReader.partitions(runDirectory: runDirectory, manifest: manifest, documentID: document, partitionIndex: partition)
                try emit(rows, LedgerTables.partitionTrajectory(rows))
                return
            }
            let rows = try LedgerReader.epochs(runDirectory: runDirectory)
            if !json { print(LedgerTables.header(manifest)) }
            try emit(rows, LedgerTables.epochs(rows))
        }
    }
}
