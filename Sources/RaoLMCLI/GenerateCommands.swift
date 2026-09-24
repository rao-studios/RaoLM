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
            for source in rendered.sources {
                let status = source.verification.map { " · \($0.rawValue)" } ?? ""
                Swift.print("  [[\(source.number)]] \(source.documentName) — \(source.documentID) p\(source.partitionIndex) tokens \(source.tokenStart)..<\(source.tokenEnd)\(status)")
            }
            let s = generation.summary
            Swift.print("")
            Swift.print("  \(s.generated) generated tokens: \(s.verbatimCovered) in verbatim spans, \(s.supportOnly) with support citations, \(s.uncited) uncited; mean confidence \(Format.f(s.meanConfidence, 2))")
            Swift.print("  bound to run \(generation.manifest.runID) epoch \(generation.manifest.epoch), checkpoint \(Format.short(generation.manifest.checkpointSHA256)), index \(Format.short(generation.manifest.indexSHA256)), corpus \(Format.short(generation.manifest.corpusHash))")
        }
    }

    static func printTraces(_ generation: CitedGeneration, limit: Int = 64) {
        Swift.print("")
        Swift.print(Format.table(
            ["#", "token", "H_lm", "H_mix", "agree", "conf", "top citation"],
            generation.traces.filter { !$0.isPrompt }.prefix(limit).map { trace in
                let citation = trace.citations.first.map { c in
                    "\(generation.partition(row: c.row)?.documentName ?? "?") p\(c.address.partitionIndex)@\(c.address.tokenOffset)"
                } ?? (trace.uncited ? "(uncited)" : "")
                return [
                    String(trace.index), Format.clip(trace.text.debugDescription, 14), Format.f(trace.lmEntropy, 2),
                    Format.f(trace.mixedEntropy, 2), Format.f(trace.agreement, 2), Format.f(trace.confidence, 2), citation,
                ]
            }))
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
            let generation = try context.generator().generate(request)
            try GenerationPrinter.print(generation, format: format, tokenizer: context.tokenizer)
            if trace { GenerationPrinter.printTraces(generation) }
            if let json {
                try generation.save(to: URL(fileURLWithPath: json))
                if format != .json { print("\nsaved \(json)") }
            }
        }
    }

    static func sliceRequest(_ spec: String, context: RunContext, params: GenerationParameters) throws -> GenerationRequest {
        let parts = spec.split(separator: ":").map(String.init)
        guard parts.count == 4, let partitionIndex = Int(parts[1]), let offset = Int(parts[2]), let length = Int(parts[3]), length > 0 else {
            throw ValidationError("--prompt-from must be DOCUMENT_ID:PARTITION:OFFSET:LENGTH")
        }
        let corpus = try context.tokenizedCorpus()
        guard let row = corpus.row(documentID: parts[0], partitionIndex: partitionIndex) else {
            throw RaoLMFailure("document \(parts[0]) partition \(partitionIndex) is not in the run's corpus", code: 66)
        }
        let partition = corpus.partitions[row]
        guard offset >= 0, offset + length <= partition.tokens.count else {
            throw RaoLMFailure("offset \(offset)+\(length) exceeds the partition's \(partition.tokens.count) tokens", code: 64)
        }
        let tokens = partition.tokens[offset..<(offset + length)].map(Int.init)
        return GenerationRequest(
            promptTokens: tokens, promptText: context.tokenizer.decode(tokens),
            promptSource: SourceAddress(
                threadID: context.manifestRef.threadID, documentID: partition.documentID, partitionIndex: partitionIndex,
                tokenOffset: offset, partitionURL: partition.url, threadPartitionID: partition.threadPartitionID),
            params: params)
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
                let endpoint = ThreadHostRecord.load(dataDirectory: global.root.threadDB)?.endpoint ?? thread.endpoint()
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
            let ledger = RunLayout.ledger(runDirectory)
            let encoder = JSONCoding.lineEncoder()
            if let fact {
                var rows: [FactEpochRow] = []
                for epoch in manifest.epochs.map(\.epoch) {
                    let url = LedgerFiles.facts(ledger, epoch: epoch)
                    guard FileManager.default.fileExists(atPath: url.path) else { continue }
                    rows += try JSONCoding.readLines(FactEpochRow.self, from: url).filter { $0.factID == fact || $0.factID.hasSuffix(fact) }
                }
                if json { for row in rows { print(String(decoding: try encoder.encode(row), as: UTF8.self)) } }
                else {
                    print(Format.table(["epoch", "mean loss", "memorised", "answer token losses"], rows.map {
                        [String($0.epoch), Format.f($0.meanLoss), $0.memorised ? "yes" : "no", $0.answerTokenLosses.map { Format.f($0, 2) }.joined(separator: " ")]
                    }))
                }
                return
            }
            if let document {
                var rows: [PartitionEpochRow] = []
                for epoch in manifest.epochs.map(\.epoch) {
                    let url = LedgerFiles.partitions(ledger, epoch: epoch)
                    guard FileManager.default.fileExists(atPath: url.path) else { continue }
                    rows += try JSONCoding.readLines(PartitionEpochRow.self, from: url).filter {
                        $0.documentID == document && (partition == nil || $0.partitionIndex == partition)
                    }
                }
                if json { for row in rows { print(String(decoding: try encoder.encode(row), as: UTF8.self)) } }
                else {
                    print(Format.table(["epoch", "partition", "tokens", "train loss", "train H", "eval loss", "eval H", "memorised", "since"], rows.map {
                        [String($0.epoch), String($0.partitionIndex), String($0.tokens), Format.f($0.train?.meanLoss), Format.f($0.train?.meanEntropy),
                         Format.f($0.eval?.meanLoss), Format.f($0.eval?.meanEntropy), Format.pct($0.eval?.memorisedFraction), $0.memorisedAtEpoch.map(String.init) ?? "—"]
                    }))
                }
                return
            }
            let rows = try JSONCoding.readLines(EpochRow.self, from: LedgerFiles.epochs(ledger))
            if json { for row in rows { print(String(decoding: try encoder.encode(row), as: UTF8.self)) } }
            else {
                print("run \(manifest.runID) (\(manifest.preset), \(Format.count(manifest.parameterCount)) parameters) on corpus \(Format.short(manifest.corpus.corpusHash))")
                print(Format.table(["epoch", "steps", "train loss", "train H", "eval loss", "eval H", "memorised", "gap", "checkpoint", "index"], rows.map {
                    [String($0.epoch), String($0.steps), Format.f($0.trainLoss), Format.f($0.trainEntropy), Format.f($0.evalLoss), Format.f($0.evalEntropy),
                     Format.pct($0.evalMemorisedFraction), Format.f($0.calibrationGap), Format.short($0.checkpointSHA256), $0.indexSHA256 == nil ? "" : Format.short($0.indexSHA256)]
                }))
            }
        }
    }
}
