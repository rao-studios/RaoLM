//
//  DemoCommand.swift
//  RaoLMCLI
//
//  WHAT: raolm demo — the whole proof of concept in one command: generate the corpus, host a
//        Thread and deposit it, export it back as a hashed snapshot, pretrain with the ledger,
//        build the provenance index, generate with citations, evaluate, and verify every cited
//        span against the live Thread.
//

import ArgumentParser
import Foundation
import RaoLM
import RaoLMWorkflows

struct Demo: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Everything end to end: corpus → Thread → snapshot → train → cite → verify.")

    @OptionGroup var global: GlobalOptions

    @Option(help: "Documents in the synthetic corpus.")
    var documents = 200

    @Option(help: "Corpus seed.")
    var corpusSeed: UInt64 = 42

    @Option(help: "Corpus slug.")
    var slug = "veldmar"

    @OptionGroup var training: TrainOptions

    @Option(help: "Run directory (default: <data root>/runs/<timestamp>-<preset>).")
    var out: String?

    @Option(help: "Thread HTTP port.")
    var httpPort = ThreadEndpoint.defaultHTTPPort

    @Option(help: "Thread gRPC port.")
    var grpcPort = ThreadEndpoint.defaultGRPCPort

    @Option(help: "Path to the thread binary.")
    var threadBinary: String?

    @Option(help: "Facts to evaluate.")
    var factsSample = 50

    @Option(help: "Thread owner id.")
    var owner = "raolm-demo"

    @Flag(help: "Leave the Thread running when done.")
    var keepThread = false

    @Flag(help: "Use the Thread already running on the ports instead of starting one.")
    var reuseThread = false

    @Flag(help: "Skip the paraphrase and fabricated-entity controls.")
    var noControls = false

    @Flag(help: "Skip the grounding measurement (SinatraHarness, with vs without each fact's source) in the evaluation.")
    var noGrounding = false

    func run() async throws {
        try await guarded {
            try Preflight.requireMetallib()
            let root = global.root
            try root.ensure()
            let runID = DataRoot.newRunID(preset: training.preset)
            let runDirectory = out.map { URL(fileURLWithPath: $0) } ?? root.run(id: runID)
            try FileManager.default.createDirectory(at: runDirectory, withIntermediateDirectories: true)
            let started = Date()
            let stop = StopSignal()
            print("RaoLM demo \(runID) → \(runDirectory.path)")

            // 1. Preflight.
            Console.section("Preflight")
            let checks = await DoctorChecks.run(root: root, threadBinary: threadBinary, httpPort: httpPort, grpcPort: grpcPort)
            for check in checks { print(check.line) }
            let blocking = checks.filter { !$0.ok && !["disk", "embedding model"].contains($0.name) && !(reuseThread && $0.name.hasSuffix("port")) }
            guard blocking.isEmpty else {
                throw RaoLMFailure("preflight failed: " + blocking.map(\.name).joined(separator: ", "), code: 78)
            }

            // 2. Corpus.
            Console.section("Corpus")
            let corpusDirectory = root.corpus(slug: slug)
            var corpus: GeneratedCorpus
            if CorpusStore.exists(at: corpusDirectory),
               let existing = try? CorpusStore.load(corpusDirectory),
               existing.manifest.seed == corpusSeed, existing.manifest.documentCount == documents,
               existing.manifest.generatorVersion == SyntheticCorpus.generatorVersion {
                corpus = existing
                print("reusing \(corpusDirectory.path)")
            } else {
                corpus = try SyntheticCorpus.generate(slug: slug, seed: corpusSeed, documentCount: documents)
                if FileManager.default.fileExists(atPath: corpusDirectory.path) { try FileManager.default.removeItem(at: corpusDirectory) }
                try CorpusStore.write(corpus, to: corpusDirectory)
                print("generated \(corpusDirectory.path)")
            }
            print("\(corpus.manifest.documentCount) documents, \(corpus.manifest.partitionCount) partitions, \(corpus.manifest.factCount) facts, corpus hash \(Format.short(corpus.manifest.corpusHash))")
            let sample = corpus.documents[0]
            print("e.g. \(sample.name): \"\(Format.clip(sample.partitions[0].text, 110))\"")

            // 3. Thread.
            Console.section("Thread")
            var host: ThreadHost?
            let endpoint: ThreadEndpoint
            var threadRef: ThreadRef?
            let binary = try ThreadBinaryLocator.locate(explicit: threadBinary)
            if reuseThread {
                let record = ThreadHostRecord.load(dataDirectory: root.threadDB)
                endpoint = record?.endpoint ?? ThreadEndpoint(httpPort: httpPort, grpcPort: grpcPort, nodeID: ThreadHost.readNodeID(dataDirectory: root.threadDB))
                _ = try await ThreadCorpusClient(endpoint: endpoint).health()
                print("using the Thread at \(endpoint.description)")
            } else {
                await ThreadHost.stopRecorded(dataDirectory: root.threadDB)
                if FileManager.default.fileExists(atPath: root.threadDB.path) {
                    try FileManager.default.removeItem(at: root.threadDB)
                    print("wiped \(root.threadDB.path)")
                }
                let configuration = ThreadHostConfiguration(
                    binary: binary, dataDirectory: root.threadDB, logFile: runDirectory.appendingPathComponent("thread.log"),
                    httpPort: httpPort, grpcPort: grpcPort, nodeID: UUID())
                let started = ThreadHost(configuration: configuration)
                print("starting \(binary.path) (log: \(configuration.logFile.path))")
                endpoint = try await started.start()
                host = started
                print("Thread \(endpoint.nodeID?.uuidString ?? "?") is healthy in open mode on http :\(endpoint.httpPort) / grpc :\(endpoint.grpcPort), embedding on device")
            }
            threadRef = ThreadRef(binary: binary.path, host: endpoint.host, httpPort: endpoint.httpPort, grpcPort: endpoint.grpcPort,
                                  dataDir: root.threadDB.path, nodeID: endpoint.nodeID?.uuidString)
            let client = ThreadCorpusClient(endpoint: endpoint)

            func shutdown() async {
                if let host, !keepThread {
                    await host.stop()
                    print("Thread stopped")
                } else if host != nil {
                    print("Thread left running on http :\(endpoint.httpPort) / grpc :\(endpoint.grpcPort); stop it with: raolm thread stop")
                }
            }

            do {
                // 4. Ingest.
                Console.section("Ingest")
                let group = "raolm-\(slug)"
                let ingestReport = try await client.index(
                    corpus.documents, slug: slug, owner: owner, group: group, groupLabel: "RaoLM \(slug) corpus", batchSize: 32
                ) { done, total in if done % 64 == 0 || done == total { print("  accepted \(done)/\(total)") } }
                print("sent \(ingestReport.documents) documents / \(ingestReport.partitions) partitions in \(Format.duration(ingestReport.seconds)); waiting for the Thread to embed and index them")
                let waitStart = Date()
                try await client.waitUntilIndexed(expected: Set(corpus.documents.map(\.id)), owner: owner, group: group, prefix: DocumentID.prefix(slug: slug))
                print("indexed in \(Format.duration(Date().timeIntervalSince(waitStart)))")
                if stop.isSet { throw RaoLMFailure("interrupted", code: 130) }

                // 5. Snapshot.
                Console.section("Snapshot")
                let snapshot = try await client.exportCorpus(owner: owner, group: group, prefix: DocumentID.prefix(slug: slug), slug: slug)
                let problems = snapshot.diff(against: corpus)
                guard problems.isEmpty else {
                    throw RaoLMFailure("the Thread export differs from the generated corpus:\n  " + problems.prefix(10).joined(separator: "\n  "), code: 70)
                }
                let snapshotDirectory = root.snapshot(hash: snapshot.corpusHash)
                try snapshot.save(to: snapshotDirectory)
                print("ExportCorpus returned \(snapshot.documentCount) documents byte-identical to the generated corpus; corpus hash \(snapshot.corpusHash)")
                print("snapshot: \(snapshotDirectory.appendingPathComponent(CorpusSnapshot.fileName).path)")

                // 6. Train.
                Console.section("Train")
                let settings = training.settings()
                _ = try settings.modelConfig()
                let tokenizer = try await RaoTokenizer.load()
                let result = try TrainingDriver.train(
                    TrainingDriver.Plan(
                        snapshot: snapshot, snapshotPath: snapshotDirectory.appendingPathComponent(CorpusSnapshot.fileName).path,
                        factsPath: corpusDirectory.appendingPathComponent("facts.jsonl"), settings: settings,
                        runDirectory: runDirectory, runID: runID, thread: threadRef),
                    tokenizer: tokenizer)
                if stop.isSet { throw RaoLMFailure("interrupted", code: 130) }
                let manifest = result.manifest
                print("eval epochs:")
                print(Format.table(LedgerTables.evalEpochs(manifest)))

                // 7. Cite.
                Console.section("Cited generation")
                let context = try await RunContext.load(runDirectory: runDirectory, allowWeakIndex: true)
                let tokenized = try context.tokenizedCorpus()
                let reader = ThreadCorpusReader(client: client, owner: owner)
                let facts = try JSONCoding.readLines(Fact.self, from: URL(fileURLWithPath: corpusDirectory.appendingPathComponent("facts.jsonl").path))
                let evaluator = FactEvaluator(context: context, corpus: tokenized, facts: facts, reader: reader)
                var params = context.defaultParameters()
                params.maxTokens = 24
                for located in evaluator.sample(2, seed: 11) {
                    let partition = tokenized.partitions[located.row]
                    let promptTokens = partition.tokens[located.contextToken..<located.answerToken].map(Int.init)
                    var generation = try context.generator().generate(GenerationRequest(
                        promptTokens: promptTokens, promptText: context.tokenizer.decode(promptTokens),
                        promptSource: SourceAddress(threadID: endpoint.nodeID?.uuidString, documentID: partition.documentID,
                                                    partitionIndex: partition.partitionIndex, tokenOffset: located.contextToken),
                        params: params))
                    _ = try await CitationVerifier.verify(&generation, reader: reader, tokenizer: context.tokenizer)
                    try generation.save(to: RunLayout.generations(runDirectory).appendingPathComponent("\(generation.generationID).json"))
                    try GenerationPrinter.print(generation, format: .markers, tokenizer: context.tokenizer)
                    print("  expected continuation: \(located.fact.answer)")
                    print("")
                }

                // 8. Evaluate + verify.
                Console.section("Evaluation")
                if !noGrounding {
                    let grounder = RaoGrounder(context: context, corpus: tokenized, budget: 120)
                    evaluator.groundingMeasurer = GroundCommandSupport.makeMeasurer(
                        grounder: grounder, runDirectory: runDirectory, threadID: context.manifestRef.threadID, saveRecords: true)
                }
                let report = try await evaluator.run(
                    options: EvalOptions(factsSample: factsSample, lambdas: [0, 0.5], primaryLambda: 0.5, seed: 7,
                                         includeControls: !noControls, saveGenerations: RunLayout.generations(runDirectory),
                                         grounding: noGrounding ? nil : GroundingEvalOptions(includeControls: !noControls, saveRecords: true))
                ) { print("  \($0)") }
                EvalPrinter.print(report, sampleRows: min(factsSample, 12))
                try JSONCoding.write(report, to: runDirectory.appendingPathComponent("eval.json"))

                // 9. Finish.
                var final = try RunManifest.load(runDirectory)
                final.status = .complete
                try final.save(to: runDirectory)
                Console.section("Done")
                print("run \(runID) complete in \(Format.duration(Date().timeIntervalSince(started)))")
                print("  run.json, ledger/, checkpoints/, provenance/, generations/, eval.json, thread.log under \(runDirectory.path)")
                if let m = report.metrics(lambda: 0.5) {
                    print("  at λ=0.5: exact answers \(Format.pct(m.exactAnswer)), citation@1 \(Format.pct(m.citationAt1Partition)), verified spans \(report.spansVerified)/\(report.spanChecks)")
                }
                print("  try: raolm generate --run \(runDirectory.path) --prompt \"\(Format.clip(sample.facts.first?.prompt ?? sample.name, 60))\"")
                await shutdown()
            } catch {
                var failed = (try? RunManifest.load(runDirectory)) ?? nil
                failed?.status = .failed
                failed?.failure = "\(error)"
                try? failed?.save(to: runDirectory)
                await shutdown()
                throw error
            }
        }
    }
}
