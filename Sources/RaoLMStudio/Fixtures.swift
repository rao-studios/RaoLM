//
//  Fixtures.swift
//  RaoLMStudio
//
//  WHAT: A backend that answers every job from a recorded run instead of MLX and a Thread,
//        so every screen can be demonstrated and tested anywhere: `raolm ui --fixtures <dir>`.
//  IN:   <dir>/fixtures.json (the corpus's slug, seed, documents, max chars; optional doctor
//        checks and Thread status) and <dir>/runs/<id>/ as a demo leaves it — run.json, ledger/,
//        generations/*.json (+ .grounding.json), eval.json — minus checkpoints and indexes.
//  PIN:  The corpus is regenerated from its seed (the generator is deterministic) and its
//        offline snapshot stands in for the Thread export; the corpus hash matches. Training
//        replays the recorded ledger; generation replays a saved generation; verification is
//        real (it needs no MLX).
//

import Foundation
import RaoLM
import RaoLMTerminal
import RaoLMWorkflows

public struct FixtureConfig: Codable, Sendable {
    public struct CorpusSpec: Codable, Sendable {
        public var slug: String
        public var seed: UInt64
        public var documents: Int
        public var maxChars: Int?
    }

    public struct ThreadSpec: Codable, Sendable {
        public var nodeID: String
        public var httpPort: Int
        public var grpcPort: Int
        public var documents: Int
        public var groups: Int
        public var owners: Int
    }

    public var corpus: CorpusSpec
    public var doctor: [DoctorCheckSpec]?
    public var thread: ThreadSpec?

    public struct DoctorCheckSpec: Codable, Sendable {
        public var name: String
        public var ok: Bool
        public var detail: String
    }
}

public final class FixtureBackend: StudioBackend, @unchecked Sendable {
    public static let snapshotKey = "fixture:snapshot"

    let directory: URL
    let root: DataRoot
    let config: FixtureConfig
    let corpus: GeneratedCorpus
    let snapshot: CorpusSnapshot
    let cache = SharedCache()
    let files: FileJobs
    let post: @Sendable (StudioEvent) -> Void
    let flag = CancelFlag()
    /// The whole training replay takes at most this long (tests set zero).
    public var replayDuration: Duration = .seconds(15)

    public init(directory: URL, post: @escaping @Sendable (StudioEvent) -> Void) throws {
        self.directory = directory
        root = DataRoot(url: directory)
        config = try JSONCoding.read(FixtureConfig.self, from: directory.appendingPathComponent("fixtures.json"))
        corpus = try SyntheticCorpus.generate(
            slug: config.corpus.slug, seed: config.corpus.seed, documentCount: config.corpus.documents, maxChars: config.corpus.maxChars ?? 600)
        snapshot = CorpusSnapshot.offline(corpus)
        self.post = post
        var files = FileJobs(root: root, cache: cache, post: post)
        files.snapshotPath = { _ in FixtureBackend.snapshotKey }
        self.files = files
    }

    public convenience init(directory: URL, mailbox: Mailbox<StudioEvent>) throws {
        try self.init(directory: directory, post: { mailbox.post($0) })
    }

    var firstRun: URL? { RunSummary.scan(root: root).runs.first?.directory }

    public func submit(_ job: StudioJob) {
        switch job {
        case .cancel:
            flag.cancel()
            return
        case .cancelTests:
            return
        default:
            break
        }
        guard let kind = job.kind else { return }
        let post = self.post
        Task.detached { [self] in
            if kind.isMLX { flag.reset() }
            post(.jobStarted(kind, LiveBackend.label(job)))
            do {
                try await perform(job)
                post(.jobFinished(kind))
            } catch {
                post(.jobFailed(kind, FailureMapping.describe(error)))
            }
        }
    }

    public func shutdown() async { flag.cancel() }

    func perform(_ job: StudioJob) async throws {
        switch job {
        case .scanRuns: files.scanRuns()
        case .doctor:
            let checks = config.doctor?.map { DoctorCheck(name: $0.name, ok: $0.ok, detail: $0.detail) }
                ?? [DoctorCheck(name: "fixtures", ok: true, detail: directory.path),
                    DoctorCheck(name: "raolm metallib", ok: Preflight.ownMetallib() != nil, detail: "not needed in fixture mode")]
            post(.doctorChecks(checks))
        case .runTests:
            post(.testFinished(69))
            throw RaoLMFailure("test suites are not available in fixture mode", code: 69)
        case .threadStatus: post(.threadStatus(threadStatus()))
        case .threadStart, .threadStop, .ingest, .pull:
            throw RaoLMFailure("fixture mode has no Thread", hint: "run raolm ui without --fixtures", code: 69)
        case .scanCorpora:
            post(.corporaLoaded([CorpusSummary(directory: root.corpus(slug: corpus.manifest.slug), manifest: corpus.manifest)]))
        case .corpusLoad(let url): post(.corpusLoaded(corpus, url))
        case .corpusGenerate(let form):
            let generated = try SyntheticCorpus.generate(slug: form.slug, seed: form.seed, documentCount: form.documents, maxChars: form.maxChars)
            post(.corpusLoaded(generated, root.corpus(slug: form.slug)))
            post(.corpusResult("generated \(generated.manifest.documentCount) documents · \(Format.short(generated.manifest.corpusHash)) (fixture mode: kept in memory)", problems: []))
        case .offlineSnapshot:
            post(.corpusResult("offline snapshot \(Format.short(snapshot.corpusHash)) · \(snapshot.documentCount) documents (fixture mode: not written)", problems: []))
        case .scanSnapshots:
            post(.snapshotsLoaded([SnapshotChoice(snapshot: snapshot, path: directory.appendingPathComponent("snapshot.json"), root: root)]))
        case .saveGeneration(let generation, _):
            post(.log("fixture mode: \(generation.generationID) not written"))
        case .loadGeneration(let url): try await files.loadGeneration(url)
        case .listGenerations(let run): files.listGenerations(run: run)
        case .partitionText(let run, let documentID, let partitionIndex):
            await cache.store(snapshot, at: Self.snapshotKey)
            try await files.partitionText(run: run, documentID: documentID, partitionIndex: partitionIndex)
        case .ledger(let run, let mode, let filter, let partition): try files.ledger(run: run, mode: mode, filter: filter, partition: partition)
        case .train: try await replayTraining()
        case .loadContext(let run, let epoch): try await loadContext(run: run, epoch: epoch)
        case .generate(let spec): try await replayGeneration(run: spec.run)
        case .verify(let generation, _, _): try await verify(generation)
        case .eval(let spec): try await replayEval(run: spec.run)
        case .ground(let spec):
            let url = GroundingRecord.url(runDirectory: spec.run, generationID: spec.generation.generationID)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw RaoLMFailure("no recorded grounding for \(spec.generation.generationID) in the fixtures", hint: "fixture mode replays; it does not measure", code: 66)
            }
            post(.grounded(try GroundingRecord.load(from: url)))
        case .cancel, .cancelTests: break
        }
    }

    func threadStatus() -> ThreadStatus {
        guard let t = config.thread else {
            return ThreadStatus(endpoint: ThreadEndpoint(), record: nil, health: nil, stats: nil, httpBusy: false, grpcBusy: false,
                                ownedByStudio: false, error: "fixture mode has no Thread", logTail: [], polledAt: Date())
        }
        let health = try? JSONDecoder().decode(ThreadHealth.self, from: Data(#"{"status":"healthy","stack":"open"}"#.utf8))
        return ThreadStatus(
            endpoint: ThreadEndpoint(httpPort: t.httpPort, grpcPort: t.grpcPort, nodeID: UUID(uuidString: t.nodeID)), record: nil,
            health: health, stats: ThreadStats(documents: t.documents, groups: t.groups, owners: t.owners),
            httpBusy: true, grpcBusy: true, ownedByStudio: false, error: nil,
            logTail: ["(fixture) Thread \(t.nodeID) serving \(t.documents) documents in open mode"], polledAt: Date())
    }

    func manifest(_ run: URL?) throws -> (URL, RunManifest) {
        guard let run = run ?? firstRun else { throw RaoLMFailure("the fixtures hold no run", code: 66) }
        return (run, try RunManifest.load(run))
    }

    func generations(_ run: URL) -> [URL] {
        let directory = RunLayout.generations(run)
        let names = (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
        return names.filter { $0.hasSuffix(".json") && !$0.hasSuffix(".grounding.json") }.sorted().map { directory.appendingPathComponent($0) }
    }

    func loadContext(run: URL, epoch: Int?) async throws {
        let (run, manifest) = try manifest(run)
        let tokenizer = try await cache.tokenizer()
        let chosen = epoch ?? manifest.latestIndexedEpoch ?? manifest.epochs.last?.epoch ?? 0
        let saved = generations(run).lazy.compactMap { try? CitedGeneration.load(from: $0) }.first
        let ref = saved?.manifest ?? ManifestRef(
            runID: manifest.runID, epoch: chosen, checkpointSHA256: manifest.epochRecord(chosen)?.checkpointSHA256 ?? "",
            indexSHA256: manifest.epochRecord(chosen)?.indexSHA256 ?? "", corpusHash: manifest.corpus.corpusHash,
            tokenizerSHA256: manifest.tokenizer.tokenizerSHA256, ledgerSHA256: nil, threadID: manifest.corpus.threadID)
        let tokenized = TokenizedCorpus(snapshot: snapshot, tokenizer: tokenizer, excluding: Set(manifest.excludedDocumentIDs))
        let facts = corpus.documents.flatMap(\.facts)
        let examples = FactLocator.locate(facts, corpus: tokenized, tokenizer: tokenizer).located.prefix(60).map { l in
            let partition = tokenized.partitions[l.row]
            return ExamplePrompt(label: "\(l.fact.kind.rawValue) · \(l.fact.subject)",
                                 slice: CorpusSlice(documentID: partition.documentID, partitionIndex: partition.partitionIndex,
                                                    offset: l.contextToken, length: l.answerToken - l.contextToken),
                                 expected: l.fact.answer)
        }
        post(.contextLoaded(ContextInfo(
            runID: manifest.runID, runDirectory: run, epoch: chosen, indexedEpochs: manifest.indexedEpochs.sorted(), manifest: ref,
            defaults: saved?.params ?? GenerationParameters(tapLayer: manifest.provenance.tapLayer, alpha: manifest.provenance.alpha),
            partitionCount: tokenized.partitions.count, indexEntries: tokenized.tokenCount,
            evalMemorised: manifest.epochRecord(chosen)?.evalMemorisedFraction ?? 0, threadID: manifest.corpus.threadID,
            owner: manifest.corpus.owner, factsPath: nil, examples: Array(examples))))
    }

    func replayTraining() async throws {
        let (run, manifest) = try manifest(nil)
        let steps = try JSONCoding.readLines(StepRow.self, from: LedgerFiles.steps(RunLayout.ledger(run)))
        guard !steps.isEmpty else { throw RaoLMFailure("the fixture run has no step ledger", code: 66) }
        post(.trainPrepared(manifest, [
            "model \(manifest.preset): \(Format.count(manifest.parameterCount)) parameters, tap layer \(manifest.provenance.tapLayer)",
            "corpus: \(manifest.corpus.documentCount) documents, \(manifest.corpus.partitionCount) partitions, \(Format.count(manifest.corpus.tokenCount)) tokens (replayed from the fixtures)",
        ]))
        let perEpoch = manifest.epochs.map(\.steps)
        post(.training(.started(totalSteps: steps.count, stepsPerEpoch: perEpoch.isEmpty ? [steps.count] : perEpoch,
                                tokensPerStep: manifest.hyperparameters.batchSize * manifest.hyperparameters.seqLen)))
        for (index, row) in steps.enumerated() {
            if flag.isCancelled {
                post(.training(.message("stopped in epoch \(row.epoch) after step \(row.step) (replay)")))
                var stopped = manifest
                stopped.status = .stopped
                post(.trainFinished(stopped))
                return
            }
            post(.training(.step(row)))
            let last = index == steps.count - 1 || steps[index + 1].epoch != row.epoch
            if last, let record = manifest.epochRecord(row.epoch) {
                if let sha = record.indexSHA256 {
                    post(.training(.indexed(epoch: row.epoch, entries: manifest.corpus.tokenCount, sha256: sha, seconds: 0.4)))
                }
                post(.training(.epoch(record)))
            }
            if replayDuration > .zero { try await Task.sleep(for: min(.milliseconds(40), replayDuration / steps.count)) }
        }
        post(.trainFinished(manifest))
    }

    func replayGeneration(run: URL) async throws {
        guard let url = generations(run).first else { throw RaoLMFailure("the fixtures hold no saved generation", code: 66) }
        let generation = try CitedGeneration.load(from: url)
        for trace in generation.traces where !trace.isPrompt {
            if flag.isCancelled { throw CancellationError() }
            post(.token(trace))
            if replayDuration > .zero { try await Task.sleep(for: .milliseconds(30)) }
        }
        let tokenizer = try await cache.tokenizer()
        post(.generated(generation, files.neighbourRows(generation, tokenizer: tokenizer)))
    }

    func verify(_ generation: CitedGeneration) async throws {
        let tokenizer = try await cache.tokenizer()
        var copy = generation
        let report = try await CitationVerifier.verify(&copy, reader: InMemoryCorpusReader(snapshot: snapshot), tokenizer: tokenizer)
        let lines = report.checks.map { "\($0.status == .verified ? "✓" : "✗") span \($0.span) \($0.documentID) p\($0.partitionIndex)@\($0.tokenOffset): \($0.status.rawValue)" }
            + ["\(report.verified)/\(report.checks.count) spans verified against the snapshot"]
        post(.verified(copy, lines, source: "snapshot"))
    }

    func replayEval(run: URL) async throws {
        let (run, _) = try manifest(run)
        let url = run.appendingPathComponent("eval.json")
        var report = try JSONCoding.read(EvalReport.self, from: url)
        let local = RunLayout.generations(run)
        for index in report.outcomes.indices {
            guard let file = report.outcomes[index].generationFile else { continue }
            let candidate = local.appendingPathComponent(URL(fileURLWithPath: file).lastPathComponent)
            report.outcomes[index].generationFile = FileManager.default.fileExists(atPath: candidate.path) ? candidate.path : nil
        }
        for line in ["λ \(report.lambdas.map { Format.f($0.lambda, 2) }.joined(separator: ", ")) on \(report.sampleSize) facts (replayed)",
                     "calibration: ECE \(Format.f(report.ece, 3))"] {
            post(.evalProgress(line))
            if replayDuration > .zero { try await Task.sleep(for: .milliseconds(120)) }
        }
        post(.evalFinished(report, url))
    }
}
