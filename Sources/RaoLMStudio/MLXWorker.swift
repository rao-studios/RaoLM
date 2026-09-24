//
//  MLXWorker.swift
//  RaoLMStudio
//
//  WHAT: The one thread that runs MLX for the studio: training, loading a run's checkpoint
//        and provenance index, cited generation, verification, the evaluation protocol and
//        the grounding measurement, one job at a time, in order.
//  PIN:  A Foundation Thread with a 64 MB stack (MLX's autodiff walks overflow the 512 KB a
//        secondary thread gets by default), never the cooperative pool — a training run blocks
//        for minutes. The model, index and grounder live here and never cross to the UI; only
//        Sendable values do. Async library calls are bridged with `blockingAwait`, whose task
//        `cancel()` reaches; training and generation poll the cancel flag between steps.
//

import Foundation
import RaoLM
import RaoLMTerminal
import RaoLMWorkflows
import SinatraHarness

final class ResultBox<T>: @unchecked Sendable {
    var value: Result<T, Error>?
}

/// What stays loaded between jobs. Touched only on the worker thread.
final class MLXSession {
    var tokenizer: RaoTokenizer?
    var context: RunContext?
    var tokenized: TokenizedCorpus?
    var grounder: RaoGrounder?
}

final class MLXWorker: @unchecked Sendable {
    private let options: StudioOptions
    private let cache: SharedCache
    private let files: FileJobs
    private let post: @Sendable (StudioEvent) -> Void
    private let condition = NSCondition()
    private var queue: [StudioJob] = []
    private var stopping = false
    private var bridge: Task<Void, Never>?
    private let flag = CancelFlag()
    private let session = MLXSession()

    init(options: StudioOptions, cache: SharedCache, files: FileJobs, post: @escaping @Sendable (StudioEvent) -> Void) {
        self.options = options
        self.cache = cache
        self.files = files
        self.post = post
        let thread = Thread { [weak self] in self?.loop() }
        thread.name = "raolm.studio.mlx"
        thread.stackSize = 64 << 20
        thread.qualityOfService = .userInitiated
        thread.start()
    }

    func enqueue(_ job: StudioJob) {
        condition.lock()
        queue.append(job)
        condition.signal()
        condition.unlock()
    }

    func cancel() {
        flag.cancel()
        condition.lock()
        let task = bridge
        condition.unlock()
        task?.cancel()
    }

    func stop() {
        condition.lock()
        stopping = true
        queue.removeAll()
        condition.signal()
        condition.unlock()
    }

    private func loop() {
        while true {
            condition.lock()
            while queue.isEmpty, !stopping { condition.wait() }
            if stopping {
                condition.unlock()
                return
            }
            let job = queue.removeFirst()
            condition.unlock()
            flag.reset()
            guard let kind = job.kind else { continue }
            post(.jobStarted(kind, label(job)))
            do {
                try autoreleasepool { try run(job) }
                post(.jobFinished(kind))
            } catch {
                post(.jobFailed(kind, FailureMapping.describe(error)))
            }
        }
    }

    private func label(_ job: StudioJob) -> String {
        switch job {
        case .train(let spec): return "preparing \(spec.runID)"
        case .loadContext(let run, let epoch): return "loading \(run.lastPathComponent)" + (epoch.map { " epoch \($0)" } ?? "")
        case .generate: return "generating"
        case .verify(_, _, let live): return live ? "verifying against the live Thread" : "verifying against the snapshot"
        case .eval: return "evaluating"
        case .ground(let spec): return "grounding (\(spec.policy)) with vs without the sources"
        default: return job.kind?.rawValue ?? ""
        }
    }

    /// Runs an async body to completion from this thread; `cancel()` cancels its task.
    private func blockingAwait<T>(_ body: @escaping @Sendable () async throws -> T) throws -> T {
        let semaphore = DispatchSemaphore(value: 0)
        let box = ResultBox<T>()
        let task = Task.detached {
            do { box.value = .success(try await body()) } catch { box.value = .failure(error) }
            semaphore.signal()
        }
        condition.lock()
        bridge = task
        condition.unlock()
        if flag.isCancelled { task.cancel() }
        semaphore.wait()
        condition.lock()
        bridge = nil
        condition.unlock()
        return try box.value!.get()
    }

    private func tokenizer() throws -> RaoTokenizer {
        if let tokenizer = session.tokenizer { return tokenizer }
        let cache = self.cache
        let loaded = try blockingAwait { try await cache.tokenizer() }
        session.tokenizer = loaded
        return loaded
    }

    /// The run's context, loading it (and its tokenized corpus) when a different run or epoch is asked for.
    @discardableResult
    private func context(run: URL, epoch: Int?) throws -> RunContext {
        if let context = session.context, context.runDirectory.standardizedFileURL == run.standardizedFileURL,
           epoch == nil || context.epoch == epoch {
            return context
        }
        try Preflight.requireMetallib()
        let tokenizer = try tokenizer()
        let context = try RunContext.load(runDirectory: run, epoch: epoch, allowWeakIndex: true, tokenizer: tokenizer)
        session.context = context
        session.tokenized = try context.tokenizedCorpus()
        session.grounder = nil
        return context
    }

    private func run(_ job: StudioJob) throws {
        switch job {
        case .train(let spec): try train(spec)
        case .loadContext(let run, let epoch): try loadContext(run: run, epoch: epoch)
        case .generate(let spec): try generate(spec)
        case .verify(let generation, let run, let live): try verify(generation, run: run, live: live)
        case .eval(let spec): try evaluate(spec)
        case .ground(let spec): try ground(spec)
        default: break
        }
    }

    private func train(_ spec: TrainSpec) throws {
        let snapshot = try CorpusSnapshot.load(from: spec.snapshotPath)
        let plan = TrainingDriver.Plan(
            snapshot: snapshot, snapshotPath: spec.snapshotPath.path, factsPath: spec.factsPath, settings: spec.settings,
            runDirectory: spec.runDirectory, runID: spec.runID, thread: spec.thread)
        let prepared = try TrainingDriver.prepare(plan, tokenizer: try tokenizer())
        post(.trainPrepared(prepared.manifest, prepared.summaryLines))
        post(.jobProgress(.train, "training \(plan.runID)"))
        let post = self.post
        do {
            let result = try prepared.run(shouldStop: { [flag] in flag.isCancelled }) { post(.training($0)) }
            post(.trainFinished(result.manifest))
        } catch {
            if var manifest = try? RunManifest.load(plan.runDirectory) {
                manifest.status = .failed
                manifest.failure = "\(error)"
                try? manifest.save(to: plan.runDirectory)
            }
            throw error
        }
    }

    private func loadContext(run: URL, epoch: Int?) throws {
        let context = try context(run: run, epoch: epoch)
        let tokenizer = try tokenizer()
        var examples: [ExamplePrompt] = []
        if let factsPath = context.manifest.corpus.factsPath, let corpus = session.tokenized,
           let facts = try? JSONCoding.readLines(Fact.self, from: URL(fileURLWithPath: factsPath)) {
            let located = FactLocator.locate(facts, corpus: corpus, tokenizer: tokenizer).located
            examples = located.prefix(60).map { l in
                let partition = corpus.partitions[l.row]
                return ExamplePrompt(
                    label: "\(l.fact.kind.rawValue) · \(l.fact.subject)",
                    slice: CorpusSlice(documentID: partition.documentID, partitionIndex: partition.partitionIndex,
                                       offset: l.contextToken, length: l.answerToken - l.contextToken),
                    expected: l.fact.answer)
            }
        }
        post(.contextLoaded(ContextInfo(
            runID: context.manifest.runID, runDirectory: context.runDirectory, epoch: context.epoch,
            indexedEpochs: context.manifest.indexedEpochs.sorted(), manifest: context.manifestRef,
            defaults: context.defaultParameters(), partitionCount: context.index.partitions.count, indexEntries: context.index.count,
            evalMemorised: context.index.info.evalMemorisedFraction, threadID: context.manifestRef.threadID,
            owner: context.manifest.corpus.owner, factsPath: context.manifest.corpus.factsPath, examples: examples)))
        if context.index.info.evalMemorisedFraction < 0.5 {
            post(.log(String(format: "epoch %d memorised only %.0f%% of the corpus: citations will be weak", context.epoch, context.index.info.evalMemorisedFraction * 100)))
        }
    }

    private func generate(_ spec: GenerateSpec) throws {
        let context = try context(run: spec.run, epoch: spec.epoch)
        let tokenizer = try tokenizer()
        let params = spec.overrides.apply(to: context.defaultParameters())
        let request: GenerationRequest
        if let slice = spec.slice {
            guard let corpus = session.tokenized else { throw RaoLMFailure("the run's corpus is not loaded", code: 70) }
            request = try slice.request(corpus: corpus, context: context, params: params)
        } else {
            request = GenerationRequest(promptTokens: tokenizer.encode(spec.promptText), promptText: spec.promptText, params: params)
        }
        let post = self.post
        let flag = self.flag
        let generation = try context.generator().generate(request) { trace in
            if flag.isCancelled { throw CancellationError() }
            post(.token(trace))
        }
        post(.generated(generation, files.neighbourRows(generation, tokenizer: tokenizer)))
    }

    private func reader(for run: URL, live: Bool, owner: String) throws -> CorpusReading {
        if live {
            let endpoint = ThreadResolve.endpoint(root: options.root, fallback: ThreadEndpoint(httpPort: options.httpPort, grpcPort: options.grpcPort))
            return ThreadCorpusReader(client: ThreadCorpusClient(endpoint: endpoint), owner: owner)
        }
        let manifest = try RunManifest.load(run)
        let cache = self.cache
        let path = manifest.corpus.snapshotPath
        return InMemoryCorpusReader(snapshot: try blockingAwait { try await cache.snapshot(at: path) })
    }

    private func verify(_ generation: CitedGeneration, run: URL, live: Bool) throws {
        let tokenizer = try tokenizer()
        let manifest = try RunManifest.load(run)
        let reader = try reader(for: run, live: live, owner: manifest.corpus.owner)
        let box = ResultBox<CitedGeneration>()
        box.value = .success(generation)
        let report = try blockingAwait { () -> VerificationReport in
            var copy = generation
            let report = try await CitationVerifier.verify(&copy, reader: reader, tokenizer: tokenizer)
            box.value = .success(copy)
            return report
        }
        let verified = try box.value!.get()
        let saved = RunLayout.generations(run).appendingPathComponent("\(verified.generationID).json")
        if FileManager.default.fileExists(atPath: saved.path) { try verified.save(to: saved) }
        var lines = report.checks.map { check in
            "\(check.status == .verified ? "✓" : "✗") span \(check.span) (\(check.kind.rawValue), \(check.length) tokens) \(check.documentID) p\(check.partitionIndex)@\(check.tokenOffset): \(check.status.rawValue)\(check.detail.map { " — \($0)" } ?? "")"
        }
        let source = live ? "the live Thread" : "the snapshot"
        lines.append("\(report.verified)/\(report.checks.count) spans verified against \(source)")
        post(.verified(verified, lines, source: live ? "live Thread" : "snapshot"))
    }

    private func grounder(_ context: RunContext, budget: TimeInterval) throws -> RaoGrounder {
        if let grounder = session.grounder, grounder.context === context { return grounder }
        guard let corpus = session.tokenized else { throw RaoLMFailure("the run's corpus is not loaded", code: 70) }
        let grounder = RaoGrounder(context: context, corpus: corpus, budget: budget)
        session.grounder = grounder
        return grounder
    }

    private func evaluate(_ spec: EvalSpec) throws {
        let context = try context(run: spec.run, epoch: spec.epoch)
        guard let factsPath = context.manifest.corpus.factsPath else {
            throw RaoLMFailure("this run has no facts.jsonl recorded; only synthetic corpora can be evaluated", code: 66)
        }
        let facts = try JSONCoding.readLines(Fact.self, from: URL(fileURLWithPath: factsPath))
        let reader: CorpusReading
        if spec.offline {
            reader = InMemoryCorpusReader(snapshot: try context.snapshot())
        } else {
            reader = try self.reader(for: spec.run, live: true, owner: spec.owner ?? context.manifest.corpus.owner)
        }
        guard let corpus = session.tokenized else { throw RaoLMFailure("the run's corpus is not loaded", code: 70) }
        let evaluator = FactEvaluator(context: context, corpus: corpus, facts: facts, reader: reader)
        if spec.grounding {
            evaluator.groundingMeasurer = try grounder(context, budget: 120).makeMeasurer(
                runDirectory: spec.run, threadID: context.manifestRef.threadID, saveRecords: true)
        }
        let options = EvalOptions(
            factsSample: spec.factsSample, lambdas: spec.lambdas, primaryLambda: spec.primaryLambda, seed: spec.seed,
            includeControls: spec.controls, saveGenerations: RunLayout.generations(spec.run),
            grounding: spec.grounding ? GroundingEvalOptions(includeControls: spec.controls, saveRecords: true) : nil)
        let post = self.post
        let report = try blockingAwait { try await evaluator.run(options: options) { post(.evalProgress($0)) } }
        let url = spec.run.appendingPathComponent("eval.json")
        try JSONCoding.write(report, to: url)
        post(.evalFinished(report, url))
    }

    private func ground(_ spec: GroundSpec) throws {
        let context = try context(run: spec.run, epoch: spec.generation.manifest.epoch)
        let grounder = try grounder(context, budget: 60)
        let policy = try GroundingSourcePolicy.parse(spec.policy)
        try grounder.checkBinding(spec.generation)
        let inputs = try grounder.inputs(for: spec.generation, policy: policy)
        var raw: GroundingRaw?
        var reason: String?
        let flag = self.flag
        do {
            raw = try grounder.score(inputs, shouldAbort: { flag.isCancelled })
        } catch let stop as GroundingScorer.Stop {
            if flag.isCancelled { throw CancellationError() }
            reason = stop.description
        }
        let scored = raw
        let skipped = reason
        let measurement = try blockingAwait { try await grounder.measurement(inputs, raw: scored, skipped: skipped) }
        let record = try grounder.record(generation: spec.generation, inputs: inputs, measurement: measurement)
        try record.save(to: GroundingRecord.url(runDirectory: spec.run, generationID: spec.generation.generationID))
        post(.grounded(record))
    }
}
