//
//  Backend.swift
//  RaoLMStudio
//
//  WHAT: Where the studio's jobs run. `LiveBackend` sends MLX jobs (train, load a run,
//        generate, verify, evaluate, ground) to the one MLX worker thread and runs everything
//        else — scans, the doctor, the Thread, the corpus, test suites, files — concurrently.
//  PIN:  Every job reports `jobStarted`, then `jobFinished` or `jobFailed` with the same
//        failure mapping and exit codes the CLI uses. Status polls and scans are deduplicated.
//

import Foundation
import RaoLM
import RaoLMTerminal
import RaoLMWorkflows

public protocol StudioBackend: AnyObject, Sendable {
    func submit(_ job: StudioJob)
    func shutdown() async
}

public struct StudioOptions: Sendable {
    public var root: DataRoot
    public var fixtures: URL?
    public var threadBinary: String?
    public var httpPort: Int
    public var grpcPort: Int
    public var owner: String

    public init(root: DataRoot, fixtures: URL? = nil, threadBinary: String? = nil, httpPort: Int = ThreadEndpoint.defaultHTTPPort,
                grpcPort: Int = ThreadEndpoint.defaultGRPCPort, owner: String = "raolm-demo") {
        self.root = root
        self.fixtures = fixtures
        self.threadBinary = threadBinary
        self.httpPort = httpPort
        self.grpcPort = grpcPort
        self.owner = owner
    }
}

/// The tokenizer and corpus snapshots, loaded once and shared by every job.
actor SharedCache {
    private var tokenizer: RaoTokenizer?
    private var snapshots: [String: CorpusSnapshot] = [:]
    private var order: [String] = []

    func tokenizer() async throws -> RaoTokenizer {
        if let tokenizer { return tokenizer }
        let loaded = try await RaoTokenizer.load()
        tokenizer = loaded
        return loaded
    }

    func snapshot(at path: String) throws -> CorpusSnapshot {
        if let cached = snapshots[path] { return cached }
        let snapshot = try CorpusSnapshot.load(from: URL(fileURLWithPath: path))
        snapshots[path] = snapshot
        order.append(path)
        if order.count > 4 { snapshots[order.removeFirst()] = nil }
        return snapshot
    }

    func store(_ snapshot: CorpusSnapshot, at path: String) {
        snapshots[path] = snapshot
        order.append(path)
    }
}

/// The file and corpus jobs both backends run the same way.
struct FileJobs: Sendable {
    let root: DataRoot
    let cache: SharedCache
    let post: @Sendable (StudioEvent) -> Void
    /// Where partitions are read from; the fixture backend points it at a regenerated corpus.
    var snapshotPath: @Sendable (RunManifest) -> String = { $0.corpus.snapshotPath }

    func scanRuns() {
        let (runs, warnings) = RunSummary.scan(root: root)
        post(.runsLoaded(runs, warnings: warnings))
    }

    func neighbourRows(_ generation: CitedGeneration, tokenizer: RaoTokenizer) -> [[NeighbourRow]] {
        generation.traces.filter { !$0.isPrompt }.map { trace in
            trace.neighbours.map { NeighbourRow(neighbour: $0, valueText: tokenizer.tokenText($0.value)) }
        }
    }

    func loadGeneration(_ url: URL) async throws {
        let generation = try CitedGeneration.load(from: url)
        let tokenizer = try await cache.tokenizer()
        // Beside the file (`generate --json x.json` → x.grounding.json), else by generation id
        // (what `eval --grounding` and the studio save, next to eval-NNN.json files).
        let run = url.deletingLastPathComponent().deletingLastPathComponent()
        let grounding = [GroundingRecord.url(besideGeneration: url), GroundingRecord.url(runDirectory: run, generationID: generation.generationID)]
            .first { FileManager.default.fileExists(atPath: $0.path) }
            .flatMap { try? GroundingRecord.load(from: $0) }
        post(.generationLoaded(generation, url, neighbourRows(generation, tokenizer: tokenizer), grounding))
    }

    func listGenerations(run: URL) {
        let directory = RunLayout.generations(run)
        let names = (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
        let urls = names.filter { $0.hasSuffix(".json") && !$0.hasSuffix(".grounding.json") }
            .map { directory.appendingPathComponent($0) }
            .sorted { a, b in
                let da = (try? a.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let db = (try? b.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return da > db
            }
        post(.generationsListed(run: run, urls))
    }

    func saveGeneration(_ generation: CitedGeneration, run: URL) throws {
        let url = RunLayout.generations(run).appendingPathComponent("\(generation.generationID).json")
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try generation.save(to: url)
        post(.generationSaved(url))
    }

    func partitionText(run: URL, documentID: String, partitionIndex: Int) async throws {
        let manifest = try RunManifest.load(run)
        let snapshot = try await cache.snapshot(at: snapshotPath(manifest))
        guard let document = snapshot.documents.first(where: { $0.id == documentID }),
              let partition = document.partitions.first(where: { $0.index == partitionIndex }) else {
            throw RaoLMFailure("\(documentID) p\(partitionIndex) is not in the run's snapshot", code: 66)
        }
        let tokenizer = try await cache.tokenizer()
        let offsets = tokenizer.byteOffsets(tokenizer.encode(partition.text))
        post(.partitionText(PartitionText(documentID: documentID, partitionIndex: partitionIndex, text: partition.text, tokenByteOffsets: offsets)))
    }

    func ledger(run: URL, mode: LedgerMode, filter: String, partition: Int?) throws {
        let manifest = try RunManifest.load(run)
        let table: TextTable
        switch mode {
        case .epochs:
            table = LedgerTables.epochs(try LedgerReader.epochs(runDirectory: run))
        case .partitions:
            guard !filter.isEmpty else { throw RaoLMFailure("name a document id", hint: "/ sets it", code: 64) }
            table = LedgerTables.partitionTrajectory(try LedgerReader.partitions(runDirectory: run, manifest: manifest, documentID: filter, partitionIndex: partition))
        case .facts:
            guard !filter.isEmpty else { throw RaoLMFailure("name a fact id (a suffix is enough)", hint: "/ sets it", code: 64) }
            table = LedgerTables.factLosses(try LedgerReader.facts(runDirectory: run, manifest: manifest, factID: filter))
        }
        post(.ledgerLoaded(LedgerData(runID: manifest.runID, header: LedgerTables.header(manifest), mode: mode, table: table)))
    }

    func scanCorpora() { post(.corporaLoaded(CorpusSummary.scan(root: root))) }
    func scanSnapshots() { post(.snapshotsLoaded(SnapshotChoice.scan(root: root))) }

    func generateCorpus(_ form: CorpusForm) throws {
        let directory = root.corpus(slug: form.slug)
        if CorpusStore.exists(at: directory), !form.force {
            throw RaoLMFailure("a corpus already exists at \(StudioApp.abbreviate(directory.path))", hint: "turn on overwrite", code: 73)
        }
        let started = Date()
        let corpus = try SyntheticCorpus.generate(slug: form.slug, seed: form.seed, documentCount: form.documents, maxChars: form.maxChars)
        if FileManager.default.fileExists(atPath: directory.path) { try FileManager.default.removeItem(at: directory) }
        try CorpusStore.write(corpus, to: directory)
        post(.corpusLoaded(corpus, directory))
        post(.corpusResult("generated \(corpus.manifest.documentCount) documents, \(corpus.manifest.partitionCount) partitions, \(corpus.manifest.factCount) facts in \(Format.duration(Date().timeIntervalSince(started))) · \(Format.short(corpus.manifest.corpusHash))", problems: []))
    }

    func offlineSnapshot(corpus url: URL) async throws {
        let corpus = try CorpusStore.load(url)
        let snapshot = CorpusSnapshot.offline(corpus)
        let directory = root.snapshot(hash: snapshot.corpusHash)
        try snapshot.save(to: directory)
        await cache.store(snapshot, at: directory.appendingPathComponent(CorpusSnapshot.fileName).path)
        post(.corpusResult("offline snapshot \(Format.short(snapshot.corpusHash)) · \(snapshot.documentCount) documents — ready to train", problems: []))
    }
}

public final class LiveBackend: StudioBackend, @unchecked Sendable {
    let options: StudioOptions
    let post: @Sendable (StudioEvent) -> Void
    let cache = SharedCache()
    let files: FileJobs
    let thread: ThreadSession
    let tests: TestRunner
    let mlx: MLXWorker
    private let lock = NSLock()
    private var inFlight: Set<JobKind> = []

    public init(options: StudioOptions, mailbox: Mailbox<StudioEvent>) {
        self.options = options
        let post: @Sendable (StudioEvent) -> Void = { mailbox.post($0) }
        self.post = post
        files = FileJobs(root: options.root, cache: cache, post: post)
        thread = ThreadSession(options: options)
        tests = TestRunner(post: post)
        mlx = MLXWorker(options: options, cache: cache, files: files, post: post)
    }

    public func submit(_ job: StudioJob) {
        switch job {
        case .cancel:
            mlx.cancel()
            return
        case .cancelTests:
            tests.cancel()
            return
        default:
            break
        }
        guard let kind = job.kind else { return }
        if kind.isMLX {
            mlx.enqueue(job)
            return
        }
        let deduplicated: Set<JobKind> = [.scanRuns, .threadStatus, .scanCorpora, .scanSnapshots, .doctor]
        lock.lock()
        if deduplicated.contains(kind), inFlight.contains(kind) {
            lock.unlock()
            return
        }
        inFlight.insert(kind)
        lock.unlock()
        let post = self.post
        Task.detached { [self] in
            post(.jobStarted(kind, Self.label(job)))
            do {
                try await perform(job)
                post(.jobFinished(kind))
            } catch {
                post(.jobFailed(kind, FailureMapping.describe(error)))
            }
            lock.withLock { _ = inFlight.remove(kind) }
        }
    }

    static func label(_ job: StudioJob) -> String {
        switch job {
        case .doctor: return "running the doctor"
        case .runTests(let filter, _, _): return "test suites" + (filter.map { " · \($0)" } ?? "")
        case .threadStart: return "starting a Thread"
        case .threadStop: return "stopping the Thread"
        case .corpusGenerate(let form): return "generating \(form.slug) (\(form.documents) documents)"
        case .ingest: return "depositing the corpus into the Thread"
        case .pull: return "exporting the corpus from the Thread"
        case .offlineSnapshot: return "taking an offline snapshot"
        case .loadGeneration: return "opening a generation"
        case .ledger: return "reading the ledger"
        default: return job.kind?.rawValue ?? ""
        }
    }

    func perform(_ job: StudioJob) async throws {
        switch job {
        case .scanRuns: files.scanRuns()
        case .doctor:
            post(.doctorChecks(await DoctorChecks.run(root: options.root, threadBinary: options.threadBinary,
                                                     httpPort: options.httpPort, grpcPort: options.grpcPort)))
        case .runTests(let filter, let mlx, let thread): try await tests.run(filter: filter, mlx: mlx, thread: thread)
        case .threadStatus(let withLog): post(.threadStatus(await thread.status(withLog: withLog)))
        case .threadStart(let fresh):
            let endpoint = try await thread.start(fresh: fresh)
            post(.log("Thread \(endpoint.nodeID?.uuidString.prefix(8).lowercased() ?? "?") is healthy on http :\(endpoint.httpPort) / grpc :\(endpoint.grpcPort)"))
            post(.threadStatus(await thread.status(withLog: true)))
        case .threadStop:
            let stopped = await thread.stop()
            post(.log(stopped ? "Thread stopped" : "no running Thread recorded"))
            post(.threadStatus(await thread.status(withLog: true)))
        case .scanCorpora: files.scanCorpora()
        case .corpusGenerate(let form): try files.generateCorpus(form)
        case .corpusLoad(let url): post(.corpusLoaded(try CorpusStore.load(url), url))
        case .ingest(let url): try await ingest(url)
        case .pull(let url): try await pull(url)
        case .offlineSnapshot(let url): try await files.offlineSnapshot(corpus: url)
        case .scanSnapshots: files.scanSnapshots()
        case .saveGeneration(let generation, let run): try files.saveGeneration(generation, run: run)
        case .loadGeneration(let url): try await files.loadGeneration(url)
        case .listGenerations(let run): files.listGenerations(run: run)
        case .partitionText(let run, let documentID, let partitionIndex):
            try await files.partitionText(run: run, documentID: documentID, partitionIndex: partitionIndex)
        case .ledger(let run, let mode, let filter, let partition): try files.ledger(run: run, mode: mode, filter: filter, partition: partition)
        default: break
        }
    }

    func ingest(_ url: URL) async throws {
        let corpus = try CorpusStore.load(url)
        let slug = corpus.manifest.slug
        let group = "raolm-\(slug)"
        let client = ThreadCorpusClient(endpoint: await thread.endpoint())
        _ = try await client.health()
        let post = self.post
        let report = try await client.index(
            corpus.documents, slug: slug, owner: options.owner, group: group, groupLabel: "RaoLM \(slug) corpus", batchSize: 32
        ) { done, total in post(.ingestProgress(done, total)) }
        post(.jobProgress(.ingest, "waiting for the Thread to embed and index \(report.documents) documents"))
        let waitStart = Date()
        try await client.waitUntilIndexed(expected: Set(corpus.documents.map(\.id)), owner: options.owner, group: group,
                                          prefix: DocumentID.prefix(slug: slug))
        post(.corpusResult("deposited \(report.documents) documents / \(report.partitions) partitions; indexed in \(Format.duration(Date().timeIntervalSince(waitStart)))", problems: []))
    }

    func pull(_ url: URL) async throws {
        let corpus = try CorpusStore.load(url)
        let slug = corpus.manifest.slug
        let endpoint = await thread.endpoint()
        let client = ThreadCorpusClient(endpoint: endpoint)
        let snapshot = try await client.exportCorpus(owner: options.owner, group: "raolm-\(slug)", prefix: DocumentID.prefix(slug: slug), slug: slug)
        guard snapshot.documentCount > 0 else {
            throw RaoLMFailure("the Thread exported no documents for owner \(options.owner)", hint: "ingest first (i)", code: 66)
        }
        let problems = snapshot.diff(against: corpus)
        let directory = options.root.snapshot(hash: snapshot.corpusHash)
        if problems.isEmpty {
            try snapshot.save(to: directory)
            await cache.store(snapshot, at: directory.appendingPathComponent(CorpusSnapshot.fileName).path)
        }
        post(.corpusResult(problems.isEmpty
            ? "ExportCorpus returned \(snapshot.documentCount) documents byte-identical to the corpus · \(Format.short(snapshot.corpusHash)) from node \(snapshot.threadID?.prefix(8).lowercased() ?? "?")"
            : "the Thread export differs from the generated corpus (not saved)", problems: problems))
    }

    public func shutdown() async {
        mlx.cancel()
        tests.cancel()
        mlx.stop()
    }
}

/// The Thread node the studio talks to, and the one it hosts, if it started one.
actor ThreadSession {
    let options: StudioOptions
    private var host: ThreadHost?

    init(options: StudioOptions) {
        self.options = options
    }

    func endpoint() -> ThreadEndpoint {
        ThreadResolve.endpoint(root: options.root, fallback: ThreadEndpoint(
            httpPort: options.httpPort, grpcPort: options.grpcPort, nodeID: ThreadHost.readNodeID(dataDirectory: options.root.threadDB)))
    }

    func status(withLog: Bool) async -> ThreadStatus {
        let record = ThreadHostRecord.load(dataDirectory: options.root.threadDB)
        let endpoint = endpoint()
        let client = ThreadCorpusClient(endpoint: endpoint)
        var health: ThreadHealth?
        var stats: ThreadStats?
        var failure: String?
        do {
            health = try await client.health(timeout: 1)
            stats = try? await client.stats()
        } catch {
            failure = "\(error)"
        }
        var owned = false
        if let host { owned = await host.isRunning }
        var tail: [String] = []
        if withLog {
            let log = record.map { URL(fileURLWithPath: $0.logFile) } ?? options.root.logs.appendingPathComponent("thread.log")
            if let data = try? Data(contentsOf: log) {
                tail = String(decoding: data.suffix(48 * 1024), as: UTF8.self).split(separator: "\n").suffix(200).map(String.init)
            }
        }
        return ThreadStatus(
            endpoint: endpoint, record: record, health: health, stats: stats,
            httpBusy: PortProbe.isListening(host: endpoint.host, port: endpoint.httpPort),
            grpcBusy: PortProbe.isListening(host: endpoint.host, port: endpoint.grpcPort),
            ownedByStudio: owned, error: failure, logTail: tail, polledAt: Date())
    }

    func start(fresh: Bool) async throws -> ThreadEndpoint {
        let dataDirectory = options.root.threadDB
        if fresh {
            await ThreadHost.stopRecorded(dataDirectory: dataDirectory)
            try? FileManager.default.removeItem(at: dataDirectory)
        }
        let binary = try ThreadBinaryLocator.locate(explicit: options.threadBinary)
        let host = ThreadHost(configuration: ThreadHostConfiguration(
            binary: binary, dataDirectory: dataDirectory, logFile: options.root.logs.appendingPathComponent("thread.log"),
            httpPort: options.httpPort, grpcPort: options.grpcPort, nodeID: nil))
        let endpoint = try await host.start()
        self.host = host
        return endpoint
    }

    func stop() async -> Bool {
        if let host, await host.isRunning {
            await host.stop()
            self.host = nil
            return true
        }
        return await ThreadHost.stopRecorded(dataDirectory: options.root.threadDB)
    }
}

/// scripts/test.sh as a child process, its output streamed line by line.
final class TestRunner: @unchecked Sendable {
    private let post: @Sendable (StudioEvent) -> Void
    private let lock = NSLock()
    private var process: Process?

    init(post: @escaping @Sendable (StudioEvent) -> Void) {
        self.post = post
    }

    static func repositoryRoot() -> URL? {
        if let explicit = ProcessInfo.processInfo.environment["RAOLM_REPO"] { return URL(fileURLWithPath: explicit) }
        var directory = Bundle.main.executableURL?.resolvingSymlinksInPath().deletingLastPathComponent()
        while let current = directory, current.path != "/" {
            if FileManager.default.fileExists(atPath: current.appendingPathComponent("Package.swift").path),
               FileManager.default.fileExists(atPath: current.appendingPathComponent("scripts/test.sh").path) {
                return current
            }
            directory = current.deletingLastPathComponent()
        }
        return nil
    }

    func run(filter: String?, mlx: Bool, thread: Bool) async throws {
        guard let repository = Self.repositoryRoot() else {
            post(.testFinished(127))
            throw RaoLMFailure("cannot find the RaoLM checkout that holds scripts/test.sh", hint: "set RAOLM_REPO", code: 66)
        }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = ["scripts/test.sh"] + (filter.map { ["--filter", $0] } ?? [])
        process.currentDirectoryURL = repository
        var environment = ProcessInfo.processInfo.environment
        environment["RAOLM_MLX_TESTS"] = mlx ? "1" : "0"
        if thread { environment["RAOLM_THREAD_TESTS"] = "1" } else { environment["RAOLM_THREAD_TESTS"] = nil }
        environment["NO_COLOR"] = "1"
        process.environment = environment
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        process.standardInput = FileHandle.nullDevice
        let post = self.post
        let buffer = LineBuffer()
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            let lines = buffer.feed(data)
            if !lines.isEmpty { post(.testOutput(lines)) }
        }
        post(.testOutput(["$ scripts/test.sh" + (filter.map { " --filter \($0)" } ?? "") + "   (RAOLM_MLX_TESTS=\(mlx ? 1 : 0)\(thread ? " RAOLM_THREAD_TESTS=1" : ""))"]))
        let code: Int32 = try await withCheckedThrowingContinuation { continuation in
            process.terminationHandler = { finished in
                pipe.fileHandleForReading.readabilityHandler = nil
                let rest = pipe.fileHandleForReading.readDataToEndOfFile()
                var lines = buffer.feed(rest)
                lines += buffer.flush()
                if !lines.isEmpty { post(.testOutput(lines)) }
                continuation.resume(returning: finished.terminationStatus)
            }
            do {
                try process.run()
                lock.withLock { self.process = process }
            } catch {
                continuation.resume(throwing: error)
            }
        }
        lock.withLock { self.process = nil }
        post(.testFinished(code))
    }

    func cancel() {
        lock.lock()
        let process = self.process
        lock.unlock()
        guard let process, process.isRunning else { return }
        // bash does not forward SIGTERM to `swift test`; stop the children first.
        let pkill = Process()
        pkill.executableURL = URL(fileURLWithPath: "/usr/bin/pkill")
        pkill.arguments = ["-TERM", "-P", String(process.processIdentifier)]
        try? pkill.run()
        pkill.waitUntilExit()
        process.terminate()
    }
}

/// Splits streamed bytes into lines, dropping ANSI escapes.
final class LineBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var pending = Data()

    func feed(_ data: Data) -> [String] {
        lock.lock()
        defer { lock.unlock() }
        pending.append(data)
        var lines: [String] = []
        while let newline = pending.firstIndex(of: 0x0A) {
            lines.append(Self.clean(String(decoding: pending[pending.startIndex..<newline], as: UTF8.self)))
            pending.removeSubrange(pending.startIndex...newline)
        }
        return lines
    }

    func flush() -> [String] {
        lock.lock()
        defer { lock.unlock() }
        guard !pending.isEmpty else { return [] }
        let line = Self.clean(String(decoding: pending, as: UTF8.self))
        pending.removeAll()
        return [line]
    }

    static func clean(_ line: String) -> String {
        var text = line.replacingOccurrences(of: "\u{1B}\\[[0-9;?]*[A-Za-z]", with: "", options: .regularExpression)
        if let carriage = text.lastIndex(of: "\r") { text = String(text[text.index(after: carriage)...]) }
        return text
    }
}
