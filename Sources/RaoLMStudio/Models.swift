//
//  Models.swift
//  RaoLMStudio
//
//  WHAT: The values screens show: run summaries scanned from the data root, a loaded run's
//        context, a Thread's status, corpora and snapshots on disk, a partition's text with
//        its token offsets, and the token strip of a generation.
//  PIN:  All Sendable values — they cross from workers to the UI. Nothing here holds MLX.
//

import Foundation
import RaoLM
import RaoLMWorkflows

public struct RunSummary: Sendable, Equatable {
    public var id: String
    public var directory: URL
    public var preset: String
    public var status: RunStatus
    public var failure: String?
    public var epochsDone: Int
    public var epochsPlanned: Int
    public var lastMemorised: Float?
    public var indexedEpochs: [Int]
    public var corpusSlug: String?
    public var corpusHash: String
    public var documents: Int
    public var partitions: Int
    public var tokens: Int
    public var threadNodeID: String?
    public var parameterCount: Int
    public var tapLayer: Int
    public var lastEvalLoss: Float?
    public var hasFacts: Bool
    public var eval: EvalBrief?
    public var generationCount: Int
    public var updatedAt: Date

    public struct EvalBrief: Sendable, Equatable {
        public var lambda: Float
        public var exact: Float
        public var citationAt1: Float
        public var auroc: Double?
        public var ece: Float
        public var spansVerified: Int
        public var spanChecks: Int
    }

    public init(manifest m: RunManifest, directory: URL) {
        id = m.runID
        self.directory = directory
        preset = m.preset
        status = m.status
        failure = m.failure
        epochsDone = m.epochs.count
        epochsPlanned = m.hyperparameters.epochs
        lastMemorised = m.epochs.last(where: { $0.evalMemorisedFraction != nil })?.evalMemorisedFraction
        lastEvalLoss = m.epochs.last(where: { $0.evalLoss != nil })?.evalLoss
        indexedEpochs = m.indexedEpochs.sorted()
        corpusSlug = m.corpus.slug
        corpusHash = m.corpus.corpusHash
        documents = m.corpus.documentCount
        partitions = m.corpus.partitionCount
        tokens = m.corpus.tokenCount
        threadNodeID = m.thread?.nodeID ?? m.corpus.threadID
        parameterCount = m.parameterCount
        tapLayer = m.provenance.tapLayer
        hasFacts = m.corpus.factsPath != nil
        updatedAt = m.updatedAt
        let evalURL = directory.appendingPathComponent("eval.json")
        if let report = try? JSONCoding.read(EvalReport.self, from: evalURL),
           let metrics = report.metrics(lambda: report.primaryLambda) ?? report.lambdas.last {
            eval = EvalBrief(
                lambda: metrics.lambda, exact: metrics.exactAnswer, citationAt1: metrics.citationAt1Partition,
                auroc: report.auroc, ece: report.ece, spansVerified: report.spansVerified, spanChecks: report.spanChecks)
        }
        let generations = (try? FileManager.default.contentsOfDirectory(atPath: RunLayout.generations(directory).path)) ?? []
        generationCount = generations.filter { $0.hasSuffix(".json") && !$0.hasSuffix(".grounding.json") }.count
    }

    /// A run left "training" with no update for ten minutes was abandoned.
    public func isStale(now: Date) -> Bool {
        [.training, .indexing, .starting].contains(status) && now.timeIntervalSince(updatedAt) > 600
    }

    /// Every run under the data root, newest first; unreadable run.json files become warnings.
    public static func scan(root: DataRoot) -> (runs: [RunSummary], warnings: [String]) {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: root.runs.path)) ?? []
        var runs: [RunSummary] = []
        var warnings: [String] = []
        for name in names.sorted() where !name.hasPrefix(".") {
            let directory = root.run(id: name)
            guard FileManager.default.fileExists(atPath: directory.appendingPathComponent(RunManifest.fileName).path) else { continue }
            do {
                runs.append(RunSummary(manifest: try RunManifest.load(directory), directory: directory))
            } catch {
                warnings.append("\(name): \(error)")
            }
        }
        return (runs.sorted { $0.updatedAt > $1.updatedAt }, warnings)
    }
}

/// A prompt the Generate screen offers: a fact's context as a corpus slice.
public struct ExamplePrompt: Sendable, Equatable {
    public var label: String
    public var slice: CorpusSlice
    public var expected: String
}

public struct ContextInfo: Sendable, Equatable {
    public var runID: String
    public var runDirectory: URL
    public var epoch: Int
    public var indexedEpochs: [Int]
    public var manifest: ManifestRef
    public var defaults: GenerationParameters
    public var partitionCount: Int
    public var indexEntries: Int
    public var evalMemorised: Float
    public var threadID: String?
    public var owner: String
    public var factsPath: String?
    public var examples: [ExamplePrompt]
}

public struct ThreadStatus: Sendable, Equatable {
    public var endpoint: ThreadEndpoint
    public var record: ThreadHostRecord?
    public var health: ThreadHealth?
    public var stats: ThreadStats?
    public var httpBusy: Bool
    public var grpcBusy: Bool
    public var ownedByStudio: Bool
    public var error: String?
    public var logTail: [String]
    public var polledAt: Date

    public var isUp: Bool { health?.status == "healthy" }
    public var nodeID: String? { endpoint.nodeID?.uuidString ?? record?.nodeID }
}

extension ThreadHostRecord: Equatable {
    public static func == (a: ThreadHostRecord, b: ThreadHostRecord) -> Bool {
        a.pid == b.pid && a.nodeID == b.nodeID && a.httpPort == b.httpPort && a.grpcPort == b.grpcPort
    }
}

public struct CorpusSummary: Sendable, Equatable {
    public var directory: URL
    public var manifest: CorpusManifest

    public static func scan(root: DataRoot) -> [CorpusSummary] {
        let base = root.url.appendingPathComponent("corpora", isDirectory: true)
        let names = (try? FileManager.default.contentsOfDirectory(atPath: base.path)) ?? []
        return names.sorted().compactMap { name in
            let directory = base.appendingPathComponent(name, isDirectory: true)
            guard let manifest = try? JSONCoding.read(CorpusManifest.self, from: directory.appendingPathComponent("manifest.json")) else { return nil }
            return CorpusSummary(directory: directory, manifest: manifest)
        }
    }
}

public struct SnapshotChoice: Sendable, Equatable {
    public var path: URL
    public var corpusHash: String
    public var slug: String?
    public var source: String
    public var threadID: String?
    public var documentCount: Int
    public var partitionCount: Int
    public var factsPath: URL?
    public var exportedAt: Date

    public init(snapshot: CorpusSnapshot, path: URL, root: DataRoot) {
        self.path = path
        corpusHash = snapshot.corpusHash
        slug = snapshot.slug
        source = snapshot.source
        threadID = snapshot.threadID
        documentCount = snapshot.documentCount
        partitionCount = snapshot.partitionCount
        exportedAt = snapshot.exportedAt
        if let slug = snapshot.slug {
            let facts = root.corpus(slug: slug).appendingPathComponent("facts.jsonl")
            factsPath = FileManager.default.fileExists(atPath: facts.path) ? facts : nil
        }
    }

    public var label: String {
        "\(slug ?? "?") \(Format.short(corpusHash)) · \(documentCount) docs · \(source)"
    }

    public static func scan(root: DataRoot) -> [SnapshotChoice] {
        let base = root.url.appendingPathComponent("snapshots", isDirectory: true)
        let names = (try? FileManager.default.contentsOfDirectory(atPath: base.path)) ?? []
        return names.compactMap { name -> SnapshotChoice? in
            let path = base.appendingPathComponent(name).appendingPathComponent(CorpusSnapshot.fileName)
            guard let snapshot = try? CorpusSnapshot.load(from: path) else { return nil }
            return SnapshotChoice(snapshot: snapshot, path: path, root: root)
        }.sorted { $0.exportedAt > $1.exportedAt }
    }
}

/// One partition's live text and where each of its tokens starts, for the metadata panel.
public struct PartitionText: Sendable, Equatable {
    public var documentID: String
    public var partitionIndex: Int
    public var text: String
    /// UTF-8 offset of each token's start, plus the total.
    public var tokenByteOffsets: [Int]

    public var key: String { "\(documentID)#\(partitionIndex)" }

    /// The byte range of tokens `start..<end`, clamped to the text.
    public func byteRange(tokens start: Int, _ end: Int) -> Range<Int>? {
        guard start >= 0, start < end, end < tokenByteOffsets.count else { return nil }
        return tokenByteOffsets[start]..<tokenByteOffsets[end]
    }
}

/// The generation as the strip shows it: prompt tokens, generated tokens coloured by
/// confidence, and `[[n]]` markers after verbatim spans.
public enum TokenStrip {
    public enum Cell: Sendable, Equatable {
        case prompt(String)
        case separator
        case token(generated: Int, text: String, confidence: Float?, uncited: Bool, verified: Bool)
        case marker(Int, verified: Bool)
    }

    public static func build(_ generation: CitedGeneration) -> [Cell] {
        var cells: [Cell] = [.prompt(generation.prompt.text), .separator]
        let (markers, sources) = CitationMarkers.markers(generation)
        var numberAt: [Int: Int] = [:]
        for marker in markers { numberAt[marker.traceIndex] = marker.number }
        var verifiedNumbers = Set<Int>()
        for source in sources where source.verification == .verified { verifiedNumbers.insert(source.number) }
        for (index, trace) in generation.traces.filter({ !$0.isPrompt }).enumerated() {
            let verified = trace.spanIndex.flatMap { i in i < generation.spans.count ? generation.spans[i] : nil }
                .map { $0.kind == .verbatim && $0.verification?.status == .verified } ?? false
            cells.append(.token(generated: index, text: trace.text, confidence: trace.confidence, uncited: trace.uncited, verified: verified))
            if let number = numberAt[trace.index] { cells.append(.marker(number, verified: verifiedNumbers.contains(number))) }
        }
        return cells
    }

    /// The same cells for tokens still streaming in (no confidence yet).
    public static func streaming(prompt: String, traces: [TokenTrace]) -> [Cell] {
        [.prompt(prompt), .separator] + traces.enumerated().map { index, trace in
            .token(generated: index, text: trace.text, confidence: nil, uncited: true, verified: false)
        }
    }

    public static func plainText(_ cells: [Cell]) -> String {
        cells.map { cell -> String in
            switch cell {
            case .prompt(let text): return text
            case .separator: return "▸"
            case .token(_, let text, _, _, _): return text
            case .marker(let n, _): return "[[\(n)]]"
            }
        }.joined()
    }
}

/// A neighbour row with its value token's text filled in.
public struct NeighbourRow: Sendable, Equatable {
    public var neighbour: Neighbour
    public var valueText: String
}

public enum LedgerMode: String, Sendable, CaseIterable {
    case epochs, partitions, facts

    public var title: String {
        switch self {
        case .epochs: return "Epochs"
        case .partitions: return "Partition"
        case .facts: return "Fact"
        }
    }
}

public struct LedgerData: Sendable {
    public var runID: String
    public var header: String
    public var mode: LedgerMode
    public var table: TextTable
}
