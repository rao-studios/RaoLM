//
//  ThreadCorpusClient.swift
//  RaoLMThread
//
//  WHAT: RaoLM's client for a Thread node: deposit a corpus (ThreadQuery.Index), wait for it
//        to land, export it back as a CorpusSnapshot (ThreadLibrary.ExportCorpus), and read
//        individual documents for verification (ThreadLibrary.Documents).
//  PIN:  Plaintext HTTP/2 gRPC on loopback, one short-lived client per call — the calls are
//        sparse and a hosted Thread may be restarted between them. Index returns before the
//        documents are queryable (Thread enqueues the write), so ingest always waits.
//

import Conduit
import Foundation
import GRPCCore
import GRPCNIOTransportHTTP2
import RaoLMCore

public struct ThreadEndpoint: Sendable, Equatable, Codable {
    public static let defaultHTTPPort = 8095
    public static let defaultGRPCPort = 9095

    public var host: String
    public var httpPort: Int
    public var grpcPort: Int
    public var nodeID: UUID?

    public init(host: String = "127.0.0.1", httpPort: Int = defaultHTTPPort, grpcPort: Int = defaultGRPCPort, nodeID: UUID? = nil) {
        self.host = host
        self.httpPort = httpPort
        self.grpcPort = grpcPort
        self.nodeID = nodeID
    }

    public var description: String { "\(host) http:\(httpPort) grpc:\(grpcPort)" }
}

public struct ThreadHealth: Codable, Sendable, Equatable {
    public var status: String
    public var timestamp: String?
    public var stack: String?
    public var app: String?
    public var contract: Int?
}

public struct ThreadStats: Sendable, Equatable {
    public var documents: Int
    public var groups: Int
    public var owners: Int

    public init(documents: Int, groups: Int, owners: Int) {
        self.documents = documents
        self.groups = groups
        self.owners = owners
    }
}

public struct IndexReport: Sendable, Equatable {
    public var documents: Int
    public var partitions: Int
    public var batches: Int
    public var seconds: Double
}

public actor ThreadCorpusClient {
    public nonisolated let endpoint: ThreadEndpoint

    public init(endpoint: ThreadEndpoint) {
        self.endpoint = endpoint
    }

    private func withClient<R: Sendable>(
        _ body: (GRPCClient<HTTP2ClientTransport.Posix>) async throws -> R
    ) async throws -> R {
        do {
            let transport = try HTTP2ClientTransport.Posix.http2NIOPosix(
                target: .ipv4(host: endpoint.host, port: endpoint.grpcPort),
                transportSecurity: .plaintext)
            return try await withGRPCClient(transport: transport) { client in
                try await body(client)
            }
        } catch let error as RPCError where error.code == .unavailable {
            throw ThreadCorpusError.notReachable(endpoint: endpoint.description, underlying: error.message)
        }
    }

    private func options(seconds: Int64) -> CallOptions {
        var options = CallOptions.defaults
        options.timeout = .seconds(seconds)
        return options
    }

    // MARK: - Health and stats

    public func health(timeout: TimeInterval = 2) async throws -> ThreadHealth {
        guard let url = URL(string: "http://\(endpoint.host):\(endpoint.httpPort)/health") else {
            throw ThreadCorpusError.notReachable(endpoint: endpoint.description, underlying: "bad url")
        }
        var request = URLRequest(url: url)
        request.timeoutInterval = timeout
        request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = timeout
        let session = URLSession(configuration: configuration)
        defer { session.finishTasksAndInvalidate() }
        do {
            let (data, response) = try await session.data(for: request)
            guard (response as? HTTPURLResponse)?.statusCode == 200 else {
                throw ThreadCorpusError.notReachable(endpoint: endpoint.description, underlying: "health returned \((response as? HTTPURLResponse)?.statusCode ?? -1)")
            }
            return try JSONDecoder().decode(ThreadHealth.self, from: data)
        } catch let error as ThreadCorpusError {
            throw error
        } catch {
            throw ThreadCorpusError.notReachable(endpoint: endpoint.description, underlying: error.localizedDescription)
        }
    }

    public func stats() async throws -> ThreadStats {
        let options = options(seconds: 10)
        let response = try await withClient { client in
            try await Thread_V1_ThreadUpdate.Client(wrapping: client).stats(Thread_V1_ThreadStatsRequest(), options: options)
        }
        return ThreadStats(
            documents: Int(response.documentCount), groups: Int(response.groupCount), owners: Int(response.ownerCount))
    }

    /// One search, so the node loads its MLX embedding model before the first Index call
    /// (the model loads lazily and the first request would otherwise pay for it).
    public func warmUpEmbedding(owner: String) async throws {
        var request = Thread_V1_ThreadSearchRequest()
        request.queryText = "raolm warm-up"
        request.ownerID = owner
        request.topK = 1
        let options = options(seconds: 600)
        _ = try await withClient { client in
            try await Thread_V1_ThreadQuery.Client(wrapping: client).search(request, options: options)
        }
    }

    // MARK: - Ingest

    public func index(
        _ documents: [CorpusDocument], slug: String, owner: String, group: String, groupLabel: String,
        batchSize: Int = 32, progress: (@Sendable (Int, Int) -> Void)? = nil
    ) async throws -> IndexReport {
        let started = Date()
        let requests = ThreadIndexRequests.make(
            documents, slug: slug, owner: owner, group: group, groupLabel: groupLabel, batchSize: batchSize)
        var done = 0
        let options = options(seconds: 600)
        for request in requests {
            do {
                let response = try await withClient { client in
                    try await Thread_V1_ThreadQuery.Client(wrapping: client).index(request, options: options)
                }
                guard response.success else {
                    throw ThreadCorpusError.indexRejected(message: "success=false for a batch of \(request.items.count)")
                }
            } catch let error as RPCError where error.code == .invalidArgument {
                throw ThreadCorpusError.indexRejected(message: error.message)
            }
            done += request.items.count
            progress?(done, documents.count)
        }
        return IndexReport(
            documents: documents.count, partitions: documents.reduce(0) { $0 + $1.partitions.count },
            batches: requests.count, seconds: Date().timeIntervalSince(started))
    }

    /// Document ids ExportCorpus currently returns for (owner, group, prefix).
    public func exportedIDs(owner: String, group: String, prefix: String) async throws -> Set<String> {
        var ids = Set<String>()
        var after = ""
        while true {
            let page = try await exportPage(owner: owner, group: group, prefix: prefix, after: after, limit: 500, includeEmbeddings: false)
            for document in page.documents { ids.insert(document.id) }
            guard page.hasMore_p, let last = page.documents.last else { break }
            after = last.id
        }
        return ids
    }

    /// Polls until every expected document id is exported (Index is asynchronous).
    public func waitUntilIndexed(
        expected: Set<String>, owner: String, group: String, prefix: String, timeout: TimeInterval = 300,
        poll: TimeInterval = 0.5
    ) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        var missing = expected
        while Date() < deadline {
            let stats = try await stats()
            if stats.documents >= expected.count {
                missing = expected.subtracting(try await exportedIDs(owner: owner, group: group, prefix: prefix))
                if missing.isEmpty { return }
            }
            try await Task.sleep(nanoseconds: UInt64(poll * 1_000_000_000))
        }
        throw ThreadCorpusError.indexTimeout(missing: missing.sorted())
    }

    // MARK: - Export

    private func exportPage(
        owner: String, group: String, prefix: String, after: String, limit: Int, includeEmbeddings: Bool
    ) async throws -> Thread_V1_ThreadExportCorpusResponse {
        var request = Thread_V1_ThreadExportCorpusRequest()
        request.ownerID = owner
        request.groupIds = [group]
        request.documentIDPrefix = prefix
        request.afterID = after
        request.limit = Int32(limit)
        request.includeEmbeddings = includeEmbeddings
        let options = options(seconds: 120)
        return try await withClient { client in
            try await Thread_V1_ThreadLibrary.Client(wrapping: client).exportCorpus(request, options: options)
        }
    }

    /// The whole corpus for (owner, group, prefix), exactly as the Thread stores it.
    public func exportCorpus(
        owner: String, group: String, prefix: String, slug: String?, pageSize: Int = 200
    ) async throws -> CorpusSnapshot {
        var documents: [SnapshotDocument] = []
        var after = ""
        while true {
            let page = try await exportPage(owner: owner, group: group, prefix: prefix, after: after, limit: pageSize, includeEmbeddings: true)
            for content in page.documents {
                let document = Self.snapshotDocument(content)
                if let slug {
                    for partition in document.partitions {
                        let expected = DocumentID.partitionURL(slug: slug, documentID: document.id, index: partition.index)
                        if let url = partition.url, url != expected {
                            throw ThreadCorpusError.partitionAddressMismatch(
                                documentID: document.id, index: partition.index, expected: expected, got: url)
                        }
                    }
                }
                documents.append(document)
            }
            guard page.hasMore_p, let last = page.documents.last else { break }
            after = last.id
        }
        return CorpusSnapshot(
            documents: documents, slug: slug, source: "thread", threadID: endpoint.nodeID?.uuidString,
            threadHost: endpoint.host, threadGRPCPort: endpoint.grpcPort, owner: owner, group: group,
            documentIDPrefix: prefix)
    }

    public func documents(ids: [String], owner: String) async throws -> [SnapshotDocument] {
        guard !ids.isEmpty else { return [] }
        var request = Thread_V1_ThreadDocumentsRequest()
        request.ownerID = owner
        request.documentIds = ids
        request.includeEmbeddings = true
        let options = options(seconds: 60)
        let response = try await withClient { client in
            try await Thread_V1_ThreadLibrary.Client(wrapping: client).documents(request, options: options)
        }
        return response.documents.map(Self.snapshotDocument)
    }

    @discardableResult
    public func remove(ids: [String], owner: String) async throws -> Int {
        var request = Thread_V1_ThreadRemoveRequest()
        request.ownerID = owner
        request.documentIds = ids
        let options = options(seconds: 60)
        let response = try await withClient { client in
            try await Thread_V1_ThreadQuery.Client(wrapping: client).remove(request, options: options)
        }
        return Int(response.removedCount)
    }

    static func snapshotDocument(_ content: Thread_V1_ThreadDocumentContent) -> SnapshotDocument {
        let described = content.partitions.count == content.texts.count
        let partitions = content.texts.enumerated().map { index, text in
            SnapshotPartition(
                index: index, text: text,
                url: described ? content.partitions[index].url : nil,
                threadPartitionID: described ? content.partitions[index].id : nil)
        }
        return SnapshotDocument(
            id: content.id, name: content.name, ownerID: content.ownerID, groupID: content.groupID,
            groupLabel: content.groupLabel, createdAt: content.createdAt, mediaType: content.mediaType,
            partitions: partitions)
    }
}

/// Reads live partition texts from a Thread for the citation verifier.
public actor ThreadCorpusReader: CorpusReading {
    private let client: ThreadCorpusClient
    private let owner: String
    private var cache: [String: [String]] = [:]

    public init(client: ThreadCorpusClient, owner: String) {
        self.client = client
        self.owner = owner
    }

    /// Fetches many documents in one call so later span checks hit the cache.
    public func prefetch(_ ids: [String]) async throws {
        let wanted = Array(Set(ids).subtracting(cache.keys)).sorted()
        guard !wanted.isEmpty else { return }
        var start = 0
        while start < wanted.count {
            let slice = Array(wanted[start..<min(start + 100, wanted.count)])
            start += 100
            for document in try await client.documents(ids: slice, owner: owner) {
                cache[document.id] = document.partitions.sorted { $0.index < $1.index }.map(\.text)
            }
        }
    }

    public func partitionText(documentID: String, partitionIndex: Int) async throws -> String? {
        if cache[documentID] == nil {
            let documents = try await client.documents(ids: [documentID], owner: owner)
            cache[documentID] = documents.first?.partitions.sorted { $0.index < $1.index }.map(\.text) ?? []
        }
        guard let partitions = cache[documentID], partitions.indices.contains(partitionIndex) else { return nil }
        return partitions[partitionIndex]
    }

    public func invalidate() {
        cache.removeAll()
    }
}
