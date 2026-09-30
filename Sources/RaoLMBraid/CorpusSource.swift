//
//  CorpusSource.swift
//  RaoLMBraid
//
//  WHAT: Where a node's corpus lives, seen from both sides: a node exports it (a snapshot to
//        build a version on, and a reader to verify spans against), and the mock feeder
//        deposits into it and withdraws from it the way a user of that Thread would.
//  IN:   A Thread node over gRPC, or — offline — a directory of document files that stands
//        in for one, with a node id of its own so citations still name a Thread.
//  PIN:  Withdrawal names documents one by one: ThreadQuery.Remove with an empty id list
//        removes every document the owner has, so an empty list is refused here.
//

import Foundation
import RaoLMCore
import RaoLMThread

public protocol CorpusSource: Sendable {
    var threadID: String? { get }
    /// The corpus as the node stores it now.
    func export() async throws -> CorpusSnapshot
    /// Live partition texts, for the citation verifier.
    func reader() -> CorpusReading
    func deposit(_ documents: [CorpusDocument]) async throws
    func withdraw(_ documentIDs: [String]) async throws
}

public enum CorpusSourceError: Error, CustomStringConvertible {
    case emptyWithdrawal
    case unreadable(String)

    public var description: String {
        switch self {
        case .emptyWithdrawal: return "refusing to withdraw an empty list of documents (the Thread would remove them all)"
        case .unreadable(let path): return "cannot read the offline document \(path)"
        }
    }
}

/// A node's Thread, over its gRPC port.
public struct ThreadCorpusSource: CorpusSource {
    public let endpoint: ThreadEndpoint
    public let slug: String
    public let owner: String

    public init(endpoint: ThreadEndpoint, slug: String, owner: String) {
        self.endpoint = endpoint
        self.slug = slug
        self.owner = owner
    }

    public var group: String { "raolm-\(slug)" }
    public var prefix: String { DocumentID.prefix(slug: slug) }
    public var threadID: String? { endpoint.nodeID?.uuidString }

    public func export() async throws -> CorpusSnapshot {
        try await ThreadCorpusClient(endpoint: endpoint).exportCorpus(owner: owner, group: group, prefix: prefix, slug: slug)
    }

    public func reader() -> CorpusReading {
        ThreadCorpusReader(client: ThreadCorpusClient(endpoint: endpoint), owner: owner)
    }

    public func deposit(_ documents: [CorpusDocument]) async throws {
        guard !documents.isEmpty else { return }
        let client = ThreadCorpusClient(endpoint: endpoint)
        _ = try await client.index(documents, slug: slug, owner: owner, group: group, groupLabel: "RaoLM braid · \(slug)", batchSize: 16)
        try await client.waitUntilIndexed(expected: Set(documents.map(\.id)), owner: owner, group: group, prefix: prefix, timeout: 240)
    }

    public func withdraw(_ documentIDs: [String]) async throws {
        guard !documentIDs.isEmpty else { throw CorpusSourceError.emptyWithdrawal }
        let client = ThreadCorpusClient(endpoint: endpoint)
        try await client.remove(ids: documentIDs, owner: owner)
        // Removal is queued: wait until the export no longer returns them.
        let gone = Set(documentIDs)
        let deadline = Date().addingTimeInterval(60)
        while Date() < deadline {
            let present = try await client.exportedIDs(owner: owner, group: group, prefix: prefix)
            if present.isDisjoint(with: gone) { return }
            try await Task.sleep(nanoseconds: 250_000_000)
        }
        throw ThreadCorpusError.indexTimeout(missing: documentIDs)
    }
}

/// The offline stand-in for a Thread: one JSON file per document.
public struct DirectoryCorpusSource: CorpusSource {
    public let directory: URL
    public let slug: String
    public let owner: String
    public let threadID: String?

    public init(directory: URL, slug: String, owner: String, threadID: String?) {
        self.directory = directory
        self.slug = slug
        self.owner = owner
        self.threadID = threadID
    }

    var documentsDirectory: URL { directory.appendingPathComponent("documents", isDirectory: true) }

    /// A stable node id for an offline node, created once.
    public static func nodeID(_ layout: NodeLayout) throws -> String {
        if let text = try? String(contentsOf: layout.offlineID, encoding: .utf8),
           let id = UUID(uuidString: text.trimmingCharacters(in: .whitespacesAndNewlines)) {
            return id.uuidString
        }
        let id = UUID().uuidString
        try FileManager.default.createDirectory(at: layout.directory, withIntermediateDirectories: true)
        try Data(id.utf8).write(to: layout.offlineID, options: .atomic)
        return id
    }

    func documents() throws -> [CorpusDocument] {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: documentsDirectory.path)) ?? []
        // "._<name>" files are macOS's extended-attribute sidecars on exFAT and FAT drives, not documents.
        return try names.filter { $0.hasSuffix(".json") && !$0.hasPrefix("._") }.sorted().map { name in
            let url = documentsDirectory.appendingPathComponent(name)
            do { return try JSONCoding.read(CorpusDocument.self, from: url) } catch { throw CorpusSourceError.unreadable(url.path) }
        }
    }

    /// When a document entered this offline Thread: its file's creation time, in milliseconds (0 if unknown).
    func createdAt(_ id: String) -> Int64 {
        let url = documentsDirectory.appendingPathComponent(id + ".json")
        guard let date = (try? FileManager.default.attributesOfItem(atPath: url.path))?[.creationDate] as? Date else { return 0 }
        return Int64((date.timeIntervalSince1970 * 1000).rounded())
    }

    public func export() async throws -> CorpusSnapshot {
        let documents = try documents().map { document in
            SnapshotDocument(
                id: document.id, name: document.name, ownerID: owner, groupID: "raolm-\(slug)", groupLabel: "RaoLM braid · \(slug)",
                createdAt: createdAt(document.id), mediaType: "text",
                partitions: document.partitions.map {
                    SnapshotPartition(index: $0.index, text: $0.text, url: $0.url, threadPartitionID: nil)
                })
        }
        return CorpusSnapshot(
            documents: documents, slug: slug, source: "offline", threadID: threadID, threadHost: nil, threadGRPCPort: nil,
            owner: owner, group: "raolm-\(slug)", documentIDPrefix: DocumentID.prefix(slug: slug), exportedAt: .wholeSecond())
    }

    public func reader() -> CorpusReading { DirectoryCorpusReader(source: self) }

    public func deposit(_ documents: [CorpusDocument]) async throws {
        try FileManager.default.createDirectory(at: documentsDirectory, withIntermediateDirectories: true)
        for document in documents {
            try JSONCoding.write(document, to: documentsDirectory.appendingPathComponent("\(document.id).json"))
        }
    }

    public func withdraw(_ documentIDs: [String]) async throws {
        guard !documentIDs.isEmpty else { throw CorpusSourceError.emptyWithdrawal }
        for id in documentIDs {
            try? FileManager.default.removeItem(at: documentsDirectory.appendingPathComponent("\(id).json"))
        }
    }
}

/// Reads partitions from the offline directory as it is now, so a withdrawn document reads as missing.
public struct DirectoryCorpusReader: CorpusReading {
    public let source: DirectoryCorpusSource

    public func partitionText(documentID: String, partitionIndex: Int) async throws -> String? {
        let url = source.documentsDirectory.appendingPathComponent("\(documentID).json")
        guard let document = try? JSONCoding.read(CorpusDocument.self, from: url),
              let partition = document.partitions.first(where: { $0.index == partitionIndex }) else { return nil }
        return partition.text
    }
}
