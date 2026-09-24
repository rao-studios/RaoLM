//
//  CorpusSnapshot.swift
//  RaoLMCore
//
//  WHAT: The corpus exactly as a Thread returned it: document ids, names, groups,
//        partition texts in stored order, and Thread's own partition ids and urls.
//  PIN:  Training only ever consumes a snapshot, and the snapshot's corpus hash goes
//        into the run manifest, so a model can prove which Thread state it saw.
//

import Foundation

public struct SnapshotPartition: Codable, Sendable, Equatable {
    public var index: Int
    public var text: String
    public var textSHA256: String
    public var url: String?
    public var threadPartitionID: String?

    public init(index: Int, text: String, url: String? = nil, threadPartitionID: String? = nil) {
        self.index = index
        self.text = text
        self.textSHA256 = ContentHash.sha256Hex(text)
        self.url = url
        self.threadPartitionID = threadPartitionID
    }
}

public struct SnapshotDocument: Codable, Sendable, Equatable, Identifiable {
    public var id: String
    public var name: String
    public var ownerID: String
    public var groupID: String
    public var groupLabel: String
    public var createdAt: Int64
    public var mediaType: String
    public var partitions: [SnapshotPartition]

    public init(
        id: String, name: String, ownerID: String, groupID: String, groupLabel: String,
        createdAt: Int64, mediaType: String, partitions: [SnapshotPartition]
    ) {
        self.id = id
        self.name = name
        self.ownerID = ownerID
        self.groupID = groupID
        self.groupLabel = groupLabel
        self.createdAt = createdAt
        self.mediaType = mediaType
        self.partitions = partitions
    }
}

public struct CorpusSnapshot: Codable, Sendable, Equatable {
    public var schemaVersion: Int
    public var corpusHash: String
    public var slug: String?
    public var source: String
    public var threadID: String?
    public var threadHost: String?
    public var threadGRPCPort: Int?
    public var owner: String
    public var group: String
    public var documentIDPrefix: String
    public var exportedAt: Date
    public var documentCount: Int
    public var partitionCount: Int
    /// Sorted by document id (ExportCorpus order).
    public var documents: [SnapshotDocument]

    public init(
        documents: [SnapshotDocument], slug: String?, source: String, threadID: String?,
        threadHost: String?, threadGRPCPort: Int?, owner: String, group: String,
        documentIDPrefix: String, exportedAt: Date = Date()
    ) {
        let sorted = documents.sorted { $0.id < $1.id }
        self.schemaVersion = 1
        self.documents = sorted
        self.slug = slug
        self.source = source
        self.threadID = threadID
        self.threadHost = threadHost
        self.threadGRPCPort = threadGRPCPort
        self.owner = owner
        self.group = group
        self.documentIDPrefix = documentIDPrefix
        self.exportedAt = exportedAt
        self.documentCount = sorted.count
        self.partitionCount = sorted.reduce(0) { $0 + $1.partitions.count }
        self.corpusHash = ContentHash.corpusHash(sorted.flatMap { document in
            document.partitions.map {
                ContentHash.CorpusEntry(documentID: document.id, partitionIndex: $0.index, textSHA256: $0.textSHA256)
            }
        })
    }

    /// A snapshot built straight from a generated corpus, without a Thread. Tests and
    /// offline experiments use it; the demo always goes through a live Thread.
    public static func offline(_ corpus: GeneratedCorpus, owner: String = "offline", group: String = "offline") -> CorpusSnapshot {
        let documents = corpus.documents.map { document in
            SnapshotDocument(
                id: document.id, name: document.name, ownerID: owner, groupID: group, groupLabel: group,
                createdAt: 0, mediaType: "text",
                partitions: document.partitions.map {
                    SnapshotPartition(index: $0.index, text: $0.text, url: $0.url, threadPartitionID: nil)
                })
        }
        return CorpusSnapshot(
            documents: documents, slug: corpus.manifest.slug, source: "offline", threadID: nil,
            threadHost: nil, threadGRPCPort: nil, owner: owner, group: group,
            documentIDPrefix: DocumentID.prefix(slug: corpus.manifest.slug), exportedAt: Date(timeIntervalSince1970: 0))
    }

    public func document(id: String) -> SnapshotDocument? {
        documents.first { $0.id == id }
    }

    /// Human-readable disagreements between what Thread returned and what was generated.
    /// Empty means the Thread holds the corpus byte for byte.
    public func diff(against corpus: GeneratedCorpus) -> [String] {
        var problems: [String] = []
        let byID = Dictionary(uniqueKeysWithValues: documents.map { ($0.id, $0) })
        let generatedIDs = Set(corpus.documents.map(\.id))
        for document in corpus.documents {
            guard let stored = byID[document.id] else {
                problems.append("\(document.id): missing from the Thread export")
                continue
            }
            if stored.partitions.count != document.partitions.count {
                problems.append("\(document.id): \(stored.partitions.count) partitions stored, \(document.partitions.count) generated")
                continue
            }
            for (storedPartition, partition) in zip(stored.partitions, document.partitions) {
                if storedPartition.text != partition.text {
                    problems.append("\(document.id) partition \(partition.index): text differs")
                }
                if let url = storedPartition.url, url != partition.url {
                    problems.append("\(document.id) partition \(partition.index): url \(url) != \(partition.url)")
                }
            }
            if stored.name != document.name {
                problems.append("\(document.id): name '\(stored.name)' != '\(document.name)'")
            }
        }
        for document in documents where !generatedIDs.contains(document.id) {
            problems.append("\(document.id): in the Thread export but not in the generated corpus")
        }
        if problems.isEmpty, corpusHash != corpus.manifest.corpusHash {
            problems.append("corpus hash \(corpusHash) != generated \(corpus.manifest.corpusHash)")
        }
        return problems
    }

    public static let fileName = "snapshot.json"

    public func save(to directory: URL) throws {
        try JSONCoding.write(self, to: directory.appendingPathComponent(Self.fileName))
    }

    public static func load(from directory: URL) throws -> CorpusSnapshot {
        let url = directory.hasDirectoryPath || !directory.lastPathComponent.hasSuffix(".json")
            ? directory.appendingPathComponent(fileName) : directory
        return try JSONCoding.read(CorpusSnapshot.self, from: url)
    }
}
