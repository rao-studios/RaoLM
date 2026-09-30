//
//  CorpusReading.swift
//  RaoLMCore
//
//  WHAT: Where a verifier reads the live text of a cited partition from. The Thread
//        implementation lives in RaoLMThread; this in-memory one serves tests and
//        offline checks.
//

import Foundation

public protocol CorpusReading: Sendable {
    /// The stored text of one partition, or nil when the document or partition is gone.
    func partitionText(documentID: String, partitionIndex: Int) async throws -> String?
}

public struct InMemoryCorpusReader: CorpusReading {
    public let texts: [String: [String]]

    public init(texts: [String: [String]]) {
        self.texts = texts
    }

    public init(snapshot: CorpusSnapshot) {
        var texts: [String: [String]] = [:]
        for document in snapshot.documents {
            texts[document.id] = document.partitions.sorted { $0.index < $1.index }.map(\.text)
        }
        self.texts = texts
    }

    public func partitionText(documentID: String, partitionIndex: Int) async throws -> String? {
        guard let partitions = texts[documentID], partitions.indices.contains(partitionIndex) else {
            return nil
        }
        return partitions[partitionIndex]
    }
}
