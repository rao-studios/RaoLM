//
//  CorpusModels.swift
//  RaoLMCore
//
//  WHAT: The corpus as RaoLM generates it and deposits it into a Thread: documents,
//        their partitions (Thread's unit of storage), and the facts each one states.
//  PIN:  A partition is the citation unit. Its address is (documentID, partitionIndex),
//        never Thread's partition id (which hashes the embedding and changes if the
//        text is re-embedded).
//

import Foundation

public enum DocumentKind: String, Codable, Sendable, CaseIterable {
    case landmark, biography, expedition, council, recipe
}

public enum FactKind: String, Codable, Sendable, CaseIterable {
    // landmark
    case completedYear, architect, height, restoredYear
    // biography
    case birthYear, birthplace, mentor
    // expedition
    case departureYear, leader, distance, members
    // council
    case foundedYear, firstSpeaker, seats
    // recipe
    case creator, grams, bakeMinutes
}

/// One fact a document states, located precisely inside its partition.
///
/// Offsets are UTF-8 byte offsets into the partition text (the synthetic corpus is
/// ASCII, so they are also character offsets). `answer` begins with the space that
/// separates it from `prompt`, so the prompt/answer boundary is also a token boundary
/// under SmolLM2's byte-level BPE.
public struct Fact: Codable, Sendable, Equatable, Identifiable {
    public var id: String
    public var kind: FactKind
    public var documentID: String
    public var partitionIndex: Int
    /// The entity the fact is about, as it appears in the text ("Kestrel Bridge").
    public var subject: String
    public var prompt: String
    public var answer: String
    public var sentence: String
    public var sentenceStart: Int
    /// Start of the sentence before this one in the same partition (or `sentenceStart`).
    public var contextStart: Int
    /// Offset of the space before the answer value.
    public var answerStart: Int
    public var answerEnd: Int
    /// Prompts for the same fact that do not occur anywhere in the corpus.
    public var paraphrases: [String]
    /// The same template about an entity that exists nowhere in the corpus.
    public var negativePrompt: String

    public init(
        id: String, kind: FactKind, documentID: String, partitionIndex: Int, subject: String,
        prompt: String, answer: String, sentence: String, sentenceStart: Int, contextStart: Int,
        answerStart: Int, answerEnd: Int, paraphrases: [String], negativePrompt: String
    ) {
        self.id = id
        self.kind = kind
        self.documentID = documentID
        self.partitionIndex = partitionIndex
        self.subject = subject
        self.prompt = prompt
        self.answer = answer
        self.sentence = sentence
        self.sentenceStart = sentenceStart
        self.contextStart = contextStart
        self.answerStart = answerStart
        self.answerEnd = answerEnd
        self.paraphrases = paraphrases
        self.negativePrompt = negativePrompt
    }
}

public struct CorpusPartition: Codable, Sendable, Equatable {
    public var index: Int
    public var text: String
    public var textSHA256: String
    /// The address RaoLM supplies to Thread for this partition.
    public var url: String

    public init(index: Int, text: String, textSHA256: String, url: String) {
        self.index = index
        self.text = text
        self.textSHA256 = textSHA256
        self.url = url
    }
}

public struct CorpusDocument: Codable, Sendable, Equatable, Identifiable {
    public var id: String
    public var name: String
    public var kind: DocumentKind
    public var subject: String
    public var partitions: [CorpusPartition]
    public var facts: [Fact]
    /// SHA-256 of the partitions joined by a blank line (the document's canonical text).
    public var textSHA256: String

    public init(
        id: String, name: String, kind: DocumentKind, subject: String,
        partitions: [CorpusPartition], facts: [Fact], textSHA256: String
    ) {
        self.id = id
        self.name = name
        self.kind = kind
        self.subject = subject
        self.partitions = partitions
        self.facts = facts
        self.textSHA256 = textSHA256
    }

    public var text: String { partitions.map(\.text).joined(separator: "\n\n") }
}

public struct CorpusManifest: Codable, Sendable, Equatable {
    public var schemaVersion: Int
    public var slug: String
    public var generator: String
    public var generatorVersion: Int
    public var seed: UInt64
    public var documentCount: Int
    public var partitionCount: Int
    public var factCount: Int
    public var chunkMaxChars: Int
    public var chunkMinChars: Int
    /// In generation order.
    public var documentIDs: [String]
    public var corpusHash: String

    public init(
        slug: String, generator: String, generatorVersion: Int, seed: UInt64, documentCount: Int,
        partitionCount: Int, factCount: Int, chunkMaxChars: Int, chunkMinChars: Int,
        documentIDs: [String], corpusHash: String
    ) {
        self.schemaVersion = 1
        self.slug = slug
        self.generator = generator
        self.generatorVersion = generatorVersion
        self.seed = seed
        self.documentCount = documentCount
        self.partitionCount = partitionCount
        self.factCount = factCount
        self.chunkMaxChars = chunkMaxChars
        self.chunkMinChars = chunkMinChars
        self.documentIDs = documentIDs
        self.corpusHash = corpusHash
    }
}

public struct GeneratedCorpus: Sendable, Equatable {
    public var manifest: CorpusManifest
    public var documents: [CorpusDocument]

    public init(manifest: CorpusManifest, documents: [CorpusDocument]) {
        self.manifest = manifest
        self.documents = documents
    }

    public var facts: [Fact] { documents.flatMap(\.facts) }

    public var corpusEntries: [ContentHash.CorpusEntry] {
        documents.flatMap { document in
            document.partitions.map {
                ContentHash.CorpusEntry(documentID: document.id, partitionIndex: $0.index, textSHA256: $0.textSHA256)
            }
        }
    }
}

/// Where a cited token lives: the Thread document, the partition inside it, and the
/// token offset inside that partition's own tokenization.
public struct SourceAddress: Codable, Sendable, Hashable {
    public var threadID: String?
    public var documentID: String
    public var partitionIndex: Int
    public var tokenOffset: Int
    public var partitionURL: String?
    public var threadPartitionID: String?

    public init(
        threadID: String? = nil, documentID: String, partitionIndex: Int, tokenOffset: Int,
        partitionURL: String? = nil, threadPartitionID: String? = nil
    ) {
        self.threadID = threadID
        self.documentID = documentID
        self.partitionIndex = partitionIndex
        self.tokenOffset = tokenOffset
        self.partitionURL = partitionURL
        self.threadPartitionID = threadPartitionID
    }
}
