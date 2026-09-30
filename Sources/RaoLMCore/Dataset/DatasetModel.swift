//
//  DatasetModel.swift
//  RaoLMCore
//
//  WHAT: The records of a braid dataset: three Threads' corpora written in three voices
//        (Ambient's readings, Craft's work, Veil's archive), the links where one Thread's
//        entity is retold in another's words, and what each document simulates about its origin.
//  PIN:  Every document is synthetic. `simulatedOrigin` borrows Rao Verified's vocabulary so
//        an attestation test can be written against it, but no document carries a Rao Verified
//        record or a seal: nothing here claims a person read, wrote or said anything.
//

import Foundation

/// A Thread's voice: which app's words its documents are written in.
public enum DatasetVoice: String, Codable, Sendable, CaseIterable {
    /// A reading companion: notes on what a person read, and their conversations with Mary.
    case ambient
    /// An on-device coding agent: session logs, release notes, postmortems, reviews.
    case craft
    /// An attribution layer for images: catalogue entries, attribution reports, wall text.
    case veil

    public var label: String { rawValue.prefix(1).uppercased() + rawValue.dropFirst() }
}

/// What a document is about. Each type belongs to one voice's world and states three facts.
public enum DatasetEntityType: String, Codable, Sendable, CaseIterable {
    case town, researcher, festival
    case library, service, incident
    case artwork, artist, collection

    /// The Thread whose own documents describe entities of this type.
    public var home: DatasetVoice {
        switch self {
        case .town, .researcher, .festival: return .ambient
        case .library, .service, .incident: return .craft
        case .artwork, .artist, .collection: return .veil
        }
    }

    public var facts: [FactKind] {
        switch self {
        case .town: return [.townFounded, .townPopulation, .townMayor]
        case .researcher: return [.researcherBorn, .researcherMentor, .researcherBook]
        case .festival: return [.festivalFirst, .festivalVisitors, .festivalFounder]
        case .library: return [.libraryAuthor, .libraryVersion, .libraryPort]
        case .service: return [.serviceOwner, .serviceLatency, .serviceLaunched]
        case .incident: return [.incidentMinutes, .incidentResponder, .incidentFixVersion]
        case .artwork: return [.artworkArtist, .artworkYear, .artworkWidth]
        case .artist: return [.artistBorn, .artistStudio, .artistTeacher]
        case .collection: return [.collectionOpened, .collectionWorks, .collectionCurator]
        }
    }

    public static func types(of voice: DatasetVoice) -> [DatasetEntityType] { allCases.filter { $0.home == voice } }
}

/// Where a document's words would have come from, in Rao Verified's vocabulary. Simulated.
public enum SimulatedOrigin: String, Codable, Sendable, CaseIterable {
    case read, written, spoken, imported, generated, system

    /// Whether a real capture with this origin could ever be verified (a live origin).
    public var live: Bool { self == .read || self == .written || self == .spoken }
}

/// How one Thread's entity reaches another Thread. None of them copies text.
public enum CrossKind: String, Codable, Sendable, CaseIterable {
    /// Every fact of the entity, retold in the other voice.
    case paraphrase
    /// One sentence quoted from the source with a small edit, and one more fact in the other voice.
    case excerpt
    /// One or two facts, briefly, in the other voice.
    case summary
    /// A paraphrase in which one fact's value disagrees with the source.
    case variant
    /// A different entity, of the other voice's own kind, that happens to share the name.
    case homonym
}

public struct DatasetDocumentRef: Codable, Sendable, Equatable, Hashable {
    public var node: String
    public var documentID: String

    public init(node: String, documentID: String) {
        self.node = node
        self.documentID = documentID
    }
}

/// One fact both sides of a link state (the same kind; the same answer unless the link is a variant).
public struct DatasetSharedFact: Codable, Sendable, Equatable {
    public var kind: FactKind
    public var sourceFact: String
    public var targetFact: String
    public var sourceAnswer: String
    public var targetAnswer: String

    public var agrees: Bool { sourceAnswer == targetAnswer }
}

/// How much text two documents share, in words (lowercased letters and digits).
public struct DatasetOverlap: Codable, Sendable, Equatable {
    public var sourceWords: Int
    public var targetWords: Int
    /// |shared word types| / |all word types|.
    public var jaccard: Float
    /// The longest run of consecutive words both documents contain.
    public var longestCommonRun: Int

    public static func measure(_ a: String, _ b: String) -> DatasetOverlap {
        let x = words(a)
        let y = words(b)
        let sx = Set(x)
        let sy = Set(y)
        let union = sx.union(sy).count
        var best = 0
        if !x.isEmpty, !y.isEmpty {
            var previous = [Int](repeating: 0, count: y.count + 1)
            for i in 1...x.count {
                var current = [Int](repeating: 0, count: y.count + 1)
                for j in 1...y.count where x[i - 1] == y[j - 1] {
                    current[j] = previous[j - 1] + 1
                    best = max(best, current[j])
                }
                previous = current
            }
        }
        return DatasetOverlap(sourceWords: x.count, targetWords: y.count,
                              jaccard: union > 0 ? Float(sx.intersection(sy).count) / Float(union) : 0, longestCommonRun: best)
    }

    public static func words(_ text: String) -> [String] {
        text.lowercased().split { !($0.isLetter || $0.isNumber) }.map(String.init)
    }
}

public struct DatasetCrosslink: Codable, Sendable, Equatable {
    public var id: String
    public var kind: CrossKind
    public var subject: String
    public var sourceType: DatasetEntityType
    public var targetType: DatasetEntityType
    public var source: DatasetDocumentRef
    public var target: DatasetDocumentRef
    /// Empty for a homonym: the two entities share a name, not facts.
    public var facts: [DatasetSharedFact]
    public var overlap: DatasetOverlap
}

/// One document's place in the dataset.
public struct DatasetDocumentMeta: Codable, Sendable, Equatable {
    public var id: String
    public var node: String
    public var kind: DocumentKind
    public var entityType: DatasetEntityType
    public var subject: String
    /// Whether the entity belongs to this Thread's own world (a homonym does).
    public var home: Bool
    /// The link this document is the target of, if any.
    public var crosslink: String?
    public var simulatedOrigin: SimulatedOrigin
    public var synthetic: Bool
    /// Where the document sits in feeding order, 0 to 1. A link's target sits just after its
    /// source's rank, so feeding the first part of every node keeps both sides of a link.
    public var rank: Double
}

/// How big a dataset is.
public struct DatasetSpec: Codable, Sendable, Equatable {
    public var name: String
    public var seed: UInt64
    /// Home documents per entity type (three types per Thread).
    public var perType: Int
    /// Per ordered pair of Threads, how many entities cross in each way.
    public var paraphrase: Int
    public var excerpt: Int
    public var summary: Int
    public var variant: Int
    /// Per ordered pair whose entity types can share a name (ambient ↔ craft, ambient ↔ veil).
    public var homonym: Int

    public init(name: String = "braid-cross-v1", seed: UInt64 = 42, perType: Int = 180, paraphrase: Int = 24, excerpt: Int = 20,
                summary: Int = 24, variant: Int = 8, homonym: Int = 8) {
        self.name = name
        self.seed = seed
        self.perType = perType
        self.paraphrase = paraphrase
        self.excerpt = excerpt
        self.summary = summary
        self.variant = variant
        self.homonym = homonym
    }

    public func count(_ kind: CrossKind) -> Int {
        switch kind {
        case .paraphrase: return paraphrase
        case .excerpt: return excerpt
        case .summary: return summary
        case .variant: return variant
        case .homonym: return homonym
        }
    }
}

public struct DatasetNodeSummary: Codable, Sendable, Equatable {
    public var name: String
    public var documents: Int
    public var home: Int
    /// Documents retelling another Thread's entity (paraphrase, excerpt, summary, variant).
    public var crossed: Int
    public var homonyms: Int
    public var partitions: Int
    public var facts: Int
    public var words: Int
    public var corpusHash: String
    /// Documents per simulated origin.
    public var origins: [String: Int]
}

public struct DatasetManifest: Codable, Sendable, Equatable {
    public var schemaVersion: Int
    public var generator: String
    public var generatorVersion: Int
    public var spec: DatasetSpec
    public var nodes: [DatasetNodeSummary]
    /// Links per kind.
    public var crosslinks: [String: Int]
    /// SHA-256 over every node's corpus hash and every link, in order: names this dataset.
    public var datasetHash: String

    public static let fileName = "manifest.json"
    public var names: [String] { nodes.map(\.name) }
}

public enum DatasetError: Error, CustomStringConvertible, Equatable {
    case invalid([String])
    case exhausted(String)
    case notADataset(String)

    public var description: String {
        switch self {
        case .invalid(let problems): return "the dataset failed validation:\n  " + problems.prefix(20).joined(separator: "\n  ")
        case .exhausted(let what): return "ran out of unique \(what); lower the dataset's size"
        case .notADataset(let path): return "no dataset manifest.json in \(path)"
        }
    }
}
