//
//  BraidLayout.swift
//  RaoLMBraid
//
//  WHAT: Where a braid keeps everything, and the records it keeps there: the shared
//        vocabulary, and per Thread node its Thread's storage, its snapshots, its versions
//        (each a standard run directory plus version.json) and which version is live.
//
//    <data root>/braid/
//      world.json                    the mock world the nodes were fed from (names, seed, shape)
//      vocabulary/<hash12>/          vocabulary.safetensors, vocabulary.json
//      umbrella/<hash12>/            a pack with a base model: the vocabulary's files, base.safetensors,
//                                    anchors, heldout.jsonl, pack.json
//      nodes/<name>/
//        node.json                   who the node is (ports, node id, processes)
//        thread-db/                  the node's Thread storage (a Thread in open mode)
//        offline-corpus/documents/   the stand-in for a Thread when running offline
//        snapshots/<hash12>/         every snapshot a version was built on
//        versions/v0001/             run.json, checkpoints/, provenance/, ledger/, version.json
//        live.json                   the version the umbrella uses
//        facts.jsonl, feed.json      what the mock feeder deposited (demo data only)
//        heldout.jsonl               documents of the node's world it was not fed, for its held-out loss
//        logs/                       node.log, thread.log
//
//  PIN:  Never the studio's own thread-db: a braid's Threads are exclusive to it.
//

import Foundation
import RaoLMCore

public struct BraidLayout: Sendable, Equatable {
    public let root: URL

    public init(root: URL) {
        self.root = root.standardizedFileURL
    }

    public init(dataRoot: DataRoot) {
        self.init(root: dataRoot.url.appendingPathComponent("braid", isDirectory: true))
    }

    public var vocabularies: URL { root.appendingPathComponent("vocabulary", isDirectory: true) }
    public func vocabulary(sha256: String) -> URL {
        vocabularies.appendingPathComponent(String(sha256.prefix(12)), isDirectory: true)
    }
    /// Umbrella packs with a base model; a pack that is a vocabulary alone lives under `vocabularies`.
    public var packs: URL { root.appendingPathComponent("umbrella", isDirectory: true) }
    public func pack(sha256: String) -> URL {
        packs.appendingPathComponent(String(sha256.prefix(12)), isDirectory: true)
    }
    public var nodes: URL { root.appendingPathComponent("nodes", isDirectory: true) }
    public func node(_ name: String) -> NodeLayout { NodeLayout(directory: nodes.appendingPathComponent(name, isDirectory: true)) }
    public var session: URL { root.appendingPathComponent("braid.json") }
    public var world: URL { root.appendingPathComponent("world.json") }
    public var logs: URL { root.appendingPathComponent("logs", isDirectory: true) }

    /// Node names are slugs: they prefix document ids and name directories.
    public static func isValidName(_ name: String) -> Bool {
        name.range(of: #"^[a-z][a-z0-9-]{1,23}$"#, options: .regularExpression) != nil
    }
}

public struct NodeLayout: Sendable, Equatable {
    public let directory: URL

    public var record: URL { directory.appendingPathComponent("node.json") }
    public var threadDB: URL { directory.appendingPathComponent("thread-db", isDirectory: true) }
    public var offlineCorpus: URL { directory.appendingPathComponent("offline-corpus", isDirectory: true) }
    public var snapshots: URL { directory.appendingPathComponent("snapshots", isDirectory: true) }
    public func snapshot(hash: String) -> URL {
        snapshots.appendingPathComponent(String(hash.prefix(12)), isDirectory: true)
    }
    public var versions: URL { directory.appendingPathComponent("versions", isDirectory: true) }
    public func version(_ number: Int) -> URL {
        versions.appendingPathComponent(String(format: "v%04d", number), isDirectory: true)
    }
    public var live: URL { directory.appendingPathComponent("live.json") }
    public var facts: URL { directory.appendingPathComponent("facts.jsonl") }
    public var heldOut: URL { directory.appendingPathComponent("heldout.jsonl") }
    public var feed: URL { directory.appendingPathComponent("feed.json") }
    public var offlineID: URL { directory.appendingPathComponent("offline-node-id") }
    public var logs: URL { directory.appendingPathComponent("logs", isDirectory: true) }
    public var nodeLog: URL { logs.appendingPathComponent("node.log") }
    public var threadLog: URL { logs.appendingPathComponent("thread.log") }

    /// Version numbers already on disk, ascending.
    public func versionNumbers() -> [Int] {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: versions.path)) ?? []
        return names.compactMap { name in name.hasPrefix("v") ? Int(name.dropFirst()) : nil }.sorted()
    }
}

// MARK: - Records

/// How a node was updated to reach a version.
public enum UpdateKind: String, Codable, Sendable {
    /// The live weights re-keyed over the new snapshot: new text is citable at once, removed text no longer.
    case reindex
    /// The blocks trained on the snapshot, from the live weights or from fresh blocks.
    case train
}

public struct GateResult: Codable, Sendable, Equatable {
    public var name: String
    public var passed: Bool
    public var detail: String

    public init(_ name: String, _ passed: Bool, _ detail: String) {
        self.name = name
        self.passed = passed
        self.detail = detail
    }
}

/// version.json: why a version exists and whether it went live.
public struct NodeVersion: Codable, Sendable, Equatable {
    public var version: Int
    public var parent: Int?
    public var kind: UpdateKind
    public var snapshotBefore: String?
    public var snapshotAfter: String
    public var snapshotPath: String
    public var change: SnapshotChange
    public var vocabularySHA256: String
    public var checkpointSHA256: String
    public var indexSHA256: String
    /// The indexed epoch inside the version's run directory.
    public var epoch: Int
    public var epochsTrained: Int
    public var memorised: Float
    public var evalLoss: Float
    public var documents: Int
    public var partitions: Int
    public var tokens: Int
    public var gates: [GateResult]
    public var promoted: Bool
    public var seconds: Double
    public var createdAt: Date
    /// The umbrella pack when it has a trunk.
    public var packSHA256: String?
    /// Mean loss on unfed documents in the Thread's own voice, and on the pack's commons sample.
    public var heldOutLoss: Float?
    public var commonsLoss: Float?
    /// λ and τ the version set from its corpus's self-trajectory, when it did.
    public var calibration: StrandCalibration?

    public static let fileName = "version.json"

    public var passed: Bool { gates.allSatisfy(\.passed) }
}

/// live.json.
public struct LivePointer: Codable, Sendable, Equatable {
    public var version: Int
    public var promotedAt: Date
}

/// node.json: written by the node process when it comes up.
public struct NodeRecord: Codable, Sendable, Equatable {
    public var name: String
    public var label: String
    public var pid: Int32
    public var threadPID: Int32?
    public var threadID: String?
    public var httpPort: Int?
    public var grpcPort: Int?
    public var offline: Bool
    public var startedAt: Date

    public static func load(_ layout: NodeLayout) -> NodeRecord? {
        try? JSONCoding.read(NodeRecord.self, from: layout.record)
    }
}

/// feed.json: which of the mock world's documents the feeder deposited into a node.
public struct FeedState: Codable, Sendable, Equatable {
    public var deposited: [String] = []
    public var withdrawn: [String] = []

    public init() {}

    public var present: [String] { deposited.filter { !withdrawn.contains($0) } }

    public static func load(_ layout: NodeLayout) -> FeedState {
        (try? JSONCoding.read(FeedState.self, from: layout.feed)) ?? FeedState()
    }
}
