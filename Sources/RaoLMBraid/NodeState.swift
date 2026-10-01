//
//  NodeState.swift
//  RaoLMBraid
//
//  WHAT: What a Thread node reports about itself while it runs: the stage of its update
//        ladder, its live and candidate versions, what it holds, how training is going, one
//        cell per partition (for the panel's database), and the gates of its last candidate.
//        Plus the pure gating functions: which update a change calls for, and whether a
//        candidate may replace the live version.
//  PIN:  Snapshots, not deltas: a node sends its whole state and a viewer replaces its copy,
//        so a dropped or coalesced update can never leave the panel inconsistent.
//

import Foundation
import RaoLMCore

public enum NodeStage: String, Codable, Sendable {
    case starting, empty, exporting, reindexing, training, indexing, gating, live, held, failed, stopped

    public var isBusy: Bool {
        switch self {
        case .starting, .exporting, .reindexing, .training, .indexing, .gating: return true
        default: return false
        }
    }
}

public struct LadderMark: Codable, Sendable, Equatable {
    public enum Status: String, Codable, Sendable { case pending, running, done, failed, skipped }

    public var name: String
    public var status: Status
    public var detail: String?

    public init(_ name: String, _ status: Status, _ detail: String? = nil) {
        self.name = name
        self.status = status
        self.detail = detail
    }
}

/// One partition of what the node holds: one pixel of the panel's database.
public struct PartitionCell: Codable, Sendable, Equatable {
    public enum State: String, Codable, Sendable {
        /// In the Thread, not yet in the live index.
        case pending
        /// In the live index: citable, not yet memorised by the weights.
        case indexed
        /// Memorised (≥ 95% of its positions below 0.1 nat).
        case learned
    }

    public var documentID: String
    /// The document's position in deposit order.
    public var document: Int
    public var partition: Int
    public var state: State
    /// Memorised fraction under the newest weights that were evaluated (nil before any).
    public var memorised: Float?
    /// The first words of the partition, for the ingest rain.
    public var glyphs: String

    public init(documentID: String, document: Int, partition: Int, state: State, memorised: Float?, glyphs: String) {
        self.documentID = documentID
        self.document = document
        self.partition = partition
        self.state = state
        self.memorised = memorised
        self.glyphs = glyphs
    }
}

public struct StrandState: Codable, Sendable, Equatable {
    public var name: String
    public var label: String
    public var threadID: String?
    public var pid: Int32?
    public var threadPID: Int32?
    public var httpPort: Int?
    public var grpcPort: Int?
    public var offline: Bool
    public var stage: NodeStage
    public var stageDetail: String?
    public var ladder: [LadderMark]
    public var liveVersion: Int?
    public var candidateVersion: Int?
    public var versions: Int
    public var documents: Int
    public var partitions: Int
    public var tokens: Int
    public var vocabularySHA256: String
    public var checkpointSHA256: String?
    public var indexSHA256: String?
    public var indexEntries: Int
    public var memorised: Float?
    public var candidateMemorised: Float?
    public var epoch: Int?
    public var epochs: Int?
    public var step: Int?
    public var steps: Int?
    public var losses: [Float]
    /// How much each block moved since the last evaluation, scaled to the most-moved block.
    public var blockActivity: [Float]
    public var factsLearned: Int?
    public var factsTotal: Int?
    public var cells: [PartitionCell]
    /// The candidate's own completion of the standing prompt (no retrieval), at its last evaluation.
    public var preview: String?
    public var lastChange: String?
    public var gates: [GateResult]
    /// Every version this node built, oldest first.
    public var history: [VersionMark]
    public var error: String?
    public var updatedAt: Date
    /// The umbrella pack the node mirrors, when it has a trunk, and the first block of the trunk.
    public var packSHA256: String?
    public var cut: Int?
    /// The live version's mean loss on unfed documents in the Thread's own voice, and on the
    /// pack's commons sample (nats per token).
    public var heldOutLoss: Float?
    public var commonsLoss: Float?

    public init(name: String, label: String, offline: Bool, vocabularySHA256: String, blocks: Int) {
        self.name = name
        self.label = label
        self.threadID = nil
        self.pid = nil
        self.threadPID = nil
        self.httpPort = nil
        self.grpcPort = nil
        self.offline = offline
        self.stage = .starting
        self.stageDetail = nil
        self.ladder = []
        self.liveVersion = nil
        self.candidateVersion = nil
        self.versions = 0
        self.documents = 0
        self.partitions = 0
        self.tokens = 0
        self.vocabularySHA256 = vocabularySHA256
        self.checkpointSHA256 = nil
        self.indexSHA256 = nil
        self.indexEntries = 0
        self.memorised = nil
        self.candidateMemorised = nil
        self.epoch = nil
        self.epochs = nil
        self.step = nil
        self.steps = nil
        self.losses = []
        self.blockActivity = Array(repeating: 0, count: blocks)
        self.factsLearned = nil
        self.factsTotal = nil
        self.cells = []
        self.preview = nil
        self.lastChange = nil
        self.gates = []
        self.history = []
        self.error = nil
        self.updatedAt = .wholeSecond()
    }

    // Recordings made before `history` existed decode with an empty one.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        name = try c.decode(String.self, forKey: .name)
        label = try c.decode(String.self, forKey: .label)
        threadID = try c.decodeIfPresent(String.self, forKey: .threadID)
        pid = try c.decodeIfPresent(Int32.self, forKey: .pid)
        threadPID = try c.decodeIfPresent(Int32.self, forKey: .threadPID)
        httpPort = try c.decodeIfPresent(Int.self, forKey: .httpPort)
        grpcPort = try c.decodeIfPresent(Int.self, forKey: .grpcPort)
        offline = try c.decode(Bool.self, forKey: .offline)
        stage = try c.decode(NodeStage.self, forKey: .stage)
        stageDetail = try c.decodeIfPresent(String.self, forKey: .stageDetail)
        ladder = try c.decode([LadderMark].self, forKey: .ladder)
        liveVersion = try c.decodeIfPresent(Int.self, forKey: .liveVersion)
        candidateVersion = try c.decodeIfPresent(Int.self, forKey: .candidateVersion)
        versions = try c.decode(Int.self, forKey: .versions)
        documents = try c.decode(Int.self, forKey: .documents)
        partitions = try c.decode(Int.self, forKey: .partitions)
        tokens = try c.decode(Int.self, forKey: .tokens)
        vocabularySHA256 = try c.decode(String.self, forKey: .vocabularySHA256)
        checkpointSHA256 = try c.decodeIfPresent(String.self, forKey: .checkpointSHA256)
        indexSHA256 = try c.decodeIfPresent(String.self, forKey: .indexSHA256)
        indexEntries = try c.decode(Int.self, forKey: .indexEntries)
        memorised = try c.decodeIfPresent(Float.self, forKey: .memorised)
        candidateMemorised = try c.decodeIfPresent(Float.self, forKey: .candidateMemorised)
        epoch = try c.decodeIfPresent(Int.self, forKey: .epoch)
        epochs = try c.decodeIfPresent(Int.self, forKey: .epochs)
        step = try c.decodeIfPresent(Int.self, forKey: .step)
        steps = try c.decodeIfPresent(Int.self, forKey: .steps)
        losses = try c.decode([Float].self, forKey: .losses)
        blockActivity = try c.decode([Float].self, forKey: .blockActivity)
        factsLearned = try c.decodeIfPresent(Int.self, forKey: .factsLearned)
        factsTotal = try c.decodeIfPresent(Int.self, forKey: .factsTotal)
        cells = try c.decode([PartitionCell].self, forKey: .cells)
        preview = try c.decodeIfPresent(String.self, forKey: .preview)
        lastChange = try c.decodeIfPresent(String.self, forKey: .lastChange)
        gates = try c.decode([GateResult].self, forKey: .gates)
        history = try c.decodeIfPresent([VersionMark].self, forKey: .history) ?? []
        error = try c.decodeIfPresent(String.self, forKey: .error)
        updatedAt = try c.decode(Date.self, forKey: .updatedAt)
        packSHA256 = try c.decodeIfPresent(String.self, forKey: .packSHA256)
        cut = try c.decodeIfPresent(Int.self, forKey: .cut)
        heldOutLoss = try c.decodeIfPresent(Float.self, forKey: .heldOutLoss)
        commonsLoss = try c.decodeIfPresent(Float.self, forKey: .commonsLoss)
    }

    /// Whether the umbrella can route to this node.
    public var isLive: Bool { liveVersion != nil }
}

/// One version in a node's history.
public struct VersionMark: Codable, Sendable, Equatable {
    public var version: Int
    public var kind: UpdateKind
    public var promoted: Bool

    public init(version: Int, kind: UpdateKind, promoted: Bool) {
        self.version = version
        self.kind = kind
        self.promoted = promoted
    }
}

extension StrandState {

}

public enum NodeEvent: Codable, Sendable, Equatable {
    case state(StrandState)
    case log(String)
    case promoted(version: Int, kind: UpdateKind)
    case held(version: Int, reason: String)
}

// MARK: - Gating

public enum UpdatePolicy {
    /// Train once added or changed partitions reach this fraction of what the node holds.
    public static let trainFraction: Float = 0.05

    /// The updates a change calls for, cheapest first. A reindex needs live weights; with none
    /// the node trains. Removals alone only reindex: the weights keep withdrawn text until the
    /// node retrains, but the index stops citing it at once.
    public static func plan(change: SnapshotChange, hasLive: Bool, partitions: Int) -> [UpdateKind] {
        guard partitions > 0 else { return [] }
        guard hasLive else { return [.train] }
        guard !change.isEmpty else { return [] }
        let grown = Float(change.added.count + change.changed.count) / Float(max(partitions, 1))
        return grown >= trainFraction ? [.reindex, .train] : [.reindex]
    }
}

public enum VersionGates {
    /// A trained candidate must have memorised this much of its corpus to go live.
    public static let memorisedFloor: Float = 0.90

    public static func memorised(_ fraction: Float, kind: UpdateKind, floor: Float = memorisedFloor) -> GateResult? {
        guard kind == .train else { return nil }
        return GateResult("memorised", fraction >= floor,
                          String(format: "%.1f%% of positions (needs %.0f%%)", fraction * 100, floor * 100))
    }

    public static func vocabulary(found: String, expected: String) -> GateResult {
        GateResult("vocabulary", found == expected,
                   found == expected ? "shared \(found.prefix(12))…" : "holds \(found.prefix(12))…, the umbrella reads \(expected.prefix(12))…")
    }

    /// The umbrella's trunk: the blocks from the cut on must be the pack's, byte for byte.
    public static func trunk(found: String, expected: String) -> GateResult {
        GateResult("trunk", found == expected,
                   found == expected ? "the umbrella's \(expected.prefix(12))…" : "holds \(found.prefix(12))…, the umbrella's is \(expected.prefix(12))…")
    }

    /// No row of the index may belong to a withdrawn document.
    public static func withdrawn(indexDocuments: Set<String>, removed: [String]) -> GateResult {
        let leaked = removed.filter { indexDocuments.contains($0) }
        return GateResult("withdrawn", leaked.isEmpty,
                          leaked.isEmpty ? (removed.isEmpty ? "nothing withdrawn" : "\(removed.count) withdrawn, none indexed")
                                         : "\(leaked.count) withdrawn document(s) still indexed")
    }

    /// Probe spans must verify against the Thread as it is now: no missing, stale or mismatched span.
    public static func verified(statuses: [VerificationStatus]) -> GateResult {
        let bad = statuses.filter { $0 != .verified }
        if statuses.isEmpty { return GateResult("verified", true, "no verbatim span to check") }
        return GateResult("verified", bad.isEmpty,
                          bad.isEmpty ? "\(statuses.count)/\(statuses.count) spans verified"
                                      : "\(bad.count) of \(statuses.count) spans \(Set(bad.map(\.rawValue)).sorted().joined(separator: ", "))")
    }
}

/// Runs an async body to completion from a thread that must not return until it has.
public enum Blocking {
    final class Box<T>: @unchecked Sendable { var value: Result<T, Error>? }

    public static func run<T>(_ body: @escaping @Sendable () async throws -> T) throws -> T {
        let semaphore = DispatchSemaphore(value: 0)
        let box = Box<T>()
        Task.detached {
            do { box.value = .success(try await body()) } catch { box.value = .failure(error) }
            semaphore.signal()
        }
        semaphore.wait()
        return try box.value!.get()
    }
}
