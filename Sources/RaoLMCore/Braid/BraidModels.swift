//
//  BraidModels.swift
//  RaoLMCore
//
//  WHAT: The values a braid passes around: what a Thread node returns for one position (its
//        retrieval hits and, when asked, its hidden state), what each Thread supplied to one
//        emitted token, and the Threads a braided generation is bound to.
//  PIN:  A node's rows and index entries are its own. A braided generation renumbers them into
//        one partition table (global row = the strand's row offset + its local row), so the
//        span and citation logic runs on it unchanged, and each partition carries its Thread.
//

import Foundation

/// One entry of a node's own provenance index, retrieved for one position.
public struct StrandHit: Codable, Sendable, Equatable {
    public var entry: Int
    /// Cosine between the query key and this entry's key, in the node's own key space.
    public var score: Float
    /// The corpus token that followed this context.
    public var value: Int
    /// Node-local positions of the context and of `value`.
    public var key: TokenPosition
    public var cited: TokenPosition
    public var sourceLoss: Float
    public var sourceEntropy: Float

    public init(
        entry: Int, score: Float, value: Int, key: TokenPosition, cited: TokenPosition, sourceLoss: Float, sourceEntropy: Float
    ) {
        self.entry = entry
        self.score = score
        self.value = value
        self.key = key
        self.cited = cited
        self.sourceLoss = sourceLoss
        self.sourceEntropy = sourceEntropy
    }
}

/// What a node returns for one position: its retrieval hits and its trajectory through its corpus.
public struct StrandStep: Codable, Sendable, Equatable {
    public var hits: [StrandHit]
    public var trajectory: StrandTrajectory?

    public init(hits: [StrandHit], trajectory: StrandTrajectory? = nil) {
        self.hits = hits
        self.trajectory = trajectory
    }
}

/// How the umbrella weighs Threads.
public enum BraidGating: String, Codable, Sendable, CaseIterable {
    /// Token by token: a memory of which Thread has been predicting the text (bounded evidence
    /// from the prompt's tokens), lifted for a Thread whose retrieval backs the leader's
    /// candidate, so a token both Threads know is shared.
    case braided
    /// By the posterior of each Thread given every token so far, unbounded: it settles on one
    /// Thread and stays there.
    case posterior
    /// By each Thread's share of one retrieval pool ranked by raw cosine across Threads.
    case retrieval
}

/// The braided gate's settings.
///
/// Settings added after the first recordings are written only when they differ from their
/// default, so the default gate's JSON, its fingerprint and every generation id made with it
/// stay what they were, and older recordings decode with the defaults.
public struct BraidGate: Codable, Sendable, Equatable {
    public enum Share: String, Codable, Sendable {
        /// A Thread gives up weight only on tokens it failed to predict.
        case variable
        /// Every Thread gives up the same fraction at every token.
        case fixed
    }

    /// How a Thread's trace of the text (its trajectory through its own corpus) enters the gate.
    public enum TrajectoryUse: String, Codable, Sendable, CaseIterable {
        /// Not at all: the gate as it was.
        case off
        /// A Thread is lifted beside the leader only as far as it traces the text as well.
        case lift
        /// Every weight is multiplied by exp(β·(trace − the mean trace of the Threads)).
        case gate
        /// Both.
        case both
    }

    /// Which Threads are asked for their hidden state while the prompt is scored.
    public enum Ask: String, Codable, Sendable, CaseIterable {
        /// Every Thread: the gate as it was.
        case gate
        /// Those whose manner at the prompt's end reaches `askFloor`; every Thread when none does,
        /// or when no Thread has a manner yet (a prompt shorter than four blocks).
        case manner
    }

    /// A Thread that gave a token less than this did not predict it.
    public var evidenceFloor: Float
    /// A Thread that gave a token at least this predicted it; more earns nothing more.
    public var evidenceCeiling: Float
    public var share: Share
    public var shareRate: Float
    /// Evidence counts in proportion to how much of the recent text the best Thread predicted.
    public var credibility: Bool
    public var credibilityRate: Float
    /// A Thread whose retrieval backs the leader's candidate is lifted towards the leader.
    public var agreement: Bool
    /// Whether tokens the mixture chose itself count as evidence.
    public var generatedEvidence: Bool
    public var trajectory: TrajectoryUse
    /// β of `trajectory: gate`.
    public var trajectoryBeta: Float
    public var ask: Ask
    public var askFloor: Float

    public init(
        evidenceFloor: Float = 0.05, evidenceCeiling: Float = 0.5, share: Share = .variable, shareRate: Float = 0.1,
        credibility: Bool = true, credibilityRate: Float = 0.3, agreement: Bool = true, generatedEvidence: Bool = false,
        trajectory: TrajectoryUse = .off, trajectoryBeta: Float = 1, ask: Ask = .gate, askFloor: Float = 0.25
    ) {
        self.evidenceFloor = evidenceFloor
        self.evidenceCeiling = evidenceCeiling
        self.share = share
        self.shareRate = shareRate
        self.credibility = credibility
        self.credibilityRate = credibilityRate
        self.agreement = agreement
        self.generatedEvidence = generatedEvidence
        self.trajectory = trajectory
        self.trajectoryBeta = trajectoryBeta
        self.ask = ask
        self.askFloor = askFloor
    }

    private enum CodingKeys: String, CodingKey {
        case evidenceFloor, evidenceCeiling, share, shareRate, credibility, credibilityRate, agreement, generatedEvidence
        case trajectory, trajectoryBeta, ask, askFloor
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = BraidGate()
        evidenceFloor = try c.decode(Float.self, forKey: .evidenceFloor)
        evidenceCeiling = try c.decode(Float.self, forKey: .evidenceCeiling)
        share = try c.decode(Share.self, forKey: .share)
        shareRate = try c.decode(Float.self, forKey: .shareRate)
        credibility = try c.decode(Bool.self, forKey: .credibility)
        credibilityRate = try c.decode(Float.self, forKey: .credibilityRate)
        agreement = try c.decode(Bool.self, forKey: .agreement)
        generatedEvidence = try c.decode(Bool.self, forKey: .generatedEvidence)
        trajectory = try c.decodeIfPresent(TrajectoryUse.self, forKey: .trajectory) ?? defaults.trajectory
        trajectoryBeta = try c.decodeIfPresent(Float.self, forKey: .trajectoryBeta) ?? defaults.trajectoryBeta
        ask = try c.decodeIfPresent(Ask.self, forKey: .ask) ?? defaults.ask
        askFloor = try c.decodeIfPresent(Float.self, forKey: .askFloor) ?? defaults.askFloor
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(evidenceFloor, forKey: .evidenceFloor)
        try c.encode(evidenceCeiling, forKey: .evidenceCeiling)
        try c.encode(share, forKey: .share)
        try c.encode(shareRate, forKey: .shareRate)
        try c.encode(credibility, forKey: .credibility)
        try c.encode(credibilityRate, forKey: .credibilityRate)
        try c.encode(agreement, forKey: .agreement)
        try c.encode(generatedEvidence, forKey: .generatedEvidence)
        let defaults = BraidGate()
        if trajectory != defaults.trajectory { try c.encode(trajectory, forKey: .trajectory) }
        if trajectoryBeta != defaults.trajectoryBeta { try c.encode(trajectoryBeta, forKey: .trajectoryBeta) }
        if ask != defaults.ask { try c.encode(ask, forKey: .ask) }
        if askFloor != defaults.askFloor { try c.encode(askFloor, forKey: .askFloor) }
    }

    /// Eight hex digits naming these settings, for a generation's id.
    public var fingerprint: String {
        let data = (try? JSONCoding.lineEncoder().encode(self)) ?? Data()
        return String(ContentHash.sha256Hex(data).prefix(8))
    }
}

/// One of the mixture's likeliest next tokens, and what each Thread supplied of it.
public struct TokenCandidate: Codable, Sendable, Equatable {
    public var token: Int
    public var text: String
    public var prob: Float
    /// Each Thread's fraction of `prob`, in the braid's strand order.
    public var parts: [Float]

    public init(token: Int, text: String, prob: Float, parts: [Float]) {
        self.token = token
        self.text = text
        self.prob = prob
        self.parts = parts
    }
}

/// What one Thread supplied to one emitted token.
public struct StrandShare: Codable, Sendable, Equatable {
    public var strand: String
    public var threadID: String?
    /// The Thread's weight before renormalising over the Threads that were asked. What it is
    /// made of depends on the gating the generation's `braid` records.
    public var gate: Float
    /// Whether the umbrella asked this Thread for its hidden state.
    public var open: Bool
    /// The best cosine among this Thread's own hits.
    public var bestScore: Float?
    /// The umbrella head's probability for the token from this Thread's hidden state.
    public var lmProb: Float?
    public var lmEntropy: Float?
    /// Retrieval weight on the token from this Thread's corpus.
    public var knn: Float
    /// The fraction of p(token) this Thread supplied. A token's shares sum to 1.
    public var share: Float
    /// Braided gate: the Thread's memory weight (which Thread has been predicting the text).
    public var memory: Float?
    /// Braided gate: how much of this Thread's retrieval backs the leader's candidate.
    public var backs: Float?
    /// What this Thread alone gave the token, λ·p_knn,t + (1−λ)·p_lm,t. Nil when it was not asked.
    public var alone: Float?
    /// The Thread's trajectory through its own corpus at the position that predicts the token.
    public var trajectory: StrandTrajectory?

    public init(
        strand: String, threadID: String?, gate: Float, open: Bool, bestScore: Float?, lmProb: Float?, lmEntropy: Float?,
        knn: Float, share: Float
    ) {
        self.strand = strand
        self.threadID = threadID
        self.gate = gate
        self.open = open
        self.bestScore = bestScore
        self.lmProb = lmProb
        self.lmEntropy = lmEntropy
        self.knn = knn
        self.share = share
    }
}

/// One Thread as a braided generation used it.
public struct BraidStrandRef: Codable, Sendable, Equatable {
    public var name: String
    public var label: String
    public var threadID: String?
    public var version: Int
    public var manifest: ManifestRef
    /// Global rows `rowOffset ..< rowOffset + rowCount` are this Thread's rows `0 ..< rowCount`.
    public var rowOffset: Int
    public var rowCount: Int
    /// Global index entries start here for this Thread.
    public var entryOffset: Int

    public init(
        name: String, label: String, threadID: String?, version: Int, manifest: ManifestRef, rowOffset: Int, rowCount: Int,
        entryOffset: Int
    ) {
        self.name = name
        self.label = label
        self.threadID = threadID
        self.version = version
        self.manifest = manifest
        self.rowOffset = rowOffset
        self.rowCount = rowCount
        self.entryOffset = entryOffset
    }

    public func contains(row: Int) -> Bool { row >= rowOffset && row < rowOffset + rowCount }
}

/// The Threads and the umbrella a braided generation is bound to.
public struct BraidRef: Codable, Sendable, Equatable {
    public var vocabularySHA256: String
    public var gateFloor: Float
    public var strands: [BraidStrandRef]
    /// How the Threads were weighed. Nil in generations recorded before it was kept (posterior).
    public var gating: BraidGating?
    /// The braided gate's settings, when that gate was used.
    public var gate: BraidGate?

    public init(
        vocabularySHA256: String, gateFloor: Float, strands: [BraidStrandRef], gating: BraidGating? = nil, gate: BraidGate? = nil
    ) {
        self.vocabularySHA256 = vocabularySHA256
        self.gateFloor = gateFloor
        self.strands = strands
        self.gating = gating
        self.gate = gate
    }

    public func strand(row: Int) -> BraidStrandRef? { strands.first { $0.contains(row: row) } }
    public func strand(named name: String) -> BraidStrandRef? { strands.first { $0.name == name } }

    /// The ManifestRef a braided generation carries: the strands' hashes folded together, so
    /// the generation id changes whenever any Thread's version does, and with the gate. The
    /// posterior keeps the id it had before gates were recorded.
    public func combinedManifest(tokenizerSHA256: String) -> ManifestRef {
        func fold(_ values: [String]) -> String { ContentHash.sha256Hex(values.joined(separator: "\n")) }
        var names = strands.map { "\($0.name)@\($0.version)" }.joined(separator: "+")
        if let gating, gating != .posterior {
            names += "/" + gating.rawValue
            if gating == .braided, let gate { names += "-" + gate.fingerprint }
        }
        return ManifestRef(
            runID: "braid:" + names, epoch: strands.map(\.version).reduce(0, +),
            checkpointSHA256: fold([vocabularySHA256] + strands.map(\.manifest.checkpointSHA256)),
            indexSHA256: fold(strands.map(\.manifest.indexSHA256)),
            corpusHash: fold(strands.map(\.manifest.corpusHash)),
            tokenizerSHA256: tokenizerSHA256, ledgerSHA256: nil, threadID: nil)
    }
}

/// A node's live version, as the umbrella needs it to route and to cite.
public struct StrandDescriptor: Codable, Sendable, Equatable {
    public var name: String
    public var label: String
    public var threadID: String?
    public var version: Int
    public var manifest: ManifestRef
    public var vocabularySHA256: String
    public var hiddenSize: Int
    public var tapLayer: Int
    public var alpha: Float
    public var defaultTau: Float
    public var defaultK: Int
    public var indexEntries: Int
    /// The partition table the node's index rows refer to, each with this Thread's id.
    public var partitions: [PartitionRef]
    /// Token 3-grams shared by two or more of this Thread's documents.
    public var sharedNgrams: PackedWords
    public var owner: String

    public init(
        name: String, label: String, threadID: String?, version: Int, manifest: ManifestRef, vocabularySHA256: String,
        hiddenSize: Int, tapLayer: Int, alpha: Float, defaultTau: Float, defaultK: Int, indexEntries: Int,
        partitions: [PartitionRef], sharedNgrams: PackedWords, owner: String
    ) {
        self.name = name
        self.label = label
        self.threadID = threadID
        self.version = version
        self.manifest = manifest
        self.vocabularySHA256 = vocabularySHA256
        self.hiddenSize = hiddenSize
        self.tapLayer = tapLayer
        self.alpha = alpha
        self.defaultTau = defaultTau
        self.defaultK = defaultK
        self.indexEntries = indexEntries
        self.partitions = partitions
        self.sharedNgrams = sharedNgrams
        self.owner = owner
    }
}

// MARK: - Packed arrays

/// Float32 values as base64 of their little-endian bytes: exact, and a third the size of JSON numbers.
public struct PackedFloats: Codable, Sendable, Equatable {
    public var base64: String
    public var count: Int

    public init(_ values: [Float]) {
        var little = values.map { $0.bitPattern.littleEndian }
        base64 = little.withUnsafeMutableBytes { Data($0) }.base64EncodedString()
        count = values.count
    }

    public var values: [Float] {
        guard let data = Data(base64Encoded: base64), data.count == count * 4 else { return [] }
        var words = [UInt32](repeating: 0, count: count)
        _ = words.withUnsafeMutableBytes { data.copyBytes(to: $0) }
        return words.map { Float(bitPattern: UInt32(littleEndian: $0)) }
    }
}

/// UInt64 values as base64 of their little-endian bytes.
public struct PackedWords: Codable, Sendable, Equatable {
    public var base64: String
    public var count: Int

    public init(_ values: [UInt64]) {
        var little = values.map { $0.littleEndian }
        base64 = little.withUnsafeMutableBytes { Data($0) }.base64EncodedString()
        count = values.count
    }

    public var values: [UInt64] {
        guard let data = Data(base64Encoded: base64), data.count == count * 8 else { return [] }
        var words = [UInt64](repeating: 0, count: count)
        _ = words.withUnsafeMutableBytes { data.copyBytes(to: $0) }
        return words.map { UInt64(littleEndian: $0) }
    }
}
