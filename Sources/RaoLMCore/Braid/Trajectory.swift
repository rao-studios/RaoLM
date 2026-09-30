//
//  Trajectory.swift
//  RaoLMCore
//
//  WHAT: A Thread's retrieval trajectory. Trace: whether the text so far follows one document the
//        Thread holds, in order, for longer than a sentence. Manner: whether the Thread's hits
//        move through their documents the way a document runs. The node computes both from the
//        hits it already retrieves and sends them with those hits.
//  PIN:  Pure: no model and no index tensors. Retrieval only seeds a chain; a chain advances only
//        when the token that arrives is the corpus's next token, so no score from one node's key
//        space enters it. The constants were fixed before any numbers (Docs/BRAID.md, "The
//        trajectory fingerprint").
//

import Foundation

/// The trajectory's constants, in tokens or steps.
public struct TrajectoryRule: Codable, Sendable, Equatable {
    /// A chain this long or shorter traces 0: longer than 95% of the dataset's sentences.
    public var phrase = 24
    /// A chain this long traces 1: about half a document.
    public var arc = 96
    /// The most a bridge skips, in the text and in the corpus; both runs must reach it.
    public var bridge = 8
    /// Chains kept, and the hits that may start one at each step.
    public var heads = 8
    public var seeds = 4
    /// Manner: steps per block, the blocks compared, the fewest that decide, and the fit window.
    public var block = 16
    public var blocks = 8
    public var minBlocks = 4
    public var fitWindow = 32

    public init() {}

    public static let standard = TrajectoryRule()

    /// clamp((length − phrase) / (arc − phrase), 0, 1).
    public func trace(length: Int) -> Float {
        guard arc > phrase else { return length > phrase ? 1 : 0 }
        return min(max(Float(length - phrase) / Float(arc - phrase), 0), 1)
    }
}

/// One Thread's trajectory at one position (after the position's token, with its hits).
public struct StrandTrajectory: Codable, Sendable, Equatable {
    /// Tokens the longest live chain has followed through one document, bridged runs included.
    public var length: Int
    /// `TrajectoryRule.trace(length:)` of the rule the node ran.
    public var trace: Float
    /// That chain's entry now (it expects the next token), where the entry's context sits, and
    /// the token it expects. Nil without a chain.
    public var entry: Int?
    public var at: TokenPosition?
    public var next: Int?
    /// Where the position's hits sit in their documents, 0 to 1, weighted as retrieval weighs them.
    public var phase: Float?
    /// The retrieval mass the node gave the tokens that came, over the last `fitWindow` steps.
    public var fit: Float
    /// Spearman of the blocks' median phases against their order; nil before `minBlocks` blocks.
    public var arc: Float?
    /// max(0, arc) · fit; nil before `minBlocks` blocks, which means "ask".
    public var manner: Float?

    public init(
        length: Int, trace: Float, entry: Int? = nil, at: TokenPosition? = nil, next: Int? = nil, phase: Float? = nil, fit: Float = 0,
        arc: Float? = nil, manner: Float? = nil
    ) {
        self.length = length
        self.trace = trace
        self.entry = entry
        self.at = at
        self.next = next
        self.phase = phase
        self.fit = fit
        self.arc = arc
        self.manner = manner
    }
}

/// Where each index entry sits in its document: what the tracker needs of a node's index.
public struct TrajectoryCorpus: Sendable {
    /// Per entry: the token that followed its context (the index's value), where its context
    /// sits, its document's ordinal, and its context's token index in that document.
    public let values: [Int32]
    public let keyRow: [Int32]
    public let keyOffset: [Int32]
    public let documents: [Int32]
    public let positions: [Int32]
    /// Per document ordinal: its tokens, the sum of its partitions' token counts.
    public let lengths: [Int32]

    /// `partitions` is the index's partition table: a document's rows in any order, each with its
    /// partition index and token count.
    public init(values: [Int32], keyRow: [Int32], keyOffset: [Int32], partitions: [PartitionRef]) {
        precondition(values.count == keyRow.count && values.count == keyOffset.count)
        var ordinal: [String: Int32] = [:]
        var byDocument: [String: [PartitionRef]] = [:]
        for partition in partitions {
            if ordinal[partition.documentID] == nil { ordinal[partition.documentID] = Int32(ordinal.count) }
            byDocument[partition.documentID, default: []].append(partition)
        }
        var rowDocument: [Int: Int32] = [:]
        var rowStart: [Int: Int32] = [:]
        var lengths = [Int32](repeating: 0, count: ordinal.count)
        for (documentID, rows) in byDocument {
            let document = ordinal[documentID]!
            var start: Int32 = 0
            for partition in rows.sorted(by: { $0.partitionIndex < $1.partitionIndex }) {
                rowDocument[partition.row] = document
                rowStart[partition.row] = start
                start += Int32(partition.tokenCount)
            }
            lengths[Int(document)] = start
        }
        self.values = values
        self.keyRow = keyRow
        self.keyOffset = keyOffset
        documents = keyRow.map { rowDocument[Int($0)] ?? -1 }
        positions = zip(keyRow, keyOffset).map { row, offset in (rowStart[Int(row)] ?? 0) + offset }
        self.lengths = lengths
    }

    public var count: Int { values.count }

    /// The entry of the next position of the same document, if the index holds it.
    public func successor(of entry: Int) -> Int? {
        let next = entry + 1
        guard entry >= 0, next < count, documents[next] == documents[entry], documents[entry] >= 0,
              positions[next] == positions[entry] + 1 else { return nil }
        return next
    }

    /// The entry's place in its document, 0 at the first token and 1 at the last.
    public func phase(of entry: Int) -> Float {
        guard entry >= 0, entry < count, documents[entry] >= 0 else { return 0 }
        let length = lengths[Int(documents[entry])]
        return length > 1 ? min(max(Float(positions[entry]) / Float(length - 1), 0), 1) : 0
    }

    public func position(of entry: Int) -> TokenPosition { TokenPosition(row: Int(keyRow[entry]), offset: Int(keyOffset[entry])) }
}

/// Follows one text through one node's corpus, a position at a time.
public struct TrajectoryTracker: Sendable {
    struct Head {
        /// The entry whose value is the token this chain expects next.
        var entry: Int
        var document: Int32
        /// Tokens counted, a bridged chain's included.
        var mass: Int
        /// Tokens matched since this run began.
        var run: Int
        /// The step the run's first token came at, and the document position of its context.
        var start: Int
        var startPosition: Int32
        /// For a dormant chain: the step it failed at, and the position it expected there.
        var ended = 0
        var endPosition: Int32 = 0
    }

    public let corpus: TrajectoryCorpus
    public let rule: TrajectoryRule
    public let tau: Float
    private var live: [Head] = []
    private var fresh: [Head] = []
    private var dormant: [Head] = []
    private var steps = 0
    private var previous: [(value: Int, weight: Float)] = []
    private var fits: [Float] = []
    private var phases: [Float] = []
    private var medians: [Float] = []

    public init(corpus: TrajectoryCorpus, rule: TrajectoryRule = .standard, tau: Float) {
        self.corpus = corpus
        self.rule = rule
        self.tau = tau
    }

    /// One position: `token` is the token at this position (the one the previous position's hits
    /// expected), `hits` what this position's key retrieved (they expect the next token).
    public mutating func step(token: Int, hits: [StrandHit]) -> StrandTrajectory {
        let step = steps
        steps += 1

        // The token that arrived: every chain that expected it moves on; a chain that did not goes
        // dormant (a seed that never matched is dropped).
        var moved: [Head] = []
        var finished: [Head] = []
        for var head in live + fresh {
            guard head.entry < corpus.count else { continue }
            if Int(corpus.values[head.entry]) == token {
                head.mass += 1
                head.run += 1
                if head.run == rule.bridge { bridge(&head) }
                if let next = corpus.successor(of: head.entry) {
                    head.entry = next
                    moved.append(head)
                } else {
                    // The document ends here: the chain counts at this position and can go no further.
                    head.ended = step + 1
                    head.endPosition = corpus.positions[head.entry] + 1
                    dormant.append(head)
                    finished.append(head)
                }
            } else if head.run > 0 {
                head.ended = step
                head.endPosition = corpus.positions[head.entry]
                dormant.append(head)
            }
        }
        // Two chains on one entry are one chain: the one with more behind it stays.
        var byEntry: [Int: Head] = [:]
        for head in moved where (byEntry[head.entry]?.mass ?? -1) < head.mass { byEntry[head.entry] = head }
        live = Array(byEntry.values.sorted { $0.mass != $1.mass ? $0.mass > $1.mass : $0.entry < $1.entry }.prefix(rule.heads))

        // This position's hits seed chains for the token that comes next.
        let sitting = Set(live.map(\.entry))
        fresh = hits.prefix(rule.seeds).compactMap { hit in
            guard hit.entry >= 0, hit.entry < corpus.count, !sitting.contains(hit.entry) else { return nil }
            return Head(entry: hit.entry, document: corpus.documents[hit.entry], mass: 0, run: 0, start: step + 1,
                        startPosition: corpus.positions[hit.entry])
        }
        dormant = Array(dormant.filter { step - $0.ended <= 2 * rule.bridge }
            .sorted { $0.mass != $1.mass ? $0.mass > $1.mass : $0.entry < $1.entry }.prefix(2 * rule.heads))

        // Manner: the mass the last position gave this token, and where this position's hits sit.
        if step > 0 {
            fits.append(previous.reduce(Float(0)) { $1.value == token ? $0 + $1.weight : $0 })
            if fits.count > rule.fitWindow { fits.removeFirst(fits.count - rule.fitWindow) }
        }
        let weights = Self.softmax(hits.map(\.score), tau: tau)
        previous = zip(hits, weights).map { (value: $0.value, weight: $1) }
        var phase: Float?
        if !hits.isEmpty {
            phase = zip(hits, weights).reduce(Float(0)) { $0 + $1.1 * corpus.phase(of: $1.0.entry) }
            phases.append(phase!)
            if phases.count == rule.block {
                medians.append(Self.median(phases))
                phases = []
                if medians.count > rule.blocks { medians.removeFirst(medians.count - rule.blocks) }
            }
        }
        let fit = fits.isEmpty ? 0 : fits.reduce(0, +) / Float(fits.count)
        let arc: Float? = medians.count >= rule.minBlocks
            ? Stats.spearman(medians.indices.map(Double.init), medians.map(Double.init)).map(Float.init) : nil

        let best = live.first
        let length = max(best?.mass ?? 0, finished.map(\.mass).max() ?? 0)
        let chain = best.flatMap { $0.mass == length ? $0 : nil }
        return StrandTrajectory(
            length: length, trace: rule.trace(length: length), entry: chain?.entry, at: chain.map { corpus.position(of: $0.entry) },
            next: chain.map { Int(corpus.values[$0.entry]) }, phase: phase, fit: fit, arc: arc, manner: arc.map { max(0, $0) * fit })
    }

    /// A run that has just reached `bridge` tokens takes on a dormant chain of the same document
    /// whose own run reached it and that ended at most `bridge` tokens before this run began, in
    /// the text and in the corpus (a changed name or value, never a sentence).
    private mutating func bridge(_ head: inout Head) {
        let candidates = dormant.indices.filter { i in
            let d = dormant[i]
            let text = head.start - d.ended
            let corpus = Int(head.startPosition - d.endPosition)
            return d.document == head.document && d.run >= rule.bridge && (0...rule.bridge).contains(text) && (0...rule.bridge).contains(corpus)
        }
        guard let pick = candidates.max(by: { dormant[$0].mass < dormant[$1].mass }) else { return }
        head.mass += dormant[pick].mass
        dormant.remove(at: pick)
    }

    static func softmax(_ scores: [Float], tau: Float) -> [Float] {
        guard let top = scores.max() else { return [] }
        let t = max(tau, 1e-6)
        let exps = scores.map { exp(($0 - top) / t) }
        let total = exps.reduce(0, +)
        return exps.map { $0 / total }
    }

    static func median(_ values: [Float]) -> Float {
        let sorted = values.sorted()
        guard !sorted.isEmpty else { return 0 }
        let mid = sorted.count / 2
        return sorted.count % 2 == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
    }
}

/// What the umbrella can check from the wire alone.
public enum TrajectoryAudit {
    /// Per position, the longest chain of hits that follow one another through one document:
    /// entry e at one position, e + 1 at the next, each expecting the token that came. Not a bound
    /// on a node's chain either way: it follows hits of any rank, where a node seeds only from its
    /// top few, so it can find short chains the node never started; and it never bridges, so it
    /// splits chains the node joins. `hits[j]` are position j's hits; `tokens[j]` the token at j.
    public static func wireLengths(tokens: [Int], hits: [[StrandHit]], documentOfRow: [Int: String]) -> [Int] {
        var lengths: [Int] = []
        var chains: [Int: Int] = [:]
        for j in hits.indices {
            var next: [Int: Int] = [:]
            if j > 0 {
                for hit in hits[j - 1] where hit.value == tokens[j] {
                    let before = hits.indices.contains(j - 2) ? hits[j - 2].first { $0.entry == hit.entry - 1 } : nil
                    let same = before.map { documentOfRow[$0.key.row] != nil && documentOfRow[$0.key.row] == documentOfRow[hit.key.row] } ?? false
                    next[hit.entry] = max(next[hit.entry] ?? 0, (same ? chains[hit.entry - 1] ?? 0 : 0) + 1)
                }
            }
            chains = next
            lengths.append(chains.values.max() ?? 0)
        }
        return lengths
    }
}
