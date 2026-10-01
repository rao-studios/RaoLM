//
//  ThreadStrand.swift
//  RaoLMProvenance
//
//  WHAT: One Thread node's live version, as the umbrella uses it: the headless transformer
//        (token embedding through the last block) and the node's provenance index. For each
//        generation it keeps a session — a KV cache, every position's last hidden state, and
//        the text's trajectory through this Thread's corpus — and answers three questions: the
//        retrieval hits and trajectory of every prompt position (`open`), those after one more
//        token (`advance`), and the hidden states the umbrella's head needs (`hidden`), with the
//        cut state beside each when the umbrella reads the node's thought (`states`).
//  PIN:  Keys are built exactly as CitedGenerator builds them (the same body, the same final
//        norm, the same ProvenanceKey), so a braid of one strand retrieves what a single model
//        retrieves. The trajectory is read from those hits and the session's tokens; it changes
//        no key, hit or hidden state. Everything here runs on the thread that loaded the model.
//

import Foundation
import MLX
import MLXLMCommon
import RaoLMCore
import RaoLMModel
import RaoLMTraining

public enum StrandError: Error, CustomStringConvertible {
    case noSession(String)
    case position(Int, count: Int)
    case notLive(String)

    public var description: String {
        switch self {
        case .noSession(let id): return "no open session \(id)"
        case .position(let position, let count): return "position \(position) is outside the session's \(count) positions"
        case .notLive(let name): return "\(name) has no live version yet"
        }
    }
}

public final class ThreadStrand {
    public let name: String
    public let label: String
    public let version: Int
    public let context: RunContext
    public let owner: String

    final class Session {
        let cache: [KVCache]
        var lasts: [MLXArray] = []
        var cuts: [MLXArray] = []
        var count = 0
        var tracker: TrajectoryTracker

        init(cache: [KVCache], tracker: TrajectoryTracker) {
            self.cache = cache
            self.tracker = tracker
        }
    }

    private var sessions: [String: Session] = [:]
    /// Where each index entry sits in its document, built once per version.
    private lazy var trajectoryCorpus = TrajectoryCorpus(
        values: index.values, keyRow: index.keyRow, keyOffset: index.keyOffset, partitions: index.partitions,
        breakLength: index.info.paragraphBreak?.count ?? 0)
    /// The umbrella pack's anchor snippets; this version's cut state on each is computed once.
    public var anchorTokens: [[Int]] = []
    private var anchorStates: [Float]?

    public init(name: String, label: String, version: Int, context: RunContext, owner: String) {
        self.name = name
        self.label = label
        self.version = version
        self.context = context
        self.owner = owner
    }

    public var threadID: String? { context.manifestRef.threadID }
    public var index: ProvenanceIndex { context.index }
    public var model: RaoTransformer { context.model }

    public func descriptor(vocabularySHA256: String, packSHA256: String? = nil) -> StrandDescriptor {
        let threadID = self.threadID
        let partitions = index.partitions.map { partition -> PartitionRef in
            var copy = partition
            copy.threadID = threadID
            return copy
        }
        return StrandDescriptor(
            name: name, label: label, threadID: threadID, version: version, manifest: context.manifestRef,
            vocabularySHA256: vocabularySHA256, hiddenSize: model.config.hiddenSize, tapLayer: index.info.tapLayer,
            alpha: index.info.alpha, defaultTau: index.info.defaultTau, defaultK: index.info.defaultK,
            indexEntries: index.count, partitions: partitions, sharedNgrams: PackedWords(index.sharedNgrams.sorted()),
            owner: owner, packSHA256: model.config.hasTrunk ? packSHA256 : nil, cut: IndexInfo.cut(of: model.config),
            anchors: anchors().map(PackedFloats.init), calibration: index.info.calibration)
    }

    /// This version's cut state on each of the pack's anchors, the mean over the anchor's positions
    /// ([anchors × hidden], row-major); nil without anchors.
    public func anchors() -> [Float]? {
        guard !anchorTokens.isEmpty else { return nil }
        if let anchorStates { return anchorStates }
        let states = UmbrellaPack.anchorStates(model: model, anchors: anchorTokens).asArray(Float.self)
        anchorStates = states
        return states
    }

    // MARK: - Sessions

    /// Prefills `tokens` and returns the hits and trajectory of every position (position j predicts token j + 1).
    public func open(session id: String, tokens: [Int], k: Int) throws -> [StrandStep] {
        guard !tokens.isEmpty else { throw ProvenanceError.emptyPrompt }
        let session = Session(
            cache: model.newCache(parameters: nil), tracker: TrajectoryTracker(corpus: trajectoryCorpus, tau: index.info.defaultTau))
        sessions[id] = session
        return try step(session, tokens: tokens, k: k)
    }

    /// Appends one token and returns the hits and trajectory of the new position.
    public func advance(session id: String, token: Int, k: Int) throws -> StrandStep {
        guard let session = sessions[id] else { throw StrandError.noSession(id) }
        return try step(session, tokens: [token], k: k).last ?? StrandStep(hits: [])
    }

    /// The last hidden state (the input of the final norm) at each of `positions`.
    public func hidden(session id: String, positions: [Int]) throws -> [[Float]] {
        guard let session = sessions[id] else { throw StrandError.noSession(id) }
        return try rows(session.lasts, positions: positions, count: session.count)
    }

    /// The last hidden state and the cut state (entering the trunk) at each of `positions`.
    public func states(session id: String, positions: [Int]) throws -> StrandStates {
        guard let session = sessions[id] else { throw StrandError.noSession(id) }
        return StrandStates(last: try rows(session.lasts, positions: positions, count: session.count),
                            cut: try rows(session.cuts, positions: positions, count: session.count))
    }

    private func rows(_ steps: [MLXArray], positions: [Int], count: Int) throws -> [[Float]] {
        for position in positions where position < 0 || position >= count {
            throw StrandError.position(position, count: count)
        }
        guard !positions.isEmpty else { return [] }
        let all = steps.count == 1 ? steps[0] : concatenated(steps, axis: 0)
        let taken = all.take(MLXArray(positions.map { Int32($0) }), axis: 0).asType(.float32)
        eval(taken)
        let width = model.config.hiddenSize
        let flat = taken.asArray(Float.self)
        return (0..<positions.count).map { Array(flat[($0 * width)..<(($0 + 1) * width)]) }
    }

    public func close(session id: String) {
        sessions[id] = nil
    }

    public var openSessions: Int { sessions.count }

    private func step(_ session: Session, tokens: [Int], k: Int) throws -> [StrandStep] {
        let input = MLXArray(tokens.map { Int32($0) }, [1, tokens.count])
        let body = model.body(input, cache: session.cache, captureTap: true, captureCut: true)
        guard let tap = body.tap else { throw ProvenanceError.corruptIndex("the model captured no tap layer") }
        let keys = ProvenanceKey.make(tap: tap, final: model.normed(body.last), alpha: index.info.alpha)
        let last = body.last[0]
        let cut = (body.cut ?? body.last)[0]
        eval(keys, last, cut)
        session.lasts.append(last)
        session.cuts.append(cut)
        let positions = input.dim(1)
        session.count += positions
        return (0..<positions).map { j in
            let found = hits(index.query(keys[0, j], k: k))
            return StrandStep(hits: found, trajectory: session.tracker.step(token: tokens[j], hits: found))
        }
    }

    public func hits(_ raw: [(entry: Int, score: Float)]) -> [StrandHit] {
        raw.map { hit in
            StrandHit(
                entry: hit.entry, score: hit.score, value: Int(index.values[hit.entry]), key: index.keyPosition(hit.entry),
                cited: index.valuePosition(hit.entry), sourceLoss: index.loss[hit.entry], sourceEntropy: index.entropy[hit.entry])
        }
    }
}

// MARK: - Links

/// A request to a strand whose answer may arrive later: the umbrella asks every Thread before
/// it waits on any, so Threads in their own processes compute at the same time.
public final class StrandCall<Value>: @unchecked Sendable {
    private let condition = NSCondition()
    private var result: Result<Value, Error>?
    private var handlers: [(Result<Value, Error>) -> Void] = []

    public init() {}

    public static func done(_ body: () throws -> Value) -> StrandCall<Value> {
        let call = StrandCall<Value>()
        call.fulfil(Result { try body() })
        return call
    }

    public func fulfil(_ result: Result<Value, Error>) {
        condition.lock()
        guard self.result == nil else {
            condition.unlock()
            return
        }
        self.result = result
        let handlers = self.handlers
        self.handlers = []
        condition.broadcast()
        condition.unlock()
        for handler in handlers { handler(result) }
    }

    /// Runs `handler` once the answer is in (at once if it already is).
    public func whenDone(_ handler: @escaping (Result<Value, Error>) -> Void) {
        condition.lock()
        if let result {
            condition.unlock()
            handler(result)
            return
        }
        handlers.append(handler)
        condition.unlock()
    }

    /// A call answered with `transform` of this one's answer.
    public func map<Other>(_ transform: @escaping (Value) throws -> Other) -> StrandCall<Other> {
        let mapped = StrandCall<Other>()
        whenDone { result in mapped.fulfil(result.flatMap { value in Result { try transform(value) } }) }
        return mapped
    }

    public func wait(timeout: TimeInterval = 120) throws -> Value {
        let deadline = Date().addingTimeInterval(timeout)
        condition.lock()
        defer { condition.unlock() }
        while result == nil {
            if !condition.wait(until: deadline) { throw StrandLinkError.timeout(seconds: timeout) }
        }
        return try result!.get()
    }
}

public enum StrandLinkError: Error, CustomStringConvertible {
    case timeout(seconds: TimeInterval)
    case closed(String)
    case remote(String, code: Int32)
    case unexpected(String)

    public var description: String {
        switch self {
        case .timeout(let seconds): return "a Thread node did not answer within \(Int(seconds)) s"
        case .closed(let name): return "the link to \(name) is closed"
        case .remote(let message, _): return message
        case .unexpected(let what): return "unexpected reply from a Thread node: \(what)"
        }
    }
}

/// A node's last hidden states and its cut states (entering the trunk) at the same positions.
public struct StrandStates: Sendable {
    public var last: [[Float]]
    /// Empty when the link cannot read them.
    public var cut: [[Float]]

    public init(last: [[Float]], cut: [[Float]]) {
        self.last = last
        self.cut = cut
    }
}

/// How the umbrella reaches one Thread node's live version.
public protocol StrandLink: AnyObject {
    var descriptor: StrandDescriptor { get }
    func open(session: String, tokens: [Int], k: Int) -> StrandCall<[StrandStep]>
    func advance(session: String, token: Int, k: Int) -> StrandCall<StrandStep>
    func hidden(session: String, positions: [Int]) -> StrandCall<[[Float]]>
    /// `hidden`, with the cut state beside each: one request for both.
    func states(session: String, positions: [Int]) -> StrandCall<StrandStates>
    func close(session: String)
}

extension StrandLink {
    public func states(session: String, positions: [Int]) -> StrandCall<StrandStates> {
        hidden(session: session, positions: positions).map { StrandStates(last: $0, cut: []) }
    }
}

/// A strand in this process: every call runs at once, on the caller's thread.
public final class LocalStrandLink: StrandLink {
    public let strand: ThreadStrand
    /// What the umbrella is told of the strand; a bench may set a calibration on it.
    public var descriptor: StrandDescriptor

    public init(strand: ThreadStrand, vocabularySHA256: String, packSHA256: String? = nil) {
        self.strand = strand
        self.descriptor = strand.descriptor(vocabularySHA256: vocabularySHA256, packSHA256: packSHA256)
    }

    public func states(session: String, positions: [Int]) -> StrandCall<StrandStates> {
        .done { try strand.states(session: session, positions: positions) }
    }

    public func open(session: String, tokens: [Int], k: Int) -> StrandCall<[StrandStep]> {
        .done { try strand.open(session: session, tokens: tokens, k: k) }
    }

    public func advance(session: String, token: Int, k: Int) -> StrandCall<StrandStep> {
        .done { try strand.advance(session: session, token: token, k: k) }
    }

    public func hidden(session: String, positions: [Int]) -> StrandCall<[[Float]]> {
        .done { try strand.hidden(session: session, positions: positions) }
    }

    public func close(session: String) { strand.close(session: session) }
}
