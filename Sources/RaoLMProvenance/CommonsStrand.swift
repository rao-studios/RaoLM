//
//  CommonsStrand.swift
//  RaoLMProvenance
//
//  WHAT: The umbrella's commons strand: the pack's base model whole, the model every node's
//        blocks start from, run by the umbrella itself as one more expert of the mixture. It
//        holds no Thread, retrieves nothing and is cited to nothing. What it supplies of a token
//        is what the base already knew, which no Thread earned, so its share goes to no one, and
//        each Thread's lift (log p_t − log p_commons) is what that Thread knows beyond it.
//  PIN:  In process, on the thread that made it, as a LocalStrandLink runs. Its λ is 0: with no
//        retrieval its distribution is its head's alone. The umbrella always asks it.
//

import Foundation
import MLX
import MLXLMCommon
import RaoLMCore
import RaoLMModel

public final class CommonsLink: StrandLink {
    public static let strandName = BraidStrandRef.commonsName

    public let model: RaoTransformer
    public var descriptor: StrandDescriptor

    final class Session {
        let cache: [KVCache]
        var lasts: [MLXArray] = []
        var cuts: [MLXArray] = []
        var count = 0

        init(cache: [KVCache]) { self.cache = cache }
    }

    private var sessions: [String: Session] = [:]

    public init(pack: UmbrellaPack, tokenizerSHA256: String) throws {
        let model = try pack.baseModel()
        self.model = model
        self.descriptor = StrandDescriptor(
            name: Self.strandName, label: "Commons", threadID: nil, version: 0,
            manifest: ManifestRef(
                runID: "commons:\(pack.name)", epoch: 0, checkpointSHA256: pack.info?.baseSHA256 ?? pack.sha256, indexSHA256: "",
                corpusHash: "", tokenizerSHA256: tokenizerSHA256, ledgerSHA256: nil, threadID: nil),
            vocabularySHA256: pack.vocabulary.sha256, hiddenSize: model.config.hiddenSize, tapLayer: model.tapLayer, alpha: 0.5,
            defaultTau: 0.05, defaultK: 16, indexEntries: 0, partitions: [], sharedNgrams: PackedWords([]), owner: "",
            packSHA256: pack.hasTrunk ? pack.sha256 : nil, cut: pack.cut, commons: true,
            anchors: pack.anchorStates.map { PackedFloats($0.asArray(Float.self)) })
    }

    public func open(session: String, tokens: [Int], k: Int) -> StrandCall<[StrandStep]> {
        .done {
            guard !tokens.isEmpty else { throw ProvenanceError.emptyPrompt }
            let state = Session(cache: model.newCache(parameters: nil))
            sessions[session] = state
            try step(state, tokens: tokens)
            return tokens.map { _ in StrandStep(hits: []) }
        }
    }

    public func advance(session: String, token: Int, k: Int) -> StrandCall<StrandStep> {
        .done {
            guard let state = sessions[session] else { throw StrandError.noSession(session) }
            try step(state, tokens: [token])
            return StrandStep(hits: [])
        }
    }

    public func hidden(session: String, positions: [Int]) -> StrandCall<[[Float]]> {
        .done {
            guard let state = sessions[session] else { throw StrandError.noSession(session) }
            return try rows(state.lasts, positions: positions, count: state.count)
        }
    }

    public func states(session: String, positions: [Int]) -> StrandCall<StrandStates> {
        .done {
            guard let state = sessions[session] else { throw StrandError.noSession(session) }
            return StrandStates(last: try rows(state.lasts, positions: positions, count: state.count),
                                cut: try rows(state.cuts, positions: positions, count: state.count))
        }
    }

    public func close(session: String) { sessions[session] = nil }

    private func step(_ state: Session, tokens: [Int]) throws {
        let body = model.body(MLXArray(tokens.map { Int32($0) }, [1, tokens.count]), cache: state.cache, captureTap: false, captureCut: true)
        let last = body.last[0]
        let cut = (body.cut ?? body.last)[0]
        eval(last, cut)
        state.lasts.append(last)
        state.cuts.append(cut)
        state.count += tokens.count
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
}
