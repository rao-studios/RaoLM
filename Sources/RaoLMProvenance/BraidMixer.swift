//
//  BraidMixer.swift
//  RaoLMProvenance
//
//  WHAT: The umbrella's arithmetic for one position of a braid, under each way of weighing
//        Threads (`BraidGating`). Every Thread is its own kNN-LM (its retrieval weighed within
//        its own index, its hidden state through the shared head); the umbrella mixes them by
//        their gates and splits the chosen token's probability exactly by Thread.
//        - braided: a memory of which Thread has been predicting the text, from bounded
//          evidence, lifted where a Thread's retrieval backs the leader's candidate.
//        - posterior: every token's likelihood multiplied, unbounded.
//        - retrieval: one pool of every Thread's hits ranked by raw cosine (the first part of
//          this file), each Thread gated by its share of the pooled weight.
//  PIN:  With one Thread this is CitationMixer step for step (same weights, same p_lm, same
//        mixture), so a braid of one reproduces CitedGenerator. A closed gate means the
//        umbrella never asks that Thread for its hidden state; the Thread with the largest
//        gate is always open. share_t(y) = g′_t·p_t(y) / p(y), so a token's shares sum to 1.
//        The braided gate compares nothing across two models' key spaces: only probabilities
//        the Threads gave to tokens.
//

import Foundation
import RaoLMCore

public enum BraidMixer {
    public static let defaultGateFloor: Float = 0.05

    public struct Pooled: Sendable, Equatable {
        public var strand: Int
        public var hit: StrandHit
        public var weight: Float
    }

    public struct Pool: Sendable, Equatable {
        /// The top-k of every Thread's hits, ranked by (score desc, strand asc, entry asc).
        public var members: [Pooled]
        /// Each Thread's share of the pooled weight.
        public var gates: [Float]
        /// Each Thread's best cosine among its own hits.
        public var best: [Float?]
        public var open: [Bool]

        public var openIndices: [Int] { open.indices.filter { open[$0] } }
    }

    /// Pools `hits` (one list per Thread, each already ranked) into the top `k`.
    public static func pool(_ hits: [[StrandHit]], k: Int, tau: Float, floor: Float = defaultGateFloor) -> Pool {
        var candidates: [(strand: Int, hit: StrandHit)] = []
        for (strand, list) in hits.enumerated() {
            for hit in list { candidates.append((strand, hit)) }
        }
        candidates.sort { a, b in
            if a.hit.score != b.hit.score { return a.hit.score > b.hit.score }
            if a.strand != b.strand { return a.strand < b.strand }
            return a.hit.entry < b.hit.entry
        }
        let top = Array(candidates.prefix(max(0, k)))
        let weights = CitationMixer.weights(top.map(\.hit.score), tau: tau)
        var gates = [Float](repeating: 0, count: hits.count)
        let members = top.enumerated().map { rank, member -> Pooled in
            gates[member.strand] += weights[rank]
            return Pooled(strand: member.strand, hit: member.hit, weight: weights[rank])
        }
        let best: [Float?] = hits.map { $0.map(\.score).max() }
        return Pool(members: members, gates: gates, best: best, open: gate(gates, floor: floor))
    }

    /// Open where the gate reaches `floor`; the largest gate is always open; no weight anywhere
    /// opens every Thread.
    public static func gate(_ gates: [Float], floor: Float) -> [Bool] {
        guard !gates.isEmpty else { return [] }
        guard gates.contains(where: { $0 > 0 }) else { return gates.map { _ in true } }
        var open = gates.map { $0 >= floor && $0 > 0 }
        if let largest = gates.indices.max(by: { gates[$0] < gates[$1] }) { open[largest] = true }
        return open
    }

    /// Gates renormalised over the open Threads (equal weights when no open Thread has any).
    public static func renormalised(_ pool: Pool) -> [Float] {
        let openTotal = pool.openIndices.reduce(Float(0)) { $0 + pool.gates[$1] }
        let openCount = Float(pool.openIndices.count)
        return pool.gates.indices.map { t in
            guard pool.open[t] else { return 0 }
            return openTotal > 0 ? pool.gates[t] / openTotal : 1 / max(openCount, 1)
        }
    }

    public struct Mix: Sendable {
        public var gatesOpen: [Float]
        /// Each open Thread's head distribution (T = 1).
        public var strandLM: [Int: [Float]]
        public var pLM: [Float]
        public var knn: [Int: Float]
        public var mixed: [Float]
        public var lambda: Float
    }

    /// Mixes the open Threads' head logits (`logits[t]` for every open t) with retrieval.
    public static func mix(_ pool: Pool, logits: [Int: [Float]], lambda: Float, vocabularySize: Int) -> Mix {
        let gPrime = renormalised(pool)
        var strandLM: [Int: [Float]] = [:]
        var pLM = [Float](repeating: 0, count: vocabularySize)
        for t in pool.openIndices {
            guard let raw = logits[t] else { continue }
            let p = CitationMixer.softmax(raw)
            strandLM[t] = p
            let g = gPrime[t]
            for i in 0..<min(p.count, vocabularySize) { pLM[i] += g * p[i] }
        }
        var knn: [Int: Float] = [:]
        for member in pool.members { knn[member.hit.value, default: 0] += member.weight }
        let l = min(max(lambda, 0), 1)
        return Mix(gatesOpen: gPrime, strandLM: strandLM, pLM: pLM, knn: knn,
                   mixed: CitationMixer.mix(pLM: pLM, knn: knn, lambda: l), lambda: l)
    }

    /// The token: argmax of the mixture at temperature 0, else a sample of the mixture built
    /// from tempered head distributions (as CitedGenerator samples).
    public static func choose(
        _ mix: Mix, pool: Pool, logits: [Int: [Float]], temperature: Float, topK: Int, rng: inout SplitMix64,
        age: (Int) -> Int64? = { _ in nil }
    ) -> Int {
        if temperature <= 0 { return CitationMixer.choose(mix.mixed, age: age) }
        var tempered = [Float](repeating: 0, count: mix.pLM.count)
        for t in pool.openIndices {
            guard let raw = logits[t] else { continue }
            let p = CitationMixer.softmax(raw, temperature: temperature)
            let g = mix.gatesOpen[t]
            for i in 0..<min(p.count, tempered.count) { tempered[i] += g * p[i] }
        }
        return CitationMixer.sample(CitationMixer.mix(pLM: tempered, knn: mix.knn, lambda: mix.lambda), topK: topK, rng: &rng)
    }

    /// What each Thread supplied to `token`.
    public static func shares(
        _ mix: Mix, pool: Pool, token: Int, names: [String], threadIDs: [String?]
    ) -> [StrandShare] {
        let total = token < mix.mixed.count ? mix.mixed[token] : 0
        var knnByStrand = [Float](repeating: 0, count: pool.gates.count)
        for member in pool.members where member.hit.value == token { knnByStrand[member.strand] += member.weight }
        return pool.gates.indices.map { t in
            let p = mix.strandLM[t].map { token < $0.count ? $0[token] : 0 }
            let part = mix.lambda * knnByStrand[t] + (1 - mix.lambda) * mix.gatesOpen[t] * (p ?? 0)
            return StrandShare(
                strand: names[t], threadID: threadIDs[t], gate: pool.gates[t], open: pool.open[t], bestScore: pool.best[t],
                lmProb: p, lmEntropy: nil, knn: knnByStrand[t],
                share: total > 0 ? part / total : pool.gates[t])
        }
    }

    /// How spread the retrieval weight is across Threads, in nats.
    public static func threadEntropy(_ gates: [Float]) -> Float {
        let total = gates.reduce(0, +)
        guard total > 0 else { return 0 }
        return CitationMath.entropy(gates.map { $0 / total })
    }
}

// MARK: - Threads as experts

extension BraidMixer {
    /// One Thread's own kNN-LM at one position.
    public struct Expert: Sendable {
        public var hits: [StrandHit]
        /// softmax(score/τ) over this Thread's own hits.
        public var weights: [Float]
        public var knn: [Int: Float]
        /// The umbrella head's distribution from this Thread's hidden state; nil when it was not asked.
        public var pLM: [Float]?
        /// λ_t = λ · lambdaScale: 1 unless the Thread set its own, 0 for the commons (no retrieval).
        public var lambdaScale: Float

        public init(hits: [StrandHit], tau: Float, logits: [Float]?, lambdaScale: Float = 1) {
            self.hits = hits
            weights = CitationMixer.weights(hits.map(\.score), tau: tau)
            var knn: [Int: Float] = [:]
            for (i, hit) in hits.enumerated() { knn[hit.value, default: 0] += weights[i] }
            self.knn = knn
            pLM = logits.map { CitationMixer.softmax($0) }
            self.lambdaScale = lambdaScale
        }

        /// This Thread's λ under the request's.
        public func lambda(_ lambda: Float) -> Float { min(max(lambda * lambdaScale, 0), 1) }

        /// p_t(y) = λ_t·p_knn,t(y) + (1−λ_t)·p_lm,t(y); nil without the head.
        public func probability(of token: Int, lambda: Float) -> Float? {
            guard let pLM else { return nil }
            let l = self.lambda(lambda)
            return l * (knn[token] ?? 0) + (1 - l) * (token < pLM.count ? pLM[token] : 0)
        }

        /// What retrieval alone guarantees of p_t(y), for a Thread whose head was not asked.
        public func lowerBound(of token: Int, lambda: Float) -> Float {
            self.lambda(lambda) * (knn[token] ?? 0)
        }

        public var bestScore: Float? { hits.map(\.score).max() }
    }

    /// The posterior over Threads from accumulated log-likelihoods under a uniform prior.
    public static func posterior(_ logLikelihoods: [Double]) -> [Float] {
        guard let top = logLikelihoods.max(), top.isFinite else {
            return logLikelihoods.map { _ in 1 / Float(max(1, logLikelihoods.count)) }
        }
        let exps = logLikelihoods.map { exp($0 - top) }
        let total = exps.reduce(0, +)
        return exps.map { Float($0 / total) }
    }

    public struct ExpertMix: Sendable {
        /// The full posterior, and the one the mixture used (renormalised over open Threads).
        public var posterior: [Float]
        public var weights: [Float]
        public var open: [Bool]
        public var pLM: [Float]
        public var knn: [Int: Float]
        public var mixed: [Float]
        /// The λ the mixture used overall, Σ_t w_t·λ_t: the request's when every Thread uses it.
        public var lambda: Float
        /// The request's λ, which each Thread scales by its own.
        public var baseLambda: Float
        /// Each Thread's weight in the head part and in the retrieval part: w_t·(1−λ_t)/(1−λ) and
        /// w_t·λ_t/λ, which are the gate's weights when every Thread's λ is the same.
        public var headWeights: [Float]
        public var knnWeights: [Float]
    }

    /// p = Σ_t w_t · (λ_t·p_knn,t + (1−λ_t)·p_lm,t) over the open Threads, w renormalised over them.
    /// When every open Thread's λ_t is the same this is λ·Σ w·p_knn + (1−λ)·Σ w·p_lm, computed so.
    public static func mix(experts: [Expert], posterior: [Float], open: [Bool], lambda: Float, vocabularySize: Int) -> ExpertMix {
        let usable = experts.indices.map { open[$0] && experts[$0].pLM != nil }
        let total = experts.indices.reduce(Float(0)) { $0 + (usable[$1] ? posterior[$1] : 0) }
        let count = Float(usable.filter { $0 }.count)
        let weights = experts.indices.map { t -> Float in
            guard usable[t] else { return 0 }
            return total > 0 ? posterior[t] / total : 1 / max(count, 1)
        }
        let lambdas = experts.map { $0.lambda(lambda) }
        let active = experts.indices.filter { weights[$0] > 0 }
        var l = min(max(lambda, 0), 1)
        var headWeights = weights
        var knnWeights = weights
        if let first = active.first, active.allSatisfy({ lambdas[$0] == lambdas[first] }) {
            l = lambdas[first]
        } else if !active.isEmpty {
            l = min(max(active.reduce(Float(0)) { $0 + weights[$1] * lambdas[$1] }, 0), 1)
            headWeights = experts.indices.map { t in l < 1 && weights[t] > 0 ? weights[t] * (1 - lambdas[t]) / (1 - l) : 0 }
            knnWeights = experts.indices.map { t in l > 0 && weights[t] > 0 ? weights[t] * lambdas[t] / l : 0 }
        }
        var pLM = [Float](repeating: 0, count: vocabularySize)
        var knn: [Int: Float] = [:]
        for t in experts.indices where weights[t] > 0 {
            let h = headWeights[t]
            if h > 0, let p = experts[t].pLM { for i in 0..<min(p.count, vocabularySize) { pLM[i] += h * p[i] } }
            let q = knnWeights[t]
            if q > 0 { for (token, value) in experts[t].knn { knn[token, default: 0] += q * value } }
        }
        return ExpertMix(posterior: posterior, weights: weights, open: usable, pLM: pLM, knn: knn,
                         mixed: CitationMixer.mix(pLM: pLM, knn: knn, lambda: l), lambda: l, baseLambda: lambda,
                         headWeights: headWeights, knnWeights: knnWeights)
    }

    public static func choose(
        _ mix: ExpertMix, experts: [Expert], logits: [Int: [Float]], temperature: Float, topK: Int, rng: inout SplitMix64,
        age: (Int) -> Int64? = { _ in nil }
    ) -> Int {
        if temperature <= 0 { return CitationMixer.choose(mix.mixed, age: age) }
        var tempered = [Float](repeating: 0, count: mix.pLM.count)
        for t in experts.indices where mix.headWeights[t] > 0 {
            guard let raw = logits[t] else { continue }
            let p = CitationMixer.softmax(raw, temperature: temperature)
            for i in 0..<min(p.count, tempered.count) { tempered[i] += mix.headWeights[t] * p[i] }
        }
        return CitationMixer.sample(CitationMixer.mix(pLM: tempered, knn: mix.knn, lambda: mix.lambda), topK: topK, rng: &rng)
    }

    /// What each Thread supplied to `token`: share_t = w_t · p_t(token) / p(token). `memory` and
    /// `backs` are the braided gate's two parts, recorded when that gate weighed the Threads. Each
    /// Thread's head entropy is left for the generator to fill from MixtureStats. With a commons
    /// strand, every Thread's lift over it is recorded; credit is the generator's, once the words
    /// the tokens belong to are whole (`TokenRoles`).
    public static func shares(
        _ mix: ExpertMix, experts: [Expert], token: Int, names: [String], threadIDs: [String?], memory: [Float]? = nil,
        backs: [Float]? = nil, commons: Int? = nil
    ) -> [StrandShare] {
        let total = token < mix.mixed.count ? mix.mixed[token] : 0
        let base = commons.flatMap { experts.indices.contains($0) ? experts[$0].probability(of: token, lambda: mix.baseLambda) : nil }
        let parts = experts.indices.map { t -> Float in
            let expert = experts[t]
            let l = expert.lambda(mix.baseLambda)
            let p = expert.pLM.map { token < $0.count ? $0[token] : 0 } ?? 0
            return mix.weights[t] * (l * (expert.knn[token] ?? 0) + (1 - l) * p)
        }
        return experts.indices.map { t in
            let expert = experts[t]
            let w = mix.weights[t]
            let p = expert.pLM.map { token < $0.count ? $0[token] : 0 }
            var share = StrandShare(
                strand: names[t], threadID: threadIDs[t], gate: mix.posterior[t], open: mix.open[t], bestScore: expert.bestScore,
                lmProb: p, lmEntropy: nil, knn: w * (expert.knn[token] ?? 0),
                share: total > 0 ? parts[t] / total : mix.weights[t])
            share.memory = memory.map { $0[t] }
            share.backs = backs.map { $0[t] }
            share.alone = expert.probability(of: token, lambda: mix.baseLambda)
            if t != commons, let base, let alone = share.alone {
                share.lift = Float(log(Double(max(alone, 1e-30))) - log(Double(max(base, 1e-30))))
            }
            return share
        }
    }

    /// The Threads the umbrella asks for hidden states: posterior at the floor, and always the largest.
    public static func gate(posterior: [Float], floor: Float) -> [Bool] { gate(posterior, floor: floor) }

    /// The mixture's `count` likeliest tokens, each split by Thread as a chosen token is.
    public static func candidates(_ mix: ExpertMix, experts: [Expert], count: Int = 3, text: (Int) -> String) -> [TokenCandidate] {
        var top: [(token: Int, prob: Float)] = []
        for (token, prob) in mix.mixed.enumerated() where prob > 0 {
            guard top.count < count || prob > (top.last?.prob ?? 0) else { continue }
            let at = top.firstIndex { prob > $0.prob } ?? top.count
            top.insert((token, prob), at: at)
            if top.count > count { top.removeLast() }
        }
        return top.map { candidate in
            let parts = experts.indices.map { t -> Float in
                guard mix.weights[t] > 0, let p = experts[t].probability(of: candidate.token, lambda: mix.baseLambda) else { return 0 }
                return mix.weights[t] * p / candidate.prob
            }
            return TokenCandidate(token: candidate.token, text: text(candidate.token), prob: candidate.prob, parts: parts)
        }
    }
}

// MARK: - The braided gate

extension BraidMixer {
    /// What the braided gate remembers of a text: which Thread has been predicting it, and how
    /// much of it anyone predicted.
    public struct GateState: Sendable, Equatable {
        /// Sums to 1. Even before any token is seen.
        public var memory: [Float]
        /// Per Thread, how much of the recent text it predicted (0 none, 1 all).
        public var credibility: [Float]
        /// Tokens observed, and how many each Thread predicted (gave at least the evidence floor).
        public var seen: Int
        public var predicted: [Int]
        /// The commons strand, when the braid has one: the memory starts on it, and a Thread's
        /// evidence is its lift over it.
        public var commons: Int?

        public init(threads: Int, commons: Int? = nil, prior: Float = 0.9) {
            let n = max(0, threads)
            memory = [Float](repeating: n > 0 ? 1 / Float(n) : 0, count: n)
            credibility = [Float](repeating: 0, count: n)
            seen = 0
            predicted = [Int](repeating: 0, count: n)
            if let commons, (0..<n).contains(commons) {
                self.commons = commons
                let p = n > 1 ? min(max(prior, 0), 1) : 1
                memory = (0..<n).map { $0 == commons ? p : (1 - p) / Float(n - 1) }
            }
        }

        /// The Thread with the most memory; the first of equals.
        public var leader: Int {
            var best = 0
            for t in memory.indices where memory[t] > memory[best] { best = t }
            return best
        }
    }

    /// The floor a Thread's gate must reach to be asked for its hidden state: `floor` is what
    /// two Threads use, and it shrinks as Threads are added so an even gate stays above it.
    public static func floor(_ floor: Float, threads: Int) -> Float {
        floor * 2 / Float(max(2, threads))
    }

    /// One token was fixed; `likelihoods[t]` is what Thread t alone gave it.
    ///
    /// Evidence is bounded: a probability counts between the floor and the ceiling, so a token
    /// moves the memory by at most ceiling/floor to 1, and a token every Thread predicted moves
    /// nothing. It counts in proportion to the best Thread's credibility. With variable share a
    /// Thread gives up weight only on a token it failed to predict, so a lead holds through text
    /// every Thread predicts and fades through text none does.
    public static func observe(_ state: inout GateState, likelihoods: [Float], gate: BraidGate) {
        if let commons = state.commons {
            observeLift(&state, likelihoods: likelihoods, commons: commons, gate: gate)
            return
        }
        let n = state.memory.count
        guard n > 0, likelihoods.count == n else { return }
        let floor = max(gate.evidenceFloor, 1e-6)
        let ceiling = max(gate.evidenceCeiling, floor)
        let hits = likelihoods.map { min(max($0, floor), ceiling) }
        let span = log(ceiling / floor)
        // 0: predicted it. 1: did not.
        let losses = hits.map { span > 0 ? log(ceiling / $0) / span : 0 }
        state.seen += 1
        for t in 0..<n where likelihoods[t] >= floor { state.predicted[t] += 1 }
        let rate = min(max(gate.credibilityRate, 0), 1)
        for t in 0..<n { state.credibility[t] = (1 - rate) * state.credibility[t] + rate * (1 - losses[t]) }
        guard n > 1 else {
            state.memory = [1]
            return
        }
        let weight = gate.credibility ? (state.credibility.max() ?? 0) : 1
        var updated = (0..<n).map { state.memory[$0] * pow(hits[$0], weight) }
        let total = updated.reduce(0, +)
        updated = total > 0 ? updated.map { $0 / total } : [Float](repeating: 1 / Float(n), count: n)
        let alpha = min(max(gate.shareRate, 0), 1)
        switch gate.share {
        case .fixed:
            state.memory = updated.map { (1 - alpha) * $0 + alpha / Float(n) }
        case .variable:
            let given = (0..<n).map { updated[$0] * (1 - pow(1 - alpha, losses[$0])) }
            let pool = given.reduce(0, +)
            state.memory = (0..<n).map { updated[$0] - given[$0] + (pool - given[$0]) / Float(n - 1) }
        }
    }

    /// The lift gate: with a commons strand, a Thread's evidence for a token is what it gave the
    /// token over what the commons gave it, p_t(x)/p_c(x), bounded to one token counting at most
    /// ceiling/floor to 1 either way; the commons' own evidence is 1. A Thread that knows no more
    /// than the base model moves nothing, so the memory stays where it started, on the commons.
    /// Credibility is how much of the recent text the best Thread out-predicted the commons, and a
    /// Thread gives up weight (variable share) only on a token it predicted worse than the commons.
    static func observeLift(_ state: inout GateState, likelihoods: [Float], commons c: Int, gate: BraidGate) {
        let n = state.memory.count
        guard n > 0, likelihoods.count == n else { return }
        let floor = max(gate.evidenceFloor, 1e-6)
        let reach = max(gate.evidenceCeiling / floor, 1 + 1e-6)
        let span = log(reach)
        let base = max(likelihoods[c], 1e-12)
        let ratios = (0..<n).map { t -> Float in t == c ? 1 : min(max(likelihoods[t] / base, 1 / reach), reach) }
        let gains = ratios.map { max(0, log($0)) / span }
        let losses = ratios.map { max(0, -log($0)) / span }
        state.seen += 1
        for t in 0..<n where likelihoods[t] >= floor { state.predicted[t] += 1 }
        let rate = min(max(gate.credibilityRate, 0), 1)
        for t in 0..<n { state.credibility[t] = (1 - rate) * state.credibility[t] + rate * gains[t] }
        guard n > 1 else {
            state.memory = [1]
            return
        }
        let weight = gate.credibility ? ((0..<n).filter { $0 != c }.map { state.credibility[$0] }.max() ?? 0) : 1
        var updated = (0..<n).map { state.memory[$0] * pow(ratios[$0], weight) }
        let total = updated.reduce(0, +)
        updated = total > 0 ? updated.map { $0 / total } : state.memory
        let alpha = min(max(gate.shareRate, 0), 1)
        switch gate.share {
        case .fixed:
            state.memory = updated.map { (1 - alpha) * $0 + alpha / Float(n) }
        case .variable:
            let given = (0..<n).map { updated[$0] * (1 - pow(1 - alpha, losses[$0])) }
            let pool = given.reduce(0, +)
            state.memory = (0..<n).map { updated[$0] - given[$0] + (pool - given[$0]) / Float(n - 1) }
        }
    }

    /// The leader's top retrieved token, the one agreement asks the other Threads to back.
    static func candidate(experts: [Expert], leader: Int) -> Int? {
        guard experts.indices.contains(leader) else { return nil }
        return experts[leader].knn.max { a, b in a.value != b.value ? a.value < b.value : a.key > b.key }?.key
    }

    /// How much of each Thread's retrieval backs the leader's top retrieved token: 1 for the
    /// leader, and 0 everywhere when the leader retrieved nothing.
    public static func agreement(experts: [Expert], leader: Int) -> [Float] {
        guard experts.indices.contains(leader) else { return experts.map { _ in 0 } }
        guard let candidate = candidate(experts: experts, leader: leader) else { return experts.indices.map { $0 == leader ? 1 : 0 } }
        return experts.indices.map { t in t == leader ? 1 : min(max(experts[t].knn[candidate] ?? 0, 0), 1) }
    }


    /// g_t ∝ m_t + o_t·(m_leader − m_t): a Thread that backs the leader's candidate in full stands
    /// beside it, one that does not keeps its memory.
    ///
    /// With the trajectory in the gate, `trace[t]` is how far Thread t's retrieval has followed one
    /// of its own documents through the text (0 to 1). `lift` lifts a Thread only as far as it
    /// traces the text as well as the leader, r_t = 1 − max(0, trace_lead − trace_t); `gate`
    /// multiplies every weight by exp(β·(trace_t − the mean trace)). Off, the gate is as it was.
    public static func weights(state: GateState, agreement: [Float], trace: [Float]? = nil, gate: BraidGate) -> [Float] {
        let memory = state.memory
        guard !memory.isEmpty else { return [] }
        var gates = memory
        let traces = trace.flatMap { $0.count == memory.count ? $0 : nil }
        let lifts = gate.trajectory == .lift || gate.trajectory == .both
        if gate.agreement, agreement.count == memory.count {
            let leader = state.leader
            for t in memory.indices where t != leader {
                if lifts, let traces {
                    let reach = 1 - max(0, traces[leader] - traces[t])
                    gates[t] = memory[t] + min(max(agreement[t], 0), 1) * reach * (memory[leader] - memory[t])
                } else {
                    gates[t] = memory[t] + min(max(agreement[t], 0), 1) * (memory[leader] - memory[t])
                }
            }
        }
        if gate.trajectory == .gate || gate.trajectory == .both, let traces {
            let mean = traces.reduce(0, +) / Float(traces.count)
            for t in gates.indices { gates[t] *= exp(gate.trajectoryBeta * (traces[t] - mean)) }
        }
        let total = gates.reduce(0, +)
        return total > 0 ? gates.map { $0 / total } : gates.map { _ in 1 / Float(gates.count) }
    }

    /// Each Thread's trace, read by the umbrella from the length of its chain with the standard
    /// rule (a node's own `trace` is not trusted); 0 for a Thread that sent no trajectory.
    public static func traces(_ trajectories: [StrandTrajectory?], rule: TrajectoryRule = .standard) -> [Float] {
        trajectories.map { $0.map { rule.trace(length: $0.length) } ?? 0 }
    }

    /// Which Threads the prompt asks for their hidden state. Every Thread, unless the gate asks by
    /// manner: then a Thread whose manner reaches the floor, or has none yet; every Thread when
    /// none is left.
    public static func asked(manner: [Float?], gate: BraidGate) -> [Bool] {
        guard gate.ask == .manner else { return manner.map { _ in true } }
        let asked = manner.map { $0.map { $0 >= gate.askFloor } ?? true }
        return asked.contains(true) ? asked : manner.map { _ in true }
    }
}
