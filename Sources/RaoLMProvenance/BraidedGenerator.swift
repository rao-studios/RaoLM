//
//  BraidedGenerator.swift
//  RaoLMProvenance
//
//  WHAT: The umbrella of a braid generating with citations. Every position: each live Thread
//        runs its headless transformer and returns its retrieval hits; the umbrella gates the
//        Threads (`BraidGating` says how), asks only the open ones for their hidden state,
//        applies the one shared final norm and tied head to each, mixes the Threads' own
//        kNN-LMs by their gates, chooses a token, and sends it to every Thread. Each trace
//        records what every Thread supplied.
//        With an umbrella pack that has a base model, the commons strand joins the Threads: the
//        base model the nodes started from, always asked, retrieving nothing and cited to
//        nothing. The gate then weighs a Thread by what it knows beyond the commons (its lift),
//        from a memory that starts on the commons, so what the base already knew is no one's.
//  OUT:  A CitedGeneration over one partition table that holds every Thread's rows, each
//        partition carrying its Thread, so spans, citations and verification run unchanged.
//  PIN:  Requests go to every Thread before the umbrella waits on any. Prompt positions are
//        scored in one batch per Thread, as a model's prefill is, so a braid of one Thread
//        reproduces CitedGenerator number for number under every gating. A Thread's own λ and
//        τ (its calibration) apply exactly as CitedGenerator applies them to that Thread alone.
//

import Foundation
import MLX
import RaoLMCore
import RaoLMModel

public struct BraidRequest: Sendable {
    public var promptTokens: [Int]
    public var promptText: String
    public var promptSource: SourceAddress?
    public var params: GenerationParameters
    /// The gate a Thread must reach to be asked for its hidden state, for two Threads; the
    /// braided gate lowers it as Threads are added.
    public var gateFloor: Float
    public var gating: BraidGating
    /// The braided gate's settings.
    public var gate: BraidGate

    public init(
        promptTokens: [Int], promptText: String, promptSource: SourceAddress? = nil, params: GenerationParameters,
        gateFloor: Float = BraidMixer.defaultGateFloor, gating: BraidGating = BraidRequest.defaultGating, gate: BraidGate = BraidRequest.defaultGate
    ) {
        self.promptTokens = promptTokens
        self.promptText = promptText
        self.promptSource = promptSource
        self.params = params
        self.gateFloor = gateFloor
        self.gating = gating
        self.gate = gate
    }

    /// How Threads are weighed when a request does not say. The gate bench's rule kept the
    /// posterior; the user chose the braided gate (Docs/BRAID.md has both).
    public static let defaultGating: BraidGating = .braided

    /// The braided gate's settings when a request does not say: the trajectory in the gate both
    /// ways at β 4, and a prompt asking only the Threads its manner points to. The trajectory
    /// bench chose both, β 4 under its rule; the owner made it and asking by manner the default
    /// (2026-09-29, Docs/BRAID.md). `BraidGate()` stays the gate without them, and it is what a
    /// recorded gate without trajectory settings reads as.
    /// The trajectory both ways at β 4 and asking by manner (the owner's choice, 2026-09-29); agreement
    /// off (2026-10-01): it lifted every Thread whose retrieval backed a token they all write, such as
    /// the space before a number, and cost the owner its lead.
    public static let defaultGate = BraidGate(agreement: false, trajectory: .both, trajectoryBeta: 4, ask: .manner)
}

/// One position of a braided generation as it happens, for a live display.
public struct BraidStep: Sendable {
    public var trace: TokenTrace
    /// Which Threads were asked for their hidden state at this position.
    public var open: [String]
}

public enum BraidError: Error, CustomStringConvertible {
    case noStrands
    case vocabulary(strand: String, found: String, expected: String)
    case hiddenSize(strand: String, found: Int, expected: Int)
    case pack(strand: String, found: String?, expected: String?)

    public var description: String {
        switch self {
        case .noStrands: return "no Thread node has a live version yet"
        case .vocabulary(let strand, let found, let expected):
            return "\(strand) embeds with vocabulary \(found.prefix(12))…, the umbrella's head is \(expected.prefix(12))…"
        case .hiddenSize(let strand, let found, let expected):
            return "\(strand) sends \(found)-wide hidden states, the umbrella's head reads \(expected)"
        case .pack(let strand, let found, let expected):
            return "\(strand) runs the umbrella pack \(found.map { String($0.prefix(12)) + "…" } ?? "none"), the umbrella's is \(expected.map { String($0.prefix(12)) + "…" } ?? "none")"
        }
    }
}

public final class BraidedGenerator {
    public let links: [StrandLink]
    public let head: UmbrellaHead
    public let tokenizer: RaoTokenizer
    public let braid: BraidRef
    public let partitionsByRow: [Int: PartitionRef]
    public let sharedNgrams: Set<UInt64>
    public var timeout: TimeInterval = 120
    /// The commons strand's place among the links, when there is one.
    public let commons: Int?
    /// Per strand: its kNN temperature (nil: the request's) and the scale of its λ.
    let taus: [Float?]
    let lambdaScales: [Float]
    private lazy var thoughtReader = ThoughtReader(descriptors: links.map(\.descriptor))

    /// `packSHA256`: the umbrella pack every Thread must run when it has a trunk.
    public init(
        links: [StrandLink], head: UmbrellaHead, tokenizer: RaoTokenizer, gateFloor: Float = BraidMixer.defaultGateFloor,
        packSHA256: String? = nil
    ) throws {
        guard !links.isEmpty else { throw BraidError.noStrands }
        var strands: [BraidStrandRef] = []
        var partitions: [Int: PartitionRef] = [:]
        var ngrams = Set<UInt64>()
        var rowOffset = 0
        var entryOffset = 0
        for link in links {
            let d = link.descriptor
            guard d.vocabularySHA256 == head.vocabularySHA256 else {
                throw BraidError.vocabulary(strand: d.name, found: d.vocabularySHA256, expected: head.vocabularySHA256)
            }
            guard d.hiddenSize == head.hiddenSize else {
                throw BraidError.hiddenSize(strand: d.name, found: d.hiddenSize, expected: head.hiddenSize)
            }
            guard d.isCommons || d.packSHA256 == packSHA256 else {
                throw BraidError.pack(strand: d.name, found: d.packSHA256, expected: packSHA256)
            }
            strands.append(BraidStrandRef(
                name: d.name, label: d.label, threadID: d.threadID, version: d.version, manifest: d.manifest,
                rowOffset: rowOffset, rowCount: d.partitions.count, entryOffset: entryOffset, commons: d.isCommons ? true : nil))
            for partition in d.partitions {
                var global = partition
                global.row = rowOffset + partition.row
                global.threadID = partition.threadID ?? d.threadID
                partitions[global.row] = global
            }
            ngrams.formUnion(d.sharedNgrams.values)
            rowOffset += d.partitions.count
            entryOffset += d.indexEntries
        }
        self.links = links
        self.head = head
        self.tokenizer = tokenizer
        self.braid = BraidRef(vocabularySHA256: head.vocabularySHA256, gateFloor: gateFloor, strands: strands, packSHA256: packSHA256)
        self.partitionsByRow = partitions
        self.sharedNgrams = ngrams
        self.commons = links.firstIndex { $0.descriptor.isCommons }
        self.taus = links.map { $0.descriptor.isCommons ? nil : $0.descriptor.calibration?.tau }
        self.lambdaScales = links.map { $0.descriptor.isCommons ? 0 : ($0.descriptor.calibration?.lambdaScale ?? 1) }
    }

    public var names: [String] { braid.strands.map(\.name) }

    /// Strand t's own kNN-LM at one position, under its own τ and λ.
    func expert(_ t: Int, hits: [StrandHit], logits: [Float]?, params: GenerationParameters) -> BraidMixer.Expert {
        BraidMixer.Expert(hits: hits, tau: taus[t] ?? params.tau, logits: logits, lambdaScale: lambdaScales[t])
    }

    /// Which strands the prompt asks for their hidden state: the gate's rule over the Threads, and
    /// always the commons.
    func askedAtPrompt(manner: [Float?], gate: BraidGate) -> [Bool] {
        guard let commons else { return BraidMixer.asked(manner: manner, gate: gate) }
        var threads = manner
        threads.remove(at: commons)
        var asked = threads.isEmpty ? [] : BraidMixer.asked(manner: threads, gate: gate)
        asked.insert(true, at: commons)
        return asked
    }

    /// The gate's open strands, and always the commons.
    func openStrands(_ gates: [Float], floor: Float) -> [Bool] {
        var open = BraidMixer.gate(gates, floor: floor)
        if let commons, open.indices.contains(commons) { open[commons] = true }
        return open
    }

    /// How far each Thread thinks what the leading Thread thinks, from their descriptions of one
    /// position; nil when the commons leads or the leader was not read.
    func thought(_ descriptions: [[Float]?], leader: Int) -> [Float]? {
        guard leader != commons, descriptions.indices.contains(leader), let lead = descriptions[leader] else { return nil }
        return descriptions.indices.map { t in
            if t == leader { return 1 }
            guard t != commons, let own = descriptions[t] else { return 0 }
            return thoughtReader.agreement(t, leader, own, lead)
        }
    }

    /// Whether strand t's thought is read under `gate`.
    func reads(_ t: Int, gate: BraidGate) -> Bool {
        gate.thoughtAgreement && t != commons && thoughtReader.reads(t)
    }

    /// Strand t's hidden states at `positions`, with its cut states when its thought is read.
    func fetch(_ t: Int, session: String, positions: [Int], gate: BraidGate) -> StrandCall<StrandStates> {
        reads(t, gate: gate)
            ? links[t].states(session: session, positions: positions)
            : links[t].hidden(session: session, positions: positions).map { StrandStates(last: $0, cut: []) }
    }

    /// A MixtureStats position for a mix: its λ recorded only when it is not the request's.
    static func position(_ mix: BraidMixer.ExpertMix, lambda: Float) -> MixtureStats.Position {
        MixtureStats.Position(weights: mix.headWeights, knn: mix.knn, lambda: mix.lambda == min(max(lambda, 0), 1) ? nil : mix.lambda)
    }

    /// `onPrompt` sees the prompt's traces once every Thread has scored it; `onStep` sees each
    /// generated token as it is chosen. Throwing from either stops the generation.
    public func generate(
        _ request: BraidRequest, onPrompt: (([TokenTrace]) throws -> Void)? = nil, onStep: ((BraidStep) throws -> Void)? = nil
    ) throws -> CitedGeneration {
        switch request.gating {
        case .braided: return try generateByGate(request, onPrompt: onPrompt, onStep: onStep)
        case .posterior: return try generateByPosterior(request, onPrompt: onPrompt, onStep: onStep)
        case .retrieval: return try generateByRetrieval(request, onPrompt: onPrompt, onStep: onStep)
        }
    }

    /// Threads weighed token by token. Every Thread scores the prompt (its hidden state at every
    /// prompt position) and the gate's memory follows what each predicted. While generating, the
    /// memory holds (unless the gate counts the tokens the mixture chose), a Thread's gate moves
    /// with how far its retrieval backs the leader's candidate, and a Thread is asked for its
    /// hidden state only when its gate reaches the floor.
    func generateByGate(
        _ request: BraidRequest, onPrompt: (([TokenTrace]) throws -> Void)?, onStep: ((BraidStep) throws -> Void)?
    ) throws -> CitedGeneration {
        let params = request.params
        let prompt = request.promptTokens
        guard !prompt.isEmpty else { throw ProvenanceError.emptyPrompt }
        let session = "gen-" + UUID().uuidString.prefix(8).lowercased()
        let eos = tokenizer.eosTokenID
        let vocabulary = head.vocabSize
        let names = self.names
        let threadIDs = braid.strands.map(\.threadID)
        let lambda = params.lambda
        let gate = request.gate
        let floor = BraidMixer.floor(request.gateFloor, threads: links.count)
        var rng = SplitMix64(seed: params.seed)
        defer { for link in links { link.close(session: session) } }

        let opened = links.map { $0.open(session: session, tokens: prompt, k: params.k) }
        let promptSteps = try opened.map { try $0.wait(timeout: timeout) }
        let everyPosition = Array(0..<prompt.count)
        // Every Thread scores the prompt, unless the gate asks by manner: then the Threads whose
        // retrieval moves through their documents the way the prompt runs (all, when none does).
        // The commons is always asked.
        let asked = askedAtPrompt(manner: promptSteps.map { $0.last?.trajectory?.manner }, gate: gate)
        let stateCalls = links.indices.map { t in asked[t] ? fetch(t, session: session, positions: everyPosition, gate: gate) : nil }
        let promptStates = try stateCalls.map { try $0?.wait(timeout: timeout) }
        let promptArrays = promptStates.map { $0.map { head.logitsArray(hiddens: $0.last) } }
        let promptLogits = promptArrays.map { $0.map { head.rows($0) } }
        let promptThoughts = links.indices.map { t in reads(t, gate: gate) ? promptStates[t].flatMap { thoughtReader.describe(t, states: $0.cut) } : nil }

        var state = BraidMixer.GateState(threads: links.count, commons: commons, prior: gate.commonsPrior)
        var traces: [TokenTrace] = []
        var mixtures: [MixtureStats.Position] = []
        for j in 1..<prompt.count {
            let experts = links.indices.map { t in expert(t, hits: promptSteps[t][j - 1].hits, logits: promptLogits[t]?[j - 1], params: params) }
            let trajectories = promptSteps.map { $0[j - 1].trajectory }
            var backs = BraidMixer.agreement(experts: experts, leader: state.leader)
            let thinking = thought(promptThoughts.map { $0?[j - 1] }, leader: state.leader)
            if let thinking { backs = zip(backs, thinking).map { max($0, $1) } }
            let gates = BraidMixer.weights(state: state, agreement: backs, trace: BraidMixer.traces(trajectories), gate: gate)
            let mix = BraidMixer.mix(experts: experts, posterior: gates, open: asked, lambda: lambda, vocabularySize: vocabulary)
            traces.append(trace(mix: mix, experts: experts, token: prompt[j], index: j, forced: true, names: names, threadIDs: threadIDs,
                                memory: state.memory, backs: backs, trajectories: trajectories, thought: thinking))
            mixtures.append(Self.position(mix, lambda: lambda))
            // A Thread that was not asked is credited with what its retrieval alone gave the token.
            let likelihoods = experts.map { $0.probability(of: prompt[j], lambda: lambda) ?? $0.lowerBound(of: prompt[j], lambda: lambda) }
            BraidMixer.observe(&state, likelihoods: likelihoods, gate: gate)
        }
        if prompt.count > 1 {
            Self.fill(&traces, from: 0, with: MixtureStats.entropies(
                heads: promptArrays.map { $0?[0..<(prompt.count - 1)] }, positions: mixtures, lambda: lambda))
        }
        try onPrompt?(traces)

        var experts = links.indices.map { t in
            expert(t, hits: promptSteps[t][prompt.count - 1].hits, logits: promptLogits[t]?[prompt.count - 1], params: params)
        }
        var trajectories = promptSteps.map { $0[prompt.count - 1].trajectory }
        var logits: [Int: [Float]] = [:]
        var arrays: [Int: MLXArray] = [:]
        var descriptions: [[Float]?] = promptThoughts.map { $0?[prompt.count - 1] }
        for t in links.indices {
            if let rows = promptLogits[t] { logits[t] = rows[prompt.count - 1] }
            if let array = promptArrays[t] { arrays[t] = array[(prompt.count - 1)..<prompt.count] }
        }
        var backs = BraidMixer.agreement(experts: experts, leader: state.leader)
        var thinking = thought(descriptions, leader: state.leader)
        if let thinking { backs = zip(backs, thinking).map { max($0, $1) } }
        var gates = BraidMixer.weights(state: state, agreement: backs, trace: BraidMixer.traces(trajectories), gate: gate)
        var open = openStrands(gates, floor: floor)
        // A Thread the prompt did not ask that the gate opens now is asked for its last position.
        for t in links.indices where open[t] && logits[t] == nil {
            let states = try fetch(t, session: session, positions: [prompt.count - 1], gate: gate).wait(timeout: timeout)
            if let hidden = states.last.first {
                let array = head.logitsArray(hiddens: [hidden])
                arrays[t] = array
                logits[t] = head.rows(array)[0]
                experts[t] = expert(t, hits: experts[t].hits, logits: logits[t], params: params)
                if reads(t, gate: gate) { descriptions[t] = thoughtReader.describe(t, states: states.cut)?.first }
            }
        }
        var generated: [Int] = []
        var stoppedOnEOS = false
        while generated.count < params.maxTokens {
            let position = prompt.count - 1 + generated.count
            let mix = BraidMixer.mix(experts: experts, posterior: gates, open: open, lambda: lambda, vocabularySize: vocabulary)
            let token = BraidMixer.choose(mix, experts: experts, logits: logits, temperature: params.temperature, topK: params.topK, rng: &rng) { [experts] in
                self.citedAge($0, mix: mix, experts: experts)
            }
            if token == eos {
                stoppedOnEOS = true
                break
            }
            var step = [trace(mix: mix, experts: experts, token: token, index: prompt.count + generated.count, forced: false,
                              names: names, threadIDs: threadIDs, memory: state.memory, backs: backs, trajectories: trajectories,
                              thought: thinking)]
            Self.fill(&step, from: 0, with: MixtureStats.entropies(
                heads: links.indices.map { arrays[$0] }, positions: [Self.position(mix, lambda: lambda)], lambda: lambda))
            traces.append(step[0])
            generated.append(token)
            try onStep?(BraidStep(trace: step[0], open: links.indices.filter { mix.open[$0] }.map { names[$0] }))
            if gate.generatedEvidence {
                // A Thread that was not asked is credited with what its retrieval alone gave the token.
                let likelihoods = experts.map { $0.probability(of: token, lambda: lambda) ?? $0.lowerBound(of: token, lambda: lambda) }
                BraidMixer.observe(&state, likelihoods: likelihoods, gate: gate)
            }
            guard generated.count < params.maxTokens else { break }
            let advanced = links.map { $0.advance(session: session, token: token, k: params.k) }
            let steps = try advanced.map { try $0.wait(timeout: timeout) }
            let hits = steps.map(\.hits)
            trajectories = steps.map(\.trajectory)
            // Retrieval alone decides who is asked: it arrives with every advance, asked or not.
            let retrieved = links.indices.map { t in expert(t, hits: hits[t], logits: nil, params: params) }
            backs = BraidMixer.agreement(experts: retrieved, leader: state.leader)
            gates = BraidMixer.weights(state: state, agreement: backs, trace: BraidMixer.traces(trajectories), gate: gate)
            open = openStrands(gates, floor: floor)
            let calls = links.indices.map { t in open[t] ? fetch(t, session: session, positions: [position + 1], gate: gate) : nil }
            logits = [:]
            arrays = [:]
            descriptions = links.map { _ in nil }
            for t in links.indices {
                if let call = calls[t] {
                    let states = try call.wait(timeout: timeout)
                    guard let hidden = states.last.first else { continue }
                    let array = head.logitsArray(hiddens: [hidden])
                    arrays[t] = array
                    logits[t] = head.rows(array)[0]
                    if reads(t, gate: gate) { descriptions[t] = thoughtReader.describe(t, states: states.cut)?.first }
                }
            }
            // Thought can lift only a Thread already asked: its cut state comes with its hidden state.
            thinking = thought(descriptions, leader: state.leader)
            if let thinking {
                backs = zip(backs, thinking).map { max($0, $1) }
                gates = BraidMixer.weights(state: state, agreement: backs, trace: BraidMixer.traces(trajectories), gate: gate)
            }
            experts = links.indices.map { t in expert(t, hits: hits[t], logits: logits[t], params: params) }
        }
        return finish(traces: traces, prompt: prompt, generated: generated, stoppedOnEOS: stoppedOnEOS, request: request)
    }

    /// The entropies MixtureStats computed for `traces[start...]`, one position each: the trace's
    /// own, and each Thread's head entropy on its share.
    static func fill(_ traces: inout [TokenTrace], from start: Int, with stats: MixtureStats.Entropies) {
        for i in stats.lm.indices where traces.indices.contains(start + i) {
            traces[start + i].lmEntropy = stats.lm[i]
            traces[start + i].mixedEntropy = stats.mixed[i]
            guard let count = traces[start + i].strands?.count else { continue }
            for t in 0..<min(count, stats.strands[i].count) { traces[start + i].strands?[t].lmEntropy = stats.strands[i][t] }
        }
    }

    /// Each Thread is its own kNN-LM; the umbrella weighs them by their posterior given the tokens
    /// so far. Every Thread scores the prompt (its hidden state at every prompt position), then
    /// only Threads whose posterior reaches the floor are asked for hidden states. A closed Thread
    /// is charged what its retrieval alone guarantees; if that lifts it back over the floor, it is
    /// scored exactly for the positions it missed and reopens.
    func generateByPosterior(
        _ request: BraidRequest, onPrompt: (([TokenTrace]) throws -> Void)?, onStep: ((BraidStep) throws -> Void)?
    ) throws -> CitedGeneration {
        let params = request.params
        let prompt = request.promptTokens
        guard !prompt.isEmpty else { throw ProvenanceError.emptyPrompt }
        let session = "gen-" + UUID().uuidString.prefix(8).lowercased()
        let eos = tokenizer.eosTokenID
        let vocabulary = head.vocabSize
        let names = self.names
        let threadIDs = braid.strands.map(\.threadID)
        let lambda = params.lambda
        var rng = SplitMix64(seed: params.seed)
        defer { for link in links { link.close(session: session) } }

        let opened = links.map { $0.open(session: session, tokens: prompt, k: params.k) }
        let promptSteps = try opened.map { try $0.wait(timeout: timeout) }
        let promptHits = promptSteps.map { $0.map(\.hits) }
        let everyPosition = Array(0..<prompt.count)
        let hiddenCalls = links.map { $0.hidden(session: session, positions: everyPosition) }
        let promptArrays = try hiddenCalls.map { head.logitsArray(hiddens: try $0.wait(timeout: timeout)) }
        let promptLogits = promptArrays.map { head.rows($0) }

        var logLikelihood = [Double](repeating: 0, count: links.count)
        var traces: [TokenTrace] = []
        var mixtures: [MixtureStats.Position] = []
        let allOpen = links.map { _ in true }
        for j in 1..<prompt.count {
            let experts = links.indices.map { t in expert(t, hits: promptHits[t][j - 1], logits: promptLogits[t][j - 1], params: params) }
            let mix = BraidMixer.mix(experts: experts, posterior: BraidMixer.posterior(logLikelihood), open: allOpen, lambda: lambda,
                                     vocabularySize: vocabulary)
            traces.append(trace(mix: mix, experts: experts, token: prompt[j], index: j, forced: true, names: names, threadIDs: threadIDs,
                                trajectories: promptSteps.map { $0[j - 1].trajectory }))
            mixtures.append(Self.position(mix, lambda: lambda))
            for t in links.indices {
                logLikelihood[t] += log(Double(max(experts[t].probability(of: prompt[j], lambda: lambda) ?? 0, 1e-30)))
            }
        }
        if prompt.count > 1 {
            Self.fill(&traces, from: 0, with: MixtureStats.entropies(
                heads: promptArrays.map { $0[0..<(prompt.count - 1)] }, positions: mixtures, lambda: lambda))
        }
        try onPrompt?(traces)

        var experts = links.indices.map { t in
            expert(t, hits: promptHits[t][prompt.count - 1], logits: promptLogits[t][prompt.count - 1], params: params)
        }
        var logits: [Int: [Float]] = Dictionary(uniqueKeysWithValues: links.indices.map { ($0, promptLogits[$0][prompt.count - 1]) })
        var arrays: [Int: MLXArray] = Dictionary(uniqueKeysWithValues: links.indices.map {
            ($0, promptArrays[$0][(prompt.count - 1)..<prompt.count])
        })
        var trajectories = promptSteps.map { $0[prompt.count - 1].trajectory }
        var open = BraidMixer.gate(posterior: BraidMixer.posterior(logLikelihood), floor: request.gateFloor)
        // Positions a closed Thread was charged its retrieval bound for: (position, token, p_knn,t(token)).
        var missed = [[(position: Int, token: Int, knn: Float)]](repeating: [], count: links.count)
        var generated: [Int] = []
        var stoppedOnEOS = false
        while generated.count < params.maxTokens {
            let position = prompt.count - 1 + generated.count
            let posterior = BraidMixer.posterior(logLikelihood)
            let mix = BraidMixer.mix(experts: experts, posterior: posterior, open: open, lambda: lambda, vocabularySize: vocabulary)
            let token = BraidMixer.choose(mix, experts: experts, logits: logits, temperature: params.temperature, topK: params.topK, rng: &rng) { [experts] in
                self.citedAge($0, mix: mix, experts: experts)
            }
            if token == eos {
                stoppedOnEOS = true
                break
            }
            var step = [trace(mix: mix, experts: experts, token: token, index: prompt.count + generated.count, forced: false,
                              names: names, threadIDs: threadIDs, trajectories: trajectories)]
            Self.fill(&step, from: 0, with: MixtureStats.entropies(
                heads: links.indices.map { arrays[$0] }, positions: [Self.position(mix, lambda: lambda)], lambda: lambda))
            traces.append(step[0])
            generated.append(token)
            try onStep?(BraidStep(trace: step[0], open: links.indices.filter { mix.open[$0] }.map { names[$0] }))
            for t in links.indices {
                if let p = experts[t].probability(of: token, lambda: lambda) {
                    logLikelihood[t] += log(Double(max(p, 1e-30)))
                } else {
                    logLikelihood[t] += log(Double(max(experts[t].lowerBound(of: token, lambda: lambda), 1e-30)))
                    missed[t].append((position, token, experts[t].knn[token] ?? 0))
                }
            }
            guard generated.count < params.maxTokens else { break }
            let advanced = links.map { $0.advance(session: session, token: token, k: params.k) }
            let steps = try advanced.map { try $0.wait(timeout: timeout) }
            let hits = steps.map(\.hits)
            trajectories = steps.map(\.trajectory)
            let next = BraidMixer.gate(posterior: BraidMixer.posterior(logLikelihood), floor: request.gateFloor)
            for t in links.indices where next[t] && !missed[t].isEmpty {
                // Reopened: score it exactly where it was only bounded.
                let rows = head.logits(hiddens: try links[t].hidden(session: session, positions: missed[t].map(\.position)).wait(timeout: timeout))
                for (i, miss) in missed[t].enumerated() {
                    let pLM = CitationMixer.softmax(rows[i])
                    let l = min(max(lambda * lambdaScales[t], 0), 1)
                    let exact = l * miss.knn + (1 - l) * (miss.token < pLM.count ? pLM[miss.token] : 0)
                    logLikelihood[t] += log(Double(max(exact, 1e-30))) - log(Double(max(l * miss.knn, 1e-30)))
                }
                missed[t].removeAll()
            }
            open = BraidMixer.gate(posterior: BraidMixer.posterior(logLikelihood), floor: request.gateFloor)
            let newPosition = position + 1
            let calls = links.indices.map { t in open[t] ? links[t].hidden(session: session, positions: [newPosition]) : nil }
            logits = [:]
            arrays = [:]
            for t in links.indices {
                if let call = calls[t], let hidden = try call.wait(timeout: timeout).first {
                    let array = head.logitsArray(hiddens: [hidden])
                    arrays[t] = array
                    logits[t] = head.rows(array)[0]
                }
            }
            experts = links.indices.map { t in expert(t, hits: hits[t], logits: logits[t], params: params) }
        }
        return finish(traces: traces, prompt: prompt, generated: generated, stoppedOnEOS: stoppedOnEOS, request: request)
    }

    /// One position of Threads as experts: the union of every open Thread's hits, each weighted by
    /// its Thread's weight times its weight within that Thread.
    private func trace(
        mix: BraidMixer.ExpertMix, experts: [BraidMixer.Expert], token: Int, index position: Int, forced: Bool, names: [String],
        threadIDs: [String?], memory: [Float]? = nil, backs: [Float]? = nil, trajectories: [StrandTrajectory?]? = nil,
        thought: [Float]? = nil
    ) -> TokenTrace {
        var pooled: [(strand: Int, hit: StrandHit, weight: Float)] = []
        for t in experts.indices where mix.knnWeights[t] > 0 {
            for (i, hit) in experts[t].hits.enumerated() { pooled.append((t, hit, mix.knnWeights[t] * experts[t].weights[i])) }
        }
        pooled.sort { a, b in
            if a.weight != b.weight { return a.weight > b.weight }
            if a.strand != b.strand { return a.strand < b.strand }
            return a.hit.entry < b.hit.entry
        }
        let neighbours = pooled.enumerated().map { rank, member -> Neighbour in
            let strand = braid.strands[member.strand]
            let hit = member.hit
            return Neighbour(
                rank: rank + 1, entry: strand.entryOffset + hit.entry, score: hit.score, weight: member.weight, value: hit.value,
                matches: hit.value == token, key: TokenPosition(row: strand.rowOffset + hit.key.row, offset: hit.key.offset),
                cited: TokenPosition(row: strand.rowOffset + hit.cited.row, offset: hit.cited.offset),
                sourceLoss: hit.sourceLoss, sourceEntropy: hit.sourceEntropy)
        }
        var trace = TokenTrace(
            index: position, token: token, text: tokenizer.tokenText(token), isPrompt: forced,
            lmEntropy: 0, knnEntropy: CitationMath.entropy(mix.knn.values),
            mixedEntropy: 0, sourceEntropy: CitationMixer.sourceEntropy(neighbours),
            lmProb: token < mix.pLM.count ? mix.pLM[token] : 0, agreement: mix.knn[token] ?? 0,
            mixedProb: token < mix.mixed.count ? mix.mixed[token] : 0, lambda: mix.lambda, neighbours: neighbours)
        trace.strands = BraidMixer.shares(mix, experts: experts, token: token, names: names, threadIDs: threadIDs,
                                          memory: memory, backs: backs, commons: commons)
        if let thought, trace.strands?.count == thought.count {
            for t in thought.indices where t != commons { trace.strands?[t].thought = thought[t] }
        }
        Self.record(trajectories, in: &trace)
        trace.threadEntropy = BraidMixer.threadEntropy(mix.posterior)
        trace.candidates = BraidMixer.candidates(mix, experts: experts) { tokenizer.tokenText($0) }
        return trace
    }

    /// One retrieval pool ranked by raw cosine across Threads, each Thread gated by its share of it.
    func generateByRetrieval(
        _ request: BraidRequest, onPrompt: (([TokenTrace]) throws -> Void)?, onStep: ((BraidStep) throws -> Void)?
    ) throws -> CitedGeneration {
        let params = request.params
        let prompt = request.promptTokens
        guard !prompt.isEmpty else { throw ProvenanceError.emptyPrompt }
        let session = "gen-" + UUID().uuidString.prefix(8).lowercased()
        let eos = tokenizer.eosTokenID
        let vocabulary = head.vocabSize
        let names = self.names
        let threadIDs = braid.strands.map(\.threadID)
        var rng = SplitMix64(seed: params.seed)
        defer { for link in links { link.close(session: session) } }

        // Prefill every Thread at once, then gate every prompt position.
        let opened = links.map { $0.open(session: session, tokens: prompt, k: params.k) }
        let promptSteps = try opened.map { try $0.wait(timeout: timeout) }
        let promptHits = promptSteps.map { $0.map(\.hits) }
        var pools = (0..<prompt.count).map { position in
            BraidMixer.pool(promptHits.map { $0[position] }, k: params.k, tau: params.tau, floor: request.gateFloor)
        }
        // Only open Threads are asked for hidden states: one batch per Thread.
        var logits = [[Int: [Float]]](repeating: [:], count: prompt.count)
        let wanted = links.indices.map { t in (0..<prompt.count).filter { pools[$0].open[t] } }
        let hiddenCalls = links.indices.map { t in wanted[t].isEmpty ? nil : links[t].hidden(session: session, positions: wanted[t]) }
        for t in links.indices {
            guard let call = hiddenCalls[t] else { continue }
            let rows = head.logits(hiddens: try call.wait(timeout: timeout))
            for (i, position) in wanted[t].enumerated() { logits[position][t] = rows[i] }
        }

        var traces: [TokenTrace] = []
        for j in 1..<prompt.count {
            let trace = step(pool: pools[j - 1], logits: logits[j - 1], index: j, forced: prompt[j], params: params,
                             vocabulary: vocabulary, names: names, threadIDs: threadIDs, trajectories: promptSteps.map { $0[j - 1].trajectory },
                             rng: &rng)
            traces.append(trace)
        }
        try onPrompt?(traces)

        var generated: [Int] = []
        var stoppedOnEOS = false
        var pool = pools[prompt.count - 1]
        var current = logits[prompt.count - 1]
        var trajectories = promptSteps.map { $0[prompt.count - 1].trajectory }
        pools.removeAll()
        logits.removeAll()
        while generated.count < params.maxTokens {
            let trace = step(pool: pool, logits: current, index: prompt.count + generated.count, forced: nil, params: params,
                             vocabulary: vocabulary, names: names, threadIDs: threadIDs, trajectories: trajectories, rng: &rng)
            if trace.token == eos {
                stoppedOnEOS = true
                break
            }
            traces.append(trace)
            generated.append(trace.token)
            try onStep?(BraidStep(trace: trace, open: pool.openIndices.map { names[$0] }))
            guard generated.count < params.maxTokens else { break }
            let advanced = links.map { $0.advance(session: session, token: trace.token, k: params.k) }
            let steps = try advanced.map { try $0.wait(timeout: timeout) }
            trajectories = steps.map(\.trajectory)
            pool = BraidMixer.pool(steps.map(\.hits), k: params.k, tau: params.tau, floor: request.gateFloor)
            let position = prompt.count + generated.count - 1
            let calls = pool.openIndices.map { t in (t, links[t].hidden(session: session, positions: [position])) }
            current = [:]
            for (t, call) in calls {
                guard let hidden = try call.wait(timeout: timeout).first else { continue }
                current[t] = head.logits(hidden: hidden)
            }
        }
        return finish(traces: traces, prompt: prompt, generated: generated, stoppedOnEOS: stoppedOnEOS, request: request)
    }

    private func finish(
        traces: [TokenTrace], prompt: [Int], generated: [Int], stoppedOnEOS: Bool, request: BraidRequest
    ) -> CitedGeneration {
        var traces = traces
        let params = request.params
        if let commons {
            // The owner's blend: form is the commons', content each strand's as it supplied it.
            TokenRoles.assignCredit(&traces, commons: names[commons], texts: (prompt + generated).map { tokenizer.tokenText($0) })
        }
        let spans = CitationSpans.annotate(
            traces: &traces, partitions: partitionsByRow, sharedNgrams: sharedNgrams, threadID: nil,
            settings: CitationSpans.Settings(rankThreshold: params.rankThreshold, minSpanLength: params.minSpanLength))
        var rows = Set<Int>()
        for trace in traces {
            for neighbour in trace.neighbours {
                rows.insert(neighbour.cited.row)
                rows.insert(neighbour.key.row)
            }
        }
        var braid = self.braid
        braid.gating = request.gating
        braid.gate = request.gating == .braided ? request.gate : nil
        let manifest = braid.combinedManifest(tokenizerSHA256: tokenizer.tokenizerSHA256)
        return CitedGeneration(
            generationID: CitedGenerator.generationID(manifest: manifest, prompt: prompt, params: params), manifest: manifest,
            prompt: GenerationPrompt(text: request.promptText, tokens: prompt, source: request.promptSource,
                                     tokenTexts: prompt.map { tokenizer.tokenText($0) }),
            params: params, tokens: generated, text: tokenizer.decode(generated), stoppedOnEOS: stoppedOnEOS,
            partitions: rows.sorted().compactMap { partitionsByRow[$0] }, traces: traces, spans: spans,
            summary: CitationSpans.summary(traces: traces, spans: spans), braid: braid)
    }

    /// One position: mix, choose (or take the forced prompt token), trace.
    private func step(
        pool: BraidMixer.Pool, logits: [Int: [Float]], index position: Int, forced: Int?, params: GenerationParameters,
        vocabulary: Int, names: [String], threadIDs: [String?], trajectories: [StrandTrajectory?], rng: inout SplitMix64
    ) -> TokenTrace {
        let mix = BraidMixer.mix(pool, logits: logits, lambda: params.lambda, vocabularySize: vocabulary)
        let token = forced ?? BraidMixer.choose(mix, pool: pool, logits: logits, temperature: params.temperature, topK: params.topK, rng: &rng) {
            self.citedAge($0, pool: pool)
        }
        let neighbours = pool.members.enumerated().map { rank, member -> Neighbour in
            let strand = braid.strands[member.strand]
            let hit = member.hit
            return Neighbour(
                rank: rank + 1, entry: strand.entryOffset + hit.entry, score: hit.score, weight: member.weight, value: hit.value,
                matches: hit.value == token, key: TokenPosition(row: strand.rowOffset + hit.key.row, offset: hit.key.offset),
                cited: TokenPosition(row: strand.rowOffset + hit.cited.row, offset: hit.cited.offset),
                sourceLoss: hit.sourceLoss, sourceEntropy: hit.sourceEntropy)
        }
        var trace = TokenTrace(
            index: position, token: token, text: tokenizer.tokenText(token), isPrompt: forced != nil,
            lmEntropy: 0, knnEntropy: CitationMath.entropy(mix.knn.values),
            mixedEntropy: 0, sourceEntropy: CitationMixer.sourceEntropy(neighbours),
            lmProb: token < mix.pLM.count ? mix.pLM[token] : 0, agreement: mix.knn[token] ?? 0,
            mixedProb: token < mix.mixed.count ? mix.mixed[token] : 0, lambda: mix.lambda, neighbours: neighbours)
        trace.strands = BraidMixer.shares(mix, pool: pool, token: token, names: names, threadIDs: threadIDs)
        Self.record(trajectories, in: &trace)
        trace.threadEntropy = BraidMixer.threadEntropy(pool.gates)
        var one = [trace]
        Self.fill(&one, from: 0, with: MixtureStats.entropies(
            heads: pool.gates.indices.map { t in mix.strandLM[t] == nil ? nil : logits[t].map { MLXArray($0, [1, $0.count]) } },
            positions: [MixtureStats.Position(weights: mix.gatesOpen, knn: mix.knn)], lambda: params.lambda))
        return one[0]
    }

    /// When the text the braid would cite `token` to was created: of the Thread that supplied the
    /// most of it, the heaviest hit that predicted it. It breaks a near-tie (`CitationMixer.choose`).
    private func citedAge(_ token: Int, mix: BraidMixer.ExpertMix, experts: [BraidMixer.Expert]) -> Int64? {
        let parts = experts.indices.map { t -> Float in
            guard mix.weights[t] > 0 else { return 0 }
            return mix.weights[t] * (experts[t].probability(of: token, lambda: mix.baseLambda) ?? experts[t].lowerBound(of: token, lambda: mix.baseLambda))
        }
        guard let t = parts.indices.max(by: { parts[$0] < parts[$1] }), parts[t] > 0 else { return nil }
        let hits = experts[t].hits
        guard let i = hits.indices.filter({ hits[$0].value == token }).max(by: { experts[t].weights[$0] < experts[t].weights[$1] })
        else { return nil }
        return partitionsByRow[braid.strands[t].rowOffset + hits[i].cited.row]?.createdAt
    }

    /// The same for one retrieval pool: the heaviest pooled hit that predicted `token`.
    private func citedAge(_ token: Int, pool: BraidMixer.Pool) -> Int64? {
        pool.members.filter { $0.hit.value == token }.max { $0.weight < $1.weight }.flatMap { member in
            partitionsByRow[braid.strands[member.strand].rowOffset + member.hit.cited.row]?.createdAt
        }
    }

    /// Each Thread's trajectory at the position that predicts the token, on its share.
    private static func record(_ trajectories: [StrandTrajectory?]?, in trace: inout TokenTrace) {
        guard let trajectories, trace.strands?.count == trajectories.count else { return }
        for t in trajectories.indices { trace.strands?[t].trajectory = trajectories[t] }
    }
}

/// Reads each cited document from the Thread it lives on, for the citation verifier.
public struct BraidCorpusReader: CorpusReading {
    public let routes: [(prefix: String, reader: CorpusReading)]

    public init(routes: [(prefix: String, reader: CorpusReading)]) {
        self.routes = routes
    }

    public func partitionText(documentID: String, partitionIndex: Int) async throws -> String? {
        guard let route = routes.first(where: { documentID.hasPrefix($0.prefix) }) else { return nil }
        return try await route.reader.partitionText(documentID: documentID, partitionIndex: partitionIndex)
    }
}
