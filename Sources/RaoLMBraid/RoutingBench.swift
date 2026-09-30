//
//  RoutingBench.swift
//  RaoLMBraid
//
//  WHAT: How well the umbrella weighs Threads. Every node's live version is loaded in this
//        process and asked four kinds of prompt, each calling for a different weighing: a fact
//        one Thread holds, a prompt that moves from one Thread's fact to another's, a subject no
//        Thread holds, and text about nothing any Thread holds. Every arm (a way of weighing)
//        answers every prompt, and `decide` applies the rule that picks the default.
//  PIN:  The rule was fixed before any numbers (Docs/BRAID.md). Nothing is written to a node.
//        Each text is scored once more with every Thread asked, to know which Threads predicted
//        each token alone: that is what a token's shares are measured against.
//

import Foundation
import RaoLMCore
import RaoLMModel
import RaoLMProvenance

/// One way of weighing Threads under test.
public struct GateArm: Codable, Sendable, Equatable {
    public var name: String
    public var gating: BraidGating
    public var gate: BraidGate
    /// Whether the rule may choose it; the others are baselines.
    public var candidate: Bool

    public init(name: String, gating: BraidGating, gate: BraidGate = BraidGate(), candidate: Bool) {
        self.name = name
        self.gating = gating
        self.gate = gate
        self.candidate = candidate
    }

    /// The baseline first, then the candidates in the order a tie goes to.
    public static var all: [GateArm] {
        var noAgreement = BraidGate()
        noAgreement.agreement = false
        var fixed = BraidGate()
        fixed.share = .fixed
        var noCredibility = BraidGate()
        noCredibility.credibility = false
        var generated = BraidGate()
        generated.generatedEvidence = true
        return [
            GateArm(name: "posterior", gating: .posterior, candidate: false),
            GateArm(name: "retrieval", gating: .retrieval, candidate: false),
            GateArm(name: "braided", gating: .braided, candidate: true),
            GateArm(name: "no agreement", gating: .braided, gate: noAgreement, candidate: true),
            GateArm(name: "fixed share", gating: .braided, gate: fixed, candidate: true),
            GateArm(name: "no credibility", gating: .braided, gate: noCredibility, candidate: true),
            GateArm(name: "generated evidence", gating: .braided, gate: generated, candidate: true),
            GateArm(name: "trajectory (default)", gating: .braided, gate: BraidRequest.defaultGate, candidate: true),
        ]
    }
}

public enum GateBenchSet: String, Codable, Sendable, CaseIterable {
    case facts, pairs, unknown, generic
    /// Reported, not part of the rule.
    case paraphrases, withdrawn
    /// A fact one Thread tells in its own words about another's entity (a dataset braid), asked
    /// of the teller; reported.
    case crossed
}

public struct GateBenchRow: Codable, Sendable, Equatable {
    public var arm: String
    public var set: GateBenchSet
    public var example: String
    public var owner: String?
    public var expected: String?
    public var answer: String
    public var exact: Bool?
    /// The owner supplied at least half of the first answer token.
    public var ownerLeads: Bool?
    public var citedToOwner: Bool?
    /// Every Thread was asked for its hidden state at the first generated token.
    public var allAsked: Bool
    /// The largest gate at the first generated token.
    public var largestGate: Float
    /// Σ over tokens some Thread predicted alone of 1 − (half the distance between the token's
    /// shares and an even split among the Threads that predicted it), and how many such tokens.
    public var fidelitySum: Float
    public var fidelityTokens: Int
    /// Per Thread, the fraction of generated positions it was asked at.
    public var asked: [String: Float]

    public init(
        arm: String, set: GateBenchSet, example: String, owner: String?, expected: String?, answer: String, exact: Bool?,
        ownerLeads: Bool?, citedToOwner: Bool?, allAsked: Bool, largestGate: Float, fidelitySum: Float = 0,
        fidelityTokens: Int = 0, asked: [String: Float] = [:]
    ) {
        self.arm = arm
        self.set = set
        self.example = example
        self.owner = owner
        self.expected = expected
        self.answer = answer
        self.exact = exact
        self.ownerLeads = ownerLeads
        self.citedToOwner = citedToOwner
        self.allAsked = allAsked
        self.largestGate = largestGate
        self.fidelitySum = fidelitySum
        self.fidelityTokens = fidelityTokens
        self.asked = asked
    }
}

public struct GateBenchSummary: Codable, Sendable, Equatable {
    public var arm: String
    public var set: GateBenchSet
    public var prompts: Int
    public var exact: Float?
    public var ownerLeads: Float?
    public var citedToOwner: Float?
    public var allAsked: Float
    public var largestGate: Float
    public var fidelity: Float?
}

public struct GateBenchDecision: Codable, Sendable, Equatable {
    /// The arm that becomes the default; nil keeps the baseline.
    public var winner: String?
    public var eligible: [String]
    /// What each candidate that is not eligible failed.
    public var failures: [String: [String]]
    /// Share fidelity on facts and pairs, per candidate.
    public var fidelity: [String: Float]
    public var summary: String
}

public struct GateBenchReport: Codable, Sendable {
    public var createdAt: Date
    public var nodes: [String: Int]
    public var arms: [GateArm]
    public var rows: [GateBenchRow]
    public var summaries: [GateBenchSummary]
    public var decision: GateBenchDecision
}

public enum RoutingBench {
    /// The world a braid's nodes were fed from: world.json when it is there; otherwise every node
    /// directory, sorted, dealt an even world from `seed` (braids made before world.json).
    public static func world(layout: BraidLayout, seed: UInt64, documentsPerNode: Int = 40) throws -> MockWorld {
        if let record = MockWorld.Record.load(layout) {
            if let dataset = record.dataset { return try MockWorld(dataset: URL(fileURLWithPath: dataset.path), names: record.names) }
            return try MockWorld(names: record.names, seed: record.seed, shape: record.shape)
        }
        let names = ((try? FileManager.default.contentsOfDirectory(atPath: layout.nodes.path)) ?? []).filter(BraidLayout.isValidName).sorted()
        return try MockWorld(names: names, seed: seed, documentsPerNode: documentsPerNode)
    }

    /// Loads each node's live version from the braid's directory (no node process runs): the
    /// nodes `names` lists, or every node directory.
    public static func strands(
        layout: BraidLayout, tokenizer: RaoTokenizer, owner: String, names only: [String]? = nil
    ) throws -> (VocabularyPack, [ThreadStrand]) {
        let pointer = try JSONCoding.read(BraidVocabulary.Current.self, from: layout.vocabularies.appendingPathComponent("current.json"))
        let vocabulary = try VocabularyPack.load(from: layout.vocabulary(sha256: pointer.sha256))
        let names = only ?? ((try? FileManager.default.contentsOfDirectory(atPath: layout.nodes.path)) ?? []).sorted()
        var strands: [ThreadStrand] = []
        for name in names {
            let node = layout.node(name)
            guard let live = try? JSONCoding.read(LivePointer.self, from: node.live) else { continue }
            let directory = node.version(live.version)
            let version = try JSONCoding.read(NodeVersion.self, from: directory.appendingPathComponent(NodeVersion.fileName))
            let context = try RunContext.load(runDirectory: directory, epoch: version.epoch, allowWeakIndex: true, tokenizer: tokenizer)
            strands.append(ThreadStrand(name: name, label: name.prefix(1).uppercased() + name.dropFirst(), version: version.version,
                                        context: context, owner: owner))
        }
        return (vocabulary, strands)
    }

    public struct Sizes: Sendable {
        public var factsPerNode = 10
        public var pairsPerOrder = 10
        public var unknown = 10
        public var paraphrases = 10

        public init() {}
    }

    public static func run(
        layout: BraidLayout, tokenizer: RaoTokenizer, world: MockWorld, sizes: Sizes = Sizes(), arms: [GateArm] = GateArm.all,
        owner: String = "raolm-braid", progress: ((String) -> Void)? = nil
    ) throws -> GateBenchReport {
        let (vocabulary, strands) = try self.strands(layout: layout, tokenizer: tokenizer, owner: owner, names: world.names)
        guard !strands.isEmpty else { throw BraidSessionError.noLiveNodes }
        let links = strands.map { LocalStrandLink(strand: $0, vocabularySHA256: vocabulary.sha256) }
        let generator = try BraidedGenerator(links: links, head: UmbrellaHead(vocabulary: vocabulary), tokenizer: tokenizer)
        let present: [BraidExample.Node] = strands.map { strand in
            (name: strand.name, label: strand.label, threadID: strand.threadID,
             documents: MockFeeder.present(node: strand.name, world: world, layout: layout.node(strand.name)))
        }
        let nodes = present.map { (name: $0.name, label: $0.label, threadID: $0.threadID, documents: world.exclusive($0.documents)) }
        let withdrawn: [BraidExample] = strands.flatMap { strand -> [BraidExample] in
            let gone = FeedState.load(layout.node(strand.name)).withdrawn.compactMap(world.document(id:))
            return BraidExample.facts(nodes: [(name: strand.name, label: strand.label, threadID: strand.threadID, documents: gone)],
                                      tokenizer: tokenizer, perNode: 4)
        }
        let sets: [(GateBenchSet, [BraidExample])] = [
            (.facts, BraidExample.facts(nodes: nodes, tokenizer: tokenizer, perNode: sizes.factsPerNode)),
            (.pairs, BraidExample.pairs(nodes: nodes, tokenizer: tokenizer, perPair: sizes.pairsPerOrder)),
            (.unknown, BraidExample.unknowns(nodes: nodes, tokenizer: tokenizer, count: sizes.unknown)),
            (.generic, BraidExample.generic(tokenizer: tokenizer)),
            (.paraphrases, BraidExample.paraphrases(nodes: nodes, tokenizer: tokenizer, count: sizes.paraphrases)),
            (.withdrawn, withdrawn),
            (.crossed, BraidExample.crossed(nodes: present, links: world.crosslinks, tokenizer: tokenizer)),
        ]
        let threadOf = Dictionary(uniqueKeysWithValues: strands.map { ($0.name, $0.threadID) })
        let names = strands.map(\.name)
        var alone: [[Int]: [Int: [String: Float]]] = [:]

        /// What each Thread alone gave every token of `tokens`, every Thread asked.
        func scoredAlone(_ tokens: [Int]) throws -> [Int: [String: Float]] {
            if let cached = alone[tokens] { return cached }
            var params = GenerationParameters(tapLayer: links[0].descriptor.tapLayer, alpha: links[0].descriptor.alpha)
            params.maxTokens = 0
            let scored = try generator.generate(BraidRequest(promptTokens: tokens, promptText: "", params: params, gating: .braided,
                                                             gate: BraidGate()))
            var byPosition: [Int: [String: Float]] = [:]
            for trace in scored.traces {
                byPosition[trace.index] = Dictionary(uniqueKeysWithValues: (trace.strands ?? []).compactMap { share in
                    share.alone.map { (share.strand, $0) }
                })
            }
            alone[tokens] = byPosition
            return byPosition
        }

        var rows: [GateBenchRow] = []
        for arm in arms {
            progress?("arm \(arm.name)")
            for (set, examples) in sets {
                for example in examples {
                    let answerLength = example.expected.map { max(1, tokenizer.encode($0).count) }
                    var params = GenerationParameters(tapLayer: links[0].descriptor.tapLayer, alpha: links[0].descriptor.alpha)
                    params.maxTokens = answerLength.map { $0 + 2 } ?? 4
                    let generation = try generator.generate(BraidRequest(
                        promptTokens: example.promptTokens, promptText: example.promptText, params: params, gating: arm.gating, gate: arm.gate))
                    let generated = generation.traces.filter { !$0.isPrompt }
                    let first = generated.first
                    let shares = first?.strands ?? []
                    var row = GateBenchRow(
                        arm: arm.name, set: set, example: example.label, owner: example.node, expected: example.expected,
                        answer: generation.text, exact: example.expected.map { generation.text.hasPrefix($0) },
                        ownerLeads: example.node.map { owner in first?.dominantStrand(threshold: 0.5)?.strand == owner },
                        citedToOwner: example.node.map { owner in
                            first?.citations.first?.address.threadID != nil && first?.citations.first?.address.threadID == threadOf[owner] ?? nil
                        },
                        allAsked: !shares.isEmpty && shares.allSatisfy(\.open), largestGate: shares.map(\.gate).max() ?? 1)
                    for name in names {
                        let open = generated.filter { $0.strands?.first { $0.strand == name }?.open == true }.count
                        row.asked[name] = generated.isEmpty ? 0 : Float(open) / Float(generated.count)
                    }
                    if set == .facts || set == .pairs || set == .paraphrases || set == .crossed {
                        let scored = try scoredAlone(example.promptTokens + generation.tokens)
                        for trace in generation.traces {
                            guard let own = scored[trace.index], let fidelity = fidelity(trace.strands ?? [], alone: own) else { continue }
                            row.fidelitySum += fidelity
                            row.fidelityTokens += 1
                        }
                    }
                    rows.append(row)
                }
            }
        }
        let summaries = summarise(rows, arms: arms)
        return GateBenchReport(
            createdAt: .wholeSecond(), nodes: Dictionary(uniqueKeysWithValues: strands.map { ($0.name, $0.version) }), arms: arms,
            rows: rows, summaries: summaries, decision: decide(rows: rows, arms: arms))
    }

    /// 1 − half the distance between a token's shares and an even split among the Threads that
    /// gave it at least `threshold` alone; nil when none did.
    public static func fidelity(_ shares: [StrandShare], alone: [String: Float], threshold: Float = 0.5) -> Float? {
        let knew = Set(alone.filter { $0.value >= threshold }.map(\.key))
        guard !knew.isEmpty, !shares.isEmpty else { return nil }
        let even = 1 / Float(knew.count)
        var distance: Float = 0
        for share in shares { distance += abs(share.share - (knew.contains(share.strand) ? even : 0)) }
        for name in knew where !shares.contains(where: { $0.strand == name }) { distance += even }
        return 1 - distance / 2
    }

    public static func summarise(_ rows: [GateBenchRow], arms: [GateArm]) -> [GateBenchSummary] {
        func rate(_ values: [Bool?]) -> Float? {
            let known = values.compactMap { $0 }
            return known.isEmpty ? nil : Float(known.filter { $0 }.count) / Float(known.count)
        }
        var result: [GateBenchSummary] = []
        for arm in arms {
            for set in GateBenchSet.allCases {
                let mine = rows.filter { $0.arm == arm.name && $0.set == set }
                guard !mine.isEmpty else { continue }
                let tokens = mine.reduce(0) { $0 + $1.fidelityTokens }
                result.append(GateBenchSummary(
                    arm: arm.name, set: set, prompts: mine.count, exact: rate(mine.map(\.exact)), ownerLeads: rate(mine.map(\.ownerLeads)),
                    citedToOwner: rate(mine.map(\.citedToOwner)), allAsked: rate(mine.map(\.allAsked)) ?? 0,
                    largestGate: Stats.mean(mine.map(\.largestGate)),
                    fidelity: tokens > 0 ? mine.reduce(0) { $0 + $1.fidelitySum } / Float(tokens) : nil))
            }
        }
        return result
    }

    /// The rule, fixed before any numbers. A candidate is eligible if it meets every requirement:
    /// - facts: exact answers at most one prompt below the baseline, and the first answer token
    ///   led by and cited to its owner wherever the baseline's was;
    /// - pairs: the second fact's owner leads and is cited at its first answer token in ≥ 90%,
    ///   with exact answers within 10 points of its facts;
    /// - unknown subjects: every Thread asked at the first generated token in ≥ 90%, the largest
    ///   gate there ≤ 0.80 on average;
    /// - generic text: every Thread asked in every prompt, the largest gate ≤ 0.65 on average.
    /// The eligible candidate with the highest share fidelity on facts and pairs wins; within
    /// 0.02 of it, the one listed first. With none eligible the baseline stays.
    public static func decide(rows: [GateBenchRow], arms: [GateArm], baseline: String = "posterior") -> GateBenchDecision {
        func of(_ arm: String, _ set: GateBenchSet) -> [GateBenchRow] { rows.filter { $0.arm == arm && $0.set == set } }
        func count(_ values: [Bool?]) -> Int { values.filter { $0 == true }.count }
        let base = of(baseline, .facts)
        let baseExact = count(base.map(\.exact))
        let baseOwned = Set(base.filter { $0.ownerLeads == true && $0.citedToOwner == true }.map(\.example))

        var eligible: [String] = []
        var failures: [String: [String]] = [:]
        var fidelity: [String: Float] = [:]
        for arm in arms where arm.candidate {
            var failed: [String] = []
            let facts = of(arm.name, .facts)
            let pairs = of(arm.name, .pairs)
            let unknown = of(arm.name, .unknown)
            let generic = of(arm.name, .generic)
            if facts.isEmpty {
                failed.append("facts: no prompts")
            } else {
                let exact = count(facts.map(\.exact))
                if exact < baseExact - 1 { failed.append("facts: \(exact) exact, the baseline \(baseExact)") }
                let lost = facts.filter { baseOwned.contains($0.example) && !($0.ownerLeads == true && $0.citedToOwner == true) }
                if !lost.isEmpty { failed.append("facts: \(lost.count) answers no longer led by and cited to their owner") }
            }
            if pairs.isEmpty {
                failed.append("pairs: no prompts")
            } else {
                let moved = Float(pairs.filter { $0.ownerLeads == true && $0.citedToOwner == true }.count) / Float(pairs.count)
                if moved < 0.9 { failed.append(String(format: "pairs: the lead moved in %.0f%%, needs 90%%", moved * 100)) }
                let pairExact = Float(count(pairs.map(\.exact))) / Float(pairs.count)
                let factExact = facts.isEmpty ? 0 : Float(count(facts.map(\.exact))) / Float(facts.count)
                if pairExact < factExact - 0.1 {
                    failed.append(String(format: "pairs: %.0f%% exact against %.0f%% on facts", pairExact * 100, factExact * 100))
                }
            }
            for (set, rows, needAsked, maxGate) in [(GateBenchSet.unknown, unknown, Float(0.9), Float(0.8)), (.generic, generic, 1, 0.65)] {
                if rows.isEmpty {
                    failed.append("\(set.rawValue): no prompts")
                    continue
                }
                let asked = Float(rows.filter(\.allAsked).count) / Float(rows.count)
                if asked < needAsked { failed.append(String(format: "%@: every Thread asked in %.0f%%, needs %.0f%%", set.rawValue, asked * 100, needAsked * 100)) }
                let gate = Stats.mean(rows.map(\.largestGate))
                if gate > maxGate { failed.append(String(format: "%@: largest gate %.2f, needs ≤ %.2f", set.rawValue, gate, maxGate)) }
            }
            let scored = facts + pairs
            let tokens = scored.reduce(0) { $0 + $1.fidelityTokens }
            fidelity[arm.name] = tokens > 0 ? scored.reduce(0) { $0 + $1.fidelitySum } / Float(tokens) : 0
            if failed.isEmpty { eligible.append(arm.name) } else { failures[arm.name] = failed }
        }
        let best = eligible.compactMap { fidelity[$0] }.max()
        let winner = best.flatMap { best in eligible.first { (fidelity[$0] ?? 0) >= best - 0.02 } }
        let summary: String
        if let winner {
            summary = String(format: "%@ wins: eligible, share fidelity %.3f", winner, fidelity[winner] ?? 0)
        } else {
            summary = "no candidate is eligible: \(baseline) stays the default"
        }
        return GateBenchDecision(winner: winner, eligible: eligible, failures: failures, fidelity: fidelity, summary: summary)
    }
}
