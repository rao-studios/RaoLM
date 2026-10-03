//
//  ScaleBench.swift
//  RaoLMBraid
//
//  WHAT: Whether the braid still finds the right Thread as Threads are added (Docs/ARCHITECTURE.md,
//        "bench-scale: the rules"): one braid per N, each fed from an N-node dataset with the same
//        per-node shape, read on the same kinds of prompts, and what one more generated token
//        costs as N grows.
//  OUT:  One point per braid (facts, prompts and unknown subjects through the umbrella, the
//        Threads asked per token, seconds per token), and a report that judges the points against
//        the smallest N's with the rules N1 and N2 and fits cost against N.
//  PIN:  In process: every live version is loaded into this process (no node processes), so the
//        N forward passes of a token run one after another on one MLX stream; seconds per token is
//        the umbrella's serial cost, where a running braid's nodes would run in parallel. The
//        rules read rates and the commons' lead, never an absolute gate bar: the gate's floor and
//        the memory's start both scale with N (BraidMixer), so "largest Thread gate ≤ 0.35" would
//        get easier as N grows.
//

import Foundation
import MLX
import RaoLMCore
import RaoLMModel
import RaoLMProvenance
import RaoLMTraining

public struct ScaleTokenCost: Codable, Sendable, Equatable {
    /// The generic prompts timed, each generated to `shortTokens` and to `longTokens`.
    public var prompts: Int
    public var shortTokens: Int
    public var longTokens: Int
    /// Mean over the prompts of (t_long − t_short) / (generated_long − generated_short).
    public var secondsPerToken: Double
    /// What a prompt costs before its first generated token: t_short − shortTokens × secondsPerToken.
    public var secondsPerPrompt: Double
    /// Over the long runs' generated tokens: the Threads asked for a hidden state (the commons not
    /// counted), and the share of tokens where every Thread was asked.
    public var askedPerToken: Float
    public var allAskedRate: Float
}

public struct ScalePoint: Codable, Sendable, Equatable {
    public var braid: String
    public var nodes: Int
    public var names: [String]
    public var pack: String?
    public var dataset: String?
    public var datasetHash: String?
    /// Every node's facts (`factsPerNode` each, or every fact).
    public var facts: UmbrellaArmResult
    /// Two-fact, generic and commons prompts.
    public var prompts: UmbrellaArmResult
    /// Subjects nobody holds.
    public var unknown: UmbrellaArmResult
    /// On subjects nobody holds: the share where the commons holds the largest gate at the first generated token.
    public var unknownCommonsLeads: Float?
    /// Over the fact answers' tokens: the Threads asked for a hidden state.
    public var askedPerFactToken: Float?
    /// The node holding the fact → [answered exactly, asked].
    public var factsByNode: [String: [Int]]
    public var nodeResults: [UmbrellaNodeResult]
    public var tokenCost: ScaleTokenCost
    /// From `braid sync`'s footprint file, when the braid was built with it.
    public var peakFootprintBytes: [String: Int]?
    public var everyFact: Bool
    public var seconds: Double
}

public struct ScaleFit: Codable, Sendable, Equatable {
    public var slope: Double
    public var intercept: Double
    public var r2: Double
}

public struct ScaleEvaluation: Codable, Sendable, Equatable {
    public var rules: [TrajectoryRuleResult]
    public var qualifies: Bool
    public var secondsFit: ScaleFit?
    public var askedFit: ScaleFit?
    public var summary: String
}

public struct ScaleReport: Codable, Sendable {
    public var createdAt: Date
    /// Sorted by N.
    public var points: [ScalePoint]
    /// The N the others are judged against (the smallest).
    public var reference: Int
    public var evaluation: ScaleEvaluation
}

public enum ScaleBench {
    // MARK: - Running

    /// A built braid's live versions loaded in process, and bench-scale's sets for it.
    struct Prepared {
        var record: MockWorld.Record
        var world: MockWorld
        var pack: UmbrellaPack
        var strands: [ThreadStrand]
        var all: UmbrellaBench.Sets
        var facts: UmbrellaBench.Sets
        var prompts: UmbrellaBench.Sets
        var unknown: UmbrellaBench.Sets
        var umbrella: BraidUmbrella
        var links: [StrandLink]
        var generator: BraidedGenerator
        var threadOf: [String: String?]
    }

    static func prepare(
        layout: BraidLayout, tokenizer: RaoTokenizer, everyFact: Bool, factsPerNode: Int, pairs: Int, owner: String, bench: String,
        progress: ((String) -> Void)?
    ) throws -> Prepared {
        // Every live version runs in this one process: cap the freed memory MLX keeps for reuse as
        // each node does (bench-scale at N = 24 grew to 69 GB without it). Allocation only.
        Memory.cacheLimit = (HypervisorSettings().cacheLimitMB ?? 2048) * 1_048_576
        guard let record = MockWorld.Record.load(layout) else {
            throw BraidSessionError.io("no world.json under \(layout.root.path): build the braid first (raolm braid sync --dataset … --feed …)")
        }
        let world = try RoutingBench.world(layout: layout, seed: record.seed)
        let n = world.names.count
        let (pack, strands) = try RoutingBench.packStrands(layout: layout, tokenizer: tokenizer, owner: owner, names: world.names)
        guard strands.count == n else { throw BraidSessionError.io("\(strands.count) of \(n) nodes are live: every node must be live") }
        guard pack.hasBase else { throw BraidSessionError.io("\(bench) needs a pack with a base model (--preset base)") }

        var sizes = UmbrellaBench.Sizes()
        sizes.factsPerNode = everyFact ? .max : factsPerNode
        sizes.pairsPerOrder = max(1, Int((Double(pairs) / Double(max(1, n * (n - 1)))).rounded(.up)))
        sizes.pairsTotal = pairs
        sizes.unknown = max(10, n)
        sizes.everyFact = false
        sizes.texts = false
        let all = UmbrellaBench.sets(world: world, base: layout, pack: pack, strands: strands, tokenizer: tokenizer, sizes: sizes)
        let factSet = UmbrellaBench.factsOnly(all)
        var promptSet = all
        promptSet.facts = []
        promptSet.unknown = []
        var unknownSet = UmbrellaBench.factsOnly(all)
        unknownSet.facts = []
        unknownSet.unknown = all.unknown
        progress?("N = \(n): \(factSet.facts.count) facts, \(promptSet.pairs.count) pairs, \(all.generic.count) generic, "
                  + "\(all.commons.count) commons, \(unknownSet.unknown.count) unknown")

        let (umbrella, links, _) = try RoutingBench.umbrella(pack: pack, strands: strands, tokenizer: tokenizer)
        let generator = try umbrella.generator(links: links)
        let threadOf = Dictionary(uniqueKeysWithValues: strands.map { ($0.name, $0.threadID) })
        return Prepared(record: record, world: world, pack: pack, strands: strands, all: all, facts: factSet, prompts: promptSet, unknown: unknownSet,
                        umbrella: umbrella, links: links, generator: generator, threadOf: threadOf)
    }

    public static func run(
        layout: BraidLayout, tokenizer: RaoTokenizer, everyFact: Bool = false, factsPerNode: Int = 10, pairs: Int = 60,
        owner: String = "raolm-braid", progress: ((String) -> Void)? = nil
    ) throws -> ScalePoint {
        let started = Date()
        let prepared = try prepare(layout: layout, tokenizer: tokenizer, everyFact: everyFact, factsPerNode: factsPerNode, pairs: pairs,
                                   owner: owner, bench: "bench-scale", progress: progress)
        let (record, world, pack, all) = (prepared.record, prepared.world, prepared.pack, prepared.all)
        let (factSet, promptSet, unknownSet) = (prepared.facts, prepared.prompts, prepared.unknown)
        let (links, generator, threadOf) = (prepared.links, prepared.generator, prepared.threadOf)
        let n = world.names.count
        let gate = BraidRequest.defaultGate

        progress?("facts")
        var observed: [CitedGeneration] = []
        let facts = try UmbrellaBench.arm("facts", generator: generator, links: links, gate: gate, sets: factSet, tokenizer: tokenizer,
                                          threadOf: threadOf, observe: { observed.append($0) })
        var tally: [String: [Int]] = [:]
        let asked = factSet.facts.filter { $0.expected != nil }
        if observed.count == asked.count {
            for (example, generation) in zip(asked, observed) {
                var counts = tally[example.node ?? "nobody"] ?? [0, 0]
                if generation.text.hasPrefix(example.expected!) { counts[0] += 1 }
                counts[1] += 1
                tally[example.node ?? "nobody"] = counts
            }
        }
        let answerTokens = observed.flatMap { $0.traces.filter { !$0.isPrompt } }
        progress?("pairs, generic and commons prompts")
        let prompts = try UmbrellaBench.arm("prompts", generator: generator, links: links, gate: gate, sets: promptSet, tokenizer: tokenizer,
                                            threadOf: threadOf)
        progress?("subjects nobody holds")
        var firsts: [TokenTrace] = []
        let unknown = try UmbrellaBench.arm("unknown", generator: generator, links: links, gate: gate, sets: unknownSet, tokenizer: tokenizer,
                                            threadOf: threadOf, observe: { generation in
            if let first = generation.traces.first(where: { !$0.isPrompt }) { firsts.append(first) }
        })
        progress?("what a token costs")
        let cost = try tokenCost(generator: generator, links: links, prompts: all.generic, gate: gate)
        let footprint = try? JSONCoding.read([String: Int].self, from: layout.root.appendingPathComponent("sync-footprint.json"))
        return ScalePoint(
            braid: layout.root.deletingLastPathComponent().lastPathComponent, nodes: n, names: world.names, pack: pack.hasBase ? pack.sha256 : nil,
            dataset: record.dataset?.name, datasetHash: record.dataset?.hash, facts: facts, prompts: prompts, unknown: unknown,
            unknownCommonsLeads: firsts.isEmpty ? nil : Float(firsts.filter(commonsLeads).count) / Float(firsts.count),
            askedPerFactToken: answerTokens.isEmpty ? nil : Float(answerTokens.map(threadsAsked).reduce(0, +)) / Float(answerTokens.count),
            factsByNode: tally, nodeResults: (try? UmbrellaBench.nodeResults("scale", layout: layout, names: world.names)) ?? [],
            tokenCost: cost, peakFootprintBytes: footprint, everyFact: everyFact, seconds: Date().timeIntervalSince(started))
    }

    /// Threads (not the commons) asked for a hidden state at a token.
    static func threadsAsked(_ trace: TokenTrace) -> Int {
        (trace.strands ?? []).filter { $0.open && $0.strand != BraidStrandRef.commonsName }.count
    }

    /// Whether the commons holds the largest gate at a token.
    static func commonsLeads(_ trace: TokenTrace) -> Bool {
        let shares = trace.strands ?? []
        guard let commons = shares.first(where: { $0.strand == BraidStrandRef.commonsName }) else { return false }
        return shares.allSatisfy { $0.strand == BraidStrandRef.commonsName || $0.gate <= commons.gate }
    }

    /// The marginal cost of one more generated token: each prompt generated short and long, after
    /// one warm-up, so the prefill and the per-generation overhead cancel.
    static func tokenCost(
        generator: BraidedGenerator, links: [StrandLink], prompts: [BraidExample], gate: BraidGate, short: Int = 8, long: Int = 24,
        routing: Bool = false, router: BraidRouterKind = .bigram
    ) throws -> ScaleTokenCost {
        func run(_ example: BraidExample, _ tokens: Int) throws -> (seconds: Double, generation: CitedGeneration) {
            var params = GenerationParameters(tapLayer: links[0].descriptor.tapLayer, alpha: links[0].descriptor.alpha)
            params.maxTokens = tokens
            let started = Date()
            let generation = try generator.generate(BraidRequest(promptTokens: example.promptTokens, promptText: example.promptText, params: params,
                                                                 gate: gate, routing: routing, router: router))
            return (Date().timeIntervalSince(started), generation)
        }
        if let first = prompts.first { _ = try run(first, short) }
        var perToken: [Double] = []
        var perPrompt: [Double] = []
        var asked: [Int] = []
        var allAsked = 0
        let threads = links.count - (generator.commons == nil ? 0 : 1)
        for example in prompts {
            let a = try run(example, short)
            let b = try run(example, long)
            let generatedA = a.generation.traces.filter { !$0.isPrompt }.count
            let generated = b.generation.traces.filter { !$0.isPrompt }
            if generated.count > generatedA {
                let token = (b.seconds - a.seconds) / Double(generated.count - generatedA)
                perToken.append(token)
                perPrompt.append(a.seconds - Double(generatedA) * token)
            }
            for trace in generated {
                let count = threadsAsked(trace)
                asked.append(count)
                if count == threads { allAsked += 1 }
            }
        }
        func mean(_ values: [Double]) -> Double { values.isEmpty ? 0 : values.reduce(0, +) / Double(values.count) }
        return ScaleTokenCost(
            prompts: prompts.count, shortTokens: short, longTokens: long, secondsPerToken: mean(perToken), secondsPerPrompt: mean(perPrompt),
            askedPerToken: asked.isEmpty ? 0 : Float(asked.reduce(0, +)) / Float(asked.count),
            allAskedRate: asked.isEmpty ? 0 : Float(allAsked) / Float(asked.count))
    }

    // MARK: - The rules (pure)

    /// Least squares of `ys` on `xs`; nil under two distinct xs.
    public static func fit(_ xs: [Double], _ ys: [Double]) -> ScaleFit? {
        guard xs.count == ys.count, Set(xs).count >= 2 else { return nil }
        let n = Double(xs.count)
        let mx = xs.reduce(0, +) / n
        let my = ys.reduce(0, +) / n
        let sxy = zip(xs, ys).reduce(0) { $0 + ($1.0 - mx) * ($1.1 - my) }
        let sxx = xs.reduce(0) { $0 + ($1 - mx) * ($1 - mx) }
        let slope = sxy / sxx
        let intercept = my - slope * mx
        let total = ys.reduce(0) { $0 + ($1 - my) * ($1 - my) }
        let residual = zip(xs, ys).reduce(0) { $0 + pow($1.1 - (intercept + slope * $1.0), 2) }
        return ScaleFit(slope: slope, intercept: intercept, r2: total > 0 ? 1 - residual / total : 1)
    }

    /// Every point against the smallest N's.
    public static func evaluate(points unsorted: [ScalePoint]) -> ScaleEvaluation {
        let points = unsorted.sorted { $0.nodes < $1.nodes }
        guard let reference = points.first else { return ScaleEvaluation(rules: [], qualifies: false, secondsFit: nil, askedFit: nil, summary: "no points") }
        func pct(_ value: Float?) -> String { value.map { String(format: "%.0f%%", $0 * 100) } ?? "—" }
        var rules: [TrajectoryRuleResult] = []
        var problems: [String] = []
        for point in points {
            if point.pack != reference.pack { problems.append("N = \(point.nodes) runs pack \(point.pack.map { String($0.prefix(12)) } ?? "none"), not the reference's") }
            let exact = point.facts.factsExactRate >= reference.facts.factsExactRate - 1 / 30 - 1e-6
            let owned = point.facts.factsOwnedRate >= 0.90 - 1e-6
            let cited = (point.facts.citation ?? 0) >= (reference.facts.citation ?? 0) - 0.02 - 1e-6
            rules.append(TrajectoryRuleResult(
                rule: "N1 finds the Thread · N = \(point.nodes)", passed: exact && owned && cited,
                detail: "exact \(point.facts.factsExact)/\(point.facts.facts) (\(pct(point.facts.factsExactRate))) against \(pct(reference.facts.factsExactRate)); "
                    + "led and cited \(pct(point.facts.factsOwnedRate)) (bar 90%); citation@1 \(pct(point.facts.citation)) against \(pct(reference.facts.citation))"))
            let leads = (point.prompts.commonsLeads ?? 0) >= 0.90 - 1e-6
            let share = (point.prompts.commonsThreadShare ?? 1) <= 0.10 + 1e-6
            rules.append(TrajectoryRuleResult(
                rule: "N2 nobody's text · N = \(point.nodes)", passed: leads && share,
                detail: "the commons leads \(pct(point.prompts.commonsLeads)) of generic and commons prompts (bar 90%); "
                    + String(format: "Threads take %.3f of commons prompts (bar 0.10)", point.prompts.commonsThreadShare ?? 1)))
        }
        let xs = points.map { Double($0.nodes) }
        let secondsFit = fit(xs, points.map(\.tokenCost.secondsPerToken))
        let askedFit = fit(xs, points.map { Double($0.tokenCost.askedPerToken) })
        let failed = rules.filter { !$0.passed }.map(\.rule)
        let qualifies = failed.isEmpty && problems.isEmpty
        let summary = qualifies ? "the gate finds the Thread at every N from \(reference.nodes) to \(points.last!.nodes)"
            : (problems + (failed.isEmpty ? [] : ["fails " + failed.joined(separator: ", ")])).joined(separator: "; ")
        return ScaleEvaluation(rules: rules, qualifies: qualifies, secondsFit: secondsFit, askedFit: askedFit, summary: summary)
    }

    public static func report(_ points: [ScalePoint]) -> ScaleReport {
        let sorted = points.sorted { $0.nodes < $1.nodes }
        return ScaleReport(createdAt: .wholeSecond(), points: sorted, reference: sorted.first?.nodes ?? 0, evaluation: evaluate(points: sorted))
    }
}
