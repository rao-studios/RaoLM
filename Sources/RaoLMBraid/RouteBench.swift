//
//  RouteBench.swift
//  RaoLMBraid
//
//  WHAT: Whether routing before asking keeps the answers and cuts the cost (Docs/ARCHITECTURE.md,
//        "Step 1"): one built braid, its live versions loaded in process, asked bench-scale's sets
//        twice, unrouted and routed, and the route itself checked against every prompt's owners.
//  OUT:  One point per braid: the route's recall and how many Threads it opens on each set, both
//        sides' facts, two-fact, generic, commons and unknown results, and what a token costs on
//        fact and generic prompts each way; the rules R1 to R3 judged within the point.
//  PIN:  Both sides run in the same process on the same loaded strands and the same sets, unrouted
//        first. In process the forward passes of a token run one after another, so a Thread that is
//        not opened saves its whole share of the token's time.
//

import Foundation
import RaoLMCore
import RaoLMModel
import RaoLMProvenance

public struct RouteSetStats: Codable, Sendable, Equatable {
    public var set: String
    public var prompts: Int
    /// Prompts with no distinctive bigram: every Thread asked.
    public var unmatched: Int
    /// Threads opened per prompt (the commons not counted).
    public var meanOpened: Double
    public var maxOpened: Int
}

public struct RouteSide: Codable, Sendable, Equatable {
    public var facts: UmbrellaArmResult
    public var prompts: UmbrellaArmResult
    public var unknown: UmbrellaArmResult
    public var factCost: ScaleTokenCost
    public var genericCost: ScaleTokenCost
    public var seconds: Double
}

public struct RoutePoint: Codable, Sendable, Equatable {
    public var braid: String
    public var nodes: Int
    public var pack: String?
    public var dataset: String?
    public var datasetHash: String?
    /// The most Threads a distinctive bigram may be held by.
    public var bound: Int
    /// Each Thread's sketch: its bigrams.
    public var sketchBigrams: [String: Int]
    /// Fact prompts whose owner is a candidate; two-fact prompts with both owners candidates.
    public var facts: Int
    public var factsFound: Int
    public var pairs: Int
    public var pairsFound: Int
    public var missed: [String]
    public var sets: [RouteSetStats]
    public var unrouted: RouteSide
    public var routed: RouteSide
    public var seconds: Double
}

public struct RouteReport: Codable, Sendable {
    public var createdAt: Date
    public var points: [RoutePoint]
    public var evaluation: ScaleEvaluation
}

public enum RouteBench {
    public static func run(
        layout: BraidLayout, tokenizer: RaoTokenizer, factsPerNode: Int = 10, pairs: Int = 60, owner: String = "raolm-braid",
        progress: ((String) -> Void)? = nil
    ) throws -> RoutePoint {
        let started = Date()
        let prepared = try ScaleBench.prepare(layout: layout, tokenizer: tokenizer, everyFact: false, factsPerNode: factsPerNode, pairs: pairs,
                                              owner: owner, bench: "bench-route", progress: progress)
        let generator = prepared.generator
        let links = prepared.links
        let n = prepared.world.names.count
        let router = generator.router
        let index = Dictionary(uniqueKeysWithValues: generator.names.enumerated().map { ($0.element, $0.offset) })

        // The route alone: who is opened for every prompt of every set.
        func stats(_ name: String, _ examples: [BraidExample]) -> RouteSetStats {
            let routes = examples.map { router.route($0.promptTokens) }
            let opened = routes.map { route in route.candidates.indices.filter { route.candidates[$0] && $0 != generator.commons }.count }
            return RouteSetStats(set: name, prompts: examples.count, unmatched: routes.filter(\.unmatched).count,
                                 meanOpened: opened.isEmpty ? 0 : Double(opened.reduce(0, +)) / Double(opened.count), maxOpened: opened.max() ?? 0)
        }
        var missed: [String] = []
        func found(_ example: BraidExample, owners: [String?]) -> Bool {
            let route = router.route(example.promptTokens)
            let ok = owners.allSatisfy { owner in owner.flatMap { index[$0] }.map { route.candidates[$0] } ?? false }
            if !ok, missed.count < 20 { missed.append(example.label) }
            return ok
        }
        let factPrompts = prepared.facts.facts.filter { $0.expected != nil }
        let pairPrompts = prepared.prompts.pairs
        let factsFound = factPrompts.filter { found($0, owners: [$0.node]) }.count
        let pairsFound = pairPrompts.filter { found($0, owners: [$0.opener, $0.node]) }.count
        let sets = [stats("facts", factPrompts), stats("pairs", pairPrompts), stats("unknown", prepared.unknown.unknown),
                    stats("generic", prepared.all.generic), stats("commons", prepared.all.commons)]
        progress?("route: \(factsFound)/\(factPrompts.count) facts and \(pairsFound)/\(pairPrompts.count) pairs reach their owners; "
                  + sets.map { String(format: "%@ %.1f opened", $0.set, $0.meanOpened) }.joined(separator: ", "))

        func side(_ routing: Bool) throws -> RouteSide {
            let began = Date()
            let gate = BraidRequest.defaultGate
            let label = routing ? "routed" : "unrouted"
            progress?("\(label): facts")
            let facts = try UmbrellaBench.arm("facts", generator: generator, links: links, gate: gate, sets: prepared.facts, tokenizer: tokenizer,
                                              threadOf: prepared.threadOf, routing: routing)
            progress?("\(label): pairs, generic and commons prompts")
            let prompts = try UmbrellaBench.arm("prompts", generator: generator, links: links, gate: gate, sets: prepared.prompts, tokenizer: tokenizer,
                                                threadOf: prepared.threadOf, routing: routing)
            progress?("\(label): subjects nobody holds")
            let unknown = try UmbrellaBench.arm("unknown", generator: generator, links: links, gate: gate, sets: prepared.unknown, tokenizer: tokenizer,
                                                threadOf: prepared.threadOf, routing: routing)
            progress?("\(label): what a token costs")
            let factCost = try ScaleBench.tokenCost(generator: generator, links: links, prompts: Array(factPrompts.prefix(8)), gate: gate, routing: routing)
            let genericCost = try ScaleBench.tokenCost(generator: generator, links: links, prompts: prepared.all.generic, gate: gate, routing: routing)
            return RouteSide(facts: facts, prompts: prompts, unknown: unknown, factCost: factCost, genericCost: genericCost,
                             seconds: Date().timeIntervalSince(began))
        }
        let unrouted = try side(false)
        let routed = try side(true)
        let sketches = Dictionary(uniqueKeysWithValues: links.filter { !$0.descriptor.isCommons }.map { ($0.descriptor.name, $0.descriptor.sketch?.count ?? 0) })
        return RoutePoint(
            braid: layout.root.deletingLastPathComponent().lastPathComponent, nodes: n, pack: prepared.pack.sha256,
            dataset: prepared.record.dataset?.name, datasetHash: prepared.record.dataset?.hash, bound: router.bound, sketchBigrams: sketches,
            facts: factPrompts.count, factsFound: factsFound, pairs: pairPrompts.count, pairsFound: pairsFound, missed: missed, sets: sets,
            unrouted: unrouted, routed: routed, seconds: Date().timeIntervalSince(started))
    }

    // MARK: - The rules (pure)

    public static func evaluate(points unsorted: [RoutePoint]) -> ScaleEvaluation {
        let points = unsorted.sorted { $0.nodes < $1.nodes }
        func pct(_ value: Float?) -> String { value.map { String(format: "%.0f%%", $0 * 100) } ?? "—" }
        func rate(_ a: Int, _ b: Int) -> Float { b > 0 ? Float(a) / Float(b) : 0 }
        var rules: [TrajectoryRuleResult] = []
        for point in points {
            let (u, r) = (point.unrouted, point.routed)
            let factsRecall = rate(point.factsFound, point.facts)
            let pairsRecall = rate(point.pairsFound, point.pairs)
            rules.append(TrajectoryRuleResult(
                rule: "R1 recall · N = \(point.nodes)", passed: point.facts > 0 && factsRecall >= 0.99 - 1e-6 && (point.pairs == 0 || pairsRecall >= 0.99 - 1e-6),
                detail: "owner a candidate on \(point.factsFound)/\(point.facts) fact prompts (\(pct(factsRecall))), both owners on "
                    + "\(point.pairsFound)/\(point.pairs) two-fact prompts (\(pct(pairsRecall))); bar 99%"))
            let exact = r.facts.factsExactRate >= u.facts.factsExactRate - 1 / 30 - 1e-6
            let cited = (r.facts.citation ?? 0) >= (u.facts.citation ?? 0) - 0.02 - 1e-6
            let moved = rate(r.prompts.pairsMoved, r.prompts.pairs) >= rate(u.prompts.pairsMoved, u.prompts.pairs) - 2 / 60 - 1e-6
            rules.append(TrajectoryRuleResult(
                rule: "R2 same answers · N = \(point.nodes)", passed: exact && cited && moved,
                detail: "exact \(r.facts.factsExact)/\(r.facts.facts) routed against \(u.facts.factsExact)/\(u.facts.facts); citation@1 "
                    + "\(pct(r.facts.citation)) against \(pct(u.facts.citation)); second owner leads \(r.prompts.pairsMoved)/\(r.prompts.pairs) "
                    + "against \(u.prompts.pairsMoved)/\(u.prompts.pairs)"))
            let ratio = u.factCost.secondsPerToken > 0 ? r.factCost.secondsPerToken / u.factCost.secondsPerToken : 0
            let opened = point.sets.first { $0.set == "facts" }?.meanOpened ?? Double(point.nodes)
            if point.nodes >= 24 {
                let cheap = opened <= Double(point.nodes) / 4 + 1e-9 && ratio > 0 && ratio <= 0.5 + 1e-9
                rules.append(TrajectoryRuleResult(
                    rule: "R3 cost · N = \(point.nodes)", passed: cheap,
                    detail: String(format: "%.1f Threads opened per fact prompt (bar %.0f); seconds per token on fact prompts %.2f× unrouted (bar 0.50×)",
                                   opened, Double(point.nodes) / 4, ratio)))
            } else if point.nodes <= 3 {
                rules.append(TrajectoryRuleResult(
                    rule: "R3 cost · N = \(point.nodes)", passed: ratio > 0 && ratio <= 1.05 + 1e-9,
                    detail: String(format: "seconds per token on fact prompts %.2f× unrouted (bar 1.05×); %.1f Threads opened per fact prompt", ratio, opened)))
            }
        }
        let failed = rules.filter { !$0.passed }.map(\.rule)
        let qualifies = !rules.isEmpty && failed.isEmpty
        return ScaleEvaluation(rules: rules, qualifies: qualifies, secondsFit: nil, askedFit: nil,
                               summary: qualifies ? "routing keeps the answers and cuts the cost at every N measured"
                                   : (rules.isEmpty ? "no points" : "fails " + failed.joined(separator: ", ")))
    }

    public static func report(_ points: [RoutePoint]) -> RouteReport {
        let sorted = points.sorted { $0.nodes < $1.nodes }
        return RouteReport(createdAt: .wholeSecond(), points: sorted, evaluation: evaluate(points: sorted))
    }
}
