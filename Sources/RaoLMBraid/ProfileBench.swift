//
//  ProfileBench.swift
//  RaoLMBraid
//
//  WHAT: Whether the knowledge profile picks the Threads a prompt needs (Docs/ARCHITECTURE.md,
//        "Step 1 v3"): one built braid, every live version in process with its profile, asked
//        bench-scale's sets. Each prompt is routed without generating (the router is stateless, so
//        this is the route a generation takes); the braid then answers unrouted, and the credit
//        the aim pays there is split by whether its Thread was opened; then it answers routed.
//  OUT:  One point per braid: the owner found, Threads opened per set, who opens with whom by
//        world, credit recall, the copies' scores (when two nodes hold the same documents), both
//        sides' answers and costs; P1 to P4 judged within the point.
//  PIN:  Credit is the repo's bits-weighted credit (`UmbrellaBench.weigh`), the commons excluded on
//        both sides of the ratio. Unrouted first, routed second, on the same loaded strands.
//

import Foundation
import RaoLMCore
import RaoLMModel
import RaoLMProvenance

public struct ProfileSetStats: Codable, Sendable, Equatable {
    public var set: String
    public var prompts: Int
    /// Prompts no Thread could be ranked for: every Thread opened.
    public var unrouted: Int
    /// Threads opened per prompt (the commons not counted), and how often each count came.
    public var meanOpened: Double
    public var maxOpened: Int
    public var histogram: [Int]
}

public struct ProfileThreadStats: Codable, Sendable, Equatable {
    public var name: String
    public var world: String?
    public var entries: Int
    public var weighted: Int
    public var liftTotal: Float
    public var spread: Float
    public var k: Int
}

public struct ProfileCopyStats: Codable, Sendable, Equatable {
    public var a: String
    public var b: String
    /// Facts located in the documents both hold, each routed.
    public var prompts: Int
    /// Prompts whose two scores differ by at most 1% of the best score, and prompts that open both.
    public var within: Int
    public var openedTogether: Int
    public var maxRelativeDifference: Float
}

/// Subject questions on a braid of subject worlds (Docs/ARCHITECTURE.md, "Dataset: three subjects").
public struct ProfileSubjectStats: Codable, Sendable, Equatable {
    /// The dataset's first question for home facts, rewritten by the rules and routed by profile.
    public var questions: Int
    /// Questions whose route opened the owner, and whose routed answer the owner led at its first token.
    public var ownerOpened: Int
    public var ownerLeads: Int
    /// Per question: Threads of another subject opened, and their summed share of the answer tokens.
    public var strayOpenedMean: Double
    public var strayShareMean: Double
    /// Answered exactly; on those, the owner's mean share of the answer tokens and how many of its tokens had positive lift.
    public var exact: Int
    public var ownerShareMean: Double?
    public var liftPositive: Int
    public var ownerTokens: Int
    /// Questions about a name two subjects share, and those whose own Thread opened and led.
    public var homonyms: Int
    public var homonymsDecided: Int
    /// Per subject world: questions, owner opened, owner led, exact.
    public var byWorld: [String: [Int]]
}

public struct ProfilePoint: Codable, Sendable, Equatable {
    public var braid: String
    public var nodes: Int
    public var pack: String?
    public var dataset: String?
    public var datasetHash: String?
    public var threads: [ProfileThreadStats]
    public var facts: Int
    public var factsFound: Int
    public var pairs: Int
    public var pairsFound: Int
    public var missed: [String]
    public var sets: [ProfileSetStats]
    /// The owner's world → the opened Threads' world → mean count per fact prompt.
    public var worlds: [String: [String: Double]]?
    /// Bits-weighted credit the unrouted generations paid Threads: to opened ones, and in all.
    public var creditOpened: Double
    public var creditTotal: Double
    public var creditRecallFacts: Float?
    public var creditRecallPairs: Float?
    /// Routed fact generations that ran on exactly the Threads routed beforehand.
    public var liveMatchesDry: Int
    public var liveChecked: Int
    public var copies: ProfileCopyStats?
    public var subjects: ProfileSubjectStats?
    public var unrouted: RouteSide
    public var routed: RouteSide
    public var seconds: Double

    public var creditRecall: Float? { creditTotal > 0 ? Float(creditOpened / creditTotal) : nil }
}

public struct ProfileReport: Codable, Sendable {
    public var createdAt: Date
    public var points: [ProfilePoint]
    public var evaluation: ScaleEvaluation
}

public enum ProfileBench {
    public static func run(
        layout: BraidLayout, tokenizer: RaoTokenizer, factsPerNode: Int = 10, pairs: Int = 60, copies requested: (String, String)? = nil,
        owner: String = "raolm-braid", progress: ((String) -> Void)? = nil
    ) throws -> ProfilePoint {
        let started = Date()
        let prepared = try ScaleBench.prepare(layout: layout, tokenizer: tokenizer, everyFact: false, factsPerNode: factsPerNode, pairs: pairs,
                                              owner: owner, bench: "bench-profile", progress: progress)
        let generator = prepared.generator
        let names = generator.names
        let commons = generator.commons
        let commonsName = commons.map { names[$0] }
        guard generator.profileRouter.profiled == prepared.strands.count else {
            throw BraidSessionError.io("\(prepared.strands.count - generator.profileRouter.profiled) of \(prepared.strands.count) live versions have no knowledge profile: run raolm braid profile --data-dir \(layout.root.deletingLastPathComponent().path) first")
        }

        // Each node's world, from the dataset it was fed from.
        var worldOf: [String: String] = [:]
        if let dataset = prepared.record.dataset,
           let manifest = try? JSONCoding.read(DatasetManifest.self, from: URL(fileURLWithPath: dataset.path).appendingPathComponent(DatasetManifest.fileName)) {
            for node in manifest.nodes { if let world = node.world { worldOf[node.name] = world } }
        }
        let threads = prepared.strands.map { strand -> ProfileThreadStats in
            let profile = strand.profile()
            return ProfileThreadStats(
                name: strand.name, world: worldOf[strand.name], entries: profile?.info.entries ?? 0, weighted: profile?.info.weighted ?? 0,
                liftTotal: profile?.info.liftTotal ?? 0, spread: profile.map { ThreadProfile.spread(centroids: $0.centroids, hidden: $0.hidden) } ?? 0,
                k: profile?.k ?? 0)
        }

        // The routes, without generating.
        func opened(_ route: ProfileRoute) -> Set<String> { Set(route.indices.filter { $0 != commons }.map { names[$0] }) }
        let factPrompts = prepared.facts.facts.filter { $0.expected != nil }
        let pairPrompts = prepared.prompts.pairs.filter { $0.expected != nil }
        var routes: [String: [ProfileRoute]] = [:]
        func stats(_ set: String, _ examples: [BraidExample]) throws -> ProfileSetStats {
            let found = try examples.map { try generator.profileRoute($0.promptTokens) }
            routes[set] = found
            let counts = found.map { opened($0).count }
            var histogram = [Int](repeating: 0, count: names.count + 1)
            for count in counts { histogram[count] += 1 }
            while histogram.count > 1, histogram.last == 0 { histogram.removeLast() }
            return ProfileSetStats(set: set, prompts: examples.count, unrouted: found.filter(\.unrouted).count,
                                   meanOpened: counts.isEmpty ? 0 : Double(counts.reduce(0, +)) / Double(counts.count),
                                   maxOpened: counts.max() ?? 0, histogram: histogram)
        }
        progress?("routing every prompt by profile")
        let sets = [try stats("facts", factPrompts), try stats("pairs", pairPrompts), try stats("unknown", prepared.unknown.unknown),
                    try stats("generic", prepared.all.generic), try stats("commons", prepared.all.commons)]
        var missed: [String] = []
        func found(_ example: BraidExample, _ route: ProfileRoute, owners: [String?]) -> Bool {
            let set = opened(route)
            let ok = owners.allSatisfy { $0.map(set.contains) ?? false }
            if !ok, missed.count < 20 { missed.append(example.label) }
            return ok
        }
        let factRoutes = routes["facts"] ?? []
        let pairRoutes = routes["pairs"] ?? []
        let factsFound = zip(factPrompts, factRoutes).filter { found($0.0, $0.1, owners: [$0.0.node]) }.count
        let pairsFound = zip(pairPrompts, pairRoutes).filter { found($0.0, $0.1, owners: [$0.0.opener, $0.0.node]) }.count
        var worlds: [String: [String: Double]]?
        if !worldOf.isEmpty {
            var sums: [String: [String: Double]] = [:]
            var prompts: [String: Int] = [:]
            for (example, route) in zip(factPrompts, factRoutes) {
                guard let ownerWorld = example.node.flatMap({ worldOf[$0] }) else { continue }
                prompts[ownerWorld, default: 0] += 1
                for name in opened(route) { sums[ownerWorld, default: [:]][worldOf[name] ?? "—", default: 0] += 1 }
            }
            worlds = Dictionary(uniqueKeysWithValues: prompts.map { world, count in
                (world, (sums[world] ?? [:]).mapValues { $0 / Double(count) })
            })
        }
        progress?(String(format: "route: owner opened on %d/%d facts and both owners on %d/%d pairs; %.1f Threads opened per fact prompt",
                         factsFound, factPrompts.count, pairsFound, pairPrompts.count, sets[0].meanOpened))

        // Both sides: the unrouted one pays the credit the route is judged by.
        func side(_ routing: Bool, facts: ((CitedGeneration) -> Void)?, prompts: ((CitedGeneration) -> Void)?) throws -> RouteSide {
            let began = Date()
            let gate = BraidRequest.defaultGate
            let label = routing ? "routed" : "unrouted"
            progress?("\(label): facts")
            let factArm = try UmbrellaBench.arm("facts", generator: generator, links: prepared.links, gate: gate, sets: prepared.facts,
                                                tokenizer: tokenizer, threadOf: prepared.threadOf, observe: facts, routing: routing, router: .profile)
            progress?("\(label): pairs, generic and commons prompts")
            let promptArm = try UmbrellaBench.arm("prompts", generator: generator, links: prepared.links, gate: gate, sets: prepared.prompts,
                                                  tokenizer: tokenizer, threadOf: prepared.threadOf, observe: prompts, routing: routing, router: .profile)
            progress?("\(label): subjects nobody holds")
            let unknownArm = try UmbrellaBench.arm("unknown", generator: generator, links: prepared.links, gate: gate, sets: prepared.unknown,
                                                   tokenizer: tokenizer, threadOf: prepared.threadOf, routing: routing, router: .profile)
            progress?("\(label): what a token costs")
            let factCost = try ScaleBench.tokenCost(generator: generator, links: prepared.links, prompts: Array(factPrompts.prefix(8)), gate: gate,
                                                    routing: routing, router: .profile)
            let genericCost = try ScaleBench.tokenCost(generator: generator, links: prepared.links, prompts: prepared.all.generic, gate: gate,
                                                       routing: routing, router: .profile)
            return RouteSide(facts: factArm, prompts: promptArm, unknown: unknownArm, factCost: factCost, genericCost: genericCost,
                             seconds: Date().timeIntervalSince(began))
        }
        var factGenerations: [CitedGeneration] = []
        var promptGenerations: [CitedGeneration] = []
        let unrouted = try side(false, facts: { factGenerations.append($0) }, prompts: { promptGenerations.append($0) })
        let factCredit = zip(factRoutes, factGenerations).map { credit($0.1.traces, opened: opened($0.0), commons: commonsName) }
        let pairCredit = zip(pairRoutes, promptGenerations.prefix(pairPrompts.count)).map { credit($0.1.traces, opened: opened($0.0), commons: commonsName) }
        func recall(_ parts: [(opened: Double, total: Double)]) -> Float? {
            let total = parts.map(\.total).reduce(0, +)
            return total > 0 ? Float(parts.map(\.opened).reduce(0, +) / total) : nil
        }
        let creditOpened = (factCredit + pairCredit).map(\.opened).reduce(0, +)
        let creditTotal = (factCredit + pairCredit).map(\.total).reduce(0, +)

        var routedFacts: [CitedGeneration] = []
        let routed = try side(true, facts: { routedFacts.append($0) }, prompts: nil)
        let liveMatchesDry = zip(factRoutes, routedFacts).filter { route, generation in
            let ran = Set((generation.braid?.strands ?? []).map(\.name)).subtracting(commonsName.map { [$0] } ?? [])
            return ran == opened(route)
        }.count

        // Copies: two nodes holding the same documents must score alike on every fact of them.
        var copyStats: ProfileCopyStats?
        var present: [String: [CorpusDocument]] = [:]
        for name in prepared.world.names { present[name] = MockFeeder.present(node: name, world: prepared.world, layout: layout.node(name)) }
        if let (a, b) = requested ?? copyPair(present.mapValues { Set($0.map(\.id)) }),
           let ia = names.firstIndex(of: a), let ib = names.firstIndex(of: b) {
            progress?("copies: \(a) and \(b)")
            let shared = Set((present[b] ?? []).map(\.id))
            let documents = (present[a] ?? []).filter { shared.contains($0.id) }
            let threadID = prepared.threadOf[a] ?? nil
            let examples = BraidExample.facts(nodes: [(name: a, label: a, threadID: threadID, documents: documents)], tokenizer: tokenizer, perNode: .max)
            var within = 0
            var together = 0
            var largest: Float = 0
            for example in examples {
                let route = try generator.profileRoute(example.promptTokens)
                guard let sa = route.scores[ia], let sb = route.scores[ib], let best = route.best, best > 0 else { continue }
                let difference = abs(sa - sb) / best
                largest = max(largest, difference)
                if difference <= 0.01 + 1e-6 { within += 1 }
                if route.candidates[ia] && route.candidates[ib] { together += 1 }
            }
            copyStats = ProfileCopyStats(a: a, b: b, prompts: examples.count, within: within, openedTogether: together, maxRelativeDifference: largest)
        }

        // Subject questions, when every node holds a subject world.
        var subjectStats: ProfileSubjectStats?
        if !worldOf.isEmpty, worldOf.values.allSatisfy({ DatasetWorld(rawValue: $0)?.subject == true }) {
            progress?("subject questions")
            subjectStats = try subjects(prepared: prepared, layout: layout, tokenizer: tokenizer, worldOf: worldOf)
        }

        return ProfilePoint(
            braid: layout.root.deletingLastPathComponent().lastPathComponent, nodes: prepared.strands.count, pack: prepared.pack.sha256,
            dataset: prepared.record.dataset?.name, datasetHash: prepared.record.dataset?.hash, threads: threads,
            facts: factPrompts.count, factsFound: factsFound, pairs: pairPrompts.count, pairsFound: pairsFound, missed: missed, sets: sets,
            worlds: worlds, creditOpened: creditOpened, creditTotal: creditTotal, creditRecallFacts: recall(factCredit),
            creditRecallPairs: recall(pairCredit), liveMatchesDry: liveMatchesDry, liveChecked: min(factRoutes.count, routedFacts.count),
            copies: copyStats, subjects: subjectStats, unrouted: unrouted, routed: routed, seconds: Date().timeIntervalSince(started))
    }

    /// The dataset's first question for each home fact (up to `perNode` a node), rewritten by the rules
    /// arm, routed by profile and answered routed, with each Thread's own context as `braid ask` asks.
    static func subjects(
        prepared: ScaleBench.Prepared, layout: BraidLayout, tokenizer: RaoTokenizer, worldOf: [String: String], perNode: Int = 30
    ) throws -> ProfileSubjectStats {
        var sizes = QuestionBench.Sizes()
        sizes.perNode = perNode
        sizes.unknownPerNode = 0
        sizes.sharedPerLink = 0
        let sets = QuestionBench.sets(world: prepared.world, questions: prepared.world, layout: layout, strands: prepared.strands,
                                      tokenizer: tokenizer, sizes: sizes)
        let generator = prepared.generator
        let names = generator.names
        let commonsName = generator.commons.map { names[$0] }
        let homonymNames = Set(prepared.world.crosslinks.filter { $0.kind == .homonym }.map(\.subject))
        var stats = ProfileSubjectStats(questions: 0, ownerOpened: 0, ownerLeads: 0, strayOpenedMean: 0, strayShareMean: 0, exact: 0,
                                        ownerShareMean: nil, liftPositive: 0, ownerTokens: 0, homonyms: 0, homonymsDecided: 0, byWorld: [:])
        var strayOpened: [Double] = []
        var strayShares: [Double] = []
        var ownerShares: [Double] = []
        for question in sets.questions {
            let world = worldOf[question.node] ?? "—"
            let rewrite = QuestionAdapter.fallback(question.text)
            let tokens = QuestionAdapter.stemTokens(rewrite.stem, tokenizer: tokenizer)
            let route = try generator.profileRoute(tokens)
            let opened = Set(route.indices.map { names[$0] }).subtracting(commonsName.map { [$0] } ?? [])
            var params = GenerationParameters(tapLayer: prepared.links[0].descriptor.tapLayer, alpha: prepared.links[0].descriptor.alpha)
            params.maxTokens = sizes.maxTokens
            let generation = try generator.generate(BraidRequest(
                promptTokens: tokens, promptText: rewrite.stem, params: params, stopAtSentenceEnd: true, question: rewrite, context: sizes.context,
                subject: QuestionAdapter.subjectTokens(stem: rewrite.stem, tokens: tokens, question: question.text, tokenizer: tokenizer),
                routing: true, router: .profile))
            let answer = generation.traces.filter { !$0.isPrompt }
            let answerTokens = Array(answer.prefix(max(1, tokenizer.encode(question.expected).count)))
            let threads = (answer.first?.strands ?? []).filter { $0.strand != commonsName }
            let leads = threads.max { $0.gate < $1.gate }?.strand == question.node
            let ownerOpened = opened.contains(question.node)
            stats.questions += 1
            if ownerOpened { stats.ownerOpened += 1 }
            if leads { stats.ownerLeads += 1 }
            let stray = opened.filter { (worldOf[$0] ?? world) != world }
            strayOpened.append(Double(stray.count))
            strayShares.append(answerTokens.isEmpty ? 0 : answerTokens.map { trace in
                Double((trace.strands ?? []).filter { stray.contains($0.strand) }.map(\.share).reduce(0, +))
            }.reduce(0, +) / Double(answerTokens.count))
            let exact = generation.text.hasPrefix(question.expected)
            if exact {
                stats.exact += 1
                let own = answerTokens.compactMap { $0.strands?.first { $0.strand == question.node } }
                if !own.isEmpty { ownerShares.append(Double(own.map(\.share).reduce(0, +)) / Double(own.count)) }
                stats.ownerTokens += own.count
                stats.liftPositive += own.filter { ($0.lift ?? 0) > 0 }.count
            }
            if homonymNames.contains(where: { question.text.contains($0) }) {
                stats.homonyms += 1
                if ownerOpened && leads { stats.homonymsDecided += 1 }
            }
            var row = stats.byWorld[world] ?? [0, 0, 0, 0]
            row[0] += 1
            if ownerOpened { row[1] += 1 }
            if leads { row[2] += 1 }
            if exact { row[3] += 1 }
            stats.byWorld[world] = row
        }
        func mean(_ values: [Double]) -> Double { values.isEmpty ? 0 : values.reduce(0, +) / Double(values.count) }
        stats.strayOpenedMean = mean(strayOpened)
        stats.strayShareMean = mean(strayShares)
        stats.ownerShareMean = ownerShares.isEmpty ? nil : mean(ownerShares)
        return stats
    }

    // MARK: - Pure

    /// The bits-weighted credit a generation paid Threads (the commons excluded): to the opened ones, and in all.
    static func credit(_ traces: [TokenTrace], opened: Set<String>, commons: String?) -> (opened: Double, total: Double) {
        var inside = 0.0
        var total = 0.0
        for trace in traces where !trace.isPrompt {
            guard let bits = trace.bits, let strands = trace.strands else { continue }
            for share in strands where share.strand != commons {
                let paid = Double(bits) * Double(share.credit ?? 0)
                total += paid
                if opened.contains(share.strand) { inside += paid }
            }
        }
        return (inside, total)
    }

    /// Two nodes that hold the same documents: the pair sharing the most, when it is at least half
    /// the smaller node's documents.
    static func copyPair(_ present: [String: Set<String>]) -> (String, String)? {
        let names = present.keys.sorted()
        var best: (String, String, Int)?
        for (i, a) in names.enumerated() {
            for b in names[(i + 1)...] {
                let shared = present[a]!.intersection(present[b]!).count
                let smaller = min(present[a]!.count, present[b]!.count)
                guard shared > 0, shared * 2 >= smaller else { continue }
                if shared > (best?.2 ?? 0) { best = (a, b, shared) }
            }
        }
        return best.map { ($0.0, $0.1) }
    }

    public static func evaluate(points unsorted: [ProfilePoint]) -> ScaleEvaluation {
        let points = unsorted.sorted { ($0.nodes, $0.braid) < ($1.nodes, $1.braid) }
        func pct(_ value: Float?) -> String { value.map { String(format: "%.0f%%", $0 * 100) } ?? "—" }
        func rate(_ a: Int, _ b: Int) -> Float { b > 0 ? Float(a) / Float(b) : 0 }
        var rules: [TrajectoryRuleResult] = []
        for point in points {
            let tag = "\(point.braid)"
            let facts = rate(point.factsFound, point.facts)
            let pairs = rate(point.pairsFound, point.pairs)
            rules.append(TrajectoryRuleResult(
                rule: "P1 owner found · \(tag)", passed: point.facts > 0 && facts >= 0.99 - 1e-6 && (point.pairs == 0 || pairs >= 0.99 - 1e-6),
                detail: "owner opened on \(point.factsFound)/\(point.facts) fact prompts (\(pct(facts))), both owners on \(point.pairsFound)/\(point.pairs) two-fact prompts (\(pct(pairs))); bar 99%"))
            let factSet = point.sets.first { $0.set == "facts" }
            let mean = factSet?.meanOpened ?? Double(point.nodes)
            rules.append(TrajectoryRuleResult(
                rule: "P2 size · \(tag)", passed: mean <= 4 + 1e-9,
                detail: String(format: "%.2f Threads opened per fact prompt (bar 4), at most %d", mean, factSet?.maxOpened ?? 0)))
            let recall = point.creditRecall
            rules.append(TrajectoryRuleResult(
                rule: "P3 credit recall · \(tag)", passed: (recall ?? 0) >= 0.95 - 1e-6,
                detail: "\(pct(recall)) of the credit the aim paid Threads unrouted went to opened ones (facts \(pct(point.creditRecallFacts)), pairs \(pct(point.creditRecallPairs))); bar 95%"))
            if let subjects = point.subjects {
                let opened = rate(subjects.ownerOpened, subjects.questions)
                let leads = rate(subjects.ownerLeads, subjects.questions)
                rules.append(TrajectoryRuleResult(
                    rule: "S1 subject routes · \(tag)", passed: subjects.questions > 0 && opened >= 0.99 - 1e-6 && leads >= 0.90 - 1e-6,
                    detail: "the owner opened on \(subjects.ownerOpened)/\(subjects.questions) subject questions (\(pct(opened)), bar 99%) and led the first answer token on \(subjects.ownerLeads) (\(pct(leads)), bar 90%)"))
                rules.append(TrajectoryRuleResult(
                    rule: "S2 no stray subject · \(tag)", passed: subjects.strayOpenedMean <= 0.10 + 1e-9 && subjects.strayShareMean <= 0.10 + 1e-9,
                    detail: String(format: "%.2f Threads of another subject opened per question (bar 0.10); they took %.3f of the answer tokens (bar 0.10)",
                                   subjects.strayOpenedMean, subjects.strayShareMean)))
                let lift = rate(subjects.liftPositive, subjects.ownerTokens)
                rules.append(TrajectoryRuleResult(
                    rule: "S3 royalties · \(tag)", passed: subjects.exact > 0 && (subjects.ownerShareMean ?? 0) >= 0.90 - 1e-6 && lift >= 0.95 - 1e-6,
                    detail: String(format: "on %d exact answers the owner took %.2f of the answer tokens (bar 0.90), its lift positive on %@ of them (bar 95%%)",
                                   subjects.exact, subjects.ownerShareMean ?? 0, pct(lift))))
                let decided = rate(subjects.homonymsDecided, subjects.homonyms)
                rules.append(TrajectoryRuleResult(
                    rule: "S4 homonyms · \(tag)", passed: subjects.homonyms > 0 && decided >= 0.90 - 1e-6,
                    detail: "on \(subjects.homonyms) questions about a shared name, the asked kind's Thread opened and led on \(subjects.homonymsDecided) (\(pct(decided)), bar 90%)"))
            }
            if let copies = point.copies {
                let passed = copies.prompts > 0 && copies.within == copies.prompts && copies.openedTogether == copies.prompts
                rules.append(TrajectoryRuleResult(
                    rule: "P4 copies · \(tag)", passed: passed,
                    detail: String(format: "%@ and %@: %d of %d facts within 1%% of the best score, %d opened both; largest difference %.2f%%",
                                   copies.a, copies.b, copies.within, copies.prompts, copies.openedTogether, copies.maxRelativeDifference * 100)))
            }
        }
        let failed = rules.filter { !$0.passed }.map(\.rule)
        let qualifies = !rules.isEmpty && failed.isEmpty
        return ScaleEvaluation(rules: rules, qualifies: qualifies, secondsFit: nil, askedFit: nil,
                               summary: qualifies ? "the profile opens the Threads a prompt needs, few of them, and the aim's credit stays inside"
                                   : (rules.isEmpty ? "no points" : "fails " + failed.joined(separator: ", ")))
    }

    public static func report(_ points: [ProfilePoint]) -> ProfileReport {
        let sorted = points.sorted { ($0.nodes, $0.braid) < ($1.nodes, $1.braid) }
        return ProfileReport(createdAt: .wholeSecond(), points: sorted, evaluation: evaluate(points: sorted))
    }
}
