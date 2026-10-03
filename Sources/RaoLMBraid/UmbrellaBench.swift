//
//  UmbrellaBench.swift
//  RaoLMBraid
//
//  WHAT: Whether the umbrella's layers earn their place. Five arms, each adding to the one before:
//        v1 (the braid as it was: tiny nodes on a seeded vocabulary), pack (base nodes, their
//        blocks warm-started from SmolLM2 under its frozen trunk), commons (+ the commons strand
//        and the lift gate), calibrated (+ each Thread's λ and τ from its corpus's
//        self-trajectory) and anchors (+ thought agreement in the gate). Every arm answers the
//        same prompts: the gate bench's four sets, general English nobody holds, held-out text
//        (each node's unfed documents and the pack's commons sample) and told texts.
//  OUT:  One report: every arm's measures, every node's, the trajectory rules on the base braid,
//        and `evaluate`'s verdict.
//  PIN:  The rules were fixed before any numbers (Docs/ARCHITECTURE.md, "bench-umbrella").
//        Nothing is written to a node. v1 runs on its own braid, fed the same dataset the same way.
//

import Foundation
import RaoLMCore
import RaoLMModel
import RaoLMProvenance
import RaoLMTraining

public struct UmbrellaArmResult: Codable, Sendable, Equatable {
    public var arm: String
    /// Fact prompts: answered exactly; the first answer token led by and cited to the owner.
    public var facts: Int
    public var factsExact: Int
    public var factsOwned: Int
    /// Over the facts' answer tokens: the top neighbour cites the fact's partition; the owner's
    /// mean share; the share whose owner lift over the commons is positive (arms with a commons).
    public var citation: Float?
    public var ownerShare: Float?
    public var liftPositive: Float?
    /// Two-fact prompts: the second owner leads and is cited at its first answer token; exact.
    public var pairs: Int
    public var pairsMoved: Int
    public var pairsExact: Int
    /// At the first generated token: every Thread asked, and the largest Thread gate.
    public var unknownAllAsked: Float
    public var unknownLargest: Float
    public var genericAllAsked: Float
    public var genericLargest: Float
    /// Generic and commons prompts: the share where the commons holds the largest gate at the first
    /// generated token, the mean largest Thread gate there, and on commons prompts the Threads'
    /// summed share of the generated tokens.
    public var commonsLeads: Float?
    public var commonsThreadGate: Float?
    public var commonsThreadShare: Float?
    /// Mean −log p(token) of the mixture over held-out text, and its tokens.
    public var heldOutNLL: Float
    public var heldOutTokens: Int
    /// Told texts: the source Thread's mean share of the fact answer tokens.
    public var toldSourceShare: Float?
    public var toldTokens: Int
    /// Voice and generic texts continued: the mean largest Thread gate at the first generated token.
    public var voiceLargest: Float?
    public var seconds: Double
    /// The owner's blend, weighted by bits (arms with a commons strand): the owner's credit on its
    /// facts' answers and the share of those answers' bits in form tokens; the commons' credit on
    /// general text; the Threads' credit on unfed text in their own voice; the source's on told answers.
    public var ownerCredit: Float? = nil
    public var formBits: Float? = nil
    public var commonsCreditGeneral: Float? = nil
    public var threadsCreditOwnVoice: Float? = nil
    public var toldSourceCredit: Float? = nil

    public var factsExactRate: Float { facts > 0 ? Float(factsExact) / Float(facts) : 0 }
    public var factsOwnedRate: Float { facts > 0 ? Float(factsOwned) / Float(facts) : 0 }
    public var pairsMovedRate: Float { pairs > 0 ? Float(pairsMoved) / Float(pairs) : 0 }
}

public struct UmbrellaNodeResult: Codable, Sendable, Equatable {
    /// "reference", "base" or "ceiling".
    public var braid: String
    public var node: String
    public var version: Int
    public var documents: Int
    public var memorised: Float
    /// Optimizer steps the node's first version took to memorise 97% (nil: it did not).
    public var stepsTo97: Int?
    public var heldOutLoss: Float?
    public var commonsLoss: Float?
    public var calibration: StrandCalibration?
}

public struct UmbrellaEvaluation: Codable, Sendable, Equatable {
    public var rules: [TrajectoryRuleResult]
    /// The arms whose rules (and every earlier arm's) hold.
    public var qualifies: [String]
    public var winner: String?
    public var reported: [String: Float]
    public var summary: String
}

public struct UmbrellaReport: Codable, Sendable {
    public var createdAt: Date
    public var pack: String?
    public var arms: [UmbrellaArmResult]
    /// The pack arm on the braid fed twice the documents.
    public var ceiling: UmbrellaArmResult?
    public var nodes: [UmbrellaNodeResult]
    public var trajectory: [TrajectoryRuleResult]
    public var evaluation: UmbrellaEvaluation
    /// The second reading (2026-10-01): every arm's fact measures on every fact prompt of each
    /// node's documents (the ceiling's too), and the rules read with them in place of the 30.
    public var allFacts: [UmbrellaArmResult]? = nil
    public var allFactsCeiling: UmbrellaArmResult? = nil
    public var allFactsEvaluation: UmbrellaEvaluation? = nil
}

public enum UmbrellaBench {
    public static let armNames = ["v1", "pack", "commons", "calibrated", "anchors"]

    /// General English no Thread holds and the base model knows.
    public static let commonsPrompts = [
        "The capital of France is", "Water freezes at a temperature of", "The largest planet in our solar system is",
        "Shakespeare wrote the play", "The chemical symbol for gold is", "The first man to walk on the moon was",
        "Photosynthesis is the process by which plants", "The Pacific Ocean is the largest", "In mathematics, the square root of sixteen is",
        "Honey is made by", "The heart pumps blood through the", "Mount Everest is the highest mountain in",
        "The Mona Lisa was painted by", "Light travels faster than", "The opposite of hot is", "Rain falls from clouds when",
        "A triangle has three sides, and a square has", "The sun rises in the", "A group of wolves is called a",
        "The Great Wall of China was built to",
    ]

    public struct Sizes: Sendable {
        public var factsPerNode = 10
        public var pairsPerOrder = 5
        public var unknown = 10
        public var heldOutPerNode = 8
        public var commonsHeldOut = 32
        public var voicePerNode = 4
        public var trajectory = TrajectoryBench.Sizes()
        /// Also read the fact rules on every fact prompt (`UmbrellaReport.allFacts`).
        public var everyFact = true
        /// At most this many two-fact prompts, spread evenly over the ordered pairs of nodes
        /// (pairs grow as N(N − 1)); nil: every one.
        public var pairsTotal: Int? = nil
        /// Build the held-out, told and continued texts (bench-scale reads neither).
        public var texts = true

        public init() {}
    }

    /// The prompts and texts every arm answers.
    struct Sets {
        var facts: [BraidExample]
        var pairs: [BraidExample]
        var unknown: [BraidExample]
        var generic: [BraidExample]
        var commons: [BraidExample]
        var heldOut: [[Int]]
        /// How many of `heldOut`, from the start, are unfed text in the Threads' own voices.
        var ownHeldOut: Int = 0
        var told: [TrajectoryText]
        var continued: [[Int]]
    }

    // MARK: - Running

    public static func run(
        base: BraidLayout, reference: BraidLayout, ceiling: BraidLayout?, tokenizer: RaoTokenizer, sizes: Sizes = Sizes(),
        owner: String = "raolm-braid", progress: ((String) -> Void)? = nil
    ) throws -> UmbrellaReport {
        guard let record = MockWorld.Record.load(base), let referenceRecord = MockWorld.Record.load(reference) else {
            throw BraidSessionError.io("both braids need a world.json: run each with raolm braid demo --dataset …")
        }
        var comparable = referenceRecord
        comparable.preset = record.preset
        comparable.arm = record.arm
        guard record.sameWorld(as: comparable) else {
            throw BraidSessionError.io("the reference braid was fed another world (\(referenceRecord.summary)); the base braid's is \(record.summary)")
        }
        let world = try RoutingBench.world(layout: base, seed: record.seed)
        let (pack, strands) = try RoutingBench.packStrands(layout: base, tokenizer: tokenizer, owner: owner, names: world.names)
        let (referencePack, referenceStrands) = try RoutingBench.packStrands(layout: reference, tokenizer: tokenizer, owner: owner, names: world.names)
        guard !strands.isEmpty, !referenceStrands.isEmpty else { throw BraidSessionError.noLiveNodes }
        guard pack.hasBase else { throw BraidSessionError.io("the base braid's nodes run no umbrella pack with a base model: start it with --preset base") }

        let sets = self.sets(world: world, base: base, pack: pack, strands: strands, tokenizer: tokenizer, sizes: sizes)
        progress?("prompts: \(sets.facts.count) facts, \(sets.pairs.count) pairs, \(sets.unknown.count) unknown, \(sets.generic.count) generic, "
                  + "\(sets.commons.count) commons, \(sets.heldOut.count) held-out texts, \(sets.told.count) told")
        let threadOf = Dictionary(uniqueKeysWithValues: strands.map { ($0.name, $0.threadID) })

        // The arms, each adding to the one before.
        struct Run {
            let name: String
            let generator: BraidedGenerator
            let links: [StrandLink]
            let gate: BraidGate
            let threadOf: [String: String?]
        }
        var runs: [Run] = []
        // v1: the reference braid as it was.
        let (_, referenceLinks, referenceGenerator) = try RoutingBench.umbrella(pack: referencePack, strands: referenceStrands, tokenizer: tokenizer)
        runs.append(Run(name: "v1", generator: referenceGenerator, links: referenceLinks, gate: BraidRequest.defaultGate,
                        threadOf: Dictionary(uniqueKeysWithValues: referenceStrands.map { ($0.name, $0.threadID) })))
        // pack: the base braid without the commons strand.
        let (umbrella, links, _) = try RoutingBench.umbrella(pack: pack, strands: strands, tokenizer: tokenizer)
        runs.append(Run(name: "pack", generator: try BraidedGenerator(links: links, head: umbrella.head, tokenizer: tokenizer, packSHA256: umbrella.packSHA256),
                        links: links, gate: BraidRequest.defaultGate, threadOf: threadOf))
        // commons: the umbrella as it runs, the commons strand among the Threads.
        runs.append(Run(name: "commons", generator: try umbrella.generator(links: links), links: links, gate: BraidRequest.defaultGate, threadOf: threadOf))
        // calibrated: each Thread's own λ and τ.
        var calibrations: [String: StrandCalibration] = [:]
        for strand in strands {
            progress?("self-trajectory of \(strand.name)")
            calibrations[strand.name] = SelfTrajectory.calibrate(
                model: strand.model, index: strand.index, corpus: try strand.context.tokenizedCorpus(), alpha: strand.index.info.alpha)
        }
        let calibrated = strands.map { strand -> LocalStrandLink in
            let link = LocalStrandLink(strand: strand, vocabularySHA256: pack.vocabulary.sha256, packSHA256: umbrella.packSHA256)
            link.descriptor.calibration = calibrations[strand.name]
            return link
        }
        let calibratedGenerator = try umbrella.generator(links: calibrated)
        runs.append(Run(name: "calibrated", generator: calibratedGenerator, links: calibrated, gate: BraidRequest.defaultGate, threadOf: threadOf))
        // anchors: thought agreement in the gate.
        var thinking = BraidRequest.defaultGate
        thinking.thoughtAgreement = true
        runs.append(Run(name: "anchors", generator: calibratedGenerator, links: calibrated, gate: thinking, threadOf: threadOf))

        var arms: [UmbrellaArmResult] = []
        for run in runs {
            progress?("arm \(run.name)")
            arms.append(try arm(run.name, generator: run.generator, links: run.links, gate: run.gate, sets: sets, tokenizer: tokenizer, threadOf: run.threadOf))
        }

        var ceilingRun: Run?
        var ceilingArm: UmbrellaArmResult?
        var nodes = try nodeResults("reference", layout: reference, names: world.names)
        nodes += try nodeResults("base", layout: base, names: world.names).map { result in
            var copy = result
            copy.calibration = calibrations[result.node]
            return copy
        }
        if let ceiling {
            let (ceilingPack, ceilingStrands) = try RoutingBench.packStrands(layout: ceiling, tokenizer: tokenizer, owner: owner, names: world.names)
            let (ceilingUmbrella, ceilingLinks, _) = try RoutingBench.umbrella(pack: ceilingPack, strands: ceilingStrands, tokenizer: tokenizer)
            let run = Run(
                name: "pack (twice the documents)",
                generator: try BraidedGenerator(links: ceilingLinks, head: ceilingUmbrella.head, tokenizer: tokenizer, packSHA256: ceilingUmbrella.packSHA256),
                links: ceilingLinks, gate: BraidRequest.defaultGate, threadOf: Dictionary(uniqueKeysWithValues: ceilingStrands.map { ($0.name, $0.threadID) }))
            progress?("the pack arm on the ceiling braid")
            ceilingArm = try arm(run.name, generator: run.generator, links: run.links, gate: run.gate, sets: factsOnly(sets), tokenizer: tokenizer,
                                 threadOf: run.threadOf)
            ceilingRun = run
            nodes += try nodeResults("ceiling", layout: ceiling, names: world.names)
        }
        progress?("the trajectory rules on the base braid")
        let trajectory = try TrajectoryBench.run(layout: base, tokenizer: tokenizer, sizes: sizes.trajectory, arms: .never, owner: owner).evaluation.rules
        var report = UmbrellaReport(
            createdAt: .wholeSecond(), pack: pack.sha256, arms: arms, ceiling: ceilingArm, nodes: nodes, trajectory: trajectory,
            evaluation: evaluate(arms: arms, ceiling: ceilingArm, nodes: nodes, trajectory: trajectory))

        // The second reading: the fact measures on every fact prompt.
        if sizes.everyFact {
            var everySizes = sizes
            everySizes.factsPerNode = .max
            let every = factsOnly(self.sets(world: world, base: base, pack: pack, strands: strands, tokenizer: tokenizer, sizes: everySizes))
            var allFacts: [UmbrellaArmResult] = []
            for run in runs + (ceilingRun.map { [$0] } ?? []) {
                progress?("every fact (\(every.facts.count) prompts): \(run.name)")
                allFacts.append(try arm(run.name, generator: run.generator, links: run.links, gate: run.gate, sets: every, tokenizer: tokenizer,
                                        threadOf: run.threadOf))
            }
            report.allFactsCeiling = ceilingRun == nil ? nil : allFacts.removeLast()
            report.allFacts = allFacts
            report.allFactsEvaluation = evaluateEveryFact(report)
        }
        return report
    }

    /// The sets with only their fact prompts.
    static func factsOnly(_ sets: Sets) -> Sets {
        var facts = sets
        facts.pairs = []
        facts.unknown = []
        facts.generic = []
        facts.commons = []
        facts.heldOut = []
        facts.ownHeldOut = 0
        facts.told = []
        facts.continued = []
        return facts
    }

    /// The rules read with every arm's fact measures taken from every fact prompt; nil without that reading.
    public static func evaluateEveryFact(_ report: UmbrellaReport) -> UmbrellaEvaluation? {
        guard let allFacts = report.allFacts else { return nil }
        let arms = report.arms.map { arm -> UmbrellaArmResult in
            guard let every = allFacts.first(where: { $0.arm == arm.arm }) else { return arm }
            var merged = arm
            merged.facts = every.facts
            merged.factsExact = every.factsExact
            merged.factsOwned = every.factsOwned
            merged.citation = every.citation
            merged.ownerShare = every.ownerShare
            merged.liftPositive = every.liftPositive
            merged.ownerCredit = every.ownerCredit
            merged.formBits = every.formBits
            return merged
        }
        return evaluate(arms: arms, ceiling: report.allFactsCeiling, nodes: report.nodes, trajectory: report.trajectory)
    }

    static func sets(
        world: MockWorld, base: BraidLayout, pack: UmbrellaPack, strands: [ThreadStrand], tokenizer: RaoTokenizer, sizes: Sizes
    ) -> Sets {
        // Texts are read the way the braid's indexes read documents: with the paragraph break, if any.
        let paragraphBreak = strands.first?.index.info.paragraphBreak ?? []
        let present: [BraidExample.Node] = strands.map { strand in
            (name: strand.name, label: strand.label, threadID: strand.threadID,
             documents: MockFeeder.present(node: strand.name, world: world, layout: base.node(strand.name)))
        }
        let nodes = present.map { (name: $0.name, label: $0.label, threadID: $0.threadID, documents: world.exclusive($0.documents)) }
        var heldOut: [[Int]] = []
        for strand in strands where sizes.texts {
            let documents = (try? JSONCoding.readLines(CorpusDocument.self, from: base.node(strand.name).heldOut)) ?? []
            heldOut += documents.prefix(sizes.heldOutPerNode).map { HeldOut.tokens($0, tokenizer: tokenizer, paragraphBreak: paragraphBreak) }
        }
        let ownHeldOut = heldOut.filter { $0.count > 1 }.count
        if sizes.texts { heldOut += pack.heldOut.prefix(sizes.commonsHeldOut).map(\.tokens) }
        let texts = sizes.texts ? TrajectoryTexts.build(world: world, layout: base, names: strands.map(\.name), tokenizer: tokenizer,
                                                        sizes: sizes.trajectory, paragraphBreak: paragraphBreak) : []
        var continued = texts.filter { $0.kind == .generic }.map(\.tokens)
        for strand in strands {
            continued += texts.filter { $0.kind == .voice && $0.owner == strand.name }.prefix(sizes.voicePerNode).map { Array($0.tokens.prefix(sizes.trajectory.minTokens)) }
        }
        return Sets(
            facts: BraidExample.facts(nodes: nodes, tokenizer: tokenizer, perNode: sizes.factsPerNode),
            pairs: budgeted(BraidExample.pairs(nodes: nodes, tokenizer: tokenizer, perPair: sizes.pairsPerOrder), total: sizes.pairsTotal),
            unknown: BraidExample.unknowns(nodes: nodes, tokenizer: tokenizer, count: sizes.unknown),
            generic: BraidExample.generic(tokenizer: tokenizer),
            commons: commonsPrompts.map { prompt in
                BraidExample(label: "commons · \(prompt)", node: nil, promptTokens: tokenizer.encode(prompt), promptText: prompt, expected: nil,
                             source: nil, kind: .generic)
            },
            heldOut: heldOut.filter { $0.count > 1 }, ownHeldOut: ownHeldOut,
            told: texts.filter { $0.kind == .told && !$0.answers.isEmpty && $0.near != nil },
            continued: continued)
    }

    /// At most `total` examples, picked at an even stride so every ordered pair of nodes is drawn alike.
    static func budgeted(_ examples: [BraidExample], total: Int?) -> [BraidExample] {
        guard let total, total >= 0, examples.count > total else { return examples }
        guard total > 0 else { return [] }
        return (0..<total).map { examples[$0 * examples.count / total] }
    }

    static func arm(
        _ name: String, generator: BraidedGenerator, links: [StrandLink], gate: BraidGate, sets: Sets, tokenizer: RaoTokenizer,
        threadOf: [String: String?], observe: ((CitedGeneration) -> Void)? = nil, routing: Bool = false, router: BraidRouterKind = .bigram
    ) throws -> UmbrellaArmResult {
        let started = Date()
        let commons = generator.commons
        let commonsName = commons.map { generator.names[$0] }
        func params(_ maxTokens: Int) -> GenerationParameters {
            var params = GenerationParameters(tapLayer: links[0].descriptor.tapLayer, alpha: links[0].descriptor.alpha)
            params.maxTokens = maxTokens
            return params
        }
        func generate(_ tokens: [Int], _ text: String, maxTokens: Int) throws -> CitedGeneration {
            let generation = try generator.generate(BraidRequest(promptTokens: tokens, promptText: text, params: params(maxTokens), gate: gate,
                                                                 routing: routing, router: router))
            observe?(generation)
            return generation
        }
        func threads(_ shares: [StrandShare]) -> [StrandShare] { shares.filter { $0.strand != commonsName } }
        func owned(_ trace: TokenTrace?, owner: String?) -> Bool {
            guard let owner, let trace else { return false }
            return trace.dominantStrand(threshold: 0.5)?.strand == owner && trace.citations.first?.address.threadID != nil
                && trace.citations.first?.address.threadID == threadOf[owner] ?? nil
        }
        // The owner's blend, weighted by bits: what `earns` take of the traces' credit.
        var weighed: [String: (credit: Double, bits: Double)] = [:]
        func weigh(_ key: String, _ traces: some Sequence<TokenTrace>, _ earns: (StrandShare) -> Bool) {
            for trace in traces {
                guard let bits = trace.bits, let strands = trace.strands else { continue }
                let credit = strands.filter(earns).reduce(0.0) { $0 + Double($1.credit ?? 0) }
                weighed[key, default: (0, 0)].credit += Double(bits) * credit
                weighed[key, default: (0, 0)].bits += Double(bits)
            }
        }
        func weighted(_ key: String) -> Float? {
            guard let value = weighed[key], value.bits > 0 else { return nil }
            return Float(value.credit / value.bits)
        }

        var result = UmbrellaArmResult(
            arm: name, facts: 0, factsExact: 0, factsOwned: 0, citation: nil, ownerShare: nil, liftPositive: nil, pairs: 0, pairsMoved: 0,
            pairsExact: 0, unknownAllAsked: 0, unknownLargest: 0, genericAllAsked: 0, genericLargest: 0, commonsLeads: nil,
            commonsThreadGate: nil, commonsThreadShare: nil, heldOutNLL: 0, heldOutTokens: 0, toldSourceShare: nil, toldTokens: 0,
            voiceLargest: nil, seconds: 0)

        // Facts.
        var cited: [Bool] = []
        var ownerShares: [Float] = []
        var lifts: [Bool] = []
        for example in sets.facts {
            guard let expected = example.expected else { continue }
            let answerLength = max(1, tokenizer.encode(expected).count)
            let generation = try generate(example.promptTokens, example.promptText, maxTokens: answerLength + 2)
            let generated = generation.traces.filter { !$0.isPrompt }
            result.facts += 1
            if generation.text.hasPrefix(expected) { result.factsExact += 1 }
            if owned(generated.first, owner: example.node) { result.factsOwned += 1 }
            for trace in generated.prefix(answerLength) {
                // Rows are the generation's own: a routed generation numbers its candidates' partitions alone.
                if let source = example.source, let top = trace.neighbours.first, let partition = generation.partition(row: top.cited.row) {
                    cited.append(partition.documentID == source.documentID && partition.partitionIndex == source.partitionIndex)
                }
                if let share = trace.strands?.first(where: { $0.strand == example.node }) {
                    ownerShares.append(share.share)
                    if commons != nil, let lift = share.lift { lifts.append(lift > 0) }
                }
            }
            if let owner = example.node {
                let answer = generated.prefix(answerLength)
                weigh("owner", answer) { $0.strand == owner }
                weigh("form", answer.filter { $0.role == .form }) { _ in true }
                weigh("form", answer.filter { $0.role == .content }) { _ in false }
            }
        }
        result.ownerCredit = weighted("owner")
        result.formBits = weighted("form")
        result.citation = cited.isEmpty ? nil : Float(cited.filter { $0 }.count) / Float(cited.count)
        result.ownerShare = ownerShares.isEmpty ? nil : Stats.mean(ownerShares)
        result.liftPositive = lifts.isEmpty ? nil : Float(lifts.filter { $0 }.count) / Float(lifts.count)

        // Two facts.
        for example in sets.pairs {
            guard let expected = example.expected else { continue }
            let generation = try generate(example.promptTokens, example.promptText, maxTokens: max(1, tokenizer.encode(expected).count) + 2)
            result.pairs += 1
            if generation.text.hasPrefix(expected) { result.pairsExact += 1 }
            if owned(generation.traces.first { !$0.isPrompt }, owner: example.node) { result.pairsMoved += 1 }
        }

        // Nobody's subject, and generic text.
        func firstGate(_ examples: [BraidExample]) throws -> (asked: Float, largest: Float) {
            var asked = 0
            var largest: [Float] = []
            for example in examples {
                let generation = try generate(example.promptTokens, example.promptText, maxTokens: 4)
                let shares = threads(generation.traces.first { !$0.isPrompt }?.strands ?? [])
                if !shares.isEmpty, shares.allSatisfy(\.open) { asked += 1 }
                largest.append(shares.map(\.gate).max() ?? 1)
            }
            return (examples.isEmpty ? 0 : Float(asked) / Float(examples.count), Stats.mean(largest))
        }
        (result.unknownAllAsked, result.unknownLargest) = try firstGate(sets.unknown)
        (result.genericAllAsked, result.genericLargest) = try firstGate(sets.generic)

        // General English: does the commons lead, and what do the Threads take of it?
        if let commonsName {
            var leads = 0
            var gates: [Float] = []
            var threadShares: [Float] = []
            let prompts = sets.generic + sets.commons
            for (i, example) in prompts.enumerated() {
                let generation = try generate(example.promptTokens, example.promptText, maxTokens: 8)
                let generated = generation.traces.filter { !$0.isPrompt }
                let shares = generated.first?.strands ?? []
                let own = shares.first { $0.strand == commonsName }?.gate ?? 0
                let most = threads(shares).map(\.gate).max() ?? 0
                if own >= most { leads += 1 }
                gates.append(most)
                if i >= sets.generic.count {
                    for trace in generated { threadShares.append(threads(trace.strands ?? []).map(\.share).reduce(0, +)) }
                    weigh("commons", generated) { $0.strand == commonsName }
                }
            }
            result.commonsLeads = prompts.isEmpty ? nil : Float(leads) / Float(prompts.count)
            result.commonsThreadGate = gates.isEmpty ? nil : Stats.mean(gates)
            result.commonsThreadShare = threadShares.isEmpty ? nil : Stats.mean(threadShares)
            result.commonsCreditGeneral = weighted("commons")
        }

        // Held-out text, scored.
        var nll = 0.0
        for (i, tokens) in sets.heldOut.enumerated() {
            let traces = try generate(tokens, "", maxTokens: 0).traces
            if i < sets.ownHeldOut { weigh("threads", traces) { $0.strand != commonsName } }
            for trace in traces {
                nll -= log(Double(max(trace.mixedProb, 1e-12)))
                result.heldOutTokens += 1
            }
        }
        result.heldOutNLL = result.heldOutTokens > 0 ? Float(nll / Double(result.heldOutTokens)) : 0
        result.threadsCreditOwnVoice = weighted("threads")

        // Told texts: what the Thread the entity came from takes of its facts' answers.
        var sourceShares: [Float] = []
        for text in sets.told {
            let traces = try generate(text.tokens, "", maxTokens: 0).traces
            let byIndex = Dictionary(uniqueKeysWithValues: traces.map { ($0.index, $0) })
            for answer in text.answers {
                for position in answer.start..<(answer.start + answer.count) where position >= 1 {
                    if let share = byIndex[position]?.strands?.first(where: { $0.strand == text.near })?.share { sourceShares.append(share) }
                    if let trace = byIndex[position] { weigh("source", [trace]) { $0.strand == text.near } }
                }
            }
        }
        result.toldSourceShare = sourceShares.isEmpty ? nil : Stats.mean(sourceShares)
        result.toldTokens = sourceShares.count
        result.toldSourceCredit = weighted("source")

        // Voice and generic texts continued.
        var largest: [Float] = []
        for tokens in sets.continued {
            let generation = try generate(tokens, "", maxTokens: 4)
            if let shares = generation.traces.first(where: { !$0.isPrompt })?.strands { largest.append(threads(shares).map(\.gate).max() ?? 1) }
        }
        result.voiceLargest = largest.isEmpty ? nil : Stats.mean(largest)
        result.seconds = Date().timeIntervalSince(started)
        return result
    }

    /// Each node's live version, and the steps its first trained version took to memorise 97%.
    static func nodeResults(_ braid: String, layout: BraidLayout, names: [String]) throws -> [UmbrellaNodeResult] {
        var results: [UmbrellaNodeResult] = []
        for name in names {
            let node = layout.node(name)
            guard let live = try? JSONCoding.read(LivePointer.self, from: node.live),
                  let version = try? JSONCoding.read(NodeVersion.self, from: node.version(live.version).appendingPathComponent(NodeVersion.fileName))
            else { continue }
            var steps: Int?
            for number in node.versionNumbers() {
                guard let record = try? JSONCoding.read(NodeVersion.self, from: node.version(number).appendingPathComponent(NodeVersion.fileName)),
                      record.kind == .train, record.parent == nil,
                      let manifest = try? RunManifest.load(node.version(number)) else { continue }
                var total = 0
                for epoch in manifest.epochs.sorted(by: { $0.epoch < $1.epoch }) {
                    total += epoch.steps
                    if let memorised = epoch.evalMemorisedFraction, memorised >= 0.97 {
                        steps = total
                        break
                    }
                }
                break
            }
            results.append(UmbrellaNodeResult(
                braid: braid, node: name, version: version.version, documents: version.documents, memorised: version.memorised,
                stepsTo97: steps, heldOutLoss: version.heldOutLoss, commonsLoss: version.commonsLoss, calibration: version.calibration))
        }
        return results
    }

    // MARK: - The rule (pure)

    /// The rules, fixed before any numbers:
    /// - U0 (pack): every base node memorises 97% within 1,000 steps; fact prompts answered exactly at
    ///   least as often as v1; citation@1 within 0.02 of v1's; the base nodes' mean held-out loss on
    ///   their own voice at most v1's; and fed twice the documents, exact and citation@1 fall by at
    ///   most 0.02.
    /// "One prompt" below means one prompt in 30, as a rate, so the rules read the same on every fact.
    /// - U1 (every arm after v1): facts exact at most one prompt below v1; first answer tokens led by
    ///   and cited to the owner at least as often as v1.
    /// - U2 (every arm after v1): the lead moves and is cited on ≥ 90% of two-fact prompts.
    /// - U3 (commons and after): the commons leads on ≥ 90% of generic and commons prompts, with the
    ///   largest Thread gate there ≤ 0.35 on average.
    /// - U4 (commons and after): the Threads' summed share of commons prompts ≤ 0.10; on facts the
    ///   owner's share ≥ 0.90 and its lift over the commons positive on ≥ 95% of answer tokens.
    /// - U5 (calibrated): held-out NLL at most the commons arm's; facts exact within one prompt of it;
    ///   citation@1 within 0.02 of it.
    /// - U6 (anchors): the source Thread's share of told answers ≥ the calibrated arm's + 0.10; the
    ///   largest Thread gate on voice and generic texts within 0.05 of it; facts exact within one prompt.
    /// - T/M: the trajectory bench's T1 to T3 and M1 to M4 pass on the base braid (pack and after).
    /// An arm qualifies when its rules and every earlier arm's hold. The winner is the last arm that
    /// qualifies; where its own measure moved by less than 0.02 over the arm before (held-out NLL for
    /// calibrated), the arm before wins.
    public static func evaluate(
        arms: [UmbrellaArmResult], ceiling: UmbrellaArmResult?, nodes: [UmbrellaNodeResult], trajectory: [TrajectoryRuleResult]
    ) -> UmbrellaEvaluation {
        func of(_ name: String) -> UmbrellaArmResult? { arms.first { $0.arm == name } }
        func pct(_ value: Float?) -> String { value.map { String(format: "%.0f%%", $0 * 100) } ?? "—" }
        func num(_ value: Float?) -> String { value.map { String(format: "%.3f", $0) } ?? "—" }
        var rules: [TrajectoryRuleResult] = []
        var reported: [String: Float] = [:]
        guard let v1 = of("v1") else {
            return UmbrellaEvaluation(rules: [], qualifies: [], winner: nil, reported: [:], summary: "no v1 arm: nothing to compare against")
        }
        var failed: [String: [String]] = [:]
        /// `arm` answers exactly at most one prompt in 30 less often than `than` (one prompt of the fixed 30).
        func level(_ arm: UmbrellaArmResult, _ than: UmbrellaArmResult) -> Bool {
            arm.facts > 0 && than.facts > 0
                && Double(arm.factsExact) / Double(arm.facts) >= Double(than.factsExact) / Double(than.facts) - 1.0 / 30 - 1e-9
        }
        func record(_ arm: String, _ rule: String, _ passed: Bool, _ detail: String) {
            rules.append(TrajectoryRuleResult(rule: "\(rule) · \(arm)", passed: passed, detail: detail))
            if !passed { failed[arm, default: []].append(rule) }
        }

        // U0: the base braid stands on its own.
        if let pack = of("pack") {
            let base = nodes.filter { $0.braid == "base" }
            let reference = nodes.filter { $0.braid == "reference" }
            let slow = base.filter { ($0.stepsTo97 ?? Int.max) > 1_000 }
            let baseLoss = base.compactMap(\.heldOutLoss)
            let referenceLoss = reference.compactMap(\.heldOutLoss)
            let lossOK = !baseLoss.isEmpty && (referenceLoss.isEmpty || Stats.mean(baseLoss) <= Stats.mean(referenceLoss))
            var detail = "97% within 1,000 steps on \(base.count - slow.count) of \(base.count) nodes (\(base.map { "\($0.node) \($0.stepsTo97.map(String.init) ?? "never")" }.joined(separator: ", ")))"
            detail += "; exact \(pct(pack.factsExactRate)) against v1's \(pct(v1.factsExactRate)); citation@1 \(pct(pack.citation)) against \(pct(v1.citation))"
            detail += "; own-voice held-out loss \(num(baseLoss.isEmpty ? nil : Stats.mean(baseLoss))) against \(num(referenceLoss.isEmpty ? nil : Stats.mean(referenceLoss)))"
            var passed = !base.isEmpty && slow.isEmpty && pack.factsExactRate >= v1.factsExactRate
                && (pack.citation ?? 0) >= (v1.citation ?? 0) - 0.02 && lossOK
            if let ceiling {
                let holds = ceiling.factsExactRate >= pack.factsExactRate - 0.02 && (ceiling.citation ?? 0) >= (pack.citation ?? 0) - 0.02
                passed = passed && holds
                detail += "; twice the documents: exact \(pct(ceiling.factsExactRate)), citation@1 \(pct(ceiling.citation))"
            } else {
                passed = false
                detail += "; no ceiling braid was given"
            }
            record("pack", "U0 base stands on its own", passed, detail)
        }
        for arm in arms where arm.arm != "v1" {
            record(arm.arm, "U1 facts", level(arm, v1) && arm.factsOwnedRate >= v1.factsOwnedRate,
                   "exact \(arm.factsExact)/\(arm.facts) (v1 \(v1.factsExact)); led and cited \(pct(arm.factsOwnedRate)) (v1 \(pct(v1.factsOwnedRate)))")
            record(arm.arm, "U2 two facts", arm.pairsMovedRate >= 0.9, "the lead moved and was cited on \(pct(arm.pairsMovedRate)) of \(arm.pairs)")
            if arm.arm != "pack" {
                let leads = arm.commonsLeads ?? 0
                let gate = arm.commonsThreadGate ?? 1
                record(arm.arm, "U3 nobody's text", leads >= 0.9 && gate <= 0.35,
                       "the commons leads on \(pct(arm.commonsLeads)) of generic and commons prompts; largest Thread gate \(num(arm.commonsThreadGate))")
                let taken = arm.commonsThreadShare ?? 1
                record(arm.arm, "U4 inherited knowledge is not attributed",
                       taken <= 0.10 && (arm.ownerShare ?? 0) >= 0.90 && (arm.liftPositive ?? 0) >= 0.95,
                       "Threads take \(num(arm.commonsThreadShare)) of commons prompts; the owner \(num(arm.ownerShare)) of fact answers, lift positive on \(pct(arm.liftPositive))")
            }
            record(arm.arm, "T/M trajectory rules", trajectory.allSatisfy(\.passed) && !trajectory.isEmpty,
                   trajectory.filter { !$0.passed }.map(\.rule).joined(separator: ", ").isEmpty ? "all pass"
                       : "failed: " + trajectory.filter { !$0.passed }.map(\.rule).joined(separator: ", "))
        }
        if let calibrated = of("calibrated"), let commons = of("commons") {
            record("calibrated", "U5 calibrated λ and τ",
                   calibrated.heldOutNLL <= commons.heldOutNLL && level(calibrated, commons)
                       && (calibrated.citation ?? 0) >= (commons.citation ?? 0) - 0.02,
                   String(format: "held-out NLL %.4f against %.4f; exact %d against %d; citation@1 %@ against %@", calibrated.heldOutNLL,
                          commons.heldOutNLL, calibrated.factsExact, commons.factsExact, pct(calibrated.citation), pct(commons.citation)))
            reported["calibrated: held-out NLL gain"] = commons.heldOutNLL - calibrated.heldOutNLL
        }
        if let anchors = of("anchors"), let calibrated = of("calibrated") {
            let rise = (anchors.toldSourceShare ?? 0) - (calibrated.toldSourceShare ?? 0)
            let gateMoved = abs((anchors.voiceLargest ?? 0) - (calibrated.voiceLargest ?? 0))
            record("anchors", "U6 anchors", rise >= 0.10 && gateMoved <= 0.05 && level(anchors, calibrated),
                   "the source's share of told answers \(num(anchors.toldSourceShare)) against \(num(calibrated.toldSourceShare)); largest gate on voice and generic moved \(num(gateMoved)); exact \(anchors.factsExact) against \(calibrated.factsExact)")
        }
        for node in nodes where node.braid == "base" {
            if let calibration = node.calibration {
                reported["\(node.node): false-chain rate"] = calibration.falseChainRate
                reported["\(node.node): τ"] = calibration.tau
                reported["\(node.node): λ scale"] = calibration.lambdaScale
            }
            if let loss = node.commonsLoss { reported["\(node.node): commons held-out loss"] = loss }
        }

        // Qualification is cumulative.
        var qualifies: [String] = []
        for name in armNames.dropFirst() {
            guard of(name) != nil, failed[name] == nil else { break }
            qualifies.append(name)
        }
        var winner = qualifies.last
        if winner == "calibrated", (reported["calibrated: held-out NLL gain"] ?? 0) < 0.02 { winner = "commons" }
        let summary: String
        if let winner {
            summary = "\(winner) wins: every rule up to it holds" + (winner != qualifies.last ? " (\(qualifies.last ?? "") qualifies but moves its measure by less than 0.02)" : "")
        } else {
            let first = armNames.dropFirst().first { failed[$0] != nil } ?? "pack"
            summary = "no arm qualifies: \(first) fails " + (failed[first] ?? []).joined(separator: ", ")
        }
        return UmbrellaEvaluation(rules: rules, qualifies: qualifies, winner: winner, reported: reported, summary: summary)
    }
}
