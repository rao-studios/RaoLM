//
//  ArchitectureBench.swift
//  RaoLMBraid
//
//  WHAT: Whether a phase-2 training arm earns its place in `base`. The arm's braid is compared with
//        today's base braid, both fed the same documents and read through the same gate: how fast
//        each node memorises, what it answers (in the braid, and alone from its facts' own
//        prompts), its loss on unfed text in its own voice, whether it writes the paragraph break
//        where a passage ends, and that no citation lands on a break.
//  OUT:  One report: both braids' measures, every node's, the trajectory rules on the arm's braid,
//        and `evaluate`'s verdict.
//  PIN:  The rules were fixed before any numbers (Docs/ARCHITECTURE.md, "Phase 2, first arm").
//        Held-out loss is scored on the partitions' own tokens in both braids, so a stream with
//        breaks is not scored on more targets than one without. Nothing is written to a node.
//

import Foundation
import MLX
import RaoLMCore
import RaoLMModel
import RaoLMProvenance
import RaoLMTraining

public struct ArchitectureCurvePoint: Codable, Sendable, Equatable {
    public var steps: Int
    public var memorised: Float
}

public struct ArchitectureNodeResult: Codable, Sendable, Equatable {
    /// "arm" or "reference".
    public var braid: String
    public var node: String
    public var version: Int
    public var memorised: Float
    /// Optimizer steps the node's first version took to memorise 97% (nil: it did not).
    public var stepsTo97: Int?
    /// Eval memorisation through the first version's training.
    public var curve: [ArchitectureCurvePoint]
    /// The first trained version's steps in all (to its anneal's end, with one), and its median
    /// seconds per training step.
    public var steps: Int? = nil
    public var stepSeconds: Double? = nil
    /// Mean loss on unfed documents in the node's own voice, on the partitions' own tokens.
    public var heldOutLoss: Float?
    /// The live version's recorded loss on the pack's commons sample.
    public var commonsLoss: Float?
    /// The node's located facts, and those it answers alone from their own prompts, greedily
    /// (every answer token its likeliest).
    public var greedyFacts: Int
    public var greedyCorrect: Int
    /// Partition boundaries in the unfed documents, read teacher-forced, and those where the node's
    /// likeliest continuation is the paragraph break: both its tokens where the node's stream has
    /// them, the first where it does not.
    public var boundaries: Int
    public var breaksPredicted: Int

    public var greedyRate: Float { greedyFacts > 0 ? Float(greedyCorrect) / Float(greedyFacts) : 0 }
    public var breakRate: Float { boundaries > 0 ? Float(breaksPredicted) / Float(boundaries) : 0 }
}

public struct ArchitectureBraidResult: Codable, Sendable, Equatable {
    /// "arm" or "reference".
    public var braid: String
    /// Fact prompts through the braid, as bench-umbrella measures them (only its facts are asked).
    public var facts: UmbrellaArmResult
    /// Fed documents continued through the braid from the end of their first partition: how many,
    /// those whose continuation opens with the paragraph break, and those that run straight on.
    public var continuations: Int
    public var breaksWritten: Int
    public var ranTogether: Int
    /// Citations and spans in every generation of the bench, and those landing on a break.
    public var citations: Int
    public var citationsOnBreaks: Int
}

public struct ArchitectureEvaluation: Codable, Sendable, Equatable {
    public var rules: [TrajectoryRuleResult]
    public var qualifies: Bool
    public var reported: [String: Float]
    public var summary: String
}

public struct ArchitectureReport: Codable, Sendable {
    public var createdAt: Date
    public var arm: String
    /// The arm the reference braid trains (nil: the recipe before arms).
    public var referenceArm: String? = nil
    public var braids: [ArchitectureBraidResult]
    public var nodes: [ArchitectureNodeResult]
    public var trajectory: [TrajectoryRuleResult]
    public var evaluation: ArchitectureEvaluation
}

public enum ArchitectureBench {
    public struct Sizes: Sendable {
        /// Fed documents per node continued from the end of their first partition.
        public var continuationsPerNode = 10
        /// The tail of the first partition a continuation starts from, and how far it runs.
        public var continuationPrompt = 48
        public var continuationTokens = 6
        /// An unfed document is read teacher-forced up to this many tokens.
        public var documentTokens = 2_048
        /// bench-umbrella's sets, with every fact prompt of each node's documents (A2, re-checked on
        /// all of them at the owner's decision; the first run asked 10 per node).
        public var umbrella = UmbrellaBench.Sizes()

        public init() {
            umbrella.factsPerNode = .max
        }
    }

    // MARK: - Running

    public static func run(
        arm armLayout: BraidLayout, reference: BraidLayout, tokenizer: RaoTokenizer, sizes: Sizes = Sizes(), owner: String = "raolm-braid",
        progress: ((String) -> Void)? = nil
    ) throws -> ArchitectureReport {
        guard let record = MockWorld.Record.load(armLayout), let referenceRecord = MockWorld.Record.load(reference) else {
            throw BraidSessionError.io("both braids need a world.json: run each with raolm braid demo --dataset …")
        }
        guard let arm = record.arm else {
            throw BraidSessionError.io("the arm's braid trains no arm: start it with raolm braid demo --arm \(HypervisorSettings.arms.joined(separator: "|"))")
        }
        guard referenceRecord.arm != record.arm else {
            throw BraidSessionError.io("both braids train the arm \(arm); the reference must train another recipe")
        }
        var comparable = referenceRecord
        comparable.arm = record.arm
        guard record.sameWorld(as: comparable) else {
            throw BraidSessionError.io("the reference braid was fed another world (\(referenceRecord.summary)); the arm's is \(record.summary)")
        }
        let world = try RoutingBench.world(layout: armLayout, seed: record.seed)
        let (pack, strands) = try RoutingBench.packStrands(layout: armLayout, tokenizer: tokenizer, owner: owner, names: world.names)
        let (referencePack, referenceStrands) = try RoutingBench.packStrands(layout: reference, tokenizer: tokenizer, owner: owner, names: world.names)
        guard !strands.isEmpty, !referenceStrands.isEmpty else { throw BraidSessionError.noLiveNodes }

        // The same fact prompts for both braids (fed the same documents): bench-umbrella's.
        var sets = UmbrellaBench.sets(world: world, base: armLayout, pack: pack, strands: strands, tokenizer: tokenizer, sizes: sizes.umbrella)
        sets.pairs = []
        sets.unknown = []
        sets.generic = []
        sets.commons = []
        sets.heldOut = []
        sets.ownHeldOut = 0
        sets.told = []
        sets.continued = []
        progress?("prompts: \(sets.facts.count) facts")

        var braids: [ArchitectureBraidResult] = []
        var nodes: [ArchitectureNodeResult] = []
        for (name, layout, pack, strands) in [("reference", reference, referencePack, referenceStrands), ("arm", armLayout, pack, strands)] {
            progress?("the \(name) braid")
            braids.append(try braid(name, layout: layout, world: world, pack: pack, strands: strands, sets: sets, tokenizer: tokenizer, sizes: sizes))
            for strand in strands {
                progress?("\(name) \(strand.name), alone")
                nodes.append(try node(name, strand: strand, layout: layout.node(strand.name), tokenizer: tokenizer, sizes: sizes))
            }
        }
        progress?("the trajectory rules on the arm's braid")
        let trajectory = try TrajectoryBench.run(layout: armLayout, tokenizer: tokenizer, sizes: sizes.umbrella.trajectory, arms: .never, owner: owner)
            .evaluation.rules
        return ArchitectureReport(
            createdAt: .wholeSecond(), arm: arm, referenceArm: referenceRecord.arm, braids: braids, nodes: nodes, trajectory: trajectory,
            evaluation: evaluate(braids: braids, nodes: nodes, trajectory: trajectory, arm: arm))
    }

    /// The braid as it runs (the commons strand among the Threads, today's default gate): its fact
    /// answers, its continuations at a passage's end, and every citation it made.
    static func braid(
        _ name: String, layout: BraidLayout, world: MockWorld, pack: UmbrellaPack, strands: [ThreadStrand], sets: UmbrellaBench.Sets,
        tokenizer: RaoTokenizer, sizes: Sizes
    ) throws -> ArchitectureBraidResult {
        let (_, links, generator) = try RoutingBench.umbrella(pack: pack, strands: strands, tokenizer: tokenizer)
        var citations = 0
        var onBreaks = 0
        func inspect(_ generation: CitedGeneration) {
            let partitions = generator.partitionsByRow
            for trace in generation.traces {
                for citation in trace.citations {
                    citations += 1
                    if let partition = partitions[citation.row], citation.address.tokenOffset >= partition.tokenCount { onBreaks += 1 }
                }
            }
            for span in generation.spans {
                citations += 1
                if let partition = partitions[span.row], span.source.tokenOffset + span.tokens.count > partition.tokenCount { onBreaks += 1 }
            }
        }
        let threadOf = Dictionary(uniqueKeysWithValues: strands.map { ($0.name, $0.threadID) })
        let facts = try UmbrellaBench.arm(
            name, generator: generator, links: links, gate: BraidRequest.defaultGate, sets: sets, tokenizer: tokenizer, threadOf: threadOf,
            observe: inspect)

        // A passage's end: does the braid write the break, or run straight on?
        let breakTokens = tokenizer.paragraphBreak
        let newlines = Set(breakTokens.prefix(1) + tokenizer.encode("\n\n"))
        var continuations = 0
        var written = 0
        var together = 0
        for strand in strands {
            let documents = MockFeeder.present(node: strand.name, world: world, layout: layout.node(strand.name))
                .filter { $0.partitions.count >= 2 }
                .prefix(sizes.continuationsPerNode)
            for document in documents {
                guard let first = document.partitions.min(by: { $0.index < $1.index }) else { continue }
                let prompt = Array(tokenizer.encode(first.text).suffix(sizes.continuationPrompt))
                var params = GenerationParameters(tapLayer: links[0].descriptor.tapLayer, alpha: links[0].descriptor.alpha)
                params.maxTokens = sizes.continuationTokens
                let generation = try generator.generate(BraidRequest(
                    promptTokens: prompt, promptText: tokenizer.decode(prompt), params: params, gate: BraidRequest.defaultGate))
                inspect(generation)
                continuations += 1
                if generation.tokens.starts(with: breakTokens) { written += 1 }
                if let next = generation.tokens.first, !newlines.contains(next) { together += 1 }
            }
        }
        return ArchitectureBraidResult(
            braid: name, facts: facts, continuations: continuations, breaksWritten: written, ranTogether: together, citations: citations,
            citationsOnBreaks: onBreaks)
    }

    /// One node alone: how it learned, its held-out loss, its facts from their own prompts, and
    /// what it predicts where a passage ends.
    static func node(_ braid: String, strand: ThreadStrand, layout node: NodeLayout, tokenizer: RaoTokenizer, sizes: Sizes) throws -> ArchitectureNodeResult {
        let model = strand.model
        let eos = Int32(tokenizer.eosTokenID)
        let paragraphBreak = strand.index.info.paragraphBreak ?? []
        let live = try JSONCoding.read(NodeVersion.self, from: node.version(strand.version).appendingPathComponent(NodeVersion.fileName))

        // Learning: the first trained version's evaluations.
        var curve: [ArchitectureCurvePoint] = []
        var stepsTo97: Int?
        var versionSteps: Int?
        var stepSeconds: Double?
        for number in node.versionNumbers() {
            guard let record = try? JSONCoding.read(NodeVersion.self, from: node.version(number).appendingPathComponent(NodeVersion.fileName)),
                  record.kind == .train, record.parent == nil,
                  let manifest = try? RunManifest.load(node.version(number)) else { continue }
            var total = 0
            for epoch in manifest.epochs.sorted(by: { $0.epoch < $1.epoch }) {
                total += epoch.steps
                guard let memorised = epoch.evalMemorisedFraction else { continue }
                curve.append(ArchitectureCurvePoint(steps: total, memorised: memorised))
                if stepsTo97 == nil, memorised >= 0.97 { stepsTo97 = total }
            }
            versionSteps = total
            let rows = (try? JSONCoding.readLines(StepRow.self, from: LedgerFiles.steps(RunLayout.ledger(node.version(number))))) ?? []
            let seconds = rows.filter { $0.tokensPerSecond > 0 }.map { Double($0.maskedTokens) / $0.tokensPerSecond }.sorted()
            if !seconds.isEmpty { stepSeconds = seconds[seconds.count / 2] }
            break
        }

        // Unfed documents in the node's own voice, read as its stream reads them.
        let documents = (try? JSONCoding.readLines(CorpusDocument.self, from: node.heldOut)) ?? []
        let scored = documents.map { HeldOut.scoredTokens($0, tokenizer: tokenizer, paragraphBreak: paragraphBreak) }
        let seqLen = strand.context.manifest.hyperparameters.seqLen
        let heldOut = HeldOut.loss(model: model, texts: scored.map(\.tokens), eos: eos, seqLen: seqLen, scored: scored.map(\.own))

        // Where a passage ends, teacher-forced.
        var boundaries = 0
        var predicted = 0
        let expected = paragraphBreak.isEmpty ? Array(tokenizer.paragraphBreak.prefix(1)) : paragraphBreak
        for document in documents where document.partitions.count >= 2 {
            var tokens: [Int] = []
            var ends: [Int] = []
            let ordered = document.partitions.sorted { $0.index < $1.index }
            for (n, partition) in ordered.enumerated() {
                if n > 0 { tokens += paragraphBreak }
                tokens += tokenizer.encode(partition.text)
                if n < ordered.count - 1 { ends.append(tokens.count - 1) }
            }
            let sequence = [eos] + tokens.prefix(sizes.documentTokens).map { Int32($0) }
            let likeliest = argMax(model.forward(MLXArray(sequence, [1, sequence.count]), captureTap: false).logits, axis: -1)
                .asArray(Int32.self)
            for end in ends where end + expected.count < sequence.count {
                // The input at sequence position end + 1 is the passage's last token.
                boundaries += 1
                if expected.indices.allSatisfy({ Int(likeliest[end + 1 + $0]) == expected[$0] }) { predicted += 1 }
            }
        }

        // Facts, from their own prompts: the corpus's tokens before the answer, no eos.
        let facts = (try? JSONCoding.readLines(Fact.self, from: node.facts)) ?? []
        let corpus = try strand.context.tokenizedCorpus()
        let located = FactLocator.locate(facts, corpus: corpus, tokenizer: tokenizer).located
            .filter { $0.answerToken > $0.contextToken && $0.answerEndToken > $0.answerToken }
        var correct = 0
        var start = 0
        while start < located.count {
            let batch = Array(located[start..<min(start + 16, located.count)])
            start += 16
            let pieces = batch.map { fact -> (prompt: [Int32], answer: [Int32]) in
                let partition = corpus.partitions[fact.row]
                return (Array(partition.tokens[fact.contextToken..<fact.answerToken]), Array(partition.tokens[fact.answerToken..<fact.answerEndToken]))
            }
            let width = pieces.map { $0.prompt.count + $0.answer.count - 1 }.max() ?? 1
            var inputs = [Int32](repeating: eos, count: batch.count * width)
            for (b, piece) in pieces.enumerated() {
                for (t, token) in (piece.prompt + piece.answer.dropLast()).enumerated() { inputs[b * width + t] = token }
            }
            let likeliest = argMax(model.forward(MLXArray(inputs, [batch.count, width]), captureTap: false).logits, axis: -1)
                .asArray(Int32.self)
            for (b, piece) in pieces.enumerated()
            where piece.answer.indices.allSatisfy({ likeliest[b * width + piece.prompt.count - 1 + $0] == piece.answer[$0] }) {
                correct += 1
            }
        }

        return ArchitectureNodeResult(
            braid: braid, node: strand.name, version: strand.version, memorised: live.memorised, stepsTo97: stepsTo97, curve: curve,
            steps: versionSteps, stepSeconds: stepSeconds, heldOutLoss: heldOut, commonsLoss: live.commonsLoss, greedyFacts: located.count, greedyCorrect: correct, boundaries: boundaries,
            breaksPredicted: predicted)
    }

    // MARK: - The rules (pure)

    /// The rules, fixed before any numbers (Docs/ARCHITECTURE.md):
    /// - A1 learning: every arm node memorises 97%, in at most 1.25× its reference node's steps.
    /// - A2 answers: facts exact at least the reference's rate minus 1/30 (one prompt of the first
    ///   30); citation@1 at least the reference's − 0.02; each node's greedy answers from its facts'
    ///   own prompts at least its reference node's − 0.02.
    /// - A3 general text: each node's held-out loss in its own voice at most its reference node's.
    /// - A4 the break: on its unfed documents each arm node's likeliest continuation is the break at
    ///   ≥ 80% of partition boundaries.
    /// - A5 provenance: no citation in the bench lands on a break; the trajectory bench's T1 to T3
    ///   and M1 to M4 pass on the arm's braid.
    /// The arm qualifies when A1 to A5 hold.
    ///
    /// The block arms (`canon`, `gated-attention`) are read on B1 to B4: learning as A1 but in at
    /// most 0.75× the reference's steps; B2 and B3 as A2 and A3; B4 provenance as A5 with A4's
    /// break as a guard on every node. `muon` is read on M1 to M4: B's, with learning also in no
    /// more training time (steps × median seconds per step) and held-out loss within 0.02.
    /// `wsd` is read on W1 to W4: each node's steps to its anneal's end at most 1.5× the
    /// reference's; exact at least the reference's + 0.02, citation@1 and each node's greedy
    /// answers at least the reference's; held-out loss within 0.02; provenance as B4.
    public static func evaluate(
        braids: [ArchitectureBraidResult], nodes: [ArchitectureNodeResult], trajectory: [TrajectoryRuleResult], arm name: String = "passage-break"
    ) -> ArchitectureEvaluation {
        let family = name == "passage-break" ? "A" : name == "muon" ? "M" : name == "wsd" ? "W" : "B"
        let blocks = family != "A"
        func pct(_ value: Float?) -> String { value.map { String(format: "%.0f%%", $0 * 100) } ?? "—" }
        func num(_ value: Float?) -> String { value.map { String(format: "%.3f", $0) } ?? "—" }
        func steps(_ value: Int?) -> String { value.map(String.init) ?? "never" }
        func rate(_ count: Int, _ total: Int) -> Double { total > 0 ? Double(count) / Double(total) : 0 }
        guard let arm = braids.first(where: { $0.braid == "arm" }), let reference = braids.first(where: { $0.braid == "reference" }) else {
            return ArchitectureEvaluation(rules: [], qualifies: false, reported: [:], summary: "both braids are needed")
        }
        let armNodes = nodes.filter { $0.braid == "arm" }
        func matching(_ node: ArchitectureNodeResult) -> ArchitectureNodeResult? {
            nodes.first { $0.braid == "reference" && $0.node == node.node }
        }
        var rules: [TrajectoryRuleResult] = []
        func rule(_ name: String, _ checks: [(passed: Bool, detail: String)], _ also: Bool = true, _ prefix: String = "") {
            rules.append(TrajectoryRuleResult(
                rule: name, passed: !checks.isEmpty && also && checks.allSatisfy(\.passed),
                detail: prefix + checks.map(\.detail).joined(separator: ", ")))
        }

        if family == "W" {
            rule("W1 learning", armNodes.map { node in
                let reference = matching(node)?.steps
                let passed = node.steps.map { found in reference.map { Double(found) <= 1.5 * Double($0) } ?? true } ?? false
                return (passed, "\(node.node) \(steps(node.steps)) (reference \(steps(reference)))")
            }, true, "steps to the end of training, at most 1.5× the reference's: ")
        } else {
            let factor = family == "A" ? 1.25 : 0.75
            rule("\(family)1 learning", armNodes.map { node in
                let reference = matching(node)
                var passed = node.stepsTo97.map { found in reference?.stepsTo97.map { Double(found) <= factor * Double($0) } ?? true } ?? false
                var detail = "\(node.node) \(steps(node.stepsTo97)) (reference \(steps(reference?.stepsTo97)))"
                if family == "M" {
                    let time = node.stepsTo97.flatMap { s in node.stepSeconds.map { Double(s) * $0 } }
                    let referenceTime = reference?.stepsTo97.flatMap { s in reference?.stepSeconds.map { Double(s) * $0 } }
                    passed = passed && (time.map { t in referenceTime.map { t <= $0 } ?? true } ?? false)
                    detail += String(format: ", %.0f s against %.0f s", time ?? -1, referenceTime ?? -1)
                }
                return (passed, detail)
            }, true, String(format: "steps to 97%% memorised, at most %.2f× the reference's", factor) + (family == "M" ? ", in no more time: " : ": "))
        }

        let gain = family == "W"
        let exact = rate(arm.facts.factsExact, arm.facts.facts)
            >= rate(reference.facts.factsExact, reference.facts.facts) + (gain ? 0.02 : -1.0 / 30) - 1e-9
        let citation = (arm.facts.citation ?? 0) >= (reference.facts.citation ?? 0) - (gain ? 0 : 0.02)
        rule("\(family)2 answers", armNodes.map { node in
            (node.greedyRate >= (matching(node)?.greedyRate ?? 0) - (gain ? 0 : 0.02),
             "\(node.node) \(pct(node.greedyRate)) of \(node.greedyFacts) (reference \(pct(matching(node)?.greedyRate)))")
        }, exact && citation,
        "facts exact \(arm.facts.factsExact)/\(arm.facts.facts) (reference \(reference.facts.factsExact)); citation@1 \(pct(arm.facts.citation)) "
            + "(reference \(pct(reference.facts.citation))); alone, greedy from their own prompts: ")

        let tolerance: Float = family == "M" || family == "W" ? 0.02 : 0
        rule("\(family)3 general text", armNodes.map { node in
            let reference = matching(node)?.heldOutLoss
            return (node.heldOutLoss.map { loss in reference.map { loss <= $0 + tolerance } ?? true } ?? false,
                    "\(node.node) \(num(node.heldOutLoss)) (reference \(num(reference)))")
        }, true, "held-out loss in their own voice: ")

        let breaks = armNodes.map { node in
            (passed: node.boundaries > 0 && node.breakRate >= 0.8, detail: "\(node.node) \(pct(node.breakRate)) of \(node.boundaries)")
        }
        if !blocks { rule("A4 the break", breaks, true, "the break likeliest at a passage's end: ") }

        let failedTrajectory = trajectory.filter { !$0.passed }.map(\.rule)
        let breaksHold = !blocks || (!breaks.isEmpty && breaks.allSatisfy(\.passed))
        rules.append(TrajectoryRuleResult(
            rule: blocks ? "\(family)4 provenance" : "A5 provenance",
            passed: arm.citationsOnBreaks == 0 && !trajectory.isEmpty && failedTrajectory.isEmpty && breaksHold,
            detail: "\(arm.citationsOnBreaks) of \(arm.citations) citations and spans on a break; trajectory rules: "
                + (trajectory.isEmpty ? "not run" : failedTrajectory.isEmpty ? "all pass" : "failed " + failedTrajectory.joined(separator: ", "))
                + (blocks ? "; the break likeliest at a passage's end: " + breaks.map(\.detail).joined(separator: ", ") : "")))

        var reported: [String: Float] = [:]
        for braid in braids { reported["\(braid.braid): seconds to answer the fact prompts"] = Float(braid.facts.seconds) }
        for braid in braids where braid.continuations > 0 {
            reported["\(braid.braid): passages run together"] = Float(braid.ranTogether) / Float(braid.continuations)
            reported["\(braid.braid): break written"] = Float(braid.breaksWritten) / Float(braid.continuations)
        }
        for node in nodes {
            if let seconds = node.stepSeconds { reported["\(node.braid) \(node.node): seconds per step"] = Float(seconds) }
            if let steps = node.steps { reported["\(node.braid) \(node.node): steps"] = Float(steps) }
            if let loss = node.commonsLoss { reported["\(node.braid) \(node.node): commons held-out loss"] = loss }
            if node.boundaries > 0 { reported["\(node.braid) \(node.node): break likeliest"] = node.breakRate }
        }
        let failed = rules.filter { !$0.passed }.map(\.rule)
        let qualifies = failed.isEmpty
        return ArchitectureEvaluation(
            rules: rules, qualifies: qualifies, reported: reported,
            summary: qualifies ? "the arm qualifies: \(blocks ? "\(family)1 to \(family)4" : "A1 to A5") hold" : "the arm does not qualify: it fails " + failed.joined(separator: ", "))
    }
}
