//
//  CommonsBench.swift
//  RaoLMBraid
//
//  WHAT: Whether a new commons is fit to be the braid's foundation (Docs/ARCHITECTURE.md, "Phase 4,
//        bench-commons"): a child pack against its parent, each under a braid fed the same
//        documents with the same recipe (the child's a rebased copy of the parent's).
//  OUT:  One report: both braids' measures on the same prompts, the held-out losses, the rules.
//  PIN:  The prompts are bench-umbrella's, cut from the child braid: every fact prompt for C2 and
//        C4's lift, its generic and commons prompts for C3 and C4's share, its questions about
//        entities nobody holds reported. C1 is the child's recipe (its first and last
//        evaluations, the same tokens) or, for a pack without one, both packs' loss on the pack's
//        held-out snippets measured now. Nothing is written to a node.
//

import Foundation
import RaoLMCore
import RaoLMModel
import RaoLMProvenance
import RaoLMTraining

public struct CommonsEvaluation: Codable, Sendable, Equatable {
    public var rules: [TrajectoryRuleResult]
    public var qualifies: Bool
    public var summary: String
}

public struct CommonsReport: Codable, Sendable {
    public var createdAt: Date
    public var child: String
    public var parent: String
    /// The child's ancestors, nearest first, by name.
    public var lineage: [String]
    /// Held-out loss per set: the parent's, then the child's.
    public var heldOut: [String: [Float]]
    /// "parent" and "child": generic, commons and unknown prompts.
    public var arms: [UmbrellaArmResult]
    /// "parent" and "child": every fact prompt.
    public var facts: [UmbrellaArmResult]
    /// "parent" and "child" → the node holding the fact → [answered exactly, asked].
    public var factsByNode: [String: [String: [Int]]]? = nil
    /// "parent" and "child" → each fact's prompt label → what the braid answered (its first line).
    public var answers: [String: [String: String]]? = nil
    public var evaluation: CommonsEvaluation
}

public enum CommonsBench {
    public static func run(
        child: BraidLayout, parent: BraidLayout, tokenizer: RaoTokenizer, sizes: UmbrellaBench.Sizes = UmbrellaBench.Sizes(), owner: String = "raolm-braid",
        progress: ((String) -> Void)? = nil
    ) throws -> CommonsReport {
        guard var childRecord = MockWorld.Record.load(child), let parentRecord = MockWorld.Record.load(parent) else {
            throw BraidSessionError.io("both braids need a world.json: run each with raolm braid demo --dataset …")
        }
        // The pack and the preset are the commons under trial, not the world: a trial of another
        // shape (raolm umbrella trial) is fed the same documents.
        childRecord.packSHA256 = parentRecord.packSHA256
        childRecord.preset = parentRecord.preset
        guard childRecord.sameWorld(as: parentRecord) else {
            throw BraidSessionError.io("the braids were fed different worlds (\(childRecord.summary) · \(parentRecord.summary))")
        }
        let world = try RoutingBench.world(layout: child, seed: childRecord.seed)
        let (childPack, childStrands) = try RoutingBench.packStrands(layout: child, tokenizer: tokenizer, owner: owner, names: world.names)
        let (parentPack, parentStrands) = try RoutingBench.packStrands(layout: parent, tokenizer: tokenizer, owner: owner, names: world.names)
        guard !childStrands.isEmpty, !parentStrands.isEmpty else { throw BraidSessionError.noLiveNodes }
        guard childPack.hasBase, parentPack.hasBase else { throw BraidSessionError.io("both braids must run a pack with a base model (--preset base)") }
        guard childPack.sha256 != parentPack.sha256 else { throw BraidSessionError.io("both braids run pack \(childPack.sha256.prefix(12)): rebase one first") }

        var lineage: [String] = []
        if let info = childPack.info, info.parent != parentPack.sha256 {
            progress?("note: the child's recorded parent is \(info.parent.map { String($0.prefix(12)) } ?? "none"), not \(parentPack.sha256.prefix(12))")
        }
        let registry = PackRegistry.load(child)
        lineage = registry.lineage(childPack.sha256).map { "\($0.name) \($0.sha256.prefix(12))" }

        let sets = UmbrellaBench.sets(world: world, base: child, pack: childPack, strands: childStrands, tokenizer: tokenizer, sizes: sizes)
        var everySizes = sizes
        everySizes.factsPerNode = .max
        let every = UmbrellaBench.factsOnly(UmbrellaBench.sets(world: world, base: child, pack: childPack, strands: childStrands, tokenizer: tokenizer, sizes: everySizes))
        var promptsOnly = sets
        promptsOnly.facts = []
        progress?("prompts: \(every.facts.count) facts, \(sets.generic.count) generic, \(sets.commons.count) commons, \(sets.unknown.count) unknown")

        var arms: [UmbrellaArmResult] = []
        var facts: [UmbrellaArmResult] = []
        var byNode: [String: [String: [Int]]] = [:]
        var answers: [String: [String: String]] = [:]
        for (name, pack, strands) in [("parent", parentPack, parentStrands), ("child", childPack, childStrands)] {
            let (umbrella, links, _) = try RoutingBench.umbrella(pack: pack, strands: strands, tokenizer: tokenizer)
            let generator = try umbrella.generator(links: links)
            let threadOf = Dictionary(uniqueKeysWithValues: strands.map { ($0.name, $0.threadID) })
            progress?("\(name): every fact")
            var observed: [CitedGeneration] = []
            facts.append(try UmbrellaBench.arm(name, generator: generator, links: links, gate: BraidRequest.defaultGate, sets: every, tokenizer: tokenizer,
                                               threadOf: threadOf, observe: { observed.append($0) }))
            // One generation per fact, in order: each fact's node, and whether it was answered exactly.
            let asked = every.facts.filter { $0.expected != nil }
            if observed.count == asked.count {
                var tally: [String: [Int]] = [:]
                for (example, generation) in zip(asked, observed) {
                    let node = example.node ?? "nobody"
                    var counts = tally[node] ?? [0, 0]
                    if generation.text.hasPrefix(example.expected!) { counts[0] += 1 }
                    counts[1] += 1
                    tally[node] = counts
                    answers[name, default: [:]]["\(node) · \(example.label) · \(example.expected!.trimmingCharacters(in: .whitespaces))"] =
                        String(generation.text.split(separator: "\n").first ?? "")
                }
                byNode[name] = tally
            }
            progress?("\(name): generic, commons and unknown prompts")
            arms.append(try UmbrellaBench.arm(name, generator: generator, links: links, gate: BraidRequest.defaultGate, sets: promptsOnly, tokenizer: tokenizer, threadOf: threadOf))
        }

        var heldOut = childPack.info?.recipe?.heldOut ?? [:]
        if heldOut.isEmpty || childPack.info?.parent != parentPack.sha256 {
            progress?("held-out loss of both commons on the pack's snippets")
            let batches = CommonsTrainer.snippetBatches(childPack.heldOut.map(\.tokens))
            heldOut = ["pack": [CommonsTrainer.loss(model: try parentPack.baseModel(), batches: batches), CommonsTrainer.loss(model: try childPack.baseModel(), batches: batches)]]
        }
        return CommonsReport(
            createdAt: .wholeSecond(), child: childPack.sha256, parent: parentPack.sha256, lineage: lineage, heldOut: heldOut, arms: arms, facts: facts,
            factsByNode: byNode.isEmpty ? nil : byNode, answers: answers.isEmpty ? nil : answers, evaluation: evaluate(heldOut: heldOut, arms: arms, facts: facts))
    }

    // MARK: - The rules (pure)

    public static func evaluate(heldOut: [String: [Float]], arms: [UmbrellaArmResult], facts: [UmbrellaArmResult]) -> CommonsEvaluation {
        func pct(_ value: Float?) -> String { value.map { String(format: "%.0f%%", $0 * 100) } ?? "—" }
        func num(_ value: Float?) -> String { value.map { String(format: "%.3f", $0) } ?? "—" }
        guard let parent = arms.first(where: { $0.arm == "parent" }), let child = arms.first(where: { $0.arm == "child" }),
              let parentFacts = facts.first(where: { $0.arm == "parent" }), let childFacts = facts.first(where: { $0.arm == "child" })
        else { return CommonsEvaluation(rules: [], qualifies: false, summary: "both braids' measures are needed") }
        var rules: [TrajectoryRuleResult] = []
        let sets = heldOut.filter { $0.value.count == 2 }.sorted { $0.key < $1.key }
        rules.append(TrajectoryRuleResult(
            rule: "C1 general English", passed: !sets.isEmpty && sets.allSatisfy { $0.value[1] <= $0.value[0] + 1e-6 },
            detail: sets.map { String(format: "%@ %.4f → %.4f", $0.key, $0.value[0], $0.value[1]) }.joined(separator: "; ")))
        let exact = childFacts.factsExactRate >= parentFacts.factsExactRate - 1 / 30 - 1e-6
        let cited = (childFacts.citation ?? 0) >= (parentFacts.citation ?? 0) - 0.02 - 1e-6
        rules.append(TrajectoryRuleResult(
            rule: "C2 Threads on it", passed: exact && cited,
            detail: "exact \(childFacts.factsExact)/\(childFacts.facts) (\(pct(childFacts.factsExactRate))) against \(pct(parentFacts.factsExactRate)); "
                + "citation@1 \(pct(childFacts.citation)) against \(pct(parentFacts.citation))"))
        rules.append(TrajectoryRuleResult(
            rule: "C3 nobody's text", passed: (child.commonsLeads ?? 0) >= (parent.commonsLeads ?? 0) - 0.02 - 1e-6,
            detail: "the commons leads \(pct(child.commonsLeads)) of generic and commons prompts against \(pct(parent.commonsLeads)) "
                + "(U3's bar 90%; largest Thread gate \(num(child.commonsThreadGate)) against \(num(parent.commonsThreadGate)), bar 0.35)"))
        let share = (child.commonsThreadShare ?? 1) <= (parent.commonsThreadShare ?? 1) + 0.02 + 1e-6
        let lift = (childFacts.liftPositive ?? 0) >= (parentFacts.liftPositive ?? 0) - 0.02 - 1e-6
        rules.append(TrajectoryRuleResult(
            rule: "C4 inherited knowledge unattributed", passed: share && lift,
            detail: "Threads take \(num(child.commonsThreadShare)) of commons prompts against \(num(parent.commonsThreadShare)) (U4's bar 0.10); "
                + "lift positive on \(pct(childFacts.liftPositive)) of fact answer tokens against \(pct(parentFacts.liftPositive)) (bar 95%)"))
        let failed = rules.filter { !$0.passed }.map(\.rule)
        return CommonsEvaluation(
            rules: rules, qualifies: failed.isEmpty,
            summary: failed.isEmpty ? "the child commons is no worse than its parent on every rule" : "fails " + failed.joined(separator: ", "))
    }
}
