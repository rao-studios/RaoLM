//
//  ThoughtBench.swift
//  RaoLMBraid
//
//  WHAT: Whether the Jacobian lens reads what a node plans (Docs/ARCHITECTURE.md, "Phase 2, the
//        Jacobian lens"). Every Thread reads bench-trajectory's texts teacher-forced, in process,
//        and at every position its cut state is read through the lens and through the identity
//        lens (the logit lens at the cut), beside the node's own next-token output.
//  OUT:  One report: per node, how often each reading agrees with the output's top token (L1) and
//        how many of the content tokens two to eight ahead each holds (L2); per told answer, which
//        Threads held its token ahead (L3); and `evaluate`'s verdict.
//  PIN:  The rules were fixed before any numbers. J is taken once through the pack's trunk and saved
//        beside the pack. Texts are read as prompts are, from position 0 with no eos. Nothing is
//        written to a node.
//

import Foundation
import MLX
import RaoLMCore
import RaoLMModel
import RaoLMProvenance
import RaoLMTraining

public struct ThoughtNodeResult: Codable, Sendable, Equatable {
    public var node: String
    /// L1, on the node's own fed documents: positions read, and those where the lens's (and the
    /// identity lens's) top token is the output's.
    public var positions: Int
    public var lensAgrees: Int
    public var identityAgrees: Int
    /// L2: content tokens two to eight ahead of a position, and those each reading's top 25 holds.
    public var ahead: Int
    public var lensAhead: Int
    public var outputAhead: Int
    public var identityAhead: Int

    public var lensAgreement: Float { positions > 0 ? Float(lensAgrees) / Float(positions) : 0 }
    public var identityAgreement: Float { positions > 0 ? Float(identityAgrees) / Float(positions) : 0 }
    public var lensAheadRate: Float { ahead > 0 ? Float(lensAhead) / Float(ahead) : 0 }
    public var outputAheadRate: Float { ahead > 0 ? Float(outputAhead) / Float(ahead) : 0 }
    public var identityAheadRate: Float { ahead > 0 ? Float(identityAhead) / Float(ahead) : 0 }
}

/// One told answer: whether each Thread's readings held its first content token in their top 25
/// at some position from 8 to 2 before it.
public struct ThoughtToldResult: Codable, Sendable, Equatable {
    public var text: String
    public var teller: String
    public var source: String
    /// The Thread that is neither teller nor source (nil with two Threads).
    public var control: String?
    public var token: String
    public var sourceLens: Bool
    public var sourceIdentity: Bool
    public var sourceOutput: Bool
    public var tellerLens: Bool
    public var controlLens: Bool?
    public var controlIdentity: Bool?
    /// The source's lens top 8, two positions before the token.
    public var workspace: [String]
}

public struct ThoughtEvaluation: Codable, Sendable, Equatable {
    public var rules: [TrajectoryRuleResult]
    public var qualifies: Bool
    public var reported: [String: Float]
    public var summary: String
}

public struct ThoughtReport: Codable, Sendable {
    public var createdAt: Date
    public var pack: String
    public var lens: JacobianLensInfo
    public var nodes: [ThoughtNodeResult]
    public var told: [ThoughtToldResult]
    public var evaluation: ThoughtEvaluation
}

public enum ThoughtBench {
    public struct Sizes: Sendable {
        /// The readings' top k.
        public var k = 25
        /// L2 and L3 look this far ahead of a position, from `nearest` on.
        public var nearest = 2
        public var farthest = 8
        /// Anchor snippets J is taken over, and channels per reverse pass.
        public var lensPrompts = 512
        public var lensBatch = 64
        public var trajectory = TrajectoryBench.Sizes()

        public init() {}
    }

    /// One Thread's readings of one text, per position: each reading's top token and top k.
    struct Readings {
        var output: [Int32]
        var lens: [Int32]
        var identity: [Int32]
        var outputTop: [Set<Int32>]
        var lensTop: [Set<Int32>]
        var identityTop: [Set<Int32>]
        var lensWorkspace: [[Int32]]
    }

    // MARK: - Running

    public static func run(
        layout: BraidLayout, tokenizer: RaoTokenizer, sizes: Sizes = Sizes(), owner: String = "raolm-braid", progress: ((String) -> Void)? = nil
    ) throws -> ThoughtReport {
        guard let record = MockWorld.Record.load(layout) else {
            throw BraidSessionError.io("the braid needs a world.json: run it with raolm braid demo --dataset …")
        }
        let world = try RoutingBench.world(layout: layout, seed: record.seed)
        let (pack, strands) = try RoutingBench.packStrands(layout: layout, tokenizer: tokenizer, owner: owner, names: world.names)
        guard !strands.isEmpty else { throw BraidSessionError.noLiveNodes }
        guard pack.hasBase, pack.hasTrunk else { throw BraidSessionError.io("the lens reads through a trunk: start the braid with --preset base") }

        let directory = layout.pack(sha256: pack.sha256)
        let lens: JacobianLens
        if let saved = try JacobianLens.load(from: directory, packSHA256: pack.sha256) {
            lens = saved
        } else {
            let prompts = pack.anchors.prefix(sizes.lensPrompts).map(\.tokens)
            progress?("the lens: J through the trunk over \(prompts.count) anchor snippets")
            lens = JacobianLens.compute(model: try pack.baseModel(), packSHA256: pack.sha256, prompts: prompts, batch: sizes.lensBatch) { done in
                if done % 64 == 0 { progress?("  \(done) of \(prompts.count)") }
            }
            try lens.save(to: directory)
        }
        progress?(String(format: "lens: split-half agreement %.3f", lens.info.splitHalfAgreement))

        let texts = TrajectoryTexts.build(
            world: world, layout: layout, names: strands.map(\.name), tokenizer: tokenizer, sizes: sizes.trajectory,
            paragraphBreak: strands.first?.index.info.paragraphBreak ?? [])
        let byName = Dictionary(uniqueKeysWithValues: strands.map { ($0.name, $0) })

        // L1 and L2: each node on its own fed documents.
        var nodes: [ThoughtNodeResult] = []
        for strand in strands {
            progress?("\(strand.name) on its own documents")
            var result = ThoughtNodeResult(node: strand.name, positions: 0, lensAgrees: 0, identityAgrees: 0, ahead: 0, lensAhead: 0, outputAhead: 0,
                                           identityAhead: 0)
            for text in texts where text.kind == .held && text.owner == strand.name {
                let readings = read(text.tokens, strand: strand, lens: lens, k: sizes.k)
                let content = TokenRoles.roles(text.tokens.map { tokenizer.tokenText($0) }).map { $0 == .content }
                for t in text.tokens.indices {
                    result.positions += 1
                    if readings.lens[t] == readings.output[t] { result.lensAgrees += 1 }
                    if readings.identity[t] == readings.output[t] { result.identityAgrees += 1 }
                    for d in sizes.nearest ... sizes.farthest where t + d < text.tokens.count && content[t + d] {
                        let token = Int32(text.tokens[t + d])
                        result.ahead += 1
                        if readings.lensTop[t].contains(token) { result.lensAhead += 1 }
                        if readings.outputTop[t].contains(token) { result.outputAhead += 1 }
                        if readings.identityTop[t].contains(token) { result.identityAhead += 1 }
                    }
                }
            }
            nodes.append(result)
        }

        // L3: told answers, read by the source, the teller and the control.
        var told: [ThoughtToldResult] = []
        let tellings = texts.filter { $0.kind == .told && !$0.answers.isEmpty }
        progress?("told texts: \(tellings.count)")
        for text in tellings {
            guard let teller = text.owner, let source = text.near, let sourceStrand = byName[source], let tellerStrand = byName[teller] else { continue }
            let control = strands.first { $0.name != teller && $0.name != source }
            let content = TokenRoles.roles(text.tokens.map { tokenizer.tokenText($0) }).map { $0 == .content }
            let sourceReadings = read(text.tokens, strand: sourceStrand, lens: lens, k: sizes.k)
            let tellerReadings = read(text.tokens, strand: tellerStrand, lens: lens, k: sizes.k)
            let controlReadings = control.map { read(text.tokens, strand: $0, lens: lens, k: sizes.k) }
            for answer in text.answers {
                guard let a = (answer.start ..< (answer.start + answer.count)).first(where: { $0 < content.count && content[$0] }),
                      a - sizes.nearest >= 0 else { continue }
                let window = max(0, a - sizes.farthest) ... (a - sizes.nearest)
                let token = Int32(text.tokens[a])
                func held(_ tops: [Set<Int32>]) -> Bool { window.contains { tops[$0].contains(token) } }
                told.append(ThoughtToldResult(
                    text: text.label, teller: teller, source: source, control: control?.name, token: tokenizer.tokenText(text.tokens[a]),
                    sourceLens: held(sourceReadings.lensTop), sourceIdentity: held(sourceReadings.identityTop),
                    sourceOutput: held(sourceReadings.outputTop), tellerLens: held(tellerReadings.lensTop),
                    controlLens: controlReadings.map { held($0.lensTop) }, controlIdentity: controlReadings.map { held($0.identityTop) },
                    workspace: sourceReadings.lensWorkspace[a - sizes.nearest].map { tokenizer.tokenText(Int($0)) }))
            }
        }
        return ThoughtReport(
            createdAt: .wholeSecond(), pack: pack.sha256, lens: lens.info, nodes: nodes, told: told,
            evaluation: evaluate(nodes: nodes, told: told, lens: lens.info))
    }

    /// A Thread reading `tokens` from position 0: its output's, the lens's and the identity lens's
    /// top token and top k at every position.
    static func read(_ tokens: [Int], strand: ThreadStrand, lens: JacobianLens, k: Int) -> Readings {
        let model = strand.model
        let body = model.body(MLXArray(tokens.map { Int32($0) }, [1, tokens.count]), captureTap: false, captureCut: true)
        let cut = body.cut ?? body.last
        let outputs = [model.head(body.last).logits, lens.logits(cut, head: model), model.head(cut).logits].map { $0[0] }
        var top: [[Int32]] = []
        var tops: [[Set<Int32>]] = []
        var ranked: [[[Int32]]] = []
        for logits in outputs {
            let best = argMax(logits, axis: -1)
            let partitioned = argPartition(-logits, kth: k - 1, axis: -1)[0..., 0 ..< k]
            let firstEight = argSort(-logits, axis: -1)[0..., 0 ..< 8]
            eval(best, partitioned, firstEight)
            top.append(best.asArray(Int32.self))
            let flat = partitioned.asArray(Int32.self)
            tops.append(tokens.indices.map { Set(flat[($0 * k) ..< (($0 + 1) * k)]) })
            let eight = firstEight.asArray(Int32.self)
            ranked.append(tokens.indices.map { Array(eight[($0 * 8) ..< (($0 + 1) * 8)]) })
        }
        return Readings(output: top[0], lens: top[1], identity: top[2], outputTop: tops[0], lensTop: tops[1], identityTop: tops[2],
                        lensWorkspace: ranked[1])
    }

    // MARK: - The rules (pure)

    /// The rules, fixed before any numbers (Docs/ARCHITECTURE.md):
    /// - L1 faithful: on every node, the lens's top token agrees with the output's at least 10
    ///   points more often than the identity lens's does.
    /// - L2 ahead: on every node, the lens's top 25 holds more of the content tokens two to eight
    ///   ahead than the output's top 25 and the identity lens's do.
    /// - L3 who knows: over told answers, the source's lens holds the answer's first content token
    ///   ahead for at least 0.20 more of them than the control's lens, and 0.10 more than the source's output.
    public static func evaluate(nodes: [ThoughtNodeResult], told: [ThoughtToldResult], lens: JacobianLensInfo? = nil) -> ThoughtEvaluation {
        func pct(_ value: Float) -> String { String(format: "%.0f%%", value * 100) }
        func rate(_ values: [Bool]) -> Float { values.isEmpty ? 0 : Float(values.filter { $0 }.count) / Float(values.count) }
        var rules: [TrajectoryRuleResult] = []
        rules.append(TrajectoryRuleResult(
            rule: "L1 faithful", passed: !nodes.isEmpty && nodes.allSatisfy { $0.positions > 0 && $0.lensAgreement >= $0.identityAgreement + 0.10 },
            detail: "the top token agrees with the output's: " + nodes.map { "\($0.node) lens \(pct($0.lensAgreement)), identity \(pct($0.identityAgreement))" }
                .joined(separator: "; ")))
        rules.append(TrajectoryRuleResult(
            rule: "L2 ahead",
            passed: !nodes.isEmpty && nodes.allSatisfy { $0.ahead > 0 && $0.lensAheadRate > $0.outputAheadRate && $0.lensAheadRate > $0.identityAheadRate },
            detail: "content tokens 2 to 8 ahead held in the top 25: " + nodes.map {
                "\($0.node) lens \(pct($0.lensAheadRate)), output \(pct($0.outputAheadRate)), identity \(pct($0.identityAheadRate))"
            }.joined(separator: "; ")))
        let controlled = told.filter { $0.controlLens != nil }
        let source = rate(told.map(\.sourceLens))
        let control = rate(controlled.compactMap(\.controlLens))
        let output = rate(told.map(\.sourceOutput))
        rules.append(TrajectoryRuleResult(
            rule: "L3 who knows", passed: !controlled.isEmpty && source >= control + 0.20 && source >= output + 0.10,
            detail: "of \(told.count) told answers, held ahead by the source's lens \(pct(source)), the control's lens \(pct(control)), "
                + "the source's output \(pct(output))"))

        var reported: [String: Float] = [
            "L3: source's identity lens": rate(told.map(\.sourceIdentity)),
            "L3: control's identity lens": rate(controlled.compactMap(\.controlIdentity)),
            "L3: teller's lens": rate(told.map(\.tellerLens)),
        ]
        if let lens { reported["lens: split-half agreement"] = lens.splitHalfAgreement }
        let failed = rules.filter { !$0.passed }.map(\.rule)
        return ThoughtEvaluation(
            rules: rules, qualifies: failed.isEmpty, reported: reported,
            summary: failed.isEmpty ? "the lens qualifies as a readout: L1 to L3 hold" : "the lens does not qualify: it fails " + failed.joined(separator: ", "))
    }
}
