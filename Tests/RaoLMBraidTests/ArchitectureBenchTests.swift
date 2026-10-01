import Foundation
import Testing

@testable import RaoLMBraid
@testable import RaoLMCore

@Suite("Architecture bench rules")
struct ArchitectureBenchTests {
    static func braid(_ name: String, exact: Int = 26, citation: Float = 0.9, onBreaks: Int = 0) -> ArchitectureBraidResult {
        ArchitectureBraidResult(
            braid: name, facts: UmbrellaBenchTests.arm(name, exact: exact, citation: citation), continuations: 30,
            breaksWritten: name == "arm" ? 27 : 2, ranTogether: name == "arm" ? 1 : 25, citations: 400, citationsOnBreaks: onBreaks)
    }

    static func node(_ braid: String, _ name: String, steps: Int? = 400, loss: Float = 1.2, greedy: Int = 81, breaks: Int = 18) -> ArchitectureNodeResult {
        ArchitectureNodeResult(
            braid: braid, node: name, version: 1, memorised: 0.97, stepsTo97: steps, curve: [ArchitectureCurvePoint(steps: steps ?? 1_000, memorised: 0.97)],
            heldOutLoss: loss, commonsLoss: 5.5, greedyFacts: 100, greedyCorrect: greedy, boundaries: 20, breaksPredicted: breaks)
    }

    static let names = ["ambient", "craft", "veil"]
    static let reference = names.map { node("reference", $0, steps: 400, loss: 1.25, greedy: 82, breaks: 1) }
    static let trajectory = [TrajectoryRuleResult(rule: "T1 arc, not phrase", passed: true, detail: "")]

    static func evaluate(arm: ArchitectureBraidResult = braid("arm"), nodes: [ArchitectureNodeResult], trajectory: [TrajectoryRuleResult] = trajectory)
        -> ArchitectureEvaluation {
        ArchitectureBench.evaluate(braids: [braid("reference"), arm], nodes: reference + nodes, trajectory: trajectory)
    }

    @Test("every rule holding, the arm qualifies; what is reported carries no threshold")
    func qualifies() {
        let evaluation = Self.evaluate(nodes: Self.names.map { Self.node("arm", $0, steps: 500, greedy: 81) })
        #expect(evaluation.qualifies, "\(evaluation.rules.filter { !$0.passed })")
        #expect(evaluation.rules.map(\.rule) == ["A1 learning", "A2 answers", "A3 general text", "A4 the break", "A5 provenance"])
        #expect(evaluation.reported["reference: passages run together"] == 25 / 30 && evaluation.reported["arm: break written"] == 27 / 30)
    }

    @Test("each rule fails on its own measure")
    func failures() {
        func failed(_ evaluation: ArchitectureEvaluation) -> [String] { evaluation.rules.filter { !$0.passed }.map(\.rule) }
        let good = Self.names.map { Self.node("arm", $0) }
        var nodes = good
        // A1: more than 1.25× the reference's steps, or never.
        nodes[0] = Self.node("arm", "ambient", steps: 501)
        #expect(failed(Self.evaluate(nodes: nodes)) == ["A1 learning"])
        nodes[0] = Self.node("arm", "ambient", steps: nil)
        #expect(failed(Self.evaluate(nodes: nodes)) == ["A1 learning"])
        // A2: more than 1/30 fewer exact (as a rate), citation@1 0.03 lower, or one node 3 points lower alone.
        #expect(failed(Self.evaluate(arm: Self.braid("arm", exact: 24), nodes: good)) == ["A2 answers"])
        #expect(failed(Self.evaluate(arm: Self.braid("arm", exact: 25), nodes: good)).isEmpty)
        // On a larger set the allowance stays a rate: 1/30 of the prompts.
        var many = Self.braid("arm")
        many.facts.facts = 680
        many.facts.factsExact = 570
        #expect(failed(Self.evaluate(arm: many, nodes: good)).isEmpty)
        many.facts.factsExact = 560
        #expect(failed(Self.evaluate(arm: many, nodes: good)) == ["A2 answers"])
        #expect(failed(Self.evaluate(arm: Self.braid("arm", citation: 0.87), nodes: good)) == ["A2 answers"])
        nodes = good
        nodes[1] = Self.node("arm", "craft", greedy: 79)
        #expect(failed(Self.evaluate(nodes: nodes)) == ["A2 answers"])
        // A3: one node's held-out loss above its reference's.
        nodes = good
        nodes[2] = Self.node("arm", "veil", loss: 1.26)
        #expect(failed(Self.evaluate(nodes: nodes)) == ["A3 general text"])
        // A4: the break likeliest at fewer than 80% of boundaries.
        nodes = good
        nodes[2] = Self.node("arm", "veil", breaks: 15)
        #expect(failed(Self.evaluate(nodes: nodes)) == ["A4 the break"])
        // A5: a citation on a break, or a trajectory rule failing.
        #expect(failed(Self.evaluate(arm: Self.braid("arm", onBreaks: 1), nodes: good)) == ["A5 provenance"])
        let broken = [TrajectoryRuleResult(rule: "M2 manner", passed: false, detail: "")]
        #expect(failed(Self.evaluate(nodes: good, trajectory: broken)) == ["A5 provenance"])
        #expect(!Self.evaluate(nodes: good, trajectory: broken).qualifies)
    }

    @Test("a block arm is read on B1 to B4: three quarters of the reference's steps, the break kept as a guard")
    func blockRules() {
        func evaluate(_ nodes: [ArchitectureNodeResult]) -> ArchitectureEvaluation {
            ArchitectureBench.evaluate(braids: [Self.braid("reference"), Self.braid("arm")], nodes: Self.reference + nodes,
                                       trajectory: Self.trajectory, arm: "canon")
        }
        let good = Self.names.map { Self.node("arm", $0, steps: 300) }
        let evaluation = evaluate(good)
        #expect(evaluation.qualifies, "\(evaluation.rules.filter { !$0.passed })")
        #expect(evaluation.rules.map(\.rule) == ["B1 learning", "B2 answers", "B3 general text", "B4 provenance"])
        #expect(evaluation.summary.contains("B1 to B4"))
        var nodes = good
        nodes[0] = Self.node("arm", "ambient", steps: 301)
        #expect(evaluate(nodes).rules.filter { !$0.passed }.map(\.rule) == ["B1 learning"])
        nodes = good
        nodes[1] = Self.node("arm", "craft", steps: 300, breaks: 15)
        #expect(evaluate(nodes).rules.filter { !$0.passed }.map(\.rule) == ["B4 provenance"])
    }

    @Test("Muon is read on M1 to M4, and warmup-stable-decay on W1 to W4")
    func optimizerRules() {
        func node(_ braid: String, _ name: String, stepsTo97: Int = 300, steps: Int = 400, seconds: Double = 0.5, greedy: Int = 81,
                  loss: Float = 1.2) -> ArchitectureNodeResult {
            var value = Self.node(braid, name, steps: stepsTo97, loss: loss, greedy: greedy)
            value.steps = steps
            value.stepSeconds = seconds
            return value
        }
        let reference = Self.names.map { node("reference", $0, stepsTo97: 400, steps: 400, seconds: 0.5, greedy: 82, loss: 1.2) }
        func evaluate(_ arm: String, _ nodes: [ArchitectureNodeResult], exact: Int = 26, citation: Float = 0.9) -> ArchitectureEvaluation {
            ArchitectureBench.evaluate(
                braids: [Self.braid("reference"), Self.braid("arm", exact: exact, citation: citation)], nodes: reference + nodes,
                trajectory: Self.trajectory, arm: arm)
        }
        func failed(_ evaluation: ArchitectureEvaluation) -> [String] { evaluation.rules.filter { !$0.passed }.map(\.rule) }

        // Muon: three quarters of the steps, in no more time; held-out loss within 0.02.
        let fast = Self.names.map { node("arm", $0, stepsTo97: 300, seconds: 0.6, loss: 1.215) }
        #expect(failed(evaluate("muon", fast)).isEmpty)
        #expect(evaluate("muon", fast).rules.map(\.rule) == ["M1 learning", "M2 answers", "M3 general text", "M4 provenance"])
        #expect(failed(evaluate("muon", Self.names.map { node("arm", $0, stepsTo97: 300, seconds: 0.7) })) == ["M1 learning"])
        #expect(failed(evaluate("muon", Self.names.map { node("arm", $0, stepsTo97: 300, seconds: 0.6, loss: 1.23) })) == ["M3 general text"])

        // Warmup-stable-decay: steps at most 1.5×; exact at least 2 points over, citation and greedy no lower.
        let annealed = Self.names.map { node("arm", $0, steps: 600, greedy: 82) }
        #expect(failed(evaluate("wsd", annealed, exact: 27, citation: 0.9)).isEmpty)
        #expect(evaluate("wsd", annealed, exact: 27).rules.map(\.rule) == ["W1 learning", "W2 answers", "W3 general text", "W4 provenance"])
        #expect(failed(evaluate("wsd", annealed, exact: 26)) == ["W2 answers"])
        #expect(failed(evaluate("wsd", annealed, exact: 27, citation: 0.89)) == ["W2 answers"])
        #expect(failed(evaluate("wsd", Self.names.map { node("arm", $0, steps: 601, greedy: 82) }, exact: 27)) == ["W1 learning"])
    }

    @Test("the block arms switch on their addition over the passage-break stream")
    func blockArms() throws {
        var settings = HypervisorSettings()
        settings.preset = "base"
        settings.arm = "canon"
        #expect(try settings.modelConfig().canon && !settings.modelConfig().attentionGate && settings.passageBreak)
        settings.arm = "gated-attention"
        #expect(try settings.modelConfig().attentionGate && !settings.modelConfig().canon && settings.passageBreak)
        settings.arm = "passage-break"
        #expect(try !settings.modelConfig().canon && !settings.modelConfig().attentionGate && settings.passageBreak)
        for arm in ["muon", "wsd"] {
            settings.arm = arm
            #expect(try settings.modelConfig() == .base && settings.passageBreak)
        }
        settings.arm = nil
        #expect(try settings.modelConfig() == .base && !settings.passageBreak)
    }
}

@Suite("Thought bench rules")
struct ThoughtBenchTests {
    static func node(_ name: String, lens: Int = 60, identity: Int = 40, lensAhead: Int = 30, outputAhead: Int = 20, identityAhead: Int = 10)
        -> ThoughtNodeResult {
        ThoughtNodeResult(node: name, positions: 100, lensAgrees: lens, identityAgrees: identity, ahead: 100, lensAhead: lensAhead,
                          outputAhead: outputAhead, identityAhead: identityAhead)
    }

    static func told(source: Bool, control: Bool, output: Bool) -> ThoughtToldResult {
        ThoughtToldResult(text: "t", teller: "ambient", source: "craft", control: "veil", token: "4", sourceLens: source, sourceIdentity: false,
                          sourceOutput: output, tellerLens: true, controlLens: control, controlIdentity: false, workspace: [])
    }

    /// Ten told answers: the source's lens holds `source` of them, the control's `control`, the source's output `output`.
    static func answers(source: Int, control: Int, output: Int) -> [ThoughtToldResult] {
        (0 ..< 10).map { told(source: $0 < source, control: $0 < control, output: $0 < output) }
    }

    @Test("L1 to L3 each fail on their own margin")
    func rules() {
        let nodes = ["ambient", "craft", "veil"].map { Self.node($0) }
        let good = ThoughtBench.evaluate(nodes: nodes, told: Self.answers(source: 7, control: 4, output: 5))
        #expect(good.qualifies, "\(good.rules.filter { !$0.passed })")
        func failed(_ nodes: [ThoughtNodeResult], _ told: [ThoughtToldResult]) -> [String] {
            ThoughtBench.evaluate(nodes: nodes, told: told).rules.filter { !$0.passed }.map(\.rule)
        }
        let told = Self.answers(source: 7, control: 4, output: 5)
        // L1: ten points over the identity lens, no fewer.
        #expect(failed([Self.node("ambient", lens: 49, identity: 40)] + nodes.dropFirst(), told) == ["L1 faithful"])
        // L2: more than the output's top 25, and the identity lens's.
        #expect(failed([Self.node("ambient", lensAhead: 20, outputAhead: 20)] + nodes.dropFirst(), told) == ["L2 ahead"])
        // L3: 0.20 over the control's lens, 0.10 over the source's own output.
        #expect(failed(nodes, Self.answers(source: 7, control: 6, output: 5)) == ["L3 who knows"])
        #expect(failed(nodes, Self.answers(source: 7, control: 4, output: 7)) == ["L3 who knows"])
        #expect(failed(nodes, []) == ["L3 who knows"])
    }
}


@Suite("Question bench rules")
struct QuestionBenchTests {
    static func arm(_ name: String, rewrite: Int = 100, exact: Int = 90, cited: Int = 85, leads: Int = 17, threadCredit: Float = 0.2,
                    decided: Int = 20, followed: Int = 19, other: Float = 0.02, ties: Int = 0) -> QuestionArmResult {
        let rewrites = name == "commons" || name == "rules"
        return QuestionArmResult(
            arm: name, questions: 100, rewriteExact: rewrite, rewriteF1: rewrites ? Float(rewrite) / 100 : nil, exact: exact, citedAnswer: cited,
            citationAt1: 0.9, unknown: 20, unknownCommonsLeads: leads, unknownThreadCredit: threadCredit, shared: 24, sharedDecided: decided,
            sharedFollowed: followed, sharedDuplicateCredit: other, sharedTies: ties, adapterMs: rewrites ? 40 : nil, seconds: 1)
    }

    static func arms(commons: QuestionArmResult = arm("commons", rewrite: 70), rules: QuestionArmResult = arm("rules"),
                     stem: QuestionArmResult = arm("stem")) -> [QuestionArmResult] {
        [commons, rules, stem, arm("slice", exact: 96), arm("paraphrase", exact: 20)]
    }

    @Test("every rule holding, the commons wins; a commons under half fidelity or far below the stem hands the win to the rules")
    func winner() {
        let good = QuestionBench.evaluate(arms: Self.arms())
        #expect(good.qualifies && good.winner == "commons", "\(good.rules.filter { !$0.passed })")
        #expect(good.rules.map(\.rule) == ["Q1 rewrite fidelity", "Q2 exact", "Q2′ context", "Q3 cited", "Q4 unknown", "Q5a retold"])
        let weak = QuestionBench.evaluate(arms: Self.arms(commons: Self.arm("commons", rewrite: 40)))
        #expect(weak.winner == "rules" && !weak.qualifies && weak.rules[0].passed == false)
        let behind = QuestionBench.evaluate(arms: Self.arms(commons: Self.arm("commons", rewrite: 70, exact: 80)))
        #expect(behind.winner == "rules" && behind.qualifies)
    }

    @Test("Q2 to Q5 each fail on their own measure, read on the winner")
    func failures() {
        func failed(_ arms: [QuestionArmResult]) -> [String] { QuestionBench.evaluate(arms: arms).rules.filter { !$0.passed }.map(\.rule) }
        #expect(failed(Self.arms(commons: Self.arm("commons", rewrite: 40), rules: Self.arm("rules", exact: 80))) == ["Q1 rewrite fidelity", "Q2 exact", "Q2′ context"])
        #expect(failed(Self.arms(commons: Self.arm("commons", rewrite: 70, cited: 80))) == ["Q3 cited"])
        #expect(failed(Self.arms(commons: Self.arm("commons", rewrite: 70, exact: 85), stem: Self.arm("stem", exact: 85))) == ["Q2′ context"])
        #expect(failed(Self.arms(commons: Self.arm("commons", rewrite: 70, leads: 15))) == ["Q4 unknown"])
        #expect(failed(Self.arms(commons: Self.arm("commons", rewrite: 70, threadCredit: 0.3))) == ["Q4 unknown"])
        #expect(failed(Self.arms(commons: Self.arm("commons", rewrite: 70, followed: 17))) == ["Q5a retold"])
        #expect(failed(Self.arms(commons: Self.arm("commons", rewrite: 70, other: 0.11))) == ["Q5a retold"])
        #expect(failed(Self.arms(commons: Self.arm("commons", rewrite: 70, decided: 10))) == ["Q5a retold"])
        #expect(QuestionBench.tokenF1("The mayor of Tillyburn is", "the mayor of Tillyburn is.") == 1)
        #expect(abs(QuestionBench.tokenF1("Tillyburn was founded in", "The town of Tillyburn was founded in") - 8.0 / 11) < 1e-6)
    }
}
