import Foundation
import Testing

@testable import RaoLMBraid
@testable import RaoLMCore

@Suite("Umbrella bench rule")
struct UmbrellaBenchTests {
    static func arm(_ name: String, exact: Int = 28, owned: Int = 27, citation: Float = 0.9, pairsMoved: Int = 28, leads: Float? = 0.95,
                    threadGate: Float? = 0.1, threadShare: Float? = 0.05, ownerShare: Float = 0.95, lift: Float? = 0.99, nll: Float = 3,
                    told: Float? = 0.2, voice: Float? = 0.3) -> UmbrellaArmResult {
        UmbrellaArmResult(
            arm: name, facts: 30, factsExact: exact, factsOwned: owned, citation: citation, ownerShare: ownerShare, liftPositive: lift, pairs: 30,
            pairsMoved: pairsMoved, pairsExact: 25, unknownAllAsked: 1, unknownLargest: 0.5, genericAllAsked: 1, genericLargest: 0.4,
            commonsLeads: leads, commonsThreadGate: threadGate, commonsThreadShare: threadShare, heldOutNLL: nll, heldOutTokens: 1000,
            toldSourceShare: told, toldTokens: 50, voiceLargest: voice, seconds: 1)
    }

    static func nodes(steps: Int? = 600, baseLoss: Float = 3, referenceLoss: Float = 6) -> [UmbrellaNodeResult] {
        ["ambient", "craft", "veil"].flatMap { name in [
            UmbrellaNodeResult(braid: "reference", node: name, version: 1, documents: 80, memorised: 0.97, stepsTo97: 800, heldOutLoss: referenceLoss,
                               commonsLoss: nil, calibration: nil),
            UmbrellaNodeResult(braid: "base", node: name, version: 1, documents: 80, memorised: 0.98, stepsTo97: steps, heldOutLoss: baseLoss,
                               commonsLoss: 3.5, calibration: nil),
        ] }
    }

    static let trajectory = [TrajectoryRuleResult(rule: "T1 arc, not phrase", passed: true, detail: "")]

    @Test("every rule holding: the last arm wins; a calibration that moves held-out loss by less than 0.02 gives way")
    func allPass() {
        let arms = [Self.arm("v1", leads: nil, threadGate: nil, threadShare: nil, lift: nil), Self.arm("pack", leads: nil, threadGate: nil, threadShare: nil, lift: nil),
                    Self.arm("commons"), Self.arm("calibrated", nll: 2.9), Self.arm("anchors", nll: 2.9, told: 0.35)]
        let ceiling = Self.arm("pack (twice the documents)")
        let evaluation = UmbrellaBench.evaluate(arms: arms, ceiling: ceiling, nodes: Self.nodes(), trajectory: Self.trajectory)
        #expect(evaluation.qualifies == ["pack", "commons", "calibrated", "anchors"], "\(evaluation.rules.filter { !$0.passed })")
        #expect(evaluation.winner == "anchors")

        // Thought agreement adds too little: the calibrated arm wins.
        var weak = arms
        weak[4] = Self.arm("anchors", nll: 2.9, told: 0.25)
        #expect(UmbrellaBench.evaluate(arms: weak, ceiling: ceiling, nodes: Self.nodes(), trajectory: Self.trajectory).winner == "calibrated")
        // The calibration barely moves held-out loss: the commons arm wins.
        var flat = weak
        flat[3] = Self.arm("calibrated", nll: 2.99)
        #expect(UmbrellaBench.evaluate(arms: flat, ceiling: ceiling, nodes: Self.nodes(), trajectory: Self.trajectory).winner == "commons")
    }

    @Test("a base node that needs more than 1,000 steps, or no ceiling braid, fails U0 and nothing qualifies")
    func baseFails() {
        let arms = [Self.arm("v1"), Self.arm("pack"), Self.arm("commons")]
        let slow = UmbrellaBench.evaluate(arms: arms, ceiling: Self.arm("pack (twice the documents)"), nodes: Self.nodes(steps: 1_400), trajectory: Self.trajectory)
        #expect(slow.qualifies.isEmpty && slow.winner == nil)
        #expect(slow.summary.contains("U0"))
        let noCeiling = UmbrellaBench.evaluate(arms: arms, ceiling: nil, nodes: Self.nodes(), trajectory: Self.trajectory)
        #expect(noCeiling.winner == nil)
    }

    @Test("Threads that take commons text, or an owner whose lift is not positive, fail U4 for the commons arm")
    func inherited() {
        let arms = [Self.arm("v1"), Self.arm("pack"), Self.arm("commons", threadShare: 0.3)]
        let evaluation = UmbrellaBench.evaluate(arms: arms, ceiling: Self.arm("pack (twice the documents)"), nodes: Self.nodes(), trajectory: Self.trajectory)
        #expect(evaluation.qualifies == ["pack"] && evaluation.winner == "pack")
        let noLift = UmbrellaBench.evaluate(arms: [Self.arm("v1"), Self.arm("pack"), Self.arm("commons", lift: 0.8)],
                                            ceiling: Self.arm("pack (twice the documents)"), nodes: Self.nodes(), trajectory: Self.trajectory)
        #expect(noLift.winner == "pack")
    }
}
