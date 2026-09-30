import Foundation
import Testing

@testable import RaoLMBraid
@testable import RaoLMCore
@testable import RaoLMProvenance

@Suite("Gate bench rule")
struct GateBenchRuleTests {
    private func row(
        _ arm: String, _ set: GateBenchSet, _ example: String, exact: Bool? = true, leads: Bool? = true, cited: Bool? = true,
        asked: Bool = true, gate: Float = 0.5, fidelity: Float = 0.9, tokens: Int = 10
    ) -> GateBenchRow {
        GateBenchRow(arm: arm, set: set, example: example, owner: exact == nil ? nil : "a", expected: exact == nil ? nil : "x", answer: "x",
                     exact: exact, ownerLeads: leads, citedToOwner: cited, allAsked: asked, largestGate: gate,
                     fidelitySum: fidelity * Float(tokens), fidelityTokens: tokens)
    }

    /// Every set for one arm, passing unless overridden.
    private func arm(
        _ name: String, factsExact: Int = 10, lostOwnership: Int = 0, pairsMoved: Int = 10, genericAsked: Int = 8, genericGate: Float = 0.5,
        unknownGate: Float = 0.5, fidelity: Float = 0.9
    ) -> [GateBenchRow] {
        var rows: [GateBenchRow] = []
        for i in 0..<10 {
            rows.append(row(name, .facts, "fact \(i)", exact: i < factsExact, leads: i >= lostOwnership, fidelity: fidelity))
            rows.append(row(name, .pairs, "pair \(i)", leads: i < pairsMoved, fidelity: fidelity))
            rows.append(row(name, .unknown, "unknown \(i)", exact: nil, leads: nil, cited: nil, gate: unknownGate))
        }
        for i in 0..<8 { rows.append(row(name, .generic, "generic \(i)", exact: nil, leads: nil, cited: nil, asked: i < genericAsked, gate: genericGate)) }
        return rows
    }

    private let arms = [
        GateArm(name: "posterior", gating: .posterior, candidate: false),
        GateArm(name: "first", gating: .braided, candidate: true),
        GateArm(name: "second", gating: .braided, candidate: true),
    ]

    @Test("the eligible arm with the best share fidelity wins; within 0.02 the one listed first")
    func winner() {
        let clear = arm("posterior") + arm("first", fidelity: 0.80) + arm("second", fidelity: 0.90)
        #expect(RoutingBench.decide(rows: clear, arms: arms).winner == "second")
        let close = arm("posterior") + arm("first", fidelity: 0.89) + arm("second", fidelity: 0.90)
        let decision = RoutingBench.decide(rows: close, arms: arms)
        #expect(decision.winner == "first")
        #expect(decision.eligible == ["first", "second"] && decision.failures.isEmpty)
        #expect(abs((decision.fidelity["first"] ?? 0) - 0.89) < 1e-5)
    }

    @Test("each requirement can rule a candidate out, and with none eligible the baseline stays")
    func requirements() {
        func failures(_ rows: [GateBenchRow]) -> [String] {
            RoutingBench.decide(rows: arm("posterior") + rows + arm("second"), arms: arms).failures["first"] ?? []
        }
        #expect(failures(arm("first", factsExact: 9)).isEmpty, "one prompt below the baseline is allowed")
        #expect(failures(arm("first", factsExact: 8)).first?.hasPrefix("facts: 8 exact") == true)
        #expect(failures(arm("first", lostOwnership: 1)).first?.contains("no longer led") == true)
        #expect(failures(arm("first", pairsMoved: 8)).first?.hasPrefix("pairs: the lead moved in 80%") == true)
        #expect(failures(arm("first", genericAsked: 7)).first?.hasPrefix("generic: every Thread asked in 88%") == true)
        #expect(failures(arm("first", genericGate: 0.7)).first?.hasPrefix("generic: largest gate 0.70") == true)
        #expect(failures(arm("first", unknownGate: 0.85)).first?.hasPrefix("unknown: largest gate 0.85") == true)
        let none = RoutingBench.decide(rows: arm("posterior") + arm("first", pairsMoved: 0) + arm("second", genericAsked: 0), arms: arms)
        #expect(none.winner == nil && none.eligible.isEmpty)
        #expect(none.summary.contains("posterior stays"))
        // A set with no prompts is not passed by default.
        let empty = RoutingBench.decide(rows: arm("posterior") + arm("first").filter { $0.set != .pairs }, arms: Array(arms.prefix(2)))
        #expect(empty.failures["first"] == ["pairs: no prompts"])
    }

    @Test("share fidelity: 1 when the shares match who predicted the token, lower when the gate credits the wrong Thread")
    func fidelity() {
        func share(_ strand: String, _ value: Float) -> StrandShare {
            StrandShare(strand: strand, threadID: nil, gate: 0, open: true, bestScore: nil, lmProb: nil, lmEntropy: nil, knn: 0, share: value)
        }
        #expect(RoutingBench.fidelity([share("a", 1), share("b", 0)], alone: ["a": 0.9, "b": 0.01]) == 1)
        #expect(RoutingBench.fidelity([share("a", 0.5), share("b", 0.5)], alone: ["a": 0.9, "b": 0.8]) == 1)
        #expect(RoutingBench.fidelity([share("a", 1), share("b", 0)], alone: ["a": 0.9, "b": 0.8]) == 0.5)
        #expect(RoutingBench.fidelity([share("a", 0), share("b", 1)], alone: ["a": 0.9, "b": 0.1]) == 0)
        #expect(RoutingBench.fidelity([share("a", 0.5), share("b", 0.5)], alone: ["a": 0.3, "b": 0.2]) == nil)
    }
}
