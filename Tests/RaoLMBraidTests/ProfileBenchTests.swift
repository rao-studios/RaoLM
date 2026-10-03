import Foundation
import Testing

@testable import RaoLMBraid
@testable import RaoLMCore
@testable import RaoLMProvenance

@Suite("bench-profile: P1 to P4, credit recall and copies")
struct ProfileBenchTests {
    static func side() -> RouteSide {
        let arm = ScaleBenchTests.arm(facts: 30, exact: 27, owned: 30, citation: 0.9)
        let cost = ScaleTokenCost(prompts: 8, shortTokens: 8, longTokens: 24, secondsPerToken: 0.01, secondsPerPrompt: 0, askedPerToken: 1, allAskedRate: 0)
        return RouteSide(facts: arm, prompts: arm, unknown: arm, factCost: cost, genericCost: cost, seconds: 0)
    }

    static func point(
        found: Int = 240, pairsFound: Int = 60, opened: Double = 1.5, creditOpened: Double = 98, copies: ProfileCopyStats? = nil
    ) -> ProfilePoint {
        ProfilePoint(
            braid: "braid-n24", nodes: 24, pack: "p", dataset: nil, datasetHash: nil, threads: [], facts: 240, factsFound: found, pairs: 60,
            pairsFound: pairsFound, missed: [],
            sets: [ProfileSetStats(set: "facts", prompts: 240, unrouted: 0, meanOpened: opened, maxOpened: 3, histogram: [0, 120, 120])],
            worlds: nil, creditOpened: creditOpened, creditTotal: 100, creditRecallFacts: nil, creditRecallPairs: nil, liveMatchesDry: 240,
            liveChecked: 240, copies: copies, subjects: nil, unrouted: side(), routed: side(), seconds: 0)
    }

    @Test("P1 owner found, P2 size, P3 credit recall: each fails on its own measure")
    func evaluate() {
        #expect(ProfileBench.evaluate(points: [Self.point()]).qualifies)
        let lost = ProfileBench.evaluate(points: [Self.point(found: 236)])
        #expect(!lost.qualifies && lost.summary.contains("P1") && !lost.summary.contains("P2"))
        #expect(!ProfileBench.evaluate(points: [Self.point(pairsFound: 58)]).qualifies)
        let wide = ProfileBench.evaluate(points: [Self.point(opened: 4.2)])
        #expect(!wide.qualifies && wide.summary.contains("P2"))
        let leaked = ProfileBench.evaluate(points: [Self.point(creditOpened: 94)])
        #expect(!leaked.qualifies && leaked.summary.contains("P3"))
        #expect(ProfileBench.evaluate(points: []).summary == "no points")
    }

    @Test("P4 is judged only where two nodes hold the same documents, and needs every fact within 1% and opened together")
    func copies() {
        #expect(!ProfileBench.evaluate(points: [Self.point()]).rules.contains { $0.rule.hasPrefix("P4") })
        let alike = ProfileCopyStats(a: "ambient", b: "craft", prompts: 50, within: 50, openedTogether: 50, maxRelativeDifference: 0.004)
        #expect(ProfileBench.evaluate(points: [Self.point(copies: alike)]).qualifies)
        var apart = alike
        apart.within = 49
        #expect(!ProfileBench.evaluate(points: [Self.point(copies: apart)]).qualifies)
        var alone = alike
        alone.openedTogether = 48
        #expect(!ProfileBench.evaluate(points: [Self.point(copies: alone)]).qualifies)
    }

    static func subjects(opened: Int = 90, leads: Int = 85, stray: Double = 0.05, strayShare: Double = 0.02, ownerShare: Double = 0.95,
                         lift: Int = 190, homonymsDecided: Int = 11) -> ProfileSubjectStats {
        ProfileSubjectStats(questions: 90, ownerOpened: opened, ownerLeads: leads, strayOpenedMean: stray, strayShareMean: strayShare, exact: 70,
                            ownerShareMean: ownerShare, liftPositive: lift, ownerTokens: 200, homonyms: 12, homonymsDecided: homonymsDecided, byWorld: [:])
    }

    @Test("S1 to S4 are judged where every node holds a subject, each on its own measure")
    func subjectRules() {
        var point = Self.point()
        point.subjects = Self.subjects()
        let passing = ProfileBench.evaluate(points: [point])
        #expect(passing.qualifies && passing.rules.filter { $0.rule.hasPrefix("S") }.count == 4)
        for (subjects, rule) in [(Self.subjects(opened: 88), "S1"), (Self.subjects(leads: 80), "S1"), (Self.subjects(stray: 0.2), "S2"),
                                 (Self.subjects(strayShare: 0.15), "S2"), (Self.subjects(ownerShare: 0.85), "S3"), (Self.subjects(lift: 180), "S3"),
                                 (Self.subjects(homonymsDecided: 10), "S4")] {
            point.subjects = subjects
            let result = ProfileBench.evaluate(points: [point])
            #expect(!result.qualifies && result.summary.contains(rule), "\(rule): \(result.summary)")
        }
    }

    @Test("copies are the pair sharing the most documents, at least half the smaller node's")
    func copyPair() {
        let copied = ProfileBench.copyPair(["ambient": Set((0..<82).map(String.init)), "craft": Set((2..<82).map(String.init)),
                                            "veil": Set((100..<180).map(String.init))])
        #expect(copied?.0 == "ambient" && copied?.1 == "craft")
        #expect(ProfileBench.copyPair(["a": ["1", "2", "3"], "b": ["3", "4", "5", "6"]]) == nil)
    }

    /// A generated trace whose Threads were paid `credits`, with `bits` of surprise.
    static func trace(bits: Float, _ credits: [String: Float], prompt: Bool = false) -> TokenTrace {
        var value = TokenTrace(index: 0, token: 1, text: "x", isPrompt: prompt, lmEntropy: 0, knnEntropy: 0, mixedEntropy: 0, sourceEntropy: 0,
                               lmProb: 0.5, agreement: 0, mixedProb: 0.5, lambda: 0.5, neighbours: [])
        value.bits = bits
        value.strands = credits.keys.sorted().map { name in
            var share = StrandShare(strand: name, threadID: nil, gate: 0.5, open: true, bestScore: nil, lmProb: nil, lmEntropy: nil, knn: 0, share: 0.5)
            share.credit = credits[name]
            return share
        }
        return value
    }

    @Test("credit recall weighs each token's credit by its bits, leaves out the commons and the prompt")
    func credit() {
        let traces = [
            Self.trace(bits: 2, ["ambient": 0.6, "craft": 0.3, "commons": 0.1]),
            Self.trace(bits: 1, ["ambient": 0, "craft": 0.8, "commons": 0.2]),
            Self.trace(bits: 9, ["ambient": 1], prompt: true),
        ]
        let paid = ProfileBench.credit(traces, opened: ["ambient"], commons: "commons")
        #expect(abs(paid.opened - 1.2) < 1e-6 && abs(paid.total - 2.6) < 1e-6)
        let all = ProfileBench.credit(traces, opened: ["ambient", "craft"], commons: "commons")
        #expect(abs(all.opened - all.total) < 1e-9)
    }
}
