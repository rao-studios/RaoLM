import Foundation
import Testing

@testable import RaoLMBraid
@testable import RaoLMCore

@Suite("bench-scale and the sync throttle")
struct ScaleBenchTests {
    static func state(_ name: String, _ stage: NodeStage, versions: Int, ladder: [LadderMark] = []) -> StrandState {
        var state = StrandState(name: name, label: name, offline: true, vocabularySHA256: "v", blocks: 4)
        state.stage = stage
        state.versions = versions
        state.ladder = ladder
        return state
    }

    @Test("one at a time: the second node is sent its sync only after the first settles; held and failed count as settled")
    func throttleOne() {
        var throttle = SyncThrottle(names: ["a", "b", "c"], atOnce: 1)
        var states = ["a": Self.state("a", .empty, versions: 0), "b": Self.state("b", .empty, versions: 0), "c": Self.state("c", .empty, versions: 0)]
        #expect(throttle.advance(states: states) == ["a"])
        // Not yet started: idle, no new version, never seen busy.
        #expect(throttle.advance(states: states).isEmpty)
        states["a"] = Self.state("a", .training, versions: 0)
        #expect(throttle.advance(states: states).isEmpty)
        // Idle with a ladder step still pending is in flight.
        states["a"] = Self.state("a", .live, versions: 1, ladder: [LadderMark("index", .pending)])
        #expect(throttle.advance(states: states).isEmpty)
        states["a"] = Self.state("a", .live, versions: 1, ladder: [LadderMark("index", .done)])
        #expect(throttle.advance(states: states) == ["b"] && throttle.finished == ["a"])
        states["b"] = Self.state("b", .held, versions: 0)
        #expect(throttle.advance(states: states) == ["c"])
        states["c"] = Self.state("c", .failed, versions: 0)
        #expect(throttle.advance(states: states).isEmpty && throttle.isDone && throttle.finished == ["a", "b", "c"])
    }

    @Test("a sync that found nothing new settles once the node was seen busy and is idle again; two at once send two")
    func throttleNoChange() {
        var throttle = SyncThrottle(names: ["a", "b", "c"], atOnce: 2)
        var states = ["a": Self.state("a", .live, versions: 2), "b": Self.state("b", .live, versions: 1), "c": Self.state("c", .live, versions: 1)]
        #expect(throttle.advance(states: states) == ["a", "b"])
        states["a"] = Self.state("a", .exporting, versions: 2)
        #expect(throttle.advance(states: states).isEmpty)
        states["a"] = Self.state("a", .live, versions: 2)
        #expect(throttle.advance(states: states) == ["c"])
        #expect(SyncThrottle(names: ["x", "y"], atOnce: 0).inFlight.isEmpty)
        var all = SyncThrottle(names: ["x", "y"], atOnce: 0)
        #expect(all.advance(states: [:]) == ["x", "y"])
    }

    static func arm(facts: Int, exact: Int, owned: Int, citation: Float, leads: Float = 1, share: Float = 0.03) -> UmbrellaArmResult {
        UmbrellaArmResult(
            arm: "x", facts: facts, factsExact: exact, factsOwned: owned, citation: citation, ownerShare: nil, liftPositive: nil, pairs: 0,
            pairsMoved: 0, pairsExact: 0, unknownAllAsked: 0, unknownLargest: 0, genericAllAsked: 0, genericLargest: 0, commonsLeads: leads,
            commonsThreadGate: nil, commonsThreadShare: share, heldOutNLL: 0, heldOutTokens: 0, toldSourceShare: nil, toldTokens: 0,
            voiceLargest: nil, seconds: 0)
    }

    static func point(_ n: Int, exact: Int, owned: Int = 30, citation: Float = 0.95, leads: Float = 1, share: Float = 0.03,
                      seconds: Double = 0.1, pack: String = "p") -> ScalePoint {
        let facts = arm(facts: 30, exact: exact, owned: owned, citation: citation)
        return ScalePoint(
            braid: "braid-n\(n)", nodes: n, names: (0..<n).map { "n\($0)" }, pack: pack, dataset: "braid-n\(n)", datasetHash: nil, facts: facts,
            prompts: arm(facts: 0, exact: 0, owned: 0, citation: 0, leads: leads, share: share), unknown: facts, unknownCommonsLeads: nil,
            askedPerFactToken: nil, factsByNode: [:], nodeResults: [],
            tokenCost: ScaleTokenCost(prompts: 8, shortTokens: 8, longTokens: 24, secondsPerToken: seconds, secondsPerPrompt: 0, askedPerToken: Float(n),
                                      allAskedRate: 1),
            peakFootprintBytes: nil, everyFact: false, seconds: 0)
    }

    @Test("N1 and N2: each point against the smallest N's facts, the owner's lead and the commons' lead")
    func evaluate() {
        let reference = Self.point(3, exact: 27)
        #expect(ScaleBench.evaluate(points: [reference, Self.point(6, exact: 26)]).qualifies)
        #expect(!ScaleBench.evaluate(points: [reference, Self.point(6, exact: 25)]).qualifies, "two facts in thirty fewer is more than 1/30")
        #expect(!ScaleBench.evaluate(points: [reference, Self.point(6, exact: 27, owned: 26)]).qualifies, "the owner leads and is cited on 87%")
        #expect(!ScaleBench.evaluate(points: [reference, Self.point(6, exact: 27, citation: 0.92)]).qualifies)
        #expect(!ScaleBench.evaluate(points: [reference, Self.point(6, exact: 27, leads: 0.85)]).qualifies)
        #expect(!ScaleBench.evaluate(points: [reference, Self.point(6, exact: 27, share: 0.12)]).qualifies)
        let other = ScaleBench.evaluate(points: [reference, Self.point(6, exact: 27, pack: "q")])
        #expect(!other.qualifies && other.summary.contains("not the reference's"))
        // The smallest N is the reference whatever the order given.
        let rules = ScaleBench.evaluate(points: [Self.point(24, exact: 27), reference]).rules
        #expect(rules.count == 4 && rules.first?.rule.hasSuffix("N = 3") == true)
    }

    @Test("the cost fit is least squares against N")
    func fit() throws {
        let fit = try #require(ScaleBench.fit([3, 6, 12, 24], [0.3, 0.6, 1.2, 2.4]))
        #expect(abs(fit.slope - 0.1) < 1e-9 && abs(fit.intercept) < 1e-9 && abs(fit.r2 - 1) < 1e-9)
        #expect(ScaleBench.fit([3], [0.3]) == nil)
        let report = ScaleBench.report([Self.point(6, exact: 27, seconds: 0.6), Self.point(3, exact: 27, seconds: 0.3)])
        #expect(report.reference == 3 && report.points.map(\.nodes) == [3, 6] && abs((report.evaluation.secondsFit?.slope ?? 0) - 0.1) < 1e-9)
    }

    @Test("the pairs budget picks evenly over the ordered pairs, and leaves a smaller set alone")
    func budget() {
        let examples = (0..<552).map { i in
            BraidExample(label: "\(i)", node: nil, promptTokens: [i], promptText: "\(i)", expected: nil, source: nil, kind: .generic)
        }
        let picked = UmbrellaBench.budgeted(examples, total: 60)
        #expect(picked.count == 60 && picked.first?.label == "0" && Set(picked.map(\.label)).count == 60)
        let gaps = zip(picked.dropFirst(), picked).map { Int($0.0.label)! - Int($0.1.label)! }
        #expect(gaps.allSatisfy { (9...10).contains($0) })
        #expect(UmbrellaBench.budgeted(examples, total: nil).count == 552 && UmbrellaBench.budgeted(Array(examples.prefix(10)), total: 60).count == 10)
    }

    @Test("H1 asks every answer token for token and shares within 1e-6; H3 asks TCP at most 1.2× pipes")
    func hosted() {
        #expect(HostedBench.evaluate(prompts: 240, identical: 240, maxShareDifference: 3e-7, ratio: 1.1).qualifies)
        let different = HostedBench.evaluate(prompts: 240, identical: 239, maxShareDifference: 0, ratio: 1.0)
        #expect(!different.qualifies && different.summary.contains("H1"))
        #expect(!HostedBench.evaluate(prompts: 240, identical: 240, maxShareDifference: 2e-6, ratio: 1.0).qualifies)
        let slow = HostedBench.evaluate(prompts: 240, identical: 240, maxShareDifference: 0, ratio: 1.25)
        #expect(!slow.qualifies && slow.summary.contains("H3") && !slow.summary.contains("H1"))
        #expect(!HostedBench.evaluate(prompts: 0, identical: 0, maxShareDifference: 0, ratio: 1.0).qualifies)
    }
}
