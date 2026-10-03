import Foundation
import Testing

@testable import RaoLMCore
@testable import RaoLMProvenance

@Suite("The knowledge profile and its router")
struct ThreadProfileTests {
    /// Points around three centres in 8 dimensions, each with its own lift.
    static func clusters() -> (states: [Float], weights: [Float], centres: [[Float]]) {
        let centres: [[Float]] = [
            [10, 0, 0, 0, 0, 0, 0, 0],
            [0, 10, 0, 0, 0, 0, 0, 0],
            [0, 0, 10, 0, 0, 0, 0, 0],
        ]
        var rng = SplitMix64(seed: 5)
        var states: [Float] = []
        var weights: [Float] = []
        for (c, centre) in centres.enumerated() {
            for _ in 0..<40 {
                states += centre.map { $0 + Float(rng.nextUnit() - 0.5) * 0.2 }
                weights.append(Float(c + 1))
            }
        }
        return (states, weights, centres)
    }

    @Test("lift is how much better the Thread predicts than the commons, never below 0")
    func lift() {
        #expect(ThreadProfile.lift(commonsLoss: [3, 1, 2], threadLoss: [1, 2, 2]) == [2, 0, 0])
    }

    @Test("weighted k-means finds the centres, each centroid carrying its points' lift, and the same seed gives the same centroids")
    func kMeans() {
        let (states, weights, centres) = Self.clusters()
        let (centroids, mass) = ThreadProfile.kMeans(states: states, hidden: 8, weights: weights, k: 3, iterations: 25, seed: 42)
        #expect(centroids.count == 24 && mass.count == 3)
        let found = (0..<3).map { Array(centroids[($0 * 8)..<(($0 + 1) * 8)]) }
        for (c, centre) in centres.enumerated() {
            let near = found.firstIndex { row in zip(row, centre).allSatisfy { abs($0 - $1) < 0.2 } }
            #expect(near != nil, "centre \(c) found")
            if let near { #expect(abs(mass[near] - Float(40 * (c + 1))) < 1e-3) }
        }
        let again = ThreadProfile.kMeans(states: states, hidden: 8, weights: weights, k: 3, iterations: 25, seed: 42)
        #expect(again.centroids == centroids && again.mass == mass)
    }

    @Test("a point of lift 0 moves nothing, and k never exceeds the weighted points")
    func zeroWeight() {
        let (states, weights, _) = Self.clusters()
        let outlier: [Float] = [0, 0, 0, 0, 0, 0, 0, 1000]
        let base = ThreadProfile.kMeans(states: states, hidden: 8, weights: weights, k: 3, iterations: 25, seed: 42)
        let with = ThreadProfile.kMeans(states: states + outlier, hidden: 8, weights: weights + [0], k: 3, iterations: 25, seed: 42)
        #expect(base.centroids == with.centroids)
        let two = ThreadProfile.kMeans(states: Array(states.prefix(16)), hidden: 8, weights: [1, 1], k: 8, iterations: 25, seed: 42)
        #expect(two.mass.count == 2)
        #expect(ThreadProfile.kMeans(states: Array(states.prefix(8)), hidden: 8, weights: [0], k: 3, iterations: 5, seed: 1).mass.isEmpty)
    }

    @Test("spread is the mean pairwise cosine of the centroids")
    func spread() {
        #expect(abs(ThreadProfile.spread(centroids: [1, 0, 0, 1], hidden: 2)) < 1e-6)
        #expect(abs(ThreadProfile.spread(centroids: [1, 0, 2, 0], hidden: 2) - 1) < 1e-6)
    }

    // MARK: - The router

    /// Four Threads with one-centroid profiles along their own axis, and the commons (no profile) last.
    static func router(bar: Float = 0.5, cap: Int = 6) -> ProfileRouter {
        let axes: [[Float]?] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], nil]
        return ProfileRouter(profiles: axes, hidden: 4, bar: bar, cap: cap)
    }

    @Test("a Thread scores by its best centroid; Threads at half the best score or more open, with the commons")
    func bar() {
        let router = Self.router()
        #expect(router.profiled == 4)
        let route = router.route(query: [1, 0.5, 0.49, 0])
        #expect(route.candidates == [true, true, false, false, true])
        #expect(route.opened == 2 && !route.unrouted)
        let two = ProfileRouter(profiles: [[1, 0, 0, 0, 1, 0], nil], hidden: 3)
        #expect(abs((two.scores(query: [0, 1, 0])[0] ?? 0) - 1) < 1e-6, "the best of a Thread's centroids is its score")
        #expect(two.scores(query: [0, 1, 0])[1] == nil)
    }

    @Test("at most cap Threads open, the highest first and ties by position")
    func cap() {
        let equal = ProfileRouter(profiles: (0..<8).map { _ in [1, 0] } + [nil], hidden: 2, cap: 6)
        let route = equal.route(query: [1, 0])
        #expect(route.opened == 6 && route.candidates == [true, true, true, true, true, true, false, false, true])
    }

    @Test("no profile anywhere, or no positive score: every Thread opens")
    func unrouted() {
        let router = Self.router()
        let negative = router.route(query: [-1, -1, -1, -1])
        #expect(negative.unrouted && negative.candidates.allSatisfy { $0 })
        let zero = router.route(query: [0, 0, 0, 0])
        #expect(zero.unrouted)
        let none = ProfileRouter(profiles: [nil, nil], hidden: 4).route(query: [1, 0, 0, 0])
        #expect(none.unrouted && none.candidates == [true, true])
    }

    @Test("identical profiles score alike and open together")
    func copies() {
        let router = ProfileRouter(profiles: [[1, 0.2, 0], [1, 0.2, 0], [0, 0, 1], nil], hidden: 3)
        let route = router.route(query: [1, 0.3, 0.1])
        #expect(route.scores[0] == route.scores[1] && route.candidates[0] && route.candidates[1] && !route.candidates[2])
    }
}
