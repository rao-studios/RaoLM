import Foundation
import Testing

@testable import RaoLMCore

@Suite("Stats")
struct StatsTests {
    @Test("Pearson: perfect, inverse, known value, and nil on degenerate input")
    func pearson() throws {
        let x: [Double] = [1, 2, 3, 4, 5]
        let linear = try #require(Stats.pearson(x, x.map { 3 * $0 - 7 }))
        #expect(abs(linear - 1) < 1e-12)
        let inverse = try #require(Stats.pearson(x, x.map { -0.5 * $0 + 2 }))
        #expect(abs(inverse + 1) < 1e-12)
        // Hand-computed: x̄ = 3, ȳ = 3.2; Σdxdy = 6, Σdx² = 10, Σdy² = 6.8 → r = 6/√68.
        let known = try #require(Stats.pearson(x, [2, 3, 2, 5, 4]))
        #expect(abs(known - 6 / 68.0.squareRoot()) < 1e-12)

        #expect(Stats.pearson([1, 2], [1, 2]) == nil)
        #expect(Stats.pearson([0.1, 0.1, 0.1, 0.1], [1, 2, 3, 4]) == nil)
        #expect(Stats.pearson([1, 2, 3], [5, 5, 5]) == nil)
        #expect(Stats.pearson([1, 2, 3], [1, 2]) == nil)
        #expect(Stats.pearson([1, 2, .nan], [1, 2, 3]) == nil)
    }

    @Test("AUROC: separable scores give 1, ties give 0.5, one class gives nil")
    func auroc() {
        #expect(Stats.auroc(scores: [0.9, 0.8, 0.2, 0.1], labels: [true, true, false, false]) == 1)
        #expect(Stats.auroc(scores: [0.5, 0.5], labels: [true, false]) == 0.5)
        #expect(Stats.auroc(scores: [0.5, 0.4], labels: [true, true]) == nil)
    }
}
