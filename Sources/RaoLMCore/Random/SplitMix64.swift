//
//  SplitMix64.swift
//  RaoLMCore
//
//  WHAT: A tiny, fully specified pseudo-random generator for everything RaoLM must
//        reproduce from a seed: the synthetic corpus, the data order, fact sampling
//        and seeded token sampling.
//  PIN:  Swift's own `Int.random(in:using:)` and `shuffled(using:)` do not promise a
//        stable algorithm across releases, so bounded draws and shuffles are written
//        here. Same seed, same bytes, on every toolchain.
//

import Foundation

public struct SplitMix64: RandomNumberGenerator, Sendable {
    public private(set) var state: UInt64

    public init(seed: UInt64) {
        self.state = seed
    }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// A value in `0 ..< bound`. Modulo reduction: the bias is below 2^-40 for every bound
    /// RaoLM uses, and the result is defined by this line rather than by the standard library.
    public mutating func nextInt(below bound: Int) -> Int {
        precondition(bound > 0, "bound must be positive")
        return Int(next() % UInt64(bound))
    }

    /// A value in `range` (inclusive).
    public mutating func nextInt(in range: ClosedRange<Int>) -> Int {
        range.lowerBound + nextInt(below: range.upperBound - range.lowerBound + 1)
    }

    /// A double in `[0, 1)` with 53 bits of randomness.
    public mutating func nextUnit() -> Double {
        Double(next() >> 11) * (1.0 / 9_007_199_254_740_992.0)
    }

    public mutating func pick<T>(_ items: [T]) -> T {
        precondition(!items.isEmpty, "cannot pick from an empty array")
        return items[nextInt(below: items.count)]
    }

    /// Fisher–Yates, drawing from this generator.
    public mutating func shuffled<T>(_ items: [T]) -> [T] {
        var result = items
        if result.count < 2 { return result }
        for i in stride(from: result.count - 1, to: 0, by: -1) {
            let j = nextInt(below: i + 1)
            if i != j { result.swapAt(i, j) }
        }
        return result
    }

    /// A derived generator for a named sub-stream, so adding draws to one part of the
    /// program does not shift the numbers another part sees.
    public static func derived(seed: UInt64, stream: UInt64) -> SplitMix64 {
        var mixer = SplitMix64(seed: seed ^ (stream &* 0xD1B5_4A32_D192_ED03))
        return SplitMix64(seed: mixer.next())
    }
}
