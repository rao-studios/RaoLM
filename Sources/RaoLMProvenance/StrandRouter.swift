//
//  StrandRouter.swift
//  RaoLMProvenance
//
//  WHAT: Routing before asking (Docs/ARCHITECTURE.md, "Step 1"): which Threads a generation
//        opens at all. Each Thread's descriptor carries its corpus's token bigrams (its sketch);
//        a prompt's bigram is distinctive when at most a quarter of the Threads hold it, and the
//        candidates are the Threads holding any distinctive bigram of the prompt, with the
//        commons always in. No distinctive bigram: every Thread, as without routing.
//  PIN:  Exact: a bigram is two token ids packed into one word (`a << 32 | b`), no hashing, so
//        a Thread holds a bigram or it does not. A link without a sketch (the commons, or a node
//        that predates sketches) is always a candidate.
//

import Foundation
import RaoLMCore

public struct StrandRoute: Sendable, Equatable {
    /// Per link: whether this generation opens it.
    public var candidates: [Bool]
    /// The prompt's distinct bigrams, and how many of them were distinctive.
    public var bigrams: Int
    public var distinctive: Int

    /// No distinctive bigram: every Thread is asked.
    public var unmatched: Bool { distinctive == 0 }
    public var opened: Int { candidates.filter { $0 }.count }
}

public struct StrandRouter: Sendable {
    /// Per link: its sketch, or nil for a link that is always a candidate.
    let sketches: [Set<UInt64>?]
    /// The Threads among the links (those with a sketch).
    public let threads: Int

    public init(sketches: [Set<UInt64>?]) {
        self.sketches = sketches
        threads = sketches.compactMap { $0 }.count
    }

    public init(descriptors: [StrandDescriptor]) {
        self.init(sketches: descriptors.map { $0.isCommons ? nil : $0.sketch.map { Set($0.values) } })
    }

    /// The most Threads a distinctive bigram may be held by: a quarter of them, at least one.
    public var bound: Int { max(1, threads / 4) }

    public static func key(_ a: Int, _ b: Int) -> UInt64 { UInt64(UInt32(truncatingIfNeeded: a)) << 32 | UInt64(UInt32(truncatingIfNeeded: b)) }

    public static func bigrams(_ tokens: [Int]) -> Set<UInt64> {
        guard tokens.count >= 2 else { return [] }
        var keys = Set<UInt64>(minimumCapacity: tokens.count)
        for i in 1..<tokens.count { keys.insert(key(tokens[i - 1], tokens[i])) }
        return keys
    }

    public func route(_ prompt: [Int]) -> StrandRoute {
        let keys = Self.bigrams(prompt)
        var candidates = sketches.map { $0 == nil }
        var distinctive = 0
        for key in keys {
            let holders = sketches.indices.filter { sketches[$0]?.contains(key) == true }
            guard !holders.isEmpty, holders.count <= bound else { continue }
            distinctive += 1
            for t in holders { candidates[t] = true }
        }
        if distinctive == 0 { candidates = sketches.map { _ in true } }
        return StrandRoute(candidates: candidates, bigrams: keys.count, distinctive: distinctive)
    }
}
