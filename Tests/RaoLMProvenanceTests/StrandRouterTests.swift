import Foundation
import Testing

@testable import RaoLMCore
@testable import RaoLMProvenance

@Suite("Routing before asking")
struct StrandRouterTests {
    /// Eight Threads: 10 and 11 are words every Thread writes; each Thread t also writes its own
    /// subject (100 + t, 200 + t); Thread 1 retells Thread 0's subject.
    static func router(commons: Bool = true) -> StrandRouter {
        var sketches: [Set<UInt64>?] = (0..<8).map { t in
            var tokens = [10, 11, 100 + t, 200 + t, 10, 11]
            if t == 1 { tokens += [100, 200] }
            return StrandRouter.bigrams(tokens)
        }
        if commons { sketches.insert(nil, at: 0) }
        return StrandRouter(sketches: sketches)
    }

    @Test("bigrams pack two token ids exactly, in order")
    func bigrams() {
        #expect(StrandRouter.bigrams([1, 2, 1, 2]) == [StrandRouter.key(1, 2), StrandRouter.key(2, 1)])
        #expect(StrandRouter.key(1, 2) != StrandRouter.key(2, 1) && StrandRouter.key(49_151, 0) == UInt64(49_151) << 32)
        #expect(StrandRouter.bigrams([7]).isEmpty)
    }

    @Test("a subject's bigram opens its holders and the commons; common bigrams open nobody")
    func one() {
        let router = Self.router()
        #expect(router.threads == 8 && router.bound == 2)
        // Thread 3's subject, written as Thread 3 writes it.
        let own = router.route([10, 11, 103, 203])
        #expect(own.candidates == [true, false, false, false, true, false, false, false, false])
        #expect(own.distinctive >= 1 && !own.unmatched && own.opened == 2)
        // Thread 0's subject is retold by Thread 1: both are candidates.
        let retold = router.route([100, 200])
        #expect(retold.candidates == [true, true, true, false, false, false, false, false, false])
    }

    @Test("two subjects open both owners; none opens every Thread")
    func twoAndNone() {
        let router = Self.router()
        let two = router.route([103, 203, 10, 11, 105, 205])
        #expect(two.candidates == [true, false, false, false, true, false, true, false, false])
        // Only bigrams every Thread holds, or none holds: nothing distinctive, so every Thread.
        let none = router.route([10, 11, 999, 998])
        #expect(none.unmatched && none.candidates.allSatisfy { $0 })
    }

    @Test("a link without a sketch is always a candidate, and a quarter bounds a distinctive bigram")
    func bound() {
        // Three Threads: a quarter is under one, so the bound is one.
        let small = StrandRouter(sketches: [StrandRouter.bigrams([1, 2]), StrandRouter.bigrams([1, 2]), StrandRouter.bigrams([3, 4]), nil])
        #expect(small.bound == 1)
        // Held by two of three: not distinctive.
        #expect(small.route([1, 2]).unmatched)
        #expect(small.route([3, 4]).candidates == [false, false, true, true])
        // From descriptors: the commons has no sketch.
        #expect(Self.router(commons: false).threads == 8)
    }
}
