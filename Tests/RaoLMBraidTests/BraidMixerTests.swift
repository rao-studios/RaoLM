import Foundation
import Testing

@testable import RaoLMCore
@testable import RaoLMProvenance

private func hit(_ entry: Int, _ score: Float, value: Int, row: Int = 0) -> StrandHit {
    StrandHit(entry: entry, score: score, value: value, key: TokenPosition(row: row, offset: entry),
              cited: TokenPosition(row: row, offset: entry + 1), sourceLoss: 0.01, sourceEntropy: 0.1)
}

@Suite("BraidMixer")
struct BraidMixerTests {
    @Test("pooling ranks every Thread's hits together and gates by the share of weight")
    func pooling() {
        let a = [hit(0, 0.99, value: 7), hit(1, 0.90, value: 8)]
        let b = [hit(0, 0.95, value: 7), hit(1, 0.60, value: 9)]
        let pool = BraidMixer.pool([a, b], k: 3, tau: 0.05)
        #expect(pool.members.map(\.strand) == [0, 1, 0])
        #expect(pool.members.map(\.hit.score) == [0.99, 0.95, 0.90])
        #expect(abs(pool.gates.reduce(0, +) - 1) < 1e-5)
        #expect(pool.gates[0] > pool.gates[1])
        #expect(pool.best == [0.99, 0.95])
        #expect(pool.open == [true, true])
        // Equal scores: the earlier strand, then the lower entry, come first.
        let tie = BraidMixer.pool([[hit(5, 0.8, value: 1)], [hit(2, 0.8, value: 1)]], k: 2, tau: 0.05)
        #expect(tie.members.map(\.strand) == [0, 1])
    }

    @Test("a Thread below the floor is closed, the largest gate is always open, no hits opens every Thread")
    func gates() {
        #expect(BraidMixer.gate([0.97, 0.03], floor: 0.05) == [true, false])
        #expect(BraidMixer.gate([0.02, 0.01], floor: 0.05) == [true, false])
        #expect(BraidMixer.gate([0, 0], floor: 0.05) == [true, true])
        #expect(BraidMixer.gate([0.5, 0, 0.5], floor: 0.05) == [true, false, true])
        let pool = BraidMixer.pool([[], []], k: 4, tau: 0.05)
        #expect(pool.open == [true, true])
        #expect(BraidMixer.renormalised(pool) == [0.5, 0.5])
    }

    @Test("a token's shares sum to one and follow the gates and the heads")
    func shares() {
        let a = [hit(0, 0.99, value: 2), hit(1, 0.97, value: 2)]
        let b = [hit(0, 0.96, value: 3), hit(1, 0.94, value: 2)]
        let pool = BraidMixer.pool([a, b], k: 4, tau: 0.05)
        // Thread 0's head puts its mass on token 2, thread 1's on token 3.
        let logits: [Int: [Float]] = [0: [0, 0, 6, 0], 1: [0, 0, 0, 6]]
        for lambda: Float in [0, 0.5, 1] {
            let mix = BraidMixer.mix(pool, logits: logits, lambda: lambda, vocabularySize: 4)
            #expect(abs(mix.mixed.reduce(0, +) - 1) < 1e-4)
            for token in 0..<4 where mix.mixed[token] > 0 {
                let shares = BraidMixer.shares(mix, pool: pool, token: token, names: ["a", "b"], threadIDs: ["A", "B"])
                #expect(abs(shares.map(\.share).reduce(0, +) - 1) < 1e-4, "λ \(lambda) token \(token)")
            }
        }
        let mix = BraidMixer.mix(pool, logits: logits, lambda: 0.5, vocabularySize: 4)
        let two = BraidMixer.shares(mix, pool: pool, token: 2, names: ["a", "b"], threadIDs: ["A", "B"])
        let three = BraidMixer.shares(mix, pool: pool, token: 3, names: ["a", "b"], threadIDs: ["A", "B"])
        #expect(two[0].share > 0.8)
        #expect(three[1].share > 0.8)
        #expect(two[0].threadID == "A" && two[0].open && two[0].lmProb != nil)
        #expect(two[0].knn > 0 && three[1].knn > 0)
    }

    @Test("a closed Thread is never mixed in, and supplies only its retrieval weight")
    func closed() {
        let a = (0..<4).map { hit($0, 0.99 - Float($0) * 0.001, value: 2) }
        let b = [hit(0, 0.70, value: 3)]
        let pool = BraidMixer.pool([a, b], k: 5, tau: 0.05)
        #expect(pool.open == [true, false])
        let mix = BraidMixer.mix(pool, logits: [0: [0, 0, 5, 0]], lambda: 0.5, vocabularySize: 4)
        #expect(mix.strandLM[1] == nil)
        #expect(mix.gatesOpen == [1, 0])
        let shares = BraidMixer.shares(mix, pool: pool, token: 3, names: ["a", "b"], threadIDs: [nil, nil])
        #expect(shares[1].lmProb == nil)
        #expect(shares[1].share > 0 && shares[1].share < 1)
        #expect(abs(shares.map(\.share).reduce(0, +) - 1) < 1e-4)
    }

    @Test("one Thread mixes exactly as CitationMixer does")
    func oneThread() {
        let hits = [hit(0, 0.93, value: 1), hit(4, 0.91, value: 2), hit(9, 0.80, value: 1)]
        let logits: [Float] = [0.1, 2.0, 1.2, -0.5, 0.3]
        let pool = BraidMixer.pool([hits], k: 3, tau: 0.05)
        let mix = BraidMixer.mix(pool, logits: [0: logits], lambda: 0.5, vocabularySize: 5)
        let weights = CitationMixer.weights(hits.map(\.score), tau: 0.05)
        var knn: [Int: Float] = [:]
        for (i, h) in hits.enumerated() { knn[h.value, default: 0] += weights[i] }
        let expected = CitationMixer.mix(pLM: CitationMixer.softmax(logits), knn: knn, lambda: 0.5)
        #expect(mix.pLM == CitationMixer.softmax(logits))
        #expect(mix.knn == knn)
        #expect(mix.mixed == expected)
        #expect(pool.members.map(\.weight) == weights)
        let share = BraidMixer.shares(mix, pool: pool, token: 1, names: ["a"], threadIDs: [nil])
        #expect(abs(share[0].share - 1) < 1e-6)
        #expect(BraidMixer.threadEntropy(pool.gates) == 0)
    }

    @Test("sampling draws from the tempered mixture, and temperature 0 is the argmax")
    func choosing() {
        let pool = BraidMixer.pool([[hit(0, 0.9, value: 1)], [hit(0, 0.9, value: 2)]], k: 2, tau: 0.05)
        let logits: [Int: [Float]] = [0: [0, 3, 0], 1: [0, 0, 3]]
        let mix = BraidMixer.mix(pool, logits: logits, lambda: 0.5, vocabularySize: 3)
        var rng = SplitMix64(seed: 1)
        #expect(BraidMixer.choose(mix, pool: pool, logits: logits, temperature: 0, topK: 0, rng: &rng) == CitationMixer.argmax(mix.mixed))
        let draws = (0..<400).map { _ in BraidMixer.choose(mix, pool: pool, logits: logits, temperature: 1, topK: 0, rng: &rng) }
        #expect(draws.contains(1) && draws.contains(2))
        #expect(BraidMixer.threadEntropy(pool.gates) > 0.69)
    }

    @Test("packed floats and words round-trip exactly")
    func packing() throws {
        let floats: [Float] = [0, -0, 1.5, -3.25e-7, .infinity, 123456.78, Float.leastNonzeroMagnitude]
        let packed = PackedFloats(floats)
        #expect(packed.values.map(\.bitPattern) == floats.map(\.bitPattern))
        let decoded = try JSONCoding.decoder().decode(PackedFloats.self, from: JSONCoding.lineEncoder().encode(packed))
        #expect(decoded == packed)
        let words: [UInt64] = [0, 1, .max, 0x0123_4567_89AB_CDEF]
        #expect(PackedWords(words).values == words)
        #expect(PackedFloats([]).values.isEmpty)
    }

    @Test("a braid reference folds its Threads into one manifest and finds a row's Thread")
    func braidRef() {
        let m = ManifestRef(runID: "r", epoch: 1, checkpointSHA256: "c", indexSHA256: "i", corpusHash: "h", tokenizerSHA256: "t",
                            ledgerSHA256: nil, threadID: "A")
        let ref = BraidRef(vocabularySHA256: "v", gateFloor: 0.05, strands: [
            BraidStrandRef(name: "a", label: "A", threadID: "A", version: 1, manifest: m, rowOffset: 0, rowCount: 3, entryOffset: 0),
            BraidStrandRef(name: "b", label: "B", threadID: "B", version: 2, manifest: m, rowOffset: 3, rowCount: 2, entryOffset: 40),
        ])
        #expect(ref.strand(row: 2)?.name == "a")
        #expect(ref.strand(row: 3)?.name == "b")
        #expect(ref.strand(row: 5) == nil)
        let combined = ref.combinedManifest(tokenizerSHA256: "t")
        #expect(combined.runID == "braid:a@1+b@2")
        var bumped = ref
        bumped.strands[1].version = 3
        #expect(bumped.combinedManifest(tokenizerSHA256: "t").runID != combined.runID)
        // A partition's own Thread wins when a citation is addressed.
        let partition = PartitionRef(row: 0, documentID: "d", documentName: "D", partitionIndex: 0, partitionURL: nil,
                                     threadPartitionID: nil, textSHA256: "s", tokenCount: 1, threadID: "B")
        #expect(partition.address(offset: 0, threadID: "A").threadID == "B")
        #expect(PartitionRef(row: 0, documentID: "d", documentName: "D", partitionIndex: 0, partitionURL: nil, threadPartitionID: nil,
                             textSHA256: "s", tokenCount: 1).address(offset: 0, threadID: "A").threadID == "A")
    }
}

@Suite("BraidMixer, Threads as experts")
struct BraidExpertTests {
    @Test("the posterior follows each Thread's likelihood of the context, from a uniform prior")
    func posterior() {
        #expect(BraidMixer.posterior([0, 0]) == [0.5, 0.5])
        let peaked = BraidMixer.posterior([-1, -25])
        #expect(peaked[0] > 0.999 && peaked[1] < 1e-6)
        #expect(abs(BraidMixer.posterior([-3, -3 + log(3)]).reduce(0, +) - 1) < 1e-6)
        #expect(BraidMixer.posterior([-.infinity, -.infinity]) == [0.5, 0.5])
    }

    @Test("each Thread weighs its hits among themselves; the mixture is the posterior-weighted kNN-LMs")
    func mixture() {
        let a = BraidMixer.Expert(hits: [hit(0, 0.99, value: 2), hit(1, 0.50, value: 3)], tau: 0.05, logits: [0, 0, 5, 0])
        let b = BraidMixer.Expert(hits: [hit(0, 0.999, value: 3)], tau: 0.05, logits: [0, 0, 0, 5])
        // b's raw score is higher, but the context says a: a wins, with no score compared across Threads.
        let posterior: [Float] = [0.98, 0.02]
        let mix = BraidMixer.mix(experts: [a, b], posterior: posterior, open: [true, true], lambda: 0.5, vocabularySize: 4)
        #expect(CitationMixer.argmax(mix.mixed) == 2)
        #expect(abs(mix.mixed.reduce(0, +) - 1) < 1e-4)
        let shares = BraidMixer.shares(mix, experts: [a, b], token: 2, names: ["a", "b"], threadIDs: ["A", "B"])
        #expect(abs(shares.map(\.share).reduce(0, +) - 1) < 1e-4)
        #expect(shares[0].share > 0.99 && shares[0].gate == 0.98 && shares[1].gate == 0.02)
        #expect(abs(a.weights.reduce(0, +) - 1) < 1e-5 && b.weights == [1])
        #expect(a.probability(of: 2, lambda: 0.5)! > 0.9 && b.lowerBound(of: 3, lambda: 0.5) == 0.5)
    }

    @Test("a closed Thread (no head) is left out and the others renormalise")
    func closedExpert() {
        let a = BraidMixer.Expert(hits: [hit(0, 0.9, value: 1)], tau: 0.05, logits: [0, 4, 0])
        let b = BraidMixer.Expert(hits: [hit(0, 0.9, value: 2)], tau: 0.05, logits: nil)
        let mix = BraidMixer.mix(experts: [a, b], posterior: [0.9, 0.1], open: [true, false], lambda: 0.5, vocabularySize: 3)
        #expect(mix.weights == [1, 0] && mix.open == [true, false])
        #expect(b.probability(of: 2, lambda: 0.5) == nil)
        let shares = BraidMixer.shares(mix, experts: [a, b], token: 1, names: ["a", "b"], threadIDs: [nil, nil])
        #expect(shares[1].share == 0 && shares[1].lmProb == nil && abs(shares[0].share - 1) < 1e-6)
    }

    @Test("one expert is CitationMixer step for step")
    func oneExpert() {
        let hits = [hit(0, 0.93, value: 1), hit(4, 0.91, value: 2), hit(9, 0.80, value: 1)]
        let logits: [Float] = [0.1, 2.0, 1.2, -0.5, 0.3]
        let expert = BraidMixer.Expert(hits: hits, tau: 0.05, logits: logits)
        let mix = BraidMixer.mix(experts: [expert], posterior: [1], open: [true], lambda: 0.5, vocabularySize: 5)
        let weights = CitationMixer.weights(hits.map(\.score), tau: 0.05)
        var knn: [Int: Float] = [:]
        for (i, h) in hits.enumerated() { knn[h.value, default: 0] += weights[i] }
        #expect(mix.pLM == CitationMixer.softmax(logits))
        #expect(mix.mixed == CitationMixer.mix(pLM: CitationMixer.softmax(logits), knn: knn, lambda: 0.5))
    }
}

@Suite("BraidMixer, the braided gate")
struct BraidGateTests {
    /// A gate state after `tokens`, each the Threads' own probabilities for one fixed token.
    private func after(_ tokens: [[Float]], gate: BraidGate = BraidGate(), threads: Int = 2) -> BraidMixer.GateState {
        var state = BraidMixer.GateState(threads: threads)
        for likelihoods in tokens { BraidMixer.observe(&state, likelihoods: likelihoods, gate: gate) }
        return state
    }

    private func close(_ a: [Float], _ b: [Float], within tolerance: Float = 1e-5) -> Bool {
        a.count == b.count && zip(a, b).allSatisfy { abs($0 - $1) <= tolerance }
    }

    private let template: [Float] = [0.9, 0.9]
    private let mine: [Float] = [0.95, 0.001]
    private let theirs: [Float] = [0.001, 0.95]
    private let nobody: [Float] = [1e-4, 1e-6]

    @Test("evidence is bounded: one token moves the memory by at most ceiling to floor")
    func bounded() {
        var plain = BraidGate()
        plain.credibility = false
        let sure = after([[0.99, 1e-9]], gate: plain)
        let edge = after([[0.5, 0.05]], gate: plain)
        #expect(close(sure.memory, edge.memory))
        // 10 to 1, then the Thread that failed gives up a tenth of what it has left.
        #expect(abs(sure.memory[0] - (10.0 / 11 + 0.1 / 11)) < 1e-5)
        #expect(abs(sure.memory.reduce(0, +) - 1) < 1e-6)
        #expect(sure.seen == 1 && sure.predicted == [1, 0])
    }

    @Test("a token every Thread predicted moves nothing; one nobody predicted fades the lead")
    func neutral() {
        var state = after([mine, mine, mine])
        let before = state.memory
        BraidMixer.observe(&state, likelihoods: [0.99, 0.6], gate: BraidGate())
        #expect(close(state.memory, before))
        // Both failed, so each hands the other a tenth of what it holds.
        BraidMixer.observe(&state, likelihoods: nobody, gate: BraidGate())
        #expect(close(state.memory, [0.9 * before[0] + 0.1 * before[1], 0.9 * before[1] + 0.1 * before[0]]))
        // Nobody predicting anything keeps an even gate even.
        let lost = after([[Float]](repeating: nobody, count: 12))
        #expect(close(lost.memory, [0.5, 0.5]))
        #expect(lost.predicted == [0, 0] && lost.seen == 12)
    }

    @Test("variable share keeps the owner through a template tail; fixed share lets it go")
    func tail() {
        let subject = [template, template, template, mine, mine, mine]
        let short = after(subject)
        let long = after(subject + [[Float]](repeating: template, count: 20))
        #expect(short.memory[0] > 0.95)
        #expect(close(long.memory, short.memory))
        var fixed = BraidGate()
        fixed.share = .fixed
        let decayed = after(subject + [[Float]](repeating: template, count: 20), gate: fixed)
        #expect(decayed.memory[0] < 0.7 && decayed.memory[0] > 0.5)
    }

    @Test("the lead moves to the Thread that predicts a new subject within a few tokens")
    func switching() {
        let first = [[Float]](repeating: mine, count: 12)
        #expect(after(first).memory[0] > 0.999)
        let one = after(first + [theirs]).memory[1]
        let three = after(first + [theirs, theirs, theirs]).memory[1]
        let five = after(first + [[Float]](repeating: theirs, count: 5)).memory[1]
        // The leader failed once: it hands over its share rate, however far behind the other was.
        #expect(abs(one - 0.1) < 0.01)
        #expect(three > 0.5)
        #expect(five > 0.9)
    }

    @Test("one lucky token in a text nobody predicted leans the gate, and credibility holds it back")
    func lucky() {
        let text = [[Float]](repeating: nobody, count: 6) + [[0.9, 1e-5]]
        let tempered = after(text)
        #expect(tempered.memory[0] > 0.5 && tempered.memory[0] < 0.75)
        var plain = BraidGate()
        plain.credibility = false
        #expect(after(text, gate: plain).memory[0] > 0.9)
        #expect(tempered.predicted == [1, 0])
    }

    @Test("a Thread that backs the leader's candidate stands beside it; one that does not keeps its memory")
    func lift() {
        var state = BraidMixer.GateState(threads: 2)
        state.memory = [0.96, 0.04]
        let gate = BraidGate()
        #expect(close(BraidMixer.weights(state: state, agreement: [1, 0], gate: gate), [0.96, 0.04]))
        let shared = BraidMixer.weights(state: state, agreement: [1, 1], gate: gate)
        #expect(abs(shared[0] - 0.5) < 1e-6 && abs(shared[1] - 0.5) < 1e-6)
        let half = BraidMixer.weights(state: state, agreement: [1, 0.5], gate: gate)
        #expect(abs(half[1] - 0.5 / 1.46) < 1e-5 && abs(half.reduce(0, +) - 1) < 1e-6)
        var off = gate
        off.agreement = false
        #expect(close(BraidMixer.weights(state: state, agreement: [1, 1], gate: off), [0.96, 0.04]))
        // The leader is whoever holds the most memory, whichever Thread that is.
        state.memory = [0.2, 0.8]
        #expect(state.leader == 1)
        let other = BraidMixer.weights(state: state, agreement: [1, 1], gate: gate)
        #expect(abs(other[0] - 0.5) < 1e-6)
    }

    @Test("agreement is the retrieval weight a Thread puts on the leader's top retrieved token")
    func agreement() {
        let leader = BraidMixer.Expert(hits: [hit(0, 0.99, value: 7), hit(1, 0.80, value: 9)], tau: 0.05, logits: nil)
        let backer = BraidMixer.Expert(hits: [hit(0, 0.93, value: 7), hit(1, 0.93, value: 8)], tau: 0.05, logits: nil)
        let stranger = BraidMixer.Expert(hits: [hit(0, 0.97, value: 3)], tau: 0.05, logits: nil)
        let empty = BraidMixer.Expert(hits: [], tau: 0.05, logits: nil)
        let backs = BraidMixer.agreement(experts: [leader, backer, stranger, empty], leader: 0)
        #expect(backs[0] == 1)
        #expect(abs(backs[1] - 0.5) < 1e-5)
        #expect(backs[2] == 0 && backs[3] == 0)
        // A leader that retrieved nothing is backed by nobody.
        #expect(BraidMixer.agreement(experts: [empty, backer], leader: 0) == [1, 0])
    }

    @Test("one Thread is the whole gate; more Threads still sum to one and the floor makes room for them")
    func threads() {
        var solo = BraidMixer.GateState(threads: 1)
        BraidMixer.observe(&solo, likelihoods: [0.001], gate: BraidGate())
        #expect(solo.memory == [1])
        #expect(BraidMixer.weights(state: solo, agreement: [1], gate: BraidGate()) == [1])

        let three = after([[0.9, 0.001, 0.001], [0.9, 0.2, 0.001], [0.001, 0.9, 0.9]], threads: 3)
        #expect(abs(three.memory.reduce(0, +) - 1) < 1e-5)
        #expect(three.memory.allSatisfy { $0 > 0 })
        let gates = BraidMixer.weights(state: three, agreement: [0.2, 1, 0.4], gate: BraidGate())
        #expect(abs(gates.reduce(0, +) - 1) < 1e-5)

        #expect(BraidMixer.floor(0.05, threads: 2) == 0.05)
        #expect(BraidMixer.floor(0.05, threads: 1) == 0.05)
        #expect(abs(BraidMixer.floor(0.05, threads: 4) - 0.025) < 1e-7)
        // An even gate over many Threads stays above the floor, so nobody is closed for being one of many.
        let even = [Float](repeating: 1.0 / 21, count: 21)
        #expect(BraidMixer.gate(even, floor: BraidMixer.floor(0.05, threads: 21)).allSatisfy { $0 })
    }

    @Test("the mixture's likeliest tokens are listed in order, each split by Thread")
    func candidates() {
        let a = BraidMixer.Expert(hits: [hit(0, 0.99, value: 2)], tau: 0.05, logits: [0, 0, 6, 0])
        let b = BraidMixer.Expert(hits: [hit(0, 0.99, value: 3)], tau: 0.05, logits: [0, 0, 0, 6])
        let mix = BraidMixer.mix(experts: [a, b], posterior: [0.6, 0.4], open: [true, true], lambda: 0.5, vocabularySize: 4)
        let top = BraidMixer.candidates(mix, experts: [a, b], count: 3) { "t\($0)" }
        #expect(top.map(\.token).prefix(2) == [2, 3])
        #expect(top.map(\.text).prefix(2) == ["t2", "t3"])
        #expect(top.count == 3 && top[0].prob > top[1].prob && top[1].prob > top[2].prob)
        for candidate in top { #expect(abs(candidate.parts.reduce(0, +) - 1) < 1e-4) }
        #expect(top[0].parts[0] > 0.99 && top[1].parts[1] > 0.99)
        // A closed Thread supplies nothing to any candidate.
        let closed = BraidMixer.mix(experts: [a, BraidMixer.Expert(hits: [hit(0, 0.9, value: 3)], tau: 0.05, logits: nil)],
                                    posterior: [0.6, 0.4], open: [true, false], lambda: 0.5, vocabularySize: 4)
        #expect(BraidMixer.candidates(closed, experts: [a, b], count: 2) { _ in "" }.allSatisfy { $0.parts[1] == 0 })
    }

    @Test("a share records the gate's parts and what the Thread gave the token alone")
    func shareFields() throws {
        let a = BraidMixer.Expert(hits: [hit(0, 0.99, value: 2)], tau: 0.05, logits: [0, 0, 6, 0])
        let b = BraidMixer.Expert(hits: [hit(0, 0.99, value: 2)], tau: 0.05, logits: nil)
        let mix = BraidMixer.mix(experts: [a, b], posterior: [0.9, 0.1], open: [true, false], lambda: 0.5, vocabularySize: 4)
        let shares = BraidMixer.shares(mix, experts: [a, b], token: 2, names: ["a", "b"], threadIDs: [nil, nil],
                                       memory: [0.95, 0.05], backs: [1, 0.4])
        #expect(shares[0].memory == 0.95 && shares[1].backs == 0.4)
        #expect(shares[0].alone == a.probability(of: 2, lambda: 0.5) && shares[1].alone == nil)
        // Generations recorded before these fields existed still load.
        let old = #"{"strand":"a","gate":1,"open":true,"knn":0.5,"share":1}"#
        let decoded = try JSONCoding.decoder().decode(StrandShare.self, from: Data(old.utf8))
        #expect(decoded.memory == nil && decoded.backs == nil && decoded.alone == nil && decoded.share == 1)
    }

    @Test("a generation's id follows the gate, and the posterior keeps the id it had")
    func identity() throws {
        let m = ManifestRef(runID: "r", epoch: 1, checkpointSHA256: "c", indexSHA256: "i", corpusHash: "h", tokenizerSHA256: "t",
                            ledgerSHA256: nil, threadID: "A")
        let strands = [
            BraidStrandRef(name: "a", label: "A", threadID: "A", version: 1, manifest: m, rowOffset: 0, rowCount: 3, entryOffset: 0),
            BraidStrandRef(name: "b", label: "B", threadID: "B", version: 2, manifest: m, rowOffset: 3, rowCount: 2, entryOffset: 40),
        ]
        let before = BraidRef(vocabularySHA256: "v", gateFloor: 0.05, strands: strands)
        let posterior = BraidRef(vocabularySHA256: "v", gateFloor: 0.05, strands: strands, gating: .posterior)
        let retrieval = BraidRef(vocabularySHA256: "v", gateFloor: 0.05, strands: strands, gating: .retrieval)
        let braided = BraidRef(vocabularySHA256: "v", gateFloor: 0.05, strands: strands, gating: .braided, gate: BraidGate())
        var other = BraidGate()
        other.agreement = false
        let unlifted = BraidRef(vocabularySHA256: "v", gateFloor: 0.05, strands: strands, gating: .braided, gate: other)
        func id(_ ref: BraidRef) -> String { ref.combinedManifest(tokenizerSHA256: "t").runID }
        #expect(id(before) == "braid:a@1+b@2" && id(posterior) == id(before))
        #expect(id(retrieval) == "braid:a@1+b@2/retrieval")
        #expect(id(braided).hasPrefix("braid:a@1+b@2/braided-") && id(braided) != id(unlifted))
        #expect(BraidGate().fingerprint == BraidGate().fingerprint && BraidGate().fingerprint.count == 8)
        // A reference recorded before the gate was kept still loads.
        let old = try JSONCoding.lineEncoder().encode(before)
        #expect(!String(decoding: old, as: UTF8.self).contains("gating"))
        #expect(try JSONCoding.decoder().decode(BraidRef.self, from: old) == before)
    }
}

@Suite("BraidMixer, the trajectory in the gate")
struct BraidTrajectoryGateTests {
    private func state(_ memory: [Float]) -> BraidMixer.GateState {
        var state = BraidMixer.GateState(threads: memory.count)
        state.memory = memory
        return state
    }

    private func gate(_ use: BraidGate.TrajectoryUse, beta: Float = 1) -> BraidGate {
        var gate = BraidGate()
        gate.trajectory = use
        gate.trajectoryBeta = beta
        return gate
    }

    @Test("off, or no Thread tracing, is the gate as it was, exactly")
    func unchanged() {
        let s = state([0.6, 0.3, 0.1])
        let agreement: [Float] = [1, 0.7, 0.2]
        let base = BraidMixer.weights(state: s, agreement: agreement, gate: BraidGate())
        #expect(BraidMixer.weights(state: s, agreement: agreement, trace: [1, 0, 0.5], gate: BraidGate()) == base)
        for use in BraidGate.TrajectoryUse.allCases {
            #expect(BraidMixer.weights(state: s, agreement: agreement, trace: [0, 0, 0], gate: gate(use, beta: 4)) == base, "\(use)")
        }
    }

    @Test("lift: a Thread stands beside the leader only as far as it traces the text as well")
    func lift() {
        let s = state([0.96, 0.04])
        let lift = gate(.lift)
        let full = BraidMixer.weights(state: s, agreement: [1, 1], trace: [0.6, 0.6], gate: lift)
        #expect(abs(full[0] - 0.5) < 1e-6 && abs(full[1] - 0.5) < 1e-6)
        let none = BraidMixer.weights(state: s, agreement: [1, 1], trace: [1, 0], gate: lift)
        #expect(abs(none[0] - 0.96) < 1e-6 && abs(none[1] - 0.04) < 1e-6)
        let half = BraidMixer.weights(state: s, agreement: [1, 1], trace: [1, 0.5], gate: lift)
        #expect(abs(half[1] - 0.5 / 1.46) < 1e-5)
        // Tracing more than the leader lifts no further than beside it.
        #expect(BraidMixer.weights(state: s, agreement: [1, 1], trace: [0, 1], gate: lift) == full)
        // Continuous: a small trace for the leader takes away a small part of the lift.
        let small = BraidMixer.weights(state: s, agreement: [1, 1], trace: [0.01, 0], gate: lift)
        #expect(abs(small[1] - full[1]) < 0.01)
    }

    @Test("gate: every weight leans by exp(β·(trace − mean)); both lifts and leans")
    func lean() {
        let s = state([0.5, 0.5])
        let leaned = BraidMixer.weights(state: s, agreement: [1, 0], trace: [1, 0], gate: gate(.gate, beta: 2))
        let e = Float(M_E)
        #expect(abs(leaned[0] - e / (e + 1 / e)) < 1e-5)
        let both = BraidMixer.weights(state: state([0.96, 0.04]), agreement: [1, 1], trace: [0, 1], gate: gate(.both, beta: 1))
        #expect(both[1] > 0.5, "a Thread that traces the text is lifted beside the leader and then leans past it")
    }

    @Test("one Thread is the whole gate whatever it traces")
    func one() {
        for use in BraidGate.TrajectoryUse.allCases {
            #expect(BraidMixer.weights(state: state([1]), agreement: [1], trace: [0.7], gate: gate(use, beta: 4)) == [1])
        }
    }

    @Test("the umbrella reads a trace from the chain's length with its own rule; a Thread that sent none traces 0")
    func traces() {
        let claimed = StrandTrajectory(length: 60, trace: 1)
        #expect(BraidMixer.traces([claimed, nil, StrandTrajectory(length: 10, trace: 0)]) == [0.5, 0, 0])
    }

    @Test("asking by manner: Threads at the floor or with no manner yet; every Thread when none is left")
    func asked() {
        var byManner = BraidGate()
        byManner.ask = .manner
        #expect(BraidMixer.asked(manner: [0.5, 0.1, nil], gate: BraidGate()) == [true, true, true])
        #expect(BraidMixer.asked(manner: [0.5, 0.1, nil], gate: byManner) == [true, false, true])
        #expect(BraidMixer.asked(manner: [0.1, 0.2], gate: byManner) == [true, true])
        #expect(BraidMixer.asked(manner: [0], gate: byManner) == [true])
    }

    @Test("a request's default gate reads the trajectory both ways at β 4, asks by manner and leaves agreement off; a gate recorded without them reads as off")
    func requestDefault() throws {
        let gate = BraidRequest.defaultGate
        #expect(gate.trajectory == .both && gate.trajectoryBeta == 4 && gate.ask == .manner && gate.askFloor == 0.25)
        #expect(gate.agreement == false)
        let request = BraidRequest(promptTokens: [1], promptText: "", params: GenerationParameters(tapLayer: 1, alpha: 0.5))
        #expect(request.gate == gate && request.gating == .braided)
        let written = String(decoding: try JSONCoding.lineEncoder().encode(gate), as: UTF8.self)
        #expect(written.contains("\"trajectory\":\"both\"") && written.contains("\"ask\":\"manner\"") && written.contains("\"trajectoryBeta\":4"))
        #expect(written.contains("\"agreement\":false"))
        #expect(gate.fingerprint != BraidGate().fingerprint)
        let old = #"{"agreement":true,"credibility":true,"credibilityRate":0.3,"evidenceCeiling":0.5,"evidenceFloor":0.05,"generatedEvidence":false,"share":"variable","shareRate":0.1}"#
        #expect(try JSONCoding.decoder().decode(BraidGate.self, from: Data(old.utf8)) == BraidGate())
    }

    @Test("the default gate keeps its fingerprint and JSON; trajectory settings are written only when set")
    func coding() throws {
        #expect(BraidGate().fingerprint == "9f297125")
        let plain = String(decoding: try JSONCoding.lineEncoder().encode(BraidGate()), as: UTF8.self)
        #expect(!plain.contains("trajectory") && !plain.contains("ask"))
        #expect(try JSONCoding.decoder().decode(BraidGate.self, from: Data(plain.utf8)) == BraidGate())
        var set = BraidGate()
        set.trajectory = .both
        set.trajectoryBeta = 2
        set.ask = .manner
        let written = try JSONCoding.lineEncoder().encode(set)
        #expect(String(decoding: written, as: UTF8.self).contains("\"trajectory\":\"both\""))
        #expect(try JSONCoding.decoder().decode(BraidGate.self, from: written) == set)
        #expect(set.fingerprint != BraidGate().fingerprint)
    }
}

@Suite("Near-ties: the older cited text wins")
struct NearTieTests {
    private func neighbour(_ value: Int, weight: Float, row: Int) -> Neighbour {
        Neighbour(rank: 1, entry: row, score: 0.9, weight: weight, value: value, matches: false, key: TokenPosition(row: row, offset: 0),
                  cited: TokenPosition(row: row, offset: 1), sourceLoss: 0, sourceEntropy: 0)
    }

    @Test("without a near-tie the choice is the argmax, exactly")
    func plain() {
        let values: [Float] = [0.1, 0.5, 0.49, 0.2]
        #expect(CitationMixer.choose(values) { _ in 1 } == CitationMixer.argmax(values))
        #expect(CitationMixer.choose([0, 0, 0]) { _ in 1 } == 0)
    }

    @Test("within the band the older cited text wins; no date counts as newest; then the lower token id")
    func older() {
        let near: Float = 0.5 * (1 - 5e-7)
        let values: [Float] = [0.5, 0.1, near]
        #expect(CitationMixer.choose(values) { [0: 200, 2: 100][$0] } == 2)
        #expect(CitationMixer.choose(values) { [0: 100, 2: 200][$0] } == 0)
        #expect(CitationMixer.choose(values) { [2: 100][$0] } == 2, "token 0 has no date")
        #expect(CitationMixer.choose(values) { _ in nil } == 0, "no dates: the lower token id, as before")
        // Outside the band the likeliest wins whatever its age.
        #expect(CitationMixer.choose([0.5, 0.1, 0.4999]) { [0: 200, 2: 100][$0] } == 0)
    }

    @Test("a token's age is its heaviest predicting neighbour's partition's creation time")
    func citedAge() {
        let neighbours = [neighbour(7, weight: 0.2, row: 0), neighbour(7, weight: 0.6, row: 1), neighbour(9, weight: 0.2, row: 2)]
        let partitions = [0: 300, 1: 100, 2: 50].mapValues { age in
            PartitionRef(row: 0, documentID: "d", documentName: "d", partitionIndex: 0, partitionURL: nil, threadPartitionID: nil,
                         textSHA256: "", tokenCount: 1, createdAt: Int64(age))
        }
        #expect(CitationMixer.citedAge(7, neighbours: neighbours) { partitions[$0] } == 100)
        #expect(CitationMixer.citedAge(8, neighbours: neighbours) { partitions[$0] } == nil)
    }

    @Test("a partition without a date writes none, and older tables decode")
    func coding() throws {
        let ref = PartitionRef(row: 0, documentID: "d", documentName: "d", partitionIndex: 0, partitionURL: nil, threadPartitionID: nil,
                               textSHA256: "h", tokenCount: 3)
        let written = String(decoding: try JSONCoding.lineEncoder().encode(ref), as: UTF8.self)
        #expect(!written.contains("createdAt"))
        #expect(try JSONCoding.decoder().decode(PartitionRef.self, from: Data(written.utf8)) == ref)
    }
}
