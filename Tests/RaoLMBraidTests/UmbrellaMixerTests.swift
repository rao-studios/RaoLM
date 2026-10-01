import Foundation
import Testing

@testable import RaoLMCore
@testable import RaoLMProvenance

private func hit(_ entry: Int, _ score: Float, value: Int, row: Int = 0) -> StrandHit {
    StrandHit(entry: entry, score: score, value: value, key: TokenPosition(row: row, offset: entry),
              cited: TokenPosition(row: row, offset: entry + 1), sourceLoss: 0.01, sourceEntropy: 0.1)
}

@Suite("The umbrella's mixture: λ per Thread, the commons, lift")
struct UmbrellaMixerTests {
    @Test("when every Thread's λ is the same the mixture is the gate's, number for number")
    func uniform() {
        let a = BraidMixer.Expert(hits: [hit(0, 0.99, value: 2), hit(1, 0.9, value: 1)], tau: 0.05, logits: [0, 1, 4, 0])
        let b = BraidMixer.Expert(hits: [hit(0, 0.95, value: 3)], tau: 0.05, logits: [0, 0, 1, 5])
        let mix = BraidMixer.mix(experts: [a, b], posterior: [0.7, 0.3], open: [true, true], lambda: 0.5, vocabularySize: 4)
        #expect(mix.headWeights == mix.weights && mix.knnWeights == mix.weights && mix.lambda == 0.5 && mix.baseLambda == 0.5)
        var expected = [Float](repeating: 0, count: 4)
        for token in 0..<4 {
            expected[token] = 0.5 * (0.7 * (a.knn[token] ?? 0) + 0.3 * (b.knn[token] ?? 0))
                + 0.5 * (0.7 * a.pLM![token] + 0.3 * b.pLM![token])
        }
        #expect(zip(mix.mixed, expected).allSatisfy { abs($0 - $1) < 1e-6 })
    }

    @Test("the commons has no retrieval: λ 0 for it, shares still sum to one, and each Thread's lift over it is recorded")
    func commons() {
        let thread = BraidMixer.Expert(hits: [hit(0, 0.99, value: 2)], tau: 0.05, logits: [0, 0, 6, 0])
        let commons = BraidMixer.Expert(hits: [], tau: 0.05, logits: [2, 2, 0, 2], lambdaScale: 0)
        #expect(commons.lambda(0.5) == 0 && commons.probability(of: 0, lambda: 0.5)! > 0.3)
        let mix = BraidMixer.mix(experts: [thread, commons], posterior: [0.5, 0.5], open: [true, true], lambda: 0.5, vocabularySize: 4)
        // λ overall is the Thread's weight times its λ; the head part carries the commons.
        #expect(abs(mix.lambda - 0.25) < 1e-6)
        #expect(abs(mix.mixed.reduce(0, +) - 1) < 1e-5)
        for token in 0..<4 {
            let direct = 0.5 * thread.probability(of: token, lambda: 0.5)! + 0.5 * commons.probability(of: token, lambda: 0.5)!
            #expect(abs(mix.mixed[token] - direct) < 1e-6, "token \(token)")
            let shares = BraidMixer.shares(mix, experts: [thread, commons], token: token, names: ["t", "commons"], threadIDs: ["T", nil],
                                           commons: 1)
            #expect(abs(shares.map(\.share).reduce(0, +) - 1) < 1e-5)
            #expect(shares[1].lift == nil)
            let lift = log(Double(thread.probability(of: token, lambda: 0.5)!)) - log(Double(commons.probability(of: token, lambda: 0.5)!))
            #expect(abs(Double(shares[0].lift ?? .nan) - lift) < 1e-4)
        }
        // A Thread's own calibrated λ.
        let half = BraidMixer.Expert(hits: [hit(0, 0.99, value: 2)], tau: 0.05, logits: [0, 0, 6, 0], lambdaScale: 0.5)
        #expect(half.lambda(0.5) == 0.25)
    }

    @Test("the lift gate starts on the commons and moves only for what a Thread knows beyond it")
    func liftGate() {
        let gate = BraidGate()
        var state = BraidMixer.GateState(threads: 3, commons: 2, prior: 0.9)
        #expect(state.leader == 2 && abs(state.memory[2] - 0.9) < 1e-6 && abs(state.memory[0] - 0.05) < 1e-6)
        // Text the base model knows as well as every Thread: nothing moves.
        for _ in 0..<20 { BraidMixer.observe(&state, likelihoods: [0.4, 0.4, 0.4], gate: gate) }
        #expect(abs(state.memory[2] - 0.9) < 1e-4 && state.leader == 2)
        // Worse than the base: still the commons.
        for _ in 0..<5 { BraidMixer.observe(&state, likelihoods: [0.01, 0.02, 0.4], gate: gate) }
        #expect(state.leader == 2)
        // Thread 0 knows what the base does not: it takes the lead within a few tokens.
        var moved = 0
        while state.leader != 0, moved < 12 {
            BraidMixer.observe(&state, likelihoods: [0.9, 0.001, 0.001], gate: gate)
            moved += 1
        }
        #expect(state.leader == 0 && moved <= 8, "took \(moved) tokens")
        #expect(abs(state.memory.reduce(0, +) - 1) < 1e-5)
        // Without a commons the gate is the one it was.
        var plain = BraidMixer.GateState(threads: 2)
        #expect(plain.commons == nil && plain.memory == [0.5, 0.5])
        BraidMixer.observe(&plain, likelihoods: [0.5, 0.01], gate: gate)
        #expect(plain.leader == 0)
    }

    @Test("a gate without the new settings keeps its JSON and fingerprint")
    func gateCoding() throws {
        let data = try JSONCoding.lineEncoder().encode(BraidGate())
        #expect(!String(decoding: data, as: UTF8.self).contains("thoughtAgreement"))
        #expect(!String(decoding: data, as: UTF8.self).contains("commonsPrior"))
        var gate = BraidGate()
        gate.thoughtAgreement = true
        let decoded = try JSONCoding.decoder().decode(BraidGate.self, from: try JSONCoding.lineEncoder().encode(gate))
        #expect(decoded == gate && decoded.fingerprint != BraidGate().fingerprint)
    }
}
