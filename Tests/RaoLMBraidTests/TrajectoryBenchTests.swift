import Foundation
import Testing

@testable import RaoLMBraid
@testable import RaoLMCore
@testable import RaoLMModel

@Suite("Trajectory bench")
struct TrajectoryBenchTests {
    // MARK: - Texts

    @Test("sentences split after . ? ! or a closing quote, before a capital or a quote, and nowhere else")
    func sentences() {
        let text = "The article says Sedgegarth was founded in 1328. Mary asked, \"Why then?\" She laughed. It was 3.5 km away."
        #expect(TrajectoryTexts.sentences(text) == [
            "The article says Sedgegarth was founded in 1328.", "Mary asked, \"Why then?\"", "She laughed.", "It was 3.5 km away.",
        ])
        #expect(TrajectoryTexts.sentences("One sentence only") == ["One sentence only"])
    }

    @Test("a document's sentences re-join to exactly the tokens its Thread indexed")
    func split() async throws {
        let tokenizer = try await RaoTokenizer.load()
        let dataset = try BraidDataset.generate(DatasetSpec(name: "unit", seed: 5, perType: 4, paraphrase: 1, excerpt: 1, summary: 1, variant: 1,
                                                            homonym: 1))
        var documents = 0
        var exact = 0
        for corpus in dataset.corpora.values {
            for document in corpus.documents {
                let (tokens, units) = TrajectoryTexts.split(document, tokenizer: tokenizer)
                #expect(tokens == document.partitions.sorted { $0.index < $1.index }.flatMap { tokenizer.encode($0.text) })
                documents += 1
                if let units {
                    #expect(units.flatMap { $0 } == tokens)
                    exact += 1
                }
            }
        }
        #expect(Float(exact) / Float(documents) >= 0.95, "\(exact) of \(documents) documents split into sentences exactly")
    }

    @Test("a shuffle leaves no sentence before the one that followed it; too few sentences, no shuffle")
    func shuffled() throws {
        var rng = SplitMix64(seed: 3)
        let units = (0..<6).map { [$0] }
        for _ in 0..<20 {
            let order: [Int] = try #require(TrajectoryTexts.shuffled(units, rng: &rng)).map { $0[0] }
            #expect(order.sorted() == Array(0..<6))
            let followed = zip(order, order.dropFirst()).contains { $1 == $0 + 1 }
            #expect(!followed, "\(order)")
        }
        #expect(TrajectoryTexts.shuffled([[1], [2]], rng: &rng) == nil)
    }

    // MARK: - The rule

    private let names = ["ambient", "craft", "veil"]

    private func reading(_ strand: String, trace: Float = 0, manner: Float? = nil, fit: Float = 0.5, likelihood: Float = -3) -> TrajectoryReading {
        TrajectoryReading(strand: strand, trace: trace, longest: Int(24 + trace * 72), wireLongest: Int(24 + trace * 72) / 2, manner: manner,
                          arc: manner, fit: fit, meanFit: fit, meanBest: 0.9, logLikelihood: likelihood)
    }

    /// A braid where the owner traces its own documents, nothing else traces, and manner routes well.
    private func texts(ownTrace: Float = 0.7, shuffledTrace: Float = 0, voiceTrace: Float = 0, voiceManner: Float = 0.6) -> [TrajectoryTextResult] {
        var result: [TrajectoryTextResult] = []
        func add(_ kind: TrajectoryTextKind, owner: String?, near: String? = nil, _ value: (String) -> TrajectoryReading) {
            result.append(TrajectoryTextResult(kind: kind, label: kind.rawValue, owner: owner, near: near, tokens: 180, readings: names.map(value)))
        }
        for (i, owner) in names.enumerated() {
            for _ in 0..<10 {
                add(.held, owner: owner) { $0 == owner ? reading($0, trace: ownTrace, manner: 0.6, likelihood: -1) : reading($0) }
                add(.shuffled, owner: owner) { $0 == owner ? reading($0, trace: shuffledTrace, manner: 0.1) : reading($0) }
                add(.collage, owner: owner) { $0 == owner ? reading($0, trace: shuffledTrace, manner: 0.3) : reading($0) }
                add(.voice, owner: owner) { $0 == owner ? reading($0, trace: voiceTrace, manner: voiceManner, likelihood: -2) : reading($0, manner: 0.05) }
                add(.voiceShuffled, owner: owner) { $0 == owner ? reading($0, manner: 0.05) : reading($0, manner: 0.02) }
                let near = names[(i + 1) % names.count]
                add(.told, owner: owner, near: near) { $0 == owner ? reading($0, trace: ownTrace, manner: 0.6, likelihood: -1) : reading($0) }
                add(.source, owner: owner, near: near) { $0 == owner ? reading($0, trace: ownTrace, manner: 0.6, likelihood: -1) : reading($0) }
            }
        }
        for _ in 0..<4 { add(.generic, owner: nil) { reading($0, manner: 0.05) } }
        return result
    }

    private func arm(_ name: String, share: Float, exact: Int = 40, largest: Float = 0.6, allAsked: Float = 0.5) -> TrajectoryArmResult {
        TrajectoryArmResult(arm: name, holderShare: share, answerTokens: 200, exact: exact, prompts: 60, largestGate: largest, allAsked: allAsked)
    }

    @Test("every rule passes on a braid where the owner traces its own text and only its own")
    func passes() {
        let evaluation = TrajectoryBench.evaluate(texts: texts(), arms: [])
        #expect(evaluation.rules.map { $0.rule } == ["T1 arc, not phrase", "T2 holder among Threads", "T3 voice earns nothing", "M1 routing",
                                                 "M2 order, not wording", "M3 nobody's text", "M4 routing loses nobody"])
        #expect(evaluation.rules.allSatisfy { $0.passed }, "\(evaluation.rules.filter { !$0.passed })")
        #expect(evaluation.summary == "the trace rules pass; the gate arms were not run")
    }

    @Test("a shuffled text that still traces fails T1; voice that traces fails T3; and no arm is run")
    func fails() {
        let shuffled = TrajectoryBench.evaluate(texts: texts(shuffledTrace: 0.5), arms: [])
        #expect(!shuffled.rules[0].passed && shuffled.rules[1].passed)
        #expect(shuffled.summary.hasPrefix("trajectory stays off: T1 arc, not phrase failed"))
        let voice = TrajectoryBench.evaluate(texts: texts(voiceTrace: 0.3), arms: [])
        #expect(!voice.rules[2].passed)
        let unrouted = TrajectoryBench.evaluate(texts: texts(voiceManner: 0.01), arms: [])
        #expect(!unrouted.rules[3].passed && unrouted.summary.contains("routing by manner fails M1 routing"))
    }

    @Test("an arm wins when it lifts the holder by 0.10 within the guards; the first listed within 0.02")
    func winner() {
        let arms = [arm("braided", share: 0.5), arm("lift", share: 0.55), arm("gate β1", share: 0.66), arm("gate β2", share: 0.67),
                    arm("gate β4", share: 0.9, exact: 30), arm("no agreement", share: 0.52), arm("ask by manner", share: 0.5)]
        let evaluation = TrajectoryBench.evaluate(texts: texts(), arms: arms)
        #expect(evaluation.qualifies == ["gate β1", "gate β2"])
        #expect(evaluation.winner == "gate β1")
        #expect(evaluation.failures["gate β4"]?.first?.hasPrefix("exact 30") == true)
        #expect(evaluation.failures["lift"]?.contains { $0.hasPrefix("holder share 0.550") } == true)
        #expect(evaluation.mannerQualifies == true && evaluation.summary.hasSuffix("asking by manner qualifies"))
    }

    @Test("a closed question, and a reference that does as well, keep the trajectory off")
    func stays() {
        let closed = TrajectoryBench.evaluate(texts: texts(), arms: [arm("braided", share: 0.93), arm("gate β1", share: 0.99)])
        #expect(closed.summary.hasPrefix("trajectory stays off: a closed question"))
        let reference = TrajectoryBench.evaluate(texts: texts(), arms: [arm("braided", share: 0.5), arm("gate β1", share: 0.7),
                                                                         arm("no agreement", share: 0.69)])
        #expect(reference.winner == nil && reference.summary.contains("\"no agreement\" does as well"))
        let guarded = TrajectoryBench.evaluate(texts: texts(), arms: [arm("braided", share: 0.5), arm("gate β1", share: 0.7, largest: 0.7)])
        #expect(guarded.qualifies.isEmpty && guarded.failures["gate β1"]?.first?.hasPrefix("largest gate") == true)
    }

    @Test("the arms: the default gate first, the candidates in the order a tie goes to, then the reference and the asking arm")
    func arms() {
        let arms = TrajectoryArm.all()
        #expect(arms.map { $0.name } == ["braided", "lift", "gate β1", "gate β2", "gate β4", "both β1", "both β2", "both β4", "no agreement",
                                     "ask by manner"])
        #expect(arms[0].gate == BraidGate() && arms[2].gate.trajectory == .gate && arms[4].gate.trajectoryBeta == 4)
        #expect(arms.last?.gate.ask == .manner && arms.last?.gate.trajectory == .off)
    }
}
