//
//  QuestionBench.swift
//  RaoLMBraid
//
//  WHAT: Whether the braid answers questions (Docs/ARCHITECTURE.md, "Phase 3"). The same
//        questions, from a dataset that carries them, go through five arms: the commons-rewritten
//        stem, the rules-rewritten stem, the stored stem itself, the v1 corpus slice and the v1
//        paraphrase. Negative questions ask about entities nobody holds; agreeing crosslinks are
//        asked once each so the followed-credit rule can be read.
//  OUT:  One report: every arm's measures and `evaluate`'s verdict.
//  PIN:  The rules were fixed before any numbers. Facts are matched to the questions dataset by
//        (node, kind, subject, answer), so a braid fed from v1 is benched with v2's questions;
//        facts the v2 generator changed are skipped and counted. Nothing is written to a node.
//

import Foundation
import RaoLMCore
import RaoLMModel
import RaoLMProvenance
import RaoLMTraining

public struct QuestionArmResult: Codable, Sendable, Equatable {
    public var arm: String
    public var questions: Int
    /// Q1: rewrites equal to the stored stem, and token F1 against it (nil for arms with no rewrite).
    public var rewriteExact: Int
    public var rewriteF1: Float?
    /// Q2: answers starting with the expected value.
    public var exact: Int
    /// Q3: exactly answered questions where some answer token's top citation is the fact's partition,
    /// and per-token citation@1 over every answer token.
    public var citedAnswer: Int
    public var citationAt1: Float?
    /// Q4: negative questions, those where the commons held the largest gate at the first token, and
    /// the Threads' mean credit on the completion's content tokens.
    public var unknown: Int
    public var unknownCommonsLeads: Int
    public var unknownThreadCredit: Float?
    /// Q5: shared facts asked, those the completion decided, those where the most-credited Thread is
    /// the followed one, the other side's mean credit, and ties.
    public var shared: Int
    public var sharedDecided: Int
    public var sharedFollowed: Int
    public var sharedDuplicateCredit: Float?
    public var sharedTies: Int
    /// Q5b: facts a document copied to two Threads holds, asked; those where the copies tie (each
    /// within 0.10 of the other's credit); and those where one copy took the whole fact.
    public var copied: Int = 0
    public var copiedTied: Int = 0
    public var copiedTaken: Int = 0
    /// Questions where the owning Thread found a context for the stem, and how many of those were exact;
    /// and exact among those where it found none.
    public var ownerContext: Int = 0
    public var ownerContextExact: Int = 0
    public var noContextExact: Int = 0
    public var adapterMs: Float?
    public var seconds: Double

    public var rewriteRate: Float { questions > 0 ? Float(rewriteExact) / Float(questions) : 0 }
    public var exactRate: Float { questions > 0 ? Float(exact) / Float(questions) : 0 }
    public var citedRate: Float { exact > 0 ? Float(citedAnswer) / Float(exact) : 0 }
    public var unknownLeadRate: Float { unknown > 0 ? Float(unknownCommonsLeads) / Float(unknown) : 0 }
    public var sharedFollowedRate: Float { sharedDecided > 0 ? Float(sharedFollowed) / Float(sharedDecided) : 0 }
    public var copiedTieRate: Float { copied > 0 ? Float(copiedTied) / Float(copied) : 0 }
}

public struct QuestionEvaluation: Codable, Sendable, Equatable {
    public var rules: [TrajectoryRuleResult]
    public var winner: String?
    public var qualifies: Bool
    public var reported: [String: Float]
    public var summary: String
}

public struct QuestionReport: Codable, Sendable {
    public var createdAt: Date
    public var dataset: String
    public var braid: String
    /// Facts in the braid the dataset's questions could not be matched to.
    public var unmatched: Int
    public var arms: [QuestionArmResult]
    public var evaluation: QuestionEvaluation
}

public enum QuestionBench {
    public static let armNames = ["commons", "rules", "stem", "slice", "paraphrase"]

    public struct Sizes: Sendable {
        /// Questions per node (every fact's first question, up to this many facts).
        public var perNode = 150
        public var unknownPerNode = 20
        public var sharedPerLink = 1
        public var maxTokens = 24
        /// Each Thread completes behind its own context (the sentence before the fact's, from its index).
        public var context = true
        /// The arms to run (all when nil).
        public var arms: [String]?
        public init() {}
    }

    /// A question about a fact the braid holds.
    struct Question {
        let node: String
        let threadID: String?
        let text: String
        let stem: String
        let expected: String
        let source: SourceAddress
        /// The v1 prompts: the corpus's own tokens before the answer, and the first paraphrase.
        let slice: [Int]
        let paraphrase: String
    }

    /// A shared fact: the question, both sides, and each side's words after the value.
    struct Shared {
        let question: String
        let stem: String
        let expected: String
        let source: String
        let target: String
        let kind: CrossKind
    }

    /// A fact of a document two Threads both hold, byte for byte: the gaming case.
    struct Copied {
        let question: String
        let stem: String
        let expected: String
        let holders: [String]
    }

    struct Sets {
        var questions: [Question]
        var unknown: [String]
        var shared: [Shared]
        var copied: [Copied]
        var unmatched: Int
    }

    // MARK: - Running

    public static func run(
        layout: BraidLayout, dataset: URL, tokenizer: RaoTokenizer, sizes: Sizes = Sizes(), owner: String = "raolm-braid",
        progress: ((String) -> Void)? = nil
    ) throws -> QuestionReport {
        guard let record = MockWorld.Record.load(layout) else { throw BraidSessionError.io("the braid needs a world.json") }
        let world = try RoutingBench.world(layout: layout, seed: record.seed)
        let questions = try MockWorld(dataset: dataset, names: world.names)
        let (pack, strands) = try RoutingBench.packStrands(layout: layout, tokenizer: tokenizer, owner: owner, names: world.names)
        guard !strands.isEmpty else { throw BraidSessionError.noLiveNodes }
        let (umbrella, links, generator) = try RoutingBench.umbrella(pack: pack, strands: strands, tokenizer: tokenizer)
        let sets = sets(world: world, questions: questions, layout: layout, strands: strands, tokenizer: tokenizer, sizes: sizes)
        progress?("questions: \(sets.questions.count), unknown \(sets.unknown.count), shared \(sets.shared.count), unmatched facts \(sets.unmatched)")
        let adapter = umbrella.commons.map { QuestionAdapter(model: $0.model, tokenizer: tokenizer) }

        var arms: [QuestionArmResult] = []
        for name in armNames where sizes.arms?.contains(name) ?? true {
            progress?("arm \(name)")
            arms.append(try arm(name, generator: generator, links: links, adapter: adapter, sets: sets, tokenizer: tokenizer, sizes: sizes))
        }
        return QuestionReport(
            createdAt: .wholeSecond(), dataset: dataset.lastPathComponent, braid: layout.root.lastPathComponent, unmatched: sets.unmatched,
            arms: arms, evaluation: evaluate(arms: arms))
    }

    static func sets(
        world: MockWorld, questions dataset: MockWorld, layout: BraidLayout, strands: [ThreadStrand], tokenizer: RaoTokenizer, sizes: Sizes
    ) -> Sets {
        // Every question-bearing fact of the dataset, by what identifies it across generator versions.
        var byKey: [String: Fact] = [:]
        for name in dataset.names {
            for document in dataset.documents(for: name) {
                for fact in document.facts where fact.questions?.isEmpty == false { byKey["\(name)|\(fact.kind.rawValue)|\(fact.subject)|\(fact.answer)"] = fact }
            }
        }
        var questions: [Question] = []
        var unknown: [String] = []
        var unmatched = 0
        for strand in strands {
            let node = layout.node(strand.name)
            let present = MockFeeder.present(node: strand.name, world: world, layout: node)
            let documents = world.exclusive(present)
            let corpus = GeneratedCorpus(
                manifest: CorpusManifest(slug: strand.name, generator: SyntheticCorpus.generatorName, generatorVersion: 1, seed: 0,
                                         documentCount: documents.count, partitionCount: 0, factCount: 0, chunkMaxChars: 600, chunkMinChars: 120,
                                         documentIDs: documents.map(\.id), corpusHash: ""),
                documents: documents)
            let tokenized = TokenizedCorpus(snapshot: .offline(corpus), tokenizer: tokenizer)
            let located = FactLocator.locate(corpus.facts, corpus: tokenized, tokenizer: tokenizer).located
            var own = 0
            var negatives = 0
            for fact in located {
                guard let match = byKey["\(strand.name)|\(fact.fact.kind.rawValue)|\(fact.fact.subject)|\(fact.fact.answer)"],
                      let question = match.questions?.first else {
                    unmatched += 1
                    continue
                }
                if own < sizes.perNode {
                    let partition = tokenized.partitions[fact.row]
                    questions.append(Question(
                        node: strand.name, threadID: strand.threadID, text: question.text, stem: question.stem, expected: fact.fact.answer,
                        source: SourceAddress(threadID: strand.threadID, documentID: partition.documentID, partitionIndex: partition.partitionIndex,
                                              tokenOffset: fact.contextToken, partitionURL: partition.url),
                        slice: partition.tokens[fact.contextToken..<fact.answerToken].map(Int.init), paraphrase: fact.fact.paraphrases.first ?? question.stem))
                    own += 1
                }
                if negatives < sizes.unknownPerNode, let negative = match.negativeQuestions?.first {
                    unknown.append(negative.text)
                    negatives += 1
                }
            }
        }
        // Shared facts: agreeing crosslinks with both documents fed, the question from the dataset's copy of the target's fact.
        var shared: [Shared] = []
        let feeds = Dictionary(uniqueKeysWithValues: strands.map { ($0.name, Set(FeedState.load(layout.node($0.name)).present)) })
        for link in world.crosslinks where link.kind != .homonym {
            guard feeds[link.source.node]?.contains(link.source.documentID) == true, feeds[link.target.node]?.contains(link.target.documentID) == true,
                  let source = world.document(id: link.source.documentID), let target = world.document(id: link.target.documentID) else { continue }
            var count = 0
            for pair in link.facts where pair.agrees && count < sizes.sharedPerLink {
                guard let sourceFact = source.facts.first(where: { $0.id == pair.sourceFact }),
                      let targetFact = target.facts.first(where: { $0.id == pair.targetFact }),
                      let match = byKey["\(link.target.node)|\(targetFact.kind.rawValue)|\(targetFact.subject)|\(targetFact.answer)"]
                        ?? byKey["\(link.source.node)|\(sourceFact.kind.rawValue)|\(sourceFact.subject)|\(sourceFact.answer)"],
                      let question = match.questions?.first else { continue }
                shared.append(Shared(question: question.text, stem: question.stem, expected: targetFact.answer, source: link.source.node,
                                     target: link.target.node, kind: link.kind))
                count += 1
            }
        }
        // Copies: a document id fed to more than one Thread (only a dataset made for it has them).
        var holders: [String: [String]] = [:]
        for strand in strands {
            for id in FeedState.load(layout.node(strand.name)).present { holders[id, default: []].append(strand.name) }
        }
        var copied: [Copied] = []
        for (id, nodes) in holders where nodes.count >= 2 {
            guard let document = world.document(id: id) else { continue }
            for fact in document.facts.prefix(2) {
                guard let match = byKey["\(nodes[0])|\(fact.kind.rawValue)|\(fact.subject)|\(fact.answer)"]
                        ?? byKey["\(nodes[1])|\(fact.kind.rawValue)|\(fact.subject)|\(fact.answer)"], let question = match.questions?.first else { continue }
                copied.append(Copied(question: question.text, stem: question.stem, expected: fact.answer, holders: nodes.sorted()))
            }
        }
        return Sets(questions: questions, unknown: unknown, shared: shared, copied: copied, unmatched: unmatched)
    }

    static func arm(
        _ name: String, generator: BraidedGenerator, links: [StrandLink], adapter: QuestionAdapter?, sets: Sets, tokenizer: RaoTokenizer, sizes: Sizes
    ) throws -> QuestionArmResult {
        let started = Date()
        let commonsName = generator.commons.map { generator.names[$0] }
        func params() -> GenerationParameters {
            var params = GenerationParameters(tapLayer: links[0].descriptor.tapLayer, alpha: links[0].descriptor.alpha)
            params.maxTokens = sizes.maxTokens
            return params
        }
        /// The prompt an arm asks for a question: its rewrite, its stem, or the v1 prompt.
        func prompt(_ question: Question) -> (tokens: [Int], text: String, rewrite: QuestionRewrite?) {
            switch name {
            case "commons":
                let rewrite = adapter?.rewrite(question.text) ?? QuestionAdapter.fallback(question.text)
                return (QuestionAdapter.stemTokens(rewrite.stem, tokenizer: tokenizer), rewrite.stem, rewrite)
            case "rules":
                let rewrite = QuestionAdapter.fallback(question.text)
                return (QuestionAdapter.stemTokens(rewrite.stem, tokenizer: tokenizer), rewrite.stem, rewrite)
            case "stem": return (QuestionAdapter.stemTokens(question.stem, tokenizer: tokenizer), question.stem, nil)
            case "slice": return (question.slice, tokenizer.decode(question.slice), nil)
            default: return (QuestionAdapter.stemTokens(question.paraphrase, tokenizer: tokenizer), question.paraphrase, nil)
            }
        }
        func request(_ tokens: [Int], _ text: String, question: String, rewrite: QuestionRewrite?) -> BraidRequest {
            BraidRequest(promptTokens: tokens, promptText: text, params: params(), stopAtSentenceEnd: true, question: rewrite, context: sizes.context,
                         subject: QuestionAdapter.subjectTokens(stem: text, tokens: tokens, question: question, tokenizer: tokenizer))
        }
        func rewriteText(_ text: String) -> QuestionRewrite? {
            switch name {
            case "commons": return adapter?.rewrite(text) ?? QuestionAdapter.fallback(text)
            case "rules": return QuestionAdapter.fallback(text)
            default: return nil
            }
        }
        var result = QuestionArmResult(
            arm: name, questions: 0, rewriteExact: 0, rewriteF1: nil, exact: 0, citedAnswer: 0, citationAt1: nil, unknown: 0,
            unknownCommonsLeads: 0, unknownThreadCredit: nil, shared: 0, sharedDecided: 0, sharedFollowed: 0, sharedDuplicateCredit: nil,
            sharedTies: 0, adapterMs: nil, seconds: 0)
        var f1s: [Float] = []
        var latencies: [Double] = []
        var cited: [Bool] = []
        let rewrites = name == "commons" || name == "rules"

        for question in sets.questions {
            let (tokens, text, rewrite) = prompt(question)
            result.questions += 1
            if let rewrite {
                latencies.append(rewrite.seconds)
                if RuleRewriter.normalised(rewrite.stem) == RuleRewriter.normalised(question.stem) { result.rewriteExact += 1 }
                f1s.append(tokenF1(rewrite.stem, question.stem))
            }
            let generation = try generator.generate(request(tokens, text, question: question.text, rewrite: rewrite))
            let generated = generation.traces.filter { !$0.isPrompt }
            let answerLength = max(1, tokenizer.encode(question.expected).count)
            let answer = generated.prefix(answerLength)
            let ownerFound = generation.braid?.strands.first { $0.name == question.node }?.context != nil
            if ownerFound { result.ownerContext += 1 }
            if generation.text.hasPrefix(question.expected) {
                result.exact += 1
                if ownerFound { result.ownerContextExact += 1 } else { result.noContextExact += 1 }
                if answer.contains(where: { trace in
                    trace.citations.first.map { generation.partition(row: $0.row) }.flatMap { $0 }.map {
                        $0.documentID == question.source.documentID && $0.partitionIndex == question.source.partitionIndex
                    } ?? false
                }) { result.citedAnswer += 1 }
            }
            for trace in answer {
                if let top = trace.neighbours.first, let partition = generator.partitionsByRow[top.cited.row] {
                    cited.append(partition.documentID == question.source.documentID && partition.partitionIndex == question.source.partitionIndex)
                }
            }
        }
        result.rewriteF1 = f1s.isEmpty ? nil : Stats.mean(f1s)
        result.citationAt1 = cited.isEmpty ? nil : Float(cited.filter { $0 }.count) / Float(cited.count)
        result.adapterMs = latencies.isEmpty ? nil : Float(latencies.reduce(0, +) / Double(latencies.count) * 1000)

        // Q4: entities nobody holds, asked the arm's way (the stem arms have no stored stem for them: the rules').
        if let commonsName, name != "slice", name != "paraphrase" {
            var credits: [Float] = []
            for text in sets.unknown {
                let rewrite = rewriteText(text) ?? QuestionAdapter.fallback(text)
                let generation = try generator.generate(request(QuestionAdapter.stemTokens(rewrite.stem, tokenizer: tokenizer), rewrite.stem, question: text, rewrite: rewrite))
                let generated = generation.traces.filter { !$0.isPrompt }
                guard let first = generated.first?.strands else { continue }
                result.unknown += 1
                let own = first.first { $0.strand == commonsName }?.gate ?? 0
                if own >= (first.filter { $0.strand != commonsName }.map(\.gate).max() ?? 0) { result.unknownCommonsLeads += 1 }
                for trace in generated where trace.role == .content {
                    credits.append((trace.strands ?? []).filter { $0.strand != commonsName }.reduce(0) { $0 + ($1.credit ?? 0) })
                }
            }
            result.unknownThreadCredit = credits.isEmpty ? nil : Stats.mean(credits)
        }

        // Q5: shared facts, decided by retrieval. The followed Thread is the one whose chain advanced
        // on the answer's tokens (`FollowedCredit` wrote it on the traces); "decided" means one side's
        // chain advanced and the other's did not advance as much. Read on the two sides only.
        if rewrites || name == "stem" {
            var duplicate: [Float] = []
            for shared in sets.shared {
                let rewrite = rewriteText(shared.question)
                let stem = rewrite?.stem ?? shared.stem
                let generation = try generator.generate(request(QuestionAdapter.stemTokens(stem, tokenizer: tokenizer), stem, question: shared.question, rewrite: rewrite))
                result.shared += 1
                guard generation.text.hasPrefix(shared.expected) else { continue }
                let generated = generation.traces.filter { !$0.isPrompt && $0.role == .content }
                let sides = [shared.source, shared.target]
                var advances: [String: Int] = [:]
                for trace in generated {
                    for strand in trace.strands ?? [] where sides.contains(strand.strand) {
                        if strand.trajectory?.next == trace.token { advances[strand.strand, default: 0] += 1 }
                    }
                }
                let a = advances[shared.source] ?? 0, b = advances[shared.target] ?? 0
                if a == b {
                    if a > 0 { result.sharedTies += 1 }
                    continue
                }
                result.sharedDecided += 1
                let truth = a > b ? shared.source : shared.target
                let other = a > b ? shared.target : shared.source
                var credit: [String: Float] = [:]
                for trace in generated { for strand in trace.strands ?? [] where sides.contains(strand.strand) { credit[strand.strand, default: 0] += strand.credit ?? 0 } }
                if let top = credit.max(by: { $0.value < $1.value }), top.key == truth, (credit[other] ?? 0) < top.value { result.sharedFollowed += 1 }
                let otherCredit = generated.map { ($0.strands ?? []).first { $0.strand == other }?.credit ?? 0 }
                if !otherCredit.isEmpty { duplicate.append(Stats.mean(otherCredit)) }
            }
            result.sharedDuplicateCredit = duplicate.isEmpty ? nil : Stats.mean(duplicate)
            // Q5b: byte-identical copies must tie; a copy that takes the whole fact is the gaming case.
            for copy in sets.copied {
                let rewrite = rewriteText(copy.question)
                let stem = rewrite?.stem ?? copy.stem
                let generation = try generator.generate(request(QuestionAdapter.stemTokens(stem, tokenizer: tokenizer), stem, question: copy.question, rewrite: rewrite))
                guard generation.text.hasPrefix(copy.expected) else { continue }
                result.copied += 1
                let generated = generation.traces.filter { !$0.isPrompt && $0.role == .content }
                var credit: [String: Float] = [:]
                for trace in generated { for strand in trace.strands ?? [] where copy.holders.contains(strand.strand) { credit[strand.strand, default: 0] += strand.credit ?? 0 } }
                let values = copy.holders.map { credit[$0] ?? 0 }
                let total = values.reduce(0, +)
                guard total > 0, let top = values.max(), let low = values.min() else { continue }
                if (top - low) / total <= 0.10 { result.copiedTied += 1 }
                if low == 0 && top > 0 { result.copiedTaken += 1 }
            }
        }
        result.seconds = Date().timeIntervalSince(started)
        return result
    }

    /// Token F1 of two stems, on whitespace-split lowercase words.
    static func tokenF1(_ a: String, _ b: String) -> Float {
        let x = RuleRewriter.normalised(a).split(separator: " ").map(String.init)
        let y = RuleRewriter.normalised(b).split(separator: " ").map(String.init)
        guard !x.isEmpty, !y.isEmpty else { return x.isEmpty && y.isEmpty ? 1 : 0 }
        var counts: [String: Int] = [:]
        for word in y { counts[word, default: 0] += 1 }
        var common = 0
        for word in x where (counts[word] ?? 0) > 0 {
            common += 1
            counts[word]! -= 1
        }
        guard common > 0 else { return 0 }
        let precision = Float(common) / Float(x.count)
        let recall = Float(common) / Float(y.count)
        return 2 * precision * recall / (precision + recall)
    }

    // MARK: - The rules (pure)

    /// Q1: rules 100%, commons ≥ 0.50. Q2: the winner's exact within 0.05 of the stem arm's. Q3: cited
    /// answers ≥ 0.90 on the winner. Q4: the commons leads ≥ 0.80 of unknown questions and the Threads'
    /// credit on them ≤ 0.25. Q5: the followed Thread is the most credited on ≥ 0.90 of decided shared
    /// facts and the other side's credit ≤ 0.05. Winner: commons if Q1 and Q2 hold for it, else rules.
    public static func evaluate(arms: [QuestionArmResult]) -> QuestionEvaluation {
        func of(_ name: String) -> QuestionArmResult? { arms.first { $0.arm == name } }
        func pct(_ value: Float?) -> String { value.map { String(format: "%.0f%%", $0 * 100) } ?? "—" }
        func num(_ value: Float?) -> String { value.map { String(format: "%.3f", $0) } ?? "—" }
        var rules: [TrajectoryRuleResult] = []
        var reported: [String: Float] = [:]
        guard let stem = of("stem"), let rulesArm = of("rules") else {
            return QuestionEvaluation(rules: [], winner: nil, qualifies: false, reported: [:], summary: "the stem and rules arms are needed")
        }
        let commons = of("commons")
        func q1(_ arm: QuestionArmResult, bar: Float) -> Bool { arm.questions > 0 && arm.rewriteRate >= bar - 1e-6 }
        func q2(_ arm: QuestionArmResult) -> Bool { arm.exactRate >= stem.exactRate - 0.05 - 1e-6 }
        let commonsWins = commons.map { q1($0, bar: 0.5) && q2($0) } ?? false
        let winner = commonsWins ? commons! : rulesArm
        rules.append(TrajectoryRuleResult(
            rule: "Q1 rewrite fidelity", passed: q1(rulesArm, bar: 1) && (commons.map { q1($0, bar: 0.5) } ?? true),
            detail: "stems reached: rules \(pct(rulesArm.rewriteRate)) (F1 \(num(rulesArm.rewriteF1))), commons \(pct(commons?.rewriteRate)) (F1 \(num(commons?.rewriteF1)))"))
        rules.append(TrajectoryRuleResult(
            rule: "Q2 exact", passed: q2(winner),
            detail: "\(winner.arm) \(winner.exact)/\(winner.questions) (\(pct(winner.exactRate))) against the stem arm's \(pct(stem.exactRate)); "
                + "slice \(pct(of("slice")?.exactRate)), paraphrase \(pct(of("paraphrase")?.exactRate))"))
        if let slice = of("slice") {
            rules.append(TrajectoryRuleResult(
                rule: "Q2′ context", passed: winner.exactRate >= slice.exactRate - 0.10 - 1e-6,
                detail: "\(winner.arm) \(pct(winner.exactRate)) against the slice's \(pct(slice.exactRate)), the ceiling"))
        }
        rules.append(TrajectoryRuleResult(
            rule: "Q3 cited", passed: winner.exact > 0 && winner.citedRate >= 0.9,
            detail: "\(winner.arm): \(winner.citedAnswer) of \(winner.exact) exact answers cited to the fact's partition (\(pct(winner.citedRate))); per-token citation@1 \(pct(winner.citationAt1))"))
        rules.append(TrajectoryRuleResult(
            rule: "Q4 unknown", passed: winner.unknown > 0 && winner.unknownLeadRate >= 0.8 && (winner.unknownThreadCredit ?? 1) <= 0.25,
            detail: "\(winner.arm): the commons leads \(winner.unknownCommonsLeads) of \(winner.unknown) (\(pct(winner.unknownLeadRate))); Threads' credit \(num(winner.unknownThreadCredit))"))
        let decidedRate = winner.shared > 0 ? Float(winner.sharedDecided) / Float(winner.shared) : 0
        rules.append(TrajectoryRuleResult(
            rule: "Q5a retold",
            passed: winner.shared > 0 && decidedRate >= 0.6 && winner.sharedFollowedRate >= 0.9 && (winner.sharedDuplicateCredit ?? 1) <= 0.10,
            detail: "\(winner.arm): \(winner.sharedDecided) of \(winner.shared) decided by retrieval (\(pct(decidedRate))); the followed Thread most credited on "
                + "\(winner.sharedFollowed) (\(pct(winner.sharedFollowedRate))); the other side's credit \(num(winner.sharedDuplicateCredit)); ties \(winner.sharedTies)"))
        if winner.copied > 0 {
            rules.append(TrajectoryRuleResult(
                rule: "Q5b copied", passed: winner.copiedTieRate >= 0.9 && winner.copiedTaken == 0,
                detail: "\(winner.arm): \(winner.copiedTied) of \(winner.copied) copied facts tie between their holders (\(pct(winner.copiedTieRate))); "
                    + "\(winner.copiedTaken) taken whole by one copy"))
        }
        for arm in arms {
            reported["\(arm.arm): exact"] = arm.exactRate
            if let ms = arm.adapterMs { reported["\(arm.arm): adapter ms"] = ms }
        }
        let failed = rules.filter { !$0.passed }.map(\.rule)
        return QuestionEvaluation(
            rules: rules, winner: winner.arm, qualifies: failed.isEmpty, reported: reported,
            summary: (failed.isEmpty ? "\(winner.arm) wins and every rule holds" : "\(winner.arm) wins; fails " + failed.joined(separator: ", ")))
    }
}
