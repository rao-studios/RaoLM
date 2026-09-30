//
//  TrajectoryBench.swift
//  RaoLMBraid
//
//  WHAT: Whether a Thread's retrieval trajectory tells its own text from a pastiche of it, and
//        whether putting the trace in the gate earns the holder its tokens. On a braid fed from a
//        braid dataset, whole documents are scored with every Thread asked: held (as written),
//        shuffled (the same sentences reordered), collage (sentence k of the k-th document),
//        voice (unfed documents in a Thread's voice), told and source (both sides of a
//        retelling), and generic prose. Gate arms run only when the trace rules pass.
//  OUT:  One report with every text's readings and the arms' results; `evaluate` applies the rule.
//  PIN:  The texts, the rule and its thresholds were fixed before any numbers (Docs/BRAID.md,
//        "Choosing the trajectory"). Nothing is written to a node.
//

import Foundation
import RaoLMCore
import RaoLMModel
import RaoLMProvenance
import RaoLMTraining

public enum TrajectoryTextKind: String, Codable, Sendable, CaseIterable {
    case held, shuffled, collage, voice, voiceShuffled, told, source, generic
}

/// A fact's answer inside a text: its first token, its length in tokens, and its words.
public struct TrajectoryAnswer: Codable, Sendable, Equatable {
    public var start: Int
    public var count: Int
    public var text: String
}

public struct TrajectoryText: Codable, Sendable, Equatable {
    public var kind: TrajectoryTextKind
    public var label: String
    public var tokens: [Int]
    /// The Thread the text belongs to: whose document it is, was cut from or is in the voice of,
    /// and the holder of a told or source text. Nil for generic prose.
    public var owner: String?
    /// Told and source: the Thread on the other side of the retelling.
    public var near: String?
    public var document: String?
    public var answers: [TrajectoryAnswer] = []
}

/// One Thread's reading of one text.
public struct TrajectoryReading: Codable, Sendable, Equatable {
    public var strand: String
    /// Mean trace from token `TrajectoryBench.from` on, and the longest chain anywhere in the text.
    public var trace: Float
    public var longest: Int
    /// The longest chain the wire's hits alone show (hits of any rank, no bridges).
    public var wireLongest: Int
    /// At the text's last position.
    public var manner: Float?
    public var arc: Float?
    public var fit: Float
    /// Token-local baselines from `from` on: the retrieval mass on the tokens that came, and the best cosine.
    public var meanFit: Float
    public var meanBest: Float
    /// Mean log of what the Thread alone gave the text's tokens, every Thread asked.
    public var logLikelihood: Float
}

public struct TrajectoryTextResult: Codable, Sendable, Equatable {
    public var kind: TrajectoryTextKind
    public var label: String
    public var owner: String?
    public var near: String?
    public var tokens: Int
    public var readings: [TrajectoryReading]

    public func reading(_ strand: String?) -> TrajectoryReading? {
        guard let strand else { return nil }
        return readings.first { $0.strand == strand }
    }
}

/// One gate under test.
public struct TrajectoryArm: Codable, Sendable, Equatable {
    public enum Role: String, Codable, Sendable {
        /// The default gate everything is measured against.
        case base
        /// May win the trace rule.
        case candidate
        /// "No agreement": if it does as well, the trajectory adds nothing.
        case reference
        /// Asking by manner.
        case manner
    }

    public var name: String
    public var gate: BraidGate
    public var role: Role

    /// In the order a tie goes to: the default gate, lift, gate β 1, 2, 4, both β 1, 2, 4, then the
    /// reference and the asking arm.
    public static func all(betas: [Float] = [1, 2, 4]) -> [TrajectoryArm] {
        var arms = [TrajectoryArm(name: "braided", gate: BraidGate(), role: .base)]
        arms.append(TrajectoryArm(name: "lift", gate: BraidGate(trajectory: .lift), role: .candidate))
        for use in [BraidGate.TrajectoryUse.gate, .both] {
            for beta in betas {
                arms.append(TrajectoryArm(name: "\(use.rawValue) β\(beta.formatted())", gate: BraidGate(trajectory: use, trajectoryBeta: beta),
                                          role: .candidate))
            }
        }
        arms.append(TrajectoryArm(name: "no agreement", gate: BraidGate(agreement: false), role: .reference))
        arms.append(TrajectoryArm(name: "ask by manner", gate: BraidGate(ask: .manner), role: .manner))
        return arms
    }
}

public struct TrajectoryArmResult: Codable, Sendable, Equatable {
    public var arm: String
    /// The holder's mean share of the fact answer tokens of told and source texts.
    public var holderShare: Float?
    public var answerTokens: Int
    /// Document-prefix prompts (up to a told or source text's last answer) answered exactly.
    public var exact: Int
    public var prompts: Int
    /// Voice and generic texts, continued four tokens: the mean largest gate at the first
    /// generated token, and the share of generated tokens with every Thread asked.
    public var largestGate: Float?
    public var allAsked: Float?
    /// Of every Thread and text scored, the share not asked for its hidden state.
    public var notAsked: Float?

    public init(
        arm: String, holderShare: Float?, answerTokens: Int, exact: Int, prompts: Int, largestGate: Float?, allAsked: Float?,
        notAsked: Float? = nil
    ) {
        self.arm = arm
        self.holderShare = holderShare
        self.answerTokens = answerTokens
        self.exact = exact
        self.prompts = prompts
        self.largestGate = largestGate
        self.allAsked = allAsked
        self.notAsked = notAsked
    }
}

public struct TrajectoryRuleResult: Codable, Sendable, Equatable {
    public var rule: String
    public var passed: Bool
    public var detail: String
}

public struct TrajectoryEvaluation: Codable, Sendable, Equatable {
    public var rules: [TrajectoryRuleResult]
    /// Candidate arms that qualify, and why each other arm did not.
    public var qualifies: [String]
    public var failures: [String: [String]]
    public var winner: String?
    public var mannerQualifies: Bool?
    /// Measures with no threshold.
    public var reported: [String: Float]
    public var summary: String
}

public struct TrajectoryReport: Codable, Sendable {
    public var createdAt: Date
    public var world: MockWorld.Record?
    public var nodes: [String: Int]
    public var documents: [String: Int]
    public var memorised: [String: Float]
    public var rule: TrajectoryRule
    public var texts: [TrajectoryTextResult]
    public var arms: [TrajectoryArmResult]
    public var evaluation: TrajectoryEvaluation
}

public enum TrajectoryBench {
    /// A text's value is its mean from this token on.
    public static let from = 24
    /// A holder traces its own text at least this; anyone else at most `low`.
    public static let holderFloor: Float = 0.40
    public static let low: Float = 0.10

    public struct Sizes: Sendable {
        /// Held, collage and voice texts per node.
        public var perNode = 20
        public var minTokens = 128
        public var maxTokens = 256
        /// Told and source texts may be shorter.
        public var minRetold = 96
        /// Voice texts per node continued in each gate arm.
        public var voiceContinued = 4
        public var seed: UInt64 = 42

        public init() {}
    }

    public enum Arms: String, Sendable, CaseIterable {
        /// Only when T1 to T3 pass, as the rule says.
        case auto
        /// Always (an arm cannot win unless T1 to T3 pass).
        case always
        case never
    }

    // MARK: - Running (loads every live version)

    public static func run(
        layout: BraidLayout, tokenizer: RaoTokenizer, sizes: Sizes = Sizes(), arms: Arms = .auto, owner: String = "raolm-braid",
        progress: ((String) -> Void)? = nil
    ) throws -> TrajectoryReport {
        let record = MockWorld.Record.load(layout)
        let world = try RoutingBench.world(layout: layout, seed: record?.seed ?? 42)
        let (vocabulary, strands) = try RoutingBench.strands(layout: layout, tokenizer: tokenizer, owner: owner, names: world.names)
        guard !strands.isEmpty else { throw BraidSessionError.noLiveNodes }
        let links = strands.map { LocalStrandLink(strand: $0, vocabularySHA256: vocabulary.sha256) }
        let generator = try BraidedGenerator(links: links, head: UmbrellaHead(vocabulary: vocabulary), tokenizer: tokenizer)
        let texts = TrajectoryTexts.build(world: world, layout: layout, names: strands.map(\.name), tokenizer: tokenizer, sizes: sizes)
        progress?("texts: " + TrajectoryTextKind.allCases.map { kind in "\(kind.rawValue) \(texts.filter { $0.kind == kind }.count)" }
            .joined(separator: " · "))

        let documentOfRow = strands.map { strand in Dictionary(uniqueKeysWithValues: strand.index.partitions.map { ($0.row, $0.documentID) }) }
        var results: [TrajectoryTextResult] = []
        for (i, text) in texts.enumerated() {
            if i % 50 == 0 { progress?("reading \(i)/\(texts.count)") }
            let traces = try scored(generator, links: links, tokens: text.tokens, gate: BraidGate())
            var readings: [TrajectoryReading] = []
            for (t, strand) in strands.enumerated() {
                let steps = try strand.open(session: "trajectory-bench", tokens: text.tokens, k: params(links, maxTokens: 0).k)
                strand.close(session: "trajectory-bench")
                let alone = traces.compactMap { trace in trace.strands?.first { $0.strand == strand.name }?.alone }
                readings.append(reading(strand: strand.name, steps: steps, tokens: text.tokens, documentOfRow: documentOfRow[t],
                                        logLikelihood: alone.isEmpty ? -30 : Stats.mean(alone.map { log(max($0, 1e-12)) })))
            }
            results.append(TrajectoryTextResult(kind: text.kind, label: text.label, owner: text.owner, near: text.near,
                                                tokens: text.tokens.count, readings: readings))
        }

        let first = evaluate(texts: results, arms: [])
        let tracePasses = first.rules.prefix(3).allSatisfy(\.passed)
        var armResults: [TrajectoryArmResult] = []
        if arms == .always || (arms == .auto && tracePasses) {
            for arm in TrajectoryArm.all() {
                progress?("arm \(arm.name)")
                armResults.append(try run(arm, texts: texts, generator: generator, links: links, sizes: sizes))
            }
        }
        var documents: [String: Int] = [:]
        var memorised: [String: Float] = [:]
        for strand in strands {
            documents[strand.name] = Set(strand.index.partitions.map(\.documentID)).count
            memorised[strand.name] = strand.index.info.evalMemorisedFraction
        }
        return TrajectoryReport(
            createdAt: .wholeSecond(), world: record, nodes: Dictionary(uniqueKeysWithValues: strands.map { ($0.name, $0.version) }),
            documents: documents, memorised: memorised, rule: .standard, texts: results, arms: armResults,
            evaluation: evaluate(texts: results, arms: armResults))
    }

    static func params(_ links: [StrandLink], maxTokens: Int) -> GenerationParameters {
        var params = GenerationParameters(tapLayer: links[0].descriptor.tapLayer, alpha: links[0].descriptor.alpha)
        params.maxTokens = maxTokens
        return params
    }

    /// Every token of `tokens` scored under `gate`, nothing generated.
    static func scored(_ generator: BraidedGenerator, links: [StrandLink], tokens: [Int], gate: BraidGate) throws -> [TokenTrace] {
        try generator.generate(BraidRequest(promptTokens: tokens, promptText: "", params: params(links, maxTokens: 0), gating: .braided,
                                            gate: gate)).traces
    }

    static func reading(
        strand: String, steps: [StrandStep], tokens: [Int], documentOfRow: [Int: String], logLikelihood: Float
    ) -> TrajectoryReading {
        let later = steps.indices.filter { $0 >= from }
        let trajectories = steps.map(\.trajectory)
        let traces = BraidMixer.traces(later.map { trajectories[$0] })
        let wire = TrajectoryAudit.wireLengths(tokens: tokens, hits: steps.map(\.hits), documentOfRow: documentOfRow)
        let last = trajectories.last ?? nil
        return TrajectoryReading(
            strand: strand, trace: traces.isEmpty ? 0 : Stats.mean(traces), longest: trajectories.compactMap { $0?.length }.max() ?? 0,
            wireLongest: wire.max() ?? 0, manner: last?.manner, arc: last?.arc, fit: last?.fit ?? 0,
            meanFit: later.isEmpty ? 0 : Stats.mean(later.map { trajectories[$0]?.fit ?? 0 }),
            meanBest: later.isEmpty ? 0 : Stats.mean(later.map { steps[$0].hits.first?.score ?? 0 }), logLikelihood: logLikelihood)
    }

    static func run(
        _ arm: TrajectoryArm, texts: [TrajectoryText], generator: BraidedGenerator, links: [StrandLink], sizes: Sizes
    ) throws -> TrajectoryArmResult {
        var shareSum: Float = 0
        var shareTokens = 0
        var exact = 0
        var prompts = 0
        var asked = 0
        var scoredThreads = 0
        for text in texts where (text.kind == .told || text.kind == .source) && !text.answers.isEmpty {
            let traces = try scored(generator, links: links, tokens: text.tokens, gate: arm.gate)
            let byIndex = Dictionary(uniqueKeysWithValues: traces.map { ($0.index, $0) })
            for answer in text.answers {
                for position in answer.start..<(answer.start + answer.count) where position >= 1 {
                    guard let share = byIndex[position]?.strands?.first(where: { $0.strand == text.owner })?.share else { continue }
                    shareSum += share
                    shareTokens += 1
                }
            }
            if let shares = traces.last?.strands {
                asked += shares.filter(\.open).count
                scoredThreads += shares.count
            }
            if let last = text.answers.last, last.start > 0 {
                prompts += 1
                let generation = try generator.generate(BraidRequest(
                    promptTokens: Array(text.tokens[0..<last.start]), promptText: "", params: params(links, maxTokens: last.count + 2),
                    gating: .braided, gate: arm.gate))
                if generation.text.trimmingCharacters(in: .whitespaces).hasPrefix(last.text.trimmingCharacters(in: .whitespaces)) { exact += 1 }
            }
        }
        var largest: [Float] = []
        var allAsked = 0
        var generatedTokens = 0
        var continued = texts.filter { $0.kind == .generic }
        for owner in Set(texts.compactMap { $0.kind == .voice ? $0.owner : nil }).sorted() {
            continued += texts.filter { $0.kind == .voice && $0.owner == owner }.prefix(sizes.voiceContinued)
        }
        for text in continued {
            let generation = try generator.generate(BraidRequest(
                promptTokens: Array(text.tokens.prefix(sizes.minTokens)), promptText: "", params: params(links, maxTokens: 4), gating: .braided,
                gate: arm.gate))
            let generated = generation.traces.filter { !$0.isPrompt }
            if let shares = generated.first?.strands { largest.append(shares.map(\.gate).max() ?? 1) }
            for trace in generated {
                generatedTokens += 1
                if (trace.strands ?? []).allSatisfy(\.open) { allAsked += 1 }
            }
        }
        return TrajectoryArmResult(
            arm: arm.name, holderShare: shareTokens > 0 ? shareSum / Float(shareTokens) : nil, answerTokens: shareTokens, exact: exact,
            prompts: prompts, largestGate: largest.isEmpty ? nil : Stats.mean(largest),
            allAsked: generatedTokens > 0 ? Float(allAsked) / Float(generatedTokens) : nil,
            notAsked: scoredThreads > 0 ? 1 - Float(asked) / Float(scoredThreads) : nil)
    }

    // MARK: - The rule (pure)

    public static func evaluate(texts: [TrajectoryTextResult], arms: [TrajectoryArmResult]) -> TrajectoryEvaluation {
        func of(_ kinds: [TrajectoryTextKind]) -> [TrajectoryTextResult] { texts.filter { kinds.contains($0.kind) } }
        func share(_ flags: [Bool]) -> Float { flags.isEmpty ? 0 : Float(flags.filter { $0 }.count) / Float(flags.count) }
        func pct(_ value: Float) -> String { String(format: "%.0f%%", value * 100) }
        func auroc(_ positives: [Float], _ negatives: [Float]) -> Float? {
            Stats.auroc(scores: (positives + negatives).map(Double.init), labels: positives.map { _ in true } + negatives.map { _ in false })
                .map(Float.init)
        }
        var rules: [TrajectoryRuleResult] = []
        var reported: [String: Float] = [:]

        // T1: the holder traces its own document as written, not the same sentences reordered or a collage.
        let held = of([.held]).compactMap { $0.reading($0.owner) }
        let apart = of([.shuffled, .collage]).compactMap { $0.reading($0.owner) }
        let t1Held = share(held.map { $0.trace >= holderFloor })
        let t1Apart = share(apart.map { $0.trace <= low })
        let t1AUROC = auroc(held.map(\.trace), apart.map(\.trace))
        rules.append(TrajectoryRuleResult(
            rule: "T1 arc, not phrase",
            passed: !held.isEmpty && !apart.isEmpty && t1Held >= 0.9 && t1Apart >= 0.9 && (t1AUROC ?? 0) >= 0.95,
            detail: "holder ≥ 0.40 on \(pct(t1Held)) of \(held.count) held; ≤ 0.10 on \(pct(t1Apart)) of \(apart.count) shuffled and collage; AUROC "
                + (t1AUROC.map { String(format: "%.3f", $0) } ?? "—")))
        reported["T1 token-local: retrieval mass on the tokens, AUROC"] = auroc(held.map(\.meanFit), apart.map(\.meanFit))
        reported["T1 token-local: best cosine, AUROC"] = auroc(held.map(\.meanBest), apart.map(\.meanBest))

        // T2: among Threads, the holder of a told or source text traces it and nobody else does.
        let retold = of([.told, .source])
        let holders = retold.compactMap { $0.reading($0.owner) }
        let others = retold.flatMap { text in text.readings.filter { $0.strand != text.owner } }
        let near = retold.compactMap { $0.reading($0.near) }
        let t2Holders = share(holders.map { $0.trace >= holderFloor })
        let t2Others = share(others.map { $0.trace <= low })
        rules.append(TrajectoryRuleResult(
            rule: "T2 holder among Threads", passed: !holders.isEmpty && t2Holders >= 0.9 && t2Others >= 0.9,
            detail: "holder ≥ 0.40 on \(pct(t2Holders)) of \(holders.count) told and source; near and other ≤ 0.10 on \(pct(t2Others)) of \(others.count) (near alone \(pct(share(near.map { $0.trace <= low }))))"))

        // T3: text in a Thread's voice that nobody holds earns no Thread a trace.
        var t3 = true
        var t3Details: [String] = []
        var falseChain = 0
        for kind in [TrajectoryTextKind.voice, .voiceShuffled, .generic] {
            let group = of([kind])
            let clean = share(group.map { $0.readings.allSatisfy { $0.trace <= low } })
            let mean = Stats.mean(group.flatMap { $0.readings.map(\.trace) })
            falseChain = max(falseChain, group.flatMap { $0.readings.map(\.longest) }.max() ?? 0)
            if group.isEmpty || clean < 0.95 || mean > 0.05 { t3 = false }
            t3Details.append("\(kind.rawValue): every Thread ≤ 0.10 on \(pct(clean)) of \(group.count), mean " + String(format: "%.3f", mean))
        }
        rules.append(TrajectoryRuleResult(rule: "T3 voice earns nothing", passed: t3, detail: t3Details.joined(separator: "; ")))
        reported["T3 longest false chain (tokens)"] = Float(falseChain)

        // M1: the voice's own Thread has the highest manner at the end of a voice text.
        let voice = of([.voice])
        func highest(_ text: TrajectoryTextResult, _ value: (TrajectoryReading) -> Float?) -> Bool {
            guard let own = text.reading(text.owner).flatMap(value) else { return false }
            return text.readings.allSatisfy { $0.strand == text.owner || (value($0) ?? -1) < own }
        }
        let m1 = share(voice.map { highest($0, \.manner) })
        rules.append(TrajectoryRuleResult(rule: "M1 routing", passed: !voice.isEmpty && m1 >= 0.8,
                                          detail: "the voice's own Thread highest on \(pct(m1)) of \(voice.count) voice texts"))
        reported["M1 fit alone: own Thread highest"] = share(voice.map { highest($0, { $0.fit }) })

        // M2: order, not wording: the own Thread's manner on voice texts against the same shuffled.
        let m2 = auroc(voice.map { $0.reading($0.owner)?.manner ?? 0 }, of([.voiceShuffled]).map { $0.reading($0.owner)?.manner ?? 0 })
        rules.append(TrajectoryRuleResult(rule: "M2 order, not wording", passed: (m2 ?? 0) >= 0.75,
                                          detail: "AUROC " + (m2.map { String(format: "%.3f", $0) } ?? "—")))

        // M3: no Thread reaches the floor on prose nobody holds.
        let floor = BraidGate().askFloor
        let generic = of([.generic])
        let m3 = generic.filter { $0.readings.contains { ($0.manner ?? 0) >= floor } }.count
        rules.append(TrajectoryRuleResult(rule: "M3 nobody's text", passed: !generic.isEmpty && m3 == 0,
                                          detail: "a Thread reached \(floor) on \(m3) of \(generic.count) generic texts"))

        // M4: asking by manner never leaves out the Thread that predicts the text best.
        let byManner = BraidGate(ask: .manner)
        let m4Texts = of([.held, .told, .source, .voice])
        var askedFlags: [Bool] = []
        let m4 = share(m4Texts.map { text in
            let asked = BraidMixer.asked(manner: text.readings.map(\.manner), gate: byManner)
            askedFlags += asked
            guard let best = text.readings.indices.max(by: { text.readings[$0].logLikelihood < text.readings[$1].logLikelihood }) else { return false }
            return asked[best]
        })
        rules.append(TrajectoryRuleResult(rule: "M4 routing loses nobody", passed: !m4Texts.isEmpty && m4 >= 0.95,
                                          detail: "the best-predicting Thread asked on \(pct(m4)) of \(m4Texts.count) texts"))
        reported["M4 hidden-state requests saved by manner"] = 1 - share(askedFlags)
        reported["collage: its Thread's manner at the end (mean)"] = Stats.mean(of([.collage]).compactMap { $0.reading($0.owner)?.manner ?? 0 })
        let readings = texts.flatMap(\.readings)
        reported["wire chain ≤ node chain"] = share(readings.map { $0.wireLongest <= $0.longest })
        let nodeLongest = Stats.mean(readings.map { Float($0.longest) })
        reported["wire chain / node chain (means)"] = nodeLongest > 0 ? Stats.mean(readings.map { Float($0.wireLongest) }) / nodeLongest : 0

        // The gate arms.
        let tracePasses = rules.prefix(3).allSatisfy(\.passed)
        let mannerPasses = rules.dropFirst(3).allSatisfy(\.passed)
        let roles = Dictionary(uniqueKeysWithValues: TrajectoryArm.all().map { ($0.name, $0.role) })
        var qualifies: [String] = []
        var failures: [String: [String]] = [:]
        var winner: String?
        var mannerQualifies: Bool?
        var summary: String
        let failedTrace = rules.prefix(3).filter { !$0.passed }.map(\.rule)
        if let base = arms.first(where: { roles[$0.arm] == .base }) {
            let baseShare = base.holderShare ?? 0
            func guards(_ arm: TrajectoryArmResult) -> [String] {
                var failed: [String] = []
                if arm.exact < base.exact - 1 { failed.append("exact \(arm.exact) of \(arm.prompts), the default's \(base.exact)") }
                if let a = arm.largestGate, let b = base.largestGate, abs(a - b) > 0.05 {
                    failed.append(String(format: "largest gate %.2f on voice and generic, the default's %.2f", a, b))
                }
                if let a = arm.allAsked, let b = base.allAsked, abs(a - b) > 0.05 {
                    failed.append(String(format: "all asked %.0f%% on voice and generic, the default's %.0f%%", a * 100, b * 100))
                }
                return failed
            }
            for arm in arms where roles[arm.arm] == .candidate {
                var failed = guards(arm)
                if !tracePasses { failed.insert("the trace rules do not pass", at: 0) }
                let share = arm.holderShare ?? 0
                if share < baseShare + 0.10 { failed.append(String(format: "holder share %.3f, needs %.3f", share, baseShare + 0.10)) }
                if failed.isEmpty { qualifies.append(arm.arm) } else { failures[arm.arm] = failed }
            }
            let best = qualifies.compactMap { name in arms.first { $0.arm == name }?.holderShare }.max()
            let reference = arms.first { roles[$0.arm] == .reference }
            let referenceMatches = best.map { best in reference.map { guards($0).isEmpty && ($0.holderShare ?? 0) >= best - 0.02 } ?? false } ?? false
            if let manner = arms.first(where: { roles[$0.arm] == .manner }) {
                let failed = guards(manner) + (mannerPasses ? [] : ["M1 to M4 do not all pass"])
                mannerQualifies = failed.isEmpty
                if !failed.isEmpty { failures[manner.arm] = failed }
            }
            if baseShare >= 0.90 {
                summary = String(format: "trajectory stays off: a closed question, the default gate already gives the holder %.3f", baseShare)
            } else if let best, referenceMatches {
                summary = String(format: "trajectory stays off: \"no agreement\" does as well (%.3f against the best arm's %.3f)",
                                 reference?.holderShare ?? 0, best)
            } else if let best {
                winner = qualifies.first { name in (arms.first { $0.arm == name }?.holderShare ?? 0) >= best - 0.02 }
                summary = String(format: "%@ wins: the holder's share %.3f, the default gate's %.3f", winner ?? "", best, baseShare)
            } else if !tracePasses {
                summary = "trajectory stays off: " + failedTrace.joined(separator: ", ") + " failed"
            } else {
                summary = "trajectory stays off: no arm lifts the holder's share by 0.10 within the guards"
            }
        } else if !tracePasses {
            summary = "trajectory stays off: " + failedTrace.joined(separator: ", ") + " failed, so no gate arm was run"
        } else {
            summary = "the trace rules pass; the gate arms were not run"
        }
        if let mannerQualifies {
            summary += mannerQualifies ? "; asking by manner qualifies" : "; asking by manner does not qualify"
        } else if !mannerPasses {
            summary += "; routing by manner fails " + rules.dropFirst(3).filter { !$0.passed }.map(\.rule).joined(separator: ", ")
        }
        return TrajectoryEvaluation(rules: rules, qualifies: qualifies, failures: failures, winner: winner, mannerQualifies: mannerQualifies,
                                    reported: reported, summary: summary)
    }
}

/// The bench's texts, built from a braid's world and what its nodes were fed.
public enum TrajectoryTexts {
    /// Prose about nothing any Thread holds.
    public static let generic = [
        """
        The tide came in slowly that afternoon, filling the channels between the sandbanks one by one. Gulls stood in a line \
        along the breakwater, facing the wind, and a single rowing boat worked its way across the harbour mouth. By the time \
        the lamps were lit on the quay the water had reached the steps, and the boats that had leaned all day on their keels \
        were floating again, turning gently on their moorings as if they had only been waiting for permission to move.
        """,
        """
        Bread wants patience more than skill. The dough is mixed in the evening and left in a cool corner, covered with a \
        cloth, while the yeast does its quiet work through the night. In the morning it has doubled and smells faintly sour. \
        It is folded, shaped and left to rise once more, then baked in the hottest oven the kitchen can manage, until the \
        crust sings as it cools on the rack and the whole house knows that breakfast is nearly ready.
        """,
        """
        A bicycle teaches its rider about hills in a way no map can. A slope that looks gentle on paper becomes a long \
        argument with the legs, and a descent that seemed steep turns out to be the best part of the day. Riders learn to \
        read the road ahead, to shift before the climb rather than halfway up it, and to save a little breath for the last \
        bend, where the gradient always seems to steepen just when the top comes into view.
        """,
        """
        In winter the garden keeps its secrets underground. The beds look bare, the branches are grey, and frost draws white \
        lines along every stem. Yet the bulbs planted in autumn are already sending roots into the cold soil, and the buds on \
        the apple tree are sealed tight against the weather, waiting. A patient gardener walks the paths on clear mornings, \
        notes what the wind has broken, and makes small plans for the first warm week of spring.
        """,
    ]

    /// A partition's sentences: split at a space after . ? ! or a closing quote, before a capital or a quote.
    static func sentences(_ text: String) -> [String] {
        let characters = Array(text)
        guard characters.count > 2 else { return text.isEmpty ? [] : [text] }
        var units: [String] = []
        var start = 0
        for i in 1..<(characters.count - 1) where characters[i] == " " {
            guard ".?!\"".contains(characters[i - 1]), characters[i + 1].isUppercase || characters[i + 1] == "\"" else { continue }
            units.append(String(characters[start..<i]))
            start = i + 1
        }
        units.append(String(characters[start...]))
        return units.filter { !$0.isEmpty }
    }

    /// A document's tokens as its Thread tokenized it (each partition on its own, nothing between),
    /// and its sentences' tokens; no sentences when re-joining them would not give those tokens.
    static func split(_ document: CorpusDocument, tokenizer: RaoTokenizer) -> (tokens: [Int], units: [[Int]]?) {
        var tokens: [Int] = []
        var units: [[Int]] = []
        var exact = true
        for partition in document.partitions.sorted(by: { $0.index < $1.index }) {
            let own = tokenizer.encode(partition.text)
            tokens += own
            let pieces = sentences(partition.text).enumerated().map { i, piece in tokenizer.encode(i == 0 ? piece : " " + piece) }
            if pieces.flatMap({ $0 }) == own { units += pieces } else { exact = false }
        }
        return (tokens, exact ? units : nil)
    }

    /// The sentences in an order where none is followed by the sentence that followed it before.
    static func shuffled(_ units: [[Int]], rng: inout SplitMix64) -> [[Int]]? {
        guard units.count >= 3 else { return nil }
        for _ in 0..<200 {
            let order = rng.shuffled(Array(units.indices))
            if zip(order, order.dropFirst()).allSatisfy({ $1 != $0 + 1 }) { return order.map { units[$0] } }
        }
        return nil
    }

    /// Where each located fact's answer sits in the document's tokens.
    static func answers(_ document: CorpusDocument, facts ids: Set<String>, tokenizer: RaoTokenizer) -> [TrajectoryAnswer] {
        let corpus = GeneratedCorpus(
            manifest: CorpusManifest(slug: "bench", generator: SyntheticCorpus.generatorName, generatorVersion: 1, seed: 0, documentCount: 1,
                                     partitionCount: 0, factCount: 0, chunkMaxChars: 600, chunkMinChars: 120, documentIDs: [document.id],
                                     corpusHash: ""),
            documents: [document])
        let tokenized = TokenizedCorpus(snapshot: .offline(corpus), tokenizer: tokenizer)
        var starts: [Int] = []
        var total = 0
        for partition in tokenized.partitions {
            starts.append(total)
            total += partition.tokens.count
        }
        return FactLocator.locate(document.facts.filter { ids.contains($0.id) }, corpus: tokenized, tokenizer: tokenizer).located
            .map { TrajectoryAnswer(start: starts[$0.row] + $0.answerToken, count: $0.answerLength, text: $0.fact.answer) }
            .sorted { $0.start < $1.start }
    }

    public static func build(
        world: MockWorld, layout: BraidLayout, names: [String], tokenizer: RaoTokenizer, sizes: TrajectoryBench.Sizes
    ) -> [TrajectoryText] {
        var rng = SplitMix64(seed: sizes.seed)
        var cache: [String: (tokens: [Int], units: [[Int]]?)] = [:]
        func pieces(_ document: CorpusDocument) -> (tokens: [Int], units: [[Int]]?) {
            if let cached = cache[document.id] { return cached }
            let made = split(document, tokenizer: tokenizer)
            cache[document.id] = made
            return made
        }
        let feeds = Dictionary(uniqueKeysWithValues: names.map { ($0, FeedState.load(layout.node($0))) })
        let whole = sizes.minTokens...sizes.maxTokens
        var texts: [TrajectoryText] = []
        for name in names {
            let present = Set(feeds[name]?.present ?? [])
            let deposited = Set(feeds[name]?.deposited ?? [])
            let own = world.exclusive(world.documents(for: name))
            let held = own.filter { present.contains($0.id) }
            for document in held.filter({ whole.contains(pieces($0).tokens.count) }).prefix(sizes.perNode) {
                texts.append(TrajectoryText(kind: .held, label: "\(name) · \(document.name)", tokens: pieces(document).tokens, owner: name,
                                            document: document.id))
                if let units = pieces(document).units, let order = shuffled(units, rng: &rng) {
                    texts.append(TrajectoryText(kind: .shuffled, label: "\(name) · \(document.name) (shuffled)", tokens: order.flatMap { $0 },
                                                owner: name, document: document.id))
                }
            }
            // Sentence k of the k-th held document: a Thread's own sentences, following none of them.
            let sources = held.compactMap { document in pieces(document).units.map { (document: document, units: $0) } }
            if sources.count >= 2 {
                for c in 0..<min(sizes.perNode, sources.count) {
                    var tokens: [Int] = []
                    var k = 0
                    while tokens.count < sizes.minTokens, k < sources.count {
                        let units = sources[(c + k) % sources.count].units
                        let unit = units[k % units.count]
                        if tokens.count + unit.count > sizes.maxTokens { break }
                        tokens += unit
                        k += 1
                    }
                    if tokens.count >= sizes.minTokens {
                        texts.append(TrajectoryText(kind: .collage, label: "\(name) · collage \(c + 1)", tokens: tokens, owner: name))
                    }
                }
            }
            let unfed = own.filter { !deposited.contains($0.id) && whole.contains(pieces($0).tokens.count) }
            for document in unfed.prefix(sizes.perNode) {
                texts.append(TrajectoryText(kind: .voice, label: "\(name) · \(document.name) (unfed)", tokens: pieces(document).tokens,
                                            owner: name, document: document.id))
                if let units = pieces(document).units, let order = shuffled(units, rng: &rng) {
                    texts.append(TrajectoryText(kind: .voiceShuffled, label: "\(name) · \(document.name) (unfed, shuffled)",
                                                tokens: order.flatMap { $0 }, owner: name, document: document.id))
                }
            }
        }
        // Both sides of every retelling whose two documents are fed.
        let retold = sizes.minRetold...sizes.maxTokens
        var sources = Set<String>()
        for link in world.crosslinks where link.kind != .homonym {
            guard feeds[link.source.node]?.present.contains(link.source.documentID) == true,
                  feeds[link.target.node]?.present.contains(link.target.documentID) == true,
                  let source = world.document(id: link.source.documentID), let target = world.document(id: link.target.documentID)
            else { continue }
            let agreeing = link.facts.filter(\.agrees)
            if retold.contains(pieces(target).tokens.count) {
                texts.append(TrajectoryText(
                    kind: .told, label: "\(link.target.node) ≈ \(link.source.node) · \(link.subject) (\(link.kind.rawValue))",
                    tokens: pieces(target).tokens, owner: link.target.node, near: link.source.node, document: target.id,
                    answers: answers(target, facts: Set(agreeing.map(\.targetFact)), tokenizer: tokenizer)))
            }
            if !sources.contains(source.id), retold.contains(pieces(source).tokens.count) {
                sources.insert(source.id)
                texts.append(TrajectoryText(
                    kind: .source, label: "\(link.source.node) → \(link.target.node) · \(link.subject)", tokens: pieces(source).tokens,
                    owner: link.source.node, near: link.target.node, document: source.id,
                    answers: answers(source, facts: Set(agreeing.map(\.sourceFact)), tokenizer: tokenizer)))
            }
        }
        for (i, passage) in generic.enumerated() {
            texts.append(TrajectoryText(kind: .generic, label: "generic \(i + 1)", tokens: tokenizer.encode(passage)))
        }
        return texts
    }
}
