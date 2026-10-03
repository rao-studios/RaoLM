//
//  HostedBench.swift
//  RaoLMBraid
//
//  WHAT: Whether a braid hosted over the network answers as the braid of child processes does
//        (Docs/ARCHITECTURE.md, "Step 2: nodes that dial in"): the same braid started twice, its
//        nodes once as children over pipes and once as processes that dial the umbrella over TCP,
//        asked the same fact prompts; every generated token and every Thread's share compared,
//        and what a generated token costs each way.
//  PIN:  The nodes run as processes either way (the generator's links are ProcessStrandLinks), so
//        the only difference is the transport. Nothing is fed or trained: the braid is used as built.
//

import Foundation
import RaoLMCore
import RaoLMModel
import RaoLMProvenance

public struct HostedReport: Codable, Sendable {
    public var createdAt: Date
    public var braid: String
    public var nodes: Int
    /// Fact prompts asked each way, and how many came out the same, token for token.
    public var prompts: Int
    public var identical: Int
    /// The largest difference in any Thread's share of any token, across the prompts both answered alike.
    public var maxShareDifference: Float
    public var firstDifference: String?
    public var pipes: ScaleTokenCost
    public var tcp: ScaleTokenCost
    /// Seconds per generated token over TCP against over pipes.
    public var ratio: Double
    public var evaluation: ScaleEvaluation
}

public enum HostedBench {
    public static func run(
        root: DataRoot, executable: URL, tokenizer: RaoTokenizer, factsPerNode: Int = 10, progress: ((String) -> Void)? = nil
    ) async throws -> HostedReport {
        var options = BraidOptions(root: root, executable: executable)
        options.offline = true
        options.adoptNodes = true
        options.syncOnStart = false
        // Nodes that only serve keep less freed memory for reuse than training needs: 24 node
        // processes at 2 GB each pushed the machine into swap (allocation only, never a number).
        options.settings.cacheLimitMB = 512
        options.restorePreset()
        let pack = try UmbrellaPacks.ensure(layout: options.layout, config: try options.settings.modelConfig(), tokenizer: tokenizer, pack: options.pack)
        guard pack.hasBase else { throw BraidSessionError.io("bench-hosted needs a braid on a pack with a base model") }

        struct Run {
            var generations: [CitedGeneration]
            var cost: ScaleTokenCost
            var nodes: Int
        }
        func ask(listen: Int?) async throws -> Run {
            var opts = options
            opts.listen = listen
            let session = try BraidSession(options: opts, vocabularySHA256: pack.vocabulary.sha256, packSHA256: pack.sha256) { _ in }
            do {
                try await session.start()
                let links = try session.links()
                guard links.count == session.names.count else {
                    throw BraidSessionError.io("\(links.count) of \(session.names.count) nodes are live: every node must be live")
                }
                let umbrella = try BraidUmbrella(pack: pack, tokenizer: tokenizer)
                let generator = try umbrella.generator(links: links)
                let nodes: [BraidExample.Node] = links.map { link in
                    (name: link.descriptor.name, label: link.descriptor.label, threadID: link.descriptor.threadID,
                     documents: session.world.exclusive(MockFeeder.present(node: link.descriptor.name, world: session.world,
                                                                            layout: options.layout.node(link.descriptor.name))))
                }
                let facts = BraidExample.facts(nodes: nodes, tokenizer: tokenizer, perNode: factsPerNode)
                var generations: [CitedGeneration] = []
                for example in facts {
                    var params = GenerationParameters(tapLayer: links[0].descriptor.tapLayer, alpha: links[0].descriptor.alpha)
                    params.maxTokens = max(1, tokenizer.encode(example.expected ?? " x").count) + 2
                    generations.append(try generator.generate(BraidRequest(promptTokens: example.promptTokens, promptText: example.promptText,
                                                                           params: params, gate: BraidRequest.defaultGate)))
                }
                let cost = try ScaleBench.tokenCost(generator: generator, links: links, prompts: BraidExample.generic(tokenizer: tokenizer),
                                                    gate: BraidRequest.defaultGate)
                await session.stop()
                return Run(generations: generations, cost: cost, nodes: links.count)
            } catch {
                await session.stop()
                throw error
            }
        }

        progress?("over pipes")
        let pipes = try await ask(listen: nil)
        progress?("over TCP, every node dialling in")
        let tcp = try await ask(listen: 0)
        let (identical, difference, first) = compare(pipes.generations, tcp.generations)
        let ratio = pipes.cost.secondsPerToken > 0 ? tcp.cost.secondsPerToken / pipes.cost.secondsPerToken : 0
        return HostedReport(
            createdAt: .wholeSecond(), braid: root.url.lastPathComponent, nodes: pipes.nodes, prompts: pipes.generations.count, identical: identical,
            maxShareDifference: difference, firstDifference: first, pipes: pipes.cost, tcp: tcp.cost, ratio: ratio,
            evaluation: evaluate(prompts: pipes.generations.count, identical: identical, maxShareDifference: difference, ratio: ratio))
    }

    /// Token for token, and every Thread's share of every token.
    static func compare(_ a: [CitedGeneration], _ b: [CitedGeneration]) -> (identical: Int, maxShareDifference: Float, first: String?) {
        var identical = 0
        var largest: Float = 0
        var first: String?
        for (x, y) in zip(a, b) {
            let tx = x.traces.filter { !$0.isPrompt }
            let ty = y.traces.filter { !$0.isPrompt }
            guard tx.map(\.token) == ty.map(\.token) else {
                if first == nil { first = "«\(x.prompt.text)»: «\(x.text)» over pipes, «\(y.text)» over TCP" }
                continue
            }
            identical += 1
            for (s, t) in zip(tx, ty) {
                for (p, q) in zip(s.strands ?? [], t.strands ?? []) { largest = max(largest, abs(p.share - q.share)) }
            }
        }
        if a.count != b.count, first == nil { first = "\(a.count) prompts over pipes, \(b.count) over TCP" }
        return (identical, largest, first)
    }

    /// H1 and H3 (H2 is the process test's: a node lost and found again).
    public static func evaluate(prompts: Int, identical: Int, maxShareDifference: Float, ratio: Double) -> ScaleEvaluation {
        let same = prompts > 0 && identical == prompts && maxShareDifference <= 1e-6
        let cheap = ratio > 0 && ratio <= 1.2 + 1e-9
        let rules = [
            TrajectoryRuleResult(rule: "H1 same answers", passed: same,
                                 detail: "\(identical) of \(prompts) fact prompts token for token; largest share difference \(String(format: "%.1e", maxShareDifference)) (bar 1e-6)"),
            TrajectoryRuleResult(rule: "H3 cost", passed: cheap,
                                 detail: String(format: "seconds per token over TCP %.2f× over pipes (bar 1.2×)", ratio)),
        ]
        let failed = rules.filter { !$0.passed }.map(\.rule)
        return ScaleEvaluation(rules: rules, qualifies: failed.isEmpty, secondsFit: nil, askedFit: nil,
                               summary: failed.isEmpty ? "the hosted braid answers as the braid of child processes, at the cost the rule allows"
                                   : "fails " + failed.joined(separator: ", "))
    }
}
