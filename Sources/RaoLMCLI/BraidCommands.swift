//
//  BraidCommands.swift
//  RaoLMCLI
//
//  WHAT: raolm braid — Thread nodes that each host a headless transformer (token embedding
//        through the last block), under one umbrella (the shared final norm, tied head, softmax
//        and the λ mix). Bare `raolm braid` opens the studio on the braid panel with its nodes
//        starting (ambient, craft and veil unless --nodes says otherwise); `demo` runs it
//        headless: start the nodes on their own processes, feed them, prompt them all, show what
//        each Thread supplied, update one, withdraw from another, move the lead to a third, and
//        ask about facts one Thread tells in its own words about another's entity.
//

import ArgumentParser
import Darwin
import Foundation
import RaoLM
import RaoLMStudio
import RaoLMWorkflows

struct BraidGroup: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "braid",
        abstract: "Run Thread nodes that each host a headless transformer, under one umbrella head.",
        discussion: """
            Each node is a `raolm node serve` process with a Thread of its own (HTTP from 8195, gRPC from 9195) and its own \
            versions under <data root>/braid/nodes/<name>. The umbrella gates the nodes by what each has been predicting, \
            asks the open ones for their last hidden state, applies the one shared final norm and tied head, and mixes the \
            Threads' own kNN-LMs. Every token records what each Thread supplied.
            """,
        subcommands: [Panel.self, Demo.self, Ask.self, List.self, Status.self, Down.self, Rebase.self, Sync.self, BenchVocabulary.self, BenchGate.self, BenchTrajectory.self, BenchUmbrella.self, BenchArchitecture.self, BenchThought.self, BenchQuestion.self, BenchCommons.self, BenchScale.self, BenchHosted.self, BenchRoute.self, Profile.self, BenchProfile.self],
        defaultSubcommand: Panel.self
    )

    struct Panel: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "panel",
            abstract: "Open the studio on the braid panel and start the nodes (the default).")

        @OptionGroup var global: GlobalOptions

        @Flag(help: "Stand-in directories instead of Thread processes (no Thread binary or embedding model needed).")
        var offline = false

        @Flag(help: "Wipe every node's storage, versions and feed first.")
        var fresh = false

        @Option(help: "Path to the thread binary.")
        var threadBinary: String?

        @Option(help: "Replay a recorded braid instead of running nodes (a fixtures directory with braid/braid-events.jsonl).")
        var fixtures: String?

        @Option(help: "Model preset each node trains: tiny, small, or base (SmolLM2-135M cut after block 20, its upper blocks the umbrella's; the braid keeps it).")
        var preset: String?

        @OptionGroup var world: BraidWorldOptions

        func run() async throws {
            try await guarded {
                guard Studio.isInteractive else {
                    throw RaoLMFailure("the braid panel needs an interactive terminal", hint: "raolm braid demo runs it headless", code: 64)
                }
                var options = StudioOptions(root: global.root, fixtures: fixtures.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) },
                                            threadBinary: threadBinary)
                options.executable = Bundle.main.executableURL?.resolvingSymlinksInPath()
                options.initialScreen = .braid
                options.braidAutoStart = true
                options.braidOffline = offline
                options.braidFresh = fresh
                var braid = BraidOptions(root: global.root, executable: options.executable ?? URL(fileURLWithPath: "/"))
                try world.apply(to: &braid)
                options.braidNodes = braid.nodes
                options.braidSeed = braid.seed
                options.braidWorld = world.choice
                options.braidPreset = preset
                let code = try await Studio.launch(options)
                if code != 0 { throw ExitCode(code) }
            }
        }
    }

    struct BraidOptionGroup: ParsableArguments {
        @Flag(help: "Stand-in directories instead of Thread processes (no Thread binary or embedding model needed).")
        var offline = false

        @Option(help: "Path to the thread binary (default: $RAOLM_THREAD_BINARY, else ../Thread/.build/release/thread).")
        var threadBinary: String?

        @Option(help: "Model preset each node trains: tiny, small, or base (SmolLM2-135M cut after block 20; default: the braid's own, else tiny).")
        var preset: String?

        @Option(help: "A training arm: passage-break (paragraph breaks in the stream, attention kept inside a document); canon, gated-attention, muon or wsd (the same, plus Canon layers, a gate on each attention head, the Muon optimizer, or a warmup-stable-decay schedule). Default: the braid's own, and passage-break for a new base braid.")
        var arm: String?

        @Option(help: "The umbrella pack to run on (a name or hash prefix from raolm umbrella list; default: the braid's own, else the current one).")
        var pack: String?

        @Option(help: "Documents a feed deposits.")
        var batch = 8

        @Flag(help: "Wipe every node's storage, versions and feed first.")
        var fresh = false

        @OptionGroup var world: BraidWorldOptions

        func options(root: DataRoot) throws -> BraidOptions {
            guard let executable = Bundle.main.executableURL?.resolvingSymlinksInPath() else { throw BraidSessionError.noExecutable }
            var options = BraidOptions(root: root, executable: executable)
            options.offline = offline
            options.threadBinary = threadBinary
            options.batch = batch
            options.fresh = fresh
            options.pack = pack
            if let preset {
                options.settings.preset = preset
                options.presetRequested = true
            }
            if let arm {
                guard HypervisorSettings.arms.contains(arm) else {
                    throw ValidationError("unknown arm \(arm); the arms are \(HypervisorSettings.arms.joined(separator: ", "))")
                }
                options.settings.arm = arm
                options.armRequested = true
            }
            try world.apply(to: &options)
            options.restorePreset()
            return options
        }
    }

    /// The braid's nodes and the mock world they are fed from.
    struct BraidWorldOptions: ParsableArguments {
        @Option(help: "The nodes, comma-separated (each name also its Thread's slug; default ambient,craft,veil, or the dataset's).")
        var nodes: String?

        @Option(help: "Feed the nodes from a braid dataset (a name in the datasets root, or a path) instead of generating a world.")
        var dataset: String?

        @Option(help: "Seed of the mock world.")
        var seed: UInt64 = 42

        var choice: BraidWorldChoice { BraidWorldChoice(nodes: nodes ?? "", dataset: dataset ?? "") }

        func apply(to options: inout BraidOptions) throws {
            try choice.apply(to: &options)
            options.seed = seed
        }
    }

    struct Demo: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Start the Thread nodes, feed them, prompt them through the umbrella, update one, withdraw from another, move the lead to a third.")

        @OptionGroup var global: GlobalOptions
        @OptionGroup var braid: BraidOptionGroup

        @Option(help: "Facts to prompt per node.")
        var prompts = 3

        @Option(help: "How the umbrella weighs Threads: braided, posterior, retrieval.")
        var gating: BraidGating = BraidRequest.defaultGating

        @Option(help: "Record every event to <dir>/braid-events.jsonl (the studio's fixtures replay it).")
        var record: String?

        @Flag(help: "Leave the nodes running at the end.")
        var keep = false

        func run() async throws {
            try await guarded {
                try Preflight.requireMetallib()
                let options = try braid.options(root: global.root)
                let tokenizer = try await RaoTokenizer.load()
                let pack = try UmbrellaPacks.ensure(layout: options.layout, config: try options.settings.modelConfig(), tokenizer: tokenizer, pack: options.pack) {
                    print("umbrella pack: \($0)")
                }
                let recorder = try record.map { try BraidRecorder(directory: URL(fileURLWithPath: $0)) }
                defer { recorder?.close() }
                let printer = BraidPrinter()
                let session = try BraidSession(
                    options: options, vocabularySHA256: pack.vocabulary.sha256, packSHA256: pack.hasBase ? pack.sha256 : nil
                ) { event in
                    recorder?.record(event)
                    printer.handle(event)
                }
                session.exampleTokenizer = tokenizer
                let umbrella = try BraidUmbrella(pack: pack, tokenizer: tokenizer)
                let stop = StopSignal()

                Console.section("Nodes")
                if let cut = pack.cut {
                    print("umbrella pack \(Format.short(pack.sha256)) (\(pack.name): blocks \(cut)+ the umbrella's frozen trunk, the commons strand asked every token) · \(options.offline ? "offline" : "Thread") mode")
                } else {
                    print("vocabulary \(Format.short(pack.vocabulary.sha256)) (seeded, head scale \(BraidVocabulary.headScale)) · \(options.offline ? "offline" : "Thread") mode")
                }
                do {
                    try await steps(session: session, umbrella: umbrella, tokenizer: tokenizer, recorder: recorder, stop: stop)
                } catch {
                    await session.stop()
                    throw error
                }
            }
        }

        func steps(
            session: BraidSession, umbrella: BraidUmbrella, tokenizer: RaoTokenizer, recorder: BraidRecorder?, stop: StopSignal
        ) async throws {
            try await session.start()
            print(Format.table(BraidTables.nodes(session.states, session: session)))
            let names = session.names
            // Share columns: every Thread, and the commons when the pack has one.
            let columns = names + (umbrella.commons == nil ? [] : [BraidStrandRef.commonsName])

            Console.section("Feed")
            let before = Dictionary(uniqueKeysWithValues: names.map { ($0, session.state($0)?.versions ?? 0) })
            for name in names {
                let documents = try await session.feed(name)
                print("\(name): deposited \(documents.count) documents into its Thread (\(documents.prefix(3).map(\.name).joined(separator: ", "))…)")
            }
            try await BraidWait.settled(session, after: before, stop: stop)
            print(Format.table(BraidTables.nodes(session.states, session: session)))

            Console.section("Prompts through the umbrella")
            print("gate: \(gating.rawValue) · " + BraidTables.gateNote(gating))
            let examples = session.examples(tokenizer: tokenizer)
            let chosen = BraidDemoPicks.pick(examples, perNode: prompts, names: names)
            session.probe(tokens: chosen.first?.promptTokens ?? [])
            var rows: [[String]] = []
            for example in chosen {
                let generation = try generate(example, umbrella: umbrella, session: session, recorder: recorder)
                var verified = generation
                let report = try await CitationVerifier.verify(&verified, reader: session.reader(), tokenizer: tokenizer)
                recorder?.record(.verified(lines: report.checks.map { "\($0.status.rawValue) \($0.documentID)" }, allVerified: report.allVerified))
                rows.append(BraidTables.route(example, verified, report: report, names: columns))
            }
            print(Format.table(BraidTables.routeHeaders(columns), rows))
            print("  share = the fraction of p(token) each Thread supplied, averaged over the answer's tokens\(umbrella.commons == nil ? "" : " (the commons' share is nobody's)"); gate = each Thread's weight at the first answer token (● asked, ○ not)")

            if let first = names.first, !stop.isSet {
                Console.section("Update: feed \(first) again")
                let mark = Dictionary(uniqueKeysWithValues: names.map { ($0, session.state($0)?.versions ?? 0) })
                let documents = try await session.feed(first, count: 2)
                if documents.isEmpty {
                    print("\(first): every document of its shard is already in its Thread, so there is nothing new to learn")
                } else {
                    print("\(first): deposited \(documents.map(\.name).joined(separator: ", "))")
                    try await BraidWait.settled(session, after: mark.filter { $0.key == first }, stop: stop)
                }
                let fresh = session.examples(tokenizer: tokenizer).filter { example in
                    example.node == first && documents.contains { $0.id == example.source?.documentID }
                }.prefix(2)
                var rows: [[String]] = []
                for example in fresh {
                    let generation = try generate(example, umbrella: umbrella, session: session, recorder: recorder)
                    var verified = generation
                    let report = try await CitationVerifier.verify(&verified, reader: session.reader(), tokenizer: tokenizer)
                    rows.append(BraidTables.route(example, verified, report: report, names: columns))
                }
                if !rows.isEmpty { print(Format.table(BraidTables.routeHeaders(columns), rows)) }
            }

            if names.count > 1, !stop.isSet {
                let name = names[1]
                Console.section("Withdraw from \(name)")
                let mark = [name: session.state(name)?.versions ?? 0]
                let withdrawnExample = session.examples(tokenizer: tokenizer).first { $0.node == name && $0.resolvedKind == .fact }
                if let document = try await session.withdraw(name) {
                    print("\(name): withdrew \(document.name)")
                    try await BraidWait.settled(session, after: mark, stop: stop)
                    let live = session.state(name)?.liveVersion.map { "v\($0)" } ?? "none"
                    print("\(name): live \(live) · the index holds \(session.state(name)?.partitions ?? 0) partitions")
                    if let example = withdrawnExample, example.source?.documentID == document.id {
                        let generation = try generate(example, umbrella: umbrella, session: session, recorder: recorder)
                        let cites = generation.traces.flatMap(\.citations).contains { $0.address.documentID == document.id }
                        print("  prompt about it now: \(Format.clip(generation.text, 40))  cites the withdrawn document: \(cites ? "yes" : "no")")
                    }
                }
            }

            if names.count > 2, !stop.isSet {
                let (opener, owner) = (names[1], names[2])
                Console.section("Move the lead: \(opener)'s fact, then \(owner)'s")
                let moves = session.examples(tokenizer: tokenizer).filter { $0.resolvedKind == .pair && $0.opener == opener && $0.node == owner }
                var rows: [[String]] = []
                for example in moves.prefix(2) {
                    let generation = try generate(example, umbrella: umbrella, session: session, recorder: recorder)
                    var verified = generation
                    let report = try await CitationVerifier.verify(&verified, reader: session.reader(), tokenizer: tokenizer)
                    rows.append(BraidTables.route(example, verified, report: report, names: columns))
                }
                if rows.isEmpty {
                    print("no example opens with \(opener)'s fact and asks \(owner)'s yet")
                } else {
                    print(Format.table(BraidTables.routeHeaders(columns), rows))
                }
            }

            let retold = session.examples(tokenizer: tokenizer).filter { $0.resolvedKind == .crossed }
            if !retold.isEmpty, !stop.isSet {
                Console.section("Told in other words: facts a Thread states about another Thread's entity")
                var rows: [[String]] = []
                for example in retold.prefix(max(1, prompts)) {
                    let generation = try generate(example, umbrella: umbrella, session: session, recorder: recorder)
                    var verified = generation
                    let report = try await CitationVerifier.verify(&verified, reader: session.reader(), tokenizer: tokenizer)
                    rows.append(BraidTables.route(example, verified, report: report, names: columns))
                }
                print(Format.table(BraidTables.routeHeaders(columns), rows))
                print("  each prompt is asked in the words of the Thread named first; the Thread after ≈ holds the same facts in its own words")
            }

            if keep {
                print("\nnodes left running: raolm braid status · raolm braid down")
            } else {
                await session.stop()
                let alive = session.states.compactMap { state -> String? in
                    let pids = [state.pid, state.threadPID].compactMap { $0 }.filter { kill($0, 0) == 0 }
                    return pids.isEmpty ? nil : "\(state.name) \(pids)"
                }
                print(alive.isEmpty ? "\nall node and Thread processes stopped" : "\nstill running: \(alive.joined(separator: ", "))")
            }
        }

        func generate(_ example: BraidExample, umbrella: BraidUmbrella, session: BraidSession, recorder: BraidRecorder?) throws -> CitedGeneration {
            let links = try session.links()
            var params = GenerationParameters(tapLayer: links[0].descriptor.tapLayer, alpha: links[0].descriptor.alpha)
            params.maxTokens = max(6, (example.expected.map { umbrella.tokenizer.encode($0).count } ?? 8) + 4)
            recorder?.record(.generating(prompt: example.promptText, nodes: links.map(\.descriptor.name)))
            let first = umbrella.tokenizer.tokenText(example.promptTokens.first ?? 0)
            let generation = try umbrella.generate(
                links: links,
                request: BraidRequest(promptTokens: example.promptTokens, promptText: example.promptText, promptSource: example.source,
                                      params: params, gating: gating),
                onPrompt: { traces in recorder?.record(.prompted(first: first, traces: traces)) }
            ) { step in
                recorder?.record(.token(BraidTokenEvent(trace: step.trace, open: step.open)))
            }
            recorder?.record(.generated(generation))
            return generation
        }
    }

    struct Ask: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Ask the braid a question: the commons rewrites it into a stem the Threads complete, and every token says who answered.",
            discussion: """
                Runs the braid's live nodes in this process (no node processes). The umbrella's question adapter turns the question into \
                a corpus-style stem with the commons model, prompted with public examples; --rewriter rules uses the dataset's own \
                question templates instead, and none asks the question as written. The answer stops at the end of its sentence.
                """)

        @OptionGroup var global: GlobalOptions

        @Argument(help: "The question.")
        var question: String

        @Option(help: "commons (default), rules or none.")
        var rewriter: String = "commons"

        @Option(help: "Most tokens of answer.")
        var maxTokens: Int = 24

        @Flag(inversion: .prefixedNo, help: "Each Thread completes the stem behind the sentence its own index recognises it in.")
        var context = true

        @Option(help: "Save the generation record to this JSON file.")
        var out: String?

        func run() async throws {
            try await guarded {
                try Preflight.requireMetallib()
                let tokenizer = try await RaoTokenizer.load()
                let layout = BraidLayout(dataRoot: global.root)
                let (pack, strands) = try RoutingBench.packStrands(layout: layout, tokenizer: tokenizer, owner: "raolm-braid")
                guard !strands.isEmpty else { throw BraidSessionError.noLiveNodes }
                let (umbrella, links, generator) = try RoutingBench.umbrella(pack: pack, strands: strands, tokenizer: tokenizer)
                let rewrite: QuestionRewrite
                switch rewriter {
                case "commons":
                    guard let commons = umbrella.commons else {
                        throw RaoLMFailure("the braid's pack has no commons model to rewrite with", hint: "use --rewriter rules", code: 64)
                    }
                    rewrite = QuestionAdapter(model: commons.model, tokenizer: tokenizer).rewrite(question, maxTokens: maxTokens)
                case "rules":
                    rewrite = QuestionAdapter.fallback(question)
                case "none":
                    rewrite = QuestionRewrite(question: question, stem: question, rewriter: "none")
                default:
                    throw ValidationError("unknown rewriter \(rewriter): commons, rules or none")
                }
                var params = GenerationParameters(tapLayer: links[0].descriptor.tapLayer, alpha: links[0].descriptor.alpha)
                params.maxTokens = maxTokens
                let stemTokens = QuestionAdapter.stemTokens(rewrite.stem, tokenizer: tokenizer)
                let generation = try generator.generate(BraidRequest(
                    promptTokens: stemTokens, promptText: rewrite.stem, params: params, stopAtSentenceEnd: true, question: rewrite,
                    context: context, subject: QuestionAdapter.subjectTokens(stem: rewrite.stem, tokens: stemTokens, question: question, tokenizer: tokenizer)))
                let alone = umbrella.commons.map { commons in
                    (name: commons.packName, text: commons.complete(stemTokens, maxTokens: maxTokens, tokenizer: tokenizer, stopAtSentenceEnd: true))
                }
                print(BraidTables.answer(generation, names: generator.names, commons: alone))
                if let out {
                    try JSONCoding.write(generation, to: URL(fileURLWithPath: (out as NSString).expandingTildeInPath))
                    print("record: \(out)")
                }
            }
        }
    }

    struct Sync: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Start the nodes, bring every node without a live version up to date (a rebased node retrains from the new base), and stop.",
            discussion: """
                Without --feed nothing is fed and nothing is withdrawn: each node trains on the documents its Thread already holds. \
                --feed n first deposits each node's next n documents of its world (a fresh braid built from a dataset, with no \
                demo's later changes). --at-once caps how many nodes train at a time, so a braid of many Threads fits in memory; \
                the peak memory of every node process is written to <data root>/braid/sync-footprint.json.
                """)

        @OptionGroup var global: GlobalOptions
        @OptionGroup var braid: BraidOptionGroup

        @Option(help: "Deposit each node's next N documents of its world before training (0: none).")
        var feed = 0

        @Option(help: "Nodes training at once (0: all).")
        var atOnce = 3

        func run() async throws {
            try await guarded {
                try Preflight.requireMetallib()
                var options = try braid.options(root: global.root)
                options.syncOnStart = false
                let tokenizer = try await RaoTokenizer.load()
                let pack = try UmbrellaPacks.ensure(layout: options.layout, config: try options.settings.modelConfig(), tokenizer: tokenizer, pack: options.pack) {
                    print("umbrella pack: \($0)")
                }
                let session = try BraidSession(options: options, vocabularySHA256: pack.vocabulary.sha256, packSHA256: pack.hasBase ? pack.sha256 : nil) { _ in }
                let stop = StopSignal()
                let started = Date()
                do {
                    try await session.start()
                    print("umbrella pack \(Format.short(pack.sha256)) (\(pack.name)) · \(session.names.count) nodes")
                    var fed: [String] = []
                    if feed > 0 {
                        for name in session.names {
                            let documents = try await session.feed(name, count: feed, sync: false)
                            if !documents.isEmpty { fed.append(name) }
                        }
                        print("fed \(fed.count) nodes \(feed) documents each")
                    }
                    let targets = session.names.filter { fed.contains($0) || session.state($0)?.liveVersion == nil }
                    if targets.isEmpty {
                        print("every node is live on this pack: nothing to do")
                    } else {
                        for name in targets {
                            if let from = session.state(name)?.rebasedFrom { print("\(name): rebased from \(from.prefix(12)), retraining") }
                        }
                        var reported: [String: String] = [:]
                        try await session.sync(targets, atOnce: atOnce, cancelled: { stop.isSet }, sent: { name, number, inFlight in
                            print("\(name): sync \(number) of \(targets.count), \(inFlight) in flight")
                        }, progress: { state in
                            let line = BraidPrinter.progress(state)
                            if reported[state.name] != line {
                                reported[state.name] = line
                                print("  \(line)")
                            }
                        })
                    }
                    print(Format.table(BraidTables.nodes(session.states, session: session)))
                    var footprint: [String: Int] = [:]
                    for state in session.states {
                        if let pid = state.pid, let bytes = BraidSession.peakFootprint(pid: pid) { footprint[state.name] = bytes }
                    }
                    if !footprint.isEmpty {
                        try JSONCoding.write(footprint, to: options.layout.root.appendingPathComponent("sync-footprint.json"))
                        let gb = footprint.values.map { Double($0) / 1e9 }
                        print(String(format: "peak memory per node process: %.1f–%.1f GB (sum %.0f GB) · %@", gb.min() ?? 0, gb.max() ?? 0,
                                     gb.reduce(0, +), Format.duration(Date().timeIntervalSince(started))))
                    }
                } catch {
                    await session.stop()
                    throw error
                }
                await session.stop()
            }
        }
    }

    struct Rebase: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Move the braid onto another umbrella pack; its nodes retrain from the new base when it next starts.",
            discussion: """
                Rebase, never migrate: a node's blocks run under the pack's trunk, so a node trained under one pack does not serve \
                under another. The braid's documents, feeds and history stay; each node's live version is retired when the braid \
                starts on the new pack, and the node trains fresh blocks from the new base. The pack must fit the braid's shape \
                (raolm umbrella list). Refused while the braid runs.
                """)

        @OptionGroup var global: GlobalOptions

        @Option(help: "The pack to run on: a name or hash prefix from raolm umbrella list.")
        var pack: String

        func run() async throws {
            try await guarded {
                let layout = BraidLayout(dataRoot: global.root)
                guard let record = MockWorld.Record.load(layout) else {
                    throw RaoLMFailure("no braid under \(layout.root.path)", hint: "raolm braid demo starts one", code: 66)
                }
                var settings = HypervisorSettings()
                settings.preset = record.preset ?? "tiny"
                let config = try settings.modelConfig()
                guard config.hasTrunk else { throw RaoLMFailure("the braid's preset (\(settings.preset)) has no trunk, so no pack to rebase onto", code: 64) }
                guard let entry = PackRegistry.load(layout).resolve(pack) else {
                    throw RaoLMFailure("no pack '\(pack)' under \(layout.packs.path)", hint: "raolm umbrella list", code: 66)
                }
                guard entry.slot == PackRegistry.slot(config) else {
                    throw RaoLMFailure("pack \(entry.sha256.prefix(12)) is \(entry.slot); the braid is \(PackRegistry.slot(config))", code: 64)
                }
                let previous = record.packSHA256
                let nodes = try BraidSession.rebase(layout: layout, pack: entry.sha256)
                print("braid \(layout.root.path): pack \(previous.map { String($0.prefix(12)) } ?? "unrecorded") → \(entry.sha256.prefix(12)) (\(entry.name))")
                for node in nodes {
                    let retired = node.version.map { version in
                        node.pack == entry.sha256 ? "v\(version) stays (already on it)" : "v\(version) retires (trained on \(node.pack.map { String($0.prefix(12)) } ?? "?"))"
                    } ?? "no live version"
                    print("  \(node.node.padding(toLength: 10, withPad: " ", startingAt: 0)) \(retired)")
                }
                print("\nraolm braid sync --data-dir \(global.root.url.path) (with the braid's --offline and --dataset) retrains every node from the new base.")
            }
        }
    }

    struct Status: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "The braid's nodes: processes, live versions, what they hold.")

        @OptionGroup var global: GlobalOptions

        func run() async throws {
            try await guarded {
                let layout = BraidLayout(dataRoot: global.root)
                let names = ((try? FileManager.default.contentsOfDirectory(atPath: layout.nodes.path)) ?? []).sorted()
                guard !names.isEmpty else {
                    print("no braid nodes under \(layout.nodes.path) (raolm braid demo starts them)")
                    return
                }
                var rows: [[String]] = []
                for name in names {
                    let node = layout.node(name)
                    let record = NodeRecord.load(node)
                    let running = record.map { kill($0.pid, 0) == 0 } ?? false
                    let live = try? JSONCoding.read(LivePointer.self, from: node.live)
                    let version = live.flatMap { try? JSONCoding.read(NodeVersion.self, from: node.version($0.version).appendingPathComponent(NodeVersion.fileName)) }
                    rows.append([
                        name, running ? "● pid \(record!.pid)" : "○ down",
                        record?.threadPID.map { "\($0)" } ?? (record?.offline == true ? "offline" : "—"),
                        record?.httpPort.map { ":\($0)/:\(record?.grpcPort ?? 0)" } ?? "—",
                        version.map { "v\($0.version) \($0.kind.rawValue)" } ?? "—",
                        "\(node.versionNumbers().count)", version.map { "\($0.documents)" } ?? "0", version.map { Format.pct($0.memorised) } ?? "—",
                        record?.threadID.map { String($0.prefix(8)).lowercased() } ?? "—",
                    ])
                }
                print(Format.table(["node", "process", "thread pid", "ports", "live", "versions", "docs", "memorised", "thread id"], rows))
            }
        }
    }

    struct List: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Every braid on this machine: the data root's, then each one in the T9 work area, as the studio lists them.")

        @OptionGroup var global: GlobalOptions

        func run() async throws {
            try await guarded {
                let entries = BraidCatalog.scan(home: global.root.url, areas: DatasetsRoot.braidAreas)
                guard !entries.isEmpty else {
                    print("no braid under \(global.root.url.path) or \(DatasetsRoot.workArea) (raolm braid demo starts one)")
                    return
                }
                let date = DateFormatter()
                date.dateFormat = "yyyy-MM-dd HH:mm"
                print(Format.table(["braid", "Threads", "model", "commons", "fed from", "mode", "last live", "where"], entries.map { entry in
                    [entry.name, entry.threads, entry.model, entry.commons ?? "—", entry.dataset ?? "mock world", entry.offline ? "offline" : "Threads",
                     entry.updated.map { date.string(from: $0) } ?? "—", entry.root.path]
                }))
            }
        }
    }

    struct Down: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Stop every braid node and its Thread left running.")

        @OptionGroup var global: GlobalOptions

        func run() async throws {
            try await guarded {
                let layout = BraidLayout(dataRoot: global.root)
                let names = ((try? FileManager.default.contentsOfDirectory(atPath: layout.nodes.path)) ?? []).sorted()
                var stopped = 0
                for name in names {
                    let node = layout.node(name)
                    let wasRunning = NodeRecord.load(node).map { kill($0.pid, 0) == 0 } ?? false
                    BraidSession.stopStale(node)
                    let thread = await ThreadHost.stopRecorded(dataDirectory: node.threadDB)
                    if wasRunning || thread {
                        stopped += 1
                        print("stopped \(name)")
                    }
                }
                print(stopped == 0 ? "no braid node was running" : "\(stopped) node(s) stopped")
            }
        }
    }

    struct BenchVocabulary: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "bench-vocabulary",
            abstract: "Measure blocks-only training against a frozen vocabulary, and choose the braid's vocabulary.",
            shouldDisplay: false)

        @OptionGroup var global: GlobalOptions

        @Option(help: "Model preset: tiny, small, smollm2-135m.")
        var preset = "tiny"

        @Option(help: "Documents in the corpus under test.")
        var documents = 24

        @Option(help: "Documents in the commons world the trained vocabulary comes from.")
        var commonsDocuments = 48

        @Option(help: "Epoch budget of every arm.")
        var epochs = 120

        @Option(help: "Sequences per optimizer step.")
        var batchSize = 4

        @Option(help: "Tokens per sequence.")
        var seqLen = 256

        @Option(help: "Peak learning rate.")
        var lr: Float = 2e-3

        @Option(help: "Eval pass every N epochs.")
        var evalEvery = 4

        @Option(help: "Memorised fraction an arm has to reach.")
        var target: Float = 0.97

        @Option(help: "Head scales of the seeded arms, comma-separated.")
        var headScales = "1,1.5,2"

        @Option(help: "Facts to evaluate per arm.")
        var factsSample = 40

        @Option(help: "Seed.")
        var seed: UInt64 = 42

        @Option(help: "Arms to train, comma-separated: baseline, seeded, commons.")
        var arms = "baseline,seeded,commons"

        @Option(help: "An earlier vocabulary-bench.json whose rows stand in for the arms not trained again.")
        var reuse: String?

        @Option(help: "Where the arms' runs go (default: <data root>/braid/bench).")
        var out: String?

        @Flag(help: "Print the report as JSON.")
        var json = false

        func run() async throws {
            try await guarded {
                try Preflight.requireMetallib()
                let scales = headScales.split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
                guard !scales.isEmpty else { throw RaoLMFailure("--head-scales names no number", code: 64) }
                let directory = out.map { URL(fileURLWithPath: $0) }
                    ?? global.root.url.appendingPathComponent("braid/bench", isDirectory: true)
                var options = VocabularyBenchOptions(directory: directory)
                options.preset = preset
                options.documents = documents
                options.commonsDocuments = commonsDocuments
                options.epochs = epochs
                options.batchSize = batchSize
                options.seqLen = seqLen
                options.lr = lr
                options.evalEvery = evalEvery
                options.target = target
                options.headScales = scales
                options.factsSample = factsSample
                options.seed = seed
                options.arms = Set(arms.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) })
                if let reuse {
                    options.reuse = try JSONCoding.read(VocabularyBenchReport.self, from: URL(fileURLWithPath: reuse)).rows
                }
                let tokenizer = try await RaoTokenizer.load()
                let quiet = json
                let report = try await VocabularyBench.run(options, tokenizer: tokenizer) { line in
                    if !quiet { print(line) }
                }
                if json {
                    print(String(decoding: try JSONCoding.prettyEncoder().encode(report), as: UTF8.self))
                    return
                }
                Console.section("Vocabulary")
                print(Format.table(BraidTables.bench(report)))
                print("")
                print("choice: \(report.choice.rawValue)" + (report.headScale.map { " (head scale \($0))" } ?? ""))
                print("  \(report.reason)")
                print("  report: \(directory.appendingPathComponent(VocabularyBenchReport.fileName).path)")
            }
        }
    }
}

extension BraidGating: ExpressibleByArgument {}

extension BraidGroup {
    struct BenchGate: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "bench-gate",
            abstract: "Ask every live node's prompts under each way of weighing Threads, and apply the rule that picks the default.",
            shouldDisplay: false,
            aliases: ["bench-routing"])

        @OptionGroup var global: GlobalOptions

        @Option(help: "Fact prompts per node.")
        var factsPerNode = 10

        @Option(help: "Two-fact prompts per ordered pair of nodes.")
        var pairs = 10

        @Option(help: "Prompts about subjects no Thread holds.")
        var unknown = 10

        @Option(help: "Reworded fact prompts (reported only).")
        var paraphrases = 10

        @Option(help: "Seed of the mock world the nodes were fed from, for braids made before world.json.")
        var seed: UInt64 = 42

        @Option(help: "Only these arms, comma-separated (the baseline, posterior, always runs).")
        var arms: String?

        @Option(help: "Write the report, every row included, to this JSON file.")
        var out: String?

        func run() async throws {
            try await guarded {
                try Preflight.requireMetallib()
                let layout = BraidLayout(dataRoot: global.root)
                let tokenizer = try await RaoTokenizer.load()
                let world = try RoutingBench.world(layout: layout, seed: seed)
                if let record = MockWorld.Record.load(layout) { Console.error("world: \(record.summary)") }
                var sizes = RoutingBench.Sizes()
                sizes.factsPerNode = factsPerNode
                sizes.pairsPerOrder = pairs
                sizes.unknown = unknown
                sizes.paraphrases = paraphrases
                var chosen = GateArm.all
                if let arms {
                    let wanted = Set(arms.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) } + ["posterior"])
                    chosen = chosen.filter { wanted.contains($0.name) }
                }
                let started = Date()
                let report = try RoutingBench.run(layout: layout, tokenizer: tokenizer, world: world, sizes: sizes, arms: chosen) { line in
                    Console.error(line)
                }
                print(BraidTables.gateBench(report))
                print(String(format: "\n%d generations in %.0f s", report.rows.count, Date().timeIntervalSince(started)))
                if let out {
                    try JSONCoding.write(report, to: URL(fileURLWithPath: out))
                    print("report: \(out)")
                }
            }
        }
    }
}

extension BraidGroup {
    struct BenchTrajectory: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "bench-trajectory",
            abstract: "Whether each Thread's retrieval trajectory tells its own text from a pastiche of it, and whether the trace in the gate earns the holder its tokens.",
            discussion: """
                Run it on a braid fed from a dataset (raolm braid demo --offline --fresh --dataset braid-cross-v1 --batch 80). Whole \
                documents are scored as written, with their sentences reordered, stitched from several documents, unfed, and on \
                both sides of a retelling. The gate arms run only when the trace rules pass (--arms always runs them anyway).
                """,
            shouldDisplay: false)

        @OptionGroup var global: GlobalOptions

        @Option(help: "Held, collage and voice texts per node.")
        var perNode = 20

        @Option(help: "Seed of the shuffles.")
        var seed: UInt64 = 42

        @Option(help: "Gate arms: auto (only when T1 to T3 pass), always, never.")
        var arms = "auto"

        @Option(help: "Write the report, every text's readings included, to this JSON file.")
        var out: String?

        @Option(help: "Print a saved report, scored again under the rule, instead of running.")
        var report: String?

        func run() async throws {
            try await guarded {
                if let report {
                    var loaded = try JSONCoding.read(TrajectoryReport.self, from: URL(fileURLWithPath: (report as NSString).expandingTildeInPath))
                    loaded.evaluation = TrajectoryBench.evaluate(texts: loaded.texts, arms: loaded.arms)
                    print(BraidTables.trajectory(loaded))
                    return
                }
                guard let mode = TrajectoryBench.Arms(rawValue: arms) else { throw RaoLMFailure("--arms takes auto, always or never", code: 64) }
                try Preflight.requireMetallib()
                let layout = BraidLayout(dataRoot: global.root)
                let tokenizer = try await RaoTokenizer.load()
                var sizes = TrajectoryBench.Sizes()
                sizes.perNode = perNode
                sizes.seed = seed
                if let record = MockWorld.Record.load(layout) { Console.error("world: \(record.summary)") }
                let started = Date()
                let result = try TrajectoryBench.run(layout: layout, tokenizer: tokenizer, sizes: sizes, arms: mode) { line in Console.error(line) }
                print(BraidTables.trajectory(result))
                print(String(format: "\n%d texts in %.0f s", result.texts.count, Date().timeIntervalSince(started)))
                if let out {
                    try JSONCoding.write(result, to: URL(fileURLWithPath: out))
                    print("report: \(out)")
                }
            }
        }
    }
}

extension BraidGroup {
    struct BenchUmbrella: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "bench-umbrella",
            abstract: "Whether the umbrella's layers earn their place: the pack, the commons strand, each Thread's own λ and τ, thought agreement.",
            discussion: """
                The base braid is --data-dir: nodes of the base preset fed from a dataset (raolm braid demo --offline --fresh \
                --dataset braid-cross-v1 --batch 80 --preset base). --reference is the same braid on the tiny preset, fed the same way; \
                --ceiling, the base braid fed twice the documents. Five arms, each adding to the one before (v1, pack, commons, \
                calibrated, anchors), answer the same prompts; the rules were fixed before any numbers (Docs/ARCHITECTURE.md).
                """,
            shouldDisplay: false)

        @OptionGroup var global: GlobalOptions

        @Option(help: "Data root of the reference braid (the tiny preset, fed the same way).")
        var reference: String?

        @Option(help: "Data root of the base braid fed twice the documents.")
        var ceiling: String?

        @Option(help: "Write the report to this JSON file.")
        var out: String?

        @Option(help: "Print a saved report, scored again under the rule, instead of running.")
        var report: String?

        func run() async throws {
            try await guarded {
                if let report {
                    var loaded = try JSONCoding.read(UmbrellaReport.self, from: URL(fileURLWithPath: (report as NSString).expandingTildeInPath))
                    loaded.evaluation = UmbrellaBench.evaluate(arms: loaded.arms, ceiling: loaded.ceiling, nodes: loaded.nodes, trajectory: loaded.trajectory)
                    loaded.allFactsEvaluation = UmbrellaBench.evaluateEveryFact(loaded)
                    print(BraidTables.umbrella(loaded))
                    return
                }
                guard let reference else { throw RaoLMFailure("bench-umbrella needs --reference, the tiny braid fed the same way", code: 64) }
                try Preflight.requireMetallib()
                let tokenizer = try await RaoTokenizer.load()
                let started = Date()
                let result = try UmbrellaBench.run(
                    base: BraidLayout(dataRoot: global.root), reference: BraidLayout(dataRoot: DataRoot.resolve(argument: reference)),
                    ceiling: ceiling.map { BraidLayout(dataRoot: DataRoot.resolve(argument: $0)) }, tokenizer: tokenizer
                ) { line in Console.error(line) }
                print(BraidTables.umbrella(result))
                print(String(format: "\n%.0f s", Date().timeIntervalSince(started)))
                if let out {
                    try JSONCoding.write(result, to: URL(fileURLWithPath: out))
                    print("report: \(out)")
                }
            }
        }
    }
}

extension BraidGroup {
    struct BenchArchitecture: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "bench-architecture",
            abstract: "Whether a phase-2 training arm earns its place in base: the arm's braid against today's base braid.",
            discussion: """
                The arm's braid is --data-dir: base nodes fed from a dataset and trained on an arm (raolm braid demo --offline --fresh \
                --dataset braid-cross-v1 --batch 80 --preset base --arm passage-break). --reference is today's base braid, fed the same \
                way. Rules A1 to A5 were fixed before any numbers (Docs/ARCHITECTURE.md).
                """,
            shouldDisplay: false)

        @OptionGroup var global: GlobalOptions

        @Option(help: "Data root of the reference braid (the base preset on today's recipe, fed the same way).")
        var reference: String?

        @Option(help: "Write the report to this JSON file.")
        var out: String?

        @Option(help: "Print a saved report, scored again under the rules, instead of running.")
        var report: String?

        func run() async throws {
            try await guarded {
                if let report {
                    var loaded = try JSONCoding.read(ArchitectureReport.self, from: URL(fileURLWithPath: (report as NSString).expandingTildeInPath))
                    loaded.evaluation = ArchitectureBench.evaluate(braids: loaded.braids, nodes: loaded.nodes, trajectory: loaded.trajectory, arm: loaded.arm)
                    print(BraidTables.architecture(loaded))
                    return
                }
                guard let reference else { throw RaoLMFailure("bench-architecture needs --reference, today's base braid fed the same way", code: 64) }
                try Preflight.requireMetallib()
                let tokenizer = try await RaoTokenizer.load()
                let started = Date()
                let result = try ArchitectureBench.run(
                    arm: BraidLayout(dataRoot: global.root), reference: BraidLayout(dataRoot: DataRoot.resolve(argument: reference)), tokenizer: tokenizer
                ) { line in Console.error(line) }
                print(BraidTables.architecture(result))
                print(String(format: "\n%.0f s", Date().timeIntervalSince(started)))
                if let out {
                    try JSONCoding.write(result, to: URL(fileURLWithPath: out))
                    print("report: \(out)")
                }
            }
        }
    }
}

extension BraidGroup {
    struct BenchThought: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "bench-thought",
            abstract: "Whether the Jacobian lens reads what a node plans: faithful to its output, ahead of it, and on the Thread that knows a retold fact.",
            discussion: """
                The braid is --data-dir: base nodes fed from a dataset (raolm braid demo --offline --fresh --dataset braid-cross-v1 \
                --batch 80 --preset base). J is taken once through the pack's trunk and saved beside the pack. Rules L1 to L3 were fixed \
                before any numbers (Docs/ARCHITECTURE.md).
                """,
            shouldDisplay: false)

        @OptionGroup var global: GlobalOptions

        @Option(help: "Write the report to this JSON file.")
        var out: String?

        @Option(help: "Print a saved report, scored again under the rules, instead of running.")
        var report: String?

        func run() async throws {
            try await guarded {
                if let report {
                    var loaded = try JSONCoding.read(ThoughtReport.self, from: URL(fileURLWithPath: (report as NSString).expandingTildeInPath))
                    loaded.evaluation = ThoughtBench.evaluate(nodes: loaded.nodes, told: loaded.told, lens: loaded.lens)
                    print(BraidTables.thought(loaded))
                    return
                }
                try Preflight.requireMetallib()
                let tokenizer = try await RaoTokenizer.load()
                let started = Date()
                let result = try ThoughtBench.run(layout: BraidLayout(dataRoot: global.root), tokenizer: tokenizer) { line in Console.error(line) }
                print(BraidTables.thought(result))
                print(String(format: "\n%.0f s", Date().timeIntervalSince(started)))
                if let out {
                    try JSONCoding.write(result, to: URL(fileURLWithPath: out))
                    print("report: \(out)")
                }
            }
        }
    }
}

extension BraidTables {
    static func thought(_ report: ThoughtReport) -> String {
        func pct(_ value: Float) -> String { String(format: "%.0f%%", value * 100) }
        var lines: [String] = [String(format: "lens: J over %d snippets of %d tokens, split-half agreement %.3f", report.lens.prompts,
                                      report.lens.promptTokens, report.lens.splitHalfAgreement), ""]
        lines.append(Format.table(
            ["node", "top token = output's: lens", "identity", "2–8 ahead in top 25: lens", "output", "identity"],
            report.nodes.map { [$0.node, pct($0.lensAgreement), pct($0.identityAgreement), pct($0.lensAheadRate), pct($0.outputAheadRate),
                                pct($0.identityAheadRate)] }))
        lines.append("")
        for told in report.told.prefix(6) {
            lines.append("\(told.text) · «\(told.token)»: source lens \(told.sourceLens ? "holds" : "misses"), control "
                         + (told.controlLens.map { $0 ? "holds" : "misses" } ?? "—") + "; the source's workspace: " + told.workspace.joined(separator: " | "))
        }
        lines.append("")
        for rule in report.evaluation.rules { lines.append("\(rule.passed ? "pass" : "FAIL")  \(rule.rule): \(rule.detail)") }
        for (key, value) in report.evaluation.reported.sorted(by: { $0.key < $1.key }) { lines.append("  \(key): \(String(format: "%.3f", value))") }
        lines.append("")
        lines.append(report.evaluation.summary)
        return lines.joined(separator: "\n")
    }
}

extension BraidGroup {
    struct BenchQuestion: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "bench-question",
            abstract: "Whether the braid answers questions: the commons' rewrite against the rules', the stem, and the v1 prompts.",
            discussion: """
                The braid is --data-dir; --dataset names a dataset with questions (braid-cross-v2 or later), whose facts are matched \
                to the braid's by node, kind, subject and answer. Rules Q1 to Q5 were fixed before any numbers (Docs/ARCHITECTURE.md).
                """,
            shouldDisplay: false)

        @OptionGroup var global: GlobalOptions

        @Option(help: "A dataset with questions: a name in the datasets root, or a path.")
        var dataset: String?

        @Option(help: "Only these arms, comma-separated (commons, rules, stem, slice, paraphrase).")
        var arms: String?

        @Flag(inversion: .prefixedNo, help: "Each Thread completes behind its own context.")
        var context = true

        @Option(help: "Write the report to this JSON file.")
        var out: String?

        @Option(help: "Print a saved report, scored again under the rules, instead of running.")
        var report: String?

        func run() async throws {
            try await guarded {
                if let report {
                    var loaded = try JSONCoding.read(QuestionReport.self, from: URL(fileURLWithPath: (report as NSString).expandingTildeInPath))
                    loaded.evaluation = QuestionBench.evaluate(arms: loaded.arms)
                    print(BraidTables.question(loaded))
                    return
                }
                guard let dataset else { throw RaoLMFailure("bench-question needs --dataset, a dataset with questions", code: 64) }
                try Preflight.requireMetallib()
                let tokenizer = try await RaoTokenizer.load()
                let started = Date()
                var sizes = QuestionBench.Sizes()
                sizes.context = context
                sizes.arms = arms.map { $0.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) } }
                let result = try QuestionBench.run(
                    layout: BraidLayout(dataRoot: global.root), dataset: try DatasetsRoot.resolve(dataset), tokenizer: tokenizer, sizes: sizes
                ) { line in Console.error(line) }
                print(BraidTables.question(result))
                print(String(format: "\n%.0f s", Date().timeIntervalSince(started)))
                if let out {
                    try JSONCoding.write(result, to: URL(fileURLWithPath: out))
                    print("report: \(out)")
                }
            }
        }
    }
}

extension BraidGroup {
    struct BenchCommons: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "bench-commons",
            abstract: "Whether a new commons is fit for the braid: a child pack's braid against its parent's, on the same documents.",
            discussion: """
                --data-dir is the braid on the child pack (a rebased copy of the parent's, retrained); --reference is the braid on \
                the parent pack. Rules C1 to C4 were fixed before any numbers (Docs/ARCHITECTURE.md, "bench-commons").
                """,
            shouldDisplay: false)

        @OptionGroup var global: GlobalOptions

        @Option(help: "The braid on the parent pack.")
        var reference: String?

        @Option(help: "Write the report to this JSON file.")
        var out: String?

        @Option(help: "Print a saved report, scored again under the rules, instead of running.")
        var report: String?

        func run() async throws {
            try await guarded {
                if let report {
                    var loaded = try JSONCoding.read(CommonsReport.self, from: URL(fileURLWithPath: (report as NSString).expandingTildeInPath))
                    loaded.evaluation = CommonsBench.evaluate(heldOut: loaded.heldOut, arms: loaded.arms, facts: loaded.facts)
                    print(BraidTables.commons(loaded))
                    return
                }
                guard let reference else { throw RaoLMFailure("bench-commons needs --reference, the braid on the parent pack", code: 64) }
                try Preflight.requireMetallib()
                let tokenizer = try await RaoTokenizer.load()
                let started = Date()
                let result = try CommonsBench.run(
                    child: BraidLayout(dataRoot: global.root), parent: BraidLayout(dataRoot: DataRoot.resolve(argument: reference)), tokenizer: tokenizer
                ) { line in Console.error(line) }
                print(BraidTables.commons(result))
                print(String(format: "\n%.0f s", Date().timeIntervalSince(started)))
                if let out {
                    try JSONCoding.write(result, to: URL(fileURLWithPath: out))
                    print("report: \(out)")
                }
            }
        }
    }

    struct BenchScale: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "bench-scale",
            abstract: "Whether the gate still finds the right Thread as Threads are added: one braid per N, judged against the smallest.",
            discussion: """
                --data-dir is a braid built from an N-node dataset (raolm braid sync --dataset braid-nN --feed 80). The run is in \
                process and writes one point; --reference adds a saved point (the smallest N) to judge it against; --report re-scores \
                saved points together and fits the cost per token against N. Rules N1 and N2 were fixed before any numbers \
                (Docs/ARCHITECTURE.md, "bench-scale").
                """,
            shouldDisplay: false)

        @OptionGroup var global: GlobalOptions

        @Option(help: "A saved report whose points this run is judged with (the smallest N's).")
        var reference: String?

        @Option(help: "Facts per node (ignored with --every-fact).")
        var factsPerNode = 10

        @Option(help: "Two-fact prompts in all, spread over the ordered pairs of nodes.")
        var pairs = 60

        @Flag(help: "Ask every fact of every node (the second reading).")
        var everyFact = false

        @Option(help: "Write the report to this JSON file.")
        var out: String?

        @Option(parsing: .upToNextOption, help: "Saved reports to score together instead of running.")
        var report: [String] = []

        func run() async throws {
            try await guarded {
                func load(_ path: String) throws -> ScaleReport {
                    try JSONCoding.read(ScaleReport.self, from: URL(fileURLWithPath: (path as NSString).expandingTildeInPath))
                }
                if !report.isEmpty {
                    var points: [Int: ScalePoint] = [:]
                    for path in report { for point in try load(path).points { points[point.nodes] = point } }
                    print(BraidTables.scale(ScaleBench.report(Array(points.values))))
                    return
                }
                try Preflight.requireMetallib()
                let tokenizer = try await RaoTokenizer.load()
                let started = Date()
                let point = try ScaleBench.run(layout: BraidLayout(dataRoot: global.root), tokenizer: tokenizer, everyFact: everyFact,
                                               factsPerNode: factsPerNode, pairs: pairs) { line in Console.error(line) }
                var points = [point]
                if let reference { points += try load(reference).points.filter { $0.nodes != point.nodes } }
                let result = ScaleBench.report(points)
                print(BraidTables.scale(result))
                print(String(format: "\n%.0f s", Date().timeIntervalSince(started)))
                if let out {
                    // The file holds this run's point alone, so reports combine by N.
                    var own = result
                    own.points = [point]
                    own.evaluation = ScaleBench.evaluate(points: points)
                    try JSONCoding.write(own, to: URL(fileURLWithPath: out))
                    print("report: \(out)")
                }
            }
        }
    }

    struct BenchHosted: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "bench-hosted",
            abstract: "Whether nodes that dial the umbrella over TCP answer as child processes over pipes do, and at what cost.",
            discussion: """
                --data-dir is a built braid with every node live. It is started twice, offline, nothing fed or trained: its nodes \
                as child processes over pipes, then as processes that dial a listener on 127.0.0.1. Rules H1 and H3 were fixed \
                before any numbers (Docs/ARCHITECTURE.md, "Step 2"); H2 is a process test.
                """,
            shouldDisplay: false)

        @OptionGroup var global: GlobalOptions

        @Option(help: "Facts per node.")
        var factsPerNode = 10

        @Option(help: "Write the report to this JSON file.")
        var out: String?

        func run() async throws {
            try await guarded {
                try Preflight.requireMetallib()
                guard let executable = Bundle.main.executableURL?.resolvingSymlinksInPath() else { throw BraidSessionError.noExecutable }
                let tokenizer = try await RaoTokenizer.load()
                let started = Date()
                let report = try await HostedBench.run(root: global.root, executable: executable, tokenizer: tokenizer,
                                                       factsPerNode: factsPerNode) { line in Console.error(line) }
                print(BraidTables.hosted(report))
                print(String(format: "\n%.0f s", Date().timeIntervalSince(started)))
                if let out {
                    try JSONCoding.write(report, to: URL(fileURLWithPath: out))
                    print("report: \(out)")
                }
            }
        }
    }
}

extension BraidGroup {
    struct BenchRoute: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "bench-route",
            abstract: "Whether routing before asking keeps the answers and cuts the cost: a built braid asked unrouted, then routed.",
            discussion: """
                --data-dir is a braid built from an N-node dataset, every node live. In process, the same sets as bench-scale; \
                --report re-scores saved points together. Rules R1 to R3 were fixed before any numbers (Docs/ARCHITECTURE.md, "Step 1").
                """,
            shouldDisplay: false)

        @OptionGroup var global: GlobalOptions

        @Option(help: "Facts per node.")
        var factsPerNode = 10

        @Option(help: "Two-fact prompts in all, spread over the ordered pairs of nodes.")
        var pairs = 60

        @Option(help: "Write the report to this JSON file.")
        var out: String?

        @Option(parsing: .upToNextOption, help: "Saved reports to score together instead of running.")
        var report: [String] = []

        func run() async throws {
            try await guarded {
                if !report.isEmpty {
                    var points: [Int: RoutePoint] = [:]
                    for path in report {
                        let saved = try JSONCoding.read(RouteReport.self, from: URL(fileURLWithPath: (path as NSString).expandingTildeInPath))
                        for point in saved.points { points[point.nodes] = point }
                    }
                    print(BraidTables.route(RouteBench.report(Array(points.values))))
                    return
                }
                try Preflight.requireMetallib()
                let tokenizer = try await RaoTokenizer.load()
                let point = try RouteBench.run(layout: BraidLayout(dataRoot: global.root), tokenizer: tokenizer, factsPerNode: factsPerNode,
                                               pairs: pairs) { line in Console.error(line) }
                let result = RouteBench.report([point])
                print(BraidTables.route(result))
                if let out {
                    try JSONCoding.write(result, to: URL(fileURLWithPath: out))
                    print("report: \(out)")
                }
            }
        }
    }
}

extension BraidTables {
    static func route(_ report: RouteReport) -> String {
        func pct(_ value: Float?) -> String { value.map { String(format: "%.0f%%", $0 * 100) } ?? "—" }
        func rate(_ a: Int, _ b: Int) -> String { b > 0 ? String(format: "%.0f%%", Double(a) / Double(b) * 100) : "—" }
        var lines: [String] = []
        for point in report.points {
            let sketch = point.sketchBigrams.values
            lines.append("\(point.braid): N = \(point.nodes), a bigram is distinctive when at most \(point.bound) Threads hold it; "
                         + "sketches \(sketch.min() ?? 0)–\(sketch.max() ?? 0) bigrams (\((sketch.max() ?? 0) * 8 / 1024) KB at most)")
            lines.append("route: owner a candidate on \(point.factsFound)/\(point.facts) facts, both owners on \(point.pairsFound)/\(point.pairs) pairs")
            lines.append(Format.table(["set", "prompts", "no distinctive bigram", "Threads opened (mean)", "max"], point.sets.map { set in
                [set.set, "\(set.prompts)", "\(set.unmatched)", String(format: "%.1f", set.meanOpened), "\(set.maxOpened)"]
            }))
            lines.append("")
            lines.append(Format.table(
                ["", "facts exact", "led + cited", "cit@1", "pairs moved", "commons leads", "Threads take", "s/token facts", "s/token generic", "s/prompt facts"],
                [("unrouted", point.unrouted), ("routed", point.routed)].map { name, side in
                    [name, "\(side.facts.factsExact)/\(side.facts.facts) (\(pct(side.facts.factsExactRate)))", pct(side.facts.factsOwnedRate),
                     pct(side.facts.citation), "\(side.prompts.pairsMoved)/\(side.prompts.pairs)", pct(side.prompts.commonsLeads),
                     side.prompts.commonsThreadShare.map { String(format: "%.3f", $0) } ?? "—",
                     String(format: "%.4f", side.factCost.secondsPerToken), String(format: "%.4f", side.genericCost.secondsPerToken),
                     String(format: "%.3f", side.factCost.secondsPerPrompt)]
                }))
            if !point.missed.isEmpty { lines.append("missed: " + point.missed.prefix(5).joined(separator: "; ")) }
            lines.append("")
        }
        for rule in report.evaluation.rules { lines.append("\(rule.passed ? "pass" : "FAIL")  \(rule.rule): \(rule.detail)") }
        lines.append(report.evaluation.summary)
        return lines.joined(separator: "\n")
    }

    static func hosted(_ report: HostedReport) -> String {
        var lines = ["\(report.braid): \(report.nodes) nodes, \(report.prompts) fact prompts each way", ""]
        lines.append(Format.table(
            ["transport", "s/token", "s/prompt", "asked/token", "all asked"],
            [("pipes", report.pipes), ("TCP", report.tcp)].map { name, cost in
                [name, String(format: "%.4f", cost.secondsPerToken), String(format: "%.3f", cost.secondsPerPrompt),
                 String(format: "%.2f", cost.askedPerToken), String(format: "%.0f%%", cost.allAskedRate * 100)]
            }))
        lines.append("")
        if let first = report.firstDifference { lines.append("first difference: \(first)") }
        for rule in report.evaluation.rules { lines.append("\(rule.passed ? "pass" : "FAIL")  \(rule.rule): \(rule.detail)") }
        lines.append(report.evaluation.summary)
        return lines.joined(separator: "\n")
    }
}

extension BraidTables {
    static func commons(_ report: CommonsReport) -> String {
        func pct(_ value: Float?) -> String { value.map { String(format: "%.0f%%", $0 * 100) } ?? "—" }
        func num(_ value: Float?) -> String { value.map { String(format: "%.3f", $0) } ?? "—" }
        var lines = ["child \(report.child.prefix(12)) against parent \(report.parent.prefix(12))" + (report.lineage.isEmpty ? "" : " · lineage " + report.lineage.joined(separator: " ← ")), ""]
        lines.append(Format.table(
            ["commons", "facts exact", "citation@1", "lift positive", "commons leads", "largest Thread gate", "Threads' share", "unknown: commons largest"],
            ["parent", "child"].compactMap { name -> [String]? in
                guard let arm = report.arms.first(where: { $0.arm == name }), let facts = report.facts.first(where: { $0.arm == name }) else { return nil }
                return [name, "\(facts.factsExact)/\(facts.facts) (\(pct(facts.factsExactRate)))", pct(facts.citation), pct(facts.liftPositive),
                        pct(arm.commonsLeads), num(arm.commonsThreadGate), num(arm.commonsThreadShare), pct(arm.unknownLargest)]
            }))
        lines.append("")
        if let byNode = report.factsByNode, let parent = byNode["parent"], let child = byNode["child"] {
            for node in parent.keys.sorted() {
                guard let p = parent[node], let c = child[node], p.count == 2, c.count == 2 else { continue }
                lines.append("facts held by \(node): parent \(p[0])/\(p[1]) · child \(c[0])/\(c[1]) (\(c[0] - p[0] >= 0 ? "+" : "")\(c[0] - p[0]))")
            }
            lines.append("")
        }
        for (set, losses) in report.heldOut.sorted(by: { $0.key < $1.key }) where losses.count == 2 {
            lines.append(String(format: "held-out %@: %.4f → %.4f (%+.4f)", set, losses[0], losses[1], losses[1] - losses[0]))
        }
        lines.append("")
        for rule in report.evaluation.rules { lines.append("\(rule.passed ? "pass" : "FAIL")  \(rule.rule): \(rule.detail)") }
        lines.append("")
        lines.append(report.evaluation.summary)
        return lines.joined(separator: "\n")
    }

    static func scale(_ report: ScaleReport) -> String {
        func pct(_ value: Float?) -> String { value.map { String(format: "%.0f%%", $0 * 100) } ?? "—" }
        func num(_ value: Float?) -> String { value.map { String(format: "%.3f", $0) } ?? "—" }
        var lines = [Format.table(
            ["N", "facts exact", "led+cited", "cit@1", "pairs moved", "commons leads", "Threads take", "unknown: commons leads",
             "asked/token (facts, generic)", "s/token", "s/prompt"],
            report.points.map { p in
                ["\(p.nodes)", "\(p.facts.factsExact)/\(p.facts.facts) (\(pct(p.facts.factsExactRate)))", pct(p.facts.factsOwnedRate), pct(p.facts.citation),
                 "\(p.prompts.pairsMoved)/\(p.prompts.pairs)", pct(p.prompts.commonsLeads), num(p.prompts.commonsThreadShare), pct(p.unknownCommonsLeads),
                 String(format: "%.1f, %.1f", p.askedPerFactToken ?? 0, p.tokenCost.askedPerToken),
                 String(format: "%.3f", p.tokenCost.secondsPerToken), String(format: "%.2f", p.tokenCost.secondsPerPrompt)]
            })]
        for p in report.points where !p.factsByNode.isEmpty {
            let nodes = p.factsByNode.sorted { $0.key < $1.key }.map { "\($0.key) \($0.value.first ?? 0)/\($0.value.last ?? 0)" }
            lines.append("N = \(p.nodes): " + nodes.joined(separator: " · "))
            if let footprint = p.peakFootprintBytes, !footprint.isEmpty {
                let gb = footprint.values.map { Double($0) / 1e9 }
                lines.append(String(format: "  peak memory per node process while training: %.1f–%.1f GB", gb.min() ?? 0, gb.max() ?? 0))
            }
        }
        lines.append("")
        for rule in report.evaluation.rules { lines.append("\(rule.passed ? "pass" : "FAIL")  \(rule.rule): \(rule.detail)") }
        if let fit = report.evaluation.secondsFit {
            lines.append(String(format: "\nseconds per token ≈ %.4f + %.4f × N (r² %.2f)", fit.intercept, fit.slope, fit.r2))
        }
        if let fit = report.evaluation.askedFit {
            lines.append(String(format: "Threads asked per token ≈ %.2f + %.3f × N (r² %.2f)", fit.intercept, fit.slope, fit.r2))
        }
        lines.append("")
        lines.append(report.evaluation.summary)
        return lines.joined(separator: "\n")
    }

    static func question(_ report: QuestionReport) -> String {
        func pct(_ value: Float?) -> String { value.map { String(format: "%.0f%%", $0 * 100) } ?? "—" }
        func num(_ value: Float?) -> String { value.map { String(format: "%.3f", $0) } ?? "—" }
        var lines: [String] = ["questions from \(report.dataset) on \(report.braid) · \(report.unmatched) facts of the braid unmatched", ""]
        lines.append(Format.table(
            ["arm", "questions", "stem reached", "F1", "exact", "cited answer", "cit@1", "unknown: commons leads", "Threads' credit",
             "shared: decided", "followed", "other's credit", "ties", "adapter ms"],
            report.arms.map { arm in
                [arm.arm, "\(arm.questions)", arm.rewriteF1 == nil ? "—" : pct(arm.rewriteRate), num(arm.rewriteF1), "\(arm.exact) (\(pct(arm.exactRate)))",
                 arm.exact > 0 ? pct(arm.citedRate) : "—", pct(arm.citationAt1), arm.unknown > 0 ? "\(arm.unknownCommonsLeads)/\(arm.unknown)" : "—",
                 num(arm.unknownThreadCredit), arm.shared > 0 ? "\(arm.sharedDecided)/\(arm.shared)" : "—",
                 arm.sharedDecided > 0 ? pct(arm.sharedFollowedRate) : "—", num(arm.sharedDuplicateCredit), arm.shared > 0 ? "\(arm.sharedTies)" : "—",
                 arm.adapterMs.map { String(format: "%.0f", $0) } ?? "—"]
            }))
        lines.append("")
        for arm in report.arms where arm.ownerContext > 0 || arm.noContextExact > 0 {
            let without = arm.questions - arm.ownerContext
            lines.append(String(format: "%@: the owner found a context for %d of %d; exact %d of those (%.0f%%), %d of the %d without (%.0f%%)", arm.arm,
                                arm.ownerContext, arm.questions, arm.ownerContextExact, arm.ownerContext > 0 ? 100 * Double(arm.ownerContextExact) / Double(arm.ownerContext) : 0,
                                arm.noContextExact, without, without > 0 ? 100 * Double(arm.noContextExact) / Double(without) : 0))
        }
        lines.append("")
        for rule in report.evaluation.rules { lines.append("\(rule.passed ? "pass" : "FAIL")  \(rule.rule): \(rule.detail)") }
        lines.append("")
        lines.append(report.evaluation.summary)
        return lines.joined(separator: "\n")
    }
}

extension BraidTables {
    static func architecture(_ report: ArchitectureReport) -> String {
        func pct(_ value: Float?) -> String { value.map { String(format: "%.0f%%", $0 * 100) } ?? "—" }
        func num(_ value: Float?) -> String { value.map { String(format: "%.3f", $0) } ?? "—" }
        func rate(_ count: Int, _ total: Int) -> String { total > 0 ? "\(count)/\(total)" : "—" }
        var lines: [String] = ["arm \(report.arm) against the reference braid (\(report.referenceArm ?? "eos-first windows"))", ""]
        lines.append(Format.table(
            ["braid", "facts exact", "led+cited", "cit@1", "break written", "ran together", "citations on a break"],
            report.braids.map { braid in
                [braid.braid, "\(braid.facts.factsExact)/\(braid.facts.facts)", pct(braid.facts.factsOwnedRate), pct(braid.facts.citation),
                 rate(braid.breaksWritten, braid.continuations), rate(braid.ranTogether, braid.continuations),
                 "\(braid.citationsOnBreaks) of \(braid.citations)"]
            }))
        lines.append("")
        lines.append(Format.table(
            ["braid", "node", "steps to 97%", "memorised", "held-out own", "held-out commons", "greedy facts", "break likeliest"],
            report.nodes.map { node in
                [node.braid, node.node, node.stepsTo97.map(String.init) ?? "never", pct(node.memorised), num(node.heldOutLoss), num(node.commonsLoss),
                 "\(node.greedyCorrect)/\(node.greedyFacts)", node.boundaries > 0 ? "\(pct(node.breakRate)) of \(node.boundaries)" : "—"]
            }))
        lines.append("")
        for rule in report.evaluation.rules { lines.append("\(rule.passed ? "pass" : "FAIL")  \(rule.rule): \(rule.detail)") }
        lines.append("")
        lines.append(report.evaluation.summary)
        return lines.joined(separator: "\n")
    }
}

extension BraidTables {
    static func umbrella(_ report: UmbrellaReport) -> String {
        func pct(_ value: Float?) -> String { value.map { String(format: "%.0f%%", $0 * 100) } ?? "—" }
        func num(_ value: Float?) -> String { value.map { String(format: "%.3f", $0) } ?? "—" }
        var lines: [String] = ["pack \(report.pack.map { String($0.prefix(12)) } ?? "none")", ""]
        // A set an arm did not run (the ceiling runs facts only) shows "—", not a zero.
        let rows = (report.arms + (report.ceiling.map { [$0] } ?? [])).map { arm -> [String] in
            [arm.arm, "\(arm.factsExact)/\(arm.facts)", arm.facts > 0 ? pct(arm.factsOwnedRate) : "—", pct(arm.citation),
             num(arm.ownerShare), pct(arm.liftPositive), arm.pairs > 0 ? pct(arm.pairsMovedRate) : "—", pct(arm.commonsLeads),
             num(arm.commonsThreadShare), arm.heldOutTokens > 0 ? String(format: "%.4f", arm.heldOutNLL) : "—",
             arm.toldTokens > 0 ? num(arm.toldSourceShare) : "—"]
        }
        lines.append(Format.table(
            ["arm", "facts exact", "led+cited", "cit@1", "owner", "lift>0", "pairs moved", "commons leads", "Threads take", "held-out NLL", "told source"],
            rows))
        lines.append("")
        for node in report.nodes {
            lines.append("\(node.braid) \(node.node) v\(node.version): \(node.documents) documents, memorised \(pct(node.memorised)), 97% after "
                         + (node.stepsTo97.map { "\($0) steps" } ?? "never") + ", held-out loss own \(num(node.heldOutLoss)) commons \(num(node.commonsLoss))"
                         + (node.calibration.map { String(format: ", false chains %.3f, τ %.3f, λ ×%.2f", $0.falseChainRate, $0.tau, $0.lambdaScale) } ?? ""))
        }
        lines.append("")
        for rule in report.evaluation.rules { lines.append("\(rule.passed ? "pass" : "FAIL")  \(rule.rule): \(rule.detail)") }
        lines.append("")
        lines.append(report.evaluation.summary)
        let credited = report.arms.filter { $0.ownerCredit != nil }
        if !credited.isEmpty {
            // The owner's blend: form is the commons', content each strand's as it supplied it.
            lines.append("")
            lines.append("credit, weighted by bits: form tokens are the commons', content tokens are each strand's as it supplied them")
            lines.append(Format.table(
                ["arm", "owner, its answers", "form share of answer bits", "commons, general text", "Threads, own unfed text", "source, told answers"],
                credited.map { arm in
                    [arm.arm, num(arm.ownerCredit), pct(arm.formBits), num(arm.commonsCreditGeneral), num(arm.threadsCreditOwnVoice),
                     num(arm.toldSourceCredit)]
                }))
        }
        if let every = report.allFacts, let evaluation = report.allFactsEvaluation {
            // The second reading: the fact rules on every fact prompt.
            let all = every + (report.allFactsCeiling.map { [$0] } ?? [])
            lines.append("")
            lines.append("every fact (\(every.first?.facts ?? 0) prompts): the fact rules read on all of them, one prompt in 30 as a rate")
            lines.append(Format.table(
                ["arm", "facts exact", "led+cited", "cit@1", "owner", "lift>0", "owner credit"],
                all.map { arm in
                    [arm.arm, "\(arm.factsExact)/\(arm.facts)", pct(arm.factsOwnedRate), pct(arm.citation), num(arm.ownerShare), pct(arm.liftPositive),
                     num(arm.ownerCredit)]
                }))
            lines.append("")
            for rule in evaluation.rules { lines.append("\(rule.passed ? "pass" : "FAIL")  \(rule.rule): \(rule.detail)") }
            lines.append("")
            lines.append("every fact: " + evaluation.summary)
        }
        return lines.joined(separator: "\n")
    }

    static func trajectory(_ report: TrajectoryReport) -> String {
        var lines: [String] = []
        let nodes = report.nodes.sorted { $0.key < $1.key }.map { name, version in
            "\(name) v\(version) (\(report.documents[name] ?? 0) documents, \(Format.pct(report.memorised[name] ?? 0)) memorised)"
        }
        lines.append("nodes: " + nodes.joined(separator: " · "))
        if let world = report.world { lines.append("world: \(world.summary)") }
        lines.append("rule: a chain of \(report.rule.phrase) tokens traces 0, of \(report.rule.arc) traces 1; bridges of \(report.rule.bridge)")
        lines.append("")
        lines.append("per text, from token \(TrajectoryBench.from): its own Thread's trace and the others'; the longest chain; manner at the end")
        var rows: [[String]] = []
        for kind in TrajectoryTextKind.allCases {
            let texts = report.texts.filter { $0.kind == kind }
            guard !texts.isEmpty else { continue }
            let own = texts.compactMap { $0.reading($0.owner) }
            let others = texts.flatMap { text in text.readings.filter { $0.strand != text.owner } }
            func share(_ readings: [TrajectoryReading], _ test: (TrajectoryReading) -> Bool) -> String {
                readings.isEmpty ? "—" : Format.pct(Float(readings.filter(test).count) / Float(readings.count))
            }
            func mean(_ values: [Float]) -> String { values.isEmpty ? "—" : Format.f(Stats.mean(values), 3) }
            rows.append([
                kind.rawValue, String(texts.count), String(Int(Stats.mean(texts.map { Float($0.tokens) }))),
                mean(own.map(\.trace)), share(own) { $0.trace >= TrajectoryBench.holderFloor }, mean(others.map(\.trace)),
                share(others) { $0.trace <= TrajectoryBench.low }, own.map(\.longest).max().map(String.init) ?? "—",
                others.map(\.longest).max().map(String.init) ?? "—", mean(own.map { $0.manner ?? 0 }), mean(others.map { $0.manner ?? 0 }),
                mean(own.map(\.meanFit)),
            ])
        }
        lines.append(Format.table(
            ["texts", "n", "tokens", "own trace", "own ≥ 0.40", "others' trace", "others ≤ 0.10", "own longest", "others' longest",
             "own manner", "others' manner", "own fit"], rows))
        lines.append("")
        let evaluation = report.evaluation
        for rule in evaluation.rules { lines.append("  \(rule.passed ? "✓" : "✗") \(rule.rule): \(rule.detail)") }
        lines.append("reported: " + evaluation.reported.sorted { $0.key < $1.key }.map { "\($0.key) \(Format.f($0.value, 3))" }.joined(separator: " · "))
        if !report.arms.isEmpty {
            lines.append("")
            lines.append("gate arms · the holder's share of told and source answer tokens; guards on document-prefix answers and continued voice and generic text")
            lines.append(Format.table(
                ["arm", "holder share", "answer tokens", "exact", "largest gate", "all asked", "not asked", "qualifies"],
                report.arms.map { arm in
                    [arm.arm, Format.f(arm.holderShare, 3), String(arm.answerTokens), "\(arm.exact)/\(arm.prompts)", Format.f(arm.largestGate, 2),
                     Format.pct(arm.allAsked), Format.pct(arm.notAsked),
                     evaluation.qualifies.contains(arm.arm) ? "yes" : (evaluation.failures[arm.arm] == nil ? "—" : "no")]
                }))
            for (arm, failed) in evaluation.failures.sorted(by: { $0.key < $1.key }) {
                lines.append("  ✗ \(arm): " + failed.joined(separator: "; "))
            }
        }
        lines.append("decision: \(evaluation.summary)")
        return lines.joined(separator: "\n")
    }
}

/// Waits for nodes to finish the update a feed started.
enum BraidWait {
    /// Waits until every node in `versions` has a new version, is held or has failed. It gives up
    /// only when no node has reported progress for `stall` seconds: a node of a larger preset takes
    /// longer, and keeps saying so.
    static func settled(_ session: BraidSession, after versions: [String: Int], stop: StopSignal, stall: TimeInterval = 600) async throws {
        var reported: [String: String] = [:]
        var progressed = Date()
        while true {
            if stop.isSet { throw CancellationError() }
            var done = true
            for (name, before) in versions {
                guard let state = session.state(name) else { done = false; continue }
                let busy = state.stage.isBusy || state.ladder.contains { $0.status == .pending || $0.status == .running }
                let finished = !busy && (state.versions > before || state.stage == .held || state.stage == .failed)
                if !finished { done = false }
                let line = BraidPrinter.progress(state)
                if reported[name] != line {
                    reported[name] = line
                    progressed = Date()
                    print("  \(line)")
                }
            }
            if done { return }
            if Date().timeIntervalSince(progressed) > stall {
                throw RaoLMFailure("the nodes made no progress for \(Int(stall)) s", hint: "raolm braid status; see <data root>/braid/nodes/*/logs", code: 75)
            }
            try await Task.sleep(nanoseconds: 500_000_000)
        }
    }
}

enum BraidDemoPicks {
    /// `perNode` facts from each node in turn, then a prompt that moves from one Thread's fact to
    /// another's, and one about a subject no Thread holds.
    static func pick(_ examples: [BraidExample], perNode: Int, names: [String]) -> [BraidExample] {
        var picked: [BraidExample] = []
        for round in 0..<perNode {
            for name in names {
                let own = examples.filter { $0.node == name && $0.resolvedKind == .fact }
                if round < own.count { picked.append(own[round]) }
            }
        }
        if let pair = examples.first(where: { $0.resolvedKind == .pair }) { picked.append(pair) }
        if let negative = examples.first(where: { $0.resolvedKind == .unknown }) { picked.append(negative) }
        if let retold = examples.first(where: { $0.resolvedKind == .crossed }) { picked.append(retold) }
        return picked
    }
}

/// Prints what matters of the event stream: node lifecycle, promotions, failures.
final class BraidPrinter: @unchecked Sendable {
    private let lock = NSLock()

    func handle(_ event: BraidEvent) {
        let line: String?
        switch event {
        case .spawned(let name, let pid): line = "\(name): node process pid \(pid)"
        case .ready(let name, let hello):
            line = "\(name): ready · " + (hello.offline ? "offline corpus" : "Thread pid \(hello.threadPID ?? 0) on :\(hello.httpPort ?? 0)/:\(hello.grpcPort ?? 0)")
                + " · node \(hello.threadID?.prefix(8).lowercased() ?? "?") · live \(hello.liveVersion.map { "v\($0)" } ?? "none")"
        case .node(let name, .promoted(let version, let kind)): line = "\(name): v\(version) live (\(kind.rawValue))"
        case .node(let name, .held(let version, let reason)): line = "\(name): v\(version) held — \(reason)"
        case .exited(let name, let status): line = status == 0 ? nil : "\(name): node process exited with \(status)"
        case .failure(let name, let message): line = "\(name ?? "braid"): \(message)"
        case .note(let message): line = "braid: \(message)"
        default: line = nil
        }
        guard let line else { return }
        lock.withLock { print(line) }
    }

    static func progress(_ state: StrandState) -> String {
        var text = "\(state.name): \(state.stage.rawValue)"
        if state.stage == .training, let epoch = state.epoch, let epochs = state.epochs {
            text += " epoch \(epoch)/\(epochs)"
            if let memorised = state.candidateMemorised { text += " · memorised \(Format.pct(memorised))" }
        } else if let detail = state.stageDetail {
            text += " · \(detail)"
        }
        return text
    }
}

enum BraidTables {
    static func nodes(_ states: [StrandState], session: BraidSession) -> TextTable {
        TextTable(
            headers: ["node", "pid", "thread pid", "ports", "thread id", "live", "versions", "docs", "partitions", "memorised"],
            rows: states.map { s in
                [
                    s.label, s.pid.map(String.init) ?? "—", s.threadPID.map(String.init) ?? (s.offline ? "offline" : "—"),
                    s.httpPort.map { ":\($0)/:\(s.grpcPort ?? 0)" } ?? "—", s.threadID.map { String($0.prefix(8)).lowercased() } ?? "—",
                    s.liveVersion.map { "v\($0)" } ?? "—", String(s.versions), String(s.documents), String(s.partitions),
                    Format.pct(s.memorised),
                ]
            })
    }

    static func gateNote(_ gating: BraidGating) -> String {
        switch gating {
        case .braided: return "token by token, by what each Thread has been predicting, lifted where Threads agree on the next token, leaning to a Thread whose documents the text follows"
        case .posterior: return "every token's likelihood multiplied since the start; it settles on one Thread"
        case .retrieval: return "each Thread's share of one pool of raw cosines across Threads"
        }
    }

    static func routeHeaders(_ names: [String]) -> [String] {
        ["prompt", "expected", "answer"] + names.map { "\($0) share" } + ["gate", "top citation", "spans"]
    }

    static func route(_ example: BraidExample, _ generation: CitedGeneration, report: VerificationReport, names: [String]) -> [String] {
        let generated = generation.traces.filter { !$0.isPrompt }
        let answerLength = max(1, example.expected.map { _ in generated.count >= 2 ? 2 : 1 } ?? 1)
        let answer = Array(generated.prefix(answerLength + 2))
        var shares: [String] = []
        for name in names {
            let values = answer.compactMap { $0.strands?.first { $0.strand == name }?.share }
            shares.append(values.isEmpty ? "—" : Format.pct(Stats.mean(values)))
        }
        let first = generated.first
        let gate = first?.strands?.map { strand in
            let label = strand.strand == BraidStrandRef.commonsName ? strand.strand : String(strand.strand.prefix(1))
            return "\(label)\(strand.open ? "●" : "○")\(Format.f(strand.gate, 2))"
        }.joined(separator: " ") ?? "—"
        let citation = first?.citations.first.map { c -> String in
            let strand = generation.braid?.strand(row: c.row)?.label ?? "?"
            return "\(strand) · \(Format.clip(generation.partition(row: c.row)?.documentName ?? "?", 22))"
        } ?? "uncited"
        let spans = report.checks.filter { $0.kind == .verbatim }
        return [
            Format.clip(example.label, 34), example.expected.map { Format.clip($0.trimmingCharacters(in: .whitespaces), 16) } ?? "—",
            Format.clip(generation.text.trimmingCharacters(in: .whitespaces), 22),
        ] + shares + [gate, citation, spans.isEmpty ? "—" : "\(spans.filter { $0.status == .verified }.count)/\(spans.count) ✓"]
    }

    /// A question's answer: how it was rewritten, what the braid said, who earned it, and where it is cited.
    static func answer(_ generation: CitedGeneration, names: [String], commons: (name: String, text: String)? = nil) -> String {
        var lines: [String] = []
        if let question = generation.prompt.question {
            lines.append("question  \(question.question)")
            lines.append(String(format: "stem      %@   (%@, %.0f ms)", question.stem, question.rewriter, question.seconds * 1000))
        } else {
            lines.append("prompt    \(generation.prompt.text)")
        }
        for strand in generation.braid?.strands ?? [] {
            if let context = strand.context { lines.append(String(format: "context   %@: “%@” (%.2f)", strand.name, context.text.trimmingCharacters(in: .whitespaces), context.score)) }
        }
        lines.append("answer    \(generation.text.trimmingCharacters(in: .whitespaces))")
        if let commons { lines.append("alone     commons (\(commons.name)): \(commons.text.trimmingCharacters(in: .whitespacesAndNewlines))") }
        let generated = generation.traces.filter { !$0.isPrompt }
        // Credit over the answer's tokens, weighted by bits, as the blend weighs it.
        var credit: [String: Double] = [:]
        var bits = 0.0
        for trace in generated {
            guard let b = trace.bits, let strands = trace.strands else { continue }
            bits += Double(b)
            for strand in strands { credit[strand.strand, default: 0] += Double(b) * Double(strand.credit ?? 0) }
        }
        if bits > 0 {
            let order = names.filter { credit[$0] != nil } + credit.keys.filter { !names.contains($0) }.sorted()
            func shown(_ name: String) -> String { name == BraidStrandRef.commonsName ? commons.map { "commons (\($0.name))" } ?? name : name }
            lines.append("credit    " + order.map { String(format: "%@ %.0f%%", shown($0), 100 * credit[$0]! / bits) }.joined(separator: " · "))
        }
        if let followed = generated.compactMap(\.followed).first { lines.append("followed  \(followed)'s document") }
        var cited: [String] = []
        for trace in generated {
            guard let citation = trace.citations.first else { continue }
            let strand = generation.braid?.strand(row: citation.row)?.label ?? "?"
            let document = generation.partition(row: citation.row)?.documentName ?? "?"
            let line = "\(strand) · \(document) · partition \(citation.address.partitionIndex)"
            if !cited.contains(line) { cited.append(line) }
        }
        lines.append(cited.isEmpty ? "cited     nothing" : "cited     " + cited.joined(separator: "\n          "))
        return lines.joined(separator: "\n")
    }

    static func gateBench(_ report: GateBenchReport) -> String {
        var lines: [String] = []
        let nodes = report.nodes.sorted { $0.key < $1.key }.map { "\($0.key) v\($0.value)" }.joined(separator: " · ")
        lines.append("nodes: \(nodes)")
        for set in GateBenchSet.allCases {
            let summaries = report.summaries.filter { $0.set == set }
            guard let first = summaries.first else { continue }
            let note: String
            switch set {
            case .facts: note = "a fact one Thread holds"
            case .pairs: note = "one Thread's fact, then another's: the lead has to move"
            case .unknown: note = "a subject no Thread holds"
            case .generic: note = "text about nothing any Thread holds"
            case .paraphrases: note = "facts in words the corpus does not use (reported only)"
            case .withdrawn: note = "facts from withdrawn documents (reported only)"
            case .crossed: note = "facts a Thread tells in its own words about another Thread's entity (reported only)"
            }
            lines.append("")
            lines.append("\(set.rawValue) · \(first.prompts) prompts · \(note)")
            lines.append(Format.table(
                ["arm", "exact", "owner leads", "cited to owner", "all asked", "largest gate", "share fidelity"],
                summaries.map { s in
                    [s.arm, s.exact.map(Format.pct) ?? "—", s.ownerLeads.map(Format.pct) ?? "—", s.citedToOwner.map(Format.pct) ?? "—",
                     Format.pct(s.allAsked), Format.f(s.largestGate, 2), s.fidelity.map { Format.f($0, 3) } ?? "—"]
                }))
        }
        lines.append("")
        let decision = report.decision
        for arm in report.arms where arm.candidate {
            if let failed = decision.failures[arm.name] {
                lines.append("  ✗ \(arm.name): " + failed.joined(separator: "; "))
            } else {
                lines.append(String(format: "  ✓ %@: eligible, share fidelity %.3f", arm.name, decision.fidelity[arm.name] ?? 0))
            }
        }
        lines.append("decision: \(decision.summary)")
        return lines.joined(separator: "\n")
    }

    static func bench(_ report: VocabularyBenchReport) -> TextTable {
        TextTable(
            headers: ["arm", "trains", "to \(Format.pct(report.target))", "seconds", "epochs", "eval loss", "memorised", "exact", "λ=0", "citation@1"],
            rows: report.rows.map { row in
                [
                    row.name, Format.count(row.trainableParameters),
                    row.epochsToTarget.map { "epoch \($0)" } ?? "—",
                    row.secondsToTarget.map { String(format: "%.1f", $0) } ?? "—",
                    String(row.epochs), Format.f(row.evalLoss), Format.pct(row.memorised), Format.pct(row.exact),
                    Format.pct(row.exactEvidenceOnly), Format.pct(row.citationAt1),
                ]
            })
    }
}

extension BraidGroup {
    struct Profile: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "profile",
            abstract: "Write each live version's knowledge profile: what its Thread knows beyond the commons, as points in the commons' states.",
            discussion: """
                Nodes write the profile at every new version. This writes it for live versions trained before profiles, \
                one node at a time, without retraining: the commons reads the Thread's corpus, each position is weighted \
                by the Thread's lift there, and k-means keeps eight centroids (Docs/ARCHITECTURE.md, "Step 1 v3").
                """,
            shouldDisplay: false)

        @OptionGroup var global: GlobalOptions

        @Option(help: "Only these nodes, comma-separated.")
        var nodes: String?

        @Flag(help: "Write it again where a profile is already there.")
        var force = false

        func run() async throws {
            try await guarded {
                try Preflight.requireMetallib()
                let tokenizer = try await RaoTokenizer.load()
                let names = nodes?.split(separator: ",").map { String($0).trimmingCharacters(in: .whitespaces) }
                let results = try ProfileBuilder.run(layout: BraidLayout(dataRoot: global.root), tokenizer: tokenizer, names: names, force: force) {
                    Console.error($0)
                }
                print(Format.table(
                    ["node", "version", "entries", "lift > 0", "lift total", "commons loss", "own loss", "spread", "k", "s"],
                    results.map { r in
                        [r.name, "v\(r.version)", "\(r.entries)", String(format: "%d (%.0f%%)", r.weighted, Double(r.weighted) / Double(max(1, r.entries)) * 100),
                         String(format: "%.0f", r.liftTotal), String(format: "%.3f", r.commonsLoss), String(format: "%.3f", r.threadLoss),
                         String(format: "%.2f", r.spread), "\(r.k)", r.skipped ? "kept" : String(format: "%.0f", r.seconds)]
                    }))
            }
        }
    }

    struct BenchProfile: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "bench-profile",
            abstract: "Whether the knowledge profile opens the Threads a prompt needs: the route, credit recall, copies, and both sides' answers.",
            discussion: """
                --data-dir is a built braid whose live versions have profiles (raolm braid profile). In process, bench-scale's \
                sets; every prompt is routed without generating, then answered unrouted (the credit the route is judged by) \
                and routed. Two nodes holding the same documents are found and checked as copies. Rules P1 to P4 were fixed \
                before any numbers (Docs/ARCHITECTURE.md, "Step 1 v3").
                """,
            shouldDisplay: false)

        @OptionGroup var global: GlobalOptions

        @Option(help: "Facts per node.")
        var factsPerNode = 10

        @Option(help: "Two-fact prompts in all, spread over the ordered pairs of nodes.")
        var pairs = 60

        @Option(help: "Two nodes that hold the same documents, a,b (default: found by shared documents).")
        var copies: String?

        @Option(help: "Write the report to this JSON file.")
        var out: String?

        @Option(parsing: .upToNextOption, help: "Saved reports to score together instead of running.")
        var report: [String] = []

        func run() async throws {
            try await guarded {
                if !report.isEmpty {
                    var points: [String: ProfilePoint] = [:]
                    for path in report {
                        let saved = try JSONCoding.read(ProfileReport.self, from: URL(fileURLWithPath: (path as NSString).expandingTildeInPath))
                        for point in saved.points { points[point.braid] = point }
                    }
                    print(BraidTables.profile(ProfileBench.report(Array(points.values))))
                    return
                }
                try Preflight.requireMetallib()
                let tokenizer = try await RaoTokenizer.load()
                let pair = copies.flatMap { value -> (String, String)? in
                    let parts = value.split(separator: ",").map { String($0).trimmingCharacters(in: .whitespaces) }
                    return parts.count == 2 ? (parts[0], parts[1]) : nil
                }
                let point = try ProfileBench.run(layout: BraidLayout(dataRoot: global.root), tokenizer: tokenizer, factsPerNode: factsPerNode,
                                                 pairs: pairs, copies: pair) { Console.error($0) }
                let result = ProfileBench.report([point])
                print(BraidTables.profile(result))
                print(String(format: "\n%.0f s", point.seconds))
                if let out {
                    try JSONCoding.write(result, to: URL(fileURLWithPath: out))
                    print("report: \(out)")
                }
            }
        }
    }
}

extension BraidTables {
    static func profile(_ report: ProfileReport) -> String {
        func pct(_ value: Float?) -> String { value.map { String(format: "%.0f%%", $0 * 100) } ?? "—" }
        var lines: [String] = []
        for point in report.points {
            lines.append("\(point.braid): \(point.nodes) Threads, each a profile of \(point.threads.first?.k ?? 0) centroids")
            lines.append(Format.table(
                ["Thread", "world", "entries", "lift > 0", "lift total", "spread"],
                point.threads.map { t in
                    [t.name, t.world ?? "—", "\(t.entries)", "\(t.weighted)", String(format: "%.0f", t.liftTotal), String(format: "%.2f", t.spread)]
                }))
            lines.append("")
            lines.append("route: owner opened on \(point.factsFound)/\(point.facts) facts, both owners on \(point.pairsFound)/\(point.pairs) pairs")
            lines.append(Format.table(["set", "prompts", "every Thread", "opened (mean)", "max", "how many opened → prompts"], point.sets.map { set in
                [set.set, "\(set.prompts)", "\(set.unrouted)", String(format: "%.2f", set.meanOpened), "\(set.maxOpened)",
                 set.histogram.enumerated().filter { $0.element > 0 }.map { "\($0.offset): \($0.element)" }.joined(separator: " · ")]
            }))
            if let worlds = point.worlds, !worlds.isEmpty {
                let columns = Set(worlds.values.flatMap(\.keys)).sorted()
                lines.append("")
                lines.append("Threads opened per fact prompt, by the owner's world (rows) and the opened Thread's world (columns)")
                lines.append(Format.table(["owner's world"] + columns, worlds.keys.sorted().map { row in
                    [row] + columns.map { String(format: "%.2f", worlds[row]?[$0] ?? 0) }
                }))
            }
            lines.append("")
            lines.append("credit recall: \(pct(point.creditRecall)) (facts \(pct(point.creditRecallFacts)), pairs \(pct(point.creditRecallPairs))); routed generations on the dry route: \(point.liveMatchesDry)/\(point.liveChecked)")
            if let subjects = point.subjects {
                lines.append("subject questions: owner opened \(subjects.ownerOpened)/\(subjects.questions), led \(subjects.ownerLeads), exact \(subjects.exact); "
                             + String(format: "%.2f stray Threads opened, %.3f of answer tokens to them; homonyms decided %d/%d",
                                      subjects.strayOpenedMean, subjects.strayShareMean, subjects.homonymsDecided, subjects.homonyms))
                lines.append(Format.table(["subject", "questions", "owner opened", "owner led", "exact"], subjects.byWorld.keys.sorted().map { world in
                    let row = subjects.byWorld[world] ?? [0, 0, 0, 0]
                    return [world] + row.map(String.init)
                }))
            }
            if let copies = point.copies {
                lines.append(String(format: "copies %@ and %@: %d/%d facts within 1%% of the best score, %d opened both, largest difference %.2f%%",
                                    copies.a, copies.b, copies.within, copies.prompts, copies.openedTogether, copies.maxRelativeDifference * 100))
            }
            lines.append(Format.table(
                ["", "facts exact", "led + cited", "cit@1", "pairs moved", "commons leads", "s/token facts", "s/token generic", "s/prompt facts"],
                [("unrouted", point.unrouted), ("routed", point.routed)].map { name, side in
                    [name, "\(side.facts.factsExact)/\(side.facts.facts) (\(pct(side.facts.factsExactRate)))", pct(side.facts.factsOwnedRate),
                     pct(side.facts.citation), "\(side.prompts.pairsMoved)/\(side.prompts.pairs)", pct(side.prompts.commonsLeads),
                     String(format: "%.4f", side.factCost.secondsPerToken), String(format: "%.4f", side.genericCost.secondsPerToken),
                     String(format: "%.3f", side.factCost.secondsPerPrompt)]
                }))
            if !point.missed.isEmpty { lines.append("missed: " + point.missed.prefix(5).joined(separator: "; ")) }
            lines.append("")
        }
        for rule in report.evaluation.rules { lines.append("\(rule.passed ? "pass" : "FAIL")  \(rule.rule): \(rule.detail)") }
        lines.append(report.evaluation.summary)
        return lines.joined(separator: "\n")
    }
}
