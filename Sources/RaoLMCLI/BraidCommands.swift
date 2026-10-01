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
        subcommands: [Panel.self, Demo.self, Status.self, Down.self, BenchVocabulary.self, BenchGate.self, BenchTrajectory.self, BenchUmbrella.self, BenchArchitecture.self, BenchThought.self],
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
                let pack = try UmbrellaPacks.ensure(layout: options.layout, config: try options.settings.modelConfig(), tokenizer: tokenizer) {
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
