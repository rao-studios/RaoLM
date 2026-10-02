//
//  UmbrellaCommand.swift
//  RaoLMCLI
//
//  WHAT: raolm umbrella — the umbrella packs a braid's data root holds. `build` makes the pack a
//        `base` braid uses from SmolLM2-135M (downloaded at the revision RaoLM's tokenizer came
//        from, cut after block 20, anchors and a held-out sample from public-domain books); `list`
//        and `show` say what is registered and where each pack came from; `use` makes a pack the
//        one braids of its shape run on (the owner's say: nothing else changes it).
//

import ArgumentParser
import Foundation
import RaoLM
import RaoLMWorkflows

struct UmbrellaGroup: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "umbrella",
        abstract: "The umbrella pack a base braid's nodes mirror: a pretrained model's vocabulary, its trunk and the commons strand.",
        subcommands: [Build.self, Import.self, List.self, Show.self, Use.self, Corpus.self, Train.self, Trial.self]
    )

    struct Import: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Make a pack from open weights on the Hugging Face hub (a Llama-family model with RaoLM's tokenizer).",
            discussion: """
                Downloads the model at a pinned revision (the repository's current one when --revision is omitted), checks that \
                RaoLM's transformer reads it as Frigate's own Llama does (C0), cuts it at --cut and registers it. A new shape \
                needs its preset (base-360m for SmolLM2-360M); braids of that shape then run on it.
                """)

        @OptionGroup var global: GlobalOptions

        @Option(help: "The repository, e.g. HuggingFaceTB/SmolLM2-360M.")
        var hub: String

        @Option(help: "The commit to pin (default: the repository's current one).")
        var revision: String?

        @Option(help: "The first block of the umbrella's trunk.")
        var cut: Int

        @Option(help: "The pack's name (default: the repository's).")
        var name: String?

        func run() async throws {
            try await guarded {
                try Preflight.requireMetallib()
                let tokenizer = try await RaoTokenizer.load()
                let layout = BraidLayout(dataRoot: global.root)
                let pinned = try revision ?? UmbrellaGroup.currentRevision(hub)
                print("importing \(hub) at \(pinned.prefix(12)), cut at block \(cut)")
                let pack = try UmbrellaPacks.build(source: PackSource.hub(repo: hub, revision: pinned, cut: cut, name: name), layout: layout, tokenizer: tokenizer) { print($0) }
                guard let info = pack.info else { return }
                UmbrellaGroup.describe(info, layout: layout, current: PackRegistry.load(layout).current[PackRegistry.slot(info.config)] == info.sha256)
            }
        }
    }

    /// A repository's current commit on the hub.
    static func currentRevision(_ repo: String) throws -> String {
        guard let url = URL(string: "https://huggingface.co/api/models/\(repo)") else { throw ValidationError("'\(repo)' is not a repository") }
        let data = try Blocking.run { try await URLSession.shared.data(from: url).0 }
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any], let sha = object["sha"] as? String else {
            throw RaoLMFailure("the hub has no model \(repo)", code: 66)
        }
        return sha
    }

    struct Train: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Continue pretraining the commons on public corpora: a new pack, its parent recorded.",
            discussion: """
                Trains the parent pack's whole base model, the embedding included, on the corpora named (each --corpus \
                name or name:weight; weights default to each corpus's size). Warmup, a stable peak, a linear anneal over the last \
                --anneal of the steps. Checkpoints every 250 steps under <data root>/braid/umbrella/runs/<run>; --run <run> \
                resumes one, and a file named STOP in the run's directory stops it at the next step, checkpointed. The new pack is \
                registered, not made current: raolm umbrella use makes it so. Run nothing else on the GPU meanwhile.
                """)

        @OptionGroup var global: GlobalOptions

        @Option(help: "The pack to train from (a name or hash prefix); the current base pack when omitted.")
        var parent: String?

        @Option(help: "A corpus to train on, name or name:weight (repeat for several).")
        var corpus: [String] = []

        @Option(help: "Tokens to train on (e.g. 5M, 50M).")
        var tokens = "5M"

        @Option(help: "The new pack's name.")
        var name = "rao-commons"

        @Option(help: "Window length in tokens.")
        var seqLen = 1024

        @Option(help: "Windows per step.")
        var batch = 8

        @Option(help: "Windows per forward pass; a step accumulates the batch in these (default: the whole batch).")
        var microBatch: Int?

        @Option(help: "Peak learning rate.")
        var lr: Float = 3e-4

        @Option(help: "The share of the steps the rate anneals over at the end.")
        var anneal: Float = 0.2

        @Option(help: "Resume this run (its directory's name).")
        var run: String?

        func run() async throws {
            try await guarded {
                try Preflight.requireMetallib()
                guard !corpus.isEmpty else { throw ValidationError("name at least one --corpus (raolm umbrella corpus list)") }
                let tokenizer = try await RaoTokenizer.load()
                let layout = BraidLayout(dataRoot: global.root)
                let parentPack = try UmbrellaPacks.ensure(layout: layout, config: RaoLMConfig.base, tokenizer: tokenizer, pack: parent)
                var sources: [TokenStream.Source] = []
                var recipeCorpora: [PackRecipe.Corpus] = []
                var heldOut: [String: [Batch]] = [:]
                for entry in corpus {
                    let parts = entry.split(separator: ":", maxSplits: 1).map(String.init)
                    let manifest = try CommonsCorpus.load(layout.corpora, parts[0])
                    let shard = try CommonsCorpus.shard(layout.corpora, parts[0], tokenizer: tokenizer, heldOut: false)
                    let weight = try parts.count > 1 ? (Double(parts[1]).flatMap { $0 > 0 ? $0 : nil } ?? { throw ValidationError("weight '\(parts[1])' must be positive") }())
                        : Double(shard.count)
                    sources.append(TokenStream.Source(name: parts[0], shard: shard, weight: weight))
                    recipeCorpora.append(PackRecipe.Corpus(name: parts[0], sha256: manifest.documentsSHA256, tokens: shard.count, weight: weight))
                    let held = try CommonsCorpus.shard(layout.corpora, parts[0], tokenizer: tokenizer, heldOut: true)
                    let stream = TokenStream(sources: [TokenStream.Source(name: parts[0], shard: held, weight: 1)], seqLen: seqLen, batchSize: 8,
                                             seed: 0, eos: Int32(tokenizer.eosTokenID))
                    let windows = TokenStream.heldOutWindows(held, seqLen: seqLen, count: 32).map { TokenStream.Window(source: 0, start: $0) }
                    heldOut[parts[0]] = stride(from: 0, to: windows.count, by: 8).map { stream.batch(Array(windows[$0 ..< min($0 + 8, windows.count)])) }
                }
                heldOut["pack"] = CommonsTrainer.snippetBatches(parentPack.heldOut.map(\.tokens))
                let runID = run ?? "commons-" + Date().formatted(.iso8601.year().month().day().time(includingFractionalSeconds: false)).replacingOccurrences(of: ":", with: "")
                let spec = CommonsTrainingSpec(
                    runID: runID, parentSHA256: parentPack.sha256, tokenizerSHA256: tokenizer.tokenizerSHA256, corpora: recipeCorpora,
                    tokens: try UmbrellaGroup.count(tokens), seqLen: seqLen, batch: batch, microBatch: microBatch, peakLR: lr, annealFraction: anneal)
                let directory = layout.commonsRuns.appendingPathComponent(runID, isDirectory: true)
                let model = try parentPack.baseModel()
                let trainer = CommonsTrainer(
                    model: model, spec: spec, stream: TokenStream(sources: sources, seqLen: seqLen, batchSize: batch, seed: spec.seed, eos: Int32(tokenizer.eosTokenID)),
                    heldOut: heldOut, directory: directory)
                Console.section("Commons training")
                print("run        \(runID) · \(directory.path)")
                print("parent     \(parentPack.name) \(parentPack.sha256.prefix(12))")
                print("corpora    " + recipeCorpora.map { "\($0.name) (\($0.tokens.formatted()) tokens, weight \(String(format: "%.3g", $0.weight)))" }.joined(separator: " · "))
                print("plan       \(spec.steps) steps × \(batch) × \(seqLen) = \(spec.tokens.formatted()) tokens · peak lr \(lr), warmup \(spec.warmupSteps), anneal the last \(spec.annealSteps) to \(spec.floorLR)")
                let stopFile = directory.appendingPathComponent("STOP")
                let started = Date()
                let state = try trainer.run(shouldStop: { FileManager.default.fileExists(atPath: stopFile.path) }) { event in
                    switch event {
                    case .resumed(let step): print("resumed at step \(step)")
                    case .step(let step, let total, let loss, let rate, let speed):
                        if step % 50 == 0 || step == total {
                            let left = Double(total - step) * Double(seqLen * batch) / max(speed, 1)
                            print(String(format: "step %5d/%d  loss %.4f  lr %.2e  %6.0f tok/s  ~%.0f min left", step, total, loss, rate, speed, left / 60))
                        }
                    case .eval(let result):
                        print("held-out at step \(result.step): " + result.losses.sorted { $0.key < $1.key }.map { String(format: "%@ %.4f", $0.key, $0.value) }.joined(separator: " · "))
                    case .checkpoint(let url): print("checkpoint \(url.lastPathComponent)")
                    }
                }
                guard state.step == spec.steps else {
                    print("stopped at step \(state.step) of \(spec.steps); resume with --run \(runID)")
                    try? FileManager.default.removeItem(at: stopFile)
                    return
                }
                var heldOutLosses: [String: [Float]] = [:]
                if let first = state.evals.first, let last = state.evals.last {
                    for (set, loss) in first.losses { heldOutLosses[set] = [loss, last.losses[set] ?? .nan] }
                }
                let recipe = PackRecipe(
                    corpora: recipeCorpora, runID: runID, steps: spec.steps, tokens: spec.tokens, seqLen: seqLen, batch: batch, peakLR: lr,
                    warmupSteps: spec.warmupSteps, annealSteps: spec.annealSteps, weightDecay: spec.weightDecay, seed: spec.seed, heldOut: heldOutLosses)
                let pack = try UmbrellaPacks.continued(model: model, parent: parentPack, name: name, recipe: recipe, layout: layout, tokenizer: tokenizer)
                print(String(format: "\ntrained in %.1f h", Date().timeIntervalSince(started) / 3600))
                guard let info = pack.info else { return }
                UmbrellaGroup.describe(info, layout: layout, current: false)
                for (set, losses) in heldOutLosses.sorted(by: { $0.key < $1.key }) {
                    print(String(format: "held-out   %@: %.4f → %.4f", set, losses[0], losses[1]))
                }
                print("\nregistered, not current: raolm umbrella use \(pack.sha256.prefix(12)) makes braids of its shape run on it")
            }
        }
    }

    /// "40M", "500k", "1.5B" or a plain count.
    static func count(_ text: String) throws -> Int {
        let trimmed = text.trimmingCharacters(in: .whitespaces).uppercased()
        let scale: Double = trimmed.hasSuffix("B") ? 1e9 : trimmed.hasSuffix("M") ? 1e6 : trimmed.hasSuffix("K") ? 1e3 : 1
        guard let value = Double(scale == 1 ? trimmed : String(trimmed.dropLast())), value > 0 else {
            throw ValidationError("'\(text)' is not a count (e.g. 40M, 500k)")
        }
        return Int(value * scale)
    }

    struct Corpus: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "The public text the commons is trained on: books, open web text, files.",
            subcommands: [Add.self, List.self, Show.self])

        struct Add: AsyncParsableCommand {
            static let configuration = CommandConfiguration(
                abstract: "Fetch, store and tokenize a corpus once (one source per corpus).",
                discussion: """
                    Project Gutenberg: --gutenberg 1342,84 or --gutenberg-top 100 (the most downloaded, English only; the pack's own \
                    books are always left out, so its held-out snippets stay unseen). A hub dataset: --hub HuggingFaceFW/fineweb-edu \
                    --subset sample-10BT --split train --max-tokens 40M, read through the datasets-server in pages at seeded offsets. \
                    Files: --files a.jsonl b.txt (a .jsonl line's "text" is one document; a .txt file is one). 1% of the documents \
                    (2 to 256) are held out and never trained on.
                    """)

            @OptionGroup var global: GlobalOptions

            @Option(help: "The corpus's name: lowercase letters, digits, dots, dashes.")
            var name: String

            @Option(help: "Project Gutenberg ids, comma-separated.")
            var gutenberg: String?

            @Option(help: "The most downloaded Project Gutenberg books, up to this many.")
            var gutenbergTop: Int?

            @Option(help: "A dataset on the Hugging Face hub, read through its datasets-server.")
            var hub: String?

            @Option(help: "The dataset's subset (its config).")
            var subset = "default"

            @Option(help: "The dataset's split.")
            var split = "train"

            @Option(help: "Stop reading the dataset after this many tokens (e.g. 40M).")
            var maxTokens = "10M"

            @Option(parsing: .upToNextOption, help: "Local .jsonl or .txt files.")
            var files: [String] = []

            func run() async throws {
                try await guarded {
                    guard CommonsCorpus.isValidName(name) else { throw ValidationError("'\(name)' is not a corpus name (lowercase letters, digits, dots, dashes)") }
                    let sources = [gutenberg != nil || gutenbergTop != nil, hub != nil, !files.isEmpty].filter { $0 }.count
                    guard sources == 1 else { throw ValidationError("give one source: --gutenberg/--gutenberg-top, --hub, or --files") }
                    let tokenizer = try await RaoTokenizer.load()
                    let layout = BraidLayout(dataRoot: global.root)
                    let progress: (String) -> Void = { print($0) }
                    let fetched: (documents: [CommonsDocument], provenance: [String: String])
                    let source: String
                    let license: String
                    if gutenberg != nil || gutenbergTop != nil {
                        let ids = try (gutenberg ?? "").split(separator: ",").map { part -> Int in
                            guard let id = Int(part.trimmingCharacters(in: .whitespaces)) else { throw ValidationError("'\(part)' is not a Gutenberg id") }
                            return id
                        }
                        let exclude = Set(PackSource.smolLM2_135M.books.map(\.id))
                        fetched = try GutenbergText.documents(
                            ids: ids, top: gutenbergTop, exclude: exclude, cache: layout.packs.appendingPathComponent("texts", isDirectory: true), progress: progress)
                        source = gutenbergTop.map { "gutenberg:top-\($0)" } ?? "gutenberg:ids"
                        license = "Public domain in the USA, from Project Gutenberg (its license header and footer removed; the trademark is not used)"
                    } else if let hub {
                        let budget = try UmbrellaGroup.count(maxTokens)
                        fetched = try DatasetRows.documents(
                            dataset: hub, config: subset, split: split, maxTokens: budget, tokenizer: tokenizer, seed: 0x5EED_C044,
                            cache: layout.packs.appendingPathComponent("texts", isDirectory: true), progress: progress)
                        source = "hub:\(hub)/\(subset)/\(split)"
                        license = hub.lowercased().contains("fineweb") ? "ODC-By 1.0, subject to CommonCrawl's terms of use" : "as the dataset card states"
                    } else {
                        fetched = try CommonsFiles.documents(files)
                        source = "files"
                        license = "the owner's files"
                    }
                    let manifest = try CommonsCorpus.write(
                        name: name, documents: fetched.documents, source: source, license: license, provenance: fetched.provenance,
                        tokenizer: tokenizer, root: layout.corpora, progress: progress)
                    UmbrellaGroup.describe(manifest, root: layout.corpora)
                }
            }
        }

        struct List: AsyncParsableCommand {
            static let configuration = CommandConfiguration(abstract: "The corpora under the data root.")

            @OptionGroup var global: GlobalOptions

            func run() async throws {
                try await guarded {
                    let layout = BraidLayout(dataRoot: global.root)
                    let corpora = CommonsCorpus.list(layout.corpora)
                    guard !corpora.isEmpty else {
                        print("no corpora under \(layout.corpora.path) (raolm umbrella corpus add makes one)")
                        return
                    }
                    print(Format.table(
                        ["corpus", "source", "documents", "held out", "training tokens", "held-out tokens", "MB"],
                        corpora.map { m in
                            let tokens = m.tokens.first
                            return [m.name, Format.clip(m.source, 40), m.documents.formatted(), "\(m.heldOutDocuments)",
                                    tokens.map { $0.train.formatted() } ?? "—", tokens.map { $0.heldOut.formatted() } ?? "—",
                                    String(format: "%.1f", Double(m.bytes) / 1e6)]
                        }))
                }
            }
        }

        struct Show: AsyncParsableCommand {
            static let configuration = CommandConfiguration(abstract: "One corpus: its source, license, split, tokens and provenance.")

            @OptionGroup var global: GlobalOptions

            @Argument var name: String

            func run() async throws {
                try await guarded {
                    let layout = BraidLayout(dataRoot: global.root)
                    UmbrellaGroup.describe(try CommonsCorpus.load(layout.corpora, name), root: layout.corpora)
                }
            }
        }
    }

    static func describe(_ manifest: CommonsCorpusManifest, root: URL) {
        Console.section("Commons corpus")
        print("corpus     \(manifest.name)")
        print("source     \(manifest.source)")
        print("license    \(manifest.license)")
        print("documents  \(manifest.documents.formatted()) (\(manifest.heldOutDocuments) held out), \(String(format: "%.1f", Double(manifest.bytes) / 1e6)) MB")
        for tokens in manifest.tokens {
            print("tokens     \(tokens.train.formatted()) to train on, \(tokens.heldOut.formatted()) held out (tokenizer \(tokens.tokenizerSHA256.prefix(12)))")
        }
        print("hash       \(manifest.documentsSHA256)")
        for (key, value) in manifest.provenance.sorted(by: { $0.key < $1.key }) where !value.isEmpty {
            print("\(key.padding(toLength: 10, withPad: " ", startingAt: 0)) \(Format.clip(value, 100))")
        }
        print("directory  \(CommonsCorpus.directory(root, manifest.name).path)")
    }

    struct List: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Every registered pack, its lineage, and which one each shape uses.")

        @OptionGroup var global: GlobalOptions

        func run() async throws {
            try await guarded {
                let layout = BraidLayout(dataRoot: global.root)
                let registry = PackRegistry.load(layout)
                guard !registry.packs.isEmpty else {
                    print("no packs under \(layout.packs.path) (raolm umbrella build makes the first)")
                    return
                }
                let current = Set(registry.current.values)
                print(Format.table(
                    ["", "pack", "name", "v", "shape", "parent", "source", "created"],
                    registry.packs.map { entry in
                        [current.contains(entry.sha256) ? "●" : "", String(entry.sha256.prefix(12)), entry.name, "\(entry.version)", entry.slot,
                         entry.parent.map { String($0.prefix(12)) } ?? "—", Format.clip(entry.source, 44),
                         entry.createdAt.formatted(date: .abbreviated, time: .shortened)]
                    }))
                print("\n● current for its shape · raolm umbrella use <pack> changes it")
            }
        }
    }

    struct Show: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "One pack: its hashes, shape, commons samples, lineage and how it was made.")

        @OptionGroup var global: GlobalOptions

        @Argument(help: "A pack's name, a prefix of its hash, or a shape (30x576-cut20).")
        var pack: String

        func run() async throws {
            try await guarded {
                let layout = BraidLayout(dataRoot: global.root)
                let registry = PackRegistry.load(layout)
                guard let entry = registry.resolve(pack) else { throw RaoLMFailure("no pack '\(pack)' under \(layout.packs.path)", hint: "raolm umbrella list", code: 66) }
                let info = try JSONCoding.read(PackInfo.self, from: layout.pack(sha256: entry.sha256).appendingPathComponent(UmbrellaPack.infoFile))
                UmbrellaGroup.describe(info, layout: layout, current: registry.current[entry.slot] == entry.sha256)
                let lineage = registry.lineage(entry.sha256)
                if !lineage.isEmpty {
                    print("lineage    " + lineage.map { "\($0.name) \($0.sha256.prefix(12))" }.joined(separator: " ← "))
                }
                if let recipe = info.recipe {
                    print("recipe     run \(recipe.runID): \(recipe.steps) steps, \(recipe.tokens.formatted()) tokens, seq \(recipe.seqLen) × batch \(recipe.batch), peak lr \(recipe.peakLR), warmup \(recipe.warmupSteps), anneal \(recipe.annealSteps)")
                    for corpus in recipe.corpora {
                        print("           \(corpus.name): \(corpus.tokens.formatted()) tokens, weight \(corpus.weight), \(corpus.sha256.prefix(12))")
                    }
                    for (set, losses) in recipe.heldOut.sorted(by: { $0.key < $1.key }) where losses.count == 2 {
                        print(String(format: "           held-out %@: %.4f → %.4f", set, losses[0], losses[1]))
                    }
                }
            }
        }
    }

    struct Use: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Make a pack the one braids of its shape run on.",
            discussion: "A braid already running on another pack keeps it until it is rebased (raolm braid rebase); its nodes then retrain from the new base.")

        @OptionGroup var global: GlobalOptions

        @Argument(help: "A pack's name or a prefix of its hash.")
        var pack: String

        func run() async throws {
            try await guarded {
                let layout = BraidLayout(dataRoot: global.root)
                var registry = PackRegistry.load(layout)
                guard let entry = registry.resolve(pack) else { throw RaoLMFailure("no pack '\(pack)' under \(layout.packs.path)", hint: "raolm umbrella list", code: 66) }
                let previous = registry.current[entry.slot]
                try registry.use(entry.sha256)
                try registry.save(layout)
                print("\(entry.slot): \(previous.map { String($0.prefix(12)) } ?? "none") → \(entry.sha256.prefix(12)) (\(entry.name))")
            }
        }
    }

    struct Trial: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "A trial braid for a pack: a reference braid's documents on it, in a new data root beside the reference.",
            discussion: """
                The step between making a commons (import, train) and choosing it (use). The reference braid's documents and feeds \
                are copied (never its versions), with the preset of the pack's shape and the pack itself, so any shape can be tried. \
                braid sync then trains every Thread from the pack's base, braid bench-commons compares the trial with the \
                reference, and the studio's catalog lists both to load and ask. The reference is only read; --data-dir is the \
                workshop the pack is registered in.
                """)

        @OptionGroup var global: GlobalOptions

        @Argument(help: "A pack's name or a prefix of its hash.")
        var pack: String

        @Option(help: "The reference braid's data root, whose documents the trial is fed.")
        var from: String

        @Option(help: "The trial's directory name, beside the reference (default: braid-<pack name>).")
        var name: String?

        func run() async throws {
            try await guarded {
                let workshop = BraidLayout(dataRoot: global.root)
                guard let entry = PackRegistry.load(workshop).resolve(pack) else {
                    throw RaoLMFailure("no pack '\(pack)' under \(workshop.packs.path)", hint: "raolm umbrella list --data-dir <workshop>", code: 66)
                }
                let reference = DataRoot.resolve(argument: from)
                let root = reference.url.deletingLastPathComponent().appendingPathComponent(name ?? "braid-\(entry.name)", isDirectory: true)
                let made = try BraidTrial.make(pack: entry, workshop: workshop, reference: BraidLayout(dataRoot: reference), into: root)
                let offline = BraidCatalog.entry(root: reference.url, name: "", isHome: false)?.offline == true
                print("trial braid  \(made.root.path)")
                print("pack         \(entry.name) \(entry.sha256.prefix(12)) (\(entry.slot)), preset \(made.preset)")
                print("documents    \(made.nodes.joined(separator: ", ")) from \(reference.url.path); no version copied, so every Thread trains from the new base")
                print("\nnext:")
                print("  raolm braid sync\(offline ? " --offline" : "") --data-dir \(made.root.path)")
                print("  raolm braid bench-commons --data-dir \(made.root.path) --reference \(reference.url.path) --out \(made.root.path)/braid/bench/commons.json")
                print("  the studio lists it: ↑↓ to it, ⏎ loads it")
                print("  raolm umbrella use \(entry.name) --data-dir \(global.root.url.path)   (the owner's say)")
            }
        }
    }

    static func describe(_ info: PackInfo, layout: BraidLayout, current: Bool) {
        Console.section("Umbrella pack")
        print("pack       \(info.sha256)\(current ? "  (current for \(PackRegistry.slot(info.config)))" : "")")
        print("seam       \(info.seam)  (what a node trained under it holds: vocabulary and trunk)")
        print("name       \(info.name), pack v\(info.version)")
        print("source     \(info.source)")
        print("shape      \(info.config.numHiddenLayers) blocks × \(info.config.hiddenSize); a node trains blocks 0..<\(info.config.cut) (\(info.config.nodeParameterCount.formatted()) parameters), blocks \(info.config.cut)..<\(info.config.numHiddenLayers) are the umbrella's")
        print("vocabulary \(info.vocabularySHA256)")
        if let difference = info.referenceMaxDifference {
            print(String(format: "reference  largest logit difference from Frigate's Llama on the same weights: %.2e (C0 holds under %.0e)", difference, ReferenceCheck.tolerance))
        }
        print("commons    \(info.anchorCount) anchors, \(info.heldOutCount) held-out snippets, from \(info.texts.map(\.title).joined(separator: ", "))")
        if let parent = info.parent { print("parent     \(parent)") }
        print("directory  \(layout.pack(sha256: info.sha256).path)")
    }

    struct Build: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Build the pack from SmolLM2-135M (downloaded once, about 270 MB) under <data root>/braid/umbrella.")

        @OptionGroup var global: GlobalOptions

        @Option(help: "The first block of the umbrella's trunk; the node trains the blocks before it.")
        var cut = RaoLMConfig.base.cut

        @Flag(help: "Build it again even when it exists.")
        var force = false

        func run() async throws {
            try await guarded {
                try Preflight.requireMetallib()
                let tokenizer = try await RaoTokenizer.load()
                var config = RaoLMConfig.base
                config.cut = cut
                try config.validate()
                let layout = BraidLayout(dataRoot: global.root)
                let progress: (String) -> Void = { print($0) }
                let pack = force
                    ? try UmbrellaPacks.build(source: try PackSource.source(for: config), layout: layout, tokenizer: tokenizer, progress: progress)
                    : try UmbrellaPacks.ensure(layout: layout, config: config, tokenizer: tokenizer, progress: progress)
                guard let info = pack.info else { return }
                let registry = PackRegistry.load(layout)
                UmbrellaGroup.describe(info, layout: layout, current: registry.current[PackRegistry.slot(info.config)] == info.sha256)
            }
        }
    }
}
