import Foundation
import MLX
import Testing

@testable import RaoLMBraid
@testable import RaoLMCore
@testable import RaoLMModel
@testable import RaoLMProvenance
@testable import RaoLMTraining

extension Rig {
    /// The rig's shape cut after its first block: block 1 is the umbrella's trunk.
    static var cutConfig: RaoLMConfig {
        var config = config
        config.cut = 1
        return config
    }

    /// A pack cut from a random model of the rig's shape, with anchors and a held-out sample.
    func pack(seed: UInt64 = 5) throws -> UmbrellaPack {
        let source = try RaoTransformer.make(config: Self.cutConfig, seed: seed)
        let vocabulary = VocabularyPack.from(model: source, tokenizerSHA256: tokenizer.tokenizerSHA256, originSHA256: "test")
        var base: [String: MLXArray] = [:]
        for (key, value) in source.parameters().flattened() where RaoTransformer.blockIndex(ofKey: key) != nil { base[key] = value }
        let trunk = base.filter { (RaoTransformer.blockIndex(ofKey: $0.key) ?? -1) >= 1 }.map { (key: $0.key, value: $0.value) }
        let anchors = (0..<12).map { i in PackSnippet(source: "test", tokens: tokenizer.encode("An anchor sentence number \(i) about the weather and the sea.")) }
        let info = PackInfo(
            name: "test", source: "test", config: Self.cutConfig,
            sha256: UmbrellaPack.fingerprint(embedding: vocabulary.embedding, norm: vocabulary.norm, trunk: trunk),
            vocabularySHA256: vocabulary.sha256, anchorCount: anchors.count, heldOutCount: 1, texts: [], tokenizerSHA256: tokenizer.tokenizerSHA256)
        return UmbrellaPack(
            vocabulary: vocabulary, info: info, base: base, anchors: anchors,
            anchorStates: UmbrellaPack.anchorStates(model: source, anchors: anchors.map(\.tokens)),
            heldOut: [PackSnippet(source: "test", tokens: tokenizer.encode("The tide came in slowly that afternoon, filling the channels."))])
    }

    func hypervisor(_ name: String, pack: UmbrellaPack) throws -> ThreadHypervisor {
        var settings = self.settings
        settings.config = Self.cutConfig
        settings.warmLR = 4e-3
        return try ThreadHypervisor(name: name, label: name.capitalized, layout: layout.node(name), pack: pack, tokenizer: tokenizer,
                                    source: try source(name), settings: settings, owner: "raolm-test")
    }
}

extension BraidMLXSuites {
    @Suite("The umbrella's pack in a braid", .serialized)
    struct UmbrellaBraidTests {
        @Test("a node on a pack starts from the base, trains only its own blocks, and the commons strand joins the mixture")
        func packBraid() async throws {
            var rig = try await Rig(names: ["ambient", "craft"], documentsPerNode: 6, seed: 7)
            defer { rig.cleanUp() }
            rig.settings.scratchSteps = 64
            rig.settings.continueSteps = 32
            rig.settings.memorisedFloor = 0
            let pack = try rig.pack()
            var nodes: [ThreadHypervisor] = []
            for name in ["ambient", "craft"] {
                _ = try await rig.feed(name, 3)
                let node = try rig.hypervisor(name, pack: pack)
                #expect(try node.sync() == [1])
                nodes.append(node)
            }
            let ambient = nodes[0]
            let live = try #require(ambient.live)
            #expect(pack.matches(live.model), "the trunk is the pack's, byte for byte")
            #expect(live.index.info.cut == 1 && live.index.info.tapLayer == 0)
            let version = try JSONCoding.read(NodeVersion.self, from: ambient.layout.version(1).appendingPathComponent(NodeVersion.fileName))
            #expect(version.gates.contains { $0.name == "trunk" && $0.passed })
            #expect(version.packSHA256 == pack.sha256)
            #expect(version.heldOutLoss != nil, "the feeder left unfed documents beside the node")
            #expect(version.commonsLoss != nil)
            #expect(ambient.state.blockActivity.count == 1, "only the node's own block moves")
            let descriptor = try #require(ambient.descriptor())
            #expect(descriptor.packSHA256 == pack.sha256 && descriptor.cut == 1)
            #expect(descriptor.anchors?.count == pack.anchors.count * Rig.config.hiddenSize)

            // Continuing keeps the trunk frozen.
            _ = try await rig.feed("ambient", 3)
            #expect(try ambient.sync().contains(3))
            #expect(pack.matches(try #require(ambient.live).model))

            // The umbrella: the Threads and the commons strand, shares exact, every Thread's lift over the commons.
            let umbrella = try BraidUmbrella(pack: pack, tokenizer: rig.tokenizer)
            let links = nodes.compactMap(\.live).map { LocalStrandLink(strand: $0, vocabularySHA256: pack.vocabulary.sha256, packSHA256: pack.sha256) }
            let example = try #require(rig.examples(nodes).first { $0.resolvedKind == .fact })
            var params = links[0].strand.context.defaultParameters()
            params.maxTokens = 6
            for thought in [false, true] {
                var gate = BraidRequest.defaultGate
                gate.thoughtAgreement = thought
                let generation = try umbrella.generate(links: links, request: BraidRequest(
                    promptTokens: example.promptTokens, promptText: example.promptText, params: params, gate: gate))
                let braid = try #require(generation.braid)
                #expect(braid.strands.map(\.name) == ["ambient", "craft", "commons"])
                #expect(braid.strands.last?.isCommons == true && braid.packSHA256 == pack.sha256)
                for trace in generation.traces {
                    let shares = try #require(trace.strands)
                    #expect(shares.count == 3)
                    #expect(abs(shares.map(\.share).reduce(0, +) - 1) < 1e-4, "shares sum to one at \(trace.index)")
                    #expect(shares[2].open, "the commons is always asked")
                    #expect(shares[2].lift == nil)
                    for share in shares.prefix(2) where share.lmProb != nil { #expect(share.lift != nil) }
                }
                if thought { #expect(generation.traces.contains { $0.strands?.contains { $0.thought != nil } == true }) }
            }
            // A Thread that runs another pack is refused.
            let other = try rig.pack(seed: 6)
            #expect(throws: BraidError.self) { _ = try BraidedGenerator(links: links, head: UmbrellaHead(vocabulary: pack.vocabulary), tokenizer: rig.tokenizer,
                                                                        packSHA256: other.sha256) }
        }

        @Test("every version writes its knowledge profile beside its index, and a profile-routed generation runs on the route computed beforehand")
        func profile() async throws {
            var rig = try await Rig(names: ["ambient", "craft"], documentsPerNode: 6, seed: 7)
            defer { rig.cleanUp() }
            rig.settings.scratchSteps = 64
            rig.settings.memorisedFloor = 0
            let pack = try rig.pack()
            var nodes: [ThreadHypervisor] = []
            for name in ["ambient", "craft"] {
                _ = try await rig.feed(name, 3)
                let node = try rig.hypervisor(name, pack: pack)
                #expect(try node.sync() == [1])
                nodes.append(node)
            }
            let live = try #require(nodes[0].live)
            let directory = RunLayout.provenance(live.context.runDirectory, epoch: live.context.epoch)
            #expect(FileManager.default.fileExists(atPath: directory.appendingPathComponent(ThreadProfileFiles.arrays).path))
            let profile = try #require(live.profile())
            #expect(profile.hidden == Rig.config.hiddenSize && profile.k >= 1 && profile.k <= ProfileSettings.k)
            #expect(profile.info.entries == live.index.count && profile.info.weighted > 0)
            #expect(profile.info.indexSHA256 == live.index.sha256 && profile.info.commonsPackSHA256 == pack.sha256)
            #expect(try #require(nodes[0].descriptor()).profile?.count == profile.k * profile.hidden)
            // The same corpus, the same commons: the same centroids.
            let again = try #require(try ThreadProfile.compute(
                commons: try pack.baseModel(), index: live.index, corpus: try live.context.tokenizedCorpus(),
                seqLen: live.context.manifest.hyperparameters.seqLen, packSHA256: pack.sha256, epoch: live.context.epoch))
            #expect(again.k == profile.k && zip(again.centroids, profile.centroids).allSatisfy { abs($0 - $1) < 1e-3 })

            let umbrella = try BraidUmbrella(pack: pack, tokenizer: rig.tokenizer)
            let links = nodes.compactMap(\.live).map { LocalStrandLink(strand: $0, vocabularySHA256: pack.vocabulary.sha256, packSHA256: pack.sha256) }
            let generator = try umbrella.generator(links: links)
            #expect(generator.profileRouter.profiled == 2)
            var params = links[0].strand.context.defaultParameters()
            params.maxTokens = 6
            for example in rig.examples(nodes).filter({ $0.resolvedKind == .fact }).prefix(3) {
                let route = try generator.profileRoute(example.promptTokens)
                #expect(route.candidates.last == true, "the commons always opens")
                let request = BraidRequest(promptTokens: example.promptTokens, promptText: example.promptText, params: params, routing: true,
                                           router: .profile)
                let routed = try generator.generate(request)
                #expect(routed.braid?.strands.map(\.name) == route.indices.map { generator.names[$0] })
                var plain = request
                plain.routing = false
                let direct = try BraidedGenerator(links: route.indices.map { generator.links[$0] }, head: generator.head, tokenizer: rig.tokenizer,
                                                  packSHA256: pack.sha256).generate(plain)
                #expect(routed.tokens == direct.tokens)
                for (a, b) in zip(routed.traces, direct.traces) {
                    #expect(zip(a.strands ?? [], b.strands ?? []).allSatisfy { $0.strand == $1.strand && abs($0.share - $1.share) < 1e-6 })
                }
            }
        }

        @Test("a braid of one Thread with its own λ and τ reproduces CitedGenerator number for number")
        func calibrated() async throws {
            var rig = try await Rig(names: ["solo"], documentsPerNode: 3, seed: 11)
            defer { rig.cleanUp() }
            rig.settings.scratchSteps = 48
            rig.settings.memorisedFloor = 0
            _ = try await rig.feed("solo", 3)
            let node = try rig.hypervisor("solo")
            #expect(try node.sync() == [1])
            var live = try #require(node.live)
            var calibration = SelfTrajectory.calibrate(
                model: live.model, index: live.index, corpus: try live.context.tokenizedCorpus(), alpha: live.index.info.alpha)
            #expect(calibration.documents == 3 && calibration.positions > 0)
            #expect(calibration.tauLogLikelihood.count == SelfTrajectory.tauGrid.count)
            #expect(calibration.lambdaScale >= SelfTrajectory.lambdaFloor && calibration.lambdaScale <= 1)
            // A calibration that moves both numbers, written where the version's index lives.
            calibration.tau = 0.035
            calibration.lambdaScale = 0.6
            var info = live.index.info
            info.calibration = calibration
            let directory = node.layout.version(1)
            try JSONCoding.write(info, to: RunLayout.provenance(directory, epoch: live.context.epoch).appendingPathComponent(ProvenanceIndexFiles.info))
            let context = try RunContext.load(runDirectory: directory, epoch: live.context.epoch, allowWeakIndex: true, tokenizer: rig.tokenizer)
            live = ThreadStrand(name: "solo", label: "Solo", version: 1, context: context, owner: "raolm-test")

            let example = try #require(rig.examples([node]).first { $0.node == "solo" && $0.resolvedKind == .fact })
            var params = context.defaultParameters()
            params.maxTokens = 10
            let single = try context.generator().generate(GenerationRequest(
                promptTokens: example.promptTokens, promptText: example.promptText, params: params))
            let link = LocalStrandLink(strand: live, vocabularySHA256: rig.vocabulary.sha256)
            #expect(link.descriptor.calibration?.tau == 0.035)
            let braided = try BraidedGenerator(links: [link], head: UmbrellaHead(vocabulary: rig.vocabulary), tokenizer: rig.tokenizer)
                .generate(BraidRequest(promptTokens: example.promptTokens, promptText: example.promptText, params: params))
            #expect(braided.tokens == single.tokens)
            for (a, b) in zip(braided.traces, single.traces) {
                #expect(a.token == b.token && abs(a.lambda - 0.3) < 1e-6 && abs(b.lambda - 0.3) < 1e-6)
                #expect(abs(a.lmProb - b.lmProb) < 1e-5 && abs(a.mixedProb - b.mixedProb) < 1e-5)
                #expect(abs(a.mixedEntropy - b.mixedEntropy) < 1e-4)
                #expect(a.neighbours.map(\.entry) == b.neighbours.map(\.entry))
                #expect(zip(a.neighbours.map(\.weight), b.neighbours.map(\.weight)).allSatisfy { abs($0 - $1) < 1e-5 })
            }
        }

        @Test("two strands' descriptions of the same state agree; of unrelated states, they do not")
        func thought() {
            var rng = SplitMix64(seed: 9)
            let width = 16
            let anchors = (0..<(24 * width)).map { _ in Float(rng.nextUnit() * 2 - 1) }
            func descriptor(_ name: String) -> StrandDescriptor {
                StrandDescriptor(
                    name: name, label: name, threadID: nil, version: 1,
                    manifest: ManifestRef(runID: name, epoch: 1, checkpointSHA256: "", indexSHA256: "", corpusHash: "", tokenizerSHA256: "",
                                          ledgerSHA256: nil, threadID: nil),
                    vocabularySHA256: "v", hiddenSize: width, tapLayer: 0, alpha: 0.5, defaultTau: 0.05, defaultK: 16, indexEntries: 0,
                    partitions: [], sharedNgrams: PackedWords([]), owner: "", anchors: PackedFloats(anchors))
            }
            let reader = ThoughtReader(descriptors: [descriptor("a"), descriptor("b")])
            #expect(reader.reads(0) && reader.reads(1))
            let state = Array(anchors[(3 * width)..<(4 * width)])
            let unrelated = (0..<width).map { _ in Float(rng.nextUnit() * 2 - 1) }
            let a = reader.describe(0, states: [state, unrelated])!
            let b = reader.describe(1, states: [state])!
            #expect(reader.agreement(0, 1, a[0], b[0]) > 0.9)
            #expect(reader.agreement(0, 1, a[1], b[0]) < 0.5)
        }
    }
}

extension BraidMLXSuites {
    @Suite("The question adapter", .serialized)
    struct QuestionAdapterTests {
        @Test("a random commons rewrites nothing useful, so the rules take over; the question keeps its subject; a known template rewrites exactly")
        func fallback() async throws {
            var rig = try await Rig(names: ["solo"], documentsPerNode: 2, seed: 3)
            defer { rig.cleanUp() }
            let pack = try rig.pack()
            let adapter = QuestionAdapter(model: try pack.baseModel(), tokenizer: rig.tokenizer)
            let rewrite = adapter.rewrite("When was Tillyburn founded?")
            // A random model writes noise: refused, and the dataset's template answers instead.
            #expect(rewrite.rewriter == "rules" && rewrite.stem == "The article says Tillyburn was founded in")
            #expect(rewrite.question == "When was Tillyburn founded?" && rewrite.seconds >= 0)
            // A question outside the templates, with noise from the model, is asked as written.
            let unknown = adapter.rewrite("Tell me everything about Tillyburn?")
            #expect(unknown.rewriter == "none" && unknown.stem == "Tell me everything about Tillyburn?")
            // The acceptance rule itself.
            #expect(QuestionAdapter.acceptable("Tillyburn was founded in", for: "When was Tillyburn founded?"))
            #expect(!QuestionAdapter.acceptable("Who founded Tillyburn", for: "When was Tillyburn founded?"))
            #expect(!QuestionAdapter.acceptable("The town was founded in", for: "When was Tillyburn founded?"))
            #expect(!QuestionAdapter.acceptable("", for: "When was Tillyburn founded?"))
            #expect(QuestionAdapter.clean(" The mayor of Paris is ___\n") == "The mayor of Paris is")
            #expect(adapter.prompt(for: "Who?").hasSuffix("Question: Who?\nStem:"))
        }
    }
}

extension BraidMLXSuites {
    @Suite("A Thread's own context for a stem", .serialized)
    struct StrandContextTests {
        @Test("the context is the sentence before the fact's, from the Thread's index, only for a subject the Thread holds; the braid completes behind it")
        func context() async throws {
            var rig = try await Rig(names: ["solo"], documentsPerNode: 3, seed: 11)
            defer { rig.cleanUp() }
            rig.settings.scratchSteps = 48
            rig.settings.memorisedFloor = 0
            _ = try await rig.feed("solo", 3)
            let node = try rig.hypervisor("solo")
            #expect(try node.sync() == [1])
            let live = try #require(node.live)
            let example = try #require(rig.examples([node]).first { $0.node == "solo" && $0.resolvedKind == .fact })
            // The fact's own prompt, as the corpus wrote it: the Thread recognises it, and the context is
            // the sentence before it, which the slice prompt already carries.
            let source = try #require(example.source)
            let document = try #require(rig.world.document(id: source.documentID))
            let partition = try #require(document.partitions.first { $0.index == source.partitionIndex })
            let fact = try #require(document.facts.first { $0.partitionIndex == source.partitionIndex && example.expected == $0.answer })
            let stem = rig.tokenizer.encode(fact.prompt)
            let found = try live.context(for: stem, k: 16, floor: 0.5)
            let context = try #require(found)
            #expect(context.documentID == source.documentID && context.score >= 0.5)
            let text = rig.tokenizer.decode(context.tokens).trimmingCharacters(in: .whitespacesAndNewlines)
            let before = String(decoding: Array(partition.text.utf8).prefix(fact.sentenceStart), as: UTF8.self)
            #expect(!text.isEmpty && before.contains(text), "context “\(text)” is not before the fact in “\(before.suffix(120))”")
            #expect(!text.contains(fact.answer.trimmingCharacters(in: .whitespaces)), "the context must not carry the answer")
            // A subject the Thread never saw: nothing.
            let foreign = rig.tokenizer.encode("Zorblax Quendrim was founded in")
            #expect(try live.context(for: foreign, subject: 0..<4, k: 16, floor: 0.5) == nil)
            // Through the braid with context on, the strand's ref records what it put before the prompt.
            var params = live.context.defaultParameters()
            params.maxTokens = 4
            let generation = try rig.generator([node]).generate(BraidRequest(
                promptTokens: stem, promptText: fact.prompt, params: params, context: true, contextFloor: 0.5))
            #expect(generation.braid?.strands.first?.context?.documentID == source.documentID)
            #expect(generation.prompt.tokens == stem, "the recorded prompt stays the stem")
        }
    }
}

extension BraidMLXSuites {
    @Suite("Pack v2: two hashes, written once, registered with lineage", .serialized)
    struct PackV2Tests {
        /// A v2 pack of the rig's cut shape; `bump` adds to one block below the cut.
        static func pack(_ rig: Rig, seed: UInt64 = 5, bump: Float = 0, parent: String? = nil, name: String = "test") throws -> UmbrellaPack {
            let source = try RaoTransformer.make(config: Rig.cutConfig, seed: seed)
            let vocabulary = VocabularyPack.from(model: source, tokenizerSHA256: rig.tokenizer.tokenizerSHA256, originSHA256: "test")
            var base: [String: MLXArray] = [:]
            for (key, value) in source.parameters().flattened() where RaoTransformer.blockIndex(ofKey: key) != nil { base[key] = value }
            let lower = try #require(base.keys.filter { RaoTransformer.blockIndex(ofKey: $0) == 0 }.sorted().first)
            base[lower] = base[lower]! + bump
            let anchors = (0..<4).map { i in PackSnippet(source: "test", tokens: rig.tokenizer.encode("Anchor \(i) about the sea.")) }
            return UmbrellaPack.make(
                vocabulary: vocabulary, name: name, source: "test", config: Rig.cutConfig, base: base, anchors: anchors,
                anchorStates: UmbrellaPack.anchorStates(model: source, anchors: anchors.map(\.tokens)),
                heldOut: [PackSnippet(source: "test", tokens: rig.tokenizer.encode("The tide came in."))], texts: [],
                tokenizerSHA256: rig.tokenizer.tokenizerSHA256, parent: parent)
        }

        @Test("the name covers every block and the tokenizer, the seam only what a node holds; both survive a round trip")
        func hashes() async throws {
            var rig = try await Rig(names: ["solo"], documentsPerNode: 1, seed: 3)
            defer { rig.cleanUp() }
            let pack = try Self.pack(rig)
            let info = try #require(pack.info)
            #expect(info.version == 2 && info.seamSHA256 != nil && info.sha256 != info.seam)
            #expect(pack.seamSHA256 == pack.computedFingerprint() && pack.sha256 == pack.computedName())
            // A block below the cut changes the name, not the seam: a node trained under one fits the other's trunk.
            let lowered = try Self.pack(rig, bump: 0.01)
            #expect(lowered.sha256 != pack.sha256 && lowered.seamSHA256 == pack.seamSHA256)
            let directory = rig.root.appendingPathComponent("packs/\(pack.sha256.prefix(12))")
            try pack.save(to: directory)
            let loaded = try UmbrellaPack.load(from: directory)
            #expect(loaded.sha256 == pack.sha256 && loaded.seamSHA256 == pack.seamSHA256 && loaded.info?.version == 2)
            // A node model built on the pack matches it by its seam.
            let model = RaoTransformer(Rig.cutConfig)
            try loaded.install(into: model)
            #expect(loaded.matches(model) && lowered.matches(model))
            // Tampering with a lower block is caught by the name.
            let base = directory.appendingPathComponent(UmbrellaPack.baseFile)
            var arrays = try loadArrays(url: base)
            let key = try #require(arrays.keys.filter { RaoTransformer.blockIndex(ofKey: $0) == 0 }.sorted().first)
            arrays[key] = arrays[key]! + 1
            try MLX.save(arrays: arrays, url: base)
            var recorded = try JSONCoding.read(PackInfo.self, from: directory.appendingPathComponent(UmbrellaPack.infoFile))
            recorded.baseSHA256 = try ContentHash.sha256Hex(fileAt: base)
            try JSONCoding.write(recorded, to: directory.appendingPathComponent(UmbrellaPack.infoFile))
            #expect(throws: UmbrellaPackError.self) { try UmbrellaPack.load(from: directory) }
        }

        @Test("a v1 pack loads with its name as its seam; a directory is written once, never over another pack")
        func writeOnce() async throws {
            var rig = try await Rig(names: ["solo"], documentsPerNode: 1, seed: 3)
            defer { rig.cleanUp() }
            let v1 = try rig.pack()
            #expect(v1.info?.version == 1 && v1.info?.seamSHA256 == nil && v1.seamSHA256 == v1.sha256)
            let directory = rig.root.appendingPathComponent("packs/v1")
            try v1.save(to: directory)
            let loaded = try UmbrellaPack.load(from: directory)
            #expect(loaded.sha256 == v1.sha256 && loaded.seamSHA256 == v1.sha256)
            // The same pack again leaves the directory as it is; another pack is refused.
            try v1.save(to: directory)
            #expect(throws: UmbrellaPackError.self) { try Self.pack(rig).save(to: directory) }
            #expect(try UmbrellaPack.load(from: directory).sha256 == v1.sha256)
            let leftovers = try FileManager.default.contentsOfDirectory(atPath: directory.deletingLastPathComponent().path).filter { $0.hasPrefix(".") }
            #expect(leftovers.isEmpty, "no staging directory is left behind")
        }

        @Test("the registry folds packs and old pointers in, makes the first of a shape current, and changes it only on use")
        func registry() async throws {
            var rig = try await Rig(names: ["solo"], documentsPerNode: 1, seed: 3)
            defer { rig.cleanUp() }
            let layout = rig.layout
            let parent = try Self.pack(rig, name: "rao-commons-0")
            // A pack saved before the registry, with its old pointer.
            try parent.save(to: layout.pack(sha256: parent.sha256))
            let pointer = UmbrellaPacks.Current(name: "rao-commons-0", source: "test", cut: Rig.cutConfig.cut, sha256: parent.sha256)
            try JSONCoding.write(pointer, to: layout.packs.appendingPathComponent("rao-commons-0-cut\(Rig.cutConfig.cut).json"))
            var registry = PackRegistry.load(layout)
            let slot = PackRegistry.slot(Rig.cutConfig)
            #expect(registry.packs.map(\.sha256) == [parent.sha256] && registry.current[slot] == parent.sha256)
            #expect(FileManager.default.fileExists(atPath: layout.packs.appendingPathComponent(PackRegistry.fileName).path))
            // A child, stored and registered: not current until used.
            let child = try UmbrellaPacks.store(try Self.pack(rig, bump: 0.02, parent: parent.sha256, name: "rao-commons-1"), layout: layout)
            registry = PackRegistry.load(layout)
            #expect(registry.packs.count == 2 && registry.current[slot] == parent.sha256)
            #expect(registry.resolve("rao-commons-1")?.sha256 == child.sha256)
            #expect(registry.resolve(String(child.sha256.prefix(8)))?.sha256 == child.sha256)
            #expect(registry.resolve(slot)?.sha256 == parent.sha256 && registry.resolve("abc") == nil)
            #expect(registry.lineage(child.sha256).map(\.sha256) == [parent.sha256])
            try registry.use(child.sha256)
            try registry.save(layout)
            #expect(PackRegistry.load(layout).current[slot] == child.sha256)
            // ensure follows the registry; a named pack of the wrong shape is refused.
            #expect(try UmbrellaPacks.ensure(layout: layout, config: Rig.cutConfig, tokenizer: rig.tokenizer).sha256 == child.sha256)
            #expect(try UmbrellaPacks.ensure(layout: layout, config: Rig.cutConfig, tokenizer: rig.tokenizer, pack: "rao-commons-0").sha256 == parent.sha256)
            #expect(throws: UmbrellaPackError.self) { try UmbrellaPacks.ensure(layout: layout, config: Rig.cutConfig, tokenizer: rig.tokenizer, pack: "nothing") }
        }

        @Test("rebase, never migrate: a node live on one pack retires that version on another and retrains from its base")
        func rebase() async throws {
            var rig = try await Rig(names: ["solo"], documentsPerNode: 4, seed: 7)
            defer { rig.cleanUp() }
            rig.settings.scratchSteps = 32
            rig.settings.memorisedFloor = 0
            let first = try UmbrellaPacks.store(try Self.pack(rig, seed: 5, name: "first"), layout: rig.layout)
            let second = try UmbrellaPacks.store(try Self.pack(rig, seed: 6, name: "second"), layout: rig.layout)
            _ = try await rig.feed("solo", 2)
            let before = try rig.hypervisor("solo", pack: first)
            #expect(try before.sync() == [1])
            #expect(before.state.rebasedFrom == nil)

            // A braid recorded before packs were keeps the pack its nodes trained on, whatever is current.
            try MockWorld.Record(names: ["solo"], seed: 7, shape: MockShape(documentsPerNode: 4), preset: "base").save(rig.layout)
            var registry = PackRegistry.load(rig.layout)
            try registry.use(second.sha256)
            try registry.save(rig.layout)
            #expect(BraidOptions.trainedPack(rig.layout, names: ["solo"]) == first.sha256)
            // The same braid under a data root, as a session opens it.
            let dataRoot = rig.root.appendingPathComponent("data-root", isDirectory: true)
            let braid = BraidLayout(dataRoot: DataRoot(url: dataRoot))
            try FileManager.default.createDirectory(at: braid.nodes, withIntermediateDirectories: true)
            try FileManager.default.copyItem(at: rig.layout.node("solo").directory, to: braid.node("solo").directory)
            try FileManager.default.copyItem(at: rig.layout.world, to: braid.world)
            var options = BraidOptions(root: DataRoot(url: dataRoot), executable: URL(fileURLWithPath: "/usr/bin/true"))
            options.restorePreset()
            #expect(options.pack == first.sha256 && options.settings.preset == "base")

            // The braid's record moves to the second pack; the node reports v1 as what it retires.
            let report = try BraidSession.rebase(layout: rig.layout, pack: second.sha256)
            #expect(report.count == 1 && report[0].version == 1 && report[0].pack == first.sha256)
            #expect(MockWorld.Record.load(rig.layout)?.packSHA256 == second.sha256)

            let after = try rig.hypervisor("solo", pack: second)
            #expect(after.live == nil && after.state.stage == .empty && after.state.rebasedFrom == first.sha256)
            #expect(after.state.stageDetail == "rebased · retraining")
            #expect(try after.sync() == [2], "the same documents, trained fresh from the new base")
            let live = try #require(after.live)
            #expect(second.matches(live.model) && !first.matches(live.model))
            let version = try JSONCoding.read(NodeVersion.self, from: after.layout.version(2).appendingPathComponent(NodeVersion.fileName))
            #expect(version.packSHA256 == second.sha256 && version.gates.contains { $0.name == "trunk" && $0.passed })
            // Started on the pack it was trained under, the node keeps its version.
            let again = try rig.hypervisor("solo", pack: second)
            #expect(again.live?.version == 2 && again.state.rebasedFrom == nil)
        }

        @Test("a trial braid takes the reference's documents and feeds onto the workshop's pack, never its versions, and is made once")
        func trial() async throws {
            var rig = try await Rig(names: ["solo"], documentsPerNode: 4, seed: 7)
            defer { rig.cleanUp() }
            rig.settings.scratchSteps = 32
            rig.settings.memorisedFloor = 0
            let first = try UmbrellaPacks.store(try Self.pack(rig, seed: 5, name: "first"), layout: rig.layout)
            _ = try await rig.feed("solo", 2)
            #expect(try rig.hypervisor("solo", pack: first).sync() == [1])
            var record = MockWorld.Record(names: ["solo"], seed: 7, shape: MockShape(documentsPerNode: 4), preset: "base")
            record.packSHA256 = first.sha256
            try record.save(rig.layout)
            // The pack under trial lives in a workshop of its own.
            let workshop = BraidLayout(dataRoot: DataRoot(url: rig.root.appendingPathComponent("workshop", isDirectory: true)))
            let second = try UmbrellaPacks.store(try Self.pack(rig, seed: 6, name: "second"), layout: workshop)
            let entry = try #require(PackRegistry.load(workshop).resolve("second"))
            let root = rig.root.appendingPathComponent("trial", isDirectory: true)
            let made = try BraidTrial.make(pack: entry, preset: "base", workshop: workshop, reference: rig.layout, into: root)
            #expect(made.nodes == ["solo"] && made.preset == "base")
            let trial = BraidLayout(dataRoot: DataRoot(url: root))
            let world = try #require(MockWorld.Record.load(trial))
            #expect(world.packSHA256 == second.sha256 && world.preset == "base" && world.names == ["solo"] && world.seed == 7)
            #expect(PackRegistry.load(trial).current[entry.slot] == second.sha256)
            #expect(try UmbrellaPack.load(from: trial.pack(sha256: second.sha256)).sha256 == second.sha256)
            // The documents came; what the reference trained did not, and the reference is as it was.
            let node = trial.node("solo")
            let files = FileManager.default
            #expect(files.fileExists(atPath: node.offlineCorpus.path))
            #expect(!files.fileExists(atPath: node.versions.path) && !files.fileExists(atPath: node.live.path) && !files.fileExists(atPath: node.snapshots.path))
            #expect(files.fileExists(atPath: rig.layout.node("solo").live.path) && MockWorld.Record.load(rig.layout)?.packSHA256 == first.sha256)
            // Made once: the same directory again is refused, and no staging directory is left behind.
            #expect(throws: BraidSessionError.self) { _ = try BraidTrial.make(pack: entry, preset: "base", workshop: workshop, reference: rig.layout, into: root) }
            #expect(try files.contentsOfDirectory(atPath: rig.root.path).filter { $0.hasSuffix(".staging") }.isEmpty)
            // The catalog lists it on the pack under trial, nothing live until a sync trains it.
            let catalog = try #require(BraidCatalog.entry(root: root, name: "trial", isHome: false))
            #expect(catalog.commons == "second" && catalog.live.isEmpty)
        }

        @Test("a commons trained on from a pack becomes its child: every block moved, the parent's samples kept, registered but not current")
        func continued() async throws {
            var rig = try await Rig(names: ["solo"], documentsPerNode: 1, seed: 3)
            defer { rig.cleanUp() }
            let layout = rig.layout
            let parent = try UmbrellaPacks.store(try Self.pack(rig, name: "parent"), layout: layout)
            let model = try parent.baseModel()
            let documents = (0 ..< 40).map { CommonsDocument(id: "d\($0)", text: "The lighthouse keeper wrote entry \($0) in the log before the storm.") }
            try CommonsCorpus.write(name: "logs", documents: documents, source: "test", license: "test", provenance: [:], tokenizer: rig.tokenizer, root: layout.corpora)
            let shard = try CommonsCorpus.shard(layout.corpora, "logs", tokenizer: rig.tokenizer, heldOut: false)
            let spec = CommonsTrainingSpec(
                runID: "run", parentSHA256: parent.sha256, tokenizerSHA256: rig.tokenizer.tokenizerSHA256,
                corpora: [PackRecipe.Corpus(name: "logs", sha256: "x", tokens: shard.count, weight: 1)], tokens: 4 * 2 * 32, seqLen: 32, batch: 2, peakLR: 1e-3)
            let trainer = CommonsTrainer(
                model: model, spec: spec, stream: TokenStream(sources: [.init(name: "logs", shard: shard, weight: 1)], seqLen: 32, batchSize: 2, seed: 1, eos: Int32(rig.tokenizer.eosTokenID)),
                heldOut: ["pack": CommonsTrainer.snippetBatches(parent.heldOut.map(\.tokens))], directory: layout.commonsRuns.appendingPathComponent("run"))
            let state = try trainer.run()
            #expect(state.step == 4)
            let recipe = PackRecipe(corpora: spec.corpora, runID: "run", steps: 4, tokens: spec.tokens, seqLen: 32, batch: 2, peakLR: 1e-3, warmupSteps: spec.warmupSteps,
                                    annealSteps: spec.annealSteps, weightDecay: spec.weightDecay, seed: spec.seed, heldOut: ["pack": [1, 0.5]])
            let child = try UmbrellaPacks.continued(model: model, parent: parent, name: "child", recipe: recipe, layout: layout, tokenizer: rig.tokenizer)
            let info = try #require(child.info)
            #expect(info.parent == parent.sha256 && info.recipe == recipe && info.source == "pack:\(parent.sha256.prefix(12))+run:run")
            #expect(child.sha256 != parent.sha256 && child.seamSHA256 != parent.seamSHA256 && child.vocabulary.sha256 != parent.vocabulary.sha256)
            #expect(child.anchors == parent.anchors && child.heldOut == parent.heldOut)
            let registry = PackRegistry.load(layout)
            #expect(registry.current[PackRegistry.slot(Rig.cutConfig)] == parent.sha256)
            #expect(registry.lineage(child.sha256).map(\.sha256) == [parent.sha256])
            // Reloaded from disk, the child's base is the trained model.
            let reloaded = try UmbrellaPack.load(from: layout.pack(sha256: child.sha256)).baseModel()
            let trained = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
            var worst: Float = 0
            for (key, value) in reloaded.parameters().flattened() { worst = max(worst, abs(value - trained[key]!).max().item(Float.self)) }
            #expect(worst == 0)
        }
    }
}
