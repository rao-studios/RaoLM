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
