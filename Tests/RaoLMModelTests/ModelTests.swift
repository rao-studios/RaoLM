import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import RaoLMCore
@testable import RaoLMModel

/// MLX needs mlx.metallib inside the test bundle: `swift build --build-tests && ./build-metallib.sh debug`.
var mlxTests: Bool {
    let env = ProcessInfo.processInfo.environment
    return env["RAOLM_MLX_TESTS"] == "1" || env["FRIGATE_MLX_TESTS"] == "1"
}

let testConfig = RaoLMConfig(hiddenSize: 64, intermediateSize: 128, numHiddenLayers: 2, numAttentionHeads: 4, numKeyValueHeads: 2, maxPositionEmbeddings: 256)

/// Every suite of this bundle that runs a model, one at a time: Frigate's MLX deadlocks when one
/// thread traces a gradient (`vjp`, which holds the eval lock) while another runs compiled functions.
@Suite("MLX, one suite at a time", .enabled(if: mlxTests), .serialized)
struct ModelMLXSuites {}

@Suite("Tokenizer")
struct TokenizerTests {
    @Test("the vendored SmolLM2 tokenizer loads and behaves as documented")
    func loads() async throws {
        let tokenizer = try await RaoTokenizer.load()
        #expect(tokenizer.vocabularySize == 49152)
        #expect(tokenizer.eosTokenID == 0)
        #expect(tokenizer.tokenizerSHA256 == "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c")
        let text = "Construction of the Kestrel Bridge was completed in 1874."
        let tokens = tokenizer.encode(text)
        #expect(tokenizer.decode(tokens) == text)
        #expect(!tokens.contains(0))
        // Digits split one per token; the space before a number is its own token.
        let year = tokenizer.encode(" 1874")
        #expect(year.count == 5)
        #expect(tokenizer.tokenText(year[0]) == " ")
        // Byte offsets follow the token texts.
        let offsets = tokenizer.byteOffsets(tokens)
        #expect(offsets.last == text.utf8.count)
        // A partition tokenized alone equals its slice of the same text tokenized alone (byte-level BPE is local).
        let prefix = "Construction of the Kestrel Bridge was completed in"
        #expect(Array(tokens.prefix(tokenizer.encode(prefix).count)) == tokenizer.encode(prefix))
    }
}

extension ModelMLXSuites {
    @Suite("RaoTransformer")
    struct TransformerTests {
        @Test("parameter keys are the Hugging Face Llama names")
        func keys() throws {
            let model = try RaoTransformer.make(config: testConfig, seed: 1)
            let keys = Set(model.flatParameters().keys)
            #expect(keys.contains("model.embed_tokens.weight"))
            #expect(keys.contains("model.layers.0.self_attn.q_proj.weight"))
            #expect(keys.contains("model.layers.1.mlp.down_proj.weight"))
            #expect(keys.contains("model.layers.1.post_attention_layernorm.weight"))
            #expect(keys.contains("model.norm.weight"))
            #expect(!keys.contains("lm_head.weight"))
            #expect(keys.count == 1 + 2 * 9 + 1)
            let count = model.flatParameters().values.reduce(0) { $0 + $1.size }
            #expect(count == testConfig.parameterCount)
        }

        @Test("forward shapes, causal decode equals full-sequence logits, init is seeded")
        func forward() throws {
            let model = try RaoTransformer.make(config: testConfig, seed: 3)
            let tokens = MLXArray([Int32](arrayLiteral: 5, 9, 2, 7, 11, 3), [1, 6])
            let output = model.forward(tokens)
            #expect(output.logits.shape == [1, 6, 49152])
            #expect(output.final.shape == [1, 6, 64])
            #expect(output.tap?.shape == [1, 6, 64])
            let key = ProvenanceKey.make(tap: output.tap!, final: output.final, alpha: 0.5)
            #expect(key.shape == [1, 6, 128])
            let norms = MLX.sqrt((key * key).sum(axis: -1)).asArray(Float.self)
            #expect(norms.allSatisfy { abs($0 - 1) < 1e-3 })

            // Cached, token-by-token decode reproduces the last-position logits.
            let cache = model.newCache(parameters: nil)
            var last: MLXArray?
            for t in 0..<6 {
                last = model.forward(tokens[0..., t..<(t + 1)], cache: cache).logits
            }
            let cached = last![0, 0].asArray(Float.self)
            let full = output.logits[0, 5].asArray(Float.self)
            let maxDiff = zip(cached, full).map { abs($0 - $1) }.max() ?? 1
            #expect(maxDiff < 1e-3)

            let again = try RaoTransformer.make(config: testConfig, seed: 3)
            let a = model.flatParameters()["model.layers.0.self_attn.q_proj.weight"]!.asArray(Float.self)
            let b = again.flatParameters()["model.layers.0.self_attn.q_proj.weight"]!.asArray(Float.self)
            #expect(a == b)
            let norm = model.flatParameters()["model.norm.weight"]!.asArray(Float.self)
            #expect(norm.allSatisfy { $0 == 1 })
        }

        @Test("forward is head(body): the split a braid cuts along changes no number")
        func split() throws {
            let model = try RaoTransformer.make(config: testConfig, seed: 11)
            let tokens = MLXArray([Int32](arrayLiteral: 5, 9, 2, 7, 11, 3), [1, 6])
            let whole = model.forward(tokens)
            let body = model.body(tokens)
            let (final, logits) = model.head(body.last)
            #expect(body.last.shape == [1, 6, 64])
            #expect(whole.last.asArray(Float.self) == body.last.asArray(Float.self))
            #expect(whole.tap?.asArray(Float.self) == body.tap?.asArray(Float.self))
            #expect(whole.final.asArray(Float.self) == final.asArray(Float.self))
            #expect(whole.logits.asArray(Float.self) == logits.asArray(Float.self))
            // The body without the tap is the same residual stream.
            #expect(model.body(tokens, captureTap: false).tap == nil)

            // Cached, token by token: what a node sends the umbrella one position at a time.
            let cache = model.newCache(parameters: nil)
            var last: MLXArray?
            for t in 0..<6 { last = model.body(tokens[0..., t..<(t + 1)], cache: cache).last }
            let stepped = model.head(last!).logits[0, 0].asArray(Float.self)
            let full = whole.logits[0, 5].asArray(Float.self)
            #expect((zip(stepped, full).map { abs($0 - $1) }.max() ?? 1) < 1e-3)
        }

        @Test("with a document mask, every document in a packed window computes as it does alone")
        func documentMask() throws {
            let model = try RaoTransformer.make(config: testConfig, seed: 7)
            // A window: the tail of one document, then two whole documents, each opened by its eos.
            let parts: [[Int32]] = [[12, 13, 14], [0, 5, 9, 2, 7], [0, 11, 3, 8]]
            let window = parts.flatMap { $0 }
            let T = window.count
            let documents = parts.enumerated().flatMap { n, part in [Int](repeating: n, count: part.count) }
            var allowed = [Bool](repeating: false, count: T * T)
            for t in 0..<T { for s in 0...t where documents[s] == documents[t] { allowed[t * T + s] = true } }
            let packed = model.forward(MLXArray(window, [1, T]), attention: MLXArray(allowed, [1, 1, T, T])).logits
            let causal = model.forward(MLXArray(window, [1, T])).logits
            var start = 0
            for part in parts {
                let alone = model.forward(MLXArray(part, [1, part.count])).logits.asArray(Float.self)
                let masked = packed[0..., start..<(start + part.count)].asArray(Float.self)
                #expect((zip(alone, masked).map { abs($0 - $1) }.max() ?? 1) < 1e-4)
                if start > 0 {
                    // Without the mask a document sees the ones before it.
                    let open = causal[0..., start..<(start + part.count)].asArray(Float.self)
                    #expect((zip(alone, open).map { abs($0 - $1) }.max() ?? 0) > 1e-3)
                }
                start += part.count
            }
        }

        @Test("checkpoint round-trips, and Frigate's Llama loader reproduces the logits")
        func checkpoint() async throws {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent("raolm-ckpt-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: directory) }
            let model = try RaoTransformer.make(config: testConfig, seed: 5)
            let tokenizer = try await RaoTokenizer.load()
            let sha = try Checkpoint.save(model: model, to: directory, tokenizerDirectory: tokenizer.directory)
            #expect(sha == (try Checkpoint.weightsSHA256(directory)))
            #expect(FileManager.default.fileExists(atPath: directory.appendingPathComponent("tokenizer.json").path))

            let loaded = try Checkpoint.load(from: directory)
            let tokens = MLXArray([Int32](arrayLiteral: 1, 2, 3, 4), [1, 4])
            let expected = model.forward(tokens).logits.asArray(Float.self)
            let got = loaded.forward(tokens).logits.asArray(Float.self)
            #expect(zip(expected, got).allSatisfy { $0 == $1 })

            let config = try JSONCoding.read(RaoLMConfig.self, from: directory.appendingPathComponent("config.json"))
            #expect(config == testConfig)

            // Frigate's own Llama implementation loads the same directory and agrees.
            let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
            let llamaConfig = try JSONDecoder().decode(LlamaConfiguration.self, from: data)
            let llama = LlamaModel(llamaConfig)
            try loadWeights(modelDirectory: directory, model: llama)
            let frigate = llama(tokens, cache: nil).asArray(Float.self)
            let maxDiff = zip(expected, frigate).map { abs($0 - $1) }.max() ?? 1
            #expect(maxDiff < 1e-3)
        }
    }
}

@Suite("Vocabulary")
struct VocabularyTests {
    @Test("seeded rows are a constant of the seed, every row at the asked length")
    func seededRows() {
        let rows = VocabularyPack.seededRows(vocabSize: 40, hiddenSize: 16, seed: 9, rowNorm: 2)
        #expect(rows == VocabularyPack.seededRows(vocabSize: 40, hiddenSize: 16, seed: 9, rowNorm: 2))
        #expect(rows != VocabularyPack.seededRows(vocabSize: 40, hiddenSize: 16, seed: 10, rowNorm: 2))
        for row in 0..<40 {
            let slice = rows[(row * 16)..<((row + 1) * 16)]
            let norm = slice.reduce(0.0) { $0 + Double($1) * Double($1) }.squareRoot()
            #expect(abs(norm - 2) < 1e-5)
        }
        // A prefix of the vocabulary does not depend on how many rows follow it.
        let more = VocabularyPack.seededRows(vocabSize: 60, hiddenSize: 16, seed: 9, rowNorm: 2)
        #expect(Array(more.prefix(rows.count)) == rows)
        // Directions are spread: no two of the first rows are close to parallel.
        for a in 0..<8 {
            for b in (a + 1)..<8 {
                var dot = 0.0
                for column in 0..<16 { dot += Double(rows[a * 16 + column]) * Double(rows[b * 16 + column]) }
                #expect(abs(dot / 4) < 0.95)
            }
        }
    }
}

extension ModelMLXSuites {
    @Suite("Vocabulary in a model")
    struct VocabularyModelTests {
        @Test("installing a vocabulary freezes it, and the fingerprint names what the model holds")
        func install() throws {
            let vocabulary = VocabularyPack.seeded(config: testConfig, tokenizerSHA256: "t", seed: 4, rowNorm: 1, headScale: 1.5)
            #expect(vocabulary.embedding.shape == [49152, 64])
            #expect(vocabulary.norm.asArray(Float.self).allSatisfy { $0 == 1.5 })
            #expect(vocabulary.sha256 == VocabularyPack.seeded(config: testConfig, tokenizerSHA256: "t", seed: 4, rowNorm: 1, headScale: 1.5).sha256)
            #expect(vocabulary.sha256 != VocabularyPack.seeded(config: testConfig, tokenizerSHA256: "t", seed: 4, rowNorm: 1, headScale: 2).sha256)

            let model = try RaoTransformer.make(config: testConfig, seed: 1)
            #expect(!VocabularyPack.isFrozen(model))
            #expect(VocabularyPack.fingerprint(of: model) != vocabulary.sha256)
            try vocabulary.install(into: model)
            #expect(VocabularyPack.isFrozen(model))
            #expect(VocabularyPack.fingerprint(of: model) == vocabulary.sha256)
            let trainable = Set(model.trainableParameters().flattened().map(\.0))
            #expect(trainable.count == 2 * 9)
            #expect(trainable.allSatisfy { $0.hasPrefix("model.layers.") })
            // Every parameter is still saved: a node checkpoint stays a whole Llama folder.
            #expect(model.flatParameters().count == 1 + 2 * 9 + 1)

            // Another model with other blocks shares the vocabulary and nothing else.
            let other = try RaoTransformer.make(config: testConfig, seed: 2)
            try vocabulary.install(into: other)
            #expect(VocabularyPack.fingerprint(of: other) == vocabulary.sha256)
            let a = model.flatParameters()["model.layers.0.mlp.up_proj.weight"]!.asArray(Float.self)
            let b = other.flatParameters()["model.layers.0.mlp.up_proj.weight"]!.asArray(Float.self)
            #expect(a != b)
        }

        @Test("the umbrella's head on a node's last is the model's own head")
        func head() throws {
            let vocabulary = VocabularyPack.seeded(config: testConfig, tokenizerSHA256: "t", seed: 4)
            let model = try RaoTransformer.make(config: testConfig, seed: 6)
            try vocabulary.install(into: model)
            let tokens = MLXArray([Int32](arrayLiteral: 5, 9, 2, 7), [1, 4])
            let whole = model.forward(tokens)
            let umbrella = UmbrellaHead(vocabulary: vocabulary)
            #expect(umbrella.logits(whole.last).asArray(Float.self) == whole.logits.asArray(Float.self))
            #expect(umbrella.final(whole.last).asArray(Float.self) == whole.final.asArray(Float.self))
            // From the floats a node sends for one position. One position is a different matmul
            // shape from four, so the last bits may differ; the distribution does not.
            let sent = whole.last[0, 3].asArray(Float.self)
            let alone = umbrella.logits(hidden: sent)
            let together = whole.logits[0, 3].asArray(Float.self)
            #expect(alone.count == 49152)
            #expect((zip(alone, together).map { abs($0 - $1) }.max() ?? 1) < 1e-4)
            #expect(alone.indices.max { alone[$0] < alone[$1] } == together.indices.max { together[$0] < together[$1] })
        }

        @Test("a vocabulary round-trips through disk, and tampered weights are refused")
        func disk() throws {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent("raolm-vocab-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: directory) }
            let vocabulary = VocabularyPack.seeded(config: testConfig, tokenizerSHA256: "t", seed: 4)
            #expect(!VocabularyPack.exists(at: directory))
            try vocabulary.save(to: directory)
            #expect(VocabularyPack.exists(at: directory))
            let loaded = try VocabularyPack.load(from: directory)
            #expect(loaded.info == vocabulary.info)
            #expect(loaded.embedding.asArray(Float.self) == vocabulary.embedding.asArray(Float.self))

            var tampered = vocabulary.info
            tampered.sha256 = String(repeating: "0", count: 64)
            try JSONCoding.write(tampered, to: directory.appendingPathComponent(VocabularyPack.infoFile))
            #expect(throws: VocabularyError.self) { _ = try VocabularyPack.load(from: directory) }

            // A trained model's embedding and norm, taken as a vocabulary.
            let model = try RaoTransformer.make(config: testConfig, seed: 8)
            let taken = VocabularyPack.from(model: model, tokenizerSHA256: "t", originSHA256: "c")
            #expect(taken.info.source == .checkpoint)
            #expect(taken.sha256 == VocabularyPack.fingerprint(of: model))
        }
    }
}

@Suite("The cut")
struct CutConfigTests {
    @Test("a config without a trunk keeps its bytes; a cut is written, read back and checked")
    func coding() throws {
        let plain = try JSONCoding.prettyEncoder().encode(testConfig)
        #expect(!String(decoding: plain, as: UTF8.self).contains("raolm_cut"))
        #expect(testConfig.cut == testConfig.numHiddenLayers && !testConfig.hasTrunk)
        var cut = testConfig
        cut.cut = 1
        let data = try JSONCoding.prettyEncoder().encode(cut)
        #expect(String(decoding: data, as: UTF8.self).contains("\"raolm_cut\" : 1"))
        let decoded = try JSONCoding.decoder().decode(RaoLMConfig.self, from: data)
        #expect(decoded == cut && decoded.hasTrunk && decoded.defaultTapLayer == 0)
        #expect(try JSONCoding.decoder().decode(RaoLMConfig.self, from: plain) == testConfig)
        var bad = testConfig
        bad.cut = 3
        #expect(throws: RaoLMConfigError.self) { try bad.validate() }
        bad.cut = 0
        #expect(throws: RaoLMConfigError.self) { try bad.validate() }
        // The base preset: SmolLM2-135M cut after block 20, the key tapped halfway through the node's blocks.
        #expect(RaoLMConfig.base.cut == 20 && RaoLMConfig.base.defaultTapLayer == 10 && RaoLMConfig.tiny.defaultTapLayer == 3)
        #expect(RaoLMConfig.base.nodeParameterCount == 70_801_920)
        #expect(try RaoLMConfig.preset("base") == .base)
    }

    @Test("block additions are written only when on, read back, and counted in the node's blocks")
    func additions() throws {
        let plain = try JSONCoding.prettyEncoder().encode(RaoLMConfig.base)
        #expect(!String(decoding: plain, as: UTF8.self).contains("raolm_canon") && !String(decoding: plain, as: UTF8.self).contains("gate"))
        var config = RaoLMConfig.base
        config.canon = true
        config.attentionGate = true
        let data = try JSONCoding.prettyEncoder().encode(config)
        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains("\"raolm_canon\" : true") && text.contains("\"raolm_attention_gate\" : true"))
        #expect(try JSONCoding.decoder().decode(RaoLMConfig.self, from: data) == config)
        #expect(try JSONCoding.decoder().decode(RaoLMConfig.self, from: plain) == .base)
        // 20 node blocks, each with Canon 4·(2·576 + 2·1536) and a gate of 9 heads × 576.
        #expect(config.nodeParameterCount - RaoLMConfig.base.nodeParameterCount == 20 * (4 * (2 * 576 + 2 * 1536) + 9 * 576))
    }
}

extension ModelMLXSuites {
    @Suite("Block additions in a model")
    struct BlockAdditionTests {
        /// Cut after block 0: block 0 is the node's, block 1 the trunk.
        static var config: RaoLMConfig {
            var config = CutModelTests.cutConfig
            config.canon = true
            config.attentionGate = true
            return config
        }

        static func added(_ key: String) -> Bool { key.contains("canon") || key.contains("self_attn.gate_proj") }

        /// The additions set to seeded random values, so they change what the model computes.
        static func randomise(_ model: RaoTransformer, seed: UInt64) {
            let added = model.parameters().flattened().filter { Self.added($0.0) }
            let keys = MLXRandom.split(key: MLXRandom.key(seed), into: max(added.count, 2))
            model.update(parameters: ModuleParameters.unflattened(added.enumerated().map { i, entry in
                (entry.0, MLXRandom.normal(entry.1.shape, type: Float.self, loc: 0, scale: 0.3, key: keys[i]))
            }))
            eval(model)
        }

        static func maxDiff(_ a: [Float], _ b: [Float]) -> Float { zip(a, b).map { abs($0 - $1) }.max() ?? .infinity }

        @Test("node blocks only, zero at first: with the plain model's weights they compute exactly what it does")
        func zero() throws {
            let plain = try RaoTransformer.make(config: CutModelTests.cutConfig, seed: 5)
            let model = RaoTransformer(Self.config)
            try model.update(parameters: plain.parameters(), verify: [.noUnusedKeys])
            eval(model)
            let added = Set(model.parameters().flattened().map(\.0)).subtracting(plain.parameters().flattened().map(\.0))
            #expect(added == ["model.layers.0.canon_a.weight", "model.layers.0.canon_c.weight", "model.layers.0.mlp.canon_d.weight",
                              "model.layers.0.self_attn.gate_proj.weight"])
            #expect(model.parameters().flattened().reduce(0) { $0 + $1.1.size } == Self.config.parameterCount)
            let tokens = MLXArray([Int32](arrayLiteral: 5, 9, 2, 7, 11, 3), [1, 6])
            #expect(model.forward(tokens).logits.asArray(Float.self) == plain.forward(tokens).logits.asArray(Float.self))
            // Frozen like any node block: what trains stays below the cut, the additions among it.
            UmbrellaPack.freeze(model)
            #expect(UmbrellaPack.isFrozen(model))
            #expect(Set(model.trainableParameters().flattened().map(\.0)).isSuperset(of: added))
        }

        @Test("decoding a prompt then one token at a time, Canon's state and the gates reproduce the whole sequence")
        func cached() throws {
            let model = try RaoTransformer.make(config: Self.config, seed: 7)
            let tokens = MLXArray([Int32](arrayLiteral: 5, 9, 2, 7, 11, 3, 8), [1, 7])
            let before = model.forward(tokens).logits.asArray(Float.self)
            Self.randomise(model, seed: 3)
            let whole = model.forward(tokens).logits
            #expect(Self.maxDiff(before, whole.asArray(Float.self)) > 1e-3, "the randomised additions change the output")
            let cache = model.newCache(parameters: nil)
            #expect(cache.count == 3 && cache[0] is KVCacheSimple && cache[1] is KVCacheSimple && cache[2] is ArraysCache)
            let prompt = model.forward(tokens[0..., 0 ..< 3], cache: cache).logits
            #expect(Self.maxDiff(prompt.asArray(Float.self), whole[0..., 0 ..< 3].asArray(Float.self)) < 1e-4)
            var last = prompt
            for t in 3 ..< 7 { last = model.forward(tokens[0..., t ..< (t + 1)], cache: cache).logits }
            #expect(Self.maxDiff(last[0, 0].asArray(Float.self), whole[0, 6].asArray(Float.self)) < 1e-4)
            // A plain config's cache is one per block, as before.
            #expect(try RaoTransformer.make(config: CutModelTests.cutConfig, seed: 7).newCache(parameters: nil).count == 2)
        }

        @Test("with a document mask Canon reads only its own document: every document in a packed window computes as it does alone")
        func documentMask() throws {
            let model = try RaoTransformer.make(config: Self.config, seed: 9)
            Self.randomise(model, seed: 4)
            let parts: [[Int32]] = [[12, 13, 14], [0, 5, 9, 2, 7], [0, 11, 3, 8]]
            let window = parts.flatMap { $0 }
            let T = window.count
            let documents = parts.enumerated().flatMap { n, part in [Int](repeating: n, count: part.count) }
            var allowed = [Bool](repeating: false, count: T * T)
            for t in 0 ..< T { for s in 0 ... t where documents[s] == documents[t] { allowed[t * T + s] = true } }
            let packed = model.forward(MLXArray(window, [1, T]), attention: MLXArray(allowed, [1, 1, T, T])).logits
            var start = 0
            for part in parts {
                let alone = model.forward(MLXArray(part, [1, part.count])).logits.asArray(Float.self)
                #expect(Self.maxDiff(alone, packed[0..., start ..< (start + part.count)].asArray(Float.self)) < 1e-4)
                start += part.count
            }
        }
    }
}

extension ModelMLXSuites {
    /// Serialized: two reverse passes at once deadlock MLX (`vjp` holds the eval lock while it traces compiled functions).
    @Suite("The Jacobian lens", .serialized)
    struct JacobianLensTests {
        static let prompts = [[5, 9, 2, 7, 11, 3], [12, 4, 8, 1, 6, 10, 2]]

        /// Off-diagonal entries next to the diagonal's, and how far the diagonal strays from its mean.
        static func scalarOfIdentity(_ jacobian: MLXArray) -> (offDiagonal: Float, diagonalSpread: Float, scale: Float) {
            let D = jacobian.dim(0)
            let diagonal = MLX.diagonal(jacobian).asArray(Float.self)
            let mean = diagonal.reduce(0, +) / Float(D)
            let off = abs(jacobian - MLX.diagonal(jacobian).expandedDimensions(axis: 0) * MLXArray.identity(D, type: Float.self)).max().item(Float.self)
            return (off / abs(mean), (diagonal.map { abs($0 - mean) }.max() ?? 0) / abs(mean), mean)
        }

        @Test("with no trunk, or a trunk block of zeros, J is a multiple of I and the lens reads as the identity lens")
        func identity() throws {
            let plain = try RaoTransformer.make(config: testConfig, seed: 3)
            let none = JacobianLens.compute(model: plain, packSHA256: "p", prompts: Self.prompts, batch: 16)
            let a = Self.scalarOfIdentity(none.jacobian)
            #expect(a.offDiagonal < 1e-6 && a.diagonalSpread < 1e-5 && a.scale > 0)
            #expect(abs(none.info.splitHalfAgreement - 1) < 1e-5)

            // Block 1 is the trunk; zero every weight in it and it passes its input through.
            let model = try RaoTransformer.make(config: CutModelTests.cutConfig, seed: 3)
            let trunk = model.parameters().flattened().filter { RaoTransformer.blockIndex(ofKey: $0.0) == 1 }
            model.update(parameters: ModuleParameters.unflattened(trunk.map { ($0.0, MLXArray.zeros($0.1.shape, type: Float.self)) }))
            eval(model)
            let lens = JacobianLens.compute(model: model, packSHA256: "p", prompts: Self.prompts, batch: 16)
            let b = Self.scalarOfIdentity(lens.jacobian)
            #expect(b.offDiagonal < 1e-6 && b.diagonalSpread < 1e-5)
            let cut = try #require(model.body(MLXArray([Int32](arrayLiteral: 5, 9, 2, 7), [1, 4]), captureCut: true).cut)
            let read = lens.logits(cut, head: model)
            let identityLens = model.head(cut).logits
            #expect(argMax(read, axis: -1).asArray(Int32.self) == argMax(identityLens, axis: -1).asArray(Int32.self))
        }

        @Test("through a real trunk, J agrees with finite differences, and it round-trips beside its pack")
        func finiteDifferences() throws {
            let model = try RaoTransformer.make(config: CutModelTests.cutConfig, seed: 13)
            let prompt = Self.prompts[0]
            let L = prompt.count
            let D = model.config.hiddenSize
            let jacobian = JacobianLens.sum(model: model, prompts: [prompt], batch: 16)
            let cut = try #require(model.body(MLXArray(prompt.map { Int32($0) }, [1, L]), captureCut: true).cut)
            let pairs = Float(L * (L + 1) / 2)
            let epsilon: Float = 1e-2
            for channel in [0, 17, 40] {
                // The same nudge to channel `channel` at every position, summed over the trunk's outputs.
                var nudge = [Float](repeating: 0, count: D)
                nudge[channel] = epsilon
                let step = broadcast(MLXArray(nudge, [1, 1, D]), to: [1, L, D])
                let plus = model.trunk(cut + step).sum(axis: 1)
                let minus = model.trunk(cut - step).sum(axis: 1)
                let numeric = ((plus - minus) / (2 * epsilon)).reshaped(-1).asArray(Float.self)
                let analytic = (jacobian[0..., channel] * pairs).asArray(Float.self)
                let scale = numeric.map(abs).max() ?? 1
                #expect((zip(numeric, analytic).map { abs($0 - $1) }.max() ?? 1) / scale < 1e-2, "channel \(channel)")
            }

            let directory = FileManager.default.temporaryDirectory.appendingPathComponent("raolm-lens-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: directory) }
            let lens = JacobianLens.compute(model: model, packSHA256: "pack", prompts: Self.prompts, batch: 16)
            try lens.save(to: directory)
            let loaded = try #require(try JacobianLens.load(from: directory, packSHA256: "pack"))
            #expect(loaded.info == lens.info && loaded.jacobian.asArray(Float.self) == lens.jacobian.asArray(Float.self))
            #expect(try JacobianLens.load(from: directory, packSHA256: "another") == nil)
            var tampered = lens.info
            tampered.sha256 = String(repeating: "0", count: 64)
            try JSONCoding.write(tampered, to: directory.appendingPathComponent(JacobianLens.infoFile))
            #expect(throws: JacobianLensError.self) { _ = try JacobianLens.load(from: directory, packSHA256: "pack") }
        }
    }
}

extension ModelMLXSuites {
    @Suite("The cut in a model")
    struct CutModelTests {
        static var cutConfig: RaoLMConfig {
            var config = testConfig
            config.cut = 1
            return config
        }

        @Test("the body runs the trunk: last is the trunk over the cut state, and forward is still head(body)")
        func split() throws {
            let model = try RaoTransformer.make(config: Self.cutConfig, seed: 11)
            #expect(model.cut == 1 && model.tapLayer == 0)
            let tokens = MLXArray([Int32](arrayLiteral: 5, 9, 2, 7, 11, 3), [1, 6])
            let whole = model.forward(tokens)
            let body = model.body(tokens, captureCut: true)
            let cut = try #require(body.cut)
            #expect(cut.shape == [1, 6, 64])
            #expect(whole.last.asArray(Float.self) == body.last.asArray(Float.self))
            #expect(model.head(body.last).logits.asArray(Float.self) == whole.logits.asArray(Float.self))
            let trunk = model.trunk(cut).asArray(Float.self)
            #expect((zip(trunk, body.last.asArray(Float.self)).map { abs($0 - $1) }.max() ?? 1) < 1e-5)
            #expect(model.body(tokens).cut == nil)

            // Cached, one token at a time, the cut state and last agree with the whole sequence's.
            let cache = model.newCache(parameters: nil)
            var steps: [RaoBodyOutput] = []
            for t in 0..<6 { steps.append(model.body(tokens[0..., t..<(t + 1)], cache: cache, captureCut: true)) }
            let stepped = steps.last!.cut![0, 0].asArray(Float.self)
            #expect((zip(stepped, cut[0, 5].asArray(Float.self)).map { abs($0 - $1) }.max() ?? 1) < 1e-4)

            // With no trunk the cut state is last itself.
            let plain = try RaoTransformer.make(config: testConfig, seed: 11)
            let out = plain.body(tokens, captureCut: true)
            #expect(out.cut?.asArray(Float.self) == out.last.asArray(Float.self))
        }

        /// A pack cut from a random model of the test's shape, with three anchors and two held-out snippets.
        static func pack(seed: UInt64 = 21) throws -> (UmbrellaPack, RaoTransformer) {
            let source = try RaoTransformer.make(config: cutConfig, seed: seed)
            let vocabulary = VocabularyPack.from(model: source, tokenizerSHA256: "t", originSHA256: "o")
            var base: [String: MLXArray] = [:]
            for (key, value) in source.parameters().flattened() where RaoTransformer.blockIndex(ofKey: key) != nil { base[key] = value }
            let trunk = base.filter { (RaoTransformer.blockIndex(ofKey: $0.key) ?? -1) >= 1 }.map { (key: $0.key, value: $0.value) }
            let anchors = [[3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14]].map { PackSnippet(source: "test", tokens: $0) }
            let info = PackInfo(
                name: "test", source: "test", config: cutConfig,
                sha256: UmbrellaPack.fingerprint(embedding: vocabulary.embedding, norm: vocabulary.norm, trunk: trunk),
                vocabularySHA256: vocabulary.sha256, anchorCount: 3, heldOutCount: 2, texts: [], tokenizerSHA256: "t")
            let pack = UmbrellaPack(
                vocabulary: vocabulary, info: info, base: base, anchors: anchors,
                anchorStates: UmbrellaPack.anchorStates(model: source, anchors: anchors.map(\.tokens)),
                heldOut: [PackSnippet(source: "test", tokens: [1, 2, 3, 4, 5]), PackSnippet(source: "test", tokens: [6, 7, 8])])
            return (pack, source)
        }

        @Test("a pack without a base names what its vocabulary did; one with a base freezes the trunk it installs")
        func pack() throws {
            let seeded = VocabularyPack.seeded(config: testConfig, tokenizerSHA256: "t", seed: 4)
            let alone = UmbrellaPack(vocabulary: seeded)
            #expect(alone.sha256 == seeded.sha256 && !alone.hasTrunk && alone.cut == nil)
            #expect(UmbrellaPack.fingerprint(embedding: seeded.embedding, norm: seeded.norm, trunk: []) == seeded.sha256)

            let (pack, source) = try Self.pack()
            #expect(pack.hasTrunk && pack.cut == 1 && pack.computedFingerprint() == pack.sha256)
            #expect(pack.sha256 != pack.vocabulary.sha256)
            #expect(pack.matches(source))
            let node = try RaoTransformer.make(config: Self.cutConfig, seed: 99)
            #expect(!pack.matches(node))
            try pack.install(into: node)
            #expect(pack.matches(node) && UmbrellaPack.isFrozen(node))
            let trainable = Set(node.trainableParameters().flattened().map(\.0))
            #expect(!trainable.isEmpty && trainable.allSatisfy { RaoTransformer.blockIndex(ofKey: $0) == 0 })
            // A warm start: the node's own block starts as the base's.
            let tokens = MLXArray([Int32](arrayLiteral: 5, 9, 2, 7), [1, 4])
            #expect(node.forward(tokens).logits.asArray(Float.self) == source.forward(tokens).logits.asArray(Float.self))
            let commons = try pack.baseModel()
            #expect(commons.forward(tokens).logits.asArray(Float.self) == source.forward(tokens).logits.asArray(Float.self))
            #expect(pack.anchorStates?.shape == [3, 64])
        }

        @Test("a pack round-trips through disk, and tampered weights are refused")
        func disk() throws {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent("raolm-pack-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: directory) }
            let (pack, _) = try Self.pack()
            try pack.save(to: directory)
            let loaded = try UmbrellaPack.load(from: directory)
            #expect(loaded.sha256 == pack.sha256 && loaded.cut == 1)
            #expect(loaded.anchors == pack.anchors && loaded.heldOut == pack.heldOut)
            #expect(loaded.anchorStates?.asArray(Float.self) == pack.anchorStates?.asArray(Float.self))
            #expect(loaded.info?.baseSHA256 != nil)
            // The directory is a vocabulary's too.
            #expect(try VocabularyPack.load(from: directory).sha256 == pack.vocabulary.sha256)
            var info = try #require(loaded.info)
            info.sha256 = String(repeating: "0", count: 64)
            try JSONCoding.write(info, to: directory.appendingPathComponent(UmbrellaPack.infoFile))
            #expect(throws: UmbrellaPackError.self) { _ = try UmbrellaPack.load(from: directory) }
        }
    }
}

extension ModelMLXSuites {
    @Suite("The importer's reference check (C0)", .serialized)
    struct ReferenceCheckTests {
        @Test("RaoLM's transformer and Frigate's Llama agree on a saved checkpoint; tampered weights do not")
        func agrees() async throws {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent("raolm-ref-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: directory) }
            let model = try RaoTransformer.make(config: testConfig, seed: 9)
            let tokenizer = try await RaoTokenizer.load()
            _ = try Checkpoint.save(model: model, to: directory, tokenizerDirectory: tokenizer.directory)
            let prompts = [tokenizer.encode("The river ran north past the mill."), tokenizer.encode("A second, longer prompt about the tide and the harbour wall.")]
            #expect(try ReferenceCheck.llama(folder: directory, model: model, prompts: prompts) < ReferenceCheck.tolerance)
            let other = try RaoTransformer.make(config: testConfig, seed: 10)
            #expect(try ReferenceCheck.llama(folder: directory, model: other, prompts: prompts) > ReferenceCheck.tolerance)
            // Weights stored in bf16, as the hub's are: both sides read them widened to float32.
            let weights = directory.appendingPathComponent(Checkpoint.weightsFile)
            try MLX.save(arrays: try loadArrays(url: weights).mapValues { $0.asType(.bfloat16) }, url: weights)
            let widened = RaoTransformer(testConfig)
            try Checkpoint.loadWeights(into: widened, from: directory)
            #expect(try ReferenceCheck.llama(folder: directory, model: widened, prompts: prompts) < ReferenceCheck.tolerance)
        }
    }
}

@Suite("Presets for imported shapes")
struct ImportedPresetTests {
    @Test("SmolLM2-360M's preset is its config, cut at 22; base-360m trains about 216M parameters per node")
    func smolLM2_360M() throws {
        let config = try RaoLMConfig.preset("base-360m")
        #expect(config.numHiddenLayers == 32 && config.hiddenSize == 960 && config.cut == 22 && config.hasTrunk)
        #expect(config.hiddenSize / config.numAttentionHeads == 64 && config.tieWordEmbeddings)
        #expect(abs(Double(config.nodeParameterCount) / 1e6 - 216) < 2, "\(config.nodeParameterCount)")
        try config.validate()
        #expect(RaoLMConfig.presetNames.contains("base-360m"))
    }
}
