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

@Suite("RaoTransformer", .enabled(if: mlxTests))
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

@Suite("Vocabulary in a model", .enabled(if: mlxTests))
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
