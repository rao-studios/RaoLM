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
