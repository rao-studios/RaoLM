//
//  RaoTokenizer.swift
//  RaoLMModel
//
//  WHAT: SmolLM2's 49,152-token byte-level BPE, loaded from the files vendored in this
//        target's resources (or a directory named by --tokenizer / RAOLM_TOKENIZER_DIR).
//  PIN:  Encoding never adds special tokens, and the loader proves it: RaoLM inserts
//        <|endoftext|> itself. Every partition is tokenized on its own, so a token offset
//        inside a partition is reproducible by re-tokenizing that partition alone — which
//        is exactly what the verifier does against the live Thread.
//

import Foundation
import Hub
import RaoLMCore
import Tokenizers

public enum RaoTokenizerError: Error, CustomStringConvertible {
    case resourceMissing(tried: [String])
    case prependsSpecialTokens(probe: [Int])
    case unreadable(String)

    public var description: String {
        switch self {
        case .resourceMissing(let tried):
            return "tokenizer files not found (tried: \(tried.joined(separator: ", "))); set RAOLM_TOKENIZER_DIR to a folder with tokenizer.json and tokenizer_config.json"
        case .prependsSpecialTokens(let probe):
            return "tokenizer adds special tokens on encode (probe \(probe)); RaoLM needs raw encoding"
        case .unreadable(let reason):
            return "tokenizer unreadable: \(reason)"
        }
    }
}

public final class RaoTokenizer: @unchecked Sendable {
    public static let environmentKey = "RAOLM_TOKENIZER_DIR"

    public let tokenizer: any Tokenizers.Tokenizer
    public let directory: URL
    public let tokenizerSHA256: String
    public let eosTokenID: Int
    public let vocabularySize: Int

    private var textCache: [Int: String] = [:]
    private let lock = NSLock()

    private init(tokenizer: any Tokenizers.Tokenizer, directory: URL, sha256: String, eos: Int, vocabularySize: Int) {
        self.tokenizer = tokenizer
        self.directory = directory
        self.tokenizerSHA256 = sha256
        self.eosTokenID = eos
        self.vocabularySize = vocabularySize
    }

    /// The vendored SmolLM2 tokenizer folder inside this target's resource bundle.
    public static func bundledDirectory() -> URL? {
        Bundle.module.url(forResource: "Tokenizer", withExtension: nil)
    }

    public static func resolveDirectory(
        _ explicit: String? = nil, environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> URL {
        var tried: [String] = []
        var candidates: [URL] = []
        if let explicit, !explicit.isEmpty { candidates.append(URL(fileURLWithPath: (explicit as NSString).expandingTildeInPath)) }
        if let value = environment[environmentKey], !value.isEmpty { candidates.append(URL(fileURLWithPath: value)) }
        if let bundled = bundledDirectory() { candidates.append(bundled) }
        for candidate in candidates {
            tried.append(candidate.path)
            if FileManager.default.fileExists(atPath: candidate.appendingPathComponent("tokenizer.json").path) {
                return candidate
            }
        }
        throw RaoTokenizerError.resourceMissing(tried: tried.isEmpty ? ["<bundle resource Tokenizer>"] : tried)
    }

    public static func load(directory: URL? = nil) async throws -> RaoTokenizer {
        let folder = try directory ?? resolveDirectory()
        let tokenizer = try await AutoTokenizer.from(modelFolder: folder)
        let sha = try ContentHash.sha256Hex(fileAt: folder.appendingPathComponent("tokenizer.json"))
        let eos = tokenizer.eosTokenId ?? 0
        let probe = tokenizer.encode(text: "a", addSpecialTokens: false)
        guard let first = probe.first, first != eos, probe.count == 1 else {
            throw RaoTokenizerError.prependsSpecialTokens(probe: probe)
        }
        return RaoTokenizer(
            tokenizer: tokenizer, directory: folder, sha256: sha, eos: eos,
            vocabularySize: try vocabularySize(folder.appendingPathComponent("tokenizer.json")))
    }

    static func vocabularySize(_ url: URL) throws -> Int {
        guard let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any],
              let model = object["model"] as? [String: Any],
              let vocab = model["vocab"] as? [String: Any]
        else { throw RaoTokenizerError.unreadable("no model.vocab in \(url.lastPathComponent)") }
        let added = (object["added_tokens"] as? [[String: Any]] ?? []).compactMap { $0["id"] as? Int }
        return max(vocab.count, (added.max() ?? -1) + 1)
    }

    /// Raw encoding: no BOS, no EOS, nothing prepended.
    public func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }

    public func decode(_ tokens: [Int]) -> String {
        tokenizer.decode(tokens: tokens, skipSpecialTokens: false)
    }

    /// One token's text (cached).
    public func tokenText(_ id: Int) -> String {
        lock.lock()
        if let cached = textCache[id] {
            lock.unlock()
            return cached
        }
        lock.unlock()
        let text = tokenizer.decode(tokens: [id], skipSpecialTokens: false)
        lock.lock()
        textCache[id] = text
        lock.unlock()
        return text
    }

    /// UTF-8 byte offset where each token starts, plus the total, for text the tokenizer
    /// produced. Exact for byte-level BPE on ASCII text.
    public func byteOffsets(_ tokens: [Int]) -> [Int] {
        var offsets = [0]
        offsets.reserveCapacity(tokens.count + 1)
        var total = 0
        for token in tokens {
            total += tokenText(token).utf8.count
            offsets.append(total)
        }
        return offsets
    }

    public var ref: TokenizerRef {
        TokenizerRef(
            id: RaoLMVersion.tokenizerID, revision: RaoLMVersion.tokenizerRevision,
            tokenizerSHA256: tokenizerSHA256, vocabSize: vocabularySize, eosTokenID: eosTokenID)
    }

    /// The files a checkpoint directory needs to load as a self-contained HF model folder.
    public static let checkpointFiles = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
}
