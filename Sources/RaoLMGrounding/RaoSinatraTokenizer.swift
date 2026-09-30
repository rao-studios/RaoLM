//
//  RaoSinatraTokenizer.swift
//  RaoLMGrounding
//
//  WHAT: RaoLM's SmolLM2 tokenizer seen through SinatraHarness's `SinatraTokenizing`, so the
//        grounding measurement reads the same ids the model and the corpus use.
//  PIN:  Sinatra matches tokens for attribution by `decode([id])` trimmed and lowercased, so
//        the byte-level "Ġ" prefix never matters; `tokenString` only feeds control-token
//        detection ("<|endoftext|>" and friends look like "<…>" and are excluded).
//

import Foundation
import RaoLMModel
import SinatraHarness

public struct RaoSinatraTokenizer: SinatraTokenizing {
    public let tokenizer: RaoTokenizer

    public init(_ tokenizer: RaoTokenizer) {
        self.tokenizer = tokenizer
    }

    /// Raw encoding: no BOS, no EOS (RaoTokenizer never adds special tokens).
    public func encode(_ text: String) -> [Int] {
        tokenizer.encode(text)
    }

    /// Single ids go through the tokenizer's cache: the measurement decodes one id at a time.
    public func decode(_ ids: [Int]) -> String {
        ids.count == 1 ? tokenizer.tokenText(ids[0]) : tokenizer.decode(ids)
    }

    /// The raw vocabulary string ("Ġthe", "<|endoftext|>").
    public func tokenString(_ id: Int) -> String? {
        tokenizer.tokenizer.convertIdToToken(id)
    }

    public var specialTokenIds: Set<Int> { [tokenizer.eosTokenID] }
}
