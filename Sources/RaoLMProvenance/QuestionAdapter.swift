//
//  QuestionAdapter.swift
//  RaoLMProvenance
//
//  WHAT: The umbrella's question adapter (the owner's choice, 2026-10-01): a question becomes the
//        corpus-style stem a Thread completes, and the commons does the rewriting, prompted with
//        a few public-knowledge examples and no training. Threads stay document-only ingestion;
//        the question lives at the umbrella. The dataset's own templates are the fallback and the
//        baseline the bench judges the commons against.
//  PIN:  The examples are fixed public facts, never a Thread's: the adapter sees no private data.
//        Each example's stem ends in a blank, "___", where the answer goes, so the commons stops
//        at the stem instead of inventing a value (plain "Q: / A:" examples had SmolLM2-135M
//        write "Tillyburn was founded in 1999"); the rewrite is cut at the blank. Greedy, until a
//        newline. A rewrite is refused (and the rules take over) when it is empty, still a
//        question, or drops a capitalised word of the question: the subject must survive the
//        rewrite or the Threads cannot find the fact.
//

import Foundation
import MLX
import RaoLMCore
import RaoLMModel

public final class QuestionAdapter {
    public let model: RaoTransformer
    public let tokenizer: RaoTokenizer
    public let examples: [(question: String, stem: String)]

    /// Public knowledge in the dataset's question kinds: years, people, counts, versions, ports, sizes.
    public static let examples: [(question: String, stem: String)] = [
        ("When was the Eiffel Tower completed?", "The Eiffel Tower was completed in"),
        ("Who wrote Pride and Prejudice?", "Pride and Prejudice was written by"),
        ("What is the population of Iceland?", "Iceland has a population of"),
        ("What port does HTTPS use by default?", "HTTPS listens on port"),
        ("How tall is Mount Everest?", "The height of Mount Everest is"),
        ("Who painted the Mona Lisa?", "The Mona Lisa is attributed to"),
        ("In what year was the Louvre opened to the public?", "The Louvre opened to the public in"),
        ("Who is the mayor of Paris?", "The mayor of Paris is"),
        ("How long did the Apollo 11 mission last?", "The Apollo 11 mission lasted"),
        ("What version of Python introduced f-strings?", "F-strings were introduced in Python version"),
    ]

    public init(model: RaoTransformer, tokenizer: RaoTokenizer, examples: [(question: String, stem: String)] = QuestionAdapter.examples) {
        self.model = model
        self.tokenizer = tokenizer
        self.examples = examples
    }

    /// The stem for `question`: the commons' rewrite when it keeps the subject, else the rules',
    /// else the question itself.
    public func rewrite(_ question: String, maxTokens: Int = 24) -> QuestionRewrite {
        let started = Date()
        let question = question.trimmingCharacters(in: .whitespacesAndNewlines)
        let rewritten = Self.clean(decode(prompt(for: question), maxTokens: maxTokens))
        if Self.acceptable(rewritten, for: question) {
            return QuestionRewrite(question: question, stem: rewritten, rewriter: "commons", seconds: Date().timeIntervalSince(started),
                                   exampleCount: examples.count)
        }
        return Self.fallback(question, seconds: Date().timeIntervalSince(started))
    }

    /// The rules' rewrite of a question, or the question itself.
    public static func fallback(_ question: String, seconds: Double = 0) -> QuestionRewrite {
        if let rules = RuleRewriter.rewrite(question) {
            return QuestionRewrite(question: question, stem: rules.stem, rewriter: "rules", seconds: seconds)
        }
        return QuestionRewrite(question: question, stem: question, rewriter: "none", seconds: seconds)
    }

    static let blank = "___"

    func prompt(for question: String) -> String {
        examples.map { "Question: \($0.question)\nStem: \($0.stem) \(Self.blank)\n\n" }.joined() + "Question: \(question)\nStem:"
    }

    /// Greedy continuation of `text` until a newline, eos or `maxTokens`.
    func decode(_ text: String, maxTokens: Int) -> String {
        let prompt = tokenizer.encode(text)
        guard !prompt.isEmpty else { return "" }
        let cache = model.newCache(parameters: nil)
        var output = model.forward(MLXArray(prompt.map { Int32($0) }, [1, prompt.count]), cache: cache, captureTap: false)
        var generated: [Int] = []
        while generated.count < maxTokens {
            let next = argMax(output.logits[0, -1]).item(Int.self)
            if next == tokenizer.eosTokenID { break }
            let piece = tokenizer.tokenText(next)
            if piece.contains("\n") { break }
            generated.append(next)
            output = model.forward(MLXArray([Int32(next)], [1, 1]), cache: cache, captureTap: false)
        }
        return tokenizer.decode(generated)
    }

    /// The words before the blank, whitespace trimmed, a trailing "?" or ":" copied from the pattern dropped.
    static func clean(_ text: String) -> String {
        var stem = text
        if let blank = stem.range(of: "_") { stem = String(stem[..<blank.lowerBound]) }
        stem = stem.trimmingCharacters(in: .whitespacesAndNewlines)
        while let last = stem.last, "?:".contains(last) { stem.removeLast() }
        return stem.trimmingCharacters(in: .whitespaces)
    }

    /// Words a question opens with; a rewrite that still opens with one repeated the question.
    static let questionWords: Set<String> = ["who", "what", "when", "where", "which", "why", "how", "whom", "whose", "is", "are", "was", "were",
                                             "do", "does", "did", "can", "could"]

    /// A stem's tokens as the corpus writes a sentence that follows another: with a leading space,
    /// so a Thread's context and the stem concatenate to the tokens of the joined text, and the
    /// stem's first token is the mid-sentence form the Thread memorised.
    public static func stemTokens(_ stem: String, tokenizer: RaoTokenizer) -> [Int] {
        tokenizer.encode(" " + stem.trimmingCharacters(in: .whitespaces))
    }

    /// The question's subject: its capitalised words after the first, in order ("the Quince Fox
    /// Festival" → "Quince Fox Festival"), as they appear in `stem`; nil when the stem lacks them.
    public static func subjectRange(in stem: String, question: String) -> Range<String.Index>? {
        let words = question.split(whereSeparator: { !$0.isLetter && !$0.isNumber }).dropFirst().map(String.init)
            .filter { $0.first?.isUppercase == true }
        guard let first = words.first, let last = words.last, let start = stem.range(of: first), let end = stem.range(of: last, range: start.lowerBound..<stem.endIndex)
        else { return nil }
        return start.lowerBound..<end.upperBound
    }

    /// The token positions of the subject in the stem's tokens: those whose text overlaps the
    /// subject's characters, located in the tokens' own decoded text (which may carry a leading space).
    public static func subjectTokens(stem: String, tokens: [Int], question: String, tokenizer: RaoTokenizer) -> Range<Int>? {
        let decoded = tokens.map { tokenizer.tokenText($0) }.joined()
        guard let range = subjectRange(in: decoded, question: question) else { return nil }
        let lower = decoded.distance(from: decoded.startIndex, to: range.lowerBound)
        let upper = decoded.distance(from: decoded.startIndex, to: range.upperBound)
        var offset = 0
        var first: Int?
        var last: Int?
        for (i, token) in tokens.enumerated() {
            let text = tokenizer.tokenText(token)
            let end = offset + text.count
            if end > lower, offset < upper {
                if first == nil { first = i }
                last = i
            }
            offset = end
        }
        guard let first, let last else { return nil }
        return first..<(last + 1)
    }

    /// A rewrite stands when it is not empty, is not itself a question, and keeps the question's
    /// capitalised words (the subject's).
    static func acceptable(_ stem: String, for question: String) -> Bool {
        guard !stem.isEmpty, !stem.hasSuffix("?") else { return false }
        if let first = stem.split(whereSeparator: \.isWhitespace).first, questionWords.contains(first.lowercased()) { return false }
        let lower = stem.lowercased()
        let capitalised = question.split(whereSeparator: { !$0.isLetter && !$0.isNumber })
            .dropFirst().map(String.init).filter { $0.first?.isUppercase == true }
        guard !capitalised.isEmpty else { return true }
        return capitalised.allSatisfy { lower.contains($0.lowercased()) }
    }
}
