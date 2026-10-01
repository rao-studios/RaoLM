//
//  TokenRoles.swift
//  RaoLMCore
//
//  WHAT: The owner's blend (2026-10-01): the commons provides a text's structure and form, the
//        Threads its content, information and taste, and the Threads stay the primary sources.
//        Every token is form (whitespace, punctuation, a function word) or content (names,
//        numbers, content words: the information and a voice's word choices). Roles go by whole
//        words, so every piece of a word takes the word's role. A form token's credit is the
//        commons'; a content token's is each strand's as it supplied it, the commons keeping what
//        it filled in. Each token counts by its bits, −log₂ of what the commons gave it, so the
//        predictable form of a text weighs little and its specifics weigh most.
//

import Foundation

public enum TokenRole: String, Codable, Sendable {
    /// Structure: whitespace, punctuation or a function word. The commons provides it.
    case form
    /// Information and a voice's word choices: names, numbers, content words. The Threads provide it.
    case content
}

public enum TokenRoles {
    /// Closed-class English words: articles, pronouns, prepositions, conjunctions, auxiliaries and
    /// a few other function words. A fixed list, the same for every Thread and every version.
    public static let functionWords: Set<String> = Set("""
        a an the this that these those my your his her its our their i you he she it we they me him us them who whom whose which what
        of in on at to for with by from about as into onto over under between through during before after above below up down out off
        near since until upon within without and or but nor so yet if then than because while when where though although whether
        is are was were be been being am has have had do does did will would can could may might shall should must
        not no there here also just very too only more most such each every some any all both either neither other another own same
        """.split(whereSeparator: \.isWhitespace).map(String.init))

    /// Contractions a tokenizer splits off a word: form, like the words they stand for.
    static let clitics: Set<String> = ["'s", "'t", "'re", "'ll", "'d", "'ve", "'m", "’s", "’t", "’re", "’ll", "’d", "’ve", "’m"]

    /// Each token's role, from the tokens' texts in order, each as the tokenizer decodes it alone
    /// (its leading space included).
    public static func roles(_ texts: [String]) -> [TokenRole] {
        var roles = [TokenRole](repeating: .content, count: texts.count)
        var start = 0
        func close(_ end: Int) {
            guard end > start else { return }
            let role = role(of: texts[start..<end].joined().trimmingCharacters(in: .whitespacesAndNewlines))
            for i in start..<end { roles[i] = role }
            start = end
        }
        for i in texts.indices.dropFirst() where startsWord(texts[i], after: texts[i - 1]) { close(i) }
        close(texts.count)
        return roles
    }

    static func role(of word: String) -> TokenRole {
        if !word.contains(where: { $0.isLetter || $0.isNumber }) { return .form }
        let lower = word.lowercased()
        if clitics.contains(lower) { return .form }
        if word.allSatisfy(\.isLetter), functionWords.contains(lower) { return .form }
        return .content
    }

    /// A token starts a word when it opens with whitespace or with neither a letter nor a digit, or
    /// when the token before it did not end in one: a run of letters and digits is one word.
    static func startsWord(_ text: String, after previous: String) -> Bool {
        guard let first = text.first else { return false }
        if first.isWhitespace || !(first.isLetter || first.isNumber) { return true }
        guard let last = previous.last else { return true }
        return !(last.isLetter || last.isNumber)
    }

    /// The blend on a braided generation's traces. `texts` are the texts of prompt ++ generated, so
    /// the word a trace's token belongs to is whole. A form token is wholly `commons`'; a content
    /// token is each strand's as it supplied it. Each trace records its role and its bits.
    public static func assignCredit(_ traces: inout [TokenTrace], commons: String, texts: [String]) {
        let roles = roles(texts)
        for i in traces.indices {
            guard var strands = traces[i].strands, roles.indices.contains(traces[i].index) else { continue }
            let role = roles[traces[i].index]
            traces[i].role = role
            if let base = strands.first(where: { $0.strand == commons })?.alone {
                traces[i].bits = Float(-log2(Double(max(base, 1e-30))))
            }
            for s in strands.indices {
                strands[s].credit = role == .form ? (strands[s].strand == commons ? 1 : 0) : strands[s].share
            }
            traces[i].strands = strands
        }
    }
}
