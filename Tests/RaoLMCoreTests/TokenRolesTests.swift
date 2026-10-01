import Foundation
import Testing

@testable import RaoLMCore

@Suite("The owner's blend: form is the commons', content the Threads'")
struct TokenRolesTests {
    @Test("roles go by whole words: whitespace, punctuation and function words are form; names, numbers and content words are content")
    func roles() {
        // As SmolLM2's tokenizer writes "They asked: what did I read about Milise Garard? Nashett's 56047."
        let texts = ["They", " asked", ":", " what", " did", " I", " read", " about", " Mil", "ise", " Gar", "ard", "?",
                     " Nas", "he", "tt", "'s", " ", "5", "6", "0", "4", "7", "."]
        let roles = TokenRoles.roles(texts)
        let form: [TokenRole] = [.form]
        func at(_ i: Int) -> TokenRole { roles[i] }
        #expect(at(0) == .form && at(1) == .content && at(2) == .form && at(3) == .form && at(4) == .form && at(5) == .form)
        #expect(at(6) == .content && at(7) == .form)
        // Every piece of a name takes the name's role, even a piece that spells a function word.
        #expect([8, 9, 10, 11].allSatisfy { at($0) == .content } && at(12) == .form)
        #expect([13, 14, 15].allSatisfy { at($0) == .content } && at(16) == .form)
        // The space before a number is form; the number, one word, is content; so is the full stop form.
        #expect([at(17)] == form && (18...22).allSatisfy { at($0) == .content } && at(23) == .form)
    }

    @Test("credit: a form token is wholly the commons'; a content token is each strand's as supplied; bits are the commons' surprise")
    func credit() {
        func share(_ strand: String, _ share: Float, alone: Float) -> StrandShare {
            var value = StrandShare(strand: strand, threadID: nil, gate: 0.5, open: true, bestScore: nil, lmProb: nil, lmEntropy: nil,
                                    knn: 0, share: share)
            value.alone = alone
            return value
        }
        func trace(_ index: Int, _ text: String, thread: Float, commons: Float, base: Float) -> TokenTrace {
            var value = TokenTrace(index: index, token: index, text: text, isPrompt: false, lmEntropy: 1, knnEntropy: 1, mixedEntropy: 1,
                                   sourceEntropy: 1, lmProb: 0.5, agreement: 0, mixedProb: 0.5, lambda: 0.5, neighbours: [])
            value.strands = [share("ambient", thread, alone: 0.9), share("commons", commons, alone: base)]
            return value
        }
        let texts = ["Kestleham", " holds", " ", "4", "9", "."]
        var traces = [trace(1, " holds", thread: 0.8, commons: 0.2, base: 0.25), trace(2, " ", thread: 0.9, commons: 0.1, base: 0.5),
                      trace(3, "4", thread: 0.95, commons: 0.05, base: 0.125), trace(5, ".", thread: 0.7, commons: 0.3, base: 1)]
        TokenRoles.assignCredit(&traces, commons: "commons", texts: texts)
        #expect(traces.map(\.role) == [.content, .form, .content, .form])
        // Content: as supplied. Form: the commons', whoever supplied it.
        #expect(traces[0].strands?.map(\.credit) == [0.8, 0.2])
        #expect(traces[1].strands?.map(\.credit) == [0, 1] && traces[3].strands?.map(\.credit) == [0, 1])
        #expect(traces[2].strands?.map(\.credit) == [0.95, 0.05])
        // Bits: −log₂ of what the commons gave the token.
        #expect(traces[0].bits == 2 && traces[1].bits == 1 && traces[2].bits == 3 && traces[3].bits == 0)
        // Without strands nothing is assigned.
        var plain = [TokenTrace(index: 0, token: 0, text: "x", isPrompt: false, lmEntropy: 1, knnEntropy: 1, mixedEntropy: 1,
                                sourceEntropy: 1, lmProb: 0.5, agreement: 0, mixedProb: 0.5, lambda: 0.5, neighbours: [])]
        TokenRoles.assignCredit(&plain, commons: "commons", texts: ["x"])
        #expect(plain[0].role == nil && plain[0].bits == nil)
    }
}
