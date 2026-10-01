import Foundation
import Testing

@testable import RaoLMCore

@Suite("The rule rewriter: the dataset's questions become their stems")
struct RuleRewriterTests {
    @Test("every question template of every kind rewrites to its stem, for a bare and a prefixed subject")
    func roundTrip() {
        var count = 0
        var shared = 0
        for kind in FactKind.allCases where DatasetVoices.questions[kind] != nil {
            for subject in ["Tillyburn", "the Quince Fox Festival", "incident DY-3878", "the Briskqueue service"] {
                for (template, question) in zip(DatasetVoices.questions[kind] ?? [], RuleRewriter.questions(of: kind, subject: subject)) {
                    let rewritten = RuleRewriter.rewrite(question)
                    // A template two kinds share ("When was {s} born?": a researcher or a painter) names
                    // one of them, and rewrites to that kind's stem; the bench's Q1 counts the cost.
                    let kinds = DatasetVoices.questions.filter { $0.value.contains(template) }.map(\.key)
                    if kinds.count > 1 {
                        shared += 1
                        #expect(rewritten.map { kinds.contains($0.kind) && $0.stem == RuleRewriter.stem(of: $0.kind, subject: subject) } == true, "\(question)")
                    } else {
                        #expect(rewritten?.kind == kind && rewritten?.stem == RuleRewriter.stem(of: kind, subject: subject), "\(question)")
                    }
                    count += 1
                }
            }
        }
        #expect(count == 27 * 3 * 4 && shared == 3 * 2 * 4, "\(count) questions, \(shared) on shared templates")
    }

    @Test("the subject keeps its own case and spacing; the template's words match in any case; other questions return nil")
    func details() {
        #expect(RuleRewriter.rewrite("when was Tillyburn founded?")?.stem == "The article says Tillyburn was founded in")
        #expect(RuleRewriter.rewrite("  Who is the mayor of  Tillyburn ? ")?.stem == "The current mayor of Tillyburn is")
        #expect(RuleRewriter.rewrite("What port does the Briskqueue service listen on?")?.stem == "By default the Briskqueue service listens on port")
        #expect(RuleRewriter.rewrite("Tell me about Tillyburn") == nil)
        #expect(RuleRewriter.rewrite("When was founded?") == nil)
        #expect(RuleRewriter.normalised("  The Mayor of Tillyburn  is. ") == "the mayor of tillyburn is")
    }
}

@Suite("Where an answer ends")
struct AnswerStopTests {
    @Test("a sentence ends at its full stop when the next token opens with whitespace; numbers and versions stay whole")
    func sentences() {
        #expect(AnswerStop.sentenceEnded(previous: "Tillyburn was founded in 1128.", next: " The"))
        #expect(AnswerStop.sentenceEnded(previous: "Is it so?", next: " Yes"))
        #expect(AnswerStop.sentenceEnded(previous: "Done.", next: "\n"))
        #expect(!AnswerStop.sentenceEnded(previous: "We pinned it at version 3.", next: "14"))
        #expect(!AnswerStop.sentenceEnded(previous: "The mayor is Jaora", next: " Seling"))
        #expect(!AnswerStop.sentenceEnded(previous: "", next: " The"))
        #expect(AnswerStop.sentenceEnded(previous: "Lasted 40 minutes. ", next: " Then"))
        // Text packed without a space after its full stop still ends there.
        #expect(AnswerStop.sentenceEnded(previous: "Tana Halman.", next: "I"))
        #expect(!AnswerStop.sentenceEnded(previous: "Tana Halman.", next: "com"))
    }
}

@Suite("Followed credit: the trajectory decides a shared fact")
struct FollowedCreditTests {
    static let partitions: [Int: PartitionRef] = [
        0: PartitionRef(row: 0, documentID: "doc-a", documentName: "A", partitionIndex: 0, partitionURL: nil, threadPartitionID: nil,
                        textSHA256: "a", tokenCount: 40),
        5: PartitionRef(row: 5, documentID: "doc-b", documentName: "B", partitionIndex: 0, partitionURL: nil, threadPartitionID: nil,
                        textSHA256: "b", tokenCount: 40),
    ]
    static let rowOffsets = ["ambient": 0, "craft": 5]

    /// A generated trace: its token, text and role, each Thread's credit and whether that Thread's chain expected the token.
    static func trace(_ index: Int, token: Int, text: String, role: TokenRole, ambient: (credit: Float, expects: Bool),
                      craft: (credit: Float, expects: Bool)) -> TokenTrace {
        func share(_ name: String, _ credit: Float, _ expects: Bool, row: Int) -> StrandShare {
            var value = StrandShare(strand: name, threadID: nil, gate: 0.5, open: true, bestScore: nil, lmProb: nil, lmEntropy: nil, knn: 0, share: 0.5)
            value.credit = credit
            value.trajectory = StrandTrajectory(
                length: 3, trace: 0, entry: 1, at: TokenPosition(row: 0, offset: 7), next: expects ? token : token + 100, phase: nil, fit: 0,
                manner: nil)
            _ = row
            return value
        }
        var value = TokenTrace(index: index, token: token, text: text, isPrompt: false, lmEntropy: 1, knnEntropy: 1, mixedEntropy: 1,
                               sourceEntropy: 1, lmProb: 0.5, agreement: 0, mixedProb: 0.5, lambda: 0.5, neighbours: [])
        value.role = role
        value.strands = [share("ambient", ambient.credit, ambient.expects, row: 0), share("craft", craft.credit, craft.expects, row: 5),
                         { var c = StrandShare(strand: "commons", threadID: nil, gate: 0.2, open: true, bestScore: nil, lmProb: nil, lmEntropy: nil,
                                               knn: 0, share: 0); c.credit = 0.1; return c }()]
        return value
    }

    @Test("the Thread whose chains advanced on the sentence takes every shared content token; the commons keeps its own")
    func followerTakes() {
        var traces = [
            Self.trace(3, token: 1, text: " 11", role: .content, ambient: (0.5, true), craft: (0.4, false)),
            Self.trace(4, token: 2, text: "28", role: .content, ambient: (0.45, true), craft: (0.45, false)),
            Self.trace(5, token: 3, text: ",", role: .form, ambient: (0, false), craft: (0, false)),
            Self.trace(6, token: 4, text: " when", role: .form, ambient: (0, true), craft: (0, false)),
        ]
        let decisions = FollowedCredit.apply(&traces, rowOffsets: Self.rowOffsets, partitions: Self.partitions, commons: "commons")
        #expect(decisions.count == 1 && decisions[0].followed == "ambient" && decisions[0].documentID == "doc-a")
        #expect(decisions[0].advances == ["ambient": 3])
        #expect(traces[0].strands?.map(\.credit) == [0.9, 0, 0.1] && traces[0].followed == "ambient")
        #expect(traces[1].strands?.map(\.credit) == [0.9, 0, 0.1])
        // Form tokens and tokens one Thread held alone are untouched.
        #expect(traces[2].followed == nil && traces[3].followed == nil)
    }

    @Test("a tie, or no chain advancing at all, changes nothing: origin alone earns nothing and so does an unfollowed copy")
    func ties() {
        var tied = [
            Self.trace(3, token: 1, text: " 11", role: .content, ambient: (0.5, true), craft: (0.4, true)),
            Self.trace(4, token: 2, text: "28", role: .content, ambient: (0.45, false), craft: (0.45, true)),
            Self.trace(5, token: 3, text: " years", role: .content, ambient: (0.5, true), craft: (0.4, false)),
        ]
        let decisions = FollowedCredit.apply(&tied, rowOffsets: Self.rowOffsets, partitions: Self.partitions, commons: "commons")
        #expect(decisions[0].followed == nil && decisions[0].advances == ["ambient": 2, "craft": 2])
        // A tie splits each shared content token evenly between the two.
        #expect(tied.allSatisfy { $0.followed == nil } && tied[0].strands?.map(\.credit) == [0.45, 0.45, 0.1])

        // Equal advances tie even when one copy's neighbours weigh more: weight never decides.
        var weighted = [
            Self.trace(3, token: 1, text: " 11", role: .content, ambient: (0.5, true), craft: (0.4, true)),
            Self.trace(4, token: 2, text: "28", role: .content, ambient: (0.5, true), craft: (0.4, true)),
        ]
        for i in weighted.indices {
            weighted[i].neighbours = [Neighbour(rank: 1, entry: 7, score: 0.99, weight: 0.9, value: weighted[i].token, matches: true,
                                                key: TokenPosition(row: 0, offset: 6), cited: TokenPosition(row: 0, offset: 7), sourceLoss: 0, sourceEntropy: 0)]
        }
        let weightedDecisions = FollowedCredit.apply(&weighted, rowOffsets: Self.rowOffsets, partitions: Self.partitions, commons: "commons")
        #expect(weightedDecisions[0].followed == nil && weightedDecisions[0].matches["ambient"] == 1.8)
        #expect(weighted.allSatisfy { $0.strands?.map(\.credit) == [0.45, 0.45, 0.1] && $0.followed == nil })

        var silent = [Self.trace(3, token: 1, text: " 11", role: .content, ambient: (0.5, false), craft: (0.4, false))]
        #expect(FollowedCredit.apply(&silent, rowOffsets: Self.rowOffsets, partitions: Self.partitions, commons: "commons")[0].followed == nil)
        #expect(silent[0].strands?.map(\.credit) == [0.5, 0.4, 0.1])
    }

    @Test("sentences are decided one by one, and the last token counts")
    func sentences() {
        var traces = [
            Self.trace(3, token: 1, text: " 11", role: .content, ambient: (0.5, true), craft: (0.4, false)),
            Self.trace(4, token: 5, text: "28", role: .content, ambient: (0.5, true), craft: (0.4, false)),
            Self.trace(5, token: 2, text: ".", role: .form, ambient: (0, false), craft: (0, false)),
            Self.trace(6, token: 3, text: " The", role: .form, ambient: (0, false), craft: (0, true)),
            Self.trace(7, token: 4, text: " mayor", role: .content, ambient: (0.4, false), craft: (0.5, true)),
        ]
        let decisions = FollowedCredit.apply(&traces, rowOffsets: Self.rowOffsets, partitions: Self.partitions, commons: "commons")
        #expect(decisions.map(\.followed) == ["ambient", "craft"])
        #expect(decisions.map(\.window) == [TokenRange(start: 0, end: 3), TokenRange(start: 3, end: 5)])
        #expect(traces[0].followed == "ambient" && traces[4].followed == "craft")
        #expect(traces[4].strands?.map(\.credit) == [0, 0.9, 0.1])
    }

    @Test("one more advance than the rival is not a decision: copies on separately trained nodes differ by a token")
    func margin() {
        var traces = [
            Self.trace(3, token: 1, text: " 11", role: .content, ambient: (0.5, true), craft: (0.4, true)),
            Self.trace(4, token: 2, text: "28", role: .content, ambient: (0.5, true), craft: (0.4, false)),
            Self.trace(5, token: 3, text: " people", role: .content, ambient: (0.5, true), craft: (0.4, true)),
        ]
        let decisions = FollowedCredit.apply(&traces, rowOffsets: Self.rowOffsets, partitions: Self.partitions, commons: "commons")
        #expect(decisions[0].followed == nil && decisions[0].advances == ["ambient": 3, "craft": 2] && decisions[0].tied == ["ambient", "craft"])
        // Undecided between two holders that advanced alike: they split what they hold evenly.
        #expect(traces.allSatisfy { $0.strands?.map(\.credit) == [0.45, 0.45, 0.1] })
        // Two more, and more than half the content tokens: decided.
        var clear = traces
        clear[2] = Self.trace(5, token: 3, text: " people", role: .content, ambient: (0.5, true), craft: (0.4, false))
        #expect(FollowedCredit.apply(&clear, rowOffsets: Self.rowOffsets, partitions: Self.partitions, commons: "commons")[0].followed == "ambient")
    }
}
