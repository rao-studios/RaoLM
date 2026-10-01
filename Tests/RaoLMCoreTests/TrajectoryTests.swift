import Foundation
import Testing

@testable import RaoLMCore

@Suite("Retrieval trajectory")
struct TrajectoryTests {
    /// Three documents: A (60 tokens over two partitions of 30), B (40), C (140). Every token id is
    /// unique, so the oracle below retrieves exactly the entry whose context is the token asked.
    private struct Index {
        let corpus: TrajectoryCorpus
        let partitions: [PartitionRef]
        let entryOfToken: [Int: Int]

        init() {
            func partition(_ row: Int, _ document: String, _ index: Int, _ tokens: Int) -> PartitionRef {
                PartitionRef(row: row, documentID: document, documentName: document, partitionIndex: index, partitionURL: nil,
                             threadPartitionID: nil, textSHA256: "\(row)", tokenCount: tokens)
            }
            partitions = [partition(0, "A", 0, 30), partition(1, "A", 1, 30), partition(2, "B", 0, 40), partition(3, "C", 0, 140)]
            var values: [Int32] = []
            var rows: [Int32] = []
            var offsets: [Int32] = []
            var entryOfToken: [Int: Int] = [:]
            // Each document's entries: every context but the last token (its value would be eos).
            for (base, length, row) in [(1000, 60, 0), (2000, 40, 2), (3000, 140, 3)] {
                for position in 0..<(length - 1) {
                    entryOfToken[base + position] = values.count
                    values.append(Int32(base + position + 1))
                    rows.append(Int32(base == 1000 ? (position < 30 ? 0 : 1) : row))
                    offsets.append(Int32(base == 1000 ? position % 30 : position))
                }
            }
            corpus = TrajectoryCorpus(values: values, keyRow: rows, keyOffset: offsets, partitions: partitions)
            self.entryOfToken = entryOfToken
        }

        func hit(_ entry: Int, score: Float = 0.99) -> StrandHit {
            StrandHit(entry: entry, score: score, value: Int(corpus.values[entry]), key: corpus.position(of: entry),
                      cited: corpus.position(of: entry), sourceLoss: 0, sourceEntropy: 0)
        }

        /// What retrieval returns at a position whose token is `token`: its own entry, if it has one.
        func hits(_ token: Int) -> [StrandHit] { entryOfToken[token].map { [hit($0)] } ?? [] }

        func run(_ text: [Int], rule: TrajectoryRule = .standard) -> [StrandTrajectory] {
            var tracker = TrajectoryTracker(corpus: corpus, rule: rule, tau: 0.05)
            return text.map { tracker.step(token: $0, hits: hits($0)) }
        }
    }

    private let index = Index()
    private func a(_ range: Range<Int>) -> [Int] { range.map { 1000 + $0 } }
    private func b(_ range: Range<Int>) -> [Int] { range.map { 2000 + $0 } }
    private func c(_ range: Range<Int>) -> [Int] { range.map { 3000 + $0 } }
    private func junk(_ count: Int) -> [Int] { (0..<count).map { 9000 + $0 } }

    @Test("trace: nothing up to a phrase, 1 at an arc")
    func trace() {
        let rule = TrajectoryRule.standard
        #expect(rule.trace(length: 0) == 0 && rule.trace(length: 24) == 0 && rule.trace(length: 60) == 0.5 && rule.trace(length: 96) == 1)
        #expect(rule.trace(length: 400) == 1)
    }

    @Test("the corpus map: documents, places across partitions, and successors that never leave a document")
    func corpus() {
        let corpus = index.corpus
        let a30 = index.entryOfToken[1030]!
        #expect(corpus.positions[a30] == 30 && corpus.documents[a30] == corpus.documents[index.entryOfToken[1000]!])
        #expect(corpus.successor(of: index.entryOfToken[1029]!) == a30, "a partition boundary inside a document")
        #expect(corpus.successor(of: index.entryOfToken[1058]!) == nil, "the next entry is B's first")
        #expect(corpus.phase(of: index.entryOfToken[1000]!) == 0 && corpus.phase(of: index.entryOfToken[1058]!) > 0.95)
        #expect(corpus.lengths.map(Int.init) == [60, 40, 140])
    }

    @Test("with a paragraph break between partitions, positions run on across it and so do chains")
    func breaks() {
        // One document of partitions 3, 2 and 4 tokens long, joined by two-token breaks addressed
        // to the partition they follow: (0, 3), (0, 4), then (1, 2), (1, 3).
        let partitions = [(0, 0, 3), (1, 1, 2), (2, 2, 4)].map { row, index, tokens in
            PartitionRef(row: row, documentID: "D", documentName: "D", partitionIndex: index, partitionURL: nil,
                         threadPartitionID: nil, textSHA256: "\(row)", tokenCount: tokens)
        }
        let addresses: [(Int32, Int32)] = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2)]
        let corpus = TrajectoryCorpus(
            values: addresses.indices.map { Int32($0 + 1) }, keyRow: addresses.map(\.0), keyOffset: addresses.map(\.1),
            partitions: partitions, breakLength: 2)
        #expect(corpus.positions.map(Int.init) == Array(0..<12))
        #expect(corpus.lengths == [13])
        #expect((0..<11).allSatisfy { corpus.successor(of: $0) == $0 + 1 }, "a chain runs through both breaks")
        #expect(corpus.successor(of: 11) == nil)
        // An index made before the break: the next partition starts right after the last token.
        let before = TrajectoryCorpus(values: [1, 2], keyRow: [0, 1], keyOffset: [2, 0], partitions: partitions)
        #expect(before.positions == [2, 3] && before.successor(of: 0) == 1 && before.lengths == [9])
    }

    @Test("a text that follows one document counts every token, across a partition boundary, to the document's end")
    func follows() {
        let steps = index.run(a(0..<60))
        #expect(steps.map(\.length) == Array(0..<60))
        #expect(steps[40].next == 1041 && steps[40].at == TokenPosition(row: 1, offset: 10))
        #expect(steps.last?.length == 59 && steps.last?.entry == nil, "the chain finished with the document")
        #expect(steps[59].trace > 0.45)
    }

    @Test("a phrase from one document and a phrase from another trace nothing")
    func phrases() {
        let steps = index.run(a(0..<20) + b(0..<20))
        #expect(steps.map(\.length).max() == 19)
        #expect(steps.allSatisfy { $0.trace == 0 })
    }

    @Test("the same sentences out of order trace nothing: no chain runs back, or forward past a bridge")
    func shuffled() {
        let steps = index.run(a(15..<30) + a(0..<15) + a(30..<45))
        #expect((steps.map(\.length).max() ?? 0) <= 15)
        #expect(steps.allSatisfy { $0.trace == 0 })
    }

    @Test("a bridge carries a chain over a changed value, never over a sentence")
    func bridge() {
        let value = index.run(a(0..<20) + junk(1) + a(21..<41))
        #expect(value.map(\.length).max() == 38, "19 before the change, 19 after, one chain")
        #expect(value.last!.trace > 0)
        let sentence = index.run(a(0..<20) + junk(12) + a(32..<52))
        #expect(sentence.map(\.length).max() == 19)
        #expect(sentence.allSatisfy { $0.trace == 0 })
        // A dropped word: the text skips a corpus token.
        let dropped = index.run(a(0..<20) + a(22..<42))
        #expect(dropped.map(\.length).max() == 38)
    }

    @Test("a chain never crosses into the next document, even at the next entry")
    func documents() {
        let steps = index.run(a(45..<60) + b(0..<15))
        #expect(steps.map(\.length).max() == 14)
    }

    @Test("manner: a text that runs through its documents in order has an arc, the same blocks reversed do not")
    func manner() {
        let forward = index.run(c(0..<140))
        #expect(forward[62].manner == nil, "fewer than four blocks: ask")
        let last = forward.last!
        #expect(last.arc == 1 && abs(last.fit - 1) < 1e-4 && abs((last.manner ?? 0) - 1) < 1e-4)
        let blocks = stride(from: 0, to: 128, by: 16).map { c($0..<($0 + 16)) }
        let reversed = index.run(blocks.reversed().flatMap { $0 })
        #expect((reversed.last?.arc ?? 0) < 0 && reversed.last?.manner == 0)
        #expect(reversed.last!.fit > 0.8, "the wording is familiar either way")
        let generic = index.run(junk(140))
        #expect(generic.last?.manner == nil && generic.last?.fit == 0, "no hits: no phase, no fit")
    }

    @Test("the wire alone: chains of hits that follow one another, a lower bound on the node's")
    func wire() {
        let text = a(0..<30) + b(0..<10)
        let documentOfRow = Dictionary(uniqueKeysWithValues: index.partitions.map { ($0.row, $0.documentID) })
        let lengths = TrajectoryAudit.wireLengths(tokens: text, hits: text.map(index.hits), documentOfRow: documentOfRow)
        #expect(lengths.prefix(30) == ArraySlice(Array(0..<30)))
        #expect(lengths.last == 9)
        let node = index.run(text)
        #expect(zip(lengths, node.map(\.length)).allSatisfy { $0 <= $1 })
    }

    @Test("a position's trajectory round-trips, and one with no chain writes no chain")
    func coding() throws {
        let full = StrandTrajectory(length: 40, trace: 0.2, entry: 5, at: TokenPosition(row: 1, offset: 2), next: 17, phase: 0.3, fit: 0.9,
                                    arc: 0.8, manner: 0.72)
        let decoded = try JSONDecoder().decode(StrandTrajectory.self, from: JSONEncoder().encode(full))
        #expect(decoded == full)
        let empty = String(decoding: try JSONEncoder().encode(StrandTrajectory(length: 0, trace: 0)), as: UTF8.self)
        #expect(!empty.contains("entry") && !empty.contains("manner"))
    }

    @Test("Spearman: ranks, ties at their average, nil when a side is constant")
    func spearman() {
        #expect(Stats.spearman([1, 2, 3, 4], [10, 20, 30, 45]) == 1)
        #expect(Stats.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1)
        #expect(Stats.ranks([3, 1, 3, 2]) == [3.5, 1, 3.5, 2])
        #expect(Stats.spearman([1, 2, 3], [5, 5, 5]) == nil)
    }
}
