import Foundation
import Testing

@testable import RaoLMCore

/// Builds traces by hand: each token lists (row, offset, rank, weight, matches) neighbours.
private func trace(_ index: Int, token: Int, prompt: Bool = false, _ neighbours: [(Int, Int, Int, Float, Bool)]) -> TokenTrace {
    let list = neighbours.map { row, offset, rank, weight, matches in
        Neighbour(
            rank: rank, entry: row * 1000 + offset, score: 0.95 - Float(rank) * 0.01, weight: weight,
            value: matches ? token : token + 1, matches: matches,
            key: TokenPosition(row: row, offset: offset - 1), cited: TokenPosition(row: row, offset: offset),
            sourceLoss: 0.01, sourceEntropy: 0.05)
    }
    return TokenTrace(
        index: index, token: token, text: "t\(token)", isPrompt: prompt, lmEntropy: 1, knnEntropy: 0.5,
        mixedEntropy: 0.7, sourceEntropy: 0.3, lmProb: 0.5, agreement: list.filter(\.matches).map(\.weight).reduce(0, +),
        mixedProb: 0.6, lambda: 0.5, neighbours: list)
}

private let partitions: [Int: PartitionRef] = [
    0: PartitionRef(row: 0, documentID: "raolm-t-aaa", documentName: "A", partitionIndex: 0, partitionURL: "raolm://t/a/p/0",
                    threadPartitionID: nil, textSHA256: "ha", tokenCount: 50, memorisedAtEpoch: 10),
    1: PartitionRef(row: 1, documentID: "raolm-t-bbb", documentName: "B", partitionIndex: 1, partitionURL: nil,
                    threadPartitionID: nil, textSHA256: "hb", tokenCount: 50),
]

@Suite("CitationSpans")
struct CitationSpansTests {
    @Test("a run advancing one offset at a time in one partition becomes a verbatim span")
    func verbatimChain() {
        var traces = [
            trace(4, token: 10, prompt: true, [(0, 5, 1, 0.8, true)]),
            trace(5, token: 11, [(0, 6, 1, 0.7, true), (1, 9, 2, 0.2, true)]),
            trace(6, token: 12, [(1, 20, 1, 0.5, true), (0, 7, 2, 0.4, true)]),
            trace(7, token: 13, [(0, 8, 3, 0.3, true)]),
            trace(8, token: 14, [(1, 3, 1, 0.9, false)]),
        ]
        let spans = CitationSpans.annotate(traces: &traces, partitions: partitions, sharedNgrams: [], threadID: "node")
        #expect(spans.count == 1)
        let span = spans[0]
        #expect(span.kind == .verbatim)
        #expect(span.tokenRange == TokenRange(start: 4, end: 8))
        #expect(span.promptTokens == 1)
        #expect(span.tokens == [10, 11, 12, 13])
        #expect(span.source.documentID == "raolm-t-aaa" && span.source.tokenOffset == 5)
        #expect(span.source.threadID == "node")
        #expect(span.ranks == [1, 1, 2, 3])
        #expect(span.distinctiveness == 1)
        // Inside the span the span's partition is cited first even when another partition weighs more.
        #expect(traces[2].citations.first?.row == 0)
        #expect(traces[2].spanIndex == 0)
        // The last token has no matching neighbour.
        #expect(traces[4].uncited)
        let summary = CitationSpans.summary(traces: traces, spans: spans)
        #expect(summary.generated == 4 && summary.verbatimCovered == 3 && summary.uncited == 1 && summary.verbatimSpans == 1)
    }

    @Test("rank above the threshold or a gap breaks the chain")
    func brokenChains() {
        var traces = [
            trace(1, token: 1, [(0, 1, 1, 0.9, true)]),
            trace(2, token: 2, [(0, 2, 4, 0.9, true)]),  // rank 4 > 3
            trace(3, token: 3, [(0, 3, 1, 0.9, true)]),
            trace(4, token: 4, [(0, 5, 1, 0.9, true)]),  // gap in offsets
        ]
        let spans = CitationSpans.annotate(traces: &traces, partitions: partitions, sharedNgrams: [], threadID: nil)
        #expect(spans.isEmpty)
        #expect(traces.allSatisfy { !$0.uncited })
    }

    @Test("overlapping chains become alternatives; boilerplate has zero distinctiveness")
    func alternatives() {
        var traces = (0..<4).map { i in
            trace(i + 1, token: 20 + i, [(0, 10 + i, 1, 0.6, true), (1, 30 + i, 2, 0.4, true)])
        }
        let shared: Set<UInt64> = [
            CitationSpans.ngramKey(20, 21, 22), CitationSpans.ngramKey(21, 22, 23),
        ]
        let spans = CitationSpans.annotate(traces: &traces, partitions: partitions, sharedNgrams: shared, threadID: nil)
        #expect(spans.count == 1)
        #expect(spans[0].row == 0)
        #expect(spans[0].alternatives.count == 1 && spans[0].alternatives[0].row == 1)
        #expect(spans[0].distinctiveness == 0)
    }

    @Test("prompt-only chains are marked promptVerbatim")
    func promptOnly() {
        var traces = (0..<3).map { i in trace(i + 1, token: i, prompt: true, [(0, i + 1, 1, 1, true)]) }
        traces.append(trace(4, token: 99, [(1, 40, 1, 1, false)]))
        let spans = CitationSpans.annotate(traces: &traces, partitions: partitions, sharedNgrams: [], threadID: nil)
        #expect(spans.count == 1 && spans[0].kind == .promptVerbatim)
        #expect(CitationSpans.summary(traces: traces, spans: spans).verbatimSpans == 0)
    }

    @Test("confidence is bounded and monotone in its factors")
    func confidence() {
        #expect(CitationMath.confidence(score: 1, support: 1, sourceLoss: 0) == 1)
        #expect(CitationMath.confidence(score: 0.4, support: 1, sourceLoss: 0) == 0)
        let weak = CitationMath.confidence(score: 0.8, support: 0.3, sourceLoss: 1)
        let strong = CitationMath.confidence(score: 0.9, support: 0.9, sourceLoss: 0.01)
        #expect(weak > 0 && weak < strong && strong <= 1)
        #expect(abs(CitationMath.entropy([0.5, 0.5]) - Float(log(2.0))) < 1e-6)
    }

    @Test("markers follow each verbatim span, one number per partition")
    func markers() {
        var traces = [
            trace(1, token: 1, [(0, 1, 1, 1, true)]),
            trace(2, token: 2, [(0, 2, 1, 1, true)]),
            trace(3, token: 3, [(0, 3, 1, 1, true)]),
            trace(4, token: 4, []),
        ]
        let spans = CitationSpans.annotate(traces: &traces, partitions: partitions, sharedNgrams: [], threadID: nil)
        let generation = CitedGeneration(
            generationID: "g", manifest: ManifestRef(runID: "r", epoch: 1, checkpointSHA256: "c", indexSHA256: "i",
                                                     corpusHash: "h", tokenizerSHA256: "t", ledgerSHA256: nil, threadID: nil),
            prompt: GenerationPrompt(text: "", tokens: [0], source: nil),
            params: GenerationParameters(tapLayer: 3, alpha: 0.5), tokens: [1, 2, 3, 4], text: "t1t2t3t4",
            stoppedOnEOS: false, partitions: Array(partitions.values), traces: traces, spans: spans,
            summary: CitationSpans.summary(traces: traces, spans: spans))
        let rendered = CitationMarkers.render(generation)
        #expect(rendered.text == "t1t2t3[[1]]t4")
        #expect(rendered.sources.count == 1 && rendered.sources[0].documentName == "A")

        // A span ending on a whitespace token gets its marker before the whitespace.
        var spaced = generation
        spaced.traces[2].text = " "
        #expect(CitationMarkers.render(spaced).text == "t1t2[[1]] t4")
    }
}
