//
//  CitationSpans.swift
//  RaoLMCore
//
//  WHAT: Turns per-token retrieval into citations: groups each token's matching
//        neighbours by partition, scores them, and finds verbatim spans — runs of emitted
//        tokens that walk a corpus partition one position at a time.
//  PIN:  Pure Swift over the traces, so the rules are tested without a model. A citation's
//        confidence is evidence of memorisation and retrieval support under the manifest's
//        exact weights. It is not an influence certificate: no counterfactual is run, and
//        `alternatives` lists the other sources a span also fits.
//

import Foundation

public enum CitationMath {
    /// Cosine 0.5 → 0, cosine 1.0 → 1.
    public static func simTerm(_ score: Float) -> Float {
        min(max((score - 0.5) / 0.5, 0), 1)
    }

    /// (simTerm(score) · support · exp(−sourceLoss))^(1/3). A geometric mean: any factor
    /// near zero pulls the confidence down.
    public static func confidence(score: Float, support: Float, sourceLoss: Float) -> Float {
        let memo = exp(-max(sourceLoss, 0))
        let product = simTerm(score) * min(max(support, 0), 1) * memo
        return product > 0 ? Float(pow(Double(product), 1.0 / 3.0)) : 0
    }

    /// −Σ p ln p over the positive entries.
    public static func entropy<S: Sequence>(_ probabilities: S) -> Float where S.Element == Float {
        var total: Double = 0
        for p in probabilities where p > 0 { total -= Double(p) * log(Double(p)) }
        return Float(total)
    }
}

public enum CitationSpans {

    public struct Settings: Sendable {
        public var rankThreshold: Int
        public var minSpanLength: Int
        public var maxCitations: Int
        public var maxAlternatives: Int

        public init(rankThreshold: Int = 3, minSpanLength: Int = 3, maxCitations: Int = 3, maxAlternatives: Int = 5) {
            self.rankThreshold = rankThreshold
            self.minSpanLength = minSpanLength
            self.maxCitations = maxCitations
            self.maxAlternatives = maxAlternatives
        }
    }

    /// Token 3-grams packed into one integer (token ids are below 2^16).
    public static func ngramKey(_ a: Int, _ b: Int, _ c: Int) -> UInt64 {
        (UInt64(UInt16(truncatingIfNeeded: a)) << 32) | (UInt64(UInt16(truncatingIfNeeded: b)) << 16)
            | UInt64(UInt16(truncatingIfNeeded: c))
    }

    public static func distinctiveness(tokens: [Int], sharedNgrams: Set<UInt64>) -> Float {
        guard tokens.count >= 3 else { return 0 }
        var shared = 0
        var total = 0
        for i in 0...(tokens.count - 3) {
            total += 1
            if sharedNgrams.contains(ngramKey(tokens[i], tokens[i + 1], tokens[i + 2])) { shared += 1 }
        }
        return 1 - Float(shared) / Float(total)
    }

    /// Every partition the token's matching neighbours come from, strongest first.
    public static func groupedCitations(
        _ trace: TokenTrace, partitions: [Int: PartitionRef], threadID: String?
    ) -> [Citation] {
        var groups: [Int: (weight: Float, best: Neighbour)] = [:]
        for neighbour in trace.neighbours where neighbour.matches {
            let row = neighbour.cited.row
            if let existing = groups[row] {
                let better = neighbour.score > existing.best.score
                    || (neighbour.score == existing.best.score && neighbour.rank < existing.best.rank)
                groups[row] = (existing.weight + neighbour.weight, better ? neighbour : existing.best)
            } else {
                groups[row] = (neighbour.weight, neighbour)
            }
        }
        let citations: [Citation] = groups.compactMap { row, group in
            guard let partition = partitions[row] else { return nil }
            return Citation(
                row: row, address: partition.address(offset: group.best.cited.offset, threadID: threadID),
                weight: group.weight, bestRank: group.best.rank, bestScore: group.best.score,
                sourceLoss: group.best.sourceLoss, memorisedAtEpoch: partition.memorisedAtEpoch,
                confidence: CitationMath.confidence(
                    score: group.best.score, support: group.weight, sourceLoss: group.best.sourceLoss))
        }
        return citations.sorted {
            if $0.weight != $1.weight { return $0.weight > $1.weight }
            if $0.bestScore != $1.bestScore { return $0.bestScore > $1.bestScore }
            return $0.row < $1.row
        }
    }

    private struct Chain {
        var start: Int
        var length: Int
        var position: TokenPosition
        var ranks: [Int]
        var weights: [Float]
        var totalWeight: Float { weights.reduce(0, +) }
    }

    /// Fills `citations`, `uncited`, `confidence` and `spanIndex` on every trace and
    /// returns the selected spans.
    ///
    /// - Parameters:
    ///   - traces: consecutive traces (trace `k` has index `traces[0].index + k`).
    ///   - partitions: the partition table rows the neighbours refer to, by row.
    ///   - sharedNgrams: token 3-grams occurring in at least two corpus documents.
    public static func annotate(
        traces: inout [TokenTrace], partitions: [Int: PartitionRef], sharedNgrams: Set<UInt64>,
        threadID: String?, settings: Settings = Settings()
    ) -> [CitedSpan] {
        // 1. Support citations per token.
        var allCitations: [[Citation]] = []
        for k in traces.indices {
            let grouped = groupedCitations(traces[k], partitions: partitions, threadID: threadID)
            allCitations.append(grouped)
            traces[k].citations = Array(grouped.prefix(settings.maxCitations))
            traces[k].uncited = grouped.isEmpty
            traces[k].confidence = grouped.first?.confidence
            traces[k].spanIndex = nil
        }

        // 2. Chain eligibility: matching, rank ≤ R, keyed by the cited (value) position.
        var eligible: [[TokenPosition: (rank: Int, weight: Float)]] = traces.map { trace in
            var map: [TokenPosition: (rank: Int, weight: Float)] = [:]
            for neighbour in trace.neighbours where neighbour.matches && neighbour.rank <= settings.rankThreshold {
                if let existing = map[neighbour.cited], existing.rank <= neighbour.rank { continue }
                map[neighbour.cited] = (neighbour.rank, neighbour.weight)
            }
            return map
        }

        // 3. Maximal chains: token k+t eligible for (row, offset + t).
        var chains: [Chain] = []
        for k in traces.indices {
            for position in eligible[k].keys.sorted() {
                let previous = TokenPosition(row: position.row, offset: position.offset - 1)
                if k > 0, eligible[k - 1][previous] != nil { continue }
                var chain = Chain(start: k, length: 0, position: position, ranks: [], weights: [])
                while k + chain.length < traces.count,
                    let info = eligible[k + chain.length][TokenPosition(row: position.row, offset: position.offset + chain.length)]
                {
                    chain.ranks.append(info.rank)
                    chain.weights.append(info.weight)
                    chain.length += 1
                }
                if chain.length >= settings.minSpanLength { chains.append(chain) }
            }
        }
        eligible.removeAll()

        // 4. Greedy selection of non-overlapping chains.
        func sortKey(_ chain: Chain) -> (Int, Float, String, Int, Int) {
            let partition = partitions[chain.position.row]
            return (chain.length, chain.totalWeight, partition?.documentID ?? "", partition?.partitionIndex ?? 0, chain.position.offset)
        }
        let ordered = chains.sorted { a, b in
            let ka = sortKey(a)
            let kb = sortKey(b)
            if ka.0 != kb.0 { return ka.0 > kb.0 }
            if ka.1 != kb.1 { return ka.1 > kb.1 }
            if ka.2 != kb.2 { return ka.2 < kb.2 }
            if ka.3 != kb.3 { return ka.3 < kb.3 }
            return ka.4 < kb.4
        }
        var selected: [Chain] = []
        var alternatives: [[SpanAlternative]] = []
        for chain in ordered {
            let range = chain.start..<(chain.start + chain.length)
            if let owner = selected.firstIndex(where: { range.overlaps($0.start..<($0.start + $0.length)) }) {
                if alternatives[owner].count < settings.maxAlternatives, let partition = partitions[chain.position.row] {
                    alternatives[owner].append(SpanAlternative(
                        row: chain.position.row,
                        source: partition.address(offset: chain.position.offset, threadID: threadID),
                        length: chain.length, cumulativeWeight: chain.totalWeight))
                }
                continue
            }
            selected.append(chain)
            alternatives.append([])
        }

        // 5. Materialize spans in token order.
        let order = selected.indices.sorted { selected[$0].start < selected[$1].start }
        var spans: [CitedSpan] = []
        for index in order {
            let chain = selected[index]
            guard let partition = partitions[chain.position.row] else { continue }
            let members = Array(traces[chain.start..<(chain.start + chain.length)])
            var confidences: [Float] = []
            for t in members.indices {
                let citation = allCitations[chain.start + t].first { $0.row == chain.position.row }
                confidences.append(citation?.confidence ?? 0)
            }
            let tokens = members.map(\.token)
            let promptTokens = members.filter(\.isPrompt).count
            let span = CitedSpan(
                kind: promptTokens == members.count ? .promptVerbatim : .verbatim,
                tokenRange: TokenRange(start: members[0].index, end: members[members.count - 1].index + 1),
                promptTokens: promptTokens, tokens: tokens, text: members.map(\.text).joined(),
                row: chain.position.row,
                source: partition.address(offset: chain.position.offset, threadID: threadID),
                documentName: partition.documentName, textSHA256: partition.textSHA256,
                ranks: chain.ranks, weights: chain.weights, confidence: confidences.min() ?? 0,
                meanConfidence: Stats.mean(confidences),
                distinctiveness: distinctiveness(tokens: tokens, sharedNgrams: sharedNgrams),
                alternatives: alternatives[index], verification: nil)
            let spanIndex = spans.count
            spans.append(span)
            for t in 0..<chain.length {
                let k = chain.start + t
                traces[k].spanIndex = spanIndex
                if let own = allCitations[k].first(where: { $0.row == chain.position.row }) {
                    var reordered = [own] + allCitations[k].filter { $0.row != chain.position.row }
                    reordered = Array(reordered.prefix(settings.maxCitations))
                    traces[k].citations = reordered
                    traces[k].confidence = own.confidence
                }
            }
        }
        return spans
    }

    public static func summary(traces: [TokenTrace], spans: [CitedSpan]) -> GenerationSummary {
        let generated = traces.filter { !$0.isPrompt }
        let verbatim = generated.filter { trace in
            guard let index = trace.spanIndex else { return false }
            return spans[index].kind == .verbatim
        }.count
        let supportOnly = generated.filter { $0.spanIndex == nil && !$0.uncited }.count
        let uncited = generated.filter(\.uncited).count
        let confidences = generated.compactMap(\.confidence)
        let verification = spans.filter { $0.kind == .verbatim }.compactMap(\.verification)
        return GenerationSummary(
            generated: generated.count, verbatimCovered: verbatim, supportOnly: supportOnly, uncited: uncited,
            meanConfidence: confidences.isEmpty ? nil : Stats.mean(confidences),
            verbatimSpans: spans.filter { $0.kind == .verbatim }.count,
            verifiedSpans: verification.isEmpty ? nil : verification.filter { $0.status == .verified }.count)
    }
}
