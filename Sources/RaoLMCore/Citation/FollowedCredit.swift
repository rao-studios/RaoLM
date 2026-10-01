//
//  FollowedCredit.swift
//  RaoLMCore
//
//  WHAT: The owner's rule for a fact several Threads hold (2026-10-01): "the trajectory is the
//        decision maker if it's the same fact". The Thread whose document the completion follows
//        takes the fact, whether it told it first or retold it; origin alone earns nothing, and
//        a copy that is not followed earns nothing. On a short prompt no trajectory has formed by
//        the answer, so the decision is made after generation, sentence by sentence, from which
//        Thread's chains the completion advanced.
//  PIN:  Each trace's `strands[t].trajectory` is Thread t's chain state at the position that
//        predicted the trace's token (`next` is the token it expected), so every generated token,
//        the last included, says whether a chain of t advanced on it. Chain advances are the only
//        evidence, read with a margin: the follower's chain must have advanced on at least
//        `margin` more tokens than any rival's, and on more than half the sentence's content
//        tokens. Two Threads holding the same document on separately trained nodes differ by a
//        token or two in what their chains expect; without the margin one token would hand the
//        whole fact to one copy. With it they tie, and a copy earns no more than a split: the
//        anti-gaming property. Where no Thread is followed but two or more advanced on the
//        sentence alike, they split their pooled credit evenly: between holders of the same
//        text the gate's preference is noise (two copies' lift differs with their seeds), and the
//        trajectory, which is the decision maker, has not decided. Only credit moves; the mixture,
//        the shares and the commons' credit are untouched.
//

import Foundation

public struct FollowDecision: Codable, Sendable, Equatable {
    /// The generated tokens of one sentence, half-open over trace indices.
    public var window: TokenRange
    /// The Thread whose document the sentence followed; nil on a tie or when no chain advanced.
    public var followed: String?
    /// On a tie, the Threads that advanced alike and split the sentence's shared credit evenly.
    public var tied: [String] = []
    /// The document it followed (the one most of the follower's advancing chains sit in).
    public var documentID: String?
    /// How many of the sentence's tokens each Thread's chain advanced on.
    public var advances: [String: Int]
    /// Each Thread's matching neighbour weight over the sentence (reported; never decides).
    public var matches: [String: Float] = [:]
}

public enum FollowedCredit {
    /// Re-assigns the Threads' credit of every content token that two or more Threads hold to
    /// the Thread whose chains the token's sentence followed. `partitions` are the braid's rows
    /// (global, as `TokenTrace.neighbours` name them); `rowOffsets` map each strand's node-local
    /// trajectory rows onto them. Returns one decision per sentence.
    /// How many more tokens the follower's chain must have advanced on than any rival's.
    public static let margin = 2

    @discardableResult
    public static func apply(
        _ traces: inout [TokenTrace], rowOffsets: [String: Int], partitions: [Int: PartitionRef], commons: String?, margin: Int = margin
    ) -> [FollowDecision] {
        var decisions: [FollowDecision] = []
        for window in sentences(traces) {
            let decision = decide(traces, window: window, rowOffsets: rowOffsets, partitions: partitions, commons: commons, margin: margin)
            decisions.append(decision)
            for i in window.start ..< window.end {
                guard traces[i].role == .content, var strands = traces[i].strands else { continue }
                let holders = strands.indices.filter { strands[$0].strand != commons && (strands[$0].credit ?? 0) > 0 }
                guard holders.count >= 2 else { continue }
                let pooled = holders.reduce(Float(0)) { $0 + (strands[$1].credit ?? 0) }
                if let follower = decision.followed, let winner = strands.firstIndex(where: { $0.strand == follower }) {
                    for h in holders { strands[h].credit = 0 }
                    strands[winner].credit = pooled
                    traces[i].followed = follower
                } else if decision.tied.count >= 2 {
                    // Undecided between Threads that advanced alike: an even split of what they hold together.
                    let sharers = holders.filter { decision.tied.contains(strands[$0].strand) }
                    guard sharers.count >= 2 else { continue }
                    let among = sharers.reduce(Float(0)) { $0 + (strands[$1].credit ?? 0) }
                    for h in sharers { strands[h].credit = among / Float(sharers.count) }
                } else {
                    continue
                }
                traces[i].strands = strands
            }
        }
        return decisions
    }

    /// The generated traces split into sentences: a window closes after a token whose text ends a
    /// sentence and before one that opens with whitespace, as `AnswerStop` reads them.
    static func sentences(_ traces: [TokenTrace]) -> [TokenRange] {
        var windows: [TokenRange] = []
        var start: Int?
        var text = ""
        for i in traces.indices {
            guard !traces[i].isPrompt else { continue }
            if let s = start, AnswerStop.sentenceEnded(previous: text, next: traces[i].text) {
                windows.append(TokenRange(start: s, end: i))
                start = nil
                text = ""
            }
            if start == nil { start = i }
            text += traces[i].text
        }
        if let s = start { windows.append(TokenRange(start: s, end: traces.count)) }
        return windows
    }

    /// Who the sentence followed: the Thread whose chain advanced on at least `margin` more tokens
    /// than any other's and on more than half the sentence's content tokens; nobody otherwise.
    /// `matches` (matching neighbour weight by Thread) is reported, never used to decide.
    static func decide(
        _ traces: [TokenTrace], window: TokenRange, rowOffsets: [String: Int], partitions: [Int: PartitionRef], commons: String?,
        margin: Int = margin
    ) -> FollowDecision {
        var advances: [String: Int] = [:]
        var matches: [String: Float] = [:]
        var documents: [String: [String: Int]] = [:]
        for i in window.start ..< window.end {
            let trace = traces[i]
            for share in trace.strands ?? [] where share.strand != commons {
                guard let trajectory = share.trajectory, trajectory.next == trace.token else { continue }
                advances[share.strand, default: 0] += 1
                if let at = trajectory.at, let offset = rowOffsets[share.strand], let partition = partitions[offset + at.row] {
                    documents[share.strand, default: [:]][partition.documentID, default: 0] += 1
                }
            }
            for neighbour in trace.neighbours where neighbour.matches {
                if let owner = strand(ofRow: neighbour.cited.row, rowOffsets: rowOffsets) {
                    matches[owner, default: 0] += neighbour.weight
                }
            }
        }
        let content = (window.start ..< window.end).filter { traces[$0].role == .content }.count
        let ranked = advances.filter { $0.value > 0 }.sorted { $0.value > $1.value }
        var followed: String?
        var tied: [String] = []
        if let best = ranked.first, best.value * 2 > content {
            if (ranked.dropFirst().first?.value ?? 0) + margin <= best.value {
                followed = best.key
            } else {
                // Within the margin of the best: they advanced alike.
                tied = ranked.filter { $0.value + margin > best.value }.map(\.key).sorted()
            }
        }
        let document = followed.flatMap { documents[$0]?.max { $0.value < $1.value }?.key }
        return FollowDecision(window: window, followed: followed, tied: tied, documentID: document, advances: advances, matches: matches)
    }

    /// The strand a global row belongs to: the one whose rows start last at or before it.
    static func strand(ofRow row: Int, rowOffsets: [String: Int]) -> String? {
        rowOffsets.filter { $0.value <= row }.max { $0.value < $1.value }?.key
    }
}
