//
//  CitationMarkers.swift
//  RaoLMCore
//
//  WHAT: Renders a cited generation with Sewn-style inline markers: `text[[1]]`, plus a
//        numbered source list, so RaoLM output drops into the family's attribution UI
//        (Sewn's Gita marker spans, MaryOS's C port).
//

import Foundation

public struct MarkedSource: Codable, Sendable, Equatable {
    public var number: Int
    public var row: Int
    public var documentID: String
    public var documentName: String
    public var partitionIndex: Int
    public var tokenStart: Int
    public var tokenEnd: Int
    public var partitionURL: String?
    public var verification: VerificationStatus?
}

public enum CitationMarkers {

    /// The generated text with `[[n]]` after each verbatim span, and the sources in
    /// order of first use. One number per partition.
    public static func render(_ generation: CitedGeneration) -> (text: String, sources: [MarkedSource]) {
        var numbers: [Int: Int] = [:]
        var sources: [MarkedSource] = []
        var text = ""
        for trace in generation.traces where !trace.isPrompt {
            text += trace.text
            guard let index = trace.spanIndex else { continue }
            let span = generation.spans[index]
            guard span.kind == .verbatim, trace.index == span.tokenRange.end - 1 else { continue }
            let number: Int
            if let existing = numbers[span.row] {
                number = existing
            } else {
                number = numbers.count + 1
                numbers[span.row] = number
                sources.append(MarkedSource(
                    number: number, row: span.row, documentID: span.source.documentID,
                    documentName: span.documentName, partitionIndex: span.source.partitionIndex,
                    tokenStart: span.source.tokenOffset, tokenEnd: span.source.tokenOffset + span.tokens.count,
                    partitionURL: span.source.partitionURL, verification: span.verification?.status))
            }
            // A span can end on a whitespace token (SmolLM2 gives the space before a number its
            // own token); the marker goes before that whitespace, where a reader expects it.
            var trailing = ""
            while let last = text.last, last.isWhitespace {
                trailing.insert(last, at: trailing.startIndex)
                text.removeLast()
            }
            text += "[[\(number)]]" + trailing
        }
        return (text, sources)
    }
}
