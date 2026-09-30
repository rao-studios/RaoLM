//
//  CitationVerifier.swift
//  RaoLMProvenance
//
//  WHAT: Re-reads every cited partition from the governed source (a live Thread, through a
//        CorpusReading) and checks each verbatim span token for token at its recorded offset.
//  PIN:  This is an integrity check of the chain Thread text ⇄ index ⇄ weights, not evidence
//        about the model: spans are built from corpus positions whose next token equals the
//        emitted token, so a span can only fail if the corpus or the pipeline changed.
//

import Foundation
import RaoLMCore
import RaoLMModel

public struct SpanCheck: Codable, Sendable, Equatable {
    public var span: Int
    public var kind: CitedSpan.Kind
    public var status: VerificationStatus
    public var documentID: String
    public var partitionIndex: Int
    public var tokenOffset: Int
    public var length: Int
    public var text: String
    public var detail: String?
}

public struct VerificationReport: Codable, Sendable, Equatable {
    public var generationID: String
    public var checkedAt: Date
    public var checks: [SpanCheck]

    public var verified: Int { checks.filter { $0.status == .verified }.count }
    public var failed: [SpanCheck] { checks.filter { $0.status != .verified } }
    public var allVerified: Bool { failed.isEmpty }
}

public enum CitationVerifier {

    public static func verify(
        _ generation: inout CitedGeneration, reader: CorpusReading, tokenizer: RaoTokenizer
    ) async throws -> VerificationReport {
        guard generation.manifest.tokenizerSHA256 == tokenizer.tokenizerSHA256 else {
            throw ProvenanceError.tokenizerMismatch(
                expected: generation.manifest.tokenizerSHA256, found: tokenizer.tokenizerSHA256)
        }
        var checks: [SpanCheck] = []
        for (i, span) in generation.spans.enumerated() {
            let (status, liveSHA, detail) = try await check(span, reader: reader, tokenizer: tokenizer)
            generation.spans[i].verification = SpanVerification(status: status, liveTextSHA256: liveSHA, detail: detail)
            checks.append(SpanCheck(
                span: i, kind: span.kind, status: status, documentID: span.source.documentID,
                partitionIndex: span.source.partitionIndex, tokenOffset: span.source.tokenOffset,
                length: span.tokens.count, text: span.text, detail: detail))
        }
        generation.summary = CitationSpans.summary(traces: generation.traces, spans: generation.spans)
        return VerificationReport(generationID: generation.generationID, checkedAt: Date(), checks: checks)
    }

    static func check(
        _ span: CitedSpan, reader: CorpusReading, tokenizer: RaoTokenizer
    ) async throws -> (VerificationStatus, String?, String?) {
        guard let live = try await reader.partitionText(
            documentID: span.source.documentID, partitionIndex: span.source.partitionIndex)
        else {
            return (.missing, nil, "partition not returned by the source")
        }
        let liveSHA = ContentHash.sha256Hex(live)
        guard liveSHA == span.textSHA256 else {
            return (.stale, liveSHA, "partition text changed since the index was built")
        }
        let tokens = tokenizer.encode(live)
        let start = span.source.tokenOffset
        let end = start + span.tokens.count
        guard start >= 0, start < tokens.count else {
            return (.mismatch, liveSHA, "offset \(start) is outside the partition's \(tokens.count) tokens")
        }
        let slice = Array(tokens[start..<min(end, tokens.count)])
        if slice == span.tokens {
            return (.verified, liveSHA, nil)
        }
        if tokenizer.decode(slice) == span.text {
            return (.tokenizerDrift, liveSHA, "same text at offset \(start) but different token ids")
        }
        return (.mismatch, liveSHA, "tokens at \(start)..<\(end) differ from the span")
    }
}
