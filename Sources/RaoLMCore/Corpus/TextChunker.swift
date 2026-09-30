//
//  TextChunker.swift
//  RaoLMCore
//
//  WHAT: Splits a document into the partitions RaoLM deposits into Thread.
//  PIN:  Paragraph == partition. Thread's gRPC Index path stores `texts` exactly as given,
//        so the strings this returns are exactly the strings Thread stores, exports and
//        RaoLM tokenizes. A paragraph over `maxChars` splits at sentence ends (then hard);
//        one under `minChars` is merged into the next, so no partition is a sliver. Never
//        returns an empty string: Thread would drop a partition whose embedding came back
//        empty, and every later partition index would shift.
//

import Foundation

public enum TextChunker {

    public static func chunk(_ text: String, maxChars: Int = 600, minChars: Int = 120) -> [String] {
        precondition(maxChars > 0 && minChars >= 0)
        let normalized = text.replacingOccurrences(of: "\r\n", with: "\n")
        var paragraphs: [String] = []
        var current: [Substring] = []
        for line in normalized.split(separator: "\n", omittingEmptySubsequences: false) {
            if line.trimmingCharacters(in: .whitespaces).isEmpty {
                if !current.isEmpty {
                    paragraphs.append(current.joined(separator: "\n"))
                    current.removeAll()
                }
            } else {
                current.append(line)
            }
        }
        if !current.isEmpty { paragraphs.append(current.joined(separator: "\n")) }

        var pieces: [String] = []
        for paragraph in paragraphs {
            let trimmed = paragraph.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            if trimmed.count <= maxChars {
                pieces.append(trimmed)
            } else {
                pieces.append(contentsOf: splitLong(trimmed, maxChars: maxChars))
            }
        }

        // Merge slivers into the following piece (the last one into its predecessor).
        var merged: [String] = []
        var carry: String?
        for piece in pieces {
            let joined = carry.map { $0 + " " + piece } ?? piece
            if joined.count < minChars {
                carry = joined
            } else {
                merged.append(joined)
                carry = nil
            }
        }
        if let carry {
            if let last = merged.popLast() {
                merged.append(last + " " + carry)
            } else {
                merged.append(carry)
            }
        }
        return merged.filter { !$0.isEmpty }
    }

    /// Greedy sentence packing up to `maxChars`; a single sentence longer than that is
    /// hard-split at the last space before the limit.
    static func splitLong(_ paragraph: String, maxChars: Int) -> [String] {
        var sentences: [String] = []
        var start = paragraph.startIndex
        var index = paragraph.startIndex
        while index < paragraph.endIndex {
            let character = paragraph[index]
            let next = paragraph.index(after: index)
            if character == "." || character == "?" || character == "!" {
                if next == paragraph.endIndex || paragraph[next] == " " || paragraph[next] == "\n" {
                    sentences.append(String(paragraph[start..<next]).trimmingCharacters(in: .whitespaces))
                    start = next
                }
            }
            index = next
        }
        let tail = String(paragraph[start...]).trimmingCharacters(in: .whitespaces)
        if !tail.isEmpty { sentences.append(tail) }

        var chunks: [String] = []
        var current = ""
        for sentence in sentences {
            for part in hardSplit(sentence, maxChars: maxChars) {
                if current.isEmpty {
                    current = part
                } else if current.count + 1 + part.count <= maxChars {
                    current += " " + part
                } else {
                    chunks.append(current)
                    current = part
                }
            }
        }
        if !current.isEmpty { chunks.append(current) }
        return chunks
    }

    static func hardSplit(_ sentence: String, maxChars: Int) -> [String] {
        var remaining = Substring(sentence)
        var parts: [String] = []
        while remaining.count > maxChars {
            let limit = remaining.index(remaining.startIndex, offsetBy: maxChars)
            let cut = remaining[..<limit].lastIndex(of: " ") ?? limit
            let head = remaining[..<cut].trimmingCharacters(in: .whitespaces)
            if !head.isEmpty { parts.append(head) }
            remaining = remaining[cut...].drop(while: { $0 == " " })
        }
        let tail = remaining.trimmingCharacters(in: .whitespaces)
        if !tail.isEmpty { parts.append(tail) }
        return parts
    }
}
