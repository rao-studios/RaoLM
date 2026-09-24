//
//  ContentHash.swift
//  RaoLMCore
//
//  WHAT: The hashing RaoLM's provenance chain rests on: text, files, and the corpus hash
//        that names exactly which Thread state a model was trained on.
//  PIN:  Every hash is lowercase hex SHA-256. The corpus hash is order-independent: it
//        sorts its entries, so the same partitions exported in any order hash the same.
//

import Crypto
import Foundation

public enum ContentHash {

    public static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// SHA-256 of the UTF-8 bytes of `text`, exactly as given (no canonicalization).
    public static func sha256Hex(_ text: String) -> String {
        sha256Hex(Data(text.utf8))
    }

    /// SHA-256 of a file's bytes, streamed in 1 MiB chunks.
    public static func sha256Hex(fileAt url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            let chunk = try handle.read(upToCount: 1 << 20) ?? Data()
            if chunk.isEmpty { break }
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// The canonical form document ids are derived from: NFC, `\r\n` → `\n`, trailing
    /// whitespace stripped from every line, and the whole trimmed.
    public static func canonical(_ text: String) -> String {
        let normalized = text.precomposedStringWithCanonicalMapping
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
        let lines = normalized.split(separator: "\n", omittingEmptySubsequences: false).map { line -> Substring in
            var end = line.endIndex
            while end > line.startIndex {
                let before = line.index(before: end)
                if line[before] == " " || line[before] == "\t" { end = before } else { break }
            }
            return line[line.startIndex..<end]
        }
        return lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public struct CorpusEntry: Sendable, Equatable {
        public var documentID: String
        public var partitionIndex: Int
        public var textSHA256: String

        public init(documentID: String, partitionIndex: Int, textSHA256: String) {
            self.documentID = documentID
            self.partitionIndex = partitionIndex
            self.textSHA256 = textSHA256
        }
    }

    /// SHA-256 over the lines `"<documentID>\t<partitionIndex>\t<textSHA256>\n"`, sorted
    /// by (documentID, partitionIndex). Recomputable by hand with `shasum -a 256`.
    public static func corpusHash(_ entries: [CorpusEntry]) -> String {
        let sorted = entries.sorted {
            ($0.documentID, $0.partitionIndex) < ($1.documentID, $1.partitionIndex)
        }
        var text = ""
        for entry in sorted {
            text += "\(entry.documentID)\t\(entry.partitionIndex)\t\(entry.textSHA256)\n"
        }
        return sha256Hex(text)
    }
}

/// Document ids, partition urls and the handles RaoLM passes to Thread.
public enum DocumentID {
    /// `raolm-<slug>-<first 24 hex of sha256(canonical text)>`. Content-addressed, never
    /// contains a `/` (Thread nests directories on `/`).
    public static func make(slug: String, canonicalText: String) -> String {
        "raolm-\(slug)-" + String(ContentHash.sha256Hex(canonicalText).prefix(24))
    }

    public static let pattern = #"^raolm-[a-z0-9-]+-[0-9a-f]{24}$"#

    public static func isValid(_ id: String) -> Bool {
        id.range(of: pattern, options: .regularExpression) != nil
    }

    /// The per-partition address RaoLM hands Thread at index time and reads back from
    /// ExportCorpus. Thread parses it with `URL(string:)` and returns `absoluteString`,
    /// which round-trips byte-for-byte for this alphabet.
    public static func partitionURL(slug: String, documentID: String, index: Int) -> String {
        "raolm://\(slug)/\(documentID)/p/\(index)"
    }

    /// The prefix every document id of a corpus shares; ExportCorpus filters on it.
    public static func prefix(slug: String) -> String {
        "raolm-\(slug)-"
    }

    /// Owner, group and slug handles: lowercase, `[a-z0-9._-]`, 2–64 characters.
    /// Thread stores a gRPC owner as given but lowercases partition owners, so mixed
    /// case would split one owner in two.
    public static func isValidHandle(_ handle: String) -> Bool {
        handle.range(of: #"^[a-z0-9][a-z0-9._-]{1,63}$"#, options: .regularExpression) != nil
    }
}
