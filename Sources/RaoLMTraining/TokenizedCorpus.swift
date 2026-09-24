//
//  TokenizedCorpus.swift
//  RaoLMTraining
//
//  WHAT: A corpus snapshot turned into tokens, with every token's source address.
//  OUT:  A partition table (row → document, partition index, tokens), and the packed
//        training stream `eos d0 eos d1 … eos` with a parallel row/offset for every token.
//  PIN:  Each partition is tokenized on its own and the id arrays are concatenated — no
//        joining whitespace, no BOS. Tokenizing a concatenation would let the BPE regex merge
//        across a partition boundary, and offsets would stop being reproducible from a single
//        partition's text, which is what the verifier re-tokenizes.
//

import Foundation
import MLX
import RaoLMCore
import RaoLMModel

public struct TokenizedPartition: Sendable {
    public let row: Int
    public let documentIndex: Int
    public let documentID: String
    public let documentName: String
    public let partitionIndex: Int
    public let tokens: [Int32]
    public let textSHA256: String
    public let url: String?
    public let threadPartitionID: String?
}

public struct TokenizedDocument: Sendable {
    public let index: Int
    public let id: String
    public let name: String
    public let rows: Range<Int>
}

public final class TokenizedCorpus: @unchecked Sendable {
    public let partitions: [TokenizedPartition]
    public let documents: [TokenizedDocument]
    public let eos: Int32
    public let slug: String?
    public let corpusHash: String
    public let threadID: String?
    public let excludedDocumentIDs: [String]

    /// eos d0 eos d1 … eos
    public let stream: [Int32]
    /// Partition row of each stream token (−1 for eos).
    public let streamRows: [Int32]
    /// Token offset inside its partition (−1 for eos).
    public let streamOffsets: [Int32]

    private let rowByAddress: [String: Int]

    public init(snapshot: CorpusSnapshot, tokenizer: RaoTokenizer, excluding: Set<String> = []) {
        let eos = Int32(tokenizer.eosTokenID)
        var partitions: [TokenizedPartition] = []
        var documents: [TokenizedDocument] = []
        var rowByAddress: [String: Int] = [:]
        var stream: [Int32] = [eos]
        var streamRows: [Int32] = [-1]
        var streamOffsets: [Int32] = [-1]

        for document in snapshot.documents where !excluding.contains(document.id) {
            let documentIndex = documents.count
            let firstRow = partitions.count
            for partition in document.partitions.sorted(by: { $0.index < $1.index }) {
                let row = partitions.count
                let tokens = tokenizer.encode(partition.text).map { Int32($0) }
                partitions.append(TokenizedPartition(
                    row: row, documentIndex: documentIndex, documentID: document.id, documentName: document.name,
                    partitionIndex: partition.index, tokens: tokens, textSHA256: partition.textSHA256,
                    url: partition.url, threadPartitionID: partition.threadPartitionID))
                rowByAddress["\(document.id)#\(partition.index)"] = row
                for (offset, token) in tokens.enumerated() {
                    stream.append(token)
                    streamRows.append(Int32(row))
                    streamOffsets.append(Int32(offset))
                }
            }
            documents.append(TokenizedDocument(index: documentIndex, id: document.id, name: document.name, rows: firstRow..<partitions.count))
            stream.append(eos)
            streamRows.append(-1)
            streamOffsets.append(-1)
        }

        self.partitions = partitions
        self.documents = documents
        self.eos = eos
        self.slug = snapshot.slug
        self.corpusHash = snapshot.corpusHash
        self.threadID = snapshot.threadID
        self.excludedDocumentIDs = snapshot.documents.map(\.id).filter { excluding.contains($0) }
        self.stream = stream
        self.streamRows = streamRows
        self.streamOffsets = streamOffsets
        self.rowByAddress = rowByAddress
    }

    public var tokenCount: Int { partitions.reduce(0) { $0 + $1.tokens.count } }

    public func row(documentID: String, partitionIndex: Int) -> Int? {
        rowByAddress["\(documentID)#\(partitionIndex)"]
    }

    /// `eos d eos` for one document, with each position's (row, offset); eos positions are −1.
    public func documentSequence(_ document: TokenizedDocument) -> (tokens: [Int32], rows: [Int32], offsets: [Int32]) {
        var tokens: [Int32] = [eos]
        var rows: [Int32] = [-1]
        var offsets: [Int32] = [-1]
        for row in document.rows {
            for (offset, token) in partitions[row].tokens.enumerated() {
                tokens.append(token)
                rows.append(Int32(row))
                offsets.append(Int32(offset))
            }
        }
        tokens.append(eos)
        rows.append(-1)
        offsets.append(-1)
        return (tokens, rows, offsets)
    }

    public func partitionRefs(memorisedAtEpoch: [Int: Int] = [:]) -> [PartitionRef] {
        partitions.map {
            PartitionRef(
                row: $0.row, documentID: $0.documentID, documentName: $0.documentName, partitionIndex: $0.partitionIndex,
                partitionURL: $0.url, threadPartitionID: $0.threadPartitionID, textSHA256: $0.textSHA256,
                tokenCount: $0.tokens.count, memorisedAtEpoch: memorisedAtEpoch[$0.row])
        }
    }

    /// Token 3-grams (packed) that occur in at least two documents.
    public func sharedNgrams() -> [UInt64] {
        var documentCount: [UInt64: Int] = [:]
        for document in documents {
            var tokens: [Int32] = []
            for row in document.rows { tokens.append(contentsOf: partitions[row].tokens) }
            guard tokens.count >= 3 else { continue }
            var seen = Set<UInt64>()
            for i in 0...(tokens.count - 3) {
                seen.insert(CitationSpans.ngramKey(Int(tokens[i]), Int(tokens[i + 1]), Int(tokens[i + 2])))
            }
            for key in seen { documentCount[key, default: 0] += 1 }
        }
        return documentCount.filter { $0.value >= 2 }.map(\.key).sorted()
    }
}

/// A fact located in token space: the partition row and token offsets of its context,
/// prompt end and answer.
public struct LocatedFact: Codable, Sendable {
    public var fact: Fact
    public var row: Int
    /// First token of the sentence before the fact's sentence (or of the fact's sentence).
    public var contextToken: Int
    /// First answer token (the prompt ends just before it).
    public var answerToken: Int
    /// One past the last answer token.
    public var answerEndToken: Int

    public var answerLength: Int { answerEndToken - answerToken }
}

public enum FactLocator {
    /// Maps each fact's byte offsets onto its partition's own tokenization. Facts whose
    /// answer does not start on a token boundary come back in `unaligned`.
    public static func locate(
        _ facts: [Fact], corpus: TokenizedCorpus, tokenizer: RaoTokenizer
    ) -> (located: [LocatedFact], unaligned: [String]) {
        var located: [LocatedFact] = []
        var unaligned: [String] = []
        var offsetsCache: [Int: [Int]] = [:]
        for fact in facts {
            guard let row = corpus.row(documentID: fact.documentID, partitionIndex: fact.partitionIndex) else {
                continue  // excluded document or not in this snapshot
            }
            let offsets: [Int]
            if let cached = offsetsCache[row] {
                offsets = cached
            } else {
                offsets = tokenizer.byteOffsets(corpus.partitions[row].tokens.map(Int.init))
                offsetsCache[row] = offsets
            }
            guard let answerToken = offsets.firstIndex(of: fact.answerStart),
                  let answerEnd = offsets.firstIndex(of: fact.answerEnd),
                  answerEnd > answerToken, answerToken > 0
            else {
                unaligned.append(fact.id)
                continue
            }
            let contextByte = max(fact.contextStart - 1, 0)
            let contextToken = fact.contextStart == 0 ? 0 : (offsets.lastIndex { $0 <= contextByte } ?? 0)
            located.append(LocatedFact(
                fact: fact, row: row, contextToken: min(contextToken, answerToken - 1),
                answerToken: answerToken, answerEndToken: answerEnd))
        }
        return (located, unaligned)
    }
}
