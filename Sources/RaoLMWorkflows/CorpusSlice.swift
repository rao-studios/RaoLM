//
//  CorpusSlice.swift
//  RaoLMWorkflows
//
//  WHAT: A prompt taken as a token slice of a corpus partition, `DOCUMENT_ID:PARTITION:OFFSET:LENGTH`,
//        so the prompt is the exact tokenization the model trained on and carries its source.
//

import Foundation
import RaoLM

public struct CorpusSlice: Sendable, Equatable {
    public var documentID: String
    public var partitionIndex: Int
    public var offset: Int
    public var length: Int

    public init(documentID: String, partitionIndex: Int, offset: Int, length: Int) {
        self.documentID = documentID
        self.partitionIndex = partitionIndex
        self.offset = offset
        self.length = length
    }

    /// Throws a code-64 failure for a malformed spec.
    public static func parse(_ spec: String) throws -> CorpusSlice {
        let parts = spec.split(separator: ":").map(String.init)
        guard parts.count == 4, let partitionIndex = Int(parts[1]), let offset = Int(parts[2]), let length = Int(parts[3]), length > 0 else {
            throw RaoLMFailure("--prompt-from must be DOCUMENT_ID:PARTITION:OFFSET:LENGTH", code: 64)
        }
        return CorpusSlice(documentID: parts[0], partitionIndex: partitionIndex, offset: offset, length: length)
    }

    public var spec: String { "\(documentID):\(partitionIndex):\(offset):\(length)" }

    public func request(corpus: TokenizedCorpus, context: RunContext, params: GenerationParameters) throws -> GenerationRequest {
        guard let row = corpus.row(documentID: documentID, partitionIndex: partitionIndex) else {
            throw RaoLMFailure("document \(documentID) partition \(partitionIndex) is not in the run's corpus", code: 66)
        }
        let partition = corpus.partitions[row]
        guard offset >= 0, offset + length <= partition.tokens.count else {
            throw RaoLMFailure("offset \(offset)+\(length) exceeds the partition's \(partition.tokens.count) tokens", code: 64)
        }
        let tokens = partition.tokens[offset..<(offset + length)].map(Int.init)
        return GenerationRequest(
            promptTokens: tokens, promptText: context.tokenizer.decode(tokens),
            promptSource: SourceAddress(
                threadID: context.manifestRef.threadID, documentID: partition.documentID, partitionIndex: partitionIndex,
                tokenOffset: offset, partitionURL: partition.url, threadPartitionID: partition.threadPartitionID),
            params: params)
    }

    public func request(context: RunContext, params: GenerationParameters) throws -> GenerationRequest {
        try request(corpus: try context.tokenizedCorpus(), context: context, params: params)
    }
}
