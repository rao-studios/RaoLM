//
//  ThreadIndexRequests.swift
//  RaoLMThread
//
//  WHAT: Pure builders for the Index requests that deposit a corpus into a Thread.
//  PIN:  One ThreadIndexItem per document: caller-chosen content-addressed id, the display
//        name, partition texts exactly as generated, one tag (so Thread runs no entity
//        extraction), and a per-partition url — RaoLM's own citation address, which Thread
//        keeps and returns through ExportCorpus. `groupID` is mandatory: Thread files a
//        document under an owner only through a group, and ExportCorpus walks only the
//        owner's groups.
//

import Conduit
import Foundation
import RaoLMCore

public enum ThreadIndexRequests {

    public static func make(
        _ documents: [CorpusDocument], slug: String, owner: String, group: String, groupLabel: String, batchSize: Int
    ) -> [Thread_V1_ThreadIndexRequest] {
        precondition(batchSize > 0)
        var requests: [Thread_V1_ThreadIndexRequest] = []
        var start = 0
        while start < documents.count {
            let batch = documents[start..<min(start + batchSize, documents.count)]
            start += batchSize
            var request = Thread_V1_ThreadIndexRequest()
            request.ownerID = owner
            request.groupID = group
            request.groupLabel = groupLabel
            request.scope = ""
            request.items = batch.map { item(for: $0, slug: slug) }
            requests.append(request)
        }
        return requests
    }

    public static func item(for document: CorpusDocument, slug: String) -> Thread_V1_ThreadIndexItem {
        var item = Thread_V1_ThreadIndexItem()
        item.documentID = document.id
        item.name = document.name
        item.texts = document.partitions.map(\.text)
        item.tags = ["raolm:\(slug)"]
        item.mediaType = "text"
        item.partitions = document.partitions.map { partition in
            var input = Thread_V1_ThreadPartitionInput()
            input.url = partition.url
            return input
        }
        return item
    }
}
