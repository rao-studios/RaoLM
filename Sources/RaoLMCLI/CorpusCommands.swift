//
//  CorpusCommands.swift
//  RaoLMCLI
//
//  WHAT: raolm corpus generate | show | ingest | pull
//

import ArgumentParser
import Foundation
import RaoLM
import RaoLMWorkflows

struct CorpusGroup: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "corpus",
        abstract: "Generate a synthetic corpus, deposit it into a Thread, and export it back.",
        subcommands: [Generate.self, Show.self, Ingest.self, Pull.self]
    )

    struct Generate: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Generate the deterministic synthetic Veldmar archive (every fact in exactly one partition).")

        @OptionGroup var global: GlobalOptions

        @Option(help: "Output directory (default: <data root>/corpora/<slug>).")
        var out: String?

        @Option(help: "Corpus slug; document ids are raolm-<slug>-<hash>.")
        var slug = "veldmar"

        @Option(help: "Number of documents.")
        var documents = 200

        @Option(help: "Generator seed.")
        var seed: UInt64 = 42

        @Option(help: "Maximum characters per partition.")
        var maxChars = 600

        @Flag(help: "Overwrite an existing corpus directory.")
        var force = false

        func run() async throws {
            try await guarded {
                let directory = out.map { URL(fileURLWithPath: $0) } ?? global.root.corpus(slug: slug)
                if CorpusStore.exists(at: directory), !force {
                    throw RaoLMFailure("a corpus already exists at \(directory.path)", hint: "pass --force to overwrite", code: 73)
                }
                let started = Date()
                let corpus = try SyntheticCorpus.generate(slug: slug, seed: seed, documentCount: documents, maxChars: maxChars)
                if FileManager.default.fileExists(atPath: directory.path) { try FileManager.default.removeItem(at: directory) }
                try CorpusStore.write(corpus, to: directory)
                print("generated \(corpus.manifest.documentCount) documents, \(corpus.manifest.partitionCount) partitions, \(corpus.manifest.factCount) facts in \(Format.duration(Date().timeIntervalSince(started)))")
                print("corpus hash \(corpus.manifest.corpusHash)")
                print(directory.path)
            }
        }
    }

    struct Show: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Summarize a corpus directory.")

        @Argument(help: "Corpus directory.")
        var directory: String

        func run() async throws {
            try await guarded {
                let corpus = try CorpusStore.load(URL(fileURLWithPath: directory))
                let m = corpus.manifest
                print("\(m.slug): \(m.documentCount) documents, \(m.partitionCount) partitions, \(m.factCount) facts (seed \(m.seed), generator v\(m.generatorVersion))")
                print("corpus hash \(m.corpusHash)")
                var kinds: [DocumentKind: Int] = [:]
                for document in corpus.documents { kinds[document.kind, default: 0] += 1 }
                print("kinds: " + DocumentKind.allCases.map { "\($0.rawValue) \(kinds[$0] ?? 0)" }.joined(separator: " · "))
                if let document = corpus.documents.first {
                    print("\nfirst document: \(document.name)  (\(document.id))")
                    for partition in document.partitions {
                        print("  [p\(partition.index)] \(partition.text)")
                    }
                    for fact in document.facts {
                        print("  fact \(fact.kind.rawValue): \(fact.prompt) ▸\(fact.answer)")
                    }
                }
            }
        }
    }

    struct Ingest: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Deposit a corpus into a running Thread (ThreadQuery.Index).")

        @OptionGroup var global: GlobalOptions

        @Argument(help: "Corpus directory (default: <data root>/corpora/veldmar).")
        var directory: String?

        @Option(help: "Thread owner id.")
        var owner = "raolm-demo"

        @Option(help: "Thread group id (default: raolm-<slug>).")
        var group: String?

        @Option(help: "Thread group label.")
        var label: String?

        @OptionGroup var thread: ThreadOptions

        @Option(help: "Documents per Index request.")
        var batch = 32

        @Flag(help: "Return as soon as the Index calls are accepted.")
        var noWait = false

        @Option(help: "Seconds to wait for indexing to finish.")
        var timeout: Double = 300

        func run() async throws {
            try await guarded {
                let url = directory.map { URL(fileURLWithPath: $0) } ?? global.root.corpus(slug: "veldmar")
                let corpus = try CorpusStore.load(url)
                let slug = corpus.manifest.slug
                let groupID = group ?? "raolm-\(slug)"
                guard DocumentID.isValidHandle(owner), DocumentID.isValidHandle(groupID) else {
                    throw RaoLMFailure("owner and group must be lowercase [a-z0-9._-], 2–64 characters", code: 64)
                }
                let client = ThreadCorpusClient(endpoint: thread.endpoint())
                _ = try await client.health()
                let report = try await client.index(
                    corpus.documents, slug: slug, owner: owner, group: groupID, groupLabel: label ?? "RaoLM \(slug) corpus",
                    batchSize: batch) { done, total in print("  indexed \(done)/\(total)") }
                print("sent \(report.documents) documents (\(report.partitions) partitions) in \(report.batches) batches, \(Format.duration(report.seconds))")
                if !noWait {
                    try await client.waitUntilIndexed(
                        expected: Set(corpus.documents.map(\.id)), owner: owner, group: groupID,
                        prefix: DocumentID.prefix(slug: slug), timeout: timeout)
                    print("all \(corpus.documents.count) documents are exported by the Thread")
                }
            }
        }
    }

    struct Pull: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Export a corpus from a Thread (ThreadLibrary.ExportCorpus) into a snapshot.")

        @OptionGroup var global: GlobalOptions

        @Option(help: "Thread owner id.")
        var owner = "raolm-demo"

        @Option(help: "Thread group id (default: raolm-<slug>).")
        var group: String?

        @Option(help: "Corpus slug (sets the document id prefix and checks partition urls).")
        var slug = "veldmar"

        @Option(help: "Document id prefix (default: raolm-<slug>-).")
        var prefix: String?

        @Option(help: "Snapshot directory (default: <data root>/snapshots/<hash12>).")
        var out: String?

        @Option(help: "A generated corpus directory to compare the export against.")
        var expectCorpus: String?

        @OptionGroup var thread: ThreadOptions

        func run() async throws {
            try await guarded {
                let nodeID = ThreadResolve.nodeID(explicit: thread.nodeId, root: global.root)
                let client = ThreadCorpusClient(endpoint: thread.endpoint(nodeID: nodeID))
                let snapshot = try await client.exportCorpus(
                    owner: owner, group: group ?? "raolm-\(slug)", prefix: prefix ?? DocumentID.prefix(slug: slug), slug: slug)
                guard snapshot.documentCount > 0 else {
                    throw RaoLMFailure("the Thread exported no documents for owner \(owner)", hint: "ingest first: raolm corpus ingest", code: 66)
                }
                if let expectCorpus {
                    let problems = snapshot.diff(against: try CorpusStore.load(URL(fileURLWithPath: expectCorpus)))
                    guard problems.isEmpty else {
                        throw RaoLMFailure("the export differs from the generated corpus:\n  " + problems.prefix(10).joined(separator: "\n  "), code: 70)
                    }
                    print("export matches the generated corpus byte for byte")
                }
                let directory = out.map { URL(fileURLWithPath: $0) } ?? global.root.snapshot(hash: snapshot.corpusHash)
                try snapshot.save(to: directory)
                print("\(snapshot.documentCount) documents, \(snapshot.partitionCount) partitions from Thread \(snapshot.threadID ?? "?")")
                print("corpus hash \(snapshot.corpusHash)")
                print(directory.appendingPathComponent(CorpusSnapshot.fileName).path)
            }
        }
    }
}
