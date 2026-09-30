//
//  DatasetCommands.swift
//  RaoLMCLI
//
//  WHAT: raolm dataset generate | verify | show — the braid datasets: three Threads' corpora in
//        three voices, with entities retold across Threads (BraidDataset).
//  IN:   Datasets live in the datasets root (`DatasetsRoot`): $RAOLM_DATASETS_DIR, else the T9
//        work area's datasets/ (/Volumes/T9/rao/projects/raolm/datasets) when that drive is
//        mounted. With neither, pass --out or a path.
//

import ArgumentParser
import Foundation
import RaoLM
import RaoLMWorkflows

struct DatasetGroup: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "dataset",
        abstract: "Generate, check and inspect braid datasets: three Threads' corpora, with entities retold across them.",
        discussion: """
            Ambient's world (towns, researchers, festivals) as reading notes and conversations; Craft's (libraries, services, \
            incidents) as session logs, release notes and postmortems; Veil's (artworks, artists, collections) as catalogue \
            entries, attribution reports and wall text. Some entities cross to another Thread in its own words: paraphrased, \
            summarised, quoted with an edit, or with one fact changed; some names are reused by a different entity. Nothing \
            crosses verbatim. Datasets live in $\(DatasetsRoot.environmentKey), else \(DatasetsRoot.workArea)/datasets.
            """,
        subcommands: [Generate.self, Verify.self, Show.self]
    )

    struct Generate: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Generate a braid dataset from a seed and write it to the datasets root.")

        @Option(help: "The dataset's name (its directory in the datasets root).")
        var name = DatasetSpec().name

        @Option(help: "Generator seed.")
        var seed: UInt64 = 42

        @Option(help: "Documents per entity type of each Thread's own world (three types per Thread).")
        var perType = DatasetSpec().perType

        @Option(help: "Per ordered pair of Threads: entities retold with every fact.")
        var paraphrase = DatasetSpec().paraphrase

        @Option(help: "Per ordered pair: entities quoted with an edit.")
        var excerpt = DatasetSpec().excerpt

        @Option(help: "Per ordered pair: entities summarised in one or two facts.")
        var summary = DatasetSpec().summary

        @Option(help: "Per ordered pair: entities retold with one fact changed.")
        var variant = DatasetSpec().variant

        @Option(help: "Per ordered pair that can share names: a different entity with the same name.")
        var homonym = DatasetSpec().homonym

        @Option(help: "Write here instead of <datasets root>/<name>.")
        var out: String?

        @Flag(help: "Overwrite an existing dataset.")
        var force = false

        func run() async throws {
            try await guarded {
                let directory = try out.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) } ?? DatasetsRoot.resolve(name)
                if BraidDataset.exists(at: directory), !force {
                    throw RaoLMFailure("a dataset already exists at \(directory.path)", hint: "pass --force to overwrite", code: 73)
                }
                let spec = DatasetSpec(name: name, seed: seed, perType: perType, paraphrase: paraphrase, excerpt: excerpt, summary: summary,
                                       variant: variant, homonym: homonym)
                let started = Date()
                Console.error("generating \(name) from seed \(seed)")
                let dataset = try BraidDataset.generate(spec)
                Console.error("validating \(dataset.manifest.nodes.reduce(0) { $0 + $1.documents }) documents")
                let problems = BraidDataset.validate(dataset)
                guard problems.isEmpty else { throw DatasetError.invalid(problems) }
                if FileManager.default.fileExists(atPath: directory.path) { try FileManager.default.removeItem(at: directory) }
                Console.error("writing \(directory.path)")
                try BraidDataset.write(dataset, to: directory)
                print(DatasetTables.summary(dataset.manifest))
                print(String(format: "\ngenerated and validated in %@ · %@", Format.duration(Date().timeIntervalSince(started)), directory.path))
            }
        }
    }

    struct Verify: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Re-read a dataset and check it: every fact where it says, no text across Threads, every link, every hash.")

        @Argument(help: "A dataset name in the datasets root, or a path.")
        var dataset: String

        func run() async throws {
            try await guarded {
                let directory = try DatasetsRoot.resolve(dataset)
                let loaded = try BraidDataset.load(directory)
                var problems = BraidDataset.validate(loaded)
                for summary in loaded.manifest.nodes {
                    guard let corpus = loaded.corpora[summary.name] else { continue }
                    let entries = corpus.documents.flatMap { document in
                        document.partitions.map { ContentHash.CorpusEntry(documentID: document.id, partitionIndex: $0.index, textSHA256: $0.textSHA256) }
                    }
                    if ContentHash.corpusHash(entries) != summary.corpusHash { problems.append("\(summary.name): the corpus hash does not match its manifest") }
                }
                if try BraidDataset.hash(loaded) != loaded.manifest.datasetHash { problems.append("the dataset hash does not match its manifest") }
                print(DatasetTables.summary(loaded.manifest))
                if problems.isEmpty {
                    print("\nverified: every fact at its offsets, no text on two Threads, every link and hash as recorded")
                } else {
                    throw DatasetError.invalid(problems)
                }
            }
        }
    }

    struct Show: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Show a dataset's counts and one example of each kind of link, both sides.")

        @Argument(help: "A dataset name in the datasets root, or a path.")
        var dataset: String

        func run() async throws {
            try await guarded {
                let loaded = try BraidDataset.load(try DatasetsRoot.resolve(dataset))
                print(DatasetTables.summary(loaded.manifest))
                var documents: [String: CorpusDocument] = [:]
                for corpus in loaded.corpora.values { for document in corpus.documents { documents[document.id] = document } }
                for kind in CrossKind.allCases {
                    guard let link = loaded.crosslinks.first(where: { $0.kind == kind }),
                          let source = documents[link.source.documentID], let target = documents[link.target.documentID] else { continue }
                    print("\n── \(kind.rawValue) · \(link.source.node) → \(link.target.node) · \(link.subject) · \(Format.f(link.overlap.jaccard, 2)) of words shared, longest run \(link.overlap.longestCommonRun)")
                    print("  \(link.source.node): \(Format.clip(source.partitions[0].text, 260))")
                    print("  \(link.target.node): \(Format.clip(target.partitions[0].text, 260))")
                    for fact in link.facts {
                        print("  \(fact.kind.rawValue): \(fact.sourceAnswer.trimmingCharacters(in: .whitespaces)) → \(fact.targetAnswer.trimmingCharacters(in: .whitespaces))\(fact.agrees ? "" : "  (differs)")")
                    }
                }
            }
        }
    }
}

enum DatasetTables {
    static func summary(_ manifest: DatasetManifest) -> String {
        var lines = ["\(manifest.spec.name) · seed \(manifest.spec.seed) · \(manifest.spec.perType) per type · hash \(Format.short(manifest.datasetHash))"]
        lines.append(Format.table(
            ["node", "documents", "own", "crossed in", "homonyms", "partitions", "facts", "words", "simulated origins"],
            manifest.nodes.map { node in
                [node.name, String(node.documents), String(node.home), String(node.crossed), String(node.homonyms), String(node.partitions),
                 String(node.facts), String(node.words),
                 node.origins.sorted { $0.key < $1.key }.map { "\($0.key) \($0.value)" }.joined(separator: " · ")]
            }))
        lines.append("links: " + CrossKind.allCases.map { "\($0.rawValue) \(manifest.crosslinks[$0.rawValue] ?? 0)" }.joined(separator: " · "))
        return lines.joined(separator: "\n")
    }
}
