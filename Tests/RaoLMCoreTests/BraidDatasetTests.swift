import Foundation
import Testing

@testable import RaoLMCore

@Suite("Braid dataset")
struct BraidDatasetTests {
    static let small = DatasetSpec(name: "test", seed: 7, perType: 12, paraphrase: 4, excerpt: 3, summary: 3, variant: 2, homonym: 2)

    @Test("every fact kind of the dataset is phrased in every voice, four ways in its own, with two paraphrases")
    func phrasings() {
        for type in DatasetEntityType.allCases {
            for kind in type.facts {
                let voices = DatasetVoices.phrasings[kind] ?? [:]
                #expect(voices[type.home]?.count == 4, "\(kind)")
                for voice in DatasetVoice.allCases where voice != type.home { #expect(voices[voice]?.count == 2, "\(kind) in \(voice)") }
                #expect(DatasetVoices.paraphrases[kind]?.count == 2, "\(kind)")
                for phrasing in voices.values.joined() {
                    #expect(phrasing.prefix.contains("{s}") || phrasing.prefix.contains("{S}"), "\(kind): \(phrasing.prefix)")
                }
            }
            for voice in DatasetVoice.allCases {
                #expect(DatasetVoices.intros[voice]?[type]?.isEmpty == false, "\(voice) \(type)")
                // Enough fillers that a document of three or four paragraphs never runs out.
                #expect(DatasetVoices.fillers(voice, type).count >= 14, "\(voice) \(type)")
            }
        }
    }

    @Test("no voice's sentence for a fact lies inside another voice's sentence for it")
    func noNesting() {
        var nested: [String] = []
        for refs in [SubjectRefs(s: "Zed", S: "Zed"), SubjectRefs(s: "the Zed service", S: "The Zed service")] {
            for (kind, voices) in DatasetVoices.phrasings {
                for (a, first) in voices {
                    for (b, second) in voices where a != b {
                        for x in first {
                            for y in second {
                                let inner = refs.fill(x.prefix) + " 7" + x.suffix
                                let outer = refs.fill(y.prefix) + " 7" + y.suffix
                                if outer.contains(inner) { nested.append("\(kind): \(a) '\(x.prefix)…\(x.suffix)' inside \(b) '\(y.prefix)…\(y.suffix)'") }
                            }
                        }
                    }
                }
            }
        }
        #expect(nested.isEmpty, "\(Set(nested).sorted())")
    }

    @Test("a small dataset generates, validates, and is the same from the same seed")
    func generate() throws {
        let dataset = try BraidDataset.generate(Self.small)
        #expect(BraidDataset.validate(dataset).isEmpty, "\(BraidDataset.validate(dataset).prefix(5))")
        #expect(dataset.names == ["ambient", "craft", "veil"])
        let again = try BraidDataset.generate(Self.small)
        #expect(again.manifest.datasetHash == dataset.manifest.datasetHash)
        var other = Self.small
        other.seed = 8
        #expect(try BraidDataset.generate(other).manifest.datasetHash != dataset.manifest.datasetHash)
        // Each node: 36 of its own, 4 + 3 + 3 + 2 from each of the two others, and its homonyms.
        for node in dataset.manifest.nodes {
            #expect(node.home == 36, "\(node.name)")
            #expect(node.crossed == 2 * 12, "\(node.name)")
            #expect(node.homonyms == (node.name == "craft" ? 2 : node.name == "veil" ? 2 : 4), "\(node.name)")
            #expect(node.documents == node.home + node.crossed + node.homonyms)
        }
        #expect(dataset.manifest.crosslinks["paraphrase"] == 24 && dataset.manifest.crosslinks["homonym"] == 8)
    }

    @Test("v2: every fact carries three questions with one stem and three negative questions, and no question occurs in any text")
    func questions() throws {
        #expect(BraidDataset.generatorVersion == 2)
        for kind in FactKind.allCases where DatasetVoices.phrasings[kind] != nil {
            #expect((DatasetVoices.questions[kind]?.count ?? 0) >= 3 && DatasetVoices.stems[kind] != nil, "\(kind)")
            for question in DatasetVoices.questions[kind] ?? [] { #expect(question.hasSuffix("?") && question.contains("{s}"), "\(question)") }
        }
        let dataset = try BraidDataset.generate(Self.small)
        let texts = dataset.names.flatMap { dataset.corpora[$0]!.documents.map { $0.partitions.map(\.text).joined(separator: "\n\n") } }
        var facts = 0
        for name in dataset.names {
            for fact in dataset.corpora[name]!.documents.flatMap(\.facts) {
                facts += 1
                let questions = try #require(fact.questions)
                #expect(questions.count >= 3 && Set(questions.map(\.stem)).count == 1)
                #expect(questions.allSatisfy { $0.text.contains(fact.subject) && $0.stem.contains(fact.subject) }, "\(fact.id)")
                // The rules rewriter reaches the stored stem from every stored question, unless the
                // question's template is shared with another kind (born: a researcher or a painter).
                let ambiguous = (DatasetVoices.questions[fact.kind] ?? []).contains { template in
                    DatasetVoices.questions.filter { $0.value.contains(template) }.count > 1
                }
                if !ambiguous { for question in questions { #expect(RuleRewriter.rewrite(question.text)?.stem == question.stem, "\(question.text)") } }
                #expect(fact.negativeQuestions?.count == questions.count && fact.negativeQuestions?.first?.text.contains(fact.subject) == false)
                for text in texts { #expect(questions.allSatisfy { !text.contains($0.text) }) }
            }
        }
        #expect(facts > 0)
        // A v1 fact, without questions, still decodes.
        let v1 = """
            {"id":"d#townFounded","kind":"townFounded","documentID":"d","partitionIndex":0,"subject":"Tillyburn","prompt":"p","answer":" 1128",
             "sentence":"p 1128.","sentenceStart":0,"contextStart":0,"answerStart":1,"answerEnd":6,"paraphrases":[],"negativePrompt":"n"}
            """
        let decoded = try JSONDecoder().decode(Fact.self, from: Data(v1.utf8))
        #expect(decoded.questions == nil && decoded.negativeQuestions == nil && decoded.subject == "Tillyburn")
    }

    @Test("links: shared facts agree except one in a variant; an excerpt quotes with an edit; a homonym shares only a name")
    func links() throws {
        let dataset = try BraidDataset.generate(Self.small)
        var documents: [String: CorpusDocument] = [:]
        for corpus in dataset.corpora.values { for document in corpus.documents { documents[document.id] = document } }
        for link in dataset.crosslinks {
            let source = try #require(documents[link.source.documentID])
            let target = try #require(documents[link.target.documentID])
            #expect(link.overlap.jaccard < 0.8 && link.overlap.longestCommonRun < DatasetOverlap.words(source.text).count)
            switch link.kind {
            case .paraphrase: #expect(link.facts.count == 3 && link.facts.allSatisfy { $0.agrees })
            case .summary: #expect((1...2).contains(link.facts.count) && link.facts.allSatisfy { $0.agrees })
            case .variant: #expect(link.facts.count == 3 && link.facts.filter { !$0.agrees }.count == 1)
            case .excerpt:
                #expect(link.facts.count == 2 && link.facts.allSatisfy { $0.agrees })
                let quoted = try #require(target.facts.first { $0.prompt.contains("\"") })
                #expect(DatasetVoices.hedges.contains { quoted.prompt.lowercased().contains($0) })
                let original = try #require(source.facts.first { $0.kind == quoted.kind })
                #expect(!target.text.contains(original.sentence))
            case .homonym:
                #expect(link.facts.isEmpty && source.subject == target.subject && link.sourceType != link.targetType)
            }
        }
    }

    @Test("feeding order: a link's target sits just after its source's place, and origins are simulated, never verified")
    func order() throws {
        let dataset = try BraidDataset.generate(Self.small)
        var rank: [String: Double] = [:]
        for rows in dataset.meta.values { for row in rows { rank[row.id] = row.rank } }
        for link in dataset.crosslinks {
            #expect((rank[link.target.documentID] ?? -1) > (rank[link.source.documentID] ?? 2))
            #expect((rank[link.target.documentID] ?? 0) - (rank[link.source.documentID] ?? 0) < 1e-3)
        }
        for rows in dataset.meta.values {
            #expect(rows.map(\.rank) == rows.map(\.rank).sorted())
            #expect(rows.allSatisfy { $0.synthetic })
        }
        // The first quarter of every node holds both sides of the links whose source lies there.
        for link in dataset.crosslinks where (rank[link.source.documentID] ?? 1) < 0.25 {
            let rows = try #require(dataset.meta[link.target.node])
            let position = try #require(rows.firstIndex { $0.id == link.target.documentID })
            #expect(Double(position) < Double(rows.count) * 0.25 + 3)
        }
    }

    @Test("written and read back, a dataset is the same")
    func files() throws {
        let dataset = try BraidDataset.generate(Self.small)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("raolm-dataset-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        try BraidDataset.write(dataset, to: directory)
        #expect(BraidDataset.exists(at: directory))
        let loaded = try BraidDataset.load(directory)
        #expect(loaded.manifest == dataset.manifest)
        #expect(loaded.crosslinks == dataset.crosslinks)
        for name in dataset.names {
            #expect(loaded.corpora[name] == dataset.corpora[name])
            #expect(loaded.meta[name] == dataset.meta[name])
        }
        let card = try String(contentsOf: directory.appendingPathComponent("README.md"), encoding: .utf8)
        #expect(card.contains(dataset.manifest.datasetHash) && card.contains("No document carries a Rao Verified record"))
    }

    @Test("the full-size dataset fits its name and value pools")
    func fullSize() throws {
        let dataset = try BraidDataset.generate(DatasetSpec())
        #expect(dataset.manifest.nodes.allSatisfy { $0.documents > 600 })
        #expect(BraidDataset.validate(dataset).isEmpty)
    }
}
