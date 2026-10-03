import Foundation
import Testing

@testable import RaoLMCore

@Suite("Braid dataset")
struct BraidDatasetTests {
    static let small = DatasetSpec(name: "test", seed: 7, perType: 12, paraphrase: 4, excerpt: 3, summary: 3, variant: 2, homonym: 2)

    @Test("every fact kind of the founding worlds is phrased in every voice, four ways in its own, with two paraphrases")
    func phrasings() {
        for type in DatasetEntityType.allCases where !type.world.subject {
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
        #expect(BraidDataset.generatorVersion == 3)
        for kind in FactKind.allCases where DatasetVoices.cores[kind] != nil {
            #expect((DatasetVoices.questions[kind]?.count ?? 0) >= 3 && DatasetVoices.stems[kind] != nil, "\(kind)")
            #expect(DatasetVoices.paraphrases[kind]?.count == 2, "\(kind)")
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

    @Test("the full-size dataset fits its name and value pools, at three nodes and at twenty-four, and the datasets on disk regenerate exactly")
    func fullSize() throws {
        let dataset = try BraidDataset.generate(DatasetSpec(name: "braid-cross-v2"))
        #expect(dataset.manifest.nodes.allSatisfy { $0.documents > 600 })
        #expect(BraidDataset.validate(dataset).isEmpty)
        // The bench datasets' shape: 24 personas, two peers each, 30 per type.
        let wide = try BraidDataset.generate(DatasetSpec(name: "braid-n24", perType: 30, paraphrase: 6, excerpt: 5, summary: 6, variant: 2,
                                                         homonym: 2, nodes: 24, peers: 2))
        #expect(wide.names.count == 24 && wide.manifest.nodes.allSatisfy { $0.home == 90 && $0.crossed == 38 })
        #expect(BraidDataset.validate(wide).isEmpty, "\(BraidDataset.validate(wide).prefix(5))")
        let three = try BraidDataset.generate(DatasetSpec(name: "braid-n3", perType: 30, paraphrase: 6, excerpt: 5, summary: 6, variant: 2,
                                                          homonym: 2, nodes: 3, peers: 2))
        // Every braid trained so far was fed these: the subject worlds must not move a single draw.
        #expect(dataset.manifest.datasetHash == "11b3aacd51172911233f6be933b933f3d44c4556f66ff92d0553cf3950e612d2", "braid-cross-v2")
        #expect(three.manifest.datasetHash == "6d506f58d8625a3d294a4f5c181c173054ccf9aca6285dbfae1627fd178d2adb", "braid-n3")
        #expect(wide.manifest.datasetHash == "096b51c0b05c1841c0dd8802287759406f74e55624dbcbea8b49033f6fca93ba", "braid-n24")
    }

    @Test("the subject worlds: Ambient writes about writing, Craft about coding and mathematics, Veil about biology, each entity invented and its own")
    func subjects() throws {
        let roster = DatasetPersonas.subjects
        #expect(roster.map(\.name) == ["ambient", "craft", "veil"] && roster.map(\.world) == [.writing, .coding, .biology])
        #expect(roster.map(\.label) == ["Writing", "Coding & math", "Biology"] && roster.allSatisfy { $0.legacy == nil })
        for persona in roster {
            #expect(persona.frames.count >= 5 && persona.frames.filter { !$0.tail.isEmpty }.count >= 2, "\(persona.name)")
            #expect(persona.kinds.count == 2 && persona.kinds.allSatisfy { (persona.headers[$0]?.count ?? 0) >= 2 })
            #expect(persona.world.types.allSatisfy { (persona.intros[$0]?.count ?? 0) >= 2 } && persona.world.types.count == 3)
            #expect(persona.foreignIntros.count >= 2 && persona.foreignIntros.allSatisfy { $0.contains("{what}") })
            #expect(persona.closings.count >= 3 && persona.excerptLeads.allSatisfy { $0.hasSuffix("\"") })
            for type in DatasetEntityType.allCases { #expect(DatasetVoices.fillers(persona, type).count >= 14) }
        }
        #expect(try DatasetPersonas.take(3, worlds: DatasetWorld.subjects).map(\.name) == ["ambient", "craft", "veil"])
        #expect(try DatasetPersonas.take(1, worlds: [.biology]).map(\.name) == ["veil"])
        #expect(throws: DatasetError.self) { _ = try DatasetPersonas.take(4, worlds: DatasetWorld.subjects) }
        #expect(throws: DatasetError.self) { _ = try DatasetPersonas.take(3, worlds: [.reading]) }

        let spec = DatasetSpec(name: "subjects", seed: 9, perType: 6, paraphrase: 2, excerpt: 1, summary: 1, variant: 1, homonym: 1,
                               nodes: 3, peers: 2, worlds: DatasetWorld.subjects)
        let dataset = try BraidDataset.generate(spec)
        #expect(BraidDataset.validate(dataset).isEmpty, "\(BraidDataset.validate(dataset).prefix(5))")
        #expect(dataset.names == ["ambient", "craft", "veil"])
        #expect(dataset.manifest.nodes.map(\.world) == ["writing", "coding", "biology"])
        #expect(dataset.manifest.nodes.map(\.label) == ["Writing", "Coding & math", "Biology"])
        for node in dataset.manifest.nodes {
            #expect(node.home == 18 && node.crossed == 2 * 5 && node.homonyms == 2, "\(node.name): \(node.home)/\(node.crossed)/\(node.homonyms)")
        }
        #expect(dataset.manifest.crosslinks["homonym"] == 6)
        let worldOf = Dictionary(uniqueKeysWithValues: dataset.manifest.nodes.map { ($0.name, $0.world ?? "") })
        #expect(dataset.crosslinks.allSatisfy { worldOf[$0.source.node] != worldOf[$0.target.node] })
        let texts = dataset.names.flatMap { dataset.corpora[$0]!.documents.map { $0.partitions.map(\.text).joined(separator: "\n\n") } }
        for persona in roster {
            let rows = try #require(dataset.meta[persona.name])
            let home = Set(rows.filter { $0.home && $0.crosslink == nil }.map(\.id))
            #expect(rows.filter { home.contains($0.id) }.allSatisfy { $0.entityType.world == persona.world }, "\(persona.name) holds only its subject")
            var coreZero = 0, facts = 0
            for document in dataset.corpora[persona.name]!.documents where home.contains(document.id) {
                let refs = BraidDataset.refs(rows.first { $0.id == document.id }!.entityType, document.subject)
                for fact in document.facts {
                    facts += 1
                    let filled = (DatasetVoices.cores[fact.kind] ?? []).map { refs.fill($0.text).lowercased() }
                    #expect(filled.contains { fact.prompt.lowercased().hasSuffix($0) }, "\(persona.name): \(fact.prompt)")
                    if fact.prompt.lowercased().hasSuffix(filled.first ?? "#") { coreZero += 1 }
                    for question in fact.questions ?? [] {
                        #expect(RuleRewriter.rewrite(question.text)?.stem == question.stem, "\(question.text)")
                        for text in texts { #expect(!text.contains(question.text)) }
                    }
                }
            }
            #expect(facts > 0 && coreZero * 3 >= facts, "\(persona.name): core 0 on \(coreZero) of \(facts)")
        }
        // A spec with worlds survives a manifest round trip; one without stays as it was.
        let encoded = try JSONEncoder().encode(spec)
        #expect(try JSONDecoder().decode(DatasetSpec.self, from: encoded) == spec)
        #expect(!String(decoding: try JSONEncoder().encode(Self.small), as: UTF8.self).contains("worlds"))
    }

    // MARK: - v3: N personas

    @Test("v3: 24 personas, eight per world in round-robin order, the founding three first, each with its own strings")
    func personas() throws {
        let all = DatasetPersonas.all
        #expect(all.count == 24 && Set(all.map(\.name)).count == 24)
        #expect(all.prefix(3).map(\.name) == ["ambient", "craft", "veil"] && all.prefix(3).allSatisfy { $0.legacy != nil })
        for (i, persona) in all.enumerated() {
            #expect(persona.world == DatasetWorld.allCases[i % 3], "\(persona.name)")
            #expect(persona.name.range(of: #"^[a-z][a-z0-9-]{1,23}$"#, options: .regularExpression) != nil, "\(persona.name)")
            guard persona.legacy == nil else { continue }
            #expect(persona.frames.count >= 5 && persona.frames.filter { !$0.tail.isEmpty }.count >= 2, "\(persona.name)")
            #expect(persona.frames.allSatisfy { !$0.lead.isEmpty || !$0.tail.isEmpty })
            #expect(persona.kinds.count == 2 && persona.kinds.allSatisfy { (persona.headers[$0]?.count ?? 0) >= 2 }, "\(persona.name)")
            #expect(persona.world.types.allSatisfy { (persona.intros[$0]?.count ?? 0) >= 2 }, "\(persona.name)")
            #expect(persona.foreignIntros.count >= 2 && persona.foreignIntros.allSatisfy { $0.contains("{what}") })
            #expect(persona.closings.count >= 3 && persona.excerptLeads.count >= 2 && persona.excerptLeads.allSatisfy { $0.hasSuffix("\"") })
            for type in DatasetEntityType.allCases { #expect(DatasetVoices.fillers(persona, type).count >= 14, "\(persona.name) \(type)") }
            for text in persona.fillers + persona.closings + persona.intros.values.joined() + persona.foreignIntros {
                #expect(text.contains("{s}") || text.contains("{S}"), "\(persona.name): \(text)")
            }
        }
        // No two personas share a string of their own; no lead ends another; no two tails are alike.
        let composed = all.filter { $0.legacy == nil } + DatasetPersonas.subjects
        let strings = composed.flatMap { p in
            p.fillers + p.closings + p.excerptLeads + p.foreignIntros + p.intros.values.joined() + p.headers.values.joined()
        }
        #expect(Set(strings).count == strings.count, "\(Dictionary(grouping: strings, by: { $0 }).filter { $0.value.count > 1 }.keys.sorted())")
        let leads = composed.flatMap { $0.frames.map(\.lead) }.filter { !$0.isEmpty }
        for a in leads { for b in leads where a != b { #expect(!b.hasSuffix(a), "'\(a)' ends '\(b)'") } }
        let tails = composed.flatMap { $0.frames.map(\.tail) }.filter { !$0.isEmpty }
        #expect(Set(tails).count == tails.count && Set(leads).count == leads.count)
        #expect(throws: DatasetError.self) { _ = try DatasetPersonas.take(25) }
    }

    @Test("v3: three cores per fact kind; the first ends the kind's stem; none holds a paraphrase or a question")
    func cores() {
        #expect(DatasetEntityType.allCases.allSatisfy { type in type.facts.allSatisfy { DatasetVoices.cores[$0] != nil } })
        for kind in FactKind.allCases where DatasetVoices.cores[kind] != nil {
            let cores = DatasetVoices.cores[kind] ?? []
            #expect(cores.count == 3, "\(kind)")
            let stem = (DatasetVoices.stems[kind] ?? "").replacingOccurrences(of: "{S}", with: "{s}").lowercased()
            #expect(stem.hasSuffix(cores.first?.text.lowercased() ?? "#"), "\(kind): '\(stem)' and '\(cores.first?.text ?? "")'")
            for core in cores {
                #expect(core.text.contains("{s}"), "\(kind)")
                for template in (DatasetVoices.paraphrases[kind] ?? []) + (DatasetVoices.questions[kind] ?? []) {
                    #expect(!core.text.lowercased().contains(template.replacingOccurrences(of: "{S}", with: "{s}").lowercased()), "\(kind)")
                }
            }
        }
    }

    @Test("v3: no persona's sentence for a fact lies inside another persona's sentence for it")
    func noNestingAcrossPersonas() {
        var nested: [String] = []
        for refs in [SubjectRefs(s: "Zed", S: "Zed"), SubjectRefs(s: "the Zed service", S: "The Zed service")] {
            for kind in FactKind.allCases where DatasetVoices.cores[kind] != nil {
                let subject = DatasetEntityType.allCases.contains { $0.world.subject && $0.facts.contains(kind) }
                let sentences = (subject ? DatasetPersonas.subjects : DatasetPersonas.all).map { persona in
                    (persona.name, Set(DatasetVoices.phrasings(kind, persona).map { refs.fill($0.prefix) + " 7" + $0.suffix }))
                }
                for (a, inner) in sentences {
                    for (b, outer) in sentences where a != b {
                        for x in inner { for y in outer where y.contains(x) { nested.append("\(kind): \(a) '\(x)' inside \(b) '\(y)'") } }
                    }
                }
            }
        }
        if !nested.isEmpty { try? Set(nested).sorted().joined(separator: "\n").write(toFile: NSTemporaryDirectory() + "raolm-nested.txt", atomically: true, encoding: .utf8) }
        #expect(nested.isEmpty, "\(nested.count) nested; listed in raolm-nested.txt in the temporary directory")
    }

    @Test("v3: six nodes generate and validate; every link joins peers; a composed persona writes in its own frames")
    func generateSix() throws {
        let spec = DatasetSpec(name: "six", seed: 5, perType: 6, paraphrase: 2, excerpt: 1, summary: 1, variant: 1, homonym: 1, nodes: 6, peers: 3)
        let dataset = try BraidDataset.generate(spec)
        #expect(BraidDataset.validate(dataset).isEmpty, "\(BraidDataset.validate(dataset).prefix(5))")
        #expect(dataset.names == DatasetPersonas.all.prefix(6).map(\.name))
        let index = Dictionary(uniqueKeysWithValues: dataset.names.enumerated().map { ($0.element, $0.offset) })
        for link in dataset.crosslinks { #expect(spec.isPeer(index[link.source.node]!, index[link.target.node]!), "\(link.id)") }
        for node in dataset.manifest.nodes {
            #expect(node.home == 18 && node.crossed <= 3 * 5 && node.persona == node.name && node.world != nil, "\(node.name)")
        }
        for persona in DatasetPersonas.all.prefix(6) where persona.legacy == nil {
            let rows = try #require(dataset.meta[persona.name])
            let documents = try #require(dataset.corpora[persona.name]?.documents)
            let home = Set(rows.filter { $0.home && $0.crosslink == nil }.map(\.id))
            var coreZero = 0, facts = 0
            for document in documents where home.contains(document.id) {
                for fact in document.facts {
                    facts += 1
                    let framed = persona.frames.contains { frame in
                        (!frame.lead.isEmpty && fact.prompt.hasPrefix(frame.lead)) || (!frame.tail.isEmpty && fact.sentence.hasSuffix(frame.tail + "."))
                    }
                    #expect(framed, "\(persona.name): \(fact.sentence)")
                    let refs = SubjectRefs(s: document.subject, S: document.subject)
                    if fact.prompt.lowercased().hasSuffix(refs.fill(DatasetVoices.cores[fact.kind]![0].text).lowercased()) { coreZero += 1 }
                }
            }
            #expect(facts > 0 && coreZero * 3 >= facts, "\(persona.name): core 0 on \(coreZero) of \(facts)")
        }
    }

    @Test("v3: a manifest written before v3 decodes as three nodes with two peers")
    func oldManifestDecodes() throws {
        let v2 = #"{"name":"braid-cross-v2","seed":42,"perType":180,"paraphrase":24,"excerpt":20,"summary":24,"variant":8,"homonym":8}"#
        let spec = try JSONDecoder().decode(DatasetSpec.self, from: Data(v2.utf8))
        #expect(spec.nodes == 3 && spec.peers == 2 && spec.worlds == nil && spec == DatasetSpec(name: "braid-cross-v2", seed: 42))
        #expect((0..<3).allSatisfy { i in (0..<3).allSatisfy { j in spec.isPeer(i, j) == (i != j) } })
    }
}
