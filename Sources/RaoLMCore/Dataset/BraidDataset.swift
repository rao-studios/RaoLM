//
//  BraidDataset.swift
//  RaoLMCore
//
//  WHAT: A dataset for a braid of three Threads, generated from a seed. Each Thread has its own
//        world in its own voice: Ambient's towns, researchers and festivals, told as reading
//        notes and conversations; Craft's libraries, services and incidents, told as session
//        logs, release notes, postmortems and reviews; Veil's artworks, artists and
//        collections, told as catalogue entries, attribution reports and wall text. Then some
//        entities cross to another Thread, retold in that Thread's voice: paraphrased,
//        summarised, quoted with an edit, or retold with one fact changed; and some names are
//        reused by a different entity on another Thread (homonyms).
//  OUT:  Per Thread a corpus in RaoLM's corpus format, documents in feeding order; a meta row
//        per document; one row per link, with the facts both sides state and how much text
//        they share; a manifest naming it all by hash.
//  PIN:  No text crosses verbatim: no partition, and no fact sentence, appears on two Threads.
//        Within a Thread every fact is stated once, at the offsets it records (FactValidator).
//        A link's target sits just after its source in feeding order, so feeding the first
//        part of every Thread keeps both sides of the links it holds.
//

import Foundation

public enum BraidDataset {
    public static let generatorName = "BraidDataset"
    public static let generatorVersion = 1
    static let maxChars = 600
    static let minChars = 120
    static let minParagraph = 220
    static let maxParagraph = 560

    public struct Generated: Sendable {
        public var manifest: DatasetManifest
        /// Per node, its corpus with documents in feeding order.
        public var corpora: [String: GeneratedCorpus]
        public var meta: [String: [DatasetDocumentMeta]]
        public var crosslinks: [DatasetCrosslink]
        /// Subjects that exist nowhere, for negative prompts (validation only; not written).
        var negatives: [String] = []

        public var names: [String] { manifest.names }
    }

    /// The pairs of entity types that can share a name across Threads: a homonym gives the
    /// source's name to a new entity of the target's type.
    static let homonymPairs: [(source: DatasetEntityType, target: DatasetEntityType)] = [
        (.town, .library), (.library, .town), (.researcher, .artist), (.artist, .researcher),
    ]

    // MARK: - Generate

    public static func generate(_ spec: DatasetSpec) throws -> Generated {
        var builder = Builder(spec: spec)
        return try builder.build()
    }

    // MARK: - Validate

    /// Everything wrong with a dataset, or an empty array.
    public static func validate(_ dataset: Generated) -> [String] {
        var problems: [String] = []
        var partitionOwner: [String: String] = [:]
        var documentOwner: [String: String] = [:]
        var documents: [String: CorpusDocument] = [:]
        for name in dataset.names {
            guard let corpus = dataset.corpora[name] else {
                problems.append("\(name): no corpus")
                continue
            }
            problems += FactValidator.validate(documents: corpus.documents, negativeSubjects: dataset.negatives,
                                               maxChars: maxChars, minChars: minChars).map { "\(name): \($0)" }
            for document in corpus.documents {
                documents[document.id] = document
                if let other = documentOwner[document.id] { problems.append("\(document.id) is on \(other) and \(name)") }
                documentOwner[document.id] = name
                for partition in document.partitions {
                    if let other = partitionOwner[partition.textSHA256], other != name {
                        problems.append("\(document.id) partition \(partition.index): the same text is on \(other)")
                    }
                    partitionOwner[partition.textSHA256] = name
                }
            }
            if dataset.meta[name]?.map(\.id) != corpus.documents.map(\.id) {
                problems.append("\(name): meta rows do not follow the corpus order")
            }
        }
        for link in dataset.crosslinks {
            guard let source = documents[link.source.documentID], let target = documents[link.target.documentID] else {
                problems.append("\(link.id): a document is missing")
                continue
            }
            if documentOwner[source.id] != link.source.node || documentOwner[target.id] != link.target.node {
                problems.append("\(link.id): a document is on the wrong node")
            }
            if link.source.node == link.target.node { problems.append("\(link.id): both sides are on \(link.source.node)") }
            // No fact sentence of the source appears verbatim in the target.
            for fact in source.facts where target.text.contains(fact.sentence) {
                problems.append("\(link.id): \(fact.id)'s sentence appears verbatim on \(link.target.node)")
            }
            let sourceFacts = Dictionary(uniqueKeysWithValues: source.facts.map { ($0.id, $0) })
            let targetFacts = Dictionary(uniqueKeysWithValues: target.facts.map { ($0.id, $0) })
            for shared in link.facts {
                guard let a = sourceFacts[shared.sourceFact], let b = targetFacts[shared.targetFact] else {
                    problems.append("\(link.id): fact \(shared.kind.rawValue) is missing on one side")
                    continue
                }
                if a.answer != shared.sourceAnswer || b.answer != shared.targetAnswer {
                    problems.append("\(link.id): fact \(shared.kind.rawValue) records the wrong answers")
                }
            }
            switch link.kind {
            case .homonym:
                if !link.facts.isEmpty || link.sourceType == link.targetType || source.subject != target.subject {
                    problems.append("\(link.id): a homonym shares its name and nothing else")
                }
            case .variant:
                if link.facts.filter({ !$0.agrees }).count != 1 { problems.append("\(link.id): a variant changes exactly one fact") }
            case .paraphrase, .excerpt, .summary:
                if link.facts.isEmpty || !link.facts.allSatisfy(\.agrees) { problems.append("\(link.id): the facts must agree") }
            }
            if link.overlap.jaccard >= 0.8 { problems.append("\(link.id): the two texts share too many words (\(link.overlap.jaccard))") }
        }
        return problems
    }

    // MARK: - Files

    /// <dir>/manifest.json, crosslinks.jsonl, README.md, and per node nodes/<name>/ (a corpus
    /// directory plus meta.jsonl).
    public static func write(_ dataset: Generated, to directory: URL) throws {
        let manager = FileManager.default
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        for name in dataset.names {
            guard let corpus = dataset.corpora[name] else { continue }
            let node = nodeDirectory(directory, name)
            try CorpusStore.write(corpus, to: node)
            let meta = try JSONLWriter(url: node.appendingPathComponent("meta.jsonl"), truncate: true)
            for row in dataset.meta[name] ?? [] { try meta.append(row) }
            meta.close()
        }
        let links = try JSONLWriter(url: directory.appendingPathComponent("crosslinks.jsonl"), truncate: true)
        for link in dataset.crosslinks { try links.append(link) }
        links.close()
        try Data(card(dataset).utf8).write(to: directory.appendingPathComponent("README.md"), options: .atomic)
        // The manifest last: a directory with one is complete.
        try JSONCoding.write(dataset.manifest, to: directory.appendingPathComponent(DatasetManifest.fileName))
    }

    public static func load(_ directory: URL) throws -> Generated {
        let manifestURL = directory.appendingPathComponent(DatasetManifest.fileName)
        guard FileManager.default.fileExists(atPath: manifestURL.path) else { throw DatasetError.notADataset(directory.path) }
        let manifest = try JSONCoding.read(DatasetManifest.self, from: manifestURL)
        var corpora: [String: GeneratedCorpus] = [:]
        var meta: [String: [DatasetDocumentMeta]] = [:]
        for name in manifest.names {
            let node = nodeDirectory(directory, name)
            corpora[name] = try CorpusStore.load(node)
            meta[name] = try JSONCoding.readLines(DatasetDocumentMeta.self, from: node.appendingPathComponent("meta.jsonl"))
        }
        let links = try JSONCoding.readLines(DatasetCrosslink.self, from: directory.appendingPathComponent("crosslinks.jsonl"))
        return Generated(manifest: manifest, corpora: corpora, meta: meta, crosslinks: links)
    }

    /// SHA-256 over every node's corpus hash and every link, in order.
    public static func hash(corpusHashes: [(name: String, hash: String)], crosslinks: [DatasetCrosslink]) throws -> String {
        let encoder = JSONCoding.lineEncoder()
        var lines = corpusHashes.map { "\($0.name)\t\($0.hash)" }
        for link in crosslinks { lines.append(String(decoding: try encoder.encode(link), as: UTF8.self)) }
        return ContentHash.sha256Hex(lines.joined(separator: "\n"))
    }

    /// The hash of a dataset as it is now (its corpora recomputed from their documents).
    public static func hash(_ dataset: Generated) throws -> String {
        let hashes = dataset.names.map { name -> (name: String, hash: String) in
            let documents = dataset.corpora[name]?.documents ?? []
            let entries = documents.flatMap { document in
                document.partitions.map { ContentHash.CorpusEntry(documentID: document.id, partitionIndex: $0.index, textSHA256: ContentHash.sha256Hex($0.text)) }
            }
            return (name, ContentHash.corpusHash(entries))
        }
        return try hash(corpusHashes: hashes, crosslinks: dataset.crosslinks)
    }

    public static func exists(at directory: URL) -> Bool {
        FileManager.default.fileExists(atPath: directory.appendingPathComponent(DatasetManifest.fileName).path)
    }

    public static func nodeDirectory(_ directory: URL, _ name: String) -> URL {
        directory.appendingPathComponent("nodes", isDirectory: true).appendingPathComponent(name, isDirectory: true)
    }

    /// The dataset card written beside the data.
    static func card(_ dataset: Generated) -> String {
        let m = dataset.manifest
        var lines = [
            "# \(m.spec.name)",
            "",
            "A braid dataset generated by RaoLM's `\(generatorName)` v\(generatorVersion) from seed \(m.spec.seed)",
            "(`raolm dataset generate --name \(m.spec.name) --seed \(m.spec.seed) --per-type \(m.spec.perType)`).",
            "Every document is synthetic. Dataset hash `\(m.datasetHash)`.",
            "",
            "Three Threads, each with its own world in its own voice. Ambient: towns, researchers and",
            "festivals, as reading notes, conversations and digests. Craft: libraries, services and",
            "incidents, as session logs, release notes, postmortems and reviews. Veil: artworks, artists",
            "and collections, as catalogue entries, attribution reports and wall text. Some entities cross",
            "to another Thread in that Thread's words, never verbatim.",
            "",
            "| Node | Documents | Own | Crossed in | Homonyms | Partitions | Facts | Words |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for node in m.nodes {
            lines.append("| \(node.name) | \(node.documents) | \(node.home) | \(node.crossed) | \(node.homonyms) | \(node.partitions) | \(node.facts) | \(node.words) |")
        }
        lines += ["", "| Link | Count | What the target holds |", "|---|---|---|"]
        let meaning: [CrossKind: String] = [
            .paraphrase: "every fact of the entity, in its own voice",
            .excerpt: "one sentence quoted with an edit, and one more fact in its own voice",
            .summary: "one or two facts, briefly",
            .variant: "every fact, one of them with a different value",
            .homonym: "a different entity with the same name",
        ]
        for kind in CrossKind.allCases {
            lines.append("| \(kind.rawValue) | \(m.crosslinks[kind.rawValue] ?? 0) | \(meaning[kind] ?? "") |")
        }
        lines += [
            "",
            "## Files",
            "",
            "- `nodes/<node>/`: a RaoLM corpus (`manifest.json`, `facts.jsonl`, `documents/<id>.json` and `.txt`), documents",
            "  in feeding order, and `meta.jsonl`: per document its kind, entity, whether it is the node's own, the link it is the",
            "  target of, its simulated origin and its rank in feeding order.",
            "- `crosslinks.jsonl`: per link, its kind, both documents, the facts both state (fact ids and answers) and how many",
            "  words the two texts share.",
            "- `manifest.json`: the spec, per node counts and corpus hashes, and the dataset hash.",
            "",
            "`simulatedOrigin` uses Rao Verified's origins (read, written, spoken, imported, generated, system) to stand for",
            "where such a document would come from. No document carries a Rao Verified record: nothing here was read, written",
            "or said by a person.",
            "",
            "Feed it to a braid: `raolm braid demo --offline --fresh --dataset \(m.spec.name) --batch 40`.",
        ]
        return lines.joined(separator: "\n") + "\n"
    }
}

// MARK: - The builder

private enum Piece {
    case sentence(String)
    case fact(kind: FactKind, prompt: String, value: String, suffix: String, paraphrases: [String], negative: String)

    var text: String {
        switch self {
        case .sentence(let text): return text
        case .fact(_, let prompt, let value, let suffix, _, _): return prompt + " " + value + suffix
        }
    }
}

private struct Entity {
    let type: DatasetEntityType
    let subject: String
    let refs: SubjectRefs
    let negative: SubjectRefs
    var answers: [FactKind: String]
}

private struct Draft {
    let kind: DocumentKind
    let name: String
    let subject: String
    let paragraphs: [[Piece]]
    let fillers: [String]
    /// The phrasing each fact used, so another Thread can quote it.
    let phrasings: [FactKind: DatasetPhrasing]
}

private struct Placed {
    var document: CorpusDocument
    var meta: DatasetDocumentMeta
    var entity: Entity
    var phrasings: [FactKind: DatasetPhrasing]
}

private struct Builder {
    let spec: DatasetSpec
    var forge: DatasetForge
    var values: DatasetValues
    var rng: SplitMix64
    var pick: SplitMix64
    var negatives: [String] = []
    var placed: [String: [Placed]] = [:]
    var present: [String: Set<String>] = [:]
    var crosslinks: [DatasetCrosslink] = []

    init(spec: DatasetSpec) {
        self.spec = spec
        forge = DatasetForge(rng: SplitMix64.derived(seed: spec.seed, stream: 21))
        rng = SplitMix64.derived(seed: spec.seed, stream: 22)
        values = DatasetValues(seed: spec.seed)
        pick = SplitMix64.derived(seed: spec.seed, stream: 24)
    }

    mutating func build() throws -> BraidDataset.Generated {
        let voices = DatasetVoice.allCases
        // Each Thread's own world, its three entity types interleaved.
        for voice in voices {
            let types = DatasetEntityType.types(of: voice)
            var own: [Placed] = []
            for index in 0..<(spec.perType * types.count) {
                let type = types[index % types.count]
                let entity = try makeEntity(type)
                let kinds = DatasetVoices.kinds(voice, type)
                let kind = kinds[(index / types.count) % kinds.count]
                var item = try placeDocument(voice: voice, kind: kind, entity: entity) { builder in
                    try builder.homeDraft(voice: voice, kind: kind, entity: entity)
                }
                item.meta.rank = (Double(index) + 0.5) / Double(spec.perType * types.count)
                own.append(item)
            }
            placed[voice.rawValue] = own
            present[voice.rawValue] = Set(own.map(\.entity.subject))
        }
        // Crossings, in a fixed order of pairs and kinds.
        var sequence = 0
        for source in voices {
            for target in voices where target != source {
                for kind in CrossKind.allCases where kind != .homonym {
                    for chosen in choose(from: source, into: target, count: spec.count(kind)) {
                        sequence += 1
                        try cross(kind, from: chosen, source: source, target: target, sequence: sequence)
                    }
                }
                for pair in BraidDataset.homonymPairs where pair.source.home == source && pair.target.home == target {
                    for chosen in choose(from: source, into: target, count: spec.homonym, type: pair.source) {
                        sequence += 1
                        try homonym(from: chosen, source: source, target: target, type: pair.target, sequence: sequence)
                    }
                }
            }
        }
        return try finish()
    }

    // MARK: Entities

    mutating func name(_ type: DatasetEntityType) throws -> String {
        switch type {
        case .town: return try forge.town()
        case .researcher, .artist: return try forge.person()
        case .festival: return try forge.festival()
        case .library: return try forge.library()
        case .service: return try forge.service()
        case .incident: return try forge.incidentCode()
        case .artwork: return try forge.artwork()
        case .collection: return try forge.collection()
        }
    }

    static func refs(_ type: DatasetEntityType, _ name: String) -> SubjectRefs {
        switch type {
        case .festival, .collection: return SubjectRefs(s: "the \(name)", S: "The \(name)")
        case .service: return SubjectRefs(s: "the \(name) service", S: "The \(name) service")
        case .incident: return SubjectRefs(s: "incident \(name)", S: "Incident \(name)")
        default: return SubjectRefs(s: name, S: name)
        }
    }

    mutating func value(_ kind: FactKind) throws -> String {
        if let taken = try values.take(kind) { return taken }
        switch kind {
        case .researcherBook: return try forge.book()
        case .artistStudio: return try forge.town()
        default: return try forge.person()
        }
    }

    mutating func makeEntity(_ type: DatasetEntityType, name forced: String? = nil) throws -> Entity {
        let subject = try forced ?? name(type)
        let negative = try name(type)
        negatives.append(negative)
        var answers: [FactKind: String] = [:]
        for kind in type.facts { answers[kind] = try value(kind) }
        return Entity(type: type, subject: subject, refs: Self.refs(type, subject), negative: Self.refs(type, negative), answers: answers)
    }

    // MARK: Crossing

    /// Entities of `source`'s own world not yet on `target`, spread over its types.
    mutating func choose(from source: DatasetVoice, into target: DatasetVoice, count: Int, type: DatasetEntityType? = nil) -> [Placed] {
        let taken = present[target.rawValue] ?? []
        let candidates = (placed[source.rawValue] ?? []).filter { item in
            item.meta.home && !taken.contains(item.entity.subject) && (type == nil || item.entity.type == type)
        }
        let chosen = Array(pick.shuffled(candidates).prefix(max(0, count)))
        present[target.rawValue, default: []].formUnion(chosen.map(\.entity.subject))
        return chosen
    }

    mutating func cross(_ kind: CrossKind, from origin: Placed, source: DatasetVoice, target: DatasetVoice, sequence: Int) throws {
        var entity = origin.entity
        var changed: FactKind?
        if kind == .variant {
            let fact = rng.pick(entity.type.facts)
            entity.answers[fact] = try value(fact)
            changed = fact
        }
        let kinds = DatasetVoices.kinds(target, entity.type)
        let documentKind = kinds[sequence % kinds.count]
        var item = try placeDocument(voice: target, kind: documentKind, entity: entity) { builder in
            try builder.crossDraft(kind, voice: target, kind: documentKind, entity: entity, origin: origin)
        }
        let id = String(format: "x%04d", sequence)
        item.meta.home = false
        item.meta.crosslink = id
        item.meta.rank = origin.meta.rank + Double(sequence) * 1e-9
        item.meta.simulatedOrigin = Self.origin(kind, voice: target, documentKind: documentKind)
        let facts = item.document.facts.compactMap { fact -> DatasetSharedFact? in
            guard let sourceFact = origin.document.facts.first(where: { $0.kind == fact.kind }) else { return nil }
            return DatasetSharedFact(kind: fact.kind, sourceFact: sourceFact.id, targetFact: fact.id, sourceAnswer: sourceFact.answer,
                                     targetAnswer: fact.answer)
        }
        assert(changed == nil || facts.contains { $0.kind == changed && !$0.agrees })
        crosslinks.append(DatasetCrosslink(
            id: id, kind: kind, subject: entity.subject, sourceType: entity.type, targetType: entity.type,
            source: DatasetDocumentRef(node: source.rawValue, documentID: origin.document.id),
            target: DatasetDocumentRef(node: target.rawValue, documentID: item.document.id), facts: facts,
            overlap: DatasetOverlap.measure(origin.document.text, item.document.text)))
        placed[target.rawValue, default: []].append(item)
    }

    mutating func homonym(from origin: Placed, source: DatasetVoice, target: DatasetVoice, type: DatasetEntityType, sequence: Int) throws {
        let entity = try makeEntity(type, name: origin.entity.subject)
        let kinds = DatasetVoices.kinds(target, type)
        let documentKind = kinds[sequence % kinds.count]
        var item = try placeDocument(voice: target, kind: documentKind, entity: entity) { builder in
            try builder.homeDraft(voice: target, kind: documentKind, entity: entity)
        }
        let id = String(format: "x%04d", sequence)
        item.meta.crosslink = id
        item.meta.rank = origin.meta.rank + Double(sequence) * 1e-9
        crosslinks.append(DatasetCrosslink(
            id: id, kind: .homonym, subject: entity.subject, sourceType: origin.entity.type, targetType: type,
            source: DatasetDocumentRef(node: source.rawValue, documentID: origin.document.id),
            target: DatasetDocumentRef(node: target.rawValue, documentID: item.document.id), facts: [],
            overlap: DatasetOverlap.measure(origin.document.text, item.document.text)))
        placed[target.rawValue, default: []].append(item)
    }

    static func origin(_ kind: CrossKind, voice: DatasetVoice, documentKind: DocumentKind) -> SimulatedOrigin {
        switch kind {
        case .excerpt: return voice == .ambient ? .read : (voice == .craft ? .written : .imported)
        case .summary: return .generated
        case .paraphrase, .variant: return .imported
        case .homonym: return DatasetVoices.origin(documentKind)
        }
    }

    // MARK: Drafting

    mutating func fill(_ template: String, _ entity: Entity, x: String) -> String {
        entity.refs.fill(template).replacingOccurrences(of: "{x}", with: x)
    }

    mutating func incidental(_ voice: DatasetVoice) throws -> String {
        switch voice {
        case .ambient: return try forge.publication()
        case .craft: return try forge.person()
        case .veil: return try forge.gallery()
        }
    }

    mutating func factPiece(_ kind: FactKind, voice: DatasetVoice, entity: Entity) -> (Piece, DatasetPhrasing) {
        let phrasing = rng.pick(DatasetVoices.phrasings[kind]![voice]!)
        let piece = Piece.fact(
            kind: kind, prompt: entity.refs.fill(phrasing.prefix), value: entity.answers[kind]!, suffix: phrasing.suffix,
            paraphrases: DatasetVoices.paraphrases[kind]!.map { entity.refs.fill($0) }, negative: entity.negative.fill(phrasing.prefix))
        return (piece, phrasing)
    }

    mutating func opening(voice: DatasetVoice, kind: DocumentKind, entity: Entity) throws -> [Piece] {
        let x = try incidental(voice)
        let header = fill(rng.pick(DatasetVoices.headers[kind]!), entity, x: x)
        let intro = fill(rng.pick(DatasetVoices.intros[voice]![entity.type]!), entity, x: x)
        return [.sentence(header), .sentence(intro)]
    }

    mutating func tail(voice: DatasetVoice, entity: Entity) -> [[Piece]] {
        var paragraphs: [[Piece]] = []
        if rng.nextUnit() < 0.6 { paragraphs.append([]) }
        if rng.nextUnit() < 0.7 { paragraphs.append([.sentence(entity.refs.fill(rng.pick(DatasetVoices.closings[voice]!)))]) }
        return paragraphs
    }

    mutating func homeDraft(voice: DatasetVoice, kind: DocumentKind, entity: Entity) throws -> Draft {
        var phrasings: [FactKind: DatasetPhrasing] = [:]
        var facts: [Piece] = []
        for fact in rng.shuffled(entity.type.facts) {
            let (piece, phrasing) = factPiece(fact, voice: voice, entity: entity)
            facts.append(piece)
            phrasings[fact] = phrasing
        }
        let paragraphs = [try opening(voice: voice, kind: kind, entity: entity) + [facts[0]], [facts[1], facts[2]]]
            + tail(voice: voice, entity: entity)
        return Draft(kind: kind, name: "\(kind.rawValue.capitalized) · \(entity.refs.S)", subject: entity.subject,
                     paragraphs: paragraphs, fillers: DatasetVoices.fillers(voice, entity.type).map { entity.refs.fill($0) }, phrasings: phrasings)
    }

    mutating func crossDraft(_ cross: CrossKind, voice: DatasetVoice, kind: DocumentKind, entity: Entity, origin: Placed) throws -> Draft {
        let order = rng.shuffled(entity.type.facts)
        var paragraphs: [[Piece]] = [try opening(voice: voice, kind: kind, entity: entity)]
        switch cross {
        case .paraphrase, .variant:
            paragraphs[0].append(factPiece(order[0], voice: voice, entity: entity).0)
            paragraphs.append([factPiece(order[1], voice: voice, entity: entity).0, factPiece(order[2], voice: voice, entity: entity).0])
        case .summary:
            paragraphs[0].append(factPiece(order[0], voice: voice, entity: entity).0)
            if rng.nextUnit() < 0.5 { paragraphs.append([factPiece(order[1], voice: voice, entity: entity).0]) }
        case .excerpt:
            paragraphs[0].append(quote(order[0], voice: voice, entity: entity, origin: origin))
            paragraphs.append([factPiece(order[1], voice: voice, entity: entity).0])
        case .homonym:
            break
        }
        paragraphs += tail(voice: voice, entity: entity)
        return Draft(kind: kind, name: "\(kind.rawValue.capitalized) · \(entity.refs.S)", subject: entity.subject,
                     paragraphs: paragraphs, fillers: DatasetVoices.fillers(voice, entity.type).map { entity.refs.fill($0) }, phrasings: [:])
    }

    /// The source's own sentence for `kind`, quoted with a hedge slipped in after its first
    /// auxiliary verb, or else after the subject, so the quote is close to the source and never
    /// contains it.
    mutating func quote(_ kind: FactKind, voice: DatasetVoice, entity: Entity, origin: Placed) -> Piece {
        let phrasing = origin.phrasings[kind] ?? DatasetVoices.phrasings[kind]![origin.entity.type.home]![0]
        let lead = rng.pick(DatasetVoices.excerptLeads[voice]!)
        let hedge = rng.pick(DatasetVoices.hedges)
        func hedged(_ refs: SubjectRefs) -> String {
            var text = refs.fill(phrasing.prefix)
            guard let mention = text.range(of: refs.S) ?? text.range(of: refs.s) else { return text + ", \(hedge)," }
            // The fact's own verb follows its subject; an earlier one ("I was surprised that …") is the note-taker's.
            for verb in [" was ", " is ", " has ", " were "] {
                if let range = text.range(of: verb, range: mention.upperBound..<text.endIndex) {
                    text.replaceSubrange(range, with: verb + hedge + " ")
                    return text
                }
            }
            text.insert(contentsOf: ", \(hedge),", at: mention.upperBound)
            return text
        }
        return .fact(kind: kind, prompt: lead + hedged(entity.refs), value: entity.answers[kind]!, suffix: phrasing.suffix + "\"",
                     paraphrases: DatasetVoices.paraphrases[kind]!.map { entity.refs.fill($0) }, negative: lead + hedged(entity.negative))
    }

    // MARK: Assembly

    fileprivate struct Failure: Error {
        let problems: [String]
    }

    /// Drafts until a draft assembles (fresh random choices each time), then places it.
    mutating func placeDocument(
        voice: DatasetVoice, kind: DocumentKind, entity: Entity, _ draft: (inout Builder) throws -> Draft
    ) throws -> Placed {
        var last: [String] = []
        for _ in 0..<32 {
            let made = try draft(&self)
            switch assemble(made, slug: voice.rawValue) {
            case .success(let document):
                let meta = DatasetDocumentMeta(
                    id: document.id, node: voice.rawValue, kind: kind, entityType: entity.type, subject: document.subject, home: true,
                    crosslink: nil, simulatedOrigin: DatasetVoices.origin(kind), synthetic: true, rank: 0)
                return Placed(document: document, meta: meta, entity: entity, phrasings: made.phrasings)
            case .failure(let failure):
                last = failure.problems
            }
        }
        throw DatasetError.invalid(last)
    }

    mutating func assemble(_ draft: Draft, slug: String) -> Result<CorpusDocument, Failure> {
        var fillers = rng.shuffled(draft.fillers)
        var texts: [String] = []
        var placedFacts: [[(kind: FactKind, prompt: String, answer: String, sentence: String, start: Int, context: Int,
                             paraphrases: [String], negative: String)]] = []
        for pieces in draft.paragraphs {
            var sentences = pieces
            var length = sentences.reduce(-1) { $0 + $1.text.utf8.count + 1 }
            while length < BraidDataset.minParagraph, let filler = fillers.popLast() {
                if length + 1 + filler.utf8.count > BraidDataset.maxParagraph { break }
                sentences.append(.sentence(filler))
                length += filler.utf8.count + 1
            }
            guard !sentences.isEmpty else { continue }
            var text = ""
            var starts: [Int] = []
            for (i, sentence) in sentences.enumerated() {
                if i > 0 { text += " " }
                starts.append(text.utf8.count)
                text += sentence.text
            }
            var facts: [(kind: FactKind, prompt: String, answer: String, sentence: String, start: Int, context: Int,
                         paraphrases: [String], negative: String)] = []
            for (i, sentence) in sentences.enumerated() {
                guard case .fact(let kind, let prompt, let value, _, let paraphrases, let negative) = sentence else { continue }
                facts.append((kind, prompt, " " + value, sentence.text, starts[i], i > 0 ? starts[i - 1] : starts[i], paraphrases, negative))
            }
            texts.append(text)
            placedFacts.append(facts)
        }
        let documentText = texts.joined(separator: "\n\n")
        var problems: [String] = []
        if TextChunker.chunk(documentText, maxChars: BraidDataset.maxChars, minChars: BraidDataset.minChars) != texts {
            problems.append("\(draft.name): the chunker does not reproduce the paragraphs")
        }
        if Set(texts).count != texts.count { problems.append("\(draft.name): a paragraph repeats") }
        for text in texts where text.utf8.count > BraidDataset.maxChars || text.utf8.count < BraidDataset.minChars {
            problems.append("\(draft.name): paragraph length \(text.utf8.count)")
        }
        for fact in placedFacts.joined() {
            if FactValidator.occurrences(of: fact.prompt, in: documentText) != 1 { problems.append("\(draft.name): '\(fact.prompt)' repeats") }
            for paraphrase in fact.paraphrases where documentText.contains(paraphrase) {
                problems.append("\(draft.name): paraphrase '\(paraphrase)' occurs")
            }
        }
        guard problems.isEmpty else { return .failure(Failure(problems: problems)) }
        let canonical = ContentHash.canonical(documentText)
        let id = DocumentID.make(slug: slug, canonicalText: canonical)
        let partitions = texts.enumerated().map { index, text in
            CorpusPartition(index: index, text: text, textSHA256: ContentHash.sha256Hex(text),
                            url: DocumentID.partitionURL(slug: slug, documentID: id, index: index))
        }
        var facts: [Fact] = []
        for (partition, list) in placedFacts.enumerated() {
            for fact in list {
                let answerStart = fact.start + fact.prompt.utf8.count
                facts.append(Fact(
                    id: "\(id)#\(fact.kind.rawValue)", kind: fact.kind, documentID: id, partitionIndex: partition, subject: draft.subject,
                    prompt: fact.prompt, answer: fact.answer, sentence: fact.sentence, sentenceStart: fact.start, contextStart: fact.context,
                    answerStart: answerStart, answerEnd: answerStart + fact.answer.utf8.count, paraphrases: fact.paraphrases,
                    negativePrompt: fact.negative))
            }
        }
        return .success(CorpusDocument(id: id, name: draft.name, kind: draft.kind, subject: draft.subject, partitions: partitions,
                                       facts: facts, textSHA256: ContentHash.sha256Hex(canonical)))
    }

    // MARK: Finish

    mutating func finish() throws -> BraidDataset.Generated {
        var corpora: [String: GeneratedCorpus] = [:]
        var meta: [String: [DatasetDocumentMeta]] = [:]
        var summaries: [DatasetNodeSummary] = []
        for voice in DatasetVoice.allCases {
            let name = voice.rawValue
            let ordered = (placed[name] ?? []).enumerated().sorted { a, b in
                a.element.meta.rank != b.element.meta.rank ? a.element.meta.rank < b.element.meta.rank : a.offset < b.offset
            }.map(\.element)
            let documents = ordered.map(\.document)
            let entries = documents.flatMap { document in
                document.partitions.map { ContentHash.CorpusEntry(documentID: document.id, partitionIndex: $0.index, textSHA256: $0.textSHA256) }
            }
            let manifest = CorpusManifest(
                slug: name, generator: BraidDataset.generatorName, generatorVersion: BraidDataset.generatorVersion, seed: spec.seed,
                documentCount: documents.count, partitionCount: entries.count, factCount: documents.reduce(0) { $0 + $1.facts.count },
                chunkMaxChars: BraidDataset.maxChars, chunkMinChars: BraidDataset.minChars, documentIDs: documents.map(\.id),
                corpusHash: ContentHash.corpusHash(entries))
            corpora[name] = GeneratedCorpus(manifest: manifest, documents: documents)
            meta[name] = ordered.map(\.meta)
            var origins: [String: Int] = [:]
            for row in ordered { origins[row.meta.simulatedOrigin.rawValue, default: 0] += 1 }
            summaries.append(DatasetNodeSummary(
                name: name, documents: documents.count, home: ordered.filter { $0.meta.home && $0.meta.crosslink == nil }.count,
                crossed: ordered.filter { !$0.meta.home }.count, homonyms: ordered.filter { $0.meta.home && $0.meta.crosslink != nil }.count,
                partitions: entries.count, facts: manifest.factCount,
                words: documents.reduce(0) { $0 + DatasetOverlap.words($1.text).count }, corpusHash: manifest.corpusHash, origins: origins))
        }
        var counts: [String: Int] = [:]
        for link in crosslinks { counts[link.kind.rawValue, default: 0] += 1 }
        let manifest = DatasetManifest(
            schemaVersion: 1, generator: BraidDataset.generatorName, generatorVersion: BraidDataset.generatorVersion, spec: spec,
            nodes: summaries, crosslinks: counts,
            datasetHash: try BraidDataset.hash(corpusHashes: summaries.map { ($0.name, $0.corpusHash) }, crosslinks: crosslinks))
        return BraidDataset.Generated(manifest: manifest, corpora: corpora, meta: meta, crosslinks: crosslinks, negatives: negatives)
    }
}
