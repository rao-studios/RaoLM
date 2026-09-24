//
//  SyntheticCorpus.swift
//  RaoLMCore
//
//  WHAT: A deterministic generator for a small fictional archive (the Veldmar world) whose
//        every fact appears in exactly one document, in exactly one partition.
//  IN:   A slug, a seed, a document count.
//  OUT:  A GeneratedCorpus: documents split into partitions (paragraph == partition),
//        each fact located by UTF-8 offsets, with paraphrased and negative prompts.
//  PIN:  The citation proof depends on uniqueness, so the generator validates as it goes:
//        each fact's prompt names its document's subject, occurs once in that document,
//        and the subject occurs in no other document. Same seed, same bytes.
//

import Foundation

public enum SyntheticCorpusError: Error, CustomStringConvertible {
    case invalidSlug(String)
    case invalidCount(Int)
    case namesExhausted(String)
    case poolExhausted(String, size: Int)
    case validation([String])

    public var description: String {
        switch self {
        case .invalidSlug(let slug):
            return "invalid corpus slug '\(slug)': use 1–32 characters of [a-z0-9-], starting with a letter or digit"
        case .invalidCount(let count):
            return "document count \(count) is out of range (1…1000)"
        case .namesExhausted(let what):
            return "ran out of unique \(what) names; lower the document count"
        case .poolExhausted(let name, let size):
            return "ran out of unique \(name) values (\(size) available); lower the document count"
        case .validation(let problems):
            return "synthetic corpus failed validation:\n  " + problems.prefix(20).joined(separator: "\n  ")
        }
    }
}

public enum SyntheticCorpus {
    public static let generatorName = "SyntheticCorpus"
    public static let generatorVersion = 1
    static let minParagraphChars = 220
    static let maxParagraphChars = 560

    public static func generate(
        slug: String = "veldmar", seed: UInt64 = 42, documentCount: Int = 200,
        maxChars: Int = 600, minChars: Int = 120
    ) throws -> GeneratedCorpus {
        guard slug.range(of: #"^[a-z0-9][a-z0-9-]{0,31}$"#, options: .regularExpression) != nil else {
            throw SyntheticCorpusError.invalidSlug(slug)
        }
        guard (1...1000).contains(documentCount) else {
            throw SyntheticCorpusError.invalidCount(documentCount)
        }
        var world = World(seed: seed)
        var documents: [CorpusDocument] = []
        documents.reserveCapacity(documentCount)
        let kinds = DocumentKind.allCases
        for index in 0..<documentCount {
            let document = try world.makeDocument(
                kind: kinds[index % kinds.count], slug: slug, maxChars: maxChars, minChars: minChars)
            documents.append(document)
        }
        let problems = FactValidator.validate(
            documents: documents, negativeSubjects: world.negativeSubjects,
            maxChars: maxChars, minChars: minChars)
        guard problems.isEmpty else { throw SyntheticCorpusError.validation(problems) }

        let entries = documents.flatMap { document in
            document.partitions.map {
                ContentHash.CorpusEntry(documentID: document.id, partitionIndex: $0.index, textSHA256: $0.textSHA256)
            }
        }
        let manifest = CorpusManifest(
            slug: slug, generator: generatorName, generatorVersion: generatorVersion, seed: seed,
            documentCount: documents.count, partitionCount: entries.count,
            factCount: documents.reduce(0) { $0 + $1.facts.count },
            chunkMaxChars: maxChars, chunkMinChars: minChars,
            documentIDs: documents.map(\.id), corpusHash: ContentHash.corpusHash(entries))
        return GeneratedCorpus(manifest: manifest, documents: documents)
    }
}

// MARK: - Templates

/// How a document refers to its subject mid-sentence (`s`) and at the start of one (`S`).
struct SubjectRefs {
    let s: String
    let S: String

    func fill(_ template: String) -> String {
        template.replacingOccurrences(of: "{s}", with: s).replacingOccurrences(of: "{S}", with: S)
    }
}

struct Phrasing {
    let prefix: String
    let suffix: String
}

struct FactSpec {
    let phrasings: [Phrasing]
    let paraphrases: [String]
}

enum FactTemplates {
    static func p(_ prefix: String, _ suffix: String) -> Phrasing { Phrasing(prefix: prefix, suffix: suffix) }

    static let specs: [FactKind: FactSpec] = [
        // landmark
        .completedYear: FactSpec(
            phrasings: [
                p("Construction of {s} was completed in", "."),
                p("Work on {s} finished in", ", after many seasons of delay."),
                p("{S} first opened to travellers in", "."),
                p("Masons set the final stone of {s} in", "."),
            ],
            paraphrases: [
                "Surveyors date the completion of {s} to the year",
                "Ask the keepers of {s} and they will name its year of completion as",
            ]),
        .architect: FactSpec(
            phrasings: [
                p("The plans of {s} were drawn by the architect", ", who signed every sheet."),
                p("The design of {s} is credited to", ", a builder from the northern towns."),
                p("Records name the architect of {s} as", "."),
                p("{S} was designed by", "."),
            ],
            paraphrases: ["The person who designed {s} was", "Credit for the design of {s} goes to"]),
        .height: FactSpec(
            phrasings: [
                p("{S} rises", " metres above the riverbank."),
                p("At its highest point {s} reaches", " metres."),
                p("Surveyors measured {s} at", " metres from base to crown."),
                p("The tallest arch of {s} stands", " metres high."),
            ],
            paraphrases: ["The height of {s} in metres is", "In metres, the full height of {s} comes to"]),
        .restoredYear: FactSpec(
            phrasings: [
                p("{S} was restored in", " after a winter flood damaged its foundations."),
                p("A long restoration of {s} began in", "."),
                p("Engineers rebuilt part of {s} in", "."),
                p("The last major repair of {s} took place in", "."),
            ],
            paraphrases: ["The year {s} was restored is", "Restoration work on {s} is dated to"]),
        // biography
        .birthYear: FactSpec(
            phrasings: [
                p("{S} was born in the year", "."),
                p("The parish register records the birth of {s} in", "."),
                p("{S} came into the world in", ", during a hard winter."),
                p("Letters place the birth of {s} in", "."),
            ],
            paraphrases: ["The year of birth of {s} is given as", "{S} was born, according to the archive, in"]),
        .birthplace: FactSpec(
            phrasings: [
                p("{S} grew up in the village of", ", beside the old mill road."),
                p("The family of {s} lived in", " for three generations."),
                p("{S} spent a childhood in", "."),
                p("The childhood home of {s} stood in", "."),
            ],
            paraphrases: ["The village where {s} grew up is", "As a child, {s} lived in the village of"]),
        .mentor: FactSpec(
            phrasings: [
                p("As an apprentice, {s} studied under", "."),
                p("{S} learned the trade from", ", a demanding teacher."),
                p("The teacher of {s} was", ", whose workshop stood by the square."),
                p("{S} trained for seven years with", "."),
            ],
            paraphrases: ["The master who trained {s} was", "{S} was apprenticed to"]),
        // expedition
        .departureYear: FactSpec(
            phrasings: [
                p("{S} departed in", "."),
                p("{S} left the harbour in the spring of", "."),
                p("Supplies for {s} were gathered through the winter before it set out in", "."),
                p("{S} began its march in", "."),
            ],
            paraphrases: ["The year in which {s} set out was", "{S} started its journey in the year"]),
        .leader: FactSpec(
            phrasings: [
                p("{S} was led by", "."),
                p("Command of {s} fell to", ", a veteran of the coast roads."),
                p("The leader of {s} was", "."),
                p("{S} followed the orders of", "."),
            ],
            paraphrases: ["The person in command of {s} was", "{S} took its orders from"]),
        .distance: FactSpec(
            phrasings: [
                p("In total {s} travelled", " kilometres."),
                p("The route of {s} covered", " kilometres of rough ground."),
                p("By its return {s} had walked", " kilometres."),
                p("The journals of {s} count", " kilometres in all."),
            ],
            paraphrases: [
                "The distance covered by {s}, in kilometres, was",
                "Over its whole route, {s} went a distance in kilometres of",
            ]),
        .members: FactSpec(
            phrasings: [
                p("{S} numbered", " members when it set out."),
                p("The roll of {s} lists", " members."),
                p("{S} counted", " travellers in its company."),
                p("The full company of {s} was", " strong."),
            ],
            paraphrases: ["The number of people in {s} was", "Counting every traveller, the size of {s} was"]),
        // council
        .foundedYear: FactSpec(
            phrasings: [
                p("{S} was founded in", "."),
                p("{S} held its first session in", "."),
                p("The charter of {s} was sealed in", "."),
                p("{S} was established in", " by a vote of the guilds."),
            ],
            paraphrases: ["The founding year of {s} is", "{S} came into being in the year"]),
        .firstSpeaker: FactSpec(
            phrasings: [
                p("The first speaker of {s} was", "."),
                p("{S} chose", " as its first speaker."),
                p("At its first session {s} elected", " to speak for it."),
                p("The founding speaker of {s} was", "."),
            ],
            paraphrases: [
                "The person first chosen to speak for {s} was",
                "Before any other, {s} was led in debate by",
            ]),
        .seats: FactSpec(
            phrasings: [
                p("{S} seated", " members."),
                p("{S} had", " seats in its hall."),
                p("By law {s} kept", " seats."),
                p("The hall of {s} held", " chairs for its members."),
            ],
            paraphrases: [
                "The number of seats in {s} was",
                "Counting every chair, the membership of {s} came to",
            ]),
        // recipe
        .creator: FactSpec(
            phrasings: [
                p("{S} was first baked by", "."),
                p("The recipe for {s} is credited to", "."),
                p("{S} was created by", ", a cook at the river inn."),
                p("Old kitchens say {s} was invented by", "."),
            ],
            paraphrases: ["The cook who first made {s} was", "Credit for inventing {s} belongs to"]),
        .grams: FactSpec(
            phrasings: [
                p("{S} calls for", " grams of flour."),
                p("A single batch of {s} needs", " grams of flour."),
                p("For {s}, bakers weigh out", " grams of flour."),
                p("The dough of {s} takes", " grams of flour."),
            ],
            paraphrases: [
                "The weight of flour in {s}, in grams, is",
                "In grams, the flour needed for {s} comes to",
            ]),
        .bakeMinutes: FactSpec(
            phrasings: [
                p("{S} bakes for", " minutes."),
                p("{S} must rest in the oven for", " minutes."),
                p("Cooks leave {s} in the oven for", " minutes."),
                p("{S} is baked for", " minutes until golden."),
            ],
            paraphrases: [
                "The baking time of {s}, in minutes, is",
                "Measured in minutes, the time {s} spends in the oven is",
            ]),
    ]

    static let commonFillers = [
        "Accounts of {s} survive in several letters kept by the archive.",
        "The archive keeps a small folio of sketches related to {s}.",
        "Later clerks added careful notes about {s} in the margins of the ledgers.",
        "Travellers who passed through Veldmar wrote often about {s}.",
        "Nothing in the surviving records contradicts this account of {s}.",
        "The story of {s} is still told at the autumn fairs.",
        "A copy of the oldest description of {s} was made by a patient clerk.",
        "Scholars who visit the archive often ask to see the papers about {s}.",
        "Some of the details about {s} are disputed in the older chronicles.",
        "The reading room keeps a map on which {s} is marked in red ink.",
    ]

    static let stones = ["limestone", "granite", "sandstone", "slate", "basalt", "marble", "flint"]
    static let occupations = ["cartographer", "glassmaker", "archivist", "clockmaker", "weaver",
                              "astronomer", "bookbinder", "herbalist", "surveyor", "bellfounder",
                              "printer", "luthier"]
    static let traits = ["patient, exacting work", "a talent for teaching",
                         "careful records kept over many decades", "an unusual sense of colour",
                         "stubborn honesty in public disputes", "work that outlasted every rival"]
}

// MARK: - World

private enum Piece {
    case sentence(String)
    case fact(kind: FactKind, prefix: String, value: String, suffix: String, paraphrases: [String], negativePrompt: String)

    var text: String {
        switch self {
        case .sentence(let text): return text
        case .fact(_, let prefix, let value, let suffix, _, _): return prefix + " " + value + suffix
        }
    }
}

private struct DocumentDraft {
    let kind: DocumentKind
    let name: String
    let subject: String
    let paragraphs: [[Piece]]
    let fillers: [String]
}

private struct World {
    var rng: SplitMix64
    var forge: NameForge
    var years: ValuePool
    var heights: ValuePool
    var distances: ValuePool
    var members: ValuePool
    var seats: ValuePool
    var grams: ValuePool
    var minutes: ValuePool
    var negativeSubjects: [String] = []

    init(seed: UInt64) {
        rng = SplitMix64.derived(seed: seed, stream: 1)
        forge = NameForge(rng: SplitMix64.derived(seed: seed, stream: 2))
        var values = SplitMix64.derived(seed: seed, stream: 3)
        years = ValuePool(name: "year", range: 1200...2199, rng: &values)
        heights = ValuePool(name: "height", range: 20...480, rng: &values)
        distances = ValuePool(name: "distance", range: 120...2980, rng: &values)
        members = ValuePool(name: "members", range: 12...240, rng: &values)
        seats = ValuePool(name: "seats", range: 7...99, rng: &values)
        grams = ValuePool(name: "grams", range: 150...990, rng: &values)
        minutes = ValuePool(name: "minutes", range: 15...240, rng: &values)
    }

    mutating func makeDocument(kind: DocumentKind, slug: String, maxChars: Int, minChars: Int) throws -> CorpusDocument {
        var lastProblems: [String] = []
        for _ in 0..<24 {
            let draft = try draftDocument(kind: kind)
            switch assemble(draft, slug: slug, maxChars: maxChars, minChars: minChars) {
            case .success(let document): return document
            case .failure(let failure): lastProblems = failure.problems
            }
        }
        throw SyntheticCorpusError.validation(lastProblems)
    }

    // MARK: Drafting

    private mutating func fact(_ kind: FactKind, _ value: String, _ refs: SubjectRefs, negative: SubjectRefs) -> Piece {
        let spec = FactTemplates.specs[kind]!
        let phrasing = rng.pick(spec.phrasings)
        return .fact(
            kind: kind, prefix: refs.fill(phrasing.prefix), value: value, suffix: phrasing.suffix,
            paraphrases: spec.paraphrases.map { refs.fill($0) },
            negativePrompt: negative.fill(phrasing.prefix))
    }

    private mutating func count(_ range: ClosedRange<Int>) -> Int { rng.nextInt(in: range) }

    private mutating func draftDocument(kind: DocumentKind) throws -> DocumentDraft {
        switch kind {
        case .landmark: return try draftLandmark()
        case .biography: return try draftBiography()
        case .expedition: return try draftExpedition()
        case .council: return try draftCouncil()
        case .recipe: return try draftRecipe()
        }
    }

    private mutating func negative(_ make: (inout NameForge) throws -> String, s: (String) -> String, S: (String) -> String) rethrows -> SubjectRefs {
        let name = try make(&forge)
        negativeSubjects.append(name)
        return SubjectRefs(s: s(name), S: S(name))
    }

    private mutating func draftLandmark() throws -> DocumentDraft {
        let name = try forge.landmark()
        let refs = SubjectRefs(s: "the \(name)", S: "The \(name)")
        let neg = try negative({ try $0.landmark() }, s: { "the \($0)" }, S: { "The \($0)" })
        let district = try forge.place()
        let river = try forge.river()
        let architect = try forge.person()
        let stone = rng.pick(FactTemplates.stones)
        let intro = rng.pick([
            "\(refs.S) stands in the \(district) quarter of Veldmar, above the slow water of the \(river).",
            "Above the \(district) quarter, \(refs.s) looks over the bend of the \(river).",
            "Few structures in Veldmar are as well documented as \(refs.s) in the \(district) quarter.",
        ])
        let a: [Piece] = [
            .sentence(intro),
            fact(.completedYear, String(try years.take()), refs, negative: neg),
            fact(.architect, architect, refs, negative: neg),
        ]
        let b: [Piece] = [
            .sentence("Its walls are built from pale \(stone) quarried in the hills above \(district)."),
            fact(.height, String(try heights.take()), refs, negative: neg),
        ]
        let c: [Piece] = [
            fact(.restoredYear, String(try years.take()), refs, negative: neg),
            .sentence("A bronze plaque near the entrance of \(refs.s) lists the names of the first masons."),
        ]
        let d: [Piece] = [
            .sentence("At dusk the lamps of \(refs.s) can be seen from the far bank of the \(river)."),
        ]
        let paragraphs = Array([a, b, c, d].prefix(count(2...4)))
        return DocumentDraft(kind: .landmark, name: refs.S, subject: name, paragraphs: paragraphs,
                             fillers: FactTemplates.commonFillers.map { refs.fill($0) })
    }

    private mutating func draftBiography() throws -> DocumentDraft {
        let name = try forge.person()
        let refs = SubjectRefs(s: name, S: name)
        let neg = try negative({ try $0.person() }, s: { $0 }, S: { $0 })
        let birthplace = try forge.place()
        let mentor = try forge.person()
        let occupation = rng.pick(FactTemplates.occupations)
        let trait = rng.pick(FactTemplates.traits)
        let intro = rng.pick([
            "\(name) was a \(occupation) of Veldmar, remembered for \(trait).",
            "Among the \(occupation)s of Veldmar, \(name) is remembered for \(trait).",
            "\(name) worked as a \(occupation) and became known for \(trait).",
        ])
        let a: [Piece] = [
            .sentence(intro),
            fact(.birthYear, String(try years.take()), refs, negative: neg),
            fact(.birthplace, birthplace, refs, negative: neg),
        ]
        let b: [Piece] = [
            fact(.mentor, mentor, refs, negative: neg),
            .sentence("Much of the work of \(name) survives in the lower rooms of the archive."),
        ]
        let c: [Piece] = [
            .sentence("In later life \(name) took on apprentices and taught them to keep honest ledgers."),
            .sentence("Several letters written by \(name) were copied into the town books."),
        ]
        let d: [Piece] = [
            .sentence("A portrait of \(name) hangs in the reading room of the archive."),
        ]
        let paragraphs = Array([a, b, c, d].prefix(count(2...4)))
        return DocumentDraft(kind: .biography, name: name, subject: name, paragraphs: paragraphs,
                             fillers: FactTemplates.commonFillers.map { refs.fill($0) })
    }

    private mutating func draftExpedition() throws -> DocumentDraft {
        let place = try forge.place()
        let name = "\(place) Expedition"
        let refs = SubjectRefs(s: "the \(name)", S: "The \(name)")
        let neg = try negative({ try $0.place() }, s: { "the \($0) Expedition" }, S: { "The \($0) Expedition" })
        let leader = try forge.person()
        let river = try forge.river()
        let intro = rng.pick([
            "\(refs.S) set out to map the far reaches of the Veldmar highlands.",
            "\(refs.S) was organised to chart the passes beyond the \(river).",
            "The archive holds the journals of \(refs.s), which crossed the eastern moors.",
        ])
        let a: [Piece] = [
            .sentence(intro),
            fact(.departureYear, String(try years.take()), refs, negative: neg),
            fact(.leader, leader, refs, negative: neg),
        ]
        let b: [Piece] = [
            fact(.distance, String(try distances.take()), refs, negative: neg),
            fact(.members, String(try members.take()), refs, negative: neg),
        ]
        let c: [Piece] = [
            .sentence("The maps drawn during \(refs.s) were copied for the harbour guild."),
            .sentence("Several members of \(refs.s) kept private diaries that later reached the archive."),
        ]
        let d: [Piece] = [
            .sentence("The return of \(refs.s) was celebrated with a feast in the square."),
        ]
        let paragraphs = Array([a, b, c, d].prefix(count(2...4)))
        return DocumentDraft(kind: .expedition, name: refs.S, subject: name, paragraphs: paragraphs,
                             fillers: FactTemplates.commonFillers.map { refs.fill($0) })
    }

    private mutating func draftCouncil() throws -> DocumentDraft {
        let place = try forge.place()
        let name = "Council of \(place)"
        let refs = SubjectRefs(s: "the \(name)", S: "The \(name)")
        let neg = try negative({ try $0.place() }, s: { "the Council of \($0)" }, S: { "The Council of \($0)" })
        let speaker = try forge.person()
        let valley = try forge.place()
        let intro = rng.pick([
            "\(refs.S) met in a long hall of timber and slate.",
            "\(refs.S) governed the markets and roads of the \(valley) valley.",
            "For generations \(refs.s) settled disputes between the guilds of the \(valley) valley.",
        ])
        let a: [Piece] = [
            .sentence(intro),
            fact(.foundedYear, String(try years.take()), refs, negative: neg),
            fact(.firstSpeaker, speaker, refs, negative: neg),
        ]
        let b: [Piece] = [
            fact(.seats, String(try seats.take()), refs, negative: neg),
            .sentence("Minutes of \(refs.s) fill eleven bound volumes in the archive."),
        ]
        let c: [Piece] = [
            .sentence("The seal of \(refs.s) shows a lantern above a pair of scales."),
            .sentence("Market fees in the \(valley) valley were set each spring by \(refs.s)."),
        ]
        let d: [Piece] = [
            .sentence("A painted banner of \(refs.s) still hangs above the archive stairs."),
        ]
        let paragraphs = Array([a, b, c, d].prefix(count(2...4)))
        return DocumentDraft(kind: .council, name: refs.S, subject: name, paragraphs: paragraphs,
                             fillers: FactTemplates.commonFillers.map { refs.fill($0) })
    }

    private mutating func draftRecipe() throws -> DocumentDraft {
        let name = try forge.dish()
        let refs = SubjectRefs(s: "the \(name)", S: "The \(name)")
        let neg = try negative({ try $0.dish() }, s: { "the \($0)" }, S: { "The \($0)" })
        let creator = try forge.person()
        let valley = try forge.place()
        let intro = rng.pick([
            "\(refs.S) is a traditional bread of the Veldmar valleys.",
            "Bakers across the \(valley) valley still prepare \(refs.s) for the harvest fair.",
            "\(refs.S) appears in more household ledgers than any other recipe in the archive.",
        ])
        let a: [Piece] = [
            .sentence(intro),
            fact(.creator, creator, refs, negative: neg),
        ]
        let b: [Piece] = [
            fact(.grams, String(try grams.take()), refs, negative: neg),
            fact(.bakeMinutes, String(try minutes.take()), refs, negative: neg),
        ]
        let c: [Piece] = [
            .sentence("Some families add dried fruit to \(refs.s), though the oldest ledgers do not."),
            .sentence("The crust of \(refs.s) is brushed with milk before baking."),
        ]
        let d: [Piece] = [
            .sentence("Wrapped in linen, \(refs.s) keeps well for a week."),
        ]
        let paragraphs = Array([a, b, c, d].prefix(count(2...4)))
        return DocumentDraft(kind: .recipe, name: refs.S, subject: name, paragraphs: paragraphs,
                             fillers: FactTemplates.commonFillers.map { refs.fill($0) })
    }

    // MARK: Assembly

    private struct AssemblyFailure: Error {
        let problems: [String]
    }

    private struct PlacedFact {
        let kind: FactKind
        let prompt: String
        let answer: String
        let sentence: String
        let sentenceStart: Int
        let contextStart: Int
        let paraphrases: [String]
        let negativePrompt: String
    }

    private mutating func assemble(
        _ draft: DocumentDraft, slug: String, maxChars: Int, minChars: Int
    ) -> Result<CorpusDocument, AssemblyFailure> {
        var fillers = rng.shuffled(draft.fillers)
        var paragraphTexts: [String] = []
        var paragraphFacts: [[PlacedFact]] = []

        for pieces in draft.paragraphs {
            var sentences = pieces
            var length = sentences.reduce(-1) { $0 + $1.text.utf8.count + 1 }
            while length < SyntheticCorpus.minParagraphChars, let filler = fillers.popLast() {
                if length + 1 + filler.utf8.count > SyntheticCorpus.maxParagraphChars { break }
                sentences.append(.sentence(filler))
                length += filler.utf8.count + 1
            }
            var text = ""
            var starts: [Int] = []
            for (i, sentence) in sentences.enumerated() {
                if i > 0 { text += " " }
                starts.append(text.utf8.count)
                text += sentence.text
            }
            var placed: [PlacedFact] = []
            for (i, sentence) in sentences.enumerated() {
                guard case .fact(let kind, let prefix, let value, _, let paraphrases, let negativePrompt) = sentence else { continue }
                placed.append(PlacedFact(
                    kind: kind, prompt: prefix, answer: " " + value, sentence: sentence.text,
                    sentenceStart: starts[i], contextStart: i > 0 ? starts[i - 1] : starts[i],
                    paraphrases: paraphrases, negativePrompt: negativePrompt))
            }
            paragraphTexts.append(text)
            paragraphFacts.append(placed)
        }

        let documentText = paragraphTexts.joined(separator: "\n\n")
        var problems: [String] = []
        if TextChunker.chunk(documentText, maxChars: maxChars, minChars: minChars) != paragraphTexts {
            problems.append("\(draft.name): chunker does not reproduce the paragraphs")
        }
        if Set(paragraphTexts).count != paragraphTexts.count {
            problems.append("\(draft.name): a paragraph repeats")
        }
        for text in paragraphTexts where text.utf8.count > maxChars || text.utf8.count < minChars {
            problems.append("\(draft.name): paragraph length \(text.utf8.count) outside \(minChars)…\(maxChars)")
        }
        for placed in paragraphFacts.joined() {
            let occurrences = FactValidator.occurrences(of: placed.prompt, in: documentText)
            if occurrences != 1 {
                problems.append("\(draft.name): prompt '\(placed.prompt)' occurs \(occurrences) times")
            }
            for paraphrase in placed.paraphrases where documentText.contains(paraphrase) {
                problems.append("\(draft.name): paraphrase '\(paraphrase)' occurs in the text")
            }
        }
        guard problems.isEmpty else { return .failure(AssemblyFailure(problems: problems)) }

        let canonical = ContentHash.canonical(documentText)
        let id = DocumentID.make(slug: slug, canonicalText: canonical)
        let partitions = paragraphTexts.enumerated().map { index, text in
            CorpusPartition(
                index: index, text: text, textSHA256: ContentHash.sha256Hex(text),
                url: DocumentID.partitionURL(slug: slug, documentID: id, index: index))
        }
        var facts: [Fact] = []
        for (partitionIndex, placedFacts) in paragraphFacts.enumerated() {
            for placed in placedFacts {
                let answerStart = placed.sentenceStart + placed.prompt.utf8.count
                facts.append(Fact(
                    id: "\(id)#\(placed.kind.rawValue)", kind: placed.kind, documentID: id,
                    partitionIndex: partitionIndex, subject: draft.subject, prompt: placed.prompt,
                    answer: placed.answer, sentence: placed.sentence, sentenceStart: placed.sentenceStart,
                    contextStart: placed.contextStart, answerStart: answerStart,
                    answerEnd: answerStart + placed.answer.utf8.count, paraphrases: placed.paraphrases,
                    negativePrompt: placed.negativePrompt))
            }
        }
        return .success(CorpusDocument(
            id: id, name: draft.name, kind: draft.kind, subject: draft.subject, partitions: partitions,
            facts: facts, textSHA256: ContentHash.sha256Hex(canonical)))
    }
}
