//
//  DatasetVoices.swift
//  RaoLMCore
//
//  WHAT: How each Thread writes. Every fact kind has phrasings in all three voices: four in
//        the voice of the world it belongs to, two in each other voice, so an entity retold on
//        another Thread states the same facts in that Thread's own words. Each voice also has
//        its document headers, an intro for every entity type, fillers, closings and the
//        lead-ins it quotes another source with.
//  PIN:  A phrasing's prefix names the subject ({s} mid-sentence, {S} at a sentence start) and
//        ends where the answer begins; the answer follows after one space. Paraphrases are
//        worded unlike every phrasing, so they never occur in a corpus.
//

import Foundation

struct DatasetPhrasing {
    let prefix: String
    let suffix: String
}

enum DatasetVoices {
    private static func p(_ prefix: String, _ suffix: String) -> DatasetPhrasing { DatasetPhrasing(prefix: prefix, suffix: suffix) }

    /// Per fact kind, per voice: how that voice states the fact.
    static let phrasings: [FactKind: [DatasetVoice: [DatasetPhrasing]]] = [
        // MARK: Ambient's world
        .townFounded: [
            .ambient: [p("The article says {s} was founded in", "."), p("According to the piece I read, {s} dates back to", "."),
                       p("{S} was settled, the guide explains, in", ", when the first mills went up."),
                       p("The page traces the founding of {s} to", ".")],
            .craft: [p("The seed data lists the founding year of {s} as", "."), p("In the fixture, {s} carries a founded year of", ".")],
            .veil: [p("The wall text notes that {s} was founded in", "."), p("The catalogue dates the founding of {s} to", ".")],
        ],
        .townPopulation: [
            .ambient: [p("{S} has a population of about", " people."), p("The census figure the article quotes for {s} is", " residents."),
                       p("I noted that {s} is home to", " people."), p("By the latest count {s} holds", " residents.")],
            .craft: [p("The population field for {s} is set to", "."), p("Our test expects {s} to report a population of", ".")],
            .veil: [p("The caption gives the population of {s} as", "."), p("When the print was made, {s} numbered", " residents.")],
        ],
        .townMayor: [
            .ambient: [p("The current mayor of {s} is", ", the article says."), p("{S} is governed by its mayor,", "."),
                       p("The interview was with the mayor of {s},", "."), p("I read that the mayor of {s} is", ".")],
            .craft: [p("The contact record for the mayor of {s} names", "."), p("In the admin panel, the mayor of {s} shows up as", ".")],
            .veil: [p("The print was commissioned by the mayor of {s},", "."), p("The lender is the mayor of {s},", ".")],
        ],
        .researcherBorn: [
            .ambient: [p("The profile says {s} was born in", "."), p("{S} was born, the essay notes, in", "."),
                       p("I learned that {s} was born in", ", which surprised me."), p("The biography opens with the birth of {s} in", ".")],
            .craft: [p("The author record for {s} has a birth year of", "."), p("We store the birth year of {s} as", ".")],
            .veil: [p("The label gives the birth year of {s} as", "."), p("The archive dates the birth of {s} to", ".")],
        ],
        .researcherMentor: [
            .ambient: [p("The piece mentions that {s} trained under", "."), p("{S} studied, the article says, with", "."),
                       p("Early on, {s} worked in the lab of", "."), p("I saved the part where {s} thanks the mentor", ".")],
            .craft: [p("The advisor field for {s} points to", "."), p("Our import maps the mentor of {s} to", ".")],
            .veil: [p("The note records that {s} was taught by", "."), p("The letter shows {s} was a student of", ".")],
        ],
        .researcherBook: [
            .ambient: [p("The best known book by {s} is", "."), p("{S} is the author of", ", the article reminds me."),
                       p("I want to read the book {s} wrote,", "."), p("The review praised the book by {s},", ".")],
            .craft: [p("The bibliography entry for {s} lists the title", "."), p("The citation test for {s} expects the book", ".")],
            .veil: [p("The vitrine holds a first edition by {s},", "."), p("The display case shows the book by {s},", ".")],
        ],
        .festivalFirst: [
            .ambient: [p("The article says {s} was first held in", "."), p("{S} began, according to the page, in", "."),
                       p("I read that the first edition of {s} took place in", "."), p("The history section dates {s} to", ".")],
            .craft: [p("The events table records the first year of {s} as", "."), p("Our calendar import starts {s} in", ".")],
            .veil: [p("The catalogue notes that {s} was first held in", "."), p("The poster series follows {s} from its first year,", ".")],
        ],
        .festivalVisitors: [
            .ambient: [p("Last year {s} drew about", " visitors."), p("The piece estimates that {s} welcomes", " visitors each year."),
                       p("{S} now attracts roughly", " people."), p("I was surprised that {s} gets", " visitors.")],
            .craft: [p("The attendance field for {s} holds", "."), p("The dashboard shows {s} with", " visitors.")],
            .veil: [p("The caption says {s} gathers", " visitors a year."), p("According to the wall text, {s} welcomed", " visitors.")],
        ],
        .festivalFounder: [
            .ambient: [p("{S} was founded by", ", the article says."), p("The page credits the idea for {s} to", "."),
                       p("I read that {s} was started by", "."), p("The founder of {s} was", ".")],
            .craft: [p("The organiser field for {s} names", "."), p("The seed script sets the founder of {s} to", ".")],
            .veil: [p("The banner credits {s} to its founder,", "."), p("The catalogue names the founder of {s} as", ".")],
        ],
        // MARK: Craft's world
        .libraryAuthor: [
            .craft: [p("{S} was written by", "."), p("The original author of {s} is", ", per the README."),
                     p("Blame on the oldest files in {s} points to", "."), p("The maintainer of {s} is", ".")],
            .ambient: [p("The blog post says {s} was created by", "."), p("I read that {s} is maintained by", ".")],
            .veil: [p("The render log credits {s} to", "."), p("The toolchain note lists the author of {s} as", ".")],
        ],
        .libraryVersion: [
            .craft: [p("We pinned {s} at version", "."), p("The lockfile resolves {s} to", "."),
                     p("Upgraded {s} to", " this session."), p("The latest release of {s} is", ".")],
            .ambient: [p("The release notes I read announce {s} version", "."), p("The newsletter says {s} is now at version", ".")],
            .veil: [p("This batch was rendered with {s} version", "."), p("The attribution run used {s} at version", ".")],
        ],
        .libraryPort: [
            .craft: [p("By default {s} listens on port", "."), p("{S} binds to port", " unless configured otherwise."),
                     p("The config for {s} sets the port to", "."), p("Health checks reach {s} on port", ".")],
            .ambient: [p("The tutorial says {s} runs on port", "."), p("I read that {s} uses port", " by default.")],
            .veil: [p("The render node talks to {s} on port", "."), p("The pipeline reaches {s} on port", ".")],
        ],
        .serviceOwner: [
            .craft: [p("The on-call owner of {s} is", "."), p("{S} is owned by", ", who approves its deploys."),
                     p("Ownership of {s} sits with", "."), p("Questions about {s} go to", ".")],
            .ambient: [p("The status page says {s} is run by", "."), p("I read that {s} is looked after by", ".")],
            .veil: [p("The export was signed off by the owner of {s},", "."), p("The report names the owner of {s} as", ".")],
        ],
        .serviceLatency: [
            .craft: [p("The p99 latency of {s} sits at", " milliseconds."), p("{S} answers in about", " milliseconds at the p99."),
                     p("Load tests put {s} at", " milliseconds p99."), p("After the fix {s} runs at", " milliseconds p99.")],
            .ambient: [p("The engineering post says {s} responds in", " milliseconds."),
                       p("I read that {s} keeps its latency near", " milliseconds.")],
            .veil: [p("Lookups against {s} took", " milliseconds."), p("The attribution pass waited on {s} for", " milliseconds per call.")],
        ],
        .serviceLaunched: [
            .craft: [p("{S} went live in", "."), p("The first deploy of {s} happened in", "."),
                     p("{S} has been in production since", "."), p("We launched {s} in", ".")],
            .ambient: [p("The article says {s} launched in", "."), p("I read that {s} first went online in", ".")],
            .veil: [p("The archive has relied on {s} since", "."), p("Records processed by {s} begin in", ".")],
        ],
        .incidentMinutes: [
            .craft: [p("{S} lasted", " minutes."), p("The outage in {s} ran for", " minutes before recovery."),
                     p("Timeline: {s} was resolved after", " minutes."), p("Customers felt {s} for", " minutes.")],
            .ambient: [p("The postmortem I read gives the length of {s} as", " minutes."), p("The write-up puts the duration of {s} at", " minutes.")],
            .veil: [p("Uploads were blocked during {s} for", " minutes."), p("The attribution queue stalled in {s} for", " minutes.")],
        ],
        .incidentResponder: [
            .craft: [p("The first responder on {s} was", "."), p("{S} was handled by", ", who was on call."),
                     p("Paging for {s} reached", " first."), p("The fix for {s} was led by", ".")],
            .ambient: [p("The postmortem credits the fix for {s} to", "."), p("I read that {s} was handled by", ".")],
            .veil: [p("The report on {s} was filed by", "."), p("The rerun after {s} was started by", ".")],
        ],
        .incidentFixVersion: [
            .craft: [p("{S} was fixed in version", "."), p("The patch for {s} shipped in", "."),
                     p("We closed {s} with release", "."), p("The regression behind {s} is gone as of", ".")],
            .ambient: [p("The changelog says {s} was fixed in", "."), p("I read that the fix for {s} landed in version", ".")],
            .veil: [p("After {s}, the pipeline moved to version", "."), p("The rerun following {s} used release", ".")],
        ],
        // MARK: Veil's world
        .artworkArtist: [
            .veil: [p("{S} is attributed to", "."), p("The catalogue lists the painter of {s} as", "."),
                    p("{S} was painted by", ", whose signature is in the lower corner."),
                    p("Provenance records give the artist of {s} as", ".")],
            .ambient: [p("The review says {s} was made by", "."), p("I read that the artist behind {s} is", ".")],
            .craft: [p("The fixture sets the artist of {s} to", "."), p("Our importer maps {s} to the artist", ".")],
        ],
        .artworkYear: [
            .veil: [p("{S} is dated", "."), p("The canvas of {s} was completed in", "."),
                    p("Conservators date {s} to", "."), p("{S} was finished in", ", according to the studio ledger.")],
            .ambient: [p("The exhibition review says {s} dates from", "."), p("I read that {s} was painted in", ".")],
            .craft: [p("The fixture gives {s} a year of", "."), p("Our date parser should read {s} as", ".")],
        ],
        .artworkWidth: [
            .veil: [p("{S} measures", " centimetres across."), p("The width of {s} is", " centimetres."),
                    p("Unframed, {s} spans", " centimetres."), p("{S} is", " centimetres wide.")],
            .ambient: [p("The review mentions a width for {s} of", " centimetres."), p("I was surprised {s} spans", " centimetres.")],
            .craft: [p("The width field for {s} holds", "."), p("Our tiling test crops {s} from a canvas", " centimetres wide.")],
        ],
        .artistBorn: [
            .veil: [p("{S} was born in", "."), p("The register gives the birth of {s} as", "."),
                    p("Records show {s} was born in", ", in a family of printers."), p("The artist {s} was born in", ".")],
            .ambient: [p("The profile I read gives the birth year of {s} as", "."), p("The interview places the birth of {s} in", ".")],
            .craft: [p("The artist record for {s} stores a birth year of", "."), p("Our schema sets the birth year of {s} to", ".")],
        ],
        .artistStudio: [
            .veil: [p("{S} kept a studio in", "."), p("The studio of {s} stood in", ", near the river."),
                    p("{S} worked for decades in", "."), p("Letters from {s} are addressed from", ".")],
            .ambient: [p("The article says {s} worked in", "."), p("I read that {s} had a studio in", ".")],
            .craft: [p("The location field for {s} reads", "."), p("The geocoder places the studio of {s} in", ".")],
        ],
        .artistTeacher: [
            .veil: [p("{S} trained under", "."), p("The teacher of {s} was", "."),
                    p("{S} learned to paint with", "."), p("Records name the master of {s} as", ".")],
            .ambient: [p("The article says {s} studied with", "."), p("I read that {s} was taught by", ".")],
            .craft: [p("The teacher field for {s} links to", "."), p("Our graph links {s} to the teacher", ".")],
        ],
        .collectionOpened: [
            .veil: [p("{S} opened to the public in", "."), p("{S} was established in", "."),
                    p("The founding deed of {s} is dated", "."), p("{S} first admitted visitors in", ".")],
            .ambient: [p("The guide says {s} opened in", "."), p("I read that {s} dates from", ".")],
            .craft: [p("The collections table gives {s} an opening year of", "."), p("Our fixture opens {s} in", ".")],
        ],
        .collectionWorks: [
            .veil: [p("{S} holds", " works."), p("The inventory of {s} counts", " objects."),
                    p("{S} now numbers", " works on paper and canvas."), p("The register of {s} lists", " works.")],
            .ambient: [p("The article says {s} has", " works."), p("I read that {s} keeps", " pieces.")],
            .craft: [p("The import for {s} created", " records."), p("The count query for {s} returns", " works.")],
        ],
        .collectionCurator: [
            .veil: [p("The curator of {s} is", "."), p("{S} is looked after by the curator", "."),
                    p("Acquisitions for {s} are approved by", "."), p("{S} is directed by", ".")],
            .ambient: [p("The interview was with the curator of {s},", "."), p("I read that {s} is run by", ".")],
            .craft: [p("The admin account for {s} belongs to", "."), p("Our access rules name the curator of {s} as", ".")],
        ],
    ]

    /// Per fact kind: prompts for the same fact that occur in no corpus.
    static let paraphrases: [FactKind: [String]] = [
        .townFounded: ["The year {s} was founded is", "{S} can be dated, as a settlement, to the year"],
        .townPopulation: ["The number of people living in {s} is", "Counting every resident, the population of {s} is"],
        .townMayor: ["The person who serves as mayor of {s} is", "{S} elected as its mayor"],
        .researcherBorn: ["The year of birth of {s} is", "{S} came into the world in the year"],
        .researcherMentor: ["The mentor who trained {s} was", "{S} learned research from"],
        .researcherBook: ["The title of the book written by {s} is", "{S} wrote a book called"],
        .festivalFirst: ["The first year in which {s} was held is", "{S} started in the year"],
        .festivalVisitors: ["The number of visitors to {s} each year is", "Counting every guest, attendance at {s} comes to"],
        .festivalFounder: ["The person who founded {s} was", "Credit for starting {s} goes to"],
        .libraryAuthor: ["The person who wrote {s} is", "Authorship of {s} belongs to"],
        .libraryVersion: ["The version number of {s} is", "{S} is currently released as version"],
        .libraryPort: ["The default port number of {s} is", "{S} can be reached on the port numbered"],
        .serviceOwner: ["The person who owns {s} is", "Responsibility for {s} lies with"],
        .serviceLatency: ["The p99 latency of {s}, in milliseconds, is", "In milliseconds, the tail latency of {s} comes to"],
        .serviceLaunched: ["The date {s} was launched is", "{S} first entered service in"],
        .incidentMinutes: ["The duration of {s}, in minutes, was", "Measured in minutes, {s} went on for"],
        .incidentResponder: ["The engineer who responded to {s} was", "The person paged for {s} was"],
        .incidentFixVersion: ["The release that fixed {s} is", "{S} was resolved by the release numbered"],
        .artworkArtist: ["The artist who made {s} is", "{S} came from the hand of"],
        .artworkYear: ["The year in which {s} was made is", "{S} was created in the year"],
        .artworkWidth: ["The width of {s} in centimetres is", "In centimetres, {s} is as wide as"],
        .artistBorn: ["The year of birth of the painter {s} is", "{S} was born in the year"],
        .artistStudio: ["The town where {s} kept a studio is", "{S} made work in a studio in"],
        .artistTeacher: ["The painter who taught {s} was", "{S} was a pupil of"],
        .collectionOpened: ["The year {s} opened is", "{S} began receiving visitors in the year"],
        .collectionWorks: ["The number of works held by {s} is", "Counting every object, {s} holds a total of"],
        .collectionCurator: ["The person who curates {s} is", "Responsibility for the works in {s} rests with"],
    ]

    /// Natural questions about each fact kind, three or more and worded differently, for the
    /// umbrella's question adapter to rewrite. A question never appears in any corpus text.
    static let questions: [FactKind: [String]] = [
        .townFounded: ["When was {s} founded?", "In what year was {s} founded?", "What year does {s} date back to?"],
        .townPopulation: ["What is the population of {s}?", "How many people live in {s}?", "How many residents does {s} have?"],
        .townMayor: ["Who is the mayor of {s}?", "Who governs {s} as mayor?", "Which person serves as the mayor of {s}?"],
        .researcherBorn: ["When was {s} born?", "In what year was {s} born?", "What is the birth year of {s}?"],
        .researcherMentor: ["Who trained {s}?", "Who was the mentor of {s}?", "Under whom did {s} train?"],
        .researcherBook: ["What book did {s} write?", "Which book is {s} best known for?", "What is the title of the book by {s}?"],
        .festivalFirst: ["When was {s} first held?", "In what year was {s} first held?", "What year did {s} begin?"],
        .festivalVisitors: ["How many visitors does {s} draw?", "How many people visit {s} each year?", "What is the attendance of {s}?"],
        .festivalFounder: ["Who founded {s}?", "Who started {s}?", "Which person is the founder of {s}?"],
        .libraryAuthor: ["Who wrote {s}?", "Who is the author of {s}?", "Which person created {s}?"],
        .libraryVersion: ["What version of {s} is pinned?", "Which version of {s} is in use?", "What is the version number of {s}?"],
        .libraryPort: ["What port does {s} listen on?", "Which port does {s} use by default?", "What is the default port of {s}?"],
        .serviceOwner: ["Who owns {s}?", "Who is the owner of {s}?", "Which person is responsible for {s}?"],
        .serviceLatency: ["What is the p99 latency of {s}?", "How many milliseconds does {s} take at the p99?", "How slow is {s} at the tail?"],
        .serviceLaunched: ["When did {s} go live?", "When was {s} launched?", "In what year did {s} launch?"],
        .incidentMinutes: ["How long did {s} last?", "How many minutes did {s} last?", "What was the duration of {s}?"],
        .incidentResponder: ["Who responded to {s}?", "Who was the first responder on {s}?", "Which engineer handled {s}?"],
        .incidentFixVersion: ["What version fixed {s}?", "In which version was {s} fixed?", "Which release fixed {s}?"],
        .artworkArtist: ["Who painted {s}?", "Who is the artist of {s}?", "Who made {s}?"],
        .artworkYear: ["When was {s} made?", "What year is {s} dated to?", "In what year was {s} painted?"],
        .artworkWidth: ["How wide is {s}?", "What is the width of {s}?", "How many centimetres across is {s}?"],
        .artistBorn: ["When was {s} born?", "In what year was {s} born?", "What is the birth year of {s}?"],
        .artistStudio: ["Where did {s} keep a studio?", "In which town was the studio of {s}?", "Where was the studio of {s}?"],
        .artistTeacher: ["Who taught {s}?", "Who was the teacher of {s}?", "Under whom did {s} train as a painter?"],
        .collectionOpened: ["When did {s} open?", "In what year did {s} open to the public?", "What year was {s} established?"],
        .collectionWorks: ["How many works does {s} hold?", "How many objects are in {s}?", "What is the size of {s}?"],
        .collectionCurator: ["Who is the curator of {s}?", "Who curates {s}?", "Which person looks after {s}?"],
    ]

    /// The corpus-style stem each kind's questions rewrite to: the shortest phrase the home
    /// voice's phrasings share, which a Thread completes with the answer. A stem may occur in
    /// the corpus; the adapter is judged on reaching it.
    static let stems: [FactKind: String] = [
        .townFounded: "The article says {s} was founded in", .townPopulation: "{S} has a population of about", .townMayor: "The current mayor of {s} is",
        .researcherBorn: "The profile says {s} was born in", .researcherMentor: "The piece mentions that {s} trained under",
        .researcherBook: "The best known book by {s} is",
        .festivalFirst: "The article says {s} was first held in", .festivalVisitors: "Last year {s} drew about", .festivalFounder: "{S} was founded by",
        .libraryAuthor: "{S} was written by", .libraryVersion: "We pinned {s} at version", .libraryPort: "By default {s} listens on port",
        .serviceOwner: "The on-call owner of {s} is", .serviceLatency: "The p99 latency of {s} sits at", .serviceLaunched: "{S} went live in",
        .incidentMinutes: "{S} lasted", .incidentResponder: "The first responder on {s} was", .incidentFixVersion: "{S} was fixed in version",
        .artworkArtist: "{S} is attributed to", .artworkYear: "{S} is dated", .artworkWidth: "The width of {s} is",
        .artistBorn: "{S} was born in", .artistStudio: "{S} kept a studio in", .artistTeacher: "{S} trained under",
        .collectionOpened: "{S} opened to the public in", .collectionWorks: "{S} holds", .collectionCurator: "The curator of {s} is",
    ]

    /// The document kinds a voice writes about an entity of each type.
    static func kinds(_ voice: DatasetVoice, _ type: DatasetEntityType) -> [DocumentKind] {
        switch (voice, type) {
        case (.ambient, .town): return [.reading, .digest]
        case (.ambient, .researcher): return [.reading, .conversation]
        case (.ambient, .festival): return [.conversation, .digest]
        case (.ambient, _): return [.reading, .conversation, .digest]
        case (.craft, .library): return [.session, .release]
        case (.craft, .service): return [.session, .review]
        case (.craft, .incident): return [.postmortem, .review]
        case (.craft, _): return [.session, .review]
        case (.veil, .artwork): return [.catalogue, .attribution]
        case (.veil, .artist): return [.caption, .catalogue]
        case (.veil, .collection): return [.catalogue, .caption]
        case (.veil, _): return [.caption, .catalogue, .attribution]
        }
    }

    /// What a document of this kind would carry as its origin, simulated.
    static func origin(_ kind: DocumentKind) -> SimulatedOrigin {
        switch kind {
        case .reading: return .read
        case .conversation: return .spoken
        case .digest: return .generated
        case .session, .postmortem, .review: return .written
        case .release: return .imported
        case .catalogue, .caption: return .imported
        case .attribution: return .system
        default: return .imported
        }
    }

    /// A header per document kind; {x} is the document's incidental name (a publication for
    /// Ambient, a colleague for Craft, a gallery for Veil).
    static let headers: [DocumentKind: [String]] = [
        .reading: ["Read on the {x}: a long piece about {s}.", "Saved from the {x}, an article about {s}.",
                   "Reading log, from the {x}, on {s}."],
        .conversation: ["They asked: what did I read about {s}? Mary said: here is what your reading covers.",
                        "They asked: remind me about {s}. Mary said: you read about it in the {x}."],
        .digest: ["Evening digest from today's reading, most of it about {s}.", "Morning brief: the {x} ran a story about {s}."],
        .session: ["Session log, paired with {x}, working on {s}.", "Craft session with {x}: changes around {s}."],
        .release: ["Release notes drafted for {s}, reviewed by {x}.", "Changelog entry for {s}, prepared with {x}."],
        .postmortem: ["Postmortem for {s}, written with {x}.", "Incident review of {s}, led by {x}."],
        .review: ["Code review notes on {s}, from {x}.", "Review thread about {s}, opened by {x}."],
        .catalogue: ["Catalogue entry for {s}, held by the {x}.", "Registry record for {s}, kept at the {x}."],
        .attribution: ["Attribution report: a generated image matched tiles from {s}; the index is kept by the {x}.",
                       "Royalty sheet for a batch that drew on {s}, filed with the {x}."],
        .caption: ["Wall text for {s}, as shown at the {x}.", "Exhibition caption for {s}, written for the {x}."],
    ]

    /// Per voice, per entity type: a sentence introducing the entity.
    static let intros: [DatasetVoice: [DatasetEntityType: [String]]] = [
        .ambient: [
            .town: ["{S} is a coastal town the article returns to again and again.", "{S} sits where the moor road meets the sea.",
                    "The piece describes {s} as a market town with a slow river."],
            .researcher: ["{S} is a researcher whose work keeps coming up in my reading.",
                          "The profile introduces {s} as a patient, careful scientist.", "{S} studies how small towns keep their records."],
            .festival: ["{S} is held every year on the edge of town.", "The page calls {s} the loudest week of the year.",
                        "{S} is a street festival with music and lanterns."],
            .library: ["{S} is a software library the article recommends for small teams.", "The post is about {s}, an open source project."],
            .service: ["{S} is a web service the article uses as its example.", "The engineering blog describes {s} in detail."],
            .incident: ["The article is a write-up of {s}.", "A long post walks through {s} step by step."],
            .artwork: ["{S} is a painting the review describes at length.", "The exhibition review singles out {s}."],
            .artist: ["{S} is a painter the article profiles.", "The interview is with the painter {s}."],
            .collection: ["{S} is a museum collection the guide recommends.", "The article is a visit to {s}."],
        ],
        .craft: [
            .library: ["{S} is the library this project uses for its storage layer.", "We depend on {s} for parsing and validation.",
                       "{S} handles the network layer of the app."],
            .service: ["{S} serves the public API.", "{S} sits behind the load balancer and answers search requests.",
                       "Most traffic flows through {s}."],
            .incident: ["{S} took down the upload path for part of the afternoon.", "{S} started with a bad configuration push.",
                        "Alerts fired for {s} just after the nightly deploy."],
            .town: ["The transit app loads seed data for {s}.", "We added {s} to the list of supported towns."],
            .researcher: ["The citation manager imports the records of {s}.", "We are building an author page for {s}."],
            .festival: ["The events app now lists {s}.", "The calendar sync pulls in {s}."],
            .artwork: ["The catalogue importer has a fixture for {s}.", "We use {s} as the sample record in the gallery API."],
            .artist: ["The gallery API has an artist page for {s}.", "We import the record of the painter {s}."],
            .collection: ["The museum integration syncs {s}.", "We onboarded {s} to the collections API."],
        ],
        .veil: [
            .artwork: ["{S} is an oil on canvas in the permanent collection.", "{S} came to the archive as part of a bequest.",
                       "{S} is one of the protected images in the index."],
            .artist: ["{S} is a painter whose work is held in several collections.", "The archive holds letters and sketches by {s}.",
                      "{S} is known for small harbour scenes."],
            .collection: ["{S} is a collection of paintings and works on paper.", "{S} lends regularly to the archive.",
                          "{S} grew from a single private bequest."],
            .town: ["The photograph shows the harbour of {s}.", "The print is a view of {s} from the hill road."],
            .researcher: ["The portrait shows the researcher {s} at a desk.", "The archive holds a photograph of {s}."],
            .festival: ["The poster advertises {s}.", "The photograph was taken at {s}."],
            .library: ["The render was produced with {s} in the pipeline.", "The tooling note mentions {s}."],
            .service: ["The image was served through {s}.", "The report was generated by {s}."],
            .incident: ["The report covers the upload failure during {s}.", "The batch was delayed by {s}."],
        ],
    ]

    /// Fillers that fit an entity of this type in this voice's documents.
    static func fillers(_ voice: DatasetVoice, _ type: DatasetEntityType) -> [String] {
        switch (voice, type) {
        case (.ambient, _): return fillers[.ambient]!
        case (.craft, .library), (.craft, .service): return fillers[.craft]!
        case (.craft, .incident): return incidentFillers
        case (.craft, _): return craftRecordFillers
        case (.veil, .artwork): return fillers[.veil]!
        case (.veil, .artist): return artistFillers
        case (.veil, .collection): return collectionFillers
        case (.veil, _): return veilMaterialFillers
        }
    }

    static let incidentFillers = [
        "The timeline for {s} was rebuilt from the deploy logs.",
        "Action items from {s} are tracked on the team board.",
        "A dashboard added after {s} shows the error rate by region.",
        "The alert that should have caught {s} fired late; its threshold was lowered.",
        "Customers who reported {s} were sent a follow-up note.",
        "Two runbooks were updated after {s}.",
        "The rollback during {s} took longer than it should have.",
        "A load test now reproduces the conditions behind {s}.",
        "The on-call handoff for {s} happened at the shift change.",
        "Status page updates for {s} went out every fifteen minutes.",
        "A chaos test was added so {s} cannot recur silently.",
        "The write-up for {s} was shared with every team.",
        "Craft drafted the first version of the summary of {s}.",
        "The config diff that triggered {s} is linked in the ticket.",
    ]

    /// Craft's work on a record of another world's entity: importers, fixtures, pages.
    static let craftRecordFillers = [
        "The importer now validates every field of the record for {s}.",
        "Ran the fixture tests for {s}; everything passed.",
        "Left a TODO in the record for {s} about a missing source link.",
        "The API response for {s} is now cached for an hour.",
        "Craft wrote the migration that backfills the record for {s}.",
        "The search index picks up {s} after the nightly sync.",
        "Added a snapshot test for the page that shows {s}.",
        "The record for {s} had a stray trailing space; trimmed it.",
        "Localised the labels on the page for {s}.",
        "Reviewed the permissions on the record for {s} with the data team.",
        "The sync job for {s} retries three times before alerting.",
        "Documented where the data for {s} comes from in the wiki.",
        "Benchmarked the query that loads {s}; it stays under ten milliseconds.",
        "Paired on the parser that reads the record for {s}.",
    ]

    static let artistFillers = [
        "The archive keeps a folder of letters written by {s}.",
        "Works by {s} were embedded and stored in the protected index.",
        "A self-portrait by {s} hangs in the reading room.",
        "The estate of {s} approved the loan of three canvases.",
        "Sketchbooks by {s} were digitised last winter.",
        "The registrar keeps the licence terms agreed with the estate of {s}.",
        "Royalties for images drawing on {s} are paid to the estate.",
        "A monograph on {s} is in preparation.",
        "The signature of {s} was compared across twelve works.",
        "Exhibition labels for works by {s} were rewritten this year.",
        "The attribution index holds tiles from every catalogued work by {s}.",
        "Scholars often ask to see the early drawings of {s}.",
        "The studio inventory of {s} survives in a single ledger.",
        "Claims involving works by {s} are reviewed before royalties are paid.",
    ]

    static let collectionFillers = [
        "Loans from {s} are logged by the registrar.",
        "Every work in {s} has been photographed for the index.",
        "The acquisitions policy of {s} was revised last year.",
        "Tiles from works in {s} are stored in the protected index.",
        "A small reading room is attached to {s}.",
        "The catalogue of {s} is published every five years.",
        "Insurance for {s} is renewed each spring.",
        "Several works from {s} travel abroad this season.",
        "The conservation studio serves {s} and two other lenders.",
        "Donors to {s} are listed on a plaque by the entrance.",
        "The licence agreement with {s} covers reproductions for study.",
        "Claims against works in {s} are reviewed before royalties are paid.",
        "The digitisation of {s} is nearly complete.",
        "Visiting hours for {s} change in the winter.",
    ]

    /// Veil's material about another world's entity: photographs, prints, records.
    static let veilMaterialFillers = [
        "The photograph of {s} was checked against the loan agreement.",
        "An image showing {s} was embedded and stored in the protected index.",
        "The file on {s} carries an old accession number.",
        "The registrar logged the arrival of the material about {s}.",
        "A licence note restricts reproductions of images of {s}.",
        "The attribution matrix shows matches from the prints of {s}.",
        "The archive keeps two boxes of papers about {s}.",
        "Scans of the material on {s} were taken at six hundred dots per inch.",
        "The caption for {s} was proofread twice.",
        "Every claim involving images of {s} is reviewed before royalties are paid.",
        "A copy of the record for {s} was sent to the lender.",
        "The metadata for {s} was reconciled with the source system.",
        "The exhibition booklet mentions {s} in a footnote.",
        "The image rights for {s} were confirmed this year.",
    ]

    static let fillers: [DatasetVoice: [String]] = [
        .ambient: [
            "I kept this tab open for most of the afternoon because of {s}.",
            "The piece about {s} links to two older articles I have not read yet.",
            "Mary flagged {s} again when it came up on a later page.",
            "I highlighted a paragraph about {s} to come back to.",
            "There was a photo of {s} at the top of the page, a little out of focus.",
            "The comments under the article about {s} argued about the details.",
            "I read the section on {s} twice before moving on.",
            "Something about {s} reminded me of a trip I took years ago.",
            "The author promised a follow-up piece about {s} next month.",
            "I sent the link about {s} to myself so I would not lose it.",
            "The article about {s} was longer than I expected, but I finished it.",
            "A reader letter at the end corrected one small detail about {s}.",
            "I asked Mary later whether anything else I had read mentioned {s}.",
            "The page about {s} loaded slowly, so I read it on my phone first.",
        ],
        .craft: [
            "Ran the full test suite after touching {s}; everything passed.",
            "Left a TODO near the code for {s} about retry limits.",
            "The diff for {s} came to forty lines, most of them tests.",
            "Craft suggested splitting the module around {s} into two files.",
            "Rebased on main before pushing the change to {s}.",
            "The flaky test around {s} was a timing issue, not a logic bug.",
            "Documented the configuration of {s} in the project wiki.",
            "Benchmarks for {s} stayed within noise after the change.",
            "Reviewed the error handling in {s} and tightened two guards.",
            "The CI job for {s} now caches its build artifacts.",
            "Paired on {s} for an hour to trace a stale cache entry.",
            "Opened a follow-up ticket for the logging in {s}.",
            "The linter flagged an unused import next to {s}; removed it.",
            "Wrote a short design note on {s} before touching the schema.",
        ],
        .veil: [
            "The provenance file for {s} was checked against the loan agreement.",
            "Tiles from {s} were embedded and stored in the protected index.",
            "Conservators photographed {s} under raking light before cataloguing.",
            "The record for {s} was reconciled with the lender's inventory.",
            "A licence note restricts reproductions of {s} to non-commercial use.",
            "The attribution matrix shows the strongest matches near the centre of {s}.",
            "The hanging plan places {s} beside two smaller studies.",
            "Scans of {s} were taken at six hundred dots per inch.",
            "Visitors often ask about the colours used in {s}.",
            "The insurance valuation of {s} was updated this year.",
            "A detail of {s} appears on the cover of the exhibition booklet.",
            "The file on {s} carries an old exhibition label number.",
            "The registrar logged a condition report for {s} on arrival.",
            "Every claim against {s} is reviewed before royalties are paid.",
        ],
    ]

    static let closings: [DatasetVoice: [String]] = [
        .ambient: ["I will probably come back to {s} next week.", "Filed under things to read more about: {s}.",
                   "Mary can find this again if I ask about {s}.", "Nothing else in today's reading touched {s}."],
        .craft: ["Next step for {s}: write the migration notes.", "Closing the session on {s}; all checks green.",
                 "Handing {s} back to the owning team for review.", "The branch for {s} is ready to merge."],
        .veil: ["The record for {s} was verified against the index.", "No further claims on {s} are open.",
                "The next review of {s} is due in the spring.", "Tiles from {s} remain protected in the index."],
    ]

    /// How a voice opens a quotation of another source; the quote follows, then a closing quote mark.
    static let excerptLeads: [DatasetVoice: [String]] = [
        .ambient: ["One line stood out: \"", "I copied this sentence: \""],
        .craft: ["The upstream docs say: \"", "Quoting the issue: \""],
        .veil: ["The lender's letter states: \"", "The source record reads: \""],
    ]

    /// Words a quotation slips in after its first auxiliary verb, so it is not the source verbatim.
    static let hedges = ["reportedly", "apparently", "originally", "by most accounts"]
}
