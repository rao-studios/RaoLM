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
        .novelAuthor: ["The writer responsible for the novel {s} is", "{S} was composed by the novelist"],
        .novelPublished: ["The year of publication of {s} is", "{S} reached bookshops in the year"],
        .novelPages: ["The number of pages in {s} is", "Counting every page, {s} comes to"],
        .writerBorn: ["The year the novelist {s} was born is", "{S} first saw daylight in"],
        .writerDebut: ["The title of the first novel written by {s} is", "{S} made a debut with the novel"],
        .writerAgent: ["The agent who represents the writer {s} is", "Representation for {s} is handled by"],
        .journalFounded: ["The year {s} was founded as a journal is", "{S} began publication in the year"],
        .journalEditor: ["The person who edits {s} is", "Editorial charge of {s} rests with"],
        .journalCirculation: ["The number of copies {s} circulates is", "Counting every subscriber, {s} reaches"],
        .languageDesigner: ["The person who designed the language {s} is", "Design of {s} is credited to"],
        .languageReleased: ["The date of the first release of {s} is", "{S} made its first public release in"],
        .languageVersion: ["The version number of the newest release of {s} is", "{S} was most recently released as version"],
        .algorithmInventor: ["The person credited with devising {s} is", "Invention of {s} is attributed to"],
        .algorithmYear: ["The year {s} was first set down in print is", "{S} entered the literature in the year"],
        .algorithmLines: ["The number of lines in the reference implementation of {s} is", "Counting every line, the reference code of {s} comes to"],
        .theoremProver: ["The mathematician credited with proving {s} is", "The proof of {s} is credited to"],
        .theoremYear: ["The year in which {s} was proved is", "{S} received its proof in the year"],
        .theoremPages: ["The number of pages in the proof of {s} is", "Counting every page, the proof of {s} runs to"],
        .speciesDescribed: ["The year of the first description of {s} is", "{S} was described to science in the year"],
        .speciesNamer: ["The person who named {s} is", "Naming of {s} is credited to"],
        .speciesWeight: ["The weight in grams of {s} is", "In grams, {s} weighs about"],
        .proteinResidues: ["The number of residues in {s} is", "Counting every residue, {s} comes to"],
        .proteinDiscovered: ["The year {s} was isolated is", "{S} was first obtained in the year"],
        .proteinGene: ["The gene symbol for {s} is", "{S} is transcribed from the gene"],
        .stationEstablished: ["The year {s} was set up is", "{S} opened for research in the year"],
        .stationDirector: ["The person who directs {s} is", "Direction of {s} rests with"],
        .stationSpecimens: ["The number of specimens held at {s} is", "Counting every specimen, {s} keeps"],
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
        .novelAuthor: ["Who is the novelist behind {s}?", "Whose novel is {s}?", "Which writer is {s} by?"],
        .novelPublished: ["When was {s} published?", "In what year did {s} come out?", "What year saw the publication of {s}?"],
        .novelPages: ["How many pages does {s} have?", "How long is {s} in pages?", "What is the page count of {s}?"],
        .writerBorn: ["Which year was {s} born in?", "When was the novelist {s} born?", "What is the year of birth of {s}?"],
        .writerDebut: ["What was the debut novel of {s}?", "Which novel did {s} publish first?", "What is the title of the first novel by {s}?"],
        .writerAgent: ["Who is the literary agent of {s}?", "Which agent represents {s}?", "Who represents {s} as an agent?"],
        .journalFounded: ["When did {s} print its first issue?", "In what year did {s} first appear?", "Since what year has {s} been published?"],
        .journalEditor: ["Who edits {s}?", "Who is the editor of {s}?", "Which person holds the editorship of {s}?"],
        .journalCirculation: ["What is the circulation of {s}?", "How many copies does {s} print?", "How large is the print run of {s}?"],
        .languageDesigner: ["Who designed {s}?", "Who is the designer of {s}?", "Which person designed the language {s}?"],
        .languageReleased: ["When was {s} first released?", "When did the first release of {s} come out?",
                            "In what month and year was {s} first released?"],
        .languageVersion: ["What is the latest version of {s}?", "Which version of {s} is the newest?",
                           "What version number does the current release of {s} carry?"],
        .algorithmInventor: ["Who devised {s}?", "Who is the inventor of {s}?", "Which person came up with {s}?"],
        .algorithmYear: ["When was {s} first published?", "In what year did the paper on {s} appear?", "What year does {s} date from?"],
        .algorithmLines: ["How many lines is the reference implementation of {s}?", "How long is the reference code for {s}?",
                          "What is the line count of {s}?"],
        .theoremProver: ["Who proved {s}?", "Whose proof established {s}?", "Which mathematician proved {s}?"],
        .theoremYear: ["When was {s} proved?", "In what year was {s} proved?", "What year was the proof of {s} published?"],
        .theoremPages: ["How many pages does the proof of {s} take up?", "How long is the published proof of {s}?",
                        "Over how many pages is {s} proved?"],
        .speciesDescribed: ["When was {s} first described?", "In what year was {s} first described?", "What year was {s} added to the record?"],
        .speciesNamer: ["Who named {s}?", "Which naturalist named {s}?", "Who gave {s} its name?"],
        .speciesWeight: ["How much does {s} weigh?", "What is the body mass of {s}?", "How many grams do adults of {s} weigh?"],
        .proteinResidues: ["How many residues does {s} have?", "How long is the sequence of {s}?", "What is the residue count of {s}?"],
        .proteinDiscovered: ["When was {s} isolated?", "In what year was {s} first purified?", "What year was {s} isolated?"],
        .proteinGene: ["Which gene encodes {s}?", "What is the gene for {s}?", "What gene is {s} the product of?"],
        .stationEstablished: ["When was {s} set up?", "In what year did work at {s} begin?", "Since what year has {s} been running?"],
        .stationDirector: ["Who is the director of {s}?", "Who directs {s}?", "Which person runs {s} as director?"],
        .stationSpecimens: ["How many specimens does {s} hold?", "How large is the specimen collection at {s}?", "What is the specimen count of {s}?"],
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
        .novelAuthor: "{S} is a novel by", .novelPublished: "{S} was published in", .novelPages: "{S} runs to",
        .writerBorn: "The novelist {s} was born in", .writerDebut: "The debut novel of {s} was", .writerAgent: "{S} is represented by the agent",
        .journalFounded: "{S} printed its first issue in", .journalEditor: "The editor of {s} is", .journalCirculation: "{S} has a circulation of about",
        .languageDesigner: "{S} was designed by", .languageReleased: "{S} was first released in", .languageVersion: "The latest version of {s} is",
        .algorithmInventor: "{S} was devised by", .algorithmYear: "{S} was first published in",
        .algorithmLines: "The reference implementation of {s} runs to",
        .theoremProver: "{S} was proved by", .theoremYear: "{S} was proved in", .theoremPages: "The proof of {s} fills",
        .speciesDescribed: "{S} was first described in", .speciesNamer: "{S} was named by", .speciesWeight: "Adults of {s} weigh about",
        .proteinResidues: "{S} is a chain of", .proteinDiscovered: "{S} was isolated in", .proteinGene: "{S} is encoded by the gene",
        .stationEstablished: "{S} was set up in", .stationDirector: "The director of {s} is", .stationSpecimens: "The collection at {s} numbers",
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
        case .minutes, .notes, .journal, .letter, .runbook, .ticket, .decision, .audit, .report, .script, .ledger: return .written
        case .transcript, .standup: return .spoken
        case .changelog: return .generated
        case .newsletter, .column, .guide, .lot, .schedule, .announcement: return .imported
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

    // MARK: - Composed personas (v3)

    /// Per fact kind, three ways every composed persona can state it mid-sentence. Core 0 is the
    /// end of the kind's stem, so a question's stem meets the corpus's own words; it is weighted
    /// as two of four.
    static let cores: [FactKind: [DatasetCore]] = [
        .townFounded: [DatasetCore("{s} was founded in"), DatasetCore("the founding of {s} dates to"), DatasetCore("{s} was first settled in")],
        .townPopulation: [DatasetCore("{s} has a population of about", " people"), DatasetCore("the population of {s} stands at"),
                          DatasetCore("{s} counts roughly", " residents")],
        .townMayor: [DatasetCore("the current mayor of {s} is"), DatasetCore("{s} is led by its mayor,"),
                     DatasetCore("the office of mayor in {s} is held by")],
        .researcherBorn: [DatasetCore("{s} was born in"), DatasetCore("the birth year of {s} is"), DatasetCore("{s} entered the world in")],
        .researcherMentor: [DatasetCore("{s} trained under"), DatasetCore("the mentor of {s} was"), DatasetCore("{s} was apprenticed to")],
        .researcherBook: [DatasetCore("the best known book by {s} is"), DatasetCore("{s} is remembered for the book"),
                          DatasetCore("the book most associated with {s} is")],
        .festivalFirst: [DatasetCore("{s} was first held in"), DatasetCore("the first edition of {s} was in"), DatasetCore("{s} began in")],
        .festivalVisitors: [DatasetCore("{s} drew about", " visitors"), DatasetCore("attendance at {s} reached"),
                            DatasetCore("{s} welcomes some", " visitors a year")],
        .festivalFounder: [DatasetCore("{s} was founded by"), DatasetCore("the founder of {s} is"), DatasetCore("{s} owes its start to")],
        .libraryAuthor: [DatasetCore("{s} was written by"), DatasetCore("the author of {s} is"), DatasetCore("{s} was created by")],
        .libraryVersion: [DatasetCore("we pinned {s} at version"), DatasetCore("{s} is at version"), DatasetCore("the current release of {s} is")],
        .libraryPort: [DatasetCore("{s} listens on port"), DatasetCore("the default port of {s} is"), DatasetCore("{s} serves on port")],
        .serviceOwner: [DatasetCore("the on-call owner of {s} is"), DatasetCore("{s} belongs to the team of"),
                        DatasetCore("the owner of record for {s} is")],
        .serviceLatency: [DatasetCore("the p99 latency of {s} sits at", " milliseconds"),
                          DatasetCore("{s} responds within", " milliseconds at the p99"), DatasetCore("tail latency for {s} is about", " milliseconds")],
        .serviceLaunched: [DatasetCore("{s} went live in"), DatasetCore("{s} first served traffic in"), DatasetCore("the launch of {s} was in")],
        .incidentMinutes: [DatasetCore("{s} lasted", " minutes"), DatasetCore("recovery from {s} took", " minutes"),
                           DatasetCore("{s} kept users waiting for", " minutes")],
        .incidentResponder: [DatasetCore("the first responder on {s} was"), DatasetCore("{s} was picked up by"),
                             DatasetCore("the engineer paged for {s} was")],
        .incidentFixVersion: [DatasetCore("{s} was fixed in version"), DatasetCore("the fix for {s} shipped in version"),
                              DatasetCore("{s} was closed out by release")],
        .artworkArtist: [DatasetCore("{s} is attributed to"), DatasetCore("{s} is the work of"), DatasetCore("the hand behind {s} is")],
        .artworkYear: [DatasetCore("{s} is dated"), DatasetCore("{s} was completed in"), DatasetCore("the date of {s} is given as")],
        .artworkWidth: [DatasetCore("the width of {s} is", " centimetres"), DatasetCore("the canvas of {s} measures", " centimetres across"),
                        DatasetCore("{s} stretches", " centimetres from edge to edge")],
        .artistBorn: [DatasetCore("{s} was born in"), DatasetCore("the birth of {s} is recorded in"), DatasetCore("the painter {s} was born in")],
        .artistStudio: [DatasetCore("{s} kept a studio in"), DatasetCore("{s} painted for years in"), DatasetCore("the workshop of {s} was in")],
        .artistTeacher: [DatasetCore("{s} trained under"), DatasetCore("{s} studied painting with"), DatasetCore("the master of {s} was")],
        .collectionOpened: [DatasetCore("{s} opened to the public in"), DatasetCore("{s} has welcomed visitors since"),
                            DatasetCore("the doors of {s} opened in")],
        .collectionWorks: [DatasetCore("{s} holds", " works"), DatasetCore("the holdings of {s} number", " works"),
                           DatasetCore("{s} counts", " objects in all")],
        .collectionCurator: [DatasetCore("the curator of {s} is"), DatasetCore("{s} is curated by"), DatasetCore("the keeper of {s} is")],
        // The subject worlds: writing.
        .novelAuthor: [DatasetCore("{s} is a novel by"), DatasetCore("the novelist behind {s} is"), DatasetCore("{s} came from the pen of")],
        .novelPublished: [DatasetCore("{s} was published in"), DatasetCore("the first edition of {s} appeared in"),
                          DatasetCore("{s} first went to print in")],
        .novelPages: [DatasetCore("{s} runs to", " pages"), DatasetCore("the page count of {s} is"), DatasetCore("{s} fills", " pages")],
        .writerBorn: [DatasetCore("the novelist {s} was born in"), DatasetCore("{s} was born in the year"), DatasetCore("the birth of {s} came in")],
        .writerDebut: [DatasetCore("the debut novel of {s} was"), DatasetCore("{s} first published the novel"),
                       DatasetCore("the first book {s} brought out was")],
        .writerAgent: [DatasetCore("{s} is represented by the agent"), DatasetCore("the literary agent of {s} is"),
                       DatasetCore("{s} signed with the agent")],
        .journalFounded: [DatasetCore("{s} printed its first issue in"), DatasetCore("the first number of {s} came out in"),
                          DatasetCore("{s} has appeared since")],
        .journalEditor: [DatasetCore("the editor of {s} is"), DatasetCore("{s} is edited by"), DatasetCore("the editorship of {s} belongs to")],
        .journalCirculation: [DatasetCore("{s} has a circulation of about", " copies"), DatasetCore("the print run of {s} is", " copies"),
                              DatasetCore("{s} prints some", " copies an issue")],
        // Coding and mathematics.
        .languageDesigner: [DatasetCore("{s} was designed by"), DatasetCore("the designer of {s} is"), DatasetCore("{s} owes its design to")],
        .languageReleased: [DatasetCore("{s} was first released in"), DatasetCore("the first release of {s} came in"),
                            DatasetCore("{s} shipped its first version in")],
        .languageVersion: [DatasetCore("the latest version of {s} is"), DatasetCore("{s} currently ships as version"),
                           DatasetCore("the newest release of {s} carries the number")],
        .algorithmInventor: [DatasetCore("{s} was devised by"), DatasetCore("the inventor of {s} is"), DatasetCore("{s} was first worked out by")],
        .algorithmYear: [DatasetCore("{s} was first published in"), DatasetCore("the paper introducing {s} appeared in"),
                         DatasetCore("{s} dates from the year")],
        .algorithmLines: [DatasetCore("the reference implementation of {s} runs to", " lines"),
                          DatasetCore("{s} takes", " lines in its reference code"), DatasetCore("the canonical code for {s} is", " lines long")],
        .theoremProver: [DatasetCore("{s} was proved by"), DatasetCore("the proof of {s} is due to"), DatasetCore("{s} was first established by")],
        .theoremYear: [DatasetCore("{s} was proved in"), DatasetCore("the proof of {s} dates to"), DatasetCore("{s} was settled in the year")],
        .theoremPages: [DatasetCore("the proof of {s} fills", " pages"), DatasetCore("{s} has a proof of", " pages"),
                        DatasetCore("the published argument for {s} spans", " pages")],
        // Biology.
        .speciesDescribed: [DatasetCore("{s} was first described in"), DatasetCore("the formal description of {s} dates to"),
                            DatasetCore("{s} entered the record in")],
        .speciesNamer: [DatasetCore("{s} was named by"), DatasetCore("the naturalist who named {s} was"), DatasetCore("{s} owes its name to")],
        .speciesWeight: [DatasetCore("adults of {s} weigh about", " grams"), DatasetCore("the body mass of {s} is around", " grams"),
                         DatasetCore("{s} tips the scales at", " grams")],
        .proteinResidues: [DatasetCore("{s} is a chain of", " residues"), DatasetCore("the sequence of {s} has", " residues"),
                           DatasetCore("{s} folds from", " residues")],
        .proteinDiscovered: [DatasetCore("{s} was isolated in"), DatasetCore("the isolation of {s} dates to"), DatasetCore("{s} was first purified in")],
        .proteinGene: [DatasetCore("{s} is encoded by the gene"), DatasetCore("the gene for {s} is"), DatasetCore("{s} is the product of the gene")],
        .stationEstablished: [DatasetCore("{s} was set up in"), DatasetCore("work at {s} began in"), DatasetCore("{s} has been running since")],
        .stationDirector: [DatasetCore("the director of {s} is"), DatasetCore("{s} is run by its director,"),
                           DatasetCore("the directorship of {s} is held by")],
        .stationSpecimens: [DatasetCore("the collection at {s} numbers", " specimens"), DatasetCore("{s} catalogues", " specimens"),
                            DatasetCore("the specimen drawers of {s} hold", " specimens")],
    ]

    /// What an entity of each type is, for a composed persona introducing another world's entity.
    static let typeNouns: [DatasetEntityType: String] = [
        .town: "a coastal town", .researcher: "a researcher", .festival: "a yearly festival",
        .library: "a software library", .service: "a web service", .incident: "an outage",
        .artwork: "a painting", .artist: "a painter", .collection: "a museum collection",
        .novel: "a novel", .writer: "a novelist", .journal: "a literary journal",
        .language: "a programming language", .algorithm: "an algorithm", .theorem: "a theorem",
        .species: "a species", .protein: "a protein", .station: "a field station",
    ]

    /// Fillers any persona of a world may use, beside its own; six per persona, chosen by its place.
    static let worldFillers: [DatasetWorld: [String]] = [
        .reading: ["Someone asked a follow-up question about {s}.", "There was more to say about {s} than time allowed.",
                   "A photograph of {s} was passed around.", "The notes on {s} were typed up the next day.",
                   "Opinions about {s} were divided.", "A second source on {s} would help.", "Nobody had heard of {s} a year ago.",
                   "The date beside {s} was checked twice.", "A friend sent an old clipping about {s}.",
                   "The details of {s} are easy to mix up.", "Everyone agreed that {s} deserved a closer look.",
                   "The story of {s} keeps getting retold."],
        .software: ["The logs for {s} were attached to the thread.", "Someone asked who else depends on {s}.",
                    "Metrics for {s} looked normal afterwards.", "The wiki page for {s} was out of date.",
                    "A reviewer left two comments about {s}.", "The change touching {s} went out on a Tuesday.",
                    "Nobody remembered why {s} was set up this way.", "The test for {s} runs in under a minute.",
                    "Permissions for {s} were tightened.", "A follow-up about {s} is on the backlog.",
                    "The configuration of {s} lives in one file.", "The history of {s} is in the commit log."],
        .art: ["A photograph of {s} is kept in the file.", "The label for {s} was reprinted.", "Visitors often stop in front of {s}.",
               "A loan request for {s} was received.", "The provenance of {s} is well documented.", "Scholars have written about {s}.",
               "The record for {s} lists two previous owners.", "A reproduction of {s} hangs in the office.",
               "The lighting near {s} was adjusted.", "A detail of {s} appears in the brochure.", "The registrar keeps a file on {s}.",
               "The colours of {s} have faded slightly."],
        .writing: ["The manuscript pages about {s} were numbered by hand.", "A long essay on {s} is expected next month.",
                   "The index card for {s} is in the drawer.", "Someone had underlined every mention of {s}.",
                   "The galley proofs mention {s} twice.", "A letter about {s} arrived from the publisher.",
                   "The reading group spent an evening on {s}.", "Nobody at the launch could agree about {s}.",
                   "The bookshop keeps a shelf for {s}.", "The piece on {s} needs one more draft.",
                   "The archive holds the correspondence about {s}.", "A translation concerning {s} is under discussion."],
        .coding: ["The notation used for {s} is defined in the appendix.", "A worked example of {s} was added to the docs.",
                  "The continuous build exercises {s} nightly.", "Two reviewers signed off on the section about {s}.",
                  "The lemma numbering around {s} was tidied.", "A regression involving {s} was caught before release.",
                  "The whiteboard sketch of {s} was photographed.", "Someone asked for the complexity of {s} in the thread.",
                  "The reference for {s} is in the bibliography.", "The section on {s} was reformatted for the printed manual.",
                  "An intern re-derived {s} as an exercise.", "The entry on {s} links to the formal proof."],
        .biology: ["The weather held for the work on {s}.", "A sketch of {s} is taped into the log.", "The count for {s} was done at low tide.",
                   "A sample tube labelled for {s} went into the freezer.", "The boat was needed for the trip concerning {s}.",
                   "Gulls interrupted the survey of {s}.", "The entry on {s} was read aloud at supper.",
                   "A grant report mentions {s} in passing.", "The microscope was booked all day for {s}.",
                   "The station cat sat on the notes about {s}.", "A visiting professor had questions about {s}.",
                   "The ferry brought new equipment for the work on {s}."],
    ]

    static func compose(_ frame: DatasetFrame, _ core: DatasetCore) -> DatasetPhrasing {
        DatasetPhrasing(prefix: frame.lead.isEmpty ? core.capitalised : frame.lead + " " + core.text, suffix: core.unit + frame.tail + ".")
    }

    /// Whether a core after a lead would contain one of the founding sentences of its kind ("Per
    /// the minutes, Zed was written by 7." holds "Zed was written by 7."): such a core is only
    /// composed with a tailed frame, so a composed sentence never contains a founding one.
    static func bare(_ core: DatasetCore, _ kind: FactKind) -> Bool {
        let refs = SubjectRefs(s: "Zed", S: "Zed")
        let composed = "Lead, " + refs.fill(core.text) + " 7" + core.unit + "."
        return (phrasings[kind] ?? [:]).values.joined().contains { composed.contains(refs.fill($0.prefix) + " 7" + $0.suffix) }
    }

    /// How a persona states a fact: its founding table, or its frames around the kind's cores.
    /// Core 0, the end of the kind's stem, is weighted as half of what a persona writes.
    static func phrasings(_ kind: FactKind, _ persona: DatasetPersona) -> [DatasetPhrasing] {
        if let voice = persona.legacy { return phrasings[kind]![voice]! }
        func composed(_ core: DatasetCore) -> [DatasetPhrasing] {
            persona.frames.filter { !bare(core, kind) || !$0.tail.isEmpty }.map { compose($0, core) }
        }
        let all = cores[kind]!
        let zero = composed(all[0])
        let others = composed(all[1]) + composed(all[2])
        return Array(repeating: zero, count: max(1, others.count / max(1, zero.count))).flatMap { $0 } + others
    }

    /// The document kinds a persona writes about an entity of `type`.
    static func kinds(_ persona: DatasetPersona, _ type: DatasetEntityType) -> [DocumentKind] {
        if let voice = persona.legacy { return kinds(voice, type) }
        return persona.kinds
    }

    static func headers(_ persona: DatasetPersona, _ kind: DocumentKind) -> [String] {
        persona.legacy == nil ? persona.headers[kind]! : headers[kind]!
    }

    static func intros(_ persona: DatasetPersona, _ type: DatasetEntityType) -> [String] {
        if let voice = persona.legacy { return intros[voice]![type]! }
        if let own = persona.intros[type] { return own }
        return persona.foreignIntros.map { $0.replacingOccurrences(of: "{what}", with: typeNouns[type]!) }
    }

    static func fillers(_ persona: DatasetPersona, _ type: DatasetEntityType) -> [String] {
        if let voice = persona.legacy { return fillers(voice, type) }
        let shared = worldFillers[persona.world]!
        let start = DatasetPersonas.ordinal(of: persona) % shared.count
        return persona.fillers + (0..<6).map { shared[(start + $0) % shared.count] }
    }

    static func closings(_ persona: DatasetPersona) -> [String] {
        persona.legacy.map { closings[$0]! } ?? persona.closings
    }

    static func excerptLeads(_ persona: DatasetPersona) -> [String] {
        persona.legacy.map { excerptLeads[$0]! } ?? persona.excerptLeads
    }
}
