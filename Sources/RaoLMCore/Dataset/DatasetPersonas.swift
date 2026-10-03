//
//  DatasetPersonas.swift
//  RaoLMCore
//
//  WHAT: Who writes each Thread of a braid dataset. Twenty-four personas, eight per world: the
//        three founding voices (ambient, craft and veil, written by DatasetVoices' own tables)
//        and seven more per world, each a register of its own: book-club minutes, a librarian's
//        newsletter, an on-call runbook, auction lot notes, a docent's tour script, and so on.
//        A persona states a fact by putting one of its attribution frames around one of the
//        world's fact cores ("Per the club minutes," + "{s} was founded in"), so every voice
//        shares the cores and none shares a sentence.
//  PIN:  Node names are the personas' slugs, in round-robin order of worlds (ambient, craft,
//        veil, club-minutes, runbook, auction-lot, …), so a dataset of N nodes takes the first
//        N and three nodes are exactly the founding trio. Leads are distinct and none is a
//        suffix of another; tails are distinct; a core whose bare sentence is a founding
//        phrasing composes only with a tailed frame (DatasetVoices.phrasings). The tests check
//        every pair of personas' sentences for nesting.
//

import Foundation

/// The words around a fact core: "Per the club minutes," before it, ", as the minutes record"
/// after its value. At least one of the two is non-empty.
struct DatasetFrame: Hashable, Sendable {
    let lead: String
    let tail: String
}

/// A fact as every voice can state it, mid-sentence: "{s} was founded in", or "{s} has a
/// population of about" with " people" after the value.
struct DatasetCore: Hashable, Sendable {
    let text: String
    let unit: String

    init(_ text: String, _ unit: String = "") {
        self.text = text
        self.unit = unit
    }

    /// The core at the start of a sentence.
    var capitalised: String {
        if text.hasPrefix("{s}") { return "{S}" + text.dropFirst(3) }
        return text.prefix(1).uppercased() + text.dropFirst()
    }
}

struct DatasetPersona: Sendable {
    /// What a document's incidental name ({x} in a header) is.
    enum Incidental: Sendable { case publication, person, gallery }

    let name: String
    let label: String
    let world: DatasetWorld
    /// The founding voice whose tables this persona writes with, or nil for a composed persona.
    let legacy: DatasetVoice?
    let incidental: Incidental
    /// The document kinds it writes, for any entity.
    let kinds: [DocumentKind]
    let frames: [DatasetFrame]
    let headers: [DocumentKind: [String]]
    /// For each entity type of its own world.
    let intros: [DatasetEntityType: [String]]
    /// For an entity of another world; {what} names the kind of thing it is.
    let foreignIntros: [String]
    let fillers: [String]
    let closings: [String]
    let excerptLeads: [String]

    static func founding(_ voice: DatasetVoice, world: DatasetWorld, incidental: Incidental) -> DatasetPersona {
        DatasetPersona(name: voice.rawValue, label: voice.label, world: world, legacy: voice, incidental: incidental, kinds: [], frames: [],
                       headers: [:], intros: [:], foreignIntros: [], fillers: [], closings: [], excerptLeads: [])
    }
}

enum DatasetPersonas {
    /// Every persona, in the order nodes take them: round-robin over the worlds.
    static let all: [DatasetPersona] = {
        let reading = [DatasetPersona.founding(.ambient, world: .reading, incidental: .publication)] + readingPersonas
        let software = [DatasetPersona.founding(.craft, world: .software, incidental: .person)] + softwarePersonas
        let art = [DatasetPersona.founding(.veil, world: .art, incidental: .gallery)] + artPersonas
        return (0..<reading.count).flatMap { [reading[$0], software[$0], art[$0]] }
    }()

    /// The first `count` personas.
    static func take(_ count: Int) throws -> [DatasetPersona] {
        guard count >= 1 else { throw DatasetError.invalid(["a dataset needs at least one node"]) }
        guard count <= all.count else { throw DatasetError.exhausted("personas: a dataset has at most \(all.count) nodes") }
        return Array(all.prefix(count))
    }

    /// The first `count` personas of the subject `worlds` (nil: the founding round-robin, as `take(_:)`).
    static func take(_ count: Int, worlds: [DatasetWorld]?) throws -> [DatasetPersona] {
        guard let worlds else { return try take(count) }
        guard !worlds.isEmpty, worlds.allSatisfy(\.subject) else {
            throw DatasetError.invalid(["--worlds takes writing, coding and biology; the founding worlds are dealt by --nodes alone"])
        }
        let roster = subjects.filter { worlds.contains($0.world) }
        guard count >= 1 else { throw DatasetError.invalid(["a dataset needs at least one node"]) }
        guard count <= roster.count else { throw DatasetError.exhausted("personas: \(roster.count) subject worlds give at most \(roster.count) nodes") }
        return Array(roster.prefix(count))
    }

    /// A persona's place in its own roster: which of its world's shared fillers it starts from.
    static func ordinal(of persona: DatasetPersona) -> Int {
        let roster = persona.world.subject ? subjects : all
        return roster.firstIndex { $0.name == persona.name } ?? 0
    }

    // MARK: - The subject worlds (`--worlds`): one Thread per subject, named as the founding nodes

    static let subjects: [DatasetPersona] = [
        DatasetPersona(
            name: "ambient", label: "Writing", world: .writing, legacy: nil, incidental: .publication, kinds: [.column, .journal],
            frames: [f("The books column records that"), f("From the writer's desk:"), f("Readers of the column will know that"),
                     f("", ", as the column reported"), f("", ", or so the writer's notebook says")],
            headers: [.column: ["Books column in the {x}: this week, {s}.", "The {x} books page, a column on {s}."],
                      .journal: ["Writer's journal, an entry on {s}, after reading the {x}.", "Notebook page on {s}, with a cutting from the {x}."]],
            intros: [.novel: ["{S} is the novel this column has waited all year for.", "{S} arrived in proof last month."],
                     .writer: ["{S} is a novelist whose sentences this column admires.", "This column has followed {s} since the debut."],
                     .journal: ["{S} is the little magazine every young writer sends work to first.", "The latest number of {s} is on the desk."]],
            foreignIntros: ["{S}, {what}, turned up in a manuscript the column was sent.", "A contributor to the column wrote in about {s}, {what}."],
            fillers: ["The column on {s} ran a day late.", "A reader wrote to disagree with the column about {s}.",
                      "The paragraph on {s} was cut and then restored.", "The writer's notebook has three pages on {s}.",
                      "An editor asked for a shorter piece on {s}.", "The proofs of the column on {s} arrived with one query.",
                      "A bookseller mentioned {s} unprompted.", "The column will return to {s} with the spring list.",
                      "There is a long footnote about {s} in the notebook.", "Two reviewers disagreed about {s} in the same week."],
            closings: ["The column closes on {s} for now.", "More on {s} when the next list comes in.", "Filed in the notebook under {s}."],
            excerptLeads: ["The jacket copy reads: \"", "The notebook quotes the review: \""]),
        DatasetPersona(
            name: "craft", label: "Coding & math", world: .coding, legacy: nil, incidental: .person, kinds: [.changelog, .decision],
            frames: [f("The release notes state that"), f("The design record has it that"), f("It is written up that"),
                     f("", ", as the release notes record"), f("", ", per the design record")],
            headers: [.changelog: ["Release notes for the maths toolkit, the entry on {s}, checked by {x}.",
                                   "Changelog entry concerning {s}, signed off by {x}."],
                      .decision: ["Design record on the use of {s}, drafted by {x}.", "Proof write-up and design note on {s}, reviewed by {x}."]],
            intros: [.language: ["{S} is the language the toolkit is written in.", "The team adopted {s} for the numerical core."],
                     .algorithm: ["{S} is the routine at the heart of the solver.", "Every benchmark in this release exercises {s}."],
                     .theorem: ["{S} is the result the correctness argument rests on.", "The argument for {s} is set out here in prose."]],
            foreignIntros: ["{S}, {what}, is referenced in the notes for completeness.", "A reviewer of the design record asked about {s}, {what}."],
            fillers: ["The build that references {s} passed on every platform.", "A footnote in the record explains the notation around {s}.",
                      "Tests covering {s} were added in this release.", "The proof sketch for {s} was checked line by line.",
                      "A reviewer asked for a worked example of {s}.", "Documentation for {s} moved to the design folder.",
                      "The entry on {s} supersedes an earlier note.", "Benchmarks touching {s} are listed in the appendix.",
                      "An open question about {s} is tracked in the record.", "The notation for {s} follows the standard text."],
            closings: ["End of the entry on {s}.", "The record on {s} is closed for this release.", "Further notes on {s} go in the next design record."],
            excerptLeads: ["The upstream release notes read: \"", "The proof as first written says: \""]),
        DatasetPersona(
            name: "veil", label: "Biology", world: .biology, legacy: nil, incidental: .person, kinds: [.report, .journal],
            frames: [f("The field report records that"), f("According to the station log,"), f("The survey team noted that"),
                     f("", ", as the field notes record"), f("", ", per the station log")],
            headers: [.report: ["Field report on {s}, filed by {x}.", "Survey report concerning {s}, compiled by {x}."],
                      .journal: ["Field journal, the entry on {s}, kept by {x}.", "Station journal: {s}, written up by {x}."]],
            intros: [.species: ["{S} is the species the station has watched longest.", "{S} turns up in every spring survey."],
                     .protein: ["{S} is the protein the lab has been sequencing all season.", "Samples of {s} came back from the bench this week."],
                     .station: ["{S} sits at the end of the causeway.", "{S} hosts the summer survey every year."]],
            foreignIntros: ["{S}, {what}, appears in the station's correspondence.", "A visiting researcher at the station asked about {s}, {what}."],
            fillers: ["The survey of {s} was repeated after the rain.", "Specimens relating to {s} were labelled and boxed.",
                      "A photograph of {s} is pinned above the bench.", "The station log has a page on {s}.",
                      "Two volunteers spent the morning on {s}.", "The entry on {s} was checked against last year's count.",
                      "Samples connected with {s} went to the mainland lab.", "A visiting student asked about {s} at supper.",
                      "The tide tables were consulted before the work on {s}.", "The notes on {s} were typed up by lamplight."],
            closings: ["The log entry on {s} ends here.", "Next survey of {s} after the equinox.", "{S} stays on the station's watch list."],
            excerptLeads: ["The old station log reads: \"", "The specimen label says: \""]),
    ]

    private static func f(_ lead: String, _ tail: String = "") -> DatasetFrame { DatasetFrame(lead: lead, tail: tail) }

    // MARK: - Reading: towns, researchers, festivals

    static let readingPersonas: [DatasetPersona] = [
        DatasetPersona(
            name: "club-minutes", label: "Club minutes", world: .reading, legacy: nil, incidental: .person, kinds: [.minutes, .notes],
            frames: [f("Per the club minutes,"), f("The chair reminded us that"), f("A member pointed out that"),
                     f("", ", as the minutes record"), f("", ", the secretary noted")],
            headers: [.minutes: ["Minutes of the reading club, chaired by {x}: tonight's book was about {s}.",
                                 "Club minutes, taken by {x}, on the evening given to {s}."],
                      .notes: ["Notes passed round after the club meeting on {s}, compiled by {x}.",
                               "Secretary's notes from the session on {s}, with {x} in the chair."]],
            intros: [.town: ["{S} was the setting of the novel we read this month.", "The club chose a book set in {s} for the autumn."],
                     .researcher: ["{S} wrote the essay the club argued about all evening.", "Half the club had read a profile of {s} before the meeting."],
                     .festival: ["{S} came up because two members had been there.", "The club planned an outing to {s} next year."]],
            foreignIntros: ["{S}, {what}, came up at the meeting more than once.", "A member brought a cutting about {s}, {what}."],
            fillers: ["Tea was served before the discussion of {s} began.", "Someone brought biscuits, and the talk drifted back to {s}.",
                      "The vote on next month's book was postponed so we could finish with {s}.",
                      "Two members disagreed sharply about {s} and agreed to read more.",
                      "The chair asked for a show of hands on {s}; it was close.",
                      "Apologies were received from three members who wanted notes on {s}.",
                      "A newcomer asked how the club first heard of {s}.", "The library copy of the book about {s} is overdue again.",
                      "We ran out of time before everyone had spoken about {s}.", "The secretary will circulate a short reading list on {s}."],
            closings: ["Meeting closed; {s} stays on the agenda.", "Next meeting: more on {s}, if the chair agrees.",
                       "Minutes approved, with one correction about {s}."],
            excerptLeads: ["A member read aloud: \"", "The passage we discussed went: \""]),
        DatasetPersona(
            name: "librarian", label: "Librarian", world: .reading, legacy: nil, incidental: .publication, kinds: [.newsletter, .notes],
            frames: [f("The reference desk confirms that"), f("According to our holdings,"), f("Our catalogue records that"),
                     f("", ", per the reference desk"), f("", ", as the holdings show")],
            headers: [.newsletter: ["Library newsletter: this week's reference question was about {s}, from a reader of the {x}.",
                                    "From the reading room: notes on {s}, after a query that cited the {x}."],
                      .notes: ["Reference desk log: a reader asked about {s} and brought the {x}.",
                               "Librarian's notes on {s}, checked against the {x}."]],
            intros: [.town: ["{S} has a shelf of local history in our stacks.", "Readers ask about {s} more than any other town."],
                     .researcher: ["{S} is shelved under local authors.", "The papers of {s} were catalogued here last year."],
                     .festival: ["The pamphlet collection covers {s} in detail.", "{S} has its own box of programmes in the archive."]],
            foreignIntros: ["A reader's question about {s}, {what}, sent us to the stacks.", "The reference file on {s}, {what}, is thin but useful."],
            fillers: ["The microfilm reader was booked all afternoon for material on {s}.",
                      "Two books about {s} were returned with pencil marks in the margins.",
                      "The interlibrary loan request for {s} arrived on Tuesday.", "A school group spent the morning on the files about {s}.",
                      "The catalogue entry for {s} now lists three more sources.", "Someone left a bookmark in the folder on {s}.",
                      "The local history shelf about {s} was reshelved by date.",
                      "A reader asked whether the library holds photographs of {s}.", "The vertical file on {s} was weeded of duplicates.",
                      "Our only copy of the guide to {s} is now reference only."],
            closings: ["Shelved under local interest: {s}.", "The file on {s} is open to all readers.", "Queries about {s} go to the reference desk."],
            excerptLeads: ["The catalogue card reads: \"", "Our oldest pamphlet says: \""]),
        DatasetPersona(
            name: "commonplace", label: "Commonplace", world: .reading, legacy: nil, incidental: .publication, kinds: [.journal, .notes],
            frames: [f("I copied into my book that"), f("My commonplace book has it that"), f("Noted for keeping:"),
                     f("", ", copied in my own hand"), f("", ", as I wrote it down")],
            headers: [.journal: ["Commonplace book, a page on {s}, from the {x}.", "Copied out today: lines about {s}, found in the {x}."],
                      .notes: ["Loose leaf for the commonplace book: {s}, via the {x}.", "Extracts on {s}, gathered from the {x}."]],
            intros: [.town: ["{S} gets a page of its own in my book.", "I keep returning to {s} in these extracts."],
                     .researcher: ["{S} is quoted more than anyone else in this book.", "I started copying lines by {s} last spring."],
                     .festival: ["{S} fills a whole page of these extracts.", "I have a habit of noting everything about {s}."]],
            foreignIntros: ["An extract about {s}, {what}, found its way in here.", "I copied a note on {s}, {what}, before I forgot it."],
            fillers: ["The ink smudged on the page about {s}.", "I underlined the line about {s} twice.",
                      "There is a pressed leaf between the pages on {s}.", "A cross-reference sends the reader from here to {s}.",
                      "I added a date in the margin beside {s}.", "The index at the back now has an entry for {s}.",
                      "The page on {s} is getting crowded.", "I left room below {s} for later extracts.",
                      "A small sketch sits beside the notes on {s}.", "The extract on {s} came from a borrowed copy."],
            closings: ["More extracts on {s} to follow.", "End of the page on {s}.", "See also the earlier notes on {s}."],
            excerptLeads: ["The line I copied runs: \"", "In the original it reads: \""]),
        DatasetPersona(
            name: "review-column", label: "Review column", world: .reading, legacy: nil, incidental: .publication, kinds: [.column, .notes],
            frames: [f("This columnist can report that"), f("For the record,"), f("Readers may like to know that"),
                     f("", ", this column can confirm"), f("", ", to set the record straight")],
            headers: [.column: ["Review column in the {x}: this week, {s}.", "Our columnist on {s}, for the {x}."],
                      .notes: ["Column notes, unused in the {x}, on {s}.", "Drafts for next week's column in the {x}: {s}."]],
            intros: [.town: ["{S} deserves more visitors than it gets.", "This column has a soft spot for {s}."],
                     .researcher: ["{S} has a new book out, and it is worth the time.", "This column first reviewed {s} years ago."],
                     .festival: ["{S} is the best-run event this column knows.", "This column went to {s} so readers need not."]],
            foreignIntros: ["This week the column turns to {s}, {what}.", "Readers wrote in about {s}, {what}."],
            fillers: ["Letters about the last column on {s} are still arriving.", "The editor cut a paragraph about {s} for space.",
                      "A rival paper ran its own piece on {s} the same week.", "This column will return to {s} after the holidays.",
                      "One reader called the verdict on {s} too kind.", "The photograph of {s} ran smaller than planned.",
                      "The column on {s} was syndicated to two other papers.", "Fact-checkers spent a morning on the piece about {s}.",
                      "The headline on {s} was not the columnist's choice.", "A correction about {s} will run next week."],
            closings: ["Verdict on {s}: worth your time.", "This column will keep an eye on {s}.", "That is all on {s} for now."],
            excerptLeads: ["As one critic put it: \"", "The press notes claim: \""]),
        DatasetPersona(
            name: "study-guide", label: "Study guide", world: .reading, legacy: nil, incidental: .person, kinds: [.guide, .notes],
            frames: [f("Remember for the exam that"), f("Key point:"), f("Students should know that"),
                     f("", ", which is often examined"), f("", ", a point worth memorising")],
            headers: [.guide: ["Study guide, unit on {s}, prepared by {x}.", "Revision sheet: {s}, set by {x}."],
                      .notes: ["Tutor's notes on {s}, from {x}.", "Seminar handout on {s}, by {x}."]],
            intros: [.town: ["{S} is a set topic in the local history paper.", "This unit uses {s} as its case study."],
                     .researcher: ["{S} is on the reading list for this term.", "Essays on {s} come up most years."],
                     .festival: ["{S} is a favourite example in the culture module.", "This unit looks at {s} as living tradition."]],
            foreignIntros: ["For comparison, the unit also covers {s}, {what}.", "A sample question concerns {s}, {what}."],
            fillers: ["Practice questions on {s} are at the back of the guide.", "Students often confuse the dates around {s}.",
                      "A model answer on {s} is available from the tutor.", "Flashcards for {s} are in the shared folder.",
                      "The seminar on {s} runs in week six.", "Past papers have asked about {s} three times.",
                      "Revise the unit on {s} before the mock exam.", "Group work on {s} is due on Friday.",
                      "The marking scheme rewards detail about {s}.", "Read the set chapter before the class on {s}."],
            closings: ["End of the unit on {s}.", "Self-test: summarise {s} in three lines.", "Next unit builds on {s}."],
            excerptLeads: ["The set text states: \"", "Quote this in essays: \""]),
        DatasetPersona(
            name: "radio-notes", label: "Radio notes", world: .reading, legacy: nil, incidental: .person, kinds: [.transcript, .notes],
            frames: [f("Listeners heard that"), f("On air we said that"), f("The presenter told listeners that"),
                     f("", ", as we said on air"), f("", ", for listeners just tuning in")],
            headers: [.transcript: ["Programme transcript: {x} on {s}.", "Radio hour with {x}, the segment on {s}."],
                      .notes: ["Producer's notes for {x}'s segment on {s}.", "Running order notes: {s}, read by {x}."]],
            intros: [.town: ["{S} was this week's place on the map.", "We broadcast live from {s} last spring."],
                     .researcher: ["{S} joined us in the studio this week.", "Our guest was {s}, down the line."],
                     .festival: ["{S} filled the second half of the programme.", "We played music recorded at {s}."]],
            foreignIntros: ["A listener asked us about {s}, {what}.", "The next item was {s}, {what}."],
            fillers: ["The phone lines stayed busy after the item on {s}.", "We ran over time on the segment about {s}.",
                      "A listener texted in a memory of {s}.", "The jingle cut in just as we finished with {s}.",
                      "The podcast version of the item on {s} is online.", "Our producer found an old recording about {s}.",
                      "Several emails asked for more on {s}.", "The weather report followed the piece on {s}.",
                      "A repeat of the item on {s} airs on Sunday.", "The studio guest had never been asked about {s} on air before."],
            closings: ["That was our item on {s}.", "More on {s} next week, same time.", "Thanks for listening to the hour on {s}."],
            excerptLeads: ["The archive tape says: \"", "From the old broadcast: \""]),
        DatasetPersona(
            name: "travel-journal", label: "Travel journal", world: .reading, legacy: nil, incidental: .person, kinds: [.journal, .letter],
            frames: [f("On the road I learned that"), f("The innkeeper told us that"), f("A local guide swore that"),
                     f("", ", or so the locals say"), f("", ", I wrote that night")],
            headers: [.journal: ["Travel journal, the day we reached {s}, with {x}.", "Journal entry: on the way to {s} with {x}."],
                      .letter: ["A letter home to {x}, written near {s}.", "Postcard to {x} about {s}."]],
            intros: [.town: ["{S} was our stop for two nights.", "We came into {s} at dusk."],
                     .researcher: ["We met {s} by chance on the train.", "{S} gave a talk at the inn where we stayed."],
                     .festival: ["We arrived just in time for {s}.", "{S} was in full swing when we got there."]],
            foreignIntros: ["Over dinner someone told us about {s}, {what}.", "A poster in the station advertised {s}, {what}."],
            fillers: ["We talked about {s} on the long walk back.", "The train was late, which left time to read about {s}.",
                      "It rained all day, so I wrote up {s}.", "The bus driver had opinions about {s}.",
                      "I sketched something to remember {s} by.", "Our room had a guidebook that mentioned {s}.",
                      "We argued over breakfast about {s}.", "A stranger on the ferry knew all about {s}.",
                      "I wrote two pages on {s} by candlelight.", "We left with more questions about {s} than we came with."],
            closings: ["Tomorrow we leave {s} behind.", "I will write more about {s} from the next stop.", "So ends the day of {s}."],
            excerptLeads: ["The guidebook claims: \"", "A sign by the road read: \""]),
    ]

    // MARK: - Software: libraries, services, incidents

    static let softwarePersonas: [DatasetPersona] = [
        DatasetPersona(
            name: "runbook", label: "Runbook", world: .software, legacy: nil, incidental: .person, kinds: [.runbook, .notes],
            frames: [f("Per the runbook,"), f("Operators should note that"), f("The runbook states that"),
                     f("", ", per the runbook"), f("", ", as the runbook warns")],
            headers: [.runbook: ["Runbook entry for {s}, maintained by {x}.", "On-call runbook: {s}, last edited by {x}."],
                      .notes: ["Operator notes on {s}, from {x}'s shift.", "Handover notes about {s}, written by {x}."]],
            intros: [.library: ["{S} is a dependency every on-call engineer should know.", "Most alerts this quarter touched {s}."],
                     .service: ["{S} pages the on-call rotation more than any other service.", "{S} has its own section in the runbook."],
                     .incident: ["{S} is the reason this runbook exists.", "The steps below were written after {s}."]],
            foreignIntros: ["An alert mentioned {s}, {what}, so it is documented here.", "The runbook covers {s}, {what}, for completeness."],
            fillers: ["Escalate to the secondary if {s} does not recover in ten minutes.", "Check the dashboard for {s} before restarting anything.",
                      "The rollback steps for {s} are in the appendix.",
                      "Do not page the owner of {s} outside working hours unless it is urgent.",
                      "Silence duplicate alerts about {s} during a known incident.", "Record every action taken on {s} in the incident channel.",
                      "The health check for {s} has a thirty second grace period.", "Capacity limits for {s} are listed in the sizing table.",
                      "A dry run against staging comes before any change to {s}.", "The last drill for {s} was run in the spring."],
            closings: ["End of the runbook entry for {s}.", "Questions about {s} go to the platform channel.", "Review this entry on {s} every quarter."],
            excerptLeads: ["The vendor runbook says: \"", "An older revision read: \""]),
        DatasetPersona(
            name: "standup", label: "Standup", world: .software, legacy: nil, incidental: .person, kinds: [.standup, .notes],
            frames: [f("At standup we confirmed that"), f("Quick update:"), f("Blocking question answered:"),
                     f("", ", said at standup"), f("", ", nothing new since yesterday")],
            headers: [.standup: ["Standup notes, {x} facilitating; one thread was {s}.", "Daily standup: {x} gave the update on {s}."],
                      .notes: ["Async standup thread on {s}, started by {x}.", "Standup follow-ups on {s}, assigned by {x}."]],
            intros: [.library: ["{S} came up in three updates today.", "Everyone is waiting on a change to {s}."],
                     .service: ["{S} was the main topic this morning.", "The team spent yesterday on {s}."],
                     .incident: ["{S} took most of yesterday's standup.", "{S} is still on the board."]],
            foreignIntros: ["A side thread at standup was about {s}, {what}.", "Someone asked about {s}, {what}, before we wrapped up."],
            fillers: ["Yesterday: reviewed the change to {s}. Today: tests.", "Blocked on access to the logs for {s}.",
                      "Will pair after lunch on {s}.", "No blockers on {s} today.", "The ticket for {s} moved to review.",
                      "Out tomorrow; someone else will cover {s}.", "Demo of the work on {s} is set for Thursday.",
                      "Estimate for {s} is two more days.", "Took the follow-up on {s} offline.", "Standup ran long because of {s}."],
            closings: ["Parking lot: {s}.", "Next standup checks in on {s}.", "Action item: write up {s}."],
            excerptLeads: ["The ticket description reads: \"", "Someone pasted this into chat: \""]),
        DatasetPersona(
            name: "ticket-thread", label: "Tickets", world: .software, legacy: nil, incidental: .person, kinds: [.ticket, .notes],
            frames: [f("Resolved in this ticket:"), f("Confirmed in the thread:"), f("The reporter says that"),
                     f("", ", per the ticket"), f("", ", marking this as answered")],
            headers: [.ticket: ["Ticket opened by {x}: question about {s}.", "Issue filed by {x} against {s}."],
                      .notes: ["Comment thread on {s}, last reply from {x}.", "Triage notes on {s}, from {x}."]],
            intros: [.library: ["{S} is the component this ticket is filed against.", "The bug report concerns {s}."],
                     .service: ["{S} is tagged on this ticket.", "A user reported strange behaviour in {s}."],
                     .incident: ["This ticket tracks the follow-ups from {s}.", "{S} is linked as the parent issue."]],
            foreignIntros: ["The ticket also mentions {s}, {what}.", "A linked issue concerns {s}, {what}."],
            fillers: ["Labels on the ticket for {s}: needs-triage, question.", "Assigned the ticket about {s} to the owning team.",
                      "Bumped the priority on {s} after a second report.", "Linked a duplicate report about {s}.",
                      "Closed as fixed; reopen if {s} misbehaves again.", "Attached the logs that mention {s}.",
                      "Moved the discussion of {s} to the design channel.", "The reporter confirmed the fix for {s} works.",
                      "Added a reproduction case for {s}.", "Milestone for {s} set to the next release."],
            closings: ["Closing the ticket on {s}.", "Leaving the ticket on {s} open for a week.", "Thanks to everyone who looked at {s}."],
            excerptLeads: ["The original report said: \"", "Quoting the first comment: \""]),
        DatasetPersona(
            name: "design-record", label: "Design record", world: .software, legacy: nil, incidental: .person, kinds: [.decision, .notes],
            frames: [f("Decided:"), f("For context,"), f("The design assumes that"),
                     f("", ", per this decision record"), f("", ", which constrains the design")],
            headers: [.decision: ["Decision record on {s}, proposed by {x}.", "Architecture decision: {s}, approved by {x}."],
                      .notes: ["Design review notes on {s}, from {x}.", "Background notes for the decision on {s}, by {x}."]],
            intros: [.library: ["{S} was chosen over two alternatives.", "This record explains why we depend on {s}."],
                     .service: ["{S} is the system this decision is about.", "{S} needed a new data model."],
                     .incident: ["{S} prompted this decision.", "Lessons from {s} shaped the design below."]],
            foreignIntros: ["The record cites {s}, {what}, as prior art.", "Alternatives considered included work around {s}, {what}."],
            fillers: ["Status of the decision on {s}: accepted.", "Consequences for {s} are listed below.",
                      "The alternative for {s} was rejected on cost.", "This supersedes an earlier record about {s}.",
                      "Reviewers signed off on {s} after one round.", "Open questions about {s} are tracked separately.",
                      "The trade-offs for {s} were discussed at length.", "A spike on {s} informed this decision.",
                      "Revisit the decision on {s} in a year.", "The record on {s} links to the benchmark results."],
            closings: ["Decision on {s} recorded.", "Follow the record on {s} for any change.", "Superseded only by a newer record on {s}."],
            excerptLeads: ["The earlier record said: \"", "The proposal stated: \""]),
        DatasetPersona(
            name: "vendor-audit", label: "Vendor audit", world: .software, legacy: nil, incidental: .person, kinds: [.audit, .report],
            frames: [f("The auditor verified that"), f("Evidence shows that"), f("Our review found that"),
                     f("", ", verified during the audit"), f("", ", per the evidence collected")],
            headers: [.audit: ["Vendor audit of {s}, led by {x}.", "Security review of {s}, signed by {x}."],
                      .report: ["Audit report on {s}, prepared by {x}.", "Findings on {s}, from {x}'s review."]],
            intros: [.library: ["{S} is a third-party component in scope for this audit.", "The audit covered {s} and its maintainers."],
                     .service: ["{S} handles customer data and was audited first.", "{S} is in scope as a critical service."],
                     .incident: ["{S} triggered a review of the vendor.", "The audit revisited {s}."]],
            foreignIntros: ["The audit sample included {s}, {what}.", "Evidence about {s}, {what}, was requested."],
            fillers: ["No material findings were raised against {s}.", "The access review for {s} found two stale accounts.",
                      "Evidence for {s} was uploaded to the audit folder.", "The vendor responded about {s} within the deadline.",
                      "A remediation plan for {s} is due next quarter.", "Controls around {s} were tested by sampling.",
                      "The data flow diagram for {s} was updated.", "Risk rating for {s}: low.",
                      "Interviews about {s} were held with two engineers.", "The audit trail for {s} is complete."],
            closings: ["Audit of {s} closed.", "Next review of {s} in twelve months.", "Findings on {s} shared with the vendor."],
            excerptLeads: ["The vendor's attestation reads: \"", "The contract clause says: \""]),
        DatasetPersona(
            name: "onboarding", label: "Onboarding", world: .software, legacy: nil, incidental: .person, kinds: [.guide, .notes],
            frames: [f("New starters should know that"), f("Worth knowing on day one:"), f("Your buddy will tell you that"),
                     f("", ", as every new hire learns"), f("", ", which surprises newcomers")],
            headers: [.guide: ["Onboarding guide, the page on {s}, kept by {x}.", "Welcome pack: getting to know {s}, from {x}."],
                      .notes: ["Buddy notes on {s}, from {x}.", "First-week notes about {s}, by {x}."]],
            intros: [.library: ["{S} is one of the first libraries you will meet here.", "You will import {s} in your first week."],
                     .service: ["{S} is a service every new engineer shadows.", "Your first ticket will probably touch {s}."],
                     .incident: ["{S} is the incident we tell every new starter about.", "Read about {s} before your first on-call shift."]],
            foreignIntros: ["You may also hear about {s}, {what}.", "Colleagues mention {s}, {what}, a lot."],
            fillers: ["Ask your buddy for access to the repository for {s}.", "The lunchtime talk on {s} is recorded.",
                      "Expect to read the code for {s} in week two.", "There is a quiz on {s} at the end of the course.",
                      "The glossary explains the jargon around {s}.", "Shadow an engineer who works on {s}.",
                      "The wiki page on {s} is the best starting point.", "Do not change {s} until your first review.",
                      "Bookmark the dashboard for {s}.", "A short video introduces {s}."],
            closings: ["That is the basics of {s}.", "Next page: going deeper on {s}.", "Ask questions about {s} in the newcomers channel."],
            excerptLeads: ["The handbook says: \"", "The welcome deck puts it: \""]),
        DatasetPersona(
            name: "changelog-bot", label: "Changelog", world: .software, legacy: nil, incidental: .person, kinds: [.changelog, .notes],
            frames: [f("Automated note:"), f("Bot summary:"), f("Generated from commit history:"),
                     f("", " (auto-generated)"), f("", " [bot]")],
            headers: [.changelog: ["Changelog for {s}, merged by {x}.", "Release digest: {s}, approved by {x}."],
                      .notes: ["Bot report on {s}, triggered by {x}.", "Weekly summary bot: {s}, subscriber {x}."]],
            intros: [.library: ["{S} had five merged changes this week.", "{S} appears in the dependency report."],
                     .service: ["{S} was deployed twice this week.", "{S} shows up in the deploy log."],
                     .incident: ["{S} is linked in the release notes.", "{S} is listed under known issues."]],
            foreignIntros: ["Mentioned in commits: {s}, {what}.", "Tag detected: {s}, {what}."],
            fillers: ["Bot: 3 commits referencing {s} since the last digest.", "Bot: no failing builds for {s} this week.",
                      "Bot: dependency update for {s} merged automatically.", "Bot: changelog entry for {s} generated from labels.",
                      "Bot: reviewers for {s} assigned by code owners.", "Bot: coverage for {s} unchanged.",
                      "Bot: stale branch touching {s} deleted.", "Bot: release notes for {s} drafted.",
                      "Bot: a security advisory for {s} was checked.", "Bot: next digest for {s} in seven days."],
            closings: ["Bot: end of digest for {s}.", "Bot: unsubscribe from {s} updates any time.", "Bot: digest for {s} archived."],
            excerptLeads: ["Bot: commit message quoted: \"", "Bot: upstream note quoted: \""]),
    ]

    // MARK: - Art: artworks, artists, collections

    static let artPersonas: [DatasetPersona] = [
        DatasetPersona(
            name: "auction-lot", label: "Auction lots", world: .art, legacy: nil, incidental: .gallery, kinds: [.lot, .report],
            frames: [f("The lot notes state that"), f("Cataloguers record that"), f("Bidders are advised that"),
                     f("", ", per the lot notes"), f("", ", as the saleroom confirmed")],
            headers: [.lot: ["Lot notes for {s}, offered at the {x}.", "Auction catalogue, a lot concerning {s}, at the {x}."],
                      .report: ["Condition summary for bidders on {s}, issued by the {x}.", "Pre-sale report on {s} from the {x}."]],
            intros: [.artwork: ["{S} comes to auction for the first time.", "{S} is the evening sale's highlight."],
                     .artist: ["Works by {s} rarely come to market.", "{S} has a strong auction record."],
                     .collection: ["{S} is selling a group of works.", "This sale includes property from {s}."]],
            foreignIntros: ["The lot includes papers concerning {s}, {what}.", "Provenance notes mention {s}, {what}."],
            fillers: ["The estimate on the lot concerning {s} was revised upward.", "Viewing for {s} is by appointment.",
                      "A telephone bidder asked about {s}.", "The reserve on the lot concerning {s} is confidential.",
                      "Condition reports on {s} are available on request.", "Buyer's premium applies to the lot concerning {s}.",
                      "The lot on {s} was withdrawn once before.", "Shipping quotes for {s} come from the saleroom.",
                      "The hammer fell quickly on the last lot like {s}.", "A collector flew in for the lot on {s}."],
            closings: ["Lot concerning {s}: sold, subject to confirmation.", "Enquiries about {s} to the specialist.",
                       "Results for {s} posted after the sale."],
            excerptLeads: ["The old sale catalogue printed: \"", "A previous owner's note says: \""]),
        DatasetPersona(
            name: "conservator", label: "Conservator", world: .art, legacy: nil, incidental: .gallery, kinds: [.report, .notes],
            frames: [f("Under examination it was confirmed that"), f("The conservation file records that"), f("Technical study indicates that"),
                     f("", ", noted at examination"), f("", ", per the treatment record")],
            headers: [.report: ["Condition report on {s}, for the {x}.", "Treatment record: {s}, studio of the {x}."],
                      .notes: ["Bench notes on {s}, conservation studio at the {x}.", "Examination notes for {s}, lent by the {x}."]],
            intros: [.artwork: ["{S} arrived with flaking paint at the edges.", "{S} was examined under ultraviolet light."],
                     .artist: ["Works by {s} use an unusual ground layer.", "The studio has treated six works by {s}."],
                     .collection: ["{S} sends its works here for treatment.", "The studio holds a framework agreement with {s}."]],
            foreignIntros: ["Documents about {s}, {what}, were checked for acidity.", "A file on {s}, {what}, went through the studio."],
            fillers: ["Surface dirt was removed from the material on {s}.", "Humidity readings for {s} stayed within range.",
                      "The varnish on the work concerning {s} was tested in a small area.",
                      "Photographs before and after treatment of {s} are filed.",
                      "The frame associated with {s} needs new hanging fixings.", "Loose fragments relating to {s} were consolidated.",
                      "The examination of {s} took two days.", "Light exposure for {s} is limited to fifty lux.",
                      "The backing board for {s} was replaced.", "Pigment samples linked to {s} went to the lab."],
            closings: ["Treatment of {s} complete.", "Recheck {s} in six months.", "{S} is stable for display."],
            excerptLeads: ["The previous treatment report noted: \"", "A restorer's label says: \""]),
        DatasetPersona(
            name: "docent-script", label: "Docent", world: .art, legacy: nil, incidental: .gallery, kinds: [.script, .notes],
            frames: [f("Point out to visitors that"), f("Tell the group that"), f("A good fact for the tour:"),
                     f("", ", visitors love to hear"), f("", ", if anyone asks")],
            headers: [.script: ["Tour script, stop on {s}, at the {x}.", "Docent script for the {x}: {s}."],
                      .notes: ["Docent's notes on {s}, for tours at the {x}.", "Tour prompts about {s}, at the {x}."]],
            intros: [.artwork: ["{S} is the fourth stop on the tour.", "Stand back so the group can see {s}."],
                     .artist: ["{S} is the painter children ask about most.", "We spend ten minutes on {s}."],
                     .collection: ["{S} gave the museum this room.", "Most of this wing comes from {s}."]],
            foreignIntros: ["If time allows, mention {s}, {what}.", "Visitors sometimes ask about {s}, {what}."],
            fillers: ["Pause here and let the group look at {s}.", "Ask the children what they notice about {s}.",
                      "Keep the group behind the line near {s}.", "Two minutes on {s} is enough for school groups.",
                      "Visitors with hearing aids can stand closer for {s}.", "Mention the gift shop postcard of {s} at the end.",
                      "Questions about {s} often come from the back.", "Point to the label for {s} rather than reading it out.",
                      "Large groups should split before {s}.", "Thank the group for their questions about {s}."],
            closings: ["Move on from {s} to the next room.", "End of the stop on {s}.", "Remind visitors they can return to {s} after the tour."],
            excerptLeads: ["The wall label reads: \"", "Our handbook puts it: \""]),
        DatasetPersona(
            name: "insurer", label: "Insurer", world: .art, legacy: nil, incidental: .gallery, kinds: [.schedule, .report],
            frames: [f("For underwriting purposes,"), f("The schedule declares that"), f("The insured confirms that"),
                     f("", ", as declared to the insurer"), f("", ", per the policy schedule")],
            headers: [.schedule: ["Insurance schedule, item {s}, for the {x}.", "Policy schedule entry for {s}, insured by the {x}."],
                      .report: ["Underwriter's report on {s}, held at the {x}.", "Loss adjuster's notes on {s}, at the {x}."]],
            intros: [.artwork: ["{S} is listed as a high-value item.", "{S} travels under nail-to-nail cover."],
                     .artist: ["Works by {s} are valued every two years.", "{S} appears on the schedule more than once."],
                     .collection: ["{S} insures its loans through this policy.", "{S} is the policyholder."]],
            foreignIntros: ["The schedule also lists {s}, {what}.", "A rider covers {s}, {what}."],
            fillers: ["The premium for {s} was renewed without change.", "Transit cover for {s} starts at the loading dock.",
                      "A valuation of {s} is due before renewal.", "No claims have been made on {s}.",
                      "The excess on {s} is set out in the policy.", "Security at the venue for {s} was inspected.",
                      "The agreed value of {s} is confidential.", "Packing for {s} must meet the insurer's standard.",
                      "Exclusions for {s} cover wear and tear.", "The broker confirmed cover for {s} by email."],
            closings: ["Cover for {s} in force.", "Schedule for {s} attached to the policy.", "Renewal for {s} due next year."],
            excerptLeads: ["The previous policy wording reads: \"", "The valuation letter states: \""]),
        DatasetPersona(
            name: "press-release", label: "Press office", world: .art, legacy: nil, incidental: .gallery, kinds: [.announcement, .notes],
            frames: [f("The museum is pleased to announce that"), f("For immediate release:"), f("Press materials confirm that"),
                     f("", ", the museum announced"), f("", ", according to the press office")],
            headers: [.announcement: ["Press release from the {x}: {s}.", "Announcement from the {x} about {s}."],
                      .notes: ["Press office notes on {s}, for the {x}.", "Media briefing notes about {s}, at the {x}."]],
            intros: [.artwork: ["{S} goes on public view this season.", "{S} returns after a long absence."],
                     .artist: ["{S} is the subject of a new exhibition.", "A retrospective of {s} opens soon."],
                     .collection: ["{S} has made a major gift.", "{S} celebrates an anniversary."]],
            foreignIntros: ["The release also mentions {s}, {what}.", "Journalists asked about {s}, {what}."],
            fillers: ["Images of {s} are available to the press on request.", "Interviews about {s} can be arranged through the press office.",
                      "The embargo on news of {s} lifts at nine.", "A press view of {s} is scheduled for Tuesday.",
                      "Quotes about {s} may be used with credit.", "The press kit for {s} includes a fact sheet.",
                      "Coverage of {s} appeared in two national papers.", "Social media posts about {s} use the official tag.",
                      "The press office will confirm details about {s}.", "High-resolution files of {s} are on the press site."],
            closings: ["Ends. Notes to editors about {s} follow.", "For more on {s}, contact the press office.",
                       "The press office thanks all who covered {s}."],
            excerptLeads: ["The director said: \"", "The earlier announcement read: \""]),
        DatasetPersona(
            name: "estate-ledger", label: "Estate ledger", world: .art, legacy: nil, incidental: .person, kinds: [.ledger, .letter],
            frames: [f("The ledger shows that"), f("The executors record that"), f("An entry in the estate books says that"),
                     f("", ", per the estate ledger"), f("", ", as the executors noted")],
            headers: [.ledger: ["Estate ledger, the page for {s}, kept by {x}.", "Ledger of the estate: {s}, entered by {x}."],
                      .letter: ["Letter from the executor {x} about {s}.", "Correspondence on {s}, signed by {x}."]],
            intros: [.artwork: ["{S} passed to the estate in the will.", "{S} is entered under paintings."],
                     .artist: ["{S} left a large estate of works.", "The estate of {s} is managed by the family."],
                     .collection: ["{S} bought from the estate twice.", "{S} holds works sold by the estate."]],
            foreignIntros: ["The ledger also records {s}, {what}.", "A note in the margin concerns {s}, {what}."],
            fillers: ["The ledger entry for {s} is in a neat hand.", "A receipt concerning {s} is pinned to the page.",
                      "The executors discussed {s} at the spring meeting.", "A copy of the entry for {s} went to the solicitor.",
                      "The estate's valuation of {s} was updated in pencil.", "Correspondence about {s} fills a thin folder.",
                      "The page for {s} carries an old stamp.", "The family asked for the record on {s}.",
                      "An index card points from {s} to this ledger.", "The entry for {s} was checked against the will."],
            closings: ["Entry for {s} closed.", "The estate holds no further papers on {s}.", "Ledger page for {s} countersigned."],
            excerptLeads: ["The will states: \"", "An older ledger has: \""]),
        DatasetPersona(
            name: "art-podcast", label: "Art podcast", world: .art, legacy: nil, incidental: .person, kinds: [.transcript, .notes],
            frames: [f("On the podcast we mentioned that"), f("Fun fact from this episode:"), f("My co-host pointed out that"),
                     f("", ", as we said on the episode"), f("", ", which blew our minds")],
            headers: [.transcript: ["Podcast transcript, episode on {s}, with {x}.", "Episode notes and transcript: {s}, hosted with {x}."],
                      .notes: ["Show notes for the episode on {s}, co-hosted by {x}.", "Research notes for {x} before the episode on {s}."]],
            intros: [.artwork: ["{S} is the painting of the week.", "We spent the whole episode on {s}."],
                     .artist: ["{S} is a painter we have wanted to cover for ages.", "Listeners voted for an episode on {s}."],
                     .collection: ["{S} invited us in for a recording.", "We recorded this episode inside {s}."]],
            foreignIntros: ["Supporters of the show asked about {s}, {what}.", "We went off topic about {s}, {what}."],
            fillers: ["Links about {s} are in the show notes.", "We recorded the bit on {s} twice.",
                      "Our editor cut a long tangent about {s}.", "Listeners sent fan art inspired by {s}.",
                      "The bonus episode goes deeper on {s}.", "We mispronounced something about {s} and fixed it in post.",
                      "The episode on {s} is our most downloaded this month.", "A guest correction about {s} is pinned in the feed.",
                      "We recommend a book about {s} at the end.", "The transcript of the bit on {s} has timestamps."],
            closings: ["That's the episode on {s}.", "Rate and review if you liked the one on {s}.", "Next week: more like {s}."],
            excerptLeads: ["Our guest read out: \"", "The museum's audio guide says: \""]),
    ]
}
