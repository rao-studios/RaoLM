//
//  DatasetForge.swift
//  RaoLMCore
//
//  WHAT: The proper names of a braid dataset, one shape per kind of thing: people, towns,
//        book titles and festivals for Ambient's world; libraries, services and incident codes
//        for Craft's; painting titles and collections for Veil's; and the incidental names a
//        document mentions (publications, colleagues, galleries). Also the pools of numbers,
//        years, dates and versions facts draw their values from.
//  PIN:  Every name is unique, and no name occurs inside another at a capitalised position.
//        Names are always written as given, so a name can only occur in text where one of its
//        capitals does: checking substrings that start at capitals is exact, and it is fast.
//

import Foundation

struct DatasetForge {
    private(set) var rng: SplitMix64
    private var used: Set<String> = []
    /// Every substring of a used name that starts at one of its capital letters.
    private var fromCapitals: Set<String> = []

    init(rng: SplitMix64) {
        self.rng = rng
    }

    /// Whether `name` equals a used name, lies inside one, or contains one.
    func conflicts(_ name: String) -> Bool {
        if used.contains(name) || fromCapitals.contains(name) { return true }
        let characters = Array(name)
        for start in characters.indices where characters[start].isUppercase {
            var piece = ""
            for character in characters[start...] {
                piece.append(character)
                if used.contains(piece) { return true }
            }
        }
        return false
    }

    mutating func register(_ name: String) {
        used.insert(name)
        let characters = Array(name)
        for start in characters.indices where characters[start].isUppercase {
            var piece = ""
            for character in characters[start...] {
                piece.append(character)
                fromCapitals.insert(piece)
            }
        }
    }

    /// Kinds of name whose base pools ran out; they draw from their wider pools from then on.
    private var widened: Set<String> = []

    private mutating func unique(
        _ what: String, _ make: (inout SplitMix64) -> String, wider: ((inout SplitMix64) -> String)? = nil
    ) throws -> String {
        if !widened.contains(what) {
            for _ in 0..<50_000 {
                let candidate = make(&rng)
                guard !conflicts(candidate) else { continue }
                register(candidate)
                return candidate
            }
            guard wider != nil else { throw DatasetError.exhausted("\(what) names") }
            widened.insert(what)
        }
        guard let wider else { throw DatasetError.exhausted("\(what) names") }
        for _ in 0..<50_000 {
            let candidate = wider(&rng)
            guard !conflicts(candidate) else { continue }
            register(candidate)
            return candidate
        }
        throw DatasetError.exhausted("\(what) names")
    }

    // Wider pools, only drawn once a base pool runs out (datasets of many nodes).
    static let moreTownA = townA + ["Ashby", "Blyth", "Corrie", "Denby", "Elmer", "Frome", "Gilling", "Hawes", "Ilmer", "Kirkby",
                                    "Ludlow", "Malden", "Nelby", "Otter", "Pickton", "Quendon", "Redby", "Selby", "Thirsk", "Ulley"]
    static let moreTownB = townB + ["borough", "field", "ton", "worth", "brook", "cliff", "dale", "end", "head", "well"]
    static let moreLibraryA = libraryA + ["Anvil", "Beryl", "Cipher", "Dynamo", "Ember", "Fathom", "Gyre", "Helix", "Iris", "Jasper",
                                          "Kestrel", "Lumina", "Mantle", "Nexus", "Opal", "Pivot", "Quiver", "Ripple", "Strata", "Tundra"]
    static let moreLibraryB = libraryB + ["bench", "frame", "kernel", "mill", "net", "port", "ring", "slate", "thread", "works"]
    static let moreServiceA = serviceA + ["Arbor", "Bastion", "Canopy", "Depot", "Estuary", "Fenway", "Gantry", "Haven", "Inlet",
                                          "Jetty", "Kennel", "Lodge", "Mooring", "Nook", "Orchard"]
    static let moreServiceB = serviceB + ["cove", "dock", "end", "ford", "grove", "hall", "mead", "pier", "quay", "row"]
    static let moreBookAdjectives = bookAdjectives + ["Bright", "Crooked", "Early", "Gentle", "Hidden", "Inland", "Lasting", "Modest",
                                                      "Outer", "Rough"]
    static let moreBookNouns = bookNouns + ["Primer", "Logbook", "Manual", "Treatise", "Catalogue", "Calendar", "Sampler", "Digest",
                                            "Herbal", "Itinerary"]
    static let morePublicationsA = publicationsA + ["Harbour", "Market", "Meadow", "Quarry", "Ridgeway", "Strand", "Shire", "Wolds",
                                                    "Weald", "Vale"]

    // MARK: People and places (every world)

    static let givenA = ["Ad", "Al", "Ar", "Ba", "Be", "Bri", "Ca", "Ce", "Da", "De", "Ed", "El", "Fa", "Fi", "Ga", "Gi", "Ha",
                         "He", "Il", "Is", "Ja", "Jo", "Ka", "Ki", "La", "Le", "Ma", "Mi", "Na", "Ni", "Od", "Ol", "Pa", "Pe",
                         "Ra", "Re", "Sa", "Se", "Ta", "Ti", "Ul", "Va", "Ve", "Ya", "Za"]
    static let givenB = ["ra", "ren", "lin", "na", "mon", "dric", "sa", "tan", "vi", "lo", "mira", "nor", "wen", "dan", "ric",
                         "sel", "tha", "vin", "la", "ron", "dis", "mae", "ko", "ruth", "bel", "gar", "lise", "neth", "ian", "ora"]
    static let surnameA = ["Ash", "Bar", "Bel", "Bray", "Cal", "Carr", "Dal", "Dun", "Eld", "Fair", "Fen", "Gar", "Hal", "Hart",
                           "Ing", "Kell", "Kor", "Lan", "Lind", "Mar", "Mor", "Nash", "Oak", "Pell", "Quin", "Rook", "Sel",
                           "Sorr", "Tam", "Thorn", "Ull", "Vane", "Wald", "Whit", "Wren", "Yor", "Zan", "Hollis", "Merrow",
                           "Penn", "Ravel", "Stroud", "Tilde", "Brandt", "Corm"]
    static let surnameB = ["well", "ford", "by", "son", "ley", "wick", "more", "ton", "stead", "hurst", "dale", "croft", "ard",
                           "ing", "ett", "ow", "ine", "ander", "ski", "gren", "holt", "mont", "vik", "sen", "ridge", "worth",
                           "land", "field", "man", "ey"]
    static let townA = ["Carrow", "Bellin", "Dunmar", "Esk", "Farra", "Glaston", "Hollin", "Ivel", "Kestle", "Lorr", "Mawn",
                        "Nethe", "Orrin", "Penhal", "Quarr", "Rill", "Sallow", "Tarn", "Umbre", "Vell", "Wyn", "Yarl", "Brom",
                        "Cairn", "Drey", "Elwy", "Fenn", "Gorse", "Harrow", "Inch", "Jessa", "Kelda", "Lune", "Mossy", "Norr",
                        "Pryce", "Rowan", "Sedge", "Tilly", "Aber"]
    static let townB = ["mere", "ford", "wick", "stow", "haven", "holme", "mouth", "port", "thwaite", "bridge", "ham", "cote",
                        "dean", "gate", "hope", "lea", "moor", "ness", "rigg", "shaw", "wold", "beck", "burn", "combe", "den",
                        "garth", "hithe", "ley", "stead", "wych"]

    mutating func person() throws -> String {
        try unique("person") { rng in
            rng.pick(Self.givenA) + rng.pick(Self.givenB) + " " + rng.pick(Self.surnameA) + rng.pick(Self.surnameB)
        }
    }

    mutating func town() throws -> String {
        try unique("town", { rng in rng.pick(Self.townA) + rng.pick(Self.townB) },
                   wider: { rng in rng.pick(Self.moreTownA) + rng.pick(Self.moreTownB) })
    }

    // MARK: Ambient's world

    static let bookAdjectives = ["Salt", "Quiet", "Paper", "Winter", "Copper", "Second", "Distant", "Borrowed", "Silent",
                                 "Northern", "Patient", "Unwritten", "Tidal", "Folded", "Honest", "Open", "Careful", "Slow",
                                 "Common", "Lower", "Measured", "Plain", "Late", "Narrow", "Wider", "Small", "Lost", "Kept"]
    static let bookNouns = ["Almanac", "Ledger", "Atlas", "Weather", "Commons", "Compass", "Hours", "Inventory", "Census",
                            "Register", "Survey", "Record", "Margins", "Grammar", "Harbourmaster", "Tables", "Parish",
                            "Notebook", "Account", "Chronicle", "Index", "Reckoning", "Tally", "Gazetteer", "Field Book"]
    static let fairColours = ["Brass", "Amber", "Cobalt", "Linen", "Juniper", "Saffron", "Indigo", "Pewter", "Clover",
                              "Heather", "Flint", "Marigold", "Russet", "Tallow", "Violet", "Willow", "Barley", "Cedar",
                              "Coral", "Ivory", "Jasper", "Lilac", "Maple", "Ochre", "Poppy", "Quince", "Sable", "Thistle"]
    static let fairThings = ["Kite", "Lantern", "Drum", "Bell", "Boat", "Ribbon", "Candle", "Mask", "Fiddle", "Kettle",
                             "Sparrow", "Wheel", "Harp", "Bonfire", "Parade", "Choir", "Hare", "Fox", "Crane", "Moon", "Star"]
    static let fairKinds = ["Festival", "Fair", "Week", "Gathering"]
    static let publicationsPre = ["", "Evening ", "Sunday ", "Morning ", "Weekend ", "Daily ", "Little ", "New "]
    static let publicationsA = ["Northfield", "Harbourside", "Lowland", "Coastal", "Riverbank", "Upland", "Midland",
                                "Borough", "Valley", "Island", "Moorland", "County", "Township", "Hillside", "Estuary",
                                "Lakeside", "Heath", "Downs", "Fenland", "Seaboard", "Marches", "Crossing", "Fells",
                                "Headland", "Tidewater"]
    static let publicationsB = ["Review", "Gazette", "Courier", "Dispatch", "Journal", "Chronicle", "Bulletin", "Post",
                                "Observer", "Weekly", "Quarterly", "Monitor"]

    mutating func book() throws -> String {
        try unique("book", { rng in "The " + rng.pick(Self.bookAdjectives) + " " + rng.pick(Self.bookNouns) },
                   wider: { rng in "The " + rng.pick(Self.moreBookAdjectives) + " " + rng.pick(Self.moreBookNouns) })
    }

    mutating func festival() throws -> String {
        try unique("festival") { rng in
            rng.pick(Self.fairColours) + " " + rng.pick(Self.fairThings) + " " + rng.pick(Self.fairKinds)
        }
    }

    mutating func publication() throws -> String {
        try unique("publication", { rng in rng.pick(Self.publicationsPre) + rng.pick(Self.publicationsA) + " " + rng.pick(Self.publicationsB) },
                   wider: { rng in rng.pick(Self.publicationsPre) + rng.pick(Self.morePublicationsA) + " " + rng.pick(Self.publicationsB) })
    }

    // MARK: Craft's world

    static let libraryA = ["Quill", "Lumen", "Tessel", "Mica", "Fjord", "Onyx", "Pylon", "Rivet", "Sprocket", "Vanta", "Wisp",
                           "Zephyr", "Argo", "Brisk", "Cobble", "Dapple", "Flux", "Glint", "Hatch", "Ingot", "Jolt", "Knot",
                           "Lattice", "Mote", "Nimbus", "Orbit", "Prism", "Quark", "Sift", "Tinder", "Umbra", "Vesper", "Weft",
                           "Zinc", "Axle", "Bramble", "Cirrus", "Delta", "Ferrous", "Gossamer"]
    static let libraryB = ["mesh", "kit", "db", "flow", "line", "forge", "stack", "wire", "grid", "base", "core", "sync",
                           "vault", "cast", "pipe", "scope", "shard", "hook", "loom", "lens", "path", "spark", "trace", "queue"]
    static let serviceA = ["Drift", "Harbor", "Toll", "Beacon", "Cinder", "Signal", "Anchor", "Sentry", "Courier", "Ferry",
                           "Pilot", "Rampart", "Shuttle", "Turnstile", "Warden", "Bellows", "Cistern", "Dispatch", "Foundry",
                           "Granary", "Kiln", "Lookout", "Outpost", "Paddock", "Rookery", "Spindle", "Lantern", "Marrow"]
    static let serviceB = ["point", "works", "house", "yard", "room", "deck", "hub", "bay", "post", "way", "stone", "light",
                           "wing", "gate", "keep", "fold", "mark", "field", "ward", "reach", "rest", "side", "tower", "lane",
                           "run", "hold", "spring", "dale"]
    static let codeLetters = Array("ABCDEFGHJKLMNPRSTUVWXYZ").map(String.init)

    mutating func library() throws -> String {
        try unique("library", { rng in rng.pick(Self.libraryA) + rng.pick(Self.libraryB) },
                   wider: { rng in rng.pick(Self.moreLibraryA) + rng.pick(Self.moreLibraryB) })
    }

    mutating func service() throws -> String {
        try unique("service", { rng in rng.pick(Self.serviceA) + rng.pick(Self.serviceB) },
                   wider: { rng in rng.pick(Self.moreServiceA) + rng.pick(Self.moreServiceB) })
    }

    mutating func incidentCode() throws -> String {
        try unique("incident") { rng in
            rng.pick(Self.codeLetters) + rng.pick(Self.codeLetters) + "-" + String(rng.nextInt(in: 1000...9999))
        }
    }

    // MARK: Veil's world

    static let scenes = ["Harbour", "Orchard", "Quarry", "Stairwell", "Window", "Estuary", "Garden", "Kitchen", "Ferry",
                         "Chapel", "Dune", "Glasshouse", "Mill", "Terrace", "Courtyard", "Lighthouse", "Station", "Meadow",
                         "Boathouse", "Cloister", "Pier", "Vineyard", "Loft", "Arcade", "Canal", "Promenade", "Harvesters",
                         "Weavers", "Sleepers", "Bathers", "Riders", "Readers", "Dancers", "Fishermen", "Musicians", "Skaters"]
    static let hours = ["Dusk", "Dawn", "Noon", "Midnight", "First Light", "Low Tide", "High Tide", "Evening", "Twilight",
                        "Nightfall", "Daybreak", "Moonrise", "Sunset"]
    static let sceneAdjectives = ["Blue", "Grey", "Drowned", "Burning", "Empty", "Golden", "Crooked", "Sleeping", "Summer",
                                  "Broken", "Painted", "Silver", "Green", "Pale", "Red", "White", "Hidden", "Flooded"]
    static let seasons = ["Spring", "Autumn", "March", "October", "August", "June", "November", "April"]
    static let collectionNames = ["Varrin", "Oskel", "Brannoch", "Dellow", "Ferris", "Galloway", "Hesketh", "Iverly",
                                  "Jarrow", "Kilbride", "Lachlan", "Merrick", "Nolde", "Ostrander", "Prewitt", "Quayle",
                                  "Rensley", "Sutter", "Tolland", "Ulverston", "Vickery", "Wexley", "Aldane", "Bexley",
                                  "Corwen", "Drummond", "Elsworth", "Farrant", "Gresley", "Hallam", "Ismay", "Keverne",
                                  "Lusk", "Maddox", "Norland", "Ormsby", "Pardew", "Rudge", "Stobart", "Treloar"]
    static let collectionKinds = ["Collection", "Bequest", "Trust", "Foundation"]
    static let galleryKinds = ["Gallery", "Rooms", "Hall", "Museum"]

    mutating func artwork() throws -> String {
        try unique("artwork") { rng in
            switch rng.nextInt(below: 4) {
            case 0: return rng.pick(Self.scenes) + " at " + rng.pick(Self.hours)
            case 1: return rng.pick(Self.sceneAdjectives) + " " + rng.pick(Self.scenes)
            case 2: return rng.pick(Self.scenes) + " in " + rng.pick(Self.seasons)
            default: return rng.pick(Self.scenes) + " with " + rng.pick(Self.scenes)
            }
        }
    }

    mutating func collection() throws -> String {
        try unique("collection") { rng in
            let name = rng.nextInt(below: 2) == 0 ? rng.pick(Self.collectionNames) : rng.pick(Self.surnameA) + rng.pick(Self.surnameB)
            return name + " " + rng.pick(Self.collectionKinds)
        }
    }

    mutating func gallery() throws -> String {
        try unique("gallery") { rng in rng.pick(Self.surnameA) + rng.pick(Self.surnameB) + " " + rng.pick(Self.galleryKinds) }
    }

    // MARK: The subject worlds (v3, `--worlds`): drawn only by a subject dataset

    // Writing: novels, writers (people), literary journals.
    static let novelAdjectives = ["Glass", "Hollow", "Wandering", "Bitter", "Gilded", "Sleepless", "Restless", "Velvet", "Burning",
                                  "Crimson", "Faded", "Unspoken", "Iron", "Lonely", "Midnight", "Mirrored", "Muted", "Painted",
                                  "Scarlet", "Shuttered", "Silken", "Smoking", "Sunken", "Tangled", "Thin", "Twelfth", "Unlit",
                                  "Vanishing", "Waking", "Wild", "Yellow", "Absent"]
    static let novelNouns = ["Orchard", "Lighthouse", "Ferryman", "Cartographer", "Inheritance", "Apprentice", "Winterhouse",
                             "Ballroom", "Vineyard", "Tenement", "Garrison", "Telegram", "Understudy", "Locksmith", "Embassy",
                             "Sanatorium", "Boarding House", "Seamstress", "Clockmaker", "Lodger", "Interpreter", "Night Train",
                             "Widow", "Regatta", "Orangery", "Bell Tower", "Pilgrim", "Harvest", "Tide Mill", "Ironworks",
                             "Mapmaker", "Glasshouse"]
    static let moreNovelAdjectives = novelAdjectives + ["Amber", "Broken", "Closing", "Darkened", "Fallen", "Gentle", "Hungry",
                                                        "Last", "Northern", "Patient"]
    static let moreNovelNouns = novelNouns + ["Almoner", "Bookbinder", "Chaperone", "Dressmaker", "Engraver", "Governess",
                                              "Hatter", "Innkeeper", "Lacemaker", "Organist"]
    static let journalStems = ["Ashl", "Belv", "Calv", "Dorr", "Elv", "Falk", "Gavr", "Holm", "Isk", "Jorv", "Kald", "Lisk", "Morv",
                               "Narr", "Ostr", "Pask", "Quarn", "Renn", "Sarv", "Tolv", "Uld", "Varn", "Wesk", "Yarr", "Zell",
                               "Brenn", "Casv", "Drav", "Frisv", "Grell"]
    static let journalEndings = ["ery", "ow", "ith", "ane", "ell", "ory", "ard", "ine"]
    static let moreJournalStems = journalStems + ["Arn", "Bost", "Cleav", "Durn", "Ethr", "Forv", "Garn", "Herv", "Kesv", "Lorv"]

    mutating func novelTitle() throws -> String {
        try unique("novel", { rng in "The " + rng.pick(Self.novelAdjectives) + " " + rng.pick(Self.novelNouns) },
                   wider: { rng in "The " + rng.pick(Self.moreNovelAdjectives) + " " + rng.pick(Self.moreNovelNouns) })
    }

    mutating func journal() throws -> String {
        try unique("journal", { rng in rng.pick(Self.journalStems) + rng.pick(Self.journalEndings) },
                   wider: { rng in rng.pick(Self.moreJournalStems) + rng.pick(Self.journalEndings) })
    }

    // Coding and mathematics: languages, algorithms, theorems.
    static let languageStems = ["Brix", "Corv", "Dax", "Edr", "Fisk", "Gorm", "Hask", "Ivr", "Jask", "Kov", "Lorn", "Mirr", "Nox",
                                "Orv", "Pax", "Rusk", "Sorv", "Tav", "Ulm", "Vex", "Wirr", "Yask", "Zorn", "Breck", "Crell",
                                "Dask", "Flen", "Grov", "Hirr", "Kesk"]
    static let languageEndings = ["el", "ic", "ith", "on", "ux", "ara", "ett", "yn"]
    static let moreLanguageStems = languageStems + ["Arx", "Bryv", "Cosk", "Drex", "Exv", "Fyrr", "Glox", "Hyv", "Jynk", "Lyx"]
    static let algorithmWords = ["Ardel", "Brannick", "Calder", "Dovrin", "Esmond", "Farrow", "Greaves", "Hadley", "Iverson",
                                 "Jessop", "Kittredge", "Larkin", "Merriman", "Norcott", "Orwin", "Pemberly", "Quarrie", "Redfern",
                                 "Sandle", "Tolliver", "Upward", "Venner", "Whitlow", "Yardley", "Abelin", "Brightwater", "Corbet",
                                 "Dunstan", "Ellery", "Fennick", "Galbraith", "Hollins", "Imrie", "Jolliffe", "Kinnaird", "Lowrie",
                                 "Mansell", "Nettleton", "Oswin", "Prideaux"]
    static let algorithmKinds = ["sort", "hashing", "search", "sieve", "pruning", "descent", "walk", "fold"]
    static let moreAlgorithmWords = algorithmWords + ["Ashdown", "Bellamy", "Coverdale", "Delamere", "Eversley", "Fothergill",
                                                      "Gilmour", "Haverford", "Ingleby", "Jardine", "Kirkwall", "Lamington",
                                                      "Mountjoy", "Newbold", "Ormerod", "Pettigrew", "Rawdon", "Silverton",
                                                      "Trelawny", "Wetherby"]
    static let mathNames = ["Orrevik", "Tessmar", "Halvard", "Brenning", "Castell", "Dahlberg", "Eckard", "Falster", "Grimvald",
                            "Hesselt", "Ingvar", "Jarnow", "Kolbeck", "Lindqvist", "Morstad", "Nyberg", "Osterholm", "Pallis",
                            "Quistgaard", "Rasmark", "Selvik", "Torvald", "Ulfsen", "Vesterby", "Wennberg", "Ystrand", "Zandvik",
                            "Arnholt", "Bjornstad", "Cederlund", "Dannevig", "Engstrom", "Frisell", "Gyllen", "Hammar", "Isaksen"]
    static let theoremKinds = ["lemma", "theorem", "inequality", "bound"]

    mutating func language() throws -> String {
        try unique("language", { rng in rng.pick(Self.languageStems) + rng.pick(Self.languageEndings) },
                   wider: { rng in rng.pick(Self.moreLanguageStems) + rng.pick(Self.languageEndings) })
    }

    mutating func algorithm() throws -> String {
        try unique("algorithm", { rng in rng.pick(Self.algorithmWords) + " " + rng.pick(Self.algorithmKinds) },
                   wider: { rng in rng.pick(Self.moreAlgorithmWords) + " " + rng.pick(Self.algorithmKinds) })
    }

    mutating func theorem() throws -> String {
        try unique("theorem") { rng in
            let first = rng.pick(Self.mathNames)
            var second = rng.pick(Self.mathNames)
            while second == first { second = rng.pick(Self.mathNames) }
            return first + "-" + second + " " + rng.pick(Self.theoremKinds)
        }
    }

    // Biology: species, proteins (and their genes), field stations.
    static let speciesPlaces = ["Abbotsley", "Brackwater", "Carrowdale", "Dunlarrow", "Eskmoor", "Fallowby", "Glenmarrow",
                                "Hollingrove", "Inverlay", "Kestwick", "Larrowby", "Moorhaven", "Netherby", "Orrinmoor",
                                "Pellingham", "Quarrendon", "Redmarsh", "Saltwick", "Tarnbury", "Ulverhay", "Vellingham",
                                "Wexmoor", "Yarlside", "Ashcombe", "Brindlemere", "Cobbleford", "Dovermarsh", "Elmstead",
                                "Fernhollow", "Greywater"]
    static let creatures = ["Finch", "Vole", "Newt", "Moth", "Beetle", "Shrew", "Plover", "Lacewing", "Mayfly", "Toad", "Snail",
                            "Limpet", "Darter", "Skipper", "Weevil", "Pipit", "Bunting", "Gudgeon", "Loach", "Sculpin", "Stonefly",
                            "Sandpiper", "Mole", "Dormouse", "Bat", "Crayfish", "Shrimp", "Hoverfly", "Leafhopper", "Wagtail"]
    static let proteinStems = ["Arv", "Bastr", "Cyl", "Dorv", "Ecl", "Fabr", "Glev", "Halv", "Ind", "Jov", "Kryst", "Lum", "Myr",
                               "Nesv", "Osk", "Pyr", "Quer", "Rhov", "Synd", "Tyr", "Ulv", "Vasc", "Wyl", "Xer", "Zyg", "Brev",
                               "Cerv", "Dyn", "Fov", "Gryph"]
    static let proteinEndings = ["in", "ase", "ulin", "erin", "idin", "ectin"]
    static let moreProteinStems = proteinStems + ["Ambr", "Blev", "Cast", "Derv", "Eskr", "Fyl", "Gorv", "Hesp", "Ilv", "Kesp"]
    static let stationWords = ["Arrowmere", "Blackhope", "Coldwater", "Dunhallow", "Eastwick", "Foxholm", "Gullbeck",
                               "Heronsgate", "Ironwold", "Kelmscot", "Lindisway", "Marrowfield", "Northwold", "Otterhallow",
                               "Puffin Rock", "Quarrymead", "Ravenholm", "Stonemere", "Thornholme", "Upperwick", "Valebridge",
                               "Westerhope", "Yewcroft", "Ashwater", "Birchwold", "Cairnholm", "Dovecote", "Eelbrook", "Fennhaven",
                               "Grayling", "Hartshead", "Inchbrae", "Juniper Bay", "Kittiwold", "Larkmoor", "Mossholme",
                               "Nettlecombe", "Ospreyhead", "Pebblecombe", "Rushmere"]
    static let stationKinds = ["Field Station", "Research Station", "Marine Station", "Biological Station", "Field Laboratory"]
    static let moreStationWords = stationWords + ["Adderley", "Bramblehope", "Curlewmoor", "Dunnockside", "Egretwater", "Fulmar Head",
                                                  "Gannetby", "Hollowmere", "Ivyholme", "Kestrel Point", "Linnetfield", "Merlinsgate",
                                                  "Nightjar Hill", "Ouzelwick", "Ptarmigan Ridge"]

    mutating func species() throws -> String {
        try unique("species") { rng in rng.pick(Self.speciesPlaces) + " " + rng.pick(Self.creatures) }
    }

    mutating func protein() throws -> String {
        try unique("protein", { rng in rng.pick(Self.proteinStems) + rng.pick(Self.proteinEndings) },
                   wider: { rng in rng.pick(Self.moreProteinStems) + rng.pick(Self.proteinEndings) })
    }

    mutating func station() throws -> String {
        try unique("station", { rng in rng.pick(Self.stationWords) + " " + rng.pick(Self.stationKinds) },
                   wider: { rng in rng.pick(Self.moreStationWords) + " " + rng.pick(Self.stationKinds) })
    }

    mutating func gene() throws -> String {
        try unique("gene") { rng in
            rng.pick(Self.codeLetters) + rng.pick(Self.codeLetters) + rng.pick(Self.codeLetters) + String(rng.nextInt(in: 1...99))
        }
    }
}

/// Values a fact kind draws without replacement, so no two of its facts share a value. A kind
/// whose pool runs out (many nodes of one world) goes on into an extension pool of values
/// outside the base range, drawn from its own stream, so a dataset that never runs out draws
/// exactly as before.
struct DatasetValues {
    private var pools: [FactKind: [String]] = [:]
    private var extensions: [FactKind: [String]] = [:]
    private var cursors: [FactKind: Int] = [:]

    static let months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                         "November", "December"]

    init(seed: UInt64) {
        var rng = SplitMix64.derived(seed: seed, stream: 23)
        func numbers(_ range: ClosedRange<Int>) -> [String] { rng.shuffled(Array(range)).map(String.init) }
        pools[.townFounded] = numbers(1050...1899)
        pools[.townPopulation] = numbers(1_200...98_000)
        pools[.researcherBorn] = numbers(1760...2004)
        pools[.festivalFirst] = numbers(1400...1999)
        pools[.festivalVisitors] = numbers(800...250_000)
        pools[.libraryPort] = numbers(1025...65_000)
        pools[.serviceLatency] = numbers(8...990)
        pools[.serviceLaunched] = rng.shuffled((1990...2026).flatMap { year in Self.months.map { "\($0) \(year)" } })
        pools[.incidentMinutes] = numbers(3...900)
        pools[.artworkYear] = numbers(1400...2020)
        pools[.artworkWidth] = numbers(18...640)
        pools[.artistBorn] = numbers(1500...1999)
        pools[.collectionOpened] = numbers(1700...2020)
        pools[.collectionWorks] = numbers(40...9_000)
        let versions = (1...9).flatMap { major in (0...40).flatMap { minor in (0...30).map { "\(major).\(minor).\($0)" } } }
        let shuffled = rng.shuffled(versions)
        pools[.libraryVersion] = Array(shuffled.prefix(shuffled.count / 2))
        pools[.incidentFixVersion] = Array(shuffled.suffix(from: shuffled.count / 2))

        var more = SplitMix64.derived(seed: seed, stream: 25)
        func extra(_ range: ClosedRange<Int>) -> [String] { more.shuffled(Array(range)).map(String.init) }
        extensions[.townFounded] = extra(800...1049)
        extensions[.townPopulation] = extra(98_001...160_000)
        extensions[.researcherBorn] = extra(1600...1759)
        extensions[.festivalFirst] = extra(1250...1399)
        extensions[.festivalVisitors] = extra(250_001...400_000)
        extensions[.libraryPort] = extra(65_001...65_535)
        extensions[.serviceLatency] = extra(991...2_400)
        extensions[.serviceLaunched] = more.shuffled((1975...1989).flatMap { year in Self.months.map { "\($0) \(year)" } })
        extensions[.incidentMinutes] = extra(901...2_000)
        extensions[.artworkYear] = extra(1250...1399)
        extensions[.artworkWidth] = extra(641...1_200)
        extensions[.artistBorn] = extra(1350...1499)
        extensions[.collectionOpened] = extra(1450...1699)
        extensions[.collectionWorks] = extra(9_001...30_000)
        let later = more.shuffled((10...19).flatMap { major in (0...40).flatMap { minor in (0...30).map { "\(major).\(minor).\($0)" } } })
        extensions[.libraryVersion] = Array(later.prefix(later.count / 2))
        extensions[.incidentFixVersion] = Array(later.suffix(from: later.count / 2))

        // The subject worlds draw on streams of their own, so the founding worlds' draws stay as they were.
        var subjects = SplitMix64.derived(seed: seed, stream: 26)
        func subject(_ range: ClosedRange<Int>) -> [String] { subjects.shuffled(Array(range)).map(String.init) }
        pools[.novelPublished] = subject(1890...2024)
        pools[.novelPages] = subject(96...880)
        pools[.writerBorn] = subject(1850...1998)
        pools[.journalFounded] = subject(1880...2015)
        pools[.journalCirculation] = subject(1_100...48_000)
        pools[.languageReleased] = subjects.shuffled((1980...2025).flatMap { year in Self.months.map { "\($0) \(year)" } })
        pools[.languageVersion] = subjects.shuffled(versions)
        pools[.algorithmYear] = subject(1936...2024)
        pools[.algorithmLines] = subject(40...2_400)
        pools[.theoremYear] = subject(1820...2020)
        pools[.theoremPages] = subject(3...140)
        pools[.speciesDescribed] = subject(1750...2020)
        pools[.speciesWeight] = subject(4...9_000)
        pools[.proteinResidues] = subject(60...2_400)
        pools[.proteinDiscovered] = subject(1900...2022)
        pools[.stationEstablished] = subject(1880...2018)
        pools[.stationSpecimens] = subject(1_500...240_000)

        var moreSubjects = SplitMix64.derived(seed: seed, stream: 27)
        func subjectExtra(_ range: ClosedRange<Int>) -> [String] { moreSubjects.shuffled(Array(range)).map(String.init) }
        extensions[.novelPublished] = subjectExtra(1800...1889)
        extensions[.novelPages] = subjectExtra(881...1_400)
        extensions[.writerBorn] = subjectExtra(1750...1849)
        extensions[.journalFounded] = subjectExtra(1800...1879)
        extensions[.journalCirculation] = subjectExtra(48_001...90_000)
        extensions[.languageReleased] = moreSubjects.shuffled((1960...1979).flatMap { year in Self.months.map { "\($0) \(year)" } })
        extensions[.languageVersion] = moreSubjects.shuffled(
            (10...19).flatMap { major in (0...40).flatMap { minor in (0...30).map { "\(major).\(minor).\($0)" } } })
        extensions[.algorithmYear] = subjectExtra(1850...1935)
        extensions[.algorithmLines] = subjectExtra(2_401...6_000)
        extensions[.theoremYear] = subjectExtra(1650...1819)
        extensions[.theoremPages] = subjectExtra(141...400)
        extensions[.speciesDescribed] = subjectExtra(1700...1749)
        extensions[.speciesWeight] = subjectExtra(9_001...40_000)
        extensions[.proteinResidues] = subjectExtra(2_401...5_000)
        extensions[.proteinDiscovered] = subjectExtra(1830...1899)
        extensions[.stationEstablished] = subjectExtra(1800...1879)
        extensions[.stationSpecimens] = subjectExtra(240_001...500_000)
    }

    /// The next value of a numeric, date or version fact; nil for facts whose values are names.
    mutating func take(_ kind: FactKind) throws -> String? {
        guard let pool = pools[kind] else { return nil }
        let cursor = cursors[kind, default: 0]
        cursors[kind] = cursor + 1
        if cursor < pool.count { return pool[cursor] }
        let extension_ = extensions[kind] ?? []
        guard cursor - pool.count < extension_.count else { throw DatasetError.exhausted("\(kind.rawValue) values") }
        return extension_[cursor - pool.count]
    }
}
