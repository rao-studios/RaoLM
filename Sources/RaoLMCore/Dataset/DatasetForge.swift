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

    private mutating func unique(_ what: String, _ make: (inout SplitMix64) -> String) throws -> String {
        for _ in 0..<50_000 {
            let candidate = make(&rng)
            guard !conflicts(candidate) else { continue }
            register(candidate)
            return candidate
        }
        throw DatasetError.exhausted("\(what) names")
    }

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
        try unique("town") { rng in rng.pick(Self.townA) + rng.pick(Self.townB) }
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
        try unique("book") { rng in "The " + rng.pick(Self.bookAdjectives) + " " + rng.pick(Self.bookNouns) }
    }

    mutating func festival() throws -> String {
        try unique("festival") { rng in
            rng.pick(Self.fairColours) + " " + rng.pick(Self.fairThings) + " " + rng.pick(Self.fairKinds)
        }
    }

    mutating func publication() throws -> String {
        try unique("publication") { rng in
            rng.pick(Self.publicationsPre) + rng.pick(Self.publicationsA) + " " + rng.pick(Self.publicationsB)
        }
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
        try unique("library") { rng in rng.pick(Self.libraryA) + rng.pick(Self.libraryB) }
    }

    mutating func service() throws -> String {
        try unique("service") { rng in rng.pick(Self.serviceA) + rng.pick(Self.serviceB) }
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
}

/// Values a fact kind draws without replacement, so no two of its facts share a value.
struct DatasetValues {
    private var pools: [FactKind: [String]] = [:]
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
    }

    /// The next value of a numeric, date or version fact; nil for facts whose values are names.
    mutating func take(_ kind: FactKind) throws -> String? {
        guard let pool = pools[kind] else { return nil }
        let cursor = cursors[kind, default: 0]
        guard cursor < pool.count else { throw DatasetError.exhausted("\(kind.rawValue) values") }
        cursors[kind] = cursor + 1
        return pool[cursor]
    }
}
