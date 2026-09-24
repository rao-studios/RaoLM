//
//  NameForge.swift
//  RaoLMCore
//
//  WHAT: Invents the proper names of the synthetic Veldmar world: people, places,
//        landmarks, rivers and dishes.
//  PIN:  Every name handed out is unique, and no name contains another as a substring.
//        That is what lets the validator prove each fact's prompt occurs in exactly one
//        document: every prompt names its document's subject, and no other document can
//        contain that subject.
//

import Foundation

struct NameForge {
    private(set) var rng: SplitMix64
    private var used: [String] = []
    private var usedSet: Set<String> = []

    init(rng: SplitMix64) {
        self.rng = rng
    }

    static let givenA = ["Or", "Ma", "Ta", "Be", "Li", "Ca", "Da", "El", "Fe", "Ga", "Ha", "Is", "Jo",
                         "Ke", "Lo", "Mi", "Ne", "Pa", "Ri", "Sa", "Te", "Va", "We", "Ya", "Zo"]
    static let givenB = ["len", "ris", "vin", "ra", "dor", "wen", "mir", "nus", "sa", "tan", "bel",
                         "gar", "lin", "ro", "thea", "nor", "vek", "dra", "mon", "sel"]
    static let surnameA = ["Bram", "Cor", "Ev", "Fal", "Gil", "Hal", "Ir", "Kes", "Lor", "Mor", "Nor",
                           "Pem", "Quen", "Rav", "Sten", "Thal", "Ur", "Vel", "Wyn", "Zell", "Tor", "Brey"]
    static let surnameB = ["wright", "more", "vale", "stone", "croft", "ridge", "wood", "fell", "ward",
                           "ley", "son", "mark", "hart", "ling", "row"]
    static let placeA = ["Amber", "Brin", "Cald", "Dray", "Elm", "Frost", "Glen", "Hart", "Ivy", "Jun",
                         "Kings", "Lark", "Marsh", "North", "Oak", "Pine", "Quarry", "Raven", "Salt",
                         "Thorn", "Umber", "Wick", "Yarrow", "Ash", "Birch", "Cold", "Dun", "Fair",
                         "Grey", "Holly"]
    static let placeB = ["fall", "wick", "ford", "mere", "haven", "stead", "holm", "reach", "moor",
                         "gate", "brook", "field", "dale", "crest", "hollow", "well", "ton", "by",
                         "stow", "ley"]
    static let landmarkA = ["Kestrel", "Heron", "Lantern", "Copper", "Silver", "Hollow", "Weeping",
                            "Iron", "Glass", "Saffron", "Cinder", "Moth", "Tallow", "Verdant", "Bell",
                            "Crane", "Ember", "Falcon", "Otter", "Tidewater"]
    static let landmarkB = ["Bridge", "Tower", "Lighthouse", "Aqueduct", "Library", "Observatory",
                            "Mill", "Gate", "Canal", "Hall", "Chapel", "Market", "Arch", "Cistern"]
    static let riverA = ["Tar", "Wen", "Os", "Bra", "Lu", "Mer", "Id", "Cal", "Es", "Te", "Al", "Fro",
                         "Ky", "Ly", "Ne"]
    static let riverB = ["n", "dle", "sel", "e", "row", "der", "k", "me", "wy", "th"]
    static let dishA = ["Honeyed", "Spiced", "Smoked", "Salted", "Golden", "Rustic", "Hearth",
                        "Harvest", "Midwinter", "Saffron", "Festival", "Pilgrim", "Lantern",
                        "Market", "Riverside"]
    static let dishB = ["Barley", "Oat", "Rye", "Plum", "Apple", "Pear", "Onion", "Leek", "Cheese",
                        "Chestnut", "Hazel", "Berry", "Fennel", "Carrot"]
    static let dishC = ["Loaf", "Pie", "Cake", "Tart", "Bun", "Pudding", "Flatbread", "Pasty", "Roll",
                        "Crumble", "Bread", "Dumpling"]

    /// Draws candidates from `make` until one is unique and substring-free, or throws.
    private mutating func unique(_ what: String, _ make: (inout SplitMix64) -> String) throws -> String {
        for _ in 0..<20_000 {
            let candidate = make(&rng)
            if usedSet.contains(candidate) { continue }
            if used.contains(where: { $0.contains(candidate) || candidate.contains($0) }) { continue }
            used.append(candidate)
            usedSet.insert(candidate)
            return candidate
        }
        throw SyntheticCorpusError.namesExhausted(what)
    }

    mutating func person() throws -> String {
        try unique("person") { rng in
            rng.pick(Self.givenA) + rng.pick(Self.givenB) + " " + rng.pick(Self.surnameA) + rng.pick(Self.surnameB)
        }
    }

    mutating func place() throws -> String {
        try unique("place") { rng in rng.pick(Self.placeA) + rng.pick(Self.placeB) }
    }

    mutating func landmark() throws -> String {
        try unique("landmark") { rng in rng.pick(Self.landmarkA) + " " + rng.pick(Self.landmarkB) }
    }

    mutating func river() throws -> String {
        try unique("river") { rng in rng.pick(Self.riverA) + rng.pick(Self.riverB) }
    }

    mutating func dish() throws -> String {
        try unique("dish") { rng in
            rng.pick(Self.dishA) + " " + rng.pick(Self.dishB) + " " + rng.pick(Self.dishC)
        }
    }
}

/// Numbers drawn without replacement, so no two facts of one kind share a value.
struct ValuePool {
    let name: String
    private var values: [Int]
    private var cursor = 0

    init(name: String, range: ClosedRange<Int>, rng: inout SplitMix64) {
        self.name = name
        self.values = rng.shuffled(Array(range))
    }

    mutating func take() throws -> Int {
        guard cursor < values.count else {
            throw SyntheticCorpusError.poolExhausted(name, size: values.count)
        }
        defer { cursor += 1 }
        return values[cursor]
    }
}
