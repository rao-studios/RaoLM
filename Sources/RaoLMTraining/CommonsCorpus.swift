//
//  CommonsCorpus.swift
//  RaoLMTraining
//
//  WHAT: A corpus the commons is trained on: public text the owner chose (public-domain books,
//        open web text, files), stored once with its provenance, split into training and held-out
//        documents by a fixed seed, and tokenized once per tokenizer into flat token files the
//        trainer maps from disk. Never a Thread's text: the commons is everyone's.
//  OUT:  <corpora>/<name>/manifest.json, documents.jsonl, tokens-<tok12>.bin (int32 little-endian,
//        the tokenizer's eos after every document) and heldout-<tok12>.bin (the same for the
//        held-out documents, never trained on).
//  PIN:  The held-out documents are drawn before tokenizing, by document, from the seed:
//        ⌊1%⌋ of them, at least 2 and at most 256. The documents' hash covers their ids and texts
//        in stored order, so a recipe that names a corpus by its hash names exactly these words.
//

import Foundation
import RaoLMCore
import RaoLMModel

public struct CommonsDocument: Codable, Sendable, Equatable {
    public var id: String
    public var text: String

    public init(id: String, text: String) {
        self.id = id
        self.text = text
    }
}

public struct CommonsCorpusManifest: Codable, Sendable, Equatable {
    public struct Tokens: Codable, Sendable, Equatable {
        public var tokenizerSHA256: String
        public var train: Int
        public var heldOut: Int
    }

    public var name: String
    /// Where the text came from, e.g. "gutenberg:top-100" or "hub:HuggingFaceFW/fineweb-edu/sample-10BT/train".
    public var source: String
    public var license: String
    public var documents: Int
    public var heldOutDocuments: Int
    public var bytes: Int
    /// SHA-256 over every document's id and text, in stored order.
    public var documentsSHA256: String
    public var tokens: [Tokens]
    public var seed: UInt64
    /// How to fetch the same text again: ids, offsets, files.
    public var provenance: [String: String]
    public var createdAt: Date

    public func tokens(for tokenizerSHA256: String) -> Tokens? { tokens.first { $0.tokenizerSHA256 == tokenizerSHA256 } }
}

public enum CommonsCorpusError: Error, CustomStringConvertible {
    case exists(String)
    case missing(String)
    case empty(String)

    public var description: String {
        switch self {
        case .exists(let name): return "a corpus named \(name) exists; corpora are written once (pick another name)"
        case .missing(let what): return "no \(what)"
        case .empty(let name): return "the corpus \(name) has no text"
        }
    }
}

/// A token file mapped from disk.
public final class TokenShard: @unchecked Sendable {
    public let url: URL
    let data: Data
    public let count: Int

    public init(url: URL) throws {
        self.url = url
        data = try Data(contentsOf: url, options: .alwaysMapped)
        count = data.count / MemoryLayout<Int32>.size
    }

    /// Tokens [start, start + length).
    public func slice(_ start: Int, _ length: Int) -> [Int32] {
        precondition(start >= 0 && start + length <= count, "token slice \(start)+\(length) outside \(count)")
        return data.withUnsafeBytes { raw in
            let base = raw.bindMemory(to: Int32.self)
            return Array(base[start ..< start + length])
        }
    }

    public subscript(index: Int) -> Int32 {
        data.withUnsafeBytes { $0.bindMemory(to: Int32.self)[index] }
    }
}

public enum CommonsCorpus {
    public static let manifestFile = "manifest.json"
    public static let documentsFile = "documents.jsonl"

    public static func directory(_ root: URL, _ name: String) -> URL { root.appendingPathComponent(name, isDirectory: true) }
    static func tag(_ tokenizerSHA256: String) -> String { String(tokenizerSHA256.prefix(12)) }
    public static func tokensURL(_ root: URL, _ name: String, tokenizerSHA256: String, heldOut: Bool) -> URL {
        directory(root, name).appendingPathComponent("\(heldOut ? "heldout" : "tokens")-\(tag(tokenizerSHA256)).bin")
    }

    public static func isValidName(_ name: String) -> Bool {
        name.range(of: #"^[a-z0-9][a-z0-9._-]{1,63}$"#, options: .regularExpression) != nil
    }

    /// How many of `count` documents are held out.
    public static func heldOutCount(_ count: Int) -> Int {
        guard count >= 4 else { return count >= 2 ? 1 : 0 }
        return min(256, max(2, count / 100))
    }

    /// Writes a new corpus: its documents, its held-out split and its tokens under `tokenizer`.
    @discardableResult
    public static func write(
        name: String, documents: [CommonsDocument], source: String, license: String, provenance: [String: String], seed: UInt64 = 0x5EED_C044,
        tokenizer: RaoTokenizer, root: URL, progress: ((String) -> Void)? = nil
    ) throws -> CommonsCorpusManifest {
        let directory = directory(root, name)
        guard !FileManager.default.fileExists(atPath: directory.appendingPathComponent(manifestFile).path) else { throw CommonsCorpusError.exists(name) }
        let documents = documents.filter { !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        guard !documents.isEmpty else { throw CommonsCorpusError.empty(name) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var data = Data()
        let encoder = JSONCoding.lineEncoder()
        for document in documents {
            data.append(try encoder.encode(document))
            data.append(0x0A)
        }
        try data.write(to: directory.appendingPathComponent(documentsFile), options: .atomic)
        var heldOut = Array(0 ..< documents.count)
        var rng = SplitMix64(seed: seed)
        heldOut.shuffle(using: &rng)
        let held = Set(heldOut.prefix(heldOutCount(documents.count)))
        var manifest = CommonsCorpusManifest(
            name: name, source: source, license: license, documents: documents.count, heldOutDocuments: held.count,
            bytes: documents.reduce(0) { $0 + $1.text.utf8.count }, documentsSHA256: hash(documents), tokens: [], seed: seed,
            provenance: provenance, createdAt: .wholeSecond())
        manifest.tokens.append(try tokenize(documents, held: held, name: name, tokenizer: tokenizer, root: root, progress: progress))
        try JSONCoding.write(manifest, to: directory.appendingPathComponent(manifestFile))
        return manifest
    }

    public static func load(_ root: URL, _ name: String) throws -> CommonsCorpusManifest {
        let url = directory(root, name).appendingPathComponent(manifestFile)
        guard FileManager.default.fileExists(atPath: url.path) else { throw CommonsCorpusError.missing("corpus \(name) under \(root.path)") }
        return try JSONCoding.read(CommonsCorpusManifest.self, from: url)
    }

    public static func list(_ root: URL) -> [CommonsCorpusManifest] {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: root.path)) ?? []
        return names.sorted().compactMap { try? load(root, $0) }
    }

    public static func documents(_ root: URL, _ name: String) throws -> [CommonsDocument] {
        try JSONCoding.readLines(CommonsDocument.self, from: directory(root, name).appendingPathComponent(documentsFile))
    }

    /// The corpus's tokens under `tokenizer`, tokenized now if they were not before.
    public static func shard(_ root: URL, _ name: String, tokenizer: RaoTokenizer, heldOut: Bool) throws -> TokenShard {
        var manifest = try load(root, name)
        if manifest.tokens(for: tokenizer.tokenizerSHA256) == nil {
            let documents = try documents(root, name)
            var order = Array(0 ..< documents.count)
            var rng = SplitMix64(seed: manifest.seed)
            order.shuffle(using: &rng)
            let held = Set(order.prefix(manifest.heldOutDocuments))
            manifest.tokens.append(try tokenize(documents, held: held, name: name, tokenizer: tokenizer, root: root, progress: nil))
            try JSONCoding.write(manifest, to: directory(root, name).appendingPathComponent(manifestFile))
        }
        return try TokenShard(url: tokensURL(root, name, tokenizerSHA256: tokenizer.tokenizerSHA256, heldOut: heldOut))
    }

    static func tokenize(
        _ documents: [CommonsDocument], held: Set<Int>, name: String, tokenizer: RaoTokenizer, root: URL, progress: ((String) -> Void)?
    ) throws -> CommonsCorpusManifest.Tokens {
        var train: [Int32] = []
        var heldOut: [Int32] = []
        let eos = Int32(tokenizer.eosTokenID)
        let started = Date()
        for (i, document) in documents.enumerated() {
            let tokens = tokenizer.encode(document.text).map(Int32.init) + [eos]
            if held.contains(i) { heldOut += tokens } else { train += tokens }
            if (i + 1) % 2000 == 0 || i + 1 == documents.count {
                let rate = Double(train.count + heldOut.count) / max(Date().timeIntervalSince(started), 1e-3)
                progress?("tokenized \(i + 1)/\(documents.count) documents, \((train.count + heldOut.count).formatted()) tokens (\(Int(rate).formatted())/s)")
            }
        }
        try save(train, to: tokensURL(root, name, tokenizerSHA256: tokenizer.tokenizerSHA256, heldOut: false))
        try save(heldOut, to: tokensURL(root, name, tokenizerSHA256: tokenizer.tokenizerSHA256, heldOut: true))
        return CommonsCorpusManifest.Tokens(tokenizerSHA256: tokenizer.tokenizerSHA256, train: train.count, heldOut: heldOut.count)
    }

    static func save(_ tokens: [Int32], to url: URL) throws {
        let data = tokens.withUnsafeBufferPointer { Data(buffer: $0) }
        try data.write(to: url, options: .atomic)
    }

    public static func hash(_ documents: [CommonsDocument]) -> String {
        var parts: [Data] = []
        for document in documents {
            parts.append(Data(document.id.utf8))
            parts.append(Data([0]))
            parts.append(Data(document.text.utf8))
            parts.append(Data([0]))
        }
        return ContentHash.sha256Hex(parts: parts)
    }
}
