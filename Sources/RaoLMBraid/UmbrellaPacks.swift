//
//  UmbrellaPacks.swift
//  RaoLMBraid
//
//  WHAT: Where a braid's umbrella pack comes from. A preset without a trunk (tiny, small) gets
//        the seeded vocabulary it always had, as a pack that is a vocabulary alone. `base` gets a
//        pack built once from SmolLM2-135M: its weights downloaded from the Hugging Face hub at a
//        pinned revision and cut after block 20, with commons samples (anchors and a held-out
//        set) cut from public-domain books on Project Gutenberg.
//  OUT:  <braid>/umbrella/<hash12>/ (UmbrellaPack's files) and <braid>/umbrella/<name>.json,
//        which names the pack a preset uses; downloaded books are kept in <braid>/umbrella/texts/.
//  PIN:  A pack's tokenizer must be RaoLM's own (SmolLM2's is the one RaoLM vendors), so a pack
//        changes no token id, tokenizer hash or citation address. Snippets are cut with a fixed
//        seed from texts recorded by their hashes. MLX: call it on the thread that will use it.
//

import Foundation
import Hub
import MLX
import RaoLMCore
import RaoLMModel

/// Where a pack's weights and commons text come from.
public struct PackSource: Sendable, Equatable {
    public struct Book: Sendable, Equatable {
        public var id: Int
        public var title: String
    }

    public var name: String
    public var repo: String
    public var revision: String
    public var cut: Int
    public var books: [Book]
    public var anchors = 512
    public var anchorTokens = 64
    public var heldOut = 64
    public var heldOutTokens = 256
    public var seed: UInt64 = 0xA11C_0C0D

    public var hubSource: String { "hub:\(repo)@\(revision)" }

    /// SmolLM2-135M at the revision RaoLM's tokenizer was vendored from, cut after block 20.
    public static let smolLM2_135M = PackSource(
        name: "smollm2-135m", repo: "HuggingFaceTB/SmolLM2-135M", revision: "93efa2f097d58c2a74874c7e644dbc9b0cee75a2", cut: 20,
        books: [
            Book(id: 1342, title: "Pride and Prejudice"), Book(id: 84, title: "Frankenstein"),
            Book(id: 11, title: "Alice's Adventures in Wonderland"), Book(id: 1661, title: "The Adventures of Sherlock Holmes"),
            Book(id: 98, title: "A Tale of Two Cities"), Book(id: 2701, title: "Moby Dick"), Book(id: 1232, title: "The Prince"),
            Book(id: 1228, title: "On the Origin of Species"),
        ])

    /// The source a preset's shape is cut from, if it has a trunk.
    public static func source(for config: RaoLMConfig) throws -> PackSource {
        var source = smolLM2_135M
        let base = RaoLMConfig.smolLM2_135M
        guard config.hiddenSize == base.hiddenSize, config.numHiddenLayers == base.numHiddenLayers,
              config.intermediateSize == base.intermediateSize, config.numAttentionHeads == base.numAttentionHeads
        else {
            throw UmbrellaPackError.shape("only SmolLM2-135M's shape (30 × 576) has a pack; this config is \(config.numHiddenLayers) × \(config.hiddenSize)")
        }
        source.cut = config.cut
        return source
    }
}

public enum UmbrellaPacks {
    /// <braid>/umbrella/<name>-cut<cut>.json: the pack a source names.
    public struct Current: Codable, Sendable, Equatable {
        public var name: String
        public var source: String
        public var cut: Int
        public var sha256: String
    }

    static func pointer(_ layout: BraidLayout, _ source: PackSource) -> URL {
        layout.packs.appendingPathComponent("\(source.name)-cut\(source.cut).json")
    }

    /// The pack a braid of `config`'s shape uses. With no trunk: the seeded vocabulary. With one:
    /// the pack built before, or one built now (it downloads the base model and the books once).
    public static func ensure(
        layout: BraidLayout, config: RaoLMConfig, tokenizer: RaoTokenizer, progress: ((String) -> Void)? = nil
    ) throws -> UmbrellaPack {
        guard config.hasTrunk else {
            return UmbrellaPack(vocabulary: try BraidVocabulary.ensure(layout: layout, config: config, tokenizer: tokenizer))
        }
        let source = try PackSource.source(for: config)
        if let current = try? JSONCoding.read(Current.self, from: pointer(layout, source)), current.source == source.hubSource,
           let pack = try? UmbrellaPack.load(from: layout.pack(sha256: current.sha256)),
           pack.info?.tokenizerSHA256 == tokenizer.tokenizerSHA256 {
            return pack
        }
        do {
            return try build(source: source, layout: layout, tokenizer: tokenizer, progress: progress)
        } catch {
            throw UmbrellaPackError.missing(
                "the \(source.name) pack could not be built (\(error)); build it once while online: raolm umbrella build --data-dir \(layout.root.deletingLastPathComponent().path)")
        }
    }

    /// Downloads the base model and the books, cuts the pack and saves it under the braid.
    public static func build(
        source: PackSource, layout: BraidLayout, tokenizer: RaoTokenizer, progress: ((String) -> Void)? = nil
    ) throws -> UmbrellaPack {
        progress?("downloading \(source.repo) at \(source.revision.prefix(8))…")
        let repo = source.repo
        let revision = source.revision
        let folder = try Blocking.run {
            try await HubApi().snapshot(from: repo, revision: revision, matching: ["config.json", "model.safetensors", "tokenizer.json"])
        }
        let tokenizerSHA = try ContentHash.sha256Hex(fileAt: folder.appendingPathComponent("tokenizer.json"))
        guard tokenizerSHA == tokenizer.tokenizerSHA256 else {
            throw UmbrellaPackError.fingerprint(expected: tokenizer.tokenizerSHA256, found: tokenizerSHA, what: "tokenizer (a pack must use RaoLM's own)")
        }
        var config = try JSONCoding.read(RaoLMConfig.self, from: folder.appendingPathComponent(Checkpoint.configFile))
        config.cut = source.cut
        try config.validate()
        guard config.tieWordEmbeddings else { throw UmbrellaPackError.shape("the base model's head is not tied to its embedding") }
        progress?("loading the base model (\(config.numHiddenLayers) × \(config.hiddenSize), cut at \(config.cut))")
        let model = RaoTransformer(config)
        try Checkpoint.loadWeights(into: model, from: folder)
        let originSHA = try ContentHash.sha256Hex(fileAt: folder.appendingPathComponent(Checkpoint.weightsFile))
        let vocabulary = VocabularyPack.from(model: model, tokenizerSHA256: tokenizer.tokenizerSHA256, originSHA256: originSHA)
        var base: [String: MLXArray] = [:]
        for (key, value) in model.parameters().flattened() where RaoTransformer.blockIndex(ofKey: key) != nil { base[key] = value }

        let samples = try commonsSamples(source: source, layout: layout, tokenizer: tokenizer, progress: progress)
        progress?("reading the base model's cut states on \(samples.anchors.count) anchors")
        let anchorStates = UmbrellaPack.anchorStates(model: model, anchors: samples.anchors.map(\.tokens))
        let trunk = base.filter { (RaoTransformer.blockIndex(ofKey: $0.key) ?? -1) >= config.cut }.map { (key: $0.key, value: $0.value) }
        let sha = UmbrellaPack.fingerprint(embedding: vocabulary.embedding, norm: vocabulary.norm, trunk: trunk)
        let info = PackInfo(
            name: source.name, source: source.hubSource, config: config, sha256: sha, vocabularySHA256: vocabulary.sha256,
            anchorCount: samples.anchors.count, heldOutCount: samples.heldOut.count, texts: samples.texts,
            tokenizerSHA256: tokenizer.tokenizerSHA256)
        let pack = UmbrellaPack(vocabulary: vocabulary, info: info, base: base, anchors: samples.anchors, anchorStates: anchorStates,
                                heldOut: samples.heldOut)
        progress?("saving pack \(sha.prefix(12))…")
        let directory = layout.pack(sha256: sha)
        try? FileManager.default.removeItem(at: directory)
        try pack.save(to: directory)
        try JSONCoding.write(Current(name: source.name, source: source.hubSource, cut: source.cut, sha256: sha), to: pointer(layout, source))
        return try UmbrellaPack.load(from: directory)
    }

    // MARK: - Commons text

    /// Anchors and held-out snippets, each starting at a paragraph of a book, spread evenly over
    /// the books, at offsets drawn from a fixed seed.
    static func commonsSamples(
        source: PackSource, layout: BraidLayout, tokenizer: RaoTokenizer, progress: ((String) -> Void)?
    ) throws -> (anchors: [PackSnippet], heldOut: [PackSnippet], texts: [PackText]) {
        var anchors: [PackSnippet] = []
        var heldOut: [PackSnippet] = []
        var texts: [PackText] = []
        var rng = SplitMix64(seed: source.seed)
        let books = source.books
        for (b, book) in books.enumerated() {
            progress?("book \(b + 1)/\(books.count): \(book.title)")
            let body = try text(of: book, layout: layout)
            let paragraphs = body.components(separatedBy: "\n\n").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { $0.count >= 200 }
            guard !paragraphs.isEmpty else { continue }
            let wantAnchors = share(source.anchors, of: books.count, at: b)
            let wantHeld = share(source.heldOut, of: books.count, at: b)
            var used = 0
            var picked = Set<Int>()
            func snippet(_ length: Int) -> [Int]? {
                for _ in 0..<64 {
                    let start = Int(rng.next() % UInt64(paragraphs.count))
                    guard !picked.contains(start) else { continue }
                    var text = ""
                    var i = start
                    while text.count < length * 8, i < paragraphs.count {
                        text += (text.isEmpty ? "" : "\n\n") + paragraphs[i]
                        i += 1
                    }
                    let tokens = tokenizer.encode(text)
                    guard tokens.count >= length else { continue }
                    for j in start..<i { picked.insert(j) }
                    return Array(tokens.prefix(length))
                }
                return nil
            }
            for _ in 0..<wantAnchors {
                guard let tokens = snippet(source.anchorTokens) else { break }
                anchors.append(PackSnippet(source: "gutenberg:\(book.id)", tokens: tokens))
                used += tokens.count
            }
            for _ in 0..<wantHeld {
                guard let tokens = snippet(source.heldOutTokens) else { break }
                heldOut.append(PackSnippet(source: "gutenberg:\(book.id)", tokens: tokens))
                used += tokens.count
            }
            texts.append(PackText(source: "gutenberg:\(book.id)", title: book.title, sha256: ContentHash.sha256Hex(body), tokens: used))
        }
        return (anchors, heldOut, texts)
    }

    /// `total` split as evenly as it goes over `count` parts: part `index`'s share.
    static func share(_ total: Int, of count: Int, at index: Int) -> Int {
        total / max(1, count) + (index < total % max(1, count) ? 1 : 0)
    }

    /// A book's body without the distributor's header and footer, lines within a paragraph joined.
    static func text(of book: PackSource.Book, layout: BraidLayout) throws -> String {
        let directory = layout.packs.appendingPathComponent("texts", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let file = directory.appendingPathComponent("pg\(book.id).txt")
        var raw: String
        if let cached = try? String(contentsOf: file, encoding: .utf8) {
            raw = cached
        } else {
            guard let url = URL(string: "https://www.gutenberg.org/cache/epub/\(book.id)/pg\(book.id).txt") else {
                throw UmbrellaPackError.missing("book \(book.id)")
            }
            let data = try Blocking.run { try await URLSession.shared.data(from: url).0 }
            raw = String(decoding: data, as: UTF8.self)
            try data.write(to: file, options: .atomic)
        }
        raw = raw.replacingOccurrences(of: "\r\n", with: "\n")
        if let start = raw.range(of: "*** START OF") {
            raw = String(raw[start.upperBound...])
            if let line = raw.firstIndex(of: "\n") { raw = String(raw[raw.index(after: line)...]) }
        }
        if let end = raw.range(of: "*** END OF") { raw = String(raw[..<end.lowerBound]) }
        return raw.components(separatedBy: "\n\n")
            .map { $0.split(separator: "\n").map { $0.trimmingCharacters(in: .whitespaces) }.joined(separator: " ") }
            .filter { !$0.isEmpty }
            .joined(separator: "\n\n")
    }
}
