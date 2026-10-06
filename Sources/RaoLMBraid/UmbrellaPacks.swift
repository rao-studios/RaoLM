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
import FrigateBridge
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

    /// SmolLM2-360M, pinned, cut after block 22.
    public static let smolLM2_360M = PackSource(
        name: "smollm2-360m", repo: "HuggingFaceTB/SmolLM2-360M", revision: "f8027fd0eaeea54caa13c31d31b9fdc459c38b49", cut: 22,
        books: smolLM2_135M.books)

    /// The source a preset's shape is cut from, if it has a trunk.
    public static func source(for config: RaoLMConfig) throws -> PackSource {
        for (candidate, shape) in [(smolLM2_135M, RaoLMConfig.smolLM2_135M), (smolLM2_360M, RaoLMConfig.smolLM2_360M)] {
            guard config.hiddenSize == shape.hiddenSize, config.numHiddenLayers == shape.numHiddenLayers,
                  config.intermediateSize == shape.intermediateSize, config.numAttentionHeads == shape.numAttentionHeads else { continue }
            var source = candidate
            source.cut = config.cut
            return source
        }
        throw UmbrellaPackError.shape(
            "no pack source for \(config.numHiddenLayers) × \(config.hiddenSize); SmolLM2-135M (30 × 576) and SmolLM2-360M (32 × 960) have one, and raolm umbrella import makes others")
    }

    /// A source for any repository on the hub, at a revision, cut where asked; its anchors from the same books.
    public static func hub(repo: String, revision: String, cut: Int, name: String? = nil) -> PackSource {
        PackSource(name: name ?? repo.split(separator: "/").last.map { String($0).lowercased() } ?? repo, repo: repo, revision: revision, cut: cut,
                   books: smolLM2_135M.books)
    }
}

public enum UmbrellaPacks {
    /// <braid>/umbrella/<name>-cut<cut>.json: the pack a source named before the registry
    /// (`PackRegistry` folds these in and no longer writes them).
    public struct Current: Codable, Sendable, Equatable {
        public var name: String
        public var source: String
        public var cut: Int
        public var sha256: String
    }

    /// The pack a braid of `config`'s shape uses. With no trunk: the seeded vocabulary. With one:
    /// `pack` when named (a registered name or hash prefix), else the registry's current pack for
    /// the shape, else one built now from the hub (it downloads the base model and the books once).
    public static func ensure(
        layout: BraidLayout, config: RaoLMConfig, tokenizer: RaoTokenizer, pack reference: String? = nil, progress: ((String) -> Void)? = nil
    ) throws -> UmbrellaPack {
        guard config.hasTrunk else {
            return UmbrellaPack(vocabulary: try BraidVocabulary.ensure(layout: layout, config: config, tokenizer: tokenizer))
        }
        let registry = PackRegistry.load(layout)
        if let reference {
            guard let entry = registry.resolve(reference) else { throw UmbrellaPackError.missing("no pack '\(reference)' under \(layout.packs.path)") }
            guard entry.slot == PackRegistry.slot(config) else {
                throw UmbrellaPackError.shape("pack \(entry.sha256.prefix(12)) is \(entry.slot), the braid's shape is \(PackRegistry.slot(config))")
            }
            return try load(entry.sha256, layout: layout, tokenizer: tokenizer)
        }
        if let sha = registry.current[PackRegistry.slot(config)], let pack = try? load(sha, layout: layout, tokenizer: tokenizer) {
            return pack
        }
        let source = try PackSource.source(for: config)
        do {
            return try build(source: source, layout: layout, tokenizer: tokenizer, progress: progress)
        } catch {
            throw UmbrellaPackError.missing(
                "the \(source.name) pack could not be built (\(error)); build it once while online: raolm umbrella build --data-dir \(layout.root.deletingLastPathComponent().path)")
        }
    }

    /// A registered pack, refused when its tokenizer is not the one in use.
    public static func load(_ sha256: String, layout: BraidLayout, tokenizer: RaoTokenizer) throws -> UmbrellaPack {
        let pack = try UmbrellaPack.load(from: layout.pack(sha256: sha256))
        guard pack.sha256 == sha256 else { throw UmbrellaPackError.fingerprint(expected: sha256, found: pack.sha256, what: "pack") }
        if let found = pack.info?.tokenizerSHA256, found != tokenizer.tokenizerSHA256 {
            throw UmbrellaPackError.fingerprint(expected: tokenizer.tokenizerSHA256, found: found, what: "tokenizer")
        }
        return pack
    }

    /// Saves a pack under the braid's data root and registers it.
    @discardableResult
    public static func store(_ pack: UmbrellaPack, layout: BraidLayout) throws -> UmbrellaPack {
        guard let info = pack.info else { throw UmbrellaPackError.noBase }
        let directory = layout.pack(sha256: info.sha256)
        try pack.save(to: directory)
        var registry = PackRegistry.load(layout)
        registry.register(info)
        try registry.save(layout)
        return try UmbrellaPack.load(from: directory)
    }

    /// A pack from a model trained on from `parent`: every block and the vocabulary as trained, the
    /// parent's anchors and held-out snippets (the same tokens, since the tokenizer is the same)
    /// with the anchors' cut states read under the new model, the parent and the recipe recorded.
    /// Registered, not made current.
    public static func continued(
        model: RaoTransformer, parent: UmbrellaPack, name: String, recipe: PackRecipe, layout: BraidLayout, tokenizer: RaoTokenizer
    ) throws -> UmbrellaPack {
        guard let info = parent.info else { throw UmbrellaPackError.noBase }
        let vocabulary = VocabularyPack.from(model: model, tokenizerSHA256: tokenizer.tokenizerSHA256, originSHA256: parent.sha256)
        var base: [String: MLXArray] = [:]
        for (key, value) in model.parameters().flattened() where RaoTransformer.blockIndex(ofKey: key) != nil { base[key] = value.asType(.float32) }
        let anchorStates = UmbrellaPack.anchorStates(model: model, anchors: parent.anchors.map(\.tokens))
        let pack = UmbrellaPack.make(
            vocabulary: vocabulary, name: name, source: "pack:\(parent.sha256.prefix(12))+run:\(recipe.runID)", config: info.config, base: base,
            anchors: parent.anchors, anchorStates: anchorStates, heldOut: parent.heldOut, texts: info.texts, tokenizerSHA256: tokenizer.tokenizerSHA256,
            parent: parent.sha256, recipe: recipe)
        return try store(pack, layout: layout)
    }

    /// Downloads the base model and the books, cuts the pack and saves it under the braid.
    public static func build(
        source: PackSource, layout: BraidLayout, tokenizer: RaoTokenizer, progress: ((String) -> Void)? = nil
    ) throws -> UmbrellaPack {
        let repo = source.repo
        let revision = source.revision
        let patterns = ["config.json", "*.safetensors", "model.safetensors.index.json", "tokenizer.json"]
        // A copy already on this Mac at that revision — in the work area or the Rao stack's
        // folder — is used where it sits; only a missing one downloads, into the T9 work area.
        let home = WorkArea.modelsHome()
        let onDisk = HubDownloader.materializedSnapshot(
            id: repo, revision: revision, matching: patterns, roots: HubDownloader.snapshotRoots(home: home))
        if onDisk == nil, !WorkArea.isAvailable() {
            throw UmbrellaPackError.missing(
                "the T9 work area \(WorkArea.url().path) is not mounted, so \(repo) has nowhere to download — plug in the T9 or set \(WorkArea.environmentKey)")
        }
        if onDisk == nil { progress?("downloading \(repo) at \(revision.prefix(8))…") }
        let folder = try onDisk ?? Blocking.run {
            try await HubDownloader(home: home).download(
                id: repo, revision: revision, matching: patterns, useLatest: false, progressHandler: { _ in })
        }
        let tokenizerSHA = try ContentHash.sha256Hex(fileAt: folder.appendingPathComponent("tokenizer.json"))
        // Sharded checkpoints are read through their index; Checkpoint.loadWeights merges the shards.
        guard tokenizerSHA == tokenizer.tokenizerSHA256 else {
            throw UmbrellaPackError.fingerprint(expected: tokenizer.tokenizerSHA256, found: tokenizerSHA, what: "tokenizer (a pack must use RaoLM's own)")
        }
        var config = try JSONCoding.read(RaoLMConfig.self, from: folder.appendingPathComponent(Checkpoint.configFile))
        config.cut = source.cut
        try config.validate()
        // By design (Docs/ARCHITECTURE.md, "The tied head stays"): the umbrella reads every strand out
        // through the embedding itself, so a commons must tie its head.
        guard config.tieWordEmbeddings else {
            throw UmbrellaPackError.shape("the base model's head is not tied to its embedding; the umbrella's readout is the embedding, so a commons must tie (SmolLM2, Llama 3.2 1B/3B, Qwen2.5 ≤ 3B, Gemma do)")
        }
        progress?("loading the base model (\(config.numHiddenLayers) × \(config.hiddenSize), cut at \(config.cut))")
        let model = RaoTransformer(config)
        try Checkpoint.loadWeights(into: model, from: folder)
        // C0: RaoLM's transformer reads the weights as Frigate's own Llama does.
        let check = try ReferenceCheck.llama(folder: folder, model: model, prompts: [tokenizer.encode("It is a truth universally acknowledged, that a single man in possession of a good fortune"),
                                                                                       tokenizer.encode("The river ran north past the mill, and the miller counted the sacks twice.")])
        progress?(String(format: "reference check: largest logit difference from Frigate's Llama %.2e", check))
        guard check < ReferenceCheck.tolerance else {
            throw UmbrellaPackError.shape(String(format: "RaoLM's transformer and Frigate's Llama disagree on %@ by %.2e (tolerance %.0e)", repo, check, ReferenceCheck.tolerance))
        }
        let originSHA = try ContentHash.sha256Hex(fileAt: folder.appendingPathComponent(Checkpoint.weightsFile))
        let vocabulary = VocabularyPack.from(model: model, tokenizerSHA256: tokenizer.tokenizerSHA256, originSHA256: originSHA)
        var base: [String: MLXArray] = [:]
        for (key, value) in model.parameters().flattened() where RaoTransformer.blockIndex(ofKey: key) != nil { base[key] = value }

        let samples = try commonsSamples(source: source, layout: layout, tokenizer: tokenizer, progress: progress)
        progress?("reading the base model's cut states on \(samples.anchors.count) anchors")
        let anchorStates = UmbrellaPack.anchorStates(model: model, anchors: samples.anchors.map(\.tokens))
        let pack = UmbrellaPack.make(
            vocabulary: vocabulary, name: source.name, source: source.hubSource, config: config, base: base, anchors: samples.anchors,
            anchorStates: anchorStates, heldOut: samples.heldOut, texts: samples.texts, tokenizerSHA256: tokenizer.tokenizerSHA256,
            referenceMaxDifference: check)
        progress?("saving pack \(pack.sha256.prefix(12))…")
        return try store(pack, layout: layout)
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

    /// A book's body, from the cache under the braid's packs (Project Gutenberg on first use).
    static func text(of book: PackSource.Book, layout: BraidLayout) throws -> String {
        try GutenbergText.body(id: book.id, cache: layout.packs.appendingPathComponent("texts", isDirectory: true))
    }
}
