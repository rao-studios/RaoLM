//
//  PackRegistry.swift
//  RaoLMBraid
//
//  WHAT: Every umbrella pack a braid's data root holds, with its lineage, and which one a braid of
//        each shape uses. The commons is Rao's own foundation model: a pack is built (from open
//        weights) or trained (from a parent pack and corpora), registered, and made current only
//        on the owner's say.
//  OUT:  <braid>/umbrella/packs.json.
//  PIN:  A slot is a shape and a cut ("30x576-cut20"): the packs that can serve a braid of that
//        shape. `use` sets a slot's current pack; nothing else does, except building the first
//        pack of an empty slot. The old pointers (<name>-cut<cut>.json) are folded in on first
//        read and left on disk.
//

import Foundation
import RaoLMCore
import RaoLMModel

public struct PackRegistry: Codable, Sendable, Equatable {
    public struct Entry: Codable, Sendable, Equatable {
        public var sha256: String
        public var seamSHA256: String
        public var name: String
        public var version: Int
        public var parent: String?
        public var source: String
        public var layers: Int
        public var hidden: Int
        public var cut: Int
        public var tokenizerSHA256: String
        public var createdAt: Date

        public init(_ info: PackInfo) {
            sha256 = info.sha256
            seamSHA256 = info.seam
            name = info.name
            version = info.version
            parent = info.parent
            source = info.source
            layers = info.config.numHiddenLayers
            hidden = info.config.hiddenSize
            cut = info.config.cut
            tokenizerSHA256 = info.tokenizerSHA256
            createdAt = info.createdAt
        }

        public var slot: String { PackRegistry.slot(layers: layers, hidden: hidden, cut: cut) }
    }

    public static let fileName = "packs.json"

    public var packs: [Entry] = []
    /// Slot → the pack's name.
    public var current: [String: String] = [:]

    public init() {}

    public static func slot(layers: Int, hidden: Int, cut: Int) -> String { "\(layers)x\(hidden)-cut\(cut)" }
    public static func slot(_ config: RaoLMConfig) -> String {
        slot(layers: config.numHiddenLayers, hidden: config.hiddenSize, cut: config.cut)
    }

    static func url(_ layout: BraidLayout) -> URL { layout.packs.appendingPathComponent(fileName) }

    /// The registry, with any pack directory or old pointer it does not list folded in.
    public static func load(_ layout: BraidLayout) -> PackRegistry {
        var registry = (try? JSONCoding.read(PackRegistry.self, from: url(layout))) ?? PackRegistry()
        let before = registry
        let directories = (try? FileManager.default.contentsOfDirectory(at: layout.packs, includingPropertiesForKeys: nil)) ?? []
        for directory in directories where !directory.lastPathComponent.hasPrefix(".") {
            let infoURL = directory.appendingPathComponent(UmbrellaPack.infoFile)
            guard let info = try? JSONCoding.read(PackInfo.self, from: infoURL), registry.entry(info.sha256) == nil else { continue }
            registry.packs.append(Entry(info))
        }
        for file in directories where file.pathExtension == "json" && file.lastPathComponent != fileName {
            guard let pointer = try? JSONCoding.read(UmbrellaPacks.Current.self, from: file),
                  let entry = registry.entry(pointer.sha256), registry.current[entry.slot] == nil else { continue }
            registry.current[entry.slot] = entry.sha256
        }
        registry.packs.sort { $0.createdAt < $1.createdAt }
        // A slot with packs and no current one takes its first, as registering it would have.
        for entry in registry.packs where registry.current[entry.slot] == nil { registry.current[entry.slot] = entry.sha256 }
        if registry != before { try? registry.save(layout) }
        return registry
    }

    public func save(_ layout: BraidLayout) throws {
        try FileManager.default.createDirectory(at: layout.packs, withIntermediateDirectories: true)
        try JSONCoding.write(self, to: Self.url(layout))
    }

    public func entry(_ sha256: String) -> Entry? { packs.first { $0.sha256 == sha256 } }

    /// A pack by its name, a prefix of its hash (at least 6 characters), or a slot's current.
    public func resolve(_ reference: String) -> Entry? {
        if let sha = current[reference], let entry = entry(sha) { return entry }
        if let exact = packs.first(where: { $0.sha256 == reference }) { return exact }
        let named = packs.filter { $0.name == reference }
        if named.count == 1 { return named[0] }
        guard reference.count >= 6 else { return nil }
        let prefixed = packs.filter { $0.sha256.hasPrefix(reference) }
        return prefixed.count == 1 ? prefixed[0] : nil
    }

    /// Registers a pack; the first pack of a slot becomes its current one.
    public mutating func register(_ info: PackInfo) {
        if entry(info.sha256) == nil { packs.append(Entry(info)) }
        let slot = Self.slot(info.config)
        if current[slot] == nil { current[slot] = info.sha256 }
    }

    /// Makes a registered pack its slot's current one.
    public mutating func use(_ sha256: String) throws {
        guard let entry = entry(sha256) else { throw UmbrellaPackError.missing("no pack \(sha256.prefix(12)) in the registry") }
        current[entry.slot] = entry.sha256
    }

    /// A pack's ancestors, nearest first.
    public func lineage(_ sha256: String) -> [Entry] {
        var chain: [Entry] = []
        var seen: Set<String> = []
        var next = entry(sha256)?.parent
        while let sha = next, !seen.contains(sha), let parent = entry(sha) {
            chain.append(parent)
            seen.insert(sha)
            next = parent.parent
        }
        return chain
    }
}
