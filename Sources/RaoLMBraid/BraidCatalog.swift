//
//  BraidCatalog.swift
//  RaoLMBraid
//
//  WHAT: Every braid on this machine, for the studio's first screen and `raolm braid list`: the
//        data root's own, then each braid in a braid area (a directory whose children are data
//        roots, as the T9 work area's are). Per braid: its Threads and their live versions, the
//        model they train, the commons it runs on, what fed it, and whether its nodes ran offline.
//  PIN:  Read only: nothing here writes into a braid. The pack registry is read as saved, never
//        folded (`PackRegistry.load` writes what it folds in).
//

import Foundation
import RaoLMCore
import RaoLMModel

public struct BraidCatalogEntry: Sendable, Equatable {
    /// "home" for the data root the studio was opened on; else the data root's directory name.
    public var name: String
    /// The data root: the braid is its braid/ directory.
    public var root: URL
    public var isHome: Bool
    public var nodes: [BraidNodeSpec]
    /// Each node's live version; a node with none is absent.
    public var live: [String: Int]
    public var preset: String
    public var arm: String?
    /// The name of the pack whose base model is the commons; nil when the braid's pack has none.
    public var commons: String?
    public var commonsSHA256: String?
    /// The dataset the nodes were fed from; nil for a generated mock world.
    public var dataset: String?
    public var documentsPerNode: Int?
    public var offline: Bool
    /// When a node last went live.
    public var updated: Date?

    public init(
        name: String, root: URL, isHome: Bool, nodes: [BraidNodeSpec], live: [String: Int], preset: String, arm: String? = nil,
        commons: String? = nil, commonsSHA256: String? = nil, dataset: String? = nil, documentsPerNode: Int? = nil, offline: Bool = false,
        updated: Date? = nil
    ) {
        self.name = name
        self.root = root
        self.isHome = isHome
        self.nodes = nodes
        self.live = live
        self.preset = preset
        self.arm = arm
        self.commons = commons
        self.commonsSHA256 = commonsSHA256
        self.dataset = dataset
        self.documentsPerNode = documentsPerNode
        self.offline = offline
        self.updated = updated
    }

    /// "Ambient v7 · Craft v5 · Veil —".
    public var threads: String {
        nodes.map { "\($0.label) " + (live[$0.name].map { "v\($0)" } ?? "—") }.joined(separator: " · ")
    }

    /// "base · passage-break".
    public var model: String { preset + (arm.map { " · \($0)" } ?? "") }
}

public enum BraidCatalog {
    /// The braid at `home` first, then those under each area, the last to go live first.
    public static func scan(home: URL, areas: [URL]) -> [BraidCatalogEntry] {
        var seen: Set<String> = [home.standardizedFileURL.path]
        let first = entry(root: home, name: "home", isHome: true).map { [$0] } ?? []
        var others: [BraidCatalogEntry] = []
        for area in areas {
            let children = (try? FileManager.default.contentsOfDirectory(at: area, includingPropertiesForKeys: nil, options: .skipsHiddenFiles)) ?? []
            for child in children where seen.insert(child.standardizedFileURL.path).inserted {
                if let entry = entry(root: child, name: child.lastPathComponent, isHome: false) { others.append(entry) }
            }
        }
        others.sort { a, b in
            let (x, y) = (a.updated ?? .distantPast, b.updated ?? .distantPast)
            return x != y ? x > y : a.name < b.name
        }
        return first + others
    }

    /// The braid in `root`/braid, or nil when there is none (no world.json and no node).
    public static func entry(root: URL, name: String, isHome: Bool) -> BraidCatalogEntry? {
        let layout = BraidLayout(root: root.appendingPathComponent("braid", isDirectory: true))
        let record = MockWorld.Record.load(layout)
        let directories = ((try? FileManager.default.contentsOfDirectory(atPath: layout.nodes.path)) ?? [])
            .filter { BraidLayout.isValidName($0) }.sorted()
        let names = record?.names ?? directories
        guard !names.isEmpty else { return nil }
        let serves = names.map { try? JSONCoding.read(NodeServerOptions.self, from: layout.node($0).directory.appendingPathComponent(NodeServerOptions.fileName)) }
        let pointers = names.map { try? JSONCoding.read(LivePointer.self, from: layout.node($0).live) }
        var live: [String: Int] = [:]
        for (name, pointer) in zip(names, pointers) { live[name] = pointer?.version }
        let nodes = zip(names, serves).map { name, serve in
            BraidNodeSpec(name: name, label: serve?.label ?? name.prefix(1).uppercased() + name.dropFirst())
        }
        let preset = record?.preset ?? serves.compactMap { $0?.settings.preset }.first ?? "tiny"
        // The pack it runs on: as recorded, else the one its live versions were trained on, else its nodes'.
        let sha = record?.packSHA256 ?? BraidOptions.trainedPack(layout, names: names) ?? serves.compactMap { $0?.packSHA256 }.first
        return BraidCatalogEntry(
            name: name, root: root.standardizedFileURL, isHome: isHome, nodes: nodes, live: live, preset: preset,
            arm: record == nil ? serves.compactMap { $0?.settings.arm }.first : record?.arm, commons: sha.map { packName($0, layout: layout) }, commonsSHA256: sha,
            dataset: record?.dataset?.name, documentsPerNode: record?.shape.documentsPerNode,
            offline: serves.contains { $0?.offline == true }, updated: pointers.compactMap { $0?.promotedAt }.max())
    }

    /// A pack's name from its pack.json, else the registry as saved, else its hash's first 12 characters.
    static func packName(_ sha256: String, layout: BraidLayout) -> String {
        if let info = try? JSONCoding.read(PackInfo.self, from: layout.pack(sha256: sha256).appendingPathComponent(UmbrellaPack.infoFile)) {
            return info.name
        }
        let registry = try? JSONCoding.read(PackRegistry.self, from: PackRegistry.url(layout))
        return registry?.entry(sha256)?.name ?? String(sha256.prefix(12))
    }
}
