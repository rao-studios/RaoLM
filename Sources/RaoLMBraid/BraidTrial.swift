//
//  BraidTrial.swift
//  RaoLMBraid
//
//  WHAT: A trial braid for a commons: a reference braid's documents on another pack, in a new data
//        root beside it, so a commons made by import or training can be trained under, benched and
//        asked like any braid. The step between making a commons and choosing it (Docs/ARCHITECTURE.md,
//        "Phase 4, the commons loop").
//  OUT:  <new root>/braid: world.json (the reference's world, the pack's preset and the pack), each
//        node's documents and feeds, and the pack under umbrella/ as its slot's current one.
//  PIN:  Rebase, never migrate: no version is copied, so every Thread trains from the new base on
//        the next sync, whatever the pack's shape. The reference is only read. A trial is made once:
//        it is staged beside its directory and renamed into place, never written over a directory.
//

import Foundation
import RaoLMCore
import RaoLMModel

public enum BraidTrial {
    /// What a node keeps that a trial leaves behind: what it trained, and the record of running it.
    static let left: Set<String> = ["versions", "snapshots", "live.json", "logs", "node.json", "serve.json"]

    public struct Made: Sendable, Equatable {
        public var root: URL
        public var preset: String
        public var nodes: [String]
    }

    /// The preset whose shape (layers, width, cut) is the pack's: what a node trains under it.
    public static func preset(for entry: PackRegistry.Entry) throws -> String {
        for name in RaoLMConfig.presetNames {
            guard let config = try? RaoLMConfig.preset(name), config.hasTrunk, PackRegistry.slot(config) == entry.slot else { continue }
            return name
        }
        throw BraidSessionError.io("no model preset has the shape \(entry.slot), so no node can train under pack \(entry.name)")
    }

    /// Makes the trial in `root` from the reference braid, with the pack from the workshop's store.
    public static func make(pack entry: PackRegistry.Entry, preset: String? = nil, workshop: BraidLayout, reference: BraidLayout, into root: URL) throws -> Made {
        let files = FileManager.default
        guard var record = MockWorld.Record.load(reference) else {
            throw BraidSessionError.io("no braid at \(reference.root.path): a reference needs a world.json")
        }
        guard !files.fileExists(atPath: root.path) else {
            throw BraidSessionError.io("\(root.path) exists: a trial is made once, in a directory of its own")
        }
        let packDirectory = workshop.pack(sha256: entry.sha256)
        guard files.fileExists(atPath: packDirectory.appendingPathComponent(UmbrellaPack.infoFile).path) else {
            throw BraidSessionError.io("pack \(entry.name) has no directory at \(packDirectory.path)")
        }
        let preset = try preset ?? Self.preset(for: entry)
        let staging = root.deletingLastPathComponent().appendingPathComponent(".\(root.lastPathComponent).staging", isDirectory: true)
        try? files.removeItem(at: staging)
        do {
            let staged = BraidLayout(dataRoot: DataRoot(url: staging))
            for name in record.names {
                let from = reference.node(name).directory
                let to = staged.node(name).directory
                try files.createDirectory(at: to, withIntermediateDirectories: true)
                for item in (try? files.contentsOfDirectory(atPath: from.path)) ?? [] where !left.contains(item) && !item.hasPrefix(".") {
                    try files.copyItem(at: from.appendingPathComponent(item), to: to.appendingPathComponent(item))
                }
            }
            try files.createDirectory(at: staged.packs, withIntermediateDirectories: true)
            try files.copyItem(at: packDirectory, to: staged.pack(sha256: entry.sha256))
            var registry = PackRegistry.load(staged)
            try registry.use(entry.sha256)
            try registry.save(staged)
            record.preset = preset
            record.packSHA256 = entry.sha256
            try record.save(staged)
            try files.moveItem(at: staging, to: root)
        } catch {
            try? files.removeItem(at: staging)
            throw error
        }
        return Made(root: root.standardizedFileURL, preset: preset, nodes: record.names)
    }
}
