//
//  ProfileBuilder.swift
//  RaoLMBraid
//
//  WHAT: Writes the knowledge profile (Docs/ARCHITECTURE.md, "Step 1 v3") for live versions that
//        were trained before profiles: each node's live version is loaded on its own, the commons
//        reads its corpus, and the profile lands beside the index as a node would have written it.
//        Nothing is retrained and no index changes.
//  PIN:  One node at a time (the pack plus one version in memory). The pack is the one the live
//        versions name; it must carry a base model, since the commons is what reads the corpus.
//

import Foundation
import MLX
import RaoLMCore
import RaoLMModel
import RaoLMProvenance
import RaoLMTraining

public struct ProfileBuildResult: Codable, Sendable, Equatable {
    public var name: String
    public var version: Int
    public var epoch: Int
    public var entries: Int
    public var weighted: Int
    public var liftTotal: Float
    public var commonsLoss: Float
    public var threadLoss: Float
    public var spread: Float
    public var k: Int
    public var seconds: Double
    /// The version already had a profile and `force` was off.
    public var skipped: Bool
}

public enum ProfileBuilder {
    public static func run(
        layout: BraidLayout, tokenizer: RaoTokenizer, names only: [String]? = nil, force: Bool = false, progress: ((String) -> Void)? = nil
    ) throws -> [ProfileBuildResult] {
        Memory.cacheLimit = (HypervisorSettings().cacheLimitMB ?? 2048) * 1_048_576
        let names = only ?? ((try? FileManager.default.contentsOfDirectory(atPath: layout.nodes.path)) ?? []).filter(BraidLayout.isValidName).sorted()
        var pack: UmbrellaPack?
        var commons: RaoTransformer?
        var results: [ProfileBuildResult] = []
        for name in names {
            let node = layout.node(name)
            guard let live = try? JSONCoding.read(LivePointer.self, from: node.live) else {
                progress?("\(name): no live version")
                continue
            }
            let directory = node.version(live.version)
            let version = try JSONCoding.read(NodeVersion.self, from: directory.appendingPathComponent(NodeVersion.fileName))
            let provenance = RunLayout.provenance(directory, epoch: version.epoch)
            if !force, let existing = try ThreadProfile.load(from: provenance) {
                results.append(result(name, version, existing, skipped: true))
                progress?("\(name): v\(version.version) already has a profile")
                continue
            }
            if pack == nil {
                guard let sha = version.packSHA256 else {
                    throw BraidSessionError.io("\(name)'s live version runs no umbrella pack: a profile needs the commons (a pack with a base model)")
                }
                let loaded = try UmbrellaPack.load(from: layout.pack(sha256: sha))
                guard loaded.hasBase else { throw BraidSessionError.io("the pack \(sha.prefix(12)) has no base model: a profile needs the commons") }
                pack = loaded
                commons = try loaded.baseModel()
            }
            guard let pack, let commons, version.packSHA256 == pack.sha256 else {
                throw BraidSessionError.io("\(name)'s live version runs another pack than the first node's")
            }
            progress?("\(name): v\(version.version), the commons reads the corpus")
            let context = try RunContext.load(runDirectory: directory, epoch: version.epoch, allowWeakIndex: true, tokenizer: tokenizer)
            guard let profile = try ThreadProfile.compute(
                commons: commons, index: context.index, corpus: try context.tokenizedCorpus(),
                seqLen: context.manifest.hyperparameters.seqLen, packSHA256: pack.sha256, epoch: version.epoch)
            else {
                progress?("\(name): predicts nothing better than the commons; no profile")
                continue
            }
            try profile.save(to: provenance)
            results.append(result(name, version, profile, skipped: false))
            Memory.clearCache()
        }
        return results
    }

    static func result(_ name: String, _ version: NodeVersion, _ profile: ThreadProfile, skipped: Bool) -> ProfileBuildResult {
        ProfileBuildResult(
            name: name, version: version.version, epoch: version.epoch, entries: profile.info.entries, weighted: profile.info.weighted,
            liftTotal: profile.info.liftTotal, commonsLoss: profile.info.commonsLoss, threadLoss: profile.info.threadLoss,
            spread: ThreadProfile.spread(centroids: profile.centroids, hidden: profile.hidden), k: profile.k,
            seconds: profile.info.seconds, skipped: skipped)
    }
}
