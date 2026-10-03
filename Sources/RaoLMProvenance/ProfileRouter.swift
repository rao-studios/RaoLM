//
//  ProfileRouter.swift
//  RaoLMProvenance
//
//  WHAT: Which Threads a generation opens, by their knowledge profiles (Docs/ARCHITECTURE.md,
//        "Step 1 v3"): each Thread scores by its best centroid's cosine with the prompt's query;
//        Threads reaching half the best score open, at most six, with the commons. That set is
//        the cluster this inference runs on; nothing is clustered in advance.
//  PIN:  Stateless and brute force: the route a bench computes without generating is the route a
//        generation takes. A link without a profile (the commons, or a node from before profiles)
//        always opens. No profile anywhere, or no positive score: every Thread opens.
//

import Foundation
import RaoLMCore

public struct ProfileRoute: Sendable, Equatable {
    /// Per link: whether this generation opens it.
    public var candidates: [Bool]
    /// Per link: its best centroid's cosine with the query; nil for a link without a profile.
    public var scores: [Float?]
    /// Every link opened because nothing could be ranked.
    public var unrouted: Bool
    /// The best score among Threads with a profile.
    public var best: Float?

    /// Threads with a profile that open (the commons and profile-less links not counted).
    public var opened: Int { zip(candidates, scores).filter { $0.0 && $0.1 != nil }.count }
    public var indices: [Int] { candidates.indices.filter { candidates[$0] } }
}

public struct ProfileRouter: Sendable {
    /// Per link: its centroids as unit rows; nil for a link that always opens.
    let profiles: [[[Float]]?]
    public let bar: Float
    public let cap: Int

    public init(profiles: [[Float]?], hidden: Int, bar: Float = ProfileSettings.bar, cap: Int = ProfileSettings.cap) {
        self.profiles = profiles.map { flat in
            guard let flat, hidden > 0, flat.count >= hidden else { return nil }
            return (0..<(flat.count / hidden)).map { Self.unit(Array(flat[($0 * hidden)..<(($0 + 1) * hidden)])) }
        }
        self.bar = bar
        self.cap = cap
    }

    public init(descriptors: [StrandDescriptor], bar: Float = ProfileSettings.bar, cap: Int = ProfileSettings.cap) {
        let hidden = descriptors.first?.hiddenSize ?? 0
        self.init(profiles: descriptors.map { $0.isCommons ? nil : $0.profile?.values }, hidden: hidden, bar: bar, cap: cap)
    }

    /// Links that carry a profile.
    public var profiled: Int { profiles.filter { $0 != nil }.count }

    static func unit(_ x: [Float]) -> [Float] {
        let norm = x.reduce(0) { $0 + $1 * $1 }.squareRoot()
        return norm > 1e-12 ? x.map { $0 / norm } : x
    }

    /// Each link's best centroid's cosine with the query.
    public func scores(query: [Float]) -> [Float?] {
        let q = Self.unit(query)
        return profiles.map { rows in
            rows.map { $0.map { row in zip(row, q).reduce(0) { $0 + $1.0 * $1.1 } }.max() ?? 0 }
        }
    }

    public func route(query: [Float]) -> ProfileRoute {
        let scores = scores(query: query)
        let ranked = scores.indices.compactMap { i in scores[i].map { (i, $0) } }
        let best = ranked.map(\.1).max()
        guard let best, best > 0 else {
            return ProfileRoute(candidates: scores.map { _ in true }, scores: scores, unrouted: true, best: best)
        }
        var candidates = scores.map { $0 == nil }
        let open = ranked.filter { $0.1 >= bar * best }
            .sorted { $0.1 != $1.1 ? $0.1 > $1.1 : $0.0 < $1.0 }
            .prefix(cap)
        for (i, _) in open { candidates[i] = true }
        return ProfileRoute(candidates: candidates, scores: scores, unrouted: false, best: best)
    }
}
