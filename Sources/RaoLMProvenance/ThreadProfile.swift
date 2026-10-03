//
//  ThreadProfile.swift
//  RaoLMProvenance
//
//  WHAT: A Thread's knowledge profile (Docs/ARCHITECTURE.md, "Step 1 v3"): what it knows beyond
//        the commons, as a few points in the commons' own state space, which every Thread shares
//        without alignment. The commons reads the Thread's corpus; each position is weighted by the
//        Thread's lift there (the commons' loss minus the Thread's, floored at 0); k-means weighted
//        by lift keeps k centroids. A prompt's query is its commons states averaged by the commons'
//        surprise at each token. The router compares the two.
//  PIN:  The commons pass runs the Thread's own eval windows over the Thread's own corpus, so its
//        entries line up one for one with the index's (the values are checked, and a mismatch
//        throws rather than writing a profile). The Thread's loss is the index's own. Nothing
//        here changes a key, a hit or a share.
//

import Foundation
import MLX
import RaoLMCore
import RaoLMModel
import RaoLMTraining

public enum ThreadProfileFiles {
    public static let arrays = "profile.safetensors"
    public static let info = "profile.json"
}

public enum ProfileError: Error, CustomStringConvertible {
    case misaligned(expected: Int, found: Int)
    case corrupt(String)

    public var description: String {
        switch self {
        case .misaligned(let expected, let found):
            return "the commons pass does not line up with the index: \(found) entries against \(expected)"
        case .corrupt(let what): return "the profile is unreadable: \(what)"
        }
    }
}

/// The profile's constants and the router's, fixed before any numbers (Docs/ARCHITECTURE.md).
public enum ProfileSettings {
    public static let k = 8
    public static let iterations = 25
    public static let seed: UInt64 = 42
    /// A Thread opens when its score reaches this share of the best score.
    public static let bar: Float = 0.5
    /// At most this many Threads open, besides the commons.
    public static let cap = 6
    public static let batchSize = 8
}

public struct ThreadProfileInfo: Codable, Sendable, Equatable {
    public var version = 1
    public var k: Int
    public var hidden: Int
    public var epoch: Int
    /// Index entries read, and those with a positive lift (the ones that place centroids).
    public var entries: Int
    public var weighted: Int
    public var liftTotal: Float
    /// Mean losses over the entries: the commons' and the Thread's own.
    public var commonsLoss: Float
    public var threadLoss: Float
    public var commonsPackSHA256: String
    public var indexSHA256: String
    public var seed: UInt64
    public var iterations: Int
    public var seconds: Double
    public var createdAt: Date
}

public struct ThreadProfile: Sendable, Equatable {
    /// [k × hidden], row-major, in the commons' final-normed state space.
    public var centroids: [Float]
    /// Each centroid's lift mass.
    public var weights: [Float]
    public var info: ThreadProfileInfo

    public var k: Int { info.k }
    public var hidden: Int { info.hidden }

    public init(centroids: [Float], weights: [Float], info: ThreadProfileInfo) {
        self.centroids = centroids
        self.weights = weights
        self.info = info
    }
}

extension ThreadProfile {
    /// Runs the commons over the Thread's corpus, weighs every index entry by the Thread's lift there
    /// and keeps the weighted centroids. Nil when the Thread predicts no position better than the commons.
    public static func compute(
        commons: RaoTransformer, index: ProvenanceIndex, corpus: TokenizedCorpus, seqLen: Int, packSHA256: String, epoch: Int,
        k: Int = ProfileSettings.k, iterations: Int = ProfileSettings.iterations, seed: UInt64 = ProfileSettings.seed
    ) throws -> ThreadProfile? {
        let started = Date()
        let pass = EvalPass.run(model: commons, corpus: corpus, seqLen: seqLen, batchSize: ProfileSettings.batchSize,
                                captureKeys: false, alpha: index.info.alpha, captureStates: true)
        var commonsLoss: [Float] = []
        var values: [Int32] = []
        commonsLoss.reserveCapacity(index.count)
        values.reserveCapacity(index.count)
        for i in 0..<pass.count where pass.indexable[i] {
            commonsLoss.append(pass.loss[i])
            values.append(pass.value[i])
        }
        guard values == index.values, let statesArray = pass.states, statesArray.dim(0) == index.count else {
            throw ProfileError.misaligned(expected: index.count, found: values.count)
        }
        let hidden = statesArray.dim(1)
        let lifts = lift(commonsLoss: commonsLoss, threadLoss: index.loss)
        let weighted = lifts.filter { $0 > 0 }.count
        guard weighted > 0 else { return nil }
        let states = statesArray.asArray(Float.self)
        let (centroids, mass) = kMeans(states: states, hidden: hidden, weights: lifts, k: min(k, weighted), iterations: iterations, seed: seed)
        let info = ThreadProfileInfo(
            k: mass.count, hidden: hidden, epoch: epoch, entries: index.count, weighted: weighted, liftTotal: lifts.reduce(0, +),
            commonsLoss: Stats.mean(commonsLoss), threadLoss: Stats.mean(index.loss), commonsPackSHA256: packSHA256,
            indexSHA256: index.sha256, seed: seed, iterations: iterations, seconds: Date().timeIntervalSince(started), createdAt: .wholeSecond())
        return ThreadProfile(centroids: centroids, weights: mass, info: info)
    }

    /// How much better the Thread predicts each entry than the commons, floored at 0.
    public static func lift(commonsLoss: [Float], threadLoss: [Float]) -> [Float] {
        zip(commonsLoss, threadLoss).map { max(0, $0 - $1) }
    }

    /// Weighted k-means: k-means++ starts drawn by weight × squared distance, then Lloyd rounds until
    /// no point moves. Points of weight 0 take no part. Returns the centroids and each one's weight mass.
    public static func kMeans(states: [Float], hidden: Int, weights: [Float], k: Int, iterations: Int, seed: UInt64)
        -> (centroids: [Float], mass: [Float])
    {
        let points = weights.indices.filter { weights[$0] > 0 }
        guard !points.isEmpty, k > 0 else { return ([], []) }
        let k = min(k, points.count)
        var rng = SplitMix64(seed: seed)
        return states.withUnsafeBufferPointer { x in
            func distance(_ p: Int, _ c: [Float], _ j: Int) -> Float {
                var sum: Float = 0
                let a = p * hidden
                let b = j * hidden
                for d in 0..<hidden {
                    let diff = x[a + d] - c[b + d]
                    sum += diff * diff
                }
                return sum
            }
            func draw(_ scores: [Float]) -> Int {
                let total = scores.reduce(0, +)
                guard total > 0 else { return points[rng.nextInt(below: points.count)] }
                var target = Float(rng.nextUnit()) * total
                for (n, score) in scores.enumerated() {
                    target -= score
                    if target <= 0 { return points[n] }
                }
                return points[points.count - 1]
            }
            var centroids = [Float](repeating: 0, count: k * hidden)
            func place(_ j: Int, at p: Int) {
                for d in 0..<hidden { centroids[j * hidden + d] = x[p * hidden + d] }
            }
            place(0, at: draw(points.map { weights[$0] }))
            var nearest = points.map { distance($0, centroids, 0) }
            for j in 1..<k {
                place(j, at: draw(points.indices.map { weights[points[$0]] * nearest[$0] }))
                for n in points.indices { nearest[n] = min(nearest[n], distance(points[n], centroids, j)) }
            }
            var assignment = [Int](repeating: -1, count: points.count)
            var mass = [Float](repeating: 0, count: k)
            for _ in 0..<max(1, iterations) {
                var moved = false
                for (n, p) in points.enumerated() {
                    var best = 0
                    var bestDistance = Float.infinity
                    for j in 0..<k {
                        let d = distance(p, centroids, j)
                        if d < bestDistance {
                            bestDistance = d
                            best = j
                        }
                    }
                    if assignment[n] != best {
                        assignment[n] = best
                        moved = true
                    }
                }
                var sums = [Float](repeating: 0, count: k * hidden)
                mass = [Float](repeating: 0, count: k)
                for (n, p) in points.enumerated() {
                    let j = assignment[n]
                    let w = weights[p]
                    mass[j] += w
                    for d in 0..<hidden { sums[j * hidden + d] += w * x[p * hidden + d] }
                }
                for j in 0..<k {
                    if mass[j] > 0 {
                        for d in 0..<hidden { centroids[j * hidden + d] = sums[j * hidden + d] / mass[j] }
                    } else {
                        // An empty centroid moves to the point that costs the most where it is.
                        var worst = points[0]
                        var worstCost: Float = -1
                        for (n, p) in points.enumerated() {
                            let cost = weights[p] * distance(p, centroids, assignment[n])
                            if cost > worstCost {
                                worstCost = cost
                                worst = p
                            }
                        }
                        place(j, at: worst)
                        moved = true
                    }
                }
                if !moved { break }
            }
            return (centroids, mass)
        }
    }

    /// A prompt as the router reads it: the commons' final-normed states, each token weighted by the
    /// commons' surprise at it given what came before (the first token weighs 0). Zeros when nothing
    /// can be weighed.
    public static func query(model: RaoTransformer, prompt: [Int]) -> [Float] {
        let hidden = model.config.hiddenSize
        guard prompt.count >= 2 else { return [Float](repeating: 0, count: hidden) }
        let input = MLXArray(prompt.map { Int32($0) }, [1, prompt.count])
        let output = model.forward(input, cache: nil, captureTap: false)
        let logits = output.logits.asType(.float32)[0, 0..<(prompt.count - 1)]
        let targets = MLXArray(prompt.dropFirst().map { Int32($0) }, [prompt.count - 1])
        let surprise = RaoLoss.tokenStats(logits: logits, targets: targets).loss
        let states = output.final.asType(.float32)[0, 1..<prompt.count]
        let total = surprise.sum()
        let weighted = (states * surprise.expandedDimensions(axis: -1)).sum(axis: 0)
        eval(weighted, total)
        let mass = total.item(Float.self)
        guard mass > 0 else { return [Float](repeating: 0, count: hidden) }
        return weighted.asArray(Float.self).map { $0 / mass }
    }

    /// The mean cosine between the centroids, pairwise: how spread the Thread's knowledge sits.
    public static func spread(centroids: [Float], hidden: Int) -> Float {
        let k = hidden > 0 ? centroids.count / hidden : 0
        guard k >= 2 else { return 1 }
        let rows = (0..<k).map { ProfileRouter.unit(Array(centroids[($0 * hidden)..<(($0 + 1) * hidden)])) }
        var sum: Float = 0
        var pairs = 0
        for a in 0..<k {
            for b in (a + 1)..<k {
                sum += zip(rows[a], rows[b]).reduce(0) { $0 + $1.0 * $1.1 }
                pairs += 1
            }
        }
        return sum / Float(pairs)
    }

    public func save(to directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try MLX.save(
            arrays: [
                "centroids": MLXArray(centroids, [k, hidden]),
                "weights": MLXArray(weights, [k]),
                "lift_total": MLXArray([info.liftTotal], [1]),
            ],
            metadata: ["format": "raolm-thread-profile", "epoch": String(info.epoch)],
            url: directory.appendingPathComponent(ThreadProfileFiles.arrays))
        try JSONCoding.write(info, to: directory.appendingPathComponent(ThreadProfileFiles.info))
    }

    /// Nil when the directory has no profile.
    public static func load(from directory: URL) throws -> ThreadProfile? {
        let url = directory.appendingPathComponent(ThreadProfileFiles.arrays)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let info = try JSONCoding.read(ThreadProfileInfo.self, from: directory.appendingPathComponent(ThreadProfileFiles.info))
        let arrays = try loadArrays(url: url)
        guard let centroids = arrays["centroids"], let weights = arrays["weights"] else { throw ProfileError.corrupt("missing arrays") }
        guard centroids.dim(0) == info.k, centroids.dim(1) == info.hidden, weights.dim(0) == info.k else {
            throw ProfileError.corrupt("centroids \(centroids.shape) against k \(info.k) × \(info.hidden)")
        }
        return ThreadProfile(centroids: centroids.asType(.float32).asArray(Float.self), weights: weights.asType(.float32).asArray(Float.self), info: info)
    }
}
