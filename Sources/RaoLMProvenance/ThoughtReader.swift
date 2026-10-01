//
//  ThoughtReader.swift
//  RaoLMProvenance
//
//  WHAT: Whether two Threads are thinking the same thing at a position, read with nothing
//        trained. Every node publishes its cut state on the umbrella pack's anchor snippets. A
//        cut state is then described by its cosines to that node's own anchor states (a relative
//        representation, Moschella et al., ICLR 2023), a description two separately trained nodes
//        can compare. Their agreement is the correlation of the two descriptions, placed for each
//        pair of nodes between how two unrelated anchors correlate (0) and how the same anchor
//        does (1).
//  PIN:  Gate only. It lifts a Thread beside the leading Thread as retrieval's agreement does; it
//        never mixes one Thread's state into another's distribution, so every share stays exact.
//        Cosines are taken about the mean anchor state, so the stream's shared offset (its rogue
//        dimensions) does not make every state look alike. MLX for the matrices, once per strand.
//

import Foundation
import MLX
import RaoLMCore

public final class ThoughtReader {
    struct Frame {
        /// [anchors, hidden]: each anchor state about the mean, at unit length.
        let anchors: MLXArray
        /// [hidden]
        let mean: MLXArray
        /// [anchors, anchors]: each anchor's own description (its row centred, at unit length).
        let descriptions: MLXArray
    }

    struct Pair: Hashable {
        let a: Int
        let b: Int
    }

    private var frames: [Int: Frame] = [:]
    private var bands: [Pair: (low: Float, high: Float)] = [:]

    /// `descriptors[t].anchors` holds strand t's cut state on each anchor; a strand without them is not read.
    public init(descriptors: [StrandDescriptor]) {
        for (t, descriptor) in descriptors.enumerated() {
            guard let packed = descriptor.anchors else { continue }
            let values = packed.values
            let width = descriptor.hiddenSize
            guard width > 0, values.count >= 2 * width, values.count % width == 0 else { continue }
            let states = MLXArray(values, [values.count / width, width])
            let count = states.dim(0)
            let mean = states.mean(axis: 0)
            let anchors = Self.unit(states - mean)
            // An anchor's cosine to itself (1) would make every anchor's description of itself
            // stand out where no position's does: it becomes the row's mean of the others.
            let gram = matmul(anchors, anchors.T)
            let identity = eye(count)
            let others = (gram.sum(axis: -1, keepDims: true) - 1) / Float(max(1, count - 1))
            let descriptions = Self.centred(gram * (1 - identity) + identity * others)
            eval(anchors, mean, descriptions)
            frames[t] = Frame(anchors: anchors, mean: mean, descriptions: descriptions)
        }
    }

    public func reads(_ t: Int) -> Bool { frames[t] != nil }

    /// Strand t's description of each of `states` ([hidden] each): unit vectors over its anchors.
    public func describe(_ t: Int, states: [[Float]]) -> [[Float]]? {
        guard let frame = frames[t], let width = states.first?.count, width == frame.mean.dim(0) else { return nil }
        let array = MLXArray(states.flatMap { $0 }, [states.count, width])
        let described = Self.centred(matmul(Self.unit(array - frame.mean), frame.anchors.T))
        eval(described)
        let count = frame.anchors.dim(0)
        let flat = described.asArray(Float.self)
        return (0..<states.count).map { Array(flat[($0 * count)..<(($0 + 1) * count)]) }
    }

    /// How far strands t and u think alike, 0 to 1, from their descriptions of one position.
    public func agreement(_ t: Int, _ u: Int, _ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty, let band = band(t, u), band.high > band.low else { return 0 }
        var dot: Float = 0
        for i in a.indices { dot += a[i] * b[i] }
        return min(max((dot - band.low) / (band.high - band.low), 0), 1)
    }

    /// For a pair of strands: the 95th percentile of the correlation between two different anchors'
    /// descriptions, and the median of the same anchor's.
    func band(_ t: Int, _ u: Int) -> (low: Float, high: Float)? {
        let key = Pair(a: min(t, u), b: max(t, u))
        if let cached = bands[key] { return cached }
        guard let x = frames[t], let y = frames[u], x.descriptions.shape == y.descriptions.shape else { return nil }
        let count = x.descriptions.dim(0)
        let same = (x.descriptions * y.descriptions).sum(axis: -1)
        var crosses: [MLXArray] = []
        for shift in 1...min(8, max(1, count - 1)) {
            let rolled = concatenated([y.descriptions[shift...], y.descriptions[0..<shift]], axis: 0)
            crosses.append((x.descriptions * rolled).sum(axis: -1))
        }
        let cross = concatenated(crosses, axis: 0)
        eval(same, cross)
        let result = (low: Stats.quantile(cross.asArray(Float.self), 0.95), high: Stats.quantile(same.asArray(Float.self), 0.5))
        bands[key] = result
        return result
    }

    static func unit(_ x: MLXArray) -> MLXArray {
        x / MLX.maximum(MLX.sqrt((x * x).sum(axis: -1, keepDims: true)), MLXArray(Float(1e-6)))
    }

    /// Each row about its own mean, at unit length: dot products of two rows are then correlations.
    static func centred(_ x: MLXArray) -> MLXArray {
        unit(x - x.mean(axis: -1, keepDims: true))
    }
}
