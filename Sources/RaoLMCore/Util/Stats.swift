//
//  Stats.swift
//  RaoLMCore
//
//  WHAT: The handful of summary statistics the ledger and the evaluation report use.
//

import Foundation

public enum Stats {

    public static func mean(_ values: [Float]) -> Float {
        guard !values.isEmpty else { return 0 }
        var total: Double = 0
        for value in values { total += Double(value) }
        return Float(total / Double(values.count))
    }

    public static func mean(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Double(values.count)
    }

    /// Linear-interpolated quantile of `values` at `q` in [0, 1]. Sorts a copy.
    public static func quantile(_ values: [Float], _ q: Double) -> Float {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        return quantileSorted(sorted, q)
    }

    public static func quantileSorted(_ sorted: [Float], _ q: Double) -> Float {
        guard !sorted.isEmpty else { return 0 }
        let position = min(max(q, 0), 1) * Double(sorted.count - 1)
        let lower = Int(position.rounded(.down))
        let upper = min(lower + 1, sorted.count - 1)
        let fraction = Float(position - Double(lower))
        return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction
    }

    /// Area under the ROC curve for scores against binary labels (Mann–Whitney U with
    /// average ranks for ties). Nil when either class is empty.
    public static func auroc(scores: [Double], labels: [Bool]) -> Double? {
        precondition(scores.count == labels.count)
        let positives = labels.filter { $0 }.count
        let negatives = labels.count - positives
        guard positives > 0, negatives > 0 else { return nil }
        let order = scores.indices.sorted { scores[$0] < scores[$1] }
        var ranks = [Double](repeating: 0, count: scores.count)
        var i = 0
        while i < order.count {
            var j = i
            while j + 1 < order.count, scores[order[j + 1]] == scores[order[i]] { j += 1 }
            let average = Double(i + j) / 2 + 1
            for k in i...j { ranks[order[k]] = average }
            i = j + 1
        }
        var positiveRankSum = 0.0
        for index in labels.indices where labels[index] { positiveRankSum += ranks[index] }
        let u = positiveRankSum - Double(positives) * Double(positives + 1) / 2
        return u / (Double(positives) * Double(negatives))
    }
}
