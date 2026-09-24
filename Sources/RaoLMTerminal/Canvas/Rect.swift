//
//  Rect.swift
//  RaoLMTerminal
//
//  WHAT: Rectangles in cell coordinates and the layout arithmetic screens use to carve them.
//  PIN:  `Length.resolve` takes fixed and percent sizes first (each clamped to what is left),
//        then shares the remainder among flex items by weight, giving the rounding remainder
//        to the last flex item. Without flex items, leftover space stays unused.
//

import Foundation

public enum Length: Sendable, Equatable {
    case fixed(Int)
    case percent(Int)
    /// A share of the remaining space, weight ≥ 1.
    case flex(Int)

    public static func resolve(_ lengths: [Length], total: Int) -> [Int] {
        var sizes = [Int](repeating: 0, count: lengths.count)
        var remaining = max(0, total)
        for (index, length) in lengths.enumerated() {
            switch length {
            case .fixed(let n):
                sizes[index] = min(max(0, n), remaining)
                remaining -= sizes[index]
            case .percent(let p):
                sizes[index] = min(max(0, total * p / 100), remaining)
                remaining -= sizes[index]
            case .flex:
                continue
            }
        }
        let flexIndices = lengths.indices.filter { if case .flex = lengths[$0] { return true } else { return false } }
        let totalWeight = flexIndices.reduce(0) { sum, index in
            if case .flex(let w) = lengths[index] { return sum + max(1, w) }
            return sum
        }
        guard totalWeight > 0 else { return sizes }
        var given = 0
        for (position, index) in flexIndices.enumerated() {
            guard case .flex(let w) = lengths[index] else { continue }
            if position == flexIndices.count - 1 {
                sizes[index] = remaining - given
            } else {
                sizes[index] = remaining * max(1, w) / totalWeight
                given += sizes[index]
            }
        }
        return sizes
    }
}

public struct Rect: Sendable, Equatable, CustomStringConvertible {
    public var x: Int
    public var y: Int
    public var width: Int
    public var height: Int

    public init(x: Int, y: Int, width: Int, height: Int) {
        self.x = x
        self.y = y
        self.width = max(0, width)
        self.height = max(0, height)
    }

    public static let zero = Rect(x: 0, y: 0, width: 0, height: 0)

    public var minX: Int { x }
    public var minY: Int { y }
    /// Exclusive.
    public var maxX: Int { x + width }
    /// Exclusive.
    public var maxY: Int { y + height }
    public var size: Size { Size(width: width, height: height) }
    public var isEmpty: Bool { width <= 0 || height <= 0 }
    public var description: String { "(\(x),\(y) \(width)×\(height))" }

    public func inset(_ n: Int) -> Rect { inset(top: n, left: n, bottom: n, right: n) }

    public func inset(top: Int = 0, left: Int = 0, bottom: Int = 0, right: Int = 0) -> Rect {
        Rect(x: x + left, y: y + top, width: max(0, width - left - right), height: max(0, height - top - bottom))
    }

    public func intersection(_ other: Rect) -> Rect {
        let x0 = max(minX, other.minX)
        let y0 = max(minY, other.minY)
        let x1 = min(maxX, other.maxX)
        let y1 = min(maxY, other.maxY)
        guard x1 > x0, y1 > y0 else { return Rect(x: x0, y: y0, width: 0, height: 0) }
        return Rect(x: x0, y: y0, width: x1 - x0, height: y1 - y0)
    }

    public func contains(x px: Int, y py: Int) -> Bool {
        px >= minX && px < maxX && py >= minY && py < maxY
    }

    /// Columns, left to right.
    public func splitHorizontally(_ lengths: [Length], gap: Int = 0) -> [Rect] {
        let gaps = max(0, lengths.count - 1) * max(0, gap)
        let widths = Length.resolve(lengths, total: width - gaps)
        var cursor = x
        return widths.map { w in
            defer { cursor += w + gap }
            return Rect(x: cursor, y: y, width: w, height: height)
        }
    }

    /// Rows, top to bottom.
    public func splitVertically(_ lengths: [Length], gap: Int = 0) -> [Rect] {
        let gaps = max(0, lengths.count - 1) * max(0, gap)
        let heights = Length.resolve(lengths, total: height - gaps)
        var cursor = y
        return heights.map { h in
            defer { cursor += h + gap }
            return Rect(x: x, y: cursor, width: width, height: h)
        }
    }

    /// The first `n` rows and the rest.
    public func top(_ n: Int) -> (Rect, Rect) {
        let n = min(max(0, n), height)
        return (Rect(x: x, y: y, width: width, height: n), Rect(x: x, y: y + n, width: width, height: height - n))
    }

    /// The last `n` rows and the rest above them.
    public func bottom(_ n: Int) -> (Rect, Rect) {
        let n = min(max(0, n), height)
        return (Rect(x: x, y: maxY - n, width: width, height: n), Rect(x: x, y: y, width: width, height: height - n))
    }

    /// A centred rectangle of at most `width`×`height`.
    public func centered(width w: Int, height h: Int) -> Rect {
        let cw = min(w, width)
        let ch = min(h, height)
        return Rect(x: x + (width - cw) / 2, y: y + (height - ch) / 2, width: cw, height: ch)
    }

    public func row(_ index: Int) -> Rect { Rect(x: x, y: y + index, width: width, height: index < height ? 1 : 0) }
}
