//
//  Charts.swift
//  RaoLMTerminal
//
//  WHAT: A sparkline, a line chart (block columns or braille dots) and a progress bar.
//  PIN:  Series longer than the plot are bucket-averaged down to it; NaN and infinite values
//        are skipped. The braille chart fills the vertical gap between consecutive samples so
//        a steep curve stays continuous.
//

import Foundation

public struct Sparkline {
    public var values: [Double]
    public var range: ClosedRange<Double>?
    public var style: Style
    public var glyphs: Glyphs

    public init(_ values: [Double], range: ClosedRange<Double>? = nil, style: Style, glyphs: Glyphs = .unicode) {
        self.values = values
        self.range = range
        self.style = style
        self.glyphs = glyphs
    }

    public func string(width: Int) -> String {
        let window = Array(values.suffix(max(0, width)))
        let finite = window.filter(\.isFinite)
        guard let lo = range?.lowerBound ?? finite.min(), let hi = range?.upperBound ?? finite.max() else {
            return String(repeating: " ", count: window.count)
        }
        let levels = glyphs.spark.count
        return window.map { value -> String in
            guard value.isFinite else { return " " }
            let fraction = hi > lo ? (min(max(value, lo), hi) - lo) / (hi - lo) : 0.5
            return glyphs.spark[min(levels - 1, max(0, Int((fraction * Double(levels - 1)).rounded())))]
        }.joined()
    }

    public func render(in rect: Rect, on canvas: inout Canvas) {
        guard !rect.isEmpty else { return }
        canvas.put(string(width: rect.width), x: rect.minX, y: rect.minY, style: style, clip: rect)
    }
}

public struct LineChart {
    public enum Mode: Sendable { case blocks, braille }

    public var values: [Double]
    public var mode: Mode
    public var showLabels: Bool
    public var range: ClosedRange<Double>?
    public var style: Style
    public var labelStyle: Style
    public var glyphs: Glyphs
    public var format: @Sendable (Double) -> String

    public init(
        _ values: [Double], mode: Mode = .braille, showLabels: Bool = true, range: ClosedRange<Double>? = nil,
        style: Style, labelStyle: Style, glyphs: Glyphs = .unicode,
        format: @escaping @Sendable (Double) -> String = { String(format: "%.3f", $0) }
    ) {
        self.values = values
        self.mode = mode
        self.showLabels = showLabels
        self.range = range
        self.style = style
        self.labelStyle = labelStyle
        self.glyphs = glyphs
        self.format = format
    }

    public static let gutter = 7

    /// Averages `values` into `count` buckets (fewer values than buckets stay as they are).
    public static func buckets(_ values: [Double], count: Int) -> [Double] {
        let finite = values.filter(\.isFinite)
        guard count > 0, finite.count > count else { return finite }
        return (0..<count).map { bucket in
            let lo = bucket * finite.count / count
            let hi = max(lo + 1, (bucket + 1) * finite.count / count)
            let slice = finite[lo..<hi]
            return slice.reduce(0, +) / Double(slice.count)
        }
    }

    public func render(in rect: Rect, on canvas: inout Canvas) {
        guard !rect.isEmpty else { return }
        let finite = values.filter(\.isFinite)
        guard let lo = range?.lowerBound ?? finite.min(), let hi = range?.upperBound ?? finite.max() else { return }
        var plot = rect
        if showLabels, rect.width > Self.gutter + 2 {
            plot = rect.inset(left: Self.gutter)
            let top = format(hi)
            let bottom = format(lo)
            for y in rect.minY..<rect.maxY {
                let label = y == rect.minY ? top : (y == rect.maxY - 1 ? bottom : "")
                let padded = String(repeating: " ", count: max(0, Self.gutter - 1 - label.count)) + label
                canvas.put(String(padded.suffix(Self.gutter - 1)), x: rect.minX, y: y, style: labelStyle, clip: rect)
                canvas.put(label.isEmpty ? glyphs.vertical : glyphs.teeLeft, x: rect.minX + Self.gutter - 1, y: y, style: labelStyle, clip: rect)
            }
        }
        guard !plot.isEmpty else { return }
        switch mode {
        case .blocks: renderBlocks(in: plot, lo: lo, hi: hi, on: &canvas)
        case .braille: renderBraille(in: plot, lo: lo, hi: hi, on: &canvas)
        }
    }

    private func fraction(_ value: Double, lo: Double, hi: Double) -> Double {
        hi > lo ? (min(max(value, lo), hi) - lo) / (hi - lo) : 0.5
    }

    private func renderBlocks(in plot: Rect, lo: Double, hi: Double, on canvas: inout Canvas) {
        let series = Self.buckets(values, count: plot.width)
        let levels = glyphs.spark.count
        for (column, value) in series.enumerated() {
            let filled = Int((fraction(value, lo: lo, hi: hi) * Double(plot.height * levels)).rounded())
            for row in 0..<plot.height {
                let level = min(levels, filled - row * levels)
                guard level > 0 else { continue }
                canvas.put(glyphs.spark[level - 1], x: plot.minX + column, y: plot.maxY - 1 - row, style: style, clip: plot)
            }
        }
    }

    private func renderBraille(in plot: Rect, lo: Double, hi: Double, on canvas: inout Canvas) {
        guard glyphs.spark.first == "▁" else {
            renderBlocks(in: plot, lo: lo, hi: hi, on: &canvas)
            return
        }
        let dotsX = plot.width * 2
        let dotsY = plot.height * 4
        let series = Self.buckets(values, count: dotsX)
        guard !series.isEmpty else { return }
        var bits = [UInt8](repeating: 0, count: plot.width * plot.height)
        func set(_ x: Int, _ y: Int) {
            guard x >= 0, x < dotsX, y >= 0, y < dotsY else { return }
            let cell = (y / 4) * plot.width + x / 2
            let masks: [[UInt8]] = [[0x01, 0x02, 0x04, 0x40], [0x08, 0x10, 0x20, 0x80]]
            bits[cell] |= masks[x % 2][y % 4]
        }
        var previous: Int?
        for (x, value) in series.enumerated() {
            let y = Int(((1 - fraction(value, lo: lo, hi: hi)) * Double(dotsY - 1)).rounded())
            if let previous {
                for fill in min(previous, y)...max(previous, y) { set(x, fill) }
            } else {
                set(x, y)
            }
            previous = y
        }
        for row in 0..<plot.height {
            for column in 0..<plot.width where bits[row * plot.width + column] != 0 {
                let scalar = Unicode.Scalar(0x2800 + UInt32(bits[row * plot.width + column]))!
                canvas.put(String(Character(scalar)), x: plot.minX + column, y: plot.minY + row, style: style, clip: plot)
            }
        }
    }
}

public struct ProgressBar {
    public var fraction: Double
    public var label: Text?
    public var trailing: Text?
    public var fillStyle: Style
    public var trackStyle: Style
    public var full: String
    public var empty: String

    public init(
        fraction: Double, label: Text? = nil, trailing: Text? = nil, fillStyle: Style, trackStyle: Style,
        full: String = "━", empty: String = "─"
    ) {
        self.fraction = fraction
        self.label = label
        self.trailing = trailing
        self.fillStyle = fillStyle
        self.trackStyle = trackStyle
        self.full = full
        self.empty = empty
    }

    public static func percent(_ fraction: Double) -> String {
        String(format: "%3.0f%%", min(max(fraction, 0), 1) * 100)
    }

    public func render(in rect: Rect, on canvas: inout Canvas) {
        guard !rect.isEmpty else { return }
        var x = rect.minX
        if let label {
            x += canvas.put(label, x: x, y: rect.minY, clip: rect) + 1
        }
        let tail = trailing ?? Text(Self.percent(fraction), style: trackStyle)
        let barWidth = max(0, rect.maxX - x - tail.width - 1)
        let clamped = fraction.isFinite ? min(max(fraction, 0), 1) : 0
        let filled = Int((clamped * Double(barWidth)).rounded())
        canvas.put(String(repeating: full, count: filled), x: x, y: rect.minY, style: fillStyle, clip: rect)
        canvas.put(String(repeating: empty, count: barWidth - filled), x: x + filled, y: rect.minY, style: trackStyle, clip: rect)
        canvas.put(tail, x: rect.maxX - tail.width, y: rect.minY, clip: rect)
    }
}
