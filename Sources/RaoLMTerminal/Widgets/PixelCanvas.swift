//
//  PixelCanvas.swift
//  RaoLMTerminal
//
//  WHAT: Pixel art in a terminal: a grid of coloured pixels drawn two per cell with half-block
//        glyphs (the upper pixel in the foreground, the lower in the background), and a share
//        bar whose segments meet inside a cell with eighth blocks.
//  PIN:  Without colour, or with ASCII glyphs, a pixel pair becomes a shade glyph by brightness,
//        so the art degrades to texture instead of vanishing. An empty pixel keeps the cell's
//        background, so the art sits on whatever ground the palette paints.
//

import Foundation

public struct PixelCanvas: Sendable, Equatable {
    /// Pixels across (one per column).
    public let width: Int
    /// Pixels down (two per row).
    public let height: Int
    public private(set) var pixels: [Color?]

    public init(width: Int, rows: Int) {
        self.width = max(0, width)
        self.height = max(0, rows) * 2
        pixels = Array(repeating: nil, count: self.width * self.height)
    }

    public var rows: Int { height / 2 }

    public subscript(x: Int, y: Int) -> Color? {
        get { x >= 0 && x < width && y >= 0 && y < height ? pixels[y * width + x] : nil }
        set {
            guard x >= 0, x < width, y >= 0, y < height else { return }
            pixels[y * width + x] = newValue
        }
    }

    public mutating func fill(x: Int, y: Int, width w: Int, height h: Int, _ color: Color?) {
        for py in max(0, y)..<min(height, y + h) {
            for px in max(0, x)..<min(width, x + w) { pixels[py * width + px] = color }
        }
    }

    /// 0…1 brightness of a colour (Rec. 709 luma); nil pixels are dark.
    public static func brightness(_ color: Color?) -> Double {
        guard let rgb = color?.rgbComponents else { return color == nil ? 0 : 0.8 }
        return (0.2126 * Double(rgb.r) + 0.7152 * Double(rgb.g) + 0.0722 * Double(rgb.b)) / 255
    }

    /// Draws the pixels with their top-left at (x, y), clipped to `clip`.
    public func render(x originX: Int, y originY: Int, clip: Rect, on canvas: inout Canvas, background: Color, glyphs: Glyphs, depth: ColorDepth) {
        let area = clip.intersection(canvas.bounds)
        let blocks = depth != .none ? (glyphs.upperHalf.flatMap { upper in glyphs.lowerHalf.map { (upper, $0) } }) : nil
        for row in 0..<rows {
            let y = originY + row
            guard y >= area.minY, y < area.maxY else { continue }
            for column in 0..<width {
                let x = originX + column
                guard x >= area.minX, x < area.maxX else { continue }
                let top = self[column, row * 2]
                let bottom = self[column, row * 2 + 1]
                if top == nil, bottom == nil { continue }
                guard let (upper, lower) = blocks else {
                    let level = max(Self.brightness(top), Self.brightness(bottom))
                    let index = min(glyphs.shades.count - 1, max(1, Int((level * Double(glyphs.shades.count - 1)).rounded())))
                    canvas.put(glyphs.shades[index], x: x, y: y, style: Style(foreground: top ?? bottom ?? .default, background: background), clip: area)
                    continue
                }
                switch (top, bottom) {
                case (let t?, let b?) where t == b:
                    canvas.put(glyphs.barFull, x: x, y: y, style: Style(foreground: t, background: background), clip: area)
                case (let t?, let b?):
                    canvas.put(upper, x: x, y: y, style: Style(foreground: t, background: b), clip: area)
                case (let t?, nil):
                    canvas.put(upper, x: x, y: y, style: Style(foreground: t, background: background), clip: area)
                case (nil, let b?):
                    canvas.put(lower, x: x, y: y, style: Style(foreground: b, background: background), clip: area)
                default:
                    break
                }
            }
        }
    }
}

/// A horizontal bar split into coloured shares that meet inside cells with eighth blocks.
public struct ShareBar: Sendable {
    public var segments: [(fraction: Double, color: Color)]
    public var track: Style
    public var glyphs: Glyphs

    public init(segments: [(fraction: Double, color: Color)], track: Style, glyphs: Glyphs) {
        self.segments = segments
        self.track = track
        self.glyphs = glyphs
    }

    /// One styled glyph per column.
    public func text(width: Int) -> Text {
        guard width > 0 else { return Text() }
        let total = width * 8
        var bounds: [(end: Int, color: Color)] = []
        var cumulative = 0.0
        for segment in segments where segment.fraction > 0 {
            cumulative += min(max(segment.fraction, 0), 1)
            bounds.append((Int((min(cumulative, 1) * Double(total)).rounded()), segment.color))
        }
        func color(at eighth: Int) -> Color? { bounds.first { eighth < $0.end }?.color }
        var text = Text()
        for cell in 0..<width {
            let start = cell * 8
            guard let left = color(at: start) else {
                text.append(glyphs.barEmpty, track)
                continue
            }
            // The first boundary inside this cell, if any.
            let split = (1..<8).first { color(at: start + $0) != left }
            guard let split else {
                text.append(glyphs.barFull, Style(foreground: left, background: track.background))
                continue
            }
            let right = color(at: start + split)
            text.append(glyphs.eighths[split - 1], Style(foreground: left, background: right ?? track.background))
        }
        return text
    }

    public func render(in rect: Rect, on canvas: inout Canvas) {
        guard !rect.isEmpty else { return }
        canvas.put(text(width: rect.width), x: rect.minX, y: rect.minY, clip: rect)
    }
}
