//
//  Banner.swift
//  RaoLMTerminal
//
//  WHAT: The RaoLM wordmark in italic slant letters with a silver→white diagonal gradient,
//        the tagline, and the braid: three strands (blue, green, orange) that converge into
//        the off-white thread and end in a gold ring — the og-image motif in one row.
//  PIN:  Every wordmark row is exactly 35 columns (tests assert it). Didone contrast comes
//        from weight, not glyphs: stems (`/ | \`) take their gradient stop, hairlines (`_ , \``)
//        sit two stops darker. No glyph changes, so alignment cannot drift.
//

import Foundation

public enum Wordmark {
    public static let rows: [String] = [
        "    ____              __    __  ___",
        "   / __ \\____ _____  / /   /  |/  /",
        "  / /_/ / __ `/ __ \\/ /   / /|_/ / ",
        " / _, _/ /_/ / /_/ / /___/ /  / /  ",
        "/_/ |_|\\__,_/\\____/_____/_/  /_/   ",
    ]
    public static let width = 35
    public static let height = 5

    /// Gradient stop 0…6 for the glyph at (row, column).
    public static func stop(row: Int, column: Int, scalar: Unicode.Scalar) -> Int {
        let base = min(6, (row + column) * 7 / 39)
        switch scalar {
        case "_", ",", "`": return max(0, base - 2)
        default: return base
        }
    }

    public static func render(at x: Int, y: Int, palette: Palette, on canvas: inout Canvas, clip: Rect? = nil) {
        for (row, line) in rows.enumerated() {
            for (column, scalar) in line.unicodeScalars.enumerated() where scalar != " " {
                let color = palette.wordmarkStops[stop(row: row, column: column, scalar: scalar)]
                canvas.put(String(Character(scalar)), x: x + column, y: y + row,
                           style: Style(foreground: color, background: palette.base.background, attributes: .bold), clip: clip)
            }
        }
    }
}

public struct Braid: Sendable {
    /// Cells of waves, ripples and tinted flat line before the plain thread.
    public var wave: Int
    public var ripple: Int
    public var tinted: Int

    public init(wave: Int, ripple: Int, tinted: Int) {
        self.wave = wave
        self.ripple = ripple
        self.tinted = tinted
    }

    public static let full = Braid(wave: 15, ripple: 12, tinted: 8)
    public static let compact = Braid(wave: 9, ripple: 8, tinted: 4)

    public func render(in rect: Rect, palette: Palette, glyphs: Glyphs, on canvas: inout Canvas) {
        guard rect.width >= 2, rect.height >= 1 else { return }
        let background = palette.base.background
        for column in 0..<(rect.width - 1) {
            let glyph: String
            let color: Color
            if column < wave {
                glyph = glyphs.wave
                color = palette.braid[column % 3]
            } else if column < wave + ripple {
                glyph = glyphs.ripple
                color = palette.braid[column % 3]
            } else if column < wave + ripple + tinted {
                glyph = glyphs.horizontal
                color = palette.braid[column % 3]
            } else {
                glyph = glyphs.horizontal
                color = palette.line
            }
            canvas.put(glyph, x: rect.minX + column, y: rect.minY, style: Style(foreground: color, background: background), clip: rect)
        }
        canvas.put(glyphs.verified, x: rect.maxX - 1, y: rect.minY, style: palette.gold, clip: rect)
    }
}

public struct Banner {
    public var version: String
    public var tagline: String
    public var status: Text?
    public var palette: Palette
    public var glyphs: Glyphs

    public static let defaultTagline = "a language model with its citations baked in"

    public init(version: String, tagline: String = Banner.defaultTagline, status: Text? = nil, palette: Palette, glyphs: Glyphs) {
        self.version = version
        self.tagline = tagline
        self.status = status
        self.palette = palette
        self.glyphs = glyphs
    }

    /// The full banner needs a tall screen; below 30 rows screens use the compact header.
    public static func shouldUseCompact(_ size: Size) -> Bool { size.height < 30 || size.width < 60 }

    /// Rows the full banner takes at `width`: the info column sits beside the wordmark from
    /// 100 columns, below it otherwise.
    public static func height(width: Int) -> Int { width >= 100 ? 6 : 7 }

    public func render(in rect: Rect, on canvas: inout Canvas) {
        guard !rect.isEmpty else { return }
        let x = rect.minX + 1
        Wordmark.render(at: x, y: rect.minY, palette: palette, on: &canvas, clip: rect)
        let wide = rect.width >= 100
        if wide {
            let infoX = x + Wordmark.width + 5
            let info = Rect(x: infoX, y: rect.minY, width: rect.maxX - infoX - 1, height: 3)
            canvas.put("v" + version, x: infoX, y: rect.minY, style: palette.dim, clip: info)
            canvas.put(tagline, x: infoX, y: rect.minY + 1, style: palette.tagline, clip: info)
            if let status { canvas.put(status.truncated(to: info.width), x: infoX, y: rect.minY + 2, clip: info) }
            Braid.full.render(in: Rect(x: rect.minX, y: rect.minY + 5, width: rect.width, height: 1), palette: palette, glyphs: glyphs, on: &canvas)
        } else {
            var line = Text("v" + version, style: palette.dim)
            line.append("  ", palette.dim)
            line.append(tagline, palette.tagline)
            canvas.put(line.truncated(to: rect.width - 2), x: x, y: rect.minY + 5, clip: rect)
            Braid.full.render(in: Rect(x: rect.minX, y: rect.minY + 6, width: rect.width, height: 1), palette: palette, glyphs: glyphs, on: &canvas)
        }
    }
}

/// One row: `RaoLM ∿∿∿~~~───◉  Title · detail          status`.
public struct CompactHeader {
    public var title: Text
    public var status: Text
    public var palette: Palette
    public var glyphs: Glyphs

    public init(title: Text, status: Text = Text(), palette: Palette, glyphs: Glyphs) {
        self.title = title
        self.status = status
        self.palette = palette
        self.glyphs = glyphs
    }

    public static let markWidth = 40

    public func render(in rect: Rect, on canvas: inout Canvas) {
        guard !rect.isEmpty else { return }
        let stops = [0, 2, 3, 4, 6]
        for (index, letter) in "RaoLM".enumerated() {
            let style = Style(foreground: palette.wordmarkStops[stops[index]], background: palette.base.background,
                              attributes: [.bold, .italic])
            canvas.put(String(letter), x: rect.minX + index, y: rect.minY, style: style, clip: rect)
        }
        let braidWidth = min(Self.markWidth - 6, max(0, rect.width - 6))
        Braid.compact.render(in: Rect(x: rect.minX + 6, y: rect.minY, width: braidWidth, height: 1),
                             palette: palette, glyphs: glyphs, on: &canvas)
        let titleX = rect.minX + Self.markWidth + 2
        let statusWidth = status.width
        let titleRoom = max(0, rect.maxX - titleX - statusWidth - 2)
        canvas.put(title.truncated(to: titleRoom), x: titleX, y: rect.minY, clip: rect)
        if statusWidth > 0, rect.maxX - statusWidth > titleX {
            canvas.put(status, x: rect.maxX - statusWidth - 1, y: rect.minY, clip: rect)
        }
    }
}
