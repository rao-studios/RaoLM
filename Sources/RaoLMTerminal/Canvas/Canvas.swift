//
//  Canvas.swift
//  RaoLMTerminal
//
//  WHAT: A grid of styled cells that widgets draw into and the renderer diffs.
//  PIN:  One Unicode scalar per cell. Zero-width scalars (combining marks, format characters,
//        variation selectors) are dropped; East Asian wide and emoji scalars take two cells, the
//        second marked as a continuation. Every glyph the RaoLM UI uses is one cell wide under
//        these rules. Writes outside the clip rectangle are ignored, never wrapped.
//

import Foundation

public struct Cell: Sendable, Equatable {
    public var scalar: Unicode.Scalar
    public var style: Style
    public var isContinuation: Bool

    public init(scalar: Unicode.Scalar = " ", style: Style = .plain, isContinuation: Bool = false) {
        self.scalar = scalar
        self.style = style
        self.isContinuation = isContinuation
    }
}

public enum TerminalWidth {
    public static func of(_ scalar: Unicode.Scalar) -> Int {
        let value = scalar.value
        if value == 0 { return 0 }
        switch scalar.properties.generalCategory {
        case .nonspacingMark, .enclosingMark, .format: return 0
        default: break
        }
        if (0xFE00...0xFE0F).contains(value) { return 0 }
        if scalar.properties.isEmojiPresentation { return 2 }
        switch value {
        case 0x1100...0x115F, 0x2E80...0x303E, 0x3041...0x33FF, 0x3400...0x4DBF, 0x4E00...0x9FFF,
             0xA000...0xA4CF, 0xAC00...0xD7A3, 0xF900...0xFAFF, 0xFE30...0xFE4F, 0xFF00...0xFF60,
             0xFFE0...0xFFE6, 0x20000...0x3FFFD:
            return 2
        default:
            return 1
        }
    }

    public static func of(_ text: String) -> Int {
        text.unicodeScalars.reduce(0) { $0 + of($1) }
    }
}

public struct Canvas: Sendable, Equatable {
    public let size: Size
    private var cells: [Cell]

    public init(size: Size, fill: Style = .plain) {
        self.size = Size(width: max(0, size.width), height: max(0, size.height))
        cells = Array(repeating: Cell(style: fill), count: self.size.width * self.size.height)
    }

    public var bounds: Rect { Rect(x: 0, y: 0, width: size.width, height: size.height) }

    public subscript(x: Int, y: Int) -> Cell {
        get {
            precondition(bounds.contains(x: x, y: y), "cell (\(x), \(y)) outside \(size)")
            return cells[y * size.width + x]
        }
        set {
            guard bounds.contains(x: x, y: y) else { return }
            cells[y * size.width + x] = newValue
        }
    }

    public mutating func clear(fill: Style) {
        for index in cells.indices { cells[index] = Cell(style: fill) }
    }

    public mutating func fill(_ rect: Rect, scalar: Unicode.Scalar = " ", style: Style) {
        let area = rect.intersection(bounds)
        guard !area.isEmpty else { return }
        for y in area.minY..<area.maxY {
            for x in area.minX..<area.maxX { cells[y * size.width + x] = Cell(scalar: scalar, style: style) }
        }
    }

    /// Rewrites the style of every cell in `rect`.
    public mutating func restyle(_ rect: Rect, _ transform: (Style) -> Style) {
        let area = rect.intersection(bounds)
        guard !area.isEmpty else { return }
        for y in area.minY..<area.maxY {
            for x in area.minX..<area.maxX {
                let index = y * size.width + x
                cells[index].style = transform(cells[index].style)
            }
        }
    }

    /// Writes `text` from (x, y) rightwards; returns the columns consumed.
    @discardableResult
    public mutating func put(_ text: String, x: Int, y: Int, style: Style, clip: Rect? = nil) -> Int {
        let area = (clip ?? bounds).intersection(bounds)
        guard y >= area.minY, y < area.maxY else { return 0 }
        var column = x
        for scalar in text.unicodeScalars {
            let (glyph, width) = Self.normalise(scalar)
            if width == 0 { continue }
            if column + width > area.maxX { break }
            if column >= area.minX {
                cells[y * size.width + column] = Cell(scalar: glyph, style: style)
                if width == 2 {
                    cells[y * size.width + column + 1] = Cell(scalar: " ", style: style, isContinuation: true)
                }
            } else if width == 2, column + 1 >= area.minX {
                cells[y * size.width + column + 1] = Cell(scalar: " ", style: style)
            }
            column += width
        }
        return column - x
    }

    @discardableResult
    public mutating func put(_ text: Text, x: Int, y: Int, clip: Rect? = nil) -> Int {
        var column = x
        for span in text.spans { column += put(span.text, x: column, y: y, style: span.style, clip: clip) }
        return column - x
    }

    /// Plain scalars of one row, continuation cells omitted — what tests assert on.
    public func line(_ y: Int) -> String {
        guard y >= 0, y < size.height else { return "" }
        var result = ""
        for x in 0..<size.width {
            let cell = cells[y * size.width + x]
            if !cell.isContinuation { result.unicodeScalars.append(cell.scalar) }
        }
        return result
    }

    public var lines: [String] { (0..<size.height).map(line) }

    /// All rows joined, trailing spaces trimmed — a readable snapshot.
    public var snapshot: String {
        lines.map { line in
            var trimmed = line
            while trimmed.last == " " { trimmed.removeLast() }
            return trimmed
        }.joined(separator: "\n")
    }

    static func normalise(_ scalar: Unicode.Scalar) -> (Unicode.Scalar, Int) {
        switch scalar {
        case "\n": return ("⏎", 1)
        case "\t": return (" ", 1)
        default:
            if scalar.value < 0x20 || scalar.value == 0x7F { return ("?", 1) }
            return (scalar, TerminalWidth.of(scalar))
        }
    }
}
