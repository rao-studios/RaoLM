//
//  TextWidgets.swift
//  RaoLMTerminal
//
//  WHAT: A single-line label and a word-wrapped paragraph of styled text.
//

import Foundation

public struct Label {
    public var text: Text
    public var align: Alignment

    public init(_ text: Text, align: Alignment = .leading) {
        self.text = text
        self.align = align
    }

    public func render(in rect: Rect, on canvas: inout Canvas) {
        guard !rect.isEmpty else { return }
        let clipped = text.truncated(to: rect.width)
        let x: Int
        switch align {
        case .leading: x = rect.minX
        case .center: x = rect.minX + (rect.width - clipped.width) / 2
        case .trailing: x = rect.maxX - clipped.width
        }
        canvas.put(clipped, x: x, y: rect.minY, clip: rect)
    }
}

public struct Paragraph {
    public var text: Text
    public var scroll: Int

    public init(_ text: Text, scroll: Int = 0) {
        self.text = text
        self.scroll = scroll
    }

    public func render(in rect: Rect, on canvas: inout Canvas) {
        guard !rect.isEmpty else { return }
        let lines = Self.wrap(text, width: rect.width)
        for (row, line) in lines.dropFirst(max(0, scroll)).prefix(rect.height).enumerated() {
            canvas.put(line, x: rect.minX, y: rect.minY + row, clip: rect)
        }
    }

    /// Plain-string wrap: breaks on spaces, hard-breaks words longer than `width`, honours `\n`.
    public static func wrap(_ string: String, width: Int) -> [String] {
        wrap(Text(string), width: width).map(\.plain)
    }

    /// Styled wrap, same rules; styles follow their characters onto the new lines.
    public static func wrap(_ text: Text, width: Int) -> [Text] {
        guard width > 0 else { return [] }
        typealias Glyph = (scalar: Unicode.Scalar, style: Style)
        enum Token { case word([Glyph]), space([Glyph]), newline }

        var tokens: [Token] = []
        var current: [Glyph] = []
        var currentIsSpace = false
        func close() {
            guard !current.isEmpty else { return }
            tokens.append(currentIsSpace ? .space(current) : .word(current))
            current.removeAll()
        }
        for span in text.spans {
            for scalar in span.text.unicodeScalars {
                if scalar == "\n" {
                    close()
                    tokens.append(.newline)
                    continue
                }
                let isSpace = scalar == " "
                if !current.isEmpty, isSpace != currentIsSpace { close() }
                currentIsSpace = isSpace
                current.append((scalar, span.style))
            }
        }
        close()

        var lines: [Text] = []
        var line: [Glyph] = []
        var lineWidth = 0
        func emit() {
            while let last = line.last, last.scalar == " " { line.removeLast() }
            var result = Text()
            var run = ""
            var runStyle: Style?
            for glyph in line {
                if glyph.style != runStyle {
                    if let runStyle { result.append(run, runStyle) }
                    run = ""
                    runStyle = glyph.style
                }
                run.unicodeScalars.append(glyph.scalar)
            }
            if let runStyle { result.append(run, runStyle) }
            lines.append(result)
            line.removeAll()
            lineWidth = 0
        }
        func measure(_ glyphs: [Glyph]) -> Int { glyphs.reduce(0) { $0 + TerminalWidth.of($1.scalar) } }

        for token in tokens {
            switch token {
            case .newline:
                emit()
            case .space(let glyphs):
                guard !line.isEmpty else { continue }
                for glyph in glyphs where lineWidth < width {
                    line.append(glyph)
                    lineWidth += 1
                }
            case .word(let glyphs):
                let w = measure(glyphs)
                if lineWidth > 0, lineWidth + w > width { emit() }
                if w <= width - lineWidth {
                    line.append(contentsOf: glyphs)
                    lineWidth += w
                } else {
                    for glyph in glyphs {
                        let gw = TerminalWidth.of(glyph.scalar)
                        if lineWidth + gw > width { emit() }
                        line.append(glyph)
                        lineWidth += gw
                    }
                }
            }
        }
        if !line.isEmpty || lines.isEmpty { emit() }
        return lines
    }
}
