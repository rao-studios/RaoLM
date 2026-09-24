import Foundation
import Testing

@testable import RaoLMTerminal

@Suite("Rect and Length")
struct RectTests {
    @Test("resolve shares flex space by weight after fixed sizes")
    func resolve() {
        #expect(Length.resolve([.fixed(10), .flex(1), .flex(2)], total: 100) == [10, 30, 60])
        #expect(Length.resolve([.flex(1), .flex(1), .flex(1)], total: 10) == [3, 3, 4])
        #expect(Length.resolve([.fixed(30), .fixed(30)], total: 50) == [30, 20])
        #expect(Length.resolve([.percent(50), .flex(1)], total: 81) == [40, 41])
    }

    @Test("inset never goes negative and intersection of disjoint rects is empty")
    func insetAndIntersection() {
        let r = Rect(x: 0, y: 0, width: 3, height: 3)
        #expect(r.inset(2).width == 0)
        #expect(r.inset(2).height == 0)
        #expect(Rect(x: 0, y: 0, width: 2, height: 2).intersection(Rect(x: 5, y: 5, width: 2, height: 2)).isEmpty)
    }

    @Test("splits tile the parent exactly")
    func splitsTile() {
        let parent = Rect(x: 2, y: 3, width: 50, height: 17)
        let rows = parent.splitVertically([.fixed(3), .flex(1), .fixed(1)])
        #expect(rows.map(\.height).reduce(0, +) == 17)
        #expect(rows.first?.minY == 3)
        #expect(rows.last?.maxY == parent.maxY)
        let columns = parent.splitHorizontally([.flex(1), .flex(2)], gap: 1)
        #expect(columns[0].maxX + 1 == columns[1].minX)
        #expect(columns[1].maxX == parent.maxX)
    }
}

@Suite("Colour")
struct ColorTests {
    @Test("xterm-256 nearest colour vectors")
    func xterm256() {
        #expect(Color.xterm256(255, 0, 0) == 196)
        #expect(Color.xterm256(0, 0, 0) == 16)
        #expect(Color.xterm256(255, 255, 255) == 231)
        #expect(Color.xterm256(128, 128, 128) == 244)
        #expect(Color.xterm256(0x4F, 0x8B, 0xE0) == 68)
        // The brand palette's pinned indices agree with nearest-colour for its hexes.
        #expect(Color.xterm256(0x4C, 0x8D, 0xF5) == 69)
        #expect(Color.xterm256(0x2E, 0xB6, 0x7D) == 36)
        #expect(Color.xterm256(0xF0, 0x88, 0x3E) == 209)
        #expect(Color.xterm256(0xD4, 0xAF, 0x37) == 179)
    }

    @Test("16-colour nearest vectors")
    func ansi16() {
        #expect(Color.ansi16(255, 0, 0) == 9)
        #expect(Color.ansi16(128, 128, 128) == 8)
        #expect(Color.ansi16(0, 0, 0) == 0)
    }

    @Test("degradation never upgrades and hex parses")
    func degrade() {
        #expect(Color(hex: 0x4F8BE0) == .rgb(0x4F, 0x8B, 0xE0))
        #expect(Color.indexed(68).degraded(to: .trueColor) == .indexed(68))
        #expect(Color.rgb(255, 0, 0).degraded(to: .ansi256) == .indexed(196))
        #expect(Color.rgb(255, 0, 0).degraded(to: .ansi16) == .ansi(9))
        #expect(Color.rgb(255, 0, 0).degraded(to: .none) == .default)
    }

    @Test("gradient has the requested stops, from first to last, monotone")
    func gradientStops() {
        let silver = Color(hex: 0xA9A9A9)
        let white = Color(hex: 0xF4F4F4)
        let stops = gradient(from: silver, to: white, steps: 35)
        #expect(stops.count == 35)
        #expect(stops.first == silver)
        #expect(stops.last == white)
        let reds = stops.compactMap { $0.rgbComponents?.r }
        #expect(reds == reds.sorted())
        #expect(gradient(from: silver, to: white, steps: 1) == [silver])
    }

    @Test("SGR encoder emits full styles on change only, and honours capabilities")
    func sgr() {
        var encoder = SGREncoder(capabilities: Capabilities(isTTY: true, colorDepth: .ansi256))
        let blue = Style(foreground: .rgb(0x4F, 0x8B, 0xE0)).bold()
        #expect(encoder.transition(to: blue) == "\u{1B}[0;1;38;5;68m")
        #expect(encoder.transition(to: blue) == "")
        #expect(encoder.reset() == "\u{1B}[0m")
        let noItalic = Capabilities(isTTY: true, colorDepth: .trueColor, italics: false)
        #expect(SGREncoder.sequence(for: Style().italic(), capabilities: noItalic) == "\u{1B}[0m")
        let mono = Capabilities(isTTY: true, colorDepth: .none)
        #expect(SGREncoder.sequence(for: Style(foreground: .rgb(1, 2, 3)).underline(), capabilities: mono) == "\u{1B}[0;4m")
        #expect(SGREncoder.sequence(for: Style(foreground: .ansi(9), background: .ansi(0)), capabilities: mono) == "\u{1B}[0m")
        let sixteen = Capabilities(isTTY: true, colorDepth: .ansi16)
        #expect(SGREncoder.sequence(for: Style(foreground: .ansi(9), background: .ansi(4)), capabilities: sixteen) == "\u{1B}[0;91;44m")
    }

    @Test("capabilities: overrides, NO_COLOR, terminal programs and the empty environment")
    func capabilities() {
        #expect(Capabilities.detect(environment: ["RAOLM_COLOR": "16", "COLORTERM": "truecolor"], isTTY: true).colorDepth == .ansi16)
        #expect(Capabilities.detect(environment: ["NO_COLOR": "1", "COLORTERM": "truecolor"], isTTY: true).colorDepth == .none)
        #expect(Capabilities.detect(environment: ["COLORTERM": "truecolor"], isTTY: true).colorDepth == .trueColor)
        #expect(Capabilities.detect(environment: ["TERM_PROGRAM": "Apple_Terminal"], isTTY: true).colorDepth == .ansi256)
        #expect(Capabilities.detect(environment: ["TERM_PROGRAM": "ghostty"], isTTY: true).colorDepth == .trueColor)
        #expect(Capabilities.detect(environment: [:], isTTY: true).colorDepth == .ansi256)
        #expect(Capabilities.detect(environment: [:], isTTY: false).colorDepth == .none)
        #expect(Capabilities.detect(environment: ["RAOLM_ASCII": "1"], isTTY: true).unicode == false)
        #expect(Capabilities.detect(environment: ["RAOLM_NO_ITALIC": "1"], isTTY: true).italics == false)
    }
}

@Suite("Canvas and renderer")
struct CanvasTests {
    @Test("put clips at the rect edge and reports columns consumed")
    func clipping() {
        var canvas = Canvas(size: Size(width: 10, height: 2))
        let used = canvas.put("hello world", x: 2, y: 0, style: .plain, clip: Rect(x: 0, y: 0, width: 7, height: 1))
        #expect(used == 5)
        #expect(canvas.line(0) == "  hello   ")
        #expect(canvas.put("x", x: 0, y: 5, style: .plain) == 0)
        canvas[99, 99] = Cell(scalar: "!")
        #expect(canvas.line(1) == "          ")
    }

    @Test("newlines show as ⏎, the UI's glyphs are one cell, CJK is two, marks are dropped")
    func widths() {
        for glyph in ["─", "│", "┌", "┐", "└", "┘", "├", "┤", "▸", "▶", "✓", "✗", "▁", "█", "◉", "∿", "≈", "●", "○", "◐", "━", "░", "…", "⏎", "«", "»"] {
            #expect(TerminalWidth.of(glyph) == 1, "\(glyph)")
        }
        var canvas = Canvas(size: Size(width: 6, height: 1))
        canvas.put("a\nb", x: 0, y: 0, style: .plain)
        #expect(canvas.line(0) == "a⏎b   ")
        var wide = Canvas(size: Size(width: 6, height: 1))
        #expect(wide.put("日x", x: 0, y: 0, style: .plain) == 3)
        #expect(wide[1, 0].isContinuation)
        #expect(wide.line(0) == "日x   ")
        var marks = Canvas(size: Size(width: 4, height: 1))
        marks.put("e\u{301}x", x: 0, y: 0, style: .plain)
        #expect(marks.line(0) == "ex  ")
    }

    @Test("renderer redraws only the changed run, and nothing for an identical frame")
    func diff() {
        let caps = Capabilities(isTTY: true, colorDepth: .ansi256)
        var renderer = Renderer(capabilities: caps)
        var a = Canvas(size: Size(width: 12, height: 3))
        a.put("first", x: 0, y: 0, style: .plain)
        a.put("second", x: 0, y: 1, style: .plain)
        let initial = String(decoding: renderer.present(a, cursor: nil), as: UTF8.self)
        #expect(initial.hasPrefix("\u{1B}[?2026h"))
        #expect(initial.hasSuffix("\u{1B}[?2026l"))
        #expect(initial.contains("\u{1B}[2J"))

        var b = a
        b.put("SEC", x: 0, y: 1, style: .plain)
        let delta = String(decoding: renderer.present(b, cursor: nil), as: UTF8.self)
        #expect(!delta.contains("\u{1B}[2J"))
        #expect(delta.contains("\u{1B}[2;1H"))
        #expect(!delta.contains("\u{1B}[1;1H"))
        #expect(delta.contains("SEC"))
        #expect(!delta.contains("first"))

        let idle = String(decoding: renderer.present(b, cursor: nil), as: UTF8.self)
        #expect(!idle.contains(";1H"))
        #expect(!idle.contains("SEC"))

        renderer.invalidate()
        let full = String(decoding: renderer.present(b, cursor: (x: 3, y: 2)), as: UTF8.self)
        #expect(full.contains("\u{1B}[2J"))
        #expect(full.contains("\u{1B}[3;4H\u{1B}[?25h"))
    }
}

@Suite("Key decoder")
struct KeyDecoderTests {
    func decode(_ bytes: [UInt8]) -> [KeyEvent] {
        var decoder = KeyDecoder()
        return decoder.feed(bytes)
    }

    @Test("CSI, SS3, modifiers and tilde keys")
    func sequences() {
        #expect(decode(Array("\u{1B}[A".utf8)) == [KeyEvent(.up)])
        #expect(decode(Array("\u{1B}OA".utf8)) == [KeyEvent(.up)])
        #expect(decode(Array("\u{1B}[1;5C".utf8)) == [KeyEvent(.right, .ctrl)])
        #expect(decode(Array("\u{1B}[1;2D".utf8)) == [KeyEvent(.left, .shift)])
        #expect(decode(Array("\u{1B}[3~".utf8)) == [KeyEvent(.delete)])
        #expect(decode(Array("\u{1B}[5~\u{1B}[6~".utf8)) == [KeyEvent(.pageUp), KeyEvent(.pageDown)])
        #expect(decode(Array("\u{1B}[Z".utf8)) == [KeyEvent(.backTab)])
        #expect(decode(Array("\u{1B}[15~".utf8)) == [KeyEvent(.function(5))])
    }

    @Test("control bytes, printable and UTF-8 characters, Alt")
    func bytes() {
        #expect(decode([0x03]) == [KeyEvent(.ctrl("c"))])
        #expect(decode([0x7F]) == [KeyEvent(.backspace)])
        #expect(decode([0x0D]) == [KeyEvent(.enter)])
        #expect(decode([0x09]) == [KeyEvent(.tab)])
        #expect(decode(Array("q".utf8)) == [KeyEvent(.char("q"))])
        #expect(decode(Array("é".utf8)) == [KeyEvent(.char("é"))])
        #expect(decode(Array("\u{1B}x".utf8)) == [KeyEvent(.char("x"), .alt)])
    }

    @Test("a lone Esc waits for flush; split UTF-8 reassembles")
    func pending() {
        var decoder = KeyDecoder()
        #expect(decoder.feed([0x1B]).isEmpty)
        #expect(decoder.hasPending)
        #expect(decoder.flush() == [KeyEvent(.escape)])
        #expect(!decoder.hasPending)

        let bytes = Array("λ".utf8)
        #expect(decoder.feed([bytes[0]]).isEmpty)
        #expect(decoder.feed([bytes[1]]) == [KeyEvent(.char("λ"))])

        #expect(decoder.feed(Array("\u{1B}[".utf8)).isEmpty)
        #expect(decoder.flush() == [KeyEvent(.escape), KeyEvent(.char("["))])
    }
}
