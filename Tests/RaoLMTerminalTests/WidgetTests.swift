import Foundation
import Testing

@testable import RaoLMTerminal

private let caps = Capabilities(isTTY: true, colorDepth: .ansi256)
private let palette = Palette(for: caps, environment: [:])

@Suite("Widgets")
struct WidgetTests {
    @Test("box draws its border, title and footer and returns the content rect")
    func box() {
        var canvas = Canvas(size: Size(width: 14, height: 4))
        let inner = Box(title: "Runs", footer: "3", style: .plain).render(in: canvas.bounds, on: &canvas)
        #expect(inner == Rect(x: 1, y: 1, width: 12, height: 2))
        #expect(canvas.line(0) == "┌─ Runs ─────┐")
        #expect(canvas.line(1) == "│            │")
        #expect(canvas.line(3) == "└──────── 3 ─┘")
        var tiny = Canvas(size: Size(width: 1, height: 1))
        #expect(Box(style: .plain).render(in: tiny.bounds, on: &tiny) == .zero)
    }

    @Test("label alignment and truncation")
    func label() {
        var canvas = Canvas(size: Size(width: 8, height: 3))
        Label("abc", align: .trailing).render(in: canvas.bounds.row(0), on: &canvas)
        Label("abc", align: .center).render(in: canvas.bounds.row(1), on: &canvas)
        Label("abcdefghijk").render(in: canvas.bounds.row(2), on: &canvas)
        #expect(canvas.line(0) == "     abc")
        #expect(canvas.line(1) == "  abc   ")
        #expect(canvas.line(2) == "abcdefg…")
    }

    @Test("paragraph wrap breaks on spaces, hard-breaks long words and honours newlines")
    func wrap() {
        #expect(Paragraph.wrap("the quick brown fox", width: 9) == ["the quick", "brown fox"])
        #expect(Paragraph.wrap("abcdefghij", width: 4) == ["abcd", "efgh", "ij"])
        #expect(Paragraph.wrap("one\ntwo three", width: 20) == ["one", "two three"])
        #expect(Paragraph.wrap("", width: 5) == [""])
        let styled = Paragraph.wrap(Text("red ", style: palette.red) + Text("blue words", style: palette.blue), width: 8)
        #expect(styled.map(\.plain) == ["red blue", "words"])
        #expect(styled[1].spans.first?.style == palette.blue)
    }

    @Test("table state: selection clamps and scroll follows it")
    func tableState() {
        var state = TableState(selected: 0)
        state.move(by: -1, rowCount: 10, visible: 5)
        #expect(state.selected == 0)
        for _ in 0..<5 { state.move(by: 1, rowCount: 10, visible: 5) }
        #expect(state.selected == 5)
        #expect(state.scroll == 1)
        state.end(rowCount: 10, visible: 5)
        #expect(state.selected == 9)
        #expect(state.scroll == 5)
        state.page(by: -1, rowCount: 10, visible: 5)
        #expect(state.selected == 5)
        state.clamp(rowCount: 3, visible: 5)
        #expect(state.selected == 2)
        #expect(state.scroll == 0)
        state.move(by: 1, rowCount: 0, visible: 5)
        #expect(state.selected == nil)
    }

    @Test("table renders header, rule, aligned cells and a full-width selection")
    func table() {
        var canvas = Canvas(size: Size(width: 24, height: 5))
        let table = Table(
            columns: [Column("name", .flex(1)), Column("n", .fixed(4), align: .trailing)],
            rows: [["alpha", "1"], ["beta", "22"], ["gamma", "333"]],
            state: TableState(selected: 1), headerStyle: .plain, ruleStyle: .plain, rowStyle: .plain,
            selectionStyle: palette.selection, marker: "▶")
        table.render(in: canvas.bounds, on: &canvas)
        #expect(canvas.line(0) == "  name                 n")
        #expect(canvas.line(1) == "  " + String(repeating: "─", count: 16) + "  " + String(repeating: "─", count: 4))
        #expect(canvas.line(2) == "  " + "alpha".padding(toLength: 16, withPad: " ", startingAt: 0) + "     1")
        #expect(canvas.line(3).hasPrefix("▶ beta"))
        #expect(canvas.line(3).hasSuffix("22"))
        #expect(canvas[5, 3].style.background == palette.selection.background)
        #expect(canvas[5, 2].style.background != palette.selection.background)
        #expect(Table.visibleRows(in: canvas.bounds, showHeader: true) == 3)
    }

    @Test("sparkline, block chart and progress bar")
    func charts() {
        #expect(Sparkline([0, 0.5, 1], style: .plain).string(width: 3) == "▁▅█")
        #expect(Sparkline([1, 2, 3, 4], style: .plain).string(width: 2) == "▁█")
        var canvas = Canvas(size: Size(width: 2, height: 2))
        LineChart([0, 1], mode: .blocks, showLabels: false, style: .plain, labelStyle: .plain).render(in: canvas.bounds, on: &canvas)
        #expect(canvas.line(0) == " █")
        #expect(canvas.line(1) == " █")
        var braille = Canvas(size: Size(width: 2, height: 1))
        LineChart([0, 1, 0, 1], mode: .braille, showLabels: false, style: .plain, labelStyle: .plain).render(in: braille.bounds, on: &braille)
        #expect(braille.line(0).unicodeScalars.allSatisfy { (0x2800...0x28FF).contains($0.value) })
        var bar = Canvas(size: Size(width: 20, height: 1))
        ProgressBar(fraction: 0.5, fillStyle: .plain, trackStyle: .plain).render(in: bar.bounds, on: &bar)
        #expect(bar.line(0) == "━━━━━━━━───────  50%")
    }

    @Test("log pane follows the tail and ring buffer evicts the oldest")
    func log() {
        var ring = RingBuffer<Int>(capacity: 3)
        for i in 1...5 { ring.append(i) }
        #expect(ring.elements == [3, 4, 5])
        #expect(ring.suffix(2) == [4, 5])
        var state = LogState(capacity: 10)
        for i in 1...6 { state.append("line \(i)") }
        var canvas = Canvas(size: Size(width: 8, height: 2))
        LogPane(state: state, style: .plain).render(in: canvas.bounds, on: &canvas)
        #expect(canvas.lines == ["line 5  ", "line 6  "])
        state.scroll(by: 2, visible: 2)
        var scrolled = Canvas(size: Size(width: 8, height: 2))
        LogPane(state: state, style: .plain).render(in: scrolled.bounds, on: &scrolled)
        #expect(scrolled.lines == ["line 3  ", "line 4  "])
    }

    @Test("text field edits, moves and kills like readline")
    func textField() {
        var field = TextFieldState("hello")
        field.handle(KeyEvent(.char("!")))
        #expect(field.text == "hello!")
        field.handle(KeyEvent(.home))
        field.handle(KeyEvent(.char(">")))
        #expect(field.text == ">hello!")
        #expect(field.cursor == 1)
        field.handle(KeyEvent(.end))
        field.handle(KeyEvent(.backspace))
        #expect(field.text == ">hello")
        field.handle(KeyEvent(.ctrl("w")))
        #expect(field.text == "")
        let changed = field.handle(KeyEvent(.backspace))
        #expect(!changed)
        field.set("abc def")
        field.handle(KeyEvent(.ctrl("u")))
        #expect(field.text == "")
        var canvas = Canvas(size: Size(width: 5, height: 1))
        let cursor = TextField(state: TextFieldState("abcdefgh"), focused: true, style: .plain, placeholderStyle: .plain,
                               cursorStyle: Style(attributes: .reverse)).render(in: canvas.bounds, on: &canvas)
        #expect(cursor?.x == 4)
        #expect(canvas.line(0) == "efgh ")
    }

    @Test("key hints drop whole hints that do not fit; tabs mark the selection")
    func chrome() {
        var canvas = Canvas(size: Size(width: 22, height: 2))
        KeyHintBar([KeyHint("q", "quit"), KeyHint("?", "help"), KeyHint("1-8", "screens")], keyStyle: .plain, labelStyle: .plain)
            .render(in: canvas.bounds.row(0), on: &canvas)
        #expect(canvas.line(0) == " [q] quit  [?] help   ")
        Tabs(["Home", "Train"], selected: 1, style: .plain, selectedStyle: palette.blue, separator: " · ")
            .render(in: canvas.bounds.row(1), on: &canvas)
        #expect(canvas.line(1).hasPrefix(" Home · Train"))
        #expect(canvas[8, 1].style == palette.blue)
    }

    @Test("wordmark rows are 35 columns and the banner ends its braid in the gold ring")
    func banner() {
        #expect(Wordmark.rows.count == Wordmark.height)
        for row in Wordmark.rows { #expect(TerminalWidth.of(row) == Wordmark.width) }
        #expect(Wordmark.stop(row: 0, column: 0, scalar: "/") == 0)
        #expect(Wordmark.stop(row: 4, column: 34, scalar: "/") == 6)
        #expect(Wordmark.stop(row: 4, column: 34, scalar: "_") == 4)

        var wide = Canvas(size: Size(width: 120, height: 6))
        Banner(version: "0.1.0", status: "status line", palette: palette, glyphs: .unicode).render(in: wide.bounds, on: &wide)
        #expect(wide.line(0).contains("v0.1.0"))
        #expect(wide.line(1).contains(Banner.defaultTagline))
        #expect(wide.line(2).contains("status line"))
        #expect(wide.line(5).hasPrefix("∿∿∿"))
        #expect(wide.line(5).hasSuffix("◉"))
        #expect(wide[119, 5].style.foreground == palette.goldColor)
        #expect(Banner.height(width: 120) == 6)

        var narrow = Canvas(size: Size(width: 80, height: 7))
        Banner(version: "0.1.0", palette: palette, glyphs: .unicode).render(in: narrow.bounds, on: &narrow)
        #expect(narrow.line(5).contains("v0.1.0  a language model"))
        #expect(narrow.line(6).hasSuffix("◉"))

        var header = Canvas(size: Size(width: 100, height: 1))
        CompactHeader(title: "Train · run", status: "◐ training", palette: palette, glyphs: .unicode).render(in: header.bounds, on: &header)
        #expect(header.line(0).hasPrefix("RaoLM ∿∿∿∿∿∿∿∿∿~~~~~~~~────────────────◉"))
        #expect(header.line(0).contains("Train · run"))
        #expect(header.line(0).hasSuffix("◐ training "))
        #expect(header[1, 0].style.attributes.contains(.italic))
        #expect(Banner.shouldUseCompact(Size(width: 120, height: 24)))
        #expect(!Banner.shouldUseCompact(Size(width: 120, height: 40)))
    }

    @Test("palette heat ramp: uncited, blue, green, thread, gold for verified")
    func heat() {
        #expect(palette.heat(confidence: nil, verified: false) == palette.heat[0])
        #expect(palette.heat(confidence: 0.2, verified: false) == palette.blue)
        #expect(palette.heat(confidence: 0.6, verified: false) == palette.green)
        #expect(palette.heat(confidence: 0.9, verified: false) == palette.heat[3])
        #expect(palette.heat(confidence: 0.2, verified: true).attributes.contains(.underline))
    }
}
