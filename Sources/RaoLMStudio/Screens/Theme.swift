//
//  Theme.swift
//  RaoLMStudio
//
//  WHAT: The few drawing helpers every screen uses: a titled panel, key/value rows, and a
//        `TextTable` from RaoLMWorkflows shown as a table widget with numeric columns
//        right-aligned — the same columns the CLI prints.
//

import Foundation
import RaoLM
import RaoLMTerminal
import RaoLMWorkflows

enum Theme {
    /// Draws a panel and returns its content rect (inset one cell on each side inside the border).
    @discardableResult
    static func panel(_ title: String, focused: Bool = false, footer: Text? = nil, _ frame: inout Frame, _ rect: Rect) -> Rect {
        var box = Box.panel(title, focused: focused, palette: frame.palette, glyphs: frame.glyphs)
        box.footer = footer
        return box.render(in: rect, on: &frame.canvas).inset(top: 0, left: 1, bottom: 0, right: 1)
    }

    static func keyValues(_ rows: [(String, Text)], _ frame: inout Frame, _ rect: Rect, labelWidth: Int = 12) {
        for (index, (label, value)) in rows.prefix(rect.height).enumerated() {
            let y = rect.minY + index
            frame.canvas.put(label, x: rect.minX, y: y, style: frame.palette.dim, clip: rect)
            let valueRect = Rect(x: rect.minX + labelWidth, y: y, width: max(0, rect.width - labelWidth), height: 1)
            frame.canvas.put(value.truncated(to: valueRect.width), x: valueRect.minX, y: y, clip: valueRect)
        }
    }

    static func isNumeric(_ cell: String) -> Bool {
        guard !cell.isEmpty else { return true }
        return cell.allSatisfy { $0.isNumber || ".%—-+e/×, ".contains($0) } && cell.contains(where: \.isNumber) || cell == "—"
    }

    /// Columns sized to their content (the last one takes the rest), numbers right-aligned.
    static func columns(for table: TextTable, width: Int, flexColumn: Int? = nil) -> [Column] {
        let count = table.headers.count
        let flex = flexColumn ?? count - 1
        return table.headers.enumerated().map { index, header in
            let cells = table.rows.map { index < $0.count ? $0[index] : "" }
            let widest = max(TerminalWidth.of(header), cells.map(TerminalWidth.of).max() ?? 0)
            let numeric = !cells.isEmpty && cells.allSatisfy(isNumeric)
            return Column(header, index == flex ? .flex(1) : .fixed(min(widest, max(6, width / 2))),
                          align: numeric ? .trailing : .leading)
        }
    }

    static func textTable(
        _ table: TextTable, state: TableState? = nil, flexColumn: Int? = nil, cellStyle: ((Int, Int, String) -> Style?)? = nil,
        _ frame: inout Frame, _ rect: Rect
    ) {
        let rows: [[Text]] = table.rows.enumerated().map { rowIndex, row in
            row.enumerated().map { column, cell in Text(cell, style: cellStyle?(rowIndex, column, cell) ?? .plain) }
        }
        Table.styled(columns: columns(for: table, width: rect.width, flexColumn: flexColumn), rows: rows, state: state,
                     palette: frame.palette, glyphs: frame.glyphs).render(in: rect, on: &frame.canvas)
    }

    static func empty(_ message: String, _ frame: inout Frame, _ rect: Rect) {
        guard !rect.isEmpty else { return }
        for (row, line) in Paragraph.wrap(message, width: max(1, rect.width - 2)).prefix(rect.height).enumerated() {
            frame.canvas.put(line, x: rect.minX + 1, y: rect.minY + row, style: frame.palette.dim, clip: rect)
        }
    }

    static func statusStyle(_ status: RunStatus, palette: Palette) -> Style {
        switch status {
        case .complete: return palette.green
        case .failed: return palette.red
        case .stopped: return palette.dim
        default: return palette.orange
        }
    }

    static func statusGlyph(_ status: RunStatus, glyphs: Glyphs) -> String {
        switch status {
        case .complete: return glyphs.check
        case .failed: return glyphs.cross
        case .stopped: return glyphs.idle
        default: return glyphs.spinner[0]
        }
    }

    static func short(_ id: String?, _ n: Int = 12) -> String {
        guard let id, !id.isEmpty else { return "—" }
        return id.count > n ? String(id.prefix(n)) + "…" : id
    }
}
