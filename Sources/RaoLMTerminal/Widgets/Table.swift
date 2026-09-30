//
//  Table.swift
//  RaoLMTerminal
//
//  WHAT: A table with a header, an under-rule, sized columns, a selected row and vertical
//        scrolling. The state is a value the screen keeps; the moves are pure.
//  PIN:  Two spaces between columns, as `Format.table` prints them, so the plain CLI and the UI
//        read alike. Text cells truncate with `…`; numbers should come in right-aligned columns.
//

import Foundation

public struct Column: Sendable {
    public var title: String
    public var width: Length
    public var align: Alignment

    public init(_ title: String, _ width: Length = .flex(1), align: Alignment = .leading) {
        self.title = title
        self.width = width
        self.align = align
    }
}

public struct TableState: Sendable, Equatable {
    public var selected: Int?
    public var scroll: Int

    public init(selected: Int? = 0, scroll: Int = 0) {
        self.selected = selected
        self.scroll = scroll
    }

    public mutating func move(by delta: Int, rowCount: Int, visible: Int) {
        guard rowCount > 0 else {
            selected = nil
            scroll = 0
            return
        }
        selected = min(max(0, (selected ?? -1) + delta), rowCount - 1)
        if selected == nil { selected = 0 }
        follow(rowCount: rowCount, visible: visible)
    }

    public mutating func page(by pages: Int, rowCount: Int, visible: Int) {
        move(by: pages * max(1, visible - 1), rowCount: rowCount, visible: visible)
    }

    public mutating func home(rowCount: Int, visible: Int) {
        selected = rowCount > 0 ? 0 : nil
        follow(rowCount: rowCount, visible: visible)
    }

    public mutating func end(rowCount: Int, visible: Int) {
        selected = rowCount > 0 ? rowCount - 1 : nil
        follow(rowCount: rowCount, visible: visible)
    }

    /// Keeps the selection in range and visible.
    public mutating func clamp(rowCount: Int, visible: Int) {
        if let s = selected { selected = rowCount > 0 ? min(max(0, s), rowCount - 1) : nil }
        follow(rowCount: rowCount, visible: visible)
    }

    mutating func follow(rowCount: Int, visible: Int) {
        let visible = max(1, visible)
        if let selected {
            if selected < scroll { scroll = selected }
            if selected >= scroll + visible { scroll = selected - visible + 1 }
        }
        scroll = min(max(0, scroll), max(0, rowCount - visible))
    }
}

public struct Table {
    public var columns: [Column]
    public var rows: [[Text]]
    public var state: TableState
    public var showHeader: Bool
    public var headerStyle: Style
    public var ruleStyle: Style
    public var rowStyle: Style
    public var selectionStyle: Style?
    public var marker: Text?
    public var glyphs: Glyphs

    public init(
        columns: [Column], rows: [[Text]], state: TableState = TableState(selected: nil), showHeader: Bool = true,
        headerStyle: Style, ruleStyle: Style, rowStyle: Style, selectionStyle: Style? = nil, marker: Text? = nil,
        glyphs: Glyphs = .unicode
    ) {
        self.columns = columns
        self.rows = rows
        self.state = state
        self.showHeader = showHeader
        self.headerStyle = headerStyle
        self.ruleStyle = ruleStyle
        self.rowStyle = rowStyle
        self.selectionStyle = selectionStyle
        self.marker = marker
        self.glyphs = glyphs
    }

    /// A table styled from the palette, with the gold `▶` marker when rows are selectable.
    public static func styled(
        columns: [Column], rows: [[Text]], state: TableState? = nil, palette: Palette, glyphs: Glyphs, showHeader: Bool = true
    ) -> Table {
        Table(
            columns: columns, rows: rows, state: state ?? TableState(selected: nil), showHeader: showHeader,
            headerStyle: palette.dim, ruleStyle: palette.border, rowStyle: palette.text,
            selectionStyle: state == nil ? nil : palette.selection,
            marker: state == nil ? nil : Text(glyphs.select, style: palette.gold), glyphs: glyphs)
    }

    public static func headerRows(in rect: Rect, showHeader: Bool) -> Int {
        guard showHeader else { return 0 }
        return rect.height >= 4 ? 2 : 1
    }

    public static func visibleRows(in rect: Rect, showHeader: Bool) -> Int {
        max(0, rect.height - headerRows(in: rect, showHeader: showHeader))
    }

    public func columnWidths(for width: Int) -> [Int] {
        let markerWidth = marker == nil ? 0 : 2
        let gutters = max(0, columns.count - 1) * 2
        return Length.resolve(columns.map(\.width), total: max(0, width - markerWidth - gutters))
    }

    public func render(in rect: Rect, on canvas: inout Canvas) {
        guard !rect.isEmpty, !columns.isEmpty else { return }
        // Leave the last column free for the scroll marker when the rows do not all fit.
        let scrolls = rows.count > Self.visibleRows(in: rect, showHeader: showHeader)
        let widths = columnWidths(for: rect.width - (scrolls ? 1 : 0))
        let markerWidth = marker == nil ? 0 : 2
        var y = rect.minY
        if showHeader {
            var x = rect.minX + markerWidth
            for (index, column) in columns.enumerated() {
                let cell = Text(column.title, style: headerStyle).padded(to: widths[index], align: column.align, style: rowStyle)
                canvas.put(cell, x: x, y: y, clip: rect)
                x += widths[index] + 2
            }
            y += 1
            if rect.height >= 4 {
                var x = rect.minX + markerWidth
                for width in widths {
                    canvas.put(String(repeating: glyphs.horizontal, count: width), x: x, y: y, style: ruleStyle, clip: rect)
                    x += width + 2
                }
                y += 1
            }
        }
        let visible = rect.maxY - y
        guard visible > 0 else { return }
        for (offset, row) in rows.dropFirst(state.scroll).prefix(visible).enumerated() {
            let index = state.scroll + offset
            let rowY = y + offset
            let selected = state.selected == index && selectionStyle != nil
            if selected, let marker { canvas.put(marker, x: rect.minX, y: rowY, clip: rect) }
            var x = rect.minX + markerWidth
            for (column, width) in widths.enumerated() {
                let content = column < row.count ? row[column] : Text()
                let styled = Text(spans: content.spans.map { span in
                    Span(span.text, span.style == .plain ? rowStyle : span.style)
                })
                canvas.put(styled.padded(to: width, align: columns[column].align, style: rowStyle), x: x, y: rowY, clip: rect)
                x += width + 2
            }
            if selected, let selectionStyle {
                let band = Rect(x: rect.minX + markerWidth, y: rowY, width: rect.width - markerWidth, height: 1)
                canvas.restyle(band) { style in
                    var s = style
                    if selectionStyle.background != .default { s.background = selectionStyle.background }
                    s.attributes.formUnion(selectionStyle.attributes)
                    return s
                }
            }
        }
        if rows.count > visible {
            // A one-cell scroll hint on the right edge: where the window sits in the rows.
            let position = rows.count <= 1 ? 0 : state.scroll * (visible - 1) / max(1, rows.count - visible)
            canvas.put(glyphs.vertical, x: rect.maxX - 1, y: y + min(visible - 1, max(0, position)), style: ruleStyle.bold(), clip: rect)
        }
    }
}
