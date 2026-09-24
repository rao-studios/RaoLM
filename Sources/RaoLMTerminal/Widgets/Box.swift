//
//  Box.swift
//  RaoLMTerminal
//
//  WHAT: A bordered panel: `┌─ Title ───┐ … └──── footer ─┘`. Returns the rectangle inside
//        the border for the panel's content.
//

import Foundation

public struct Box {
    public var title: Text?
    public var footer: Text?
    public var style: Style
    public var glyphs: Glyphs

    public init(title: Text? = nil, footer: Text? = nil, style: Style, glyphs: Glyphs = .unicode) {
        self.title = title
        self.footer = footer
        self.style = style
        self.glyphs = glyphs
    }

    /// A panel titled in `palette.title` when focused, `palette.dim` otherwise.
    public static func panel(_ title: String, focused: Bool = false, palette: Palette, glyphs: Glyphs) -> Box {
        Box(title: Text(title, style: focused ? palette.title : palette.text),
            style: focused ? palette.focusBorder : palette.border, glyphs: glyphs)
    }

    public static func content(of rect: Rect) -> Rect { rect.inset(1) }

    @discardableResult
    public func render(in rect: Rect, on canvas: inout Canvas) -> Rect {
        guard rect.width >= 2, rect.height >= 2 else { return .zero }
        let h = glyphs.horizontal
        let top = rect.minY
        let bottom = rect.maxY - 1
        let horizontalRun = String(repeating: h, count: rect.width - 2)
        canvas.put(glyphs.topLeft + horizontalRun + glyphs.topRight, x: rect.minX, y: top, style: style, clip: rect)
        canvas.put(glyphs.bottomLeft + horizontalRun + glyphs.bottomRight, x: rect.minX, y: bottom, style: style, clip: rect)
        for y in (top + 1)..<bottom {
            canvas.put(glyphs.vertical, x: rect.minX, y: y, style: style, clip: rect)
            canvas.put(glyphs.vertical, x: rect.maxX - 1, y: y, style: style, clip: rect)
        }
        let inner = rect.inset(top: 0, left: 2, bottom: 0, right: 2)
        if let title, !title.isEmpty, rect.width > 6 {
            let label = (Text(" ", style: style) + title + Text(" ", style: style)).truncated(to: max(0, rect.width - 4))
            canvas.put(label, x: rect.minX + 2, y: top, clip: inner)
        }
        if let footer, !footer.isEmpty, rect.width > 6 {
            let label = (Text(" ", style: style) + footer + Text(" ", style: style)).truncated(to: max(0, rect.width - 4))
            canvas.put(label, x: rect.maxX - 2 - label.width, y: bottom, clip: inner)
        }
        return Self.content(of: rect)
    }
}
