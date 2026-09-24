//
//  Chrome.swift
//  RaoLMTerminal
//
//  WHAT: The key-hint row (`[q] quit  [?] help`) and the screen tabs row.
//

import Foundation

public struct KeyHint: Sendable, Equatable {
    public var key: String
    public var label: String

    public init(_ key: String, _ label: String) {
        self.key = key
        self.label = label
    }
}

public struct KeyHintBar {
    public var hints: [KeyHint]
    public var keyStyle: Style
    public var labelStyle: Style

    public init(_ hints: [KeyHint], keyStyle: Style, labelStyle: Style) {
        self.hints = hints
        self.keyStyle = keyStyle
        self.labelStyle = labelStyle
    }

    public var text: Text {
        var text = Text(" ", style: labelStyle)
        for (index, hint) in hints.enumerated() {
            if index > 0 { text.append("  ", labelStyle) }
            text.append("[", labelStyle)
            text.append(hint.key, keyStyle)
            text.append("] ", labelStyle)
            text.append(hint.label, labelStyle)
        }
        return text
    }

    public func render(in rect: Rect, on canvas: inout Canvas) {
        guard !rect.isEmpty else { return }
        // Drop hints from the end rather than cutting one in half.
        var shown = hints
        while !shown.isEmpty, KeyHintBar(shown, keyStyle: keyStyle, labelStyle: labelStyle).text.width > rect.width {
            shown.removeLast()
        }
        canvas.put(KeyHintBar(shown, keyStyle: keyStyle, labelStyle: labelStyle).text, x: rect.minX, y: rect.minY, clip: rect)
    }
}

public struct Tabs {
    public var titles: [Text]
    public var selected: Int
    public var style: Style
    public var selectedStyle: Style
    public var separator: Text

    public init(_ titles: [Text], selected: Int, style: Style, selectedStyle: Style, separator: Text) {
        self.titles = titles
        self.selected = selected
        self.style = style
        self.selectedStyle = selectedStyle
        self.separator = separator
    }

    public func render(in rect: Rect, on canvas: inout Canvas) {
        guard !rect.isEmpty else { return }
        var text = Text(" ", style: style)
        for (index, title) in titles.enumerated() {
            if index > 0 { text.append(separator) }
            let chosen = index == selected
            text.append(Text(spans: title.spans.map { span in
                Span(span.text, span.style == .plain ? (chosen ? selectedStyle : style) : span.style)
            }))
        }
        canvas.put(text.truncated(to: rect.width), x: rect.minX, y: rect.minY, clip: rect)
    }
}
