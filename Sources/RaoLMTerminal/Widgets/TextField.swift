//
//  TextField.swift
//  RaoLMTerminal
//
//  WHAT: A single-line editor: insert, delete, cursor movement and the readline chords
//        (Ctrl-A/E/U/K/W), scrolling horizontally to keep the cursor in view.
//

import Foundation

public struct TextFieldState: Sendable, Equatable {
    public private(set) var text: String
    /// In characters.
    public private(set) var cursor: Int

    public init(_ text: String = "") {
        self.text = text
        cursor = text.count
    }

    public mutating func set(_ text: String) {
        self.text = text
        cursor = text.count
    }

    /// True when the key edited the text or moved the cursor.
    @discardableResult
    public mutating func handle(_ event: KeyEvent) -> Bool {
        var characters = Array(text)
        switch event.key {
        case .char(let c) where !event.modifiers.contains(.alt):
            characters.insert(c, at: cursor)
            cursor += 1
        case .backspace:
            guard cursor > 0 else { return false }
            characters.remove(at: cursor - 1)
            cursor -= 1
        case .delete, .ctrl("d"):
            guard cursor < characters.count else { return false }
            characters.remove(at: cursor)
        case .left, .ctrl("b"):
            guard cursor > 0 else { return false }
            cursor -= 1
        case .right, .ctrl("f"):
            guard cursor < characters.count else { return false }
            cursor += 1
        case .home, .ctrl("a"):
            cursor = 0
        case .end, .ctrl("e"):
            cursor = characters.count
        case .ctrl("u"):
            characters.removeSubrange(0..<cursor)
            cursor = 0
        case .ctrl("k"):
            characters.removeSubrange(cursor..<characters.count)
        case .ctrl("w"):
            var start = cursor
            while start > 0, characters[start - 1] == " " { start -= 1 }
            while start > 0, characters[start - 1] != " " { start -= 1 }
            characters.removeSubrange(start..<cursor)
            cursor = start
        default:
            return false
        }
        text = String(characters)
        return true
    }
}

public struct TextField {
    public var state: TextFieldState
    public var placeholder: String
    public var focused: Bool
    public var style: Style
    public var placeholderStyle: Style
    public var cursorStyle: Style

    public init(state: TextFieldState, placeholder: String = "", focused: Bool, style: Style, placeholderStyle: Style, cursorStyle: Style) {
        self.state = state
        self.placeholder = placeholder
        self.focused = focused
        self.style = style
        self.placeholderStyle = placeholderStyle
        self.cursorStyle = cursorStyle
    }

    /// Draws the field and returns the cursor cell when focused.
    @discardableResult
    public func render(in rect: Rect, on canvas: inout Canvas) -> (x: Int, y: Int)? {
        guard !rect.isEmpty else { return nil }
        let characters = Array(state.text)
        if characters.isEmpty, !focused {
            canvas.put(placeholder, x: rect.minX, y: rect.minY, style: placeholderStyle, clip: rect)
            return nil
        }
        let visible = max(1, rect.width - 1)
        let start = max(0, state.cursor - visible)
        let shown = String(characters[start..<min(characters.count, start + visible + 1)])
        canvas.put(shown, x: rect.minX, y: rect.minY, style: style, clip: rect)
        if characters.isEmpty, focused, !placeholder.isEmpty {
            canvas.put(placeholder, x: rect.minX + 1, y: rect.minY, style: placeholderStyle, clip: rect)
        }
        guard focused else { return nil }
        let x = rect.minX + TerminalWidth.of(String(characters[start..<state.cursor]))
        let under = state.cursor < characters.count ? String(characters[state.cursor]) : " "
        canvas.put(under, x: x, y: rect.minY, style: cursorStyle, clip: rect)
        return (x, rect.minY)
    }
}
