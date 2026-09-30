//
//  Style.swift
//  RaoLMTerminal
//
//  WHAT: Text attributes and colours, styled text, and the SGR encoder that turns a style
//        change into the shortest correct escape sequence for the terminal's capabilities.
//  PIN:  Every transition emits `\e[0;…m` — a reset followed by the full style — instead of
//        the attribute-off codes (22/23/24/27), whose support varies. Colours degrade here, at
//        emission time, so styles stay depth-independent and testable.
//

import Foundation

public struct Attributes: OptionSet, Sendable, Hashable {
    public let rawValue: UInt8
    public init(rawValue: UInt8) { self.rawValue = rawValue }

    public static let bold = Attributes(rawValue: 1)
    public static let dim = Attributes(rawValue: 2)
    public static let italic = Attributes(rawValue: 4)
    public static let underline = Attributes(rawValue: 8)
    public static let reverse = Attributes(rawValue: 16)
}

public struct Style: Sendable, Hashable {
    public var foreground: Color
    public var background: Color
    public var attributes: Attributes

    public init(foreground: Color = .default, background: Color = .default, attributes: Attributes = []) {
        self.foreground = foreground
        self.background = background
        self.attributes = attributes
    }

    public static let plain = Style()

    public func fg(_ color: Color) -> Style { var copy = self; copy.foreground = color; return copy }
    public func bg(_ color: Color) -> Style { var copy = self; copy.background = color; return copy }
    public func adding(_ more: Attributes) -> Style { var copy = self; copy.attributes.formUnion(more); return copy }
    public func removing(_ some: Attributes) -> Style { var copy = self; copy.attributes.subtract(some); return copy }
    public func bold() -> Style { adding(.bold) }
    public func dim() -> Style { adding(.dim) }
    public func italic() -> Style { adding(.italic) }
    public func underline() -> Style { adding(.underline) }
    public func reverse() -> Style { adding(.reverse) }

    /// `other`'s non-default colours win; attributes are unioned.
    public func merging(_ other: Style) -> Style {
        Style(
            foreground: other.foreground == .default ? foreground : other.foreground,
            background: other.background == .default ? background : other.background,
            attributes: attributes.union(other.attributes))
    }
}

public struct Span: Sendable, Equatable {
    public var text: String
    public var style: Style

    public init(_ text: String, _ style: Style = .plain) {
        self.text = text
        self.style = style
    }
}

public struct Text: Sendable, Equatable, ExpressibleByStringLiteral, CustomStringConvertible {
    public var spans: [Span]

    public init(_ string: String = "", style: Style = .plain) {
        spans = string.isEmpty ? [] : [Span(string, style)]
    }

    public init(spans: [Span]) { self.spans = spans.filter { !$0.text.isEmpty } }

    public init(stringLiteral value: String) { self.init(value) }

    public var plain: String { spans.map(\.text).joined() }
    public var description: String { plain }
    public var width: Int { spans.reduce(0) { $0 + TerminalWidth.of($1.text) } }
    public var isEmpty: Bool { spans.allSatisfy { $0.text.isEmpty } }

    public mutating func append(_ text: String, _ style: Style = .plain) {
        guard !text.isEmpty else { return }
        spans.append(Span(text, style))
    }

    public mutating func append(_ other: Text) { spans.append(contentsOf: other.spans) }

    public static func + (lhs: Text, rhs: Text) -> Text { Text(spans: lhs.spans + rhs.spans) }

    /// Clips to `width` columns, ending in `ellipsis` when anything was cut.
    public func truncated(to width: Int, ellipsis: String = "…") -> Text {
        guard self.width > width else { return self }
        guard width > 0 else { return Text() }
        let budget = max(0, width - TerminalWidth.of(ellipsis))
        var used = 0
        var result: [Span] = []
        var lastStyle = spans.first?.style ?? .plain
        outer: for span in spans {
            var piece = ""
            for scalar in span.text.unicodeScalars {
                let w = TerminalWidth.of(scalar)
                if used + w > budget {
                    if !piece.isEmpty { result.append(Span(piece, span.style)) }
                    lastStyle = span.style
                    break outer
                }
                piece.unicodeScalars.append(scalar)
                used += w
            }
            if !piece.isEmpty { result.append(Span(piece, span.style)) }
            lastStyle = span.style
        }
        result.append(Span(ellipsis, lastStyle))
        return Text(spans: result)
    }

    /// Pads with spaces (in `style`) to exactly `width` columns, truncating when longer.
    public func padded(to width: Int, align: Alignment = .leading, style: Style = .plain) -> Text {
        let clipped = truncated(to: width)
        let gap = max(0, width - clipped.width)
        guard gap > 0 else { return clipped }
        switch align {
        case .leading: return clipped + Text(String(repeating: " ", count: gap), style: style)
        case .trailing: return Text(String(repeating: " ", count: gap), style: style) + clipped
        case .center:
            let left = gap / 2
            return Text(String(repeating: " ", count: left), style: style) + clipped
                + Text(String(repeating: " ", count: gap - left), style: style)
        }
    }
}

public enum Alignment: Sendable, Equatable {
    case leading, center, trailing
}

public struct SGREncoder: Sendable {
    public let capabilities: Capabilities
    public private(set) var current: Style?

    public init(capabilities: Capabilities) {
        self.capabilities = capabilities
    }

    /// The escape sequence that moves the terminal from the current style to `style`; empty
    /// when nothing changes.
    public mutating func transition(to style: Style) -> String {
        if let current, current == style { return "" }
        current = style
        return Self.sequence(for: style, capabilities: capabilities)
    }

    public mutating func reset() -> String {
        current = .plain
        return "\u{1B}[0m"
    }

    /// Forget the terminal's state: the next transition emits a full style.
    public mutating func invalidate() { current = nil }

    public static func sequence(for style: Style, capabilities: Capabilities) -> String {
        var codes = ["0"]
        let a = style.attributes
        if a.contains(.bold) { codes.append("1") }
        if a.contains(.dim) { codes.append("2") }
        if a.contains(.italic), capabilities.italics { codes.append("3") }
        if a.contains(.underline) { codes.append("4") }
        if a.contains(.reverse) { codes.append("7") }
        if let fg = colorCode(style.foreground.degraded(to: capabilities.colorDepth), background: false) { codes.append(fg) }
        if let bg = colorCode(style.background.degraded(to: capabilities.colorDepth), background: true) { codes.append(bg) }
        return "\u{1B}[" + codes.joined(separator: ";") + "m"
    }

    static func colorCode(_ color: Color, background: Bool) -> String? {
        switch color {
        case .default: return nil
        case .rgb(let r, let g, let b): return "\(background ? 48 : 38);2;\(r);\(g);\(b)"
        case .indexed(let index): return "\(background ? 48 : 38);5;\(index)"
        case .ansi(let index):
            let i = Int(index & 0x0F)
            if i < 8 { return String((background ? 40 : 30) + i) }
            return String((background ? 100 : 90) + i - 8)
        }
    }
}
