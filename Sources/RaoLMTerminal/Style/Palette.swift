//
//  Palette.swift
//  RaoLMTerminal
//
//  WHAT: The RaoLM brand as terminal styles: a charcoal ground, the silver→white wordmark,
//        the three braid strands (blue Ambient, green Craft, orange Veil) converging into the
//        off-white thread, and the gold ring. Semantic tokens map onto them.
//  PIN:  Each token pins its own 256- and 16-colour choice instead of trusting nearest-colour
//        degradation (nearest 16 for the brand green is cyan). Gold is reserved for
//        confidence 1.0, verified spans, the selection rule and the braid's terminus.
//        RAOLM_UI_THEME=terminal leaves the terminal's own background and foreground alone.
//

import Foundation

public struct Palette: Sendable {
    public var depth: ColorDepth
    public var paintsBackground: Bool

    public var base: Style
    public var text: Style
    public var dim: Style
    public var muted: Style
    public var title: Style
    public var border: Style
    public var focusBorder: Style

    public var blue: Style
    public var green: Style
    public var orange: Style
    public var gold: Style
    public var red: Style

    public var ok: Style
    public var warn: Style
    public var error: Style
    public var selection: Style
    public var cursor: Style

    public var hintKey: Style
    public var hintLabel: Style
    public var tagline: Style

    public var wordmarkStops: [Color]
    public var line: Color
    public var braid: [Color]
    public var goldColor: Color

    /// Confidence ramp for generated tokens: uncited, < 0.5, < 0.8, < 1.0, verified.
    public var heat: [Style]

    public init(for capabilities: Capabilities, environment: [String: String] = ProcessInfo.processInfo.environment) {
        let depth = capabilities.colorDepth
        self.depth = depth
        func pick(_ hex: UInt32, _ x256: UInt8, _ ansi: UInt8) -> Color {
            switch depth {
            case .trueColor: return Color(hex: hex)
            case .ansi256: return .indexed(x256)
            case .ansi16: return .ansi(ansi)
            case .none: return .default
            }
        }
        let paints = depth >= .ansi256 && environment["RAOLM_UI_THEME"]?.lowercased() != "terminal"
        paintsBackground = paints
        let charcoal = paints ? pick(0x141414, 233, 0) : Color.default
        let fg = paints ? pick(0xE6E2D8, 254, 7) : Color.default
        let greyDim = pick(0x8A8A8A, 245, 8)
        let greyMuted = pick(0x6E6E6E, 242, 8)
        let borderColor = pick(0x3A3A3A, 237, 8)
        let white = pick(0xF4F4F4, 255, 15)
        let blueColor = pick(0x4C8DF5, 69, 12)
        let greenColor = pick(0x2EB67D, 36, 2)
        let orangeColor = pick(0xF0883E, 209, 3)
        let goldColor = pick(0xD4AF37, 179, 3)
        let redColor = pick(0xE5484D, 167, 1)
        let mono = depth == .none

        base = Style(foreground: fg, background: charcoal)
        text = Style(foreground: fg, background: charcoal)
        dim = mono ? Style(attributes: .dim) : Style(foreground: greyDim, background: charcoal)
        muted = mono ? Style(attributes: .dim) : Style(foreground: greyMuted, background: charcoal)
        title = Style(foreground: paints ? white : .default, background: charcoal, attributes: .bold)
        border = mono ? Style(attributes: .dim) : Style(foreground: borderColor, background: charcoal)
        focusBorder = mono ? Style() : Style(foreground: greyDim, background: charcoal)
        blue = Style(foreground: blueColor, background: charcoal)
        green = Style(foreground: greenColor, background: charcoal)
        orange = Style(foreground: orangeColor, background: charcoal)
        gold = Style(foreground: goldColor, background: charcoal, attributes: depth <= .ansi16 ? .bold : [])
        red = Style(foreground: redColor, background: charcoal)
        ok = green
        warn = orange
        error = mono ? Style(attributes: .bold) : red
        selection = mono || depth == .ansi16
            ? Style(attributes: .reverse)
            : Style(foreground: white, background: pick(0x2A2A2A, 235, 0), attributes: .bold)
        cursor = Style(foreground: fg, background: charcoal, attributes: .reverse)
        hintKey = mono ? Style(attributes: .bold) : Style(foreground: fg, background: charcoal, attributes: .bold)
        hintLabel = dim
        tagline = dim.italic()

        switch depth {
        case .trueColor:
            wordmarkStops = [0xA9A9A9, 0xB5B5B5, 0xC2C2C2, 0xCECECE, 0xDBDBDB, 0xE7E7E7, 0xF4F4F4].map { Color(hex: $0) }
        case .ansi256:
            wordmarkStops = [248, 249, 251, 252, 253, 254, 255].map { Color.indexed($0) }
        case .ansi16:
            wordmarkStops = [7, 7, 7, 15, 15, 15, 15].map { Color.ansi($0) }
        case .none:
            wordmarkStops = Array(repeating: .default, count: 7)
        }
        if !paints && depth >= .ansi256 {
            // On the terminal's own background the silver end may vanish on a light theme;
            // keep the gradient but anchor it in mid-greys.
            wordmarkStops = depth == .trueColor
                ? [0x7A7A7A, 0x858585, 0x919191, 0x9C9C9C, 0xA8A8A8, 0xB3B3B3, 0xBFBFBF].map { Color(hex: $0) }
                : [243, 244, 245, 246, 247, 248, 249].map { Color.indexed($0) }
        }
        line = paints ? fg : pick(0xB8B4AA, 250, 7)
        braid = [blueColor, greenColor, orangeColor]
        self.goldColor = goldColor

        heat = [
            muted,
            blue,
            green,
            mono || depth == .ansi16 ? Style(foreground: fg, background: charcoal, attributes: .bold) : text,
            Style(foreground: goldColor, background: charcoal, attributes: depth <= .ansi16 ? [.bold, .underline] : [.underline]),
        ]
    }

    /// The heat style for a token: nil confidence means uncited.
    public func heat(confidence: Float?, verified: Bool) -> Style {
        if verified { return heat[4] }
        guard let confidence, confidence.isFinite else { return heat[0] }
        if confidence < 0.5 { return heat[1] }
        if confidence < 0.8 { return heat[2] }
        return heat[3]
    }

    /// Keeps the ground: a style with a default background gets the palette's.
    public func onBase(_ style: Style) -> Style {
        guard style.background == .default else { return style }
        return style.bg(base.background)
    }
}

/// The glyphs the UI draws, with ASCII stand-ins (RAOLM_ASCII=1 or TERM=dumb).
public struct Glyphs: Sendable {
    public var check, cross, live, idle, verified, prompt, select, ellipsis, newline, dash, dot: String
    public var spinner: [String]
    public var spark: [String]
    public var horizontal, vertical, topLeft, topRight, bottomLeft, bottomRight, teeLeft, teeRight: String
    public var progressFull, progressEmpty, barFull, barEmpty: String
    public var wave, ripple: String

    public static let unicode = Glyphs(
        check: "✓", cross: "✗", live: "●", idle: "○", verified: "◉", prompt: "▸", select: "▶", ellipsis: "…",
        newline: "⏎", dash: "—", dot: "·", spinner: ["◐", "◓", "◑", "◒"],
        spark: ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"],
        horizontal: "─", vertical: "│", topLeft: "┌", topRight: "┐", bottomLeft: "└", bottomRight: "┘",
        teeLeft: "┤", teeRight: "├", progressFull: "━", progressEmpty: "─", barFull: "█", barEmpty: "░",
        wave: "∿", ripple: "~")

    public static let ascii = Glyphs(
        check: "v", cross: "x", live: "*", idle: "o", verified: "@", prompt: ">", select: ">", ellipsis: "~",
        newline: "~", dash: "-", dot: ".", spinner: ["|", "/", "-", "\\"],
        spark: [".", ".", ":", ":", "-", "=", "#", "#"],
        horizontal: "-", vertical: "|", topLeft: "+", topRight: "+", bottomLeft: "+", bottomRight: "+",
        teeLeft: "+", teeRight: "+", progressFull: "=", progressEmpty: "-", barFull: "#", barEmpty: ".",
        wave: "~", ripple: "-")

    public static func `for`(_ capabilities: Capabilities) -> Glyphs { capabilities.unicode ? .unicode : .ascii }
}
