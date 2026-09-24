//
//  Capabilities.swift
//  RaoLMTerminal
//
//  WHAT: What the terminal can show: colour depth, italics, Unicode glyphs.
//  IN:   The environment. First match wins: RAOLM_COLOR (truecolor|256|16|none) → NO_COLOR →
//        COLORTERM=truecolor|24bit → TERM_PROGRAM (iTerm.app, WezTerm, ghostty → truecolour;
//        Apple_Terminal → 256) → TERM (…direct → truecolour, …256color → 256, dumb → none) →
//        256 on a TTY, none otherwise.
//  PIN:  TERM, COLORTERM and TERM_PROGRAM may all be empty (editor-hosted shells), so the
//        fallback is a conservative 256 colours rather than none. NO_COLOR removes colour only;
//        RAOLM_ASCII=1 swaps the Unicode glyphs for ASCII.
//

import Foundation

public enum ColorDepth: Int, Sendable, Comparable, CustomStringConvertible {
    case none = 0, ansi16, ansi256, trueColor

    public static func < (lhs: ColorDepth, rhs: ColorDepth) -> Bool { lhs.rawValue < rhs.rawValue }

    public var description: String {
        switch self {
        case .none: return "none"
        case .ansi16: return "16"
        case .ansi256: return "256"
        case .trueColor: return "truecolor"
        }
    }
}

public struct Capabilities: Sendable, Equatable {
    public var isTTY: Bool
    public var colorDepth: ColorDepth
    public var italics: Bool
    public var unicode: Bool
    /// `\e[?2026h/l` (synchronized output); terminals that do not know it ignore it.
    public var synchronizedOutput: Bool

    public init(isTTY: Bool, colorDepth: ColorDepth, italics: Bool = true, unicode: Bool = true, synchronizedOutput: Bool = true) {
        self.isTTY = isTTY
        self.colorDepth = colorDepth
        self.italics = italics
        self.unicode = unicode
        self.synchronizedOutput = synchronizedOutput
    }

    public static func detect(environment: [String: String], isTTY: Bool) -> Capabilities {
        func value(_ key: String) -> String { environment[key]?.lowercased() ?? "" }
        let depth: ColorDepth
        switch value("RAOLM_COLOR") {
        case "truecolor", "24bit", "true": depth = .trueColor
        case "256": depth = .ansi256
        case "16": depth = .ansi16
        case "none", "0", "mono": depth = .none
        default:
            let term = value("TERM")
            let program = environment["TERM_PROGRAM"] ?? ""
            if !(environment["NO_COLOR"] ?? "").isEmpty {
                depth = .none
            } else if ["truecolor", "24bit"].contains(value("COLORTERM")) {
                depth = .trueColor
            } else if ["iTerm.app", "WezTerm", "ghostty"].contains(program) {
                depth = .trueColor
            } else if program == "Apple_Terminal" {
                depth = .ansi256
            } else if term.contains("direct") {
                depth = .trueColor
            } else if term.contains("256color") {
                depth = .ansi256
            } else if term == "dumb" {
                depth = .none
            } else {
                depth = isTTY ? .ansi256 : .none
            }
        }
        let unicode = environment["RAOLM_ASCII"].map { $0.isEmpty || $0 == "0" } ?? true
        return Capabilities(
            isTTY: isTTY, colorDepth: depth, italics: environment["RAOLM_NO_ITALIC"] == nil,
            unicode: unicode && value("TERM") != "dumb")
    }
}
