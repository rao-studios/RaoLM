//
//  KeyDecoder.swift
//  RaoLMTerminal
//
//  WHAT: Turns the bytes a terminal sends in raw mode into key events: printable characters
//        (UTF-8 assembled across reads), control keys, CSI (`\e[A`, `\e[1;5C`, `\e[3~`, `\e[Z`)
//        and SS3 (`\eOA`) sequences, and Esc+key as Alt.
//  PIN:  A lone Esc cannot be told apart from the start of a sequence until more bytes arrive
//        or a short timeout passes; the reader calls `flush()` on that timeout. Raw mode turns
//        off ISIG, so Ctrl-C arrives here as `.ctrl("c")`, not as SIGINT.
//

import Foundation

public struct Modifiers: OptionSet, Sendable, Hashable {
    public let rawValue: UInt8
    public init(rawValue: UInt8) { self.rawValue = rawValue }

    public static let shift = Modifiers(rawValue: 1)
    public static let alt = Modifiers(rawValue: 2)
    public static let ctrl = Modifiers(rawValue: 4)
}

public enum Key: Sendable, Hashable {
    case char(Character)
    case enter, tab, backTab, backspace, delete, insert, escape
    case up, down, left, right, home, end, pageUp, pageDown
    case function(Int)
    /// A control chord, lowercase letter (Ctrl-C is `.ctrl("c")`).
    case ctrl(Character)
    case unknown([UInt8])
}

public struct KeyEvent: Sendable, Hashable, CustomStringConvertible {
    public var key: Key
    public var modifiers: Modifiers

    public init(_ key: Key, _ modifiers: Modifiers = []) {
        self.key = key
        self.modifiers = modifiers
    }

    public static func char(_ c: Character) -> KeyEvent { KeyEvent(.char(c)) }

    public var description: String {
        var prefix = ""
        if modifiers.contains(.ctrl) { prefix += "ctrl-" }
        if modifiers.contains(.alt) { prefix += "alt-" }
        if modifiers.contains(.shift) { prefix += "shift-" }
        return prefix + "\(key)"
    }
}

public struct KeyDecoder: Sendable {
    private var pending: [UInt8] = []

    public init() {}

    public var hasPending: Bool { !pending.isEmpty }

    public mutating func feed(_ bytes: [UInt8]) -> [KeyEvent] {
        pending.append(contentsOf: bytes)
        var events: [KeyEvent] = []
        var index = 0
        while index < pending.count {
            guard let (event, length) = Self.parse(pending, at: index) else { break }
            if let event { events.append(event) }
            index += length
        }
        pending.removeFirst(index)
        return events
    }

    /// Resolves whatever is pending as if no more bytes will come: a lone Esc becomes
    /// `.escape`; the rest of a cut-off sequence becomes plain characters.
    public mutating func flush() -> [KeyEvent] {
        guard !pending.isEmpty else { return [] }
        var bytes = pending
        pending.removeAll()
        var events: [KeyEvent] = []
        if bytes.first == 0x1B {
            events.append(KeyEvent(.escape))
            bytes.removeFirst()
        }
        var rest = KeyDecoder()
        events += rest.feed(bytes)
        if rest.hasPending {
            // A cut-off UTF-8 sequence or another lone Esc.
            events += rest.pending.map { $0 == 0x1B ? KeyEvent(.escape) : KeyEvent(.unknown([$0])) }
        }
        return events
    }

    /// One event and its byte length, `(nil, n)` for bytes to skip, or nil when more bytes
    /// are needed.
    static func parse(_ bytes: [UInt8], at start: Int) -> (KeyEvent?, Int)? {
        let byte = bytes[start]
        switch byte {
        case 0x1B:
            return parseEscape(bytes, at: start)
        case 0x0D, 0x0A:
            return (KeyEvent(.enter), 1)
        case 0x09:
            return (KeyEvent(.tab), 1)
        case 0x7F, 0x08:
            return (KeyEvent(.backspace), 1)
        case 0x00:
            return (KeyEvent(.ctrl(" ")), 1)
        case 0x01...0x1A:
            return (KeyEvent(.ctrl(Character(Unicode.Scalar(byte + 0x60)))), 1)
        case 0x1C...0x1F:
            return (KeyEvent(.ctrl(Character(Unicode.Scalar(byte + 0x40)))), 1)
        case 0x20..<0x80:
            return (KeyEvent(.char(Character(Unicode.Scalar(byte)))), 1)
        default:
            return parseUTF8(bytes, at: start)
        }
    }

    private static func parseUTF8(_ bytes: [UInt8], at start: Int) -> (KeyEvent?, Int)? {
        let lead = bytes[start]
        let length: Int
        switch lead {
        case 0xC0...0xDF: length = 2
        case 0xE0...0xEF: length = 3
        case 0xF0...0xF7: length = 4
        default: return (KeyEvent(.unknown([lead])), 1)
        }
        guard start + length <= bytes.count else { return nil }
        let slice = Array(bytes[start..<(start + length)])
        guard slice.dropFirst().allSatisfy({ $0 & 0xC0 == 0x80 }) else { return (KeyEvent(.unknown([lead])), 1) }
        let string = String(decoding: slice, as: UTF8.self)
        if string.count == 1, let character = string.first, character != "\u{FFFD}" {
            return (KeyEvent(.char(character)), length)
        }
        return (KeyEvent(.unknown(slice)), length)
    }

    private static func parseEscape(_ bytes: [UInt8], at start: Int) -> (KeyEvent?, Int)? {
        guard start + 1 < bytes.count else { return nil }
        let next = bytes[start + 1]
        switch next {
        case 0x5B:  // [
            return parseCSI(bytes, at: start)
        case 0x4F:  // O
            guard start + 2 < bytes.count else { return nil }
            let final = bytes[start + 2]
            let key: Key? = ss3Key(final)
            return (KeyEvent(key ?? .unknown(Array(bytes[start...(start + 2)]))), 3)
        case 0x1B:
            return (KeyEvent(.escape), 1)
        default:
            guard let (event, length) = parse(bytes, at: start + 1) else { return nil }
            guard var event else { return (nil, length + 1) }
            event.modifiers.insert(.alt)
            return (event, length + 1)
        }
    }

    private static func ss3Key(_ final: UInt8) -> Key? {
        switch final {
        case 0x41: return .up
        case 0x42: return .down
        case 0x43: return .right
        case 0x44: return .left
        case 0x48: return .home
        case 0x46: return .end
        case 0x50: return .function(1)
        case 0x51: return .function(2)
        case 0x52: return .function(3)
        case 0x53: return .function(4)
        default: return nil
        }
    }

    private static func parseCSI(_ bytes: [UInt8], at start: Int) -> (KeyEvent?, Int)? {
        var index = start + 2
        var parameters = ""
        while index < bytes.count {
            let byte = bytes[index]
            if (0x30...0x3F).contains(byte) || (0x20...0x2F).contains(byte) {
                parameters.append(Character(Unicode.Scalar(byte)))
                index += 1
                continue
            }
            guard (0x40...0x7E).contains(byte) else {
                // Not a CSI after all: report the ESC [ and let the rest parse normally.
                return (KeyEvent(.unknown([0x1B, 0x5B])), 2)
            }
            let length = index - start + 1
            let raw = Array(bytes[start..<(index + 1)])
            let numbers = parameters.split(separator: ";", omittingEmptySubsequences: false).map { Int($0) }
            var modifiers: Modifiers = []
            if numbers.count >= 2, let code = numbers[1], code > 1 {
                let bits = UInt8(truncatingIfNeeded: code - 1)
                if bits & 1 != 0 { modifiers.insert(.shift) }
                if bits & 2 != 0 { modifiers.insert(.alt) }
                if bits & 4 != 0 { modifiers.insert(.ctrl) }
            }
            let key: Key
            switch byte {
            case 0x41: key = .up
            case 0x42: key = .down
            case 0x43: key = .right
            case 0x44: key = .left
            case 0x48: key = .home
            case 0x46: key = .end
            case 0x5A: key = .backTab
            case 0x50: key = .function(1)
            case 0x51: key = .function(2)
            case 0x52: key = .function(3)
            case 0x53: key = .function(4)
            case 0x7E:
                switch numbers.first ?? nil {
                case 1, 7: key = .home
                case 4, 8: key = .end
                case 2: key = .insert
                case 3: key = .delete
                case 5: key = .pageUp
                case 6: key = .pageDown
                case let n? where (11...15).contains(n): key = .function(n - 10)
                case let n? where (17...21).contains(n): key = .function(n - 11)
                case 23: key = .function(11)
                case 24: key = .function(12)
                default: key = .unknown(raw)
                }
            default:
                key = .unknown(raw)
            }
            return (KeyEvent(key, modifiers), length)
        }
        return nil
    }
}
