//
//  Display.swift
//  RaoLMTerminal
//
//  WHAT: The seam between the app loop and a screen: a real terminal (`Terminal`) or an
//        in-memory one (`MemoryDisplay`) that tests drive without a TTY.
//

import Foundation

public struct Size: Sendable, Equatable, CustomStringConvertible {
    public var width: Int
    public var height: Int

    public init(width: Int, height: Int) {
        self.width = width
        self.height = height
    }

    public var description: String { "\(width)×\(height)" }
}

public enum TerminationSignal: Sendable, Equatable {
    case interrupt, terminate, hangup
}

public enum TerminalEvent: Sendable, Equatable {
    case key(KeyEvent)
    case resize(Size)
    case signal(TerminationSignal)
    case inputClosed
}

public enum TerminalError: Error, Sendable, CustomStringConvertible {
    case notATTY
    case termios(errno: Int32)

    public var description: String {
        switch self {
        case .notATTY: return "standard input and output must be a terminal"
        case .termios(let code): return "could not configure the terminal (\(String(cString: strerror(code))))"
        }
    }
}

public protocol Display: AnyObject, Sendable {
    var capabilities: Capabilities { get }
    /// Re-queried on every call.
    var size: Size { get }
    var events: AsyncStream<TerminalEvent> { get }
    /// Raw mode, alternate screen, hidden cursor, cleared screen.
    func enter() throws
    /// Idempotent: safe to call twice and from any exit path.
    func restore()
    /// Buffered until `flush`.
    func write(_ bytes: [UInt8])
    func flush()
}
