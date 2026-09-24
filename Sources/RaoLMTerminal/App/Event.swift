//
//  Event.swift
//  RaoLMTerminal
//
//  WHAT: What reaches an app's update function, and what it can ask for in return.
//

import Foundation

public enum Event<Message: Sendable>: Sendable {
    case key(KeyEvent)
    case resize(Size)
    case tick(Date)
    case signal(TerminationSignal)
    case message(Message)
    case inputClosed
}

public enum Command: Sendable, Equatable {
    case none
    case quit(code: Int32)
    /// Forget what the terminal shows and redraw every cell.
    case redraw

    public static let quit = Command.quit(code: 0)
}
