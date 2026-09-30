//
//  Renderer.swift
//  RaoLMTerminal
//
//  WHAT: Turns a canvas into the bytes that bring the terminal from the previous frame to
//        this one: per row, only the run between the first and last changed cell is redrawn,
//        each run absolutely positioned, all inside synchronized output.
//  PIN:  Pure: it returns bytes and the display writes them, so the diff is testable. The SGR
//        state is forgotten at the start of every frame (anything may have printed since),
//        and a size change or `invalidate()` clears the screen and redraws every row.
//

import Foundation

public struct Renderer: Sendable {
    public let capabilities: Capabilities
    public private(set) var front: Canvas?
    private var encoder: SGREncoder

    public init(capabilities: Capabilities) {
        self.capabilities = capabilities
        encoder = SGREncoder(capabilities: capabilities)
    }

    public mutating func invalidate() { front = nil }

    public mutating func present(_ back: Canvas, cursor: (x: Int, y: Int)?) -> [UInt8] {
        var out = ""
        if capabilities.synchronizedOutput { out += "\u{1B}[?2026h" }
        out += "\u{1B}[?25l"
        encoder.invalidate()
        let full = front == nil || front?.size != back.size
        if full { out += "\u{1B}[0m\u{1B}[2J" }
        let width = back.size.width
        for y in 0..<back.size.height {
            var first = -1
            var last = -1
            if full {
                first = 0
                last = width - 1
            } else if let front {
                for x in 0..<width where front[x, y] != back[x, y] {
                    if first < 0 { first = x }
                    last = x
                }
            }
            guard first >= 0 else { continue }
            // Never start a run on the second half of a wide glyph.
            while first > 0, back[first, y].isContinuation { first -= 1 }
            out += "\u{1B}[\(y + 1);\(first + 1)H"
            for x in first...last {
                let cell = back[x, y]
                if cell.isContinuation { continue }
                out += encoder.transition(to: cell.style)
                out.unicodeScalars.append(cell.scalar)
            }
        }
        out += encoder.reset()
        if let cursor {
            out += "\u{1B}[\(cursor.y + 1);\(cursor.x + 1)H\u{1B}[?25h"
        }
        if capabilities.synchronizedOutput { out += "\u{1B}[?2026l" }
        front = back
        return Array(out.utf8)
    }
}
