//
//  LogPane.swift
//  RaoLMTerminal
//
//  WHAT: A bounded scroll-back of lines that follows the tail until the user scrolls up.
//

import Foundation

public struct RingBuffer<Element: Sendable>: Sendable {
    public let capacity: Int
    private var storage: [Element] = []
    private var start = 0

    public init(capacity: Int) {
        self.capacity = max(1, capacity)
    }

    public var count: Int { storage.count }
    public var isEmpty: Bool { storage.isEmpty }

    public mutating func append(_ element: Element) {
        if storage.count < capacity {
            storage.append(element)
        } else {
            storage[start] = element
            start = (start + 1) % capacity
        }
    }

    public subscript(index: Int) -> Element { storage[(start + index) % storage.count] }

    public var elements: [Element] { (0..<count).map { self[$0] } }

    public func suffix(_ n: Int) -> [Element] {
        let n = min(max(0, n), count)
        return ((count - n)..<count).map { self[$0] }
    }

    public mutating func removeAll() {
        storage.removeAll()
        start = 0
    }
}

public struct LogState: Sendable {
    public var lines: RingBuffer<Text>
    /// 0 follows the tail; n > 0 shows the window ending n lines above it.
    public var offset: Int = 0

    public init(capacity: Int = 2000) {
        lines = RingBuffer(capacity: capacity)
    }

    public mutating func append(_ line: Text) {
        let full = lines.count == lines.capacity
        lines.append(line)
        if offset > 0, !full { offset += 1 }
    }

    public mutating func append(_ line: String, _ style: Style = .plain) { append(Text(line, style: style)) }

    public mutating func scroll(by delta: Int, visible: Int) {
        offset = min(max(0, offset + delta), max(0, lines.count - visible))
    }

    public mutating func toEnd() { offset = 0 }

    public mutating func clear() {
        lines.removeAll()
        offset = 0
    }
}

public struct LogPane {
    public var state: LogState
    public var style: Style

    public init(state: LogState, style: Style) {
        self.state = state
        self.style = style
    }

    public func render(in rect: Rect, on canvas: inout Canvas) {
        guard !rect.isEmpty else { return }
        let end = max(0, state.lines.count - state.offset)
        let startIndex = max(0, end - rect.height)
        for (row, index) in (startIndex..<end).enumerated() {
            let line = state.lines[index]
            let styled = Text(spans: line.spans.map { Span($0.text, $0.style == .plain ? style : $0.style) })
            canvas.put(styled.truncated(to: rect.width), x: rect.minX, y: rect.minY + row, clip: rect)
        }
    }
}
