//
//  MemoryDisplay.swift
//  RaoLMTerminal
//
//  WHAT: A headless display for tests and screenshots: a fixed size, events a test sends, and
//        every byte the app wrote.
//

import Foundation

public final class MemoryDisplay: Display, @unchecked Sendable {
    public let capabilities: Capabilities
    public let events: AsyncStream<TerminalEvent>
    private let continuation: AsyncStream<TerminalEvent>.Continuation
    private let lock = NSLock()
    private var currentSize: Size
    private var pending: [UInt8] = []
    private var written: [UInt8] = []
    private var flushes = 0
    private var enters = 0
    private var restores = 0

    public init(size: Size, capabilities: Capabilities = Capabilities(isTTY: true, colorDepth: .ansi256)) {
        self.currentSize = size
        self.capabilities = capabilities
        (events, continuation) = AsyncStream.makeStream(of: TerminalEvent.self, bufferingPolicy: .unbounded)
    }

    public var size: Size {
        lock.lock()
        defer { lock.unlock() }
        return currentSize
    }

    public func resize(to size: Size) {
        lock.lock()
        currentSize = size
        lock.unlock()
        continuation.yield(.resize(size))
    }

    public func send(_ event: TerminalEvent) { continuation.yield(event) }
    public func send(key: KeyEvent) { continuation.yield(.key(key)) }
    public func closeInput() { continuation.yield(.inputClosed) }

    public func enter() throws {
        lock.lock()
        enters += 1
        lock.unlock()
    }

    public func restore() {
        lock.lock()
        restores += 1
        lock.unlock()
    }

    public func write(_ bytes: [UInt8]) {
        lock.lock()
        pending.append(contentsOf: bytes)
        lock.unlock()
    }

    public func flush() {
        lock.lock()
        written.append(contentsOf: pending)
        pending.removeAll()
        flushes += 1
        lock.unlock()
    }

    public var output: [UInt8] {
        lock.lock()
        defer { lock.unlock() }
        return written
    }

    public var flushCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return flushes
    }

    public var enterCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return enters
    }

    public var restoreCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return restores
    }
}
