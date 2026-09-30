//
//  Mailbox.swift
//  RaoLMTerminal
//
//  WHAT: How background work talks to the UI: a mailbox any thread may post to, and a flag
//        the UI sets to ask a worker to stop.
//  PIN:  Posting yields into the app's AsyncStream, which is thread-safe; the worker never
//        touches UI state. Workers poll the cancel flag between steps.
//

import Foundation
import Synchronization

public final class Mailbox<Message: Sendable>: Sendable {
    private let sink: @Sendable (Message) -> Void

    public init(_ sink: @escaping @Sendable (Message) -> Void) {
        self.sink = sink
    }

    public func post(_ message: Message) { sink(message) }

    /// A mailbox that drops everything (tests, previews).
    public static func discarding() -> Mailbox<Message> { Mailbox { _ in } }
}

public final class CancelFlag: Sendable {
    private let flag = Atomic<Bool>(false)

    public init() {}

    public func cancel() { flag.store(true, ordering: .releasing) }
    public func reset() { flag.store(false, ordering: .releasing) }
    public var isCancelled: Bool { flag.load(ordering: .acquiring) }
}
