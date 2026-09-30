//
//  InputReader.swift
//  RaoLMTerminal
//
//  WHAT: Reads a terminal's input on a private serial queue and turns it into key events.
//  PIN:  Blocking `read` inside a DispatchSource handler, as MaryPi's console does: the source
//        only fires when bytes are ready. O_NONBLOCK is never set — the flag lives on the shared
//        file description and would leak into the parent shell. The decoder and the Esc
//        timeout's generation counter are only touched on `queue`.
//

import Foundation

final class InputReader: @unchecked Sendable {
    private let fd: Int32
    private let queue: DispatchQueue
    private let escapeTimeout: DispatchTimeInterval
    private let sink: @Sendable (TerminalEvent) -> Void
    private var decoder = KeyDecoder()
    private var generation = 0
    private let lock = NSLock()
    private var source: DispatchSourceRead?

    init(fd: Int32, queue: DispatchQueue, escapeTimeout: DispatchTimeInterval = .milliseconds(50),
         sink: @escaping @Sendable (TerminalEvent) -> Void) {
        self.fd = fd
        self.queue = queue
        self.escapeTimeout = escapeTimeout
        self.sink = sink
    }

    func start() {
        let source = DispatchSource.makeReadSource(fileDescriptor: fd, queue: queue)
        source.setEventHandler { [weak self] in self?.readAvailable() }
        lock.lock()
        self.source = source
        lock.unlock()
        source.resume()
    }

    func stop() {
        lock.lock()
        let source = self.source
        self.source = nil
        lock.unlock()
        source?.cancel()
    }

    private func readAvailable() {
        var buffer = [UInt8](repeating: 0, count: 4096)
        let count = buffer.withUnsafeMutableBytes { read(fd, $0.baseAddress, $0.count) }
        if count < 0, errno == EINTR || errno == EAGAIN { return }
        guard count > 0 else {
            stop()
            sink(.inputClosed)
            return
        }
        for event in decoder.feed(Array(buffer[0..<count])) { sink(.key(event)) }
        if decoder.hasPending {
            generation &+= 1
            let expected = generation
            queue.asyncAfter(deadline: .now() + escapeTimeout) { [weak self] in
                guard let self, self.generation == expected else { return }
                for event in self.decoder.flush() { self.sink(.key(event)) }
            }
        }
    }
}
