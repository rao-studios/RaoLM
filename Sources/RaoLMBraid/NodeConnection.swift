//
//  NodeConnection.swift
//  RaoLMBraid
//
//  WHAT: A node the umbrella talks to, however it is reached: a child process over pipes
//        (`NodeProcess`), or a node that dialled a hosting umbrella over TCP (`NodeSocket`,
//        accepted by `BraidListener`). Both carry the same wire: one JSON object per line,
//        numbered requests, replies by number, events at any time (NodeProtocol).
//  PIN:  A socket's first line is the node's announce (its name), read before the connection
//        is handed over. A connection that closes fails every request still waiting on it, as a
//        node process that exits does. TCP_NODELAY on both ends: a token's requests are small
//        and each waits on its reply.
//

import Darwin
import Foundation
import RaoLMCore
import RaoLMProvenance

/// What the session and the strand links need of a node, however it is reached.
public protocol NodeConnection: AnyObject, Sendable {
    var name: String { get }
    /// The node's process id (on its own machine, for a socket).
    var pid: Int32 { get }
    var isRunning: Bool { get }
    func send(_ op: NodeRequest.Op) -> StrandCall<NodeReply>
    func stop(grace: TimeInterval) async
}

extension NodeProcess: NodeConnection {}

/// Numbered requests and their replies over one wire, whatever carries it.
final class WireClient: @unchecked Sendable {
    let name: String
    private let lock = NSLock()
    private let splitter = LineSplitter()
    private var nextID = 1
    private var pending: [Int: StrandCall<NodeReply>] = [:]
    private var closed = false
    var onEvent: @Sendable (NodeEvent) -> Void = { _ in }

    init(name: String) {
        self.name = name
    }

    var isClosed: Bool { lock.withLock { closed } }

    func send(_ op: NodeRequest.Op, write: (Data) throws -> Void) -> StrandCall<NodeReply> {
        let call = StrandCall<NodeReply>()
        lock.lock()
        guard !closed else {
            lock.unlock()
            call.fulfil(.failure(StrandLinkError.closed(name)))
            return call
        }
        let id = nextID
        nextID += 1
        pending[id] = call
        lock.unlock()
        do {
            try write(try NodeWire.encode(NodeRequest(id: id, op: op)))
        } catch {
            lock.withLock { _ = pending.removeValue(forKey: id) }
            call.fulfil(.failure(StrandLinkError.closed(name)))
        }
        return call
    }

    func received(_ data: Data) { lines(splitter.feed(data)) }

    func lines(_ lines: [Data]) {
        for line in lines {
            guard let message = try? NodeWire.decode(NodeOutput.self, line: line) else { continue }
            switch message {
            case .reply(let id, let reply):
                lock.withLock { pending.removeValue(forKey: id) }?.fulfil(.success(reply))
            case .failure(let id, let message, let code):
                lock.withLock { pending.removeValue(forKey: id) }?.fulfil(.failure(StrandLinkError.remote(message, code: code)))
            case .event(let event):
                onEvent(event)
            case .announce:
                break
            }
        }
    }

    /// Fails everything still waiting; later sends fail at once.
    func close() {
        lock.lock()
        closed = true
        let waiting = Array(pending.values)
        pending.removeAll()
        lock.unlock()
        for call in waiting { call.fulfil(.failure(StrandLinkError.closed(name))) }
    }
}

/// A node that dialled the umbrella: the umbrella's side of its TCP connection.
public final class NodeSocket: NodeConnection, @unchecked Sendable {
    public let name: String
    public let pid: Int32
    private let handle: FileHandle
    private let writeLock = NSLock()
    private let client: WireClient
    private let exitLock = NSLock()
    private var exitReported = false
    public var onEvent: @Sendable (NodeEvent) -> Void {
        get { client.onEvent }
        set { client.onEvent = newValue }
    }
    public var onExit: @Sendable (Int32) -> Void = { _ in }

    init(name: String, pid: Int32, fd: Int32) {
        self.name = name
        self.pid = pid
        handle = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
        client = WireClient(name: name)
    }

    public var isRunning: Bool { !client.isClosed }

    /// Starts reading, after any lines that arrived with the announce.
    func start(after early: [Data]) {
        client.lines(early)
        handle.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard let self else { return }
            if data.isEmpty { self.closed() } else { self.client.received(data) }
        }
    }

    public func send(_ op: NodeRequest.Op) -> StrandCall<NodeReply> {
        client.send(op) { data in try writeLock.withLock { try handle.write(contentsOf: data) } }
    }

    private func closed() {
        handle.readabilityHandler = nil
        client.close()
        let first = exitLock.withLock { () -> Bool in
            defer { exitReported = true }
            return !exitReported
        }
        if first { onExit(-1) }
    }

    /// Drops the connection (a missed heartbeat, or a node replaced by its reconnection).
    public func drop() {
        shutdown(handle.fileDescriptor, SHUT_RDWR)
        closed()
    }

    /// Asks the node to shut down, then drops the connection.
    public func stop(grace: TimeInterval = 15) async {
        guard isRunning else { return }
        let call = send(.shutdown)
        _ = try? await withTimeout(grace) { try await call.value() }
        drop()
    }

    private func withTimeout<T: Sendable>(_ seconds: TimeInterval, _ body: @escaping @Sendable () async throws -> T) async throws -> T {
        try await withThrowingTaskGroup(of: T.self) { group in
            group.addTask { try await body() }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(seconds * 1e9))
                throw CancellationError()
            }
            let first = try await group.next()!
            group.cancelAll()
            return first
        }
    }
}

/// Accepts nodes that dial the hosting umbrella and hands each over once it has said who it is.
public final class BraidListener: @unchecked Sendable {
    public let host: String
    public private(set) var port: Int
    private var fd: Int32 = -1
    private let lock = NSLock()
    private var stopped = false
    /// Called for every node that announced itself, on the listener's thread.
    public var onNode: @Sendable (NodeSocket) -> Void = { _ in }

    public init(host: String = "127.0.0.1", port: Int) {
        self.host = host
        self.port = port
    }

    public func start() throws {
        fd = try Self.listen(host: host, port: port)
        port = Self.boundPort(fd) ?? port
        let thread = Thread { [weak self] in self?.acceptLoop() }
        thread.name = "raolm.braid.listener"
        thread.start()
    }

    public func stop() {
        lock.withLock { stopped = true }
        if fd >= 0 {
            shutdown(fd, SHUT_RDWR)
            Darwin.close(fd)
        }
    }

    private func acceptLoop() {
        while !lock.withLock({ stopped }) {
            let client = accept(fd, nil, nil)
            guard client >= 0 else {
                if lock.withLock({ stopped }) { return }
                continue
            }
            Self.tune(client)
            // Who it is: the first line, within ten seconds.
            var timeout = timeval(tv_sec: 10, tv_usec: 0)
            setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
            let splitter = LineSplitter()
            var lines: [Data] = []
            var buffer = [UInt8](repeating: 0, count: 64 * 1024)
            while lines.isEmpty {
                let count = read(client, &buffer, buffer.count)
                guard count > 0 else { break }
                lines = splitter.feed(Data(buffer[0..<count]))
            }
            var off = timeval(tv_sec: 0, tv_usec: 0)
            setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &off, socklen_t(MemoryLayout<timeval>.size))
            guard let first = lines.first, case .announce(let name, let pid)? = try? NodeWire.decode(NodeOutput.self, line: first) else {
                Darwin.close(client)
                continue
            }
            let socket = NodeSocket(name: name, pid: pid, fd: client)
            onNode(socket)
            socket.start(after: Array(lines.dropFirst()))
        }
    }

    // MARK: - Sockets

    static func tune(_ fd: Int32) {
        var one: Int32 = 1
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, socklen_t(MemoryLayout<Int32>.size))
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout<Int32>.size))
    }

    static func addresses(host: String, port: Int, passive: Bool) -> UnsafeMutablePointer<addrinfo>? {
        var hints = addrinfo()
        hints.ai_family = AF_UNSPEC
        hints.ai_socktype = SOCK_STREAM
        hints.ai_protocol = IPPROTO_TCP
        if passive { hints.ai_flags = AI_PASSIVE }
        var result: UnsafeMutablePointer<addrinfo>?
        guard getaddrinfo(host, String(port), &hints, &result) == 0 else { return nil }
        return result
    }

    static func listen(host: String, port: Int) throws -> Int32 {
        guard let list = addresses(host: host, port: port, passive: true) else { throw BraidSessionError.io("cannot resolve \(host)") }
        defer { freeaddrinfo(list) }
        var node: UnsafeMutablePointer<addrinfo>? = list
        while let info = node {
            let fd = socket(info.pointee.ai_family, info.pointee.ai_socktype, info.pointee.ai_protocol)
            if fd >= 0 {
                var one: Int32 = 1
                setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, socklen_t(MemoryLayout<Int32>.size))
                if bind(fd, info.pointee.ai_addr, info.pointee.ai_addrlen) == 0, Darwin.listen(fd, 64) == 0 { return fd }
                Darwin.close(fd)
            }
            node = info.pointee.ai_next
        }
        throw BraidSessionError.io("cannot listen on \(host):\(port): \(String(cString: strerror(errno)))")
    }

    static func boundPort(_ fd: Int32) -> Int? {
        var address = sockaddr_storage()
        var length = socklen_t(MemoryLayout<sockaddr_storage>.size)
        let ok = withUnsafeMutablePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(fd, $0, &length) == 0 }
        }
        guard ok else { return nil }
        switch Int32(address.ss_family) {
        case AF_INET:
            return withUnsafePointer(to: &address) { $0.withMemoryRebound(to: sockaddr_in.self, capacity: 1) { Int(UInt16(bigEndian: $0.pointee.sin_port)) } }
        case AF_INET6:
            return withUnsafePointer(to: &address) { $0.withMemoryRebound(to: sockaddr_in6.self, capacity: 1) { Int(UInt16(bigEndian: $0.pointee.sin6_port)) } }
        default:
            return nil
        }
    }

    /// The node's side: a connected socket to the umbrella, or nil when it cannot be reached.
    public static func dial(host: String, port: Int) -> Int32? {
        guard let list = addresses(host: host, port: port, passive: false) else { return nil }
        defer { freeaddrinfo(list) }
        var node: UnsafeMutablePointer<addrinfo>? = list
        while let info = node {
            let fd = socket(info.pointee.ai_family, info.pointee.ai_socktype, info.pointee.ai_protocol)
            if fd >= 0 {
                if Darwin.connect(fd, info.pointee.ai_addr, info.pointee.ai_addrlen) == 0 {
                    tune(fd)
                    return fd
                }
                Darwin.close(fd)
            }
            node = info.pointee.ai_next
        }
        return nil
    }
}
