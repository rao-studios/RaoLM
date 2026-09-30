//
//  Terminal.swift
//  RaoLMTerminal
//
//  WHAT: The real terminal: raw mode, the alternate screen, the cursor, its size, and its
//        input, resize and termination signals as one event stream.
//  PIN:  Every exit path restores the terminal. Normal and thrown exits go through the app's
//        `defer { display.restore() }`; SIGINT/SIGTERM/SIGHUP become events (the app decides);
//        `exit()` anywhere runs the atexit hook; a crash (SIGSEGV, SIGBUS, SIGILL, SIGTRAP,
//        SIGABRT, SIGFPE) runs a C handler that only writes a preallocated reset sequence and
//        calls tcsetattr — both async-signal-safe — then re-raises with the default action so
//        crash reports still happen. Raw mode keeps OPOST|ONLCR (as MaryPi's console does) so a
//        stray `\n` still returns the carriage. `enter()` drains the line-buffered stdout FILE*
//        first so nothing printed earlier lands on the alternate screen.
//

import Darwin
import Foundation

public final class Terminal: Display, @unchecked Sendable {
    public let capabilities: Capabilities
    public let events: AsyncStream<TerminalEvent>
    private let continuation: AsyncStream<TerminalEvent>.Continuation
    private let input: Int32
    private let output: Int32
    private let environment: [String: String]

    private let lock = NSLock()
    private var active = false
    private var original = termios()
    private var buffer = OutputBuffer()
    private var reader: InputReader?
    private var signalSources: [DispatchSourceSignal] = []
    private var previousHandlers: [(Int32, sig_t?)] = []
    private var savedStandardError: Int32 = -1
    private let inputQueue = DispatchQueue(label: "raolm.terminal.input")
    private let signalQueue = DispatchQueue(label: "raolm.terminal.signals")

    public init(
        input: Int32 = STDIN_FILENO, output: Int32 = STDOUT_FILENO,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) {
        self.input = input
        self.output = output
        self.environment = environment
        let isTTY = isatty(input) == 1 && isatty(output) == 1
        self.capabilities = Capabilities.detect(environment: environment, isTTY: isTTY)
        (events, continuation) = AsyncStream.makeStream(of: TerminalEvent.self, bufferingPolicy: .unbounded)
    }

    /// Whether both standard input and output are terminals.
    public static var isInteractive: Bool {
        isatty(STDIN_FILENO) == 1 && isatty(STDOUT_FILENO) == 1
            && ProcessInfo.processInfo.environment["TERM"] != "dumb"
    }

    public var size: Size {
        if let size = Self.querySize(fd: output) { return size }
        if let columns = environment["COLUMNS"].flatMap(Int.init), let lines = environment["LINES"].flatMap(Int.init),
           columns > 0, lines > 0 {
            return Size(width: columns, height: lines)
        }
        return Size(width: 80, height: 24)
    }

    public static func querySize(fd: Int32) -> Size? {
        var window = winsize()
        guard ioctl(fd, UInt(TIOCGWINSZ), &window) == 0, window.ws_col > 0, window.ws_row > 0 else { return nil }
        return Size(width: Int(window.ws_col), height: Int(window.ws_row))
    }

    public func enter() throws {
        lock.lock()
        defer { lock.unlock() }
        guard !active else { return }
        guard isatty(input) == 1, isatty(output) == 1 else { throw TerminalError.notATTY }
        fflush(stdout)
        var settings = termios()
        guard tcgetattr(input, &settings) == 0 else { throw TerminalError.termios(errno: errno) }
        original = settings
        var raw = settings
        cfmakeraw(&raw)
        raw.c_oflag |= tcflag_t(OPOST | ONLCR)
        guard tcsetattr(input, TCSANOW, &raw) == 0 else { throw TerminalError.termios(errno: errno) }
        CrashRestore.arm(input: input, output: output, settings: original)

        OutputBuffer.writeAll(Array(ANSI.enter.utf8), to: output)

        let continuation = self.continuation
        let output = self.output
        for (number, event) in [(SIGWINCH, nil), (SIGINT, TerminationSignal.interrupt),
                                (SIGTERM, .terminate), (SIGHUP, .hangup)] as [(Int32, TerminationSignal?)] {
            previousHandlers.append((number, signal(number, SIG_IGN)))
            let source = DispatchSource.makeSignalSource(signal: number, queue: signalQueue)
            source.setEventHandler {
                if let event {
                    continuation.yield(.signal(event))
                } else if let size = Terminal.querySize(fd: output) {
                    continuation.yield(.resize(size))
                }
            }
            source.resume()
            signalSources.append(source)
        }
        let reader = InputReader(fd: input, queue: inputQueue) { continuation.yield($0) }
        reader.start()
        self.reader = reader
        active = true
    }

    public func restore() {
        lock.lock()
        defer { lock.unlock() }
        guard active else { return }
        reader?.stop()
        reader = nil
        for source in signalSources { source.cancel() }
        signalSources.removeAll()
        for (number, handler) in previousHandlers { signal(number, handler) }
        previousHandlers.removeAll()
        buffer.append(ANSI.restore)
        buffer.flush(to: output)
        var settings = original
        tcsetattr(input, TCSANOW, &settings)
        if savedStandardError >= 0 {
            dup2(savedStandardError, STDERR_FILENO)
            close(savedStandardError)
            savedStandardError = -1
        }
        CrashRestore.disarm()
        active = false
        continuation.finish()
    }

    public func write(_ bytes: [UInt8]) {
        lock.lock()
        buffer.append(bytes)
        lock.unlock()
    }

    public func flush() {
        lock.lock()
        buffer.flush(to: output)
        lock.unlock()
    }

    /// Sends standard error to `url` until `restore()`: MLX, Metal and library warnings would
    /// otherwise print over the screen.
    public func captureStandardError(to url: URL) throws {
        lock.lock()
        defer { lock.unlock() }
        guard savedStandardError < 0 else { return }
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let fd = open(url.path, O_WRONLY | O_CREAT | O_APPEND, 0o644)
        guard fd >= 0 else { throw TerminalError.termios(errno: errno) }
        fflush(stderr)
        savedStandardError = dup(STDERR_FILENO)
        dup2(fd, STDERR_FILENO)
        close(fd)
    }
}

enum ANSI {
    static let enter = "\u{1B}[?1049h\u{1B}[?25l\u{1B}[2J\u{1B}[H"
    static let restore = "\u{1B}[?2026l\u{1B}[0m\u{1B}[?25h\u{1B}[?1049l"
}

/// State the crash and exit hooks read. Written only while no hook can run (before arming,
/// after disarming), so the C handlers see a consistent snapshot.
enum CrashRestore {
    nonisolated(unsafe) static var armed: Int32 = 0
    nonisolated(unsafe) static var input: Int32 = -1
    nonisolated(unsafe) static var output: Int32 = -1
    nonisolated(unsafe) static var settings = termios()
    nonisolated(unsafe) static var sequence: UnsafeMutablePointer<UInt8>?
    nonisolated(unsafe) static var sequenceLength = 0
    nonisolated(unsafe) static var installed = false

    static func arm(input: Int32, output: Int32, settings: termios) {
        armed = 0
        Self.input = input
        Self.output = output
        Self.settings = settings
        if sequence == nil {
            let bytes = Array(ANSI.restore.utf8)
            let pointer = UnsafeMutablePointer<UInt8>.allocate(capacity: bytes.count)
            pointer.initialize(from: bytes, count: bytes.count)
            sequence = pointer
            sequenceLength = bytes.count
        }
        if !installed {
            installed = true
            atexit { CrashRestore.restoreIfArmed() }
            for number in [SIGSEGV, SIGBUS, SIGILL, SIGTRAP, SIGABRT, SIGFPE] {
                signal(number, crashHandler)
            }
        }
        armed = 1
    }

    static func disarm() { armed = 0 }

    static func restoreIfArmed() {
        guard armed != 0 else { return }
        armed = 0
        if let sequence { _ = Darwin.write(output, sequence, sequenceLength) }
        var copy = settings
        _ = tcsetattr(input, TCSANOW, &copy)
    }
}

private let crashHandler: @convention(c) (Int32) -> Void = { number in
    CrashRestore.restoreIfArmed()
    signal(number, SIG_DFL)
    raise(number)
}
