//
//  App.swift
//  RaoLMTerminal
//
//  WHAT: The full-screen app loop: one stream merging key presses, resizes, termination
//        signals, a 100 ms tick and worker messages; an update function over a state value;
//        a render function into a frame; at most one frame per 33 ms.
//  PIN:  A burst of worker messages coalesces into one draw per frame interval, and the last
//        message is always drawn. Keys go through the same path (≤ 33 ms latency). Below the
//        minimum size the app draws a notice instead of calling `render`. A second interrupt
//        (Ctrl-C or SIGINT) within 1.5 s returns 130 whatever `update` says, so a stuck worker
//        can never trap the user. `defer { display.restore() }` covers every way out of `run`.
//

import Foundation

public struct Frame {
    public var canvas: Canvas
    public let palette: Palette
    public let glyphs: Glyphs
    /// Where to show the hardware cursor, if anywhere.
    public var cursor: (x: Int, y: Int)?

    public init(canvas: Canvas, palette: Palette, glyphs: Glyphs, cursor: (x: Int, y: Int)? = nil) {
        self.canvas = canvas
        self.palette = palette
        self.glyphs = glyphs
        self.cursor = cursor
    }

    public var bounds: Rect { canvas.bounds }
    public var size: Size { canvas.size }
}

@MainActor
public final class App<State, Message: Sendable> {
    public struct Configuration: Sendable {
        public var frameInterval: Duration
        public var tickInterval: Duration
        public var minimumSize: Size
        public var showStats: Bool

        public init(
            frameInterval: Duration = .milliseconds(33), tickInterval: Duration = .milliseconds(100),
            minimumSize: Size = Size(width: 80, height: 24),
            showStats: Bool = ProcessInfo.processInfo.environment["RAOLM_TUI_STATS"] == "1"
        ) {
            self.frameInterval = frameInterval
            self.tickInterval = tickInterval
            self.minimumSize = minimumSize
            self.showStats = showStats
        }
    }

    private enum Internal: Sendable {
        case event(Event<Message>)
        case frame
    }

    public nonisolated let mailbox: Mailbox<Message>
    public let palette: Palette
    public let glyphs: Glyphs
    public private(set) var state: State

    private let display: any Display
    private let configuration: Configuration
    private let update: @MainActor (inout State, Event<Message>) -> Command
    private let render: @MainActor (State, inout Frame) -> Void
    private let stream: AsyncStream<Internal>
    private let continuation: AsyncStream<Internal>.Continuation
    private var renderer: Renderer
    private var canvas: Canvas
    private var framePending = false
    private var frameTask: Task<Void, Never>?
    private var lastDraw: ContinuousClock.Instant?
    private var lastInterrupt: ContinuousClock.Instant?
    private var frames: [ContinuousClock.Instant] = []
    private var lastBytes = 0

    public init(
        display: any Display, initial: State, configuration: Configuration = Configuration(),
        palette: Palette? = nil,
        update: @escaping @MainActor (inout State, Event<Message>) -> Command,
        render: @escaping @MainActor (State, inout Frame) -> Void
    ) {
        self.display = display
        self.state = initial
        self.configuration = configuration
        self.update = update
        self.render = render
        self.palette = palette ?? Palette(for: display.capabilities)
        self.glyphs = Glyphs.for(display.capabilities)
        renderer = Renderer(capabilities: display.capabilities)
        canvas = Canvas(size: display.size)
        let (stream, continuation) = AsyncStream.makeStream(of: Internal.self, bufferingPolicy: .unbounded)
        self.stream = stream
        self.continuation = continuation
        mailbox = Mailbox { continuation.yield(.event(.message($0))) }
    }

    /// Runs until `update` returns `.quit` (its code), input closes (0), or a double
    /// interrupt (130).
    public func run() async throws -> Int32 {
        try display.enter()
        defer { display.restore() }
        let continuation = self.continuation
        let display = self.display
        let forwarder = Task.detached {
            for await event in display.events {
                switch event {
                case .key(let key): continuation.yield(.event(.key(key)))
                case .resize(let size): continuation.yield(.event(.resize(size)))
                case .signal(let signal): continuation.yield(.event(.signal(signal)))
                case .inputClosed: continuation.yield(.event(.inputClosed))
                }
            }
        }
        let interval = configuration.tickInterval
        let ticker = Task.detached {
            while !Task.isCancelled {
                try? await Task.sleep(for: interval)
                if Task.isCancelled { break }
                continuation.yield(.event(.tick(Date())))
            }
        }
        defer {
            forwarder.cancel()
            ticker.cancel()
            frameTask?.cancel()
        }

        draw()
        for await item in stream {
            switch item {
            case .frame:
                framePending = false
                draw()
            case .event(let event):
                if case .resize = event { renderer.invalidate() }
                if isInterrupt(event) {
                    let now = ContinuousClock.now
                    if let lastInterrupt, now - lastInterrupt < .milliseconds(1500) { return 130 }
                    lastInterrupt = now
                }
                switch update(&state, event) {
                case .quit(let code): return code
                case .redraw:
                    renderer.invalidate()
                    requestFrame()
                case .none:
                    requestFrame()
                }
                if case .inputClosed = event { return 0 }
            }
        }
        return 0
    }

    private func isInterrupt(_ event: Event<Message>) -> Bool {
        switch event {
        case .signal(.interrupt), .key(KeyEvent(.ctrl("c"))): return true
        default: return false
        }
    }

    private func requestFrame() {
        guard !framePending else { return }
        let now = ContinuousClock.now
        let elapsed = lastDraw.map { now - $0 } ?? configuration.frameInterval
        if elapsed >= configuration.frameInterval {
            draw()
            return
        }
        framePending = true
        let wait = configuration.frameInterval - elapsed
        let continuation = self.continuation
        frameTask = Task.detached {
            try? await Task.sleep(for: wait)
            continuation.yield(.frame)
        }
    }

    private func draw() {
        let size = display.size
        if canvas.size != size {
            canvas = Canvas(size: size)
            renderer.invalidate()
        }
        canvas.clear(fill: palette.base)
        var frame = Frame(canvas: canvas, palette: palette, glyphs: glyphs)
        if size.width < configuration.minimumSize.width || size.height < configuration.minimumSize.height {
            let message = "terminal is \(size); RaoLM needs \(configuration.minimumSize)"
            let lines = Paragraph.wrap(message, width: max(1, size.width))
            let top = max(0, (size.height - lines.count) / 2)
            for (row, line) in lines.enumerated() {
                frame.canvas.put(line, x: max(0, (size.width - TerminalWidth.of(line)) / 2), y: top + row, style: palette.warn)
            }
        } else {
            render(state, &frame)
        }
        let now = ContinuousClock.now
        if configuration.showStats {
            frames.append(now)
            frames.removeAll { now - $0 > .seconds(1) }
            let stats = " \(frames.count) fps · \(lastBytes) B "
            frame.canvas.put(stats, x: max(0, size.width - stats.count), y: 0, style: palette.dim.reverse())
        }
        let bytes = renderer.present(frame.canvas, cursor: frame.cursor)
        lastBytes = bytes.count
        display.write(bytes)
        display.flush()
        canvas = frame.canvas
        lastDraw = now
    }
}
