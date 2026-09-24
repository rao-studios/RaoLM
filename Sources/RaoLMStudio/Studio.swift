//
//  Studio.swift
//  RaoLMStudio
//
//  WHAT: Opens the studio: the terminal, the app loop, the live or fixture backend, and the
//        jobs the first screen needs.
//  PIN:  The UI draws on a duplicate of the terminal's descriptor while standard output and
//        error go to <data root>/logs/studio.log, so a library that prints (a tokenizer
//        warning, MLX, Metal) can never scribble over the screen. Both come back before the
//        function returns, and the terminal is restored on every path out of `App.run`.
//

import Darwin
import Foundation
import RaoLM
import RaoLMTerminal
import RaoLMWorkflows

@MainActor
final class BackendBox {
    var backend: StudioBackend?
}

/// Points standard output and error at a file until `restore()`.
final class StreamRedirect {
    private var saved: [(Int32, Int32)] = []

    init(to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let fd = open(url.path, O_WRONLY | O_CREAT | O_APPEND, 0o644)
        guard fd >= 0 else { throw RaoLMFailure("cannot open \(url.path) for the studio log", code: 73) }
        fflush(stdout)
        fflush(stderr)
        for target in [STDOUT_FILENO, STDERR_FILENO] {
            saved.append((target, dup(target)))
            dup2(fd, target)
        }
        close(fd)
    }

    func restore() {
        fflush(stdout)
        fflush(stderr)
        for (target, copy) in saved {
            dup2(copy, target)
            close(copy)
        }
        saved.removeAll()
    }
}

public enum Studio {
    /// Standard input and output are a terminal (and not a dumb one).
    public static var isInteractive: Bool { Terminal.isInteractive }

    @MainActor
    public static func launch(_ options: StudioOptions) async throws -> Int32 {
        let tty = dup(STDOUT_FILENO)
        guard tty >= 0 else { throw RaoLMFailure("cannot open the terminal", code: 70) }
        defer { close(tty) }
        let terminal = Terminal(input: STDIN_FILENO, output: tty)
        let redirect = try StreamRedirect(to: options.root.logs.appendingPathComponent("studio.log"))
        defer { redirect.restore() }

        var state = StudioState(root: options.fixtures.map { DataRoot(url: $0) } ?? options.root, fixtures: options.fixtures != nil, owner: options.owner)
        state.size = terminal.size
        let box = BackendBox()
        let app = App<StudioState, StudioEvent>(
            display: terminal, initial: state,
            update: { state, event in
                let jobs = StudioApp.handle(event, state: &state)
                for job in jobs { box.backend?.submit(job) }
                if let code = state.quitCode { return .quit(code: code) }
                if state.redraw {
                    state.redraw = false
                    return .redraw
                }
                return .none
            },
            render: { state, frame in StudioApp.render(state, &frame) })
        let backend: StudioBackend
        if let fixtures = options.fixtures {
            backend = try FixtureBackend(directory: fixtures, mailbox: app.mailbox)
        } else {
            try options.root.ensure()
            backend = LiveBackend(options: options, mailbox: app.mailbox)
        }
        box.backend = backend
        for job in StudioApp.initialJobs(state) { backend.submit(job) }
        let code = try await app.run()
        await backend.shutdown()
        return code
    }
}
