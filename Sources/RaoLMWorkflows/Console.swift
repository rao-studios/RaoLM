//
//  Console.swift
//  RaoLMWorkflows
//
//  WHAT: Plain-terminal helpers: line-buffered stdout, errors to stderr, 72-column section
//        rules; the Metal library preflight; and a SIGINT/SIGTERM flag that lets a command shut
//        its children down instead of dying.
//

import Foundation

public enum Console {
    public static func setup() {
        setvbuf(stdout, nil, _IOLBF, 0)
    }

    public static func error(_ message: String) {
        FileHandle.standardError.write(Data((message + "\n").utf8))
    }

    public static func section(_ title: String) {
        print("\n── \(title) " + String(repeating: "─", count: max(0, 72 - title.count)))
    }
}

public enum Preflight {
    /// The Metal library MLX loads for this executable.
    public static func ownMetallib() -> URL? {
        guard let executable = Bundle.main.executableURL?.resolvingSymlinksInPath() else { return nil }
        let directory = executable.deletingLastPathComponent()
        for name in ["mlx.metallib", "Resources/default.metallib", "default.metallib"] {
            let url = directory.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        return nil
    }

    /// MLX aborts the process on the first GPU op without a metallib; fail with a fix instead.
    public static func requireMetallib() throws {
        guard ownMetallib() == nil else { return }
        let path = Bundle.main.executableURL?.deletingLastPathComponent().path ?? "the raolm binary"
        let configuration = path.contains("/Release") || path.contains("/release") ? "release" : "debug"
        throw RaoLMFailure(
            "no mlx.metallib beside \(path); MLX cannot run without it",
            hint: "./build-metallib.sh \(configuration)   (after swift build)", code: 78)
    }
}

/// Stops on SIGINT/SIGTERM without killing the process, so children can be shut down.
public final class StopSignal: @unchecked Sendable {
    private let lock = NSLock()
    private var fired = false
    private var sources: [DispatchSourceSignal] = []

    public init(_ signals: [Int32] = [SIGINT, SIGTERM]) {
        for number in signals {
            signal(number, SIG_IGN)
            let source = DispatchSource.makeSignalSource(signal: number, queue: .global())
            source.setEventHandler { [weak self] in self?.fire() }
            source.resume()
            sources.append(source)
        }
    }

    private func fire() {
        lock.lock()
        fired = true
        lock.unlock()
    }

    public var isSet: Bool {
        lock.lock()
        defer { lock.unlock() }
        return fired
    }
}
