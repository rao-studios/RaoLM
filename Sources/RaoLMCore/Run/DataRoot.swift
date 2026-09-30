//
//  DataRoot.swift
//  RaoLMCore
//
//  WHAT: Where RaoLM keeps everything it produces.
//  IN:   --data-dir, then RAOLM_DATA_DIR, then ~/Documents/raolm-db.
//  OUT:  thread-db/ (the demo Thread's storage), corpora/<slug>/, snapshots/<hash12>/,
//        runs/<run id>/, logs/.
//  PIN:  Never ~/Documents/thread-db: that is the default of a real Thread node, and the
//        demo wipes its own Thread storage on every run.
//

import Foundation

public struct DataRoot: Sendable, Equatable {
    public static let environmentKey = "RAOLM_DATA_DIR"

    public let url: URL

    public init(url: URL) {
        self.url = url.standardizedFileURL
    }

    public static func resolve(
        argument: String?, environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> DataRoot {
        if let argument, !argument.isEmpty {
            return DataRoot(url: URL(fileURLWithPath: expandTilde(argument), isDirectory: true))
        }
        if let value = environment[environmentKey], !value.isEmpty {
            return DataRoot(url: URL(fileURLWithPath: expandTilde(value), isDirectory: true))
        }
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Documents")
        return DataRoot(url: documents.appendingPathComponent("raolm-db", isDirectory: true))
    }

    static func expandTilde(_ path: String) -> String {
        (path as NSString).expandingTildeInPath
    }

    public var threadDB: URL { url.appendingPathComponent("thread-db", isDirectory: true) }
    public var logs: URL { url.appendingPathComponent("logs", isDirectory: true) }
    public var runs: URL { url.appendingPathComponent("runs", isDirectory: true) }

    public func corpus(slug: String) -> URL {
        url.appendingPathComponent("corpora", isDirectory: true).appendingPathComponent(slug, isDirectory: true)
    }

    public func snapshot(hash: String) -> URL {
        url.appendingPathComponent("snapshots", isDirectory: true)
            .appendingPathComponent(String(hash.prefix(12)), isDirectory: true)
    }

    public func run(id: String) -> URL {
        runs.appendingPathComponent(id, isDirectory: true)
    }

    /// `20260924-091500-tiny`.
    public static func newRunID(preset: String, now: Date = Date()) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone.current
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return "\(formatter.string(from: now))-\(preset)"
    }

    public func ensure() throws {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }
}
