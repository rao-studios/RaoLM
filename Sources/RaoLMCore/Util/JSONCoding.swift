//
//  JSONCoding.swift
//  RaoLMCore
//
//  WHAT: The one JSON encoder and decoder RaoLM writes manifests, snapshots, ledgers
//        and generations with.
//  PIN:  Sorted keys and ISO-8601 dates, so the same value always encodes to the same
//        bytes and a manifest's hash means something.
//

import Foundation

public enum JSONCoding {

    /// Pretty, sorted, stable. For files people read (manifests, snapshots, generations).
    public static func prettyEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        encoder.nonConformingFloatEncodingStrategy = .convertToString(
            positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
        return encoder
    }

    /// Compact, sorted, stable. For JSONL rows.
    public static func lineEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        encoder.nonConformingFloatEncodingStrategy = .convertToString(
            positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
        return encoder
    }

    public static func decoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        decoder.nonConformingFloatDecodingStrategy = .convertFromString(
            positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
        return decoder
    }

    public static func write<T: Encodable>(_ value: T, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let data = try prettyEncoder().encode(value)
        try data.write(to: url, options: .atomic)
    }

    public static func read<T: Decodable>(_ type: T.Type, from url: URL) throws -> T {
        try decoder().decode(type, from: Data(contentsOf: url))
    }

    /// Every non-empty line of a JSONL file, decoded.
    public static func readLines<T: Decodable>(_ type: T.Type, from url: URL) throws -> [T] {
        let text = try String(contentsOf: url, encoding: .utf8)
        let decoder = decoder()
        return try text.split(separator: "\n").map { line in
            try decoder.decode(type, from: Data(line.utf8))
        }
    }
}

/// An append-only JSONL file. One encodable value per line.
public final class JSONLWriter: @unchecked Sendable {
    public let url: URL
    private let handle: FileHandle
    private let encoder = JSONCoding.lineEncoder()
    private let lock = NSLock()

    public init(url: URL, truncate: Bool = false) throws {
        self.url = url
        let manager = FileManager.default
        try manager.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        if truncate || !manager.fileExists(atPath: url.path) {
            manager.createFile(atPath: url.path, contents: nil)
        }
        self.handle = try FileHandle(forWritingTo: url)
        try handle.seekToEnd()
    }

    deinit {
        try? handle.close()
    }

    public func append<T: Encodable>(_ value: T) throws {
        var data = try encoder.encode(value)
        data.append(0x0A)
        lock.lock()
        defer { lock.unlock() }
        try handle.write(contentsOf: data)
    }

    public func close() {
        lock.lock()
        defer { lock.unlock() }
        try? handle.synchronize()
        try? handle.close()
    }
}
