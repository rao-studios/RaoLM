//
//  NodeProtocol.swift
//  RaoLMBraid
//
//  WHAT: How the umbrella and a node process talk: one JSON object per line over the node's
//        standard input and output. The umbrella sends numbered requests; the node answers
//        each by number and also sends events (its state, log lines, promotions) at any time.
//  PIN:  Only retrieval hits, the node's trajectory through its corpus (a few numbers a
//        position) and hidden states cross for generation — never weights, logits or corpus
//        text. Float arrays travel as base64 of their little-endian bytes, exact.
//

import Foundation
import RaoLMCore

public struct NodeRequest: Codable, Sendable {
    public var id: Int
    public var op: Op

    public enum Op: Codable, Sendable {
        case hello
        case describe
        /// Run the update ladder now.
        case sync
        /// The standing prompt a training candidate completes at each evaluation.
        case probe(tokens: [Int])
        case open(session: String, tokens: [Int], k: Int)
        case advance(session: String, token: Int, k: Int)
        case hidden(session: String, positions: [Int])
        case close(session: String)
        /// Stop the running update at its next step.
        case cancel
        case shutdown

        /// Answered from the live version between training steps rather than queued behind an update.
        public var isServing: Bool {
            switch self {
            case .describe, .probe, .open, .advance, .hidden, .close: return true
            default: return false
            }
        }
    }

    public init(id: Int, op: Op) {
        self.id = id
        self.op = op
    }
}

public struct NodeHello: Codable, Sendable, Equatable {
    public var name: String
    public var label: String
    public var pid: Int32
    public var threadPID: Int32?
    public var threadID: String?
    public var httpPort: Int?
    public var grpcPort: Int?
    public var offline: Bool
    public var liveVersion: Int?
    public var vocabularySHA256: String
    public var state: StrandState
}

public enum NodeReply: Codable, Sendable {
    case hello(NodeHello)
    case described(StrandDescriptor?)
    case accepted
    case opened([StrandStep])
    case hits(StrandStep)
    case hidden([PackedFloats])
    case ok
}

public enum NodeOutput: Codable, Sendable {
    case reply(id: Int, reply: NodeReply)
    case failure(id: Int, message: String, code: Int32)
    case event(NodeEvent)
}

/// Encodes and decodes one message per line.
public enum NodeWire {
    public static func encode<T: Encodable>(_ value: T) throws -> Data {
        var data = try JSONCoding.lineEncoder().encode(value)
        data.append(0x0A)
        return data
    }

    public static func decode<T: Decodable>(_ type: T.Type, line: Data) throws -> T {
        try JSONCoding.decoder().decode(type, from: line)
    }
}

/// Splits a byte stream into lines, keeping partial lines until their newline arrives.
public final class LineSplitter: @unchecked Sendable {
    private let lock = NSLock()
    private var pending = Data()

    public init() {}

    public func feed(_ data: Data) -> [Data] {
        lock.lock()
        defer { lock.unlock() }
        pending.append(data)
        var lines: [Data] = []
        while let newline = pending.firstIndex(of: 0x0A) {
            let line = pending[pending.startIndex..<newline]
            if !line.isEmpty { lines.append(Data(line)) }
            pending.removeSubrange(pending.startIndex...newline)
        }
        return lines
    }
}
