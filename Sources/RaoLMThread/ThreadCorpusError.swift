//
//  ThreadCorpusError.swift
//  RaoLMThread
//
//  WHAT: Everything that can go wrong talking to or hosting a Thread, each with the fix.
//

import Foundation

public enum ThreadCorpusError: Error, CustomStringConvertible {
    case binaryNotFound(searched: [String])
    case metallibMissing(binary: String)
    case portBusy(port: Int, what: String)
    case exitedDuringStartup(status: Int32, logTail: String)
    case startupTimeout(seconds: Double, logTail: String)
    case notOpenMode(stack: String)
    case notReachable(endpoint: String, underlying: String)
    case indexRejected(message: String)
    case indexTimeout(missing: [String])
    case exportIncomplete(expected: Int, got: Int)
    case partitionAddressMismatch(documentID: String, index: Int, expected: String, got: String)

    public var description: String {
        switch self {
        case .binaryNotFound(let searched):
            return "no Thread binary found (searched: \(searched.joined(separator: ", "))). Build it with: cd ../Thread && swift build -c release && ./build-metallib.sh release — or pass --thread-binary / set RAOLM_THREAD_BINARY"
        case .metallibMissing(let binary):
            return "no mlx.metallib beside \(binary); without it Thread's --use-mlx silently falls back to the Mistral API. Fix: cd ../Thread && ./build-metallib.sh release"
        case .portBusy(let port, let what):
            return "port \(port) (\(what)) is already in use; stop that process (raolm thread stop) or pass a different --http-port/--grpc-port"
        case .exitedDuringStartup(let status, let tail):
            return "the Thread exited during startup (status \(status))" + (tail.isEmpty ? "" : ":\n\(tail)")
        case .startupTimeout(let seconds, let tail):
            return "the Thread did not become healthy within \(Int(seconds)) s" + (tail.isEmpty ? "" : ":\n\(tail)")
        case .notOpenMode(let stack):
            return "the Thread on this port runs in stack mode '\(stack)', not open mode — it is not the Thread RaoLM started"
        case .notReachable(let endpoint, let underlying):
            return "cannot reach the Thread at \(endpoint): \(underlying). Start one with: raolm thread start"
        case .indexRejected(let message):
            return "the Thread rejected an Index request: \(message)"
        case .indexTimeout(let missing):
            return "timed out waiting for \(missing.count) document(s) to finish indexing (first: \(missing.prefix(3).joined(separator: ", ")))"
        case .exportIncomplete(let expected, let got):
            return "ExportCorpus returned \(got) of \(expected) expected documents"
        case .partitionAddressMismatch(let documentID, let index, let expected, let got):
            return "\(documentID) partition \(index): Thread returned url '\(got)', expected '\(expected)'"
        }
    }
}
