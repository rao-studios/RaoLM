//
//  Failure.swift
//  RaoLMWorkflows
//
//  WHAT: A failure with a message, an optional fix, and a sysexits-style code — and the one
//        mapping from every RaoLM error type to it, shared by the CLI's `guarded` and the
//        studio's status bar.
//  PIN:  Codes: 64 usage, 65 data, 66 no input, 69 unavailable, 70 software, 73 can't create,
//        75 temporary, 78 configuration; 3 means verification found a mismatch; 130 interrupted.
//

import Foundation
import RaoLM

public struct RaoLMFailure: Error, CustomStringConvertible, Sendable, Equatable {
    public var message: String
    public var hint: String?
    public var code: Int32

    public init(_ message: String, hint: String? = nil, code: Int32 = 70) {
        self.message = message
        self.hint = hint
        self.code = code
    }

    public var description: String { message }
}

public enum FailureMapping {
    /// The failure to report for `error`, with RaoLM's exit code for its type.
    public static func describe(_ error: Error) -> RaoLMFailure {
        switch error {
        case let failure as RaoLMFailure:
            return failure
        case let error as ThreadCorpusError:
            switch error {
            case .indexTimeout: return RaoLMFailure(error.description, code: 75)
            case .indexRejected, .exportIncomplete, .partitionAddressMismatch: return RaoLMFailure(error.description, code: 65)
            default: return RaoLMFailure(error.description, code: 69)
            }
        case let error as RaoTokenizerError:
            return RaoLMFailure(error.description, code: 78)
        case let error as RunManifestError:
            return RaoLMFailure(error.description, code: 66)
        case let error as CorpusStoreError:
            return RaoLMFailure(error.description, hint: "generate one with: raolm corpus generate", code: 66)
        case let error as ProvenanceError:
            return RaoLMFailure(error.description, code: 65)
        case let error as SyntheticCorpusError:
            return RaoLMFailure(error.description, code: 65)
        case let error as RaoLMConfigError:
            return RaoLMFailure(error.description, code: 64)
        case let error as CheckpointError:
            return RaoLMFailure(error.description, code: 66)
        case let error as GroundingError:
            return RaoLMFailure(error.description, hint: error.hint, code: error.exitCode)
        case is CancellationError:
            return RaoLMFailure("cancelled", code: 130)
        default:
            if let mapped = extraMappings(error) { return mapped }
            return RaoLMFailure("\(error)", code: 70)
        }
    }

    nonisolated(unsafe) private static var extras: [(Error) -> RaoLMFailure?] = []

    /// Lets a target this one cannot see (the CLI, the grounding harness) teach the mapping its
    /// own error types. Register once at start-up, before any failure is described.
    public static func register(_ mapping: @escaping (Error) -> RaoLMFailure?) { extras.append(mapping) }

    private static func extraMappings(_ error: Error) -> RaoLMFailure? {
        for mapping in extras {
            if let failure = mapping(error) { return failure }
        }
        return nil
    }
}
