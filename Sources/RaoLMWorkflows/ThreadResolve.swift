//
//  ThreadResolve.swift
//  RaoLMWorkflows
//
//  WHAT: Which Thread a command talks to: the one raolm hosts (its host record under the data
//        root) before the host/ports given on the command line, and which node id it has.
//

import Foundation
import RaoLM

public enum ThreadResolve {
    public static func endpoint(root: DataRoot, fallback: ThreadEndpoint) -> ThreadEndpoint {
        ThreadHostRecord.load(dataDirectory: root.threadDB)?.endpoint ?? fallback
    }

    /// The explicit id, else the hosted Thread's record, else its `node-id` file.
    public static func nodeID(explicit: String?, root: DataRoot) -> UUID? {
        explicit.flatMap(UUID.init)
            ?? ThreadHostRecord.load(dataDirectory: root.threadDB)?.endpoint.nodeID
            ?? ThreadHost.readNodeID(dataDirectory: root.threadDB)
    }
}
