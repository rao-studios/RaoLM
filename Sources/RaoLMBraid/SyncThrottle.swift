//
//  SyncThrottle.swift
//  RaoLMBraid
//
//  WHAT: At most `atOnce` nodes training at a time. A braid of many Threads cannot train every
//        node at once (each training node of the base preset holds several GB); the throttle
//        sends a sync to the next waiting node when one in flight settles.
//  PIN:  Pure: the session feeds it node states and sends what it returns. A node is in flight
//        from its sync until it is idle with nothing on its ladder and has a new version, is
//        held or has failed, or has been seen busy and gone idle (a sync that found nothing new).
//

import Darwin
import Foundation

public struct SyncThrottle: Sendable, Equatable {
    public struct Flight: Sendable, Equatable {
        public var versions: Int
        public var sawBusy: Bool
    }

    /// Nodes training at once; 0 or less: all of them.
    public let atOnce: Int
    public private(set) var pending: [String]
    public private(set) var inFlight: [String: Flight] = [:]
    public private(set) var finished: [String] = []

    public init(names: [String], atOnce: Int) {
        self.atOnce = atOnce
        pending = names
    }

    public var isDone: Bool { pending.isEmpty && inFlight.isEmpty }

    public static func busy(_ state: StrandState) -> Bool {
        state.stage.isBusy || state.ladder.contains { $0.status == .pending || $0.status == .running }
    }

    /// Whether a node sent a sync when it had `versions` versions is done with it.
    public static func settled(_ state: StrandState, after versions: Int, sawBusy: Bool = false) -> Bool {
        guard !busy(state) else { return false }
        return state.versions > versions || state.stage == .held || state.stage == .failed || sawBusy
    }

    /// Retires the nodes in flight that settled, then returns the nodes to sync now (marking
    /// them in flight at their current version count).
    public mutating func advance(states: [String: StrandState]) -> [String] {
        for name in inFlight.keys.sorted() {
            guard let state = states[name], let flight = inFlight[name] else { continue }
            if Self.settled(state, after: flight.versions, sawBusy: flight.sawBusy) {
                inFlight[name] = nil
                finished.append(name)
            } else if Self.busy(state) {
                inFlight[name]?.sawBusy = true
            }
        }
        var send: [String] = []
        let limit = atOnce <= 0 ? Int.max : atOnce
        while inFlight.count < limit, !pending.isEmpty {
            let name = pending.removeFirst()
            inFlight[name] = Flight(versions: states[name]?.versions ?? 0, sawBusy: false)
            send.append(name)
        }
        return send
    }
}

extension BraidSession {
    /// Syncs `names`, at most `atOnce` at a time, and returns once every one has settled. Throws
    /// when no node in flight has moved for `stall` seconds, or when `cancelled` says so.
    public func sync(
        _ names: [String], atOnce: Int, stall: TimeInterval = 600, cancelled: @escaping () -> Bool = { false },
        sent: ((_ node: String, _ number: Int, _ inFlight: Int) -> Void)? = nil, progress: ((StrandState) -> Void)? = nil
    ) async throws {
        var throttle = SyncThrottle(names: names, atOnce: atOnce)
        var seen: [String: String] = [:]
        var moved = Date()
        var number = 0
        while true {
            if cancelled() { throw CancellationError() }
            var states: [String: StrandState] = [:]
            for name in names { states[name] = state(name) }
            for name in throttle.advance(states: states) {
                try sync(name)
                number += 1
                moved = Date()
                sent?(name, number, throttle.inFlight.count)
            }
            if throttle.isDone { return }
            for name in throttle.inFlight.keys.sorted() {
                guard let state = states[name] else { continue }
                let line = "\(state.stage.rawValue) \(state.epoch ?? -1) \(state.step ?? -1) \(state.versions) \(state.candidateMemorised ?? -1)"
                if seen[name] != line {
                    seen[name] = line
                    moved = Date()
                    progress?(state)
                }
            }
            if Date().timeIntervalSince(moved) > stall {
                throw BraidSessionError.io("no node in flight made progress for \(Int(stall)) s (\(throttle.inFlight.keys.sorted().joined(separator: ", ")))")
            }
            try await Task.sleep(nanoseconds: 500_000_000)
        }
    }

    /// The most physical memory a process has held over its life, in bytes (nil when it is gone).
    public static func peakFootprint(pid: Int32) -> Int? {
        var info = rusage_info_v4()
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(pid, RUSAGE_INFO_V4, $0) }
        }
        return result == 0 ? Int(info.ri_lifetime_max_phys_footprint) : nil
    }
}
