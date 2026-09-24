//
//  ThreadScreen.swift
//  RaoLMStudio
//
//  WHAT: 4 Thread: the Thread node RaoLM hosts — its identity, health, what it stores, its
//        ports, and the tail of its log — with start and stop.
//

import Foundation
import RaoLM
import RaoLMTerminal
import RaoLMWorkflows

enum ThreadScreen: StudioScreen {
    static func render(_ state: StudioState, _ frame: inout Frame, _ rect: Rect) {
        let palette = frame.palette
        let glyphs = frame.glyphs
        let (top, logRect) = rect.top(min(10, rect.height / 2))
        let halves = top.splitHorizontally([.flex(1), .flex(1)])
        let status = state.status.thread
        let up = status?.isUp == true
        let badge = up ? Text("\(glyphs.live) healthy", style: palette.green) : Text("\(glyphs.idle) down", style: palette.dim)
        let node = Theme.panel("Node", footer: badge, &frame, halves[0])
        guard let status else {
            Theme.empty("checking the Thread…", &frame, node)
            return
        }
        Theme.keyValues([
            ("node id", Text(status.nodeID ?? "unknown", style: status.nodeID == nil ? palette.muted : palette.text)),
            ("endpoint", Text("\(status.endpoint.host)  http :\(status.endpoint.httpPort)  grpc :\(status.endpoint.grpcPort)")),
            ("process", status.record.map { Text("pid \($0.pid) · started \(Self.time($0.startedAt))" + (status.ownedByStudio ? " · by the studio" : "")) }
                ?? Text("not hosted by raolm", style: palette.muted)),
            ("binary", Text(status.record?.binary ?? "—", style: palette.dim)),
            ("storage", Text(StudioApp.abbreviate(state.root.threadDB.path), style: palette.dim)),
            ("ports", Text("http \(status.httpBusy ? "in use" : "free") · grpc \(status.grpcBusy ? "in use" : "free")",
                           style: up || !(status.httpBusy || status.grpcBusy) ? palette.dim : palette.orange)),
        ], &frame, node, labelWidth: 10)

        let health = Theme.panel("Health", &frame, halves[1])
        var rows: [(String, Text)] = []
        if let h = status.health {
            rows.append(("status", Text(h.status, style: h.status == "healthy" ? palette.green : palette.orange)))
            rows.append(("stack", Text(h.stack ?? "—") + Text(h.stack == nil || h.stack == "open" ? "  (open mode, on-device embedding)" : "", style: palette.dim)))
            rows.append(("contract", Text(h.contract.map(String.init) ?? "—")))
        } else {
            rows.append(("status", Text(status.error ?? "not reachable", style: palette.dim)))
        }
        if let s = status.stats {
            rows.append(("documents", Text(Format.count(s.documents))))
            rows.append(("groups", Text("\(s.groups)    owners \(s.owners)")))
        }
        rows.append(("polled", Text(Self.time(status.polledAt), style: palette.dim)))
        Theme.keyValues(rows, &frame, health, labelWidth: 11)

        let logFooter = Text(status.record.map { StudioApp.abbreviate($0.logFile) } ?? "", style: palette.dim)
        let inner = Theme.panel("Log", footer: logFooter, &frame, logRect)
        if state.thread.log.lines.isEmpty {
            Theme.empty(up ? "no log lines yet" : "s starts a Thread in open mode on \(StudioApp.abbreviate(state.root.threadDB.path))" + (state.thread.fresh ? " after wiping it (fresh)" : ""), &frame, inner)
        } else {
            LogPane(state: state.thread.log, style: palette.dim).render(in: inner, on: &frame.canvas)
        }
    }

    static func time(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        return formatter.string(from: date)
    }

    static func handle(_ key: KeyEvent, _ state: inout StudioState) -> [StudioJob]? {
        switch key.key {
        case .char("s"):
            if state.status.thread?.isUp == true {
                state.status.message = "a Thread is already running"
                return []
            }
            return [.threadStart(fresh: state.thread.fresh)]
        case .char("x"): return [.threadStop]
        case .char("r"): return [.threadStatus(withLog: true)]
        case .char("f"):
            state.thread.fresh.toggle()
            state.status.message = state.thread.fresh ? "next start wipes the Thread's storage" : "next start keeps the Thread's storage"
        case .up, .char("k"): state.thread.log.scroll(by: 1, visible: 8)
        case .down, .char("j"): state.thread.log.scroll(by: -1, visible: 8)
        default: return nil
        }
        return []
    }

    static func hints(_ state: StudioState) -> [KeyHint] {
        [KeyHint("s", "start"), KeyHint("x", "stop"), KeyHint("f", state.thread.fresh ? "fresh: on" : "fresh: off"), KeyHint("r", "refresh")]
    }
}
