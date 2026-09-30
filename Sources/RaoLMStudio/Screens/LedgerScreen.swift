//
//  LedgerScreen.swift
//  RaoLMStudio
//
//  WHAT: 8 Ledger: the entropy ledger — per-epoch summaries, one document's partitions
//        across epochs, or one fact's answer-token losses — with the columns `raolm ledger`
//        prints.
//

import Foundation
import RaoLM
import RaoLMTerminal
import RaoLMWorkflows

enum LedgerScreen: StudioScreen {
    static func job(_ state: StudioState) -> StudioJob? {
        guard let run = state.ledger.run else { return nil }
        let partition = Int(state.ledger.partition.text.trimmingCharacters(in: .whitespaces))
        return .ledger(run: run, mode: state.ledger.mode, filter: state.ledger.filter.text.trimmingCharacters(in: .whitespaces), partition: partition)
    }

    static func render(_ state: StudioState, _ frame: inout Frame, _ rect: Rect) {
        let palette = frame.palette
        let (controls, tableRect) = rect.top(3)
        let inner = Theme.panel("Ledger" + (state.ledger.run.map { " · " + $0.lastPathComponent } ?? ""), focused: state.ledger.editing != nil, &frame, controls)
        var x = inner.minX
        for mode in LedgerMode.allCases {
            let chosen = mode == state.ledger.mode
            x += frame.canvas.put(" \(mode.title) ", x: x, y: inner.minY, style: chosen ? palette.blue.bold().reverse() : palette.dim, clip: inner) + 1
        }
        if state.ledger.mode != .epochs {
            x += 2
            let label = state.ledger.mode == .partitions ? "document" : "fact id"
            x += frame.canvas.put(label + " ", x: x, y: inner.minY, style: palette.dim, clip: inner)
            let field = Rect(x: x, y: inner.minY, width: min(40, inner.maxX - x), height: 1)
            TextField(state: state.ledger.filter, placeholder: "/ to set (suffix match for facts)", focused: state.ledger.editing == 0,
                      style: palette.title, placeholderStyle: palette.muted, cursorStyle: palette.cursor).render(in: field, on: &frame.canvas)
            x = field.maxX + 2
            if state.ledger.mode == .partitions {
                x += frame.canvas.put("partition ", x: x, y: inner.minY, style: palette.dim, clip: inner)
                TextField(state: state.ledger.partition, placeholder: "all", focused: state.ledger.editing == 1,
                          style: palette.title, placeholderStyle: palette.muted, cursorStyle: palette.cursor)
                    .render(in: Rect(x: x, y: inner.minY, width: 6, height: 1), on: &frame.canvas)
            }
        }
        guard let data = state.ledger.data, state.ledger.run != nil else {
            let body = Theme.panel("Rows", &frame, tableRect)
            Theme.empty(state.ledger.run == nil ? "r picks a run, or select one on 1 Home and press l." : "⏎ loads the ledger.", &frame, body)
            return
        }
        let body = Theme.panel(data.header, footer: Text("\(data.table.rows.count) rows", style: palette.dim), &frame, tableRect)
        if data.table.rows.isEmpty {
            Theme.empty(data.mode == .epochs ? "no epochs recorded yet" : "no rows match — documents are raolm-<slug>-<hash>, facts match by suffix", &frame, body)
            return
        }
        Theme.textTable(data.table, state: state.ledger.table, cellStyle: { _, _, cell in
            cell == "yes" ? palette.green : (cell == "no" ? palette.dim : nil)
        }, &frame, body)
    }

    static func handle(_ key: KeyEvent, _ state: inout StudioState) -> [StudioJob]? {
        if let editing = state.ledger.editing {
            switch key.key {
            case .enter:
                state.ledger.editing = nil
                return [job(state)].compactMap { $0 }
            case .escape:
                state.ledger.editing = nil
            case .tab where state.ledger.mode == .partitions:
                state.ledger.editing = editing == 0 ? 1 : 0
            default:
                if editing == 0 { state.ledger.filter.handle(key) } else { state.ledger.partition.handle(key) }
            }
            return []
        }
        let rows = state.ledger.data?.table.rows.count ?? 0
        let visible = max(3, state.size.height - 12)
        switch key.key {
        case .char("m"), .right, .left:
            let all = LedgerMode.allCases
            let delta = key.key == .left ? all.count - 1 : 1
            state.ledger.mode = all[(all.firstIndex(of: state.ledger.mode)! + delta) % all.count]
            state.ledger.data = nil
            return state.ledger.mode == .epochs ? [job(state)].compactMap { $0 } : []
        case .char("/"):
            if state.ledger.mode != .epochs { state.ledger.editing = 0 }
        case .char("p") where state.ledger.mode == .partitions:
            state.ledger.editing = 1
        case .enter:
            return [job(state)].compactMap { $0 }
        case .char("r"):
            let runs = state.home.runs
            state.overlay = .picker(Picker(title: "Run ledger", items: runs.map { "\($0.id)  \($0.preset)  \($0.epochsDone) epochs" },
                                           values: runs.map(\.directory), table: TableState(selected: 0), purpose: .run(.ledger)))
        case .up, .char("k"): state.ledger.table.move(by: -1, rowCount: rows, visible: visible)
        case .down, .char("j"): state.ledger.table.move(by: 1, rowCount: rows, visible: visible)
        case .pageUp: state.ledger.table.page(by: -1, rowCount: rows, visible: visible)
        case .pageDown: state.ledger.table.page(by: 1, rowCount: rows, visible: visible)
        case .char("g"):
            guard let run = state.ledger.run else { return [] }
            var jobs = GenerateScreen.select(run: run, epoch: nil, state: &state)
            jobs += StudioApp.go(to: .generate, state: &state)
            return jobs
        default:
            return nil
        }
        return []
    }

    static func hints(_ state: StudioState) -> [KeyHint] {
        if state.ledger.editing != nil { return [KeyHint("⏎", "load"), KeyHint("esc", "done")] }
        return [KeyHint("m", "mode"), KeyHint("/", "filter"), KeyHint("⏎", "load"), KeyHint("r", "run"), KeyHint("g", "generate")]
    }
}
