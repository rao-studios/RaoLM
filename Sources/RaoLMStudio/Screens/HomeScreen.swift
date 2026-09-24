//
//  HomeScreen.swift
//  RaoLMStudio
//
//  WHAT: 1 Home: every run under the data root, and the selected run's model, corpus,
//        Thread node and evaluation at a glance.
//

import Foundation
import RaoLM
import RaoLMTerminal
import RaoLMWorkflows

enum HomeScreen: StudioScreen {
    static func visibleRows(_ state: StudioState) -> Int { max(3, state.size.height - 26) }

    static func render(_ state: StudioState, _ frame: inout Frame, _ rect: Rect) {
        let palette = frame.palette
        let glyphs = frame.glyphs
        let detailHeight = rect.height >= 22 ? 8 : 0
        let (detailRect, listRect) = rect.bottom(detailHeight)
        let inner = Theme.panel("Runs · \(state.home.runs.count)", focused: true, &frame, listRect)
        if state.home.runs.isEmpty {
            Theme.empty("No runs under \(StudioApp.abbreviate(state.root.runs.path)) yet. Generate a corpus on 3 Corpus, take an offline snapshot (O), then train one on 5 Train — or run `raolm demo`.", &frame, inner)
        } else {
            let wide = inner.width >= 100
            var columns = [Column("run id", .flex(1)), Column("preset", .fixed(6)), Column("status", .fixed(10)),
                           Column("epochs", .fixed(6), align: .trailing), Column("mem", .fixed(4), align: .trailing)]
            if wide { columns += [Column("indexed", .fixed(9)), Column("corpus", .fixed(20)), Column("node", .fixed(8))] }
            columns += [Column("eval", .fixed(4)), Column("gens", .fixed(4), align: .trailing)]
            let rows: [[Text]] = state.home.runs.map { run in
                let stale = run.isStale(now: state.now)
                var row: [Text] = [
                    Text(run.id), Text(run.preset),
                    Text("\(Theme.statusGlyph(run.status, glyphs: glyphs)) \(stale ? "stale" : run.status.rawValue)",
                         style: stale ? palette.muted : Theme.statusStyle(run.status, palette: palette)),
                    Text("\(run.epochsDone)/\(run.epochsPlanned)"),
                    Text(Format.pct(run.lastMemorised)),
                ]
                if wide {
                    row += [
                        run.indexedEpochs.isEmpty ? Text("\(glyphs.idle) —", style: palette.muted)
                            : Text("\(glyphs.check) \(run.indexedEpochs.map(String.init).joined(separator: ","))", style: palette.green),
                        Text("\(run.corpusSlug ?? "?") \(run.corpusHash.prefix(8))", style: palette.dim),
                        Text(run.threadNodeID.map { String($0.prefix(8)).lowercased() } ?? "offline", style: palette.dim),
                    ]
                }
                row += [run.eval == nil ? Text(glyphs.dash, style: palette.muted) : Text(glyphs.check, style: palette.green),
                        Text(String(run.generationCount))]
                return row
            }
            Table.styled(columns: columns, rows: rows, state: state.home.table, palette: palette, glyphs: glyphs)
                .render(in: inner, on: &frame.canvas)
        }
        guard detailHeight > 0 else { return }
        let halves = detailRect.splitHorizontally([.flex(1), .flex(1)])
        guard let run = state.selectedRun else {
            Theme.panel("Run", &frame, halves[0])
            Theme.panel("Corpus", &frame, halves[1])
            return
        }
        let left = Theme.panel(run.id, &frame, halves[0])
        var evalText = Text("not evaluated — e to evaluate", style: palette.dim)
        if let e = run.eval {
            evalText = Text("λ \(Format.f(e.lambda, 2)) · exact \(Format.pct(e.exact)) · citation@1 \(Format.pct(e.citationAt1))"
                            + (e.auroc.map { String(format: " · AUROC %.3f", $0) } ?? ""), style: palette.text)
        }
        Theme.keyValues([
            ("model", Text("\(run.preset) · \(Format.count(run.parameterCount)) parameters · tap layer \(run.tapLayer)")),
            ("training", Text("epoch \(run.epochsDone)/\(run.epochsPlanned) · eval loss \(Format.f(run.lastEvalLoss)) nat")),
            ("memorised", Text("\(Format.pct(run.lastMemorised)) of corpus positions")),
            ("index", run.indexedEpochs.isEmpty ? Text("none yet", style: palette.muted)
                : Text("epochs \(run.indexedEpochs.map(String.init).joined(separator: ", "))", style: palette.green)),
            ("eval", evalText),
            ("status", run.failure.map { Text($0, style: palette.red) } ?? Text(run.status.rawValue, style: Theme.statusStyle(run.status, palette: palette))),
        ], &frame, left)
        let right = Theme.panel("Corpus", &frame, halves[1])
        let thread = state.status.thread
        Theme.keyValues([
            ("documents", Text("\(Format.count(run.documents))    partitions \(Format.count(run.partitions))")),
            ("tokens", Text(Format.count(run.tokens))),
            ("hash", Text(Format.short(run.corpusHash), style: palette.dim)),
            ("slug", Text(run.corpusSlug ?? "—")),
            ("node", Text(run.threadNodeID ?? "offline snapshot", style: run.threadNodeID == nil ? palette.muted : palette.dim)),
            ("thread", thread.map { t in
                t.isUp ? Text("\(glyphs.live) http :\(t.endpoint.httpPort) · grpc :\(t.endpoint.grpcPort)", style: palette.green)
                    : Text("\(glyphs.idle) not running", style: palette.dim)
            } ?? Text("—", style: palette.muted)),
        ], &frame, right)
    }

    static func handle(_ key: KeyEvent, _ state: inout StudioState) -> [StudioJob]? {
        let count = state.home.runs.count
        let visible = visibleRows(state)
        switch key.key {
        case .up, .char("k"): state.home.table.move(by: -1, rowCount: count, visible: visible)
        case .down, .char("j"): state.home.table.move(by: 1, rowCount: count, visible: visible)
        case .pageUp: state.home.table.page(by: -1, rowCount: count, visible: visible)
        case .pageDown: state.home.table.page(by: 1, rowCount: count, visible: visible)
        case .home: state.home.table.home(rowCount: count, visible: visible)
        case .end: state.home.table.end(rowCount: count, visible: visible)
        case .char("r"): return [.scanRuns]
        case .enter, .char("g"):
            guard let run = state.selectedRun else { return [] }
            guard !run.indexedEpochs.isEmpty else {
                state.status.error = RaoLMFailure("\(run.id) has no provenance index yet", hint: "train it until an indexed epoch", code: 66)
                return []
            }
            var jobs = GenerateScreen.select(run: run.directory, epoch: nil, state: &state)
            jobs += StudioApp.go(to: .generate, state: &state)
            return jobs
        case .char("e"):
            guard let run = state.selectedRun else { return [] }
            state.eval.run = run.directory
            return StudioApp.go(to: .eval, state: &state)
        case .char("l"):
            guard let run = state.selectedRun else { return [] }
            state.ledger.run = run.directory
            var jobs = StudioApp.go(to: .ledger, state: &state)
            if let job = LedgerScreen.job(state) { jobs.append(job) }
            return jobs
        case .char("t"):
            return StudioApp.go(to: .train, state: &state)
        default:
            return nil
        }
        return []
    }

    static func hints(_ state: StudioState) -> [KeyHint] {
        [KeyHint("⏎", "generate"), KeyHint("e", "eval"), KeyHint("l", "ledger"), KeyHint("t", "train"), KeyHint("r", "rescan")]
    }
}
