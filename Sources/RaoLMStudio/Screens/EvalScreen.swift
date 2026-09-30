//
//  EvalScreen.swift
//  RaoLMStudio
//
//  WHAT: 7 Eval: the citation evaluation protocol — the λ ablation, calibration of citation
//        confidence (ECE, AUROC), the paraphrase / fabricated-entity / leave-out controls, span
//        verification — and, with grounding on, the two-fold harness: how grounded, drifting and
//        at risk of hallucination each group's answers are against their true source.
//

import Foundation
import RaoLM
import RaoLMTerminal
import RaoLMWorkflows

enum EvalScreen: StudioScreen {
    static func render(_ state: StudioState, _ frame: inout Frame, _ rect: Rect) {
        let palette = frame.palette
        let glyphs = frame.glyphs
        let columns = rect.splitHorizontally([.fixed(min(40, rect.width / 3)), .flex(1)])
        let formHeight = state.eval.form.fields.count + 4
        let (formRect, progressRect) = columns[0].top(formHeight)
        let formFocused = !state.eval.focusOutcomes
        let run = state.run(at: state.eval.run)
        let form = Theme.panel("Evaluate", focused: formFocused, footer: Text(run?.id ?? "no run — r", style: run == nil ? palette.orange : palette.dim), &frame, formRect)
        if let run, !run.hasFacts {
            frame.canvas.put("this run has no facts.jsonl", x: form.minX, y: form.minY, style: palette.orange, clip: form)
        }
        state.eval.form.render(in: form.inset(top: 1), focused: formFocused, frame: &frame, labelWidth: 15)
        let progress = Theme.panel("Progress", &frame, progressRect)
        if state.eval.progress.lines.isEmpty {
            Theme.empty("Each fact's prompt is the corpus tokens before its answer. Offline verification re-reads spans from the run's snapshot; turn it off to verify against the live Thread.", &frame, progress)
        } else {
            LogPane(state: state.eval.progress, style: palette.dim).render(in: progress, on: &frame.canvas)
        }

        guard let report = state.eval.report else {
            let inner = Theme.panel("Report", &frame, columns[1])
            var message = "⏎ on “Run evaluation”. The last report of a run loads from its eval.json."
            if let run, run.eval != nil { message = "l loads \(run.id)/eval.json · " + message }
            Theme.empty(message, &frame, inner)
            return
        }
        let right = columns[1]
        let lambdas = EvalTables.lambdas(report)
        let calibration = report.calibration.filter { $0.count > 0 }
        let (lambdaRect, afterLambda) = right.top(lambdas.rows.count + 4)
        let lambdaInner = Theme.panel("λ sweep · \(report.sampleSize) facts · epoch \(report.epoch)", &frame, lambdaRect)
        Theme.textTable(lambdas, flexColumn: 7, cellStyle: { row, _, _ in
            abs(report.lambdas[row].lambda - report.primaryLambda) < 1e-6 ? palette.title : nil
        }, &frame, lambdaInner)

        var notes: [Text] = [Text(EvalTables.calibrationLine(report), style: palette.text)]
        if let spans = EvalTables.spansLine(report) {
            notes.append(Text(spans, style: report.spansVerified == report.spanChecks ? palette.gold : palette.orange))
        }
        for line in EvalTables.controlLines(report) { notes.append(Text(line, style: palette.text)) }
        if let grounding = report.grounding {
            for line in GroundingTables.evalLines(grounding).prefix(1) { notes.append(Text(line, style: palette.dim)) }
        }
        let middleHeight = min(max(calibration.count, 4) + 2, max(6, afterLambda.height / 3))
        let (middle, afterMiddle) = afterLambda.top(middleHeight)
        let middleColumns = middle.splitHorizontally([.fixed(min(34, middle.width / 2)), .flex(1)])
        let calibrationInner = Theme.panel("Calibration: accuracy by confidence", &frame, middleColumns[0])
        for (row, bin) in calibration.suffix(calibrationInner.height).enumerated() {
            let label = Text(String(format: "%.1f–%.1f %3d", bin.lower, bin.upper, bin.count), style: palette.dim)
            let verified = bin.accuracy >= 0.999
            ProgressBar(fraction: Double(bin.accuracy), label: label,
                        trailing: Text(Format.pct(bin.accuracy).padding(toLength: 5, withPad: " ", startingAt: 0) + (verified ? glyphs.verified : " "),
                                       style: verified ? palette.gold : palette.text),
                        fillStyle: palette.heat(confidence: bin.meanConfidence, verified: false), trackStyle: palette.muted,
                        full: glyphs.barFull, empty: glyphs.barEmpty).render(in: calibrationInner.row(row), on: &frame.canvas)
        }
        let notesInner = Theme.panel("Controls and verification", &frame, middleColumns[1])
        let wrapped = notes.flatMap { Paragraph.wrap($0, width: max(1, notesInner.width)) }
        for (row, line) in wrapped.prefix(notesInner.height).enumerated() {
            frame.canvas.put(line, x: notesInner.minX, y: notesInner.minY + row, clip: notesInner)
        }

        var outcomesRect = afterMiddle
        if let grounding = report.grounding, afterMiddle.height > grounding.groups.count + 9 {
            let (groundingRect, rest) = afterMiddle.top(grounding.groups.count + 4)
            let inner = Theme.panel(GroundingTables.evalSectionTitle + " · each answer with vs without its true source", &frame, groundingRect)
            var groups = GroundingTables.evalGroups(grounding)
            for index in groups.rows.indices { groups.rows[index][0] = groups.rows[index][0].components(separatedBy: " ").first ?? "" }
            Theme.textTable(groups, flexColumn: 0, cellStyle: { _, column, cell in
                column == 7 ? ((Float(cell) ?? 0) > 0.3 ? palette.orange : palette.text) : nil
            }, &frame, inner)
            outcomesRect = rest
        }
        let inner = Theme.panel("Outcomes · \(report.outcomes.count)", focused: state.eval.focusOutcomes,
                                footer: Text("tab, then ⏎ opens the generation", style: palette.dim), &frame, outcomesRect)
        let grounded = report.grounding != nil
        let citationColumn = inner.width >= 100
        let rows: [[Text]] = report.outcomes.map { o in
            var row: [Text] = [
                Text(o.kind.rawValue), Text(o.expected.trimmingCharacters(in: .whitespaces)), Text(o.generated.trimmingCharacters(in: .whitespaces).replacingOccurrences(of: "\n", with: glyphs.newline), style: o.exact ? palette.green : palette.red),
            ]
            if citationColumn {
                row.append(Text((o.topCitationName ?? "—") + (o.topCitation.map { " p\($0.partitionIndex)@\($0.tokenOffset)" } ?? "")))
            }
            row += [
                o.answerCoveredByVerbatimSpan ? Text(glyphs.verified, style: palette.gold) : Text(glyphs.dash, style: palette.muted),
                Text(o.spansChecked.map { "\(o.spansVerified ?? 0)/\($0)" } ?? "—"),
                Text(Format.f(o.meanAnswerConfidence, 2), style: palette.heat(confidence: o.meanAnswerConfidence, verified: false)),
            ]
            if grounded {
                let cells = GroundingTables.outcomeCells(o.grounding)
                row += [Text(cells[0]), Text(cells[1], style: (Float(cells[1]) ?? 0) > 0.3 ? palette.orange : palette.text)]
            }
            return row
        }
        var outcomeColumns = [Column("fact", .fixed(13)), Column("expected", .flex(1)), Column("generated", .flex(1))]
        if citationColumn { outcomeColumns.append(Column("top citation", .flex(2))) }
        outcomeColumns += [Column("span", .fixed(4)), Column("ver", .fixed(3), align: .trailing), Column("conf", .fixed(4), align: .trailing)]
        if grounded { outcomeColumns += [Column("ι", .fixed(5), align: .trailing), Column("risk", .fixed(4), align: .trailing)] }
        Table.styled(columns: outcomeColumns, rows: rows, state: state.eval.focusOutcomes ? state.eval.outcomes : nil,
                     palette: palette, glyphs: glyphs).render(in: inner, on: &frame.canvas)
    }

    static func handle(_ key: KeyEvent, _ state: inout StudioState) -> [StudioJob]? {
        if !state.eval.form.editing {
            switch key.key {
            case .char("r"):
                let runs = state.home.runs.filter { !$0.indexedEpochs.isEmpty }
                state.overlay = .picker(Picker(title: "Run to evaluate", items: runs.map { "\($0.id)  \($0.preset)" + ($0.hasFacts ? "" : "  (no facts)") },
                                               values: runs.map(\.directory), table: TableState(selected: 0), purpose: .run(.eval)))
                return []
            case .char("l"):
                guard let run = state.eval.run else { return [] }
                let url = run.appendingPathComponent("eval.json")
                do {
                    state.eval.report = try JSONCoding.read(EvalReport.self, from: url)
                    state.eval.savedTo = url
                    state.eval.outcomes = TableState(selected: 0)
                } catch {
                    state.status.error = RaoLMFailure("no eval.json in \(run.lastPathComponent)", code: 66)
                }
                return []
            case .tab, .backTab:
                state.eval.focusOutcomes.toggle()
                return []
            case .char("c"):
                return [.cancel]
            default:
                break
            }
            if state.eval.focusOutcomes, let report = state.eval.report {
                let visible = max(3, state.size.height / 3)
                switch key.key {
                case .up, .char("k"): state.eval.outcomes.move(by: -1, rowCount: report.outcomes.count, visible: visible)
                case .down, .char("j"): state.eval.outcomes.move(by: 1, rowCount: report.outcomes.count, visible: visible)
                case .pageUp: state.eval.outcomes.page(by: -1, rowCount: report.outcomes.count, visible: visible)
                case .pageDown: state.eval.outcomes.page(by: 1, rowCount: report.outcomes.count, visible: visible)
                case .enter:
                    guard let index = state.eval.outcomes.selected, index < report.outcomes.count,
                          let file = report.outcomes[index].generationFile else {
                        state.status.message = "this outcome's generation was not saved"
                        return []
                    }
                    return [.loadGeneration(URL(fileURLWithPath: file))]
                default: return nil
                }
                return []
            }
        }
        switch state.eval.form.handle(key) {
        case .changed: return []
        case .submit: return submit(&state)
        case .unhandled: return nil
        }
    }

    static func submit(_ state: inout StudioState) -> [StudioJob] {
        guard let run = state.eval.run else {
            state.status.error = RaoLMFailure("no run selected", hint: "r picks one", code: 66)
            return []
        }
        guard state.status.mlxJob == nil else {
            state.status.error = RaoLMFailure("the MLX worker is busy: \(state.status.mlxJob!.label)", hint: "press c to cancel it", code: 75)
            return []
        }
        let form = state.eval.form
        let lambdas = form["lambdas"].split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
        guard let facts = form.int("facts"), facts > 0, !lambdas.isEmpty, let primary = form.float("primary"), let seed = form.uint("seed") else {
            state.status.error = RaoLMFailure("check the evaluation settings (facts > 0, λ list like 0,0.5)", code: 64)
            return []
        }
        state.eval.progress.clear()
        state.eval.report = nil
        return [.eval(EvalSpec(run: run, epoch: nil, factsSample: facts, lambdas: lambdas, primaryLambda: primary, seed: seed,
                               controls: form.bool("controls"), offline: form.bool("offline"), grounding: form.bool("grounding"),
                               owner: nil))]
    }

    static func hints(_ state: StudioState) -> [KeyHint] {
        if state.eval.form.editing { return [KeyHint("⏎", "commit"), KeyHint("esc", "done"), KeyHint("tab", "next field")] }
        return [KeyHint("⏎", state.eval.focusOutcomes ? "open generation" : "edit/run"), KeyHint("tab", "outcomes"),
                KeyHint("r", "run"), KeyHint("l", "load eval.json"), KeyHint("c", "cancel")]
    }
}
