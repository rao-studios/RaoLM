//
//  TrainScreen.swift
//  RaoLMStudio
//
//  WHAT: 5 Train: pretraining from a corpus snapshot with the settings `raolm train` takes,
//        and the run live: progress and ETA, the loss curve, entropy, learning rate, gradient
//        norm and throughput, the epoch table with checkpoint and index hashes, and events.
//  PIN:  ETA = remaining steps × the smoothed step time + the eval epochs still to come × the
//        overhead the last eval epoch measured. Cancelling stops at the next step; the run keeps
//        its completed epochs and is marked `stopped`.
//

import Foundation
import RaoLM
import RaoLMTerminal
import RaoLMWorkflows

enum TrainScreen: StudioScreen {
    static func apply(_ event: TrainingEvent, state: inout StudioState) {
        guard var live = state.train.live else { return }
        switch event {
        case .started(let total, let perEpoch, let tokens):
            live.totalSteps = total
            live.stepsPerEpoch = perEpoch
            live.events.append("training: \(total) steps (\(perEpoch.first ?? 0)/epoch × \(live.epochs) epochs), \(Format.count(tokens)) tokens/step")
        case .step(let row):
            if let previous = live.lastStep {
                let seconds = max(0, row.wallClockSeconds - previous.wallClockSeconds)
                if row.epoch == previous.epoch {
                    live.stepSeconds = live.stepSeconds.map { $0 * 0.9 + seconds * 0.1 } ?? seconds
                }
            }
            live.lastStep = row
            live.losses.append(Double(row.loss))
            live.entropies.append(Double(row.entropy.mean))
            live.gradNorms.append(Double(row.gradNorm))
            live.tokensPerSecond.append(row.tokensPerSecond)
            live.lrs.append(Double(row.lr))
            let perEpoch = live.stepsPerEpoch.first ?? 0
            var label = "training epoch \(row.epoch)/\(live.epochs) step \(row.step + 1)/\(perEpoch)"
            if let eta = live.eta { label += " · ETA \(Format.duration(eta))" }
            state.status.jobs[.train]?.label = label
        case .epoch(let record):
            live.records.append(record)
            if record.evalLoss != nil, let stepSeconds = live.stepSeconds {
                live.evalOverhead = max(0, record.wallClockSeconds - Double(record.steps) * stepSeconds)
            }
            live.events.append(TrainingConsolePrinter.epochLine(record))
        case .indexed(let epoch, let entries, let sha, let seconds):
            live.events.append("provenance index for epoch \(epoch): \(Format.count(entries)) entries, sha \(Format.short(sha)), \(Format.duration(seconds))")
        case .earlyStop(let epoch, let memorised):
            live.events.append(String(format: "early stop after epoch %d: %.1f%% of positions memorised", epoch, memorised * 100))
        case .message(let text):
            live.events.append(text)
        }
        state.train.live = live
    }

    static func render(_ state: StudioState, _ frame: inout Frame, _ rect: Rect) {
        let palette = frame.palette
        let columns = rect.splitHorizontally([.fixed(min(44, rect.width / 3)), .flex(1)])
        let formHeight = min(columns[0].height, state.train.form.fields.count + 3)
        let (formRect, noteRect) = columns[0].top(formHeight)
        let editing = state.status.mlxJob == nil
        let form = Theme.panel("Pretrain", focused: editing, &frame, formRect)
        state.train.form.render(in: form.inset(top: 1), focused: editing, frame: &frame, labelWidth: 14)
        if noteRect.height > 2 {
            let note = Theme.panel("Snapshot", &frame, noteRect)
            if let index = state.train.form.choiceIndex("snapshot"), index < state.train.snapshots.count {
                let s = state.train.snapshots[index]
                Theme.keyValues([
                    ("corpus", Text("\(s.slug ?? "?") · \(s.documentCount) docs · \(s.partitionCount) parts")),
                    ("hash", Text(Format.short(s.corpusHash), style: palette.dim)),
                    ("source", Text(s.source + (s.threadID.map { " · node \(Theme.short($0.lowercased(), 8))" } ?? ""), style: palette.dim)),
                    ("facts", s.factsPath == nil ? Text("none — no fact ledger or eval", style: palette.orange) : Text("facts.jsonl found", style: palette.green)),
                ], &frame, note, labelWidth: 8)
            } else {
                Theme.empty("No snapshots yet. On 3 Corpus: p pulls one from the Thread, O takes one offline.", &frame, note)
            }
        }
        renderLive(state, &frame, columns[1])
    }

    static func renderLive(_ state: StudioState, _ frame: inout Frame, _ rect: Rect) {
        let palette = frame.palette
        let glyphs = frame.glyphs
        guard let live = state.train.live else {
            let inner = Theme.panel("Run", &frame, rect)
            Theme.empty("Pick a snapshot and settings, then ⏎ on “Start training”. The tiny preset memorises a small corpus in a few minutes on Apple Silicon; 2 epochs with eval every 1 is a quick smoke test.", &frame, inner)
            return
        }
        let (progressRect, rest) = rect.top(4)
        let status: Text
        if let finished = live.finished {
            status = Text("\(Theme.statusGlyph(finished.status, glyphs: glyphs)) \(finished.status.rawValue)", style: Theme.statusStyle(finished.status, palette: palette))
        } else if live.cancelRequested {
            status = Text("\(StudioApp.spinner(state, glyphs)) stopping at the next step", style: palette.orange)
        } else {
            status = Text("\(StudioApp.spinner(state, glyphs)) training", style: palette.orange)
        }
        let progress = Theme.panel(live.runID, footer: status, &frame, progressRect)
        let fraction = live.totalSteps > 0 ? Double(live.globalStep) / Double(live.totalSteps) : 0
        let epoch = live.lastStep?.epoch ?? 0
        var tail = Text(ProgressBar.percent(fraction), style: palette.text)
        if let eta = live.eta { tail.append("  eta \(Format.duration(eta))", palette.dim) }
        ProgressBar(fraction: live.finished?.status == .complete ? 1 : fraction, label: Text("epoch \(epoch)/\(live.epochs)", style: palette.dim), trailing: tail,
                    fillStyle: palette.blue, trackStyle: palette.muted, full: glyphs.progressFull, empty: glyphs.progressEmpty)
            .render(in: progress.row(0), on: &frame.canvas)
        let elapsed = state.now.timeIntervalSince(live.startedAt)
        let line = Text("step \(Format.count(live.globalStep))/\(Format.count(live.totalSteps)) · elapsed \(Format.duration(elapsed)) · "
                        + (live.summary.first ?? ""), style: palette.dim)
        frame.canvas.put(line.truncated(to: progress.width), x: progress.minX, y: progress.minY + 1, clip: progress)

        let chartHeight = max(6, min(12, rest.height / 2))
        let (chartsRect, lower) = rest.top(chartHeight)
        let chartColumns = chartsRect.splitHorizontally([.flex(3), .fixed(20)])
        let lossInner = Theme.panel("Loss (train)  ·  entropy below", &frame, chartColumns[0])
        let (entropyRow, lossArea) = lossInner.bottom(1)
        LineChart(live.losses.elements, mode: .braille, style: palette.blue, labelStyle: palette.dim, glyphs: glyphs)
            .render(in: lossArea, on: &frame.canvas)
        var entropyLabel = Text("H      ", style: palette.dim)
        entropyLabel.append(Sparkline(live.entropies.elements, style: palette.green, glyphs: glyphs).string(width: max(0, entropyRow.width - 22)), palette.green)
        entropyLabel.append("  \(Format.f(live.lastStep?.entropy.mean)) nat", palette.dim)
        frame.canvas.put(entropyLabel, x: entropyRow.minX, y: entropyRow.minY, clip: entropyRow)

        let tiles = chartColumns[1].splitVertically([.flex(1), .flex(1), .flex(1)])
        tile("lr", String(format: "%.2e", live.lastStep?.lr ?? 0), live.lrs.elements, palette.dim, &frame, tiles[0])
        tile("grad", Format.f(live.lastStep?.gradNorm, 2) + (live.lastStep?.clipped == true ? " clip" : ""), live.gradNorms.elements, palette.orange, &frame, tiles[1])
        tile("tok/s", Format.count(Int(live.lastStep?.tokensPerSecond ?? 0)), live.tokensPerSecond.elements, palette.green, &frame, tiles[2])

        let lowerColumns = lower.splitVertically([.flex(3), .flex(2)])
        let epochs = Theme.panel("Epochs · \(live.records.count)/\(live.epochs)", &frame, lowerColumns[0])
        let rows: [[Text]] = live.records.map { r in
            [Text(String(r.epoch)), Text(String(r.steps)), Text(Format.f(r.trainLoss)), Text(Format.f(r.trainEntropy)),
             Text(Format.f(r.evalLoss)), Text(Format.f(r.evalEntropy)),
             Text(Format.pct(r.evalMemorisedFraction), style: r.evalMemorisedFraction == nil ? palette.muted : palette.green),
             Text(Format.f(r.calibrationGap)), Text(r.checkpointSHA256.map { String($0.prefix(8)) } ?? "—", style: palette.dim),
             r.indexSHA256.map { Text("\(glyphs.check) \($0.prefix(8))", style: palette.gold) } ?? Text(glyphs.dash, style: palette.muted)]
        }
        let visible = Table.visibleRows(in: epochs, showHeader: true)
        Table(columns: [Column("epoch", .fixed(5), align: .trailing), Column("steps", .fixed(5), align: .trailing),
                        Column("train loss", .fixed(10), align: .trailing), Column("train H", .fixed(7), align: .trailing),
                        Column("eval loss", .fixed(9), align: .trailing), Column("eval H", .fixed(6), align: .trailing),
                        Column("memorised", .fixed(9), align: .trailing), Column("gap", .fixed(6), align: .trailing),
                        Column("ckpt", .fixed(8)), Column("index", .flex(1))],
              rows: rows, state: TableState(selected: nil, scroll: max(0, rows.count - visible)),
              headerStyle: palette.dim, ruleStyle: palette.border, rowStyle: palette.text, glyphs: glyphs)
            .render(in: epochs, on: &frame.canvas)
        let events = Theme.panel("Events", &frame, lowerColumns[1])
        LogPane(state: live.events, style: palette.dim).render(in: events, on: &frame.canvas)
    }

    static func tile(_ title: String, _ value: String, _ series: [Double], _ style: Style, _ frame: inout Frame, _ rect: Rect) {
        let inner = Theme.panel(title, &frame, rect)
        guard inner.height > 0 else { return }
        frame.canvas.put(value, x: inner.minX, y: inner.minY, style: frame.palette.title, clip: inner)
        if inner.height > 1 {
            Sparkline(series, style: style, glyphs: frame.glyphs).render(in: inner.row(1), on: &frame.canvas)
        }
    }

    static func handle(_ key: KeyEvent, _ state: inout StudioState) -> [StudioJob]? {
        if key.key == .char("c"), !state.train.form.editing {
            guard state.status.jobs[.train] != nil else { return [] }
            state.train.live?.cancelRequested = true
            return [.cancel]
        }
        if key.key == .char("g"), !state.train.form.editing, let run = state.train.live?.finished {
            let directory = state.root.run(id: run.runID)
            var jobs = GenerateScreen.select(run: directory, epoch: nil, state: &state)
            jobs += StudioApp.go(to: .generate, state: &state)
            return jobs
        }
        if key.key == .char("r"), !state.train.form.editing { return [.scanSnapshots] }
        switch state.train.form.handle(key) {
        case .changed: return []
        case .submit: return submit(&state)
        case .unhandled: return nil
        }
    }

    static func submit(_ state: inout StudioState) -> [StudioJob] {
        guard state.status.mlxJob == nil else {
            state.status.error = RaoLMFailure("the MLX worker is busy: \(state.status.mlxJob!.label)", hint: "press c to cancel it", code: 75)
            return []
        }
        guard let index = state.train.form.choiceIndex("snapshot"), index < state.train.snapshots.count else {
            state.status.error = RaoLMFailure("no snapshot to train on", hint: "3 Corpus: p pulls one from the Thread, O takes one offline", code: 66)
            return []
        }
        let form = state.train.form
        var settings = TrainingSettings()
        settings.preset = form["preset"]
        guard let epochs = form.int("epochs"), epochs > 0, let batch = form.int("batchSize"), batch > 0,
              let seqLen = form.int("seqLen"), seqLen > 8, let lr = form.float("lr"), lr > 0,
              let evalEvery = form.int("evalEvery"), evalEvery >= 0, let indexEvery = form.int("indexEvery"), indexEvery >= 0,
              let alpha = form.float("alpha"), let seed = form.uint("seed"), let earlyStop = form.float("earlyStop"),
              let exclude = form.int("exclude"), exclude >= 0, let keep = form.int("keep"), keep >= 1
        else {
            state.status.error = RaoLMFailure("a training setting is not a valid number", code: 64)
            return []
        }
        let tap = form["tapLayer"].trimmingCharacters(in: .whitespaces)
        if !tap.isEmpty {
            guard let layer = Int(tap), layer >= 0 else {
                state.status.error = RaoLMFailure("tap layer must be empty or a layer index", code: 64)
                return []
            }
            settings.tapLayer = layer
        }
        settings.epochs = epochs
        settings.batchSize = batch
        settings.seqLen = seqLen
        settings.lr = lr
        settings.evalEvery = evalEvery
        settings.indexEvery = indexEvery
        settings.alpha = alpha
        settings.seed = seed
        settings.earlyStop = earlyStop
        settings.excludeDocuments = exclude
        settings.keepCheckpoints = keep
        do { _ = try settings.modelConfig() } catch {
            state.status.error = FailureMapping.describe(error)
            return []
        }
        let choice = state.train.snapshots[index]
        let runID = DataRoot.newRunID(preset: settings.preset)
        var threadRef: ThreadRef?
        if let thread = state.status.thread, thread.isUp, let record = thread.record, choice.threadID != nil {
            threadRef = ThreadRef(binary: record.binary, host: record.host, httpPort: record.httpPort, grpcPort: record.grpcPort,
                                  dataDir: state.root.threadDB.path, nodeID: record.nodeID)
        }
        state.train.live = nil
        return [.train(TrainSpec(settings: settings, snapshotPath: choice.path, factsPath: choice.factsPath, runID: runID,
                                 runDirectory: state.root.run(id: runID), thread: threadRef))]
    }

    static func hints(_ state: StudioState) -> [KeyHint] {
        if state.train.form.editing { return [KeyHint("⏎", "commit"), KeyHint("esc", "done"), KeyHint("tab", "next field")] }
        var hints = [KeyHint("⏎", "edit/start"), KeyHint("←→", "choose")]
        if state.status.jobs[.train] != nil { hints.append(KeyHint("c", "cancel")) }
        if state.train.live?.finished != nil { hints.append(KeyHint("g", "generate")) }
        hints.append(KeyHint("r", "rescan"))
        return hints
    }
}
