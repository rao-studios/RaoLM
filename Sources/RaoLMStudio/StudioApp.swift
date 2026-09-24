//
//  StudioApp.swift
//  RaoLMStudio
//
//  WHAT: The studio as three pure functions over `StudioState`: `handle` (a key, tick, resize,
//        signal or backend event in, jobs out), `reduce` (backend events), and `render` (the
//        banner, the screen tabs, the screen, key hints, the status bar and any overlay).
//  PIN:  Pure so tests drive it without a terminal or MLX. Keys go to an open overlay first,
//        then to a field being edited, then to the global keys, then to the screen.
//

import Foundation
import RaoLM
import RaoLMTerminal
import RaoLMWorkflows

protocol StudioScreen {
    static func render(_ state: StudioState, _ frame: inout Frame, _ rect: Rect)
    /// Nil when the key means nothing on this screen.
    static func handle(_ key: KeyEvent, _ state: inout StudioState) -> [StudioJob]?
    static func hints(_ state: StudioState) -> [KeyHint]
}

public enum StudioApp {
    static func screen(_ screen: Screen) -> StudioScreen.Type {
        switch screen {
        case .home: return HomeScreen.self
        case .doctor: return DoctorScreen.self
        case .corpus: return CorpusScreen.self
        case .thread: return ThreadScreen.self
        case .train: return TrainScreen.self
        case .generate: return GenerateScreen.self
        case .eval: return EvalScreen.self
        case .ledger: return LedgerScreen.self
        }
    }

    /// What the studio asks for as soon as it opens.
    public static func initialJobs(_ state: StudioState) -> [StudioJob] {
        [.scanRuns, .doctor, .threadStatus(withLog: false), .scanCorpora, .scanSnapshots]
    }

    // MARK: - Handle

    public static func handle(_ event: Event<StudioEvent>, state: inout StudioState) -> [StudioJob] {
        switch event {
        case .message(let message):
            return reduce(message, state: &state)
        case .tick(let date):
            return tick(date, state: &state)
        case .resize(let size):
            state.size = size
            return []
        case .signal(let signal):
            if signal == .interrupt { return requestQuit(&state) }
            state.quitCode = 130
            return [.cancel, .cancelTests]
        case .inputClosed:
            state.quitCode = 0
            return [.cancel, .cancelTests]
        case .key(let key):
            return handleKey(key, state: &state)
        }
    }

    static func handleKey(_ key: KeyEvent, state: inout StudioState) -> [StudioJob] {
        if key == KeyEvent(.ctrl("c")) { return requestQuit(&state) }
        if state.overlay != nil { return handleOverlay(key, state: &state) }
        if state.editing { return screen(state.screen).handle(key, &state) ?? [] }
        state.status.error = nil
        switch key.key {
        case .char(let c) where ("1"..."8").contains(c) && key.modifiers.isEmpty:
            return go(to: Screen(rawValue: Int(String(c))!)!, state: &state)
        case .char("]"):
            return go(to: Screen(rawValue: state.screen.rawValue % 8 + 1)!, state: &state)
        case .char("["):
            return go(to: Screen(rawValue: (state.screen.rawValue + 6) % 8 + 1)!, state: &state)
        case .char("?"):
            state.overlay = .help
            return []
        case .char("q"):
            return requestQuit(&state)
        case .char("Q"):
            state.quitCode = 130
            return [.cancel, .cancelTests]
        case .ctrl("l"):
            state.redraw = true
            return []
        case .escape:
            if let previous = state.history.popLast() { state.screen = previous }
            return []
        default:
            return screen(state.screen).handle(key, &state) ?? []
        }
    }

    /// Switches screens and asks for whatever the new one shows.
    public static func go(to target: Screen, state: inout StudioState, remember: Bool = true) -> [StudioJob] {
        guard target != state.screen else { return [] }
        if remember { state.history.append(state.screen) }
        if state.history.count > 20 { state.history.removeFirst() }
        state.screen = target
        switch target {
        case .home: return [.scanRuns]
        case .thread: return [.threadStatus(withLog: true)]
        case .corpus: return [.scanCorpora]
        case .train: return [.scanSnapshots]
        case .generate:
            if state.generate.run == nil, state.generate.generation == nil, let run = state.selectedRun, !run.indexedEpochs.isEmpty {
                return GenerateScreen.select(run: run.directory, epoch: nil, state: &state)
            }
            return []
        case .eval:
            if state.eval.run == nil { state.eval.run = state.selectedRun?.directory }
            return []
        case .ledger:
            if state.ledger.run == nil, let run = state.selectedRun {
                state.ledger.run = run.directory
                return [LedgerScreen.job(state)].compactMap { $0 }
            }
            return []
        case .doctor:
            return []
        }
    }

    static func runningBlocksQuit(_ state: StudioState) -> Bool {
        state.status.jobs.values.contains { $0.kind.isMLX || $0.kind == .tests || $0.kind == .ingest || $0.kind == .threadStart }
    }

    static func requestQuit(_ state: inout StudioState) -> [StudioJob] {
        if state.fixtures {
            state.quitCode = 0
            return [.cancel, .cancelTests]
        }
        if runningBlocksQuit(state) {
            state.overlay = .confirmQuit
            return []
        }
        if state.status.thread?.ownedByStudio == true, state.status.thread?.isUp == true {
            state.overlay = .confirmStopThread
            return []
        }
        state.quitCode = 0
        return []
    }

    static func handleOverlay(_ key: KeyEvent, state: inout StudioState) -> [StudioJob] {
        guard let overlay = state.overlay else { return [] }
        switch overlay {
        case .help:
            state.overlay = nil
            return []
        case .confirmQuit:
            switch key.key {
            case .char("y"), .enter:
                state.overlay = .cancelling(since: state.now)
                state.quitAfterJobs = true
                return [.cancel, .cancelTests]
            case .char("Q"):
                state.quitCode = 130
                return [.cancel, .cancelTests]
            default:
                state.overlay = nil
                return []
            }
        case .confirmStopThread:
            switch key.key {
            case .char("s"), .char("y"):
                state.overlay = .cancelling(since: state.now)
                state.quitAfterJobs = true
                return [.threadStop]
            case .char("l"), .enter:
                state.quitCode = 0
                return []
            default:
                state.overlay = nil
                return []
            }
        case .cancelling:
            if key.key == .char("Q") { state.quitCode = 130 }
            return []
        case .picker(var picker):
            let visible = max(1, min(picker.items.count, state.size.height - 10))
            switch key.key {
            case .up, .char("k"): picker.table.move(by: -1, rowCount: picker.items.count, visible: visible)
            case .down, .char("j"): picker.table.move(by: 1, rowCount: picker.items.count, visible: visible)
            case .pageUp: picker.table.page(by: -1, rowCount: picker.items.count, visible: visible)
            case .pageDown: picker.table.page(by: 1, rowCount: picker.items.count, visible: visible)
            case .escape, .char("q"):
                state.overlay = nil
                return []
            case .enter:
                state.overlay = nil
                guard let index = picker.table.selected, index < picker.values.count else { return [] }
                let url = picker.values[index]
                switch picker.purpose {
                case .generation:
                    return [.loadGeneration(url)]
                case .run(let screen):
                    switch screen {
                    case .generate: return GenerateScreen.select(run: url, epoch: nil, state: &state)
                    case .eval:
                        state.eval.run = url
                        return []
                    case .ledger:
                        state.ledger.run = url
                        return [LedgerScreen.job(state)].compactMap { $0 }
                    default: return []
                    }
                }
            default: break
            }
            state.overlay = .picker(picker)
            return []
        case .params(var form):
            if !form.editing, key.key == .escape {
                state.overlay = nil
                return []
            }
            let outcome = form.handle(key)
            if case .submit = outcome {
                do {
                    state.generate.overrides = try form.overrides()
                    state.overlay = nil
                    state.status.message = "generation parameters updated"
                } catch {
                    state.status.error = FailureMapping.describe(error)
                    state.overlay = .params(form)
                }
                return []
            }
            state.overlay = .params(form)
            return []
        }
    }

    static func tick(_ date: Date, state: inout StudioState) -> [StudioJob] {
        state.now = date
        state.spinner &+= 1
        var jobs: [StudioJob] = []
        if case .cancelling(let since) = state.overlay {
            let busy = runningBlocksQuit(state) || state.status.jobs[.threadStop] != nil
            if !busy {
                if state.status.thread?.ownedByStudio == true, state.status.thread?.isUp == true,
                   state.status.jobs[.threadStop] == nil, !state.quitAfterJobsStoppedThread {
                    state.overlay = .confirmStopThread
                } else {
                    state.quitCode = 0
                }
            } else if date.timeIntervalSince(since) > 15 {
                state.quitCode = 130
            }
        }
        if [.home, .thread].contains(state.screen), state.status.jobs[.threadStatus] == nil,
           state.status.lastThreadPoll.map({ date.timeIntervalSince($0) > 3 }) ?? true {
            state.status.lastThreadPoll = date
            jobs.append(.threadStatus(withLog: state.screen == .thread))
        }
        return jobs
    }

    // MARK: - Reduce

    public static func reduce(_ event: StudioEvent, state: inout StudioState) -> [StudioJob] {
        switch event {
        case .jobStarted(let kind, let label):
            state.status.jobs[kind] = JobStatus(kind: kind, label: label, since: state.now)
        case .jobProgress(let kind, let label):
            state.status.jobs[kind]?.label = label
        case .jobFinished(let kind):
            state.status.jobs[kind] = nil
            if kind == .threadStop, state.quitAfterJobs { state.quitAfterJobsStoppedThread = true }
        case .jobFailed(let kind, let failure):
            state.status.jobs[kind] = nil
            if failure.code == 130 {
                state.status.message = "\(kind.rawValue) cancelled"
            } else {
                state.status.error = failure
            }
            state.doctor.log.append(Text("\(kind.rawValue): \(failure.message)", style: .plain))
            if kind == .threadStatus { state.status.lastThreadPoll = state.now }
        case .log(let message):
            state.status.message = message
        case .runsLoaded(let runs, let warnings):
            let selectedID = state.selectedRun?.id
            state.home.runs = runs
            state.home.warnings = warnings
            if let selectedID, let index = runs.firstIndex(where: { $0.id == selectedID }) {
                state.home.table.selected = index
            }
            state.home.table.clamp(rowCount: runs.count, visible: HomeScreen.visibleRows(state))
        case .doctorChecks(let checks):
            state.doctor.checks = checks
        case .testOutput(let lines):
            for line in lines { state.doctor.log.append(DoctorScreen.styled(line)) }
        case .testFinished(let code):
            state.doctor.running = false
            state.doctor.exitCode = code
            state.doctor.log.append(Text("exit \(code)", style: .plain))
            state.status.message = code == 0 ? "tests passed" : "tests failed (exit \(code))"
        case .threadStatus(let status):
            state.status.thread = status
            state.status.lastThreadPoll = state.now
            if !status.logTail.isEmpty {
                state.thread.log.clear()
                for line in status.logTail { state.thread.log.append(line) }
            }
        case .corporaLoaded(let corpora):
            state.corpus.corpora = corpora
            state.corpus.corporaTable.clamp(rowCount: corpora.count, visible: 8)
        case .corpusLoaded(let corpus, let directory):
            state.corpus.loaded = corpus
            state.corpus.loadedDirectory = directory
            state.corpus.documentsTable = TableState(selected: 0)
        case .corpusResult(let message, let problems):
            state.corpus.ingest = nil
            state.corpus.lines.insert((message, problems.isEmpty), at: 0)
            for problem in problems.prefix(6).reversed() { state.corpus.lines.insert(("  " + problem, false), at: 1) }
            state.corpus.lines = Array(state.corpus.lines.prefix(12))
            state.status.message = message
            return [.scanCorpora, .scanSnapshots]
        case .ingestProgress(let done, let total):
            state.corpus.ingest = (done, total)
        case .snapshotsLoaded(let snapshots):
            state.train.snapshots = snapshots
            state.train.form.setChoices("snapshot", snapshots.map(\.label))
        case .trainPrepared(let manifest, let summary):
            var live = TrainLive(runID: manifest.runID, runDirectory: URL(fileURLWithPath: manifest.corpus.snapshotPath), startedAt: state.now, summary: summary)
            live.runDirectory = state.root.run(id: manifest.runID)
            live.epochs = manifest.hyperparameters.epochs
            live.evalEvery = manifest.hyperparameters.evalEvery
            state.train.live = live
            return [.scanRuns]
        case .training(let event):
            TrainScreen.apply(event, state: &state)
        case .trainFinished(let manifest):
            state.train.live?.finished = manifest
            state.status.message = "run \(manifest.runID) \(manifest.status.rawValue)"
            return [.scanRuns]
        case .contextLoaded(let info):
            state.generate.context = info
            state.generate.run = info.runDirectory
            state.status.message = "loaded \(info.runID) epoch \(info.epoch)"
        case .token(let trace):
            state.generate.streaming.append(trace)
        case .generated(let generation, let rows):
            GenerateScreen.show(generation, file: nil, rows: rows, grounding: nil, state: &state)
            return GenerateScreen.partitionJobs(&state)
        case .verified(let generation, let lines, let source):
            let rows = state.generate.neighbourRows
            let grounding = state.generate.grounding
            GenerateScreen.show(generation, file: state.generate.generationFile, rows: rows, grounding: grounding, keepCursor: true, state: &state)
            state.generate.verification = lines
            state.generate.verificationSource = source
            state.status.message = lines.last
        case .generationSaved(let url):
            state.generate.generationFile = url
            state.status.message = "saved \(url.lastPathComponent)"
        case .generationLoaded(let generation, let url, let rows, let grounding):
            let run = url.deletingLastPathComponent().deletingLastPathComponent()
            let needsContext = state.generate.context?.runDirectory.standardizedFileURL != run.standardizedFileURL
                || state.generate.context?.epoch != generation.manifest.epoch
            state.generate.run = run
            if needsContext { state.generate.context = nil }
            GenerateScreen.show(generation, file: url, rows: rows, grounding: grounding, state: &state)
            var jobs = go(to: .generate, state: &state)
            if needsContext { jobs.append(.loadContext(run: run, epoch: generation.manifest.epoch)) }
            return jobs + GenerateScreen.partitionJobs(&state)
        case .generationsListed(let run, let urls):
            guard !urls.isEmpty else {
                state.status.message = "no saved generations in \(run.lastPathComponent)"
                return []
            }
            state.overlay = .picker(Picker(
                title: "Saved generations · \(run.lastPathComponent)", items: urls.map { $0.deletingPathExtension().lastPathComponent },
                values: urls, table: TableState(selected: 0), purpose: .generation))
        case .partitionText(let text):
            state.generate.partitionTexts[text.key] = text
        case .evalProgress(let line):
            state.eval.progress.append(line)
            state.status.jobs[.eval]?.label = "eval · " + line
        case .evalFinished(let report, let url):
            state.eval.report = report
            state.eval.savedTo = url
            state.eval.outcomes = TableState(selected: 0)
            state.status.message = "evaluation written to \(url.lastPathComponent)"
            return [.scanRuns]
        case .ledgerLoaded(let data):
            state.ledger.data = data
            state.ledger.table = TableState(selected: 0)
        case .grounded(let record):
            state.generate.grounding = record
            state.generate.tab = .grounding
            state.status.message = record.measured
                ? "grounded against \(record.sources.count) source\(record.sources.count == 1 ? "" : "s")"
                : "grounding skipped: \(record.measurement.skippedReason ?? "?")"
        }
        return []
    }

    // MARK: - Render

    public static func render(_ state: StudioState, _ frame: inout Frame) {
        let palette = frame.palette
        let glyphs = frame.glyphs
        var body = frame.bounds
        if state.screen == .home, !Banner.shouldUseCompact(frame.size) {
            let height = Banner.height(width: frame.size.width)
            let (top, rest) = body.top(height)
            Banner(version: state.version, status: bannerStatus(state, palette: palette, glyphs: glyphs), palette: palette, glyphs: glyphs)
                .render(in: top, on: &frame.canvas)
            body = rest
        } else {
            let (top, rest) = body.top(1)
            CompactHeader(title: headerTitle(state, palette: palette), status: headerStatus(state, palette: palette, glyphs: glyphs),
                          palette: palette, glyphs: glyphs).render(in: top, on: &frame.canvas)
            body = rest
        }
        let (tabsRow, afterTabs) = body.top(1)
        renderTabs(state, &frame, tabsRow)
        let (statusRow, aboveStatus) = afterTabs.bottom(1)
        let (hintsRow, screenRect) = aboveStatus.bottom(1)
        screen(state.screen).render(state, &frame, screenRect)
        KeyHintBar(screen(state.screen).hints(state) + globalHints(state), keyStyle: palette.hintKey, labelStyle: palette.hintLabel)
            .render(in: hintsRow, on: &frame.canvas)
        renderStatus(state, &frame, statusRow)
        if let overlay = state.overlay { renderOverlay(overlay, state, &frame) }
    }

    static func globalHints(_ state: StudioState) -> [KeyHint] {
        state.editing ? [] : [KeyHint("1-8", "screens"), KeyHint("?", "help"), KeyHint("q", "quit")]
    }

    static func bannerStatus(_ state: StudioState, palette: Palette, glyphs: Glyphs) -> Text {
        var text = state.fixtures ? Text("fixtures  \(glyphs.dot)  ", style: palette.orange) : Text()
        text.append(threadBadge(state, palette: palette, glyphs: glyphs))
        text.append("  \(glyphs.dot)  ", palette.muted)
        text.append("\(state.home.runs.count) run\(state.home.runs.count == 1 ? "" : "s")", palette.dim)
        text.append("  \(glyphs.dot)  ", palette.muted)
        text.append(abbreviate(state.root.url.path), palette.dim)
        return text
    }

    static func threadBadge(_ state: StudioState, palette: Palette, glyphs: Glyphs) -> Text {
        guard let thread = state.status.thread else { return Text("thread \(glyphs.idle) unknown", style: palette.dim) }
        if thread.isUp {
            let node = thread.nodeID.map { String($0.prefix(8)).lowercased() } ?? "?"
            return Text("thread ", style: palette.dim) + Text(glyphs.live, style: palette.green)
                + Text(" \(node) :\(thread.endpoint.httpPort)", style: palette.dim)
        }
        return Text("thread ", style: palette.dim) + Text(glyphs.idle, style: palette.muted) + Text(" down", style: palette.dim)
    }

    static func headerTitle(_ state: StudioState, palette: Palette) -> Text {
        var text = Text(state.screen.title, style: palette.title)
        let detail: String?
        switch state.screen {
        case .generate: detail = state.generate.context.map { "\($0.runID) · epoch \($0.epoch) · λ \(Format.f(state.generate.overrides.lambda, 2))" }
        case .train: detail = state.train.live.map { $0.runID }
        case .eval: detail = state.eval.run?.lastPathComponent
        case .ledger: detail = state.ledger.run?.lastPathComponent
        case .corpus: detail = state.corpus.loaded?.manifest.slug
        case .thread: detail = state.status.thread?.nodeID
        default: detail = nil
        }
        if let detail { text.append(" · " + detail, palette.dim) }
        return text
    }

    static func headerStatus(_ state: StudioState, palette: Palette, glyphs: Glyphs) -> Text {
        if let job = state.status.headline {
            return Text(spinner(state, glyphs) + " ", style: palette.orange) + Text(job.label, style: palette.dim)
        }
        return threadBadge(state, palette: palette, glyphs: glyphs)
    }

    static func spinner(_ state: StudioState, _ glyphs: Glyphs) -> String {
        glyphs.spinner[state.spinner % glyphs.spinner.count]
    }

    static func renderTabs(_ state: StudioState, _ frame: inout Frame, _ rect: Rect) {
        let palette = frame.palette
        let busy = Set(state.status.jobs.values.filter(\.kind.isVisible).map { Screen.owner(of: $0.kind) })
        let titles: [Text] = Screen.allCases.map { screen in
            let chosen = screen == state.screen
            var text = Text("\(screen.rawValue) ", style: chosen ? palette.gold : palette.muted)
            text.append(screen.title, chosen ? palette.blue.bold() : palette.dim)
            if busy.contains(screen) { text.append(" " + spinner(state, frame.glyphs), palette.orange) }
            return text
        }
        Tabs(titles, selected: state.screen.rawValue - 1, style: palette.dim, selectedStyle: palette.blue.bold(),
             separator: Text(rect.width < 100 ? "  " : "   ", style: palette.dim)).render(in: rect, on: &frame.canvas)
    }

    static func renderStatus(_ state: StudioState, _ frame: inout Frame, _ rect: Rect) {
        let palette = frame.palette
        let glyphs = frame.glyphs
        frame.canvas.fill(rect, style: palette.selection.removing([.bold, .reverse]).fg(palette.dim.foreground))
        let band = palette.selection.background
        func on(_ style: Style) -> Style { band == .default ? style : style.bg(band) }
        var left = Text(" ", style: on(palette.dim))
        if state.fixtures { left.append("fixtures  ", on(palette.orange)) }
        left.append(Text(spans: threadBadge(state, palette: palette, glyphs: glyphs).spans.map { Span($0.text, on($0.style)) }))
        if let job = state.status.headline {
            left.append("  ", on(palette.dim))
            left.append(spinner(state, glyphs) + " ", on(palette.orange))
            left.append(job.label, on(palette.text))
        } else {
            left.append("  " + abbreviate(state.root.url.path), on(palette.muted))
        }
        var right = Text()
        if let error = state.status.error {
            right.append("raolm: " + error.message, on(palette.error))
            if let hint = error.hint { right.append(" — " + hint, on(palette.dim)) }
        } else if let message = state.status.message {
            right.append(message, on(palette.dim))
        }
        let leftWidth = min(left.width, rect.width * 3 / 5)
        frame.canvas.put(left.truncated(to: leftWidth), x: rect.minX, y: rect.minY, clip: rect)
        let rightRoom = max(0, rect.width - leftWidth - 3)
        let shown = right.truncated(to: rightRoom)
        frame.canvas.put(shown, x: rect.maxX - shown.width - 1, y: rect.minY, clip: rect)
    }

    // MARK: - Overlays

    static func renderOverlay(_ overlay: Overlay, _ state: StudioState, _ frame: inout Frame) {
        let palette = frame.palette
        let glyphs = frame.glyphs
        frame.canvas.restyle(frame.bounds) { $0.adding(.dim) }
        switch overlay {
        case .help:
            let width = min(frame.size.width - 4, 100)
            let lines = helpLines(glyphs, width: width - 5)
            let rect = frame.bounds.centered(width: width, height: min(frame.size.height - 2, lines.count + 2))
            frame.canvas.fill(rect, style: palette.base)
            let inner = Box(title: Text("Help", style: palette.title), footer: Text("any key closes", style: palette.dim),
                            style: palette.focusBorder, glyphs: glyphs).render(in: rect, on: &frame.canvas)
            for (row, line) in lines.prefix(inner.height).enumerated() {
                frame.canvas.put(line.truncated(to: inner.width - 1), x: inner.minX + 1, y: inner.minY + row, clip: inner)
            }
        case .confirmQuit:
            let label = state.status.headline?.label ?? "a job"
            dialog("Quit", [
                Text("\(label) is still running.", style: palette.text),
                Text(""),
                Text("[y] cancel it and quit    [Q] quit now    [esc] stay", style: palette.dim),
            ], &frame)
        case .confirmStopThread:
            dialog("Quit", [
                Text("The studio started a Thread on :\(state.status.thread?.endpoint.httpPort ?? 0).", style: palette.text),
                Text(""),
                Text("[s] stop it and quit    [l] leave it running and quit    [esc] stay", style: palette.dim),
            ], &frame)
        case .cancelling:
            dialog("Quitting", [
                Text(spinner(state, glyphs) + " waiting for running work to stop…", style: palette.orange),
                Text(""),
                Text("[Q] quit now", style: palette.dim),
            ], &frame)
        case .picker(let picker):
            let width = min(frame.size.width - 8, max(50, (picker.items.map(\.count).max() ?? 20) + 10))
            let height = min(frame.size.height - 6, picker.items.count + 2)
            let rect = frame.bounds.centered(width: width, height: max(3, height))
            frame.canvas.fill(rect, style: palette.base)
            let inner = Box(title: Text(picker.title, style: palette.title), footer: Text("⏎ open  esc close", style: palette.dim),
                            style: palette.focusBorder, glyphs: glyphs).render(in: rect, on: &frame.canvas)
            Table.styled(columns: [Column("", .flex(1))], rows: picker.items.map { [Text($0)] }, state: picker.table,
                         palette: palette, glyphs: glyphs, showHeader: false).render(in: inner, on: &frame.canvas)
        case .params(let form):
            let rect = frame.bounds.centered(width: 50, height: form.fields.count + 4)
            frame.canvas.fill(rect, style: palette.base)
            let inner = Box(title: Text("Generation parameters", style: palette.title), footer: Text("esc close", style: palette.dim),
                            style: palette.focusBorder, glyphs: glyphs).render(in: rect, on: &frame.canvas)
            form.render(in: inner.inset(top: 1), focused: true, frame: &frame)
        }
    }

    static func dialog(_ title: String, _ lines: [Text], _ frame: inout Frame) {
        let palette = frame.palette
        let width = min(frame.size.width - 4, max(40, (lines.map(\.width).max() ?? 30) + 6))
        let rect = frame.bounds.centered(width: width, height: lines.count + 4)
        frame.canvas.fill(rect, style: palette.base)
        let inner = Box(title: Text(title, style: palette.title), style: palette.focusBorder, glyphs: frame.glyphs)
            .render(in: rect, on: &frame.canvas)
        for (row, line) in lines.enumerated() {
            frame.canvas.put(line, x: inner.minX + 2, y: inner.minY + 1 + row, clip: inner)
        }
    }

    static func helpLines(_ glyphs: Glyphs, width: Int = 94) -> [Text] {
        let sections: [(String, [(String, String)])] = [
            ("screens", [("1 Home", "runs"), ("2 Tests", "doctor, test suites"), ("3 Corpus", "dataset generation"), ("4 Thread", "node"),
                         ("5 Train", "pretraining"), ("6 Generate", "output contribution debugger"), ("7 Eval", "precision"), ("8 Ledger", "entropy ledger")]),
            ("move", [("↑↓ jk", "rows"), ("←→ hl", "tokens, choices"), ("tab", "next pane or panel"), ("[ ]", "previous / next screen"), ("esc", "back")]),
            ("forms", [("⏎", "edit a field, again to commit"), ("space ←→", "toggle, choose"), ("⏎ on ▸", "run the action")]),
            ("generate", [("/", "prompt"), ("x", "next fact prompt"), ("m", "text or corpus slice"), ("P", "parameters"), ("⏎", "generate"),
                          ("v V", "verify offline / live"), ("G A", "ground on source / all cited"), ("s o", "save / open")]),
            ("jobs", [("c", "cancel the running job"), ("r", "refresh"), ("Q", "quit now")]),
        ]
        var lines: [Text] = []
        for (title, entries) in sections {
            var line = Text(title.padding(toLength: 10, withPad: " ", startingAt: 0), style: .plain)
            for (key, label) in entries {
                line.append("[" + key + "] ", .plain)
                line.append(label + "   ", .plain)
            }
            lines.append(contentsOf: hanging(line, width: width))
        }
        lines.append(Text(""))
        for line in [
            "legend    \(glyphs.check) ok  \(glyphs.cross) fail  \(glyphs.live) live  \(glyphs.idle) idle  \(glyphs.spinner[0]) busy  \(glyphs.verified) verified  \(glyphs.select) selected",
            "heat      dim uncited · blue < 0.5 · green < 0.8 · white < 1.0 · gold underlined = verified span",
            "grounding \(glyphs.live) grounded (ι ≥ 0.5)  ◐ unsupported  \(glyphs.cross) contradicted (ι ≤ −0.5)  \(glyphs.dot) function word",
            "",
            "cli       raolm doctor · corpus generate|ingest|pull · thread start|stop · train · generate · verify · ground · eval · ledger",
        ] { lines.append(contentsOf: hanging(Text(line), width: width)) }
        return lines
    }

    /// Wraps with a 10-column hanging indent under the section label.
    static func hanging(_ line: Text, width: Int) -> [Text] {
        guard !line.isEmpty else { return [line] }
        let first = Paragraph.wrap(line, width: max(20, width))
        guard first.count > 1 else { return first }
        var result = [first[0]]
        let remainder = first.dropFirst().reduce(Text()) { $0 + Text(" ") + $1 }
        for piece in Paragraph.wrap(remainder, width: max(10, width - 10)) {
            result.append(Text("          ") + piece)
        }
        return result
    }

    static func abbreviate(_ path: String) -> String {
        let home = NSHomeDirectory()
        return path.hasPrefix(home) ? "~" + path.dropFirst(home.count) : path
    }
}

extension StudioState {
    /// Set once the Thread-stop job of a quit has finished, so the studio does not ask again.
    public var quitAfterJobsStoppedThread: Bool {
        get { _quitAfterJobsStoppedThread }
        set { _quitAfterJobsStoppedThread = newValue }
    }
}
