import Foundation
import Testing

@testable import RaoLM
@testable import RaoLMStudio
@testable import RaoLMTerminal
@testable import RaoLMWorkflows

final class Collector: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [StudioEvent] = []

    func post(_ event: StudioEvent) { lock.withLock { events.append(event) } }

    func drain() -> [StudioEvent] {
        lock.withLock {
            defer { events.removeAll() }
            return events
        }
    }
}

/// The fixture backend's jobs, folded through the studio's reducer, rendered into a canvas.
struct Harness {
    static var fixtures: URL {
        Bundle.module.url(forResource: "Fixtures", withExtension: nil)!.appendingPathComponent("demo")
    }

    var backend: FixtureBackend
    let collector = Collector()
    var state: StudioState

    init() throws {
        let collector = self.collector
        backend = try FixtureBackend(directory: Self.fixtures, post: { collector.post($0) })
        backend.replayDuration = .zero
        state = StudioState(root: DataRoot(url: Self.fixtures), fixtures: true)
        state.size = Size(width: 120, height: 40)
    }

    mutating func run(_ job: StudioJob) async throws {
        try await backend.perform(job)
        var followUps: [StudioJob] = []
        for event in collector.drain() { followUps += StudioApp.reduce(event, state: &state) }
        for job in followUps {
            try? await backend.perform(job)
            for event in collector.drain() { _ = StudioApp.reduce(event, state: &state) }
        }
    }

    mutating func key(_ key: KeyEvent) async throws {
        let jobs = StudioApp.handle(.key(key), state: &state)
        for job in jobs { try await run(job) }
    }

    func render(_ size: Size = Size(width: 120, height: 40), depth: ColorDepth = .ansi256) -> Canvas {
        let capabilities = Capabilities(isTTY: true, colorDepth: depth)
        var frame = Frame(canvas: Canvas(size: size), palette: Palette(for: capabilities, environment: [:]), glyphs: .unicode)
        StudioApp.render(state, &frame)
        return frame.canvas
    }

    /// Every cell with its colours, for drawing a screenshot of a frame (RAOLM_DUMP_SCREEN=<file>).
    func dump(_ canvas: Canvas, to path: String) throws {
        func rgb(_ color: Color) -> [Int]? { color.rgbComponents.map { [Int($0.r), Int($0.g), Int($0.b)] } }
        var rows: [[[String: Any]]] = []
        for y in 0..<canvas.size.height {
            var row: [[String: Any]] = []
            for x in 0..<canvas.size.width {
                let cell = canvas[x, y]
                var entry: [String: Any] = ["c": String(Character(cell.scalar)), "cont": cell.isContinuation]
                if let fg = rgb(cell.style.foreground) { entry["fg"] = fg }
                if let bg = rgb(cell.style.background) { entry["bg"] = bg }
                entry["a"] = Int(cell.style.attributes.rawValue)
                row.append(entry)
            }
            rows.append(row)
        }
        try JSONSerialization.data(withJSONObject: rows).write(to: URL(fileURLWithPath: path))
    }

    var run: URL { state.home.runs[0].directory }

    func generations(grounded: Bool) -> [URL] {
        let directory = RunLayout.generations(run)
        let names = (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
        return names.filter { $0.hasSuffix(".json") && !$0.hasSuffix(".grounding.json") }.sorted()
            .map { directory.appendingPathComponent($0) }
            .filter { url in
                guard grounded else { return true }
                guard let generation = try? CitedGeneration.load(from: url) else { return false }
                return FileManager.default.fileExists(atPath: GroundingRecord.url(runDirectory: run, generationID: generation.generationID).path)
            }
    }
}

@Suite("Studio screens", .serialized)
struct StudioScreenTests {
    @Test("1 Home: the banner, the run table and the run's detail")
    func home() async throws {
        var h = try Harness()
        try await h.run(.scanRuns)
        try await h.run(.threadStatus(withLog: false))
        #expect(h.state.home.runs.count == 1)
        let text = h.render().snapshot
        #expect(text.contains("v\(RaoLMVersion.string)"))
        #expect(text.contains("a language model with its citations baked in"))
        #expect(text.contains(h.state.home.runs[0].id))
        #expect(text.contains("✓ complete"))
        #expect(text.contains("◉"))
        #expect(text.contains("1 Home"))
        #expect(text.contains("fixtures"))
    }

    @Test("every screen renders at 80×24 with the compact header")
    func narrow() async throws {
        var h = try Harness()
        try await h.run(.scanRuns)
        for screen in Screen.allCases {
            h.state.screen = screen
            let canvas = h.render(Size(width: 80, height: 24))
            #expect(canvas.line(0).hasPrefix("RaoLM ∿"), "\(screen)")
            #expect(canvas.line(23).contains("thread"), "\(screen)")
        }
    }

    @Test("2 Tests: the doctor's checks")
    func doctor() async throws {
        var h = try Harness()
        try await h.run(.doctor)
        h.state.screen = .doctor
        let text = h.render().snapshot
        #expect(text.contains("✓ fixtures"))
        #expect(text.contains("Test suites"))
    }

    @Test("3 Corpus: the regenerated archive's documents and a partition with its facts")
    func corpus() async throws {
        var h = try Harness()
        try await h.run(.scanCorpora)
        let corpus = h.state.corpus.corpora[0]
        try await h.run(.corpusLoad(corpus.directory))
        h.state.screen = .corpus
        let text = h.render().snapshot
        let first = try #require(h.state.corpus.loaded?.documents.first)
        #expect(text.contains(first.name))
        #expect(text.contains("[p0]"))
        #expect(text.contains("fact "))
    }

    @Test("4 Thread: the node's identity and health")
    func thread() async throws {
        var h = try Harness()
        try await h.run(.threadStatus(withLog: true))
        h.state.screen = .thread
        let text = h.render().snapshot
        let node = try #require(h.state.status.thread?.nodeID)
        #expect(text.contains(node))
        #expect(text.contains("healthy"))
    }

    @Test("5 Train: replaying the ledger fills progress, the loss chart and the epoch table")
    func train() async throws {
        var h = try Harness()
        try await h.run(.scanRuns)
        let spec = TrainSpec(settings: TrainingSettings(), snapshotPath: URL(fileURLWithPath: "/fixture"), factsPath: nil,
                             runID: "replay", runDirectory: h.run, thread: nil)
        try await h.run(.train(spec))
        let live = try #require(h.state.train.live)
        #expect(live.finished != nil)
        #expect(!live.records.isEmpty)
        #expect(live.globalStep == live.totalSteps)
        h.state.screen = .train
        let text = h.render().snapshot
        #expect(text.contains("100%"))
        #expect(text.contains("train loss"))
        #expect(text.contains("Events"))
        #expect(text.unicodeScalars.contains { (0x2800...0x28FF).contains($0.value) })
    }

    @Test("6 Generate: the token strip, the token's numbers, citations and the cited partition")
    func generate() async throws {
        var h = try Harness()
        try await h.run(.scanRuns)
        try await h.run(.loadContext(run: h.run, epoch: nil))
        #expect(h.state.generate.context != nil)
        #expect(!(h.state.generate.context?.examples.isEmpty ?? true))
        let file = try #require(h.generations(grounded: false).first)
        try await h.run(.loadGeneration(file))
        #expect(h.state.screen == .generate)
        let generation = try #require(h.state.generate.generation)
        // Put the cursor on a cited token so the partition panel has something to show.
        let cited = generation.traces.filter { !$0.isPrompt }.firstIndex { !$0.citations.isEmpty } ?? 0
        h.state.generate.cursor = cited
        for job in GenerateScreen.partitionJobs(&h.state) { try await h.run(job) }
        let text = h.render().snapshot
        #expect(text.contains("▸"))
        #expect(text.contains("H_lm"))
        #expect(text.contains("Citations"))
        #expect(text.contains("thread node"))
        #expect(text.contains("raolm://"))
        #expect(!h.state.generate.partitionTexts.isEmpty)

        try await h.key(KeyEvent(.tab))
        #expect(h.state.generate.tab == .neighbours)
        #expect(h.render().snapshot.contains("cited r/o"))
        try await h.key(KeyEvent(.char("v")))
        #expect(h.state.generate.verification.last?.contains("spans verified against the snapshot") == true)
    }

    @Test("6 Generate: a grounded generation shows the class band, ι and the attribution")
    func grounding() async throws {
        var h = try Harness()
        try await h.run(.scanRuns)
        let file = try #require(h.generations(grounded: true).first)
        try await h.run(.loadGeneration(file))
        let record = try #require(h.state.generate.grounding)
        #expect(record.measured)
        h.state.generate.tab = .grounding
        let text = h.render().snapshot
        #expect(text.contains("grounding "))
        #expect(text.contains("ι · KL"))
        #expect(text.contains("A_p"))
        #expect(text.contains("uptake"))
        #expect(text.contains("grounded"))
    }

    @Test("7 Eval: the λ sweep, calibration, controls, grounding groups and outcomes")
    func eval() async throws {
        var h = try Harness()
        try await h.run(.scanRuns)
        h.state.eval.run = h.run
        try await h.run(.eval(EvalSpec(run: h.run, epoch: nil, factsSample: 8, lambdas: [0, 0.5], primaryLambda: 0.5, seed: 7,
                                       controls: true, offline: true, grounding: true, owner: nil)))
        let report = try #require(h.state.eval.report)
        h.state.screen = .eval
        let text = h.render(Size(width: 140, height: 50)).snapshot
        #expect(text.contains("λ sweep"))
        #expect(text.contains("ECE"))
        #expect(text.contains("citation@1"))
        #expect(text.contains("Outcomes · \(report.outcomes.count)"))
        if report.grounding != nil { #expect(text.contains("Grounding (two-fold)")) }
        #expect(report.outcomes.contains { $0.generationFile != nil })
    }

    @Test("9 Braid: a recorded session replays into a panel per node (ambient, craft, veil), the umbrella and the strands")
    func braid() async throws {
        var h = try Harness()
        h.state.screen = .braid
        let idle = h.render().snapshot
        #expect(idle.contains("S starts") && idle.contains("Ambient, Craft, Veil"))
        #expect(idle.contains("RMSNorm (final) → tied head → softmax"))
        try await h.run(.braid(.start(offline: true, fresh: false)))
        #expect(h.state.braid.running)
        #expect(h.state.braid.nodes.map(\.name) == ["ambient", "craft", "veil"])
        #expect(h.state.braid.states.values.allSatisfy { $0.isLive })
        let generation = try #require(h.state.braid.generation)
        #expect(generation.braid?.strands.count == 3)
        #expect(!h.state.braid.examples.isEmpty)
        // Walk to a token one Thread supplied, as the panel colours it.
        let traces = h.state.braid.shownTraces
        let owned = traces.firstIndex { $0.dominantStrand() != nil } ?? 0
        h.state.braid.cursor = h.state.braid.firstAnswer + owned
        let canvas = h.render()
        let text = canvas.snapshot
        if ProcessInfo.processInfo.environment["RAOLM_PRINT_BRAID"] == "1" { print(text) }
        if let path = ProcessInfo.processInfo.environment["RAOLM_DUMP_SCREEN"] {
            try h.dump(h.render(Size(width: 120, height: 40), depth: .trueColor), to: path)
        }
        #expect(text.contains("Umbrella"))
        #expect(text.contains("Ambient") && text.contains("Craft") && text.contains("Veil"))
        #expect(text.contains("gates"))
        #expect(text.contains("cites"))
        #expect(text.contains("◉"))
        #expect(text.contains("9 Braid"))
        #expect(text.contains("snapshot"))
        #expect(text.contains("Tokens so far"))
        // The databases are drawn with half blocks in the strands' colours, three columns wide at 120.
        #expect(text.contains("▀") || text.contains("▄"))
        // Keys: tab moves the focus, x picks an example, ⏎ is refused only when the worker is busy.
        try await h.key(KeyEvent(.tab))
        #expect(h.state.braid.focus == 1)
        try await h.key(KeyEvent(.char("x")))
        #expect(h.state.braid.exampleIndex >= 0 && !h.state.braid.prompt.text.isEmpty)
        try await h.key(KeyEvent(.enter))
        #expect(h.state.braid.generation != nil)
        try await h.key(KeyEvent(.char("v")))
        #expect(!h.state.braid.verification.isEmpty)
        // The prompt is walkable: every Thread scored it, and the gate formed there.
        #expect(!h.state.braid.promptTraces.isEmpty, "the recording keeps the prompt's traces")
        #expect(h.state.braid.cursor == h.state.braid.firstAnswer)
        try await h.key(KeyEvent(.home))
        #expect(h.state.braid.cursor == 0)
        let prompt = h.render().snapshot
        #expect(prompt.contains("prompt 2/"))
        #expect(prompt.contains("next ") && prompt.contains("why ") && prompt.contains("knew "))
        // g cycles the gate, t switches to sampling, a asks each Thread alone as well.
        let gating = h.state.braid.gating
        try await h.key(KeyEvent(.char("g")))
        #expect(h.state.braid.gating != gating)
        try await h.key(KeyEvent(.char("t")))
        #expect(h.state.braid.temperature > 0)
        try await h.key(KeyEvent(.char("a")))
        #expect(h.state.braid.alone.count == 3)
        #expect(h.render().snapshot.contains("alone "))
        // And it fits a small terminal.
        let small = h.render(Size(width: 80, height: 24)).snapshot
        #expect(small.contains("Umbrella") && small.contains("Ambient"))
        // The animation clock moves pulses along and reveals streamed tokens.
        let before = h.state.braid.phase
        _ = StudioApp.handle(.tick(Date()), state: &h.state)
        #expect(h.state.braid.phase == before + 1)
    }

    @Test("8 Ledger: epochs, and a document's partitions")
    func ledger() async throws {
        var h = try Harness()
        try await h.run(.scanRuns)
        h.state.ledger.run = h.run
        try await h.run(.ledger(run: h.run, mode: .epochs, filter: "", partition: nil))
        h.state.screen = .ledger
        #expect(h.render().snapshot.contains("train loss"))
        let document = try #require(h.backend.corpus.documents.first?.id)
        try await h.run(.ledger(run: h.run, mode: .partitions, filter: document, partition: nil))
        #expect(!(h.state.ledger.data?.table.rows.isEmpty ?? true))
    }
}

@Suite("Studio state")
struct StudioStateTests {
    @Test("a braid that refuses to start stops saying 'starting' and says why; the next start clears it")
    func braidStartRefused() {
        var state = StudioState(root: DataRoot(url: URL(fileURLWithPath: "/tmp/raolm-studio-test")))
        state.braid.started = true
        state.braid.nodes = BraidNodeSpec.defaults
        _ = BraidScreen.apply(.failure(nil, message: "ambient was fed raolm-ambient-6a1f99, which this mock world does not have"), state: &state)
        #expect(!state.braid.started && state.braid.startFailure?.hasPrefix("ambient was fed") == true)
        #expect(state.status.error != nil)
        _ = BraidScreen.apply(.starting(Array(BraidNodeSpec.defaults.prefix(2)), offline: true), state: &state)
        #expect(state.braid.started && state.braid.startFailure == nil && state.braid.nodes.count == 2)
        _ = BraidScreen.apply(.note("kept this braid's own nodes"), state: &state)
        #expect(state.status.message == "kept this braid's own nodes")
        // A node's own failure while the braid runs does not stop the braid.
        state.braid.running = true
        _ = BraidScreen.apply(.failure("craft", message: "held"), state: &state)
        #expect(state.braid.started && state.braid.startFailure == nil)
    }

    func state() -> StudioState {
        var state = StudioState(root: DataRoot(url: URL(fileURLWithPath: "/tmp/raolm-studio-test")))
        state.size = Size(width: 120, height: 40)
        return state
    }

    func snapshot(_ state: StudioState) -> String {
        var frame = Frame(canvas: Canvas(size: state.size), palette: Palette(for: Capabilities(isTTY: true, colorDepth: .ansi256), environment: [:]),
                          glyphs: .unicode)
        StudioApp.render(state, &frame)
        return frame.canvas.snapshot
    }

    func type(_ text: String, _ state: inout StudioState) {
        for character in text { _ = StudioApp.handle(.key(KeyEvent(.char(character))), state: &state) }
    }

    @Test("9 Braid: d names the dataset feeds come from, and the braid restarts on it")
    func braidDataset() throws {
        var s = state()
        s.screen = .braid
        #expect(snapshot(s).contains("d names the dataset f and F feed from"))
        #expect(BraidScreen.hints(s).contains(KeyHint("d", "dataset: mock world")))

        // d opens the dialog. ⏎ on nothing is refused there; digits and q are text, not screens or quit.
        _ = StudioApp.handle(.key(KeyEvent(.char("d"))), state: &s)
        guard case .braidDataset(let opened) = s.overlay else {
            Issue.record("d opens the dataset dialog")
            return
        }
        #expect(opened.field.text.isEmpty)
        #expect(StudioApp.handle(.key(KeyEvent(.enter)), state: &s).isEmpty)
        guard case .braidDataset(let refused) = s.overlay else {
            Issue.record("an empty name keeps the dialog open")
            return
        }
        #expect(refused.error != nil)
        let path = "/Volumes/T9/q9/braid-cross-v1"
        type(path, &s)
        let dialog = snapshot(s)
        #expect(dialog.contains("Feed from a dataset") && dialog.contains(path) && dialog.contains("wiped"))

        // ⏎ only looks for the dataset: nothing changes until it is found.
        let look = StudioApp.handle(.key(KeyEvent(.enter)), state: &s)
        guard look.count == 1, case .braid(.dataset(let asked)) = look[0] else {
            Issue.record("⏎ looks for the dataset")
            return
        }
        #expect(asked == path && s.overlay == nil && s.screen == .braid && s.quitCode == nil)
        _ = StudioApp.reduce(.jobFailed(.braid, RaoLMFailure("no braid dataset at \(path)", code: 66)), state: &s)
        #expect(s.status.error != nil && !s.braid.started && s.braid.world.isEmpty)

        // Found, with the braid stopped: it starts on the dataset, and nodes fed from another world start over.
        let first = StudioApp.reduce(.braidDataset(path), state: &s)
        guard first.count == 1, case .braid(.start(let offline, let fresh, let world, let switching, _)) = first[0] else {
            Issue.record("a found dataset starts the braid")
            return
        }
        #expect(!offline && !fresh && switching && world == BraidWorldChoice(dataset: path) && s.braid.started)
        _ = StudioApp.reduce(.braidSource(MockDatasetSource(path: path, name: "braid-cross-v1", hash: "f4bd9ea99d56")), state: &s)
        _ = StudioApp.reduce(.braid(.starting(BraidNodeSpec.defaults, offline: false)), state: &s)

        // While the nodes start, d waits.
        _ = StudioApp.handle(.key(KeyEvent(.char("d"))), state: &s)
        #expect(s.overlay == nil && s.status.message == "the nodes are still starting")
        _ = StudioApp.reduce(.braid(.started), state: &s)
        #expect(BraidScreen.hints(s).contains(KeyHint("d", "dataset")))

        // Running: the dialog shows where feeds come from; another dataset stops the braid, then starts it again.
        _ = StudioApp.handle(.key(KeyEvent(.char("d"))), state: &s)
        guard case .braidDataset(let running) = s.overlay else {
            Issue.record("d opens the dialog while the braid runs")
            return
        }
        #expect(running.field.text == path)
        _ = StudioApp.handle(.key(KeyEvent(.escape)), state: &s)
        #expect(s.overlay == nil && s.braid.world.dataset == path)
        let stop = StudioApp.reduce(.braidDataset("/sets/other"), state: &s)
        guard stop.count == 1, case .braid(.stop) = stop[0] else {
            Issue.record("a running braid stops first")
            return
        }
        #expect(s.braid.restarting)
        let again = StudioApp.reduce(.braid(.stopped), state: &s)
        guard again.count == 1, case .braid(.start(_, _, let next, let switchingAgain, _)) = again[0] else {
            Issue.record("and starts again once stopped")
            return
        }
        #expect(next.dataset == "/sets/other" && switchingAgain && s.braid.started && !s.braid.restarting)

        // An ordinary stop stays stopped; S then starts the chosen dataset, refusing another world as before.
        s.braid.running = true
        let stopped = StudioApp.reduce(.braid(.stopped), state: &s)
        guard stopped.count == 1, case .braid(.catalog) = stopped[0] else {
            Issue.record("an ordinary stop lists the braids again")
            return
        }
        #expect(!s.braid.started)
        #expect(BraidScreen.hints(s).contains(KeyHint("d", "dataset: braid-cross-v1")))
        let plain = StudioApp.handle(.key(KeyEvent(.char("S"))), state: &s)
        guard plain.count == 1, case .braid(.start(_, _, let kept, let switchingPlain, _)) = plain[0] else {
            Issue.record("S starts the braid")
            return
        }
        #expect(kept.dataset == "/sets/other" && !switchingPlain)
    }

    @Test("Ctrl-U empties the braid prompt from anywhere, example and all, and the next examples leave it empty")
    func braidClearPrompt() {
        var s = state()
        s.screen = .braid
        s.braid.started = true
        s.braid.running = true
        let example = BraidExample(label: "Ambient · Tillyburn", node: "ambient", promptTokens: [1, 2, 3],
                                   promptText: "The article says Tillyburn was founded in", expected: " 1128", source: nil, kind: .fact)
        _ = BraidScreen.apply(.examples([example]), state: &s)
        #expect(s.braid.prompt.text == "The article says Tillyburn was founded in" && s.braid.exampleTokens == [1, 2, 3])
        _ = StudioApp.handle(.key(KeyEvent(.ctrl("u"))), state: &s)
        #expect(s.braid.prompt.text.isEmpty && s.braid.exampleTokens == nil && s.braid.exampleLabel == nil && s.braid.editingPrompt)
        #expect(BraidScreen.hints(s).contains { $0.key == "^U" })
        // Leaving the editor empty, a feed's new examples do not put the long example back.
        _ = StudioApp.handle(.key(KeyEvent(.escape)), state: &s)
        _ = BraidScreen.apply(.examples([example, example]), state: &s)
        #expect(s.braid.prompt.text.isEmpty && !s.braid.editingPrompt)
        // Typing works at once after Ctrl-U from browsing.
        _ = StudioApp.handle(.key(KeyEvent(.ctrl("u"))), state: &s)
        for c in "Who is the mayor of Tillyburn?" { _ = StudioApp.handle(.key(KeyEvent(.char(c))), state: &s) }
        #expect(s.braid.prompt.text == "Who is the mayor of Tillyburn?")
    }

    @Test("r turns the question adapter off and on: off, a typed question is completed as written")
    func braidQuestionToggle() {
        var s = state()
        s.screen = .braid
        s.braid.started = true
        s.braid.running = true
        var live = StrandState(name: "ambient", label: "Ambient", offline: true, vocabularySHA256: "v", blocks: 4)
        live.liveVersion = 1
        s.braid.states["ambient"] = live
        s.braid.prompt.set("Who is the mayor of Tillyburn?")
        func asked() -> Bool? {
            let jobs = BraidScreen.generate(&s)
            s.braid.generating = false
            guard jobs.count == 1, case .braid(.generate(let spec)) = jobs[0] else { return nil }
            return spec.question
        }
        #expect(s.braid.questions && asked() == true && BraidScreen.hints(s).contains(KeyHint("r", "Q/A: on")))
        _ = StudioApp.handle(.key(KeyEvent(.char("r"))), state: &s)
        #expect(!s.braid.questions && asked() == false && BraidScreen.hints(s).contains(KeyHint("r", "Q/A: off")))
        _ = StudioApp.handle(.key(KeyEvent(.char("r"))), state: &s)
        #expect(s.braid.questions && asked() == true)
        // A prompt that is not a question is completed either way.
        s.braid.prompt.set("The mayor of Tillyburn is")
        #expect(asked() == false)
    }

    @Test("the commons is shown by its pack name once the braid says which pack it runs")
    func braidCommonsAlone() {
        var s = state()
        s.screen = .braid
        #expect(BraidScreen.label(BraidStrandRef.commonsName, s) == "Commons")
        _ = StudioApp.reduce(.braidCommons("rao-commons-1"), state: &s)
        #expect(s.braid.commonsPack == "rao-commons-1")
        #expect(BraidScreen.label(BraidStrandRef.commonsName, s) == "Commons · rao-commons-1")
        _ = StudioApp.reduce(.braidCommons(nil), state: &s)
        #expect(BraidScreen.label(BraidStrandRef.commonsName, s) == "Commons")
    }

    @Test("9 Braid: the catalog lists every braid; ↑↓ chooses one with its own nodes and mode, ⏎ loads it from its data root")
    func braidCatalog() {
        var s = state()
        s.screen = .braid
        let home = BraidCatalogEntry(
            name: "home", root: URL(fileURLWithPath: "/tmp/raolm-studio-test"), isHome: true,
            nodes: [BraidNodeSpec(name: "ambient", label: "Ambient"), BraidNodeSpec(name: "craft", label: "Craft")], live: ["ambient": 7, "craft": 10],
            preset: "tiny")
        let other = BraidCatalogEntry(
            name: "braid-commons-1", root: URL(fileURLWithPath: "/work/braid-commons-1"), isHome: false, nodes: BraidNodeSpec.defaults,
            live: ["ambient": 3, "craft": 3], preset: "base", arm: "passage-break", commons: "rao-commons-1", commonsSHA256: "ae6f5b5d208a903e",
            dataset: "braid-cross-v1", offline: true)
        // The first catalog chooses the studio's own braid.
        _ = StudioApp.reduce(.braidCatalog([home, other]), state: &s)
        #expect(BraidScreen.selectedBraid(s) == home && s.braid.nodes == home.nodes && !s.braid.offline)
        #expect(BraidScreen.hints(s).contains(KeyHint("⏎", "load")))
        let text = snapshot(s)
        #expect(text.contains("Braids · 2") && text.contains("braid-commons-1") && text.contains("rao-commons-1") && text.contains("2/3 live"))
        // ↓ chooses the next: its nodes, its mode, its own world, never fresh.
        s.braid.fresh = true
        s.braid.world = BraidWorldChoice(dataset: "/sets/home")
        _ = StudioApp.handle(.key(KeyEvent(.down)), state: &s)
        #expect(BraidScreen.selectedBraid(s) == other && s.braid.nodes.count == 3 && s.braid.offline && !s.braid.fresh && s.braid.world.isEmpty)
        #expect(snapshot(s).contains("rao-commons-1 ae6f5b5d208a"))
        // ⏎ loads it from its data root.
        let jobs = StudioApp.handle(.key(KeyEvent(.enter)), state: &s)
        guard jobs.count == 1, case .braid(.start(let offline, let fresh, let world, _, let root)) = jobs[0] else {
            Issue.record("⏎ starts the chosen braid")
            return
        }
        #expect(offline && !fresh && world.isEmpty && root == other.root && s.braid.started && s.braid.loaded == other)
        // Stopped, the catalog is asked for again, and the selection stays on the braid that ran.
        s.braid.running = true
        let stopped = StudioApp.reduce(.braid(.stopped), state: &s)
        guard stopped.count == 1, case .braid(.catalog) = stopped[0] else {
            Issue.record("a stop lists the braids again")
            return
        }
        _ = StudioApp.reduce(.braidCatalog([home, other]), state: &s)
        #expect(BraidScreen.selectedBraid(s) == other && snapshot(s).contains("Braids · 2"))
    }

    @Test("9 Braid: the answer by Thread has a row for the commons when the answer has one")
    func braidCommonsRow() async throws {
        var h = try Harness()
        h.state.screen = .braid
        try await h.run(.braid(.start(offline: true, fresh: false)))
        _ = StudioApp.reduce(.braidCommons("rao-commons-1"), state: &h.state)
        func row(_ text: String) -> Bool { text.split(separator: "\n").contains { $0.contains("Commons · rao-commons-1") && $0.contains("· led ") } }
        // The recording has no commons strand: no row.
        #expect(!row(h.render().snapshot))
        var generation = try #require(h.state.braid.generation)
        // The commons supplies 80% of every answer token, the Threads the rest as they did.
        for i in generation.traces.indices where !generation.traces[i].isPrompt {
            for j in generation.traces[i].strands?.indices ?? 0 ..< 0 { generation.traces[i].strands?[j].share *= 0.2 }
            generation.traces[i].strands?.append(StrandShare(
                strand: BraidStrandRef.commonsName, threadID: nil, gate: 1, open: true, bestScore: nil, lmProb: nil, lmEntropy: nil, knn: 0, share: 0.8))
        }
        h.state.braid.generation = generation
        let text = h.render().snapshot
        #expect(row(text))
        let answered = generation.traces.filter { !$0.isPrompt }.count
        #expect(text.contains("led \(answered)/\(answered)"))
    }

    @Test("a braid start refused before any node came up lets S start again, and says why")
    func braidStartFailedEarly() {
        var s = state()
        s.braid.started = true
        _ = StudioApp.reduce(.jobFailed(.braidUmbrella, RaoLMFailure("no braid dataset at /nowhere/ds", code: 66)), state: &s)
        #expect(!s.braid.started && s.braid.startFailure == "no braid dataset at /nowhere/ds")
        // A generation that fails while the braid runs leaves it running.
        s.braid.started = true
        s.braid.running = true
        s.braid.startFailure = nil
        _ = StudioApp.reduce(.jobFailed(.braidUmbrella, RaoLMFailure("the MLX worker is busy", code: 75)), state: &s)
        #expect(s.braid.started && s.braid.startFailure == nil)
    }

    @Test("a failed job shows its error until the next key")
    func errors() {
        var s = state()
        _ = StudioApp.reduce(.jobStarted(.pull, "pulling"), state: &s)
        #expect(s.status.headline?.label == "pulling")
        _ = StudioApp.reduce(.jobFailed(.pull, RaoLMFailure("no Thread", hint: "start one", code: 69)), state: &s)
        #expect(s.status.error?.code == 69)
        #expect(s.status.jobs.isEmpty)
        _ = StudioApp.handle(.key(KeyEvent(.char("j"))), state: &s)
        #expect(s.status.error == nil)
    }

    @Test("screens switch with 1–9 and [ ], esc goes back, ? opens help")
    func navigation() {
        var s = state()
        _ = StudioApp.handle(.key(KeyEvent(.char("5"))), state: &s)
        #expect(s.screen == .train)
        _ = StudioApp.handle(.key(KeyEvent(.char("]"))), state: &s)
        #expect(s.screen == .generate)
        _ = StudioApp.handle(.key(KeyEvent(.char("["))), state: &s)
        _ = StudioApp.handle(.key(KeyEvent(.char("["))), state: &s)
        #expect(s.screen == .thread)
        _ = StudioApp.handle(.key(KeyEvent(.escape)), state: &s)
        #expect(s.screen == .train)
        _ = StudioApp.handle(.key(KeyEvent(.char("?"))), state: &s)
        if case .help = s.overlay {} else { Issue.record("help did not open") }
        _ = StudioApp.handle(.key(KeyEvent(.char("x"))), state: &s)
        #expect(s.overlay == nil)
    }

    @Test("quitting while an MLX job runs asks, cancels, and quits when the job ends")
    func quitWhileBusy() {
        var s = state()
        _ = StudioApp.reduce(.jobStarted(.train, "training"), state: &s)
        #expect(StudioApp.handle(.key(KeyEvent(.char("q"))), state: &s).isEmpty)
        if case .confirmQuit = s.overlay {} else { Issue.record("no confirmation") }
        let jobs = StudioApp.handle(.key(KeyEvent(.char("y"))), state: &s)
        #expect(jobs.contains { if case .cancel = $0 { return true } else { return false } })
        _ = StudioApp.handle(.tick(Date()), state: &s)
        #expect(s.quitCode == nil)
        _ = StudioApp.reduce(.jobFinished(.train), state: &s)
        _ = StudioApp.handle(.tick(Date()), state: &s)
        #expect(s.quitCode == 0)
    }

    @Test("an MLX action while the worker is busy is refused with a status error")
    func busy() {
        var s = state()
        s.screen = .generate
        s.generate.context = ContextInfo(
            runID: "r", runDirectory: URL(fileURLWithPath: "/tmp/r"), epoch: 1, indexedEpochs: [1],
            manifest: ManifestRef(runID: "r", epoch: 1, checkpointSHA256: "", indexSHA256: "", corpusHash: "", tokenizerSHA256: "", ledgerSHA256: nil, threadID: nil),
            defaults: GenerationParameters(tapLayer: 1, alpha: 0.5), partitionCount: 1, indexEntries: 1, evalMemorised: 1, threadID: nil,
            owner: "o", factsPath: nil, examples: [])
        s.generate.run = s.generate.context?.runDirectory
        s.generate.prompt.set("hello")
        _ = StudioApp.reduce(.jobStarted(.eval, "evaluating"), state: &s)
        let jobs = StudioApp.handle(.key(KeyEvent(.enter)), state: &s)
        #expect(jobs.isEmpty)
        #expect(s.status.error?.code == 75)
    }

    @Test("forms: fields edit, toggles and choices cycle, digits only in numbers, the action submits")
    func forms() {
        var form = FormState([
            FormField("n", "n", .integer, "4"), FormField("on", "on", .toggle, "off"),
            FormField("pick", "pick", .choice(["a", "b"]), "a"), FormField("go", "Go", .action),
        ])
        #expect(form.handle(KeyEvent(.enter)) == .changed)
        #expect(form.editing)
        form.handle(KeyEvent(.char("x")))
        form.handle(KeyEvent(.char("2")))
        form.handle(KeyEvent(.enter))
        #expect(form.int("n") == 42)
        form.handle(KeyEvent(.down))
        form.handle(KeyEvent(.char(" ")))
        #expect(form.bool("on"))
        form.handle(KeyEvent(.down))
        form.handle(KeyEvent(.right))
        #expect(form["pick"] == "b")
        #expect(form.choiceIndex("pick") == 1)
        form.handle(KeyEvent(.down))
        #expect(form.handle(KeyEvent(.enter)) == .submit("go"))
        #expect(form.handle(KeyEvent(.down)) == .unhandled)
    }

    @Test("the strip's [[n]] numbers are the CLI's markers")
    func stripNumbers() async throws {
        var h = try Harness()
        try await h.run(.scanRuns)
        for file in h.generations(grounded: false).prefix(6) {
            let generation = try CitedGeneration.load(from: file)
            let strip = TokenStrip.build(generation).compactMap { cell -> Int? in
                if case .marker(let n, _) = cell { return n } else { return nil }
            }
            let rendered = CitationMarkers.render(generation).text
            let regex = try Regex("\\[\\[([0-9]+)\\]\\]")
            let cli = rendered.matches(of: regex).compactMap { Int(String(rendered[$0.range].dropFirst(2).dropLast(2))) }
            #expect(strip == cli)
        }
    }

    @Test("run scans skip an unreadable run.json with a warning")
    func scan() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("raolm-scan-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let bad = root.appendingPathComponent("runs/broken")
        try FileManager.default.createDirectory(at: bad, withIntermediateDirectories: true)
        try Data("{".utf8).write(to: bad.appendingPathComponent("run.json"))
        let (runs, warnings) = RunSummary.scan(root: DataRoot(url: root))
        #expect(runs.isEmpty)
        #expect(warnings.count == 1)
    }

    @Test("ETA counts the remaining steps and the eval epochs still to come")
    func eta() {
        var live = TrainLive(runID: "r", runDirectory: URL(fileURLWithPath: "/tmp"), startedAt: Date(), summary: [])
        live.totalSteps = 100
        live.epochs = 10
        live.evalEvery = 5
        live.stepSeconds = 0.5
        live.evalOverhead = 3
        live.lastStep = StepRow(epoch: 3, step: 9, globalStep: 29, lr: 1e-3, loss: 1, entropy: EntropySummary(values: [1]),
                                lossMinusEntropy: 0, gradNorm: 1, clipped: false, tokens: 1, maskedTokens: 1, tokensPerSecond: 1, wallClockSeconds: 1)
        // 70 steps left at 0.5 s, and the eval epochs 5 and 10.
        #expect(live.eta == 70 * 0.5 + 2 * 3)
    }
}
