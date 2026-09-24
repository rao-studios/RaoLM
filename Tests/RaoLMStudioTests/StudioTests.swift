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

    func render(_ size: Size = Size(width: 120, height: 40)) -> Canvas {
        let capabilities = Capabilities(isTTY: true, colorDepth: .ansi256)
        var frame = Frame(canvas: Canvas(size: size), palette: Palette(for: capabilities, environment: [:]), glyphs: .unicode)
        StudioApp.render(state, &frame)
        return frame.canvas
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
    func state() -> StudioState {
        var state = StudioState(root: DataRoot(url: URL(fileURLWithPath: "/tmp/raolm-studio-test")))
        state.size = Size(width: 120, height: 40)
        return state
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

    @Test("screens switch with 1–8 and [ ], esc goes back, ? opens help")
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
