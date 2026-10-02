//
//  StudioState.swift
//  RaoLMStudio
//
//  WHAT: Everything the studio shows, as one value: the current screen and overlay, the
//        status bar, and each screen's own state.
//

import Foundation
import RaoLM
import RaoLMTerminal
import RaoLMWorkflows

public enum Screen: Int, CaseIterable, Sendable {
    case home = 1, doctor, corpus, thread, train, generate, eval, ledger, braid

    public var title: String {
        switch self {
        case .home: return "Home"
        case .doctor: return "Tests"
        case .corpus: return "Corpus"
        case .thread: return "Thread"
        case .train: return "Train"
        case .generate: return "Generate"
        case .eval: return "Eval"
        case .ledger: return "Ledger"
        case .braid: return "Braid"
        }
    }

    /// The screens a job's progress shows on.
    public static func owner(of kind: JobKind) -> Screen {
        switch kind {
        case .doctor, .tests: return .doctor
        case .threadStatus, .threadStart, .threadStop: return .thread
        case .scanCorpora, .corpusGenerate, .corpusLoad, .ingest, .pull, .offlineSnapshot: return .corpus
        case .scanSnapshots, .train: return .train
        case .loadContext, .generate, .verify, .saveGeneration, .loadGeneration, .listGenerations, .partitionText, .ground: return .generate
        case .eval: return .eval
        case .ledger: return .ledger
        case .scanRuns: return .home
        case .braid, .braidUmbrella: return .braid
        }
    }
}

public struct Picker: Sendable {
    public enum Purpose: Sendable, Equatable { case run(Screen), generation }
    public var title: String
    public var items: [String]
    public var values: [URL]
    public var table: TableState
    public var purpose: Purpose
}

/// One line of text asked for in a dialog: ⏎ takes it, esc leaves things as they were.
public struct InputPrompt: Sendable {
    public var title: String
    public var help: [String]
    public var field: TextFieldState
    /// Why the last ⏎ was refused.
    public var error: String?
}

public enum Overlay: Sendable {
    case help
    case confirmQuit
    case confirmStopThread
    case cancelling(since: Date)
    case picker(Picker)
    case params(FormState)
    /// The dataset the braid's feeds come from (d on the braid panel).
    case braidDataset(InputPrompt)
}

public struct JobStatus: Sendable, Equatable {
    public var kind: JobKind
    public var label: String
    public var since: Date
}

public struct StatusState: Sendable {
    public var thread: ThreadStatus?
    public var jobs: [JobKind: JobStatus] = [:]
    public var message: String?
    public var error: RaoLMFailure?
    public var lastThreadPoll: Date?

    /// The job the status bar shows: the MLX job first, then any visible one.
    public var headline: JobStatus? {
        jobs.values.filter { $0.kind.isMLX }.first
            ?? jobs.values.filter { $0.kind.isVisible }.sorted { $0.since < $1.since }.first
    }

    public var mlxJob: JobStatus? { jobs.values.first { $0.kind.isMLX } }
}

public struct HomeState: Sendable {
    public var runs: [RunSummary] = []
    public var warnings: [String] = []
    public var table = TableState(selected: 0)
}

public struct DoctorState: Sendable {
    public var checks: [DoctorCheck] = []
    public var filter = TextFieldState("")
    public var editingFilter = false
    public var mlx = true
    public var thread = false
    public var log = LogState(capacity: 5000)
    public var running = false
    public var exitCode: Int32?
    public var startedAt: Date?
}

public struct CorpusState: Sendable {
    public enum Pane: Int, Sendable, CaseIterable { case form, corpora, documents }
    public var form = FormState.corpus()
    public var pane: Pane = .form
    public var corpora: [CorpusSummary] = []
    public var corporaTable = TableState(selected: 0)
    public var loaded: GeneratedCorpus?
    public var loadedDirectory: URL?
    public var documentsTable = TableState(selected: 0)
    public var lines: [(String, Bool)] = []
    public var ingest: (done: Int, total: Int)?
}

public struct ThreadScreenState: Sendable {
    public var fresh = false
    public var log = LogState(capacity: 400)
}

public struct TrainLive: Sendable {
    public var runID: String
    public var runDirectory: URL
    public var startedAt: Date
    public var summary: [String]
    public var totalSteps = 0
    public var stepsPerEpoch: [Int] = []
    public var epochs = 0
    public var evalEvery = 0
    public var lastStep: StepRow?
    public var losses = RingBuffer<Double>(capacity: 4096)
    public var entropies = RingBuffer<Double>(capacity: 4096)
    public var gradNorms = RingBuffer<Double>(capacity: 512)
    public var tokensPerSecond = RingBuffer<Double>(capacity: 512)
    public var lrs = RingBuffer<Double>(capacity: 512)
    public var records: [EpochRecord] = []
    public var stepSeconds: Double?
    public var evalOverhead: Double = 0
    public var events = LogState(capacity: 400)
    public var finished: RunManifest?
    public var cancelRequested = false

    public var globalStep: Int { (lastStep?.globalStep ?? -1) + 1 }

    /// Remaining steps at the measured step rate, plus the eval epochs still to come at the
    /// overhead the last eval epoch took.
    public var eta: Double? {
        guard let stepSeconds, finished == nil, totalSteps > 0 else { return nil }
        let remainingSteps = max(0, totalSteps - globalStep)
        let currentEpoch = lastStep?.epoch ?? 1
        let remainingEvals = ((currentEpoch)...max(currentEpoch, epochs)).filter {
            $0 == epochs || (evalEvery > 0 && $0 % evalEvery == 0)
        }.filter { epoch in !records.contains { $0.epoch == epoch } }.count
        return Double(remainingSteps) * stepSeconds + Double(remainingEvals) * evalOverhead
    }
}

public struct TrainState: Sendable {
    public var form = FormState.training(snapshots: [])
    public var snapshots: [SnapshotChoice] = []
    public var live: TrainLive?
}

public struct GenerateState: Sendable {
    public enum Tab: Int, Sendable, CaseIterable { case citations, neighbours, spans, grounding
        public var title: String {
            switch self {
            case .citations: return "Citations"
            case .neighbours: return "Neighbours"
            case .spans: return "Spans"
            case .grounding: return "Grounding"
            }
        }
    }

    public var run: URL?
    public var context: ContextInfo?
    public var prompt = TextFieldState("")
    public var editingPrompt = false
    public var sliceMode = false
    public var exampleIndex = -1
    public var overrides = GenerationOverrides()
    public var streaming: [TokenTrace] = []
    public var generation: CitedGeneration?
    public var generationFile: URL?
    public var neighbourRows: [[NeighbourRow]] = []
    public var cursor = 0
    public var tab: Tab = .citations
    public var tables: [Tab: TableState] = [:]
    public var partitionTexts: [String: PartitionText] = [:]
    public var requestedTexts: Set<String> = []
    public var verification: [String] = []
    public var verificationSource: String?
    public var grounding: GroundingRecord?

    public func table(_ tab: Tab) -> TableState { tables[tab] ?? TableState(selected: 0) }
}

public struct EvalState: Sendable {
    public var run: URL?
    public var form = FormState.eval()
    public var progress = LogState(capacity: 400)
    public var report: EvalReport?
    public var savedTo: URL?
    public var outcomes = TableState(selected: 0)
    public var focusOutcomes = false
}

public struct LedgerState: Sendable {
    public var run: URL?
    public var mode: LedgerMode = .epochs
    public var filter = TextFieldState("")
    public var partition = TextFieldState("")
    public var editing: Int?  // 0 filter, 1 partition
    public var data: LedgerData?
    public var table = TableState(selected: 0)
}

/// The braid panel: the nodes as they report themselves, the umbrella's last generation, and
/// the animation clock.
public struct BraidScreenState: Sendable {
    /// A pulse of one generated token: up a node's block stack, along its strand, into the umbrella.
    public struct Pulse: Sendable, Equatable {
        public var strand: String
        public var start: Int
        public var strength: Float
        public var open: Bool
    }

    public var started = false
    public var running = false
    /// Why the last start was refused, until the next one.
    public var startFailure: String?
    public var offline = false
    public var fresh = false
    public var autoStart = false
    /// The nodes and dataset a start asks for: what the studio was opened with, until a dataset
    /// is chosen on the panel (d) or another braid in the catalog.
    public var world = BraidWorldChoice()
    /// What the studio was opened with, for its own braid when it is chosen again.
    public var launchWorld = BraidWorldChoice()
    /// Every braid on this machine, the studio's own first; the selected one is what a start loads.
    public var catalog: [BraidCatalogEntry] = []
    public var catalogTable = TableState(selected: 0)
    /// The braid running, or last started (nil before the first start).
    public var loaded: BraidCatalogEntry?
    /// The dataset the running braid feeds its nodes from; nil: a generated mock world.
    public var source: MockDatasetSource?
    /// The braid is stopping to start again on the dataset just chosen.
    public var restarting = false
    public var nodes: [BraidNodeSpec] = []
    public var states: [String: StrandState] = [:]
    public var pids: [String: Int32] = [:]
    public var focus = 0
    public var prompt = TextFieldState("")
    public var editingPrompt = false
    public var examples: [BraidExample] = []
    public var exampleIndex = -1
    /// The corpus tokens of the chosen example, while the prompt field still shows its text.
    public var exampleTokens: [Int]?
    public var exampleSource: SourceAddress?
    public var exampleExpected: String?
    public var exampleLabel: String?
    /// The Thread holding the chosen example's answer (nil: an entity no Thread holds).
    public var exampleNode: String?
    public var streaming: [BraidTokenEvent] = []
    /// The prompt's traces: each prompt token as every Thread predicted it, and where the gate stood.
    public var promptTraces: [TokenTrace] = []
    /// The prompt's first token, which has no trace.
    public var promptFirst: String?
    public var generation: CitedGeneration?
    public var generating = false
    public var revealed = 0
    /// Into `walkable`: the prompt's traces, then the generated ones.
    public var cursor = 0
    /// How the next generation weighs Threads, whether it samples, and whether each Thread also
    /// answers alone.
    public var gating: BraidGating = BraidRequest.defaultGating
    public var temperature: Float = 0
    public var askAlone = false
    /// Each Thread's own answer to the current generation's prompt.
    public var alone: [String: String] = [:]
    /// The commons' pack by name ("rao-commons-1"); nil when the braid runs none.
    public var commonsPack: String?
    public var verification: [String] = []
    public var verifiedAll: Bool?
    public var events = LogState(capacity: 400)
    public var phase = 0
    public var pulses: [Pulse] = []
    /// When each node was last fed (animation phase), for the ingest rain.
    public var fedAt: [String: Int] = [:]
    public var lambda: Float = 0.5

    public init() {}

    public var focusedNode: String? { focus < nodes.count ? nodes[focus].name : nil }

    /// The generated traces the strip has revealed so far.
    public var shownTraces: [TokenTrace] {
        if let generation { return generation.traces.filter { !$0.isPrompt } }
        return Array(streaming.prefix(revealed).map(\.trace))
    }

    /// What the cursor walks: the prompt's traces, then the generated ones.
    public var walkable: [TokenTrace] { promptTraces + shownTraces }

    /// The cursor's position of the first generated token.
    public var firstAnswer: Int { promptTraces.count }
}

public struct StudioState: Sendable {
    public var root: DataRoot
    public var fixtures: Bool
    public var version: String
    public var owner: String
    public var screen: Screen = .home
    public var history: [Screen] = []
    public var overlay: Overlay?
    public var status = StatusState()
    public var home = HomeState()
    public var doctor = DoctorState()
    public var corpus = CorpusState()
    public var thread = ThreadScreenState()
    public var train = TrainState()
    public var generate = GenerateState()
    public var eval = EvalState()
    public var ledger = LedgerState()
    public var braid = BraidScreenState()
    public var size = Size(width: 120, height: 40)
    public var now = Date()
    public var spinner = 0
    public var quitCode: Int32?
    public var quitAfterJobs = false
    var _quitAfterJobsStoppedThread = false
    public var redraw = false

    public init(root: DataRoot, fixtures: Bool = false, version: String = RaoLMVersion.string, owner: String = "raolm-demo") {
        self.root = root
        self.fixtures = fixtures
        self.version = version
        self.owner = owner
    }

    /// A text field or form field is taking keystrokes: global keys are off.
    public var editing: Bool {
        if case .params(let form) = overlay { return form.editing }
        if case .braidDataset = overlay { return true }
        switch screen {
        case .doctor: return doctor.editingFilter
        case .corpus: return corpus.form.editing
        case .train: return train.form.editing
        case .generate: return generate.editingPrompt
        case .eval: return eval.form.editing
        case .ledger: return ledger.editing != nil
        case .braid: return braid.editingPrompt
        default: return false
        }
    }

    /// The run a screen works on: its own choice, else the one selected on Home.
    public var selectedRun: RunSummary? {
        guard let selected = home.table.selected, selected < home.runs.count else { return nil }
        return home.runs[selected]
    }

    public func run(at url: URL?) -> RunSummary? {
        guard let url else { return nil }
        return home.runs.first { $0.directory.standardizedFileURL == url.standardizedFileURL }
    }
}
