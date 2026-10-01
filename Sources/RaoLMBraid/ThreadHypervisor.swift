//
//  ThreadHypervisor.swift
//  RaoLMBraid
//
//  WHAT: What runs inside one Thread node: its headless transformer's versions and the update
//        loop that keeps them current. `sync` exports the node's corpus, diffs it against the
//        live version's snapshot, and climbs the ladder cheapest first — reindex the live
//        weights over the new snapshot (new text citable at once, withdrawn text no longer),
//        then, when enough changed, train the blocks from the live weights against the frozen
//        shared vocabulary — and a candidate goes live only when every gate passes. With an
//        umbrella pack that has a base model, a node's first blocks start as the base's, and the
//        trunk above the cut is frozen with the vocabulary: the node trains blocks 0 ..< cut.
//  OUT:  versions/vNNNN (a standard run directory plus version.json), live.json, and a stream
//        of StrandState snapshots for the panel.
//  PIN:  Everything here runs on the node's one MLX thread. `service` is called between
//        training steps and between ladder stages, so the node keeps answering the umbrella
//        while it trains; a generation that started on the previous version finishes on it.
//

import Foundation
import MLX
import RaoLMCore
import RaoLMModel
import RaoLMProvenance
import RaoLMTraining

public struct HypervisorSettings: Codable, Sendable, Equatable {
    public var preset = "tiny"
    public var batchSize = 4
    public var seqLen = 256
    public var lr: Float = 2e-3
    /// Optimizer steps a node's first version (fresh blocks) may take; the frozen-vocabulary
    /// experiment memorised 24 documents in about 620.
    public var scratchSteps = 1_000
    /// Steps when continuing from the live blocks.
    public var continueSteps = 500
    /// Epoch ceilings on top of the step budgets.
    public var scratchEpochs = 600
    public var continueEpochs = 400
    /// Evaluate about every this many steps (the epoch cadence follows the corpus size).
    public var evalSteps = 40
    /// Training stops early once an evaluation memorises this much.
    public var earlyStop: Float = 0.97
    public var seed: UInt64 = 42
    /// Facts from the changed documents a candidate is probed with before it goes live.
    public var probes = 3
    /// Versions kept whole; older ones keep their records and lose their weights and index.
    public var keepVersions = 3
    /// A trained candidate must memorise this much to go live.
    public var memorisedFloor: Float = VersionGates.memorisedFloor
    /// The most freed Metal memory a node process keeps for reuse, in MB (nil: MLX's default,
    /// which can reach most of the machine's memory). Many nodes share one machine, so each is
    /// capped; the cap changes allocation, never a number the node computes.
    public var cacheLimitMB: Int? = 2048
    /// A model shape other than the preset's (tests use a small one).
    public var config: RaoLMConfig?
    /// The peak learning rate when the blocks start from a pretrained base (nil: 5e-4). A rate
    /// that trains blocks from scratch would wash out what the base knew.
    public var warmLR: Float?
    /// Set λ and the kNN temperature from the corpus's self-trajectory at every version (nil: no).
    public var calibrate: Bool?
    /// A phase-2 training arm (nil: the reference recipe). `passage-break`: a document's
    /// partitions are joined by the tokenizer's paragraph break in every stream, attention stays
    /// inside a document, and no window opens with eos in place of its first token. `canon` and
    /// `gated-attention` train the same stream with Canon layers, or a gate on each attention
    /// head, in the node's blocks. `muon` trains it with Muon for the blocks' weight matrices;
    /// `wsd` with a warmup-stable-decay schedule that anneals once a version has learned.
    public var arm: String?

    /// The arms a braid can be started with.
    public static let arms = ["passage-break", "canon", "gated-attention", "muon", "wsd"]
    /// The arm a new braid of the base preset trains unless another is named: the passage break,
    /// adopted on 2026-10-01 once it held every rule of bench-architecture. A braid recorded
    /// without an arm keeps the recipe it was trained with (eos-first windows on a pack).
    public static let baseArm = "passage-break"

    public init() {}

    /// Whether the nodes train on the passage-break stream: every arm does.
    public var passageBreak: Bool { arm.map(Self.arms.contains) ?? false }

    public func modelConfig() throws -> RaoLMConfig {
        var config = try self.config ?? RaoLMConfig.preset(preset)
        if arm == "canon" { config.canon = true }
        if arm == "gated-attention" { config.attentionGate = true }
        try config.validate()
        return config
    }
}

public final class ThreadHypervisor {
    public let name: String
    public let label: String
    public let layout: NodeLayout
    public let pack: UmbrellaPack
    public var vocabulary: VocabularyPack { pack.vocabulary }
    public let tokenizer: RaoTokenizer
    public let source: CorpusSource
    public let settings: HypervisorSettings
    public let owner: String
    public let config: RaoLMConfig

    public private(set) var live: ThreadStrand?
    public private(set) var liveVersion: NodeVersion?
    public private(set) var state: StrandState
    private var liveSnapshot: CorpusSnapshot?
    /// Memorised fraction per `documentID#index` under the newest evaluated weights, and under the live ones.
    private var memorised: [String: Float] = [:]
    private var liveMemorised: [String: Float] = [:]
    private var lastCellsSnapshot: CorpusSnapshot?

    public var onState: (StrandState) -> Void = { _ in }
    public var onEvent: (NodeEvent) -> Void = { _ in }
    public var service: () -> Void = {}
    public var shouldCancel: () -> Bool = { false }
    /// The standing prompt a candidate completes at each evaluation.
    public var probeTokens: [Int] = []

    public convenience init(
        name: String, label: String, layout: NodeLayout, vocabulary: VocabularyPack, tokenizer: RaoTokenizer,
        source: CorpusSource, settings: HypervisorSettings, owner: String
    ) throws {
        try self.init(name: name, label: label, layout: layout, pack: UmbrellaPack(vocabulary: vocabulary), tokenizer: tokenizer,
                      source: source, settings: settings, owner: owner)
    }

    public init(
        name: String, label: String, layout: NodeLayout, pack: UmbrellaPack, tokenizer: RaoTokenizer,
        source: CorpusSource, settings: HypervisorSettings, owner: String
    ) throws {
        self.name = name
        self.label = label
        self.layout = layout
        self.pack = pack
        self.tokenizer = tokenizer
        self.source = source
        self.settings = settings
        self.owner = owner
        let config = try settings.modelConfig()
        let vocabulary = pack.vocabulary
        guard config.hiddenSize == vocabulary.info.hiddenSize, config.vocabSize == vocabulary.info.vocabSize else {
            throw VocabularyError.shape(expected: [config.vocabSize, config.hiddenSize],
                                        found: [vocabulary.info.vocabSize, vocabulary.info.hiddenSize], what: "vocabulary")
        }
        if let base = pack.info?.config, pack.hasBase {
            guard base.numHiddenLayers == config.numHiddenLayers, base.cut == config.cut, base.intermediateSize == config.intermediateSize,
                  base.numAttentionHeads == config.numAttentionHeads, base.numKeyValueHeads == config.numKeyValueHeads
            else {
                throw UmbrellaPackError.shape(
                    "the node is \(config.numHiddenLayers) blocks cut at \(config.cut), the pack \(base.numHiddenLayers) cut at \(base.cut)")
            }
        }
        self.config = config
        self.state = StrandState(name: name, label: label, offline: source is DirectoryCorpusSource,
                                 vocabularySHA256: vocabulary.sha256, blocks: config.cut)
        state.threadID = source.threadID
        state.packSHA256 = pack.hasTrunk ? pack.sha256 : nil
        state.cut = pack.cut
        try FileManager.default.createDirectory(at: layout.versions, withIntermediateDirectories: true)
        try loadLive()
    }

    /// The pack's sha when it has a trunk: what a node's records and descriptor name beside the vocabulary.
    public var packSHA256: String? { pack.hasTrunk ? pack.sha256 : nil }

    /// A strand over a loaded version, reading the pack's anchors.
    private func strand(version: Int, context: RunContext) -> ThreadStrand {
        let strand = ThreadStrand(name: name, label: label, version: version, context: context, owner: owner)
        strand.anchorTokens = pack.anchors.map(\.tokens)
        return strand
    }

    /// Whether a model holds the pack: its vocabulary, and its trunk when there is one.
    private func checkPack(_ model: RaoTransformer) throws {
        let found = VocabularyPack.fingerprint(of: model)
        guard found == vocabulary.sha256 else { throw VocabularyError.mismatch(node: found, umbrella: vocabulary.sha256) }
        if pack.hasTrunk {
            let trunk = UmbrellaPack.fingerprint(of: model)
            guard trunk == pack.sha256 else { throw UmbrellaPackError.fingerprint(expected: pack.sha256, found: trunk, what: "trunk") }
        }
    }

    // MARK: - Live version

    /// Every version on disk, for the panel's history.
    private func refreshHistory() {
        state.versions = layout.versionNumbers().count
        state.history = layout.versionNumbers().compactMap { number in
            (try? JSONCoding.read(NodeVersion.self, from: layout.version(number).appendingPathComponent(NodeVersion.fileName)))
                .map { VersionMark(version: $0.version, kind: $0.kind, promoted: $0.promoted) }
        }
    }

    private func loadLive() throws {
        refreshHistory()
        guard let pointer = try? JSONCoding.read(LivePointer.self, from: layout.live) else {
            state.stage = .empty
            return
        }
        let directory = layout.version(pointer.version)
        do {
            let version = try JSONCoding.read(NodeVersion.self, from: directory.appendingPathComponent(NodeVersion.fileName))
            let context = try RunContext.load(runDirectory: directory, epoch: version.epoch, allowWeakIndex: true, tokenizer: tokenizer)
            try checkPack(context.model)
            let snapshot = try CorpusSnapshot.load(from: URL(fileURLWithPath: version.snapshotPath))
            adopt(strand(version: version.version, context: context), version: version, snapshot: snapshot)
            if let memorisedMap = try? liveMemorisedMap(directory: directory, epoch: version.epoch, corpus: context.tokenizedCorpus()) {
                liveMemorised = memorisedMap
                memorised = memorisedMap
            }
            refreshCells(snapshot)
            state.stage = .live
        } catch {
            log("v\(pointer.version) could not be loaded (\(error)); the node starts without a live version")
            state.stage = .empty
        }
    }

    private func adopt(_ strand: ThreadStrand, version: NodeVersion, snapshot: CorpusSnapshot) {
        live = strand
        liveVersion = version
        liveSnapshot = snapshot
        state.liveVersion = version.version
        state.checkpointSHA256 = version.checkpointSHA256
        state.indexSHA256 = version.indexSHA256
        state.indexEntries = strand.index.count
        state.memorised = version.memorised
        state.documents = version.documents
        state.partitions = version.partitions
        state.tokens = version.tokens
        state.threadID = strand.threadID ?? source.threadID
        state.heldOutLoss = version.heldOutLoss
        state.commonsLoss = version.commonsLoss
    }

    public func descriptor() -> StrandDescriptor? { live?.descriptor(vocabularySHA256: vocabulary.sha256, packSHA256: packSHA256) }

    /// The processes a node process reports: itself and its Thread.
    public func setProcess(pid: Int32?, threadPID: Int32?, httpPort: Int?, grpcPort: Int?) {
        state.pid = pid
        state.threadPID = threadPID
        state.httpPort = httpPort
        state.grpcPort = grpcPort
    }

    // MARK: - The ladder

    static let ladderNames = ["snapshot", "diff", "reindex", "train", "index", "gates", "live"]

    private func mark(_ name: String, _ status: LadderMark.Status, _ detail: String? = nil) {
        if let i = state.ladder.firstIndex(where: { $0.name == name }) {
            state.ladder[i] = LadderMark(name, status, detail)
        }
    }

    private func stage(_ stage: NodeStage, _ detail: String? = nil) {
        state.stage = stage
        state.stageDetail = detail
        publish()
    }

    public func publish() {
        state.updatedAt = Date()
        onState(state)
    }

    private func log(_ message: String) {
        onEvent(.log(message))
    }

    /// Brings the node up to date with its corpus. Returns the versions it promoted.
    @discardableResult
    public func sync() throws -> [Int] {
        state.error = nil
        state.gates = []
        state.ladder = Self.ladderNames.map { LadderMark($0, .pending) }
        mark("snapshot", .running)
        stage(.exporting, "exporting the corpus")
        let snapshot: CorpusSnapshot
        do {
            let source = self.source
            snapshot = try Blocking.run { try await source.export() }
        } catch {
            fail("export failed: \(error)")
            throw error
        }
        service()
        mark("snapshot", .done, "\(snapshot.documentCount) docs · \(String(snapshot.corpusHash.prefix(8)))")
        let change = snapshot.changes(since: liveSnapshot)
        state.lastChange = change.summary
        mark("diff", .done, change.summary)
        refreshCells(snapshot)

        let plan = UpdatePolicy.plan(change: change, hasLive: live != nil, partitions: snapshot.partitionCount)
        guard !plan.isEmpty else {
            for name in ["reindex", "train", "index", "gates"] { mark(name, .skipped) }
            mark("live", live == nil ? .skipped : .done, live.map { "v\($0.version) unchanged" })
            stage(live == nil ? .empty : .live, snapshot.partitionCount == 0 ? "no documents yet" : "up to date")
            return []
        }
        let snapshotDirectory = layout.snapshot(hash: snapshot.corpusHash)
        let snapshotPath = snapshotDirectory.appendingPathComponent(CorpusSnapshot.fileName)
        if !FileManager.default.fileExists(atPath: snapshotPath.path) { try snapshot.save(to: snapshotDirectory) }

        var promoted: [Int] = []
        do {
            if plan.contains(.reindex), live != nil {
                if let version = try reindex(snapshot, path: snapshotPath.path, change: change) { promoted.append(version) }
            } else {
                mark("reindex", .skipped)
            }
            service()
            if shouldCancel() { throw CancellationError() }
            if plan.contains(.train) {
                // A reindex promoted just now changed nothing the blocks see: diff against the version trained on.
                if let version = try train(snapshot, path: snapshotPath.path, change: change) { promoted.append(version) }
            } else {
                mark("train", .skipped)
            }
        } catch is CancellationError {
            log("update cancelled")
            for mark in state.ladder where mark.status == .running || mark.status == .pending { self.mark(mark.name, .failed, "cancelled") }
            stage(live == nil ? .empty : .live, "cancelled")
            return promoted
        } catch {
            fail("\(error)")
            throw error
        }
        // The ladder is finished: whatever a held candidate or a skipped step left pending is settled.
        for mark in state.ladder where mark.status == .pending { self.mark(mark.name, .skipped) }
        publish()
        return promoted
    }

    /// Epochs for a step budget: small corpora take few steps per epoch, so a fixed epoch count
    /// would starve them. Evaluations follow the same scale (about every `evalSteps` steps).
    static func schedule(tokens: Int, seqLen: Int, batchSize: Int, targetSteps: Int, maxEpochs: Int, evalSteps: Int = 40)
        -> (epochs: Int, evalEvery: Int) {
        let windows = max(1, tokens / max(1, seqLen))
        let stepsPerEpoch = max(1, (windows + batchSize - 1) / batchSize)
        let epochs = min(max(1, maxEpochs), max(8, (targetSteps + stepsPerEpoch - 1) / stepsPerEpoch))
        return (epochs, max(1, evalSteps / stepsPerEpoch))
    }

    private func fail(_ message: String) {
        for mark in state.ladder where mark.status == .running { self.mark(mark.name, .failed) }
        state.error = message
        log(message)
        stage(.failed, message)
    }

    private func nextVersionNumber() -> Int { (layout.versionNumbers().max() ?? 0) + 1 }

    // MARK: - Reindex

    private func reindex(_ snapshot: CorpusSnapshot, path: String, change: SnapshotChange) throws -> Int? {
        guard let live, let parent = liveVersion else { return nil }
        let started = Date()
        let number = nextVersionNumber()
        state.candidateVersion = number
        mark("reindex", .running, "v\(number)")
        stage(.reindexing, "re-keying v\(parent.version)'s weights over \(snapshot.partitionCount) partitions")
        let directory = layout.version(number)
        let parentDirectory = layout.version(parent.version)
        let epoch = 1
        let checkpoint = RunLayout.checkpoint(directory, epoch: epoch)
        try Self.link(RunLayout.checkpoint(parentDirectory, epoch: parent.epoch), to: checkpoint)

        // The stream the live weights were trained and indexed on.
        let corpus = TokenizedCorpus(snapshot: snapshot, tokenizer: tokenizer, paragraphBreak: live.index.info.paragraphBreak ?? [])
        let alpha = live.index.info.alpha
        let evalResult = EvalPass.run(model: live.model, corpus: corpus, seqLen: settings.seqLen, batchSize: 8, captureKeys: true, alpha: alpha)
        service()
        let info = IndexInfo(
            epoch: epoch, tapLayer: live.index.info.tapLayer, alpha: alpha, keyDims: ProvenanceKey.dimensions(for: live.model.config),
            count: evalResult.indexableCount, checkpointSHA256: parent.checkpointSHA256, corpusHash: snapshot.corpusHash,
            tokenizerSHA256: tokenizer.tokenizerSHA256, threadID: snapshot.threadID, evalLoss: evalResult.meanLoss,
            evalMemorisedFraction: evalResult.memorisedFraction, cut: IndexInfo.cut(of: live.model.config),
            paragraphBreak: live.index.info.paragraphBreak)
        let indexDirectory = RunLayout.provenance(directory, epoch: epoch)
        let indexSHA = try ProvenanceIndexer.write(eval: evalResult, corpus: corpus, info: info, memorisedAtEpoch: [:], to: indexDirectory)

        let parentManifest = try RunManifest.load(parentDirectory)
        var manifest = RunManifest(
            runID: String(format: "%@-v%04d", name, number), preset: parentManifest.preset, model: live.model.config,
            tokenizer: tokenizer.ref, corpus: corpusRef(snapshot, path: path, corpus: corpus),
            hyperparameters: parentManifest.hyperparameters, provenance: parentManifest.provenance)
        manifest.vocabularySHA256 = vocabulary.sha256
        manifest.packSHA256 = packSHA256
        manifest.paragraphBreak = parentManifest.paragraphBreak
        manifest.notes.append("Reindex of v\(parent.version): its weights re-keyed over snapshot \(snapshot.corpusHash.prefix(12)); nothing trained.")
        manifest.epochs = [EpochRecord(
            epoch: epoch, steps: 0, trainLoss: .nan, trainEntropy: .nan, evalLoss: evalResult.meanLoss,
            evalEntropy: evalResult.meanEntropy, evalMemorisedFraction: evalResult.memorisedFraction,
            calibrationGap: evalResult.meanLoss - evalResult.meanEntropy, checkpointPath: checkpoint.path,
            checkpointSHA256: parent.checkpointSHA256, indexPath: indexDirectory.path, indexSHA256: indexSHA,
            wallClockSeconds: Date().timeIntervalSince(started))]
        manifest.indexedEpochs = [epoch]
        manifest.status = .complete
        try manifest.save(to: directory)

        record(evalResult: evalResult, corpus: corpus)
        mark("reindex", .done, "v\(number)")
        mark("index", .done, "\(evalResult.indexableCount) entries")
        return try finish(
            kind: .reindex, number: number, directory: directory, epoch: epoch, epochsTrained: 0, snapshot: snapshot, path: path,
            change: change, started: started)
    }

    /// Hard-links a checkpoint's files (a reindex shares its parent's weights), copying if linking fails.
    static func link(_ source: URL, to target: URL) throws {
        let manager = FileManager.default
        try manager.createDirectory(at: target, withIntermediateDirectories: true)
        for name in try manager.contentsOfDirectory(atPath: source.path) {
            let from = source.appendingPathComponent(name)
            let to = target.appendingPathComponent(name)
            if manager.fileExists(atPath: to.path) { try manager.removeItem(at: to) }
            do { try manager.linkItem(at: from, to: to) } catch { try manager.copyItem(at: from, to: to) }
        }
    }

    private func corpusRef(_ snapshot: CorpusSnapshot, path: String, corpus: TokenizedCorpus) -> CorpusRef {
        CorpusRef(
            slug: snapshot.slug, corpusHash: snapshot.corpusHash, snapshotPath: path, source: snapshot.source,
            threadID: snapshot.threadID, owner: snapshot.owner, group: snapshot.group, documentCount: corpus.documents.count,
            partitionCount: corpus.partitions.count, tokenCount: corpus.tokenCount,
            factsPath: FileManager.default.fileExists(atPath: layout.facts.path) ? layout.facts.path : nil)
    }

    // MARK: - Train

    private func train(_ snapshot: CorpusSnapshot, path: String, change: SnapshotChange) throws -> Int? {
        let started = Date()
        let number = nextVersionNumber()
        state.candidateVersion = number
        let parent = liveVersion
        let model: RaoTransformer
        if let parent, let live {
            model = try Checkpoint.load(
                from: RunLayout.checkpoint(layout.version(parent.version), epoch: parent.epoch), tapLayer: live.index.info.tapLayer)
            try checkPack(model)
            // The vocabulary and the trunk stay frozen: only the node's own blocks continue.
            pack.freeze(model)
        } else {
            // Fresh blocks: the base's own when the pack has one (a warm start), else seeded.
            model = try RaoTransformer.make(config: config, seed: settings.seed &+ UInt64(number), tapLayer: config.defaultTapLayer)
            try pack.install(into: model)
        }
        let corpus = TokenizedCorpus(
            snapshot: snapshot, tokenizer: tokenizer, paragraphBreak: settings.passageBreak ? tokenizer.paragraphBreak : [])
        let located = FactLocator.locate(loadFacts(), corpus: corpus, tokenizer: tokenizer).located
        let (epochs, evalEvery) = Self.schedule(
            tokens: corpus.tokenCount, seqLen: settings.seqLen, batchSize: settings.batchSize,
            targetSteps: parent == nil ? settings.scratchSteps : settings.continueSteps,
            maxEpochs: parent == nil ? settings.scratchEpochs : settings.continueEpochs, evalSteps: settings.evalSteps)
        // The node decides when to stop, not the trainer: the whole corpus and what changed must
        // both be learned, or a new document would go live half-remembered.
        var hyper = TrainingHyperparameters(
            batchSize: settings.batchSize, seqLen: settings.seqLen, epochs: epochs, peakLR: pack.hasBase ? (settings.warmLR ?? 5e-4) : settings.lr,
            seed: settings.seed &+ UInt64(number), evalEvery: evalEvery, indexEvery: 0, keepCheckpoints: 1,
            earlyStopMemorised: nil, checkpointEvaluatedOnly: true,
            eosFirstWindows: pack.hasBase && !settings.passageBreak ? true : nil, maskDocuments: settings.passageBreak ? true : nil)
        if settings.arm == "muon" {
            hyper.optimizer = "muon"
            hyper.muonScale = 0.2 * Float(config.hiddenSize).squareRoot()
        }
        // With a warmup-stable-decay schedule a version that learns anneals before it is indexed.
        let stableDecay = settings.arm == "wsd"
        if stableDecay {
            hyper.schedule = "wsd"
            hyper.annealFraction = 0.2
            hyper.annealMinSteps = 50
        }
        let provenance = ProvenanceSettings(tapLayer: model.tapLayer, alpha: 0.5)
        let directory = layout.version(number)
        var manifest = RunManifest(
            runID: String(format: "%@-v%04d", name, number), preset: settings.preset, model: config, tokenizer: tokenizer.ref,
            corpus: corpusRef(snapshot, path: path, corpus: corpus), hyperparameters: hyper, provenance: provenance)
        manifest.vocabularySHA256 = vocabulary.sha256
        manifest.packSHA256 = packSHA256
        if pack.hasTrunk {
            manifest.notes.append(parent.map { "Blocks 0..<\(config.cut) continued from v\($0.version); the vocabulary and the umbrella's trunk stay frozen." }
                ?? "Blocks 0..<\(config.cut) start as the base model's (\(pack.name)); the vocabulary and the umbrella's trunk stay frozen.")
        } else {
            manifest.notes.append(parent.map { "Blocks continued from v\($0.version); the shared vocabulary stays frozen." }
                ?? "Fresh blocks against the shared vocabulary, which stays frozen.")
        }

        let changedKeys = parent == nil ? nil : Set(change.added + change.changed)
        let changedDocuments = Set(change.addedDocuments)
        let changedFacts = Set(located.filter { parent == nil || changedDocuments.contains($0.fact.documentID) }.map(\.fact.id))
        state.epoch = 0
        state.epochs = epochs
        state.step = 0
        state.steps = nil
        state.losses = []
        state.candidateMemorised = nil
        state.factsTotal = located.count
        state.factsLearned = 0
        mark("train", .running, String(format: "v%d · from %@", number, parent.map { "v\($0.version)" } ?? "fresh blocks"))
        stage(.training, parent.map { "continuing v\($0.version)'s blocks" } ?? "fresh blocks")
        var previous = Self.blockParameters(model)
        var learned = false
        var lastEvaluated: EpochRecord?
        let trainer = Pretrainer(
            model: model, corpus: corpus, tokenizer: tokenizer, facts: located, hyper: hyper, provenance: provenance,
            runDirectory: directory, manifest: manifest)
        var result = try trainer.run(shouldStop: { [self] in
            self.service()
            return (learned && !stableDecay) || self.shouldCancel()
        }, shouldAnneal: { learned }) { [self] event in
            switch event {
            case .started(let total, _, _):
                self.state.steps = total
            case .step(let row):
                self.state.step = row.globalStep + 1
                self.state.epoch = row.epoch
                self.state.losses.append(row.loss)
                if self.state.losses.count > 240 { self.state.losses.removeFirst(self.state.losses.count - 240) }
                self.publish()
            case .epoch(let record):
                guard let overall = record.evalMemorisedFraction else { break }
                lastEvaluated = record
                self.state.candidateMemorised = overall
                let factsMemorised = self.evaluated(epoch: record.epoch, runDirectory: directory, corpus: corpus)
                self.state.blockActivity = Self.activity(model: model, previous: &previous)
                self.state.preview = self.preview(model)
                let changed = changedKeys.map { keys in keys.isEmpty ? overall : Stats.mean(keys.map { self.memorised[$0] ?? 0 }) } ?? overall
                let facts = changedFacts.isEmpty ? 1 : Float(factsMemorised.intersection(changedFacts).count) / Float(changedFacts.count)
                if !learned, overall >= self.settings.earlyStop, changed >= self.settings.earlyStop, facts >= 0.9 {
                    learned = true
                    self.log(String(format: "v%d learned after epoch %d: %.1f%% memorised, %.1f%% of what changed, %.0f%% of its facts",
                                    number, record.epoch, overall * 100, changed * 100, facts * 100))
                }
                self.publish()
            case .indexed(let epoch, let entries, _, _):
                self.mark("index", .done, "epoch \(epoch) · \(entries) entries")
                self.stage(.indexing, "indexed epoch \(epoch)")
            case .message(let line):
                self.log("v\(number): \(line)")
            case .earlyStop:
                break
            }
            self.service()
        }
        if result.status == .stopped {
            guard learned, !shouldCancel(), let record = lastEvaluated, let checkpointSHA = record.checkpointSHA256 else {
                mark("train", .failed, "cancelled")
                throw CancellationError()
            }
            // Stopped because it learned: index the epoch it stopped after (the model holds its weights).
            stage(.indexing, "indexing epoch \(record.epoch)")
            let evalResult = EvalPass.run(model: model, corpus: corpus, seqLen: settings.seqLen, batchSize: 8, captureKeys: true, alpha: provenance.alpha)
            let info = IndexInfo(
                epoch: record.epoch, tapLayer: model.tapLayer, alpha: provenance.alpha, keyDims: ProvenanceKey.dimensions(for: model.config),
                count: evalResult.indexableCount, checkpointSHA256: checkpointSHA, corpusHash: snapshot.corpusHash,
                tokenizerSHA256: tokenizer.tokenizerSHA256, threadID: snapshot.threadID, evalLoss: evalResult.meanLoss,
                evalMemorisedFraction: evalResult.memorisedFraction, cut: IndexInfo.cut(of: model.config),
                paragraphBreak: corpus.paragraphBreak.isEmpty ? nil : corpus.paragraphBreak.map(Int.init))
            let indexDirectory = RunLayout.provenance(directory, epoch: record.epoch)
            let sha = try ProvenanceIndexer.write(
                eval: evalResult, corpus: corpus, info: info, memorisedAtEpoch: Self.memorisedAtEpoch(directory: directory, epoch: record.epoch),
                to: indexDirectory)
            if let i = result.epochs.lastIndex(where: { $0.epoch == record.epoch }) {
                result.epochs[i].indexPath = indexDirectory.path
                result.epochs[i].indexSHA256 = sha
            }
            result.indexedEpochs.append(record.epoch)
            mark("index", .done, "epoch \(record.epoch) · \(evalResult.indexableCount) entries")
        }
        result.status = .complete
        try result.save(to: directory)
        guard let epoch = result.latestIndexedEpoch else { throw ProvenanceError.corruptIndex("v\(number) built no index") }
        mark("train", .done, "v\(number) · \(result.epochs.count) epochs")
        return try finish(
            kind: .train, number: number, directory: directory, epoch: epoch, epochsTrained: result.epochs.count, snapshot: snapshot,
            path: path, change: change, started: started)
    }

    /// The first epoch each partition row was memorised at, from the ledger of `epoch`.
    static func memorisedAtEpoch(directory: URL, epoch: Int) -> [Int: Int] {
        let rows = (try? JSONCoding.readLines(PartitionEpochRow.self, from: LedgerFiles.partitions(RunLayout.ledger(directory), epoch: epoch))) ?? []
        var map: [Int: Int] = [:]
        for row in rows { if let at = row.memorisedAtEpoch { map[row.row] = at } }
        return map
    }

    private func loadFacts() -> [Fact] {
        (try? JSONCoding.readLines(Fact.self, from: layout.facts)) ?? []
    }

    /// Per-block parameters of the node's own blocks (below the cut), copied: the optimizer
    /// updates arrays in place.
    static func blockParameters(_ model: RaoTransformer) -> [Int: [MLXArray]] {
        var blocks: [Int: [MLXArray]] = [:]
        for (key, value) in model.parameters().flattened() {
            guard let block = RaoTransformer.blockIndex(ofKey: key), block < model.cut else { continue }
            blocks[block, default: []].append(value + 0)
        }
        for arrays in blocks.values { eval(arrays) }
        return blocks
    }

    /// How far each block moved since `previous` (relative to its size), scaled to the most-moved block.
    static func activity(model: RaoTransformer, previous: inout [Int: [MLXArray]]) -> [Float] {
        let now = blockParameters(model)
        var moved: [Float] = []
        for block in now.keys.sorted() {
            guard let old = previous[block], old.count == now[block]?.count else { moved.append(0); continue }
            var delta = MLXArray(Float(0))
            var size = MLXArray(Float(0))
            for (a, b) in zip(now[block]!, old) {
                delta = delta + ((a - b) * (a - b)).sum()
                size = size + (b * b).sum()
            }
            let ratio = MLX.sqrt(delta / MLX.maximum(size, MLXArray(Float(1e-12))))
            eval(ratio)
            moved.append(ratio.item(Float.self))
        }
        previous = now
        let top = moved.max() ?? 0
        return moved.map { top > 0 ? $0 / top : 0 }
    }

    /// The model's own greedy completion of the standing prompt: what it has learned, without retrieval.
    private func preview(_ model: RaoTransformer, maxTokens: Int = 12) -> String? {
        let prompt = probeTokens
        guard !prompt.isEmpty else { return nil }
        let cache = model.newCache(parameters: nil)
        var logits = model.forward(MLXArray(prompt.map { Int32($0) }, [1, prompt.count]), cache: cache, captureTap: false)
            .logits[0, prompt.count - 1]
        var generated: [Int] = []
        for _ in 0..<maxTokens {
            let next = argMax(logits).item(Int.self)
            if next == tokenizer.eosTokenID { break }
            generated.append(next)
            logits = model.forward(MLXArray([Int32(next)], [1, 1]), cache: cache, captureTap: false).logits[0, 0]
        }
        return tokenizer.decode(generated)
    }

    // MARK: - Evaluation → cells

    /// Reads an evaluated epoch's ledger into the cells; returns the facts it had memorised.
    @discardableResult
    private func evaluated(epoch: Int, runDirectory: URL, corpus: TokenizedCorpus) -> Set<String> {
        let ledger = RunLayout.ledger(runDirectory)
        if let rows = try? JSONCoding.readLines(PartitionEpochRow.self, from: LedgerFiles.partitions(ledger, epoch: epoch)) {
            for row in rows {
                if let fraction = row.eval?.memorisedFraction { memorised["\(row.documentID)#\(row.partitionIndex)"] = fraction }
            }
        }
        var learned = Set<String>()
        if let facts = try? JSONCoding.readLines(FactEpochRow.self, from: LedgerFiles.facts(ledger, epoch: epoch)) {
            learned = Set(facts.filter(\.memorised).map(\.factID))
            state.factsLearned = learned.count
            state.factsTotal = facts.count
        }
        if let snapshot = lastCellsSnapshot { refreshCells(snapshot) }
        return learned
    }

    private func record(evalResult: EvalResult, corpus: TokenizedCorpus) {
        for (row, stats) in evalResult.partitionStats(rowCount: corpus.partitions.count).enumerated() {
            guard let stats else { continue }
            let partition = corpus.partitions[row]
            memorised["\(partition.documentID)#\(partition.partitionIndex)"] = stats.memorisedFraction
        }
    }

    private func liveMemorisedMap(directory: URL, epoch: Int, corpus: TokenizedCorpus) throws -> [String: Float] {
        let rows = try JSONCoding.readLines(PartitionEpochRow.self, from: LedgerFiles.partitions(RunLayout.ledger(directory), epoch: epoch))
        var map: [String: Float] = [:]
        for row in rows { if let fraction = row.eval?.memorisedFraction { map["\(row.documentID)#\(row.partitionIndex)"] = fraction } }
        return map
    }

    /// One cell per partition the node holds now, in the order the feeder deposited documents.
    private func refreshCells(_ snapshot: CorpusSnapshot) {
        lastCellsSnapshot = snapshot
        let order = FeedState.load(layout).deposited
        let rank = Dictionary(order.enumerated().map { ($0.element, $0.offset) }, uniquingKeysWith: { a, _ in a })
        let documents = snapshot.documents.sorted { (rank[$0.id] ?? Int.max, $0.id) < (rank[$1.id] ?? Int.max, $1.id) }
        let indexed = Set(live?.index.partitions.map { "\($0.documentID)#\($0.partitionIndex)" } ?? [])
        var cells: [PartitionCell] = []
        for (d, document) in documents.enumerated() {
            for partition in document.partitions {
                let key = "\(document.id)#\(partition.index)"
                let fraction = memorised[key]
                let state: PartitionCell.State = !indexed.contains(key) ? .pending
                    : ((liveMemorised[key] ?? 0) >= 0.95 ? .learned : .indexed)
                cells.append(PartitionCell(documentID: document.id, document: d, partition: partition.index, state: state,
                                           memorised: fraction, glyphs: String(partition.text.prefix(18))))
            }
        }
        state.cells = cells
    }

    // MARK: - Gates and promotion

    private func finish(
        kind: UpdateKind, number: Int, directory: URL, epoch: Int, epochsTrained: Int, snapshot: CorpusSnapshot, path: String,
        change: SnapshotChange, started: Date
    ) throws -> Int? {
        mark("gates", .running)
        stage(.gating, "gating v\(number)")
        var gates: [GateResult] = []
        var context: RunContext
        do {
            context = try RunContext.load(runDirectory: directory, epoch: epoch, allowWeakIndex: true, tokenizer: tokenizer)
            gates.append(GateResult("chain", true, "index ↔ checkpoint \(context.checkpointSHA256.prefix(8)) ↔ tokenizer"))
        } catch {
            gates.append(GateResult("chain", false, "\(error)"))
            return hold(kind: kind, number: number, directory: directory, epoch: epoch, gates: gates, snapshot: snapshot,
                        path: path, change: change, started: started, epochsTrained: epochsTrained)
        }
        if settings.calibrate == true {
            // λ and τ from how the corpus traces itself, written into the index the version serves.
            stage(.gating, "v\(number): reading the corpus's self-trajectory")
            let calibration = SelfTrajectory.calibrate(
                model: context.model, index: context.index, corpus: try context.tokenizedCorpus(), alpha: context.index.info.alpha)
            var info = context.index.info
            info.calibration = calibration
            try JSONCoding.write(info, to: RunLayout.provenance(directory, epoch: epoch).appendingPathComponent(ProvenanceIndexFiles.info))
            context = try RunContext.load(runDirectory: directory, epoch: epoch, allowWeakIndex: true, tokenizer: tokenizer)
            service()
        }
        gates.append(VersionGates.vocabulary(found: VocabularyPack.fingerprint(of: context.model), expected: vocabulary.sha256))
        if pack.hasTrunk {
            gates.append(VersionGates.trunk(found: UmbrellaPack.fingerprint(of: context.model), expected: pack.sha256))
        }
        let fraction = context.index.info.evalMemorisedFraction
        if let gate = VersionGates.memorised(fraction, kind: kind, floor: settings.memorisedFloor) { gates.append(gate) }
        gates.append(VersionGates.withdrawn(indexDocuments: Set(context.index.partitions.map(\.documentID)), removed: change.removedDocuments))
        service()
        gates.append(VersionGates.verified(statuses: try probe(context, change: change)))
        service()
        state.gates = gates

        let strand = self.strand(version: number, context: context)
        let corpus = try context.tokenizedCorpus()
        let losses = heldOutLosses(context.model, paragraphBreak: context.index.info.paragraphBreak ?? [])
        service()
        var version = NodeVersion(
            version: number, parent: liveVersion?.version, kind: kind, snapshotBefore: liveSnapshot?.corpusHash,
            snapshotAfter: snapshot.corpusHash, snapshotPath: path, change: change, vocabularySHA256: vocabulary.sha256,
            checkpointSHA256: context.checkpointSHA256, indexSHA256: context.index.sha256, epoch: epoch, epochsTrained: epochsTrained,
            memorised: fraction, evalLoss: context.index.info.evalLoss, documents: corpus.documents.count,
            partitions: corpus.partitions.count, tokens: corpus.tokenCount, gates: gates, promoted: gates.allSatisfy(\.passed),
            seconds: Date().timeIntervalSince(started), createdAt: .wholeSecond())
        version.packSHA256 = packSHA256
        version.heldOutLoss = losses.own
        version.commonsLoss = losses.commons
        version.calibration = context.index.info.calibration
        try JSONCoding.write(version, to: directory.appendingPathComponent(NodeVersion.fileName))
        refreshHistory()
        guard version.promoted else {
            return hold(version: version)
        }
        try JSONCoding.write(LivePointer(version: number, promotedAt: .wholeSecond()), to: layout.live)
        adopt(strand, version: version, snapshot: snapshot)
        liveMemorised = memorised
        refreshCells(snapshot)
        state.candidateVersion = nil
        state.candidateMemorised = nil
        mark("gates", .done, "\(gates.count)/\(gates.count)")
        mark("live", .done, "v\(number) · \(kind.rawValue)")
        onEvent(.promoted(version: number, kind: kind))
        log(String(format: "v%d live (%@): %d docs, %d partitions, memorised %.1f%%", number, kind.rawValue, version.documents,
                   version.partitions, fraction * 100))
        stage(.live, "v\(number) live")
        retire()
        return number
    }

    private func hold(
        kind: UpdateKind, number: Int, directory: URL, epoch: Int, gates: [GateResult], snapshot: CorpusSnapshot, path: String,
        change: SnapshotChange, started: Date, epochsTrained: Int
    ) -> Int? {
        let version = NodeVersion(
            version: number, parent: liveVersion?.version, kind: kind, snapshotBefore: liveSnapshot?.corpusHash,
            snapshotAfter: snapshot.corpusHash, snapshotPath: path, change: change, vocabularySHA256: vocabulary.sha256,
            checkpointSHA256: "", indexSHA256: "", epoch: epoch, epochsTrained: epochsTrained, memorised: 0, evalLoss: .nan,
            documents: snapshot.documentCount, partitions: snapshot.partitionCount, tokens: 0, gates: gates, promoted: false,
            seconds: Date().timeIntervalSince(started), createdAt: .wholeSecond())
        try? JSONCoding.write(version, to: directory.appendingPathComponent(NodeVersion.fileName))
        refreshHistory()
        state.gates = gates
        return hold(version: version)
    }

    private func hold(version: NodeVersion) -> Int? {
        let failed = version.gates.filter { !$0.passed }
        let reason = failed.map { "\($0.name): \($0.detail)" }.joined(separator: "; ")
        memorised = liveMemorised
        if let snapshot = lastCellsSnapshot { refreshCells(snapshot) }
        state.candidateVersion = nil
        mark("gates", .failed, failed.map(\.name).joined(separator: ", "))
        mark("live", .skipped, live.map { "v\($0.version) keeps serving" })
        onEvent(.held(version: version.version, reason: reason))
        log("v\(version.version) held: \(reason)")
        stage(.held, "v\(version.version) held · \(failed.map(\.name).joined(separator: ", "))")
        return nil
    }

    /// A model's mean loss on the unfed documents the feeder left beside the node (its own
    /// voice), and on the pack's commons sample. Own documents are read as the version's stream reads them.
    func heldOutLosses(_ model: RaoTransformer, paragraphBreak: [Int] = []) -> (own: Float?, commons: Float?) {
        let documents = (try? JSONCoding.readLines(CorpusDocument.self, from: layout.heldOut)) ?? []
        let eos = Int32(tokenizer.eosTokenID)
        let own = documents.isEmpty ? nil
            : HeldOut.loss(
                model: model, texts: documents.map { HeldOut.tokens($0, tokenizer: tokenizer, paragraphBreak: paragraphBreak) }, eos: eos,
                seqLen: settings.seqLen)
        let commons = pack.heldOut.isEmpty ? nil : HeldOut.loss(model: model, texts: pack.heldOut.map(\.tokens), eos: eos, seqLen: settings.seqLen)
        return (own, commons)
    }

    /// Probes a candidate with facts from the changed documents; the spans it emits must verify
    /// against the node's corpus as it is now.
    private func probe(_ context: RunContext, change: SnapshotChange) throws -> [VerificationStatus] {
        guard settings.probes > 0 else { return [] }
        let corpus = try context.tokenizedCorpus()
        let changed = Set(change.addedDocuments)
        let facts = loadFacts()
        var located = FactLocator.locate(facts.filter { changed.contains($0.documentID) }, corpus: corpus, tokenizer: tokenizer).located
        if located.isEmpty { located = FactLocator.locate(facts, corpus: corpus, tokenizer: tokenizer).located }
        var statuses: [VerificationStatus] = []
        let reader = source.reader()
        let tokenizer = self.tokenizer
        for fact in located.prefix(settings.probes) {
            let partition = corpus.partitions[fact.row]
            let prompt = partition.tokens[fact.contextToken..<fact.answerToken].map(Int.init)
            var params = context.defaultParameters()
            params.maxTokens = fact.answerLength + 2
            let generation = try context.generator().generate(GenerationRequest(
                promptTokens: prompt, promptText: tokenizer.decode(prompt), params: params))
            let report = try Blocking.run {
                var copy = generation
                return try await CitationVerifier.verify(&copy, reader: reader, tokenizer: tokenizer)
            }
            statuses += report.checks.filter { $0.kind == .verbatim }.map(\.status)
            service()
        }
        return statuses
    }

    /// Keeps the newest versions whole (and the live one); older ones keep their records only.
    private func retire() {
        let numbers = layout.versionNumbers()
        let keep = Set(numbers.suffix(max(1, settings.keepVersions))).union([liveVersion?.version].compactMap { $0 })
        for number in numbers where !keep.contains(number) {
            let directory = layout.version(number)
            for sub in ["checkpoints", "provenance"] {
                try? FileManager.default.removeItem(at: directory.appendingPathComponent(sub, isDirectory: true))
            }
        }
    }
}
