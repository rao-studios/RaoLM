//
//  BraidScreen.swift
//  RaoLMStudio
//
//  WHAT: 9 Braid: decentralised intelligence, drawn. Two (or more) Thread nodes, each its own
//        process with a Thread of its own, each hosting a headless transformer — the token
//        embedding through the last block — beside its provenance database. Above them one
//        umbrella: the shared final norm, the tied head, softmax and p = λ·p_knn + (1−λ)·p_lm.
//        Feed a node and watch its ladder climb (snapshot, diff, reindex, train, index, gates,
//        live) and its database fill; prompt the umbrella and see, token by token, which Thread
//        supplied it, which gates opened, and which partition it is cited to.
//

import Foundation
import RaoLM
import RaoLMTerminal
import RaoLMWorkflows

enum BraidScreen: StudioScreen {
    /// "A", "A and B", "A, B and C".
    static func listed(_ items: [String]) -> String {
        guard items.count > 1 else { return items.first ?? "" }
        return items.dropLast().joined(separator: ", ") + " and " + items[items.count - 1]
    }

    /// Ticks one token pulse takes: up the block stack, then along the strand to the ring.
    static let pulseTicks = 10

    // MARK: - Events

    static func apply(_ event: BraidEvent, state: inout StudioState) -> [StudioJob] {
        switch event {
        case .starting(let specs, let offline):
            state.braid.nodes = specs
            state.braid.started = true
            state.braid.running = false
            state.braid.startFailure = nil
            state.braid.offline = offline
            state.braid.states = [:]
            state.braid.events.append("starting \(listed(specs.map(\.label))) · \(offline ? "offline corpora" : "a Thread each")")
        case .spawned(let name, let pid):
            state.braid.pids[name] = pid
            state.braid.events.append("\(name): node process pid \(pid)")
        case .ready(let name, let hello):
            state.braid.states[name] = hello.state
            state.braid.events.append("\(name): ready · " + (hello.offline ? "offline corpus" : "Thread pid \(hello.threadPID ?? 0) :\(hello.httpPort ?? 0)")
                                      + " · live \(hello.liveVersion.map { "v\($0)" } ?? "none")")
        case .node(let name, let nodeEvent):
            switch nodeEvent {
            case .state(let s):
                state.braid.states[name] = s
            case .log(let line):
                state.braid.events.append("\(name): \(line)")
            case .promoted(let version, let kind):
                state.braid.events.append(Text("▲ \(name) v\(version) live (\(kind.rawValue))", style: .plain))
                state.status.message = "\(name) v\(version) live (\(kind.rawValue))"
            case .held(let version, let reason):
                state.braid.events.append("■ \(name) v\(version) held: \(reason)")
                state.status.message = "\(name) v\(version) held"
            }
        case .exited(let name, let status):
            state.braid.events.append("\(name): node process exited (\(status))")
            state.braid.states[name]?.stage = .stopped
        case .started:
            state.braid.running = true
            state.status.message = "\(state.braid.nodes.count) Thread nodes up · f feeds the focused one"
        case .fed(let name, let documents):
            state.braid.fedAt[name] = state.braid.phase
            state.braid.events.append("\(name) ← \(documents.count) document\(documents.count == 1 ? "" : "s"): \(documents.prefix(3).joined(separator: ", "))\(documents.count > 3 ? "…" : "")")
            if documents.isEmpty { state.status.message = "\(name) has been fed its whole shard" }
        case .withdrew(let name, let document):
            state.braid.events.append("\(name) ✗ withdrew \(document)")
        case .examples(let examples):
            // The first examples to arrive fill an empty prompt; later ones never refill one the owner cleared.
            let first = state.braid.examples.isEmpty
            state.braid.examples = examples
            if first, state.braid.exampleLabel == nil, state.braid.prompt.text.isEmpty, !examples.isEmpty, !state.braid.editingPrompt {
                choose(example: 0, state: &state)
            } else if let tokens = state.braid.exampleTokens {
                // The list reorders as documents arrive: keep the chosen example, find its new place.
                state.braid.exampleIndex = examples.firstIndex { $0.promptTokens == tokens } ?? -1
            }
        case .generating:
            state.braid.generating = true
            state.braid.generation = nil
            state.braid.streaming = []
            state.braid.promptTraces = []
            state.braid.promptFirst = nil
            state.braid.alone = [:]
            state.braid.revealed = 0
            state.braid.cursor = 0
            state.braid.verification = []
            state.braid.verifiedAll = nil
        case .prompted(let first, let traces):
            state.braid.promptFirst = first
            state.braid.promptTraces = traces
            state.braid.cursor = traces.count
        case .token(let token):
            state.braid.streaming.append(token)
        case .generated(let generation):
            state.braid.generation = generation
            state.braid.generating = false
            let prompt = generation.traces.filter(\.isPrompt)
            if !prompt.isEmpty { state.braid.promptTraces = prompt }
            if let first = generation.prompt.tokenTexts?.first { state.braid.promptFirst = first }
            state.braid.revealed = generation.traces.filter { !$0.isPrompt }.count
            state.braid.cursor = state.braid.firstAnswer
        case .verified(let lines, let allVerified):
            state.braid.verification = lines
            state.braid.verifiedAll = allVerified
        case .stopped:
            state.braid.running = false
            state.braid.started = false
            state.braid.events.append("every node and its Thread stopped")
            for name in state.braid.states.keys { state.braid.states[name]?.stage = .stopped }
            // Stopped to switch dataset (d): start again, on it.
            if state.braid.restarting {
                state.braid.restarting = false
                return [start(&state, switching: true)]
            }
            // Back to the catalog, with the live versions the run left.
            return [.braid(.catalog)]
        case .failure(let name, let message):
            state.braid.events.append(Text("\(name ?? "braid"): \(message)", style: .plain))
            state.status.error = RaoLMFailure(message, code: 69)
            // The braid itself did not start: nothing is starting any more.
            if name == nil, !state.braid.running {
                state.braid.started = false
                state.braid.startFailure = message
            }
        case .note(let message):
            state.braid.events.append(Text("braid: \(message)", style: .plain))
            state.status.message = message
        }
        return []
    }

    /// The animation clock: pulses travel, streamed tokens are revealed at a pace you can follow.
    static func tick(_ state: inout StudioState) {
        state.braid.phase &+= 1
        let phase = state.braid.phase
        if state.braid.revealed < state.braid.streaming.count, state.braid.generation == nil || state.braid.revealed < state.braid.streaming.count {
            let token = state.braid.streaming[state.braid.revealed]
            state.braid.revealed += 1
            for share in token.trace.strands ?? [] {
                state.braid.pulses.append(.init(strand: share.strand, start: phase, strength: share.gate, open: share.open))
            }
        }
        state.braid.pulses.removeAll { phase - $0.start > pulseTicks }
    }

    static func choose(example index: Int, state: inout StudioState) {
        let examples = state.braid.examples
        guard !examples.isEmpty else { return }
        let i = (index % examples.count + examples.count) % examples.count
        let example = examples[i]
        state.braid.exampleIndex = i
        state.braid.exampleTokens = example.promptTokens
        state.braid.exampleSource = example.source
        state.braid.exampleExpected = example.expected
        state.braid.exampleLabel = example.label
        state.braid.exampleNode = example.node
        state.braid.prompt.set(example.promptText.trimmingCharacters(in: .whitespaces))
    }

    // MARK: - Keys

    static func handle(_ key: KeyEvent, _ state: inout StudioState) -> [StudioJob]? {
        if case .ctrl("u") = key.key, state.braid.started {
            // One key from anywhere: an empty prompt, ready to type (the example goes with it).
            state.braid.prompt.set("")
            state.braid.exampleTokens = nil
            state.braid.exampleSource = nil
            state.braid.exampleExpected = nil
            state.braid.exampleLabel = nil
            state.braid.exampleNode = nil
            state.braid.exampleIndex = -1
            state.braid.editingPrompt = true
            return []
        }
        if state.braid.editingPrompt {
            switch key.key {
            case .escape:
                state.braid.editingPrompt = false
                return []
            case .enter:
                state.braid.editingPrompt = false
                return generate(&state)
            default:
                if state.braid.prompt.handle(key) {
                    // Typed text is tokenized as typed; it is no longer the example's corpus slice.
                    state.braid.exampleTokens = nil
                    state.braid.exampleSource = nil
                    state.braid.exampleExpected = nil
                    state.braid.exampleLabel = nil
                    state.braid.exampleNode = nil
                    state.braid.exampleIndex = -1
                }
                return []
            }
        }
        // Before a start: ↑↓ choose a braid in the catalog, ⏎ loads it.
        if !state.braid.started, !state.braid.catalog.isEmpty {
            let selected = state.braid.catalogTable.selected ?? 0
            switch key.key {
            case .up, .char("k"):
                select(braid: selected - 1, state: &state)
                return []
            case .down, .char("j"):
                select(braid: selected + 1, state: &state)
                return []
            case .enter:
                return [start(&state)]
            default:
                break
            }
        }
        let nodes = state.braid.nodes
        switch key.key {
        case .char("S"):
            guard !state.braid.started else {
                state.status.message = "the braid is already running"
                return []
            }
            return [start(&state)]
        case .char("X"):
            guard state.braid.started else { return [] }
            return [.braid(.stop)]
        case .char("o"):
            guard !state.braid.started else {
                state.status.message = "stop the braid (X) to change its mode"
                return []
            }
            state.braid.offline.toggle()
            state.status.message = state.braid.offline ? "next start: offline corpora (no Thread processes)" : "next start: a Thread process per node"
        case .char("R"):
            guard !state.braid.started else {
                state.status.message = "stop the braid (X) first"
                return []
            }
            state.braid.fresh.toggle()
            state.status.message = state.braid.fresh ? "next start wipes every node's storage and versions" : "next start keeps the nodes' storage"
        case .char("d"):
            guard state.braid.running || !state.braid.started, !state.braid.restarting else {
                state.status.message = "the nodes are still starting"
                return []
            }
            guard !state.braid.generating else {
                state.status.message = "the umbrella is answering: c cancels it"
                return []
            }
            state.overlay = .braidDataset(datasetPrompt(state))
        case .tab, .backTab:
            guard !nodes.isEmpty else { return [] }
            let delta = key.key == .tab ? 1 : nodes.count - 1
            state.braid.focus = (state.braid.focus + delta) % nodes.count
        case .char("f"):
            guard state.braid.running, let node = state.braid.focusedNode else { return notRunning(&state) }
            return [.braid(.feed(node: node))]
        case .char("F"):
            guard state.braid.running else { return notRunning(&state) }
            return nodes.map { .braid(.feed(node: $0.name)) }
        case .char("w"):
            guard state.braid.running, let node = state.braid.focusedNode else { return notRunning(&state) }
            return [.braid(.withdraw(node: node))]
        case .char("c"):
            guard let node = state.braid.focusedNode else { return [] }
            if state.braid.generating { return [.cancel] }
            return [.braid(.cancel(node: node))]
        case .char("/"), .char("p"):
            state.braid.editingPrompt = true
        case .char("x"):
            guard !state.braid.examples.isEmpty else {
                state.status.message = "no examples yet: feed a node (f) and let it go live"
                return []
            }
            choose(example: state.braid.exampleIndex + 1, state: &state)
        case .enter:
            return generate(&state)
        case .char("a"):
            return generate(&state, alone: true)
        case .char("g"):
            let order: [BraidGating] = [.braided, .posterior, .retrieval]
            state.braid.gating = order[((order.firstIndex(of: state.braid.gating) ?? 0) + 1) % order.count]
            state.status.message = "gate: \(state.braid.gating.rawValue) · " + gateDescription(state.braid.gating)
        case .char("r"):
            state.braid.questions.toggle()
            state.status.message = state.braid.questions
                ? "Q/A on: a prompt ending in \"?\" is rewritten by the commons into a stem the Threads complete"
                : "Q/A off: every prompt is completed as written, a question too"
        case .char("t"):
            state.braid.temperature = state.braid.temperature > 0 ? 0 : 0.8
            state.status.message = state.braid.temperature > 0
                ? "sampling: each token is drawn from the mixture, so every Thread gets chosen in proportion"
                : "the likeliest token each time"
        case .left, .char("h"):
            state.braid.cursor = max(0, state.braid.cursor - 1)
        case .right, .char("l"):
            state.braid.cursor = min(max(0, state.braid.walkable.count - 1), state.braid.cursor + 1)
        case .home:
            state.braid.cursor = 0
        case .end:
            state.braid.cursor = max(0, state.braid.walkable.count - 1)
        case .char("v"):
            guard let generation = state.braid.generation else { return [] }
            return [.braid(.verify(generation))]
        case .char("-"), .char("_"):
            state.braid.lambda = max(0, (state.braid.lambda * 20 - 1).rounded() / 20)
            state.status.message = String(format: "λ %.2f: next generation leans %@ on retrieval", state.braid.lambda, state.braid.lambda < 0.5 ? "less" : "more")
        case .char("+"), .char("="):
            state.braid.lambda = min(1, (state.braid.lambda * 20 + 1).rounded() / 20)
            state.status.message = String(format: "λ %.2f: next generation leans %@ on retrieval", state.braid.lambda, state.braid.lambda < 0.5 ? "less" : "more")
        default:
            return nil
        }
        return []
    }

    static func notRunning(_ state: inout StudioState) -> [StudioJob] {
        state.status.message = state.braid.started ? "the nodes are still starting" : "S starts the Thread nodes"
        return []
    }

    // MARK: - The dataset feeds come from (d)

    static func datasetPrompt(_ state: StudioState) -> InputPrompt {
        InputPrompt(
            title: "Feed from a dataset", help: [
                "A path to a braid dataset, or its name in the datasets root (the T9's, or $\(DatasetsRoot.environmentKey)).",
                "The braid restarts on it with the dataset's own nodes; f and F then feed from it.",
                "Nodes fed from anything else start over: their storage and versions are wiped.",
            ], field: TextFieldState(state.braid.source?.path ?? state.braid.world.dataset))
    }

    /// The dataset is there: the braid stops if it runs, and starts on the dataset.
    static func restart(on dataset: String, state: inout StudioState) -> [StudioJob] {
        state.braid.world = BraidWorldChoice(dataset: dataset)
        guard state.braid.started else { return [start(&state, switching: true)] }
        state.braid.restarting = true
        state.status.message = "stopping the braid to restart it on \(dataset)"
        return [.braid(.stop)]
    }

    static func start(_ state: inout StudioState, switching: Bool = false) -> StudioJob {
        state.braid.started = true
        let entry = selectedBraid(state)
        if let entry { state.braid.loaded = entry }
        return .braid(.start(offline: state.braid.offline, fresh: state.braid.fresh, world: state.braid.world, switching: switching, root: entry?.root))
    }

    // MARK: - The catalog: every braid on this machine

    static func selectedBraid(_ state: StudioState) -> BraidCatalogEntry? {
        guard let index = state.braid.catalogTable.selected, index < state.braid.catalog.count else { return nil }
        return state.braid.catalog[index]
    }

    /// Chooses a braid: the next start loads it, with its own nodes and mode, and never fresh.
    static func select(braid index: Int, state: inout StudioState) {
        let catalog = state.braid.catalog
        guard !catalog.isEmpty else { return }
        let i = min(max(0, index), catalog.count - 1)
        state.braid.catalogTable.selected = i
        let entry = catalog[i]
        state.braid.nodes = entry.nodes
        state.braid.offline = entry.offline
        state.braid.fresh = false
        state.braid.focus = 0
        state.braid.world = entry.isHome ? state.braid.launchWorld : BraidWorldChoice()
    }

    /// The catalog arrived: the selection stays on the same braid, and the first catalog chooses
    /// the studio's own unless a braid is already running.
    static func catalogLoaded(_ entries: [BraidCatalogEntry], state: inout StudioState) {
        let first = state.braid.catalog.isEmpty
        let selected = (selectedBraid(state) ?? state.braid.loaded)?.root
        state.braid.catalog = entries
        guard !entries.isEmpty else {
            state.braid.catalogTable = TableState(selected: nil)
            return
        }
        let index = entries.firstIndex { $0.root == selected } ?? 0
        state.braid.catalogTable = TableState(selected: index)
        if state.braid.started {
            // `raolm braid` started the studio's own before any catalog: that is the one loaded.
            state.braid.loaded = state.braid.loaded.flatMap { loaded in entries.first { $0.root == loaded.root } } ?? entries.first { $0.isHome }
        } else {
            if let loaded = state.braid.loaded { state.braid.loaded = entries.first { $0.root == loaded.root } ?? loaded }
            if first { select(braid: index, state: &state) }
        }
    }

    /// What f and F feed from, in a few words.
    static func sourceLabel(_ state: StudioState) -> String {
        if let source = state.braid.source { return source.name }
        let dataset = state.braid.world.dataset
        return dataset.isEmpty ? "mock world" : (dataset as NSString).lastPathComponent
    }

    static func gateDescription(_ gating: BraidGating) -> String {
        switch gating {
        case .braided: return "token by token: who has been predicting the text, lifted where Threads agree, leaning to a Thread whose documents the text follows"
        case .posterior: return "every token's likelihood multiplied since the start: it settles on one Thread"
        case .retrieval: return "each Thread's share of one pool of raw cosines across Threads"
        }
    }

    static func generate(_ state: inout StudioState, alone: Bool = false) -> [StudioJob] {
        guard state.braid.running else { return notRunning(&state) }
        guard state.braid.states.values.contains(where: \.isLive) else {
            state.status.error = RaoLMFailure("no node has a live version yet", hint: "f feeds the focused node", code: 66)
            return []
        }
        guard state.status.mlxJob == nil else {
            state.status.error = RaoLMFailure("the MLX worker is busy: \(state.status.mlxJob!.label)", hint: "press c to cancel it", code: 75)
            return []
        }
        let text = state.braid.prompt.text
        guard !text.trimmingCharacters(in: .whitespaces).isEmpty || state.braid.exampleTokens != nil else {
            state.status.error = RaoLMFailure("the prompt is empty", hint: "/ types one, x takes a fact from a Thread", code: 64)
            return []
        }
        let answerTokens = state.braid.exampleExpected.map { max(2, $0.count / 3) } ?? 12
        // A typed question goes through the umbrella's adapter, the "?" says so, unless r turned it off.
        let question = state.braid.questions && state.braid.exampleTokens == nil && text.trimmingCharacters(in: .whitespacesAndNewlines).hasSuffix("?")
        state.braid.generating = true
        return [.braid(.generate(BraidGenerateSpec(
            promptTokens: state.braid.exampleTokens, promptText: text, source: state.braid.exampleSource,
            lambda: state.braid.lambda, maxTokens: question ? 24 : min(32, answerTokens + 10), gating: state.braid.gating,
            temperature: state.braid.temperature, alone: alone, question: question)))]
    }

    static func hints(_ state: StudioState) -> [KeyHint] {
        if state.braid.editingPrompt { return [KeyHint("⏎", "generate"), KeyHint("^U", "clear"), KeyHint("esc", "stop editing")] }
        guard state.braid.started else {
            let start = state.braid.catalog.isEmpty ? [KeyHint("S", "start")] : [KeyHint("⏎", "load"), KeyHint("↑↓", "braid")]
            return start + [KeyHint("o", state.braid.offline ? "mode: offline" : "mode: Threads"),
                            KeyHint("R", state.braid.fresh ? "fresh: on" : "fresh: off"), KeyHint("d", "dataset: \(sourceLabel(state))")]
        }
        var hints = [KeyHint("f", "feed"), KeyHint("d", "dataset"), KeyHint("tab", "node"), KeyHint("x", "example"), KeyHint("^U", "clear"), KeyHint("⏎", "ask"),
                     KeyHint("a", "alone"), KeyHint("r", state.braid.questions ? "Q/A: on" : "Q/A: off"), KeyHint("←→", "token")]
        if state.braid.generation != nil { hints.append(KeyHint("v", "verify")) }
        hints += [KeyHint("g", "gate"), KeyHint("w", "withdraw"), KeyHint("X", "stop")]
        return hints
    }

    // MARK: - Render

    static func render(_ state: StudioState, _ frame: inout Frame, _ rect: Rect) {
        if !state.braid.started, !state.braid.catalog.isEmpty {
            renderCatalog(state, &frame, rect)
            return
        }
        guard state.braid.started || !state.braid.states.isEmpty else {
            renderIdle(state, &frame, rect)
            return
        }
        let compact = rect.height < 30
        let umbrellaHeight = compact ? 7 : 10 + max(2, state.braid.nodes.count)
        let nodeHeight = compact ? max(7, rect.height - umbrellaHeight - 1 - 3) : min(14, max(11, (rect.height - umbrellaHeight - 1) / 2))
        let (umbrellaRect, belowUmbrella) = rect.top(umbrellaHeight)
        let (strandRow, belowStrands) = belowUmbrella.top(1)
        let (nodesRect, bottomRect) = belowStrands.top(nodeHeight)
        renderUmbrella(state, &frame, umbrellaRect, compact: compact)
        let names = state.braid.nodes.map(\.name)
        let columns = nodesRect.splitHorizontally(Array(repeating: .flex(1), count: max(1, names.count)))
        for (index, name) in names.enumerated() where index < columns.count {
            renderNode(state, name: name, index: index, &frame, columns[index], compact: compact)
        }
        let phase = state.braid.phase
        let pulses: [(strand: Int, progress: Double, strength: Float, open: Bool)] = state.braid.pulses.compactMap { pulse in
            guard let index = names.firstIndex(of: pulse.strand) else { return nil }
            let t = Double(phase - pulse.start) / Double(pulseTicks)
            guard t > 0.4 else { return nil }
            return (index, min(1, (t - 0.4) / 0.6), pulse.strength, pulse.open)
        }
        BraidArt.strands(centres: columns.prefix(names.count).map { $0.minX + $0.width / 2 }, rect: strandRow, pulses: pulses,
                         palette: frame.palette, glyphs: frame.glyphs, frame: &frame)
        if bottomRect.height >= 3 { renderBottom(state, &frame, bottomRect) }
    }

    /// The idle screen's strands: the first half of the nodes run in from the left to the gold
    /// ring, the rest from the right; with an odd count the middle node's strand frames the ring.
    static func idleStrands(count: Int, palette: Palette, glyphs: Glyphs, side: Int = 17) -> Text {
        let n = max(1, count)
        let left = Array(0..<(n / 2))
        let centre: Int? = n % 2 == 1 ? n / 2 : nil
        let right = Array(((n + 1) / 2)..<n)
        func half(_ strands: [Int], mirrored: Bool) -> Text {
            guard !strands.isEmpty else { return Text(String(repeating: " ", count: side), style: palette.muted) }
            var cells: [(String, Int)] = []
            for i in 0..<side {
                let t = Double(i) / Double(side)
                let glyph = t < 0.4 ? glyphs.wave : (t < 0.72 ? glyphs.ripple : glyphs.horizontal)
                cells.append((glyph, strands[min(strands.count - 1, i * strands.count / side)]))
            }
            if mirrored { cells.reverse() }
            var text = Text()
            for (glyph, strand) in cells { text.append(glyph, Style(foreground: BraidArt.strandColor(strand, palette), background: palette.base.background)) }
            return text
        }
        var text = half(left, mirrored: false)
        let centreStyle = centre.map { Style(foreground: BraidArt.strandColor($0, palette), background: palette.base.background) }
        text.append(centreStyle == nil ? " " : glyphs.horizontal, centreStyle ?? palette.muted)
        text.append(glyphs.verified, palette.gold)
        text.append(centreStyle == nil ? " " : glyphs.horizontal, centreStyle ?? palette.muted)
        text.append(half(right.reversed(), mirrored: true))
        return text
    }

    /// The catalog: every braid on this machine, and the selected one's Threads, commons and model.
    static func renderCatalog(_ state: StudioState, _ frame: inout Frame, _ rect: Rect) {
        let palette = frame.palette
        let glyphs = frame.glyphs
        let catalog = state.braid.catalog
        let detailHeight = rect.height >= 18 ? 10 : 0
        let (detailRect, listRect) = rect.bottom(detailHeight)
        let inner = Theme.panel("Braids · \(catalog.count) · ↑↓ chooses one, ⏎ loads it", focused: true, &frame, listRect)
        if !inner.isEmpty {
            let wide = inner.width >= 100
            var columns = [Column("braid", .flex(1)), Column("commons", .fixed(14)), Column("Threads", .fixed(9)), Column("model", .fixed(22))]
            if wide { columns += [Column("fed from", .fixed(15)), Column("mode", .fixed(7)), Column("last live", .fixed(9))] }
            let rows: [[Text]] = catalog.map { entry in
                let live = entry.nodes.filter { entry.live[$0.name] != nil }.count
                var row = [
                    Text(entry.name, style: entry.isHome ? palette.title : palette.text),
                    entry.commons.map { Text($0, style: palette.gold) } ?? Text(glyphs.dash, style: palette.muted),
                    Text(live == entry.nodes.count ? "\(live) live" : "\(live)/\(entry.nodes.count) live", style: live == 0 ? palette.muted : palette.text),
                    Text(entry.model, style: palette.dim),
                ]
                if wide {
                    row += [Text(entry.dataset ?? "mock world", style: palette.dim), Text(entry.offline ? "offline" : "Threads", style: palette.dim),
                            Text(entry.updated.map { $0.formatted(.dateTime.month(.abbreviated).day()) } ?? glyphs.dash, style: palette.dim)]
                }
                return row
            }
            var table = state.braid.catalogTable
            table.clamp(rowCount: rows.count, visible: Table.visibleRows(in: inner, showHeader: true))
            Table.styled(columns: columns, rows: rows, state: table, palette: palette, glyphs: glyphs).render(in: inner, on: &frame.canvas)
        }
        guard detailHeight > 0, let entry = selectedBraid(state) else { return }
        let detail = Theme.panel(entry.name + (entry.isHome ? " · the data root's own braid" : ""), &frame, detailRect)
        guard !detail.isEmpty else { return }
        func key(_ name: String) -> Text { Text(name.padding(toLength: 10, withPad: " ", startingAt: 0), style: palette.muted) }
        var threads = key("Threads")
        for (index, node) in entry.nodes.enumerated() {
            if index > 0 { threads.append(" · ", palette.dim) }
            threads.append(node.label, Style(foreground: BraidArt.strandColor(index, palette), background: palette.base.background, attributes: .bold))
            threads.append(entry.live[node.name].map { " v\($0)" } ?? " not live", palette.dim)
        }
        var commons = key("commons")
        if let name = entry.commons {
            commons.append(name, palette.gold)
            commons.append((entry.commonsSHA256.map { " " + $0.prefix(12) } ?? "") + " · the base model under every Thread: it answers what no Thread holds, and is never cited", palette.dim)
        } else {
            commons.append("none · a \(entry.preset) braid's umbrella is the shared vocabulary alone", palette.muted)
        }
        var lines = [
            key("where") + Text(StudioApp.abbreviate(entry.root.path), style: palette.dim),
            threads,
            commons,
            key("model") + Text(entry.model, style: palette.text)
                + Text(entry.commons == nil ? "" : " · each Thread trains its lower blocks under the commons' trunk", style: palette.dim),
            key("fed from") + Text(entry.dataset ?? "a generated mock world", style: palette.text),
            Text(""),
            Text("⏎ loads it: \(entry.nodes.count) Thread node\(entry.nodes.count == 1 ? "" : "s"), each on ", style: palette.text)
                + Text(state.braid.offline ? "an offline corpus" : "a Thread of its own", style: palette.title) + Text(" (o switches)", style: palette.dim),
        ]
        if state.braid.fresh { lines.append(Text("fresh: the next start wipes every node's storage and versions", style: palette.red)) }
        if let failure = state.braid.startFailure { lines.append(Text("\(glyphs.cross) the last start: \(failure)", style: palette.red)) }
        for (row, line) in lines.prefix(detail.height).enumerated() {
            frame.canvas.put(line.truncated(to: detail.width), x: detail.minX, y: detail.minY + row, clip: detail)
        }
    }

    static func renderIdle(_ state: StudioState, _ frame: inout Frame, _ rect: Rect) {
        let palette = frame.palette
        let glyphs = frame.glyphs
        let inner = Theme.panel("Braid · one headless transformer per Thread, one umbrella over all", focused: true, &frame, rect)
        let nodes = state.braid.nodes.isEmpty ? BraidNodeSpec.defaults : state.braid.nodes
        var lines: [Text] = [
            Text("S starts \(nodes.count) Thread nodes (\(nodes.map(\.label).joined(separator: ", "))), each its own process with ", style: palette.text)
                + Text(state.braid.offline ? "an offline corpus" : "a Thread of its own", style: palette.title)
                + Text(" (o switches), exclusive to RaoLM: ports from 8195, storage under <data root>/braid.", style: palette.text),
            Text(""),
            Text("  umbrella   ", style: palette.muted) + Text("RMSNorm (final) → tied head → softmax → p = λ·p_knn + (1−λ)·p_lm", style: palette.gold),
            Text("             ", style: palette.muted) + Text("gates each Thread token by token by what it has been predicting, asks only open ones for their hidden state", style: palette.dim),
            Text("  strands    ", style: palette.muted) + idleStrands(count: nodes.count, palette: palette, glyphs: glyphs),
            Text("  nodes      ", style: palette.muted) + Text("token embedding → blocks → last hidden state", style: palette.text)
                + Text("   (the shared vocabulary is frozen; each Thread trains only its blocks)", style: palette.dim),
            Text("             ", style: palette.muted) + Text("a provenance index over its own corpus · an update ladder: snapshot, diff, reindex, train, index, gates, live", style: palette.dim),
            Text(""),
            Text("Then: f feeds the focused node mock documents (tab switches, F feeds all), x picks a fact from a Thread, ⏎ asks the umbrella,", style: palette.text),
            Text("←→ walks the tokens, the prompt's too: each is coloured by the Thread that supplied it, with every gate and its citation.", style: palette.text),
            Text(""),
            Text("d names the dataset f and F feed from (a path, or a name in the datasets root) and starts the braid on it. Now: ", style: palette.text)
                + Text(state.braid.world.dataset.isEmpty ? "this braid's own, else a generated mock world" : state.braid.world.dataset, style: palette.title),
            Text(""),
            Text("CLI: raolm braid demo [--offline] runs the same thing headless · raolm braid status · raolm braid down", style: palette.muted),
        ]
        if state.braid.fresh { lines.append(Text("fresh: the next start wipes every node's storage and versions", style: palette.dim.italic())) }
        for (row, line) in lines.prefix(inner.height).enumerated() {
            frame.canvas.put(line.truncated(to: inner.width), x: inner.minX, y: inner.minY + row, clip: inner)
        }
    }

    // MARK: Umbrella

    static func renderUmbrella(_ state: StudioState, _ frame: inout Frame, _ rect: Rect, compact: Bool) {
        let palette = frame.palette
        let glyphs = frame.glyphs
        let names = state.braid.nodes.map(\.name)
        let live = state.braid.states.values.filter(\.isLive).count
        // The gate of the generation shown, and the one the next generation will use when it differs.
        let shownGate = state.braid.generation.map { $0.braid?.gating ?? .posterior }
        var footer = Text(state.braid.loaded.map { "\($0.name) · " } ?? "", style: palette.title)
        footer.append("gate \((shownGate ?? state.braid.gating).rawValue)", palette.dim)
        if let shownGate, shownGate != state.braid.gating { footer.append(" · next \(state.braid.gating.rawValue)", palette.dim.italic()) }
        footer.append(state.braid.temperature > 0 ? " · sampling" : " · likeliest", palette.dim)
        footer.append(String(format: " · λ %.2f", state.braid.lambda), palette.dim)
        footer.append(" · \(live)/\(names.count) Threads live", palette.muted)
        if let generation = state.braid.generation {
            if let rewrite = generation.prompt.question { footer.append(" · rewrite \(rewrite.rewriter)", palette.dim) }
            if let followed = generation.traces.compactMap(\.followed).first { footer.append(" · followed \(followed)", palette.gold) }
        }
        let title = compact ? "Umbrella · norm → tied head → softmax → Σ g·(λ mix)"
            : "Umbrella · RMSNorm (final) → tied head → softmax → p = Σ g·(λ·p_knn + (1−λ)·p_lm)"
        let inner = Theme.panel(title, focused: true, footer: footer, &frame, rect)
        guard !inner.isEmpty else { return }
        let traces = state.braid.walkable
        let stripHeight = compact ? 2 : 3
        let strip = Rect(x: inner.minX, y: inner.minY, width: inner.width, height: min(stripHeight, inner.height))
        renderStrip(state, &frame, strip)
        let detail = inner.inset(top: strip.height, left: 0, bottom: 0, right: 0)
        guard !detail.isEmpty else { return }
        guard !traces.isEmpty else {
            let waiting = state.braid.generating ? "\(StudioApp.spinner(state, glyphs)) the umbrella is asking the Threads…"
                : (state.braid.states.values.contains(where: \.isLive) ? "⏎ asks the umbrella · a asks each Thread alone too · x takes a fact from a Thread · / types a prompt"
                    : "feed a node (f) and wait for it to go live; the umbrella routes only to live Threads")
            Theme.empty(waiting, &frame, detail)
            return
        }
        let cursor = min(state.braid.cursor, traces.count - 1)
        let trace = traces[cursor]
        let inPrompt = cursor < state.braid.firstAnswer
        // A generation recorded before gates were kept was weighed by the posterior.
        let gating: BraidGating
        if let generation = state.braid.generation {
            gating = generation.braid?.gating ?? .posterior
        } else if trace.strands?.first?.memory != nil {
            gating = .braided
        } else {
            gating = state.braid.gating == .braided ? .posterior : state.braid.gating
        }
        let shares = trace.strands ?? []
        let order = shares.map(\.strand)
        func colour(_ strand: String, _ fallback: Int) -> Color {
            strand == BraidStrandRef.commonsName ? palette.goldColor : BraidArt.strandColor(names.firstIndex(of: strand) ?? fallback, palette)
        }
        func display(_ text: String) -> String { text.replacingOccurrences(of: "\n", with: glyphs.newline) }
        var rows: [Text] = []
        // Shares.
        let position = inPrompt ? "prompt \(trace.index + 1)/\(state.braid.promptTraces.count + 1)"
            : "token \(cursor - state.braid.firstAnswer + 1)/\(state.braid.shownTraces.count)"
        var shareLine = Text("\(position) «\(display(trace.text))»  ", style: palette.title)
        shareLine.append("p \(Format.f(trace.mixedProb, 2))  ", palette.dim)
        let bar = ShareBar(segments: shares.enumerated().map { (Double($0.element.share), colour($0.element.strand, $0.offset)) },
                           track: palette.muted, glyphs: glyphs)
        shareLine.append(bar.text(width: compact ? 16 : 24))
        for (i, share) in shares.enumerated() {
            shareLine.append("  " + label(share.strand, state) + " ", Style(foreground: colour(share.strand, i), background: palette.base.background))
            shareLine.append(Format.pct(share.share), palette.text)
        }
        shareLine.append("   H_thread \(Format.f(trace.threadEntropy, 2))", palette.dim)
        rows.append(shareLine)
        // What the mixture expected here, and who supplied each candidate.
        if !compact {
            var next = Text("next   ", style: palette.muted)
            if let candidates = trace.candidates, !candidates.isEmpty {
                for (i, candidate) in candidates.enumerated() {
                    if i > 0 { next.append("   ", palette.dim) }
                    next.append("«\(display(candidate.text))» ", candidate.token == trace.token ? palette.title : palette.text)
                    next.append(Format.pct(candidate.prob) + " ", palette.dim)
                    let parts = candidate.parts.enumerated().map { part in
                        (Double(part.element), colour(part.offset < order.count ? order[part.offset] : "", part.offset))
                    }
                    next.append(ShareBar(segments: parts, track: palette.muted, glyphs: glyphs).text(width: 6))
                }
            } else {
                next.append("(not recorded for this token)", palette.muted)
            }
            rows.append(next)
        }
        // Gates.
        if compact {
            var gates = Text("gates  ", style: palette.muted)
            for (i, share) in shares.enumerated() {
                gates.append(label(share.strand, state) + " ", Style(foreground: colour(share.strand, i), background: palette.base.background))
                gates.append(share.open ? glyphs.live : glyphs.idle, share.open ? Style(foreground: colour(share.strand, i), background: palette.base.background) : palette.muted)
                gates.append(" \(Format.f(share.gate, 2))  ", share.open ? palette.text : palette.muted)
            }
            rows.append(gates)
        } else {
            let prompt = state.braid.promptTraces.count
            let width = max(8, shares.map { label($0.strand, state).count }.max() ?? 8)
            let leader = shares.indices.max { (shares[$0].memory ?? 0) < (shares[$1].memory ?? 0) }
            for (i, share) in shares.enumerated() {
                let tint = Style(foreground: colour(share.strand, i), background: palette.base.background)
                var line = Text(i == 0 ? "gates  " : "       ", style: palette.muted)
                line.append(label(share.strand, state).padding(toLength: width, withPad: " ", startingAt: 0) + " ", tint)
                line.append(share.open ? glyphs.live : glyphs.idle, share.open ? tint : palette.muted)
                line.append(" \(Format.f(share.gate, 2)) \(share.open ? "asked " : "closed")", share.open ? palette.text : palette.muted)
                if let memory = share.memory { line.append(" · memory \(Format.f(memory, 2))", palette.dim) }
                if let backs = share.backs { line.append(i == leader ? " · leads     " : " · backs \(Format.f(backs, 2))", palette.dim) }
                if prompt > 0 { line.append(" · knew \(knew(state, share.strand))/\(prompt)", palette.dim) }
                line.append(" · alone " + (share.alone.map { Format.f($0, 2) } ?? "—"), palette.dim)
                // With a commons strand: what this Thread knows of the token beyond the base model.
                if let lift = share.lift { line.append(" · lift \(Format.f(lift, 1))", lift > 0 ? palette.text : palette.dim) }
                if let thought = share.thought, i != leader { line.append(" · thinks alike \(Format.f(thought, 2))", palette.dim) }
                // Its trajectory: how many tokens its retrieval has followed one of its documents, and how its hits move through them.
                if let trajectory = share.trajectory {
                    line.append(" · traced \(trajectory.length)", trajectory.length > TrajectoryRule.standard.phrase ? palette.text : palette.dim)
                    line.append(" · manner " + (trajectory.manner.map { Format.f($0, 2) } ?? "—"), palette.dim)
                }
                line.append(" · best \(Format.f(share.bestScore, 3))", palette.dim)
                rows.append(line)
            }
        }
        rows.append(why(state, trace: trace, gating: gating, palette: palette))
        if !compact {
            var entropies = Text("inner  ", style: palette.muted)
            entropies.append("H_lm \(Format.f(max(0, trace.lmEntropy), 2)) · H_knn \(Format.f(max(0, trace.knnEntropy), 2)) · H_mix \(Format.f(max(0, trace.mixedEntropy), 2)) · H_source \(Format.f(max(0, trace.sourceEntropy), 2))", palette.dim)
            entropies.append(" · agree \(Format.f(trace.agreement, 2)) · confidence ", palette.dim)
            entropies.append(Format.f(trace.confidence, 2), palette.heat(confidence: trace.confidence, verified: false))
            rows.append(entropies)
        }
        // Citation.
        var cites = Text("cites  ", style: palette.muted)
        if let citation = trace.citations.first, let generation = state.braid.generation ?? nil {
            let strand = generation.braid?.strand(row: citation.row)
            cites.append((strand?.label ?? "?") + " · ", Style(foreground: colour(strand?.name ?? "", 0), background: palette.base.background))
            cites.append(generation.partition(row: citation.row)?.documentName ?? "?", palette.title)
            cites.append(" p\(citation.address.partitionIndex)@\(citation.address.tokenOffset) · conf \(Format.f(citation.confidence, 2))", palette.dim)
            if let spanIndex = trace.spanIndex, spanIndex < generation.spans.count {
                let span = generation.spans[spanIndex]
                if let status = span.verification?.status {
                    cites.append("  \(status == .verified ? glyphs.verified : glyphs.cross) \(status.rawValue)", status == .verified ? palette.gold : palette.red)
                } else {
                    cites.append("  span \(spanIndex + 1) · v verifies", palette.muted)
                }
            }
            cites.append("  thread \(citation.address.threadID.map { String($0.prefix(8)).lowercased() } ?? "—")", palette.muted)
        } else if let citation = trace.citations.first {
            cites.append("row \(citation.row) · p\(citation.address.partitionIndex)@\(citation.address.tokenOffset)", palette.dim)
        } else if inPrompt {
            cites.append("a prompt token: what the Threads predicted is above; answers are cited", palette.muted)
        } else {
            cites.append(trace.citations.isEmpty && state.braid.generation != nil ? "uncited: no retrieved context predicted this token" : "…", palette.muted)
        }
        if !compact { rows.append(cites) }
        for (i, row) in rows.prefix(detail.height).enumerated() {
            frame.canvas.put(row.truncated(to: detail.width), x: detail.minX, y: detail.minY + i, clip: detail)
        }
    }

    /// How many of the prompt's tokens a Thread predicted (gave at least the gate's evidence floor).
    static func knew(_ state: StudioState, _ strand: String) -> Int {
        let floor = state.braid.generation?.braid?.gate?.evidenceFloor ?? BraidGate().evidenceFloor
        return state.braid.promptTraces.filter { ($0.strands?.first { $0.strand == strand }?.alone ?? 0) >= floor }.count
    }

    /// Whether no Thread predicted even a third of the prompt.
    static func unrecognised(_ state: StudioState) -> Bool {
        let prompt = state.braid.promptTraces.count
        guard prompt > 0, state.braid.promptTraces.contains(where: { $0.strands?.contains { $0.alone != nil } == true }) else { return false }
        return state.braid.nodes.allSatisfy { knew(state, $0.name) * 3 < prompt }
    }

    /// Why the gate stands where it does at this token, in words.
    static func why(_ state: StudioState, trace: TokenTrace, gating: BraidGating, palette: Palette) -> Text {
        let head = Text("why    ", style: palette.muted)
        let shares = trace.strands ?? []
        guard shares.count > 1 else { return head + Text("one Thread supplies every token", style: palette.dim) }
        switch gating {
        case .posterior, .retrieval:
            return head + Text("\(gating.rawValue) gate: " + gateDescription(gating) + " (g switches the gate)", style: palette.dim)
        case .braided:
            break
        }
        let prompt = state.braid.promptTraces.count
        let leader = shares.indices.max { (shares[$0].memory ?? shares[$0].gate) < (shares[$1].memory ?? shares[$1].gate) } ?? 0
        let lead = shares[leader]
        let name = { (share: StrandShare) in label(share.strand, state) }
        if unrecognised(state) {
            return head + Text("no Thread recognises this prompt: the gate stays even, every Thread is asked, and the answer is a guess",
                               style: palette.dim.italic())
        }
        if let gate = state.braid.generation?.braid?.gate, gate.trajectory != .off {
            let phrase = TrajectoryRule.standard.phrase
            let lengths = shares.map { $0.trajectory?.length ?? 0 }
            if let top = lengths.indices.max(by: { lengths[$0] < lengths[$1] }), lengths[top] > phrase,
               lengths.indices.allSatisfy({ $0 == top || lengths[$0] <= phrase }) {
                return head + Text("the text follows one of \(name(shares[top]))'s documents (\(lengths[top]) tokens in order), so the gate leans to it",
                                   style: palette.text)
            }
        }
        if let other = shares.indices.first(where: { $0 != leader && (shares[$0].backs ?? 0) >= 0.5 }) {
            return head + Text("\(name(lead)) leads, and \(name(shares[other]))'s retrieval backs the same next token, so they share it", style: palette.text)
        }
        if (lead.memory ?? 0) >= 0.8 {
            let others = shares.indices.filter { $0 != leader }.map { "\(name(shares[$0])) \(knew(state, shares[$0].strand))" }.joined(separator: ", ")
            let detail = prompt > 0 ? ": it predicted \(knew(state, lead.strand)) of the prompt's \(prompt) tokens, \(others)" : ": it has been predicting the text"
            return head + Text("\(name(lead)) leads" + detail, style: palette.text)
        }
        return head + Text("the Threads know this phrasing alike: no Thread leads, so the gate is shared", style: palette.text)
    }

    static func label(_ name: String, _ state: StudioState) -> String {
        if name == BraidStrandRef.commonsName { return state.braid.commonsPack.map { "Commons · \($0)" } ?? "Commons" }
        return state.braid.nodes.first { $0.name == name }?.label ?? name
    }

    /// A prompt token: coloured by the Thread that supplied it, muted when no Thread predicted it.
    static func promptStyle(_ trace: TokenTrace, strands: [String], floor: Float, palette: Palette) -> Style {
        let alone = (trace.strands ?? []).compactMap(\.alone)
        if !alone.isEmpty, alone.allSatisfy({ $0 < floor }) { return palette.muted }
        return BraidArt.tokenStyle(trace, strands: strands, palette: palette, dimUncited: false)
    }

    static func renderStrip(_ state: StudioState, _ frame: inout Frame, _ rect: Rect) {
        let palette = frame.palette
        let glyphs = frame.glyphs
        let names = state.braid.nodes.map(\.name)
        let floor = state.braid.generation?.braid?.gate?.evidenceFloor ?? BraidGate().evidenceFloor
        func display(_ text: String) -> String { text.replacingOccurrences(of: "\n", with: glyphs.newline) }
        var pieces: [(String, Style)] = []
        let cursor = state.braid.cursor
        let prompt = state.braid.promptTraces
        let generated = state.braid.shownTraces
        if !prompt.isEmpty {
            pieces.append((display(state.braid.promptFirst ?? ""), palette.dim))
            for (i, trace) in prompt.enumerated() {
                var style = promptStyle(trace, strands: names, floor: floor, palette: palette)
                if i == cursor { style = style.reverse() }
                pieces.append((display(trace.text), style))
            }
            pieces.append((" ", palette.dim))
        } else {
            let promptText = (state.braid.generation?.prompt.text ?? (state.braid.generating || !generated.isEmpty ? state.braid.prompt.text : ""))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if let rewrite = state.braid.generation?.prompt.question {
                // The question, then the stem the umbrella rewrote it to.
                pieces.append(("Q " + rewrite.question + " ▸ ", palette.dim.italic()))
                pieces.append((rewrite.stem + " ", palette.dim))
            } else if !promptText.isEmpty {
                pieces.append((promptText + " ", palette.dim))
            }
        }
        if !generated.isEmpty || state.braid.generating { pieces.append((glyphs.prompt, palette.gold)) }
        for (i, trace) in generated.enumerated() {
            var style = BraidArt.tokenStyle(trace, strands: names, palette: palette)
            if let generation = state.braid.generation, let spanIndex = trace.spanIndex, spanIndex < generation.spans.count,
               generation.spans[spanIndex].verification?.status == .verified {
                style = style.underline()
            }
            if prompt.count + i == cursor, state.braid.generation != nil { style = style.reverse() }
            pieces.append((display(trace.text), style))
        }
        if state.braid.generating, state.braid.revealed >= state.braid.streaming.count {
            pieces.append((" " + StudioApp.spinner(state, glyphs), palette.gold))
        }
        // Wrap, keeping the cursor's line in view.
        var lines: [[(String, Style)]] = [[]]
        var width = 0
        var cursorLine = 0
        for (text, style) in pieces {
            // A token that fits on a line is not split across two.
            let tokenWidth = TerminalWidth.of(text)
            if style != palette.dim, tokenWidth <= rect.width, width + tokenWidth > rect.width {
                lines.append([])
                width = 0
            }
            for scalar in text.unicodeScalars {
                let w = TerminalWidth.of(scalar)
                if width + w > rect.width {
                    lines.append([])
                    width = 0
                }
                lines[lines.count - 1].append((String(Character(scalar)), style))
                width += w
            }
            if style.attributes.contains(.reverse) { cursorLine = lines.count - 1 }
        }
        let first = max(0, min(max(cursorLine - rect.height + 1, lines.count - rect.height), lines.count - rect.height))
        for (row, line) in lines.dropFirst(first).prefix(rect.height).enumerated() {
            var x = rect.minX
            for (piece, style) in line { x += frame.canvas.put(piece, x: x, y: rect.minY + row, style: style, clip: rect) }
        }
    }

    // MARK: Nodes

    /// A node's stage. Busy stages take the node's own strand colour, so orange means Veil only.
    static func stageBadge(_ s: StrandState, state: StudioState, glyphs: Glyphs, palette: Palette, colour: Color) -> Text {
        let spin = StudioApp.spinner(state, glyphs)
        let busy = Style(foreground: colour, background: palette.base.background)
        switch s.stage {
        case .live: return Text("\(glyphs.live) live v\(s.liveVersion ?? 0)", style: palette.green)
        case .empty: return Text("\(glyphs.idle) empty", style: palette.muted)
        case .starting: return Text("\(spin) starting", style: busy)
        case .exporting: return Text("\(spin) snapshot", style: busy)
        case .reindexing: return Text("\(spin) reindex v\(s.candidateVersion ?? 0)", style: busy)
        case .training:
            return Text("\(spin) training v\(s.candidateVersion ?? 0) · \(s.epoch ?? 0)/\(s.epochs ?? 0)", style: busy)
        case .indexing: return Text("\(spin) indexing v\(s.candidateVersion ?? 0)", style: busy)
        case .gating: return Text("\(spin) gates v\(s.candidateVersion ?? 0)", style: busy)
        case .held: return Text("■ held · live v\(s.liveVersion.map(String.init) ?? "—")", style: palette.dim.adding(.bold))
        case .failed: return Text("\(glyphs.cross) failed", style: palette.red)
        case .stopped: return Text("\(glyphs.idle) stopped", style: palette.muted)
        }
    }

    static func renderNode(_ state: StudioState, name: String, index: Int, _ frame: inout Frame, _ rect: Rect, compact: Bool) {
        let palette = frame.palette
        let glyphs = frame.glyphs
        let colour = BraidArt.strandColor(index, palette)
        let focused = state.braid.focus == index
        guard let s = state.braid.states[name] else {
            if let failure = state.braid.startFailure, !state.braid.started {
                let inner = Theme.panel(label(name, state), focused: focused,
                                        footer: Text("\(glyphs.cross) not started", style: palette.red), &frame, rect)
                Theme.empty("the braid did not start: \(failure)", &frame, inner)
                return
            }
            let inner = Theme.panel(label(name, state), focused: focused, footer: Text("\(StudioApp.spinner(state, glyphs)) starting", style: Style(foreground: colour, background: palette.base.background)), &frame, rect)
            let pid = state.braid.pids[name].map { "node process pid \($0) · starting its Thread and loading its live version…" } ?? "spawning the node process…"
            Theme.empty(pid, &frame, inner)
            return
        }
        var box = Box.panel("", focused: focused, palette: palette, glyphs: glyphs)
        box.title = Text(s.label, style: Style(foreground: colour, background: palette.base.background, attributes: [.bold]))
            + Text(" · \(s.threadID.map { String($0.prefix(8)).lowercased() } ?? "offline")", style: palette.dim)
        box.footer = stageBadge(s, state: state, glyphs: glyphs, palette: palette, colour: colour)
        let inner = box.render(in: rect, on: &frame.canvas).inset(top: 0, left: 1, bottom: 0, right: 1)
        guard inner.height >= 3 else { return }
        let (ladderRow, body) = inner.bottom(1)
        renderLadder(s, &frame, ladderRow, colour: colour)
        let phase = state.braid.phase
        let cited = citedCells(state, name: name, cells: s.cells)
        // The database needs its 14 columns and room for text beside it; the block stack 9 more.
        let showArt = !compact && body.width >= BraidArt.databaseWidth + 1 + 20 && body.height >= BraidArt.databaseRows + 1
        let showStack = showArt && body.width >= 44
        var textRect = body
        if showArt {
            let art = BraidArt.database(s, colour: colour, palette: palette, phase: phase, cited: cited, training: s.stage == .training)
            let origin = (x: body.minX, y: body.minY + 1)
            art.render(x: origin.x, y: origin.y, clip: body, on: &frame.canvas, background: palette.base.background, glyphs: glyphs,
                       depth: palette.depth)
            BraidArt.rain(s, origin: origin, clip: Rect(x: body.minX, y: body.minY, width: BraidArt.databaseWidth, height: BraidArt.databaseRows + 1),
                          colour: colour, palette: palette, phase: phase, frame: &frame)
            let caption = "\(s.cells.count) partition\(s.cells.count == 1 ? "" : "s")"
            if body.height > BraidArt.databaseRows + 1 {
                frame.canvas.put(Text(caption, style: palette.muted).truncated(to: BraidArt.databaseWidth), x: body.minX,
                                 y: body.minY + BraidArt.databaseRows + 1, clip: body)
            }
            if showStack {
                let stackHeight = min(body.height, max(3, s.blockActivity.count + 2))
                let stackRect = Rect(x: body.minX + BraidArt.databaseWidth + 1, y: body.minY, width: 9, height: stackHeight)
                BraidArt.hypervisor(s, rect: stackRect, colour: colour, pulse: pulse(state, name: name), palette: palette, phase: phase, frame: &frame)
            }
            textRect = body.inset(left: BraidArt.databaseWidth + 1 + (showStack ? 9 + 1 : 0))
        }
        renderNodeText(s, state: state, &frame, textRect, colour: colour)
    }

    /// Where the newest token's pulse is inside a node's stack (0…1), while it is still climbing.
    static func pulse(_ state: StudioState, name: String) -> Double? {
        guard let pulse = state.braid.pulses.last(where: { $0.strand == name }), pulse.open else { return nil }
        let t = Double(state.braid.phase - pulse.start) / Double(pulseTicks)
        return t <= 0.5 ? t / 0.5 : nil
    }

    /// Which of a node's cells the token under the cursor cites.
    static func citedCells(_ state: StudioState, name: String, cells: [PartitionCell]) -> Set<Int> {
        guard let generation = state.braid.generation, let strand = generation.braid?.strand(named: name) else { return [] }
        let traces = state.braid.walkable
        guard !traces.isEmpty else { return [] }
        let trace = traces[min(state.braid.cursor, traces.count - 1)]
        var result = Set<Int>()
        for citation in trace.citations where strand.contains(row: citation.row) {
            let document = citation.address.documentID
            let partition = citation.address.partitionIndex
            if let k = cells.firstIndex(where: { $0.documentID == document && $0.partition == partition }) { result.insert(k) }
        }
        return result
    }

    static func renderNodeText(_ s: StrandState, state: StudioState, _ frame: inout Frame, _ rect: Rect, colour: Color) {
        let palette = frame.palette
        let glyphs = frame.glyphs
        guard !rect.isEmpty else { return }
        var lines: [Text] = []
        let processes = "pid \(s.pid.map(String.init) ?? "—")" + (s.offline ? " · offline" : " · thread \(s.threadPID.map(String.init) ?? "—")")
        lines.append(Text(processes, style: palette.text) + Text(s.httpPort.map { " :\($0)" } ?? "", style: palette.dim))
        lines.append(Text("\(s.documents) docs · \(s.partitions) parts · \(Format.count(s.tokens)) tok", style: palette.text))
        let versions = "v\(s.liveVersion.map(String.init) ?? "—") live · \(s.versions) version\(s.versions == 1 ? "" : "s")"
        lines.append(Text(versions, style: palette.text) + Text(s.lastChange.map { " · \($0.components(separatedBy: " ").first ?? $0)" } ?? "", style: palette.dim))
        var memory = Text("memorised ", style: palette.muted) + Text(Format.pct(s.memorised), style: palette.text)
        if let candidate = s.candidateMemorised, s.stage == .training || s.stage == .gating || s.stage == .indexing {
            memory.append(" → v\(s.candidateVersion ?? 0) ", palette.muted)
            memory.append(Format.pct(candidate), Style(foreground: colour, background: palette.base.background))
        }
        if let learned = s.factsLearned, let total = s.factsTotal, total > 0 { memory.append(" · facts \(learned)/\(total)", palette.dim) }
        lines.append(memory)
        if s.heldOutLoss != nil || s.commonsLoss != nil {
            // Loss on text the node never saw: unfed documents in its own voice, and the commons sample.
            lines.append(Text("held out ", style: palette.muted) + Text(Format.f(s.heldOutLoss, 2), style: palette.text)
                + Text(s.commonsLoss.map { " · commons \(Format.f($0, 2))" } ?? "", style: palette.dim))
        }
        if s.stage == .training || !s.losses.isEmpty {
            let spark = Sparkline(s.losses.map(Double.init), style: Style(foreground: colour, background: palette.base.background), glyphs: glyphs)
                .string(width: max(4, min(18, rect.width - 16)))
            lines.append(Text("loss ", style: palette.muted) + Text(spark, style: Style(foreground: colour, background: palette.base.background))
                + Text(" \(Format.f(s.losses.last, 3))", style: palette.dim))
        }
        lines.append(Text("ckpt \(s.checkpointSHA256.map { String($0.prefix(8)) } ?? "—") · idx \(s.indexSHA256.map { String($0.prefix(6)) } ?? "—")", style: palette.dim))
        if let preview = s.preview, !preview.isEmpty {
            lines.append(Text("says ", style: palette.muted) + Text(preview.replacingOccurrences(of: "\n", with: glyphs.newline), style: Style(foreground: Color.mix(colour, palette.line, 0.5), background: palette.base.background, attributes: .italic)))
        }
        if !s.history.isEmpty {
            var history = Text("versions ", style: palette.muted)
            for mark in s.history.suffix(4) {
                history.append("v\(mark.version) ", mark.version == s.liveVersion ? Style(foreground: colour, background: palette.base.background, attributes: .bold) : palette.dim)
                history.append(mark.kind == .train ? "train" : "reidx", palette.dim)
                history.append(mark.promoted ? glyphs.check : "■", mark.promoted ? palette.green : palette.dim.adding(.bold))
                history.append(" ", palette.dim)
            }
            lines.append(history)
        }
        if let error = s.error { lines.append(Text(error, style: palette.red)) }
        if !s.gates.isEmpty, s.stage == .held || s.stage == .gating || s.stage == .live {
            var gates = Text("gates ", style: palette.muted)
            for gate in s.gates { gates.append("\(gate.passed ? glyphs.check : glyphs.cross)\(gate.name) ", gate.passed ? palette.green : palette.red) }
            lines.append(gates)
        }
        for (row, line) in lines.prefix(rect.height).enumerated() {
            frame.canvas.put(line.truncated(to: rect.width), x: rect.minX, y: rect.minY + row, clip: rect)
        }
    }

    static func renderLadder(_ s: StrandState, _ frame: inout Frame, _ rect: Rect, colour: Color) {
        let palette = frame.palette
        let glyphs = frame.glyphs
        var text = Text()
        for (i, mark) in s.ladder.enumerated() {
            if i > 0 { text.append(" ", palette.muted) }
            let style: Style
            let symbol: String
            switch mark.status {
            case .done: style = Style(foreground: colour, background: palette.base.background); symbol = glyphs.check
            case .running: style = Style(foreground: colour, background: palette.base.background).adding(.bold); symbol = glyphs.spinner[(s.updatedAt.hashValue & 3)]
            case .failed: style = palette.red; symbol = glyphs.cross
            case .skipped: style = palette.muted; symbol = glyphs.dot
            case .pending: style = palette.muted; symbol = glyphs.idle
            }
            text.append(mark.name, mark.status == .pending || mark.status == .skipped ? palette.muted : palette.dim)
            if mark.name == "diff", let detail = mark.detail, mark.status == .done {
                text.append(" " + (detail.components(separatedBy: " ").filter { $0 != "−0" && $0 != "~0" && $0 != "+0" }.joined(separator: " ")), style)
            } else {
                text.append(symbol, style)
            }
        }
        if s.ladder.isEmpty { text = Text(s.stage == .empty ? "no documents yet · f feeds this node" : "waiting for its first update", style: palette.muted) }
        frame.canvas.put(text.truncated(to: rect.width), x: rect.minX, y: rect.minY, clip: rect)
    }

    // MARK: Bottom

    static func renderBottom(_ state: StudioState, _ frame: inout Frame, _ rect: Rect) {
        let palette = frame.palette
        let glyphs = frame.glyphs
        let columns = rect.splitHorizontally([.flex(3), .flex(2)])
        let footer = Text(state.braid.exampleLabel == nil ? "typed" : (state.braid.exampleIndex >= 0 ? "example \(state.braid.exampleIndex + 1)/\(state.braid.examples.count)" : "example"), style: palette.dim)
        let tokens = Theme.panel("Tokens so far ▸ every live Thread's embedding", focused: state.braid.editingPrompt, footer: footer, &frame, columns[0])
        if !tokens.isEmpty {
            frame.canvas.put("prompt", x: tokens.minX, y: tokens.minY, style: palette.dim, clip: tokens)
            let field = Rect(x: tokens.minX + 7, y: tokens.minY, width: max(0, tokens.width - 7), height: 1)
            TextField(state: state.braid.prompt, placeholder: "/ types a prompt · x takes a fact from a Thread", focused: state.braid.editingPrompt,
                      style: palette.title, placeholderStyle: palette.muted, cursorStyle: palette.cursor).render(in: field, on: &frame.canvas)
            var row = tokens.minY + 1
            if row < tokens.maxY, let label = state.braid.exampleLabel {
                var line = Text(label, style: palette.muted)
                if let expected = state.braid.exampleExpected { line.append("  expects", palette.muted); line.append(expected, palette.gold) }
                if state.braid.exampleNode == nil { line.append("  (no Thread holds this)", palette.dim.italic()) }
                if let opener = state.braid.examples.first(where: { $0.promptTokens == state.braid.exampleTokens })?.opener {
                    line.append("  (opens with \(Self.label(opener, state))'s fact)", palette.dim)
                }
                frame.canvas.put(line.truncated(to: tokens.width), x: tokens.minX, y: row, clip: tokens)
                row += 1
            }
            // The token ids flowing into the nodes.
            let ids = state.braid.generation?.prompt.tokens ?? state.braid.exampleTokens ?? []
            let generated = state.braid.shownTraces.map(\.token)
            if row < tokens.maxY, !(ids + generated).isEmpty {
                var chips = Text("ids ", style: palette.muted)
                for id in ids.suffix(18) { chips.append("\(id) ", palette.dim) }
                if !generated.isEmpty { chips.append(glyphs.prompt + " ", palette.gold) }
                let names = state.braid.nodes.map(\.name)
                for (i, trace) in state.braid.shownTraces.enumerated() {
                    var style = BraidArt.tokenStyle(trace, strands: names, palette: palette)
                    if i == state.braid.cursor { style = style.reverse() }
                    chips.append("\(generated[i])", style)
                    chips.append(" ", palette.dim)
                }
                frame.canvas.put(chips.truncated(to: tokens.width), x: tokens.minX, y: row, clip: tokens)
                row += 1
            }
            let shown = state.braid.shownTraces
            let commons = BraidStrandRef.commonsName
            let hasCommons = shown.contains { $0.strands?.contains { $0.strand == commons } == true }
            if row + 1 < tokens.maxY, !shown.isEmpty {
                // A blank line above the legend, when the legend, the Threads and the commons still fit below it.
                if row + 2 + state.braid.nodes.count + (hasCommons ? 1 : 0) < tokens.maxY { row += 1 }
                let legend = state.braid.promptTraces.isEmpty ? "" : " · knew = prompt tokens it predicted"
                frame.canvas.put(Text("the answer, by Thread" + legend, style: palette.muted).truncated(to: tokens.width), x: tokens.minX, y: row, clip: tokens)
                row += 1
                let prompt = state.braid.promptTraces.count
                let width = max(9, ((hasCommons ? [label(commons, state)] : []) + state.braid.nodes.map(\.label)).map(\.count).max() ?? 0) + 1
                func part(_ strand: String) -> (shares: [StrandShare], mean: Float, led: Int) {
                    let shares = shown.compactMap { $0.strands?.first { $0.strand == strand } }
                    let mean = shares.isEmpty ? 0 : shares.map(\.share).reduce(0, +) / Float(shares.count)
                    return (shares, mean, shown.filter { $0.dominantStrand()?.strand == strand }.count)
                }
                for (index, spec) in state.braid.nodes.enumerated() where row < tokens.maxY - 1 {
                    let (shares, mean, led) = part(spec.name)
                    let opened = shares.filter(\.open).count
                    let colour = BraidArt.strandColor(index, palette)
                    var line = Text(spec.label.padding(toLength: width, withPad: " ", startingAt: 0), style: Style(foreground: colour, background: palette.base.background, attributes: .bold))
                    line.append(ShareBar(segments: [(Double(mean), colour)], track: palette.muted, glyphs: glyphs).text(width: 12))
                    line.append(" \(Format.pct(mean))", palette.text)
                    line.append(" · led \(led)/\(shown.count) · asked \(opened)/\(shares.count)", palette.dim)
                    if prompt > 0 { line.append(" · knew \(knew(state, spec.name))/\(prompt)", palette.dim) }
                    frame.canvas.put(line.truncated(to: tokens.width), x: tokens.minX, y: row, clip: tokens)
                    row += 1
                }
                // The commons' part of the answer: what the base model supplied, no Thread's.
                if hasCommons, row < tokens.maxY - 1 {
                    let (_, mean, led) = part(commons)
                    var line = Text(label(commons, state).padding(toLength: width, withPad: " ", startingAt: 0),
                                    style: Style(foreground: palette.goldColor, background: palette.base.background, attributes: .bold))
                    line.append(ShareBar(segments: [(Double(mean), palette.goldColor)], track: palette.muted, glyphs: glyphs).text(width: 12))
                    line.append(" \(Format.pct(mean))", palette.text)
                    line.append(" · led \(led)/\(shown.count)", palette.dim)
                    frame.canvas.put(line.truncated(to: tokens.width), x: tokens.minX, y: row, clip: tokens)
                    row += 1
                }
                if unrecognised(state), row < tokens.maxY - 1 {
                    frame.canvas.put(Text("no Thread recognises this prompt: the answer is a guess", style: palette.dim.italic()).truncated(to: tokens.width),
                                     x: tokens.minX, y: row, clip: tokens)
                    row += 1
                }
                for (index, spec) in state.braid.nodes.enumerated() where row < tokens.maxY - 1 {
                    guard let answer = state.braid.alone[spec.name] else { continue }
                    let colour = BraidArt.strandColor(index, palette)
                    let line = Text("alone    ", style: palette.muted) + Text(spec.label + " ", style: Style(foreground: colour, background: palette.base.background))
                        + Text(answer.replacingOccurrences(of: "\n", with: glyphs.newline).trimmingCharacters(in: .whitespaces), style: palette.text)
                    frame.canvas.put(line.truncated(to: tokens.width), x: tokens.minX, y: row, clip: tokens)
                    row += 1
                }
                // The RaoLM base model alone: what the commons says with no Thread asked.
                if let answer = state.braid.alone[BraidStrandRef.commonsName], row < tokens.maxY - 1 {
                    let line = Text("alone    ", style: palette.muted) + Text(label(BraidStrandRef.commonsName, state) + " ", style: palette.gold)
                        + Text(answer.replacingOccurrences(of: "\n", with: glyphs.newline).trimmingCharacters(in: .whitespaces), style: palette.text)
                    frame.canvas.put(line.truncated(to: tokens.width), x: tokens.minX, y: row, clip: tokens)
                    row += 1
                }
                if let expected = state.braid.exampleExpected, let generation = state.braid.generation,
                   generation.prompt.tokens == state.braid.exampleTokens, row < tokens.maxY - 1 {
                    let right = generation.text.hasPrefix(expected)
                    frame.canvas.put(Text("answer ", style: palette.muted) + Text(right ? "\(glyphs.check) matches the fact" : "\(glyphs.cross) differs from the fact", style: right ? palette.green : palette.red)
                        + Text("  (\(expected.trimmingCharacters(in: .whitespaces)))", style: palette.dim), x: tokens.minX, y: row, clip: tokens)
                    row += 1
                }
            }
            if row < tokens.maxY, !state.braid.verification.isEmpty {
                let ok = state.braid.verifiedAll == true
                frame.canvas.put(Text("\(ok ? glyphs.verified : glyphs.cross) " + (state.braid.verification.last ?? ""), style: ok ? palette.gold : palette.red)
                    .truncated(to: tokens.width), x: tokens.minX, y: row, clip: tokens)
                row += 1
            }
            if row < tokens.maxY {
                let legend = state.braid.nodes.enumerated().reduce(Text("colour ", style: palette.muted)) { text, pair in
                    text + Text(pair.element.label + " ", style: Style(foreground: BraidArt.strandColor(pair.offset, palette), background: palette.base.background))
                } + Text("shared ", style: Style(foreground: palette.line, background: palette.base.background)) + Text("· underline = verified span", style: palette.muted)
                frame.canvas.put(legend.truncated(to: tokens.width), x: tokens.minX, y: tokens.maxY - 1, clip: tokens)
            }
        }
        let events = Theme.panel("Events", &frame, columns[1])
        LogPane(state: state.braid.events, style: palette.dim).render(in: events, on: &frame.canvas)
    }
}
