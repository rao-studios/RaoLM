//
//  GenerateScreen.swift
//  RaoLMStudio
//
//  WHAT: 6 Generate: cited generation and the output-contribution debugger. The strip shows
//        prompt ▸ generation with every token coloured by citation confidence and `[[n]]` after
//        each verbatim span; the cursor picks a token. Beside it: the token's four entropies,
//        p_LM, kNN agreement and confidence (RaoLM) and, once grounded, its influence ι, KL,
//        drift and risk with vs without the source (SinatraHarness); the per-partition
//        citations, the retrieved neighbours, the spans and the grounding attribution; and the
//        cited partition's Thread node, document, partition, token offset and text.
//

import Foundation
import RaoLM
import RaoLMTerminal
import RaoLMWorkflows
import SinatraHarness

enum GenerateScreen: StudioScreen {
    // MARK: - State changes

    static func select(run: URL, epoch: Int?, state: inout StudioState) -> [StudioJob] {
        state.generate.run = run
        state.generate.context = nil
        state.generate.generation = nil
        state.generate.generationFile = nil
        state.generate.grounding = nil
        state.generate.streaming = []
        state.generate.exampleIndex = -1
        return [.loadContext(run: run, epoch: epoch)]
    }

    static func show(_ generation: CitedGeneration, file: URL?, rows: [[NeighbourRow]], grounding: GroundingRecord?,
                     keepCursor: Bool = false, state: inout StudioState) {
        state.generate.generation = generation
        state.generate.generationFile = file
        state.generate.neighbourRows = rows
        state.generate.grounding = grounding ?? (keepCursor ? state.generate.grounding : nil)
        state.generate.streaming = []
        if !keepCursor {
            state.generate.cursor = 0
            state.generate.tables = [:]
            state.generate.verification = []
            state.generate.verificationSource = nil
            if state.generate.tab == .grounding, grounding == nil { state.generate.tab = .citations }
        }
    }

    static func generated(_ state: StudioState) -> [TokenTrace] {
        state.generate.generation?.traces.filter { !$0.isPrompt } ?? []
    }

    /// The partition the lower panel shows, its address, the token range to highlight, and
    /// which `[[n]]` it is.
    struct Selection {
        var ref: PartitionRef
        var address: SourceAddress
        var range: Range<Int>
        var verification: VerificationStatus?
        var number: Int?
        var why: String
    }

    static func selection(_ state: StudioState) -> Selection? {
        guard let generation = state.generate.generation else { return nil }
        let traces = generated(state)
        let cursor = min(state.generate.cursor, max(0, traces.count - 1))
        let numbers = Dictionary(CitationMarkers.sources(generation).map { ($0.row, $0.number) }, uniquingKeysWith: { a, _ in a })
        func spanRange(_ span: CitedSpan) -> Range<Int> { span.source.tokenOffset..<(span.source.tokenOffset + span.tokens.count) }

        switch state.generate.tab {
        case .spans:
            guard let index = state.generate.table(.spans).selected, index < generation.spans.count else { return nil }
            let span = generation.spans[index]
            guard let ref = generation.partition(row: span.row) else { return nil }
            return Selection(ref: ref, address: span.source, range: spanRange(span), verification: span.verification?.status,
                             number: numbers[span.row], why: "span \(index + 1)")
        case .grounding:
            guard let record = state.generate.grounding, let index = state.generate.table(.grounding).selected,
                  index < record.partitions.count else { break }
            let row = record.partitions[index]
            guard let ref = generation.partitions.first(where: { $0.documentID == row.documentID && $0.partitionIndex == row.partitionIndex })
            else { break }
            return Selection(ref: ref, address: ref.address(offset: 0, threadID: generation.manifest.threadID), range: 0..<0,
                             verification: nil, number: numbers[ref.row], why: "grounding source")
        default:
            break
        }
        guard cursor < traces.count else { return nil }
        let trace = traces[cursor]
        var citation = trace.citations.first
        if state.generate.tab == .citations, let index = state.generate.table(.citations).selected, index < trace.citations.count {
            citation = trace.citations[index]
        }
        guard let citation, let ref = generation.partition(row: citation.row) else { return nil }
        var range = citation.address.tokenOffset..<(citation.address.tokenOffset + 1)
        var verification: VerificationStatus?
        if let spanIndex = trace.spanIndex, spanIndex < generation.spans.count {
            let span = generation.spans[spanIndex]
            if span.row == citation.row {
                range = spanRange(span)
                verification = span.verification?.status
            }
        }
        return Selection(ref: ref, address: citation.address, range: range, verification: verification,
                         number: numbers[citation.row], why: "token \(cursor + 1)")
    }

    /// Asks for the selected partition's text when it is not loaded yet.
    static func partitionJobs(_ state: inout StudioState) -> [StudioJob] {
        guard let selection = selection(state) else { return [] }
        let key = "\(selection.ref.documentID)#\(selection.ref.partitionIndex)"
        guard state.generate.partitionTexts[key] == nil, !state.generate.requestedTexts.contains(key) else { return [] }
        guard let run = state.generate.run ?? state.generate.generationFile?.deletingLastPathComponent().deletingLastPathComponent() else { return [] }
        state.generate.requestedTexts.insert(key)
        return [.partitionText(run: run, documentID: selection.ref.documentID, partitionIndex: selection.ref.partitionIndex)]
    }

    // MARK: - Render

    static func render(_ state: StudioState, _ frame: inout Frame, _ rect: Rect) {
        let palette = frame.palette
        guard state.generate.run != nil || state.generate.generation != nil else {
            let inner = Theme.panel("Generate", &frame, rect)
            let indexed = state.home.runs.filter { !$0.indexedEpochs.isEmpty }
            Theme.empty(indexed.isEmpty
                ? "No run has a provenance index yet. Train one on 5 Train (the final epoch is always indexed)."
                : "r picks a run (\(indexed.count) with an index), or select one on 1 Home and press ⏎. o opens a saved generation.", &frame, inner)
            return
        }
        let stripHeight = max(5, min(9, rect.height / 4))
        let partitionHeight = max(6, min(9, rect.height / 4))
        let (stripRect, rest) = rect.top(stripHeight)
        let (partitionRect, middle) = rest.bottom(partitionHeight)
        renderStrip(state, &frame, stripRect)
        let columns = middle.splitHorizontally([.fixed(min(38, middle.width / 3)), .flex(1)])
        renderToken(state, &frame, columns[0])
        renderTabs(state, &frame, columns[1])
        renderPartition(state, &frame, partitionRect)
        _ = palette
    }

    static func stripCells(_ state: StudioState) -> [TokenStrip.Cell] {
        if let generation = state.generate.generation { return TokenStrip.build(generation) }
        let prompt = state.generate.sliceMode ? state.generate.prompt.text : state.generate.prompt.text
        return TokenStrip.streaming(prompt: prompt, traces: state.generate.streaming)
    }

    static func renderStrip(_ state: StudioState, _ frame: inout Frame, _ rect: Rect) {
        let palette = frame.palette
        let glyphs = frame.glyphs
        let traces = generated(state)
        var title = "Prompt \(glyphs.prompt) generation"
        if let generation = state.generate.generation {
            let sources = CitationMarkers.sources(generation).count
            title += " · \(generation.tokens.count) tokens · \(sources) source\(sources == 1 ? "" : "s")"
        } else if !state.generate.streaming.isEmpty {
            title += " · \(state.generate.streaming.count) tokens so far"
        }
        var footer = Text(state.generate.sliceMode ? "corpus slice" : "text", style: palette.dim)
        if let source = state.generate.verificationSource, let generation = state.generate.generation {
            let checks = generation.spans.filter { $0.kind == .verbatim && $0.verification != nil }
            let ok = checks.filter { $0.verification?.status == .verified }.count
            footer = Text("\(ok)/\(checks.count) spans \(ok == checks.count ? glyphs.check : glyphs.cross) \(source)",
                          style: ok == checks.count ? palette.gold : palette.red)
        }
        let inner = Theme.panel(title, focused: state.generate.editingPrompt, footer: footer, &frame, rect)
        guard !inner.isEmpty else { return }

        // Row 0: the prompt field.
        let label = state.generate.sliceMode ? "slice " : "prompt"
        frame.canvas.put(label, x: inner.minX, y: inner.minY, style: palette.dim, clip: inner)
        let field = Rect(x: inner.minX + 7, y: inner.minY, width: inner.width - 7, height: 1)
        let placeholder = state.generate.sliceMode ? "DOCUMENT_ID:PARTITION:OFFSET:LENGTH — x cycles fact prompts"
            : "/ to type a prompt · x for a fact prompt from the corpus · m for a corpus slice"
        TextField(state: state.generate.prompt, placeholder: placeholder, focused: state.generate.editingPrompt,
                  style: palette.title, placeholderStyle: palette.muted, cursorStyle: palette.cursor).render(in: field, on: &frame.canvas)
        if state.generate.exampleIndex >= 0, let context = state.generate.context, state.generate.exampleIndex < context.examples.count,
           !state.generate.editingPrompt {
            let example = context.examples[state.generate.exampleIndex]
            let note = Text("  \(example.label) \(glyphs.prompt)", style: palette.dim) + Text(example.expected, style: palette.gold)
            let x = field.minX + TerminalWidth.of(state.generate.prompt.text) + 1
            frame.canvas.put(note.truncated(to: max(0, field.maxX - x)), x: x, y: inner.minY, clip: field)
        }

        // The strip.
        let hasBand = state.generate.grounding?.measured == true
        let stripArea = inner.inset(top: 1, left: 0, bottom: hasBand ? 1 : 0, right: 0)
        let cursor = min(state.generate.cursor, max(0, traces.count - 1))
        var lines: [[(String, Style)]] = [[]]
        var width = 0
        func add(_ text: String, _ style: Style, atomic: Bool) {
            let shown = text.replacingOccurrences(of: "\n", with: glyphs.newline)
            let w = TerminalWidth.of(shown)
            if atomic, w <= stripArea.width, width + w > stripArea.width {
                lines.append([])
                width = 0
            }
            for scalar in shown.unicodeScalars {
                let sw = TerminalWidth.of(scalar)
                if width + sw > stripArea.width {
                    lines.append([])
                    width = 0
                }
                lines[lines.count - 1].append((String(Character(scalar)), style))
                width += sw
            }
        }
        let cells = stripCells(state)
        if state.generate.generation == nil, state.generate.streaming.isEmpty {
            Theme.empty(state.generate.context == nil ? "loading the run…" : "⏎ generates from the prompt", &frame, stripArea)
            return
        }
        for cell in cells {
            switch cell {
            case .prompt(let text):
                add(text, palette.dim, atomic: false)
            case .separator:
                add(glyphs.prompt, palette.gold, atomic: true)
            case .token(let index, let text, let confidence, let uncited, let verified):
                var style = uncited && confidence == nil && state.generate.generation != nil
                    ? palette.heat[0] : palette.heat(confidence: confidence, verified: verified)
                if state.generate.generation == nil { style = palette.text }
                if index == cursor, state.generate.generation != nil { style = style.adding(.reverse) }
                add(text, style, atomic: true)
            case .marker(let number, let verified):
                add("[[\(number)]]", verified ? palette.gold : palette.muted, atomic: true)
            }
        }
        // Keep the cursor's line in view.
        var cursorLine = 0
        for (index, line) in lines.enumerated() where line.contains(where: { $0.1.attributes.contains(.reverse) }) { cursorLine = index }
        let first = max(0, min(cursorLine - stripArea.height + 1, lines.count - stripArea.height))
        for (row, line) in lines.dropFirst(first).prefix(stripArea.height).enumerated() {
            var x = stripArea.minX
            for (piece, style) in line { x += frame.canvas.put(piece, x: x, y: stripArea.minY + row, style: style, clip: stripArea) }
        }
        if hasBand, let record = state.generate.grounding {
            let y = inner.maxY - 1
            var band = Text("grounding ", style: palette.dim)
            for row in record.tokens where !row.isEOS {
                var style = classStyle(row.kind, palette)
                if row.step == cursor { style = style.adding(.reverse) }
                band.append(classGlyph(row.kind, glyphs), style)
            }
            let s = record.measurement.summary
            band.append("   grounded \(Format.pct(s.grounding)) · drift \(Format.f(s.drift, 2)) · risk \(Format.f(s.hallucinationRisk, 2)) · Σι \(GroundingTables.signed(s.contextDependence)) nats", palette.dim)
            frame.canvas.put(band.truncated(to: inner.width), x: inner.minX, y: y, clip: inner)
        }
    }

    static func classGlyph(_ kind: GroundingClass, _ glyphs: Glyphs) -> String {
        switch kind {
        case .grounded: return glyphs.live
        case .unsupported: return glyphs.spinner[0]
        case .contradicted: return glyphs.cross
        case .function: return glyphs.dot
        }
    }

    static func classStyle(_ kind: GroundingClass, _ palette: Palette) -> Style {
        switch kind {
        case .grounded: return palette.green
        case .unsupported: return palette.orange
        case .contradicted: return palette.red
        case .function: return palette.muted
        }
    }

    static func bar(_ value: Float?, max: Float, width: Int, _ glyphs: Glyphs) -> String {
        guard let value, value.isFinite, max > 0 else { return "" }
        let filled = Int((Swift.min(Swift.max(value / max, 0), 1) * Float(width)).rounded())
        return String(repeating: glyphs.barFull, count: filled) + String(repeating: glyphs.barEmpty, count: width - filled)
    }

    static func renderToken(_ state: StudioState, _ frame: inout Frame, _ rect: Rect) {
        let palette = frame.palette
        let glyphs = frame.glyphs
        let traces = generated(state)
        guard !traces.isEmpty else {
            let inner = Theme.panel("Token", &frame, rect)
            Theme.empty("←→ moves through the generated tokens once there are some.", &frame, inner)
            return
        }
        let cursor = min(state.generate.cursor, traces.count - 1)
        let trace = traces[cursor]
        let inner = Theme.panel("Token \(cursor + 1)/\(traces.count) «\(trace.text.replacingOccurrences(of: "\n", with: glyphs.newline))»", focused: true, &frame, rect)
        let entropyMax = Swift.max(trace.lmEntropy, trace.knnEntropy, trace.mixedEntropy, 1)
        func entropy(_ raw: Float) -> Text {
            let value = Swift.max(0, raw)  // entropies are ≥ 0; float noise can print as -0.00
            return Text(Format.f(value, 2).padding(toLength: 6, withPad: " ", startingAt: 0)) + Text(bar(value, max: entropyMax, width: 8, glyphs), style: palette.blue)
        }
        var confidence = Text(Format.f(trace.confidence, 2) + "  ")
        let pips = Int(((trace.confidence ?? 0) * 4).rounded())
        confidence.append(String(repeating: glyphs.live, count: pips) + String(repeating: glyphs.idle, count: 4 - pips),
                          palette.heat(confidence: trace.confidence, verified: false))
        var rows: [(String, Text)] = [
            ("H_lm", entropy(trace.lmEntropy)),
            ("H_knn", entropy(trace.knnEntropy)),
            ("H_mix", entropy(trace.mixedEntropy)),
            ("H_source", entropy(trace.sourceEntropy)),
            ("p_lm", Text(Format.f(trace.lmProb, 3)) + Text("   agree ", style: palette.dim) + Text(Format.f(trace.agreement, 2))),
            ("p_mix", Text(Format.f(trace.mixedProb, 3)) + Text("   λ ", style: palette.dim) + Text(Format.f(trace.lambda, 2))),
            ("confidence", confidence),
            ("cited", trace.uncited ? Text("\(glyphs.idle) uncited", style: palette.muted)
                : Text("\(glyphs.check) \(trace.citations.count) partition\(trace.citations.count == 1 ? "" : "s")", style: palette.green)
                    + Text(trace.spanIndex.map { " · span \($0 + 1)" } ?? "", style: palette.dim)),
        ]
        if let record = state.generate.grounding, record.measured, let row = record.tokens.first(where: { $0.step == cursor }) {
            rows.append(("ι · KL", Text(GroundingTables.signed(row.influence), style: row.influence >= 0 ? palette.green : palette.red)
                + Text(" nats · KL \(Format.f(row.contextKL, 2))", style: palette.dim)))
            rows.append(("drift", Text(Format.f(row.drift, 2)) + Text("  risk ", style: palette.dim) + Text(Format.f(row.risk, 2), style: row.risk > 0.3 ? palette.orange : palette.text)))
            rows.append(("H ctx|bare", Text("\(Format.f(row.entropyCtx, 2)) | \(Format.f(row.entropyBare, 2))")))
            rows.append(("class", Text("\(classGlyph(row.kind, glyphs)) \(row.kind.rawValue)", style: classStyle(row.kind, palette))
                + Text(row.drift > 0.05 ? "  tune \((row.tuneText ?? "#\(row.tune)").debugDescription)" : "", style: palette.dim)))
        } else {
            rows.append(("grounding", Text("G: with vs without source", style: palette.muted)))
        }
        Theme.keyValues(rows, &frame, inner, labelWidth: 11)
    }

    static func renderTabs(_ state: StudioState, _ frame: inout Frame, _ rect: Rect) {
        let palette = frame.palette
        let glyphs = frame.glyphs
        var title = Text()
        for (index, tab) in GenerateState.Tab.allCases.enumerated() {
            if index > 0 { title.append(" │ ", palette.border) }
            title.append(tab.title, tab == state.generate.tab ? palette.blue.bold() : palette.dim)
        }
        let box = Box(title: title, footer: Text("tab switches", style: palette.muted), style: palette.focusBorder, glyphs: glyphs)
        let inner = box.render(in: rect, on: &frame.canvas).inset(top: 0, left: 1, bottom: 0, right: 1)
        guard let generation = state.generate.generation else {
            Theme.empty(state.generate.streaming.isEmpty ? "" : "generating…", &frame, inner)
            return
        }
        let traces = generated(state)
        let cursor = min(state.generate.cursor, max(0, traces.count - 1))
        let tableState = state.generate.table(state.generate.tab)
        switch state.generate.tab {
        case .citations:
            guard cursor < traces.count else { return }
            let trace = traces[cursor]
            if trace.citations.isEmpty {
                Theme.empty("No retrieved context predicted this token: it is uncited by construction. Neighbours shows what retrieval found instead.", &frame, inner)
                return
            }
            let attribution = Dictionary((state.generate.grounding?.partitions ?? []).map { ("\($0.documentID)#\($0.partitionIndex)", $0.nats) },
                                         uniquingKeysWith: { a, _ in a })
            let hasAP = !attribution.isEmpty
            let rows: [[Text]] = trace.citations.enumerated().map { index, c in
                let name = generation.partition(row: c.row)?.documentName ?? "?"
                var row: [Text] = [
                    Text(String(index + 1)), Text(name), Text("p\(c.address.partitionIndex)"), Text(String(c.address.tokenOffset)),
                    Text(Format.f(c.weight, 2)), Text(String(c.bestRank)), Text(Format.f(c.bestScore, 3)), Text(Format.f(c.sourceLoss, 3)),
                    Text(c.memorisedAtEpoch.map(String.init) ?? "—", style: palette.dim),
                    Text(Format.f(c.confidence, 2), style: palette.heat(confidence: c.confidence, verified: false)),
                ]
                if hasAP { row.append(Text(Format.f(attribution["\(c.address.documentID)#\(c.address.partitionIndex)"] ?? nil, 2))) }
                return row
            }
            var columns = [Column("#", .fixed(2), align: .trailing), Column("document", .flex(1)), Column("p", .fixed(3)),
                           Column("off", .fixed(4), align: .trailing), Column("weight", .fixed(6), align: .trailing),
                           Column("rank", .fixed(4), align: .trailing), Column("score", .fixed(6), align: .trailing),
                           Column("src loss", .fixed(8), align: .trailing), Column("mem@", .fixed(4), align: .trailing),
                           Column("conf", .fixed(5), align: .trailing)]
            if hasAP { columns.append(Column("A_p", .fixed(5), align: .trailing)) }
            Table.styled(columns: columns, rows: rows, state: tableState, palette: palette, glyphs: glyphs).render(in: inner, on: &frame.canvas)
        case .neighbours:
            guard cursor < state.generate.neighbourRows.count else {
                Theme.empty("no neighbours recorded", &frame, inner)
                return
            }
            let rows: [[Text]] = state.generate.neighbourRows[cursor].map { r in
                let n = r.neighbour
                return [Text(String(n.rank)), Text(Format.f(n.score, 3)), Text(Format.f(n.weight, 3)),
                        Text(r.valueText.debugDescription), n.matches ? Text(glyphs.check, style: palette.green) : Text(glyphs.cross, style: palette.muted),
                        Text("r\(n.key.row) o\(n.key.offset)", style: palette.dim), Text("r\(n.cited.row) o\(n.cited.offset)", style: palette.dim),
                        Text(Format.f(n.sourceLoss, 3)), Text(Format.f(n.sourceEntropy, 3))]
            }
            Table.styled(columns: [Column("rank", .fixed(4), align: .trailing), Column("score", .fixed(6), align: .trailing),
                                   Column("weight", .fixed(6), align: .trailing), Column("value", .flex(1)), Column("✓", .fixed(1)),
                                   Column("key r/o", .fixed(10)), Column("cited r/o", .fixed(10)),
                                   Column("src loss", .fixed(8), align: .trailing), Column("src H", .fixed(6), align: .trailing)],
                         rows: rows, state: tableState, palette: palette, glyphs: glyphs).render(in: inner, on: &frame.canvas)
        case .spans:
            if generation.spans.isEmpty {
                Theme.empty("No verbatim spans: no run of tokens reproduced a corpus partition at rank ≤ 3.", &frame, inner)
                return
            }
            let rows: [[Text]] = generation.spans.enumerated().map { index, span in
                let status = span.verification?.status
                let statusStyle = status == .verified ? palette.gold : (status == nil || status == .unverified ? palette.dim : palette.red)
                return [Text(String(index + 1)), Text(span.kind == .verbatim ? "verbatim" : "prompt", style: span.kind == .verbatim ? .plain : palette.dim),
                        Text("\(span.tokenRange.start)..<\(span.tokenRange.end)"),
                        Text("\(span.documentName) p\(span.source.partitionIndex)@\(span.source.tokenOffset)"),
                        Text(String(span.tokens.count)), Text(Format.f(span.confidence, 2)), Text(Format.f(span.distinctiveness, 2)),
                        Text(status?.rawValue ?? "unchecked", style: statusStyle)]
            }
            Table.styled(columns: [Column("#", .fixed(2), align: .trailing), Column("kind", .fixed(8)), Column("range", .fixed(9)),
                                   Column("source", .flex(1)), Column("toks", .fixed(4), align: .trailing),
                                   Column("conf", .fixed(5), align: .trailing), Column("distinct", .fixed(8), align: .trailing),
                                   Column("status", .fixed(10))],
                         rows: rows, state: tableState, palette: palette, glyphs: glyphs).render(in: inner, on: &frame.canvas)
        case .grounding:
            guard let record = state.generate.grounding else {
                Theme.empty("G measures this generation with SinatraHarness: the same output is scored with its source partitions in front of the prompt and without them. ι = log p(with) − log p(without) per token; drift = KL − ι; risk = 1 − p on tokens the source did not back. A grounds against every cited partition.", &frame, inner)
                return
            }
            let summary = GroundingTables.summaryLines(record)
            let statsLines = record.measured ? GroundingTables.statsLines(record) : []
            let tableHeight = record.measured ? min(record.partitions.count + 2, 6) : 0
            let wrapped = (Array(summary.dropFirst()) + statsLines.prefix(2)).enumerated().flatMap { index, line in
                Paragraph.wrap(Text(line, style: index == 0 ? palette.text : palette.dim), width: max(1, inner.width))
            }
            let textLines = Array(wrapped.prefix(max(0, inner.height - tableHeight - 1)))
            for (row, line) in textLines.enumerated() {
                frame.canvas.put(line, x: inner.minX, y: inner.minY + row, clip: inner)
            }
            let tableRect = inner.inset(top: textLines.count + 1, left: 0, bottom: 0, right: 0)
            guard record.measured, tableRect.height > 2 else { return }
            let rows: [[Text]] = record.partitions.map { p in
                [Text(p.documentName), Text("p\(p.partitionIndex)"), Text(Format.f(p.citationWeight, 2)), Text(Format.f(p.meanCitationConfidence, 2)),
                 Text(Format.f(p.nats, 2), style: palette.title), Text(Format.pct(p.uptake)), Text(Format.f(p.missed, 2)),
                 Text(Format.pct(p.intent)), Text(Format.pct(p.coverage)), Text(Format.pct(p.parrot))]
            }
            Table.styled(columns: [Column("source", .flex(1)), Column("p", .fixed(3)), Column("cite w", .fixed(6), align: .trailing),
                                   Column("conf", .fixed(4), align: .trailing), Column("A_p", .fixed(5), align: .trailing),
                                   Column("uptake", .fixed(6), align: .trailing), Column("missed", .fixed(6), align: .trailing),
                                   Column("intent", .fixed(6), align: .trailing), Column("cover", .fixed(5), align: .trailing),
                                   Column("parrot", .fixed(6), align: .trailing)],
                         rows: rows, state: tableState, palette: palette, glyphs: glyphs).render(in: tableRect, on: &frame.canvas)
        }
    }

    static func renderPartition(_ state: StudioState, _ frame: inout Frame, _ rect: Rect) {
        let palette = frame.palette
        let glyphs = frame.glyphs
        guard let selection = selection(state) else {
            let inner = Theme.panel("Partition", &frame, rect)
            Theme.empty(state.generate.generation == nil ? "" : "This token has no citation. Pick another token, or the Spans tab.", &frame, inner)
            return
        }
        let ref = selection.ref
        let title = "Partition" + (selection.number.map { " [[\($0)]]" } ?? "") + " · \(selection.why)"
        var footer: Text?
        if let status = selection.verification {
            footer = Text("\(status == .verified ? glyphs.verified : glyphs.cross) \(status.rawValue)", style: status == .verified ? palette.gold : palette.red)
        }
        let inner = Theme.panel(title, footer: footer, &frame, rect)
        let threadID = selection.address.threadID ?? state.generate.generation?.manifest.threadID
        var rows: [(String, Text)] = [
            ("thread node", Text(threadID ?? "offline snapshot (no node)", style: threadID == nil ? palette.muted : palette.text)
                + Text("   document ", style: palette.dim) + Text(ref.documentID) + Text("  \(ref.documentName)", style: palette.title)),
            ("partition", Text("p\(ref.partitionIndex) · offset \(selection.address.tokenOffset)", style: palette.text)
                + Text(" · \(ref.tokenCount) tokens · memorised @ epoch \(ref.memorisedAtEpoch.map(String.init) ?? "—") · textSHA256 \(Format.short(ref.textSHA256))", style: palette.dim)),
            ("url", Text(ref.partitionURL ?? "—", style: palette.blue) + Text("   thread partition ", style: palette.dim)
                + Text(Theme.short(ref.threadPartitionID, 18), style: palette.dim)),
        ]
        let key = "\(ref.documentID)#\(ref.partitionIndex)"
        let textRect = inner.inset(top: rows.count, left: 0, bottom: 0, right: 0)
        Theme.keyValues(rows, &frame, inner, labelWidth: 12)
        rows.removeAll()
        guard let text = state.generate.partitionTexts[key] else {
            Theme.empty("reading the partition…", &frame, textRect)
            return
        }
        var styled = Text()
        if let bytes = text.byteRange(tokens: selection.range.lowerBound, selection.range.upperBound) {
            let utf8 = Array(text.text.utf8)
            let lower = min(bytes.lowerBound, utf8.count)
            let upper = min(bytes.upperBound, utf8.count)
            styled.append(String(decoding: utf8[..<lower], as: UTF8.self), palette.dim)
            styled.append(String(decoding: utf8[lower..<upper], as: UTF8.self), palette.gold.underline())
            styled.append(String(decoding: utf8[upper...], as: UTF8.self), palette.dim)
        } else {
            styled.append(text.text, palette.dim)
        }
        let lines = Paragraph.wrap(styled, width: max(1, textRect.width))
        // Start at the line holding the highlight.
        var start = 0
        for (index, line) in lines.enumerated() where line.spans.contains(where: { $0.style.attributes.contains(.underline) }) {
            start = index
            break
        }
        start = max(0, min(start, lines.count - textRect.height))
        for (row, line) in lines.dropFirst(start).prefix(textRect.height).enumerated() {
            frame.canvas.put(line, x: textRect.minX, y: textRect.minY + row, clip: textRect)
        }
    }

    // MARK: - Keys

    static func tableRowCount(_ state: StudioState) -> Int {
        guard let generation = state.generate.generation else { return 0 }
        let traces = generated(state)
        let cursor = min(state.generate.cursor, max(0, traces.count - 1))
        switch state.generate.tab {
        case .citations: return cursor < traces.count ? traces[cursor].citations.count : 0
        case .neighbours: return cursor < state.generate.neighbourRows.count ? state.generate.neighbourRows[cursor].count : 0
        case .spans: return generation.spans.count
        case .grounding: return state.generate.grounding?.partitions.count ?? 0
        }
    }

    static func handle(_ key: KeyEvent, _ state: inout StudioState) -> [StudioJob]? {
        if state.generate.editingPrompt {
            switch key.key {
            case .escape:
                state.generate.editingPrompt = false
                return []
            case .enter:
                state.generate.editingPrompt = false
                return generate(&state)
            default:
                state.generate.prompt.handle(key)
                return []
            }
        }
        let count = generated(state).count
        switch key.key {
        case .char("/"), .char("p"):
            state.generate.editingPrompt = true
        case .char("m"):
            state.generate.sliceMode.toggle()
            state.generate.prompt.set("")
            state.generate.exampleIndex = -1
        case .char("x"):
            guard let context = state.generate.context, !context.examples.isEmpty else {
                state.status.message = "this run has no located facts to take prompts from"
                return []
            }
            state.generate.exampleIndex = (state.generate.exampleIndex + 1) % context.examples.count
            state.generate.sliceMode = true
            state.generate.prompt.set(context.examples[state.generate.exampleIndex].slice.spec)
        case .enter:
            return generate(&state)
        case .left, .char("h"):
            state.generate.cursor = max(0, state.generate.cursor - 1)
            state.generate.tables[.citations] = TableState(selected: 0)
            state.generate.tables[.neighbours] = TableState(selected: 0)
            return partitionJobs(&state)
        case .right, .char("l"):
            state.generate.cursor = min(max(0, count - 1), state.generate.cursor + 1)
            state.generate.tables[.citations] = TableState(selected: 0)
            state.generate.tables[.neighbours] = TableState(selected: 0)
            return partitionJobs(&state)
        case .home:
            state.generate.cursor = 0
            return partitionJobs(&state)
        case .end:
            state.generate.cursor = max(0, count - 1)
            return partitionJobs(&state)
        case .tab, .backTab:
            let all = GenerateState.Tab.allCases
            let delta = key.key == .tab ? 1 : all.count - 1
            state.generate.tab = all[(state.generate.tab.rawValue + delta) % all.count]
            return partitionJobs(&state)
        case .up, .char("k"), .down, .char("j"), .pageUp, .pageDown:
            var table = state.generate.table(state.generate.tab)
            let rows = tableRowCount(state)
            let visible = max(3, state.size.height / 4)
            switch key.key {
            case .up, .char("k"): table.move(by: -1, rowCount: rows, visible: visible)
            case .down, .char("j"): table.move(by: 1, rowCount: rows, visible: visible)
            case .pageUp: table.page(by: -1, rowCount: rows, visible: visible)
            default: table.page(by: 1, rowCount: rows, visible: visible)
            }
            state.generate.tables[state.generate.tab] = table
            return partitionJobs(&state)
        case .char("v"), .char("V"):
            guard let generation = state.generate.generation, let run = runDirectory(state) else { return [] }
            return [.verify(generation, run: run, live: key.key == .char("V"))]
        case .char("s"):
            guard let generation = state.generate.generation, let run = runDirectory(state) else { return [] }
            return [.saveGeneration(generation, run: run)]
        case .char("o"):
            guard let run = runDirectory(state) ?? state.selectedRun?.directory else {
                state.status.message = "pick a run first (r)"
                return []
            }
            return [.listGenerations(run: run)]
        case .char("G"), .char("A"):
            guard let generation = state.generate.generation, let run = runDirectory(state) else { return [] }
            let policy = key.key == .char("A") ? "all" : (generation.prompt.source != nil ? "fact" : "top")
            return [.ground(GroundSpec(generation: generation, run: run, policy: policy))]
        case .char("P"):
            state.overlay = .params(FormState.params(state.generate.overrides))
        case .char(","), .char("."):
            guard let context = state.generate.context, let index = context.indexedEpochs.firstIndex(of: context.epoch) else { return [] }
            let next = key.key == .char(".") ? index + 1 : index - 1
            guard next >= 0, next < context.indexedEpochs.count else { return [] }
            return select(run: context.runDirectory, epoch: context.indexedEpochs[next], state: &state)
        case .char("r"):
            let runs = state.home.runs.filter { !$0.indexedEpochs.isEmpty }
            guard !runs.isEmpty else {
                state.status.message = "no run has a provenance index yet"
                return []
            }
            state.overlay = .picker(Picker(
                title: "Run to generate from", items: runs.map { "\($0.id)  \($0.preset)  epochs \($0.indexedEpochs.map(String.init).joined(separator: ","))" },
                values: runs.map(\.directory), table: TableState(selected: 0), purpose: .run(.generate)))
        case .char("e"):
            state.eval.run = runDirectory(state)
            return StudioApp.go(to: .eval, state: &state)
        case .char("c"):
            return [.cancel]
        default:
            return nil
        }
        return []
    }

    static func runDirectory(_ state: StudioState) -> URL? {
        state.generate.run ?? state.generate.generationFile?.deletingLastPathComponent().deletingLastPathComponent()
    }

    static func generate(_ state: inout StudioState) -> [StudioJob] {
        guard let context = state.generate.context else {
            state.status.error = RaoLMFailure("no run loaded", hint: "r picks one", code: 66)
            return []
        }
        guard state.status.mlxJob == nil else {
            state.status.error = RaoLMFailure("the MLX worker is busy: \(state.status.mlxJob!.label)", hint: "press c to cancel it", code: 75)
            return []
        }
        let text = state.generate.prompt.text
        var slice: CorpusSlice?
        if state.generate.sliceMode {
            do { slice = try CorpusSlice.parse(text.trimmingCharacters(in: .whitespaces)) } catch {
                state.status.error = FailureMapping.describe(error)
                return []
            }
        } else if text.trimmingCharacters(in: .whitespaces).isEmpty {
            state.status.error = RaoLMFailure("the prompt is empty", hint: "/ to type one, x for a fact prompt", code: 64)
            return []
        }
        state.generate.generation = nil
        state.generate.grounding = nil
        state.generate.streaming = []
        state.generate.verification = []
        state.generate.verificationSource = nil
        return [.generate(GenerateSpec(run: context.runDirectory, epoch: context.epoch, promptText: text, slice: slice,
                                       overrides: state.generate.overrides))]
    }

    static func hints(_ state: StudioState) -> [KeyHint] {
        if state.generate.editingPrompt { return [KeyHint("⏎", "generate"), KeyHint("esc", "stop editing")] }
        var hints = [KeyHint("/", "prompt"), KeyHint("x", "fact prompt"), KeyHint("⏎", "generate"), KeyHint("←→", "token"), KeyHint("tab", "panel")]
        if state.generate.generation != nil {
            hints += [KeyHint("v", "verify"), KeyHint("G", "ground"), KeyHint("s", "save")]
        }
        hints += [KeyHint("o", "open"), KeyHint("P", "params"), KeyHint("r", "run")]
        return hints
    }
}
