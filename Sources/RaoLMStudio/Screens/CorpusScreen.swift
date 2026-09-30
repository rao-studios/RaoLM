//
//  CorpusScreen.swift
//  RaoLMStudio
//
//  WHAT: 3 Corpus: generate the synthetic Veldmar archive, browse its documents, partitions
//        and facts, deposit it into the Thread, export it back as a hashed snapshot — or take
//        an offline snapshot so training needs no Thread.
//

import Foundation
import RaoLM
import RaoLMTerminal
import RaoLMWorkflows

enum CorpusScreen: StudioScreen {
    static func render(_ state: StudioState, _ frame: inout Frame, _ rect: Rect) {
        let palette = frame.palette
        let glyphs = frame.glyphs
        let columns = rect.splitHorizontally([.fixed(min(46, rect.width / 3)), .flex(1)])
        let formHeight = state.corpus.form.fields.count + 3
        let (formRect, leftRest) = columns[0].top(formHeight)
        let (listRect, resultRect) = leftRest.top(max(4, leftRest.height - 8))
        let form = Theme.panel("Generate", focused: state.corpus.pane == .form, &frame, formRect)
        state.corpus.form.render(in: form.inset(top: 1), focused: state.corpus.pane == .form, frame: &frame, labelWidth: 12)

        let list = Theme.panel("Corpora · \(state.corpus.corpora.count)", focused: state.corpus.pane == .corpora, &frame, listRect)
        if state.corpus.corpora.isEmpty {
            Theme.empty("none under \(StudioApp.abbreviate(state.root.url.path))/corpora", &frame, list)
        } else {
            let rows: [[Text]] = state.corpus.corpora.map { c in
                let loaded = c.directory.standardizedFileURL == state.corpus.loadedDirectory?.standardizedFileURL
                return [Text(c.manifest.slug, style: loaded ? palette.gold : .plain), Text(String(c.manifest.documentCount)),
                        Text(String(c.manifest.factCount)), Text(Format.short(c.manifest.corpusHash), style: palette.dim)]
            }
            Table.styled(columns: [Column("slug", .flex(1)), Column("docs", .fixed(5), align: .trailing),
                                   Column("facts", .fixed(5), align: .trailing), Column("hash", .fixed(13))],
                         rows: rows, state: state.corpus.pane == .corpora ? state.corpus.corporaTable : nil,
                         palette: palette, glyphs: glyphs).render(in: list, on: &frame.canvas)
        }

        var footer: Text?
        if let (done, total) = state.corpus.ingest {
            footer = Text("\(StudioApp.spinner(state, glyphs)) \(done)/\(total)", style: palette.orange)
        }
        let results = Theme.panel("Thread and snapshots", footer: footer, &frame, resultRect)
        if let (done, total) = state.corpus.ingest, results.height > 0 {
            ProgressBar(fraction: total > 0 ? Double(done) / Double(total) : 0, label: Text("ingest", style: palette.dim),
                        fillStyle: palette.blue, trackStyle: palette.muted, full: glyphs.progressFull, empty: glyphs.progressEmpty)
                .render(in: results.row(0), on: &frame.canvas)
        }
        let lineRect = state.corpus.ingest == nil ? results : results.inset(top: 1)
        if state.corpus.lines.isEmpty {
            Theme.empty("i deposits the loaded corpus into the Thread · p exports it back and checks it byte for byte · O snapshots it offline", &frame, lineRect)
        }
        for (row, (line, ok)) in state.corpus.lines.prefix(lineRect.height).enumerated() {
            let style = line.hasPrefix("  ") ? palette.red : (ok ? palette.text : palette.orange)
            frame.canvas.put(Text(line, style: style).truncated(to: lineRect.width), x: lineRect.minX, y: lineRect.minY + row, clip: lineRect)
        }

        renderDocuments(state, &frame, columns[1])
    }

    static func renderDocuments(_ state: StudioState, _ frame: inout Frame, _ rect: Rect) {
        let palette = frame.palette
        guard let corpus = state.corpus.loaded else {
            let inner = Theme.panel("Documents", &frame, rect)
            Theme.empty("Select a corpus (tab to the list, ⏎) or generate one. The archive is deterministic: every fact is stated exactly once, in one partition.", &frame, inner)
            return
        }
        let m = corpus.manifest
        let (tableRect, previewRect) = rect.top(max(6, rect.height * 2 / 5))
        let focused = state.corpus.pane == .documents
        let inner = Theme.panel("\(m.slug) · \(m.documentCount) documents · \(m.partitionCount) partitions · \(m.factCount) facts",
                                focused: focused, footer: Text(Format.short(m.corpusHash), style: palette.dim), &frame, tableRect)
        let rows: [[Text]] = corpus.documents.map { d in
            [Text(d.name), Text(d.kind.rawValue, style: palette.dim), Text(String(d.partitions.count)), Text(String(d.facts.count)),
             Text(d.id, style: palette.muted)]
        }
        Table.styled(columns: [Column("document", .fixed(28)), Column("kind", .fixed(10)), Column("parts", .fixed(5), align: .trailing),
                               Column("facts", .fixed(5), align: .trailing), Column("id", .flex(1))],
                     rows: rows, state: state.corpus.documentsTable, palette: palette, glyphs: frame.glyphs)
            .render(in: inner, on: &frame.canvas)

        guard let index = state.corpus.documentsTable.selected, index < corpus.documents.count else { return }
        let document = corpus.documents[index]
        let preview = Theme.panel(document.name, &frame, previewRect)
        var lines: [Text] = []
        for partition in document.partitions {
            var text = Text("[p\(partition.index)] ", style: palette.gold)
            text.append(highlight(partition.text, facts: document.facts.filter { $0.partitionIndex == partition.index }, palette: palette))
            lines.append(contentsOf: Paragraph.wrap(text, width: max(1, preview.width)))
            lines.append(Text(partition.url, style: palette.muted))
        }
        for fact in document.facts {
            var text = Text("fact \(fact.kind.rawValue): ", style: palette.dim)
            text.append(fact.prompt, palette.text)
            text.append(" \(frame.glyphs.prompt)", palette.muted)
            text.append(fact.answer, palette.gold)
            lines.append(text)
        }
        for (row, line) in lines.prefix(preview.height).enumerated() {
            frame.canvas.put(line.truncated(to: preview.width), x: preview.minX, y: preview.minY + row, clip: preview)
        }
    }

    /// The partition text with each fact's answer bytes in gold.
    static func highlight(_ text: String, facts: [Fact], palette: Palette) -> Text {
        let bytes = Array(text.utf8)
        var ranges = facts.map { max(0, $0.answerStart)..<min(bytes.count, $0.answerEnd) }.filter { !$0.isEmpty }.sorted { $0.lowerBound < $1.lowerBound }
        var result = Text()
        var cursor = 0
        while let range = ranges.first {
            ranges.removeFirst()
            guard range.lowerBound >= cursor else { continue }
            result.append(String(decoding: bytes[cursor..<range.lowerBound], as: UTF8.self), palette.text)
            result.append(String(decoding: bytes[range], as: UTF8.self), palette.gold.underline())
            cursor = range.upperBound
        }
        result.append(String(decoding: bytes[cursor...], as: UTF8.self), palette.text)
        return result
    }

    static func selectedCorpus(_ state: StudioState) -> URL? {
        if let loaded = state.corpus.loadedDirectory { return loaded }
        guard let index = state.corpus.corporaTable.selected, index < state.corpus.corpora.count else { return nil }
        return state.corpus.corpora[index].directory
    }

    static func handle(_ key: KeyEvent, _ state: inout StudioState) -> [StudioJob]? {
        if state.corpus.form.editing || state.corpus.pane == .form {
            switch state.corpus.form.handle(key) {
            case .changed: return []
            case .submit: return submit(&state)
            case .unhandled: break
            }
        }
        switch key.key {
        case .tab:
            state.corpus.pane = CorpusState.Pane(rawValue: (state.corpus.pane.rawValue + 1) % 3)!
            if state.corpus.pane == .documents, state.corpus.loaded == nil { state.corpus.pane = .form }
            return []
        case .backTab:
            state.corpus.pane = CorpusState.Pane(rawValue: (state.corpus.pane.rawValue + 2) % 3)!
            return []
        case .up, .char("k"), .down, .char("j"), .pageUp, .pageDown:
            let delta: Int
            switch key.key {
            case .up, .char("k"): delta = -1
            case .down, .char("j"): delta = 1
            case .pageUp: delta = -10
            default: delta = 10
            }
            if state.corpus.pane == .corpora {
                state.corpus.corporaTable.move(by: delta, rowCount: state.corpus.corpora.count, visible: 8)
            } else if state.corpus.pane == .documents, let corpus = state.corpus.loaded {
                state.corpus.documentsTable.move(by: delta, rowCount: corpus.documents.count, visible: max(3, state.size.height / 3))
            }
            return []
        case .enter:
            if state.corpus.pane == .corpora, let index = state.corpus.corporaTable.selected, index < state.corpus.corpora.count {
                state.corpus.pane = .documents
                return [.corpusLoad(state.corpus.corpora[index].directory)]
            }
            return []
        case .char("i"):
            guard let corpus = selectedCorpus(state) else { return noCorpus(&state) }
            state.corpus.ingest = (0, state.corpus.loaded?.documents.count ?? 0)
            return [.ingest(corpus: corpus)]
        case .char("p"):
            guard let corpus = selectedCorpus(state) else { return noCorpus(&state) }
            return [.pull(corpus: corpus)]
        case .char("O"):
            guard let corpus = selectedCorpus(state) else { return noCorpus(&state) }
            return [.offlineSnapshot(corpus: corpus)]
        case .char("r"):
            return [.scanCorpora]
        default:
            return nil
        }
    }

    static func noCorpus(_ state: inout StudioState) -> [StudioJob] {
        state.status.error = RaoLMFailure("no corpus selected", hint: "generate one, or tab to the list and press ⏎", code: 66)
        return []
    }

    static func submit(_ state: inout StudioState) -> [StudioJob] {
        let form = state.corpus.form
        let slug = form["slug"].trimmingCharacters(in: .whitespaces)
        guard DocumentID.isValidHandle(slug) || slug.range(of: "^[a-z0-9]+$", options: .regularExpression) != nil,
              let documents = form.int("documents"), documents > 0, let seed = form.uint("seed"),
              let maxChars = form.int("maxChars"), maxChars >= 120
        else {
            state.status.error = RaoLMFailure("check the form: lowercase slug, documents > 0, a numeric seed, max chars ≥ 120", code: 64)
            return []
        }
        return [.corpusGenerate(CorpusForm(slug: slug, documents: documents, seed: seed, maxChars: maxChars, force: form.bool("force")))]
    }

    static func hints(_ state: StudioState) -> [KeyHint] {
        if state.corpus.form.editing { return [KeyHint("⏎", "commit"), KeyHint("esc", "done"), KeyHint("tab", "next field")] }
        return [KeyHint("tab", "pane"), KeyHint("⏎", state.corpus.pane == .form ? "edit/run" : "open"), KeyHint("i", "ingest"),
                KeyHint("p", "pull"), KeyHint("O", "offline snapshot")]
    }
}
