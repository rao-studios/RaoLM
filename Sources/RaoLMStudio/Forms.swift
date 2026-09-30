//
//  Forms.swift
//  RaoLMStudio
//
//  WHAT: A keyboard form: text, integer and decimal fields, choices, toggles and an action
//        row. ↑↓ (j k) move, Enter edits a field (Enter or Esc commits, Tab commits and moves on),
//        toggles and choices change with Enter, Space or ←→, and Enter on the action row submits.
//        Tab outside a field is left to the screen, which uses it to switch panes.
//  PIN:  Fields keep text; screens parse on submit and report a bad value in the status bar,
//        so a half-typed number never throws away the rest of the form.
//

import Foundation
import RaoLM
import RaoLMTerminal
import RaoLMWorkflows

public struct FormField: Sendable {
    public enum Kind: Sendable, Equatable {
        case text, integer, number
        case choice([String])
        case toggle
        case action
    }

    public var key: String
    public var label: String
    public var kind: Kind
    public var input: TextFieldState
    public var help: String

    public init(_ key: String, _ label: String, _ kind: Kind, _ value: String = "", help: String = "") {
        self.key = key
        self.label = label
        self.kind = kind
        self.input = TextFieldState(value)
        self.help = help
    }

    public var value: String { input.text }
}

public struct FormState: Sendable {
    public enum Outcome: Equatable { case unhandled, changed, submit(String) }

    public var fields: [FormField]
    public var focus = 0
    public var editing = false

    public init(_ fields: [FormField]) { self.fields = fields }

    public subscript(key: String) -> String {
        get { fields.first { $0.key == key }?.value ?? "" }
        set {
            guard let index = fields.firstIndex(where: { $0.key == key }) else { return }
            fields[index].input.set(newValue)
        }
    }

    public func int(_ key: String) -> Int? { Int(self[key].trimmingCharacters(in: .whitespaces)) }
    public func uint(_ key: String) -> UInt64? { UInt64(self[key].trimmingCharacters(in: .whitespaces)) }
    public func float(_ key: String) -> Float? { Float(self[key].trimmingCharacters(in: .whitespaces)) }
    public func bool(_ key: String) -> Bool { self[key] == "on" }

    /// The option index of a choice field.
    public func choiceIndex(_ key: String) -> Int? {
        guard let field = fields.first(where: { $0.key == key }), case .choice(let options) = field.kind else { return nil }
        return options.firstIndex(of: field.value)
    }

    public mutating func setChoices(_ key: String, _ options: [String], keepIndex: Bool = true) {
        guard let index = fields.firstIndex(where: { $0.key == key }) else { return }
        let old = choiceIndex(key)
        fields[index].kind = .choice(options)
        let pick = keepIndex ? min(old ?? 0, max(0, options.count - 1)) : 0
        fields[index].input.set(options.isEmpty ? "" : options[pick])
    }

    public var focusedField: FormField? { focus < fields.count ? fields[focus] : nil }

    public mutating func handle(_ key: KeyEvent) -> Outcome {
        guard !fields.isEmpty else { return .unhandled }
        focus = min(max(0, focus), fields.count - 1)
        if editing {
            switch key.key {
            case .enter, .escape:
                editing = false
                return .changed
            case .tab, .down:
                editing = false
                focus = min(fields.count - 1, focus + 1)
                return .changed
            case .backTab, .up:
                editing = false
                focus = max(0, focus - 1)
                return .changed
            default:
                if case .char(let c) = key.key, !accepts(c, kind: fields[focus].kind) { return .changed }
                return fields[focus].input.handle(key) ? .changed : .changed
            }
        }
        switch key.key {
        case .up, .char("k"):
            guard focus > 0 else { return .unhandled }
            focus -= 1
            return .changed
        case .down, .char("j"):
            guard focus < fields.count - 1 else { return .unhandled }
            focus += 1
            return .changed
        case .enter:
            switch fields[focus].kind {
            case .text, .integer, .number:
                editing = true
                return .changed
            case .choice, .toggle:
                cycle(by: 1)
                return .changed
            case .action:
                return .submit(fields[focus].key)
            }
        case .char(" "):
            switch fields[focus].kind {
            case .choice, .toggle:
                cycle(by: 1)
                return .changed
            default:
                return .unhandled
            }
        case .left, .right:
            switch fields[focus].kind {
            case .choice, .toggle:
                cycle(by: key.key == .left ? -1 : 1)
                return .changed
            default:
                return .unhandled
            }
        default:
            return .unhandled
        }
    }

    private func accepts(_ c: Character, kind: FormField.Kind) -> Bool {
        switch kind {
        case .integer: return c.isNumber || c == "-"
        case .number: return c.isNumber || c == "." || c == "-" || c == "e" || c == ","
        default: return true
        }
    }

    private mutating func cycle(by delta: Int) {
        switch fields[focus].kind {
        case .toggle:
            fields[focus].input.set(fields[focus].value == "on" ? "off" : "on")
        case .choice(let options):
            guard !options.isEmpty else { return }
            let current = options.firstIndex(of: fields[focus].value) ?? 0
            fields[focus].input.set(options[(current + delta + options.count) % options.count])
        default:
            break
        }
    }

    /// Label column, value column; the focused row carries the gold marker.
    public func render(in rect: Rect, focused: Bool, frame: inout Frame, labelWidth: Int = 16) {
        let palette = frame.palette
        let glyphs = frame.glyphs
        let visible = rect.height
        let scroll = max(0, min(focus - visible + 1, fields.count - visible))
        for (row, index) in (scroll..<min(fields.count, scroll + visible)).enumerated() {
            let field = fields[index]
            let y = rect.minY + row
            let isFocus = focused && index == focus
            if isFocus { frame.canvas.put(glyphs.select, x: rect.minX, y: y, style: palette.gold, clip: rect) }
            let x = rect.minX + 2
            if case .action = field.kind {
                let label = Text(" \(glyphs.prompt) \(field.label) ", style: isFocus ? palette.selection.bold() : palette.title)
                frame.canvas.put(label, x: x, y: y, clip: rect)
                continue
            }
            frame.canvas.put(field.label, x: x, y: y, style: isFocus ? palette.text : palette.dim, clip: rect)
            let valueRect = Rect(x: x + labelWidth, y: y, width: max(0, rect.maxX - x - labelWidth), height: 1)
            switch field.kind {
            case .toggle:
                let on = field.value == "on"
                frame.canvas.put(on ? "\(glyphs.live) on" : "\(glyphs.idle) off", x: valueRect.minX, y: y,
                                 style: on ? palette.green : palette.dim, clip: valueRect)
            case .choice(let options):
                let arrows = options.count > 1 ? " ‹›" : ""
                let text = Text(field.value.isEmpty ? "—" : field.value, style: palette.text) + Text(arrows, style: palette.muted)
                frame.canvas.put(text.truncated(to: valueRect.width), x: valueRect.minX, y: y, clip: valueRect)
            default:
                if isFocus && editing {
                    frame.cursor = nil
                    TextField(state: field.input, focused: true, style: palette.title, placeholderStyle: palette.muted,
                              cursorStyle: palette.cursor).render(in: valueRect, on: &frame.canvas)
                } else {
                    frame.canvas.put(field.value.isEmpty ? "—" : field.value, x: valueRect.minX, y: y,
                                     style: isFocus ? palette.title : palette.text, clip: valueRect)
                }
            }
        }
    }

    // MARK: - The studio's forms

    public static func corpus() -> FormState {
        FormState([
            FormField("slug", "slug", .text, "veldmar", help: "document ids are raolm-<slug>-<hash>"),
            FormField("documents", "documents", .integer, "200"),
            FormField("seed", "seed", .integer, "42"),
            FormField("maxChars", "max chars", .integer, "600", help: "per partition"),
            FormField("force", "overwrite", .toggle, "off"),
            FormField("generate", "Generate corpus", .action),
        ])
    }

    public static func training(snapshots: [String]) -> FormState {
        let defaults = TrainingSettings()
        return FormState([
            FormField("snapshot", "snapshot", .choice(snapshots), snapshots.first ?? ""),
            FormField("preset", "preset", .choice(["tiny", "small", "smollm2-135m"]), defaults.preset),
            FormField("epochs", "epochs", .integer, String(defaults.epochs)),
            FormField("batchSize", "batch size", .integer, String(defaults.batchSize)),
            FormField("seqLen", "seq len", .integer, String(defaults.seqLen)),
            FormField("lr", "peak lr", .number, "2e-3"),
            FormField("evalEvery", "eval every", .integer, String(defaults.evalEvery)),
            FormField("indexEvery", "index every", .integer, String(defaults.indexEvery)),
            FormField("tapLayer", "tap layer", .text, "", help: "empty: layers/2"),
            FormField("alpha", "alpha", .number, "0.5"),
            FormField("seed", "seed", .integer, String(defaults.seed)),
            FormField("earlyStop", "early stop", .number, "0.98"),
            FormField("exclude", "hold out docs", .integer, "0"),
            FormField("keep", "keep ckpts", .integer, String(defaults.keepCheckpoints)),
            FormField("train", "Start training", .action),
        ])
    }

    public static func eval() -> FormState {
        FormState([
            FormField("facts", "facts sample", .integer, "50"),
            FormField("lambdas", "λ values", .text, "0,0.25,0.5,0.75"),
            FormField("primary", "primary λ", .number, "0.5"),
            FormField("seed", "seed", .integer, "7"),
            FormField("controls", "controls", .toggle, "on"),
            FormField("offline", "offline verify", .toggle, "on"),
            FormField("grounding", "grounding", .toggle, "on"),
            FormField("run", "Run evaluation", .action),
        ])
    }

    public static func params(_ o: GenerationOverrides) -> FormState {
        FormState([
            FormField("lambda", "λ (retrieval)", .number, Format.f(o.lambda, 2)),
            FormField("tau", "τ", .number, o.tau.map { Format.f($0, 3) } ?? ""),
            FormField("k", "k neighbours", .integer, o.k.map(String.init) ?? ""),
            FormField("temperature", "temperature", .number, Format.f(o.temperature, 2)),
            FormField("topK", "top-k", .integer, String(o.topK)),
            FormField("maxTokens", "max tokens", .integer, String(o.maxTokens)),
            FormField("seed", "seed", .integer, String(o.seed)),
            FormField("apply", "Apply", .action),
        ])
    }

    public func overrides() throws -> GenerationOverrides {
        var o = GenerationOverrides()
        guard let lambda = float("lambda"), (0...1).contains(lambda) else { throw RaoLMFailure("λ must be between 0 and 1", code: 64) }
        o.lambda = lambda
        o.tau = self["tau"].isEmpty ? nil : float("tau")
        o.k = self["k"].isEmpty ? nil : int("k")
        guard let temperature = float("temperature"), temperature >= 0 else { throw RaoLMFailure("temperature must be ≥ 0", code: 64) }
        o.temperature = temperature
        o.topK = max(0, int("topK") ?? 0)
        guard let maxTokens = int("maxTokens"), maxTokens > 0 else { throw RaoLMFailure("max tokens must be > 0", code: 64) }
        o.maxTokens = maxTokens
        o.seed = uint("seed") ?? 42
        return o
    }
}
