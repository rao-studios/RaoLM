//
//  DoctorScreen.swift
//  RaoLMStudio
//
//  WHAT: 2 Tests: the doctor's checks, and RaoLM's test suites run through scripts/test.sh
//        with their output streaming in.
//

import Foundation
import RaoLM
import RaoLMTerminal
import RaoLMWorkflows

enum DoctorScreen: StudioScreen {
    static func styled(_ line: String) -> Text {
        let lower = line.lowercased()
        if line.contains("✘") || lower.contains(" failed") || lower.contains("error:") { return Text(line, style: Style(foreground: .indexed(167))) }
        if line.contains("✔") || lower.contains(" passed") { return Text(line, style: Style(foreground: .indexed(36))) }
        if line.contains("↳") || line.contains("◇") { return Text(line, style: Style(foreground: .indexed(245))) }
        return Text(line)
    }

    static func logRows(_ state: StudioState) -> Int { max(3, state.size.height - 8 - state.doctor.checks.count - 6) }

    static func render(_ state: StudioState, _ frame: inout Frame, _ rect: Rect) {
        let palette = frame.palette
        let glyphs = frame.glyphs
        let checksHeight = min(rect.height / 2, max(4, state.doctor.checks.count + 2))
        let (checksRect, testsRect) = rect.top(checksHeight)
        let ok = state.doctor.checks.filter(\.ok).count
        let footer = state.doctor.checks.isEmpty ? nil : Text("\(ok)/\(state.doctor.checks.count) ok", style: ok == state.doctor.checks.count ? palette.green : palette.orange)
        let checksInner = Theme.panel("Doctor", footer: footer, &frame, checksRect)
        if state.doctor.checks.isEmpty {
            Theme.empty(state.status.jobs[.doctor] != nil ? "running checks…" : "press r to run the checks", &frame, checksInner)
        } else {
            for (row, check) in state.doctor.checks.prefix(checksInner.height).enumerated() {
                var line = Text(check.ok ? glyphs.check : glyphs.cross, style: check.ok ? palette.green : palette.red)
                line.append(" " + check.name.padding(toLength: 17, withPad: " ", startingAt: 0), palette.text)
                line.append(check.detail, check.ok ? palette.dim : palette.orange)
                frame.canvas.put(line.truncated(to: checksInner.width), x: checksInner.minX, y: checksInner.minY + row, clip: checksInner)
            }
        }

        var footerText: Text?
        if state.doctor.running {
            footerText = Text("\(StudioApp.spinner(state, glyphs)) running", style: palette.orange)
        } else if let code = state.doctor.exitCode {
            footerText = code == 0 ? Text("\(glyphs.check) exit 0", style: palette.green) : Text("\(glyphs.cross) exit \(code)", style: palette.red)
        }
        let inner = Theme.panel("Test suites · scripts/test.sh", focused: state.doctor.editingFilter, footer: footerText, &frame, testsRect)
        let (controls, logRect) = inner.top(2)
        frame.canvas.put("filter", x: controls.minX, y: controls.minY, style: palette.dim, clip: controls)
        let field = Rect(x: controls.minX + 8, y: controls.minY, width: 30, height: 1)
        TextField(state: state.doctor.filter, placeholder: "all suites (/ sets one)",
                  focused: state.doctor.editingFilter, style: palette.title, placeholderStyle: palette.muted,
                  cursorStyle: palette.cursor).render(in: field, on: &frame.canvas)
        var flags = Text("   MLX suites ", style: palette.dim)
        flags.append(state.doctor.mlx ? "\(glyphs.live) on" : "\(glyphs.idle) off", state.doctor.mlx ? palette.green : palette.dim)
        flags.append("   Thread suite ", palette.dim)
        flags.append(state.doctor.thread ? "\(glyphs.live) on" : "\(glyphs.idle) off", state.doctor.thread ? palette.green : palette.dim)
        frame.canvas.put(flags, x: field.maxX + 1, y: controls.minY, clip: controls)
        if state.fixtures {
            Theme.empty("Test suites need the repository checkout; they are not available in fixture mode.", &frame, logRect)
        } else if state.doctor.log.lines.isEmpty {
            Theme.empty("t runs the suites (the MLX ones build the Metal library first and take a few minutes). Output streams here.", &frame, logRect)
        } else {
            LogPane(state: state.doctor.log, style: palette.text).render(in: logRect, on: &frame.canvas)
        }
    }

    static func handle(_ key: KeyEvent, _ state: inout StudioState) -> [StudioJob]? {
        if state.doctor.editingFilter {
            switch key.key {
            case .enter, .escape: state.doctor.editingFilter = false
            default: state.doctor.filter.handle(key)
            }
            return []
        }
        let visible = logRows(state)
        switch key.key {
        case .char("r"): return [.doctor]
        case .char("/"): state.doctor.editingFilter = true
        case .char("m"): state.doctor.mlx.toggle()
        case .char("T"): state.doctor.thread.toggle()
        case .char("t"), .enter:
            guard !state.doctor.running else {
                state.status.message = "the suites are already running"
                return []
            }
            state.doctor.running = true
            state.doctor.exitCode = nil
            state.doctor.log.clear()
            let filter = state.doctor.filter.text.trimmingCharacters(in: .whitespaces)
            return [.runTests(filter: filter.isEmpty ? nil : filter, mlx: state.doctor.mlx, thread: state.doctor.thread)]
        case .char("c"):
            return [.cancelTests]
        case .up, .char("k"): state.doctor.log.scroll(by: 1, visible: visible)
        case .down, .char("j"): state.doctor.log.scroll(by: -1, visible: visible)
        case .pageUp: state.doctor.log.scroll(by: visible, visible: visible)
        case .pageDown: state.doctor.log.scroll(by: -visible, visible: visible)
        case .end: state.doctor.log.toEnd()
        default: return nil
        }
        return []
    }

    static func hints(_ state: StudioState) -> [KeyHint] {
        state.doctor.editingFilter ? [KeyHint("⏎", "set filter"), KeyHint("esc", "done")]
            : [KeyHint("t", "run suites"), KeyHint("/", "filter"), KeyHint("m", "MLX"), KeyHint("T", "Thread"),
               KeyHint("c", "cancel"), KeyHint("r", "re-check")]
    }
}
