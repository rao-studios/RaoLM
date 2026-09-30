import Foundation
import Testing

@testable import RaoLMTerminal

@MainActor
@Suite("App loop", .serialized)
struct AppTests {
    struct Model {
        var messages = 0
        var keys: [KeyEvent] = []
        var closed = false
    }

    enum Message: Sendable { case bump, quit(Int32) }

    func makeApp(_ display: MemoryDisplay, frame: Duration = .milliseconds(33)) -> App<Model, Message> {
        App(
            display: display, initial: Model(),
            configuration: .init(frameInterval: frame, tickInterval: .seconds(60), minimumSize: Size(width: 20, height: 5), showStats: false),
            update: { model, event in
                switch event {
                case .message(.bump): model.messages += 1
                case .message(.quit(let code)): return .quit(code: code)
                case .key(let key):
                    model.keys.append(key)
                    if key == KeyEvent(.char("q")) { return .quit }
                case .inputClosed: model.closed = true
                default: break
                }
                return .none
            },
            render: { model, frame in
                frame.canvas.put("messages \(model.messages)", x: 0, y: 0, style: .plain)
            })
    }

    @Test("a burst of messages coalesces into few frames, and quit returns its code and restores once")
    func coalescing() async throws {
        let display = MemoryDisplay(size: Size(width: 40, height: 10))
        let app = makeApp(display, frame: .milliseconds(200))
        let mailbox = app.mailbox
        Task.detached {
            for _ in 0..<50 { mailbox.post(.bump) }
            try? await Task.sleep(for: .milliseconds(500))
            mailbox.post(.quit(7))
        }
        let code = try await app.run()
        #expect(code == 7)
        #expect(app.state.messages == 50)
        #expect(display.flushCount <= 4)
        #expect(display.enterCount == 1)
        #expect(display.restoreCount == 1)
        // Frames after the first redraw only the cells that changed: "0" then "50".
        let output = String(decoding: display.output, as: UTF8.self)
        #expect(output.contains("messages 0"))
        #expect(output.hasSuffix("\u{1B}[?2026l"))
        #expect(output.contains("50"))
    }

    @Test("keys reach update; closing input ends the loop")
    func keysAndClose() async throws {
        let display = MemoryDisplay(size: Size(width: 40, height: 10))
        let app = makeApp(display)
        display.send(key: KeyEvent(.char("x")))
        display.closeInput()
        let code = try await app.run()
        #expect(code == 0)
        #expect(app.state.keys == [KeyEvent(.char("x"))])
        #expect(app.state.closed)
    }

    @Test("a second interrupt returns 130 even when update ignores it")
    func doubleInterrupt() async throws {
        let display = MemoryDisplay(size: Size(width: 40, height: 10))
        let app = makeApp(display)
        display.send(.signal(.interrupt))
        display.send(key: KeyEvent(.ctrl("c")))
        let code = try await app.run()
        #expect(code == 130)
        #expect(display.restoreCount == 1)
    }

    @Test("a screen below the minimum size shows the notice instead of rendering")
    func tooSmall() async throws {
        let display = MemoryDisplay(size: Size(width: 18, height: 4))
        let app = makeApp(display)
        display.send(key: KeyEvent(.char("q")))
        _ = try await app.run()
        let output = String(decoding: display.output, as: UTF8.self)
        #expect(output.contains("needs 20×5"))
        #expect(!output.contains("messages"))
    }
}
