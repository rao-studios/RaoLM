import Foundation
import Testing

@testable import RaoLMBraid
@testable import RaoLMCore

/// MLX needs mlx.metallib inside the test bundle: scripts/test.sh puts it there.
var mlxTests: Bool {
    let env = ProcessInfo.processInfo.environment
    return env["RAOLM_MLX_TESTS"] == "1" || env["FRIGATE_MLX_TESTS"] == "1"
}

@Suite("Vocabulary choice")
struct VocabularyChoiceTests {
    func row(_ name: String, to epochs: Int?, exact: Float) -> VocabularyBenchRow {
        VocabularyBenchRow(
            name: name, vocabulary: "v", trainableParameters: 1, epochsToTarget: epochs, secondsToTarget: epochs.map(Double.init),
            epochs: epochs ?? 120, seconds: 1, evalLoss: 0.05, memorised: epochs == nil ? 0.6 : 0.98, exact: exact,
            exactEvidenceOnly: exact, citationAt1: 0.9, runDirectory: "")
    }

    @Test("seeded wins inside twice the baseline's epochs and five points of its answers")
    func seeded() {
        let decision = VocabularyBench.decide([
            row("baseline", to: 40, exact: 0.9), row("seeded ×1.0", to: nil, exact: 0.2),
            row("seeded ×1.5", to: 72, exact: 0.86), row("seeded ×2.0", to: 64, exact: 0.88),
            row("commons, warm blocks", to: 20, exact: 0.95),
        ])
        #expect(decision.choice == .seeded)
        #expect(decision.headScale == 2)
    }

    @Test("a seeded arm that is too slow or answers too little hands over to the commons")
    func commons() {
        let slow = VocabularyBench.decide([
            row("baseline", to: 40, exact: 0.9), row("seeded ×1.5", to: 84, exact: 0.9),
            row("commons, fresh blocks", to: 60, exact: 0.88), row("commons, warm blocks", to: 36, exact: 0.87),
        ])
        #expect(slow.choice == .commons)
        #expect(slow.reason.contains("commons, warm blocks"))
        let weak = VocabularyBench.decide([
            row("baseline", to: 40, exact: 0.9), row("seeded ×1.5", to: 44, exact: 0.8),
            row("commons, fresh blocks", to: 60, exact: 0.88),
        ])
        #expect(weak.choice == .commons)
    }

    @Test("when no blocks-only arm passes, each Thread keeps its own vocabulary")
    func fallback() {
        let decision = VocabularyBench.decide([
            row("baseline", to: 40, exact: 0.9), row("seeded ×1.5", to: nil, exact: 0.5),
            row("commons, warm blocks", to: 100, exact: 0.9),
        ])
        #expect(decision.choice == .perThread)
        #expect(decision.headScale == nil)
    }
}
