import Foundation
import Testing

@testable import RaoLM
@testable import RaoLMWorkflows

@Suite("Workflows")
struct WorkflowsTests {
    @Test("training settings default to the CLI's hyperparameters, and early stop 0 disables it")
    func settings() {
        var settings = TrainingSettings()
        let hyper = settings.hyperparameters()
        #expect(hyper.batchSize == 4)
        #expect(hyper.seqLen == 512)
        #expect(hyper.epochs == 60)
        #expect(hyper.peakLR == 2e-3)
        #expect(hyper.evalEvery == 10)
        #expect(hyper.earlyStopMemorised == 0.98)
        settings.earlyStop = 0
        #expect(settings.hyperparameters().earlyStopMemorised == nil)
        settings.preset = "enormous"
        #expect(throws: RaoLMConfigError.self) { try settings.modelConfig() }
    }

    @Test("the console printer keeps `raolm train`'s exact lines and throttles steps after the third")
    func printer() {
        var clock = Date(timeIntervalSince1970: 0)
        var lines: [String] = []
        var hyper = TrainingSettings().hyperparameters()
        hyper.epochs = 2
        let printer = TrainingConsolePrinter(hyper: hyper, out: { lines.append($0) }, now: { clock })
        printer.handle(.started(totalSteps: 8, stepsPerEpoch: [4, 4], tokensPerStep: 2048))
        #expect(lines.last == "training: 8 steps (4/epoch × 2 epochs), 2,048 tokens/step")
        func step(_ global: Int) -> StepRow {
            StepRow(epoch: 1, step: global, globalStep: global, lr: 2e-4, loss: 3.25, entropy: EntropySummary(values: [2.5]),
                    lossMinusEntropy: 0.75, gradNorm: 1.5, clipped: true, tokens: 2048, maskedTokens: 2000,
                    tokensPerSecond: 41200, wallClockSeconds: Double(global))
        }
        for global in 0..<5 {
            clock = clock.addingTimeInterval(1)
            printer.handle(.step(step(global)))
        }
        #expect(lines.count == 4)
        #expect(lines[1] == "  epoch 1 step 1/4  loss 3.250  entropy 2.500  lr 2.00e-04  grad 1.50  41200 tok/s")
        clock = clock.addingTimeInterval(6)
        printer.handle(.step(step(5)))
        #expect(lines.count == 5)
        let record = EpochRecord(epoch: 1, steps: 4, trainLoss: 3, trainEntropy: 2.5, evalLoss: 1.25, evalEntropy: 1,
                                 evalMemorisedFraction: 0.5, wallClockSeconds: 12)
        #expect(TrainingConsolePrinter.epochLine(record) == "epoch 1  train loss 3.000  entropy 2.500  eval loss 1.250  memorised 50.0%  (12.0 s)")
    }

    @Test("failures map to the CLI's exit codes")
    func failureCodes() {
        #expect(FailureMapping.describe(ThreadCorpusError.indexTimeout(missing: ["d"])).code == 75)
        #expect(FailureMapping.describe(RunManifestError.notARun("/x")).code == 66)
        let corpus = FailureMapping.describe(CorpusStoreError.notACorpus("/x"))
        #expect(corpus.code == 66)
        #expect(corpus.hint == "generate one with: raolm corpus generate")
        #expect(FailureMapping.describe(CancellationError()).code == 130)
        #expect(FailureMapping.describe(RaoLMFailure("x", code: 3)).code == 3)
        struct Other: Error {}
        #expect(FailureMapping.describe(Other()).code == 70)
    }

    @Test("corpus slices parse and reject malformed specs with a usage code")
    func slices() throws {
        let slice = try CorpusSlice.parse("raolm-veldmar-abc:2:10:5")
        #expect(slice == CorpusSlice(documentID: "raolm-veldmar-abc", partitionIndex: 2, offset: 10, length: 5))
        #expect(slice.spec == "raolm-veldmar-abc:2:10:5")
        for bad in ["a:b:c:d", "doc:1:2", "doc:1:2:0", ""] {
            #expect(throws: RaoLMFailure.self) { try CorpusSlice.parse(bad) }
            #expect((try? CorpusSlice.parse(bad)) == nil)
        }
    }

    @Test("tables render with Format.table's gutters, as the CLI prints them")
    func tables() {
        let record = EpochRecord(epoch: 2, steps: 10, trainLoss: 1.5, trainEntropy: 1.25, evalLoss: nil, evalEntropy: nil,
                                 evalMemorisedFraction: nil, checkpointSHA256: "0123456789abcdef", wallClockSeconds: 3)
        let table = LedgerTables.epochs([EpochRow(record: record)])
        #expect(table.headers.first == "epoch")
        #expect(table.rows == [["2", "10", "1.500", "1.250", "—", "—", "—", "—", "0123456789ab…", ""]])
        let text = Format.table(TextTable(headers: ["a", "bb"], rows: [["1", "2"]]))
        #expect(text == "  a  bb\n  ─  ──\n  1  2")
    }
}
