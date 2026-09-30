//
//  LedgerReader.swift
//  RaoLMWorkflows
//
//  WHAT: Reads a run's entropy ledger back: per-epoch summaries, one document's partitions
//        across epochs, or one fact's answer-token losses across eval epochs.
//

import Foundation
import RaoLM

public enum LedgerReader {
    public static func epochs(runDirectory: URL) throws -> [EpochRow] {
        try JSONCoding.readLines(EpochRow.self, from: LedgerFiles.epochs(RunLayout.ledger(runDirectory)))
    }

    public static func partitions(
        runDirectory: URL, manifest: RunManifest, documentID: String, partitionIndex: Int? = nil
    ) throws -> [PartitionEpochRow] {
        let ledger = RunLayout.ledger(runDirectory)
        var rows: [PartitionEpochRow] = []
        for epoch in manifest.epochs.map(\.epoch) {
            let url = LedgerFiles.partitions(ledger, epoch: epoch)
            guard FileManager.default.fileExists(atPath: url.path) else { continue }
            rows += try JSONCoding.readLines(PartitionEpochRow.self, from: url).filter {
                $0.documentID == documentID && (partitionIndex == nil || $0.partitionIndex == partitionIndex)
            }
        }
        return rows
    }

    /// Matches the fact id exactly or by suffix.
    public static func facts(runDirectory: URL, manifest: RunManifest, factID: String) throws -> [FactEpochRow] {
        let ledger = RunLayout.ledger(runDirectory)
        var rows: [FactEpochRow] = []
        for epoch in manifest.epochs.map(\.epoch) {
            let url = LedgerFiles.facts(ledger, epoch: epoch)
            guard FileManager.default.fileExists(atPath: url.path) else { continue }
            rows += try JSONCoding.readLines(FactEpochRow.self, from: url).filter { $0.factID == factID || $0.factID.hasSuffix(factID) }
        }
        return rows
    }
}
