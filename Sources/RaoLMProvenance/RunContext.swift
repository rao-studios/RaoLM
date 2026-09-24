//
//  RunContext.swift
//  RaoLMProvenance
//
//  WHAT: Loads everything a trained run needs to generate with citations — manifest,
//        tokenizer, checkpoint, provenance index — and refuses combinations whose hashes
//        disagree.
//

import Foundation
import MLX
import RaoLMCore
import RaoLMModel
import RaoLMTraining

public final class RunContext {
    public let runDirectory: URL
    public let manifest: RunManifest
    public let epoch: Int
    public let tokenizer: RaoTokenizer
    public let model: RaoTransformer
    public let index: ProvenanceIndex
    public let checkpointSHA256: String
    public let manifestRef: ManifestRef

    private init(
        runDirectory: URL, manifest: RunManifest, epoch: Int, tokenizer: RaoTokenizer, model: RaoTransformer,
        index: ProvenanceIndex, checkpointSHA256: String, manifestRef: ManifestRef
    ) {
        self.runDirectory = runDirectory
        self.manifest = manifest
        self.epoch = epoch
        self.tokenizer = tokenizer
        self.model = model
        self.index = index
        self.checkpointSHA256 = checkpointSHA256
        self.manifestRef = manifestRef
    }

    public static func load(
        runDirectory: URL, epoch requested: Int? = nil, allowWeakIndex: Bool = false, tokenizerDirectory: URL? = nil
    ) async throws -> RunContext {
        let manifest = try RunManifest.load(runDirectory)
        guard let latest = manifest.latestIndexedEpoch else { throw RunManifestError.noIndex(runDirectory.path) }
        let epoch = requested ?? latest
        guard manifest.indexedEpochs.contains(epoch) else {
            throw RunManifestError.epochNotIndexed(epoch, available: manifest.indexedEpochs.sorted())
        }
        let tokenizer = try await RaoTokenizer.load(directory: tokenizerDirectory)
        guard tokenizer.tokenizerSHA256 == manifest.tokenizer.tokenizerSHA256 else {
            throw ProvenanceError.tokenizerMismatch(expected: manifest.tokenizer.tokenizerSHA256, found: tokenizer.tokenizerSHA256)
        }
        let checkpointDirectory = RunLayout.checkpoint(runDirectory, epoch: epoch)
        guard FileManager.default.fileExists(atPath: checkpointDirectory.appendingPathComponent(Checkpoint.weightsFile).path) else {
            throw ProvenanceError.missingCheckpoint(checkpointDirectory.path)
        }
        let checkpointSHA = try Checkpoint.weightsSHA256(checkpointDirectory)
        let index = try ProvenanceIndex(directory: RunLayout.provenance(runDirectory, epoch: epoch))
        guard index.info.checkpointSHA256 == checkpointSHA else {
            throw ProvenanceError.checkpointMismatch(index: index.info.checkpointSHA256, checkpoint: checkpointSHA)
        }
        if !allowWeakIndex, index.info.evalMemorisedFraction < 0.5 {
            throw ProvenanceError.weakIndex(epoch: epoch, memorised: index.info.evalMemorisedFraction)
        }
        let model = try Checkpoint.load(from: checkpointDirectory, tapLayer: index.info.tapLayer)
        let record = manifest.epochRecord(epoch)
        let ref = ManifestRef(
            runID: manifest.runID, epoch: epoch, checkpointSHA256: checkpointSHA, indexSHA256: index.sha256,
            corpusHash: manifest.corpus.corpusHash, tokenizerSHA256: tokenizer.tokenizerSHA256,
            ledgerSHA256: record?.partitionLedgerSHA256, threadID: manifest.corpus.threadID)
        return RunContext(
            runDirectory: runDirectory, manifest: manifest, epoch: epoch, tokenizer: tokenizer, model: model,
            index: index, checkpointSHA256: checkpointSHA, manifestRef: ref)
    }

    public func generator() -> CitedGenerator {
        CitedGenerator(model: model, tokenizer: tokenizer, index: index, manifestRef: manifestRef)
    }

    public func defaultParameters() -> GenerationParameters {
        GenerationParameters(
            lambda: 0.5, tau: index.info.defaultTau, k: index.info.defaultK, tapLayer: index.info.tapLayer,
            alpha: index.info.alpha)
    }

    public func snapshot() throws -> CorpusSnapshot {
        try CorpusSnapshot.load(from: URL(fileURLWithPath: manifest.corpus.snapshotPath))
    }

    public func tokenizedCorpus() throws -> TokenizedCorpus {
        TokenizedCorpus(snapshot: try snapshot(), tokenizer: tokenizer, excluding: Set(manifest.excludedDocumentIDs))
    }
}
