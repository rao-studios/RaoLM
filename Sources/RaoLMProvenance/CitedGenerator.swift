//
//  CitedGenerator.swift
//  RaoLMProvenance
//
//  WHAT: Generation with citations: every step retrieves the corpus positions whose keys are
//        nearest the current one, mixes their next tokens into the model's distribution,
//        chooses a token, and records a TokenTrace (entropies, neighbours, citations).
//  OUT:  A CitedGeneration bound to the run's manifest hashes, with verbatim spans found.
//  PIN:  Prompt positions get teacher-forced traces from the prefill, so a verbatim span may
//        start inside the prompt. Some tokens are uncited by construction — no retrieved
//        context predicted them — and the trace says so. The softmax, the mixture and the
//        chosen token are computed on the CPU (CitationMixer); only the entropies a trace
//        reports come from the GPU (MixtureStats), as they do in a braid.
//

import Foundation
import MLX
import MLXLMCommon
import RaoLMCore
import RaoLMModel

public struct GenerationRequest {
    public var promptTokens: [Int]
    public var promptText: String
    public var promptSource: SourceAddress?
    public var params: GenerationParameters

    public init(promptTokens: [Int], promptText: String, promptSource: SourceAddress? = nil, params: GenerationParameters) {
        self.promptTokens = promptTokens
        self.promptText = promptText
        self.promptSource = promptSource
        self.params = params
    }
}

public final class CitedGenerator {
    public let model: RaoTransformer
    public let tokenizer: RaoTokenizer
    public let index: ProvenanceIndex
    public let manifestRef: ManifestRef

    public init(model: RaoTransformer, tokenizer: RaoTokenizer, index: ProvenanceIndex, manifestRef: ManifestRef) {
        self.model = model
        self.tokenizer = tokenizer
        self.index = index
        self.manifestRef = manifestRef
    }

    /// `onToken` sees each generated token as it is chosen; throwing from it (for example a
    /// `CancellationError`) stops the generation and rethrows.
    public func generate(_ request: GenerationRequest, onToken: ((TokenTrace) throws -> Void)? = nil) throws -> CitedGeneration {
        let params = request.params
        guard !request.promptTokens.isEmpty else { throw ProvenanceError.emptyPrompt }
        let prompt = request.promptTokens
        let eos = tokenizer.eosTokenID
        var rng = SplitMix64(seed: params.seed)

        let cache = model.newCache(parameters: nil)
        let promptArray = MLXArray(prompt.map { Int32($0) }, [1, prompt.count])
        var output = model.forward(promptArray, cache: cache, captureTap: true)
        var keys = ProvenanceKey.make(tap: output.tap!, final: output.final, alpha: params.alpha)
        eval(output.logits, keys)

        var traces: [TokenTrace] = []
        // Teacher-forced traces for prompt tokens 1..<n (position j-1 predicts token j).
        let vocabulary = output.logits.dim(-1)
        let promptLogits = output.logits[0].asType(.float32)
        let flat = promptLogits.asArray(Float.self)
        var mixtures: [MixtureStats.Position] = []
        for j in 1..<prompt.count {
            let (trace, knn) = step(
                logits: Array(flat[((j - 1) * vocabulary)..<(j * vocabulary)]), key: keys[0, j - 1], index: j, forced: prompt[j],
                params: params, rng: &rng)
            traces.append(trace)
            mixtures.append(MixtureStats.Position(weights: [1], knn: knn))
        }
        if prompt.count > 1 {
            Self.fill(&traces, from: 0, with: MixtureStats.entropies(
                heads: [promptLogits[0..<(prompt.count - 1)]], positions: mixtures, lambda: params.lambda))
        }

        var generated: [Int] = []
        var stoppedOnEOS = false
        var position = prompt.count - 1
        while generated.count < params.maxTokens {
            let logits = output.logits[0, position].asType(.float32)
            var (trace, knn) = step(
                logits: logits.asArray(Float.self), key: keys[0, position], index: prompt.count + generated.count,
                forced: nil, params: params, rng: &rng)
            var one = [trace]
            Self.fill(&one, from: 0, with: MixtureStats.entropies(
                heads: [logits.reshaped(1, -1)], positions: [MixtureStats.Position(weights: [1], knn: knn)], lambda: params.lambda))
            trace = one[0]
            if trace.token == eos {
                stoppedOnEOS = true
                break
            }
            traces.append(trace)
            generated.append(trace.token)
            try onToken?(trace)
            guard generated.count < params.maxTokens else { break }
            output = model.forward(MLXArray([Int32(trace.token)], [1, 1]), cache: cache, captureTap: true)
            keys = ProvenanceKey.make(tap: output.tap!, final: output.final, alpha: params.alpha)
            eval(output.logits, keys)
            position = 0
        }

        let threadID = manifestRef.threadID
        let spans = CitationSpans.annotate(
            traces: &traces, partitions: index.partitionsByRow, sharedNgrams: index.sharedNgrams, threadID: threadID,
            settings: CitationSpans.Settings(rankThreshold: params.rankThreshold, minSpanLength: params.minSpanLength))
        var rows = Set<Int>()
        for trace in traces {
            for neighbour in trace.neighbours {
                rows.insert(neighbour.cited.row)
                rows.insert(neighbour.key.row)
            }
        }
        let partitions = rows.sorted().compactMap { index.partitionsByRow[$0] }
        let summary = CitationSpans.summary(traces: traces, spans: spans)
        let generationID = Self.generationID(manifest: manifestRef, prompt: prompt, params: params)
        return CitedGeneration(
            generationID: generationID, manifest: manifestRef,
            prompt: GenerationPrompt(text: request.promptText, tokens: prompt, source: request.promptSource),
            params: params, tokens: generated, text: tokenizer.decode(generated), stoppedOnEOS: stoppedOnEOS,
            partitions: partitions, traces: traces, spans: spans, summary: summary)
    }

    /// The entropies MixtureStats computed for `traces[start...]`, one position each.
    static func fill(_ traces: inout [TokenTrace], from start: Int, with stats: MixtureStats.Entropies) {
        for i in stats.lm.indices where traces.indices.contains(start + i) {
            traces[start + i].lmEntropy = stats.lm[i]
            traces[start + i].mixedEntropy = stats.mixed[i]
        }
    }

    /// One position: retrieve, mix, choose (or take the forced prompt token), trace. The trace's
    /// head and mixture entropies are left for MixtureStats; the retrieval distribution it needs
    /// is returned beside it.
    private func step(
        logits logitValues: [Float], key: MLXArray, index position: Int, forced: Int?, params: GenerationParameters,
        rng: inout SplitMix64
    ) -> (TokenTrace, [Int: Float]) {
        let pLM = CitationMixer.softmax(logitValues)
        let hits = index.query(key, k: params.k)
        var neighbours = CitationMixer.neighbours(hits: hits, index: index, tau: params.tau)
        let knn = CitationMixer.knnDistribution(neighbours)
        let mixed = CitationMixer.mix(pLM: pLM, knn: knn, lambda: params.lambda)

        let token: Int
        if let forced {
            token = forced
        } else if params.temperature <= 0 {
            token = CitationMixer.choose(mixed) { CitationMixer.citedAge($0, neighbours: neighbours) { index.partitionsByRow[$0] } }
        } else {
            let tempered = CitationMixer.mix(
                pLM: CitationMixer.softmax(logitValues, temperature: params.temperature), knn: knn, lambda: params.lambda)
            token = CitationMixer.sample(tempered, topK: params.topK, rng: &rng)
        }
        for i in neighbours.indices { neighbours[i].matches = neighbours[i].value == token }

        let trace = TokenTrace(
            index: position, token: token, text: tokenizer.tokenText(token), isPrompt: forced != nil,
            lmEntropy: 0, knnEntropy: CitationMath.entropy(knn.values),
            mixedEntropy: 0, sourceEntropy: CitationMixer.sourceEntropy(neighbours),
            lmProb: token < pLM.count ? pLM[token] : 0, agreement: knn[token] ?? 0,
            mixedProb: token < mixed.count ? mixed[token] : 0, lambda: params.lambda, neighbours: neighbours)
        return (trace, knn)
    }

    static func generationID(manifest: ManifestRef, prompt: [Int], params: GenerationParameters) -> String {
        let encoder = JSONCoding.lineEncoder()
        let paramsData = (try? encoder.encode(params)) ?? Data()
        let material = "\(manifest.runID)|\(manifest.epoch)|\(manifest.checkpointSHA256)|\(prompt)|" + String(decoding: paramsData, as: UTF8.self)
        return "gen-" + String(ContentHash.sha256Hex(material).prefix(16))
    }
}
