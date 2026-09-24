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
//        context predicted them — and the trace says so.
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

    public func generate(_ request: GenerationRequest, onToken: ((TokenTrace) -> Void)? = nil) throws -> CitedGeneration {
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
        for j in 1..<prompt.count {
            let trace = step(
                logits: output.logits[0, j - 1], key: keys[0, j - 1], index: j, forced: prompt[j],
                params: params, rng: &rng)
            traces.append(trace)
        }

        var generated: [Int] = []
        var stoppedOnEOS = false
        var position = prompt.count - 1
        while generated.count < params.maxTokens {
            let trace = step(
                logits: output.logits[0, position], key: keys[0, position], index: prompt.count + generated.count,
                forced: nil, params: params, rng: &rng)
            if trace.token == eos {
                stoppedOnEOS = true
                break
            }
            traces.append(trace)
            generated.append(trace.token)
            onToken?(trace)
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

    /// One position: retrieve, mix, choose (or take the forced prompt token), trace.
    private func step(
        logits: MLXArray, key: MLXArray, index position: Int, forced: Int?, params: GenerationParameters,
        rng: inout SplitMix64
    ) -> TokenTrace {
        let logitValues = logits.asType(.float32).asArray(Float.self)
        let pLM = CitationMixer.softmax(logitValues)
        let hits = index.query(key, k: params.k)
        var neighbours = CitationMixer.neighbours(hits: hits, index: index, tau: params.tau)
        let knn = CitationMixer.knnDistribution(neighbours)
        let mixed = CitationMixer.mix(pLM: pLM, knn: knn, lambda: params.lambda)

        let token: Int
        if let forced {
            token = forced
        } else if params.temperature <= 0 {
            token = CitationMixer.argmax(mixed)
        } else {
            let tempered = CitationMixer.mix(
                pLM: CitationMixer.softmax(logitValues, temperature: params.temperature), knn: knn, lambda: params.lambda)
            token = CitationMixer.sample(tempered, topK: params.topK, rng: &rng)
        }
        for i in neighbours.indices { neighbours[i].matches = neighbours[i].value == token }

        return TokenTrace(
            index: position, token: token, text: tokenizer.tokenText(token), isPrompt: forced != nil,
            lmEntropy: CitationMath.entropy(pLM), knnEntropy: CitationMath.entropy(knn.values),
            mixedEntropy: CitationMath.entropy(mixed), sourceEntropy: CitationMixer.sourceEntropy(neighbours),
            lmProb: token < pLM.count ? pLM[token] : 0, agreement: knn[token] ?? 0,
            mixedProb: token < mixed.count ? mixed[token] : 0, lambda: params.lambda, neighbours: neighbours)
    }

    static func generationID(manifest: ManifestRef, prompt: [Int], params: GenerationParameters) -> String {
        let encoder = JSONCoding.lineEncoder()
        let paramsData = (try? encoder.encode(params)) ?? Data()
        let material = "\(manifest.runID)|\(manifest.epoch)|\(manifest.checkpointSHA256)|\(prompt)|" + String(decoding: paramsData, as: UTF8.self)
        return "gen-" + String(ContentHash.sha256Hex(material).prefix(16))
    }
}
