import Foundation
import Testing

@testable import RaoLMCore

@Suite("SyntheticCorpus")
struct SyntheticCorpusTests {
    @Test("same seed gives byte-identical manifests; a different seed does not")
    func determinism() throws {
        let a = try SyntheticCorpus.generate(seed: 7, documentCount: 30)
        let b = try SyntheticCorpus.generate(seed: 7, documentCount: 30)
        let c = try SyntheticCorpus.generate(seed: 8, documentCount: 30)
        let encoder = JSONCoding.prettyEncoder()
        #expect(try encoder.encode(a.manifest) == encoder.encode(b.manifest))
        #expect(a.documents == b.documents)
        #expect(a.manifest.corpusHash != c.manifest.corpusHash)
    }

    @Test("structure: 2–4 partitions, lengths in range, paragraph == partition")
    func structure() throws {
        let corpus = try SyntheticCorpus.generate(seed: 42, documentCount: 60)
        #expect(corpus.documents.count == 60)
        #expect(corpus.manifest.documentCount == 60)
        for document in corpus.documents {
            #expect((2...4).contains(document.partitions.count))
            #expect(DocumentID.isValid(document.id))
            #expect(TextChunker.chunk(document.text) == document.partitions.map(\.text))
            for partition in document.partitions {
                #expect(partition.text.utf8.count >= 120 && partition.text.utf8.count <= 600)
                #expect(partition.url == DocumentID.partitionURL(slug: "veldmar", documentID: document.id, index: partition.index))
                #expect(partition.text.allSatisfy { $0.isASCII })
            }
            #expect(!document.facts.isEmpty)
        }
        #expect(Set(corpus.documents.map(\.kind)) == Set(DocumentKind.allCases))
    }

    @Test("every fact is stated exactly once, at its recorded offsets")
    func factsAreUnique() throws {
        let corpus = try SyntheticCorpus.generate(seed: 42, documentCount: 200)
        #expect(FactValidator.validate(documents: corpus.documents, negativeSubjects: [], maxChars: 600, minChars: 120).isEmpty)
        let allText = corpus.documents.map(\.text).joined(separator: "\n\n")
        for fact in corpus.facts.prefix(150) {
            #expect(FactValidator.occurrences(of: fact.prompt, in: allText) == 1, "\(fact.prompt)")
            #expect(!allText.contains(fact.negativePrompt), "\(fact.negativePrompt)")
            for paraphrase in fact.paraphrases {
                #expect(!allText.contains(paraphrase))
            }
            let document = try #require(corpus.documents.first { $0.id == fact.documentID })
            let bytes = Array(document.partitions[fact.partitionIndex].text.utf8)
            #expect(String(decoding: bytes[fact.answerStart..<fact.answerEnd], as: UTF8.self) == fact.answer)
            #expect(fact.contextStart <= fact.sentenceStart)
        }
        #expect(corpus.facts.count > 500)
    }

    @Test("store round-trip and offline snapshot hash")
    func storeAndSnapshot() throws {
        let corpus = try SyntheticCorpus.generate(seed: 3, documentCount: 12)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("raolm-corpus-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        try CorpusStore.write(corpus, to: directory)
        let loaded = try CorpusStore.load(directory)
        #expect(loaded == corpus)
        let facts = try JSONCoding.readLines(Fact.self, from: directory.appendingPathComponent("facts.jsonl"))
        #expect(facts == corpus.facts)

        let snapshot = CorpusSnapshot.offline(corpus)
        #expect(snapshot.corpusHash == corpus.manifest.corpusHash)
        #expect(snapshot.diff(against: corpus).isEmpty)
        try snapshot.save(to: directory)
        #expect(try CorpusSnapshot.load(from: directory) == snapshot)

        var mutated = snapshot
        mutated.documents[0].partitions[0].text += " tampered"
        #expect(!mutated.diff(against: corpus).isEmpty)
    }

    @Test("invalid slugs and counts are rejected")
    func invalidInput() {
        #expect(throws: SyntheticCorpusError.self) { try SyntheticCorpus.generate(slug: "Bad Slug", documentCount: 3) }
        #expect(throws: SyntheticCorpusError.self) { try SyntheticCorpus.generate(documentCount: 0) }
    }
}

@Suite("DataRoot and manifests")
struct DataRootTests {
    @Test("argument beats environment beats default")
    func resolution() {
        #expect(DataRoot.resolve(argument: "/tmp/a", environment: ["RAOLM_DATA_DIR": "/tmp/b"]).url.path == "/tmp/a")
        #expect(DataRoot.resolve(argument: nil, environment: ["RAOLM_DATA_DIR": "/tmp/b"]).url.path == "/tmp/b")
        #expect(DataRoot.resolve(argument: nil, environment: [:]).url.path.hasSuffix("Documents/raolm-db"))
        let root = DataRoot(url: URL(fileURLWithPath: "/tmp/r"))
        #expect(root.snapshot(hash: "0123456789abcdef").lastPathComponent == "0123456789ab")
        #expect(root.threadDB.lastPathComponent == "thread-db")
    }

    @Test("config presets and HF config decoding")
    func config() throws {
        #expect(RaoLMConfig.tiny.parameterCount == 17_304_832)
        try RaoLMConfig.tiny.validate()
        let hf = """
            {"architectures":["LlamaForCausalLM"],"hidden_size":576,"intermediate_size":1536,
             "num_hidden_layers":30,"num_attention_heads":9,"num_key_value_heads":3,"vocab_size":49152,
             "max_position_embeddings":8192,"rope_theta":100000,"rms_norm_eps":1e-05,
             "tie_word_embeddings":true,"model_type":"llama","extra_key":1}
            """
        let decoded = try JSONDecoder().decode(RaoLMConfig.self, from: Data(hf.utf8))
        #expect(decoded.hiddenSize == 576 && decoded.numKeyValueHeads == 3 && decoded.tieWordEmbeddings)
        #expect(throws: RaoLMConfigError.self) { try RaoLMConfig.preset("huge") }
    }

    @Test("run manifest round-trips")
    func manifest() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("raolm-run-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        var manifest = RunManifest(
            runID: "r1", preset: "tiny", model: .tiny,
            tokenizer: TokenizerRef(id: "t", revision: "r", tokenizerSHA256: "h", vocabSize: 49152, eosTokenID: 0),
            corpus: CorpusRef(slug: "veldmar", corpusHash: "c", snapshotPath: "/s", source: "offline", threadID: nil,
                              owner: "o", group: "g", documentCount: 1, partitionCount: 2, tokenCount: 3),
            hyperparameters: TrainingHyperparameters(), provenance: .defaults(for: .tiny))
        manifest.epochs.append(EpochRecord(epoch: 1, steps: 10, trainLoss: 2, trainEntropy: 3, wallClockSeconds: 1))
        manifest.indexedEpochs = [1]
        try manifest.save(to: directory)
        let loaded = try RunManifest.load(directory)
        #expect(loaded.runID == "r1" && loaded.epochs.count == 1 && loaded.latestIndexedEpoch == 1)
        #expect(loaded.provenance.tapLayer == 3)
    }
}
