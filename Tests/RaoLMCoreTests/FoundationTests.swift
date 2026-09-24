import Foundation
import Testing

@testable import RaoLMCore

@Suite("SplitMix64")
struct SplitMix64Tests {
    @Test("known answers for seed 0")
    func knownAnswers() {
        var rng = SplitMix64(seed: 0)
        #expect(rng.next() == 0xE220_A839_7B1D_CDAF)
        #expect(rng.next() == 0x6E78_9E6A_A1B9_65F4)
        #expect(rng.next() == 0x06C4_5D18_8009_454F)
    }

    @Test("bounded draws stay in range and shuffles are deterministic")
    func boundsAndShuffles() {
        var rng = SplitMix64(seed: 99)
        for _ in 0..<1000 {
            let value = rng.nextInt(below: 7)
            #expect((0..<7).contains(value))
        }
        var a = SplitMix64(seed: 5)
        var b = SplitMix64(seed: 5)
        #expect(a.shuffled(Array(0..<50)) == b.shuffled(Array(0..<50)))
        #expect(Set(a.shuffled(Array(0..<50))) == Set(0..<50))
    }
}

@Suite("ContentHash and DocumentID")
struct ContentHashTests {
    @Test("sha256 matches the standard test vector")
    func vector() {
        #expect(ContentHash.sha256Hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    }

    @Test("file hash equals in-memory hash")
    func fileHash() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("raolm-hash-\(UUID().uuidString)")
        let data = Data((0..<3_000_000).map { UInt8(truncatingIfNeeded: $0 &* 31) })
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(try ContentHash.sha256Hex(fileAt: url) == ContentHash.sha256Hex(data))
    }

    @Test("canonical form normalizes line endings and trailing whitespace")
    func canonical() {
        #expect(ContentHash.canonical("  a  \r\nb\t\n\n") == "a\nb")
        #expect(ContentHash.canonical("x") == "x")
    }

    @Test("corpus hash ignores entry order")
    func corpusHashOrder() {
        let entries = [
            ContentHash.CorpusEntry(documentID: "b", partitionIndex: 0, textSHA256: "1"),
            ContentHash.CorpusEntry(documentID: "a", partitionIndex: 1, textSHA256: "2"),
            ContentHash.CorpusEntry(documentID: "a", partitionIndex: 0, textSHA256: "3"),
        ]
        #expect(ContentHash.corpusHash(entries) == ContentHash.corpusHash(entries.reversed()))
        #expect(ContentHash.corpusHash(entries) == ContentHash.sha256Hex("a\t0\t3\na\t1\t2\nb\t0\t1\n"))
    }

    @Test("document ids and partition urls round-trip through URL")
    func ids() {
        let id = DocumentID.make(slug: "veldmar", canonicalText: "hello")
        #expect(DocumentID.isValid(id))
        #expect(!id.contains("/"))
        let url = DocumentID.partitionURL(slug: "veldmar", documentID: id, index: 3)
        #expect(URL(string: url)?.absoluteString == url)
        #expect(DocumentID.isValidHandle("raolm-demo"))
        #expect(!DocumentID.isValidHandle("Raolm"))
        #expect(!DocumentID.isValidHandle("a"))
    }
}

@Suite("TextChunker")
struct TextChunkerTests {
    @Test("paragraph equals partition")
    func paragraphs() {
        let a = String(repeating: "alpha beta. ", count: 20).trimmingCharacters(in: .whitespaces)
        let b = String(repeating: "gamma delta. ", count: 20).trimmingCharacters(in: .whitespaces)
        #expect(TextChunker.chunk("\(a)\n\n\(b)\n") == [a, b])
    }

    @Test("long paragraphs split at sentence ends and never exceed the limit")
    func longParagraph() {
        let sentence = "This sentence is about forty characters."
        let paragraph = Array(repeating: sentence, count: 40).joined(separator: " ")
        let chunks = TextChunker.chunk(paragraph, maxChars: 200, minChars: 10)
        #expect(chunks.count > 1)
        #expect(chunks.allSatisfy { $0.count <= 200 && $0.hasSuffix(".") })
        #expect(chunks.joined(separator: " ") == paragraph)
    }

    @Test("slivers merge and whitespace-only input yields nothing")
    func slivers() {
        let long = String(repeating: "word ", count: 40).trimmingCharacters(in: .whitespaces)
        #expect(TextChunker.chunk("tiny\n\n\(long)", maxChars: 600, minChars: 50) == ["tiny \(long)"])
        #expect(TextChunker.chunk("   \n\n  \n") == [])
        #expect(TextChunker.chunk("x").allSatisfy { !$0.isEmpty })
    }
}
