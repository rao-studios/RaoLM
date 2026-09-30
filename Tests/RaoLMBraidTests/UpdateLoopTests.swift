import Foundation
import Testing

@testable import RaoLMBraid
@testable import RaoLMCore

private func snapshot(_ documents: [(String, [String])]) -> CorpusSnapshot {
    CorpusSnapshot(
        documents: documents.map { id, texts in
            SnapshotDocument(id: id, name: id, ownerID: "o", groupID: "g", groupLabel: "g", createdAt: 0, mediaType: "text",
                             partitions: texts.enumerated().map { SnapshotPartition(index: $0.offset, text: $0.element) })
        },
        slug: "s", source: "offline", threadID: nil, threadHost: nil, threadGRPCPort: nil, owner: "o", group: "g", documentIDPrefix: "raolm-s-")
}

@Suite("Update loop, pure")
struct UpdateLoopTests {
    @Test("snapshot changes: additions, removals and edits by partition text hash")
    func changes() {
        let before = snapshot([("a", ["one", "two"]), ("b", ["three"])])
        let after = snapshot([("a", ["one", "TWO"]), ("c", ["four", "five"])])
        let change = after.changes(since: before)
        #expect(change.added == ["c#0", "c#1"])
        #expect(change.removed == ["b#0"])
        #expect(change.changed == ["a#1"])
        #expect(change.addedDocuments == ["c"] && change.removedDocuments == ["b"])
        #expect(change.summary == "+2 −1 ~1")
        #expect(after.changes(since: after).isEmpty)
        let first = before.changes(since: nil)
        #expect(first.added.count == 3 && first.removed.isEmpty)
    }

    @Test("the policy reindexes first, trains once enough changed, and only reindexes a removal")
    func policy() {
        let grown = SnapshotChange(added: ["x#0", "x#1"], addedDocuments: ["x"])
        #expect(UpdatePolicy.plan(change: grown, hasLive: false, partitions: 10) == [.train])
        #expect(UpdatePolicy.plan(change: grown, hasLive: true, partitions: 10) == [.reindex, .train])
        #expect(UpdatePolicy.plan(change: SnapshotChange(added: ["x#0"]), hasLive: true, partitions: 100) == [.reindex])
        #expect(UpdatePolicy.plan(change: SnapshotChange(removed: ["x#0"], removedDocuments: ["x"]), hasLive: true, partitions: 10) == [.reindex])
        #expect(UpdatePolicy.plan(change: SnapshotChange(), hasLive: true, partitions: 10).isEmpty)
        #expect(UpdatePolicy.plan(change: grown, hasLive: false, partitions: 0).isEmpty)
    }

    @Test("gates: memorised only for training, withdrawn documents must be gone, spans must verify")
    func gates() {
        #expect(VersionGates.memorised(0.95, kind: .reindex) == nil)
        #expect(VersionGates.memorised(0.95, kind: .train)?.passed == true)
        #expect(VersionGates.memorised(0.85, kind: .train)?.passed == false)
        #expect(VersionGates.memorised(0.85, kind: .train, floor: 0.8)?.passed == true)
        #expect(VersionGates.withdrawn(indexDocuments: ["a", "b"], removed: ["c"]).passed)
        #expect(!VersionGates.withdrawn(indexDocuments: ["a", "b"], removed: ["b"]).passed)
        #expect(VersionGates.verified(statuses: []).passed)
        #expect(VersionGates.verified(statuses: [.verified, .verified]).passed)
        #expect(!VersionGates.verified(statuses: [.verified, .stale]).passed)
        #expect(VersionGates.vocabulary(found: "a", expected: "a").passed && !VersionGates.vocabulary(found: "a", expected: "b").passed)
    }

    @Test("the schedule gives a small corpus as many steps as a large one, and evaluates on the same scale")
    func schedule() {
        let small = ThreadHypervisor.schedule(tokens: 1_400, seqLen: 256, batchSize: 4, targetSteps: 1_000, maxEpochs: 600)
        let large = ThreadHypervisor.schedule(tokens: 14_000, seqLen: 256, batchSize: 4, targetSteps: 1_000, maxEpochs: 600)
        #expect(small.epochs == 500 && small.evalEvery == 20)
        #expect(large.epochs == 72 && large.evalEvery == 2)
        #expect(ThreadHypervisor.schedule(tokens: 10, seqLen: 256, batchSize: 4, targetSteps: 10_000, maxEpochs: 300).epochs == 300)
        #expect(ThreadHypervisor.schedule(tokens: 10_000_000, seqLen: 256, batchSize: 4, targetSteps: 10, maxEpochs: 300).epochs == 8)
    }

    @Test("one world dealt into shards: disjoint names, stable prefixes, every kind in every shard")
    func shards() throws {
        let slugs = ["ambient", "craft", "veil"]
        let small = try SyntheticCorpus.generateShards(slugs: slugs, seed: 5, documentsPerShard: 4)
        let large = try SyntheticCorpus.generateShards(slugs: slugs, seed: 5, documentsPerShard: 7)
        #expect(small.map(\.documents.count) == [4, 4, 4])
        for (i, slug) in slugs.enumerated() {
            #expect(small[i].documents.allSatisfy { $0.id.hasPrefix("raolm-\(slug)-") })
            #expect(large[i].documents.prefix(4).map(\.id) == small[i].documents.map(\.id))
        }
        let subjects = large.map { Set($0.documents.map(\.subject)) }
        for i in subjects.indices { for j in subjects.indices where i < j { #expect(subjects[i].isDisjoint(with: subjects[j])) } }
        #expect(Set(large[0].documents.map(\.kind)).count == DocumentKind.veldmar.count)
        #expect(large[0].manifest.documentIDs == large[0].documents.map(\.id))
        #expect(throws: SyntheticCorpusError.self) { _ = try SyntheticCorpus.generateShards(slugs: ["Bad"], documentsPerShard: 1) }
    }

    @Test("the wire round-trips requests, replies and events")
    func wire() throws {
        let request = NodeRequest(id: 7, op: .advance(session: "s", token: 42, k: 16))
        let decodedRequest = try NodeWire.decode(NodeRequest.self, line: try NodeWire.encode(request).dropLast())
        guard case .advance(let session, let token, let k) = decodedRequest.op else { Issue.record("wrong op"); return }
        #expect(decodedRequest.id == 7 && session == "s" && token == 42 && k == 16)
        #expect(decodedRequest.op.isServing && !NodeRequest.Op.sync.isServing)
        let hidden = NodeOutput.reply(id: 3, reply: .hidden([PackedFloats([1, -2.5, 3e-7])]))
        let back = try NodeWire.decode(NodeOutput.self, line: try NodeWire.encode(hidden))
        guard case .reply(3, .hidden(let packed)) = back else { Issue.record("wrong reply"); return }
        #expect(packed[0].values == [1, -2.5, 3e-7])
        let hit = StrandHit(entry: 4, score: 0.9, value: 17, key: TokenPosition(row: 1, offset: 2), cited: TokenPosition(row: 1, offset: 3),
                            sourceLoss: 0.01, sourceEntropy: 0.2)
        let trajectory = StrandTrajectory(length: 30, trace: 0.08, entry: 5, at: TokenPosition(row: 1, offset: 3), next: 18, phase: 0.4,
                                          fit: 0.9, arc: 0.7, manner: 0.63)
        let steps = [StrandStep(hits: [hit], trajectory: trajectory), StrandStep(hits: [])]
        guard case .reply(4, .opened(let opened)) = try NodeWire.decode(NodeOutput.self, line: try NodeWire.encode(NodeOutput.reply(id: 4, reply: .opened(steps))))
        else { Issue.record("wrong opened reply"); return }
        #expect(opened == steps)
        guard case .reply(5, .hits(let advanced)) = try NodeWire.decode(NodeOutput.self, line: try NodeWire.encode(NodeOutput.reply(id: 5, reply: .hits(steps[0]))))
        else { Issue.record("wrong hits reply"); return }
        #expect(advanced == steps[0])
        var state = StrandState(name: "ambient", label: "Ambient", offline: true, vocabularySHA256: "v", blocks: 6)
        state.stage = .training
        state.cells = [PartitionCell(documentID: "raolm-ambient-0", document: 0, partition: 1, state: .indexed, memorised: 0.5, glyphs: "The Kestrel")]
        let event = NodeOutput.event(.state(state))
        guard case .event(.state(let decoded)) = try NodeWire.decode(NodeOutput.self, line: try NodeWire.encode(event)) else {
            Issue.record("wrong event")
            return
        }
        #expect(decoded == state)
        let splitter = LineSplitter()
        #expect(splitter.feed(Data("{\"a\":1}\n{\"b\"".utf8)).count == 1)
        #expect(splitter.feed(Data(":2}\n\n".utf8)).map { String(decoding: $0, as: UTF8.self) } == ["{\"b\":2}"])
    }
}
