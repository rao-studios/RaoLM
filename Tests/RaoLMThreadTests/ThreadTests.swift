import Conduit
import Foundation
import Testing

@testable import RaoLMCore
@testable import RaoLMThread

/// Host/client suites need a built Thread (../Thread/.build/release/thread) and are opt-in.
private var threadTests: Bool {
    ProcessInfo.processInfo.environment["RAOLM_THREAD_TESTS"] == "1"
}

private func fixtureCorpus() throws -> GeneratedCorpus {
    try SyntheticCorpus.generate(slug: "fixture", seed: 7, documentCount: 8)
}

@Suite("ThreadIndexRequests")
struct ThreadIndexRequestsTests {
    @Test("requests carry ids, names, texts, one tag and per-partition urls")
    func requestShape() throws {
        let corpus = try fixtureCorpus()
        let requests = ThreadIndexRequests.make(
            corpus.documents, slug: "fixture", owner: "raolm-test", group: "raolm-fixture", groupLabel: "Fixture", batchSize: 3)
        #expect(requests.count == 3)
        #expect(requests.map(\.items.count) == [3, 3, 2])
        for request in requests {
            #expect(request.ownerID == "raolm-test" && request.groupID == "raolm-fixture" && request.groupLabel == "Fixture")
            #expect(request.scope == "")
            for item in request.items {
                let document = try #require(corpus.documents.first { $0.id == item.documentID })
                #expect(item.name == document.name)
                #expect(item.texts == document.partitions.map(\.text))
                #expect(item.tags == ["raolm:fixture"])
                #expect(item.mediaType == "text")
                #expect(item.partitions.count == item.texts.count)
                #expect(item.partitions.map(\.url) == document.partitions.map(\.url))
                #expect(item.partitions.allSatisfy { $0.embedding.isEmpty })
            }
        }
    }

    @Test("document content converts into a snapshot document")
    func conversion() {
        var content = Thread_V1_ThreadDocumentContent()
        content.id = "raolm-x-abc"
        content.name = "Doc"
        content.texts = ["one", "two"]
        var p0 = Thread_V1_ThreadPartitionOutput()
        p0.id = "111"
        p0.url = "raolm://x/raolm-x-abc/p/0"
        var p1 = Thread_V1_ThreadPartitionOutput()
        p1.id = "222"
        p1.url = "raolm://x/raolm-x-abc/p/1"
        content.partitions = [p0, p1]
        let document = ThreadCorpusClient.snapshotDocument(content)
        #expect(document.partitions.map(\.text) == ["one", "two"])
        #expect(document.partitions.map(\.threadPartitionID) == ["111", "222"])
        #expect(document.partitions[1].url == "raolm://x/raolm-x-abc/p/1")
    }

    @Test("child environment drops stack and key variables")
    func environment() {
        let env = ThreadHost.childEnvironment([
            "PATH": "/usr/bin", "HOME": "/Users/x", "HF_HOME": "/models", "RAO_HOME": "/rao", "RAO_APP": "craft",
            "AMBIENT_STACK_SECRET": "s", "THREAD_NODE_ID": "n", "MISTRAL_API_KEY": "k",
        ])
        #expect(env == ["PATH": "/usr/bin", "HOME": "/Users/x", "HF_HOME": "/models"])
    }

    @Test("launch arguments name the data dir, node id and on-device embedding")
    func arguments() {
        let id = UUID()
        let configuration = ThreadHostConfiguration(
            binary: URL(fileURLWithPath: "/bin/thread"), dataDirectory: URL(fileURLWithPath: "/tmp/db"),
            logFile: URL(fileURLWithPath: "/tmp/log"), httpPort: 1, grpcPort: 2)
        let arguments = ThreadHost.arguments(configuration, nodeID: id)
        #expect(arguments.contains("--use-mlx") && arguments.contains("--no-graph-extraction"))
        #expect(arguments.contains(id.uuidString) && arguments.contains("/tmp/db"))
        #expect(!arguments.contains("--mothership-host"))
    }

    @Test("binary candidates prefer the explicit path, then the environment")
    func locator() {
        let candidates = ThreadBinaryLocator.candidates(
            explicit: "/opt/thread", environment: ["RAOLM_THREAD_BINARY": "/env/thread"], roots: [URL(fileURLWithPath: "/repo/RaoLM")])
        #expect(candidates[0].path == "/opt/thread" && candidates[1].path == "/env/thread")
        #expect(candidates.contains { $0.path == "/repo/Thread/.build/release/thread" })
    }
}

@Suite("PortProbe")
struct PortProbeTests {
    @Test("a fresh port is not listening")
    func freePortIsFree() throws {
        let port = try #require(PortProbe.freePort())
        #expect(port > 0)
        #expect(!PortProbe.isListening(port: port))
    }
}

@Suite("ThreadHost + ThreadCorpusClient", .enabled(if: threadTests), .serialized)
struct ThreadIntegrationTests {
    static func makeHost() throws -> (ThreadHost, URL) {
        let binary = try ThreadBinaryLocator.locate(roots: [URL(fileURLWithPath: FileManager.default.currentDirectoryPath)])
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("RaoLMThreadTests-\(UUID().uuidString)")
        let http = try #require(PortProbe.freePort())
        let grpc = try #require(PortProbe.freePort())
        let host = ThreadHost(configuration: ThreadHostConfiguration(
            binary: binary, dataDirectory: directory.appendingPathComponent("thread-db"),
            logFile: directory.appendingPathComponent("thread.log"), httpPort: http, grpcPort: grpc, nodeID: UUID(),
            startupTimeout: 120))
        return (host, directory)
    }

    @Test("launch, ingest eight documents, export them byte for byte, verify a partition, stop")
    func roundTrip() async throws {
        let (host, directory) = try Self.makeHost()
        defer { try? FileManager.default.removeItem(at: directory) }
        let endpoint = try await host.start()
        do {
            let client = ThreadCorpusClient(endpoint: endpoint)
            let health = try await client.health()
            #expect(health.status == "healthy" && health.stack == "open")
            #expect(ThreadHost.readNodeID(dataDirectory: host.configuration.dataDirectory) == endpoint.nodeID)

            let corpus = try fixtureCorpus()
            let owner = "raolm-test"
            let group = "raolm-fixture"
            _ = try await client.index(corpus.documents, slug: "fixture", owner: owner, group: group, groupLabel: "Fixture", batchSize: 3)
            try await client.waitUntilIndexed(expected: Set(corpus.documents.map(\.id)), owner: owner, group: group, prefix: "raolm-fixture-", timeout: 240)

            let snapshot = try await client.exportCorpus(owner: owner, group: group, prefix: "raolm-fixture-", slug: "fixture")
            #expect(snapshot.documentCount == 8)
            #expect(snapshot.diff(against: corpus).isEmpty)
            #expect(snapshot.corpusHash == corpus.manifest.corpusHash)
            #expect(snapshot.documents.allSatisfy { $0.partitions.allSatisfy { ($0.threadPartitionID ?? "").isEmpty == false } })
            #expect(snapshot.threadID == endpoint.nodeID?.uuidString)

            let two = Array(corpus.documents.prefix(2).map(\.id))
            let fetched = try await client.documents(ids: two, owner: owner)
            #expect(Set(fetched.map(\.id)) == Set(two))
            let reader = ThreadCorpusReader(client: client, owner: owner)
            let text = try await reader.partitionText(documentID: corpus.documents[0].id, partitionIndex: 1)
            #expect(text == corpus.documents[0].partitions[1].text)
            #expect(try await reader.partitionText(documentID: "raolm-fixture-missing", partitionIndex: 0) == nil)

            // Another owner sees nothing; a second host on the same ports is refused.
            let other = try await client.exportCorpus(owner: "someone-else", group: group, prefix: "raolm-fixture-", slug: "fixture")
            #expect(other.documentCount == 0)
            let clash = ThreadHost(configuration: host.configuration)
            await #expect(throws: ThreadCorpusError.self) { try await clash.start() }
        } catch {
            await host.stop()
            throw error
        }
        await host.stop()
        #expect(!(await host.isRunning))
    }
}
